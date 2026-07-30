import re
from collections import Counter
from pathlib import Path

import pandas as pd

from data_loader import load_matches
from elo_ratings import STARTING_ELO, apply_training_window

DRAW_SIZE = 128
SURFACE = "Grass"

ROUND_NAMES = [
    "Round of 128",
    "Round of 64",
    "Round of 32",
    "Round of 16",
    "Quarterfinals",
    "Semifinals",
    "Final",
]

ATP_DRAW_MD_PATH = Path(__file__).resolve().parent.parent / "data" / "wimbledon_2026_draw.md"
WTA_DRAW_MD_PATH = Path(__file__).resolve().parent.parent / "data" / "wimbledon_2026_wta_draw.md"
ATP_RATINGS_PATH = Path(__file__).resolve().parent.parent / "output" / "player_elo_ratings_atp.csv"
WTA_RATINGS_PATH = Path(__file__).resolve().parent.parent / "output" / "player_elo_ratings_wta.csv"

ATP_MATCH_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "atp_tennis.csv"
WTA_MATCH_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "wta_tennis.csv"

# matches lines from the draw markdown - seed and status (wildcard/qualifier/etc) are both optional
DRAW_LINE_RE = re.compile(
    r"^\d+\.\s+(?P<lastname>[^,]+),\s+(?P<firstname>[^\[\(]+?)"
    r"\s*(?:\[(?P<seed>\d+)\])?\s*(?:\((?P<status>[A-Z])\))?\s*$"
)

# csv initials always have a dot in them (B., J-L., J.M.)
INITIALS_TOKEN_RE = re.compile(r"^[A-Za-z](?:[.\-][A-Za-z]*)*\.$")

# manual overrides for players whose draw name and Elo-CSV name don't share a common
# lastname/initials shape (e.g. csv has an extra surname word, or a different first name
# entirely) - keyed by normalized (draw lastname, draw firstname). lives in code instead of
# the csv so it survives elo_ratings.py regenerating player_elo_ratings.csv from scratch.
ATP_NAME_ALIASES = {
    ("merida", "daniel"): "Merida Aguilar D.",
    ("vallejo", "adolfo daniel"): "Vallejo D.",
}

WTA_NAME_ALIASES = {
    ("wang", "xinyu"): "Wang Xin.",
    ("pliskova", "karolina"): "Pliskova Ka.",
    ("osorio", "camila"): "Osorio M.",
}


def _split_words(text):
    return [w for w in re.split(r"[\s\-]+", text.strip()) if w]


def parse_draw(path):
# reads the draw md file into a list of 128 entries, in bracket order
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            match = DRAW_LINE_RE.match(line.strip())
            if not match:
                continue
            entries.append({
                "lastname": match.group("lastname").strip(),
                "firstname": match.group("firstname").strip(),
                "seed": match.group("seed"),
                "status": match.group("status"),
            })
    return entries


def _normalize_lastname(text):
    return " ".join(_split_words(text)).lower()


def _initials_from_words(words):
    return "".join(w[0] for w in words).upper()

# splits a csv name like "Van De Zandschulp B." into (lastname, initials)
def _split_csv_name(csv_name):
    tokens = csv_name.split()
    boundary = len(tokens)
    while boundary > 0 and INITIALS_TOKEN_RE.match(tokens[boundary - 1]):
        boundary -= 1
    lastname_tokens = tokens[:boundary]
    initials_tokens = tokens[boundary:]
    lastname = _normalize_lastname(" ".join(lastname_tokens))
    initials = "".join(re.sub(r"[.\-]", "", tok) for tok in initials_tokens).upper()
    return lastname, initials


def _ratings_total_matches(ratings_df):
    return ratings_df["hard_matches"] + ratings_df["clay_matches"] + ratings_df["grass_matches"]


def _build_ratings_index(ratings_df):
# tier 1 lookup table: (lastname, full initials) to csv name, most-played player wins ties
    index = {}
    for csv_name, matches in zip(ratings_df["player"], _ratings_total_matches(ratings_df)):
        key = _split_csv_name(csv_name.strip())
        if key not in index or matches > index[key][1]:
            index[key] = (csv_name, matches)
    return {key: name for key, (name, _) in index.items()}


def _build_first_initial_index(ratings_df):
    index = {}
    for csv_name, matches in zip(ratings_df["player"], _ratings_total_matches(ratings_df)):
        lastname, initials = _split_csv_name(csv_name.strip())
        if not initials:
            continue
        index.setdefault((lastname, initials[0]), []).append((csv_name, matches))
    return index


def _lastnames_in_training_window(match_data_path):
    df = apply_training_window(load_matches(match_data_path))
    player_names = pd.concat([df["Player_1"], df["Player_2"]]).unique()
    return {_split_csv_name(name.strip())[0] for name in player_names}


def _has_training_history(lastname, known_lastnames):
    #true if lastname matches a known one exactly, or is a prefix/suffix of one
    return any(
        known == lastname or known.startswith(lastname + " ") or lastname.startswith(known + " ")
        for known in known_lastnames
    )


def match_draw_to_ratings(draw_entries, ratings_df, name_aliases, match_data_path):
    exact_index = _build_ratings_index(ratings_df)
    first_initial_index = _build_first_initial_index(ratings_df)
    known_lastnames = None  # computed lazily; only needed if tiers 1-2 both miss

    ratings_df = ratings_df.copy()
    existing_names = set(ratings_df["player"])
    new_rows = []

    names = []
    resolutions = []
    for entry in draw_entries:
        lastname = _normalize_lastname(entry["lastname"])
        firstname_words = _split_words(entry["firstname"])
        full_initials = _initials_from_words(firstname_words)
        first_initial = firstname_words[0][0].upper() if firstname_words else ""
        firstname_key = " ".join(w.lower() for w in firstname_words)

        # tier 0: explicit manual alias, checked first since it's a known-correct override.
        # falls through to the normal tiers if the aliased name isn't in the csv (e.g. it
        # got renamed upstream) instead of trusting a stale alias.
        csv_name = name_aliases.get((lastname, firstname_key))
        csv_name = csv_name if csv_name in existing_names else None
        tier = 0 if csv_name is not None else None

        if csv_name is None:
            csv_name = exact_index.get((lastname, full_initials))
            tier = 1 if csv_name is not None else None

        if csv_name is None:
            candidates = first_initial_index.get((lastname, first_initial), [])
            if len(candidates) == 1:
                csv_name = candidates[0][0]
                tier = 2

        # tier 3: only give a fresh STARTING_ELO placeholder if this player genuinely has no matches in the training window 

        if csv_name is None:
            if known_lastnames is None:
                known_lastnames = _lastnames_in_training_window(match_data_path)
            if not _has_training_history(lastname, known_lastnames):
                csv_name = f"{lastname.title()} {first_initial}."
                tier = 3
                if csv_name not in existing_names:
                    new_rows.append({
                        "player": csv_name,
                        "hard_elo": STARTING_ELO,
                        "clay_elo": STARTING_ELO,
                        "grass_elo": STARTING_ELO,
                        "overall_elo": STARTING_ELO,
                        "hard_matches": 0,
                        "clay_matches": 0,
                        "grass_matches": 0,
                    })
                    existing_names.add(csv_name)

        names.append(csv_name)
        resolutions.append({
            **entry,
            "expected_key": (lastname, full_initials),
            "tier": tier,
            "csv_name": csv_name,
        })

    if new_rows:
        ratings_df = pd.concat([ratings_df, pd.DataFrame(new_rows)], ignore_index=True)

    return names, resolutions, ratings_df


def get_matchups(players):
    if len(players) % 2 != 0:
        raise ValueError(f"Cannot pair an odd number of players: {len(players)}")
    return list(zip(players[0::2], players[1::2]))


def validate_draw(draw):
    size = len(draw)
    if size != DRAW_SIZE:
        raise ValueError(f"Expected a {DRAW_SIZE}-player draw, got {size}")
    if any(player is None for player in draw):
        missing = sum(1 for player in draw if player is None)
        raise ValueError(f"Draw contains {missing} unresolved slot(s) — fix unmatched names before simulating")
    if len(set(draw)) != size:
        raise ValueError("Draw contains duplicate players")


_draw_entries = parse_draw(ATP_DRAW_MD_PATH)
_ratings_df = pd.read_csv(ATP_RATINGS_PATH)
DRAW, RESOLUTIONS, _updated_ratings_df = match_draw_to_ratings(
    _draw_entries, _ratings_df, ATP_NAME_ALIASES, ATP_MATCH_DATA_PATH
)
UNMATCHED = [r for r in RESOLUTIONS if r["tier"] is None]

if len(_updated_ratings_df) != len(_ratings_df):
    _updated_ratings_df.to_csv(ATP_RATINGS_PATH, index=False)

_wta_draw_entries = parse_draw(WTA_DRAW_MD_PATH)
_wta_ratings_df = pd.read_csv(WTA_RATINGS_PATH)
WTA_DRAW, WTA_RESOLUTIONS, _updated_wta_ratings_df = match_draw_to_ratings(
    _wta_draw_entries, _wta_ratings_df, WTA_NAME_ALIASES, WTA_MATCH_DATA_PATH
)
WTA_UNMATCHED = [r for r in WTA_RESOLUTIONS if r["tier"] is None]

if len(_updated_wta_ratings_df) != len(_wta_ratings_df):
    _updated_wta_ratings_df.to_csv(WTA_RATINGS_PATH, index=False)


if __name__ == "__main__":
    print(f"Parsed {len(_draw_entries)} draw entries from {ATP_DRAW_MD_PATH.name}")
    print(f"Matched {len(DRAW) - len(UNMATCHED)}/{len(DRAW)} players to Elo ratings")

    tier_counts = Counter(r["tier"] for r in RESOLUTIONS)
    print(f"  Tier 0 (manual alias override): {tier_counts.get(0, 0)}")
    print(f"  Tier 1 (exact lastname + full initials): {tier_counts.get(1, 0)}")
    print(f"  Tier 2 (lastname + first initial, unique candidate): {tier_counts.get(2, 0)}")
    print(f"  Tier 3 (no training-window history, STARTING_ELO placeholder): {tier_counts.get(3, 0)}")
    print(f"  Unresolved: {tier_counts.get(None, 0)}")

    non_tier1 = [r for r in RESOLUTIONS if r["tier"] != 1]
    if non_tier1:
        print("\nNon-tier-1 matches (review these):")
        for entry in non_tier1:
            seed = f"[{entry['seed']}]" if entry["seed"] else ""
            tier_label = f"tier {entry['tier']}" if entry["tier"] is not None else "UNRESOLVED"
            print(f"  [{tier_label}] {entry['lastname']}, {entry['firstname']} {seed} -> {entry['csv_name']}")

    if UNMATCHED:
        print("\nUnmatched names (check spelling/format against the Elo CSV):")
        for entry in UNMATCHED:
            seed = f"[{entry['seed']}]" if entry["seed"] else ""
            print(f"  {entry['lastname']}, {entry['firstname']} {seed}  (looked for key={entry['expected_key']})")
    else:
        validate_draw(DRAW)
        print("\nAll 128 players matched. Draw is ready to simulate.")
        for round_index, round_name in enumerate(ROUND_NAMES):
            players_remaining = DRAW_SIZE // (2 ** round_index)
            print(f"Round {round_index + 1} ({round_name}): {players_remaining} players, "
                  f"{players_remaining // 2} matchups")
