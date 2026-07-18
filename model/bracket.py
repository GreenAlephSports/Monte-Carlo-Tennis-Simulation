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

DRAW_MD_PATH = Path(__file__).resolve().parent.parent / "data" / "wimbledon_2026_draw.md"
RATINGS_PATH = Path(__file__).resolve().parent.parent / "output" / "player_elo_ratings.csv"

DRAW_LINE_RE = re.compile(
    r"^\d+\.\s+(?P<lastname>[^,]+),\s+(?P<firstname>[^\[\(]+?)"
    r"\s*(?:\[(?P<seed>\d+)\])?\s*(?:\((?P<status>[A-Z])\))?\s*$"
)

# A CSV name's trailing "initials" tokens are letters/dots/hyphens containing
# at least one dot (e.g. "B.", "J-L.", "J.M."); lastname tokens never contain a dot.
INITIALS_TOKEN_RE = re.compile(r"^[A-Za-z](?:[.\-][A-Za-z]*)*\.$")


def _split_words(text):
    return [w for w in re.split(r"[\s\-]+", text.strip()) if w]


def parse_draw(path=DRAW_MD_PATH):
    """Parse the markdown draw file into a list of 128 entries in bracket order."""
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


def _split_csv_name(csv_name):
    """Split an Elo CSV name like 'Van De Zandschulp B.' into (lastname, initials letters)."""
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
    """Map (normalized lastname, full initials) -> best CSV player name (most total matches wins ties)."""
    index = {}
    for csv_name, matches in zip(ratings_df["player"], _ratings_total_matches(ratings_df)):
        key = _split_csv_name(csv_name.strip())
        if key not in index or matches > index[key][1]:
            index[key] = (csv_name, matches)
    return {key: name for key, (name, _) in index.items()}


def _build_first_initial_index(ratings_df):
    """Map (normalized lastname, first initial) -> list of candidate CSV player names."""
    index = {}
    for csv_name, matches in zip(ratings_df["player"], _ratings_total_matches(ratings_df)):
        lastname, initials = _split_csv_name(csv_name.strip())
        if not initials:
            continue
        index.setdefault((lastname, initials[0]), []).append((csv_name, matches))
    return index


def _lastnames_in_training_window():
    """Normalized lastnames of every player with at least one match in the current Elo training window."""
    df = apply_training_window(load_matches())
    player_names = pd.concat([df["Player_1"], df["Player_2"]]).unique()
    return {_split_csv_name(name.strip())[0] for name in player_names}


def _has_training_history(lastname, known_lastnames):
    """True if `lastname` matches a known one exactly, or is a whole-word prefix/suffix of one
    (e.g. draw lastname 'merida' vs CSV's fuller 'merida aguilar') — avoids tier 3 false positives
    on players whose surname is recorded with more or fewer words than the draw uses.
    """
    return any(
        known == lastname or known.startswith(lastname + " ") or lastname.startswith(known + " ")
        for known in known_lastnames
    )


def match_draw_to_ratings(draw_entries, ratings_df):
    """Resolve each draw entry to an Elo CSV player name via a 3-tier fallback chain:
    (1) exact lastname + full initials, (2) lastname + first initial when unambiguous,
    (3) a fresh STARTING_ELO placeholder for players with zero rows in the training window.
    Returns (names, resolutions, updated_ratings_df) where resolutions records the tier used
    (1, 2, 3, or None for unresolved) for every draw entry, in draw order.
    """
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

        csv_name = exact_index.get((lastname, full_initials))
        tier = 1 if csv_name is not None else None

        if csv_name is None:
            candidates = first_initial_index.get((lastname, first_initial), [])
            if len(candidates) == 1:
                csv_name = candidates[0][0]
                tier = 2

        if csv_name is None:
            if known_lastnames is None:
                known_lastnames = _lastnames_in_training_window()
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
    """Pair adjacent players in the current round: [p0, p1, p2, p3] -> [(p0, p1), (p2, p3)]."""
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


_draw_entries = parse_draw()
_ratings_df = pd.read_csv(RATINGS_PATH)
DRAW, RESOLUTIONS, _updated_ratings_df = match_draw_to_ratings(_draw_entries, _ratings_df)
UNMATCHED = [r for r in RESOLUTIONS if r["tier"] is None]

# Persist any tier-3 placeholder rows so win_probability.py's own CSV read picks them up too.
if len(_updated_ratings_df) != len(_ratings_df):
    _updated_ratings_df.to_csv(RATINGS_PATH, index=False)


if __name__ == "__main__":
    print(f"Parsed {len(_draw_entries)} draw entries from {DRAW_MD_PATH.name}")
    print(f"Matched {len(DRAW) - len(UNMATCHED)}/{len(DRAW)} players to Elo ratings")

    tier_counts = Counter(r["tier"] for r in RESOLUTIONS)
    print(f"  Tier 1 (exact lastname + full initials): {tier_counts.get(1, 0)}")
    print(f"  Tier 2 (lastname + first initial, unique candidate): {tier_counts.get(2, 0)}")
    print(f"  Tier 3 (no training-window history, STARTING_ELO placeholder): {tier_counts.get(3, 0)}")
    print(f"  Unresolved: {tier_counts.get(None, 0)}")

    non_tier1 = [r for r in RESOLUTIONS if r["tier"] != 1]
    if non_tier1:
        print("\nNon-tier-1 matches (review these):")
        for entry in non_tier1:
            seed = f"[{entry['seed']}]" if entry["seed"] else ""
            tier_label = f"tier {entry['tier']}" if entry["tier"] else "UNRESOLVED"
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
