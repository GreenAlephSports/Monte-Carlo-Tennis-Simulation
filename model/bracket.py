import re
from pathlib import Path

import pandas as pd

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


def _build_ratings_index(ratings_df):
    """Map (normalized lastname, initials) -> best CSV player name (most total matches wins ties)."""
    index = {}
    total_matches = (
        ratings_df["hard_matches"] + ratings_df["clay_matches"] + ratings_df["grass_matches"]
    )
    for csv_name, matches in zip(ratings_df["player"], total_matches):
        key = _split_csv_name(csv_name.strip())
        if key not in index or matches > index[key][1]:
            index[key] = (csv_name, matches)
    return {key: name for key, (name, _) in index.items()}


def match_draw_to_ratings(draw_entries, ratings_df):
    """Resolve each draw entry to its Elo CSV player name. Returns (names, unmatched)."""
    ratings_index = _build_ratings_index(ratings_df)

    names = []
    unmatched = []
    for entry in draw_entries:
        lastname = _normalize_lastname(entry["lastname"])
        initials = _initials_from_words(_split_words(entry["firstname"]))
        csv_name = ratings_index.get((lastname, initials))
        names.append(csv_name)
        if csv_name is None:
            unmatched.append({**entry, "expected_key": (lastname, initials)})

    return names, unmatched


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
DRAW, UNMATCHED = match_draw_to_ratings(_draw_entries, _ratings_df)


if __name__ == "__main__":
    print(f"Parsed {len(_draw_entries)} draw entries from {DRAW_MD_PATH.name}")
    print(f"Matched {len(DRAW) - len(UNMATCHED)}/{len(DRAW)} players to Elo ratings")

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
