import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bracket_schema import DRAW_SIZE, BracketValidationError, load_bracket_yaml
from data_loader import load_matches
from elo_ratings import STARTING_ELO, apply_training_window

ROUND_NAMES = [
    "Round of 128",
    "Round of 64",
    "Round of 32",
    "Round of 16",
    "Quarterfinals",
    "Semifinals",
    "Final",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

ATP_RATINGS_PATH = OUTPUT_DIR / "player_elo_ratings_atp.csv"
WTA_RATINGS_PATH = OUTPUT_DIR / "player_elo_ratings_wta.csv"

ATP_MATCH_DATA_PATH = DATA_DIR / "atp_tennis.csv"
WTA_MATCH_DATA_PATH = DATA_DIR / "wta_tennis.csv"

# csv initials always have a dot in them (B., J-L., J.M.)
INITIALS_TOKEN_RE = re.compile(r"^[A-Za-z](?:[.\-][A-Za-z]*)*\.$")

# manual overrides for players whose bracket-file name and Elo-CSV name don't share a common
# lastname/initials shape (e.g. csv has an extra surname word, or drops a given name entirely) -
# keyed by the exact name string as written in the bracket YAML. lives in code instead of the csv
# so it survives elo_ratings.py regenerating the ratings csv from scratch.
ATP_NAME_ALIASES = {
    "Merida D.": "Merida Aguilar D.",
    "Vallejo A.D.": "Vallejo D.",
}

WTA_NAME_ALIASES = {
    "Wang X.": "Wang Xin.",
    "Pliskova K.": "Pliskova Ka.",
    "Osorio C.": "Osorio M.",
}


@dataclass(frozen=True)
class TourConfig:
    ratings_path: Path
    match_data_path: Path
    name_aliases: dict


TOUR_CONFIG = {
    "ATP": TourConfig(ATP_RATINGS_PATH, ATP_MATCH_DATA_PATH, ATP_NAME_ALIASES),
    "WTA": TourConfig(WTA_RATINGS_PATH, WTA_MATCH_DATA_PATH, WTA_NAME_ALIASES),
}


def _split_words(text):
    return [w for w in re.split(r"[\s\-]+", text.strip()) if w]


def _normalize_lastname(text):
    return " ".join(_split_words(text)).lower()


# splits a csv-style name like "Van De Zandschulp B." into (lastname, initials).
# bracket YAML player names are expected in this same format, so this same splitter is used on
# both sides of the match - that's what makes tier 1 (exact lastname + initials) hit cleanly.
def _split_csv_name(name):
    tokens = name.split()
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


def _lastnames_in_training_window(match_data_path, cutoff_date):
    df = apply_training_window(load_matches(match_data_path), cutoff_date)
    player_names = pd.concat([df["Player_1"], df["Player_2"]]).unique()
    return {_split_csv_name(name.strip())[0] for name in player_names}


def _has_training_history(lastname, known_lastnames):
    #true if lastname matches a known one exactly, or is a prefix/suffix of one
    return any(
        known == lastname or known.startswith(lastname + " ") or lastname.startswith(known + " ")
        for known in known_lastnames
    )


def match_draw_to_ratings(players, ratings_df, name_aliases, match_data_path, cutoff_date):
    exact_index = _build_ratings_index(ratings_df)
    first_initial_index = _build_first_initial_index(ratings_df)
    known_lastnames = None  # computed lazily; only needed if tiers 1-2 both miss

    ratings_df = ratings_df.copy()
    existing_names = set(ratings_df["player"])
    new_rows = []

    names = []
    resolutions = []
    for entry in players:
        raw_name = entry.name
        lastname, full_initials = _split_csv_name(raw_name)
        first_initial = full_initials[0] if full_initials else ""

        # tier 0: explicit manual alias, checked first since it's a known-correct override.
        # falls through to the normal tiers if the aliased name isn't in the csv (e.g. it
        # got renamed upstream) instead of trusting a stale alias.
        csv_name = name_aliases.get(raw_name)
        csv_name = csv_name if csv_name in existing_names else None
        tier = 0 if csv_name is not None else None

        # tier 1: bracket names are written in ratings-csv format already, so splitting both
        # sides the same way (_split_csv_name) usually hits this tier directly.
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
                known_lastnames = _lastnames_in_training_window(match_data_path, cutoff_date)
            if not _has_training_history(lastname, known_lastnames):
                csv_name = raw_name
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
            "seed": entry.seed,
            "name": raw_name,
            "status": entry.status,
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bracket.py <bracket.yaml>")
        sys.exit(1)

    try:
        bracket = load_bracket_yaml(sys.argv[1])
    except BracketValidationError as e:
        print(e)
        sys.exit(1)

    tour_config = TOUR_CONFIG[bracket.tour]
    ratings_df = pd.read_csv(tour_config.ratings_path)
    draw, resolutions, _updated_ratings_df = match_draw_to_ratings(
        bracket.players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date
    )
    unmatched = [r for r in resolutions if r["tier"] is None]

    print(f"Parsed {len(bracket.players)} players from {bracket.source_path.name} "
          f"({bracket.tournament} {bracket.year} {bracket.tour}, {bracket.surface})")
    print(f"Matched {len(draw) - len(unmatched)}/{len(draw)} players to Elo ratings")

    tier_counts = Counter(r["tier"] for r in resolutions)
    print(f"  Tier 0 (manual alias override): {tier_counts.get(0, 0)}")
    print(f"  Tier 1 (exact lastname + full initials): {tier_counts.get(1, 0)}")
    print(f"  Tier 2 (lastname + first initial, unique candidate): {tier_counts.get(2, 0)}")
    print(f"  Tier 3 (no training-window history, STARTING_ELO placeholder): {tier_counts.get(3, 0)}")
    print(f"  Unresolved: {tier_counts.get(None, 0)}")

    non_tier1 = [r for r in resolutions if r["tier"] != 1]
    if non_tier1:
        print("\nNon-tier-1 matches (review these):")
        for entry in non_tier1:
            seed = f"[{entry['seed']}]" if entry["seed"] else ""
            tier_label = f"tier {entry['tier']}" if entry["tier"] is not None else "UNRESOLVED"
            print(f"  [{tier_label}] {entry['name']} {seed} -> {entry['csv_name']}")

    if unmatched:
        print("\nUnmatched names (check spelling/format against the Elo CSV):")
        for entry in unmatched:
            seed = f"[{entry['seed']}]" if entry["seed"] else ""
            print(f"  {entry['name']} {seed}  (looked for key={entry['expected_key']})")
    else:
        validate_draw(draw)
        print("\nAll 128 players matched. Draw is ready to simulate.")
        for round_index, round_name in enumerate(ROUND_NAMES):
            players_remaining = DRAW_SIZE // (2 ** round_index)
            print(f"Round {round_index + 1} ({round_name}): {players_remaining} players, "
                  f"{players_remaining // 2} matchups")
