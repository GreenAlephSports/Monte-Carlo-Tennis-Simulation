import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bracket_schema import BracketValidationError, load_bracket_yaml
from data_loader import load_matches
from elo_ratings import STARTING_ELO, apply_training_window

# maps the player count in a round to its conventional Grand-Slam-style name; falls back to
# "Round of N" for any size that doesn't land on one of these (shouldn't happen in practice
# since round sizes always halve down from a validated bracket size)
ROUND_NAME_BY_SIZE = {
    128: "Round of 128",
    64: "Round of 64",
    32: "Round of 32",
    16: "Round of 16",
    8: "Quarterfinals",
    4: "Semifinals",
    2: "Final",
}


def round_name_for_size(size):
    return ROUND_NAME_BY_SIZE.get(size, f"Round of {size}")


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
    "Vallejo A.": "Vallejo D.",  # single-initial spelling, e.g. from PDF-extracted draws
    "Tirante T.": "Tirante T.A.",  # single-initial spelling, e.g. from PDF-extracted draws
    # these keys are in ESPN-match-alias form ("Lastname I.", built from a live ESPN display
    # name - see hybrid_simulation.match_espn_name_to_draw), not bracket-YAML form like the
    # entries above. brackets/montreal_2026.yaml is PDF-extracted (parse_atp_draw.py), which has
    # a known word-gap issue on tightly-kerned multi-word surnames (see parse_atp_draw.py's own
    # docstring) - these six players' surnames came out glued/mis-split, so ESPN's normally-
    # spaced real name can't be pattern-matched to the draw's mangled one without an explicit
    # alias. Needed to backtest that bracket against real ESPN results.
    "Busta P.": "Carrenobusta P.",
    "Perricard G.": "Mpetshiperricard G.",
    "Zandschulp B.": "Vandezandschulp B.",
    "Cerundolo J.": "Cerundolo J.M.",
    "Assche L.": "Vanassche L.",
    "Landaluce M.": "Andaluce M.",
    "Carabelli C.": "Ugocarabelli C.",
    "Minaur A.": "Deminaur A.",
    # ESPN-match-alias form (see the block above) for two more match_espn_name_to_draw edge cases
    # the tiered lastname/suffix matcher can't handle on its own: an apostrophe collapses "O'Connell"
    # into a single ESPN token that can't split into the CSV's two-word lastname ("O Connell C."),
    # and "J.J." as a dotted-initials *firstname* (not a compound lastname) never gets its dots
    # stripped before the suffix comparison, so "Wolf J.J." can't be reached from "J.J. Wolf" either.
    "O'Connell C.": "O Connell C.",
    "Wolf J.": "Wolf J.J.",
}

WTA_NAME_ALIASES = {
    # only safe while a bracket has exactly one "Wang X."-truncated player - Wang Xiyu and Wang
    # Xinyu both truncate to "Wang X.", and a single alias can't tell them apart. A draw with
    # both (e.g. brackets/wta_toronto_2026.yaml) must spell each entry out fully instead
    # (Wang Xiy. / Wang Xin., matching the ratings csv) rather than relying on this alias.
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
            elif len(candidates) > 1:
                # a truncated single-letter query (e.g. "Ruse E." for "Elena-Gabriela") can
                # still be unambiguous even with multiple first-initial candidates, if those
                # candidates are just punctuation variants of the same compound initials
                # (e.g. "Ruse E.G." and "Ruse E-G." both normalize to "EG") - group by full
                # initials and resolve if they all agree, most-played wins the tiebreak (same
                # convention as the tier-1 exact index)
                by_full_initials = {}
                for candidate_name, match_count in candidates:
                    candidate_initials = _split_csv_name(candidate_name)[1]
                    if candidate_initials not in by_full_initials or match_count > by_full_initials[candidate_initials][1]:
                        by_full_initials[candidate_initials] = (candidate_name, match_count)
                if len(by_full_initials) == 1:
                    csv_name = next(iter(by_full_initials.values()))[0]
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


# draw order is normally just list order, but a PDF-extracted bracket carries an explicit
# 'position' (bracket slot number) per player because some slots are omitted from the source
# entirely (the "phantom" opponent of a bye has no row to extract). when every player has a
# position, that's the authoritative draw order; otherwise trust the YAML's list order as-is.
def order_by_draw_position(players):
    if players and all(p.position is not None for p in players):
        return sorted(players, key=lambda p: p.position)
    return list(players)


def get_matchups(players):
    if len(players) % 2 != 0:
        raise ValueError(f"Cannot pair an odd number of players: {len(players)}")
    return list(zip(players[0::2], players[1::2]))


def validate_draw(draw):
    if any(player is None for player in draw):
        missing = sum(1 for player in draw if player is None)
        raise ValueError(f"Draw contains {missing} unresolved slot(s) — fix unmatched names before simulating")
    if len(set(draw)) != len(draw):
        raise ValueError("Draw contains duplicate players")


def _is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


# byes let a player skip Round 1 and advance straight to Round 2 - used for draws smaller than a
# clean power-of-two bracket size (e.g. a 96-player Masters draw padded out with 32 byes). works
# unchanged for a draw with zero byes (a clean 128-draw like a Grand Slam).
def split_byes(items, byes):
    non_bye = [item for item, bye in zip(items, byes) if not bye]
    bye_items = [item for item, bye in zip(items, byes) if bye]
    return non_bye, bye_items


def validate_bracket_structure(byes):
    non_bye_count = sum(1 for bye in byes if not bye)
    bye_count = len(byes) - non_bye_count

    if non_bye_count % 2 != 0:
        raise ValueError(
            f"Number of non-bye players must be even to pair up for Round 1, got {non_bye_count} "
            f"non-bye player(s) (plus {bye_count} bye(s))"
        )

    round1_result = bye_count + non_bye_count // 2
    if not _is_power_of_two(round1_result):
        raise ValueError(
            f"Not a valid bracket size: Round 1 has {non_bye_count} non-bye players "
            f"({non_bye_count // 2} matchups) plus {bye_count} bye(s), leaving {round1_result} players "
            f"for Round 2 onward — {round1_result} is not a power of two"
        )

    return non_bye_count, bye_count


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bracket.py <bracket.yaml>")
        sys.exit(1)

    try:
        bracket = load_bracket_yaml(sys.argv[1])
    except BracketValidationError as e:
        print(e)
        sys.exit(1)

    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    try:
        validate_bracket_structure(byes)
    except ValueError as e:
        print(f"{bracket.source_path}: {e}")
        sys.exit(1)

    tour_config = TOUR_CONFIG[bracket.tour]
    ratings_df = pd.read_csv(tour_config.ratings_path)
    draw, resolutions, _updated_ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date
    )
    unmatched = [r for r in resolutions if r["tier"] is None]

    print(f"Parsed {len(players)} players from {bracket.source_path.name} "
          f"({bracket.tournament} {bracket.year} {bracket.tour}, {bracket.surface})")
    print(f"Matched {len(draw) - len(unmatched)}/{len(draw)} players to Elo ratings")

    tier_counts = Counter(r["tier"] for r in resolutions)
    print(f"  Tier 0 (manual alias override): {tier_counts.get(0, 0)}")
    print(f"  Tier 1 (exact lastname + full initials): {tier_counts.get(1, 0)}")
    print(f"  Tier 2 (lastname + first initial, unique candidate or compound-initials match): {tier_counts.get(2, 0)}")
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
        non_bye_count, bye_count = validate_bracket_structure(byes)
        print(f"\nAll {len(draw)} players matched. Draw is ready to simulate.")

        total_round1 = non_bye_count + bye_count
        bye_note = f", {bye_count} bye(s) advance automatically" if bye_count else ""
        print(f"Round 1 ({round_name_for_size(total_round1)}): {non_bye_count} players play "
              f"({non_bye_count // 2} matchups){bye_note}")

        remaining = bye_count + non_bye_count // 2
        round_number = 2
        while remaining > 1:
            print(f"Round {round_number} ({round_name_for_size(remaining)}): {remaining} players, "
                  f"{remaining // 2} matchups")
            remaining //= 2
            round_number += 1
