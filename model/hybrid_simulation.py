"""Hybrid simulation: treats real match results (from live_scores.py / ESPN) as known up through
a given round, then Monte Carlo-simulates everything after that - even rounds that have already
actually happened in reality, if they're past the cutoff.

Round numbering matches the bracket's own round structure (1 = the non-bye Round 1 pairing, then
2, 3, ... up through the Final), not ESPN's labels directly - those are mapped onto it. Real
results can only be applied for a *prefix* of rounds starting at 1: if rounds 1..N are all fully
known, the field entering round N+1 is deterministic (no randomness used yet), so that becomes
the fixed starting point for Monte Carlo simulation of the rest. There's no such thing as "round 3
known but round 2 unknown" - the field a simulated round 2 would produce doesn't match reality, so
round 3 can't be pinned to real results without round 2 having been real too.

Usage:
    python model/hybrid_simulation.py brackets/wta_toronto_2026.yaml --through-round 3
    python model/hybrid_simulation.py brackets/wta_toronto_2026.yaml --all-rounds
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket import (  # noqa: E402
    TOUR_CONFIG, get_matchups, match_draw_to_ratings, order_by_draw_position,
    split_byes, validate_bracket_structure, validate_draw,
)
from bracket_schema import BracketValidationError, load_bracket_yaml  # noqa: E402
from data_loader import load_matches  # noqa: E402
from elo_ratings import calculate_elo_ratings  # noqa: E402
from live_scores import LiveScoresError, extract_matches, fetch_scoreboard  # noqa: E402
from simulate import N_SIMULATIONS, run_simulations_from_field  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

STANDARD_TAIL_ROUNDS = ["Quarterfinal", "Semifinal", "Final"]
NUMBERED_ROUND_RE = re.compile(r"^Round (\d+)$")
TOUR_SINGLES_CATEGORY = {"atp": "Men's Singles", "wta": "Women's Singles"}


def build_round_sequence(round_labels):
    """Orders ESPN round labels into the main-draw sequence: numeric 'Round N' labels first
    (ascending), then Quarterfinal/Semifinal/Final. Anything else (e.g. 'Qualifying Final') is
    pre-draw and excluded - those players already appear as regular entries in the main draw."""
    numbered = sorted(
        (label for label in round_labels if NUMBERED_ROUND_RE.match(label)),
        key=lambda label: int(NUMBERED_ROUND_RE.match(label).group(1)),
    )
    tail = [label for label in STANDARD_TAIL_ROUNDS if label in round_labels]
    return numbered + tail


def _lastname_and_suffix(csv_name):
    """Simpler split than bracket._split_csv_name: last whitespace token is always treated as
    the truncated-firstname suffix (whatever convention it uses - single dotted initial, e.g.
    'E.', multi-letter compound initials, e.g. 'E.G.', or a literal N-char prefix used to
    disambiguate a collision, e.g. 'Xiy.'), everything before it is the lastname. Good enough
    for approximate matching against messy real-world ESPN name strings; bracket.py's stricter
    splitter cares about exact tier-1/tier-2 lookup semantics, which isn't the goal here."""
    tokens = csv_name.split()
    if len(tokens) < 2:
        return csv_name.lower(), ""
    lastname = " ".join(tokens[:-1]).lower()
    suffix = re.sub(r"[.\-]", "", tokens[-1]).lower()
    return lastname, suffix


def _firstname_matches_suffix(firstname_part, suffix):
    """The suffix might be a literal prefix of a single-word firstname (e.g. 'xiy' matching
    'Xiyu' - a disambiguating truncation) or the per-word initials of a compound/hyphenated
    firstname (e.g. 'eg' matching 'Elena-Gabriela')."""
    if not suffix:
        return False
    firstname_part = firstname_part.lower()
    if firstname_part.startswith(suffix):
        return True
    initials = "".join(w[0] for w in re.split(r"[\s\-]+", firstname_part) if w)
    return initials == suffix or initials.startswith(suffix)


def match_espn_name_to_draw(espn_name, draw_csv_names, name_aliases=None):
    """Matches an ESPN full display name (e.g. 'Iga Swiatek') to one of the bracket's already-
    resolved ratings-csv-format names (e.g. 'Swiatek I.'). Tries the draw name's lastname
    against both the trailing AND leading word(s) of the ESPN name - most names are Western
    order (Firstname Lastname), but ESPN represents some players (seen here: Chinese players
    Wang Xiyu/Wang Xinyu/Zhang Shuai) in native order (Lastname Firstname) instead, and there's
    no reliable per-player signal for which. Also checks the manual alias table for the small
    number of players whose ratings-csv name doesn't share an initial with their real first
    name at all (e.g. 'Osorio M.' for Camila Osorio) - unrecoverable by name similarity alone.
    Returns None if zero or multiple candidates match, rather than guess."""
    espn_words = espn_name.split()
    candidates = set()

    for csv_name in draw_csv_names:
        lastname, suffix = _lastname_and_suffix(csv_name)
        if not suffix:
            continue
        lastname_words = lastname.split()
        n = len(lastname_words)
        if n == 0 or len(espn_words) < n + 1:
            continue
        for lastname_at_tail in (True, False):
            candidate_lastname = [w.lower() for w in (espn_words[-n:] if lastname_at_tail else espn_words[:n])]
            if candidate_lastname != lastname_words:
                continue
            firstname_part = "".join(espn_words[:-n] if lastname_at_tail else espn_words[n:])
            if _firstname_matches_suffix(firstname_part, suffix):
                candidates.add(csv_name)

    if name_aliases and len(espn_words) >= 2:
        alias_key = f"{espn_words[-1].title()} {espn_words[0][0].upper()}."
        alias_target = name_aliases.get(alias_key)
        if alias_target in draw_csv_names:
            candidates.add(alias_target)

    return next(iter(candidates)) if len(candidates) == 1 else None


def build_real_results_by_round(espn_matches, draw_csv_names, name_aliases=None):
    """Returns (results_by_round, round_sequence, unresolved_names). results_by_round[n] maps
    frozenset({player_a, player_b}) -> winner, for round n. Only completed matches with a clear
    winner and both players resolvable to the draw contribute a result."""
    round_labels = {m["round"] for m in espn_matches if m["round"]}
    round_sequence = build_round_sequence(round_labels)
    round_index = {label: i + 1 for i, label in enumerate(round_sequence)}

    results_by_round = defaultdict(dict)
    unresolved_names = set()
    for m in espn_matches:
        round_num = round_index.get(m["round"])
        if round_num is None:
            continue
        if m["status_state"] != "post" or not m["winner"]:
            continue

        p1 = match_espn_name_to_draw(m["player_1"], draw_csv_names, name_aliases)
        p2 = match_espn_name_to_draw(m["player_2"], draw_csv_names, name_aliases)
        if p1 is None:
            unresolved_names.add(m["player_1"])
        if p2 is None:
            unresolved_names.add(m["player_2"])
        if p1 is None or p2 is None:
            continue

        winner_csv = p1 if m["winner"] == m["player_1"] else p2
        results_by_round[round_num][frozenset((p1, p2))] = winner_csv

    return results_by_round, round_sequence, unresolved_names


def replay_real_rounds(non_bye_players, bye_players, results_by_round):
    """Replays as many rounds as have a COMPLETE set of real results, starting at round 1 and
    stopping at the first round with any unresolved matchup. Returns a list `fields` where
    fields[n] is the real field entering round n+1 (fields[0] = pre-Round-1 non-bye field);
    len(fields) - 1 is the highest round fully known from real results.

    Round 1's matchups come from get_matchups(non_bye_players) - that pairing genuinely matches
    the real bracket, since a bye's phantom opponent never sits between two real non-bye players.
    Round 2 onward is different: the simulated field is "Round 1 winners + byes" concatenated
    (plenty good enough for plain Monte Carlo simulation, where pairing order doesn't affect
    aggregate win probabilities), which does NOT reconstruct the true bracket-tree adjacency a
    real Round 2 pairing follows. So for round 2+, ESPN's own reported pairings are used as the
    source of truth directly, rather than re-deriving an expected pairing order ourselves - the
    round only counts as fully known if its real matches account for every player in the field
    exactly once."""
    fields = [list(non_bye_players)]
    current_field = list(non_bye_players)
    round_num = 1
    while len(current_field) > 1:
        round_results = results_by_round.get(round_num, {})
        if round_num == 1:
            matchups = get_matchups(current_field)
        else:
            matchups = [tuple(pair) for pair in round_results.keys()]
            players_in_matchups = [p for pair in matchups for p in pair]
            if set(players_in_matchups) != set(current_field) or len(players_in_matchups) != len(current_field):
                return fields

        winners = []
        for pair in matchups:
            winner = round_results.get(frozenset(pair))
            if winner is None:
                return fields
            winners.append(winner)
        current_field = winners + list(bye_players) if round_num == 1 else winners
        fields.append(current_field)
        round_num += 1
    return fields


def _report(label, draw, champion_counts, n_simulations, output_path):
    results = pd.DataFrame({
        "player": draw,
        "win_count": [champion_counts.get(player, 0) for player in draw],
    })
    results["tournament_win_probability"] = results["win_count"] / n_simulations
    results = results.sort_values("tournament_win_probability", ascending=False).reset_index(drop=True)
    results.to_csv(output_path, index=False)

    print(f"\n=== {label} === (saved to {output_path})")
    print(results.head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bracket_path", type=Path)
    parser.add_argument("--through-round", type=int, default=None,
                         help="rounds 1..N are treated as known (real); everything after is "
                              "simulated normally regardless of what actually happened")
    parser.add_argument("--all-rounds", action="store_true",
                         help="generate one snapshot per fully-known real round (1..latest), "
                              "each to its own labeled output file")
    parser.add_argument("--simulations", type=int, default=N_SIMULATIONS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if not args.all_rounds and args.through_round is None:
        parser.error("pass --through-round N or --all-rounds")

    try:
        bracket = load_bracket_yaml(args.bracket_path)
    except BracketValidationError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    try:
        validate_bracket_structure(byes)
    except ValueError as e:
        print(f"{args.bracket_path}: {e}", file=sys.stderr)
        sys.exit(1)

    tour_config = TOUR_CONFIG[bracket.tour]
    matches_history = load_matches(tour_config.match_data_path)
    ratings_df = calculate_elo_ratings(matches_history, bracket.start_date)
    ratings_df = ratings_df.sort_values("overall_elo", ascending=False).reset_index(drop=True)

    draw, resolutions, ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date
    )
    unmatched = [r for r in resolutions if r["tier"] is None]
    if unmatched:
        print("Unmatched bracket names, fix before running a hybrid simulation:", file=sys.stderr)
        for entry in unmatched:
            print(f"  {entry['name']}", file=sys.stderr)
        sys.exit(1)
    validate_draw(draw)
    non_bye_players, bye_players = split_byes(draw, byes)

    try:
        espn_data = fetch_scoreboard(bracket.tour.lower())
    except LiveScoresError as e:
        print(f"ERROR fetching live results: {e}", file=sys.stderr)
        sys.exit(1)
    espn_matches, stats = extract_matches(espn_data)
    category = TOUR_SINGLES_CATEGORY[bracket.tour.lower()]
    tournament_matches = [
        m for m in espn_matches if m["tournament"] == bracket.tournament and m["category"] == category
    ]
    if not tournament_matches:
        print(
            f"ERROR: no live matches found for tournament {bracket.tournament!r} / {category} - "
            f"check the tournament name matches ESPN's exactly.",
            file=sys.stderr,
        )
        sys.exit(1)

    results_by_round, round_sequence, unresolved_names = build_real_results_by_round(
        tournament_matches, draw, tour_config.name_aliases
    )
    if unresolved_names:
        print(f"WARNING: {len(unresolved_names)} ESPN player name(s) could not be matched to the "
              f"draw (their matches are excluded from real results): {sorted(unresolved_names)}",
              file=sys.stderr)

    fields = replay_real_rounds(non_bye_players, bye_players, results_by_round)
    max_known_round = len(fields) - 1
    print(f"Round sequence observed: {round_sequence}")
    print(f"Real results fully known through round {max_known_round} of the main draw "
          f"(then simulated normally from there)")
    if max_known_round == 0:
        print("No fully-known real round available - nothing to replay yet.", file=sys.stderr)
        sys.exit(1)

    bracket_stem = args.bracket_path.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def run_snapshot(n):
        if not (0 <= n <= max_known_round):
            print(f"ERROR: --through-round {n} is out of range (0..{max_known_round} known)", file=sys.stderr)
            sys.exit(1)
        starting_field = fields[n]
        champion_counts = run_simulations_from_field(starting_field, bracket.surface, args.simulations, tour_config.ratings_path)
        output_path = args.output_dir / f"{bracket_stem}_through_round_{n}.csv"
        _report(f"Through round {n} ({len(starting_field)} players remaining)", draw, champion_counts, args.simulations, output_path)

    if args.all_rounds:
        for n in range(1, max_known_round + 1):
            run_snapshot(n)
    else:
        run_snapshot(args.through_round)


if __name__ == "__main__":
    main()
