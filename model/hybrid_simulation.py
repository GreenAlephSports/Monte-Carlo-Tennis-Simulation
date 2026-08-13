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

A round doesn't have to be fully *decided* to be usable, though - only its matchup structure has
to be fully *known*, which is a genuinely weaker, separate condition. Round 1's pairing is always
known (fixed by the bracket itself, via get_matchups - never depends on any match actually being
played). Round 2+'s pairing is only known once ESPN has published real names on both sides of
every match in that round - which ESPN does well before those matches are played, since it
returns every round's competitions upfront and only uses 'TBD' for a slot still waiting on an
earlier round's real outcome (see build_known_pairings_by_round). Whichever round comes right
after the last fully-decided one, if its matchups are fully known, can be partially replayed:
whichever of its pairings already have a real winner are pinned, and the rest are Monte
Carlo-simulated like any other match. `--through-round N` falls back to this automatically for
that one round when it isn't fully decided yet but has at least one final result and a fully
known matchup set.

Usage:
    python model/hybrid_simulation.py brackets/wta_toronto_2026.yaml --through-round 3
    python model/hybrid_simulation.py brackets/wta_toronto_2026.yaml --all-rounds
"""
import argparse
import random
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
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from live_scores import LiveScoresError, extract_matches, fetch_scoreboard  # noqa: E402
from simulate import N_SIMULATIONS, run_simulations_from_field, run_simulations_partial_round  # noqa: E402

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


def _round_index(espn_matches):
    """Shared by build_real_results_by_round and build_known_pairings_by_round so both agree on
    the same round numbering - both derive round_sequence from the identical input list (not
    filtered by match status first), so calling this once per (tournament, category) match list
    and reusing the result, or just calling both functions with that same list, keeps them
    consistent automatically."""
    round_labels = {m["round"] for m in espn_matches if m["round"]}
    round_sequence = build_round_sequence(round_labels)
    return round_sequence, {label: i + 1 for i, label in enumerate(round_sequence)}


def build_real_results_by_round(espn_matches, draw_csv_names, name_aliases=None):
    """Returns (results_by_round, round_sequence, unresolved_names). results_by_round[n] maps
    frozenset({player_a, player_b}) -> winner, for round n. Only completed matches with a clear
    winner and both players resolvable to the draw contribute a result - a match that hasn't
    been played yet, even with two real (non-'TBD') names, contributes nothing here; see
    build_known_pairings_by_round for that weaker, separate signal."""
    round_sequence, round_index = _round_index(espn_matches)

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


def build_known_pairings_by_round(espn_matches, draw_csv_names, name_aliases=None):
    """Returns (known_pairings_by_round, round_sequence, unresolved_names). known_pairings_by_round[n]
    is a set of frozenset({player_a, player_b}) for round n - every match where BOTH players are
    already resolvable to the draw (i.e. neither side is still 'TBD'), regardless of whether the
    match has actually been played yet. A genuinely separate, weaker signal than
    build_real_results_by_round's "decided" results: ESPN returns every round's competitions
    upfront, using 'TBD' only for a slot still waiting on an earlier round's real outcome (see
    espn_bracket.py's own docstring) - so a round's matchup structure is often fully knowable well
    before any of its matches conclude."""
    round_sequence, round_index = _round_index(espn_matches)

    known_pairings_by_round = defaultdict(set)
    unresolved_names = set()
    for m in espn_matches:
        round_num = round_index.get(m["round"])
        if round_num is None:
            continue
        p1_name, p2_name = m["player_1"], m["player_2"]
        if not p1_name or not p2_name or p1_name == "TBD" or p2_name == "TBD":
            continue

        p1 = match_espn_name_to_draw(p1_name, draw_csv_names, name_aliases)
        p2 = match_espn_name_to_draw(p2_name, draw_csv_names, name_aliases)
        if p1 is None:
            unresolved_names.add(p1_name)
        if p2 is None:
            unresolved_names.add(p2_name)
        if p1 is None or p2 is None:
            continue

        known_pairings_by_round[round_num].add(frozenset((p1, p2)))

    return known_pairings_by_round, round_sequence, unresolved_names


def known_matchups_for_round(round_num, current_field, known_pairings_by_round):
    """Returns round_num's full matchup list if it's completely determined, else None.

    Round 1's matchups are always determined - get_matchups(current_field) reconstructs the real
    bracket pairing directly from the draw itself, independent of anything ESPN reports (a bye's
    phantom opponent never sits between two real non-bye players). Round 2+ has no such
    independent source: the "winners + byes" concatenation that forms a simulated field does NOT
    reconstruct true bracket-tree adjacency, so round 2+'s pairing can only come from ESPN's own
    known_pairings_by_round - and only counts as determined if those known pairings account for
    every player in current_field exactly once (anything less means some slot is still 'TBD',
    waiting on a match this function doesn't have visibility into)."""
    if round_num == 1:
        return get_matchups(current_field)

    known_pairs = known_pairings_by_round.get(round_num, set())
    players_in_known_pairs = [p for pair in known_pairs for p in pair]
    if set(players_in_known_pairs) != set(current_field) or len(players_in_known_pairs) != len(current_field):
        return None
    return [tuple(pair) for pair in known_pairs]


def replay_real_rounds(non_bye_players, bye_players, results_by_round, known_pairings_by_round=None):
    """Replays as many rounds as have a COMPLETE set of real results, starting at round 1 and
    stopping at the first round whose matchup structure isn't fully known (known_matchups_for_round
    returns None) or that has any unresolved matchup within a known structure. Returns a list
    `fields` where fields[n] is the real field entering round n+1 (fields[0] = pre-Round-1
    non-bye field); len(fields) - 1 is the highest round fully known (matchups AND results) from
    real results.

    known_pairings_by_round is optional (round 1 never needs it - see known_matchups_for_round);
    omitting it just means round 2+ can never be treated as known here, matching this function's
    original round-1-only behavior."""
    known_pairings_by_round = known_pairings_by_round or {}
    fields = [list(non_bye_players)]
    current_field = list(non_bye_players)
    round_num = 1
    while len(current_field) > 1:
        matchups = known_matchups_for_round(round_num, current_field, known_pairings_by_round)
        if matchups is None:
            return fields

        round_results = results_by_round.get(round_num, {})
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible Monte Carlo results")
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
    matches_history = load_matches_for_tour(bracket.tour)
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
    known_pairings_by_round, _round_sequence_2, unresolved_names_2 = build_known_pairings_by_round(
        tournament_matches, draw, tour_config.name_aliases
    )
    unresolved_names = unresolved_names | unresolved_names_2
    if unresolved_names:
        print(f"WARNING: {len(unresolved_names)} ESPN player name(s) could not be matched to the "
              f"draw (their matches are excluded from real results): {sorted(unresolved_names)}",
              file=sys.stderr)

    fields = replay_real_rounds(non_bye_players, bye_players, results_by_round, known_pairings_by_round)
    max_known_round = len(fields) - 1
    print(f"Round sequence observed: {round_sequence}")
    print(f"Real results fully known through round {max_known_round} of the main draw "
          f"(then simulated normally from there)")

    # the round right after the last fully-decided one may still be partially replayable: its
    # matchup structure might already be fully known (round 1, always; round 2+, only if ESPN
    # has published real names for every pairing) even though not every match in it has been
    # played yet. Not a round-1 special case - round 1 is just the instance of this that's
    # always available, since its matchups never depend on ESPN reporting anything.
    target_round = max_known_round + 1
    partial_field = fields[max_known_round]
    partial_matchups = known_matchups_for_round(target_round, partial_field, known_pairings_by_round)
    partial_known_results = {}
    if partial_matchups is not None:
        round_results = results_by_round.get(target_round, {})
        partial_known_results = {
            frozenset(pair): round_results[frozenset(pair)]
            for pair in partial_matchups if frozenset(pair) in round_results
        }

    if partial_matchups is not None and partial_known_results:
        print(f"Round {target_round} matchups are fully known ({len(partial_field)} players) - "
              f"{len(partial_known_results)}/{len(partial_matchups)} already final - "
              f"--through-round {target_round} will pin those and simulate the rest.")
    elif max_known_round == 0:
        print("No fully-known real round available, and the next round's matchups and/or "
              "results aren't usable yet either - nothing to replay yet.", file=sys.stderr)
        sys.exit(1)

    bracket_stem = args.bracket_path.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def run_snapshot(n):
        partial = n == target_round and partial_matchups is not None and partial_known_results
        if not partial and not (0 <= n <= max_known_round):
            print(f"ERROR: --through-round {n} is out of range (0..{max_known_round} known)", file=sys.stderr)
            sys.exit(1)
        # reseeded per-round (not once for the whole run) so a standalone --through-round N
        # always reproduces exactly the same snapshot as round N within an --all-rounds run -
        # neither draws from wherever the global random stream happened to be left by whatever
        # ran before it.
        random.seed(args.seed + n)
        if partial:
            # byes only join in right after round 1 - for any later partial round they're
            # already folded into partial_field (fields[max_known_round]).
            extra_after = bye_players if n == 1 else []
            champion_counts = run_simulations_partial_round(
                partial_field, extra_after, partial_known_results, bracket.surface,
                args.simulations, tour_config.ratings_path,
            )
            output_path = args.output_dir / f"{bracket_stem}_through_round_{n}_partial.csv"
            _report(
                f"Through round {n} - PARTIAL ({len(partial_known_results)}/{len(partial_matchups)} "
                f"round-{n} matches final, rest of round {n} and beyond simulated)",
                draw, champion_counts, args.simulations, output_path,
            )
            return
        starting_field = fields[n]
        champion_counts = run_simulations_from_field(starting_field, bracket.surface, args.simulations, tour_config.ratings_path)
        output_path = args.output_dir / f"{bracket_stem}_through_round_{n}.csv"
        _report(f"Through round {n} ({len(starting_field)} players remaining)", draw, champion_counts, args.simulations, output_path)

    if args.all_rounds:
        for n in range(1, max_known_round + 1):
            run_snapshot(n)
        if partial_matchups is not None and partial_known_results:
            run_snapshot(target_round)
    else:
        run_snapshot(args.through_round)


if __name__ == "__main__":
    main()
