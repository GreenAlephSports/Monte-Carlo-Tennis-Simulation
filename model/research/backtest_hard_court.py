"""Backtests the model's pre-tournament predictions against the real, final outcomes of the two
hard-court US Open tune-up events: Montreal (ATP) and Toronto/National Bank Open (WTA). Both are
now fully concluded, so every round has a known real result.

Real results come from the same auto-updating Kaggle match-history dataset used for Elo (see
build_real_results_from_kaggle below), not a live ESPN call - both tournaments are fully
concluded, and Kaggle's per-round match rows already give every result directly, with no need for
ESPN's live-feed-specific "pairing known but not yet played" handling. Reuses hybrid_simulation.
py's own round replay (replay_real_rounds) and round-snapshot simulation (run_simulations_from_
field, _report) rather than re-deriving match outcomes or win probabilities from scratch - this
script only adds the before/after comparison on top of that existing machinery.

"Pre-tournament prediction" = the plain full-bracket simulation already sitting in output/ (the
same file any live run_tournament.py call for that bracket produces) - pre-cutoff Elo, zero real
results incorporated. Round-by-round snapshots are regenerated fresh via the current (reseed-
per-round) hybrid_simulation logic for internal consistency, rather than reusing whichever older
on-disk snapshot files happen to already exist.

Usage:
    python model/backtest_hard_court.py
"""
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import (  # noqa: E402
    TOUR_CONFIG, match_draw_to_ratings, match_name_to_pool, order_by_draw_position, split_byes,
    validate_bracket_structure, validate_draw,
)
from bracket_schema import BracketValidationError, load_bracket_yaml  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from hybrid_simulation import (  # noqa: E402
    _report, build_round_sequence, known_matchups_for_round, replay_real_rounds,
)
from simulate import N_SIMULATIONS, run_simulations_from_field, run_simulations_partial_round  # noqa: E402
from win_probability import win_probability  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
SEED = 42

# Kaggle's Round labels for these datasets ('1st Round', 'Quarterfinals', 'The Final', ...) don't
# match hybrid_simulation.build_round_sequence's expected vocabulary ('Round 1', 'Quarterfinal',
# 'Final', ...) - this translates before reusing that same ordering logic, so both the live-ESPN
# path (hybrid_simulation.py) and this static-Kaggle path agree on round numbering.
KAGGLE_ROUND_LABELS = {
    "1st Round": "Round 1", "2nd Round": "Round 2", "3rd Round": "Round 3", "4th Round": "Round 4",
    "5th Round": "Round 5", "Quarterfinals": "Quarterfinal", "Semifinals": "Semifinal", "The Final": "Final",
}

# (bracket path, pre-tournament full-simulation CSV already sitting in output/, Kaggle tournament
# name) - both hard-court US Open tune-ups. The 2026 Kaggle ATP/WTA match-history datasets (see
# data_loader_kaggle.py) already carry complete, round-labeled results for both events under the
# name "Canadian Open" (their ATP/WTA feeds are separate files, so this name alone disambiguates
# Montreal (ATP) from Toronto (WTA) without needing an event id) - both tournaments are fully
# concluded, and Kaggle's auto-updating pull already has every round through the final. Matched by
# tournament name + a date window around the bracket's start_date, same disambiguation need the old
# ESPN event-id lookup had (the bracket YAML's "Montreal Open" name has no match in either feed).
KAGGLE_TOURNAMENT_NAME = "Canadian Open"
TOURNAMENTS = [
    (Path("brackets/montreal_2026.yaml"), OUTPUT_DIR / "montreal_open_2026_simulation_results_atp.csv", KAGGLE_TOURNAMENT_NAME),
    (Path("brackets/wta_toronto_2026.yaml"), OUTPUT_DIR / "national_bank_open_presented_by_rogers_2026_simulation_results_wta.csv", KAGGLE_TOURNAMENT_NAME),
]


def build_real_results_from_kaggle(matches_df, kaggle_tournament_name, start_date, draw_csv_names, name_aliases=None):
    """Kaggle-sourced equivalent of hybrid_simulation's build_real_results_by_round +
    build_known_pairings_by_round combined, for an already-fully-concluded tournament: every real
    match row already carries its final Winner, so - unlike the ESPN live-feed case those two
    functions were built for - there's no "pairing known but not yet played" distinction to make;
    a round's known pairings and its decided results are identical.

    Player_1/Player_2/Winner are already in ratings-csv "Lastname I." shape (this dataframe IS the
    same live-pulled match history match_draw_to_ratings resolves the draw against), but Kaggle's
    own name spelling isn't always internally consistent with the draw's resolved names: compound
    surnames sometimes disagree on whether to keep the internal space ("De Minaur A." vs.
    "Deminaur A."), Kaggle itself inconsistently truncates a compound surname to only one of its
    words across different match rows ("Mpetshi G." vs. "Mpetshi Perricard G."), and a name can
    carry the same kind of PDF-extraction/manual-alias mismatch the draw's own resolution already
    handles elsewhere ("Landaluce M." vs. the draw's "Andaluce M."). Resolved via bracket.py's
    match_name_to_pool - the same tiered fuzzy-matching system (manual alias, exact lastname+
    initials, first-initial-unique, glued-lastname prefix/suffix) match_draw_to_ratings already
    uses to match bracket names against the Elo ratings csv - so a name that's genuinely not in
    the draw (e.g. a withdrawal replaced by a lucky loser after the bracket YAML was written)
    still correctly stays unresolved rather than being guessed at.

    Returns (results_by_round, known_pairings_by_round, round_sequence, unresolved_names)."""
    window = matches_df[
        (matches_df["Tournament"] == kaggle_tournament_name)
        & (matches_df["Date"] >= start_date - pd.Timedelta(days=2))
        & (matches_df["Date"] < start_date + pd.Timedelta(days=21))
    ]

    round_labels = {KAGGLE_ROUND_LABELS[r] for r in window["Round"].unique() if r in KAGGLE_ROUND_LABELS}
    round_sequence = build_round_sequence(round_labels)
    round_index = {label: i + 1 for i, label in enumerate(round_sequence)}

    draw_csv_names = list(draw_csv_names)
    resolved_cache = {}

    def resolve(kaggle_name):
        if kaggle_name not in resolved_cache:
            resolved_cache[kaggle_name] = match_name_to_pool(kaggle_name, draw_csv_names, name_aliases)
        return resolved_cache[kaggle_name]

    results_by_round = defaultdict(dict)
    known_pairings_by_round = defaultdict(set)
    unresolved_names = set()
    for row in window.itertuples():
        round_num = round_index.get(KAGGLE_ROUND_LABELS.get(row.Round))
        if round_num is None:
            continue
        p1, p2 = resolve(row.Player_1), resolve(row.Player_2)
        winner = resolve(row.Winner)
        if p1 is None:
            unresolved_names.add(row.Player_1)
        if p2 is None:
            unresolved_names.add(row.Player_2)
        if p1 is None or p2 is None:
            continue
        pair = frozenset((p1, p2))
        known_pairings_by_round[round_num].add(pair)
        if winner in (p1, p2):
            results_by_round[round_num][pair] = winner

    return results_by_round, known_pairings_by_round, round_sequence, unresolved_names


def analyze_tournament(bracket_path, pretournament_csv_path, kaggle_tournament_name, n_simulations=N_SIMULATIONS,
                        use_rank_adjustment=True, use_confidence_calibration=True, use_layoff_adjustment=True):
    bracket = load_bracket_yaml(bracket_path)
    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    validate_bracket_structure(byes)

    tour_config = TOUR_CONFIG[bracket.tour]
    matches_history = load_matches_for_tour(bracket.tour)
    ratings_df = calculate_elo_ratings(matches_history, bracket.start_date)
    ratings_df = ratings_df.sort_values("overall_elo", ascending=False).reset_index(drop=True)

    draw, resolutions, ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date
    )
    unmatched = [r for r in resolutions if r["tier"] is None]
    if unmatched:
        raise RuntimeError(f"Unmatched bracket names for {bracket_path}: {[r['name'] for r in unmatched]}")

    # win_probability()/run_simulations_from_field read Elo from this file, not from ratings_df in
    # memory - save it now so every lookup below uses THIS bracket's cutoff, not whatever tour/
    # cutoff last happened to write here (mirrors run_tournament.py's own save step).
    tour_config.ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(tour_config.ratings_path, index=False)

    validate_draw(draw)
    non_bye_players, bye_players = split_byes(draw, byes)

    # No live ESPN call needed here: both tournaments are fully concluded, and Kaggle's
    # auto-updating dataset (already loaded above as matches_history, for Elo) already carries
    # every round's real results. build_real_results_from_kaggle gives results_by_round and
    # known_pairings_by_round directly from it - see that function's docstring for why a fully-
    # concluded event needs no "known but not yet played" distinction the way the live ESPN path
    # (hybrid_simulation.py, used for in-progress tournaments) does.
    results_by_round, known_pairings_by_round, round_sequence, unresolved = build_real_results_from_kaggle(
        matches_history, kaggle_tournament_name, bracket.start_date, draw, tour_config.name_aliases
    )
    if unresolved:
        print(f"WARNING: {len(unresolved)} Kaggle player name(s) could not be matched to the "
              f"draw (their matches are excluded from real results): {sorted(unresolved)}", file=sys.stderr)

    fields = replay_real_rounds(non_bye_players, bye_players, results_by_round, known_pairings_by_round)
    max_known_round = len(fields) - 1
    if not results_by_round:
        raise RuntimeError(f"No real results found for {bracket_path} yet")

    # fields[n] (n>=1)'s order isn't real bracket adjacency (byes get merged in as "round winners +
    # byes" appended in bulk, not interleaved at their true position) - normally this would need
    # reconstructing before run_simulations_from_field below plays any further rounds via plain
    # positional pairing (see hybrid_simulation.reconstruct_leaves_by_round2_slot, which needs
    # ESPN's Round 1/2 structure to do that reconstruction and so isn't available on this all-
    # Kaggle path). It's a genuine no-op here, not a gap: both tournaments are fully concluded, so
    # fields[max_known_round] always has exactly 1 player (the real champion) - there are no
    # further rounds left for run_simulations_from_field to actually pair anyone in, so no bracket
    # order is ever consulted. This would need addressing before reusing this path on a
    # still-in-progress tournament.

    # elimination round per player, taken directly from every decided real match Kaggle has -
    # NOT from fields[] progression, so a data gap elsewhere (an earlier round's missing row, or
    # another match in the same round) never hides a result this function otherwise has: e.g. the
    # live Kaggle pull is missing a small number of Round 1 rows for both 2026 hard-court events
    # (a player simply has no recorded Round 1 match, though their later rounds ARE present) -
    # every OTHER real match in that same round still directly tells us who won and who didn't.
    eliminated_in_round = {}
    for n, round_results in results_by_round.items():
        for pair, winner in round_results.items():
            loser = next(p for p in pair if p != winner)
            eliminated_in_round[loser] = n

    final_round = len(round_sequence)
    final_results = results_by_round.get(final_round, {})
    champion = next(iter(final_results.values())) if len(final_results) == 1 else None
    all_players = set(non_bye_players) | set(bye_players)
    finalists = [champion] if champion is not None else sorted(p for p in all_players if p not in eliminated_in_round)

    # checkpoint 0 = pre-tournament prediction, the plain full-bracket simulation already sitting
    # in output/ - exactly what a live run_tournament.py call for this bracket produces
    pretournament = pd.read_csv(pretournament_csv_path).set_index("player")["tournament_win_probability"]
    snapshots = {0: pretournament}

    # checkpoints 1..max_known_round: reuse hybrid_simulation's own simulate+report path
    # (run_simulations_from_field + _report), one round at a time, same reseed-per-round scheme
    # hybrid_simulation.py itself now uses - regenerated fresh here for internal consistency
    # rather than reused from whatever's already on disk from an earlier/pre-fix run.
    bracket_stem = bracket_path.stem
    for n in range(1, max_known_round + 1):
        random.seed(SEED + n)
        champion_counts = run_simulations_from_field(fields[n], bracket.surface, n_simulations, tour_config.ratings_path)
        snapshot_path = OUTPUT_DIR / f"{bracket_stem}_through_round_{n}.csv"
        _report(f"{bracket.tournament} - through round {n}", draw, champion_counts, n_simulations, snapshot_path)
        snapshots[n] = pd.read_csv(snapshot_path).set_index("player")["tournament_win_probability"]

    # the round right after the last fully-decided one may still be partially replayable: its
    # matchup structure might already be fully known (round 1, always; round 2+, only if every
    # pairing is known) even though a handful of its results are the same kind of Kaggle data gap
    # eliminated_in_round works around above - same "pin what's decided, simulate the rest"
    # technique hybrid_simulation.py's live --through-round path already uses for an in-progress
    # round (run_simulations_partial_round), reused here for a gap instead of a not-yet-played match.
    target_round = max_known_round + 1
    if target_round <= len(round_sequence):
        partial_field = fields[max_known_round]
        partial_matchups = known_matchups_for_round(target_round, partial_field, known_pairings_by_round)
        if partial_matchups is not None:
            partial_known_results = {
                frozenset(pair): results_by_round[target_round][frozenset(pair)]
                for pair in partial_matchups if frozenset(pair) in results_by_round.get(target_round, {})
            }
            if partial_known_results:
                random.seed(SEED + target_round)
                extra_after = bye_players if target_round == 1 else []
                champion_counts = run_simulations_partial_round(
                    partial_field, extra_after, partial_known_results, bracket.surface,
                    n_simulations, tour_config.ratings_path,
                )
                snapshot_path = OUTPUT_DIR / f"{bracket_stem}_through_round_{target_round}_partial.csv"
                _report(
                    f"{bracket.tournament} - through round {target_round} (PARTIAL: "
                    f"{len(partial_known_results)}/{len(partial_matchups)} final)",
                    draw, champion_counts, n_simulations, snapshot_path,
                )
                snapshots[target_round] = pd.read_csv(snapshot_path).set_index("player")["tournament_win_probability"]

    # per-match calibration: for every real match Kaggle has a decided result for, did the model's
    # pre-match favorite (surface Elo, via the same win_probability() every simulated match uses)
    # actually win it? Uses every round with any real results, not just fully-known ones - a
    # match's outcome is knowable on its own regardless of whether its round completed cleanly.
    match_rows = []
    for n in sorted(results_by_round):
        for pair, winner in results_by_round[n].items():
            a, b = tuple(pair)
            prob_a = win_probability(
                a, b, bracket.surface, tour_config.ratings_path,
                use_rank_adjustment=use_rank_adjustment,
                use_confidence_calibration=use_confidence_calibration,
                use_layoff_adjustment=use_layoff_adjustment,
            )
            favorite = a if prob_a >= 0.5 else b
            match_rows.append({
                "round": n, "player_a": a, "player_b": b,
                "favorite": favorite, "favorite_prob": max(prob_a, 1 - prob_a),
                "winner": winner, "favorite_won": favorite == winner,
            })

    return {
        "bracket": bracket, "round_sequence": round_sequence, "max_known_round": max_known_round,
        "eliminated_in_round": eliminated_in_round, "champion": champion, "finalists": finalists,
        "snapshots": snapshots, "match_calibration": pd.DataFrame(match_rows),
    }


def _round_label(result, n):
    return result["round_sequence"][n - 1] if n - 1 <= len(result["round_sequence"]) - 1 else f"Round {n}"


def outcome_label(player, result):
    if player == result["champion"]:
        return "WON title"
    n = result["eliminated_in_round"].get(player)
    return f"lost {_round_label(result, n)}" if n is not None else "unresolved"


def print_report(name, result):
    print(f"\n{'=' * 90}\n{name}\n{'=' * 90}")
    print(f"Rounds fully known from real results: {result['max_known_round']} of "
          f"{len(result['round_sequence'])} ({result['round_sequence']})")
    if result["champion"] is not None:
        print(f"Champion (real): {result['champion']}")
    else:
        print(f"Champion not resolvable from real results ({len(result['finalists'])} player(s) "
              f"never recorded as eliminated): {result['finalists']}")

    print("\n--- Pre-tournament top 8 favorites vs. actual result (so far) ---")
    pretournament = result["snapshots"][0].sort_values(ascending=False)
    rows = [
        {"rank": rank, "player": player, "pretournament_win_prob": round(prob, 4),
         "actual_result": outcome_label(player, result)}
        for rank, (player, prob) in enumerate(pretournament.head(8).items(), start=1)
    ]
    print(pd.DataFrame(rows).to_string(index=False))

    tracked = [result["champion"]] if result["champion"] is not None else result["finalists"]
    label = "the ACTUAL champion" if result["champion"] is not None else "each unresolved finalist"
    print(f"\n--- Model's win probability for {label}, by checkpoint ---")
    print(f"(tracking: {tracked})")
    traj_rows = []
    for n in sorted(result["snapshots"]):
        partial_note = " (partial)" if n > result["max_known_round"] else ""
        row = {"checkpoint": "Pre-tournament" if n == 0 else f"After {_round_label(result, n)}{partial_note}"}
        for player in tracked:
            prob = result["snapshots"][n].get(player, float("nan"))
            row[player] = round(prob, 4) if pd.notna(prob) else None
        traj_rows.append(row)
    print(pd.DataFrame(traj_rows).to_string(index=False))

    calib = result["match_calibration"]
    if len(calib):
        print(f"\n--- Per-match calibration ({len(calib)} real matches across "
              f"{calib['round'].nunique()} known rounds) ---")
        by_round = calib.groupby("round").agg(
            matches=("favorite_won", "size"),
            avg_favorite_prob=("favorite_prob", "mean"),
            favorite_win_rate=("favorite_won", "mean"),
        ).reset_index()
        by_round["round_label"] = by_round["round"].apply(lambda n: _round_label(result, n))
        print(by_round[["round_label", "matches", "avg_favorite_prob", "favorite_win_rate"]].to_string(index=False))
        print(f"\nOverall: model's favorite actually won {calib['favorite_won'].mean():.1%} of "
              f"{len(calib)} real matches (model's average assigned favorite probability: "
              f"{calib['favorite_prob'].mean():.1%})")


if __name__ == "__main__":
    for bracket_path, pretournament_csv, kaggle_tournament_name in TOURNAMENTS:
        try:
            result = analyze_tournament(bracket_path, pretournament_csv, kaggle_tournament_name)
        except (BracketValidationError, RuntimeError, FileNotFoundError) as e:
            print(f"ERROR analyzing {bracket_path}: {e}", file=sys.stderr)
            continue
        b = result["bracket"]
        print_report(f"{b.tournament} {b.year} ({b.tour}, {b.surface})", result)
