"""Backtests the model's pre-tournament predictions against the real, final outcomes of the two
hard-court US Open tune-up events: Montreal (ATP) and Toronto/National Bank Open (WTA). Both are
now fully concluded, so every round has a known real result.

Reuses hybrid_simulation.py's own real-result parsing (build_real_results_by_round,
replay_real_rounds) and round-snapshot simulation (run_simulations_from_field, _report) rather
than re-deriving match outcomes or win probabilities from scratch - this script only adds the
before/after comparison on top of that existing machinery.

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
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket import (  # noqa: E402
    TOUR_CONFIG, match_draw_to_ratings, order_by_draw_position, split_byes, validate_bracket_structure, validate_draw,
)
from bracket_schema import BracketValidationError, load_bracket_yaml  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from hybrid_simulation import (  # noqa: E402
    TOUR_SINGLES_CATEGORY, _report, build_known_pairings_by_round, build_real_results_by_round,
    reconstruct_leaves_by_round2_slot, replay_real_rounds, true_bracket_order,
)
from live_scores import LiveScoresError, extract_matches, fetch_scoreboard  # noqa: E402
from simulate import N_SIMULATIONS, run_simulations_from_field  # noqa: E402
from win_probability import win_probability  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
SEED = 42

# (bracket path, pre-tournament full-simulation CSV already sitting in output/, ESPN event id) -
# both hard-court US Open tune-ups. Matched by event id, not tournament name: ESPN bundles both
# draws under one event ("National Bank Open presented by Rogers", id 421-2026) even though the
# men's draw (Montreal) and women's draw (Toronto) are different cities - the bracket YAML's
# "Montreal Open" name has no match in ESPN's feed at all, so name-matching silently finds zero
# real results for it. See live_scores.py's --event-id option, added for the same reason.
NATIONAL_BANK_OPEN_EVENT_ID = "421-2026"
TOURNAMENTS = [
    (Path("brackets/montreal_2026.yaml"), OUTPUT_DIR / "montreal_open_2026_simulation_results_atp.csv", NATIONAL_BANK_OPEN_EVENT_ID),
    (Path("brackets/wta_toronto_2026.yaml"), OUTPUT_DIR / "national_bank_open_presented_by_rogers_2026_simulation_results_wta.csv", NATIONAL_BANK_OPEN_EVENT_ID),
]


def analyze_tournament(bracket_path, pretournament_csv_path, event_id, n_simulations=N_SIMULATIONS):
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

    espn_data = fetch_scoreboard(bracket.tour.lower())
    espn_matches, _ = extract_matches(espn_data)
    category = TOUR_SINGLES_CATEGORY[bracket.tour.lower()]
    tournament_matches = [
        m for m in espn_matches if m["event_id"] == event_id and m["category"] == category
    ]
    if not tournament_matches:
        raise RuntimeError(f"No live matches found for event_id={event_id!r} / {category}")

    results_by_round, round_sequence, _unresolved = build_real_results_by_round(
        tournament_matches, draw, tour_config.name_aliases
    )
    # known_pairings_by_round is required for replay_real_rounds to advance past Round 1 at all -
    # Round 2+'s matchup structure has no other source (see known_matchups_for_round's docstring
    # in hybrid_simulation.py). Omitting it silently truncates replay to Round 1 regardless of how
    # much of the real tournament actually finished - confirmed the hard way: this call used to
    # omit it and produced "Rounds fully known ... 1 of 7" for an already-fully-concluded event.
    known_pairings_by_round, _round_sequence_2, _unresolved_2 = build_known_pairings_by_round(
        tournament_matches, draw, tour_config.name_aliases
    )
    fields = replay_real_rounds(non_bye_players, bye_players, results_by_round, known_pairings_by_round)
    max_known_round = len(fields) - 1
    if max_known_round == 0:
        raise RuntimeError(f"No fully-known real round for {bracket_path} yet")

    # fields[n] (n>=1) is only reliable for *which* players remain - its own order isn't real
    # bracket adjacency (byes get merged in as "round winners + byes" appended in bulk, not
    # interleaved at their true position - see reconstruct_leaves_by_round2_slot's docstring).
    # run_simulations_from_field below plays every subsequent round via plain positional pairing,
    # so it needs fields[n] resorted into TRUE bracket order first, or it simulates rounds 2+
    # against a bracket structure that doesn't match the real draw.
    leaves_by_slot = reconstruct_leaves_by_round2_slot(tournament_matches, non_bye_players, bye_players)
    leaf_position = {name: i for i, (name, _is_bye) in enumerate(true_bracket_order(leaves_by_slot))}
    fields = [
        sorted(field, key=lambda name: leaf_position.get(name, len(leaf_position))) for field in fields
    ]

    # elimination round per player, from real results only (1-indexed)
    eliminated_in_round = {}
    for n in range(1, max_known_round + 1):
        for player in fields[n - 1]:
            if player not in fields[n]:
                eliminated_in_round[player] = n
    finalists = fields[max_known_round]
    champion = finalists[0] if len(finalists) == 1 else None

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

    # per-match calibration: for every real match in a fully-known round, did the model's
    # pre-match favorite (surface Elo, via the same win_probability() every simulated match uses)
    # actually win it?
    match_rows = []
    for n in range(1, max_known_round + 1):
        for pair, winner in results_by_round.get(n, {}).items():
            a, b = tuple(pair)
            prob_a = win_probability(a, b, bracket.surface, tour_config.ratings_path)
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
        print(f"Not yet fully concluded in the live feed - {len(result['finalists'])} player(s) "
              f"still alive heading into {_round_label(result, result['max_known_round'] + 1)}: "
              f"{result['finalists']}")

    print("\n--- Pre-tournament top 8 favorites vs. actual result (so far) ---")
    pretournament = result["snapshots"][0].sort_values(ascending=False)
    rows = [
        {"rank": rank, "player": player, "pretournament_win_prob": round(prob, 4),
         "actual_result": outcome_label(player, result)}
        for rank, (player, prob) in enumerate(pretournament.head(8).items(), start=1)
    ]
    print(pd.DataFrame(rows).to_string(index=False))

    tracked = [result["champion"]] if result["champion"] is not None else result["finalists"]
    label = "the ACTUAL champion" if result["champion"] is not None else "each still-alive finalist"
    print(f"\n--- Model's win probability for {label}, by checkpoint ---")
    print(f"(tracking: {tracked})")
    traj_rows = []
    for n in range(0, result["max_known_round"] + 1):
        row = {"checkpoint": "Pre-tournament" if n == 0 else f"After {_round_label(result, n)}"}
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
    for bracket_path, pretournament_csv, event_id in TOURNAMENTS:
        try:
            result = analyze_tournament(bracket_path, pretournament_csv, event_id)
        except (BracketValidationError, LiveScoresError, RuntimeError, FileNotFoundError) as e:
            print(f"ERROR analyzing {bracket_path}: {e}", file=sys.stderr)
            continue
        b = result["bracket"]
        print_report(f"{b.tournament} {b.year} ({b.tour}, {b.surface})", result)
