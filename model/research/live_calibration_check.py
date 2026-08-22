"""Adapts backtest_hard_court.py's per-match calibration methodology to Cincinnati, which is
still IN PROGRESS - so unlike Montreal/Toronto (fully concluded, real results read from the
post-hoc Kaggle dataset), this reads real results from live ESPN state instead, via the same
build_real_results_by_round machinery bracket_export.py itself uses.

Same core question as backtest_hard_court.py's calibration section, same methodology: for every
real, DECIDED match so far, was the model's PRE-TOURNAMENT frozen-Elo favorite (win_probability()
against the ratings snapshot cut off at the bracket's start_date - zero real Cincinnati results
baked in) actually the one who won? This is honest pre-match calibration, not hindsight - Elo is
never updated with in-tournament results before being asked to "predict" them.

Deliberately does NOT reuse backtest_hard_court.py's snapshot-trajectory checkpoints (pre-
tournament / after-each-round win probabilities) - those need reconstruct_leaves_by_round2_slot's
true-bracket-adjacency logic for a still-in-progress event, which is a bigger lift this question
doesn't need. The per-match calibration table alone answers exactly what was asked: does assigned
confidence line up with the real outcome, at whatever sample size actually exists right now.

Usage:
    python model/research/live_calibration_check.py
"""
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import (  # noqa: E402
    TOUR_CONFIG, match_draw_to_ratings, order_by_draw_position, split_byes, validate_bracket_structure,
    validate_draw,
)
from bracket_export import resolve_qualifier_placeholders  # noqa: E402
from bracket_schema import BracketValidationError, load_bracket_yaml  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from hybrid_simulation import TOUR_SINGLES_CATEGORY, build_real_results_by_round  # noqa: E402
from live_scores import LiveScoresError, extract_matches, fetch_scoreboard  # noqa: E402
from win_probability import win_probability  # noqa: E402

MIN_SAMPLE_FOR_ANY_CONFIDENCE = 30  # below this, a CI is printed but flagged as barely informative

TOURNAMENTS = [
    Path("brackets/cincinnati_2026_atp_demo.yaml"),
    Path("brackets/cincinnati_2026_wta.yaml"),
]


def analyze_live_tournament(bracket_path):
    bracket = load_bracket_yaml(bracket_path)
    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    validate_bracket_structure(byes)

    tour_config = TOUR_CONFIG[bracket.tour]
    matches_history = load_matches_for_tour(bracket.tour)
    ratings_df = calculate_elo_ratings(matches_history, bracket.start_date)
    ratings_df = ratings_df.sort_values("overall_elo", ascending=False).reset_index(drop=True)

    espn_data = fetch_scoreboard(bracket.tour.lower())
    espn_matches, _ = extract_matches(espn_data)
    category = TOUR_SINGLES_CATEGORY[bracket.tour.lower()]
    tournament_matches = [
        m for m in espn_matches if m["tournament"] == bracket.tournament and m["category"] == category
    ]
    if not tournament_matches:
        raise RuntimeError(f"No live matches found for {bracket.tournament!r} / {category}")

    # same qualifier-placeholder resolution bracket_export.py runs - must happen before
    # match_draw_to_ratings, same reason as there (see that function's own docstring).
    players, _qualifier_warnings = resolve_qualifier_placeholders(
        players, byes, tournament_matches, ratings_df, tour_config.name_aliases
    )

    draw, resolutions, ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date
    )
    unmatched = [r for r in resolutions if r["tier"] is None]
    if unmatched:
        raise RuntimeError(f"Unmatched bracket names for {bracket_path}: {[r['name'] for r in unmatched]}")

    # PRE-TOURNAMENT ratings, frozen - written now, before any real Cincinnati result is
    # incorporated, so win_probability() below reflects the model's genuine pre-match prediction,
    # not a rating that has already absorbed the very outcome it's being asked to "predict".
    tour_config.ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(tour_config.ratings_path, index=False)

    validate_draw(draw)
    split_byes(draw, byes)  # structural check only (byes never appear as a real-match participant)

    results_by_round, round_sequence, unresolved = build_real_results_by_round(
        tournament_matches, draw, tour_config.name_aliases
    )
    if unresolved:
        print(f"WARNING: {len(unresolved)} ESPN player name(s) could not be matched to the draw "
              f"(their matches are excluded from calibration): {sorted(unresolved)}", file=sys.stderr)

    match_rows = []
    for n in sorted(results_by_round):
        round_label = round_sequence[n - 1] if n - 1 < len(round_sequence) else f"Round {n}"
        for pair, winner in results_by_round[n].items():
            a, b = tuple(pair)
            prob_a = win_probability(a, b, bracket.surface, tour_config.ratings_path)
            favorite = a if prob_a >= 0.5 else b
            match_rows.append({
                "round": n, "round_label": round_label, "player_a": a, "player_b": b,
                "favorite": favorite, "favorite_prob": max(prob_a, 1 - prob_a),
                "winner": winner, "favorite_won": favorite == winner,
            })

    return bracket, pd.DataFrame(match_rows), round_sequence


def _wilson_ci(n_success, n, z=1.96):
    """Wilson score interval - unlike the normal approximation used elsewhere in this codebase's
    larger-sample tests, this stays well-behaved (never goes below 0 or above 1, doesn't
    degenerate at p near 0 or 1) at the small sample sizes an in-progress tournament actually has
    right now, which is the case this function exists for."""
    if n == 0:
        return float("nan"), float("nan")
    p = n_success / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half_width = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def print_report(name, calib, round_sequence=None):
    print(f"\n{'=' * 90}\n{name}\n{'=' * 90}")
    if not len(calib):
        print("No real, decided matches yet - nothing to calibrate against.")
        return

    n = len(calib)
    n_favorite_won = int(calib["favorite_won"].sum())
    win_rate = n_favorite_won / n
    avg_prob = calib["favorite_prob"].mean()
    lo, hi = _wilson_ci(n_favorite_won, n)

    if round_sequence:
        print(f"{n} real matches decided so far, across {calib['round'].nunique()} of "
              f"{len(round_sequence)} rounds ({round_sequence})")
        by_round = calib.groupby("round_label", sort=False).agg(
            matches=("favorite_won", "size"),
            avg_favorite_prob=("favorite_prob", "mean"),
            favorite_win_rate=("favorite_won", "mean"),
        )
        by_round = by_round.reindex([r for r in round_sequence if r in by_round.index])
        print(by_round.to_string(formatters={
            "avg_favorite_prob": "{:.1%}".format, "favorite_win_rate": "{:.1%}".format,
        }))
    else:
        print(f"{n} real matches decided so far (combined across tournaments)")

    print(f"\nOverall: model's favorite actually won {win_rate:.1%} of {n} real matches "
          f"(model's average assigned favorite probability: {avg_prob:.1%})")
    print(f"95% Wilson CI on the actual favorite-win-rate: [{lo:.1%}, {hi:.1%}]")

    if n < MIN_SAMPLE_FOR_ANY_CONFIDENCE:
        print(f"\nCAUTION: n={n} is well below the ~{MIN_SAMPLE_FOR_ANY_CONFIDENCE}-match floor "
              f"this codebase's other calibration checks require before treating a rate as "
              f"meaningful (see weather_upset_test.py's own MIN_COVERAGE_FOR_TEST gate). At this "
              f"size a single early upset or chalk result swings the headline number by 10+ "
              f"points, and the CI above is correspondingly wide - this is an early, honest read "
              f"of an in-progress tournament, not a settled calibration verdict on the same footing "
              f"as the fully-concluded Montreal (94 matches) / Toronto (94 matches) backtests.")


if __name__ == "__main__":
    all_calib = []
    for bracket_path in TOURNAMENTS:
        try:
            bracket, calib, round_sequence = analyze_live_tournament(bracket_path)
        except (BracketValidationError, RuntimeError, LiveScoresError) as e:
            print(f"ERROR analyzing {bracket_path}: {e}", file=sys.stderr)
            continue
        print_report(f"{bracket.tournament} {bracket.year} ({bracket.tour}, {bracket.surface})", calib, round_sequence)
        all_calib.append(calib)

    if len(all_calib) > 1 and sum(len(c) for c in all_calib) > 0:
        combined = pd.concat(all_calib, ignore_index=True)
        print_report(f"COMBINED (ATP + WTA, {len(all_calib)} tournaments)", combined)
