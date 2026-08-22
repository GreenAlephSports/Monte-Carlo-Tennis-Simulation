"""Real head-to-head calibration test: model vs. market, not "who picks more winners" but who is
better CALIBRATED - lower log-loss/Brier against the real outcome wins. Favorite-win-rate (what
calibration_log.py reports) can't distinguish the model from the market at all, since whichever
one is closer to 50/50 on a given match will usually agree on who the favorite is; log-loss/Brier
penalize CONFIDENCE, not just direction, so this is the only fair side-by-side.

Then, critically, the same comparison sliced by |model_prob_a - market_prob_a| (disagreement size):
a model that's only competitive with the market on matches where they already agree isn't finding
anything - the only place a genuine edge could exist is where they meaningfully disagree. If the
model's relative log-loss advantage doesn't grow in the high-disagreement bucket, that's a real,
honest null result, not a reason to force one.

Market data availability, stated plainly: calibration_log.py's persisted log does NOT carry a
market probability at all (only favorite_prob, which is the model's own number) - this had to be
sourced fresh here, not read back from that log. Two sources exist in this project, with very
different coverage:
  - kaggle_concluded matches (Montreal/Toronto) carry real historical bookmaker odds directly in
    the Kaggle dataset itself (Odd_1/Odd_2, de-vigged the same way ev_comparison.py already does)
    - this script re-derives market probabilities from there, matching backtest_hard_court.py's
      own tournament/date windowing.
  - live_espn matches have NO recoverable market data: The Odds API doesn't serve historical
    odds through the endpoint this project uses, and calibration_log.py never captured
    market_prob_a at logging time (only as of this investigation). So every live_espn-sourced
    match in the log is EXCLUDED here, not silently defaulted to anything - reported explicitly.

Usage:
    python model/research/model_vs_market_calibration.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from backtest_hard_court import KAGGLE_ROUND_LABELS  # noqa: E402
from backtest_hard_court import TOURNAMENTS as CONCLUDED_TOURNAMENTS  # noqa: E402
from bracket import (  # noqa: E402
    TOUR_CONFIG, match_draw_to_ratings, match_name_to_pool, order_by_draw_position,
    validate_bracket_structure, validate_draw,
)
from bracket_schema import load_bracket_yaml  # noqa: E402
from elite_opponent_residual_test import log_loss  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from ev_comparison import implied_probabilities  # noqa: E402
from hybrid_simulation import build_round_sequence  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from win_probability import win_probability  # noqa: E402

DISAGREEMENT_BUCKETS = [
    ("<5pp", lambda d: d < 0.05),
    ("5-10pp", lambda d: 0.05 <= d < 0.10),
    ("10pp+", lambda d: d >= 0.10),
]


def _prepare_ratings(bracket):
    """Same setup calibration_log.py's _prepare_ratings does - recompute Elo up to the bracket's
    start_date, match the draw, write the ratings CSV win_probability() reads from."""
    tour_config = TOUR_CONFIG[bracket.tour]
    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    validate_bracket_structure(byes)

    matches_history = load_matches_for_tour(bracket.tour)
    ratings_df = calculate_elo_ratings(matches_history, bracket.start_date)
    ratings_df = ratings_df.sort_values("overall_elo", ascending=False).reset_index(drop=True)

    draw, resolutions, ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date
    )
    unmatched = [r for r in resolutions if r["tier"] is None]
    if unmatched:
        raise RuntimeError(f"Unmatched bracket names for {bracket.tournament}: {[r['name'] for r in unmatched]}")

    tour_config.ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(tour_config.ratings_path, index=False)

    validate_draw(draw)
    return tour_config, draw, matches_history


def collect_rows(bracket_path, kaggle_tournament_name):
    """One row per real, decided match with BOTH a model and a real market probability - mirrors
    calibration_log.collect_concluded_rows' windowing exactly, but additionally requires valid
    Odd_1/Odd_2 (the market side calibration_log.py never captures) and keeps player_a-perspective
    probabilities (not just favorite_prob) so log-loss/Brier can be computed consistently."""
    bracket = load_bracket_yaml(bracket_path)
    tour_config, draw, matches_history = _prepare_ratings(bracket)

    window = matches_history[
        (matches_history["Tournament"] == kaggle_tournament_name)
        & (matches_history["Date"] >= bracket.start_date - pd.Timedelta(days=2))
        & (matches_history["Date"] < bracket.start_date + pd.Timedelta(days=21))
    ]
    round_labels = {KAGGLE_ROUND_LABELS[r] for r in window["Round"].unique() if r in KAGGLE_ROUND_LABELS}
    round_sequence = build_round_sequence(round_labels)
    round_index = {label: i + 1 for i, label in enumerate(round_sequence)}

    resolved_cache = {}

    def resolve(name):
        if name not in resolved_cache:
            resolved_cache[name] = match_name_to_pool(name, draw, tour_config.name_aliases)
        return resolved_cache[name]

    rows, unresolved, no_odds = [], set(), 0
    for row in window.itertuples():
        round_label = KAGGLE_ROUND_LABELS.get(row.Round)
        if round_index.get(round_label) is None:
            continue
        p1, p2, winner = resolve(row.Player_1), resolve(row.Player_2), resolve(row.Winner)
        if p1 is None:
            unresolved.add(row.Player_1)
        if p2 is None:
            unresolved.add(row.Player_2)
        if p1 is None or p2 is None or winner not in (p1, p2):
            continue
        if pd.isna(row.Odd_1) or pd.isna(row.Odd_2) or row.Odd_1 <= 1 or row.Odd_2 <= 1:
            no_odds += 1
            continue

        model_prob_a = win_probability(p1, p2, bracket.surface, tour_config.ratings_path)
        market_prob_a, _market_prob_b = implied_probabilities(row.Odd_1, row.Odd_2)
        rows.append({
            "tour": bracket.tour, "tournament": bracket.tournament, "round": round_label,
            "player_a": p1, "player_b": p2, "actual_a_win": int(winner == p1),
            "model_prob_a": model_prob_a, "market_prob_a": market_prob_a,
        })

    if unresolved:
        print(f"WARNING: {len(unresolved)} Kaggle player name(s) unresolved for {bracket.tournament} - "
              f"excluded: {sorted(unresolved)}", file=sys.stderr)
    if no_odds:
        print(f"  {bracket.tournament}: {no_odds} real match(es) had no usable Odd_1/Odd_2 in the "
              f"Kaggle data - excluded (not defaulted to anything)", file=sys.stderr)
    return rows


def disagreement_bucket(d):
    for name, test in DISAGREEMENT_BUCKETS:
        if test(d):
            return name
    raise ValueError(d)


def report_bucket(label, g):
    n = len(g)
    model_ll, market_ll = g["model_loss"].mean(), g["market_loss"].mean()
    model_br, market_br = g["model_brier"].mean(), g["market_brier"].mean()
    observed, lo, hi = cluster_bootstrap_ci(g, "market_loss", "model_loss", group_col="player_a")
    winner = "model" if observed > 0 else "market"
    print(f"\n{label}: n={n}")
    print(f"  Log-loss  - model: {model_ll:.4f}   market: {market_ll:.4f}   "
          f"(lower is better; {winner} scores better)")
    print(f"  Brier     - model: {model_br:.4f}   market: {market_br:.4f}")
    print(f"  Player-clustered improvement (market_loss - model_loss, >0 = model better): "
          f"{observed:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
    return observed, lo, hi


def run():
    all_rows = []
    for bracket_path, _pretournament_csv, kaggle_tournament_name in CONCLUDED_TOURNAMENTS:
        print(f"Collecting {bracket_path.stem}...")
        rows = collect_rows(bracket_path, kaggle_tournament_name)
        print(f"  {len(rows)} real match(es) with both a model and a real market probability")
        all_rows.extend(rows)

    print(f"\nNOTE: live_espn-sourced matches in calibration_log.csv (Cincinnati) are excluded "
          f"entirely from this comparison - no recoverable market probability exists for them "
          f"(The Odds API has no historical endpoint this project uses, and calibration_log.py "
          f"never captured market_prob_a at logging time before now). This comparison covers only "
          f"kaggle_concluded matches (Montreal/Toronto), which carry real Odd_1/Odd_2.", file=sys.stderr)

    if not all_rows:
        print("\nNo matches with both a model and market probability - nothing to compare.")
        return

    df = pd.DataFrame(all_rows)
    df["model_loss"] = log_loss(df["actual_a_win"].values, df["model_prob_a"].values)
    df["market_loss"] = log_loss(df["actual_a_win"].values, df["market_prob_a"].values)
    df["model_brier"] = (df["actual_a_win"] - df["model_prob_a"]) ** 2
    df["market_brier"] = (df["actual_a_win"] - df["market_prob_a"]) ** 2
    df["disagreement"] = (df["model_prob_a"] - df["market_prob_a"]).abs()
    df["disagreement_bucket"] = df["disagreement"].apply(disagreement_bucket)

    print(f"\n{'=' * 90}\nOVERALL: {len(df)} real, decided matches with both a model and a real "
          f"market probability\n{'=' * 90}")
    report_bucket("Overall (all matches pooled)", df)

    print(f"\n{'=' * 90}\nSliced by disagreement size (|model - market|) - the only place a real "
          f"edge could exist\n{'=' * 90}")
    bucket_order = [b[0] for b in DISAGREEMENT_BUCKETS]
    bucket_results = []
    for b in bucket_order:
        g = df[df["disagreement_bucket"] == b]
        if len(g) == 0:
            print(f"\n{b}: n=0 - no matches in this bucket")
            continue
        observed, lo, hi = report_bucket(f"Disagreement {b}", g)
        bucket_results.append((b, len(g), observed, lo, hi))

    print(f"\n{'=' * 90}\nSummary: does the model's relative edge grow with disagreement size?\n{'=' * 90}")
    summary = pd.DataFrame(bucket_results, columns=["bucket", "n", "model_advantage", "ci_lo", "ci_hi"])
    print(summary.to_string(index=False, formatters={
        "model_advantage": "{:+.4f}".format, "ci_lo": "{:+.4f}".format, "ci_hi": "{:+.4f}".format,
    }))
    if len(bucket_results) >= 2:
        is_increasing = summary["model_advantage"].is_monotonic_increasing
        print(f"\nDoes the model's log-loss advantage over the market increase monotonically from "
              f"<5pp -> 5-10pp -> 10pp+ disagreement? {'YES' if is_increasing else 'NO'} - "
              f"{'this is the pattern a genuine edge would produce' if is_increasing else 'this is NOT the pattern a genuine edge would produce - report this honestly rather than cherry-picking a favorable bucket'}")


if __name__ == "__main__":
    run()
