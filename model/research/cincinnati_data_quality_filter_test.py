"""Tests the thin-data hypothesis directly, with real numbers, instead of pattern-matching by name.

cincinnati_paper_trading_backtest_tennisdata.py's 190-match, real-odds backtest came back
essentially flat-to-negative overall (see that script's own docstring/output). Eyeballing the
sorted-by-EV table, the largest edges looked concentrated among players who "sound like" thin-data
qualifiers/lucky losers - but that was a name-based guess, not a measurement. This script measures
it directly:

  1. For every +EV bet found (same opportunity set as the tennisdata backtest, rebuilt here with
     the same frozen-at-start_date ratings), pulls each side's REAL hard_matches count and
     days_since_last_match from the ratings CSV _prepare_ratings() already writes (elo_ratings.py's
     calculate_elo_ratings computes both columns as a side effect of building the Elo ratings
     themselves - no new calculation, just reading columns that already exist).
  2. Reports whether extreme-EV bets (>= EXTREME_EV_THRESHOLD) are disproportionately low-hard_matches
     compared to the full opportunity set - a real comparison of two real distributions, not a
     pattern-matched anecdote.
  3. Reruns the full backtest under a systematic grid of two real filters:
       - min_hard_matches in {10, 20, 30}: require BOTH player and opponent to have at least this
         many hard-court matches behind their frozen rating (a data-quality gate).
       - min_edge_pp in {5, 10, 15}: require the model/market probability gap to be at least this
         many percentage points (a materiality gate, same spirit as live_odds_value_scan.py's
         MIN_EDGE_PP but tested as a real grid here instead of one fixed constant).
     Every one of the 9 (+ baseline) combinations reports ROI/win-rate AND the surviving bet count,
     since a filter that leaves 3 bets isn't a meaningfully different result from having none.

Usage:
    python model/research/cincinnati_data_quality_filter_test.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bracket import match_name_to_pool  # noqa: E402
from bracket_schema import load_bracket_yaml  # noqa: E402
from calibration_log import _prepare_ratings  # noqa: E402
from cincinnati_paper_trading_backtest_tennisdata import (  # noqa: E402
    BRACKET_PATHS, KELLY_FRACTIONS, REFERENCE_BANKROLL, FLAT_STAKE,
    fetch_source_csv, kelly_fraction, size_and_settle, summarize, print_summary_table,
)
from ev_comparison import implied_probabilities  # noqa: E402
from win_probability import win_probability  # noqa: E402

EXTREME_EV_THRESHOLD = 0.50  # +50% EV, matching the user's own "extreme" cutoff
MIN_HARD_MATCHES_GRID = [10, 20, 30]
MIN_EDGE_PP_GRID = [5.0, 10.0, 15.0]


def build_opportunity_rows_with_data_quality(tour):
    """Same opportunity-finding logic as cincinnati_paper_trading_backtest_tennisdata.py's
    build_opportunity_rows_for_tour, extended to also attach each side's real hard_matches count
    and days_since_last_match, read straight from the ratings CSV _prepare_ratings() writes (not
    re-derived - the exact same frozen-at-start_date numbers win_probability() itself used)."""
    bracket = load_bracket_yaml(BRACKET_PATHS[tour])
    tour_config, draw, _matches_history = _prepare_ratings(bracket)

    ratings_df = pd.read_csv(tour_config.ratings_path).set_index("player")
    hard_matches = ratings_df["hard_matches"].to_dict()
    days_since_last = ratings_df["days_since_last_match"].to_dict()

    csv_path = fetch_source_csv(tour)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    pool = set(draw)
    rows = []
    for row in df.itertuples(index=False):
        winner_raw, loser_raw = row.Winner, row.Loser
        avg_w, avg_l = row.AvgW, row.AvgL
        if pd.isna(avg_w) or pd.isna(avg_l):
            continue

        winner = match_name_to_pool(winner_raw, pool, tour_config.name_aliases)
        loser = match_name_to_pool(loser_raw, pool, tour_config.name_aliases)
        if winner is None or loser is None:
            continue

        market_winner, market_loser = implied_probabilities(avg_w, avg_l)
        model_winner = win_probability(winner, loser, bracket.surface, tour_config.ratings_path)
        model_loser = 1 - model_winner

        for player, opponent, model_p, market_p, won in [
            (winner, loser, model_winner, market_winner, True),
            (loser, winner, model_loser, market_loser, False),
        ]:
            ev_per_unit = model_p / market_p - 1
            if ev_per_unit <= 0:
                continue
            rows.append({
                "tour": tour, "round": row.Round, "date": row.Date,
                "bet_on": player, "opponent": opponent,
                "model_prob": model_p, "market_prob": market_p, "ev_per_unit": ev_per_unit,
                "edge_pp": (model_p - market_p) * 100,
                "decimal_odds": 1 / market_p, "won": won,
                "player_hard_matches": hard_matches.get(player),
                "opponent_hard_matches": hard_matches.get(opponent),
                "player_days_since_last_match": days_since_last.get(player),
                "opponent_days_since_last_match": days_since_last.get(opponent),
            })
    return pd.DataFrame(rows)


def report_thin_data_hypothesis(all_opps):
    print(f"\n=== Thin-data hypothesis: extreme-EV bets (>= +{EXTREME_EV_THRESHOLD:.0%} EV) vs. the "
          f"full opportunity set ===")
    all_opps = all_opps.copy()
    all_opps["min_hard_matches"] = all_opps[["player_hard_matches", "opponent_hard_matches"]].min(axis=1)

    extreme = all_opps[all_opps["ev_per_unit"] >= EXTREME_EV_THRESHOLD]
    rest = all_opps[all_opps["ev_per_unit"] < EXTREME_EV_THRESHOLD]

    print(f"\n{len(extreme)} of {len(all_opps)} bets clear +{EXTREME_EV_THRESHOLD:.0%} EV "
          f"({len(extreme)/len(all_opps):.1%} of all opportunities).")
    print(f"\nmin(player, opponent) hard_matches - the weaker-supported side of each bet:")
    print(f"  Extreme-EV bets  (n={len(extreme):>3}): mean={extreme['min_hard_matches'].mean():>6.1f}  "
          f"median={extreme['min_hard_matches'].median():>5.1f}")
    print(f"  Remaining bets   (n={len(rest):>3}): mean={rest['min_hard_matches'].mean():>6.1f}  "
          f"median={rest['min_hard_matches'].median():>5.1f}")

    for threshold in MIN_HARD_MATCHES_GRID:
        pct_extreme = (extreme["min_hard_matches"] < threshold).mean() if len(extreme) else float("nan")
        pct_rest = (rest["min_hard_matches"] < threshold).mean() if len(rest) else float("nan")
        print(f"  Share with min_hard_matches < {threshold}: extreme-EV = {pct_extreme:.1%}, "
              f"rest = {pct_rest:.1%}")

    print(f"\n--- Every extreme-EV bet (raw numbers, not name pattern-matching) ---")
    cols = ["tour", "round", "bet_on", "player_hard_matches", "player_days_since_last_match",
            "opponent", "opponent_hard_matches", "opponent_days_since_last_match",
            "model_prob", "market_prob", "ev_per_unit", "won"]
    display = extreme.sort_values("ev_per_unit", ascending=False)[cols]
    print(display.to_string(index=False, formatters={
        "model_prob": "{:.1%}".format, "market_prob": "{:.1%}".format, "ev_per_unit": "{:+.1%}".format,
    }))


def run_filter_grid(all_opps):
    print(f"\n=== Systematic filter grid: min_hard_matches x min_edge_pp ===")
    all_opps = all_opps.copy()
    all_opps["min_hard_matches"] = all_opps[["player_hard_matches", "opponent_hard_matches"]].min(axis=1)

    baseline = size_and_settle(all_opps)
    base_summary = summarize(baseline)
    print(f"\nBaseline (no filter, n={len(all_opps)}):")
    print(f"  Flat: win% {base_summary['flat']['win_rate']:.1%}, ROI {base_summary['flat']['roi_pct']:+.1f}%   "
          f"Kelly 0.25x: ROI {base_summary['kelly_0.25']['roi_pct']:+.1f}%   "
          f"Kelly 0.5x: ROI {base_summary['kelly_0.5']['roi_pct']:+.1f}%")

    header = (f"{'min_hard':>9} {'min_edge_pp':>11} {'n_bets':>7} {'win%':>7} "
              f"{'flat_ROI%':>10} {'k0.25_ROI%':>11} {'k0.5_ROI%':>10}")
    print(f"\n{header}")
    print("-" * len(header))

    results = []
    for min_hm in MIN_HARD_MATCHES_GRID:
        for min_edge in MIN_EDGE_PP_GRID:
            filtered = all_opps[
                (all_opps["min_hard_matches"] >= min_hm) & (all_opps["edge_pp"] >= min_edge)
            ]
            n = len(filtered)
            if n == 0:
                print(f"{min_hm:>9} {min_edge:>10.0f}pp {n:>7} {'--':>7} {'--':>10} {'--':>11} {'--':>10}  "
                      f"(zero bets survive)")
                results.append({"min_hard_matches": min_hm, "min_edge_pp": min_edge, "n_bets": 0})
                continue
            settled = size_and_settle(filtered)
            s = summarize(settled)
            print(f"{min_hm:>9} {min_edge:>10.0f}pp {n:>7} {s['flat']['win_rate']:>6.1%} "
                  f"{s['flat']['roi_pct']:>+9.1f}% {s['kelly_0.25']['roi_pct']:>+10.1f}% "
                  f"{s['kelly_0.5']['roi_pct']:>+9.1f}%")
            results.append({
                "min_hard_matches": min_hm, "min_edge_pp": min_edge, "n_bets": n,
                "win_rate": s["flat"]["win_rate"], "flat_roi": s["flat"]["roi_pct"],
                "kelly_0.25_roi": s["kelly_0.25"]["roi_pct"], "kelly_0.5_roi": s["kelly_0.5"]["roi_pct"],
            })
    return pd.DataFrame(results)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    all_opps = pd.concat(
        [build_opportunity_rows_with_data_quality(t) for t in ("ATP", "WTA")], ignore_index=True
    )
    print(f"\n{len(all_opps)} total +EV opportunities loaded (both tours), with real hard_matches "
          f"and days_since_last_match attached for both sides of every bet.")

    report_thin_data_hypothesis(all_opps)
    grid_df = run_filter_grid(all_opps)

    n_reasonable = grid_df[grid_df["n_bets"] >= 20]
    print(f"\n=== Verdict ===")
    print(f"{len(grid_df[grid_df['n_bets'] > 0])}/{len(grid_df)} grid cells have any surviving bets at "
          f"all; {len(n_reasonable)}/{len(grid_df)} retain at least 20 bets (an arbitrary but stated "
          f"'still a real sample' floor - anything below that is reported but shouldn't be read as a "
          f"stable result).")
    if len(n_reasonable):
        best = n_reasonable.sort_values("flat_roi", ascending=False).iloc[0]
        print(f"Best flat-stake ROI among cells with >=20 bets: min_hard_matches>={best['min_hard_matches']:.0f}, "
              f"min_edge_pp>={best['min_edge_pp']:.0f} -> n={best['n_bets']:.0f}, "
              f"win_rate={best['win_rate']:.1%}, flat ROI={best['flat_roi']:+.1f}%, "
              f"Kelly 0.25x ROI={best['kelly_0.25_roi']:+.1f}%")
    print(f"See the full grid above for every combination's real bet count and ROI - report exactly "
          f"what it shows, not just the best cell.")


if __name__ == "__main__":
    main()
