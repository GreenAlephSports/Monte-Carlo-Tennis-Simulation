"""Answers a real question the point-estimate backtest can't: how much of the filtered result's
+8.5%/+11.8% ROI is real signal vs. one lucky realization of a modest (n=34) sample?

No new data - same cached market odds and same frozen-at-start_date model probabilities as
cincinnati_paper_trading_backtest_tennisdata.py (nothing is refetched). What's new is simulating
each bet's win/loss outcome as a Bernoulli draw at the model's own claimed probability, 20,000
times, instead of relying on the single real outcome that actually happened. This is a variance
check on the model's own assumptions, not a different backtest - if the model's probabilities are
roughly right, the realized ROI should sit comfortably inside the simulated distribution; if it's
an outlier, that's the sample warning you.

Run for both the filtered (n=34, min_hard_matches>=30, min_edge_pp>=10) and unfiltered (n=190) sets,
so the artifact can show side by side how much tighter the filtered distribution is.

Usage:
    python model/research/cincinnati_montecarlo_variance_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cincinnati_paper_trading_backtest_tennisdata import (  # noqa: E402
    build_opportunity_rows_for_tour, size_and_settle, summarize,
    MIN_HARD_MATCHES, MIN_EDGE_PP, KELLY_FRACTIONS,
)

N_TRIALS = 20_000
RNG_SEED = 20260825  # fixed seed = reproducible report numbers, not a hidden random search


def simulate_roi_distribution(opps, stake_col, rng):
    """Vectorized: draw N_TRIALS x n_bets Bernoulli outcomes at each bet's own model_prob (not the
    real, single outcome that happened), settle every trial at the same real decimal_odds and
    stake this bet was actually sized at, and return the resulting ROI% for every trial."""
    model_p = opps["model_prob"].to_numpy()
    stake = opps[stake_col].to_numpy()
    decimal_odds = opps["decimal_odds"].to_numpy()

    draws = rng.random((N_TRIALS, len(opps))) < model_p  # True = simulated win
    pnl_if_win = stake * (decimal_odds - 1)
    pnl_if_loss = -stake
    trial_pnl = np.where(draws, pnl_if_win, pnl_if_loss).sum(axis=1)

    total_staked = stake.sum()
    return trial_pnl / total_staked * 100  # ROI% per trial


def report_variance_check(label, opps, rng):
    opps = size_and_settle(opps)
    real_summary = summarize(opps)

    print(f"\n=== {label} (n={len(opps)}) ===")
    for stake_label, stake_col in [("flat", "stake_flat"), ("kelly_0.25", "stake_kelly_0.25")]:
        roi_dist = simulate_roi_distribution(opps, stake_col, rng)
        real_roi = real_summary["flat" if stake_label == "flat" else "kelly_0.25"]["roi_pct"]
        p5, p50, p95 = np.percentile(roi_dist, [5, 50, 95])
        prob_loss = (roi_dist < 0).mean() * 100
        print(f"  [{stake_label}] realized ROI: {real_roi:+.1f}%  |  "
              f"{N_TRIALS:,}-trial simulated distribution (same odds/probs, resampled outcomes): "
              f"mean {roi_dist.mean():+.1f}%  median {p50:+.1f}%  "
              f"P5-P95 [{p5:+.1f}%, {p95:+.1f}%]  P(ROI<0) = {prob_loss:.1f}%")
        yield {
            "label": label, "n_bets": len(opps), "stake": stake_label, "real_roi": real_roi,
            "mean": roi_dist.mean(), "median": p50, "p5": p5, "p95": p95, "prob_loss": prob_loss,
        }


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    rng = np.random.default_rng(RNG_SEED)
    all_opps = pd.concat([build_opportunity_rows_for_tour(t) for t in ("ATP", "WTA")], ignore_index=True)
    filtered_opps = all_opps[
        (all_opps["min_hard_matches"] >= MIN_HARD_MATCHES) & (all_opps["edge_pp"] >= MIN_EDGE_PP)
    ]

    print(f"Monte Carlo variance check - {N_TRIALS:,} simulated trials per set, same cached market "
          f"odds and model probabilities as the real backtest, only the bet-by-bet outcome resampled.")

    results = list(report_variance_check("BEFORE filtering (unfiltered, all +EV bets)", all_opps, rng))
    results += list(report_variance_check("AFTER filtering (standing result)", filtered_opps, rng))

    out_df = pd.DataFrame(results)
    out_path = Path(__file__).resolve().parent.parent.parent / "output" / "cincinnati_montecarlo_variance.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
