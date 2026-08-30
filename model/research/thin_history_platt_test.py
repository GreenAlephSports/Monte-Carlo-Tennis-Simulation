"""Third fix attempt for thin-history calibration, mechanistically distinct from the first two:
thin_history_rank_blend_test.py (blend the RATING toward rank) and thin_history_shrinkage_test.py
(blend the RATING toward the population mean) both tried to fix the player's Elo number itself.
This one never touches the rating at all - it recalibrates the PROBABILITY win_probability()
produces, the exact same mechanism win_probability._apply_confidence_calibration already uses in
production (Platt scaling: calibrated = sigmoid(PLATT_B * logit(raw)), one free parameter, intercept
fixed at exactly 0 by the mirrored-observation symmetry every row in this dataset already has), just
fit on a different population: rows where the player themselves has <10 career matches, instead of
gated by how confident the raw prediction already is.

Motivation: historical_bracket_calibration.py's round-depth breakdown found the tournament-level
Round-of-64 overconfidence (-2.02% aggregate gap) is almost entirely concentrated in thin-history
players (-15.4% gap, CI excludes zero, vs -0.5% and not significant for solid 30+-match players) -
the same population both prior rating-side fixes already failed to help. A probability-side
correction is a genuinely different lever: it doesn't ask "is 1500 the right default Elo for this
player," it asks "regardless of what the rating implies, is the WIN PROBABILITY the model outputs
for this population systematically too extreme (or not extreme enough)."

Methodology - same rigor and same reused machinery as every prior test in this series:
  - Population/dataset construction reused directly from thin_history_rank_blend_test.build_dataset
    (frozen per-tournament-edition Elo, player_matches_before column) - not reimplemented.
  - Chronological tournament-edition 80/20 train/test split (elite_opponent_residual_test.
    TRAIN_FRACTION).
  - Treatment population: rows where player_matches_before < THIN_THRESHOLD (default 10, same
    threshold as both prior thin-history tests) - the row's OWN perspective, not "either side," so
    the mirrored dataset naturally scores both this player's matches and (separately, as their own
    row) the opponent's perspective on the same match only if the opponent is ALSO thin.
  - Platt B is grid-searched on TRAIN-era thin rows only, minimizing train log-loss, then validated
    held-out on TEST-era thin rows the value was never chosen to fit - never re-derived from test
    data, same discipline as the rank-trajectory-lag weight grid search.
  - Compared against three baselines on held-out thin rows: raw (uncalibrated) Elo, and production's
    EXISTING global Platt correction (PLATT_B=0.9205, applied unconditionally) - a population-
    specific fit is only worth anything if it beats what's already deployed, not just raw Elo.
  - Same "does it actually tame extreme predictions, and in the right direction" check that caught
    thin_history_rank_blend_test.py's harm: share of held-out thin-population predictions more
    extreme than 70/30, before vs after, plus a finer 0-2/3-9 match sub-bucket breakdown (the same
    split that showed rank-blend was harmful specifically at the thinnest end).
  - Player-clustered bootstrap CI throughout (survivorship_upset_test.cluster_bootstrap_ci).

Usage:
    python model/research/thin_history_platt_test.py [--tour ATP|WTA] [--thin-threshold N] [--max-editions N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import TRAIN_FRACTION, log_loss, logit, sigmoid  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from thin_history_rank_blend_test import DEFAULT_THIN_THRESHOLD, build_dataset  # noqa: E402

# grid centered on 1.0 (no-op); <1 shrinks toward 50/50 (less confident), >1 amplifies (more
# confident) - fine enough (0.05 steps) to distinguish from production's existing global PLATT_B=
# 0.9205 without over-fitting a population this small.
CANDIDATE_B = [round(b, 2) for b in np.arange(0.30, 1.71, 0.05)]
PRODUCTION_GLOBAL_PLATT_B = 0.9205  # win_probability.PLATT_B - the correction already in production
EXTREME_THRESHOLD = 0.70  # |pred - 0.5| > this counts as an "extreme" prediction (30pp+ favorite)


def fit_platt_b(train_thin):
    """Grid search B minimizing train-era log-loss on the thin population, mirroring
    rank_trajectory_lag_test.py's CANDIDATE_WEIGHTS grid-search pattern."""
    best_b, best_loss = None, float("inf")
    print("--- Grid search (train-era thin rows only) ---")
    for b in CANDIDATE_B:
        calibrated = train_thin["pred_win"].apply(lambda p: sigmoid(b * logit(p)))
        loss = log_loss(train_thin["actual_win"].values, calibrated.values).mean()
        marker = ""
        if loss < best_loss:
            best_loss, best_b = loss, b
            marker = "  <- best so far"
        if b in (0.30, 0.50, 0.70, 0.9205, 1.00, 1.20, 1.50, 1.70) or b == best_b:
            print(f"  B={b:.2f}: train log-loss = {loss:.4f}{marker}")
    print(f"  -> selected B={best_b:.2f} (lowest train-era log-loss)\n")
    return best_b


def extreme_share(preds):
    return ((preds - 0.5).abs() > (EXTREME_THRESHOLD - 0.5)).mean()


def score_population(df, label, b=None):
    """Raw vs. Platt-corrected (if b given) log-loss/Brier/extreme-share for one population slice."""
    raw_loss = log_loss(df["actual_win"].values, df["pred_win"].values)
    raw_extreme = extreme_share(df["pred_win"])
    print(f"  {label} (n={len(df)}): raw log-loss={raw_loss.mean():.4f}  "
          f"raw extreme(|p-50%|>{EXTREME_THRESHOLD - 0.5:.0%})share={raw_extreme:.1%}")
    if b is not None and len(df):
        corrected = df["pred_win"].apply(lambda p: sigmoid(b * logit(p)))
        corr_loss = log_loss(df["actual_win"].values, corrected.values)
        corr_extreme = extreme_share(corrected)
        print(f"    -> B={b:.2f} corrected log-loss={corr_loss.mean():.4f}  "
              f"corrected extreme share={corr_extreme:.1%}")
        return raw_loss, corr_loss
    return raw_loss, None


def run(tour, thin_threshold, max_editions=None):
    if max_editions is not None:
        print(f"*** QUICK CHECK ON RECENT DATA ONLY (--max-editions {max_editions}) - NOT the "
              f"full-historical verdict. This population is naturally small even at full scale, so "
              f"a recent-only window may leave too few rows to mean anything - read the n's below "
              f"before trusting any number here. ***\n")
    matches = load_matches_for_tour(tour)
    preds, editions = build_dataset(matches, max_editions=max_editions)

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    print(f"{tour}: {len(editions)} tournament editions, train = first {len(train_editions)}, "
          f"test = remaining {len(test_editions)}")

    thin = preds[preds["player_matches_before"] < thin_threshold].copy()
    train_thin = thin[thin["edition_id"].isin(train_editions)]
    test_thin = thin[thin["edition_id"].isin(test_editions)]
    print(f"Thin population (player_matches_before < {thin_threshold}): {len(train_thin)} train-era "
          f"rows, {len(test_thin)} test-era rows ({thin['player'].nunique()} distinct players total)\n")

    if len(train_thin) < 50 or len(test_thin) < 50:
        print("Too few thin-population rows in train or test era to fit/validate a Platt correction "
              "- stopping (population as defined is too small for this dataset/window).")
        return

    best_b = fit_platt_b(train_thin)

    print(f"--- Held-out validation on the thin population (n={len(test_thin)}) ---")
    raw_loss, fitted_loss = score_population(test_thin, "raw vs. fitted-B", b=best_b)
    _, global_loss = score_population(test_thin, "raw vs. production's existing global Platt", b=PRODUCTION_GLOBAL_PLATT_B)

    scored = test_thin.copy()
    scored["raw_loss"] = raw_loss
    scored["fitted_loss"] = fitted_loss
    scored["global_loss"] = global_loss

    print()
    obs_fit, lo_fit, hi_fit = cluster_bootstrap_ci(scored, "raw_loss", "fitted_loss", group_col="player")
    verdict_fit = "IMPROVES" if lo_fit > 0 else ("HURTS" if hi_fit < 0 else "NO SIGNIFICANT EFFECT")
    print(f"  Fitted-B vs. raw Elo, player-clustered improvement (raw - fitted, >0 = better): "
          f"{obs_fit:+.4f}, 95% CI [{lo_fit:+.4f}, {hi_fit:+.4f}] -> {verdict_fit}")

    obs_glob, lo_glob, hi_glob = cluster_bootstrap_ci(scored, "raw_loss", "global_loss", group_col="player")
    verdict_glob = "IMPROVES" if lo_glob > 0 else ("HURTS" if hi_glob < 0 else "NO SIGNIFICANT EFFECT")
    print(f"  Production's existing global Platt vs. raw Elo (thin pop. only), improvement: "
          f"{obs_glob:+.4f}, 95% CI [{lo_glob:+.4f}, {hi_glob:+.4f}] -> {verdict_glob}")

    obs_h2h, lo_h2h, hi_h2h = cluster_bootstrap_ci(scored, "global_loss", "fitted_loss", group_col="player")
    verdict_h2h = "fitted-B BEATS the existing global correction" if lo_h2h > 0 else (
        "existing global correction BEATS a thin-specific fit" if hi_h2h < 0 else "NOT distinguishable")
    print(f"  Head-to-head, fitted-B vs. existing global Platt (global - fitted, >0 = fitted-B "
          f"better): {obs_h2h:+.4f}, 95% CI [{lo_h2h:+.4f}, {hi_h2h:+.4f}] -> {verdict_h2h}")

    # finer sub-bucket breakdown - the exact check that caught rank-blend being actively harmful at
    # the thinnest end even though its overall/aggregate number looked directionally fine.
    print(f"\n--- Sub-bucket breakdown (does the fitted correction help evenly, or hurt the thinnest end?) ---")
    for lo_n, hi_n, name in [(0, 3, "0-2 matches"), (3, thin_threshold, f"3-{thin_threshold - 1} matches")]:
        bucket = test_thin[(test_thin["player_matches_before"] >= lo_n) & (test_thin["player_matches_before"] < hi_n)]
        if len(bucket) < 20:
            print(f"  {name}: n={len(bucket)} - too few to report")
            continue
        score_population(bucket, name, b=best_b)

    if len(test_thin) < 200:
        print(f"\n  CAUTION: n={len(test_thin)} test-era thin rows is a small sample - treat this as "
              f"early/directional, not a settled verdict, regardless of which way it points.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
    parser.add_argument("--thin-threshold", type=int, default=DEFAULT_THIN_THRESHOLD)
    parser.add_argument("--max-editions", type=int, default=None,
                         help="quick-check mode: only score the most recent N tournament editions "
                              "(before the 80/20 split), instead of the full lookback window")
    args = parser.parse_args()
    run(args.tour, args.thin_threshold, max_editions=args.max_editions)
