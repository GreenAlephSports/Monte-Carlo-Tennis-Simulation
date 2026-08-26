"""Tests a recent-form correction: for each player, at the moment of each real match, compute
their actual win rate over their previous N (10, with 15 as a robustness check) real matches and
compare it to what Elo alone predicted for those same matches ("recent_form_residual" - positive
means they've been overperforming their Elo lately, negative means underperforming). Does this
residual carry real predictive signal for the player's NEXT match, beyond what their current Elo
rating already captures?

This is additive on top of a stable Elo rating (like the rank-gap/layoff/confidence-calibration
corrections already in production) - NOT a change to the lookback window itself (that was tested
and rejected tonight: see lookback_full_historical_test.py). A player's Elo can be static while
still running hot or cold recently; this tests whether that short-term wobble is informative.

Run at FULL historical scale from the start, both tours, ~2,800 tournament editions - tonight's
lookback-window test already showed a single-tournament (and even two-tournament) result can look
real and then evaporate or reverse at scale, so a small-sample first pass here would just repeat
that mistake.

Methodology:
  - Frozen per-tournament-edition Elo via elite_opponent_residual_test.build_frozen_predictions -
    the SAME single continuously-updated overall_elo (no windowing) already used for that test and
    veteran_decline_test.py, since the lookback window itself isn't what's under test here.
  - recent_form_residual per player-perspective row = mean(actual_win) - mean(pred_win) over that
    player's own previous WINDOW real matches (chronologically, shifted by one match so the
    current match itself is never included) - undefined (NaN, dropped) until a player has that
    many prior recorded matches.
  - Chronological tournament-edition 80/20 train/test split, computed per tour (same convention as
    lookback_full_historical_test.py), then combined for the headline number.
  - Rather than an arbitrary bucket cut (which risks cherry-picking a favorable tail), the primary
    fit is a single continuous logistic-regression coefficient beta on
    adjusted_logit = logit(pred_win) + beta * recent_form_residual, fit by 1D Newton-Raphson on
    TRAIN-era rows only - the same "single fitted global constant" shape as this project's own
    production rank-gap adjustment and Platt-scaling confidence calibration, so a positive result
    here would slot into the pipeline the same way those did.
  - Train-era significance (z from the Newton-Raphson Hessian) AND held-out validation (apply the
    fitted beta to TEST-era rows, player-clustered bootstrap CI on the log-loss improvement) are
    both reported - same "VERDICT" pattern as veteran_decline_test.py. A descriptive hot/cold
    tercile breakdown is also shown for interpretability, but the beta fit is the real test.

Usage:
    python model/research/recent_form_test.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import EPS, build_frozen_predictions, log_loss, logit, sigmoid  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

TRAIN_FRACTION = 0.8
PRIMARY_WINDOW = 10
ROBUSTNESS_WINDOW = 15


def add_recent_form(preds, window):
    """recent_form_residual as of each row = that player's own (actual - predicted) averaged over
    their previous `window` real matches, strictly excluding the current one. preds must already
    be sorted by date (build_frozen_predictions emits rows in chronological edition order, but
    re-sort defensively since groupby+rolling is order-sensitive)."""
    preds = preds.sort_values(["player", "date"], kind="stable").copy()
    roll_actual = preds.groupby("player")["actual_win"].transform(
        lambda s: s.rolling(window, min_periods=window).mean())
    roll_pred = preds.groupby("player")["pred_win"].transform(
        lambda s: s.rolling(window, min_periods=window).mean())
    preds["recent_actual_incl"] = roll_actual
    preds["recent_pred_incl"] = roll_pred
    preds["recent_form_actual"] = preds.groupby("player")["recent_actual_incl"].shift(1)
    preds["recent_form_pred"] = preds.groupby("player")["recent_pred_incl"].shift(1)
    preds["recent_form_residual"] = preds["recent_form_actual"] - preds["recent_form_pred"]
    return preds.drop(columns=["recent_actual_incl", "recent_pred_incl"])


def fit_beta_newton(offset, x, y, iters=100, tol=1e-10):
    """1D logistic regression with a fixed per-row offset: P(y=1) = sigmoid(offset + beta*x).
    Newton-Raphson on the single free parameter beta; returns (beta, standard_error)."""
    beta = 0.0
    for _ in range(iters):
        z = offset + beta * x
        p = 1 / (1 + np.exp(-np.clip(z, -35, 35)))
        grad = np.sum((y - p) * x)
        hess = -np.sum(p * (1 - p) * x * x)
        if hess == 0:
            break
        step = grad / hess
        beta -= step
        if abs(step) < tol:
            break
    z = offset + beta * x
    p = 1 / (1 + np.exp(-np.clip(z, -35, 35)))
    hess = -np.sum(p * (1 - p) * x * x)
    se = math.sqrt(-1 / hess) if hess < 0 else float("nan")
    return beta, se


def build_tour_predictions(tour):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)
    preds["tour"] = tour
    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    print(f"{tour}: {len(editions)} editions ({editions['edition_start'].min().date()} to "
          f"{editions['edition_start'].max().date()}); train = first {len(train_editions)}, "
          f"test = most recent {len(test_editions)} (from "
          f"{editions['edition_start'].iloc[split_idx].date()})")
    return preds, train_editions, test_editions


def run_for_window(window, all_data):
    print(f"\n{'#' * 90}\nWINDOW = last {window} real matches\n{'#' * 90}")

    train_parts, test_parts = [], []
    for tour, (preds, train_editions, test_editions) in all_data.items():
        p = add_recent_form(preds, window)
        p = p[p["recent_form_residual"].notna()].copy()
        train_parts.append(p[p["edition_id"].isin(train_editions)])
        test_parts.append(p[p["edition_id"].isin(test_editions)])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    print(f"Rows with a defined recent_form_residual (>= {window} prior real matches for that "
          f"player): {len(train)} train-era, {len(test)} test-era (both tours combined)")

    # --- primary test: single fitted continuous beta, same shape as production's rank-gap/Platt
    # corrections, fit on train, validated held-out on test.
    offset = train["pred_win"].apply(logit).values
    x = train["recent_form_residual"].values
    y = train["actual_win"].values
    beta, se = fit_beta_newton(offset, x, y)
    z = beta / se if se == se and se != 0 else float("nan")
    print(f"\nTrain-era fitted beta (adjusted_logit = logit(pred_win) + beta * recent_form_residual): "
          f"{beta:+.4f} (SE={se:.4f}, z={z:+.2f}, "
          f"{'|z|>1.96, nominally significant' if abs(z) > 1.96 else 'not significant on its own'})")

    test = test.copy()
    test["adjusted_pred"] = test.apply(
        lambda r: sigmoid(logit(r["pred_win"]) + beta * r["recent_form_residual"]), axis=1)
    test["raw_loss"] = log_loss(test["actual_win"].values, test["pred_win"].values)
    test["adj_loss"] = log_loss(test["actual_win"].values, test["adjusted_pred"].values)

    print(f"\nHeld-out test era ({len(test)} rows, {test['player'].nunique()} distinct players):")
    print(f"  Raw Elo         : mean log-loss = {test['raw_loss'].mean():.4f}")
    print(f"  Form-adjusted   : mean log-loss = {test['adj_loss'].mean():.4f}")

    observed, lo, hi = cluster_bootstrap_ci(test, "raw_loss", "adj_loss", group_col="player")
    ci_excludes_zero = lo > 0 or hi < 0
    print(f"  Mean per-match log-loss improvement (raw - adjusted, >0 = adjustment better), "
          f"player-clustered: {observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"\n  VERDICT @ window={window}: train z={z:+.2f} "
          f"({'nominally significant' if abs(z) > 1.96 else 'not significant on its own'}); "
          f"held-out CI {'excludes zero - real, held-out-validated effect' if ci_excludes_zero else 'straddles zero - NOT validated out of sample'}")

    # --- descriptive hot/cold tercile breakdown on the held-out test era, for interpretability
    # only (bucket edges from TEST era itself here since this is descriptive, not a second fit)
    q1, q2 = test["recent_form_residual"].quantile([1 / 3, 2 / 3])
    test["form_bucket"] = np.select(
        [test["recent_form_residual"] <= q1, test["recent_form_residual"] >= q2],
        ["cold (bottom tercile)", "hot (top tercile)"], default="neutral (middle tercile)")
    print(f"\nDescriptive breakdown, held-out test era, terciles by recent_form_residual "
          f"(cutpoints {q1:+.3f} / {q2:+.3f}):")
    desc = test.groupby("form_bucket").agg(
        n=("actual_win", "size"),
        assigned=("pred_win", "mean"),
        actual=("actual_win", "mean"),
        mean_recent_form_residual=("recent_form_residual", "mean"),
    ).reindex(["cold (bottom tercile)", "neutral (middle tercile)", "hot (top tercile)"])
    desc["gap"] = desc["assigned"] - desc["actual"]
    print(desc.to_string(formatters={
        "assigned": "{:.1%}".format, "actual": "{:.1%}".format, "gap": "{:+.1%}".format,
        "mean_recent_form_residual": "{:+.3f}".format,
    }))

    return beta, z, observed, lo, hi


def run():
    all_data = {}
    for tour in ["ATP", "WTA"]:
        all_data[tour] = build_tour_predictions(tour)

    results = {}
    for window in [PRIMARY_WINDOW, ROBUSTNESS_WINDOW]:
        results[window] = run_for_window(window, all_data)

    print(f"\n{'=' * 90}\nSUMMARY ACROSS WINDOWS\n{'=' * 90}")
    for window, (beta, z, observed, lo, hi) in results.items():
        ci_excludes_zero = lo > 0 or hi < 0
        print(f"  window={window:>2}: beta={beta:+.4f} (train z={z:+.2f}), held-out improvement "
              f"{observed:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] -> "
              f"{'VALIDATED' if ci_excludes_zero and lo > 0 else ('WORSE' if hi < 0 else 'NOT validated')}")


if __name__ == "__main__":
    run()
