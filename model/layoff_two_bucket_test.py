"""Fits a two-bucket layoff adjustment - <90 days since last match = no adjustment, 90+ days = a
single flat logit penalty - mirroring the single-threshold approach survivorship_upset_test.py
settled on for the upset-boost signal (a full graded scheme that looked good in training but
didn't hold up out-of-sample in the middle buckets, versus one threshold that captured nearly all
of the benefit with one parameter). layoff_test.py's own 5-bucket run showed the same pattern:
the 60_90d bucket doesn't fit a clean gradient in either tour, but the 90d_plus bucket is the
single most robust, most consistent effect (largest |z|, largest held-out per-bucket gain) in
both. This script tests whether collapsing everything below 90 days to "no adjustment" keeps
most of the 5-bucket version's held-out benefit.

Same fitting discipline as every prior test in this series: frozen per-tournament-edition Elo,
chronological tournament-level 80/20 train/test split, held-out validation of the fitted
adjustment against raw Elo, player-clustered bootstrap confidence intervals. Boundary check
included: recent play (<90 days) must get exactly zero adjustment by construction, not as a near-
zero fit accident - the single-parameter shift is only ever added to the 90+ bucket's rows.

Usage:
    python model/layoff_two_bucket_test.py [--tour ATP|WTA]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from elite_opponent_residual_test import TRAIN_FRACTION, log_loss, logit, sigmoid  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from layoff_test import build_frozen_predictions, build_layoff_dataset  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

LAYOFF_THRESHOLD_DAYS = 90


def run(tour):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)
    layoff_df = build_layoff_dataset(matches, preds)
    # no_prior_match rows (a player's first-ever recorded match, zero career history) are a
    # different population from a known player returning after a layoff, same exclusion
    # layoff_test.py applies to its own monotonicity check - the two-bucket adjustment is a
    # returning-player-rust story specifically, not a cold-start story.
    layoff_df = layoff_df[layoff_df["bucket"] != "no_prior_match"].copy()
    layoff_df["is_long_layoff"] = layoff_df["days_since_last"] >= LAYOFF_THRESHOLD_DAYS

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    train = layoff_df[layoff_df["edition_id"].isin(train_editions)]
    test = layoff_df[layoff_df["edition_id"].isin(test_editions)]

    long_train = train[train["is_long_layoff"]]
    actual_rate, pred_rate = long_train["actual_win"].mean(), long_train["pred_win"].mean()
    shift = logit(actual_rate) - logit(pred_rate)

    print(f"{tour}: fitted on {len(long_train)} train-era 90+ day rows "
          f"(actual win rate {actual_rate:.1%} vs. Elo's predicted {pred_rate:.1%})")
    print(f"  Fitted logit shift for >= {LAYOFF_THRESHOLD_DAYS}d layoff: {shift:+.4f} "
          f"(applied to the disadvantaged player; boundary is exact by construction - every row "
          f"with days_since_last < {LAYOFF_THRESHOLD_DAYS} gets shift = 0.0000)")

    test = test.copy()
    test["adjusted_pred"] = test.apply(
        lambda r: sigmoid(logit(r["pred_win"]) + shift) if r["is_long_layoff"] else r["pred_win"], axis=1)
    test["raw_loss"] = log_loss(test["actual_win"].values, test["pred_win"].values)
    test["adj_loss"] = log_loss(test["actual_win"].values, test["adjusted_pred"].values)
    test["raw_brier"] = (test["actual_win"] - test["pred_win"]) ** 2
    test["adj_brier"] = (test["actual_win"] - test["adjusted_pred"]) ** 2

    observed, lo, hi = cluster_bootstrap_ci(test, "raw_loss", "adj_loss")

    # boundary check: confirm the <90d rows really did get zero adjustment, not just "close to
    # zero" - the whole point of the single-threshold design.
    short = test[~test["is_long_layoff"]]
    max_short_delta = (short["adjusted_pred"] - short["pred_win"]).abs().max()
    print(f"  Boundary check: max |adjusted - raw| for the {len(short)} test-era rows under "
          f"{LAYOFF_THRESHOLD_DAYS}d = {max_short_delta:.10f} (must be exactly 0)")

    long_test = test[test["is_long_layoff"]]
    print(f"\n--- Held-out validation: {len(test)} test-era rows ({len(long_test)} in the 90+ day "
          f"bucket), {test['player'].nunique()} players ---")
    print(f"  Raw Elo          : log-loss = {test['raw_loss'].mean():.4f}, Brier = {test['raw_brier'].mean():.4f}")
    print(f"  Two-bucket adj.  : log-loss = {test['adj_loss'].mean():.4f}, Brier = {test['adj_brier'].mean():.4f}")
    print(f"  Mean per-match log-loss improvement (raw - adjusted), player-clustered: "
          f"{observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")

    only_long = test[test["is_long_layoff"]]
    long_improvement = (only_long["raw_loss"] - only_long["adj_loss"]).mean()
    print(f"  Within the 90+ day bucket alone: log-loss improvement = {long_improvement:+.4f} "
          f"over {len(only_long)} rows")

    return shift, observed, lo, hi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
    args = parser.parse_args()
    run(args.tour)
