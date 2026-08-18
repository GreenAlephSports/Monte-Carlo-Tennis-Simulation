"""Fits a three-bucket layoff adjustment - <60 days = no adjustment, 60-90 days = one flat logit
shift, 90+ days = a second flat logit shift - as a middle ground between the full 5-bucket
gradient (layoff_test.py) and the single-threshold 90+-only collapse (layoff_two_bucket_test.py).

The two-bucket collapse failed its own held-out validation gate (overall CI included zero in both
tours) specifically because it discarded the 14_30d/30_60d buckets' real, individually-significant
contributions along with the noisier 60_90d one. This version uses the boundary the user actually
asked for - <60d untouched, 60-90d one shift, 90d+ a second, presumably larger shift - which still
leaves the 30_60d bucket's own real 5-bucket-run effect uncaptured (zeroed out same as under_14d/
14_30d, since it falls below the 60-day line). The held-out numbers below will show whether folding
60_90d and 90d+ into two parameters is still enough to clear the validation gate the two-bucket
version missed, or whether 30_60d's dropped contribution was actually load-bearing.

Same fitting discipline as every prior test in this series: frozen per-tournament-edition Elo,
chronological tournament-level 80/20 train/test split, held-out validation against raw Elo,
player-clustered bootstrap confidence intervals, and a boundary check that <60d rows get exactly
zero adjustment by construction.

Usage:
    python model/layoff_three_bucket_test.py [--tour ATP|WTA]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from elite_opponent_residual_test import TRAIN_FRACTION, log_loss, logit, sigmoid  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from layoff_test import build_frozen_predictions, build_layoff_dataset  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

BUCKETS = [
    ("under_60d", lambda d: d < 60),
    ("60_90d", lambda d: 60 <= d < 90),
    ("90d_plus", lambda d: d >= 90),
]


def three_bucket_for_days(days):
    for name, test in BUCKETS:
        if test(days):
            return name
    raise ValueError(days)


def run(tour):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)
    layoff_df = build_layoff_dataset(matches, preds)
    layoff_df = layoff_df[layoff_df["bucket"] != "no_prior_match"].copy()
    layoff_df["bucket3"] = layoff_df["days_since_last"].apply(three_bucket_for_days)

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    train = layoff_df[layoff_df["edition_id"].isin(train_editions)]
    test = layoff_df[layoff_df["edition_id"].isin(test_editions)]

    shift_by_bucket = {"under_60d": 0.0}
    for name in ("60_90d", "90d_plus"):
        g = train[train["bucket3"] == name]
        actual_rate, pred_rate = g["actual_win"].mean(), g["pred_win"].mean()
        shift = logit(actual_rate) - logit(pred_rate)
        shift_by_bucket[name] = shift
        print(f"{tour}: {name} fitted on {len(g)} train-era rows "
              f"(actual {actual_rate:.1%} vs. Elo's predicted {pred_rate:.1%}) -> shift {shift:+.4f}")

    test = test.copy()
    test["adjusted_pred"] = test.apply(
        lambda r: sigmoid(logit(r["pred_win"]) + shift_by_bucket[r["bucket3"]]), axis=1)
    test["raw_loss"] = log_loss(test["actual_win"].values, test["pred_win"].values)
    test["adj_loss"] = log_loss(test["actual_win"].values, test["adjusted_pred"].values)
    test["raw_brier"] = (test["actual_win"] - test["pred_win"]) ** 2
    test["adj_brier"] = (test["actual_win"] - test["adjusted_pred"]) ** 2

    observed, lo, hi = cluster_bootstrap_ci(test, "raw_loss", "adj_loss")

    short = test[test["bucket3"] == "under_60d"]
    max_short_delta = (short["adjusted_pred"] - short["pred_win"]).abs().max()
    print(f"  Boundary check: max |adjusted - raw| for the {len(short)} test-era rows under 60d = "
          f"{max_short_delta:.10f} (must be exactly 0)")

    print(f"\n--- Held-out validation: {len(test)} test-era rows, {test['player'].nunique()} players ---")
    print(f"  Raw Elo          : log-loss = {test['raw_loss'].mean():.4f}, Brier = {test['raw_brier'].mean():.4f}")
    print(f"  Three-bucket adj.: log-loss = {test['adj_loss'].mean():.4f}, Brier = {test['adj_brier'].mean():.4f}")
    print(f"  Mean per-match log-loss improvement (raw - adjusted), player-clustered: "
          f"{observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")

    print("\n  Per-bucket breakdown of the same test rows:")
    by_bucket = test.groupby("bucket3").apply(
        lambda g: __import__("pandas").Series({
            "n": len(g), "raw_loss": g["raw_loss"].mean(), "adj_loss": g["adj_loss"].mean(),
            "improvement": (g["raw_loss"] - g["adj_loss"]).mean(),
        }), include_groups=False,
    ).reindex([b[0] for b in BUCKETS]).reset_index()
    print(by_bucket.to_string(index=False, formatters={
        "raw_loss": "{:.4f}".format, "adj_loss": "{:.4f}".format, "improvement": "{:+.4f}".format,
    }))

    return shift_by_bucket, observed, lo, hi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
    args = parser.parse_args()
    run(args.tour)
