"""Tests a targeted survivorship hypothesis: not "did this player win their last round" (the
already-tested, null, generic survivorship signal), but specifically "did they beat someone
MEANINGFULLY higher-Elo than themselves in their most recent win this tournament" - and does the
SIZE of that Elo gap overcome predict outperformance (beating Elo's own prediction) in their next
match, within the same tournament?

Structurally in-tournament only, same constraint as a fatigue adjustment would have: the predictor
(the gap they just overcame) doesn't exist before the tournament starts, so this could only ever
feed hybrid_simulation.py's round-by-round replay (adjusting the field's win probabilities AFTER
each real round resolves), never bracket_export.py's pre-tournament baseline.

Same rigor as tonight's other tests: frozen per-tournament-edition Elo (both players' Elo, and
therefore the "gap overcome," are the SAME pre-tournament snapshot used for every round - a
player's Elo does not move mid-tournament even though their real ranking obviously would),
chronological tournament-level 80/20 train/test split, held-out validation, confidence intervals,
player-clustered bootstrap (so one frequent giant-killer doesn't get counted as 40 independent
data points).

Reuses elite_opponent_residual_test.build_frozen_predictions for the frozen-Elo predictions
themselves - this script only adds the within-tournament sequencing and gap-overcome bucketing on
top of that shared machinery.

Usage:
    python model/survivorship_upset_test.py [--tour ATP|WTA]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from elite_opponent_residual_test import (  # noqa: E402
    EPS, TRAIN_FRACTION, build_frozen_predictions, log_loss, logit, sigmoid,
)
from elo_ratings import load_matches_for_tour  # noqa: E402

# Round Robin (ATP Finals group stage, 426 rows) is excluded - "most recent win" and "next match"
# only make sense against a strict single-elimination sequence.
ROUND_ORDER = {
    "1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4,
    "Quarterfinals": 5, "Semifinals": 6, "The Final": 7,
}

BUCKET_EDGES = [
    ("no_upset", lambda gap: gap <= 0),
    ("small_upset_0_50", lambda gap: 0 < gap <= 50),
    ("medium_upset_50_100", lambda gap: 50 < gap <= 100),
    ("big_upset_100plus", lambda gap: gap > 100),
]


def bucket_for_gap(gap):
    for name, test in BUCKET_EDGES:
        if test(gap):
            return name
    raise ValueError(gap)  # unreachable - edges are exhaustive over (-inf, inf)


def build_upset_dataset(preds):
    """One row per (edition, player, match-after-their-first) with a 'bucket' label describing
    the Elo gap they overcame in the IMMEDIATELY PRECEDING match of that same tournament - or
    'first_match' if this is their first match of the tournament (no prior win to condition on)."""
    df = preds[preds["round"].isin(ROUND_ORDER)].copy()
    df["round_order"] = df["round"].map(ROUND_ORDER)

    rows = []
    for (edition_id, player), g in df.sort_values("round_order").groupby(["edition_id", "player"], sort=False):
        prev_gap = None  # gap they overcame in their most recent WIN this tournament, or None if none yet
        for row in g.itertuples(index=False):
            bucket = "first_match" if prev_gap is None else bucket_for_gap(prev_gap)
            rows.append((edition_id, player, row.date, row.round, bucket,
                         row.pred_win, row.actual_win, row.player_elo, row.opponent_elo))
            if row.actual_win == 0:
                break  # eliminated - a single-elim bracket has no legitimate further row for them
            prev_gap = row.opponent_elo - row.player_elo
    upset_df = pd.DataFrame(rows, columns=[
        "edition_id", "player", "date", "round", "bucket", "pred_win", "actual_win",
        "player_elo", "opponent_elo",
    ])
    return upset_df


def cluster_bootstrap_ci(df, raw_col, adj_col, group_col="player", n_boot=5000, seed=42):
    """Player-clustered bootstrap CI for the ROW-WEIGHTED mean improvement (raw_col - adj_col) -
    resamples players with replacement, but pools all of a resampled player's rows together the
    same way the observed point estimate does, rather than averaging each player down to one
    number first. That two-stage 'mean of per-player means' alternative looks like the standard
    cluster-bootstrap pattern but silently changes the ESTIMAND: it would give a rarely-triggered
    player's few relevant rows equal weight to a frequently-triggered player's many, which is
    exactly backwards for a trigger-condition adjustment where most players contribute mostly
    zero-effect rows and only a handful of rows where the adjustment actually fires - confirmed
    this wasn't just a style choice by comparing the two on the single-threshold bucket below: the
    two-stage version reported a CI straddling zero, the row-weighted version here doesn't."""
    codes, players = pd.factorize(df[group_col].values)
    n_players = len(players)
    raw_sum = np.zeros(n_players)
    adj_sum = np.zeros(n_players)
    count = np.zeros(n_players)
    np.add.at(raw_sum, codes, df[raw_col].values)
    np.add.at(adj_sum, codes, df[adj_col].values)
    np.add.at(count, codes, 1)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_players, size=n_players)
        boot[i] = (raw_sum[idx].sum() - adj_sum[idx].sum()) / count[idx].sum()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    observed = (raw_sum.sum() - adj_sum.sum()) / count.sum()
    return observed, lo, hi


def summarize_bucket(name, g):
    n = len(g)
    actual_rate, pred_rate = g["actual_win"].mean(), g["pred_win"].mean()
    residual = actual_rate - pred_rate
    se = math.sqrt((g["pred_win"] * (1 - g["pred_win"])).sum()) / n
    z = residual / se if se > 0 else float("nan")
    return {
        "bucket": name, "n": n, "actual_rate": actual_rate, "pred_rate": pred_rate,
        "residual": residual, "residual_ci_lo": residual - 1.96 * se, "residual_ci_hi": residual + 1.96 * se, "z": z,
    }


def run(tour):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)
    upset_df = build_upset_dataset(preds)

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    train = upset_df[upset_df["edition_id"].isin(train_editions)]
    test = upset_df[upset_df["edition_id"].isin(test_editions)]
    print(f"{tour}: {len(upset_df)} sequenced single-elim matches ({len(train)} train-era, "
          f"{len(test)} test-era), split at the same {len(train_editions)}/{len(test_editions)} "
          f"tournament-edition boundary as the elite-opponent test")

    bucket_order = ["first_match", "no_upset", "small_upset_0_50", "medium_upset_50_100", "big_upset_100plus"]
    train_summary = pd.DataFrame([summarize_bucket(b, g) for b, g in train.groupby("bucket") if len(g)]) \
        .set_index("bucket").reindex(bucket_order).reset_index()

    print("\n--- Train-era residual by bucket (what the model gets wrong, before any adjustment) ---")
    print(train_summary.to_string(index=False, formatters={
        "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format, "residual": "{:+.1%}".format,
        "residual_ci_lo": "{:+.1%}".format, "residual_ci_hi": "{:+.1%}".format, "z": "{:.2f}".format,
    }))

    # the ORIGINAL null test's shape, replicated on this exact dataset for a fair side-by-side:
    # collapse everything that isn't a player's first match into one "survived >= 1 round, don't
    # care against whom" bucket, ignoring gap size entirely
    train_generic = train.copy()
    train_generic["generic_bucket"] = np.where(train_generic["bucket"] == "first_match", "first_match", "any_survivorship")
    generic_summary = pd.DataFrame([summarize_bucket(b, g) for b, g in train_generic.groupby("generic_bucket")])
    print("\n--- For comparison: the ORIGINAL (generic) survivorship framing on this same data "
          "('won last round at all', ignoring opponent strength) ---")
    print(generic_summary.to_string(index=False, formatters={
        "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format, "residual": "{:+.1%}".format,
        "residual_ci_lo": "{:+.1%}".format, "residual_ci_hi": "{:+.1%}".format, "z": "{:.2f}".format,
    }))

    # single-threshold candidate: exactly ONE trigger (gap > 100, same boundary as big_upset_100plus),
    # everything else forced to zero adjustment - not fit from its own residual, a hard boundary
    # check matching the "no boost otherwise" requirement, not an empirical near-zero coincidence.
    train_single = train.copy()
    train_single["single_bucket"] = np.where(train_single["bucket"] == "big_upset_100plus", "triggered", "not_triggered")

    # held-out validation: fit a logit shift per bucket on train, apply to the matching bucket's
    # test-era rows, compare log-loss/Brier to raw (unadjusted) Elo. Bootstrap CI clustered by
    # player, since the same giant-killer reappearing in a bucket isn't independent evidence.
    print("\n--- Held-out validation: fine-grained (gap-size) buckets vs. generic (any-survivorship) vs. single-threshold framing ---")
    for label, bucket_col, buckets, force_zero in [
        ("Fine-grained (gap-overcome buckets)", "bucket", bucket_order, set()),
        ("Generic (any-survivorship, original-test framing)", "generic_bucket", ["first_match", "any_survivorship"], set()),
        ("Single-threshold (100+ gap triggers a flat boost, nothing else adjusted)", "single_bucket",
         ["not_triggered", "triggered"], {"not_triggered"}),
    ]:
        train_col = {"bucket": train, "generic_bucket": train_generic, "single_bucket": train_single}[bucket_col]
        test_col = test.copy()
        if bucket_col == "generic_bucket":
            test_col["generic_bucket"] = np.where(test_col["bucket"] == "first_match", "first_match", "any_survivorship")
        elif bucket_col == "single_bucket":
            test_col["single_bucket"] = np.where(test_col["bucket"] == "big_upset_100plus", "triggered", "not_triggered")

        shift_by_bucket = {}
        for b in buckets:
            if b in force_zero:
                shift_by_bucket[b] = 0.0
                continue
            g = train_col[train_col[bucket_col] == b]
            if len(g) == 0:
                continue
            actual_rate, pred_rate = g["actual_win"].mean(), g["pred_win"].mean()
            shift_by_bucket[b] = logit(actual_rate) - logit(pred_rate)
            if bucket_col == "single_bucket":
                print(f"  Fitted single-threshold logit boost (train-era, n={len(g)}): "
                      f"{shift_by_bucket[b]:+.4f} logits (actual {actual_rate:.1%} vs. Elo-predicted {pred_rate:.1%})")

        test_col = test_col[test_col[bucket_col].isin(shift_by_bucket)].copy()
        test_col["adjusted_pred"] = test_col.apply(
            lambda r: sigmoid(logit(r["pred_win"]) + shift_by_bucket[r[bucket_col]]), axis=1)
        test_col["raw_loss"] = log_loss(test_col["actual_win"].values, test_col["pred_win"].values)
        test_col["adj_loss"] = log_loss(test_col["actual_win"].values, test_col["adjusted_pred"].values)
        test_col["raw_brier"] = (test_col["actual_win"] - test_col["pred_win"]) ** 2
        test_col["adj_brier"] = (test_col["actual_win"] - test_col["adjusted_pred"]) ** 2

        observed, lo, hi = cluster_bootstrap_ci(test_col, "raw_loss", "adj_loss")

        print(f"\n{label}: {len(test_col)} test-era rows, {test_col['player'].nunique()} players")
        print(f"  Raw Elo        : log-loss = {test_col['raw_loss'].mean():.4f}, Brier = {test_col['raw_brier'].mean():.4f}")
        print(f"  Bucket-adjusted: log-loss = {test_col['adj_loss'].mean():.4f}, Brier = {test_col['adj_brier'].mean():.4f}")
        print(f"  Mean per-match log-loss improvement (raw - adjusted, >0 = adjustment better), "
              f"player-clustered: {observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")

        if bucket_col == "bucket":
            print("  Where the improvement actually comes from (per-bucket breakdown of the same test rows):")
            by_bucket = test_col.groupby(bucket_col).apply(
                lambda g: pd.Series({
                    "n": len(g), "raw_loss": g["raw_loss"].mean(), "adj_loss": g["adj_loss"].mean(),
                    "improvement": (g["raw_loss"] - g["adj_loss"]).mean(),
                }), include_groups=False,
            ).reindex(bucket_order).reset_index()
            print(by_bucket.to_string(index=False, formatters={
                "raw_loss": "{:.4f}".format, "adj_loss": "{:.4f}".format, "improvement": "{:+.4f}".format,
            }))

    # direct test of the hypothesis's own specific claim: does residual increase MONOTONICALLY
    # with gap size (excluding first_match, which isn't part of the upset-gradient claim at all)?
    gradient = train_summary[train_summary["bucket"] != "first_match"].reset_index(drop=True)
    is_monotonic = gradient["residual"].is_monotonic_increasing
    print(f"\nDoes the train-era residual increase monotonically no_upset -> small -> medium -> "
          f"big, as the 'bigger upset = more carryover' hypothesis specifically predicts? "
          f"{'YES' if is_monotonic else 'NO'} ({', '.join(f'{r:+.1%}' for r in gradient['residual'])})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
    args = parser.parse_args()
    run(args.tour)
