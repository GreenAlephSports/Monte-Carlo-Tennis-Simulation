"""Tests whether time since a player's last recorded match (a layoff/rust proxy, computed
directly from existing Kaggle match dates - no new data source needed) predicts underperformance
relative to Elo. This directly tests the mechanism suggested by the real Draper case found
earlier: a long layoff not being reflected in static pre-tournament Elo, which tracks who beat
whom but never how recently they last played.

Unlike the elite-opponent and gap-overcome (survivorship) tests, days-since-last-match is known
BEFORE a tournament even starts (it only needs the player's own last real match, anywhere, in any
earlier tournament) - so if this signal turns out to be real, it could feed bracket_export.py's
pre-tournament baseline directly, not just hybrid_simulation.py's round-by-round replay the way
the gap-overcome adjustment is structurally limited to.

days_since_last_match is computed per MATCH ROW, not frozen per tournament edition the way Elo
is: a player's rust plausibly wears off as their own tournament goes on, so round 2+ of the same
event correctly lands in the <14-day bucket on its own (they just played round 1) without a
separate within-tournament decay model - the same "a fixed pre-tournament snapshot, refreshed
every subsequent real result" idea hybrid_simulation.py's round replay already uses for Elo
itself, just applied to a different quantity.

Same rigor as every prior test in this series: frozen per-tournament-edition Elo, reused directly
from elite_opponent_residual_test.build_frozen_predictions (same predictions, same chronological
tournament-level 80/20 train/test split - never a random match-level split, which would leak a
player's own future form into their own training residual), held-out validation of the
bucket-adjusted vs. raw prediction, and player-clustered bootstrap confidence intervals (reused
from survivorship_upset_test.py, for the same reason: one frequent long-layoff returner shouldn't
count as N independent data points).

Usage:
    python model/layoff_test.py [--tour ATP|WTA]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from elite_opponent_residual_test import TRAIN_FRACTION, build_frozen_predictions, log_loss, logit, sigmoid  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci, summarize_bucket  # noqa: E402

LAYOFF_BUCKET_EDGES = [
    ("under_14d", lambda d: d < 14),
    ("14_30d", lambda d: 14 <= d < 30),
    ("30_60d", lambda d: 30 <= d < 60),
    ("60_90d", lambda d: 60 <= d < 90),
    ("90d_plus", lambda d: d >= 90),
]
BUCKET_ORDER = ["no_prior_match"] + [name for name, _ in LAYOFF_BUCKET_EDGES]


def bucket_for_days(days):
    if pd.isna(days):
        return "no_prior_match"  # this player's first-ever recorded match in the dataset
    for name, test in LAYOFF_BUCKET_EDGES:
        if test(days):
            return name
    raise ValueError(days)  # unreachable - edges are exhaustive over [0, inf)


def compute_days_since_last_match(matches):
    """One row per (edition_id, date, round, player, opponent) - the same granularity as one leg
    of build_frozen_predictions' player-perspective rows - giving the real calendar gap since that
    player's own previous recorded match, in EITHER tour history (not just this tournament, and
    not just this edition): a player returning from a long injury layoff has their last match in
    some earlier tournament, not this one. Keying on (..., round, player) alone isn't quite enough
    - Round Robin group-stage play (e.g. the ATP Finals) has a player facing multiple different
    opponents under the identical 'Round Robin' round label, sometimes the same day - opponent is
    included to disambiguate those, not because layoff itself depends on who they played."""
    long = pd.concat([
        matches[["Tournament", "Date", "Round", "Player_1", "Player_2"]]
        .rename(columns={"Player_1": "player", "Player_2": "opponent"}),
        matches[["Tournament", "Date", "Round", "Player_2", "Player_1"]]
        .rename(columns={"Player_2": "player", "Player_1": "opponent"}),
    ], ignore_index=True)
    long["edition_id"] = long["Tournament"] + " " + long["Date"].dt.year.astype(str)
    long = long.sort_values(["player", "Date"], kind="stable").reset_index(drop=True)
    long["prev_date"] = long.groupby("player")["Date"].shift(1)
    long["days_since_last"] = (long["Date"] - long["prev_date"]).dt.days
    # a genuine data-quality dupe (identical Tournament/Date/Round/player/opponent twice) would
    # otherwise silently multiply rows on the merge below - collapse defensively rather than let
    # that happen
    return long[["edition_id", "Date", "Round", "player", "opponent", "days_since_last"]].drop_duplicates(
        subset=["edition_id", "Date", "Round", "player", "opponent"]
    ).rename(columns={"Date": "date", "Round": "round"})


def build_layoff_dataset(matches, preds):
    layoff = compute_days_since_last_match(matches)
    # validate="one_to_one" is the real check here: raises loudly if either side turns out to have
    # a duplicate (edition_id, date, round, player, opponent) key, instead of silently
    # duplicating/dropping rows and quietly corrupting the residual estimates downstream.
    merged = preds.merge(
        layoff, on=["edition_id", "date", "round", "player", "opponent"], how="left", validate="one_to_one"
    )
    assert len(merged) == len(preds), "layoff merge changed row count - duplicate match keys somewhere"
    merged["bucket"] = merged["days_since_last"].apply(bucket_for_days)
    return merged


def run(tour):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)
    layoff_df = build_layoff_dataset(matches, preds)

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    train = layoff_df[layoff_df["edition_id"].isin(train_editions)]
    test = layoff_df[layoff_df["edition_id"].isin(test_editions)]
    print(f"{tour}: {len(layoff_df)} player-perspective rows ({len(train)} train-era, "
          f"{len(test)} test-era), split at the same {len(train_editions)}/{len(test_editions)} "
          f"tournament-edition boundary as the elite-opponent and gap-overcome tests")

    train_summary = pd.DataFrame([summarize_bucket(b, g) for b, g in train.groupby("bucket") if len(g)]) \
        .set_index("bucket").reindex(BUCKET_ORDER).reset_index()

    print("\n--- Train-era residual by days-since-last-match bucket (what the model gets wrong, "
          "before any adjustment) ---")
    print(train_summary.to_string(index=False, formatters={
        "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format, "residual": "{:+.1%}".format,
        "residual_ci_lo": "{:+.1%}".format, "residual_ci_hi": "{:+.1%}".format, "z": "{:.2f}".format,
    }))

    # held-out validation: fit a logit shift per bucket on train, apply to the matching bucket's
    # test-era rows, compare log-loss/Brier to raw (unadjusted) Elo. Player-clustered bootstrap CI.
    shift_by_bucket = {}
    for b in BUCKET_ORDER:
        g = train[train["bucket"] == b]
        if len(g) == 0:
            continue
        actual_rate, pred_rate = g["actual_win"].mean(), g["pred_win"].mean()
        shift_by_bucket[b] = logit(actual_rate) - logit(pred_rate)

    test_col = test[test["bucket"].isin(shift_by_bucket)].copy()
    test_col["adjusted_pred"] = test_col.apply(
        lambda r: sigmoid(logit(r["pred_win"]) + shift_by_bucket[r["bucket"]]), axis=1)
    test_col["raw_loss"] = log_loss(test_col["actual_win"].values, test_col["pred_win"].values)
    test_col["adj_loss"] = log_loss(test_col["actual_win"].values, test_col["adjusted_pred"].values)
    test_col["raw_brier"] = (test_col["actual_win"] - test_col["pred_win"]) ** 2
    test_col["adj_brier"] = (test_col["actual_win"] - test_col["adjusted_pred"]) ** 2

    observed, lo, hi = cluster_bootstrap_ci(test_col, "raw_loss", "adj_loss")

    print(f"\n--- Held-out validation: {len(test_col)} test-era rows, {test_col['player'].nunique()} players ---")
    print(f"  Raw Elo          : log-loss = {test_col['raw_loss'].mean():.4f}, Brier = {test_col['raw_brier'].mean():.4f}")
    print(f"  Bucket-adjusted  : log-loss = {test_col['adj_loss'].mean():.4f}, Brier = {test_col['adj_brier'].mean():.4f}")
    print(f"  Mean per-match log-loss improvement (raw - adjusted, >0 = adjustment better), "
          f"player-clustered: {observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")

    print("\n  Where the improvement actually comes from (per-bucket breakdown of the same test rows):")
    by_bucket = test_col.groupby("bucket").apply(
        lambda g: pd.Series({
            "n": len(g), "raw_loss": g["raw_loss"].mean(), "adj_loss": g["adj_loss"].mean(),
            "improvement": (g["raw_loss"] - g["adj_loss"]).mean(),
        }), include_groups=False,
    ).reindex(BUCKET_ORDER).reset_index()
    print(by_bucket.to_string(index=False, formatters={
        "raw_loss": "{:.4f}".format, "adj_loss": "{:.4f}".format, "improvement": "{:+.4f}".format,
    }))

    # direct test of the hypothesis's own specific claim: does residual decrease (i.e. get more
    # NEGATIVE - actual win rate falling further below Elo's prediction) monotonically as the
    # layoff gets longer? (excluding no_prior_match, which isn't part of the layoff gradient claim
    # at all - it's a different population: a player with zero recorded history, not a known
    # returning player.)
    gradient = train_summary[train_summary["bucket"] != "no_prior_match"].reset_index(drop=True)
    is_monotonic = gradient["residual"].is_monotonic_decreasing
    print(f"\nDoes the train-era residual decrease monotonically under_14d -> 14_30d -> 30_60d -> "
          f"60_90d -> 90d_plus, as the 'longer layoff = more underperformance vs. Elo' hypothesis "
          f"specifically predicts? {'YES' if is_monotonic else 'NO'} "
          f"({', '.join(f'{r:+.1%}' for r in gradient['residual'])})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
    args = parser.parse_args()
    run(args.tour)
