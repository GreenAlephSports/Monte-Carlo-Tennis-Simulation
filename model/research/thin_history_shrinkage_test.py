"""Second fix attempt for thin-history calibration, after thin_history_rank_blend_test.py's
rank-blend held up to nothing (no significant held-out effect, and actively worse in the 0-2
career-match bucket for ATP - the exact population it was meant to help).

Different hypothesis this time, and deliberately simpler: the pipeline already shrinks a thin
SURFACE rating toward the player's own overall_elo (elo_ratings.SURFACE_BLEND_K), but never
questions overall_elo itself. For a player with very few TOTAL career matches, overall_elo sits
close to STARTING_ELO=1500 largely by construction (few K-factor updates have moved it anywhere) -
not because 1500 is a good estimate of that player's true skill, just because the model hasn't seen
enough of them to move off the default. This test asks: does shrinking a thin player's overall_elo
toward the population's real average Elo (not toward rank, not toward anything external - just "we
don't know much about you, so assume you're closer to typical than your barely-updated number
suggests") help, using the exact same weighted-shrinkage FORM the pipeline already validated
(weight = matches / (matches + K)), but fitting its own K2 rather than reusing SURFACE_BLEND_K.

Methodology (same rigor as every test tonight, same dataset construction reused directly from
thin_history_rank_blend_test.build_dataset - frozen per-edition Elo, chronological tournament-
edition 80/20 split):
  - The shrinkage prior is the REAL mean overall_elo of the train-era "solid" population (>=30
    career matches - the same trustworthy-player definition used to fit the rank map in the prior
    test), not a bare assumption of 1500 - stated and reported, not hardcoded blindly.
  - K2 is FIT on train-era data only, via grid search minimizing log-loss on the train-era THIN
    population specifically (the population this correction targets, not the whole dataset -
    fitting against the general population would let a K2 that's good on ordinary matches mask
    doing nothing useful for thin ones). Never touches test-era data during fitting.
  - Held out on test-era thin population only, log-loss/Brier vs. raw, player-clustered bootstrap
    CI (survivorship_upset_test.cluster_bootstrap_ci) - identical validation discipline to the
    rank-blend test, so the two results are directly comparable.
  - Same "does it actually tame the extreme predictions" check that correctly killed the rank-blend
    idea: share of held-out thin-population predictions more extreme than 70/30, before vs after.

Usage:
    python model/research/thin_history_shrinkage_test.py [--tour ATP|WTA] [--thin-threshold N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import TRAIN_FRACTION, log_loss  # noqa: E402
from elo_ratings import STARTING_ELO, load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from thin_history_rank_blend_test import SOLID_MATCHES, build_dataset  # noqa: E402

DEFAULT_THIN_THRESHOLD = 10
K2_GRID = [3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]


def expected_score_vec(a, b):
    return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))


def is_thin(df, threshold):
    return (df["player_matches_before"] < threshold) | (df["opponent_matches_before"] < threshold)


def shrink_elo(raw_elo, matches_before, prior, k2, threshold):
    """weight = matches / (matches + k2), same functional form as elo_ratings.SURFACE_BLEND_K's
    surface-to-overall blend - just applied to overall_elo itself, toward a population prior,
    gated to only players below the thin threshold (a no-op above it, by construction)."""
    weight = np.where(matches_before < threshold, matches_before / (matches_before + k2), 1.0)
    return weight * raw_elo + (1 - weight) * prior


def apply_shrinkage(df, prior, k2, threshold):
    player_shrunk = shrink_elo(df["player_elo"].to_numpy(float), df["player_matches_before"].to_numpy(float), prior, k2, threshold)
    opp_shrunk = shrink_elo(df["opponent_elo"].to_numpy(float), df["opponent_matches_before"].to_numpy(float), prior, k2, threshold)
    return expected_score_vec(player_shrunk, opp_shrunk)


def run(tour, thin_threshold):
    matches = load_matches_for_tour(tour)
    preds, editions = build_dataset(matches)

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    train = preds[preds["edition_id"].isin(train_editions)].copy()
    test = preds[preds["edition_id"].isin(test_editions)].copy()
    print(f"{tour}: {len(editions)} tournament editions, train = first {len(train_editions)} "
          f"(through {editions['edition_start'].iloc[split_idx - 1].date()}), "
          f"test = remaining {len(test_editions)} (from {editions['edition_start'].iloc[split_idx].date()})")
    print(f"{len(train)} train-era rows, {len(test)} test-era rows\n")

    # real population prior, not a bare assumption of 1500 - mean overall_elo of the train-era
    # "solid" population (>=30 career matches), deduped to one observation per (player, edition)
    solid = train[train["player_matches_before"] >= SOLID_MATCHES].drop_duplicates(subset=["player", "edition_id"])
    prior = solid["player_elo"].mean()
    print(f"Shrinkage prior: mean overall_elo of {len(solid)} train-era solid-player-edition "
          f"observations (>= {SOLID_MATCHES} career matches) = {prior:.1f} "
          f"(vs. STARTING_ELO={STARTING_ELO} - {'close to' if abs(prior - STARTING_ELO) < 15 else 'notably different from'} the raw default)")

    train_thin = train[is_thin(train, thin_threshold)].copy()
    print(f"Train-era thin population (>= 1 side < {thin_threshold} career matches): "
          f"{len(train_thin)} of {len(train)} rows ({len(train_thin) / len(train):.1%})\n")

    # fit K2 on TRAIN thin population only - grid search, log-loss
    print(f"--- Fitting K2 via grid search on train-era thin population only ---")
    best_k2, best_loss = None, float("inf")
    fit_rows = []
    for k2 in K2_GRID:
        pred = apply_shrinkage(train_thin, prior, k2, thin_threshold)
        loss = log_loss(train_thin["actual_win"].values, pred).mean()
        fit_rows.append((k2, loss))
        if loss < best_loss:
            best_k2, best_loss = k2, loss
    raw_train_loss = log_loss(train_thin["actual_win"].values, train_thin["pred_win"].values).mean()
    print(f"  Raw Elo train-era thin-population log-loss (baseline to beat): {raw_train_loss:.4f}")
    for k2, loss in fit_rows:
        flag = "  <- best" if k2 == best_k2 else ""
        print(f"  K2={k2:>4}: train thin-population log-loss = {loss:.4f}{flag}")
    print(f"\n  Fitted K2 = {best_k2} (train thin-pop log-loss {best_loss:.4f} vs raw {raw_train_loss:.4f}, "
          f"{'IMPROVES' if best_loss < raw_train_loss else 'DOES NOT IMPROVE'} train-era fit)")

    # held-out validation on TEST thin population, using the fitted K2 - never touched during fitting
    test_thin = test[is_thin(test, thin_threshold)].copy()
    test_thin["shrunk_pred"] = apply_shrinkage(test_thin, prior, best_k2, thin_threshold)
    test_thin["raw_loss"] = log_loss(test_thin["actual_win"].values, test_thin["pred_win"].values)
    test_thin["shrunk_loss"] = log_loss(test_thin["actual_win"].values, test_thin["shrunk_pred"].values)
    test_thin["raw_brier"] = (test_thin["actual_win"] - test_thin["pred_win"]) ** 2
    test_thin["shrunk_brier"] = (test_thin["actual_win"] - test_thin["shrunk_pred"]) ** 2

    print(f"\n--- Held-out validation on test-era thin population (n={len(test_thin)}), K2={best_k2} ---")
    print(f"  Raw Elo (current pipeline) : log-loss = {test_thin['raw_loss'].mean():.4f}, "
          f"Brier = {test_thin['raw_brier'].mean():.4f}")
    print(f"  Elo-shrunk                 : log-loss = {test_thin['shrunk_loss'].mean():.4f}, "
          f"Brier = {test_thin['shrunk_brier'].mean():.4f}")

    observed, lo, hi = cluster_bootstrap_ci(test_thin, "raw_loss", "shrunk_loss")
    print(f"  Mean per-row log-loss improvement (raw - shrunk, >0 = shrinkage better), "
          f"player-clustered: {observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    verdict = "IMPROVES" if lo > 0 else ("HURTS" if hi < 0 else "NO SIGNIFICANT EFFECT (CI straddles zero)")
    print(f"  -> Overall_elo shrinkage {verdict} calibration on the thin population, held out.")

    print(f"\n--- Where along 0-{thin_threshold} the shrinkage's effect actually lands ---")
    bucket_edges = [("0-2", lambda n: n <= 2), ("3-5", lambda n: 3 <= n <= 5),
                    (f"6-{thin_threshold - 1}", lambda n: 6 <= n < thin_threshold)]
    thinner_side_matches = test_thin[["player_matches_before", "opponent_matches_before"]].min(axis=1)
    for label, test_fn in bucket_edges:
        bucket = test_thin[thinner_side_matches.apply(test_fn)]
        if len(bucket) == 0:
            print(f"  {label:>6} career matches: n=0, skipped")
            continue
        extreme_raw = (bucket["pred_win"].sub(0.5).abs() > 0.3).mean()
        extreme_shrunk = (bucket["shrunk_pred"].sub(0.5).abs() > 0.3).mean()
        print(f"  {label:>6} career matches (n={len(bucket):>4}): raw log-loss={bucket['raw_loss'].mean():.4f}  "
              f"shrunk log-loss={bucket['shrunk_loss'].mean():.4f}  "
              f"share of |pred-50%|>30pp: raw={extreme_raw:.1%} -> shrunk={extreme_shrunk:.1%}")

    print(f"\n--- Most extreme raw predictions in the thin test population ---")
    extreme_rows = test_thin.reindex(test_thin["pred_win"].sub(0.5).abs().sort_values(ascending=False).index).head(10)
    print(extreme_rows[["player", "opponent", "player_matches_before", "opponent_matches_before",
                         "pred_win", "shrunk_pred", "actual_win"]]
          .to_string(index=False, formatters={"pred_win": "{:.1%}".format, "shrunk_pred": "{:.1%}".format}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
    parser.add_argument("--thin-threshold", type=int, default=DEFAULT_THIN_THRESHOLD)
    args = parser.parse_args()
    run(args.tour, args.thin_threshold)
