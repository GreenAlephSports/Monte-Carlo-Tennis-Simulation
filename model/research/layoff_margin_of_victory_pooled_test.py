"""Pools ATP + WTA together into one combined test population for the margin-of-victory decay
hypothesis (layoff_margin_of_victory_test.py), per direct request: the underlying hypothesis -
does match_1 margin of victory predict whether the match_2 layoff residual still shows up - isn't
tour-specific in principle, and neither tour cleared the held-out significance bar on its own with
~250 90d_plus-starting players each (ATP +0.0029 CI [-0.0022,+0.0083], WTA +0.0007 CI
[-0.0036,+0.0051], both straddling zero). Pooling roughly doubles the effective n without needing
any new data.

tour is kept as a control variable, not dropped: this script fits BOTH a pooled (tour-blind) shift
per margin bucket and a tour-interaction (margin_bucket x tour) shift on the same train data, then
compares them held-out - if tour genuinely doesn't matter, the simpler pooled fit should do at
least as well as the interaction fit, which has to split its (necessarily smaller) samples four
ways instead of two.

Everything else matches layoff_margin_of_victory_test.py exactly: frozen per-tournament-edition
Elo, a chronological 80/20 train/test split done PER TOUR (never a random or cross-tour split -
each tour's own tournament calendar defines its own train/test boundary, then the two tours' train
rows are pooled and the two tours' test rows are pooled), held-out validation against raw Elo AND
the CURRENT production flat shift (which is tour-specific - ATP and WTA have different fitted
90d_plus penalties - so the comparison uses each row's own tour's flat shift, not a single global
number), player-clustered bootstrap CIs (clustered on tour+player, since a player name could in
principle - if vanishingly unlikely - collide across tours and shouldn't be pooled as one cluster).

Usage:
    python model/research/layoff_margin_of_victory_pooled_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import TRAIN_FRACTION, build_frozen_predictions, log_loss, logit, sigmoid  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from layoff_margin_of_victory_test import attach_prior_margin, build_margin_lookup  # noqa: E402
from layoff_test import build_layoff_dataset  # noqa: E402
from layoff_within_tournament_decay_test import build_within_tournament_sequences, fit_shifts  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci, summarize_bucket  # noqa: E402
from win_probability import LAYOFF_BUCKET_EDGES_ATP, LAYOFF_BUCKET_EDGES_WTA  # noqa: E402

MARGIN_BUCKETS = ["dominant_win_straight_sets", "grinding_win_needed_decider"]


def score_pooled(df, bucket_col, shift_by_bucket):
    """Same shape as layoff_within_tournament_decay_test.score, but the 'current production flat
    shift' comparison point is read per-row from current_flat_shift (tour-specific) instead of a
    single scalar - ATP and WTA currently have different fitted 90d_plus shifts."""
    out = df[df[bucket_col].isin(shift_by_bucket)].copy()
    out["adjusted_pred"] = out.apply(lambda r: sigmoid(logit(r["pred_win"]) + shift_by_bucket[r[bucket_col]]), axis=1)
    out["flat_pred"] = out.apply(lambda r: sigmoid(logit(r["pred_win"]) + r["current_flat_shift"]), axis=1)
    out["raw_loss"] = log_loss(out["actual_win"].values, out["pred_win"].values)
    out["adj_loss"] = log_loss(out["actual_win"].values, out["adjusted_pred"].values)
    out["flat_loss"] = log_loss(out["actual_win"].values, out["flat_pred"].values)
    out["raw_brier"] = (out["actual_win"] - out["pred_win"]) ** 2
    out["adj_brier"] = (out["actual_win"] - out["adjusted_pred"]) ** 2
    out["flat_brier"] = (out["actual_win"] - out["flat_pred"]) ** 2
    return out


def build_tour_dataset(tour, match_numbers):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)
    layoff_df = build_layoff_dataset(matches, preds)
    seq = build_within_tournament_sequences(layoff_df)
    margin_lookup = build_margin_lookup(matches)
    seq = attach_prior_margin(seq, margin_lookup)

    treatment = seq[seq["first_match_bucket"] == "90d_plus"].copy()
    pop = treatment[
        treatment["match_number"].isin(match_numbers) & (treatment["prior_score_consistent"] == True)  # noqa: E712
    ].copy()
    pop["margin_bucket"] = np.where(pop["prior_straight_sets"], "dominant_win_straight_sets", "grinding_win_needed_decider")

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    pop["is_train"] = pop["edition_id"].isin(train_editions)
    pop["is_test"] = pop["edition_id"].isin(test_editions)
    pop["tour"] = tour
    pop["player_id"] = tour + "::" + pop["player"]

    bucket_edges = LAYOFF_BUCKET_EDGES_ATP if tour == "ATP" else LAYOFF_BUCKET_EDGES_WTA
    pop["current_flat_shift"] = next(shift for name, _test, shift in bucket_edges if name == "90d_plus")

    n_m1 = len(treatment[treatment["match_number"] == 1])
    print(f"  {tour}: {n_m1} players started in 90d_plus, {len(pop)} usable rows for match_number "
          f"in {sorted(match_numbers)} with a score-consistent prior-match margin "
          f"(margin split: {dict(pop['margin_bucket'].value_counts())})")
    return pop


def run_pooled(label, match_numbers, min_confidence=20):
    print(f"\n{'=' * 90}\n{label}\n{'=' * 90}")
    per_tour = [build_tour_dataset(tour, match_numbers) for tour in ("ATP", "WTA")]
    pooled = pd.concat(per_tour, ignore_index=True)
    train, test = pooled[pooled["is_train"]], pooled[pooled["is_test"]]
    print(f"Combined: {len(pooled)} total rows ({len(train)} train-era / {len(test)} test-era) "
          f"across both tours, vs. ~half that testing either tour alone")

    if len(train) < min_confidence or len(test) < min_confidence:
        print(f"\n  Still below the {min_confidence}-row floor even pooled - inconclusive by "
              f"sample size, not by the hypothesis failing.")
        return

    # --- train-era residual by margin bucket, pooled (primary read) ---
    train_summary = pd.DataFrame([summarize_bucket(b, g) for b, g in train.groupby("margin_bucket") if len(g)])
    print("\n--- Train-era residual by match margin, POOLED across ATP+WTA (primary read) ---")
    print(train_summary.to_string(index=False, formatters={
        "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format, "residual": "{:+.1%}".format,
        "residual_ci_lo": "{:+.1%}".format, "residual_ci_hi": "{:+.1%}".format, "z": "{:.2f}".format,
    }))

    # --- same breakdown per tour, side by side - is there a real tour-specific difference, or is
    # pooling defensible? ---
    print("\n--- Same breakdown, PER TOUR (context: is there a real tour-specific difference?) ---")
    for tour, g_tour in train.groupby("tour"):
        tour_summary = pd.DataFrame([summarize_bucket(b, g) for b, g in g_tour.groupby("margin_bucket") if len(g)])
        print(f"  {tour}:")
        print(tour_summary.to_string(index=False, formatters={
            "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format, "residual": "{:+.1%}".format,
        }).replace("\n", "\n  "))

    # --- held-out: pooled (tour-blind) fit vs. current flat (tour-specific) vs. raw Elo ---
    shift_pooled = fit_shifts(train, "margin_bucket", MARGIN_BUCKETS)
    for b, s in shift_pooled.items():
        print(f"  Fitted POOLED shift, {b}: {s:+.4f} logits (n={len(train[train['margin_bucket'] == b])})")

    scored = score_pooled(test, "margin_bucket", shift_pooled)
    if len(scored) == 0:
        print("\n  No test-era rows survived - inconclusive by sample size.")
        return

    print(f"\n--- Held-out validation: {len(scored)} test-era rows, {scored['player_id'].nunique()} "
          f"players across both tours ---")
    print(f"  Raw Elo (no layoff adj.)     : log-loss = {scored['raw_loss'].mean():.4f}, Brier = {scored['raw_brier'].mean():.4f}")
    print(f"  Current flat shift (per-tour): log-loss = {scored['flat_loss'].mean():.4f}, Brier = {scored['flat_brier'].mean():.4f}")
    print(f"  Pooled margin-adjusted       : log-loss = {scored['adj_loss'].mean():.4f}, Brier = {scored['adj_brier'].mean():.4f}")

    obs_flat, lo_flat, hi_flat = cluster_bootstrap_ci(scored, "flat_loss", "adj_loss", group_col="player_id")
    print(f"  Pooled margin-adjusted vs. CURRENT flat shift, player-clustered improvement "
          f"(flat - pooled_adj, >0 = pooled model better): {obs_flat:+.4f}, 95% CI [{lo_flat:+.4f}, {hi_flat:+.4f}]")
    obs_raw, lo_raw, hi_raw = cluster_bootstrap_ci(scored, "raw_loss", "adj_loss", group_col="player_id")
    print(f"  Pooled margin-adjusted vs. raw Elo, player-clustered improvement "
          f"(raw - pooled_adj, >0 = pooled model better): {obs_raw:+.4f}, 95% CI [{lo_raw:+.4f}, {hi_raw:+.4f}]")

    # --- does modeling tour separately (margin x tour interaction) beat the simpler pooled fit? ---
    train_i = train.copy()
    train_i["bucket_tour"] = train_i["margin_bucket"] + "::" + train_i["tour"]
    interaction_buckets = [f"{b}::{t}" for b in MARGIN_BUCKETS for t in ("ATP", "WTA")]
    shift_interaction = fit_shifts(train_i, "bucket_tour", interaction_buckets)
    test_i = test.copy()
    test_i["bucket_tour"] = test_i["margin_bucket"] + "::" + test_i["tour"]
    scored_interaction = score_pooled(test_i, "bucket_tour", shift_interaction)

    common = scored.index.intersection(scored_interaction.index)
    if len(common) and len(shift_interaction) == len(interaction_buckets):
        h2h = scored.loc[common, ["player_id"]].copy()
        h2h["pooled_loss"] = scored.loc[common, "adj_loss"].values
        h2h["interaction_loss"] = scored_interaction.loc[common, "adj_loss"].values
        obs_h2h, lo_h2h, hi_h2h = cluster_bootstrap_ci(h2h, "interaction_loss", "pooled_loss", group_col="player_id")
        print(f"\n  Control check - does splitting by tour (margin x tour interaction fit) beat the "
              f"simpler pooled fit? Player-clustered improvement (interaction - pooled, >0 = pooled "
              f"model, i.e. tour doesn't add anything, is BETTER): {obs_h2h:+.4f}, "
              f"95% CI [{lo_h2h:+.4f}, {hi_h2h:+.4f}]")
    else:
        print("\n  Control check skipped: at least one tour x margin-bucket cell is empty in train "
              "or test - not enough data to fit the interaction model at all, which is itself "
              "informative about how thin this cut gets once split four ways.")

    if len(test) < 30:
        print(f"\n  CAUTION: n={len(test)} test-era rows, while roughly double either tour alone, "
              f"is still below the ~30-row floor this codebase's other calibration checks require - "
              f"directional, not a settled verdict, even pooled.")


if __name__ == "__main__":
    run_pooled("PRIMARY (pooled): match_1 margin of victory -> match_2 residual, ATP+WTA combined", {2})
    run_pooled(
        "SUPPLEMENTARY (pooled, more power): prior-match margin -> residual, "
        "pooled across match_number >= 2, ATP+WTA combined",
        set(range(2, 20)),
    )
