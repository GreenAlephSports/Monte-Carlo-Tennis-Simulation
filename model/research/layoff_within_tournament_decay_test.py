"""Tests whether the 90d+ layoff penalty should DECAY within a tournament, rather than being
applied as the same flat shift to every round the way win_probability.py currently does it: since
days_since_last_match is frozen at the pre-tournament ratings snapshot and never refreshed
mid-event (see get_days_since_last_match's docstring), a player who started 90+ days rusty
currently eats the identical -0.2145 (ATP) / -0.2500 (WTA) logit penalty in their 4th-round match
as in their 1st-round match, even though by round 4 they've played 3 real competitive matches since
that layoff ended.

Population: players whose FIRST match of a tournament edition falls in the 90d_plus layoff bucket
(same bucketing as layoff_test.py, computed from real Kaggle match dates - this is exactly the
population LAYOFF_BUCKET_EDGES_*'s 90d_plus shift targets). For every match this player plays in
that same tournament (round 1, 2, 3, ... until eliminated), residual = actual_win - pred_win is
computed against RAW frozen-per-edition Elo (via elite_opponent_residual_test.build_frozen_predictions)
- deliberately WITHOUT any layoff adjustment baked in, so this measures the same underperformance
the flat shift is meant to capture, at every match number, not just the first.

Two decay drivers are compared head-to-head on match_number >= 2 rows only (match 1 has no
"intervening tournament performance" to condition on):
  - match-count decay: shift depends only on how many matches into the tournament they are
    (match_2 / match_3 / match_4_plus) - "rust wears off with time/matches played," full stop.
  - performance-based decay: shift depends on whether they've actually been beating their own
    frozen-Elo prediction so far this tournament (cumulative actual - cumulative pred over all
    PRIOR matches this event) - "rust wears off when you show you're playing well," not just when
    the calendar/round advances.

Same rigor as every prior script in this series: frozen per-tournament-edition Elo (reused
directly from elite_opponent_residual_test, no in-tournament Elo lookahead), chronological
tournament-level 80/20 train/test split, held-out validation against both raw Elo AND the current
production flat-shift model, and player-clustered bootstrap confidence intervals (one frequent
long-layoff returner shouldn't count as N independent data points).

Usage:
    python model/research/layoff_within_tournament_decay_test.py [--tour ATP|WTA]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import TRAIN_FRACTION, build_frozen_predictions, log_loss, logit, sigmoid  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from layoff_test import build_layoff_dataset  # noqa: E402
from survivorship_upset_test import ROUND_ORDER, cluster_bootstrap_ci, summarize_bucket  # noqa: E402
from win_probability import LAYOFF_BUCKET_EDGES_ATP, LAYOFF_BUCKET_EDGES_WTA  # noqa: E402

MIN_SAMPLE_FOR_ANY_CONFIDENCE = 30

MATCH_NUMBER_BUCKETS = ["match_1", "match_2", "match_3", "match_4_plus"]


def match_number_bucket(n):
    if n >= 4:
        return "match_4_plus"
    return f"match_{n}"


def build_within_tournament_sequences(layoff_df):
    """One row per (edition, player, match) for every player, sequenced by round within their own
    tournament run - same single-elim traversal as survivorship_upset_test.build_upset_dataset
    (Round Robin excluded, stop after a player's first loss). Unlike that function, this keeps
    every row (not just post-first-match ones) and carries match_number plus two decay candidates:
    the player's OWN days_since_last-bucket at match 1 of this tournament (the treatment label),
    and prior_gap_overcome = opponent_elo - player_elo in their immediately PRECEDING match this
    tournament (None for match_number == 1, which has no preceding match yet). This is deliberately
    the same "Elo gap overcome in their most recent win" signal survivorship_upset_test.py uses,
    not a raw actual-vs-predicted residual: every row with match_number >= 2 belongs to a player who
    necessarily WON their previous match (single elimination), so cumulative actual-minus-predicted
    would be trivially positive for nearly the whole population - not a real performance signal.
    Gap overcome doesn't have that problem: a heavy favorite winning as expected has a gap near/
    below zero, while an underdog who just upset a much-higher-Elo opponent has a large positive
    gap - real variation, not a survivorship artifact."""
    df = layoff_df[layoff_df["round"].isin(ROUND_ORDER)].copy()
    df["round_order"] = df["round"].map(ROUND_ORDER)

    rows = []
    for (edition_id, player), g in df.sort_values("round_order").groupby(["edition_id", "player"], sort=False):
        match_number = 0
        prev_gap = None  # opponent_elo - player_elo in their most recent WIN this tournament
        first_match_bucket = None
        for row in g.itertuples(index=False):
            match_number += 1
            if match_number == 1:
                first_match_bucket = row.bucket
            rows.append((
                edition_id, player, row.opponent, row.date, row.round, match_number, first_match_bucket,
                prev_gap, row.pred_win, row.actual_win,
            ))
            if row.actual_win == 0:
                break  # eliminated - no legitimate further row for them this tournament
            prev_gap = row.opponent_elo - row.player_elo

    return pd.DataFrame(rows, columns=[
        "edition_id", "player", "opponent", "date", "round", "match_number", "first_match_bucket",
        "prior_gap_overcome", "pred_win", "actual_win",
    ])


def fit_shifts(train, bucket_col, buckets):
    shift_by_bucket = {}
    for b in buckets:
        g = train[train[bucket_col] == b]
        if len(g) == 0:
            continue
        actual_rate, pred_rate = g["actual_win"].mean(), g["pred_win"].mean()
        shift_by_bucket[b] = logit(actual_rate) - logit(pred_rate)
    return shift_by_bucket


def score(df, bucket_col, shift_by_bucket, flat_shift=None):
    """Adds adjusted_pred / raw_loss / adj_loss / flat_loss (if flat_shift given) columns."""
    out = df[df[bucket_col].isin(shift_by_bucket)].copy()
    out["adjusted_pred"] = out.apply(
        lambda r: sigmoid(logit(r["pred_win"]) + shift_by_bucket[r[bucket_col]]), axis=1)
    out["raw_loss"] = log_loss(out["actual_win"].values, out["pred_win"].values)
    out["adj_loss"] = log_loss(out["actual_win"].values, out["adjusted_pred"].values)
    out["raw_brier"] = (out["actual_win"] - out["pred_win"]) ** 2
    out["adj_brier"] = (out["actual_win"] - out["adjusted_pred"]) ** 2
    if flat_shift is not None:
        out["flat_pred"] = out["pred_win"].apply(lambda p: sigmoid(logit(p) + flat_shift))
        out["flat_loss"] = log_loss(out["actual_win"].values, out["flat_pred"].values)
        out["flat_brier"] = (out["actual_win"] - out["flat_pred"]) ** 2
    return out


def run(tour):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)
    layoff_df = build_layoff_dataset(matches, preds)
    seq = build_within_tournament_sequences(layoff_df)

    treatment = seq[seq["first_match_bucket"] == "90d_plus"].copy()
    treatment["mn_bucket"] = treatment["match_number"].apply(match_number_bucket)
    print(f"{tour}: {treatment['player'].nunique()} distinct players with >=1 tournament run that "
          f"started in the 90d_plus layoff bucket, {len(treatment)} total in-tournament match rows "
          f"across those runs (match_number distribution: "
          f"{dict(treatment['mn_bucket'].value_counts().reindex(MATCH_NUMBER_BUCKETS).fillna(0).astype(int))})")

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    train = treatment[treatment["edition_id"].isin(train_editions)]
    test = treatment[treatment["edition_id"].isin(test_editions)]
    print(f"Same tournament-edition boundary as the rest of this series: {len(train)} train-era "
          f"rows, {len(test)} test-era rows\n")

    # --- 1. train-era residual by match number: does it shrink toward zero as the tournament goes on? ---
    train_summary = pd.DataFrame([summarize_bucket(b, g) for b, g in train.groupby("mn_bucket") if len(g)]) \
        .set_index("bucket").reindex(MATCH_NUMBER_BUCKETS).reset_index()
    print("--- Train-era residual (actual - raw-Elo-predicted) by match number within the "
          "tournament, 90d_plus-starting players only ---")
    print(train_summary.to_string(index=False, formatters={
        "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format, "residual": "{:+.1%}".format,
        "residual_ci_lo": "{:+.1%}".format, "residual_ci_hi": "{:+.1%}".format, "z": "{:.2f}".format,
    }))
    gradient = train_summary.dropna(subset=["residual"]).reset_index(drop=True)
    is_decaying = gradient["residual"].is_monotonic_increasing
    print(f"\nDoes the residual move monotonically toward zero (increase) match_1 -> match_2 -> "
          f"match_3 -> match_4_plus, as the 'rust wears off as the tournament goes on' hypothesis "
          f"predicts? {'YES' if is_decaying else 'NO'} "
          f"({', '.join(f'{r:+.1%}' for r in gradient['residual'])})")

    # --- 2. held-out: match-number-decay-aware model vs. the CURRENT flat single-shift production
    #    model (win_probability.py applies the same 90d_plus shift to every round) vs. raw Elo ---
    bucket_edges = LAYOFF_BUCKET_EDGES_ATP if tour == "ATP" else LAYOFF_BUCKET_EDGES_WTA
    flat_shift = next(shift for name, _test, shift in bucket_edges if name == "90d_plus")
    print(f"\nCurrent production flat shift for 90d_plus ({tour}): {flat_shift:+.4f} logits, "
          f"applied identically to every round of the tournament")

    shift_by_mn = fit_shifts(train, "mn_bucket", MATCH_NUMBER_BUCKETS)
    for b, s in shift_by_mn.items():
        print(f"  Fitted train-era match-number shift, {b}: {s:+.4f} logits (n={len(train[train['mn_bucket'] == b])})")

    test_scored = score(test, "mn_bucket", shift_by_mn, flat_shift=flat_shift)
    print(f"\n--- Held-out validation: {len(test_scored)} test-era rows, "
          f"{test_scored['player'].nunique()} players ---")
    print(f"  Raw Elo (no layoff adj.)   : log-loss = {test_scored['raw_loss'].mean():.4f}, "
          f"Brier = {test_scored['raw_brier'].mean():.4f}")
    print(f"  Current flat 90d_plus shift: log-loss = {test_scored['flat_loss'].mean():.4f}, "
          f"Brier = {test_scored['flat_brier'].mean():.4f}")
    print(f"  Match-number decay-aware   : log-loss = {test_scored['adj_loss'].mean():.4f}, "
          f"Brier = {test_scored['adj_brier'].mean():.4f}")

    obs_vs_flat, lo_vs_flat, hi_vs_flat = cluster_bootstrap_ci(test_scored, "flat_loss", "adj_loss")
    print(f"  Decay-aware vs. CURRENT flat shift, player-clustered improvement "
          f"(flat - decay_aware, >0 = decay-aware better): {obs_vs_flat:+.4f}, "
          f"95% bootstrap CI [{lo_vs_flat:+.4f}, {hi_vs_flat:+.4f}]")
    obs_vs_raw, lo_vs_raw, hi_vs_raw = cluster_bootstrap_ci(test_scored, "raw_loss", "adj_loss")
    print(f"  Decay-aware vs. raw Elo, player-clustered improvement "
          f"(raw - decay_aware, >0 = decay-aware better): {obs_vs_raw:+.4f}, "
          f"95% bootstrap CI [{lo_vs_raw:+.4f}, {hi_vs_raw:+.4f}]")

    if len(test) < MIN_SAMPLE_FOR_ANY_CONFIDENCE:
        print(f"\n  CAUTION: n={len(test)} test-era rows is well below the "
              f"~{MIN_SAMPLE_FOR_ANY_CONFIDENCE}-row floor this codebase's other calibration checks "
              f"require - 90d_plus-starting tournament runs are rare by construction, and this read "
              f"should be treated as early/directional, not a settled verdict.")

    # --- 3. what actually drives the decay: match count alone, or how the player has actually
    #    performed in this tournament so far? Head-to-head on match_number >= 2 rows only - match 1
    #    has no "prior tournament performance" to condition on, so it's excluded from this specific
    #    comparison (both candidate models agree there; the comparison is about what happens next). ---
    print(f"\n{'=' * 90}\nMatch-count decay vs. performance-based decay (match_number >= 2 only)\n{'=' * 90}")
    mn2plus = treatment[treatment["match_number"] >= 2].copy()
    # gap > 0: beat a higher-Elo opponent in their most recent match (real form signal). gap <= 0:
    # won as the (near-)favorite, as raw Elo would already expect - not a form signal either way.
    mn2plus["perf_bucket"] = np.where(
        mn2plus["prior_gap_overcome"] > 0, "beat_stronger_opponent", "won_as_expected_or_favorite")

    train2 = mn2plus[mn2plus["edition_id"].isin(train_editions)]
    test2 = mn2plus[mn2plus["edition_id"].isin(test_editions)]
    print(f"{len(train2)} train-era / {len(test2)} test-era match_number>=2 rows "
          f"(perf_bucket split, train era: "
          f"{dict(train2['perf_bucket'].value_counts())})")

    if len(train2) == 0 or len(test2) == 0:
        print("\n  Not enough match_number>=2 rows for the 90d_plus population to run this "
              "comparison at all - inconclusive by sample size, not by the hypothesis failing.")
        return

    shift_by_mn2 = fit_shifts(train2, "mn_bucket", MATCH_NUMBER_BUCKETS)
    shift_by_perf = fit_shifts(train2, "perf_bucket", ["beat_stronger_opponent", "won_as_expected_or_favorite"])
    for b, s in shift_by_perf.items():
        print(f"  Fitted train-era performance-bucket shift, {b}: {s:+.4f} logits "
              f"(n={len(train2[train2['perf_bucket'] == b])})")

    scored_mn = score(test2, "mn_bucket", shift_by_mn2)
    scored_perf = score(test2, "perf_bucket", shift_by_perf)
    # align on the same row set (both should already cover all of test2, but intersect defensively
    # so the head-to-head bootstrap compares the exact same rows under both models)
    common_idx = scored_mn.index.intersection(scored_perf.index)
    scored_mn, scored_perf = scored_mn.loc[common_idx], scored_perf.loc[common_idx]

    print(f"\n--- Held-out validation: {len(common_idx)} test-era match_number>=2 rows, "
          f"{scored_mn['player'].nunique()} players ---")
    print(f"  Raw Elo               : log-loss = {scored_mn['raw_loss'].mean():.4f}, "
          f"Brier = {scored_mn['raw_brier'].mean():.4f}")
    print(f"  Match-count decay     : log-loss = {scored_mn['adj_loss'].mean():.4f}, "
          f"Brier = {scored_mn['adj_brier'].mean():.4f}")
    print(f"  Performance-based decay: log-loss = {scored_perf['adj_loss'].mean():.4f}, "
          f"Brier = {scored_perf['adj_brier'].mean():.4f}")

    obs_mn, lo_mn, hi_mn = cluster_bootstrap_ci(scored_mn, "raw_loss", "adj_loss")
    print(f"\n  Match-count decay vs. raw Elo, player-clustered improvement: {obs_mn:+.4f}, "
          f"95% CI [{lo_mn:+.4f}, {hi_mn:+.4f}]")
    obs_perf, lo_perf, hi_perf = cluster_bootstrap_ci(scored_perf, "raw_loss", "adj_loss")
    print(f"  Performance-based decay vs. raw Elo, player-clustered improvement: {obs_perf:+.4f}, "
          f"95% CI [{lo_perf:+.4f}, {hi_perf:+.4f}]")

    # direct head-to-head: put both models' adjusted predictions on the same rows and diff them
    head_to_head = scored_mn[["player", "raw_loss"]].copy()
    head_to_head["matchcount_loss"] = scored_mn["adj_loss"].values
    head_to_head["performance_loss"] = scored_perf["adj_loss"].values
    obs_h2h, lo_h2h, hi_h2h = cluster_bootstrap_ci(head_to_head, "matchcount_loss", "performance_loss")
    print(f"\n  Head-to-head: performance-based vs. match-count decay, player-clustered improvement "
          f"(matchcount_loss - performance_loss, >0 = performance-based better): {obs_h2h:+.4f}, "
          f"95% CI [{lo_h2h:+.4f}, {hi_h2h:+.4f}]")

    if len(test2) < MIN_SAMPLE_FOR_ANY_CONFIDENCE:
        print(f"\n  CAUTION: n={len(test2)} test-era match_number>=2 rows is well below the "
              f"~{MIN_SAMPLE_FOR_ANY_CONFIDENCE}-row floor - this head-to-head is directional only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
    args = parser.parse_args()
    run(args.tour)
