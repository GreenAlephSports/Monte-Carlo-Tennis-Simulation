"""Refits the under_14d layoff bucket using ONLY genuine cross-tournament rows (a player's previous
recorded match was in a DIFFERENT tournament edition, within 14 days) - splitting out the ~64-65%
of the original under_14d population (confirmed tonight) that's actually same-tournament round-to-
round timing (a player who just won an earlier round of the SAME event), which is a different,
mostly-uninformative population diluting the pooled fit.

Same rigor as the original layoff_test.py: frozen per-tournament-edition overall Elo (build_frozen_
predictions), chronological tournament-level 80/20 train/test split, player-clustered bootstrap
held-out validation. Only the under_14d bucket's composition changes here - every other bucket
(14_30d, 30_60d, 60_90d, 90d_plus) is untouched, since a same-tournament round gap essentially never
exceeds 14 days in normal single-elimination play, so contamination there is structurally minimal.

Usage:
    python model/research/layoff_under14_cross_tournament_refit.py [--tour ATP|WTA]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import TRAIN_FRACTION, build_frozen_predictions, log_loss, logit, sigmoid  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci, summarize_bucket  # noqa: E402

LAYOFF_BUCKET_EDGES = [
    ("14_30d", lambda d: 14 <= d < 30),
    ("30_60d", lambda d: 30 <= d < 60),
    ("60_90d", lambda d: 60 <= d < 90),
    ("90d_plus", lambda d: d >= 90),
]
BUCKET_ORDER = ["no_prior_match", "under_14d_same_tourney", "under_14d_cross_tourney"] + \
    [name for name, _ in LAYOFF_BUCKET_EDGES]

ORIGINAL_POOLED_UNDER14 = {"ATP": 0.0489, "WTA": 0.0346}


def bucket_for_days(days, same_tournament):
    if pd.isna(days):
        return "no_prior_match"
    if days < 14:
        return "under_14d_same_tourney" if same_tournament else "under_14d_cross_tourney"
    for name, test in LAYOFF_BUCKET_EDGES:
        if test(days):
            return name
    raise ValueError(days)


def compute_days_since_last_match(matches):
    long = pd.concat([
        matches[["Tournament", "Date", "Round", "Player_1", "Player_2"]]
        .rename(columns={"Player_1": "player", "Player_2": "opponent"}),
        matches[["Tournament", "Date", "Round", "Player_2", "Player_1"]]
        .rename(columns={"Player_2": "player", "Player_1": "opponent"}),
    ], ignore_index=True)
    long["edition_id"] = long["Tournament"] + " " + long["Date"].dt.year.astype(str)
    long = long.sort_values(["player", "Date"], kind="stable").reset_index(drop=True)
    long["prev_date"] = long.groupby("player")["Date"].shift(1)
    long["prev_edition"] = long.groupby("player")["edition_id"].shift(1)
    long["days_since_last"] = (long["Date"] - long["prev_date"]).dt.days
    long["same_tournament_as_prev"] = long["edition_id"] == long["prev_edition"]
    return long[["edition_id", "Date", "Round", "player", "opponent", "days_since_last", "same_tournament_as_prev"]].drop_duplicates(
        subset=["edition_id", "Date", "Round", "player", "opponent"]
    ).rename(columns={"Date": "date", "Round": "round"})


def build_layoff_dataset(matches, preds):
    layoff = compute_days_since_last_match(matches)
    merged = preds.merge(
        layoff, on=["edition_id", "date", "round", "player", "opponent"], how="left", validate="one_to_one"
    )
    assert len(merged) == len(preds), "layoff merge changed row count - duplicate match keys somewhere"
    merged["bucket"] = merged.apply(
        lambda r: bucket_for_days(r["days_since_last"], r["same_tournament_as_prev"]), axis=1)
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
    print(f"{tour}: {len(layoff_df)} player-perspective rows ({len(train)} train-era, {len(test)} test-era)")

    train_summary = pd.DataFrame([summarize_bucket(b, g) for b, g in train.groupby("bucket") if len(g)]) \
        .set_index("bucket").reindex(BUCKET_ORDER).reset_index()
    print("\n--- Train-era residual by bucket (under_14d now split same-tourney vs cross-tourney) ---")
    print(train_summary.to_string(index=False, formatters={
        "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format, "residual": "{:+.1%}".format,
        "residual_ci_lo": "{:+.1%}".format, "residual_ci_hi": "{:+.1%}".format, "z": "{:.2f}".format,
    }))

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

    print(f"\n--- Held-out per-bucket validation ({len(test_col)} test-era rows) ---")
    for b in BUCKET_ORDER:
        g = test_col[test_col["bucket"] == b]
        if len(g) < 10:
            print(f"  {b:<24}: n={len(g):<6} - too few to validate")
            continue
        observed, lo, hi = cluster_bootstrap_ci(g, "raw_loss", "adj_loss", group_col="player")
        sig = "  <- excludes zero" if (lo > 0 or hi < 0) else ""
        fitted = shift_by_bucket.get(b, float("nan"))
        print(f"  {b:<24}: n={len(g):<6} fitted_shift={fitted:+.4f}  held-out log-loss improvement="
              f"{observed:+.4f} CI[{lo:+.4f},{hi:+.4f}]{sig}")

    cross_shift = shift_by_bucket.get("under_14d_cross_tourney")
    same_shift = shift_by_bucket.get("under_14d_same_tourney")
    pooled = ORIGINAL_POOLED_UNDER14[tour]
    print(f"\n{'=' * 100}\n{tour} SUMMARY: isolated cross-tournament under_14d shift vs. original pooled number\n{'=' * 100}")
    print(f"  Original pooled under_14d shift (production, live): {pooled:+.4f}")
    if same_shift is not None:
        print(f"  Same-tournament-only shift (the ~64-65% diluting population): {same_shift:+.4f}")
    if cross_shift is not None:
        print(f"  Cross-tournament-only shift (the genuine 'arrived fresh' signal): {cross_shift:+.4f}")
        delta = cross_shift - pooled
        print(f"  Difference from pooled: {delta:+.4f}")
        meaningfully_different = abs(delta) > 0.02  # ~3-4 Elo-point-equivalent threshold, disclosed judgment call
        print(f"  Meaningfully different from the current production number? "
              f"{'YES' if meaningfully_different else 'NO - close enough that the pooled number held up fine'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default=None, choices=["ATP", "WTA"])
    args = parser.parse_args()
    tours = [args.tour] if args.tour else ["ATP", "WTA"]
    for t in tours:
        run(t)
        print()
