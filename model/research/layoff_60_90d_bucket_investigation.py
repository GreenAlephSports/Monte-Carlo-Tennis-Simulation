"""Investigates the unexplained 60_90d layoff-bucket irregularity flagged by layoff_test.py: the
train-era residual doesn't move monotonically with layoff length - it partially RECOVERS at
60_90d before dropping again at 90d_plus (ATP: 30_60d -4.0% -> 60_90d -2.0% -> 90d_plus -5.2%;
WTA: 30_60d -1.9% -> 60_90d -0.6% -> 90d_plus -6.7%). Both tours show the same bump, which argues
against pure noise, but the mechanism is unknown.

This is a pure descriptive/composition investigation, not another held-out-validation test - the
goal is to characterize WHO is in the 60_90d bucket and WHY their residual looks different, not to
propose or validate a new adjustment. Four candidate explanations are checked, each by the same
method: split the bucket into subgroups, check whether the residual differs meaningfully across
subgroups, and check whether dropping any one dominant subgroup moves the bucket's average residual:
  1. Surface - is 60_90d disproportionately hard/clay/grass relative to the other buckets?
  2. Tournament tier (Series column: Grand Slam / Masters 1000 / ATP500 / ATP250 / etc, or WTA's
     equivalent tiers) - are 60_90d returns disproportionately happening at lower-tier, less
     competitive events (a warm-up-event selection effect)?
  3. Time of year - is 60_90d disproportionately an off-season return (e.g. the Australian Open
     swing, where a 60-90 day gap is mechanically produced by the tour's own winter break, not an
     individual injury/rust story)?
  4. Player concentration - is a small number of players cycling through short injury breaks
     repeatedly responsible for most of the bucket's rows, such that the bucket's average is really
     a few individuals' idiosyncratic pattern rather than a population-level effect?

Reuses build_layoff_dataset (layoff_test.py) and build_frozen_predictions
(elite_opponent_residual_test.py) exactly as already validated - no new Elo or bucketing logic.

Usage:
    python model/research/layoff_60_90d_bucket_investigation.py [--tour ATP|WTA]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import build_frozen_predictions  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from layoff_test import build_layoff_dataset  # noqa: E402
from survivorship_upset_test import summarize_bucket  # noqa: E402

TARGET_BUCKET = "60_90d"
NEIGHBOR_BUCKETS = ["30_60d", "90d_plus"]  # for side-by-side comparison, not re-derivation


def run(tour):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)
    layoff_df = build_layoff_dataset(matches, preds)

    # attach Tournament/Series/Surface/Month from the raw matches table - layoff_df only carries
    # what build_layoff_dataset kept (edition_id/date/round/player/opponent/bucket/pred/actual),
    # so tier and month need to be merged back in from the source rows.
    has_tier = "Series" in matches.columns  # WTA's Kaggle dataset has no tournament-tier column at all
    tier_cols = ["Series"] if has_tier else []
    meta = pd.concat([
        matches[["Tournament", "Date", "Round", "Surface", "Player_1", "Player_2"] + tier_cols]
        .rename(columns={"Player_1": "player", "Player_2": "opponent"}),
        matches[["Tournament", "Date", "Round", "Surface", "Player_2", "Player_1"] + tier_cols]
        .rename(columns={"Player_2": "player", "Player_1": "opponent"}),
    ], ignore_index=True)
    meta["edition_id"] = meta["Tournament"] + " " + meta["Date"].dt.year.astype(str)
    meta = meta.rename(columns={"Date": "date", "Round": "round"}).drop_duplicates(
        subset=["edition_id", "date", "round", "player", "opponent"]
    )
    df = layoff_df.merge(meta, on=["edition_id", "date", "round", "player", "opponent"], how="left", validate="one_to_one")
    df["month"] = df["date"].dt.month

    target = df[df["bucket"] == TARGET_BUCKET].copy()
    print(f"{tour}: {len(target)} player-perspective rows in the {TARGET_BUCKET} bucket "
          f"({target['player'].nunique()} distinct players)")
    overall_actual, overall_pred = target["actual_win"].mean(), target["pred_win"].mean()
    print(f"Bucket-wide residual (all {TARGET_BUCKET} rows, for reference): "
          f"actual {overall_actual:.1%} vs. predicted {overall_pred:.1%} "
          f"(gap {overall_actual - overall_pred:+.1%})")

    def report_split(label, col):
        print(f"\n--- Split by {label} ---")
        summary = pd.DataFrame([summarize_bucket(str(v), g) for v, g in target.groupby(col) if len(g) >= 10])
        if len(summary) == 0:
            print("  No subgroup has >= 10 rows - too thin to say anything.")
            return summary
        summary = summary.sort_values("n", ascending=False)
        print(summary[["bucket", "n", "actual_rate", "pred_rate", "residual", "z"]].to_string(index=False, formatters={
            "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format, "residual": "{:+.1%}".format, "z": "{:.2f}".format,
        }))
        return summary

    # 1. Surface
    surface_summary = report_split("surface", "Surface")

    # 2. Tournament tier (ATP only - WTA's Kaggle dataset carries no tier/Series column)
    if has_tier:
        tier_summary = report_split("tournament tier (Series)", "Series")
    else:
        print("\n--- Split by tournament tier ---\n  Skipped: this tour's dataset has no tier column.")

    # 3. Time of year - group into tour-calendar quarters rather than raw month, more legible
    def season_bucket(m):
        if m in (12, 1, 2):
            return "Dec-Feb (Australian swing / winter)"
        if m in (3, 4, 5):
            return "Mar-May (Sunshine Double / clay swing)"
        if m in (6, 7, 8):
            return "Jun-Aug (grass / US hard swing)"
        return "Sep-Nov (Asian swing / indoor season)"
    target["season"] = target["month"].apply(season_bucket)
    season_summary = report_split("time of year", "season")

    # 4. Player concentration
    print(f"\n--- Player concentration ---")
    per_player = target.groupby("player").size().sort_values(ascending=False)
    top5 = per_player.head(5)
    print(f"Rows per player: median {per_player.median():.0f}, max {per_player.max()}, "
          f"{(per_player == 1).sum()} of {len(per_player)} players appear exactly once")
    print(f"Top 5 most frequent players in this bucket:\n{top5.to_string()}")

    print("\nDropping the single most frequent player and re-checking the bucket-wide residual:")
    top_player = per_player.index[0]
    without_top = target[target["player"] != top_player]
    wa, wp = without_top["actual_win"].mean(), without_top["pred_win"].mean()
    print(f"  Without {top_player} ({per_player.iloc[0]} rows removed, {len(without_top)} remain): "
          f"actual {wa:.1%} vs. predicted {wp:.1%} (gap {wa - wp:+.1%}) "
          f"vs. full-bucket gap {overall_actual - overall_pred:+.1%}")

    print("\nDropping the top 5 most frequent players and re-checking:")
    without_top5 = target[~target["player"].isin(top5.index)]
    w5a, w5p = without_top5["actual_win"].mean(), without_top5["pred_win"].mean()
    print(f"  Without top 5 ({len(target) - len(without_top5)} rows removed, {len(without_top5)} remain): "
          f"actual {w5a:.1%} vs. predicted {w5p:.1%} (gap {w5a - w5p:+.1%}) "
          f"vs. full-bucket gap {overall_actual - overall_pred:+.1%}")

    # side-by-side with neighboring buckets, for the same subgroup breakdowns, so a candidate
    # explanation has to explain why 60_90d looks DIFFERENT from its neighbors, not just why
    # 60_90d itself has variation
    print(f"\n{'=' * 90}\nSide-by-side with neighboring buckets ({', '.join(NEIGHBOR_BUCKETS)}), "
          f"same composition axes\n{'=' * 90}")
    for b in NEIGHBOR_BUCKETS:
        neighbor = df[df["bucket"] == b]
        na, npred = neighbor["actual_win"].mean(), neighbor["pred_win"].mean()
        print(f"\n{b}: {len(neighbor)} rows, {neighbor['player'].nunique()} players, "
              f"bucket-wide gap {na - npred:+.1%}")
        axes = [("Surface", "surface")] + ([("Series", "tier")] if has_tier else [])
        for col, name in axes:
            vc = neighbor.groupby(col).apply(
                lambda g: pd.Series({"n": len(g), "actual-pred": g["actual_win"].mean() - g["pred_win"].mean()}),
                include_groups=False,
            )
            vc = vc[vc["n"] >= 10].sort_values("n", ascending=False)
            if len(vc):
                print(f"  by {name}: " + ", ".join(f"{idx}(n={int(r.n)}, gap={r['actual-pred']:+.1%})" for idx, r in vc.iterrows()))

    # compare the SAME composition mix (surface/tier shares) between 60_90d and its neighbors -
    # if 60_90d's mix is basically identical to 30_60d/90d_plus, composition isn't the explanation
    print(f"\n--- Composition mix comparison: is {TARGET_BUCKET}'s surface/tier mix actually "
          f"different from its neighbors, or about the same? ---")
    for col, name in [("Surface", "surface")] + ([("Series", "tier")] if has_tier else []):
        mix = pd.DataFrame({
            b: df[df["bucket"] == b][col].value_counts(normalize=True) for b in [TARGET_BUCKET] + NEIGHBOR_BUCKETS
        }).fillna(0)
        print(f"\n  {name} share of rows:")
        print(mix.to_string(formatters={c: "{:.1%}".format for c in mix.columns}))

    # standardization: how much of the 60_90d bucket's smaller-magnitude gap is just its
    # different composition mix? Reweight 60_90d's OWN per-subgroup residuals using the OVERALL
    # dataset's subgroup mix (not the bucket's own mix) - if the standardized gap moves toward the
    # neighbors' gaps, composition is doing real explanatory work; if it barely moves, it isn't.
    print(f"\n--- Standardization: reweighting {TARGET_BUCKET}'s own per-subgroup residuals by "
          f"the OVERALL dataset's subgroup mix ---")
    print(f"  Raw {TARGET_BUCKET} gap: {overall_actual - overall_pred:+.2%} (n={len(target)})")
    for col, name in [("Surface", "surface")] + ([("Series", "tier")] if has_tier else []):
        overall_mix = df[col].value_counts(normalize=True)
        per_sub = target.groupby(col).apply(
            lambda g: pd.Series({"n": len(g), "gap": g["actual_win"].mean() - g["pred_win"].mean()}),
            include_groups=False,
        )
        eligible = per_sub[per_sub["n"] >= 10]
        weight_covered = sum(overall_mix.get(s, 0) for s in eligible.index)
        std_gap = sum(overall_mix.get(s, 0) * eligible.loc[s, "gap"] for s in eligible.index) / weight_covered
        print(f"  {name}-standardized gap: {std_gap:+.2%} "
              f"({'moved toward neighbors -> real explanatory contribution' if abs(std_gap) > abs(overall_actual - overall_pred) + 0.002 else 'barely moved -> composition on this axis explains little'})")
    neighbor_avg_gap = np.average(
        [df[df['bucket'] == b]['actual_win'].mean() - df[df['bucket'] == b]['pred_win'].mean() for b in NEIGHBOR_BUCKETS],
        weights=[len(df[df['bucket'] == b]) for b in NEIGHBOR_BUCKETS],
    )
    print(f"  (for reference, the n-weighted average gap of the two neighboring buckets is "
          f"{neighbor_avg_gap:+.2%} - that's the target this bucket's gap would need to reach for "
          f"composition to fully explain the irregularity)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
    args = parser.parse_args()
    run(args.tour)
