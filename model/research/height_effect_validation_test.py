"""Validates the height_diff -> outperformance-beyond-Elo effect found by height_serve_proxy_test.py
to the same standard this project applies before any correction reaches production: full historical
scale (not a truncated/recent-only cut), a decade-by-decade stability check (the same check that
caught decay3's per-era heterogeneity in decay3_full_historical_test.py), and a population-
composition scrutiny across tour/ranking-tier/age/player-concentration (the same axes
layoff_60_90d_bucket_investigation.py used to characterize the 60-90d layoff bucket rather than take
its pooled average at face value).

This is NOT a re-run of the original hypothesis test (that already happened, with its own
significance/CI verdict) - it's a robustness audit of a REAL finding, checking whether it holds up
or is secretly a period effect / subpopulation artifact before treating it as decision-worthy.

Full historical scale: uses the exact same load_matches_for_tour + build_frozen_predictions walk-
forward as height_serve_proxy_test.py - the complete Kaggle history each tour's loader returns
(ATP 2000-, WTA 2006-), no lookback window or recent-only truncation anywhere in this pipeline.

Decade stability: matches decay3_full_historical_test.py's own (date.year // 5) * 5 bucketing and
per-bucket coefficient + player-clustered bootstrap CI, applied to the height_diff term specifically
(controlling for elo_diff in every bucket, never pooled skill-blind).

Population composition, real subgroup splits (not just the pooled average):
  - Tour (ATP vs WTA separately - not just as a covariate)
  - Ranking tier: min(rank_a, rank_b) at match time (from the raw Kaggle Rank_1/Rank_2 columns,
    the real official ranking on the match date, not a re-derived proxy) - is this a top-of-the-
    game effect or does it hold among journeyman-level players too?
  - Age tier: (match year - Wikipedia birth year) averaged across both players - is this a
    peak-athleticism-era effect, or does it show up for veterans and teenagers alike? Real birth
    years (model/research/wikipedia_handedness_scrape.py --backfill-height --fields birth_year),
    not a tour-tenure proxy.
  - Player concentration: does dropping the most extreme-height players change the coefficient
    materially, or is this a real population-level pattern? Same check
    layoff_60_90d_bucket_investigation.py runs for its dominant-player concern.

Usage:
    python model/research/height_effect_validation_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import build_frozen_predictions  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from height_serve_proxy_test import cluster_bootstrap_coef, load_height_map, ols  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
HANDEDNESS_PATH = OUTPUT_DIR / "player_handedness.csv"
MIN_BUCKET_N = 300  # below this, a subgroup coefficient isn't trustworthy enough to report a verdict on


def load_birth_year_map():
    """{player_csv_name: birth_year (int)} - only cleanly-resolved rows."""
    df = pd.read_csv(HANDEDNESS_PATH, keep_default_na=False, dtype=str)
    usable = df[(df["birth_year"] != "") & df["birth_year"].notna()]
    return {row.player: int(row.birth_year) for row in usable.itertuples(index=False)}


def build_rank_lookup(matches):
    """Long-format (edition_id, date, round, player, opponent) -> rank_self, using the raw Kaggle
    Rank_1/Rank_2 columns directly - the real official ranking AS OF that match's date, not a
    re-derived snapshot (no lookahead concern: this is contemporaneous ranking data the tour itself
    published going into the match, same as Rank_1/Rank_2 usage elsewhere in this project, e.g.
    elite_opponent_residual_test's own opponent_rank tracking)."""
    df = matches.copy()
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)
    long = pd.concat([
        df.assign(player=df["Player_1"], opponent=df["Player_2"], rank_self=df["Rank_1"]),
        df.assign(player=df["Player_2"], opponent=df["Player_1"], rank_self=df["Rank_2"]),
    ], ignore_index=True)
    long = long.rename(columns={"Date": "date", "Round": "round"})
    return long[["edition_id", "date", "round", "player", "opponent", "rank_self"]].drop_duplicates(
        subset=["edition_id", "date", "round", "player", "opponent"]
    )


def build_dataset(tour, matches, height, birth_year):
    preds, editions = build_frozen_predictions(matches)
    rank_lookup = build_rank_lookup(matches)
    preds = preds.merge(rank_lookup, on=["edition_id", "date", "round", "player", "opponent"], how="left")

    neutral = preds[preds["player"] < preds["opponent"]].copy()
    neutral = neutral.rename(columns={
        "player": "player_a", "opponent": "player_b",
        "player_elo": "elo_a", "opponent_elo": "elo_b", "actual_win": "won_a",
        "rank_self": "rank_a",
    })
    # rank_b comes from the OPPONENT-perspective row of the same match - merge it in separately
    opp_rank = preds.rename(columns={"player": "player_b", "opponent": "player_a", "rank_self": "rank_b"})[
        ["edition_id", "date", "round", "player_a", "player_b", "rank_b"]
    ]
    neutral = neutral.merge(opp_rank, on=["edition_id", "date", "round", "player_a", "player_b"], how="left")

    neutral["tour"] = tour
    neutral["height_a"] = neutral["player_a"].map(height)
    neutral["height_b"] = neutral["player_b"].map(height)
    neutral["birth_a"] = neutral["player_a"].map(birth_year)
    neutral["birth_b"] = neutral["player_b"].map(birth_year)
    neutral["elo_diff"] = neutral["elo_a"] - neutral["elo_b"]
    neutral["match_year"] = neutral["date"].dt.year
    neutral["decade"] = (neutral["match_year"] // 5) * 5

    return neutral[[
        "tour", "edition_id", "date", "match_year", "decade", "player_a", "player_b",
        "elo_diff", "won_a", "height_a", "height_b", "birth_a", "birth_b", "rank_a", "rank_b",
    ]]


def fit_height_coef(df):
    """Returns (coef, se, z, lo, hi) for height_diff in won_a ~ elo_diff + height_diff, plus the
    player-clustered bootstrap CI - the exact same regression height_serve_proxy_test.py runs,
    reused verbatim so every subgroup verdict below is directly comparable to the headline result."""
    y = df["won_a"].values.astype(float)
    x_cols = ["elo_diff", "height_diff"]
    X = df[x_cols].values.astype(float)
    beta, se = ols(y, X)
    z = beta[2] / se[2]
    lo, hi = cluster_bootstrap_coef(df, "won_a", x_cols, 1, ["player_a", "player_b"], n_boot=2000)
    return beta[2], se[2], z, lo, hi


def report_subgroup(label, df):
    if len(df) < MIN_BUCKET_N:
        print(f"  {label:<38} n={len(df):<7} TOO THIN (<{MIN_BUCKET_N}) - no verdict")
        return None
    coef, se, z, lo, hi = fit_height_coef(df)
    sig = abs(z) > 1.96 and (lo > 0 or hi < 0)
    tag = "SIGNIFICANT" if sig else "not significant"
    print(f"  {label:<38} n={len(df):<7} coef={coef:+.6f}  z={z:+.2f}  "
          f"boot CI=[{lo:+.6f},{hi:+.6f}]  {tag}")
    return coef


def run():
    height = load_height_map()
    birth_year = load_birth_year_map()
    print(f"Height known for {len(height)} players. Birth year known for {len(birth_year)} players "
          f"(model/research/wikipedia_handedness_scrape.py --backfill-height --fields birth_year).")

    frames = []
    for tour in ("ATP", "WTA"):
        matches = load_matches_for_tour(tour)
        rows = build_dataset(tour, matches, height, birth_year)
        print(f"{tour}: {len(rows)} real historical matches, full available history "
              f"({rows['date'].min().date()} to {rows['date'].max().date()}) - no lookback "
              f"truncation, same as the original hypothesis test.")
        frames.append(rows)
    all_rows = pd.concat(frames, ignore_index=True)

    usable = all_rows.dropna(subset=["height_a", "height_b"]).copy()
    usable["height_diff"] = usable["height_a"] - usable["height_b"]
    print(f"\n{len(all_rows)} total real matches; {len(usable)} have known height for both players "
          f"({len(usable) / len(all_rows):.1%}) - this is the population every check below runs on.")

    print(f"\n{'=' * 92}\nHEADLINE (reference): full-sample coefficient, same as height_serve_proxy_test.py\n{'=' * 92}")
    report_subgroup("Full sample, both tours pooled", usable)

    # --- 1. Decade stability (same bucketing as decay3_full_historical_test.py) ---
    print(f"\n{'=' * 92}\nDECADE STABILITY (does the effect hold consistently across eras, or "
          f"concentrate in one period?)\n{'=' * 92}")
    for decade in sorted(usable["decade"].unique()):
        bucket = usable[usable["decade"] == decade]
        yr_range = f"{decade}-{decade + 4}"
        report_subgroup(yr_range, bucket)

    # --- 2. Tour split ---
    print(f"\n{'=' * 92}\nPOPULATION COMPOSITION: by tour\n{'=' * 92}")
    for tour in ("ATP", "WTA"):
        report_subgroup(tour, usable[usable["tour"] == tour])

    # --- 3. Ranking tier ---
    print(f"\n{'=' * 92}\nPOPULATION COMPOSITION: by ranking tier (min real rank of the two "
          f"players, at match time)\n{'=' * 92}")
    ranked = usable.dropna(subset=["rank_a", "rank_b"]).copy()
    ranked["best_rank"] = ranked[["rank_a", "rank_b"]].min(axis=1)
    print(f"({len(ranked)} of {len(usable)} rows have a known rank for both players)")
    tier_edges = [(0, 50, "top-50 involved"), (50, 150, "51-150"), (150, float("inf"), "151+")]
    for lo_r, hi_r, label in tier_edges:
        bucket = ranked[(ranked["best_rank"] > lo_r) & (ranked["best_rank"] <= hi_r)]
        report_subgroup(f"best_rank {label}", bucket)

    # --- 4. Age tier ---
    print(f"\n{'=' * 92}\nPOPULATION COMPOSITION: by age tier (real Wikipedia birth year, "
          f"match_year - birth_year, averaged across both players)\n{'=' * 92}")
    aged = usable.dropna(subset=["birth_a", "birth_b"]).copy()
    aged["age_a"] = aged["match_year"] - aged["birth_a"]
    aged["age_b"] = aged["match_year"] - aged["birth_b"]
    aged["avg_age"] = (aged["age_a"] + aged["age_b"]) / 2
    print(f"({len(aged)} of {len(usable)} rows have a known birth year for both players)")
    age_edges = [(0, 23, "<23 (young)"), (23, 29, "23-29 (prime)"), (29, 99, "29+ (veteran)")]
    for lo_a, hi_a, label in age_edges:
        bucket = aged[(aged["avg_age"] >= lo_a) & (aged["avg_age"] < hi_a)]
        report_subgroup(f"avg age {label}", bucket)

    # --- 5. Player concentration ---
    print(f"\n{'=' * 92}\nPLAYER CONCENTRATION: is this a population-level pattern or a handful "
          f"of extreme-height players' idiosyncratic result?\n{'=' * 92}")
    heights_by_player = pd.concat([
        usable[["player_a", "height_a"]].rename(columns={"player_a": "player", "height_a": "height"}),
        usable[["player_b", "height_b"]].rename(columns={"player_b": "player", "height_b": "height"}),
    ]).drop_duplicates(subset=["player"])
    tallest5 = heights_by_player.nlargest(5, "height")["player"].tolist()
    shortest5 = heights_by_player.nsmallest(5, "height")["player"].tolist()
    print(f"Tallest 5 in this population: {tallest5}")
    print(f"Shortest 5 in this population: {shortest5}")
    without_extremes = usable[
        ~usable["player_a"].isin(tallest5 + shortest5) & ~usable["player_b"].isin(tallest5 + shortest5)
    ]
    report_subgroup("Full sample (reference)", usable)
    report_subgroup("Excluding tallest-5 and shortest-5 players", without_extremes)

    print(f"\n{'=' * 92}\nDone. A production-ready effect should show: a headline coefficient that "
          f"survives decade-by-decade (no single era carrying it), holds in both tours "
          f"individually, doesn't flip sign across ranking tiers or age tiers, and doesn't "
          f"collapse when the most extreme-height players are excluded.\n{'=' * 92}")


if __name__ == "__main__":
    run()
