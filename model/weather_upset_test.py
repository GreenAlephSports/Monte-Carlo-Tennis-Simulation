"""Tests the "extreme heat causes more upsets" hypothesis: on days that were unusually hot for
that match's location (top decile of same-location temperature), does Elo's predicted favorite
lose more often than on days with moderate conditions?

Geographic/weather layer, built for this test specifically:
  - tournament_locations.py: static Kaggle-tournament-name -> (city, lat, lon) alias table. Row
    coverage is reported FIRST, before anything else runs - see run()'s opening block - because a
    thin, unrepresentative covered subset would make any result here unfalsifiable in practice.
  - weather_fetch.py: one Open-Meteo historical-archive call per (location, tournament-edition
    date range), disk-cached, returning real daily max temperature (C) and mean relative
    humidity (%) for that location and those exact calendar days.

Same rigor as every prior test in this series: frozen per-tournament-edition Elo (reused directly
from elite_opponent_residual_test.build_frozen_predictions, run on the FULL match history so Elo
still evolves correctly across editions this test can't resolve a location for - only the
downstream weather join is restricted to the resolved/outdoor subset), a chronological
tournament-edition-level 80/20 train/test split, held-out validation of the heat-adjusted vs. raw
prediction, and a bootstrap CI clustered by EDITION rather than by player - matches at the same
event in the same week share the same weather, so they are not independent evidence about "does
heat correlate with upsets" the way that same event's outcomes usually would be for a per-player
question.

Methodology notes / simplifications (stated up front, not buried):
  - Restricted to Court == "Outdoor" - weather is meaningless (or worse, misleadingly attributes an
    indoor climate-controlled match to the day's outdoor temperature) for indoor events.
  - "Favorite" = the player build_frozen_predictions' frozen pre-tournament-edition Elo gives
    pred_win > 0.5 for that specific match; "upset" = that favorite lost. Matches with pred_win
    exactly 0.5 (a dead-even Elo tie) are dropped - there's no favorite to test.
  - The top-decile threshold is fit on TRAIN-era weather-covered matches only, then applied as a
    fixed Celsius cutoff to test-era matches - it is a real held-out cutoff, not read off the
    combined train+test distribution.
  - Humidity is reported as a secondary, exploratory bucket analysis (same method, top-decile
    mean relative humidity vs. the rest) - the hypothesis as stated is specifically about heat.

Usage:
    python model/weather_upset_test.py [--tour ATP|WTA|BOTH]
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
from survivorship_upset_test import cluster_bootstrap_ci, summarize_bucket  # noqa: E402
from tournament_locations import resolve_location  # noqa: E402
from weather_fetch import fetch_daily_weather_batch  # noqa: E402

MIN_COVERAGE_FOR_TEST = 0.30  # below this, refuse to run the hypothesis test at all


def report_location_coverage(matches, tour):
    matches = matches.copy()
    matches["resolved"] = matches["Tournament"].apply(resolve_location)
    n_total = len(matches)
    n_resolved = matches["resolved"].notna().sum()
    print(f"\n{tour}: tournament-location coverage = {n_resolved}/{n_total} rows "
          f"({n_resolved / n_total:.1%}) resolved to a known city via tournament_locations.py")

    outdoor = matches[matches["Court"] == "Outdoor"]
    n_outdoor_resolved = (outdoor["resolved"].notna()).sum()
    print(f"{tour}: of those, {n_outdoor_resolved}/{len(outdoor)} Outdoor-court rows "
          f"({n_outdoor_resolved / max(len(outdoor), 1):.1%}) are both resolved AND outdoor - "
          f"this Outdoor+resolved subset is what actually feeds the weather join below "
          f"({n_outdoor_resolved / n_total:.1%} of all {tour} rows)")

    unresolved_counts = matches.loc[matches["resolved"].isna(), "Tournament"].value_counts()
    print(f"{tour}: largest unresolved tournament names (by row count) - "
          f"either a genuinely ambiguous host city (documented in tournament_locations.py's "
          f"explicit exclusions) or a name not yet covered:")
    print(unresolved_counts.head(8).to_string())

    return matches


def build_weather_by_edition(outdoor_resolved):
    """One row per (edition_id, date) with tmax/humidity_mean, built from a single Open-Meteo call
    per (location, edition) date-range rather than one call per match date."""
    outdoor_resolved = outdoor_resolved.copy()
    outdoor_resolved["canonical_key"] = outdoor_resolved["resolved"].apply(lambda r: r[0])
    outdoor_resolved["lat"] = outdoor_resolved["resolved"].apply(lambda r: r[3])
    outdoor_resolved["lon"] = outdoor_resolved["resolved"].apply(lambda r: r[4])
    outdoor_resolved["edition_id"] = outdoor_resolved["Tournament"] + " " + outdoor_resolved["Date"].dt.year.astype(str)

    edition_ranges = outdoor_resolved.groupby(["edition_id", "lat", "lon"])["Date"].agg(
        date_min="min", date_max="max"
    ).reset_index()
    requests_needed = [
        (row.lat, row.lon, row.date_min.strftime("%Y-%m-%d"), row.date_max.strftime("%Y-%m-%d"))
        for row in edition_ranges.itertuples(index=False)
    ]
    weather_by_range = fetch_daily_weather_batch(requests_needed)

    rows = []
    for row in edition_ranges.itertuples(index=False):
        key = (row.lat, row.lon, row.date_min.strftime("%Y-%m-%d"), row.date_max.strftime("%Y-%m-%d"))
        by_date = weather_by_range.get(key, {})
        for date_str, vals in by_date.items():
            rows.append((row.edition_id, pd.Timestamp(date_str), vals["tmax"], vals["humidity_mean"]))
    weather_df = pd.DataFrame(rows, columns=["edition_id", "date", "tmax", "humidity_mean"])
    return weather_df.drop_duplicates(subset=["edition_id", "date"])


def bucket_top_decile(values, threshold):
    return np.where(values >= threshold, "top_decile", "rest")


def bucket_quintile(values, edges):
    labels = ["q1_coolest", "q2", "q3", "q4", "q5_hottest"]
    idx = np.searchsorted(edges, values, side="right")
    return [labels[min(i, 4)] for i in idx]


def run_single_threshold_and_quintiles(train, test, value_col, label):
    print(f"\n=== {label}: single-threshold (top decile vs. rest) held-out validation ===")

    threshold = train[value_col].quantile(0.90)
    print(f"Train-era top-decile cutoff (90th percentile of train-era {value_col}): {threshold:.1f}")

    train = train.copy()
    test = test.copy()
    train["bucket"] = bucket_top_decile(train[value_col].values, threshold)
    test["bucket"] = bucket_top_decile(test[value_col].values, threshold)

    print(f"Train: {(train['bucket'] == 'top_decile').sum()} top-decile rows, "
          f"{(train['bucket'] == 'rest').sum()} rest; "
          f"Test: {(test['bucket'] == 'top_decile').sum()} top-decile rows, "
          f"{(test['bucket'] == 'rest').sum()} rest")

    summary = pd.DataFrame([summarize_bucket(b, g) for b, g in train.groupby("bucket")]) \
        .set_index("bucket").reindex(["rest", "top_decile"]).reset_index()
    summary = summary.rename(columns={"actual_rate": "actual_favorite_win_rate", "pred_rate": "pred_favorite_win_rate"})
    print("\nTrain-era favorite-win rate by bucket (residual = actual - Elo-predicted; "
          "MORE NEGATIVE residual = more upsets than Elo expects):")
    print(summary.to_string(index=False, formatters={
        "actual_favorite_win_rate": "{:.1%}".format, "pred_favorite_win_rate": "{:.1%}".format,
        "residual": "{:+.1%}".format, "residual_ci_lo": "{:+.1%}".format, "residual_ci_hi": "{:+.1%}".format,
        "z": "{:.2f}".format,
    }))

    shift_by_bucket = {}
    for b in ["rest", "top_decile"]:
        g = train[train["bucket"] == b]
        if len(g) == 0:
            continue
        actual_rate, pred_rate = g["actual_win"].mean(), g["pred_win"].mean()
        shift_by_bucket[b] = logit(actual_rate) - logit(pred_rate)

    test_col = test[test["bucket"].isin(shift_by_bucket)].copy()
    test_col["adjusted_pred"] = test_col.apply(lambda r: sigmoid(logit(r["pred_win"]) + shift_by_bucket[r["bucket"]]), axis=1)
    test_col["raw_loss"] = log_loss(test_col["actual_win"].values, test_col["pred_win"].values)
    test_col["adj_loss"] = log_loss(test_col["actual_win"].values, test_col["adjusted_pred"].values)

    if len(test_col) >= 5 and test_col["edition_id"].nunique() >= 2:
        observed, lo, hi = cluster_bootstrap_ci(test_col, "raw_loss", "adj_loss", group_col="edition_id")
        print(f"\nHeld-out test-era validation: {len(test_col)} rows, {test_col['edition_id'].nunique()} editions")
        print(f"  Raw Elo             : log-loss = {test_col['raw_loss'].mean():.4f}")
        print(f"  Heat-bucket-adjusted: log-loss = {test_col['adj_loss'].mean():.4f}")
        print(f"  Mean per-match log-loss improvement (raw - adjusted, >0 = heat bucket helps), "
              f"edition-clustered: {observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    else:
        print(f"\nHeld-out test-era validation: only {len(test_col)} usable rows across "
              f"{test_col['edition_id'].nunique()} editions - too few for a meaningful bootstrap CI, skipping")

    # finer breakdown for readability, train-era only (not a second held-out test, just showing
    # where the single-threshold train summary above is coming from)
    edges = train[value_col].quantile([0.2, 0.4, 0.6, 0.8]).values
    train["quintile"] = bucket_quintile(train[value_col].values, edges)
    quint_order = ["q1_coolest", "q2", "q3", "q4", "q5_hottest"]
    quint_summary = pd.DataFrame([summarize_bucket(b, g) for b, g in train.groupby("quintile") if len(g)]) \
        .set_index("bucket").reindex(quint_order).reset_index()
    print(f"\nTrain-era residual by {value_col} quintile (context - not itself the held-out test):")
    print(quint_summary.to_string(index=False, formatters={
        "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format, "residual": "{:+.1%}".format,
        "residual_ci_lo": "{:+.1%}".format, "residual_ci_hi": "{:+.1%}".format, "z": "{:.2f}".format,
    }))
    is_monotonic = quint_summary["residual"].is_monotonic_decreasing
    print(f"Does the train-era residual decrease monotonically q1 (coolest) -> q5 (hottest), as "
          f"'hotter = more upsets' specifically predicts? {'YES' if is_monotonic else 'NO'} "
          f"({', '.join(f'{r:+.1%}' for r in quint_summary['residual'])})")

    return summary


def run(tour):
    if tour == "BOTH":
        frames = []
        for t in ["ATP", "WTA"]:
            m = load_matches_for_tour(t)
            m["Tournament"] = t + ": " + m["Tournament"]  # keep edition_id namespaced per tour
            frames.append(m)
        matches = pd.concat(frames, ignore_index=True)
    else:
        matches = load_matches_for_tour(tour)

    matches = report_location_coverage(matches, tour)

    outdoor_resolved = matches[(matches["Court"] == "Outdoor") & matches["resolved"].notna()].copy()
    weather_df = build_weather_by_edition(outdoor_resolved)
    print(f"\n{tour}: weather successfully retrieved for {weather_df['edition_id'].nunique()} "
          f"tournament editions, {len(weather_df)} unique (edition, date) days")

    preds, editions = build_frozen_predictions(matches)
    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])

    # one row per real match (the favorite's own perspective row - pred_win > 0.5), then join
    # weather by (edition_id, date). This inner join is the real, final coverage number the
    # hypothesis test runs on - reported explicitly, since it will be well below the raw row
    # coverage above (Outdoor-only, weather-fetch-successful, has-a-favorite all further narrow it).
    favorite_rows = preds[preds["pred_win"] > 0.5 + EPS].copy()
    weather_matches = favorite_rows.merge(weather_df, on=["edition_id", "date"], how="inner")

    n_total_matches = len(favorite_rows)
    n_with_weather = len(weather_matches)
    print(f"\n{tour}: {n_total_matches} real matches have a clear Elo favorite; "
          f"{n_with_weather} of those ({n_with_weather / max(n_total_matches, 1):.1%}) have a "
          f"resolved location AND successfully-fetched weather for their exact match date - "
          f"THIS is the sample the hypothesis test below actually runs on")

    coverage_frac = n_with_weather / max(n_total_matches, 1)
    if coverage_frac < MIN_COVERAGE_FOR_TEST or n_with_weather < 200:
        print(f"\nCoverage of the weather-testable sample ({coverage_frac:.1%}, n={n_with_weather}) is "
              f"below the {MIN_COVERAGE_FOR_TEST:.0%} / 200-match floor this script requires before "
              f"treating a result as meaningful. STOPPING HERE rather than reporting a heat-vs-upset "
              f"finding from a thin, unrepresentative subset.")
        return

    train = weather_matches[weather_matches["edition_id"].isin(train_editions)]
    test = weather_matches[weather_matches["edition_id"].isin(test_editions)]
    print(f"Train/test split (same chronological tournament-edition boundary as the other tests "
          f"in this series): {len(train)} train-era weather-matched matches "
          f"({train['edition_id'].nunique()} editions), {len(test)} test-era "
          f"({test['edition_id'].nunique()} editions)")

    if len(train) < 100 or len(test) < 30:
        print(f"\nTrain ({len(train)}) or test ({len(test)}) sample is too small for a trustworthy "
              f"held-out split even though raw coverage passed the floor above - STOPPING HERE.")
        return

    run_single_threshold_and_quintiles(train, test, "tmax", "TEMPERATURE (primary hypothesis)")
    run_single_threshold_and_quintiles(train, test, "humidity_mean", "HUMIDITY (secondary, exploratory)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA", "BOTH"])
    args = parser.parse_args()
    run(args.tour)
