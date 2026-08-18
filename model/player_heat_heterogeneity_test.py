"""Tests player-level heterogeneity in heat performance - distinct from weather_upset_test.py's
population-level "does extreme heat cause more upsets overall" question. Here the question is:
does any INDIVIDUAL player perform better or worse than their own Elo predicts, specifically in
top-decile-heat matches, in a way that's stable enough to say something real about that player -
beyond what raw Elo alone already knows?

Same rigor as elite_opponent_residual_test.py's "capitulator vs. big-game hunter" test, which this
mirrors structurally (frozen per-tournament-edition Elo, chronological tournament-edition 80/20
train/test split, per-player logit-space shift fit on train and validated on held-out test,
player-clustered bootstrap CI) - and, critically, the SAME diagnostic that invalidated that
earlier finding: heterogeneity or held-out "improvement" concentrated in players whose Elo was
still adjusting during the training window (few prior career matches, so overall_elo is close to
STARTING_ELO and swings hard per result) is not real per-player signal - it's raw-Elo's own known
unreliability for undertrained players, showing up in a subset slice instead of overall.

Geographic/weather layer reused as-is from weather_upset_test.py (tournament_locations.py +
weather_fetch.py + build_weather_by_edition) - see that module's docstring for the coverage
caveats. The top-decile heat threshold is refit here on THIS test's own train-era sample (all
player-perspective rows, not just the favorite's) rather than reusing weather_upset_test.py's
numeric cutoff verbatim - the row population differs (every player's own perspective, not one row
per match), so refitting on the same train editions with the same 90th-percentile method is the
correct way to keep this "the same threshold" in the way that matters (same definition, same
train-only fitting discipline), not literally the same float.

Methodology notes / simplifications (stated up front, not buried):
  - Restricted to Court == "Outdoor", same reason as weather_upset_test.py.
  - Per-player residual = (actual win rate in that player's own hot-weather matches) - (Elo's
    average predicted win rate for those same matches), train era only, converted to a logit-space
    shift and applied to that SAME player's test-era hot-weather matches.
  - "Prior career matches" (the Elo-adjustment diagnostic) = how many matches (of any kind, any
    weather, any opponent) that player had already played, tour-wide, before the tournament
    EDITION in which a given hot-weather match happened - computed at the same edition-start
    granularity build_frozen_predictions freezes Elo at, so it lines up with what "the Elo used
    for this match" actually knew at the time. EARLY_CAREER_MATCH_THRESHOLD (20) is a heuristic,
    not a fitted value - roughly the point past which a player's Elo has absorbed enough results
    to stop being dominated by the STARTING_ELO prior.

Usage:
    python model/player_heat_heterogeneity_test.py [--tour ATP|WTA] [--min-train N] [--min-test N]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from elite_opponent_residual_test import (  # noqa: E402
    EPS, TRAIN_FRACTION, build_frozen_predictions, log_loss, logit, sigmoid,
)
from elo_ratings import load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from weather_upset_test import build_weather_by_edition, report_location_coverage  # noqa: E402

MIN_TEST_HOT_MATCHES = 4
EARLY_CAREER_MATCH_THRESHOLD = 20


def compute_prior_match_counts(matches, editions):
    """dict[(edition_id, player)] -> number of matches that player had already played, anywhere
    in the tour's history, strictly before this edition started. Mirrors build_frozen_predictions'
    own edition-by-edition loop exactly (same edition_id column, same chronological edition order)
    so "prior matches" lines up with the same frozen-Elo snapshot each edition's rows were scored
    against - a player's count is the number of results their edition-start Elo had already
    absorbed, not a count as of the exact match date within the edition."""
    df = matches.copy()
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)

    counts_by_edition = {}
    running_count = {}
    for edition_id in editions["edition_id"]:
        edition_matches = df[df["edition_id"] == edition_id]
        for player in pd.concat([edition_matches["Player_1"], edition_matches["Player_2"]]).unique():
            counts_by_edition[(edition_id, player)] = running_count.get(player, 0)
        for row in edition_matches.itertuples(index=False):
            running_count[row.Player_1] = running_count.get(row.Player_1, 0) + 1
            running_count[row.Player_2] = running_count.get(row.Player_2, 0) + 1
    return counts_by_edition


def run(tour, min_train_hot, min_test_hot):
    matches = load_matches_for_tour(tour)
    matches = report_location_coverage(matches, tour)

    outdoor_resolved = matches[(matches["Court"] == "Outdoor") & matches["resolved"].notna()].copy()
    weather_df = build_weather_by_edition(outdoor_resolved)
    print(f"\n{tour}: weather successfully retrieved for {weather_df['edition_id'].nunique()} "
          f"tournament editions, {len(weather_df)} unique (edition, date) days")

    preds, editions = build_frozen_predictions(matches)
    prior_counts = compute_prior_match_counts(matches, editions)
    preds["prior_career_matches"] = preds.apply(
        lambda r: prior_counts.get((r["edition_id"], r["player"]), 0), axis=1)

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    print(f"{tour}: {len(editions)} tournament editions; train = first {len(train_editions)} "
          f"(through {editions['edition_start'].iloc[split_idx - 1].date()}), "
          f"test = remaining {len(test_editions)} (from {editions['edition_start'].iloc[split_idx].date()})")

    # join weather onto EVERY player-perspective row (not just the favorite's) - this is a
    # per-player question, not a favorite/upset one.
    weather_matches = preds.merge(weather_df, on=["edition_id", "date"], how="inner")
    n_with_weather = len(weather_matches)
    n_total = len(preds)
    print(f"\n{tour}: {n_total} player-perspective rows total; {n_with_weather} "
          f"({n_with_weather / max(n_total, 1):.1%}) have a resolved location AND successfully-"
          f"fetched weather for their exact match date - this is the sample the per-player heat "
          f"test runs on")

    train_all = weather_matches[weather_matches["edition_id"].isin(train_editions)]
    test_all = weather_matches[weather_matches["edition_id"].isin(test_editions)]

    threshold = train_all["tmax"].quantile(0.90)
    print(f"\nTrain-era top-decile heat cutoff (90th percentile of train-era tmax, this test's own "
          f"player-perspective sample): {threshold:.1f}C")

    train_hot = train_all[train_all["tmax"] >= threshold].copy()
    test_hot = test_all[test_all["tmax"] >= threshold].copy()
    print(f"Train: {len(train_hot)} hot-weather player-perspective rows across "
          f"{train_hot['player'].nunique()} distinct players; Test: {len(test_hot)} rows across "
          f"{test_hot['player'].nunique()} distinct players")

    # --- per-player train-era residual in hot matches only ---
    train_stats = train_hot.groupby("player").agg(
        n_train=("actual_win", "size"),
        actual_rate=("actual_win", "mean"),
        pred_rate=("pred_win", "mean"),
        median_prior_matches=("prior_career_matches", "median"),
    ).reset_index()
    train_stats["residual"] = train_stats["actual_rate"] - train_stats["pred_rate"]
    train_stats["logit_shift"] = train_stats.apply(
        lambda r: logit(r["actual_rate"]) - logit(r["pred_rate"]), axis=1)
    train_stats["early_career"] = train_stats["median_prior_matches"] < EARLY_CAREER_MATCH_THRESHOLD

    eligible = train_stats[train_stats["n_train"] >= min_train_hot].copy()
    print(f"\nPlayers with >= {min_train_hot} hot-weather matches in the train era: {len(eligible)} "
          f"of {train_stats['player'].nunique()} players who played a hot-weather match at all")

    if len(eligible) < 5:
        print(f"\nOnly {len(eligible)} eligible players - too few for a meaningful heterogeneity "
              f"test. STOPPING HERE.")
        return

    # --- heterogeneity: is the spread of per-player hot-weather residuals bigger than sampling
    # noise alone would produce if every player secretly matched their own Elo in hot weather? ---
    def z_score(row):
        p = min(max(row["pred_rate"], EPS), 1 - EPS)
        se = math.sqrt(p * (1 - p) / row["n_train"])
        return row["residual"] / se

    eligible["z"] = eligible.apply(z_score, axis=1)
    chi2_stat = (eligible["z"] ** 2).sum()
    df_chi2 = len(eligible) - 1
    chi2_p = stats.chi2.sf(chi2_stat, df_chi2) if df_chi2 > 0 else float("nan")
    n_extreme = (eligible["z"].abs() > 1.96).sum()
    print(f"\nHeterogeneity check: {n_extreme}/{len(eligible)} eligible players have |z| > 1.96 "
          f"(~{n_extreme / len(eligible):.0%}, vs. ~5% expected under 'everyone matches their own "
          f"Elo in hot weather'); chi-square goodness-of-fit stat = {chi2_stat:.1f} on {df_chi2} df, "
          f"p = {chi2_p:.2g}")

    # --- Elo-still-adjusting diagnostic, run BEFORE trusting the heterogeneity number above ---
    extreme = eligible[eligible["z"].abs() > 1.96].sort_values("z", key=abs, ascending=False)
    print(f"\nElo-still-adjusting diagnostic for the {len(extreme)} |z| > 1.96 player(s) - median "
          f"prior tour-wide career matches at the time of their train-era hot-weather matches "
          f"(flagged 'early_career' if < {EARLY_CAREER_MATCH_THRESHOLD}, the same artifact that "
          f"invalidated the elite-opponent capitulator/hunter finding):")
    if len(extreme):
        print(extreme[["player", "n_train", "residual", "z", "median_prior_matches", "early_career"]]
              .to_string(index=False, formatters={"residual": "{:+.1%}".format, "z": "{:.2f}".format}))
    n_extreme_early = int(extreme["early_career"].sum())
    print(f"{n_extreme_early}/{len(extreme)} of the extreme players are flagged early_career.")

    # rerun the SAME heterogeneity test restricted to established players only (median prior
    # matches >= threshold) - does the chi-square signal survive once undertrained-Elo players are
    # excluded, or was it concentrated in exactly the players Elo can't be trusted for yet?
    established = eligible[~eligible["early_career"]].copy()
    if len(established) >= 5:
        chi2_est = (established["z"] ** 2).sum()
        df_est = len(established) - 1
        p_est = stats.chi2.sf(chi2_est, df_est) if df_est > 0 else float("nan")
        n_extreme_est = (established["z"].abs() > 1.96).sum()
        print(f"\nRestricted to the {len(established)} established players (median prior matches "
              f">= {EARLY_CAREER_MATCH_THRESHOLD}): {n_extreme_est}/{len(established)} have "
              f"|z| > 1.96, chi-square = {chi2_est:.1f} on {df_est} df, p = {p_est:.2g}")
    else:
        print(f"\nOnly {len(established)} established (non-early-career) players eligible - too "
              f"few to meaningfully rerun the heterogeneity test on that subset alone.")

    # --- held-out validation: does a player's train-era hot-weather logit shift predict their
    # OWN test-era hot-weather performance better than raw Elo alone? ---
    shift_by_player = dict(zip(eligible["player"], eligible["logit_shift"]))
    early_by_player = dict(zip(eligible["player"], eligible["early_career"]))
    test_eligible = test_hot[test_hot["player"].isin(shift_by_player)].copy()
    test_eligible["early_career"] = test_eligible["player"].map(early_by_player)
    test_eligible["adjusted_pred"] = test_eligible.apply(
        lambda r: sigmoid(logit(r["pred_win"]) + shift_by_player[r["player"]]), axis=1)
    test_eligible["raw_loss"] = log_loss(test_eligible["actual_win"].values, test_eligible["pred_win"].values)
    test_eligible["adj_loss"] = log_loss(test_eligible["actual_win"].values, test_eligible["adjusted_pred"].values)
    test_eligible["raw_brier"] = (test_eligible["actual_win"] - test_eligible["pred_win"]) ** 2
    test_eligible["adj_brier"] = (test_eligible["actual_win"] - test_eligible["adjusted_pred"]) ** 2

    n_test_players = test_eligible["player"].nunique()
    print(f"\nHeld-out validation: {len(test_eligible)} test-era hot-weather rows belonging to "
          f"eligible players, across {n_test_players} distinct players who played a hot-weather "
          f"match again in the test era")

    def _report_validation(df_subset, label):
        if len(df_subset) < 5 or df_subset["player"].nunique() < 2:
            print(f"  [{label}] only {len(df_subset)} rows / {df_subset['player'].nunique()} "
                  f"players - too few for a meaningful bootstrap CI, skipping")
            return
        print(f"  [{label}] Raw Elo        : log-loss = {df_subset['raw_loss'].mean():.4f}, "
              f"Brier = {df_subset['raw_brier'].mean():.4f}")
        print(f"  [{label}] Player-adjusted: log-loss = {df_subset['adj_loss'].mean():.4f}, "
              f"Brier = {df_subset['adj_brier'].mean():.4f}")
        observed, lo, hi = cluster_bootstrap_ci(df_subset, "raw_loss", "adj_loss", group_col="player")
        print(f"  [{label}] Mean per-match log-loss improvement (raw - adjusted, >0 = adjustment "
              f"helps), player-clustered: {observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")

    _report_validation(test_eligible, "ALL eligible players")
    _report_validation(test_eligible[~test_eligible["early_career"]], "established players only")
    _report_validation(test_eligible[test_eligible["early_career"]], "early-career players only")

    # per-player reliability table
    test_stats = test_hot.groupby("player").agg(n_test=("actual_win", "size")).reset_index()
    report = eligible.merge(test_stats, on="player", how="left")
    report["n_test"] = report["n_test"].fillna(0).astype(int)
    report = report.sort_values("residual", key=abs, ascending=False)

    print(f"\nPer-player hot-weather residual table, train era, sorted by |residual| "
          f"(n_train >= {min_train_hot} shown):")
    print(report[["player", "n_train", "actual_rate", "pred_rate", "residual", "z",
                   "median_prior_matches", "early_career", "n_test"]]
          .to_string(index=False, formatters={
              "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format,
              "residual": "{:+.1%}".format, "z": "{:.2f}".format,
          }))

    validated = report[report["n_test"] >= min_test_hot]
    print(f"\n{len(validated)} of those players ALSO have >= {min_test_hot} hot-weather matches in "
          f"the held-out test era - the only players for whom this analysis can claim any real "
          f"look at whether their train-era heat residual actually generalized:")
    print(validated[["player", "n_train", "residual", "early_career", "n_test"]].to_string(
        index=False, formatters={"residual": "{:+.1%}".format}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
    parser.add_argument("--min-train", type=int, default=8,
                         help="min train-era hot-weather matches to estimate a per-player residual at all")
    parser.add_argument("--min-test", type=int, default=MIN_TEST_HOT_MATCHES,
                         help="min test-era hot-weather matches for a player to count in the held-out reliability table")
    args = parser.parse_args()
    run(args.tour, args.min_train, args.min_test)
