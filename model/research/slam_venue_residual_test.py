"""Tests a "stable venue-specific performance residual" hypothesis, genuinely distinct from every
recency/reaction-speed mechanism tested tonight: does a player's win rate AT ONE SPECIFIC GRAND
SLAM, pooled across multiple editions of that same Slam, deviate from what their contemporaneous
overall Elo predicted for those same matches - in a way stable enough to help predict that same
player's FUTURE matches at that SAME Slam?

This is not about reacting faster to recent form (every mechanism tested earlier tonight - adaptive
K, signature-win boost, tiered recency weighting - was). It's asking whether a Slam is a durable,
repeatable "own venue" effect for a given player (surface aside - Elo already has surface-specific
ratings; a genuine venue effect would be a DIFFERENT axis - crowd, altitude/conditions, scheduling,
draw-quality artifacts of seeding at that specific event, etc.) A concrete named case this was asked
to check directly: does Osaka's US Open history (2x champion, 2018 and 2020) show a residual beyond
what her Elo at the time already predicted?

Same rigor as elite_opponent_residual_test.py, which this is structurally cloned from: frozen
per-tournament-edition Elo (single continuously-updated overall_elo, not the production pipeline's
surface-specific/lookback-windowed Elo - same stated simplification as that script, for the same
reason: recomputing the full windowed pipeline per edition is O(editions x window matches) and
grouping/attribution is the question here, not window mechanics), a chronological tournament-EDITION
80/20 train/test split (never random-match, which would leak a player's own future Slam form into
their own training residual), player-AND-VENUE-clustered residual estimation, and held-out
validation of the adjusted vs. raw prediction with a player-clustered bootstrap CI.

Per-(player, slam) residual = (actual win rate at that Slam, train era) - (Elo's average predicted
win rate for those same matches), converted to a logit-space shift and applied to that player's
TEST-era matches at that SAME Slam.

Usage:
    python model/research/slam_venue_residual_test.py [--tour ATP|WTA] [--min-train N]

FINAL VERDICT (2026-08-27): REJECTED, both tours, decisively - held-out log-loss got WORSE after
applying the venue-specific adjustment, not better:
  - WTA: 690 eligible (player, Slam) pairs (>=8 train matches), 1551 held-out rows across 252 pairs
    that recurred in the test era. Mean per-match log-loss improvement -0.0368, 95% bootstrap CI
    [-0.0591, -0.0158] - CI excludes zero, entirely on the "worse" side. Heterogeneity check: 37/690
    pairs (5.4%) have |z|>1.96, statistically indistinguishable from the ~5% expected by chance alone
    if every player's Slam performance secretly matched their own contemporaneous Elo exactly
    (chi-square p=0.37, not significant) - there is no real detectable population-level heterogeneity
    to even adjust FOR.
  - ATP: 837 eligible pairs, 1477 held-out rows across 236 recurring pairs. Improvement -0.0221, 95%
    CI [-0.0420, -0.0031] - also excludes zero, also worse. Heterogeneity check here DOES clear
    significance (54/837 = 6.5%, chi-square p=4.1e-05) - unlike WTA, there's real excess spread in
    train-era residuals - but it still doesn't generalize: this is the same overfitting signature as
    several of tonight's other rejected mechanisms (real-looking in-sample structure that is mostly
    noise dressed as signal once you actually test it on new data).

Named-case check, Osaka N. (WTA): a real train-era residual exists at BOTH her hard-court Slams -
Australian Open (n=29, actual 82.8% vs. Elo-predicted 63.7%, +19.0%, z=+2.13) and US Open (n=26,
actual 84.6% vs. predicted 65.0%, +19.6%, z=+2.10) - but NOT at French Open (-6.3%, z=-0.46) or
Wimbledon (+0.5%, z~0). That pattern - same-sized effect at both hard-court Slams, ~nothing at the
other two - is a hard-court-skill signature this single non-surface-specific overall_elo can't see,
not evidence of a US-Open-SPECIFIC venue effect; a true venue effect would show up at US Open without
also showing up at Melbourne. And with 690 WTA pairs tested at a 95% threshold, ~34-35 would clear
|z|>1.96 by pure chance; 37 did, so Osaka's two data points are exactly consistent with chance rather
than a real, isolable effect - especially given the population-level held-out test above already
shows the adjustment doesn't generalize even where the heterogeneity check does look real (ATP).

This closes the "stable venue-specific residual" hypothesis: genuinely distinct from the recency-
family mechanisms tested earlier the same night (adaptive K, signature-win boost, tiered recency
weighting - all also rejected, see elo_k_factor_full_historical_test.py, signature_win_boost_test.py,
tiered_recency_elo_test.py), but landing on the same practical conclusion by an independent route.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elo_ratings import K_FACTOR, STARTING_ELO, expected_score, load_matches_for_tour  # noqa: E402

TRAIN_FRACTION = 0.8
MIN_TEST_MATCHES = 3
EPS = 1e-3

SLAM_NAMES = {"Australian Open", "French Open", "US Open", "Wimbledon"}


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def log_loss(actual, pred):
    pred = np.clip(pred, EPS, 1 - EPS)
    return -(actual * np.log(pred) + (1 - actual) * np.log(1 - pred))


def build_frozen_predictions(df):
    """One player-perspective row per (match, player). player_elo/opponent_elo/pred_win are frozen
    at the tournament EDITION's start - identical for every round of that edition. Elo is then
    updated with the edition's real results before moving to the next (chronologically later)
    edition. Identical construction to elite_opponent_residual_test.py's build_frozen_predictions."""
    df = df[df["Date"].notna()].copy()
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)
    edition_start = df.groupby("edition_id")["Date"].transform("min")
    editions = (
        df.assign(edition_start=edition_start)[["edition_id", "edition_start", "Tournament"]]
        .drop_duplicates()
        .sort_values("edition_start")
        .reset_index(drop=True)
    )

    overall_elo = {}
    rows = []
    for edition_id in editions["edition_id"]:
        edition_matches = df[df["edition_id"] == edition_id]
        tournament = edition_matches["Tournament"].iloc[0]

        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            elo_p1 = overall_elo.get(p1, STARTING_ELO)
            elo_p2 = overall_elo.get(p2, STARTING_ELO)
            pred_p1 = expected_score(elo_p1, elo_p2)
            win1 = 1 if winner == p1 else 0
            rows.append((edition_id, tournament, row.Date, row.Round, p1, p2, elo_p1, pred_p1, win1))
            rows.append((edition_id, tournament, row.Date, row.Round, p2, p1, elo_p2, 1 - pred_p1, 1 - win1))

        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            overall_elo.setdefault(p1, STARTING_ELO)
            overall_elo.setdefault(p2, STARTING_ELO)
            score1 = 1.0 if winner == p1 else 0.0
            exp1 = expected_score(overall_elo[p1], overall_elo[p2])
            overall_elo[p1] += K_FACTOR * (score1 - exp1)
            overall_elo[p2] += K_FACTOR * ((1 - score1) - (1 - exp1))

    preds = pd.DataFrame(rows, columns=[
        "edition_id", "tournament", "date", "round", "player", "opponent", "player_elo",
        "pred_win", "actual_win",
    ])
    return preds, editions


def run(tour, min_train_matches, highlight):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    print(f"{tour}: {len(editions)} tournament editions "
          f"({editions['edition_start'].min().date()} to {editions['edition_start'].max().date()}); "
          f"train = first {len(train_editions)} editions (through "
          f"{editions['edition_start'].iloc[split_idx - 1].date()}), "
          f"test = remaining {len(test_editions)} editions "
          f"(from {editions['edition_start'].iloc[split_idx].date()})")

    slam = preds[preds["tournament"].isin(SLAM_NAMES)].copy()
    train_slam = slam[slam["edition_id"].isin(train_editions)]
    test_slam = slam[slam["edition_id"].isin(test_editions)]
    print(f"Slam player-perspective rows: {len(train_slam)} train-era, {len(test_slam)} test-era "
          f"(tournaments: {sorted(SLAM_NAMES)})")

    # population-level baseline: pooled over every player, does Elo's average prediction match
    # reality at Slams at all? (context, not the main test - Slams are best-of-5 for ATP, which
    # this single overall_elo doesn't model, so some population-level gap here is expected)
    pop_actual, pop_pred = train_slam["actual_win"].mean(), train_slam["pred_win"].mean()
    print(f"\nPopulation baseline (train era, all players pooled, all 4 Slams): actual win rate = "
          f"{pop_actual:.1%}, Elo's average predicted rate = {pop_pred:.1%} "
          f"(gap = {pop_actual - pop_pred:+.1%}, n={len(train_slam)})")

    train_stats = train_slam.groupby(["player", "tournament"]).agg(
        n_train=("actual_win", "size"), actual_rate=("actual_win", "mean"), pred_rate=("pred_win", "mean"),
    ).reset_index()
    train_stats["residual"] = train_stats["actual_rate"] - train_stats["pred_rate"]
    train_stats["logit_shift"] = train_stats.apply(
        lambda r: logit(r["actual_rate"]) - logit(r["pred_rate"]), axis=1)

    eligible = train_stats[train_stats["n_train"] >= min_train_matches].copy()
    print(f"\n(player, Slam) pairs with >= {min_train_matches} train-era matches at that specific "
          f"Slam (a real per-pair estimate is even attemptable): {len(eligible)} of "
          f"{len(train_stats)} pairs who played that Slam at all")

    # heterogeneity check: is the spread of per-(player,Slam) residuals bigger than sampling noise
    # alone would produce if every player secretly performed at every Slam exactly as their own
    # contemporaneous Elo predicted?
    def z_score(row):
        p = min(max(row["pred_rate"], EPS), 1 - EPS)
        se = math.sqrt(p * (1 - p) / row["n_train"])
        return row["residual"] / se

    eligible["z"] = eligible.apply(z_score, axis=1)
    chi2_stat = (eligible["z"] ** 2).sum()
    df_chi2 = len(eligible) - 1
    chi2_p = stats.chi2.sf(chi2_stat, df_chi2) if df_chi2 > 0 else float("nan")
    n_extreme = (eligible["z"].abs() > 1.96).sum()
    print(f"Heterogeneity check: {n_extreme}/{len(eligible)} eligible (player, Slam) pairs have "
          f"|z| > 1.96 (~{n_extreme / len(eligible):.0%}, vs. ~5% expected under 'everyone's Slam "
          f"performance matches their own contemporaneous Elo'); chi-square goodness-of-fit stat = "
          f"{chi2_stat:.1f} on {df_chi2} df, p = {chi2_p:.2g}")

    # held-out test: for eligible (player, Slam) pairs, does applying the train-era logit shift to
    # that SAME player's test-era matches at that SAME Slam beat raw (unadjusted) Elo?
    shift_by_pair = {(r.player, r.tournament): r.logit_shift for r in eligible.itertuples()}
    test_slam = test_slam.copy()
    test_slam["pair_key"] = list(zip(test_slam["player"], test_slam["tournament"]))
    test_eligible = test_slam[test_slam["pair_key"].isin(shift_by_pair)].copy()
    test_eligible["adjusted_pred"] = test_eligible.apply(
        lambda r: sigmoid(logit(r["pred_win"]) + shift_by_pair[r["pair_key"]]), axis=1)
    test_eligible["raw_loss"] = log_loss(test_eligible["actual_win"].values, test_eligible["pred_win"].values)
    test_eligible["adj_loss"] = log_loss(test_eligible["actual_win"].values, test_eligible["adjusted_pred"].values)
    test_eligible["raw_brier"] = (test_eligible["actual_win"] - test_eligible["pred_win"]) ** 2
    test_eligible["adj_brier"] = (test_eligible["actual_win"] - test_eligible["adjusted_pred"]) ** 2

    n_test_pairs = test_eligible["pair_key"].nunique()
    print(f"\nHeld-out validation: {len(test_eligible)} test-era rows belonging to eligible "
          f"(player, Slam) pairs, across {n_test_pairs} distinct pairs who actually played that "
          f"same Slam again in the test era")
    if len(test_eligible):
        print(f"  Raw Elo         : log-loss = {test_eligible['raw_loss'].mean():.4f}, "
              f"Brier = {test_eligible['raw_brier'].mean():.4f}")
        print(f"  Venue-adjusted  : log-loss = {test_eligible['adj_loss'].mean():.4f}, "
              f"Brier = {test_eligible['adj_brier'].mean():.4f}")

        # cluster (by pair) bootstrap on the per-match log-loss improvement, so one player's Slam
        # matches don't count as independent evidence many times over
        rng = np.random.default_rng(42)
        per_pair_delta = test_eligible.groupby("pair_key", group_keys=False).apply(
            lambda g: (g["raw_loss"] - g["adj_loss"]).mean(), include_groups=False)
        n_pairs = len(per_pair_delta)
        boot_means = [
            per_pair_delta.iloc[rng.integers(0, n_pairs, size=n_pairs)].mean()
            for _ in range(5000)
        ]
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        observed = per_pair_delta.mean()
        print(f"  Mean per-match log-loss improvement from the venue adjustment (raw - adjusted, "
              f">0 = adjustment better), (player,Slam)-clustered: {observed:+.4f}, "
              f"95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")

    # per-pair reliability table
    test_stats = test_slam.groupby(["player", "tournament"]).agg(n_test=("actual_win", "size")).reset_index()
    report = eligible.merge(test_stats, on=["player", "tournament"], how="left")
    report["n_test"] = report["n_test"].fillna(0).astype(int)
    report["reliability"] = np.select(
        [report["n_train"] >= 20, report["n_train"] >= min_train_matches],
        ["solid", "thin"], default="excluded",
    )
    report = report.sort_values("residual", key=abs, ascending=False)

    print(f"\nPer-(player, Slam) residual table, train era, sorted by |residual| "
          f"(n_train >= {min_train_matches} shown; 'solid' = n_train >= 20, else 'thin'), top 30:")
    print(report[["player", "tournament", "n_train", "actual_rate", "pred_rate", "residual", "n_test", "reliability"]]
          .head(30)
          .to_string(index=False, formatters={
              "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format, "residual": "{:+.1%}".format,
          }))

    validated = report[report["n_test"] >= MIN_TEST_MATCHES]
    print(f"\n{len(validated)} of those pairs ALSO have >= {MIN_TEST_MATCHES} matches at that same "
          f"Slam in the held-out test era - the only pairs for whom this analysis can claim any "
          f"real look at whether the train-era residual actually generalized:")
    print(validated[["player", "tournament", "n_train", "residual", "n_test"]].to_string(
        index=False, formatters={"residual": "{:+.1%}".format}))

    if highlight:
        print(f"\n{'=' * 70}\nNamed-case lookup: {highlight}\n{'=' * 70}")
        for slam_name in sorted(SLAM_NAMES):
            row = train_stats[(train_stats["player"] == highlight) & (train_stats["tournament"] == slam_name)]
            if row.empty:
                print(f"  {slam_name}: no train-era matches on record for {highlight}")
                continue
            r = row.iloc[0]
            test_row = test_stats[(test_stats["player"] == highlight) & (test_stats["tournament"] == slam_name)]
            n_test_h = int(test_row["n_test"].iloc[0]) if not test_row.empty else 0
            print(f"  {slam_name}: train n={int(r['n_train'])}, actual={r['actual_rate']:.1%}, "
                  f"Elo-predicted={r['pred_rate']:.1%}, residual={r['residual']:+.1%}"
                  + (f", z={eligible.loc[(eligible['player'] == highlight) & (eligible['tournament'] == slam_name), 'z'].iloc[0]:+.2f}"
                     if ((eligible["player"] == highlight) & (eligible["tournament"] == slam_name)).any() else "")
                  + f", n_test={n_test_h}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="WTA", choices=["ATP", "WTA"])
    parser.add_argument("--min-train", type=int, default=8,
                         help="min train-era matches at that specific Slam to estimate a per-pair residual at all")
    parser.add_argument("--highlight", default=None, help="ratings-csv-format player name to print a full 4-Slam breakdown for regardless of eligibility thresholds")
    args = parser.parse_args()
    run(args.tour, args.min_train, args.highlight)
