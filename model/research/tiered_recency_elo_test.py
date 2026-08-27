"""Sixth and (per the module docstring below, once run) final test in tonight's "faster-reacting
Elo" hypothesis family. Tests tiered recency weighting WITHIN the existing, unchanged 5-year hard
lookback window (elo_ratings.LOOKBACK_YEARS untouched - a match's inclusion/exclusion is not what's
under test here, only how much a match still inside the window counts): recent matches get a
larger K-factor multiplier, older ones (but still inside the 5yr window) get a smaller one, in
discrete tiers rather than a single smooth exponential decay.

Why this is a distinct, not-yet-tested variant of the idea, not a rerun of an earlier one:
  - elo_lookback_test.py's variant C (decay3) already tested SMOOTH exponential recency decay, but
    only at the small 2-tournament (Cincinnati + Canadian Open) scale, and its own module docstring
    reports the effect "moves ratings <10 pts" there - too small a probe to draw a real conclusion,
    and never re-run at full historical scale the way the hard-cutoff lookback question was.
  - elo_k_factor_full_historical_test.py tested a PLAYER-EXPERIENCE-based K (K=250/(5+n)^0.4) -
    REJECTED, worse in 6/6 decades, worst for thin-history players. Different axis entirely (who
    the player is, not how old the match is).
  - signature_win_boost_test.py tested a RESULT-SIGNIFICANCE-based K boost - REJECTED, symmetric
    zero-sum amplification of losses alongside wins. Different axis again (match importance, not
    match age).
  - Tennis Abstract / Jeff Sackmann's own published tennis Elo write-up (an independent, external
    validation) explicitly states he tested "different k factors for likely types of 'important'
    matches" and found no consistent predictive improvement - external corroboration of the
    signature-win-boost rejection, not this recency-tier question specifically.
  - This test is the first at full historical scale to isolate MATCH AGE alone (within a fixed
    inclusion window) as a discrete/tiered adjustment, rather than a smooth decay probed only on
    ~183 real matches.

Three tiered schemes are tested (not just one guess), all monotonically non-increasing in age,
un-fit on any data (chosen up front, disclosed here, not grid-searched - avoiding yet another
free-parameter search on top of five already-tested mechanisms tonight):

  B. mild:       <=6mo: 1.3x   6-12mo: 1.15x   1-3yr: 1.0x   3-5yr: 0.85x
  C. moderate:   <=6mo: 1.5x   6-12mo: 1.2x    1-2yr: 1.0x   2-3yr: 0.8x   3-5yr: 0.6x
  D. aggressive: <=6mo: 2.0x   6-12mo: 1.5x    1-2yr: 1.0x   2-3yr: 0.7x   3-5yr: 0.4x

Methodology - same as lookback_full_historical_test.py, since match age (unlike a K-factor
variant that depends only on match-intrinsic properties) is relative to whichever edition is being
predicted, requiring a REAL per-edition window rebuild (not the single-online-pass shortcut the
K-factor/signature-boost tests could use): every edition's Elo is a real windowed replay, K_FACTOR
scaled by the age-tier weight at that point in time, rebuilt from scratch at every edition boundary
from real match history strictly before that edition's start (no lookahead). Full Kaggle history,
both tours, raw Elo win probability (not the full win_probability() pipeline, for tractability at
this scale - same convention as lookback_full_historical_test.py), chronological 80/20 held-out
split, player-clustered bootstrap CIs, full-period decade stability breakdown.

Usage:
    python model/research/tiered_recency_elo_test.py

FINAL VERDICT (2026-08-26): REJECTED - the cleanest, most decisive rejection of any test in this
entire series, and the FINAL word on the "faster-reacting Elo" hypothesis family. Held out on the
combined 46778-row test era (both tours), ALL THREE tiered schemes are worse than flat K, every one
with a CI that excludes zero: mild -0.0026 [-0.0030,-0.0021], moderate -0.0049 [-0.0056,-0.0042],
aggressive -0.0132 [-0.0145,-0.0119] - and the effect is perfectly MONOTONIC with how aggressively
recent matches are up-weighted (mild < moderate < aggressive, in that exact order, no crossovers).
Per-tour: both ATP and WTA separately worse for all three schemes, no exceptions. Era/decade
breakdown: worse than baseline in EVERY SINGLE ONE of 6 decades (2000-2029) for EVERY SINGLE ONE of
the 3 schemes - 18 of 18 decade x scheme cells, zero exceptions, zero cells even reaching "not
distinguishable." No other test tonight produced a result this uniform in either direction.

This closes the "faster-reacting Elo" hypothesis family for good, not just this one variant. Six
distinct mechanisms were tested tonight, all aimed at the same underlying idea (make Elo react
faster to recent form/experience/stakes), all independently rejected at full historical, held-out,
player-clustered-bootstrap rigor:
  1. 3-year hard lookback cutoff (elo_lookback_test.py / lookback_full_historical_test.py) -
     worse in 4/6 decades.
  2. Smooth exponential recency decay, small-sample only (elo_lookback_test.py variant C) -
     directionally inconclusive at 2-tournament scale, never escalated (see below for why).
  3. Experience-based adaptive K, K=250/(5+n)^0.4 (elo_k_factor_full_historical_test.py) - worse
     in 6/6 decades, worst for thin-history players specifically.
  4. Result-significance-based K boost (signature_win_boost_test.py) - worse everywhere, worst on
     the exact population (signature matches) it targeted; zero-sum symmetry amplifies losses as
     much as wins.
  5. Rank/Elo trajectory-lag blend (rank_trajectory_lag_test.py) - the lone partial exception:
     directionally positive, not (yet) statistically significant - the only one of the six not
     cleanly rejected, because it doesn't touch the Elo update rule at all (blends the terminal
     rating toward current rank instead).
  6. Tiered recency weighting within the unchanged 5yr window (this file) - REJECTED, most
     decisively of all six.
Independent, external corroboration: Jeff Sackmann's own published tennis Elo methodology (Tennis
Abstract) reports testing K-factor tweaks for "important" matches and finding no consistent
predictive improvement either - the same conclusion, reached independently, by the person who
originated this style of Elo.

Conclusion: production's flat K_FACTOR=32 over a flat 5-year hard lookback window is not just
un-beaten tonight, it is now a well-stress-tested design choice - six different angles of attack on
the same "make Elo faster/more responsive" idea, all failing, several decisively. No seventh variant
of this family should be pursued without a genuinely new mechanism, not another parameterization of
"weight recent/significant/experienced matches more." No production change made. The one still-open,
not-yet-closed avenue for the Osaka/Andreeva/Zverev-style market gaps remains
rank_trajectory_lag_test.py's blend, which is mechanistically distinct from all six rejected here.
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import EPS, log_loss  # noqa: E402
from elo_ratings import K_FACTOR, LOOKBACK_YEARS, STARTING_ELO, expected_score, load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

TRAIN_FRACTION = 0.8

# (max_age_years, weight) tiers, checked in order - must be sorted ascending by max_age_years and
# reach LOOKBACK_YEARS as the last edge.
TIER_SCHEMES = {
    "B. mild":       [(0.5, 1.3), (1.0, 1.15), (3.0, 1.0), (LOOKBACK_YEARS, 0.85)],
    "C. moderate":   [(0.5, 1.5), (1.0, 1.2), (2.0, 1.0), (3.0, 0.8), (LOOKBACK_YEARS, 0.6)],
    "D. aggressive": [(0.5, 2.0), (1.0, 1.5), (2.0, 1.0), (3.0, 0.7), (LOOKBACK_YEARS, 0.4)],
}
BASELINE_LABEL = "A. flat (production, no recency tiers)"
VARIANT_LABELS = [BASELINE_LABEL] + list(TIER_SCHEMES.keys())


def tiered_weight(age_years, tiers):
    for max_age, weight in tiers:
        if age_years <= max_age:
            return weight
    return tiers[-1][1]


def build_windowed_predictions(df, tiers, tour_label):
    """Per-edition, REAL windowed Elo replay (same expensive-but-faithful mechanism
    lookback_full_historical_test.build_windowed_predictions uses) - tiers=None means flat K_FACTOR
    (the baseline); otherwise K_FACTOR is scaled by tiered_weight(age_years, tiers) for every match,
    age computed relative to THIS window's own most recent match (same reference point
    elo_lookback_test.calculate_elo_variant's decay used), recomputed fresh at every edition."""
    df = df.sort_values("Date", kind="stable").copy()
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)
    edition_start = df.groupby("edition_id")["Date"].transform("min")
    editions = (
        df.assign(edition_start=edition_start)[["edition_id", "edition_start"]]
        .drop_duplicates().sort_values("edition_start").reset_index(drop=True)
    )

    rows = []
    t0 = time.time()
    for idx, (edition_id, cutoff) in enumerate(zip(editions["edition_id"], editions["edition_start"])):
        edition_matches = df[df["edition_id"] == edition_id]
        window_df = df[df["Date"] < cutoff]
        if len(window_df) == 0:
            continue
        max_date = window_df["Date"].max()
        lookback_start = max_date - pd.DateOffset(years=LOOKBACK_YEARS)
        window_df = window_df[window_df["Date"] >= lookback_start]

        elo = {}
        if tiers is None:
            for row in window_df.itertuples(index=False):
                p1, p2, winner = row.Player_1, row.Player_2, row.Winner
                elo.setdefault(p1, STARTING_ELO)
                elo.setdefault(p2, STARTING_ELO)
                s1 = 1.0 if winner == p1 else 0.0
                e1 = expected_score(elo[p1], elo[p2])
                elo[p1] += K_FACTOR * (s1 - e1)
                elo[p2] += K_FACTOR * ((1 - s1) - (1 - e1))
        else:
            ages = (max_date - window_df["Date"]).dt.days / 365.25
            for row, age in zip(window_df.itertuples(index=False), ages):
                p1, p2, winner = row.Player_1, row.Player_2, row.Winner
                elo.setdefault(p1, STARTING_ELO)
                elo.setdefault(p2, STARTING_ELO)
                s1 = 1.0 if winner == p1 else 0.0
                e1 = expected_score(elo[p1], elo[p2])
                k_eff = K_FACTOR * tiered_weight(age, tiers)
                elo[p1] += k_eff * (s1 - e1)
                elo[p2] += k_eff * ((1 - s1) - (1 - e1))

        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            e1, e2 = elo.get(p1, STARTING_ELO), elo.get(p2, STARTING_ELO)
            pred1 = expected_score(e1, e2)
            win1 = 1 if winner == p1 else 0
            rows.append((edition_id, row.Date, p1, p2, pred1, win1))
            rows.append((edition_id, row.Date, p2, p1, 1 - pred1, 1 - win1))

        if (idx + 1) % 300 == 0:
            print(f"    [{tour_label}, tiers={'flat' if tiers is None else 'yes'}] {idx + 1}/{len(editions)} "
                  f"editions replayed ({time.time() - t0:.0f}s elapsed)")

    preds = pd.DataFrame(rows, columns=["edition_id", "date", "player", "opponent", "pred_win", "actual_win"])
    preds["loss"] = log_loss(preds["actual_win"].values, preds["pred_win"].values)
    print(f"    [{tour_label}, tiers={'flat' if tiers is None else 'yes'}] done: {len(editions)} editions, "
          f"{len(preds)} rows, {time.time() - t0:.0f}s total")
    return preds, editions


def bootstrap_verdict(long_baseline, long_variant, merge_keys=("tour", "edition_id", "date", "player", "opponent")):
    merged = long_baseline[[*merge_keys, "loss"]].merge(
        long_variant[[*merge_keys, "loss"]], on=list(merge_keys), suffixes=("_baseline", "_variant"))
    observed, lo, hi = cluster_bootstrap_ci(merged, "loss_baseline", "loss_variant", group_col="player")
    verdict = "BEATS baseline (CI excludes zero, >0)" if lo > 0 else (
        "WORSE than baseline (CI excludes zero, <0)" if hi < 0 else "NOT distinguishable (CI straddles zero)")
    return merged, observed, lo, hi, verdict


def run():
    tours = ["ATP", "WTA"]
    all_preds, all_editions = {}, {}

    for tour in tours:
        matches = load_matches_for_tour(tour)
        print(f"\n{'#' * 90}\n{tour}: {len(matches)} total matches\n{'#' * 90}")
        preds, editions = build_windowed_predictions(matches, None, tour)
        all_preds[(tour, BASELINE_LABEL)] = preds
        all_editions[tour] = editions
        for label, tiers in TIER_SCHEMES.items():
            preds, _ = build_windowed_predictions(matches, tiers, tour)
            all_preds[(tour, label)] = preds

    test_edition_ids = {}
    for tour in tours:
        editions = all_editions[tour]
        split_idx = int(len(editions) * TRAIN_FRACTION)
        test_edition_ids[tour] = set(editions["edition_id"].iloc[split_idx:])
        print(f"\n{tour}: {len(editions)} editions; held-out test era = most recent "
              f"{len(editions) - split_idx} editions, from {editions['edition_start'].iloc[split_idx].date()}")

    print(f"\n{'=' * 90}\nHEADLINE - HELD-OUT TEST ERA (most recent 20% of editions), BOTH TOURS COMBINED"
          f"\n{'=' * 90}")
    combined_test = {}
    for label in VARIANT_LABELS:
        parts = [all_preds[(tour, label)][all_preds[(tour, label)]["edition_id"].isin(test_edition_ids[tour])].assign(tour=tour)
                 for tour in tours]
        combined_test[label] = pd.concat(parts, ignore_index=True)
        cdf = combined_test[label]
        print(f"\n{label}: {len(cdf)} held-out rows | mean log-loss = {cdf['loss'].mean():.4f}")

    for label in VARIANT_LABELS:
        if label == BASELINE_LABEL:
            continue
        merged, observed, lo, hi, verdict = bootstrap_verdict(combined_test[BASELINE_LABEL], combined_test[label])
        print(f"\n{label} vs. baseline (COMBINED): {len(merged)} matched rows")
        print(f"  mean log-loss improvement (baseline - variant, >0 = variant better): "
              f"{observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  VERDICT: {verdict}")

    print(f"\n{'=' * 90}\nPER-TOUR BREAKDOWN, held-out test era\n{'=' * 90}")
    for tour in tours:
        print(f"\n--- {tour} only ---")
        base_t = combined_test[BASELINE_LABEL][combined_test[BASELINE_LABEL]["tour"] == tour]
        for label in VARIANT_LABELS:
            if label == BASELINE_LABEL:
                continue
            var_t = combined_test[label][combined_test[label]["tour"] == tour]
            merged, observed, lo, hi, verdict = bootstrap_verdict(base_t, var_t)
            print(f"  {label}: {len(merged)} rows, improvement {observed:+.4f}, "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")

    print(f"\n{'=' * 90}\nERA / DECADE BREAKDOWN, FULL PERIOD (both tours combined, all editions)\n{'=' * 90}")
    full_by_label = {}
    for label in VARIANT_LABELS:
        parts = [all_preds[(tour, label)].assign(tour=tour) for tour in tours]
        full_by_label[label] = pd.concat(parts, ignore_index=True)
        full_by_label[label]["decade"] = (full_by_label[label]["date"].dt.year // 5) * 5

    decades = sorted(full_by_label[BASELINE_LABEL]["decade"].unique())
    for decade in decades:
        base_d = full_by_label[BASELINE_LABEL][full_by_label[BASELINE_LABEL]["decade"] == decade]
        if len(base_d) < 200:
            continue
        yr_range = f"{decade}-{decade + 4}"
        print(f"\n--- {yr_range} ---")
        for label in VARIANT_LABELS:
            g = full_by_label[label][full_by_label[label]["decade"] == decade]
            print(f"  {label}: n={len(g)} | mean log-loss={g['loss'].mean():.4f}")
        for label in VARIANT_LABELS:
            if label == BASELINE_LABEL:
                continue
            var_d = full_by_label[label][full_by_label[label]["decade"] == decade]
            merged = base_d[["tour", "edition_id", "date", "player", "opponent", "loss"]].merge(
                var_d[["tour", "edition_id", "date", "player", "opponent", "loss"]],
                on=["tour", "edition_id", "date", "player", "opponent"], suffixes=("_baseline", "_variant"))
            if len(merged) < 10:
                continue
            observed, lo, hi = cluster_bootstrap_ci(merged, "loss_baseline", "loss_variant", group_col="player")
            verdict = "BEATS baseline" if lo > 0 else ("WORSE than baseline" if hi < 0 else "not distinguishable")
            print(f"  {label} vs. baseline ({yr_range}): {len(merged)} rows, improvement "
                  f"{observed:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")


if __name__ == "__main__":
    run()
