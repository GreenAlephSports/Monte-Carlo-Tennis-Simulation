"""Definitive, full-historical validation of LOOKBACK_YEARS: hard3 (3yr hard cutoff) vs. the
current production 5yr hard cutoff, across the ENTIRE Kaggle match history for both tours (tens
of thousands of real matches, ~2,800 tournament editions, 2000-2026 ATP / 2006-2026 WTA) - not a
single tournament (Cincinnati) or a two-tournament sample (Cincinnati + Canadian Open), which
earlier tonight showed the effect shrinking and losing per-tournament significance as more data
was added. This is the real test of whether that pattern continues (fixable, genuine, but small)
or reverses (real, stable, worth a foundational change).

Methodology - deliberately more faithful to the real lookback mechanism than the OTHER
"same rigor" tests tonight (elite_opponent_residual_test.py, veteran_decline_test.py), which both
explicitly use a single continuously-updated overall_elo with NO windowing at all (a documented
simplification there, since the window itself wasn't what those tests were probing). Here the
window IS the thing under test, so every tournament edition's Elo is a REAL windowed replay -
elo_ratings.calculate_elo_ratings/apply_training_window's exact mechanism, rebuilt from scratch at
every edition boundary from real match history strictly before that edition's start (no
lookahead), for each of the two lookback_years values being compared. Overall Elo only (no
surface-blending) - same simplification elite_opponent_residual_test/veteran_decline_test already
made, and necessary to keep ~2,800 x 2 variants x 2 tours full-window replays tractable.

Raw Elo win probability (expected_score) is compared directly - not the full win_probability()
pipeline (rank-adjustment/layoff/confidence-calibration), same convention as
elite_opponent_residual_test.py and veteran_decline_test.py, and necessary for this to run in
reasonable time (the full pipeline's per-match ratings CSV write/read, used for the small
Cincinnati/Canadian-Open checks tonight, would not scale to this).

Unlike the OTHER production corrections tested tonight, this comparison fits NO parameters on the
train-era data - hard3 and baseline are both fully mechanical Elo-window variants, so every
edition's prediction is already genuinely out-of-sample relative to that edition's own Elo by
construction (frozen before the edition's own results are known). The headline number below still
uses the standard chronological 80/20 tournament-edition split (test = the most recent 20% of
editions) to match the same rigor/process as every other correction tonight; the full-period
decade breakdown is reported as an additional, honest stability check across eras, not a second
independent "held-out" claim - clarified here rather than overclaiming a train/test split where
none was structurally required.

Age/veteran breakdown is ATP-only, same disclosed limitation as veteran_decline_test.py and
elo_lookback_test.py: no WTA birthdate source was found (Jeff Sackmann's tennis_wta repo is gone,
same as tennis_atp), only Tennismylife/TML-Database's ATP_Database.csv.

Usage:
    python model/research/lookback_full_historical_test.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import match_name_to_pool  # noqa: E402
from elite_opponent_residual_test import EPS, log_loss  # noqa: E402
from elo_ratings import K_FACTOR, STARTING_ELO, expected_score, load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from veteran_decline_test import load_birthdates  # noqa: E402

TRAIN_FRACTION = 0.8
AGE_THRESHOLD = 33
VARIANTS = {
    "A. baseline (5yr hard cutoff, production)": 5,
    "B. hard3 (3yr hard cutoff)": 3,
}
BASELINE_LABEL = "A. baseline (5yr hard cutoff, production)"


def build_windowed_predictions(df, lookback_years, tour_label):
    """Per-edition, REAL windowed Elo replay (not a single online pass) - the exact
    apply_training_window/calculate_elo_ratings mechanism, rebuilt from scratch at every edition
    boundary. O(editions x window_size); this is the expensive, deliberately-not-approximated part
    of this test."""
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
        lookback_start = max_date - pd.DateOffset(years=lookback_years)
        window_df = window_df[window_df["Date"] >= lookback_start]

        elo = {}
        for row in window_df.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            elo.setdefault(p1, STARTING_ELO)
            elo.setdefault(p2, STARTING_ELO)
            s1 = 1.0 if winner == p1 else 0.0
            e1 = expected_score(elo[p1], elo[p2])
            elo[p1] += K_FACTOR * (s1 - e1)
            elo[p2] += K_FACTOR * ((1 - s1) - (1 - e1))

        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            e1, e2 = elo.get(p1, STARTING_ELO), elo.get(p2, STARTING_ELO)
            pred1 = expected_score(e1, e2)
            win1 = 1 if winner == p1 else 0
            rows.append((edition_id, row.Date, p1, p2, pred1, win1))
            rows.append((edition_id, row.Date, p2, p1, 1 - pred1, 1 - win1))

        if (idx + 1) % 300 == 0:
            print(f"    [{tour_label}, lookback={lookback_years}yr] {idx + 1}/{len(editions)} "
                  f"editions replayed ({time.time() - t0:.0f}s elapsed)")

    preds = pd.DataFrame(rows, columns=["edition_id", "date", "player", "opponent", "pred_win", "actual_win"])
    preds["loss"] = log_loss(preds["actual_win"].values, preds["pred_win"].values)
    print(f"    [{tour_label}, lookback={lookback_years}yr] done: {len(editions)} editions, "
          f"{len(preds)} player-perspective rows, {time.time() - t0:.0f}s total")
    return preds, editions


def attach_age(preds, birthdate_by_name):
    pool_names = list(birthdate_by_name.index)
    unique_players = preds["player"].unique()
    resolved = {p: match_name_to_pool(p, pool_names) for p in unique_players}
    preds = preds.copy()
    preds["birthdate"] = preds["player"].map(resolved).map(birthdate_by_name)
    preds["age_years"] = (preds["date"] - preds["birthdate"]).dt.days / 365.25
    return preds


def bootstrap_verdict(long_baseline, long_variant, merge_keys=("edition_id", "date", "player", "opponent")):
    merged = long_baseline[[*merge_keys, "loss"]].merge(
        long_variant[[*merge_keys, "loss"]], on=list(merge_keys), suffixes=("_baseline", "_variant"))
    observed, lo, hi = cluster_bootstrap_ci(merged, "loss_baseline", "loss_variant", group_col="player")
    verdict = "BEATS baseline (CI excludes zero, >0)" if lo > 0 else (
        "WORSE than baseline (CI excludes zero, <0)" if hi < 0 else "NOT distinguishable (CI straddles zero)")
    return merged, observed, lo, hi, verdict


def run():
    tours = ["ATP", "WTA"]
    all_preds = {}    # (tour, variant_label) -> preds df
    all_editions = {}  # tour -> editions df (same regardless of variant)

    for tour in tours:
        matches = load_matches_for_tour(tour)
        print(f"\n{'#' * 90}\n{tour}: {len(matches)} total matches, "
              f"{matches['Date'].min().date()} to {matches['Date'].max().date()}\n{'#' * 90}")
        for label, lookback_years in VARIANTS.items():
            preds, editions = build_windowed_predictions(matches, lookback_years, f"{tour}")
            all_preds[(tour, label)] = preds
            all_editions[tour] = editions

    # chronological 80/20 split per tour (editions are identical across variants within a tour)
    test_edition_ids = {}
    for tour in tours:
        editions = all_editions[tour]
        split_idx = int(len(editions) * TRAIN_FRACTION)
        test_edition_ids[tour] = set(editions["edition_id"].iloc[split_idx:])
        print(f"\n{tour}: {len(editions)} editions total; held-out test era = most recent "
              f"{len(editions) - split_idx} editions, from "
              f"{editions['edition_start'].iloc[split_idx].date()} to "
              f"{editions['edition_start'].iloc[-1].date()}")

    # ============================================================================
    # HEADLINE: held-out (last 20% of editions), BOTH tours combined
    # ============================================================================
    print(f"\n{'=' * 90}\nHEADLINE - HELD-OUT TEST ERA (most recent 20% of tournament editions), "
          f"BOTH TOURS COMBINED\n{'=' * 90}")
    combined_test = {}
    for label in VARIANTS:
        parts = []
        for tour in tours:
            df = all_preds[(tour, label)]
            parts.append(df[df["edition_id"].isin(test_edition_ids[tour])].assign(tour=tour))
        combined_test[label] = pd.concat(parts, ignore_index=True)
        cdf = combined_test[label]
        print(f"\n{label}: {len(cdf)} held-out player-perspective rows ({cdf['edition_id'].nunique()} editions)")
        print(f"  mean log-loss = {cdf['loss'].mean():.4f}")
        print(f"  favorite calibration: assigned P(win) = {cdf['pred_win'].mean():.1%}, "
              f"actual win rate = {cdf['actual_win'].mean():.1%}")

    merge_keys = ("tour", "edition_id", "date", "player", "opponent")
    for label in VARIANTS:
        if label == BASELINE_LABEL:
            continue
        merged, observed, lo, hi, verdict = bootstrap_verdict(
            combined_test[BASELINE_LABEL], combined_test[label], merge_keys=merge_keys)
        print(f"\n{label} vs. baseline (HELD-OUT, COMBINED): {len(merged)} matched rows")
        print(f"  mean log-loss improvement (baseline - variant, >0 = variant better): "
              f"{observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  VERDICT: {verdict}")

    # ============================================================================
    # PER-TOUR breakdown, held-out era
    # ============================================================================
    print(f"\n{'=' * 90}\nPER-TOUR BREAKDOWN, held-out test era\n{'=' * 90}")
    for tour in tours:
        print(f"\n--- {tour} only ---")
        base_t = combined_test[BASELINE_LABEL][combined_test[BASELINE_LABEL]["tour"] == tour]
        for label in VARIANTS:
            if label == BASELINE_LABEL:
                continue
            var_t = combined_test[label][combined_test[label]["tour"] == tour]
            merged, observed, lo, hi, verdict = bootstrap_verdict(base_t, var_t, merge_keys=merge_keys)
            print(f"  {label}: {len(merged)} rows, improvement {observed:+.4f}, "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")

    # ============================================================================
    # AGE BREAKDOWN (ATP only - no WTA birthdate source), held-out era
    # ============================================================================
    print(f"\n{'=' * 90}\nAGE BREAKDOWN (ATP only, no WTA birthdate source), held-out test era, "
          f"age >= {AGE_THRESHOLD} = veteran\n{'=' * 90}")
    birthdate_by_name = load_birthdates()
    atp_test = {label: combined_test[label][combined_test[label]["tour"] == "ATP"].copy() for label in VARIANTS}
    for label in VARIANTS:
        atp_test[label] = attach_age(atp_test[label], birthdate_by_name)
    coverage = atp_test[BASELINE_LABEL]["age_years"].notna().mean()
    print(f"Age match coverage (ATP held-out rows): {coverage:.1%}")

    for group_label, cond in [("VETERAN (age >= 33)", lambda d: d["age_years"] >= AGE_THRESHOLD),
                               ("PRIME-AGE (age < 33)", lambda d: d["age_years"] < AGE_THRESHOLD)]:
        print(f"\n--- {group_label} ---")
        base_g = atp_test[BASELINE_LABEL][cond(atp_test[BASELINE_LABEL])]
        for label in VARIANTS:
            g = atp_test[label][cond(atp_test[label])]
            n = len(g)
            if n == 0:
                continue
            print(f"  {label}: n={n} | assigned P(win)={g['pred_win'].mean():.1%} | "
                  f"actual win rate={g['actual_win'].mean():.1%} | "
                  f"gap={g['pred_win'].mean() - g['actual_win'].mean():+.1%} | "
                  f"mean log-loss={g['loss'].mean():.4f}")
        for label in VARIANTS:
            if label == BASELINE_LABEL:
                continue
            var_g = atp_test[label][cond(atp_test[label])]
            merged = base_g[["edition_id", "date", "player", "opponent", "loss"]].merge(
                var_g[["edition_id", "date", "player", "opponent", "loss"]],
                on=["edition_id", "date", "player", "opponent"], suffixes=("_baseline", "_variant"))
            if len(merged) < 10:
                print(f"  {label} vs. baseline ({group_label}): only {len(merged)} rows - too few to bootstrap")
                continue
            observed, lo, hi = cluster_bootstrap_ci(merged, "loss_baseline", "loss_variant", group_col="player")
            verdict = "BEATS baseline" if lo > 0 else ("WORSE than baseline" if hi < 0 else "not distinguishable")
            print(f"  {label} vs. baseline ({group_label}): {len(merged)} rows, improvement "
                  f"{observed:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")

    # ============================================================================
    # ERA / DECADE breakdown, FULL period (both tours) - not just the held-out 20%, since this
    # comparison fits no parameters on train data (see module docstring) so every edition already
    # qualifies as out-of-sample relative to its own frozen pre-edition Elo; this is a stability
    # check across history, not a second independent held-out claim.
    # ============================================================================
    print(f"\n{'=' * 90}\nERA / DECADE BREAKDOWN, FULL PERIOD (both tours combined, all editions - "
          f"see module docstring for why the full period is valid here)\n{'=' * 90}")
    full_by_label = {}
    for label in VARIANTS:
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
        for label in VARIANTS:
            g = full_by_label[label][full_by_label[label]["decade"] == decade]
            print(f"  {label}: n={len(g)} | mean log-loss={g['loss'].mean():.4f} | "
                  f"favorite calib: assigned={g['pred_win'].mean():.1%} vs actual={g['actual_win'].mean():.1%}")
        for label in VARIANTS:
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
