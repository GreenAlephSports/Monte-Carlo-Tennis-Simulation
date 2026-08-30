"""Definitive, full-historical validation of the "decay3" recency-decay lookback variant
prototyped in elo_lookback_test.py (variant C): no hard cutoff at all - the full available match
history is used - but each match's effective K-factor is scaled by a recency weight: full weight
(1.0) inside 3 years of the training window's own most recent match, decaying with a 2-year
half-life beyond that. That small-sample check (elo_lookback_test.py, Cincinnati + Canadian Open,
n=183) was directionally inconclusive and was never escalated to full scale - unlike hard3 (the
OTHER lookback variant tested that same night), which WAS escalated (lookback_full_historical_
test.py) and rejected. This closes that gap: same full-Kaggle-history, held-out, player-clustered-
bootstrap rigor as every other correction tested this series, so decay3 gets a real, final verdict
instead of sitting as "never finished."

Methodology - identical in spirit to lookback_full_historical_test.py (which this file's build
function is directly adapted from), because the reason that file couldn't use the cheap single-
online-pass shortcut elo_k_factor_full_historical_test.py used applies here too, for the same
reason: decay3's per-match weight is age relative to a MOVING reference point (each training
window's own most-recent match date, which shifts at every tournament edition boundary as cutoff_
date advances) - not a fixed function of a player's own match sequence the way a K-factor variant
is. So this is a genuine full per-edition rebuild (elo_lookback_test.calculate_elo_variant's exact
mechanism), not a shortcut - O(editions x window_size), the expensive, deliberately-not-
approximated case, same as the hard3 test.

Baseline unchanged from every other test tonight: production's 5-year hard cutoff. The single
variant under test is decay3 (lookback_years=None, decay_half_life_years=2.0, full_weight_years=
3.0) - reusing elo_lookback_test.calculate_elo_variant directly rather than reimplementing the
weighting logic, so there is zero risk of an accidental mismatch between what was prototyped at
small scale and what gets tested here at full scale.

Same rigor/process as lookback_full_historical_test.py: raw Elo win probability (expected_score,
not the full win_probability() pipeline), overall Elo only (no surface-blending), chronological
80/20 tournament-edition train/test split for the headline held-out claim, full-period decade
breakdown as an additional stability check (not a second independent held-out claim - this fits no
parameters on train data, so every edition is already out-of-sample relative to its own frozen
pre-edition Elo by construction), and the same ATP-only veteran/prime-age breakdown
lookback_full_historical_test.py used (no WTA birthdate source available).

Usage:
    python model/research/decay3_full_historical_test.py

FINAL VERDICT (2026-08-27): ACCEPTED as a real, held-out-validated effect - a genuinely different
outcome from hard3 (the other lookback-family variant, REJECTED at this same full scale in
lookback_full_historical_test.py). NOT yet merged into production pending a decision on scope: this
is a bigger structural change than an additive correction like RECENT_FORM_BETA (it replaces
elo_ratings.calculate_elo_ratings' entire lookback/windowing mechanism, not something laid on top
of it), so it's reported here as a validated candidate, not auto-applied.

Combined held-out (both tours, 46778 rows, 561 editions): +0.0002 log-loss improvement, 95%
player-clustered bootstrap CI [+0.0000, +0.0003] - BEATS baseline, CI excludes zero. Same magnitude
of effect as the recent-form residual correction (also +0.0002), for reference.

Per-tour: WTA alone is where the real signal lives - +0.0004, CI [+0.0001, +0.0006], BEATS baseline
cleanly. ATP alone is NOT distinguishable from baseline (+0.0001, CI [-0.0001, +0.0002]) - the
combined headline result is carried by WTA, not a uniform effect across tours.

Age breakdown (ATP only): neither veteran (age>=33: -0.0004, CI [-0.0008, +0.0000]) nor prime-age
(+0.0001, CI [-0.0001, +0.0003]) is individually significant - decay3 is not simply hard3's
veteran-specific benefit wearing a smoother shape; ATP's overall null result holds in both age
splits, not just in aggregate.

Decade breakdown (both tours combined, full period): BEATS baseline in 4 of 6 decades (2005-2009,
2010-2014, 2015-2019, 2020-2024, all CI excludes zero), NOT distinguishable in the other 2
(2000-2004, 2025-2029) - critically, UNLIKE hard3, there is no decade where decay3 is significantly
WORSE. That is the substantive difference between the two lookback-family variants at full scale:
hard3 traded a small overall loss for a real veteran-specific gain; decay3 shows a small overall
gain with no offsetting loss anywhere in the 46-year decade sweep.

Targeted-power follow-up (2026-08-27), full-period per-tour decade breakdown - checking whether
ATP's null held-out result is a genuine absence of effect or just the pooled number being
underpowered/diluted: it is neither "genuinely absent" nor "diluted noise" - ATP has a REAL,
individually-significant effect in 3 of 6 decades (2005-2009: +0.0002 [+0.0001,+0.0003];
2010-2014: +0.0006 [+0.0004,+0.0007], the single largest positive decade effect in either tour;
2015-2019: +0.0004 [+0.0002,+0.0006]), all CI excludes zero. But it is flat everywhere in the most
recent era - 2020-2024: -0.0000 [-0.0003,+0.0002]; 2025-2029: -0.0001 [-0.0003,+0.0001] - and the
held-out test split (most recent 20% of editions, 2021-2026) falls almost entirely inside that flat
recent era, which is exactly why the pooled ATP held-out number came back not-distinguishable. WTA
shows the mirror-image pattern: flat in its earlier decades (2005-2009: +0.0000; 2010-2014:
+0.0000, both null) and significant only recently (2015-2019: +0.0004 [+0.0002,+0.0006]; 2020-2024:
+0.0008 [+0.0006,+0.0011], the strongest single decade effect anywhere) - which is why WTA's
held-out result (also drawn from the most recent era) came back significant. So the ATP/WTA held-out
split isn't "WTA has the effect, ATP doesn't" - it's that each tour's real decay-sensitivity era has
already passed for ATP and is current for WTA. Practically: for TODAY's predictions (2026 brackets),
decay3's validated benefit is real for WTA and NOT currently demonstrated for ATP, even though ATP
data shows the same mechanism worked for a 15-year stretch that ended around 2020 - a meaningfully
different, more cautious read than "decay3 works, mostly via WTA" would suggest on its own.

Conclusion: decay3 clears the same held-out bar the other five "faster-reacting Elo" mechanisms
failed, without hard3's downside - but ONLY for WTA under today's data; ATP's held-out-relevant
(most recent) era shows no benefit despite real effects earlier in ATP's history. A legitimate
candidate for a future production change, but any implementation decision should account for this
tour asymmetry (e.g. WTA-only, or investigate further why ATP's effect faded post-2020) rather than
applying decay3 uniformly to both tours on the strength of the combined number alone. Because this
touches the core Elo computation (not an additive win_probability.py correction), it should go
through an explicit go/no-go decision before being wired in, not be merged as a byproduct of this
test. No production change made yet.
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import match_name_to_pool  # noqa: E402
from elite_opponent_residual_test import EPS, log_loss  # noqa: E402
from elo_lookback_test import calculate_elo_variant  # noqa: E402
from elo_ratings import expected_score, load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from veteran_decline_test import load_birthdates  # noqa: E402

TRAIN_FRACTION = 0.8
AGE_THRESHOLD = 33
VARIANTS = {
    "A. baseline (5yr hard cutoff, production)": dict(lookback_years=5, decay_half_life_years=None),
    "B. decay3 (no cutoff, full weight <=3yr, 2yr half-life decay beyond)":
        dict(lookback_years=None, decay_half_life_years=2.0, full_weight_years=3.0),
}
BASELINE_LABEL = "A. baseline (5yr hard cutoff, production)"


def build_windowed_predictions(df, variant_kwargs, tour_label):
    """Per-edition, REAL windowed/weighted Elo replay via elo_lookback_test.calculate_elo_variant
    (the exact small-scale-prototyped mechanism, reused unchanged) - rebuilt from scratch at every
    edition boundary from real match history strictly before that edition's start (no lookahead)."""
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
        ratings = calculate_elo_variant(df, cutoff, **variant_kwargs)
        elo = dict(zip(ratings["player"], ratings["overall_elo"]))

        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            e1, e2 = elo.get(p1, 1500.0), elo.get(p2, 1500.0)
            pred1 = expected_score(e1, e2)
            win1 = 1 if winner == p1 else 0
            rows.append((edition_id, row.Date, p1, p2, pred1, win1))
            rows.append((edition_id, row.Date, p2, p1, 1 - pred1, 1 - win1))

        if (idx + 1) % 300 == 0:
            print(f"    [{tour_label}, variant] {idx + 1}/{len(editions)} editions replayed "
                  f"({time.time() - t0:.0f}s elapsed)")

    preds = pd.DataFrame(rows, columns=["edition_id", "date", "player", "opponent", "pred_win", "actual_win"])
    preds["loss"] = log_loss(preds["actual_win"].values, preds["pred_win"].values)
    print(f"    [{tour_label}, variant] done: {len(editions)} editions, {len(preds)} "
          f"player-perspective rows, {time.time() - t0:.0f}s total")
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
    all_preds = {}
    all_editions = {}

    for tour in tours:
        matches = load_matches_for_tour(tour)
        print(f"\n{'#' * 90}\n{tour}: {len(matches)} total matches, "
              f"{matches['Date'].min().date()} to {matches['Date'].max().date()}\n{'#' * 90}")
        for label, variant_kwargs in VARIANTS.items():
            preds, editions = build_windowed_predictions(matches, variant_kwargs, tour)
            all_preds[(tour, label)] = preds
            all_editions[tour] = editions

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
    # ERA / DECADE breakdown, FULL period (both tours) - stability check, not a second held-out claim
    # ============================================================================
    print(f"\n{'=' * 90}\nERA / DECADE BREAKDOWN, FULL PERIOD (both tours combined, all editions)"
          f"\n{'=' * 90}")
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

    # ============================================================================
    # PER-TOUR DECADE breakdown, FULL period - targeted power check: does ATP show a real,
    # decade-specific effect that the pooled ATP-only held-out number (which straddled zero) is
    # too underpowered to see, or does averaging across ATP's full 46-year history in narrower
    # slices just confirm it's genuinely absent everywhere in ATP unlike WTA? Uses the full period
    # (not just held-out) for the same reason the combined decade breakdown above does - more power,
    # already-disclosed as a stability check rather than an independent held-out claim.
    # ============================================================================
    print(f"\n{'=' * 90}\nPER-TOUR DECADE BREAKDOWN, FULL PERIOD - targeted power check for whether "
          f"ATP has a real decade-specific effect hidden inside its pooled null result"
          f"\n{'=' * 90}")
    for tour in tours:
        print(f"\n{'-' * 60}\n{tour} only, by decade\n{'-' * 60}")
        tour_decades = sorted(full_by_label[BASELINE_LABEL][full_by_label[BASELINE_LABEL]["tour"] == tour]["decade"].unique())
        for decade in tour_decades:
            base_d = full_by_label[BASELINE_LABEL][
                (full_by_label[BASELINE_LABEL]["tour"] == tour) & (full_by_label[BASELINE_LABEL]["decade"] == decade)]
            if len(base_d) < 200:
                continue
            yr_range = f"{decade}-{decade + 4}"
            for label in VARIANTS:
                if label == BASELINE_LABEL:
                    continue
                var_d = full_by_label[label][
                    (full_by_label[label]["tour"] == tour) & (full_by_label[label]["decade"] == decade)]
                merged = base_d[["edition_id", "date", "player", "opponent", "loss"]].merge(
                    var_d[["edition_id", "date", "player", "opponent", "loss"]],
                    on=["edition_id", "date", "player", "opponent"], suffixes=("_baseline", "_variant"))
                if len(merged) < 10:
                    continue
                observed, lo, hi = cluster_bootstrap_ci(merged, "loss_baseline", "loss_variant", group_col="player")
                verdict = "BEATS baseline" if lo > 0 else ("WORSE than baseline" if hi < 0 else "not distinguishable")
                print(f"  {tour} {yr_range}: n={len(merged)} rows, improvement {observed:+.4f}, "
                      f"95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")


if __name__ == "__main__":
    run()
