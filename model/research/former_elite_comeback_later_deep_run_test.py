"""Extends former_elite_comeback_deep_run_test.py's explicitly-disclosed scope gap: that script
only counted a "deep run after a comeback" if it happened in the SAME tournament as the comeback
window, and only counted Semifinal/Final. Both restrictions were named there as real limitations,
not settled conclusions - four of the real, sourced injury cases from this session's own research
(Djokovic's 2018 elbow surgery -> Wimbledon 2018 title, Federer's 2016 knee surgery -> Australian
Open 2017 title, Zverev's 2022 French Open ankle surgery -> French Open 2023 semifinal a year
later, Sharapova's 2008 shoulder surgery -> French Open 2012 title three years later) motivated
this follow-up directly: Zverev's and Sharapova's real comebacks are structurally invisible to a
same-tournament-only design by construction, no matter how real the underlying recovery was.

THIS script: a "comeback window" (first match of an edition in a real, recorded layoff bucket -
LAYOFF_BUCKET taxonomy, >= 30 days since that player's last recorded match, exactly as before) now
counts as followed by a deep run if the player reaches QUARTERFINALS OR BETTER (broadened from
Semifinal/Final - round_order >= 5, survivorship_upset_test.ROUND_ORDER's own scale) in ANY of
their real tournament editions - not just the comeback edition itself - within WINDOW_DAYS_PRIMARY
(365 days / 12 months) of that comeback window's start, with WINDOW_DAYS_SENSITIVITY (548 days /
~18 months) reported as a sensitivity check. The former-elite-below-peak gate (gap_below_peak >=
FORMER_ELITE_GAP_THRESHOLD, years_of_history >= MIN_HISTORY_YEARS) is evaluated AT THE COMEBACK
WINDOW itself, never at the later deep-run edition - evaluating it later would be circular (a real
deep run mechanically raises current Elo, shrinking gap_below_peak as a DIRECT consequence of the
very outcome being tested for, not an independent pre-condition).

Right-censoring, handled explicitly rather than silently miscounted: a comeback window in, say,
mid-2026 does not have a full 12/18-month forward window of real match data available in this
dataset yet - those instances are EXCLUDED from both the comeback and baseline pools (not counted
as "no deep run found", which would be a real, avoidable bias toward under-counting recent cases),
using each tour's own last recorded match date as the horizon.

INFERENTIAL TEST (beyond pure population-counting): does having a real recorded layoff (comeback
bucket) predict reaching a QF+ deep run within the window MORE than an ordinary below-peak run with
no real recent layoff does (baseline = below-peak editions whose first_match_bucket is a normal
in-season gap, under_14d/14_30d - i.e. down in form for some other reason, not coming back from a
recorded absence)? Player-clustered bootstrap CI on the difference in hit rate
(former_elite_vs_current_top_test.cluster_bootstrap_group_diff, reused directly - a hit-rate
difference is the same "mean(actual - 0) vs mean(actual - 0) between two groups" shape that function
already computes), split into a chronological train/test stability check (same TRAIN_FRACTION
boundary as the rest of this project) - not a fit/validate pair, since nothing is fit here, purely
a robustness check on whether the same pattern holds across two independent eras.

SPOT-CHECK, the actual point of this extension: Djokovic 2018, Federer 2017, Nadal 2022, and
Zverev's 2023 delayed comeback are looked up directly and reported whether they are now correctly
captured - honestly, including printing WHY a case is or isn't captured (gap_below_peak too small,
years_of_history coverage, bucket boundary) rather than silently passing or failing.

Usage:
    python model/research/former_elite_comeback_later_deep_run_test.py
    python model/research/former_elite_comeback_later_deep_run_test.py --gap-threshold 150 --min-history-years 2
"""
import argparse
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import TRAIN_FRACTION, build_frozen_predictions  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from former_elite_vs_current_top_test import (  # noqa: E402
    FORMER_ELITE_GAP_THRESHOLD_DEFAULT, MIN_HISTORY_YEARS_DEFAULT, cluster_bootstrap_group_diff,
)
from layoff_test import build_layoff_dataset  # noqa: E402
from layoff_within_tournament_decay_test import build_within_tournament_sequences  # noqa: E402
from peak_elo_recovery_test import add_peak_features  # noqa: E402
from survivorship_upset_test import ROUND_ORDER  # noqa: E402

COMEBACK_BUCKETS = {"30_60d", "60_90d", "90d_plus"}       # >= 30 real days off - same as before
BASELINE_BUCKETS = {"under_14d", "14_30d"}                 # normal in-season gap, not a real layoff
WINDOW_DAYS_PRIMARY = 365       # ~12 months
WINDOW_DAYS_SENSITIVITY = 548   # ~18 months
DEEP_ROUND_ORDER_MIN = ROUND_ORDER["Quarterfinals"]         # broadened from Semifinals this time
MIN_INSTANCES_FOR_STATS = 20


def build_run_level(tour, gap_threshold, min_history_years):
    """One row per (player, edition): deepest round reached, first_match_bucket (that run's OWN
    comeback classification), gap_below_peak/years_of_history AS OF that edition - UNFILTERED by
    the former-elite gate (the gate is applied by the caller only to the STARTING comeback window,
    never to the later editions being scanned for a deep run, per the docstring's circularity note)."""
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)
    preds = add_peak_features(preds)
    preds["gap_below_peak"] = (preds["career_peak_prior"] - preds["player_elo"]).clip(lower=0)

    layoff_df = build_layoff_dataset(matches, preds)
    seq = build_within_tournament_sequences(layoff_df)
    seq["round_order"] = seq["round"].map(ROUND_ORDER)

    peak_lvl = preds[["player", "edition_id", "gap_below_peak", "years_of_history"]].drop_duplicates(
        subset=["player", "edition_id"])
    seq = seq.merge(peak_lvl, on=["player", "edition_id"], how="left", validate="many_to_one")

    edition_start = editions.set_index("edition_id")["edition_start"]
    run_level = seq.groupby(["edition_id", "player"], sort=False).agg(
        deepest_round_order=("round_order", "max"),
        first_match_bucket=("first_match_bucket", "first"),
        gap_below_peak=("gap_below_peak", "first"),
        years_of_history=("years_of_history", "first"),
    ).reset_index()
    run_level["edition_start"] = run_level["edition_id"].map(edition_start)
    run_level["tour"] = tour
    data_last_date = matches["Date"].max()
    return run_level, editions, data_last_date


def build_player_index(run_level):
    """player -> (sorted edition_start ndarray, matching deepest_round_order ndarray, matching
    edition_id ndarray) - used for a binary-search window lookup instead of a per-row full scan."""
    idx = {}
    for player, g in run_level.sort_values("edition_start").groupby("player", sort=False):
        idx[player] = (
            g["edition_start"].values.astype("datetime64[ns]"),
            g["deepest_round_order"].values,
            g["edition_id"].values,
        )
    return idx


def deep_run_in_window(idx, player, start, window_days):
    """Best (deepest_round_order, edition_id, edition_start, days_later) among that player's real
    editions with edition_start in [start, start + window_days] inclusive - None if no editions
    fall in that window at all (distinct from 'found editions but none deep enough', which returns
    a real round_order that just doesn't clear DEEP_ROUND_ORDER_MIN)."""
    if player not in idx:
        return None
    dates, rounds, eids = idx[player]
    end = start + pd.Timedelta(days=window_days)
    lo = bisect_left(dates, np.datetime64(start))
    hi = bisect_right(dates, np.datetime64(end))
    if lo >= hi:
        return None
    window_rounds = rounds[lo:hi]
    best_i = lo + int(np.argmax(window_rounds))
    days_later = (dates[best_i] - np.datetime64(start)) / np.timedelta64(1, "D")
    return rounds[best_i], eids[best_i], dates[best_i], days_later


def build_pools(run_level, gap_threshold, min_history_years):
    base = run_level[(run_level["gap_below_peak"] >= gap_threshold) & (run_level["years_of_history"] >= min_history_years)]
    comeback_pool = base[base["first_match_bucket"].isin(COMEBACK_BUCKETS)].copy()
    baseline_pool = base[base["first_match_bucket"].isin(BASELINE_BUCKETS)].copy()
    return comeback_pool, baseline_pool


def score_pool(pool, idx, data_last_date, window_days, pool_label):
    n_before_censor = len(pool)
    eligible = pool[pool["edition_start"] + pd.Timedelta(days=window_days) <= data_last_date].copy()
    n_censored = n_before_censor - len(eligible)

    hits, best_rounds = [], []
    for row in eligible.itertuples(index=False):
        result = deep_run_in_window(idx, row.player, row.edition_start, window_days)
        if result is None:
            hits.append(0)
            best_rounds.append(np.nan)
        else:
            round_order, _eid, _date, _days = result
            hits.append(1 if round_order >= DEEP_ROUND_ORDER_MIN else 0)
            best_rounds.append(round_order)
    eligible["hit"] = hits
    eligible["best_round_order"] = best_rounds
    eligible["zero"] = 0.0

    print(f"  {pool_label}: {n_before_censor} instances, {n_censored} excluded (insufficient "
          f"follow-up - within {window_days}d of this dataset's last recorded match), "
          f"{len(eligible)} scored -> hit rate {eligible['hit'].mean():.1%}" if len(eligible) else
          f"  {pool_label}: {n_before_censor} instances, {n_censored} excluded, 0 scored")
    return eligible


def run_for_window(window_days, all_comeback, all_baseline, idx, data_last_date_by_tour, editions_by_tour, label):
    print(f"\n{'=' * 90}\n{label} (window = {window_days} days)\n{'=' * 90}")

    comeback_scored = pd.concat([
        score_pool(all_comeback[all_comeback["tour"] == t], idx, data_last_date_by_tour[t], window_days, f"COMEBACK pool [{t}]")
        for t in ["ATP", "WTA"]
    ], ignore_index=True)
    baseline_scored = pd.concat([
        score_pool(all_baseline[all_baseline["tour"] == t], idx, data_last_date_by_tour[t], window_days, f"BASELINE pool [{t}]")
        for t in ["ATP", "WTA"]
    ], ignore_index=True)

    if len(comeback_scored) < MIN_INSTANCES_FOR_STATS or len(baseline_scored) < MIN_INSTANCES_FOR_STATS:
        print(f"\n  TOO FEW SCORED INSTANCES (comeback n={len(comeback_scored)}, baseline "
              f"n={len(baseline_scored)}, need >= {MIN_INSTANCES_FOR_STATS} each) - not computing a "
              f"hit-rate comparison at this window.")
        return None

    comeback_scored["pool"] = "comeback"
    baseline_scored["pool"] = "baseline"
    combined = pd.concat([comeback_scored, baseline_scored], ignore_index=True)
    obs, lo, hi, n_valid = cluster_bootstrap_group_diff(combined, "hit", "zero", "pool", "comeback", "baseline")
    print(f"\n  Hit-rate difference (comeback - baseline), player-clustered bootstrap "
          f"({n_valid}/5000 valid replicates): observed {obs:+.1%}, "
          + (f"95% CI [{lo:+.1%}, {hi:+.1%}]" if n_valid >= 20 else "too few valid replicates for a CI"))
    if n_valid >= 20:
        verdict = ("comeback pool reaches QF+ MORE often (CI excludes zero, >0)" if lo > 0 else
                   ("comeback pool reaches QF+ LESS often (CI excludes zero, <0)" if hi < 0 else
                    "NOT distinguishable (CI straddles zero)"))
        print(f"  VERDICT (full sample): {verdict}")

    for tour in ["ATP", "WTA"]:
        editions = editions_by_tour[tour]
        split_idx = int(len(editions) * TRAIN_FRACTION)
        train_editions = set(editions["edition_id"].iloc[:split_idx])
        tour_combined = combined[combined["tour"] == tour]
        for split_label, mask in [("train-era", tour_combined["edition_id"].isin(train_editions)),
                                   ("test-era", ~tour_combined["edition_id"].isin(train_editions))]:
            sub = tour_combined[mask]
            n_c, n_b = (sub["pool"] == "comeback").sum(), (sub["pool"] == "baseline").sum()
            if n_c < MIN_INSTANCES_FOR_STATS or n_b < MIN_INSTANCES_FOR_STATS:
                print(f"  {tour} {split_label}: comeback n={n_c}, baseline n={n_b} - too few to bootstrap")
                continue
            obs_s, lo_s, hi_s, nv_s = cluster_bootstrap_group_diff(sub, "hit", "zero", "pool", "comeback", "baseline")
            v = ("REAL >0" if nv_s >= 20 and lo_s > 0 else ("REAL <0" if nv_s >= 20 and hi_s < 0 else "not distinguishable"))
            print(f"  {tour} {split_label}: comeback n={n_c} (hit {sub[sub['pool']=='comeback']['hit'].mean():.1%}), "
                  f"baseline n={n_b} (hit {sub[sub['pool']=='baseline']['hit'].mean():.1%}), "
                  f"diff {obs_s:+.1%} -> {v}")

    return {"window_days": window_days, "n_comeback": len(comeback_scored), "n_baseline": len(baseline_scored),
            "observed": obs, "lo": lo, "hi": hi, "n_valid": n_valid}


def spot_check(name, tour, date_lo, date_hi, gap_threshold, min_history_years, all_run_level, idx, note):
    print(f"\n--- SPOT CHECK: {name} ({tour}), {date_lo} to {date_hi} - {note} ---")
    g = all_run_level[
        (all_run_level["tour"] == tour) & (all_run_level["player"] == name)
        & (all_run_level["edition_start"] >= pd.Timestamp(date_lo)) & (all_run_level["edition_start"] <= pd.Timestamp(date_hi))
    ].sort_values("edition_start")
    if g.empty:
        print(f"  No editions found for '{name}' in this window - check the exact draw-name spelling "
              f"(Kaggle uses 'Lastname F.' style).")
        return
    for row in g.itertuples(index=False):
        deepest_round = [k for k, v in ROUND_ORDER.items() if v == row.deepest_round_order]
        deepest_round = deepest_round[0] if deepest_round else f"order={row.deepest_round_order}"
        is_comeback = row.first_match_bucket in COMEBACK_BUCKETS
        is_former_elite = (row.gap_below_peak >= gap_threshold) and (row.years_of_history >= min_history_years)
        print(f"  {row.edition_id} (start {row.edition_start.date()}): first_match_bucket="
              f"{row.first_match_bucket}, deepest round reached={deepest_round}, "
              f"gap_below_peak={row.gap_below_peak:.1f}, years_of_history={row.years_of_history:.1f} "
              f"-> comeback window? {is_comeback}; former-elite-below-peak gate met? {is_former_elite}")
        if is_comeback and is_former_elite:
            for window_days, wlabel in [(WINDOW_DAYS_PRIMARY, "12mo"), (WINDOW_DAYS_SENSITIVITY, "18mo")]:
                result = deep_run_in_window(idx, name, row.edition_start, window_days)
                if result is None:
                    print(f"    [{wlabel} window] no editions found in window")
                    continue
                round_order, eid, date, days_later = result
                rname = [k for k, v in ROUND_ORDER.items() if v == round_order]
                rname = rname[0] if rname else f"order={round_order}"
                hit = round_order >= DEEP_ROUND_ORDER_MIN
                print(f"    [{wlabel} window] deepest reached anywhere in window: {rname} at {eid} "
                      f"({date.astype('datetime64[D]')}, {days_later:.0f} days later) -> "
                      f"QF+ hit? {hit}")


def run(gap_threshold, min_history_years):
    run_levels, editions_by_tour, data_last_date_by_tour = [], {}, {}
    for tour in ["ATP", "WTA"]:
        rl, editions, last_date = build_run_level(tour, gap_threshold, min_history_years)
        editions_by_tour[tour] = editions
        data_last_date_by_tour[tour] = last_date
        print(f"{tour}: {len(rl)} total (player, edition) runs, data through {last_date.date()}")
        run_levels.append(rl)
    all_run_level = pd.concat(run_levels, ignore_index=True)
    idx = build_player_index(all_run_level)

    comeback_pool, baseline_pool = build_pools(all_run_level, gap_threshold, min_history_years)
    print(f"\n{'=' * 90}\nPOOLS (former-elite-below-peak gate applied AT the starting edition only)\n"
          f"{'=' * 90}")
    print(f"COMEBACK pool (first_match_bucket >= 30 real days off): {len(comeback_pool)} instances, "
          f"{comeback_pool['player'].nunique()} distinct players")
    print(f"BASELINE pool (first_match_bucket = normal in-season gap): {len(baseline_pool)} instances, "
          f"{baseline_pool['player'].nunique()} distinct players")

    results = []
    for window_days, label in [(WINDOW_DAYS_PRIMARY, "PRIMARY"), (WINDOW_DAYS_SENSITIVITY, "SENSITIVITY")]:
        r = run_for_window(window_days, comeback_pool, baseline_pool, idx, data_last_date_by_tour, editions_by_tour, label)
        if r is not None:
            results.append(r)

    print(f"\n{'=' * 90}\nSPOT CHECKS - real, sourced injury cases from this session's research\n{'=' * 90}")
    spot_check("Djokovic N.", "ATP", "2018-01-01", "2018-12-31", gap_threshold, min_history_years,
               all_run_level, idx, "elbow surgery Feb 2018 -> real comeback culminated in Wimbledon 2018 title")
    spot_check("Federer R.", "ATP", "2016-06-01", "2017-06-01", gap_threshold, min_history_years,
               all_run_level, idx, "meniscus surgery Feb 2016, out 2nd half of 2016 -> Australian Open 2017 title")
    spot_check("Nadal R.", "ATP", "2021-08-01", "2022-03-01", gap_threshold, min_history_years,
               all_run_level, idx, "foot procedure Sep 2021 -> Australian Open 2022 title")
    spot_check("Zverev A.", "ATP", "2022-06-01", "2023-07-01", gap_threshold, min_history_years,
               all_run_level, idx, "ankle ligament surgery June 2022 -> French Open 2023 semifinal, ~1yr later")

    print(f"\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
    for r in results:
        months = r["window_days"] / 30.44
        print(f"  ~{months:.0f}mo window: comeback n={r['n_comeback']} baseline n={r['n_baseline']} "
              f"hit-rate diff={r['observed']:+.1%} "
              + (f"95% CI [{r['lo']:+.1%}, {r['hi']:+.1%}]" if r["n_valid"] >= 20 else "(no CI - too few valid replicates)"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gap-threshold", type=float, default=FORMER_ELITE_GAP_THRESHOLD_DEFAULT)
    parser.add_argument("--min-history-years", type=float, default=MIN_HISTORY_YEARS_DEFAULT)
    args = parser.parse_args()
    run(args.gap_threshold, args.min_history_years)
