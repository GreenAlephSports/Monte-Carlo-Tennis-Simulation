"""Matched-pairs version of the peak-Elo hypothesis, requested explicitly as a check on whether it
is meaningfully different from the already-tested peak_elo_recovery_test.py regression or
substantially the same question in a different shape.

HONEST FRAMING, stated up front rather than left implicit: this IS testing the same underlying
latent quantity peak_elo_recovery_test.py tested - "does a player's career-peak Elo carry real
predictive information beyond what their CURRENT Elo already captures" - not a new hypothesis. The
two designs are different ESTIMATORS of a related-but-not-identical estimand, not different
questions:
  - peak_elo_recovery_test.py: a continuous logistic regression, adjusted_logit = logit(pred_win) +
    beta*gap_below_peak/100, fit across EVERY real match with a defined career_peak_prior
    (regardless of the current-Elo gap in that specific match) - a global, parametric estimate of
    the marginal effect of gap_below_peak, pooling near-even matchups with wildly lopsided ones.
    NOT VALIDATED held-out (see that script's own printed verdict).
  - THIS script: restricts to the single narrowest, cleanest slice where the peak signal would be
    easiest to see if real - matches where the two players' CURRENT Elo is within
    MATCH_BAND_ELO_PTS (30, with 25 as a stricter check) of each other, i.e. Elo alone says this is
    close to a coin flip - and asks whether the side with a MEANINGFULLY higher career peak
    (>= MEANINGFUL_PEAK_GAP_PTS, 100 Elo) wins more often than Elo's own (near-50%) prediction for
    that side. Non-parametric in the sense that it doesn't rely on the logistic offset correctly
    extrapolating across large Elo gaps - but it is answering the SAME question, at one specific,
    cleanly-controlled point in the space that question lives in.

Prior expectation, disclosed before running rather than after: given peak_elo_recovery_test.py's
regression already integrates over this near-zero-gap region as PART of its overall (not validated)
estimate, the base-rate expectation is that this matched-pairs slice replicates the same null -
a real, robust effect should show up in BOTH designs, and this script's own verdict explicitly
states whether that expectation held or not, rather than presenting a null result here as if it
were newly-discovered independent evidence, or a positive result here as fully independent
confirmation if the pooled regression didn't show it (a real but narrow effect concentrated at this
one slice, diluted to insignificance in the pooled regression, is an honest possible outcome that
would need to be flagged as SUCH - not as "yet more confirmation" of the same finding).

Design:
  1. Frozen per-tournament-edition Elo (elite_opponent_residual_test.build_frozen_predictions) +
     career_peak_prior/years_of_history for BOTH sides of every real match (peak_elo_recovery_test.
     add_peak_features for the 'player' side, a self-join - same pattern former_elite_form_signal_
     test.add_opponent_recent_form already used, round included in the join key for the same
     edition_id-collision reason documented there - for the 'opponent' side).
  2. One row per REAL MATCH (not per player-perspective - preds has two mirror rows per match;
     deduplicated here on (edition_id, date, round, frozenset(player, opponent)) to avoid double-
     counting every match twice, which wouldn't bias the win-rate but would overstate n and corrupt
     the player-clustered bootstrap).
  3. Eligible = current-Elo gap <= MATCH_BAND_ELO_PTS AND both sides have >= MIN_HISTORY_YEARS
     coverage AND |peak_gap| >= MEANINGFUL_PEAK_GAP_PTS.
  4. higher_peak_player = whichever side has the higher career_peak_prior on that row; residual =
     actual_win_for_higher_peak_side - pred_win_for_higher_peak_side (NOT vs. a naive 50% - Elo's
     own prediction already reflects the small edge from the current-Elo gap inside the match band,
     so the honest test is whether peak explains anything BEYOND that, exactly the same logic as
     every other correction in this pipeline).
  5. Player-clustered bootstrap CI on mean residual (reusing survivorship_upset_test.
     cluster_bootstrap_ci directly, clustered by higher_peak_player identity - the side being
     credited on each row), computed on TRAIN-era and TEST-era rows separately (chronological
     tournament-edition split, same boundary as the rest of this project) for robustness, since
     there is no parameter to fit here that a train/test split would protect against overfitting -
     the split is reported as a stability check, not a fit/validate pair.

Usage:
    python model/research/peak_elo_matched_pairs_test.py
    python model/research/peak_elo_matched_pairs_test.py --match-band 25 --peak-gap 150
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import TRAIN_FRACTION, build_frozen_predictions  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from former_elite_vs_current_top_test import MIN_HISTORY_YEARS_DEFAULT  # noqa: E402
from peak_elo_recovery_test import add_peak_features  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

MATCH_BAND_CANDIDATES = [30, 25]   # Elo points - "current Elo essentially tied"
MEANINGFUL_PEAK_GAP_DEFAULT = 100.0  # Elo points - "meaningfully higher" career peak


def add_opponent_peak(preds):
    """Self-join preds onto itself, player/opponent swapped, to read the OPPONENT's own
    career_peak_prior/years_of_history (peak_elo_recovery_test.add_peak_features already computed
    these for the 'player' side) for the SAME real match - not recomputed. Join key includes
    'round' and is deduplicated on the swapped side before merging, same edition_id-collision
    defense former_elite_form_signal_test.add_opponent_recent_form already documented and used."""
    swapped = preds[["edition_id", "date", "round", "player", "opponent", "career_peak_prior", "years_of_history"]].copy()
    swapped = swapped.rename(columns={
        "player": "opponent", "opponent": "player",
        "career_peak_prior": "opponent_career_peak_prior", "years_of_history": "opponent_years_of_history",
    })
    swapped = swapped.drop_duplicates(subset=["edition_id", "date", "round", "player", "opponent"])
    merged = preds.merge(swapped, on=["edition_id", "date", "round", "player", "opponent"], how="left")
    assert len(merged) == len(preds), "opponent-peak self-join changed row count"
    return merged


def build_match_level(tour, min_history_years):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)
    preds = add_peak_features(preds)
    preds = add_opponent_peak(preds)

    preds["pair_key"] = preds.apply(
        lambda r: (r["edition_id"], r["date"], r["round"], frozenset((r["player"], r["opponent"]))), axis=1)
    match_level = preds.drop_duplicates(subset="pair_key", keep="first").copy()
    match_level["tour"] = tour

    match_level["current_elo_gap"] = (match_level["player_elo"] - match_level["opponent_elo"]).abs()
    match_level["peak_gap"] = match_level["career_peak_prior"] - match_level["opponent_career_peak_prior"]
    match_level["coverage_ok"] = (
        (match_level["years_of_history"] >= min_history_years)
        & (match_level["opponent_years_of_history"] >= min_history_years)
        & match_level["career_peak_prior"].notna() & match_level["opponent_career_peak_prior"].notna()
    )
    return match_level, editions


def run_for_band(match_band, peak_gap_threshold, all_match_level, editions_by_tour):
    print(f"\n{'=' * 90}\nMATCH_BAND = {match_band} Elo pts (current Elo essentially tied), "
          f"MEANINGFUL_PEAK_GAP = {peak_gap_threshold:.0f} Elo pts\n{'=' * 90}")

    eligible = all_match_level[
        all_match_level["coverage_ok"]
        & (all_match_level["current_elo_gap"] <= match_band)
        & (all_match_level["peak_gap"].abs() >= peak_gap_threshold)
    ].copy()
    print(f"Eligible matched-pair matches: {len(eligible)}, "
          f"{pd.concat([eligible['player'], eligible['opponent']]).nunique()} distinct players involved")

    if len(eligible) == 0:
        print("  No eligible matches at this band/threshold - cannot test.")
        return None

    higher_is_player = eligible["peak_gap"] > 0
    eligible["higher_peak_player_name"] = np.where(higher_is_player, eligible["player"], eligible["opponent"])
    eligible["actual_win_higher_peak"] = np.where(higher_is_player, eligible["actual_win"], 1 - eligible["actual_win"])
    eligible["pred_win_higher_peak"] = np.where(higher_is_player, eligible["pred_win"], 1 - eligible["pred_win"])
    eligible["player"] = eligible["higher_peak_player_name"]  # for cluster_bootstrap_ci's default group_col

    naive_rate = eligible["actual_win_higher_peak"].mean()
    elo_rate = eligible["pred_win_higher_peak"].mean()
    print(f"  Higher-career-peak side: actual win rate {naive_rate:.1%}, Elo's own predicted rate for "
          f"that side {elo_rate:.1%} (already reflects the small edge from being the current-Elo-"
          f"favored side within the match band) -> naive residual {naive_rate - elo_rate:+.1%}")

    observed, lo, hi = cluster_bootstrap_ci(eligible, "actual_win_higher_peak", "pred_win_higher_peak")
    print(f"\n  Player-clustered bootstrap (ALL eligible rows, both tours, full history): "
          f"mean residual (actual - Elo-predicted) for the higher-peak side = {observed:+.1%}, "
          f"95% CI [{lo:+.1%}, {hi:+.1%}]")
    verdict_all = "REAL (CI excludes zero, >0)" if lo > 0 else ("REAL, WRONG SIGN (CI excludes zero, <0)" if hi < 0 else "NOT distinguishable (CI straddles zero)")
    print(f"  VERDICT (full sample): {verdict_all}")

    # train/test stability check (not a fit/validate pair - nothing is fit here - just a robustness split)
    for tour in ["ATP", "WTA"]:
        editions = editions_by_tour[tour]
        split_idx = int(len(editions) * TRAIN_FRACTION)
        train_editions = set(editions["edition_id"].iloc[:split_idx])
        tour_elig = eligible[eligible["tour"] == tour]
        train_e = tour_elig[tour_elig["edition_id"].isin(train_editions)]
        test_e = tour_elig[~tour_elig["edition_id"].isin(train_editions)]
        print(f"\n  {tour} stability check: {len(train_e)} train-era / {len(test_e)} test-era rows")
        for label, sub in [("train-era", train_e), ("test-era", test_e)]:
            if len(sub) < 20:
                print(f"    {label}: n={len(sub)} - too few to bootstrap")
                continue
            obs_s, lo_s, hi_s = cluster_bootstrap_ci(sub, "actual_win_higher_peak", "pred_win_higher_peak")
            v = "REAL >0" if lo_s > 0 else ("REAL <0" if hi_s < 0 else "not distinguishable")
            print(f"    {label}: n={len(sub)}, residual {obs_s:+.1%}, 95% CI [{lo_s:+.1%}, {hi_s:+.1%}] -> {v}")

    return {"match_band": match_band, "n": len(eligible), "observed": observed, "lo": lo, "hi": hi}


def run(min_history_years, peak_gap_threshold):
    frames, editions_by_tour = [], {}
    for tour in ["ATP", "WTA"]:
        ml, editions = build_match_level(tour, min_history_years)
        editions_by_tour[tour] = editions
        n_coverage = ml["coverage_ok"].sum()
        print(f"{tour}: {len(ml)} real matches (deduplicated), {n_coverage} with peak/history coverage "
              f"on both sides ({n_coverage / len(ml):.1%})")
        frames.append(ml)
    all_match_level = pd.concat(frames, ignore_index=True)

    results = []
    for band in MATCH_BAND_CANDIDATES:
        r = run_for_band(band, peak_gap_threshold, all_match_level, editions_by_tour)
        if r is not None:
            results.append(r)

    print(f"\n{'=' * 90}\nSUMMARY & HONEST COMPARISON TO peak_elo_recovery_test.py\n{'=' * 90}")
    for r in results:
        print(f"  MATCH_BAND={r['match_band']:<3} n={r['n']:<6} residual={r['observed']:+.1%} "
              f"95% CI [{r['lo']:+.1%}, {r['hi']:+.1%}]")
    any_real_positive = any(r["lo"] > 0 for r in results)
    if any_real_positive:
        print("\nAt least one band shows a REAL positive residual: the higher-career-peak side wins "
              "more than Elo's own prediction, even holding current Elo essentially fixed. Since this "
              "is a narrower, cleaner slice than peak_elo_recovery_test.py's pooled regression, this "
              "would need to be reconciled with that script's own (not-validated) result explicitly - "
              "either a real, localized effect that pooling diluted, or (if the pooled regression WAS "
              "validated positive) simple confirmation via a different estimator, not new independent "
              "evidence of a previously-unknown effect.")
    else:
        print("\nNo band shows a real positive residual - this matched-pairs design REPLICATES the null "
              "already found by peak_elo_recovery_test.py's regression, via an independent, non-"
              "parametric estimator. This is not new evidence against the peak-Elo hypothesis so much "
              "as the SAME null showing up a second, more direct way - consistent with, not additional "
              "to, that script's own verdict. REJECTED, same conclusion as before, honestly labeled as "
              "such rather than presented as a fresh independent test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-band", type=int, default=None, help="override: run only this one band")
    parser.add_argument("--peak-gap", type=float, default=MEANINGFUL_PEAK_GAP_DEFAULT)
    parser.add_argument("--min-history-years", type=float, default=MIN_HISTORY_YEARS_DEFAULT)
    args = parser.parse_args()
    if args.match_band is not None:
        MATCH_BAND_CANDIDATES = [args.match_band]
    run(args.min_history_years, args.peak_gap)
