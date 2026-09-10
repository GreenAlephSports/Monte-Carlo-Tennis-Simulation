"""Tests whether UPSET_BOOST_LOGIT_SHIFT (win_probability.py) - the "just beat a >100-Elo-point
favorite this tournament, get a carryover boost in your NEXT match" signal - should scale with the
BENEFICIARY's own proven historical ceiling (career-peak Elo), not just the size of the gap
overcome (already tested and rejected as its own axis in upset_boost_scaling_test.py: neither a
graduated-bucket nor continuous gap-size scaling cleared held-out validation).

This is a DIFFERENT covariate, not a re-run of that rejected one: holding the gap overcome AND the
player's CURRENT Elo fixed, does the size of the carryover boost also depend on how high that
player's own career-peak Elo has been? The mechanistic story: a proven top-level player (Osaka,
returning from injury/absence, currently at a middling Elo but with a real Slam-winning peak in her
history) beating a big favorite is a "form returning to a real, previously-demonstrated ceiling"
signal - different information than the identical upset scored by a player with no comparable peak
(a first career signature win, genuinely uncertain whether it repeats). Directly relevant to the
injury-recovery question this project has separately investigated (peak_elo_recovery_test.py): "a
proven player beating someone while returning from injury" is exactly this mechanism's target case.

Explicitly distinct from signature_win_boost_test.py (REJECTED 2026-08-26), which is a DIFFERENT,
already-failed idea despite superficial similarity - both involve "beating a strong opponent", but:
  - signature-win-boost: a K-FACTOR multiplier on the ELO UPDATE for a match meeting a fixed bar
    (top-5 opponent or Slam final), applied identically regardless of who the beneficiary is. It is
    SYMMETRIC BY CONSTRUCTION - K_FACTOR*mult updates BOTH players in that one match, so a player's
    boosted losses to elite opponents (frequent, if they play elite opponents often) mechanically
    outweigh their boosted wins. That symmetric penalty is WHY it was rejected: Osaka -0.7pts net,
    Andreeva -15.2pts net (worse than flat K), because both are established players who lose to
    elite competition more often than they beat it, and the boost punishes those losses as hard as
    it rewards the wins.
  - upset-boost (this test's base mechanism, production since before tonight): a LOGIT-SPACE SHIFT
    to the WINNER's NEXT-MATCH win probability, gated on having just WON against a >100-Elo
    favorite. It only ever fires on a row that exists because the player WON their previous match
    this tournament (survivorship_upset_test.build_upset_dataset's sequencing: the loop BREAKS on a
    loss, so a losing player generates no further within-tournament row for any shift to touch).
    There is no "next match" to boost after an elimination, so this mechanism cannot construct a
    symmetric across-players loss penalty the way a K-factor-on-the-match design can - checked
    empirically below (POPULATION-CONSTRUCTION CHECK), not just asserted from reading the code.
    The one thing it does NOT dodge: the boosted player's own next match can still be a LOSS (they
    beat one favorite and then lose to the next one) - but that's not a symmetric "punish the
    opponent" penalty, it's the same row type any predictive signal is graded on, and it is already
    fully priced into this test's held-out log-loss (a boosted-then-lost row contributes its own
    log-loss cost to the SAME population's held-out number, exactly like every other correction in
    this pipeline). So this test's gating (scaling the shift by the beneficiary's career_peak_prior)
    inherits upset-boost's asymmetric-by-construction structure, NOT signature-win-boost's symmetric
    one - explicitly checked, not assumed, in the POPULATION-CONSTRUCTION CHECK section below.

Peak definitions (both tested independently, same as peak_elo_recovery_test.py, reusing its exact
no-lookahead peak machinery): career_peak_prior (all-time high-to-date overall_elo, no lookahead)
and peak_2y_prior (trailing-730-day version). Both are ABSOLUTE peak levels (not gap-below-peak) -
the hypothesis here is specifically "how high has this player's ceiling ever been", independent of
where their current Elo sits, so two players with identical current Elo and identical gap overcome
can still differ on this covariate. Same coverage requirement as peak_elo_recovery_test.py (>=
MIN_HISTORY_YEARS of real tracked history for a meaningful peak estimate), reported before any
conclusion.

Design (within the upset-boost-eligible population - prev_gap > UPSET_BOOST_ELO_GAP_THRESHOLD=100,
same population UPSET_BOOST_LOGIT_SHIFT fires on in production):
  A. Flat shift (current production design), refit fresh on this population/split for an
     apples-to-apples baseline - same precedent as upset_boost_scaling_test.py's own Candidate 1.
  B. Peak-interaction: adjusted_logit = logit(pred_win) + BASE_SHIFT + PEAK_COEF *
     (peak_prior - STARTING_ELO)/100, both parameters jointly fit by 2D MLE (scipy) on train-era
     rows of this population. Held out against raw Elo AND head-to-head against Candidate A on the
     identical test rows (same discipline as upset_boost_scaling_test.py's graduated/continuous
     candidates - both comparisons must clear for a verdict of "real").

Same full rigor as the rest of this project's Elo research: frozen per-tournament-edition Elo
(elite_opponent_residual_test.build_frozen_predictions), chronological tournament-edition 80/20
train/test split, held-out validation, player-clustered bootstrap CIs
(survivorship_upset_test.cluster_bootstrap_ci), both tours, real historical data throughout.

Usage:
    python model/research/upset_boost_peak_ceiling_test.py
    python model/research/upset_boost_peak_ceiling_test.py --min-history-years 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import (  # noqa: E402
    EPS, TRAIN_FRACTION, build_frozen_predictions, log_loss, logit, sigmoid,
)
from elo_ratings import STARTING_ELO, load_matches_for_tour  # noqa: E402
from peak_elo_recovery_test import add_peak_features  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from upset_boost_scaling_test import build_gap_dataset  # noqa: E402
from win_probability import UPSET_BOOST_ELO_GAP_THRESHOLD, UPSET_BOOST_LOGIT_SHIFT  # noqa: E402

MIN_HISTORY_YEARS_DEFAULT = 2.0
PROVEN_PEAK_MARGIN = 300.0  # Elo pts above STARTING_ELO - descriptive "real, high peak" cut only


def fit_flat_and_interaction(train, peak_col):
    """Candidate A (flat) is the beta=0 special case fit by closed form (same as
    upset_boost_scaling_test.py's Candidate 1). Candidate B (interaction) jointly fits BASE_SHIFT
    and PEAK_COEF by MLE with a fixed per-row offset = logit(pred_win), same offset-GLM pattern as
    upset_boost_scaling_test.fit_continuous_beta, extended to 2 free parameters via scipy since
    fit_beta_newton (recent_form_test.py) is 1D-only."""
    flat_actual, flat_pred = train["actual_win"].mean(), train["pred_win"].mean()
    flat_shift = logit(flat_actual) - logit(flat_pred)

    offset = train["pred_win"].apply(logit).values
    x = ((train[peak_col] - STARTING_ELO) / 100.0).values
    y = train["actual_win"].values

    def neg_log_lik(params):
        base, coef = params
        z = np.clip(offset + base + coef * x, -35, 35)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, EPS, 1 - EPS)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

    result = minimize(neg_log_lik, x0=[flat_shift, 0.0], method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    base_fit, coef_fit = result.x
    return flat_shift, base_fit, coef_fit


def held_out_report(label, test_rows, adjusted_col_fn):
    t = test_rows.copy()
    t["adjusted_pred"] = t.apply(adjusted_col_fn, axis=1)
    t["raw_loss"] = log_loss(t["actual_win"].values, t["pred_win"].values)
    t["adj_loss"] = log_loss(t["actual_win"].values, t["adjusted_pred"].values)
    observed, lo, hi = cluster_bootstrap_ci(t, "raw_loss", "adj_loss")
    print(f"\n{label}: {len(t)} test-era rows, {t['player'].nunique()} players")
    print(f"  Raw Elo   : log-loss = {t['raw_loss'].mean():.4f}")
    print(f"  Adjusted  : log-loss = {t['adj_loss'].mean():.4f}")
    print(f"  Mean per-match log-loss improvement (raw - adjusted, >0 = better), player-clustered: "
          f"{observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    return observed, lo, hi, t


def run_for_peak_col(peak_col, label, train_pop, test_pop):
    print(f"\n{'=' * 90}\n{label}\n{'=' * 90}")
    train = train_pop[train_pop[peak_col].notna()].copy()
    test = test_pop[test_pop[peak_col].notna()].copy()
    print(f"{len(train)} train-era / {len(test)} test-era upset-boost-eligible rows "
          f"(gap>{UPSET_BOOST_ELO_GAP_THRESHOLD}, coverage-filtered), "
          f"{train['player'].nunique()} / {test['player'].nunique()} distinct players")
    print(f"  mean {peak_col}: train={train[peak_col].mean():.1f}, test={test[peak_col].mean():.1f} "
          f"(current production UPSET_BOOST_LOGIT_SHIFT={UPSET_BOOST_LOGIT_SHIFT:+.4f} for reference; "
          f"refit fresh below for a fair comparison)")

    flat_shift, base_fit, coef_fit = fit_flat_and_interaction(train, peak_col)
    print(f"\n  Candidate A (flat, refit on this population/split): {flat_shift:+.4f} logits")
    print(f"  Candidate B (peak-interaction, joint MLE): BASE_SHIFT={base_fit:+.4f}, "
          f"PEAK_COEF={coef_fit:+.4f} per 100 Elo pts of ({peak_col} - {STARTING_ELO:.0f})")
    for peak_example in (STARTING_ELO, STARTING_ELO + 150, STARTING_ELO + 300, STARTING_ELO + 500):
        shift = base_fit + coef_fit * (peak_example - STARTING_ELO) / 100.0
        print(f"    at {peak_col}={peak_example:.0f}: implied shift = {shift:+.4f} logits")

    obs_a, lo_a, hi_a, _ = held_out_report(
        "Candidate A vs. raw Elo", test, lambda r: sigmoid(logit(r["pred_win"]) + flat_shift))
    obs_b, lo_b, hi_b, _ = held_out_report(
        "Candidate B (peak-interaction) vs. raw Elo", test,
        lambda r: sigmoid(logit(r["pred_win"]) + base_fit + coef_fit * (r[peak_col] - STARTING_ELO) / 100.0))

    # head-to-head: does B actually beat A on the IDENTICAL test rows?
    common = test.copy()
    common["flat_pred"] = common["pred_win"].apply(lambda p: sigmoid(logit(p) + flat_shift))
    common["interact_pred"] = common.apply(
        lambda r: sigmoid(logit(r["pred_win"]) + base_fit + coef_fit * (r[peak_col] - STARTING_ELO) / 100.0), axis=1)
    common["flat_loss"] = log_loss(common["actual_win"].values, common["flat_pred"].values)
    common["interact_loss"] = log_loss(common["actual_win"].values, common["interact_pred"].values)
    obs_ba, lo_ba, hi_ba = cluster_bootstrap_ci(common, "flat_loss", "interact_loss")
    print(f"\n  Head-to-head, peak-interaction vs. flat (identical {len(common)} test rows): "
          f"improvement {obs_ba:+.4f}, 95% CI [{lo_ba:+.4f}, {hi_ba:+.4f}]")

    # descriptive: proven-high-peak vs. no-comparable-peak, holding this population (gap>100) fixed
    high_peak = test[peak_col] >= STARTING_ELO + PROVEN_PEAK_MARGIN
    desc = pd.DataFrame({
        "population": [f"no comparable peak (< {PROVEN_PEAK_MARGIN:.0f} above start)",
                       f"proven high peak (>= {PROVEN_PEAK_MARGIN:.0f} above start)"],
        "n": [len(test[~high_peak]), len(test[high_peak])],
        "assigned": [test.loc[~high_peak, "pred_win"].mean(), test.loc[high_peak, "pred_win"].mean()],
        "actual": [test.loc[~high_peak, "actual_win"].mean(), test.loc[high_peak, "actual_win"].mean()],
    })
    desc["residual_pp"] = desc["actual"] - desc["assigned"]
    print(f"\n  Descriptive: actual vs. Elo-assigned win rate, held-out test rows only, split at "
          f"{peak_col} >= {STARTING_ELO + PROVEN_PEAK_MARGIN:.0f}:")
    print(desc.to_string(index=False, formatters={
        "assigned": "{:.1%}".format, "actual": "{:.1%}".format, "residual_pp": "{:+.1%}".format}))

    clears = lo_b > 0 and lo_ba > 0
    print(f"\n  VERDICT ({label}): {'VALIDATED' if clears else 'NOT validated'} "
          f"(needs CI>0 vs. raw Elo AND vs. flat; got vs.raw=[{lo_b:+.4f},{hi_b:+.4f}], "
          f"vs.flat=[{lo_ba:+.4f},{hi_ba:+.4f}])")
    return {"peak_col": peak_col, "coef_fit": coef_fit, "obs_b": obs_b, "lo_b": lo_b, "hi_b": hi_b,
            "obs_ba": obs_ba, "lo_ba": lo_ba, "hi_ba": hi_ba, "clears": clears}


def check_population_construction(gap_df):
    """Empirical check of the claim made in the module docstring: does this population contain ANY
    row representing a 'next match after a LOSS to a big favorite' - i.e. could the peak-interaction
    shift ever apply a symmetric penalty the way signature-win-boost's K-factor did? By
    build_gap_dataset's own sequencing (copied unchanged from survivorship_upset_test.build_upset_
    dataset), the per-player loop BREAKS the first time actual_win==0, so no further row is ever
    emitted for that player in that edition - there should be exactly zero such rows. Verified here,
    not just asserted from reading the code. Vectorized (groupby + shift), not a per-row Python
    loop - the manual-loop version of this check was too slow to finish on the full dataset."""
    eligible = gap_df[gap_df["prev_gap"] > UPSET_BOOST_ELO_GAP_THRESHOLD]
    # a row in this population exists only if the PRECEDING row (same player/edition) was a WIN -
    # cross-check via groupby-shift: for every row, look at that same group's previous row's
    # actual_win. If build_gap_dataset's break-on-loss sequencing is doing its job, no eligible row
    # should ever have a previous row with actual_win==0.
    ordered = gap_df.sort_values(["edition_id", "player", "date"], kind="stable").copy()
    ordered["prev_actual_win"] = ordered.groupby(["edition_id", "player"])["actual_win"].shift(1)
    n_after_loss = int(((ordered["prev_gap"] > UPSET_BOOST_ELO_GAP_THRESHOLD) & (ordered["prev_actual_win"] == 0)).sum())
    print(f"  Eligible (gap>{UPSET_BOOST_ELO_GAP_THRESHOLD}) rows in this population: {len(eligible)}")
    print(f"  Of those, rows immediately following a LOSS by the same player (would be the symmetric-"
          f"penalty failure mode): {n_after_loss}")
    print(f"  {'CONFIRMED' if n_after_loss == 0 else 'NOT CONFIRMED'}: this mechanism cannot construct "
          f"a symmetric loss-penalty row the way signature-win-boost's K-factor could "
          f"(signature-win-boost applied to BOTH players of a match regardless of outcome; upset-boost "
          f"only ever fires on a next-match row that exists because the prior match was WON).")


def run(min_history_years):
    print(f"{'=' * 90}\nPOPULATION-CONSTRUCTION CHECK (both tours) - does this mechanism inherit "
          f"upset-boost's asymmetric structure, or signature-win-boost's symmetric one?\n{'=' * 90}")

    all_train, all_test = {"career_peak_prior": [], "peak_2y_prior": []}, {"career_peak_prior": [], "peak_2y_prior": []}
    coverage_rows = []

    for tour in ["ATP", "WTA"]:
        matches = load_matches_for_tour(tour)
        preds, editions = build_frozen_predictions(matches)
        gap_df = build_gap_dataset(preds)

        print(f"\n--- {tour} ---")
        check_population_construction(gap_df)

        peaked = add_peak_features(preds)
        peak_cols = peaked[["player", "edition_id", "career_peak_prior", "peak_2y_prior", "years_of_history"]] \
            .drop_duplicates(subset=["player", "edition_id"])
        gap_df = gap_df.merge(peak_cols, on=["player", "edition_id"], how="left", validate="many_to_one")

        eligible = gap_df[gap_df["prev_gap"] > UPSET_BOOST_ELO_GAP_THRESHOLD].copy()
        eligible = eligible[eligible["years_of_history"] >= min_history_years]
        n_total = len(gap_df[gap_df["prev_gap"] > UPSET_BOOST_ELO_GAP_THRESHOLD])
        coverage_rows.append((tour, n_total, len(eligible), len(eligible) / n_total if n_total else float("nan"),
                               eligible["player"].nunique()))

        split_idx = int(len(editions) * TRAIN_FRACTION)
        train_editions = set(editions["edition_id"].iloc[:split_idx])
        test_editions = set(editions["edition_id"].iloc[split_idx:])
        train = eligible[eligible["edition_id"].isin(train_editions)]
        test = eligible[eligible["edition_id"].isin(test_editions)]
        for col in ("career_peak_prior", "peak_2y_prior"):
            all_train[col].append(train)
            all_test[col].append(test)

    print(f"\n{'=' * 90}\nCOVERAGE - upset-boost-eligible (gap>{UPSET_BOOST_ELO_GAP_THRESHOLD}) rows with "
          f">= {min_history_years:.1f} years of real tracked history (required for a meaningful peak "
          f"estimate)\n{'=' * 90}")
    cov = pd.DataFrame(coverage_rows, columns=["tour", "n_eligible_gap_rows", "n_coverage_ok", "coverage_share", "n_players"])
    print(cov.to_string(index=False, formatters={"coverage_share": "{:.1%}".format}))

    results = {}
    for col, label in [("career_peak_prior", "CAREER-HIGH PEAK (all-time, to-date)"),
                        ("peak_2y_prior", "TRAILING 2-YEAR PEAK")]:
        train = pd.concat(all_train[col], ignore_index=True)
        test = pd.concat(all_test[col], ignore_index=True)
        results[label] = run_for_peak_col(col, label, train, test)

    print(f"\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
    for label, r in results.items():
        print(f"  {label:<40} PEAK_COEF={r['coef_fit']:+.4f}, vs.raw {r['obs_b']:+.4f} "
              f"[{r['lo_b']:+.4f},{r['hi_b']:+.4f}], vs.flat {r['obs_ba']:+.4f} "
              f"[{r['lo_ba']:+.4f},{r['hi_ba']:+.4f}] -> {'VALIDATED' if r['clears'] else 'NOT validated'}")
    any_validated = any(r["clears"] for r in results.values())
    print(f"\n{'=' * 90}\nOVERALL VERDICT\n{'=' * 90}")
    if any_validated:
        print("At least one peak definition beats BOTH raw Elo and the flat production design, "
              "held out, on the upset-boost-eligible population. See per-definition breakdown above "
              "for the fitted PEAK_COEF and which definition cleared.")
    else:
        print("Neither peak definition clears held-out validation against both raw Elo and the flat "
              "production design. History-gating the upset-boost shift by career-peak Elo does not "
              "improve on the existing flat design - REJECTED. The population-construction check above "
              "still stands regardless: whatever the outcome, this mechanism does not carry "
              "signature-win-boost's symmetric-loss-penalty failure mode, because it structurally "
              "cannot generate a 'next match after a loss' row for the shift to apply to.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-history-years", type=float, default=MIN_HISTORY_YEARS_DEFAULT)
    args = parser.parse_args()
    run(args.min_history_years)
