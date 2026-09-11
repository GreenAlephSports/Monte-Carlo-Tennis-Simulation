"""Genuinely different framing from every other test in this family (upset_boost_scaling_test.py,
upset_boost_peak_ceiling_test.py, former_elite_vs_current_top_test.py, former_elite_form_signal_
test.py): none of those ever looked at the SHAPE of a real deep tournament run, only at single
matches or streak length. This one asks: for a former-elite player currently well below their own
career peak, who is ALSO coming off a real comeback window (a recorded layoff before this
tournament, using this project's own already-validated LAYOFF_BUCKET_EDGES taxonomy - see
layoff_test.py), and who goes on to reach a Semifinal or Final in that SAME tournament run - does
the match-by-match trajectory of that specific run look distinctive (margin of victory trending up
round over round, or lower variance / more consistent) compared to how a below-peak player's
matches look otherwise?

Scope, disclosed explicitly rather than assumed: "at some point after one of their comeback
windows" is operationalized here as the SAME tournament run (the comeback-tagged edition IS the
deep-run edition) - not a later, separate tournament following an earlier comeback. A "later
tournament" version would require an additional, non-obvious design choice (which of a player's
possibly-several past comebacks a later deep run should be attributed to, and how far back to look)
and is out of scope here; this is the tightest, least ambiguous reading of the question, not a
weaker substitute for it.

Population construction, each step reusing already-existing, already-validated machinery rather
than reimplemented:
  1. Frozen per-tournament-edition Elo (elite_opponent_residual_test.build_frozen_predictions) +
     career_peak_prior/years_of_history (peak_elo_recovery_test.add_peak_features) - same
     gap_below_peak = clip(career_peak_prior - player_elo, 0) >= FORMER_ELITE_GAP_THRESHOLD (150,
     same value as every other test in this family) and >= MIN_HISTORY_YEARS coverage bar.
  2. Real layoff bucket per match (layoff_test.build_layoff_dataset, days since that player's own
     last recorded match in EITHER tour history) and within-tournament match sequencing
     (layoff_within_tournament_decay_test.build_within_tournament_sequences) - gives match_number
     and first_match_bucket (that run's ENTIRE bucket, fixed at match 1) per real single-elimination
     run, breaking on the player's first loss, same convention as every sequencing function in this
     project.
  3. games_margin (games won minus games lost, signed) and straight_sets per real match
     (layoff_margin_of_victory_test.build_margin_lookup, parsed from the Kaggle Score string,
     score_consistent rows only - same ~0.03% inconsistent-row exclusion already documented there).
  4. COMEBACK_BUCKETS = {"30_60d", "60_90d", "90d_plus"} - a real, recorded gap of >= 30 days
     before this tournament's first match, using this codebase's own layoff-bucket boundaries
     rather than inventing a new threshold. Disclosed choice, not hidden: a stricter >= 60 days
     definition is also reported as a sensitivity check.
  5. DEEP_RUN = a qualifying run's deepest round reached is Semifinals or The Final (round_order
     >= 6, survivorship_upset_test.ROUND_ORDER's own scale).
  6. Population = former-elite-below-peak (edition-level) runs where first_match_bucket is a
     COMEBACK bucket AND the run is a DEEP_RUN. Reported honestly BEFORE any conclusion, both as a
     count and (since this is expected to be a very small number) as an explicit per-instance
     listing - player, tour, edition, deepest round, first_match_bucket, the real match-by-match
     games_margin sequence.

Baseline (the "what a typical below-peak player's matches look like otherwise" comparison): every
OTHER real match by a former-elite-below-peak player (same gap_below_peak/coverage filter) that is
NOT part of a comeback-deep-run instance - i.e. disjoint from the population above, not a superset
of it.

Trajectory statistics (only computed if MIN_RUNS_FOR_STATS is met - see the honesty gate below):
  - Slope of games_margin on match_number (plain OLS, pooled match-level points, NOT one point per
    run - a run with 5 matches contributes 5 points, same "row-weighted" convention as this
    project's own cluster_bootstrap_ci), compared between the deep-run population and the baseline,
    player-clustered bootstrap CI on the DIFFERENCE in slope.
  - Variance of games_margin within each population (descriptive only if sample is too thin for a
    meaningful bootstrap CI on a variance ratio - reported plainly as such, not dressed up).

If the real "comeback deep run" population is too small for ANY of this to mean something (a real,
expected possibility given how narrow the intersection of four independently rare conditions is:
former-elite AND well-below-peak AND coming off a real layoff AND then reaching a Semifinal/Final
in that SAME tournament), this script says so explicitly rather than forcing a held-out verdict off
a handful of rows - MIN_RUNS_FOR_STATS gates every inferential claim below the per-instance listing.

Usage:
    python model/research/former_elite_comeback_deep_run_test.py
    python model/research/former_elite_comeback_deep_run_test.py --gap-threshold 150 --min-history-years 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import build_frozen_predictions  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from former_elite_vs_current_top_test import (  # noqa: E402
    FORMER_ELITE_GAP_THRESHOLD_DEFAULT, MIN_HISTORY_YEARS_DEFAULT,
)
from layoff_margin_of_victory_test import build_margin_lookup  # noqa: E402
from layoff_test import build_layoff_dataset  # noqa: E402
from layoff_within_tournament_decay_test import build_within_tournament_sequences  # noqa: E402
from peak_elo_recovery_test import add_peak_features  # noqa: E402
from survivorship_upset_test import ROUND_ORDER  # noqa: E402

COMEBACK_BUCKETS_PRIMARY = {"30_60d", "60_90d", "90d_plus"}     # >= 30 real days off before this run
COMEBACK_BUCKETS_STRICT = {"60_90d", "90d_plus"}                # sensitivity check, >= 60 days
DEEP_ROUND_ORDER_MIN = ROUND_ORDER["Semifinals"]                 # reached SF or better
MIN_RUNS_FOR_STATS = 8    # real deep-run instances needed before ANY slope/variance claim is made
MIN_MATCHES_FOR_STATS = 25  # pooled match-level rows needed on top of that


def ols_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0 or len(x) < 2:
        return float("nan")
    return float(((x - xm) * (y - ym)).sum() / denom)


def cluster_bootstrap_slope_diff(df_a, df_b, x_col, y_col, player_col="player", n_boot=5000, seed=42):
    """Player-clustered bootstrap CI on (slope_a - slope_b), the two OLS slopes fit independently
    within each population. Replicates where either resampled group has fewer than 2 distinct
    match_number values (slope undefined) are dropped and the valid-replicate count is reported -
    the same 'unstable CI is itself part of the honesty signal' convention former_elite_vs_current_
    top_test.cluster_bootstrap_group_diff already uses for a mean-difference statistic, extended
    here to a slope-difference statistic."""
    players_a = df_a[player_col].unique()
    players_b = df_b[player_col].unique()
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        samp_a = rng.choice(players_a, size=len(players_a), replace=True)
        samp_b = rng.choice(players_b, size=len(players_b), replace=True)
        rows_a = df_a[df_a[player_col].isin(samp_a)]
        rows_b = df_b[df_b[player_col].isin(samp_b)]
        if rows_a[x_col].nunique() < 2 or rows_b[x_col].nunique() < 2:
            continue
        slope_a = ols_slope(rows_a[x_col], rows_a[y_col])
        slope_b = ols_slope(rows_b[x_col], rows_b[y_col])
        if np.isnan(slope_a) or np.isnan(slope_b):
            continue
        boot.append(slope_a - slope_b)
    n_valid = len(boot)
    observed = ols_slope(df_a[x_col], df_a[y_col]) - ols_slope(df_b[x_col], df_b[y_col])
    if n_valid < 20:
        return observed, float("nan"), float("nan"), n_valid
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return observed, lo, hi, n_valid


def build_population(tour, gap_threshold, min_history_years):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)
    preds = add_peak_features(preds)
    preds["gap_below_peak"] = (preds["career_peak_prior"] - preds["player_elo"]).clip(lower=0)

    layoff_df = build_layoff_dataset(matches, preds)
    seq = build_within_tournament_sequences(layoff_df)
    seq["round_order"] = seq["round"].map(ROUND_ORDER)

    margin_lookup = build_margin_lookup(matches)
    seq = seq.merge(margin_lookup, on=["edition_id", "date", "round", "player", "opponent"], how="left")

    peak_lvl = preds[["player", "edition_id", "gap_below_peak", "years_of_history"]].drop_duplicates(
        subset=["player", "edition_id"])
    seq = seq.merge(peak_lvl, on=["player", "edition_id"], how="left", validate="many_to_one")

    former_elite = seq[
        (seq["years_of_history"] >= min_history_years) & (seq["gap_below_peak"] >= gap_threshold)
    ].copy()
    former_elite["tour"] = tour
    return former_elite


def summarize_runs(former_elite, comeback_buckets, label):
    run_level = former_elite.groupby(["tour", "edition_id", "player"], sort=False).agg(
        n_matches=("match_number", "max"),
        deepest_round_order=("round_order", "max"),
        first_match_bucket=("first_match_bucket", "first"),
    ).reset_index()
    deep_runs = run_level[
        run_level["first_match_bucket"].isin(comeback_buckets)
        & (run_level["deepest_round_order"] >= DEEP_ROUND_ORDER_MIN)
    ]
    print(f"\n{label}: {len(deep_runs)} real comeback-deep-run instances "
          f"({deep_runs['player'].nunique()} distinct players)")
    return deep_runs, run_level


def run(gap_threshold, min_history_years):
    all_former_elite = []
    for tour in ["ATP", "WTA"]:
        fe = build_population(tour, gap_threshold, min_history_years)
        n_runs = fe.groupby(["edition_id", "player"]).ngroups
        print(f"{tour}: {len(fe)} former-elite-below-peak sequenced rows "
              f"(gap_below_peak >= {gap_threshold:.0f}, years_of_history >= {min_history_years:.1f}), "
              f"{n_runs} distinct tournament runs, {fe['player'].nunique()} distinct players")
        all_former_elite.append(fe)
    former_elite = pd.concat(all_former_elite, ignore_index=True)

    print(f"\n{'=' * 90}\nPOPULATION - comeback-deep-run instances (>= 30 real days off before this "
          f"SAME tournament, then reached Semifinal or Final in it)\n{'=' * 90}")
    deep_runs, run_level = summarize_runs(former_elite, COMEBACK_BUCKETS_PRIMARY, "PRIMARY (>= 30 days off)")
    deep_runs_strict, _ = summarize_runs(former_elite, COMEBACK_BUCKETS_STRICT, "SENSITIVITY (>= 60 days off)")

    if len(deep_runs) == 0:
        print("\nZERO real instances of this exact scenario in the full historical record, either tour. "
              "Nothing to analyze - the population this hypothesis is about does not exist in the data "
              "at the >= 30-day comeback bar. REJECTED by absence of any real cases, not by a failed fit.")
        return

    print("\n--- Every real instance found (PRIMARY definition) ---")
    listing = deep_runs.merge(
        former_elite[["tour", "edition_id", "player", "match_number", "round", "games_margin",
                       "straight_sets", "score_consistent"]],
        on=["tour", "edition_id", "player"], how="left",
    ).sort_values(["tour", "edition_id", "player", "match_number"])
    for (tour, edition_id, player), g in listing.groupby(["tour", "edition_id", "player"], sort=False):
        bucket = g["first_match_bucket"].iloc[0]
        seq_str = ", ".join(
            f"m{int(r.match_number)}:{r.round}"
            + (f" margin={r.games_margin:+.0f}" if pd.notna(r.games_margin) else " margin=n/a(score unparseable)")
            for r in g.itertuples(index=False)
        )
        print(f"  {tour} {edition_id} - {player} (comeback bucket: {bucket}): {seq_str}")

    n_runs = len(deep_runs)
    if n_runs < MIN_RUNS_FOR_STATS:
        print(f"\n{'=' * 90}\nVERDICT: TOO FEW REAL INSTANCES TO VALIDATE ANYTHING "
              f"({n_runs} run(s), need >= {MIN_RUNS_FOR_STATS})\n{'=' * 90}")
        print("The instances above are the complete, honest record of this exact scenario in the "
              "historical data. Any trajectory/variance claim computed from this few real cases would "
              "not be distinguishable from noise, and this script will not manufacture a confidence "
              "interval that implies otherwise. This is not a rejection of the underlying idea - it is "
              "an honest 'not enough real cases exist to test it,' full stop.")
        return

    # ============================================================================
    # Only reached if n_runs >= MIN_RUNS_FOR_STATS - real trajectory/variance statistics
    # ============================================================================
    deep_match_rows = former_elite.merge(
        deep_runs[["tour", "edition_id", "player"]], on=["tour", "edition_id", "player"], how="inner")
    deep_match_rows = deep_match_rows[deep_match_rows["score_consistent"] == True]  # noqa: E712

    baseline_rows = former_elite.merge(
        deep_runs[["tour", "edition_id", "player"]], on=["tour", "edition_id", "player"],
        how="left", indicator=True)
    baseline_rows = baseline_rows[(baseline_rows["_merge"] == "left_only") & (baseline_rows["score_consistent"] == True)]  # noqa: E712

    print(f"\nPooled match-level rows with a usable (score-consistent) games_margin: "
          f"deep-run population n={len(deep_match_rows)}, baseline (all other former-elite-below-peak "
          f"matches) n={len(baseline_rows)}")

    if len(deep_match_rows) < MIN_MATCHES_FOR_STATS:
        print(f"\nVERDICT: {n_runs} runs meets the run-count floor, but only {len(deep_match_rows)} "
              f"pooled match-level rows have a usable margin - below the {MIN_MATCHES_FOR_STATS}-row "
              f"floor for a trajectory/variance statistic to mean anything. Reporting the per-instance "
              f"listing above as the honest result; no slope or variance claim follows from this few rows.")
        return

    print(f"\n{'=' * 90}\nTRAJECTORY: slope of games_margin on match_number, deep-run population vs. "
          f"baseline\n{'=' * 90}")
    slope_deep = ols_slope(deep_match_rows["match_number"], deep_match_rows["games_margin"])
    slope_base = ols_slope(baseline_rows["match_number"], baseline_rows["games_margin"])
    print(f"  Deep-run slope (margin per additional round): {slope_deep:+.2f} games/round (n={len(deep_match_rows)})")
    print(f"  Baseline slope:                                {slope_base:+.2f} games/round (n={len(baseline_rows)})")

    obs, lo, hi, n_valid = cluster_bootstrap_slope_diff(
        deep_match_rows, baseline_rows, "match_number", "games_margin")
    print(f"\n  Difference in slope (deep-run - baseline), player-clustered bootstrap "
          f"({n_valid}/5000 valid replicates):")
    if n_valid < 20:
        print("    Too few valid bootstrap replicates (too few distinct players in the deep-run "
              "population) - INCONCLUSIVE, cannot report a CI here even though the run-count floor "
              "was met.")
    else:
        verdict = ("deep runs trend MORE upward (CI excludes zero, >0)" if lo > 0 else
                   ("deep runs trend LESS upward (CI excludes zero, <0)" if hi < 0 else
                    "NOT distinguishable (CI straddles zero)"))
        print(f"    observed {obs:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}] -> {verdict}")

    print(f"\n{'=' * 90}\nVARIANCE: games_margin spread, deep-run population vs. baseline (descriptive - "
          f"a formal variance-ratio CI is not attempted at this sample size)\n{'=' * 90}")
    var_deep, var_base = deep_match_rows["games_margin"].var(), baseline_rows["games_margin"].var()
    std_deep, std_base = deep_match_rows["games_margin"].std(), baseline_rows["games_margin"].std()
    print(f"  Deep-run:  var={var_deep:.1f}, std={std_deep:.1f} (n={len(deep_match_rows)})")
    print(f"  Baseline:  var={var_base:.1f}, std={std_base:.1f} (n={len(baseline_rows)})")
    print(f"  Variance ratio (deep-run / baseline): {var_deep / var_base:.2f}" if var_base else "  n/a")
    print("  Read descriptively only - with this few deep-run players, a variance-ratio CI would be "
          "dominated by which specific players happened to be resampled, not a real population "
          "estimate; not computed for that reason, consistent with this script's honesty discipline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gap-threshold", type=float, default=FORMER_ELITE_GAP_THRESHOLD_DEFAULT)
    parser.add_argument("--min-history-years", type=float, default=MIN_HISTORY_YEARS_DEFAULT)
    args = parser.parse_args()
    run(args.gap_threshold, args.min_history_years)
