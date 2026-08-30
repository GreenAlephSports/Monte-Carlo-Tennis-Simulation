"""Tests the same rank/Elo-blend idea thin_history_rank_blend_test.py prototyped, but for a
genuinely different population and a different hypothesis: NOT players with too little match
history (that population was tested there and REJECTED - see that file's FINAL VERDICT, actively
harmful in the 0-2-match bucket). This tests established players (plenty of career matches, a
real, trustworthy continuously-updated Elo) whose current ranking has moved sharply relative to
what their Elo still says - i.e. Elo is LAGGING a real recent trajectory shift, not missing data.
The concrete motivating cases are Rybakina E. and Andreeva M. (WTA): players whose real-world rank
trajectory (a big recent climb or slide) plausibly outpaces how fast a continuously-updated Elo can
move, since Elo only updates K_FACTOR points per result while rank can jump many places off a
single strong tournament.

Population definition (data-driven, not guessed): among "solid" players (>= SOLID_MATCHES career
matches - the SAME threshold thin_history_rank_blend_test.py used to define a trustworthy Elo, here
used to EXCLUDE the thin-data population instead of study it), compute each row's rank/Elo
disagreement gap = rank_elo_fn(current_rank) - player_elo (positive = real rank says this player is
BETTER than their Elo credits them for; negative = the reverse). The "trajectory-lag" population is
the top quartile of |gap| in the TRAIN era only (train-era threshold applied unchanged to test era,
never re-derived from test data) - real players whose rank and Elo meaningfully disagree despite an
established, trustworthy Elo.

Correction tested: blend player_elo toward rank_elo_fn(rank) at a single fixed weight, the same
functional form thin_history_rank_blend_test.py used (weighted average, not a free-form fit) - the
weight itself is chosen by a small grid search (0.05/0.10/0.15/0.20/0.25) restricted to TRAIN-era
trajectory-lag rows only (minimizing train log-loss), then validated held-out on test-era
trajectory-lag rows the weight was never chosen to fit. This is a genuinely new free parameter
(unlike thin_history's reuse of the pipeline's own SURFACE_BLEND_K), so it's disclosed plainly here
rather than pretending it's parameter-free - grid-searched on train only, never touching test.

Same rigor as every other test in this series: frozen per-tournament-edition Elo (reuses
thin_history_rank_blend_test.build_dataset directly), chronological 80/20 train/test split
(elite_opponent_residual_test.TRAIN_FRACTION), held-out validation, player-clustered bootstrap CI
(survivorship_upset_test.cluster_bootstrap_ci).

Usage:
    python model/research/rank_trajectory_lag_test.py [--tour WTA|ATP]

FINAL VERDICT (2026-08-26, WTA): NOT statistically significant, held out - not added to production,
but closer to significant than thin_history_rank_blend_test.py's rejection and worth revisiting with
more data. On the WTA trajectory-lag population (solid players, >=30 matches, in the train-era top
quartile of |rank-implied Elo - actual Elo|, n=4813 held-out test-era rows), a train-era-grid-
searched blend weight of 0.20 toward rank-implied Elo improves held-out log-loss by +0.0020 (0.6118
raw -> 0.6098 blended), 95% player-clustered bootstrap CI [-0.0004, +0.0049] - technically straddles
zero, but only just (the lower bound is a hair below 0), unlike thin_history's clearly negative/
harmful result. Directionally this DOES look like a real, distinct effect from the thin-history
case - it's positive rather than harmful, and lands close to the significance boundary on a
population of thousands rather than hundreds of rows. Rybakina E. and Andreeva M. (the motivating
cases) are both confirmed to fall inside the trajectory-lag population at points in the dataset,
though at their most recent test-era rows their gap magnitude (-32 and -36 pts respectively) is
actually below the 82-pt population threshold - i.e. the specific "right now" mismatch for these
two players is smaller than the population this test defined, so this test validates the GENERAL
hypothesis more than it confirms these two players are currently large outliers. Conclusion: a
genuinely different result from thin-history's rejection (different population, different sign),
plausible but not yet held-out-significant - a reasonable candidate to re-test as more 2026 data
accumulates, or with a wider gap-threshold/more granular weight grid, rather than a settled reject.
No production change made.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import match_name_to_pool  # noqa: E402
from elite_opponent_residual_test import TRAIN_FRACTION, log_loss  # noqa: E402
from elo_ratings import expected_score, load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from thin_history_rank_blend_test import SOLID_MATCHES, build_dataset, fit_rank_to_elo  # noqa: E402

CANDIDATE_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25]  # fraction of the blend toward rank-implied Elo
MOTIVATING_PLAYERS = ["Rybakina E.", "Andreeva M."]


def compute_gap(df, rank_elo_fn):
    df = df.copy()
    df["player_rank_elo"] = df["player_rank"].apply(lambda r: rank_elo_fn(r) if pd.notna(r) else None)
    df["gap"] = df["player_rank_elo"] - df["player_elo"]
    return df


def trajectory_lag_mask(df, gap_threshold):
    return (
        (df["player_matches_before"] >= SOLID_MATCHES)
        & df["player_rank"].notna()
        & (df["gap"].abs() >= gap_threshold)
    )


def apply_blend(df, weight):
    df = df.copy()
    df["player_elo_blend"] = (1 - weight) * df["player_elo"] + weight * df["player_rank_elo"]
    # opponent side left unblended unless the opponent is ALSO in the trajectory-lag population for
    # their own row (handled naturally since every player appears once per match as "player" - the
    # opponent's own row, elsewhere in the dataframe, gets its own blend independently).
    df["opponent_elo_blend"] = df["opponent_elo"]
    df["blended_pred"] = df.apply(
        lambda r: expected_score(r["player_elo_blend"], r["opponent_elo_blend"]), axis=1,
    )
    return df


def run(tour, max_editions=None):
    if max_editions is not None:
        print(f"*** QUICK CHECK ON RECENT DATA ONLY (--max-editions {max_editions}) - NOT the "
              f"full-historical verdict. A different (smaller, more recent) editions pool than the "
              f"full-scale run, with its own fresh train/test split and its own fresh weight grid "
              f"search - a rough out-of-sample consistency check, not a replacement. ***\n")
    matches = load_matches_for_tour(tour)
    preds, editions = build_dataset(matches, max_editions=max_editions)

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    train = preds[preds["edition_id"].isin(train_editions)]
    test = preds[preds["edition_id"].isin(test_editions)]
    print(f"{tour}: {len(editions)} tournament editions, train = first {len(train_editions)} "
          f"(through {editions['edition_start'].iloc[split_idx - 1].date()}), "
          f"test = remaining {len(test_editions)} (from {editions['edition_start'].iloc[split_idx].date()})")
    print(f"{len(train)} train-era rows, {len(test)} test-era rows\n")

    rank_elo_fn = fit_rank_to_elo(train)
    train = compute_gap(train, rank_elo_fn)
    test = compute_gap(test, rank_elo_fn)

    # train-era-derived threshold (top quartile of |gap| among solid, ranked train-era rows) -
    # applied UNCHANGED to test era, never re-derived from it.
    solid_train = train[(train["player_matches_before"] >= SOLID_MATCHES) & train["player_rank"].notna()]
    gap_threshold = solid_train["gap"].abs().quantile(0.75)
    print(f"Solid-player (>= {SOLID_MATCHES} matches) train-era |rank-implied Elo - actual Elo| "
          f"75th percentile: {gap_threshold:.1f} pts -> trajectory-lag population = solid rows at "
          f"or above this gap\n")

    train_lag = train[trajectory_lag_mask(train, gap_threshold)].copy()
    test_lag = test[trajectory_lag_mask(test, gap_threshold)].copy()
    print(f"Train-era trajectory-lag rows: {len(train_lag)} of {len(train)} solid+ranked rows")
    print(f"Test-era (held-out) trajectory-lag rows: {len(test_lag)} of {len(test)}\n")

    if len(train_lag) < 50 or len(test_lag) < 50:
        print("Too few trajectory-lag rows in train or test era to fit/validate a blend weight - "
              "stopping (population as defined is too small for this dataset/tour).")
        return

    # grid search the blend weight on TRAIN-era trajectory-lag rows only, minimizing train log-loss
    best_weight, best_train_loss = None, float("inf")
    print("--- Grid search (train-era trajectory-lag rows only) ---")
    for w in CANDIDATE_WEIGHTS:
        blended = apply_blend(train_lag, w)
        loss = log_loss(blended["actual_win"].values, blended["blended_pred"].values).mean()
        print(f"  weight={w:.2f}: train log-loss = {loss:.4f}")
        if loss < best_train_loss:
            best_train_loss, best_weight = loss, w
    print(f"  -> selected weight={best_weight:.2f} (lowest train-era log-loss)\n")

    # held-out validation at the selected weight, on test-era trajectory-lag rows the weight was
    # never chosen to fit
    test_blended = apply_blend(test_lag, best_weight)
    test_blended["raw_loss"] = log_loss(test_blended["actual_win"].values, test_blended["pred_win"].values)
    test_blended["blend_loss"] = log_loss(test_blended["actual_win"].values, test_blended["blended_pred"].values)
    test_blended["raw_brier"] = (test_blended["actual_win"] - test_blended["pred_win"]) ** 2
    test_blended["blend_brier"] = (test_blended["actual_win"] - test_blended["blended_pred"]) ** 2

    print(f"--- Held-out validation on the trajectory-lag population (n={len(test_blended)}) ---")
    print(f"  Raw Elo (current pipeline)  : log-loss = {test_blended['raw_loss'].mean():.4f}, "
          f"Brier = {test_blended['raw_brier'].mean():.4f}")
    print(f"  Rank-blended (weight={best_weight:.2f}) : log-loss = {test_blended['blend_loss'].mean():.4f}, "
          f"Brier = {test_blended['blend_brier'].mean():.4f}")

    observed, lo, hi = cluster_bootstrap_ci(test_blended, "raw_loss", "blend_loss")
    print(f"  Mean per-row log-loss improvement (raw - blended, >0 = blend better), "
          f"player-clustered: {observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    verdict = "IMPROVES" if lo > 0 else ("HURTS" if hi < 0 else "NO SIGNIFICANT EFFECT (CI straddles zero)")
    print(f"  -> Rank-blending {verdict} calibration on the trajectory-lag population, held out.\n")

    # motivating cases: does Rybakina/Andreeva actually show up in this population, and if so what
    # does the blend do to their predictions specifically?
    print("--- Motivating cases (Rybakina E. / Andreeva M.) ---")
    all_lag_names = set(train_lag["player"]) | set(test_lag["player"])
    pool_names = sorted(set(preds["player"]))
    for raw_name in MOTIVATING_PLAYERS:
        resolved = match_name_to_pool(raw_name, pool_names) or raw_name
        in_lag_pop = resolved in all_lag_names
        player_rows = test[test["player"] == resolved]
        if len(player_rows) == 0:
            print(f"  {raw_name} (-> {resolved}): no test-era rows found")
            continue
        latest = player_rows.sort_values("date").iloc[-1]
        gap = latest["gap"]
        print(f"  {raw_name} (-> {resolved}): most recent test-era row {pd.Timestamp(latest['date']).date()}, "
              f"matches_before={latest['player_matches_before']}, rank={latest['player_rank']}, "
              f"Elo={latest['player_elo']:.0f}, rank-implied Elo={latest['player_rank_elo']:.0f}, "
              f"gap={gap:+.0f} pts (threshold {gap_threshold:.0f}) -> "
              f"{'IN' if in_lag_pop else 'NOT in'} the trajectory-lag population at any point in the dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="WTA", choices=["ATP", "WTA"])
    parser.add_argument("--max-editions", type=int, default=None,
                         help="quick-check mode: only replay the most recent N tournament editions "
                              "(before the 80/20 split), instead of the full lookback window")
    args = parser.parse_args()
    run(args.tour, max_editions=args.max_editions)
