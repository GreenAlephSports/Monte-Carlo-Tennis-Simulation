"""Recency refit of the rank-gap correction (win_probability.RANK_ADJUSTMENT_C/D) - restricted to
ONLY 2021-2026 ATP data, both for fitting and for validation, instead of the full 2000-2026 history
the current fixed constants were implicitly fit on.

Motivation: rank_gap_original_reproduction.py already showed the ORIGINAL pattern reproduces on a
train-era (older) population but the EXISTING fixed formula shows no held-out benefit on the
(mostly-recent) test-era split of full history - i.e. the correction that was real for the era it
was fit on may no longer describe how ranking-vs-Elo divergence behaves in the CURRENT game. This
script asks the direct follow-up: is that because the constants are stale (a recency refit restores
a real, significant benefit), or because the underlying signal itself has disappeared (even a
same-era refit shows nothing significant held out)?

Methodology - same rigor as every other correction in this project:
  - ATP only, production's real surface-specific Elo (elo_ratings.calculate_elo_ratings: 5yr hard
    lookback, per-surface blend+damping), frozen per tournament edition.
  - Population restricted to matches from tournament editions starting on/after 2021-01-01 through
    the latest data (2026-08). All Elo history before an edition's own cutoff (including pre-2021
    matches) is still used to COMPUTE that edition's Elo snapshot - only the editions being
    predicted/fit are restricted to the recent window, not the Elo lookback itself.
  - Gated to |Elo diff| <= 50 (RANK_ADJUSTMENT_ELO_WINDOW) - same near-coin-flip population the
    correction is scoped to in production; rank known both sides.
  - Chronological tournament-level 80/20 split WITHIN the 2021-2026 population (elite_opponent_
    residual_test.TRAIN_FRACTION convention, recomputed on this restricted edition list rather than
    reusing the full-history boundary).
  - C, D refit on TRAIN-era rows only, via MLE (minimize held-out-style log-loss on train), then
    validated on TEST-era rows with player-clustered bootstrap CIs - not fit and validated on the
    same rows.
  - The EXISTING fixed formula is evaluated on the exact same test-era split, side by side, so the
    comparison is apples-to-apples (same population, same split) rather than against the full-
    history reproduction script's numbers.

Usage:
    python model/research/rank_adjustment_recency_refit.py [--max-editions N]
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import TRAIN_FRACTION, log_loss  # noqa: E402
from elo_ratings import SURFACES, calculate_elo_ratings, expected_score, load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from win_probability import RANK_ADJUSTMENT_C, RANK_ADJUSTMENT_D, _apply_rank_adjustment  # noqa: E402

RECENCY_CUTOFF = pd.Timestamp("2021-01-01")
RANK_ADJUSTMENT_ELO_WINDOW = 50


def build_all_editions(df):
    df = df.copy()
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)
    edition_start = df.groupby("edition_id")["Date"].transform("min")
    editions = (
        df.assign(edition_start=edition_start)[["edition_id", "edition_start"]]
        .drop_duplicates().sort_values("edition_start").reset_index(drop=True)
    )
    return df, editions


def build_predictions(df, tour, target_editions, max_editions=None):
    """One player-perspective row per (match, player) for the given editions, using real production
    calculate_elo_ratings frozen at each edition's own start date - full match history (including
    pre-2021 matches) is passed in as df so the 5yr lookback behaves exactly as it would in
    production; only the EDITIONS being predicted are restricted by the caller."""
    target = target_editions if max_editions is None else target_editions.tail(max_editions).reset_index(drop=True)

    rows = []
    t0 = time.time()
    for i, (edition_id, cutoff) in enumerate(zip(target["edition_id"], target["edition_start"])):
        ratings_df = calculate_elo_ratings(df, cutoff, tour=tour)
        idx = ratings_df.set_index("player")
        edition_matches = df[df["edition_id"] == edition_id]
        for row in edition_matches.itertuples(index=False):
            p1, p2, winner, surface = row.Player_1, row.Player_2, row.Winner, row.Surface
            if surface not in SURFACES:
                continue
            if p1 not in idx.index or p2 not in idx.index:
                continue
            col = f"{surface.lower()}_elo"
            elo1, elo2 = idx.loc[p1, col], idx.loc[p2, col]
            rank1, rank2 = idx.loc[p1, "current_rank"], idx.loc[p2, "current_rank"]
            pred1 = expected_score(elo1, elo2)
            win1 = 1 if winner == p1 else 0
            rows.append((edition_id, row.Date, p1, p2, elo1, elo2, pred1, win1, rank1, rank2))
            rows.append((edition_id, row.Date, p2, p1, elo2, elo1, 1 - pred1, 1 - win1, rank2, rank1))
        if (i + 1) % 25 == 0:
            print(f"    [{tour}] {i + 1}/{len(target)} editions replayed with real production Elo "
                  f"({time.time() - t0:.0f}s elapsed)")

    preds = pd.DataFrame(rows, columns=[
        "edition_id", "date", "player", "opponent", "player_elo", "opponent_elo",
        "pred_win", "actual_win", "own_rank", "opponent_rank",
    ])
    print(f"    [{tour}] done: {len(target)} editions, {len(preds)} rows, {time.time() - t0:.0f}s total")
    return preds


def _adjusted_with_params(rows, c, d):
    def apply_one(r):
        rank_a, rank_b = r["own_rank"], r["opponent_rank"]
        if pd.isna(rank_a) or pd.isna(rank_b) or rank_a == rank_b:
            return r["pred_win"]
        gap = abs(rank_a - rank_b)
        shift = c * math.log1p(gap / d)
        sign = 1.0 if rank_a < rank_b else -1.0
        logit_p = math.log(r["pred_win"] / (1 - r["pred_win"]))
        return 1 / (1 + math.exp(-(logit_p + sign * shift)))
    return rows.apply(apply_one, axis=1)


def fit_c_d(train_rows):
    """MLE refit of C, D on TRAIN-era rows only: minimize mean log-loss of the adjusted prediction
    against actual outcomes. Initialized at the existing production constants (a warm start, not a
    prior/regularizer - the objective is plain unregularized log-loss)."""
    def objective(params):
        c, d = params
        if c <= 0 or d <= 1:
            return 1e6
        adjusted = _adjusted_with_params(train_rows, c, d).values
        return log_loss(train_rows["actual_win"].values, adjusted).mean()

    result = minimize(
        objective, x0=[RANK_ADJUSTMENT_C, RANK_ADJUSTMENT_D],
        method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 2000},
    )
    return result.x[0], result.x[1], result


def report(label, rows, c, d):
    rows = rows.dropna(subset=["own_rank", "opponent_rank"])
    if len(rows) < 20:
        print(f"  {label:<28}: n={len(rows):<6} - too few rank-known rows to evaluate")
        return None
    adjusted = _adjusted_with_params(rows, c, d)
    rows = rows.assign(
        raw_loss=log_loss(rows["actual_win"].values, rows["pred_win"].values),
        adj_loss=log_loss(rows["actual_win"].values, adjusted.values),
    )
    observed, lo, hi = cluster_bootstrap_ci(rows, "raw_loss", "adj_loss", group_col="player")
    sig = "  <- excludes zero (real benefit)" if (lo > 0 or hi < 0) else "  (includes zero, not significant)"
    print(f"  {label:<28}: n={len(rows):<6} held-out log-loss improvement (raw-adj)={observed:+.5f} "
          f"CI[{lo:+.5f},{hi:+.5f}]{sig}")
    return observed, lo, hi


def run(max_editions=None):
    if max_editions is not None:
        print(f"*** QUICK CHECK ({'--max-editions ' + str(max_editions)}) - NOT the full 2021-2026 "
              f"verdict. Rerun without --max-editions before trusting this. ***\n")

    matches = load_matches_for_tour("ATP")
    df, all_editions = build_all_editions(matches)

    recent_editions = all_editions[all_editions["edition_start"] >= RECENCY_CUTOFF].reset_index(drop=True)
    split_idx = int(len(recent_editions) * TRAIN_FRACTION)
    train_ids = set(recent_editions["edition_id"].iloc[:split_idx])
    test_ids = set(recent_editions["edition_id"].iloc[split_idx:])
    print(f"ATP, 2021-2026 only: {len(recent_editions)} tournament editions "
          f"({recent_editions['edition_start'].min().date()} to {recent_editions['edition_start'].max().date()})")
    print(f"  train = first {split_idx} editions (through "
          f"{recent_editions['edition_start'].iloc[split_idx - 1].date()})")
    print(f"  test  = remaining {len(test_ids)} editions "
          f"(from {recent_editions['edition_start'].iloc[split_idx].date()})")
    print(f"  Elo lookback still uses FULL history (df passed in unrestricted) - only the predicted "
          f"editions are restricted to 2021-2026.\n")

    print(f"Replaying {len(train_ids)} train-era editions with real production Elo...")
    train_target = recent_editions[recent_editions["edition_id"].isin(train_ids)].reset_index(drop=True)
    train_preds = build_predictions(df, "ATP", train_target, max_editions=max_editions)

    print(f"\nReplaying {len(test_ids)} test-era editions with real production Elo...")
    test_target = recent_editions[recent_editions["edition_id"].isin(test_ids)].reset_index(drop=True)
    test_preds = build_predictions(df, "ATP", test_target, max_editions=max_editions)

    train_preds["elo_diff"] = (train_preds["player_elo"] - train_preds["opponent_elo"]).abs()
    test_preds["elo_diff"] = (test_preds["player_elo"] - test_preds["opponent_elo"]).abs()

    train_pop = train_preds[
        (train_preds["elo_diff"] <= RANK_ADJUSTMENT_ELO_WINDOW)
        & train_preds["own_rank"].notna() & train_preds["opponent_rank"].notna()
    ].copy()
    test_pop = test_preds[
        (test_preds["elo_diff"] <= RANK_ADJUSTMENT_ELO_WINDOW)
        & test_preds["own_rank"].notna() & test_preds["opponent_rank"].notna()
    ].copy()
    print(f"\n|Elo diff|<=50, rank known both sides: train n={len(train_pop)} rows "
          f"(~{len(train_pop)//2} matches), test n={len(test_pop)} rows (~{len(test_pop)//2} matches)")

    if len(train_pop) < 100 or len(test_pop) < 20:
        print("\nToo few rows to fit/validate reliably - aborting. Rerun without --max-editions.")
        return

    print(f"\n{'=' * 100}\nSTEP 1 - refit C, D on TRAIN-era (2021-2026 only) rows via MLE\n{'=' * 100}")
    new_c, new_d, opt_result = fit_c_d(train_pop)
    print(f"  existing production constants: C={RANK_ADJUSTMENT_C:.4f}, D={RANK_ADJUSTMENT_D:.2f}")
    print(f"  recency-refit constants:        C={new_c:.4f}, D={new_d:.2f}  "
          f"(optimizer: {'converged' if opt_result.success else 'DID NOT CONVERGE - treat with caution'}, "
          f"{opt_result.nit} iterations)")

    print(f"\n{'=' * 100}\nSTEP 2 - held-out validation on TEST-era (2021-2026 only) rows, "
          f"side by side\n{'=' * 100}")
    report("EXISTING fixed formula", test_pop, RANK_ADJUSTMENT_C, RANK_ADJUSTMENT_D)
    result_refit = report("RECENCY-REFIT formula", test_pop, new_c, new_d)

    print(f"\n{'=' * 100}\nVERDICT\n{'=' * 100}")
    if result_refit is not None:
        observed, lo, hi = result_refit
        if lo > 0:
            print("Recency-refit constants show a REAL, significant held-out benefit on 2021-2026 "
                  "data (CI excludes zero, positive) where the existing constants do not. The "
                  "underlying rank-gap signal is still real in the current game - it just needs "
                  "updated constants (C, D above) for the current era. This is an evidence-backed "
                  "production update: replace RANK_ADJUSTMENT_C/D with the recency-refit values.")
        else:
            print("Recency-refit constants do NOT show a significant held-out benefit either (CI "
                  "includes zero, or negative) - refitting C/D on recent-only data doesn't rescue "
                  "the correction. This points to the underlying rank-vs-Elo divergence signal "
                  "itself having disappeared in the current game, not just stale constants. The "
                  "honest move is disabling RANK_ADJUSTMENT (use_rank_adjustment default) in "
                  "production until a real, held-out-validated replacement is found, rather than "
                  "leaving a known-broken correction live.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-editions", type=int, default=None,
                         help="quick-check mode: only replay the most recent N editions of each "
                              "(train, test) split instead of the full 2021-2026 population")
    args = parser.parse_args()
    run(max_editions=args.max_editions)
