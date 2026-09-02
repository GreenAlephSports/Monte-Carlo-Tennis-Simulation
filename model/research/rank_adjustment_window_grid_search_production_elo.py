"""Re-runs rank_adjustment_window_grid_search.py's question - is RANK_ADJUSTMENT_ELO_WINDOW=50
(win_probability.py) actually the best-performing gate for the rank-gap correction? - but using
PRODUCTION'S ACTUAL SURFACE-SPECIFIC ELO (elo_ratings.calculate_elo_ratings: 5-year hard lookback,
per-surface blending toward overall_elo weighted by surface sample size, surface-mismatch damping,
decay3 for WTA) instead of the research series' simplified single-online-pass overall Elo.

This directly tests whether the first grid search's surprising result (no cutoff showed a
significant held-out benefit; several bands, incl. 40-50, were significantly NEGATIVE) was an
artifact of that simplification, or a real, present-day breakdown of the correction - by rebuilding
the EXACT ratings snapshot win_probability() itself reads (same function, same columns:
{surface}_elo, current_rank), frozen per tournament edition, no shortcuts.

Cost note (why this is a separate, slower script rather than a flag on the first one):
calculate_elo_ratings() is called once per TEST-era edition (not every edition - the rank-gap
formula itself, RANK_ADJUSTMENT_C/D, is held fixed and not being re-fit here, so only held-out
predictions are needed) with cutoff_date = that edition's own start - a REAL walk-forward replay of
up to 5 years of real match history each time (per elite_opponent_residual_test.build_frozen_
predictions' own docstring: this is the O(editions x window matches) cost the research series'
simplification exists to avoid; unavoidable here since the whole point is testing production's
actual windowed Elo, not a cheaper approximation of it).

Same rigor otherwise: chronological tournament-level 80/20 edition split (edition boundary
computed once, from the FULL edition list, identical convention to every other test tonight),
held-out log-loss, player-clustered bootstrap CIs. ATP only - the correction is fit and documented
as ATP-only in win_probability.py.

Usage:
    python model/research/rank_adjustment_window_grid_search_production_elo.py [--max-editions N]
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import TRAIN_FRACTION, log_loss  # noqa: E402
from elo_ratings import SURFACES, calculate_elo_ratings, expected_score, load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from win_probability import RANK_ADJUSTMENT_C, RANK_ADJUSTMENT_D, _apply_rank_adjustment  # noqa: E402

CANDIDATE_WINDOWS = [30, 40, 50, 60, 75]
MARGINAL_BANDS = [
    ("0_30", 0, 30), ("30_40", 30, 40), ("40_50", 40, 50), ("50_60", 50, 60),
    ("60_75", 60, 75), ("75_100", 75, 100), ("100_plus", 100, float("inf")),
]


def build_all_editions(df):
    df = df.copy()
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)
    edition_start = df.groupby("edition_id")["Date"].transform("min")
    editions = (
        df.assign(edition_start=edition_start)[["edition_id", "edition_start"]]
        .drop_duplicates().sort_values("edition_start").reset_index(drop=True)
    )
    return df, editions


def build_production_test_predictions(df, tour, test_edition_ids, max_editions=None):
    """One player-perspective row per (match, player) for TEST-era editions only, using the real
    production ratings snapshot (calculate_elo_ratings, frozen at that edition's own start date) -
    same surface_elo/current_rank columns win_probability() itself reads. Non-{Hard,Clay,Grass}
    surface matches are skipped (production's rank-gap gate operates on surface Elo, which is only
    defined for those three)."""
    editions = df[["edition_id", "Date"]].drop_duplicates(subset=["edition_id"])
    test_editions = (
        df[df["edition_id"].isin(test_edition_ids)][["edition_id"]]
        .drop_duplicates()
        .merge(df.groupby("edition_id")["Date"].min().rename("edition_start"), on="edition_id")
        .sort_values("edition_start")
        .reset_index(drop=True)
    )
    if max_editions is not None:
        test_editions = test_editions.tail(max_editions).reset_index(drop=True)

    rows = []
    t0 = time.time()
    for i, (edition_id, cutoff) in enumerate(zip(test_editions["edition_id"], test_editions["edition_start"])):
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
            print(f"    [{tour}] {i + 1}/{len(test_editions)} test editions replayed with real "
                  f"production Elo ({time.time() - t0:.0f}s elapsed)")

    preds = pd.DataFrame(rows, columns=[
        "edition_id", "date", "player", "opponent", "player_elo", "opponent_elo",
        "pred_win", "actual_win", "own_rank", "opponent_rank",
    ])
    print(f"    [{tour}] done: {len(test_editions)} test editions, {len(preds)} rows, "
          f"{time.time() - t0:.0f}s total")
    return preds


def report_window(label, rows):
    rows = rows.dropna(subset=["own_rank", "opponent_rank"])
    if len(rows) < 20:
        print(f"  {label:<10}: n={len(rows):<6} - too few rank-known rows to evaluate")
        return
    adjusted = rows.apply(
        lambda r: _apply_rank_adjustment(r["pred_win"], r["own_rank"], r["opponent_rank"]), axis=1)
    rows = rows.assign(
        raw_loss=log_loss(rows["actual_win"].values, rows["pred_win"].values),
        adj_loss=log_loss(rows["actual_win"].values, adjusted.values),
    )
    observed, lo, hi = cluster_bootstrap_ci(rows, "raw_loss", "adj_loss", group_col="player")
    sig = "  <- excludes zero" if (lo > 0 or hi < 0) else ""
    print(f"  {label:<10}: n={len(rows):<6} held-out log-loss improvement (raw-adj)={observed:+.5f} "
          f"CI[{lo:+.5f},{hi:+.5f}]{sig}")


def run(max_editions=None):
    if max_editions is not None:
        print(f"*** QUICK CHECK ON RECENT DATA ONLY (--max-editions {max_editions}) - NOT the "
              f"full-historical verdict. Rerun without --max-editions before trusting this. ***\n")

    matches = load_matches_for_tour("ATP")
    df, editions = build_all_editions(matches)

    split_idx = int(len(editions) * TRAIN_FRACTION)
    test_edition_ids = set(editions["edition_id"].iloc[split_idx:])
    print(f"ATP: {len(editions)} tournament editions "
          f"({editions['edition_start'].min().date()} to {editions['edition_start'].max().date()}); "
          f"train = first {split_idx} editions (through "
          f"{editions['edition_start'].iloc[split_idx - 1].date()}), "
          f"test = remaining {len(test_edition_ids)} editions "
          f"(from {editions['edition_start'].iloc[split_idx].date()}) - "
          f"SAME edition boundary as the simplified-Elo grid search")
    print(f"Fixed formula being gated: RANK_ADJUSTMENT_C={RANK_ADJUSTMENT_C}, "
          f"RANK_ADJUSTMENT_D={RANK_ADJUSTMENT_D} (unchanged - only the |Elo diff| gate varies)")
    print(f"\nReplaying {len(test_edition_ids)} test-era editions with REAL production "
          f"calculate_elo_ratings (5yr lookback, surface blend+damping) - this is the slow part...")

    test = build_production_test_predictions(df, "ATP", test_edition_ids, max_editions=max_editions)
    test["elo_diff"] = (test["player_elo"] - test["opponent_elo"]).abs()

    print(f"\n{'=' * 100}\nCUMULATIVE: held-out improvement over ALL test rows with |Elo diff| <= W "
          f"(nested supersets - what production would see at each candidate gate), PRODUCTION "
          f"SURFACE ELO\n{'=' * 100}")
    for w in CANDIDATE_WINDOWS:
        report_window(f"W<={w}", test[test["elo_diff"] <= w])

    print(f"\n{'=' * 100}\nMARGINAL: held-out improvement inside each non-overlapping Elo-diff band, "
          f"PRODUCTION SURFACE ELO\n{'=' * 100}")
    for name, lo_edge, hi_edge in MARGINAL_BANDS:
        band = test[(test["elo_diff"] > lo_edge) & (test["elo_diff"] <= hi_edge)] if lo_edge > 0 \
            else test[test["elo_diff"] <= hi_edge]
        report_window(name, band)

    print(f"\n{'=' * 100}\nVERDICT (production surface Elo, vs. the simplified-Elo grid search)\n{'=' * 100}")
    print("If these numbers now show real, CI-excluding-zero positive improvement (esp. at/near "
          "W=50) where the simplified-Elo run showed none/negative, the earlier result was an "
          "artifact of the research-series Elo simplification (no surface split, no 5yr window, no "
          "blending/damping) and production's actual correction is fine. If the pattern is the "
          "same - no cutoff clearly beats the others, and/or the 40-50 or 100+ bands are still "
          "significantly negative - that confirms this is a real, present-day breakdown of the "
          "correction, not a methodology artifact, and the Elo-source difference is ruled out as "
          "the explanation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-editions", type=int, default=None,
                         help="quick-check mode: only replay the most recent N test-era editions "
                              "instead of all of them")
    args = parser.parse_args()
    run(max_editions=args.max_editions)
