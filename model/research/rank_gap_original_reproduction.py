"""Harness-trust check, not a new test: reproduce the ORIGINAL rank-gap validation cited in
win_probability.py's own docstring (RANK_ADJUSTMENT_C/D, RANK_ADJUSTMENT_ELO_WINDOW) as closely as
its own description allows, and see whether TODAY's data + THIS harness's production Elo still
shows the same finding it originally reported:

    "among 17,955 historical ATP matches with |Elo diff| <= 50 (i.e. Elo calls them a near-
    coin-flip), current ATP ranking predicted a real, monotonic excess win rate for the better-
    ranked player that Elo alone missed (up to +15.7pp for a 100+ rank-gap bucket, holding up
    across every Elo-window and rank-range sensitivity check tried)... had the best held-out
    log-likelihood on a tournament-level 80/20 split."

DISCLOSURE, up front: the original backtest script that produced this was never committed to this
repository (confirmed: `git log --all --diff-filter=D` finds no deleted rank-gap script, and the
commit that first introduced RANK_ADJUSTMENT_C/D, 7c43c142, adds only the constants to
win_probability.py - no accompanying script). Its exact rank-gap bucket edges and the nonlinear fit
procedure for C/D are therefore not recoverable verbatim. This script reproduces every part of the
methodology that IS fully specified by that docstring, verbatim:
  - ATP only, production's real surface-specific Elo (elo_ratings.calculate_elo_ratings: 5yr hard
    lookback, per-surface blend+damping) frozen per tournament edition - not the research series'
    simplified overall-Elo shortcut.
  - Population: |Elo diff| <= 50 (RANK_ADJUSTMENT_ELO_WINDOW itself).
  - A monotonic rank-gap bucketing with a named "100+" top bucket (reconstructed as 0-50/50-100/
    100+, the same 3-band convention this project already uses for an analogous gap variable in
    survivorship_upset_test.py's BUCKET_EDGES - the closest available precedent for "a monotonic
    gap bucketing with a 100+ top band" in this codebase).
  - Tournament-level chronological 80/20 train/test split (elite_opponent_residual_test.
    TRAIN_FRACTION, the same split every correction in this project uses).
  - The EXISTING FIXED formula (RANK_ADJUSTMENT_C=1.0629, RANK_ADJUSTMENT_D=260.72) - not re-fit
    here (re-deriving those two constants from scratch via nonlinear regression is a different,
    larger undertaking than a reproduction check; this script instead asks the more direct
    question the original docstring's own bucket table already answers: does the RAW empirical
    actual-vs-Elo-predicted excess win rate, bucketed by rank gap, still look like what was
    originally reported?).

Two independent checks, reported side by side:
  1. TRAIN-ERA bucket table (same era style of population the original fit would have used): does
     the raw actual-minus-predicted excess still come out real, monotonic, and ~15pp-scale in the
     100+ bucket? This is the direct "does the harness still find the original pattern" check.
  2. TEST-ERA held-out validation of the EXISTING fixed formula (same population, same |Elo diff|
     <=50 gate): does it still show a held-out log-likelihood benefit? (This reuses the same
     machinery as rank_adjustment_window_grid_search_production_elo.py's W<=50 result - reported
     here again for direct side-by-side comparison against the train-era pattern.)

If (1) reproduces the original pattern - real, monotonic, ~15pp-scale in the top bucket - the
harness is trustworthy and today's held-out null (2) is a genuine, new, present-day finding (recent
years behave differently from the 2000-2021 era the correction was implicitly fit on), not a sign
of a broken test harness. If (1) does NOT reproduce - the pattern is absent, non-monotonic, or
much smaller even on the original-style training population - that points to something wrong in
the shared harness itself (calculate_elo_ratings, the rank columns, the bucket/log-loss machinery),
and every other null result tonight needs to be treated as suspect until that's found.

Usage:
    python model/research/rank_gap_original_reproduction.py [--max-editions N]
"""
import argparse
import math
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

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
CACHE_PATH = OUTPUT_DIR / "_rank_gap_reproduction_atp_predictions.csv"

RANK_GAP_BUCKETS = [
    ("0_50", 0, 50), ("50_100", 50, 100), ("100_plus", 100, float("inf")),
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


def build_production_predictions(df, tour, editions, max_editions=None):
    """Same mechanism as rank_adjustment_window_grid_search_production_elo.py's
    build_production_test_predictions, generalized to run over an arbitrary (here: ALL, not just
    test-era) set of editions - real production calculate_elo_ratings snapshot, frozen at each
    edition's own start date, real surface Elo + current_rank columns."""
    target = editions if max_editions is None else editions.tail(max_editions).reset_index(drop=True)

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
        if (i + 1) % 50 == 0:
            print(f"    [{tour}] {i + 1}/{len(target)} editions replayed with real production Elo "
                  f"({time.time() - t0:.0f}s elapsed)")

    preds = pd.DataFrame(rows, columns=[
        "edition_id", "date", "player", "opponent", "player_elo", "opponent_elo",
        "pred_win", "actual_win", "own_rank", "opponent_rank",
    ])
    print(f"    [{tour}] done: {len(target)} editions, {len(preds)} rows, {time.time() - t0:.0f}s total")
    return preds


def bucket_for_gap(gap):
    for name, lo, hi in RANK_GAP_BUCKETS:
        if lo < gap <= hi if lo > 0 else gap <= hi:
            return name
    return "100_plus"


def run(max_editions=None):
    if max_editions is not None:
        print(f"*** QUICK CHECK ON RECENT DATA ONLY (--max-editions {max_editions}) - NOT the "
              f"full-historical reproduction. Rerun without --max-editions before trusting this. ***\n")

    matches = load_matches_for_tour("ATP")
    df, editions = build_all_editions(matches)
    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_edition_ids = set(editions["edition_id"].iloc[:split_idx])
    test_edition_ids = set(editions["edition_id"].iloc[split_idx:])
    print(f"ATP: {len(editions)} tournament editions "
          f"({editions['edition_start'].min().date()} to {editions['edition_start'].max().date()}); "
          f"train = first {split_idx} editions (through "
          f"{editions['edition_start'].iloc[split_idx - 1].date()}), "
          f"test = remaining {len(test_edition_ids)} editions "
          f"(from {editions['edition_start'].iloc[split_idx].date()})")

    if CACHE_PATH.exists() and max_editions is None:
        print(f"\nUsing cached full-population predictions at {CACHE_PATH} "
              f"(delete this file to force a fresh replay)")
        preds = pd.read_csv(CACHE_PATH, parse_dates=["date"])
    else:
        print(f"\nReplaying {len(editions)} editions (ALL - train+test) with REAL production "
              f"calculate_elo_ratings - this is the slow part, ~15-20 min for full ATP history...")
        preds = build_production_predictions(df, "ATP", editions, max_editions=max_editions)
        if max_editions is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            preds.to_csv(CACHE_PATH, index=False)
            print(f"Cached full-population predictions to {CACHE_PATH}")

    preds["elo_diff"] = (preds["player_elo"] - preds["opponent_elo"]).abs()
    preds["era"] = preds["edition_id"].apply(
        lambda e: "train" if e in train_edition_ids else ("test" if e in test_edition_ids else "other"))

    near_coinflip = preds[(preds["elo_diff"] <= 50) & preds["own_rank"].notna() & preds["opponent_rank"].notna()]
    n_matches = len(near_coinflip) // 2
    print(f"\n{'=' * 100}\nPOPULATION CHECK: |Elo diff| <= 50, rank known both sides "
          f"(production surface Elo, full ATP history)\n{'=' * 100}")
    print(f"  {len(near_coinflip)} player-perspective rows = ~{n_matches} matches "
          f"(original docstring cites 17,955 matches - today's data runs later, through "
          f"{editions['edition_start'].max().date()}, so an exact match isn't expected, but this "
          f"should be the same ORDER OF MAGNITUDE if the harness is reproducing the same "
          f"population definition)")

    near_coinflip = near_coinflip.copy()
    near_coinflip["rank_gap"] = (near_coinflip["own_rank"] - near_coinflip["opponent_rank"]).abs()
    near_coinflip["better_ranked_won"] = (
        ((near_coinflip["own_rank"] < near_coinflip["opponent_rank"]) & (near_coinflip["actual_win"] == 1)) |
        ((near_coinflip["own_rank"] > near_coinflip["opponent_rank"]) & (near_coinflip["actual_win"] == 0))
    )
    near_coinflip["bucket"] = near_coinflip["rank_gap"].apply(bucket_for_gap)

    print(f"\n{'=' * 100}\nCHECK 1 - TRAIN-ERA bucket table: does the original PATTERN reproduce?\n{'=' * 100}")
    print("(actual win rate FOR THE BETTER-RANKED PLAYER vs. Elo's average predicted win rate for "
          "that same better-ranked player, by rank-gap bucket - the original claim was a real, "
          "monotonic, up-to-+15.7pp-at-100+ excess)")
    train_pop = near_coinflip[near_coinflip["era"] == "train"]
    for name, lo, hi in RANK_GAP_BUCKETS:
        g = train_pop[train_pop["bucket"] == name]
        # restrict to the better-ranked player's own perspective row for a clean "predicted win
        # rate for the favored-by-rank side" comparison, matching the docstring's framing
        g_better = g[((g["own_rank"] < g["opponent_rank"]))]
        if len(g_better) < 20:
            print(f"  {name:<10}: n={len(g_better):<6} - too few to evaluate")
            continue
        actual = g_better["better_ranked_won"].mean()
        pred = g_better["pred_win"].mean()
        excess = actual - pred
        se = math.sqrt((g_better["pred_win"] * (1 - g_better["pred_win"])).sum()) / len(g_better)
        z = excess / se if se > 0 else float("nan")
        print(f"  {name:<10}: n={len(g_better):<6}  actual={actual:.1%}  elo_pred={pred:.1%}  "
              f"excess={excess:+.1%}  z={z:.2f}")

    print(f"\n{'=' * 100}\nCHECK 2 - TEST-ERA held-out validation of the EXISTING fixed formula "
          f"(same population)\n{'=' * 100}")
    test_pop = near_coinflip[near_coinflip["era"] == "test"].copy()
    adjusted = test_pop.apply(
        lambda r: _apply_rank_adjustment(r["pred_win"], r["own_rank"], r["opponent_rank"]), axis=1)
    test_pop = test_pop.assign(
        raw_loss=log_loss(test_pop["actual_win"].values, test_pop["pred_win"].values),
        adj_loss=log_loss(test_pop["actual_win"].values, adjusted.values),
    )
    observed, lo, hi = cluster_bootstrap_ci(test_pop, "raw_loss", "adj_loss", group_col="player")
    sig = "excludes zero" if (lo > 0 or hi < 0) else "includes zero, not significant"
    print(f"  n={len(test_pop)} test-era rows, |Elo diff|<=50")
    print(f"  Held-out log-loss improvement (raw - adjusted, existing fixed formula): {observed:+.5f} "
          f"CI[{lo:+.5f},{hi:+.5f}] ({sig})")

    print(f"\n{'=' * 100}\nVERDICT\n{'=' * 100}")
    print("If CHECK 1's excess column is real, positive, and roughly monotonic across the three "
          "buckets (comparable order of magnitude to the original +15.7pp at 100+) - the harness "
          "DOES reproduce the original finding on the original-style (train-era, production-Elo) "
          "population. Combined with CHECK 2 still coming back null/negative on the held-out "
          "test-era population, that's a real, present-day breakdown of the correction (recent "
          "years no longer show the pattern the correction was built on), NOT a harness problem - "
          "the harness correctly detects the original signal where it exists and correctly detects "
          "its absence where it doesn't. If CHECK 1 ALSO fails to reproduce (excess near zero, "
          "non-monotonic, or wrong-signed even in the train-era, same-methodology population), that "
          "means something in the shared harness (calculate_elo_ratings, the rank columns, or the "
          "log-loss/bucket machinery itself) has changed or is broken, and every other null result "
          "tonight needs to be re-examined.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-editions", type=int, default=None,
                         help="quick-check mode: only replay the most recent N editions instead of "
                              "full ATP history (skips the on-disk cache)")
    args = parser.parse_args()
    run(max_editions=args.max_editions)
