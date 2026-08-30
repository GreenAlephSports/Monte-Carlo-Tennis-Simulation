"""Fits and validates a flat damping correction for the surface-mismatch overconfidence/
underconfidence finding from tonight's real-market tests (surface_mismatch_market_test.py,
surface_mismatch_raw_elo_test.py, surface_mismatch_magnitude_test.py): when a player's own
surface_elo diverges from their overall_elo by more than SURFACE_MISMATCH_THRESHOLD (50, matching
tonight's own selection threshold), pull the divergence back toward overall_elo by a FIXED number
of Elo points (not a magnitude-scaled function - tonight's dose-response regression on the small
real-market sample (n=772) was not significant, so a flat damp is what the data actually supports,
per the user's own framing of this task).

Run at FULL historical scale (both tours, real Kaggle match history, not the 1,749-match real-market
sample - that sample was needed for the ORIGINAL finding because it required real market odds as an
independent yardstick, but fitting/validating a pure calibration correction only needs real outcomes,
which the full historical dataset has orders of magnitude more of). Same frozen-per-tournament-
edition-Elo, chronological 80/20 train/test split convention as every other production correction
(rank-gap, layoff, recent-form, Platt). One simplification, same precedent as elite_opponent_
residual_test.build_frozen_predictions and recent_form_test.py: a single continuously-updated Elo
walk-forward (overall + per-surface, blended via elo_ratings.SURFACE_BLEND_K) rather than
recomputing with the production 5yr lookback window / decay3 per edition - recomputing per-edition
at full historical scale (thousands of editions) doesn't finish in reasonable time, and (per those
scripts' own precedent) shouldn't matter for a "does damping large surface/overall splits help"
question, since the lookback window's purpose is dropping long-inactive players, not changing
surface-blend dynamics.

Usage:
    python model/research/surface_mismatch_damping_fit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elo_ratings import K_FACTOR, STARTING_ELO, SURFACE_BLEND_K, SURFACES, expected_score, load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

TRAIN_FRACTION = 0.8
SURFACE_MISMATCH_THRESHOLD = 50.0
DAMP_GRID = list(range(0, 121, 10))
EPS = 1e-3


def log_loss(actual, pred):
    pred = np.clip(pred, EPS, 1 - EPS)
    return -(actual * np.log(pred) + (1 - actual) * np.log(1 - pred))


def build_frozen_predictions_surface(df, max_editions=None):
    """Player-perspective rows, frozen per tournament edition - same shape/discipline as
    elite_opponent_residual_test.build_frozen_predictions, extended to also track SURFACE-blended
    Elo (same SURFACE_BLEND_K formula elo_ratings.calculate_elo_ratings uses) alongside overall_elo,
    both frozen at the edition's own start (no in-tournament lookahead)."""
    df = df.dropna(subset=["Date"]).copy()  # drops the 2 WTA NaT rows that otherwise upcast .dt.year to float
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)
    edition_start = df.groupby("edition_id")["Date"].transform("min")
    editions = (
        df.assign(edition_start=edition_start)[["edition_id", "edition_start"]]
        .drop_duplicates().sort_values("edition_start").reset_index(drop=True)
    )

    overall_elo = {}
    surface_elo = {s: {} for s in SURFACES}
    surface_matches = {s: {} for s in SURFACES}
    rows = []

    for edition_id in editions["edition_id"]:
        edition_matches = df[df["edition_id"] == edition_id]
        snap_overall = dict(overall_elo)
        snap_surface = {s: dict(surface_elo[s]) for s in SURFACES}
        snap_counts = {s: dict(surface_matches[s]) for s in SURFACES}

        def blended(surface, player, overall):
            raw = snap_surface[surface].get(player, STARTING_ELO)
            cnt = snap_counts[surface].get(player, 0)
            w = cnt / (cnt + SURFACE_BLEND_K)
            return w * raw + (1 - w) * overall

        for row in edition_matches.itertuples(index=False):
            p1, p2, winner, surface = row.Player_1, row.Player_2, row.Winner, row.Surface
            if surface not in SURFACES:
                continue
            oe1, oe2 = snap_overall.get(p1, STARTING_ELO), snap_overall.get(p2, STARTING_ELO)
            se1, se2 = blended(surface, p1, oe1), blended(surface, p2, oe2)
            pred1 = expected_score(se1, se2)
            win1 = 1 if winner == p1 else 0
            rows.append((edition_id, row.Date, row.Round, surface, p1, p2, oe1, oe2, se1, se2, pred1, win1))
            rows.append((edition_id, row.Date, row.Round, surface, p2, p1, oe2, oe1, se2, se1, 1 - pred1, 1 - win1))

        for row in edition_matches.itertuples(index=False):
            p1, p2, winner, surface = row.Player_1, row.Player_2, row.Winner, row.Surface
            overall_elo.setdefault(p1, STARTING_ELO)
            overall_elo.setdefault(p2, STARTING_ELO)
            score1 = 1.0 if winner == p1 else 0.0
            exp1 = expected_score(overall_elo[p1], overall_elo[p2])
            overall_elo[p1] += K_FACTOR * (score1 - exp1)
            overall_elo[p2] += K_FACTOR * ((1 - score1) - (1 - exp1))
            if surface in SURFACES:
                ratings, counts = surface_elo[surface], surface_matches[surface]
                ratings.setdefault(p1, STARTING_ELO)
                ratings.setdefault(p2, STARTING_ELO)
                exp1s = expected_score(ratings[p1], ratings[p2])
                ratings[p1] += K_FACTOR * (score1 - exp1s)
                ratings[p2] += K_FACTOR * ((1 - score1) - (1 - exp1s))
                counts[p1] = counts.get(p1, 0) + 1
                counts[p2] = counts.get(p2, 0) + 1

    preds = pd.DataFrame(rows, columns=[
        "edition_id", "date", "round", "surface", "player", "opponent",
        "player_overall_elo", "opponent_overall_elo", "player_surface_elo", "opponent_surface_elo",
        "pred_win", "actual_win",
    ])
    if max_editions is not None:
        editions = editions.tail(max_editions).reset_index(drop=True)
        keep = set(editions["edition_id"])
        preds = preds[preds["edition_id"].isin(keep)].reset_index(drop=True)
    return preds, editions


def damp_elo(surface_elo, overall_elo, damp_points, threshold=SURFACE_MISMATCH_THRESHOLD):
    mismatch = surface_elo - overall_elo
    abs_m = np.abs(mismatch)
    excess = np.maximum(abs_m - damp_points, threshold)
    new_abs_m = np.where(abs_m > threshold, np.minimum(abs_m, excess), abs_m)
    return overall_elo + np.sign(mismatch) * new_abs_m


def add_damped_pred(df, damp_points):
    dp = damp_elo(df["player_surface_elo"].values, df["player_overall_elo"].values, damp_points)
    do = damp_elo(df["opponent_surface_elo"].values, df["opponent_overall_elo"].values, damp_points)
    return expected_score(dp, do)


def build_tour(tour):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions_surface(matches)
    preds["tour"] = tour
    preds["mismatch"] = preds["player_surface_elo"] - preds["player_overall_elo"]
    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    print(f"{tour}: {len(editions)} editions ({editions['edition_start'].min().date()} to "
          f"{editions['edition_start'].max().date()}); train={len(train_editions)}, test={len(test_editions)}")
    return preds, train_editions, test_editions


def report_bucket(label, df, damp_points):
    if len(df) < 20:
        print(f"  {label}: n={len(df)} - too small")
        return
    raw_loss = log_loss(df["actual_win"].values, df["pred_win"].values)
    damped_pred = add_damped_pred(df, damp_points)
    damped_loss = log_loss(df["actual_win"].values, damped_pred)
    d = df.assign(raw_loss=raw_loss, damped_loss=damped_loss, damped_pred=damped_pred)

    ll_diff, ll_lo, ll_hi = cluster_bootstrap_ci(d, "raw_loss", "damped_loss", group_col="player")
    raw_gap, raw_lo, raw_hi = cluster_bootstrap_ci(
        d.assign(_a=d["actual_win"], _s=d["pred_win"]), "_a", "_s", group_col="player")
    damp_gap, damp_lo, damp_hi = cluster_bootstrap_ci(
        d.assign(_a=d["actual_win"], _s=d["damped_pred"]), "_a", "_s", group_col="player")
    print(f"  {label}: n={len(d)}")
    print(f"    Calibration gap (actual-pred): RAW={raw_gap:+.1%} CI[{raw_lo:+.1%},{raw_hi:+.1%}]  "
          f"DAMPED={damp_gap:+.1%} CI[{damp_lo:+.1%},{damp_hi:+.1%}]")
    print(f"    Log-loss improvement (raw-damped, >0=damped better): {ll_diff:+.4f} "
          f"CI[{ll_lo:+.4f},{ll_hi:+.4f}]" + ("  <- excludes zero" if ll_lo > 0 or ll_hi < 0 else ""))


def run():
    all_data = {tour: build_tour(tour) for tour in ("ATP", "WTA")}
    train = pd.concat([p[p["edition_id"].isin(te)] for p, te, _ in all_data.values()], ignore_index=True)
    test = pd.concat([p[p["edition_id"].isin(tt)] for p, _, tt in all_data.values()], ignore_index=True)
    print(f"\nTotal player-perspective rows: {len(train)} train-era, {len(test)} test-era (both tours)")

    print(f"\n{'=' * 90}\nGrid search DAMP_POINTS on TRAIN era (full population mean log-loss - "
          f"unaffected rows are numerically identical across candidates, so this is equivalent to "
          f"minimizing just the affected subset)\n{'=' * 90}")
    best_d, best_loss = None, float("inf")
    for d in DAMP_GRID:
        pred = add_damped_pred(train, d)
        mean_loss = log_loss(train["actual_win"].values, pred).mean()
        flag = ""
        if mean_loss < best_loss:
            best_loss, best_d = mean_loss, d
            flag = "  <- best so far"
        print(f"  DAMP_POINTS={d:>4}: train mean log-loss={mean_loss:.5f}{flag}")
    print(f"\nSelected DAMP_POINTS={best_d} (threshold={SURFACE_MISMATCH_THRESHOLD:.0f}) from TRAIN era only")

    print(f"\n{'=' * 90}\nHELD-OUT VALIDATION (test era, fitted DAMP_POINTS={best_d})\n{'=' * 90}")
    report_bucket("Full population", test, best_d)
    report_bucket(f"Affected (|mismatch| >= {SURFACE_MISMATCH_THRESHOLD:.0f}, either direction)",
                  test[test["mismatch"].abs() >= SURFACE_MISMATCH_THRESHOLD], best_d)
    report_bucket("Specialist-direction (mismatch >= +50)", test[test["mismatch"] >= SURFACE_MISMATCH_THRESHOLD], best_d)
    report_bucket("Mismatch-weakness-direction (mismatch <= -50)", test[test["mismatch"] <= -SURFACE_MISMATCH_THRESHOLD], best_d)
    report_bucket("UNAFFECTED (|mismatch| < 50) - should be numerically identical", test[test["mismatch"].abs() < SURFACE_MISMATCH_THRESHOLD], best_d)

    print(f"\n{'=' * 90}\nPer-tour held-out breakdown (affected subset only)\n{'=' * 90}")
    for tour in ("ATP", "WTA"):
        t = test[(test["tour"] == tour) & (test["mismatch"].abs() >= SURFACE_MISMATCH_THRESHOLD)]
        report_bucket(f"{tour} affected", t, best_d)


if __name__ == "__main__":
    run()
