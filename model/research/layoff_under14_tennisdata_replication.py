"""Independent replication of layoff_under14_cross_tournament_refit.py's finding, using a
DIFFERENT data vendor (tennis-data.co.uk season archives, already fetched tonight for the injury-
retirement test - data/tennis_data_seasons/, both tours, 2013-2026) rather than pooling it with the
Kaggle-based fit, which would just double-count many of the same real matches under a different
vendor's formatting and artificially deflate the confidence intervals without adding real
information. This is a genuine out-of-sample check: same question (is the under_14d bucket's
validated shift diluted by same-tournament round-to-round rows vs genuine cross-tournament
turnaround?), different real-world data source, same chronological 80/20 held-out discipline.

Usage:
    python model/research/layoff_under14_tennisdata_replication.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elo_ratings import K_FACTOR, STARTING_ELO, expected_score  # noqa: E402
from injury_retirement_penalty_test import load_all  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

TRAIN_FRACTION = 0.8
EPS = 1e-3


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def log_loss(actual, pred):
    pred = np.clip(pred, EPS, 1 - EPS)
    return -(actual * np.log(pred) + (1 - actual) * np.log(1 - pred))


def bucket_for(days, same_tournament):
    if days is None or days != days:
        return "no_prior_match"
    if days < 14:
        return "under_14d_same_tourney" if same_tournament else "under_14d_cross_tourney"
    if days < 30:
        return "14_30d"
    if days < 60:
        return "30_60d"
    if days < 90:
        return "60_90d"
    return "90d_plus"


def build_frozen_predictions(df, tour):
    df = df.copy()
    df["edition_id"] = df["Tournament"].astype(str) + " " + df["Date"].dt.year.astype(str)
    edition_start = df.groupby("edition_id")["Date"].transform("min")
    editions = (
        df.assign(edition_start=edition_start)[["edition_id", "edition_start"]]
        .drop_duplicates().sort_values("edition_start").reset_index(drop=True)
    )

    overall_elo = {}
    last_match_date = {}
    last_match_edition = {}
    rows = []

    for edition_id in editions["edition_id"]:
        edition_matches = df[df["edition_id"] == edition_id].sort_values("Date", kind="stable")
        snap_elo = dict(overall_elo)  # Elo frozen for the whole edition, matches production
        # date/edition trackers update INCREMENTALLY round-by-round within this edition (unlike
        # Elo) - a player's own calendar history is a real, progressively-known fact, not
        # something that should stay frozen at the tournament's start the way Elo does. This is
        # the bug fix: the original version only committed these after the WHOLE edition finished,
        # so round 2+ of the same event always compared against the PRE-tournament last match
        # instead of the player's own most recent round of THIS event - silently making
        # same_tourney impossible to ever observe (confirmed: n=0 in the first run).
        running_date = dict(last_match_date)
        running_edition = dict(last_match_edition)

        for row in edition_matches.itertuples(index=False):
            w, l = row.Winner, row.Loser
            elo_w = snap_elo.get(w, STARTING_ELO)
            elo_l = snap_elo.get(l, STARTING_ELO)
            pred_w = expected_score(elo_w, elo_l)

            for player, pred, win in [(w, pred_w, 1), (l, 1 - pred_w, 0)]:
                prev_date = running_date.get(player)
                prev_edition = running_edition.get(player)
                days_since = (row.Date - prev_date).days if prev_date is not None else None
                same_tourney = prev_edition == edition_id
                rows.append((edition_id, row.Date, tour, player, pred, win,
                             bucket_for(days_since, same_tourney)))

            running_date[w] = row.Date
            running_date[l] = row.Date
            running_edition[w] = edition_id
            running_edition[l] = edition_id

        last_match_date = running_date
        last_match_edition = running_edition

        for row in edition_matches.itertuples(index=False):
            w, l = row.Winner, row.Loser
            overall_elo.setdefault(w, STARTING_ELO)
            overall_elo.setdefault(l, STARTING_ELO)
            exp_w = expected_score(overall_elo[w], overall_elo[l])
            overall_elo[w] += K_FACTOR * (1 - exp_w)
            overall_elo[l] += K_FACTOR * (0 - (1 - exp_w))

    preds = pd.DataFrame(rows, columns=["edition_id", "date", "tour", "player", "pred_win", "actual_win", "bucket"])
    return preds, editions


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    all_preds, editions_by_tour = [], {}
    for tour in ("ATP", "WTA"):
        raw = load_all(tour)
        preds, editions = build_frozen_predictions(raw, tour)
        all_preds.append(preds)
        editions_by_tour[tour] = editions
        print(f"{tour}: {len(raw)} real matches, {len(editions)} editions, {len(preds)} player-perspective rows")

    preds = pd.concat(all_preds, ignore_index=True)

    train_editions, test_editions = set(), set()
    for tour, editions in editions_by_tour.items():
        split_idx = int(len(editions) * TRAIN_FRACTION)
        train_editions |= set(editions["edition_id"].iloc[:split_idx])
        test_editions |= set(editions["edition_id"].iloc[split_idx:])
    train = preds[preds["edition_id"].isin(train_editions)]
    test = preds[preds["edition_id"].isin(test_editions)]
    print(f"\nTrain-era: {len(train)} rows, test-era: {len(test)} rows (both tours combined)")

    for tour in ("ATP", "WTA", "BOTH"):
        t_train = train if tour == "BOTH" else train[train["tour"] == tour]
        t_test = test if tour == "BOTH" else test[test["tour"] == tour]
        print(f"\n{'=' * 100}\n{tour}\n{'=' * 100}")
        for bucket in ["under_14d_same_tourney", "under_14d_cross_tourney"]:
            g_train = t_train[t_train["bucket"] == bucket]
            if len(g_train) < 20:
                print(f"  {bucket:<24}: n_train={len(g_train)} - too few")
                continue
            actual_rate, pred_rate = g_train["actual_win"].mean(), g_train["pred_win"].mean()
            shift = logit(actual_rate) - logit(pred_rate)

            g_test = t_test[t_test["bucket"] == bucket]
            if len(g_test) < 10:
                print(f"  {bucket:<24}: n_train={len(g_train):<6} fitted_shift={shift:+.4f}  n_test={len(g_test)} too few to validate")
                continue
            raw_loss = log_loss(g_test["actual_win"].values, g_test["pred_win"].values)
            adj_pred = g_test.apply(lambda r: sigmoid(logit(r["pred_win"]) + shift), axis=1)
            adj_loss = log_loss(g_test["actual_win"].values, adj_pred.values)
            d = g_test.assign(raw_loss=raw_loss, adj_loss=adj_loss)
            diff, lo, hi = cluster_bootstrap_ci(d, "raw_loss", "adj_loss", group_col="player")
            sig = "  <- excludes zero" if (lo > 0 or hi < 0) else ""
            print(f"  {bucket:<24}: n_train={len(g_train):<6} n_test={len(g_test):<6} fitted_shift={shift:+.4f}  "
                  f"held-out log-loss improvement={diff:+.4f} CI[{lo:+.4f},{hi:+.4f}]{sig}")


if __name__ == "__main__":
    main()
