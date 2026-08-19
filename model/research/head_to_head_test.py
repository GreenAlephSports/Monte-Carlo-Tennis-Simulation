"""Tests whether head-to-head history between two specific players predicts outperformance
relative to what their CURRENT Elo difference alone already predicts - a genuinely new signal,
not a re-measurement of the same match Elo already knows about. The candidate mechanism: a
player who beat this exact opponent the last time they met might have a persistent stylistic
edge (a bad matchup for the other player) that a single aggregate Elo number doesn't capture.

Definition: a "rematch" is any match where this exact pair has met at least once before,
anywhere in the dataset. The "h2h favorite" for a rematch is whichever player won their most
recent prior meeting. The test: does the h2h favorite win the rematch MORE often than Elo alone
would predict? Conditioned on how long ago that prior meeting was (a stale result from 4 years
ago plausibly carries less signal than one from 3 months ago) and whether it was on the same
surface (a hard-court beatdown says less about a clay rematch).

Same rigor as every other test in this repo: frozen per-tournament-edition Elo (reuses
elite_opponent_residual_test.build_frozen_predictions - no in-tournament lookahead, no
mid-tournament rating movement), chronological tournament-level 80/20 train/test split, held-out
validation (log-loss/Brier, raw Elo vs. h2h-adjusted), player-clustered bootstrap CI.

Stated up front: most pairs in the dataset meet 0 or 1 times, so the population of ELIGIBLE
rematch rows is a small fraction of all matches, and the population of rematches that ALSO fall
in the held-out test era is smaller still. This script reports those counts honestly at every
stage rather than presenting only whichever cut happens to look significant.

Usage:
    python model/research/head_to_head_test.py [--tour ATP|WTA]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import (  # noqa: E402
    EPS, TRAIN_FRACTION, build_frozen_predictions, log_loss, logit, sigmoid,
)
from elo_ratings import load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

RECENCY_BUCKETS = [
    ("within_180d", lambda d: d <= 180),
    ("181d_to_720d", lambda d: 180 < d <= 720),
    ("over_720d", lambda d: d > 720),
]
MIN_BUCKET_TEST_ROWS = 20  # below this, a per-bucket held-out fit is noise, not a finding


def recency_bucket(days):
    for name, test in RECENCY_BUCKETS:
        if test(days):
            return name
    raise ValueError(days)


def build_h2h_dataset(matches, preds):
    """One row per rematch, from the perspective of the player who won the PRIOR meeting (the
    'h2h favorite'). The mirror-image row (their opponent's perspective) is redundant - actual_win
    and pred_win are complements of each other - so only one row per rematch is kept, not two."""
    m = matches.copy()
    m["edition_id"] = m["Tournament"] + " " + m["Date"].dt.year.astype(str)
    m = m.sort_values("Date", kind="stable")

    perspective = {
        (row.edition_id, row.date, row.round, row.player, row.opponent): row
        for row in preds.itertuples(index=False)
    }

    pair_history = defaultdict(list)  # pair_key -> chronological list of {date, surface, winner}
    rows = []
    for row in m.itertuples(index=False):
        p1, p2, winner = row.Player_1, row.Player_2, row.Winner
        pair_key = tuple(sorted((p1, p2)))
        history = pair_history[pair_key]
        if history:
            prior = history[-1]
            h2h_favorite = prior["winner"]
            other = p2 if h2h_favorite == p1 else p1
            key = (row.edition_id, row.Date, row.Round, h2h_favorite, other)
            pr = perspective.get(key)
            if pr is not None:
                days_since = (row.Date - prior["date"]).days
                rows.append((
                    row.edition_id, row.Date, row.Round, h2h_favorite, other,
                    pr.player_elo, pr.opponent_elo, pr.pred_win, pr.actual_win,
                    days_since, row.Surface == prior["surface"], len(history), pair_key,
                ))
        history.append({"date": row.Date, "surface": row.Surface, "winner": winner})

    h2h_df = pd.DataFrame(rows, columns=[
        "edition_id", "date", "round", "player", "opponent", "player_elo", "opponent_elo",
        "pred_win", "actual_win", "days_since_prior", "same_surface", "n_prior_meetings", "pair_key",
    ])
    h2h_df["recency_bucket"] = h2h_df["days_since_prior"].map(recency_bucket)
    return h2h_df


def fit_shift(train_g):
    actual_rate, pred_rate = train_g["actual_win"].mean(), train_g["pred_win"].mean()
    return logit(actual_rate) - logit(pred_rate), actual_rate, pred_rate


def held_out_report(label, train_g, test_g):
    if len(train_g) == 0 or len(test_g) < MIN_BUCKET_TEST_ROWS:
        print(f"\n{label}: {len(train_g)} train rows, {len(test_g)} test rows - "
              f"below the {MIN_BUCKET_TEST_ROWS}-row test-era floor, skipping held-out fit "
              f"(would just be fitting noise)")
        return None

    shift, actual_rate, pred_rate = fit_shift(train_g)
    test_g = test_g.copy()
    test_g["adjusted_pred"] = test_g["pred_win"].apply(lambda p: sigmoid(logit(p) + shift))
    test_g["raw_loss"] = log_loss(test_g["actual_win"].values, test_g["pred_win"].values)
    test_g["adj_loss"] = log_loss(test_g["actual_win"].values, test_g["adjusted_pred"].values)
    test_g["raw_brier"] = (test_g["actual_win"] - test_g["pred_win"]) ** 2
    test_g["adj_brier"] = (test_g["actual_win"] - test_g["adjusted_pred"]) ** 2

    observed, lo, hi = cluster_bootstrap_ci(test_g, "raw_loss", "adj_loss")
    print(f"\n{label}: train n={len(train_g)} (actual {actual_rate:.1%} vs. Elo-predicted "
          f"{pred_rate:.1%}, fitted shift {shift:+.4f} logits); test n={len(test_g)}, "
          f"{test_g['player'].nunique()} distinct h2h-favorite players")
    print(f"  Raw Elo      : log-loss = {test_g['raw_loss'].mean():.4f}, Brier = {test_g['raw_brier'].mean():.4f}")
    print(f"  H2H-adjusted : log-loss = {test_g['adj_loss'].mean():.4f}, Brier = {test_g['adj_brier'].mean():.4f}")
    print(f"  Mean per-match log-loss improvement (raw - adjusted, >0 = h2h signal helps), "
          f"player-clustered: {observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    return observed, lo, hi


def run(tour):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])

    h2h_df = build_h2h_dataset(matches, preds)
    train = h2h_df[h2h_df["edition_id"].isin(train_editions)]
    test = h2h_df[h2h_df["edition_id"].isin(test_editions)]

    distinct_pairs = h2h_df["pair_key"].nunique()
    meetings_per_pair = h2h_df.groupby("pair_key").size()
    print(f"{tour}: {len(matches)} total matches -> {len(h2h_df)} rematch rows (a real prior "
          f"meeting exists), {len(train)} train-era / {len(test)} test-era")
    print(f"Distinct pairs that generated at least one rematch: {distinct_pairs} "
          f"(median {meetings_per_pair.median():.0f} rematches/pair, "
          f"max {meetings_per_pair.max()}, {(meetings_per_pair == 1).sum()} pairs with exactly 1)")
    print(f"Sanity check on thinness: {h2h_df['player'].nunique()} distinct players ever appear as "
          f"the h2h favorite, vs. {matches['Player_1'].nunique()} total distinct players in the dataset")

    # 1. Overall: does the h2h favorite (winner of the most recent prior meeting) beat Elo's
    # prediction in the rematch, full stop, no conditioning?
    held_out_report("Overall (any rematch, no recency/surface conditioning)", train, test)

    # train-era descriptive residual, unconditional
    actual_rate, pred_rate = train["actual_win"].mean(), train["pred_win"].mean()
    print(f"\nTrain-era descriptive residual (all rematches pooled): actual {actual_rate:.1%} vs. "
          f"Elo-predicted {pred_rate:.1%} (gap = {actual_rate - pred_rate:+.1%}, n={len(train)})")

    # 2. Recency conditioning
    print("\n--- Recency-conditioned (train-era descriptive) ---")
    recency_order = [b[0] for b in RECENCY_BUCKETS]
    recency_summary = []
    for b in recency_order:
        g = train[train["recency_bucket"] == b]
        if len(g) == 0:
            continue
        ar, pr = g["actual_win"].mean(), g["pred_win"].mean()
        recency_summary.append({"bucket": b, "n": len(g), "actual_rate": ar, "pred_rate": pr, "residual": ar - pr})
    recency_summary_df = pd.DataFrame(recency_summary)
    if len(recency_summary_df):
        print(recency_summary_df.to_string(index=False, formatters={
            "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format, "residual": "{:+.1%}".format,
        }))
        is_decaying = recency_summary_df["residual"].is_monotonic_decreasing
        print(f"Does the residual shrink monotonically as the prior meeting gets staler "
              f"(within_180d -> 181d_to_720d -> over_720d), as a 'fresher h2h signal = more "
              f"predictive' story would predict? {'YES' if is_decaying else 'NO'} "
              f"({', '.join(f'{r:+.1%}' for r in recency_summary_df['residual'])})")

    print("\n--- Recency-conditioned held-out validation (only where the test-era bucket clears "
          f"the {MIN_BUCKET_TEST_ROWS}-row floor) ---")
    for b in recency_order:
        held_out_report(f"Recency bucket: {b}", train[train["recency_bucket"] == b], test[test["recency_bucket"] == b])

    # 3. Surface conditioning
    print("\n--- Surface-conditioned (train-era descriptive) ---")
    surface_summary = []
    for same_surface in (True, False):
        g = train[train["same_surface"] == same_surface]
        if len(g) == 0:
            continue
        ar, pr = g["actual_win"].mean(), g["pred_win"].mean()
        surface_summary.append({
            "same_surface": same_surface, "n": len(g), "actual_rate": ar, "pred_rate": pr, "residual": ar - pr,
        })
    surface_summary_df = pd.DataFrame(surface_summary)
    if len(surface_summary_df):
        print(surface_summary_df.to_string(index=False, formatters={
            "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format, "residual": "{:+.1%}".format,
        }))

    print("\n--- Surface-conditioned held-out validation ---")
    for same_surface in (True, False):
        label = f"Surface match: {'same surface as prior meeting' if same_surface else 'different surface'}"
        held_out_report(label, train[train["same_surface"] == same_surface], test[test["same_surface"] == same_surface])

    # 4. Best-case cut: recent AND same surface - the strongest form of the hypothesis, if the
    # sample can even support testing it
    strong = train[(train["recency_bucket"] == "within_180d") & (train["same_surface"])]
    strong_test = test[(test["recency_bucket"] == "within_180d") & (test["same_surface"])]
    held_out_report("Strongest cut: prior meeting <=180 days ago AND same surface", strong, strong_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
    args = parser.parse_args()
    run(args.tour)
