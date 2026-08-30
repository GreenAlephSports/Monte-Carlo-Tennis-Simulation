"""Tests the "capitulator vs. big-game hunter" hypothesis: does a player's own historical
performance against elite (top-ranked) opponents deviate from what their Elo predicts, in a way
that's stable enough to improve predictions for that SAME player's future matches against elite
opponents - beyond what raw Elo alone already knows?

Same rigor as the rank-gap/Platt-scaling backtests in win_probability.py's own module docstring
comments: frozen per-tournament-edition Elo (every match in a tournament is scored off the SAME
pre-tournament rating - no in-tournament lookahead), a chronological tournament-level 80/20
train/test split (never a random match-level split, which would leak a player's own future form
into their own training residual), and held-out validation of the adjusted vs. raw prediction.

Methodology notes / simplifications (stated up front, not buried):
  - Uses a single continuously-updated overall_elo (not the pipeline's surface-specific Elo, and
    not recomputed with the pipeline's 5-year lookback window per tournament) - recomputing
    per-edition with the full windowing logic is O(editions x window matches) and doesn't finish
    in reasonable time; a single online pass is the standard simplification and shouldn't matter
    for a career-long "does this player run hot/cold vs elite opponents" question, since the
    lookback window mostly exists to drop long-inactive players, not to change matchup dynamics.
  - "Elite opponent" = opponent's real ATP ranking, as of the tournament's start (last known
    Rank_1/Rank_2 from any earlier match), was <= ELITE_RANK_THRESHOLD. Deliberately NOT gated on
    the player's OWN rank - this captures both "top player beats another top player" and
    "underdog upsets/loses to a top player", which is the actual population the hypothesis is
    about.
  - Per-player residual = (actual win rate vs. elite opponents) - (Elo's average predicted win
    rate for those same matches), computed on TRAIN-era matches only, then converted to a single
    logit-space shift and applied to that player's TEST-era elite-opponent predictions - additive
    in logit space for the same reason the codebase's own rank-gap adjustment is (keeps adjusted
    probabilities in [0,1] and composes cleanly with an already-extreme raw prediction).

Usage:
    python model/elite_opponent_residual_test.py [--tour ATP|WTA] [--elite-rank N] [--min-train N]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elo_ratings import K_FACTOR, STARTING_ELO, expected_score, load_matches_for_tour  # noqa: E402

TRAIN_FRACTION = 0.8
MIN_TEST_ELITE_MATCHES = 5
EPS = 1e-3


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def log_loss(actual, pred):
    pred = np.clip(pred, EPS, 1 - EPS)
    return -(actual * np.log(pred) + (1 - actual) * np.log(1 - pred))


def build_frozen_predictions(df, max_editions=None):
    """One player-perspective row per (match, player). player_elo/opponent_elo/pred_win are
    frozen at the tournament EDITION's start - identical for every round of that edition, exactly
    like win_probability() sees a fixed pre-tournament rating for every simulated round. Elo/rank
    are then updated with the edition's real results before moving to the next (chronologically
    later) edition.

    max_editions: if set, restricts the RETURNED population to the most recent max_editions
    editions (chronologically last) - a QUICK-CHECK mode for fast iteration while developing a new
    test, not a substitute for a full-historical run. The warm-up walk-forward still replays the
    FULL history first (this function's own loop is cheap regardless of max_editions - the real
    per-edition cost lives in windowed-rebuild-style builds like tiered_recency_elo_test.py's, not
    here), so player_elo/matches_played/current_rank at the start of the kept window are exactly
    what a full run would show, not reset to STARTING_ELO/zero. Truncating the INPUT df instead
    (an earlier version of this parameter did that) silently breaks any population defined by
    accumulated history - e.g. a "solid player, >=30 career matches" filter can never be satisfied
    inside a small recent-only window built from a cold start, a real failure caught while quick-
    checking rank_trajectory_lag_test.py (0 solid-player rows at max_editions=20 under the old
    truncate-first design). Never use a max_editions result as a final verdict; see this project's
    own K-factor small-scale-vs-full-scale history (Cincinnati-alone) for why a small, recent-only
    sample has fooled this project before."""
    df = df.copy()
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)
    edition_start = df.groupby("edition_id")["Date"].transform("min")
    editions = (
        df.assign(edition_start=edition_start)[["edition_id", "edition_start"]]
        .drop_duplicates()
        .sort_values("edition_start")
        .reset_index(drop=True)
    )

    overall_elo, current_rank = {}, {}
    rows = []
    for edition_id in editions["edition_id"]:
        edition_matches = df[df["edition_id"] == edition_id]

        snapshot_elo = dict(overall_elo)
        snapshot_rank = dict(current_rank)
        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            elo_p1 = snapshot_elo.get(p1, STARTING_ELO)
            elo_p2 = snapshot_elo.get(p2, STARTING_ELO)
            pred_p1 = expected_score(elo_p1, elo_p2)
            win1 = 1 if winner == p1 else 0
            rows.append((edition_id, row.Date, row.Round, p1, p2, elo_p1, elo_p2, pred_p1, win1, snapshot_rank.get(p2)))
            rows.append((edition_id, row.Date, row.Round, p2, p1, elo_p2, elo_p1, 1 - pred_p1, 1 - win1, snapshot_rank.get(p1)))

        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            overall_elo.setdefault(p1, STARTING_ELO)
            overall_elo.setdefault(p2, STARTING_ELO)
            score1 = 1.0 if winner == p1 else 0.0
            exp1 = expected_score(overall_elo[p1], overall_elo[p2])
            overall_elo[p1] += K_FACTOR * (score1 - exp1)
            overall_elo[p2] += K_FACTOR * ((1 - score1) - (1 - exp1))
            if pd.notna(row.Rank_1) and row.Rank_1 > 0:
                current_rank[p1] = row.Rank_1
            if pd.notna(row.Rank_2) and row.Rank_2 > 0:
                current_rank[p2] = row.Rank_2

    preds = pd.DataFrame(rows, columns=[
        "edition_id", "date", "round", "player", "opponent", "player_elo", "opponent_elo",
        "pred_win", "actual_win", "opponent_rank",
    ])
    if max_editions is not None:
        editions = editions.tail(max_editions).reset_index(drop=True)
        keep = set(editions["edition_id"])
        preds = preds[preds["edition_id"].isin(keep)].reset_index(drop=True)
    return preds, editions


def run(tour, elite_rank_threshold, min_train_matches, max_editions=None):
    if max_editions is not None:
        print(f"*** QUICK CHECK ON RECENT DATA ONLY (--max-editions {max_editions}) - NOT the "
              f"full-historical verdict. Rerun without --max-editions before trusting this. ***\n")
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches, max_editions=max_editions)

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    print(f"{tour}: {len(editions)} tournament editions "
          f"({editions['edition_start'].min().date()} to {editions['edition_start'].max().date()}); "
          f"train = first {len(train_editions)} editions (through "
          f"{editions['edition_start'].iloc[split_idx - 1].date()}), "
          f"test = remaining {len(test_editions)} editions "
          f"(from {editions['edition_start'].iloc[split_idx].date()})")

    elite = preds[preds["opponent_rank"].notna() & (preds["opponent_rank"] <= elite_rank_threshold)].copy()
    train_elite = elite[elite["edition_id"].isin(train_editions)]
    test_elite = elite[elite["edition_id"].isin(test_editions)]
    print(f"Elite-opponent matches (opponent ranked <= {elite_rank_threshold} at tournament start): "
          f"{len(train_elite)} train-era player-perspective rows, {len(test_elite)} test-era rows")

    # population-level baseline: averaged over EVERY player, does facing an elite opponent see
    # real outcomes deviate systematically from what Elo predicts? (context, not the main test)
    pop_actual, pop_pred = train_elite["actual_win"].mean(), train_elite["pred_win"].mean()
    print(f"\nPopulation baseline (train era, all players pooled): actual win rate vs. elite "
          f"opponents = {pop_actual:.1%}, Elo's average predicted rate = {pop_pred:.1%} "
          f"(gap = {pop_actual - pop_pred:+.1%}, n={len(train_elite)})")

    train_stats = train_elite.groupby("player").agg(
        n_train=("actual_win", "size"), actual_rate=("actual_win", "mean"), pred_rate=("pred_win", "mean"),
    ).reset_index()
    train_stats["residual"] = train_stats["actual_rate"] - train_stats["pred_rate"]
    train_stats["logit_shift"] = train_stats.apply(
        lambda r: logit(r["actual_rate"]) - logit(r["pred_rate"]), axis=1)

    eligible = train_stats[train_stats["n_train"] >= min_train_matches].copy()
    print(f"\nPlayers with >= {min_train_matches} elite-opponent matches in the train era "
          f"(a real per-player estimate is even attemptable): {len(eligible)} of "
          f"{train_stats['player'].nunique()} players who faced an elite opponent at all")

    # heterogeneity check: is the spread of per-player residuals bigger than sampling noise alone
    # would produce if every player secretly had the SAME true vs.-elite performance as their own
    # Elo predicts? (standardized residual z_i, then a chi-square goodness-of-fit test)
    def z_score(row):
        p = min(max(row["pred_rate"], EPS), 1 - EPS)
        se = math.sqrt(p * (1 - p) / row["n_train"])
        return row["residual"] / se

    eligible["z"] = eligible.apply(z_score, axis=1)
    chi2_stat = (eligible["z"] ** 2).sum()
    df_chi2 = len(eligible) - 1
    chi2_p = stats.chi2.sf(chi2_stat, df_chi2) if df_chi2 > 0 else float("nan")
    n_extreme = (eligible["z"].abs() > 1.96).sum()
    print(f"Heterogeneity check: {n_extreme}/{len(eligible)} eligible players have |z| > 1.96 "
          f"(~{n_extreme / len(eligible):.0%}, vs. ~5% expected under 'everyone matches their own "
          f"Elo vs. elite opponents'); chi-square goodness-of-fit stat = {chi2_stat:.1f} on "
          f"{df_chi2} df, p = {chi2_p:.2g}")

    # held-out test: for eligible players only, does applying their train-era logit shift to
    # test-era elite-opponent matches beat raw (unadjusted) Elo?
    shift_by_player = dict(zip(eligible["player"], eligible["logit_shift"]))
    test_eligible = test_elite[test_elite["player"].isin(shift_by_player)].copy()
    test_eligible["adjusted_pred"] = test_eligible.apply(
        lambda r: sigmoid(logit(r["pred_win"]) + shift_by_player[r["player"]]), axis=1)
    test_eligible["raw_loss"] = log_loss(test_eligible["actual_win"].values, test_eligible["pred_win"].values)
    test_eligible["adj_loss"] = log_loss(test_eligible["actual_win"].values, test_eligible["adjusted_pred"].values)
    test_eligible["raw_brier"] = (test_eligible["actual_win"] - test_eligible["pred_win"]) ** 2
    test_eligible["adj_brier"] = (test_eligible["actual_win"] - test_eligible["adjusted_pred"]) ** 2

    n_test_players = test_eligible["player"].nunique()
    print(f"\nHeld-out validation: {len(test_eligible)} test-era elite-opponent rows belonging to "
          f"eligible players, across {n_test_players} distinct players who actually faced an "
          f"elite opponent again in the test era")
    if len(test_eligible):
        print(f"  Raw Elo       : log-loss = {test_eligible['raw_loss'].mean():.4f}, "
              f"Brier = {test_eligible['raw_brier'].mean():.4f}")
        print(f"  Player-adjusted: log-loss = {test_eligible['adj_loss'].mean():.4f}, "
              f"Brier = {test_eligible['adj_brier'].mean():.4f}")

        # cluster (by player) bootstrap on the per-match log-loss improvement, so one prolific
        # player's matches don't count as independent evidence 40 times over
        rng = np.random.default_rng(42)
        players = test_eligible["player"].unique()
        per_player_delta = test_eligible.groupby("player").apply(
            lambda g: (g["raw_loss"] - g["adj_loss"]).mean(), include_groups=False)
        boot_means = [
            per_player_delta.loc[rng.choice(players, size=len(players), replace=True)].mean()
            for _ in range(5000)
        ]
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        observed = per_player_delta.mean()
        print(f"  Mean per-match log-loss improvement from the adjustment (raw - adjusted, "
              f">0 = adjustment better), player-clustered: {observed:+.4f}, "
              f"95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")

    # per-player reliability table
    test_stats = test_elite.groupby("player").agg(n_test=("actual_win", "size")).reset_index()
    report = eligible.merge(test_stats, on="player", how="left")
    report["n_test"] = report["n_test"].fillna(0).astype(int)
    report["reliability"] = np.select(
        [report["n_train"] >= 30, report["n_train"] >= min_train_matches],
        ["solid", "thin"], default="excluded",
    )
    report = report.sort_values("residual", key=abs, ascending=False)

    print(f"\nPer-player residual table, train era, sorted by |residual| "
          f"(n_train >= {min_train_matches} shown; 'solid' = n_train >= 30, else 'thin'):")
    print(report[["player", "n_train", "actual_rate", "pred_rate", "residual", "n_test", "reliability"]]
          .to_string(index=False, formatters={
              "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format, "residual": "{:+.1%}".format,
          }))

    validated = report[report["n_test"] >= MIN_TEST_ELITE_MATCHES]
    print(f"\n{len(validated)} of those players ALSO have >= {MIN_TEST_ELITE_MATCHES} elite-opponent "
          f"matches in the held-out test era - the only players for whom this analysis can claim any "
          f"real look at whether their train-era residual actually generalized, as opposed to just "
          f"having a train-era estimate at all:")
    print(validated[["player", "n_train", "residual", "n_test"]].to_string(
        index=False, formatters={"residual": "{:+.1%}".format}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
    parser.add_argument("--elite-rank", type=int, default=20, help="opponent rank threshold defining 'elite'")
    parser.add_argument("--min-train", type=int, default=15, help="min train-era elite-opponent matches to estimate a per-player residual at all")
    parser.add_argument("--max-editions", type=int, default=None,
                         help="quick-check mode: only replay the most recent N tournament editions "
                              "(before the 80/20 split), instead of the full lookback window")
    args = parser.parse_args()
    run(args.tour, args.elite_rank, args.min_train, max_editions=args.max_editions)
