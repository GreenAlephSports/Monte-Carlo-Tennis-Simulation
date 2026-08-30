"""Tests a mechanically distinct hypothesis from every mechanism closed tonight: does the
OFFICIAL WTA/ATP ranking-points trajectory over a recent rolling window (rate of change, not
level) carry predictive signal beyond what a continuously-updated Elo rating already captures?

This is NOT another Elo-recency/lookback variant (hard3/decay3, both already closed - decay3
ACCEPTED for WTA in decay3_full_historical_test.py) and NOT the static rank-vs-Elo level gap
(rank_trajectory_lag_test.py, NOT significant). Ranking points and Elo are two independent scoring
systems built on fundamentally different logic: Elo is a smoothed, symmetric, match-by-match
update (K_FACTOR points win or lose, same magnitude regardless of tournament importance) averaged
over a multi-year window; the tour's own point system is a rolling 52-week SUM of tournament-tier-
weighted results with full credit the moment a result happens and zero smoothing - a single deep
run at a big event can move it sharply in a matter of weeks. If the market anchors more to "how
many ranking points has this player racked up lately" than to a slow-moving Elo average, that gap
would show up here as real, held-out predictive signal in points_trajectory that Elo doesn't have
- a genuinely new, checkable variable, not a re-parameterization of anything already tested.

Points trajectory, defined precisely: for a given player-perspective row (a real match, as of that
match's date), log(most recent known ranking points at/before this match) minus log(most recent
known ranking points at/before [this match's date - WINDOW]). Log-scale specifically so the signal
is a relative (percent) change, comparable across eras/players with wildly different point totals
(a top-3 player's point scale vs. a Q-round player's) rather than a raw point-count delta that
would be dominated by the biggest-point players regardless of real momentum. Undefined (NaN,
dropped) unless the player has a real points reading from at least WINDOW days before this match -
never imputed to zero, since "just turned pro, no ranking history that far back" and "genuinely flat
lately" are different situations. Primary window = 70 days (10 weeks), 84 days (12 weeks) as a
robustness check, per the two concrete windows this hypothesis was framed around.

Methodology (same rigor as every other correction tested tonight):
  - Frozen per-tournament-edition Elo predictions (elite_opponent_residual_test.build_frozen_
    predictions, reused unchanged - the same single continuously-updated overall_elo baseline
    recent_form_test.py and veteran_decline_test.py already use as their Elo reference point).
  - Points-trajectory itself is computed from real historical Rank_1/Pts_1/Rank_2/Pts_2 fact-
    lookups (a player's own officially-reported points are known publicly before their next match
    starts - no lookahead risk, same as how Rank_1/Rank_2 are already used unguarded elsewhere in
    this series, e.g. signature_win_boost_test.py, thin_history_rank_blend_test.py).
  - Chronological tournament-edition 80/20 train/test split, computed per tour then combined for
    the headline (recent_form_test.py's exact convention).
  - Primary test: single fitted continuous logistic-regression coefficient beta on
    adjusted_logit = logit(pred_win) + beta * points_trajectory, fit by 1D Newton-Raphson on
    TRAIN-era rows only (recent_form_test.fit_beta_newton, reused unchanged) - the same "one
    global fitted constant" shape as every additive correction already in production. Held-out
    validation on TEST-era rows: apply the fitted beta, player-clustered bootstrap CI on the
    log-loss improvement (survivorship_upset_test.cluster_bootstrap_ci).
  - Both tours tested (ATP ranking points ARE available in the same Kaggle columns as WTA -
    confirmed directly: Pts_1/Pts_2 present in both the ATP and WTA live datasets), combined
    headline plus a per-tour breakdown to check whether this is a uniform effect or tour-specific
    (the same targeted-power discipline decay3's ATP/WTA split used).

Usage:
    python model/research/points_trajectory_test.py

FINAL VERDICT (2026-08-27): REJECTED - not added to production. Both tours combined, both windows,
full historical scale (ATP 68591 matches 2000-2026, WTA 45357 matches 2007-2026):

  window=70d (10wk): train-era beta=+0.1143, SE=0.0148, z=+7.73 (nominally significant, train-era
  only) - but held-out (45827 test-era rows, most recent 20% of editions, 903 distinct players):
  log-loss improvement +0.0001, 95% player-clustered bootstrap CI [-0.0002, +0.0003] - CI straddles
  zero, NOT validated. Per-tour: ATP +0.0002 CI [-0.0001,+0.0005], WTA -0.0002 CI [-0.0006,+0.0003]
  - neither distinguishable from baseline; WTA's point estimate is even slightly negative.

  window=84d (12wk) robustness check: same pattern, train z=+7.57, held-out improvement +0.0001 CI
  [-0.0002, +0.0003] - NOT validated. Per-tour: ATP +0.0002 CI [-0.0001,+0.0005], WTA -0.0001 CI
  [-0.0005,+0.0003] - both still not distinguishable.

  The descriptive tercile breakdown shows the correlation is real and in the expected direction
  (bottom tercile, falling >5%/70d: 48.6% actual win rate vs 49.3% assigned; top tercile, rising
  >10%/70d: 51.7% actual vs 50.6% assigned - a real +1.1pp gap) - so a rising points trajectory DOES
  associate with modest outperformance in the raw data. But the same pattern that sank hard3, the
  adaptive-K formula, and four of the six original lookback/recency mechanisms shows up again here:
  a train-era-significant coefficient (z=7.7, p<<0.001) that does not survive out-of-sample
  validation once measured on data the beta was never fit to. Points trajectory over a 10-12 week
  window is likely capturing much of the same signal recent_form_residual (also 10 matches, already
  ACCEPTED and live in production) already captures - a player racking up points is, almost by
  definition, a player who has recently been winning matches Elo already saw, so it is not
  surprising the correlation exists but the incremental information beyond Elo+recent_form is not
  demonstrated. Conclusion: mechanically distinct hypothesis, cleanly tested, genuinely rejected
  on the evidence - not a re-run of anything closed tonight, but the market/model gap is NOT
  explained by this mechanism specifically. No production change made.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import EPS, build_frozen_predictions, log_loss, logit, sigmoid  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from recent_form_test import fit_beta_newton  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

TRAIN_FRACTION = 0.8
PRIMARY_WINDOW_DAYS = 70   # 10 weeks
ROBUST_WINDOW_DAYS = 84    # 12 weeks


def build_points_history(df):
    """Long-format (player, date, points) event history built directly from real Pts_1/Pts_2
    readings - independent of the frozen-Elo mechanism, since a player's own official ranking
    points are a public fact known before their next match, not something that needs an
    edition-boundary snapshot to avoid lookahead."""
    events = pd.concat([
        df[["Player_1", "Date", "Pts_1"]].rename(columns={"Player_1": "player", "Date": "date", "Pts_1": "points"}),
        df[["Player_2", "Date", "Pts_2"]].rename(columns={"Player_2": "player", "Date": "date", "Pts_2": "points"}),
    ], ignore_index=True)
    events = events.dropna(subset=["points", "date"])
    events = events[events["points"] > 0]
    events = events.sort_values(["player", "date"], kind="stable")

    history = {}
    for player, g in events.groupby("player", sort=False):
        history[player] = (g["date"].values.astype("datetime64[ns]"), g["points"].values.astype(float))
    return history


def points_asof(history, player, date):
    """Most recent known points reading at or before `date`, or None if the player has no
    reading that far back."""
    entry = history.get(player)
    if entry is None:
        return None
    dates, points = entry
    idx = np.searchsorted(dates, np.datetime64(date), side="right") - 1
    if idx < 0:
        return None
    return points[idx]


def add_points_trajectory(preds, history, window_days):
    """log(points now) - log(points as of window_days earlier), per player-perspective row.
    NaN (dropped downstream) unless both readings exist and are positive."""
    offset = pd.Timedelta(days=window_days)
    trajectories = np.full(len(preds), np.nan)
    dates = preds["date"].values
    players = preds["player"].values
    for i in range(len(preds)):
        now = points_asof(history, players[i], dates[i])
        before = points_asof(history, players[i], pd.Timestamp(dates[i]) - offset)
        if now is not None and before is not None and now > 0 and before > 0:
            trajectories[i] = math.log(now) - math.log(before)
    preds = preds.copy()
    preds["points_trajectory"] = trajectories
    return preds


def build_tour_data(tour):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)
    preds["tour"] = tour
    history = build_points_history(matches)
    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    print(f"{tour}: {len(editions)} editions ({editions['edition_start'].min().date()} to "
          f"{editions['edition_start'].max().date()}); train = first {len(train_editions)}, "
          f"test = most recent {len(test_editions)} (from "
          f"{editions['edition_start'].iloc[split_idx].date()})")
    return preds, history, train_editions, test_editions


def run_for_window(window_days, all_data):
    print(f"\n{'#' * 90}\nWINDOW = last {window_days} days ({window_days / 7:.1f} weeks)\n{'#' * 90}")

    tour_frames = {}
    for tour, (preds, history, train_editions, test_editions) in all_data.items():
        p = add_points_trajectory(preds, history, window_days)
        p = p[p["points_trajectory"].notna()].copy()
        tour_frames[tour] = (p, train_editions, test_editions)
        print(f"  {tour}: {len(p)} of {len(preds)} rows have a defined points_trajectory "
              f"(both a now-reading and a >= {window_days}d-earlier reading)")

    train = pd.concat([p[p["edition_id"].isin(te)] for p, te, _ in tour_frames.values()], ignore_index=True)
    test = pd.concat([p[p["edition_id"].isin(tt)] for p, _, tt in tour_frames.values()], ignore_index=True)
    print(f"\nCombined (both tours): {len(train)} train-era rows, {len(test)} test-era rows")

    offset = train["pred_win"].apply(logit).values
    x = train["points_trajectory"].values
    y = train["actual_win"].values
    beta, se = fit_beta_newton(offset, x, y)
    z = beta / se if se == se and se != 0 else float("nan")
    print(f"\nTrain-era fitted beta (adjusted_logit = logit(pred_win) + beta * points_trajectory): "
          f"{beta:+.4f} (SE={se:.4f}, z={z:+.2f}, "
          f"{'|z|>1.96, nominally significant' if abs(z) > 1.96 else 'not significant on its own'})")

    test = test.copy()
    test["adjusted_pred"] = test.apply(
        lambda r: sigmoid(logit(r["pred_win"]) + beta * r["points_trajectory"]), axis=1)
    test["raw_loss"] = log_loss(test["actual_win"].values, test["pred_win"].values)
    test["adj_loss"] = log_loss(test["actual_win"].values, test["adjusted_pred"].values)

    print(f"\nHeld-out test era, BOTH TOURS COMBINED ({len(test)} rows, {test['player'].nunique()} distinct players):")
    print(f"  Raw Elo              : mean log-loss = {test['raw_loss'].mean():.4f}")
    print(f"  Trajectory-adjusted  : mean log-loss = {test['adj_loss'].mean():.4f}")

    observed, lo, hi = cluster_bootstrap_ci(test, "raw_loss", "adj_loss", group_col="player")
    ci_excludes_zero = lo > 0 or hi < 0
    print(f"  Mean per-match log-loss improvement (raw - adjusted, >0 = adjustment better), "
          f"player-clustered: {observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    combined_verdict = (
        "VALIDATED (beats baseline, held out)" if ci_excludes_zero and lo > 0
        else ("WORSE than baseline, held out" if hi < 0 else "NOT validated (CI straddles zero)"))
    print(f"\n  VERDICT @ window={window_days}d: train z={z:+.2f} "
          f"({'nominally significant' if abs(z) > 1.96 else 'not significant on its own'}); "
          f"held-out -> {combined_verdict}")

    # --- per-tour breakdown, held-out era, same fitted beta (checks whether the combined result
    # is a uniform effect across tours or carried by one)
    print(f"\n--- Per-tour breakdown, held-out test era (same train-fitted beta applied to each) ---")
    per_tour_results = {}
    for tour in tour_frames:
        t = test[test["tour"] == tour]
        if len(t) < 10:
            print(f"  {tour}: only {len(t)} held-out rows - too few to bootstrap")
            continue
        obs_t, lo_t, hi_t = cluster_bootstrap_ci(t, "raw_loss", "adj_loss", group_col="player")
        verdict_t = "BEATS baseline" if lo_t > 0 else ("WORSE than baseline" if hi_t < 0 else "not distinguishable")
        print(f"  {tour}: n={len(t)}, improvement {obs_t:+.4f}, 95% CI [{lo_t:+.4f}, {hi_t:+.4f}] -> {verdict_t}")
        per_tour_results[tour] = (obs_t, lo_t, hi_t)

    # --- descriptive hot/cold tercile breakdown, held-out era, for interpretability only
    q1, q2 = test["points_trajectory"].quantile([1 / 3, 2 / 3])
    test["traj_bucket"] = np.select(
        [test["points_trajectory"] <= q1, test["points_trajectory"] >= q2],
        ["falling (bottom tercile)", "rising (top tercile)"], default="flat (middle tercile)")
    print(f"\nDescriptive breakdown, held-out test era, terciles by points_trajectory "
          f"(cutpoints {q1:+.3f} / {q2:+.3f} log-points over {window_days}d):")
    desc = test.groupby("traj_bucket").agg(
        n=("actual_win", "size"),
        assigned=("pred_win", "mean"),
        actual=("actual_win", "mean"),
        mean_points_trajectory=("points_trajectory", "mean"),
    ).reindex(["falling (bottom tercile)", "flat (middle tercile)", "rising (top tercile)"])
    desc["gap"] = desc["assigned"] - desc["actual"]
    print(desc.to_string(formatters={
        "assigned": "{:.1%}".format, "actual": "{:.1%}".format, "gap": "{:+.1%}".format,
        "mean_points_trajectory": "{:+.3f}".format,
    }))

    return beta, z, observed, lo, hi, per_tour_results


def run():
    all_data = {}
    for tour in ["ATP", "WTA"]:
        all_data[tour] = build_tour_data(tour)

    results = {}
    for window_days in [PRIMARY_WINDOW_DAYS, ROBUST_WINDOW_DAYS]:
        results[window_days] = run_for_window(window_days, all_data)

    print(f"\n{'=' * 90}\nSUMMARY ACROSS WINDOWS\n{'=' * 90}")
    for window_days, (beta, z, observed, lo, hi, per_tour) in results.items():
        ci_excludes_zero = lo > 0 or hi < 0
        print(f"  window={window_days:>3}d: beta={beta:+.4f} (train z={z:+.2f}), held-out improvement "
              f"{observed:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] -> "
              f"{'VALIDATED' if ci_excludes_zero and lo > 0 else ('WORSE' if hi < 0 else 'NOT validated')}")
        for tour, (obs_t, lo_t, hi_t) in per_tour.items():
            v_t = "BEATS" if lo_t > 0 else ("WORSE" if hi_t < 0 else "n/a")
            print(f"      {tour}: {obs_t:+.4f} CI [{lo_t:+.4f}, {hi_t:+.4f}] -> {v_t}")


if __name__ == "__main__":
    run()
