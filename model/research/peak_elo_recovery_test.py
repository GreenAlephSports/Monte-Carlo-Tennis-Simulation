"""Tests a DISTINCT hypothesis from tonight's already-rejected comeback/layoff-lag work: not
whether current Elo updates too slowly after a break (rejected - see champion_comeback_test.py /
layoff_within_tournament_decay_test.py), but whether a SECOND number - a player's own past peak
Elo, which current Elo has fully "forgotten" once enough time/results have passed - carries real
predictive information for players who are meaningfully below that peak right now. Mechanistically
this is a "proven top-level ability, currently underrated by a single current-form snapshot"
signal, not a recency/rust signal - current_elo already reflects every real result since the peak,
so any residual benefit from adding peak_elo back in has to come from something current_elo itself
doesn't capture (a stable per-player ceiling/talent signal that a form-driven rating can undershoot
during a slump, injury spell, or - Zheng Qinwen's real case - a run of SEVERAL shorter absences
rather than one clean layoff cf. this project's own single-layoff-length correction).

Two separate peak definitions are tested independently, each against the SAME population and
split, since they are two different empirical claims:
  - career_peak: max overall_elo across every earlier tournament edition in this dataset (all-time
    high-to-date, no lookahead).
  - peak_2y: max overall_elo across earlier editions within the trailing 2 calendar years only (a
    "recent-enough peak" version - excludes a career-defining run from a decade ago that arguably
    says less about current ability than a 18-months-ago peak would).
  gap_below_peak = max(peak - current_elo, 0) for either definition - clipped at zero because the
  hypothesis is specifically about being below a past high, not about being AT a fresh career-best
  (there peak == current by construction and carries no extra information).

Coverage requirement, reported before any conclusion: a "peak" computed from a rookie's first two
tournaments isn't a meaningful ceiling estimate, so both tests are restricted to rows where the
player already has >= MIN_HISTORY_YEARS of real tracked history (their own first recorded match in
this dataset to this edition's start) - this is a real, reportable filter, not an afterthought, and
sample size after applying it is printed explicitly.

Same rigor as every other test tonight: frozen per-tournament-edition Elo (single continuously-
updated overall_elo, no windowing - same simplification build_frozen_predictions itself documents,
consistent with elite_opponent/recent_form/veteran_decline), chronological tournament-edition 80/20
train/test split, held-out validation, and player-clustered bootstrap confidence intervals. Primary
fit is a single continuous logistic coefficient beta on gap_below_peak/100 (same "one fitted global
constant" shape as RECENT_FORM_BETA/rank-gap/Platt), fit via 1D Newton-Raphson on train-era rows
only, exactly like recent_form_test.py's own beta fit - reused directly from there.

Usage:
    python model/research/peak_elo_recovery_test.py
    python model/research/peak_elo_recovery_test.py --min-history-years 2 --gap-threshold 100
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import build_frozen_predictions, log_loss, logit, sigmoid  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from recent_form_test import fit_beta_newton  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

TRAIN_FRACTION = 0.8
MIN_HISTORY_YEARS_DEFAULT = 2.0
GAP_THRESHOLD_DEFAULT = 100.0  # Elo points - "meaningfully below peak" for the descriptive breakdown
TRAILING_WINDOW_DAYS = 730


def _edition_level_elo(preds):
    """One row per (player, edition_id): that player's frozen player_elo for the edition (already
    identical across every match-row of theirs within the edition) plus the edition's start date -
    the actual trajectory checkpoints peak/career-length are computed from. Sorted chronologically
    per player."""
    lvl = (
        preds[["player", "edition_id", "date", "player_elo"]]
        .assign(edition_start=lambda d: d.groupby("edition_id")["date"].transform("min"))
        .drop(columns="date")
        .drop_duplicates(subset=["player", "edition_id"])
        .sort_values(["player", "edition_start"], kind="stable")
        .reset_index(drop=True)
    )
    return lvl


def _trailing_window_prior_max(dates, values, window_days):
    """Sliding-window (time-based, not count-based) max, STRICTLY prior to each row - dates must
    already be sorted ascending. O(n) amortized via a decreasing monotonic deque: front is always
    the current window's max: evict-from-front anything older than the window, evict-from-back
    anything <= the incoming value (it can never be the max again once a later, >= value exists),
    then read the max BEFORE pushing the current row (no lookahead - the row's own value must
    never contribute to its own peak)."""
    dq = deque()  # (date, value), value strictly decreasing front->back
    out = []
    for date, value in zip(dates, values):
        cutoff = date - pd.Timedelta(days=window_days)
        while dq and dq[0][0] < cutoff:
            dq.popleft()
        out.append(dq[0][1] if dq else np.nan)
        while dq and dq[-1][1] <= value:
            dq.pop()
        dq.append((date, value))
    return out


def add_peak_features(preds):
    """Merges career_peak_prior, peak_2y_prior, and years_of_history back onto every per-match
    row (edition-level values broadcast to every match in that edition, same frozen-per-edition
    convention as player_elo itself)."""
    lvl = _edition_level_elo(preds)

    out_frames = []
    for player, g in lvl.groupby("player", sort=False):
        g = g.sort_values("edition_start", kind="stable").reset_index(drop=True)
        g["career_peak_prior"] = g["player_elo"].shift(1).cummax()
        g["peak_2y_prior"] = _trailing_window_prior_max(
            g["edition_start"].values, g["player_elo"].values, TRAILING_WINDOW_DAYS
        )
        g["years_of_history"] = (g["edition_start"] - g["edition_start"].iloc[0]).dt.days / 365.25
        out_frames.append(g)
    lvl = pd.concat(out_frames, ignore_index=True)

    merged = preds.merge(
        lvl[["player", "edition_id", "career_peak_prior", "peak_2y_prior", "years_of_history"]],
        on=["player", "edition_id"], how="left", validate="many_to_one",
    )
    assert len(merged) == len(preds), "peak-feature merge changed row count"
    return merged


def _fit_and_validate(train, test, peak_col, label):
    train = train[train[peak_col].notna()].copy()
    test = test[test[peak_col].notna()].copy()
    train["gap_below_peak"] = (train[peak_col] - train["player_elo"]).clip(lower=0)
    test["gap_below_peak"] = (test[peak_col] - test["player_elo"]).clip(lower=0)

    print(f"\n{'-' * 90}\n{label}: {len(train)} train-era / {len(test)} test-era rows "
          f"({train['player'].nunique()} / {test['player'].nunique()} distinct players)")
    print(f"  mean gap_below_peak: train={train['gap_below_peak'].mean():.1f} Elo, "
          f"test={test['gap_below_peak'].mean():.1f} Elo; "
          f"share exactly at peak (gap=0): train={(train['gap_below_peak'] == 0).mean():.1%}, "
          f"test={(test['gap_below_peak'] == 0).mean():.1%}")

    offset = train["pred_win"].apply(logit).values
    x = (train["gap_below_peak"] / 100.0).values
    y = train["actual_win"].values
    beta, se = fit_beta_newton(offset, x, y)
    z = beta / se if se == se and se != 0 else float("nan")
    print(f"  Train-era fitted beta (adjusted_logit = logit(pred_win) + beta * gap_below_peak/100): "
          f"{beta:+.4f} (SE={se:.4f}, z={z:+.2f}, "
          f"{'|z|>1.96, nominally significant' if abs(z) > 1.96 else 'not significant on its own'})")

    test = test.copy()
    test["adjusted_pred"] = test.apply(
        lambda r: sigmoid(logit(r["pred_win"]) + beta * (r["gap_below_peak"] / 100.0)), axis=1)
    test["raw_loss"] = log_loss(test["actual_win"].values, test["pred_win"].values)
    test["adj_loss"] = log_loss(test["actual_win"].values, test["adjusted_pred"].values)

    observed, lo, hi = cluster_bootstrap_ci(test, "raw_loss", "adj_loss", group_col="player")
    ci_excludes_zero = lo > 0 or hi < 0
    print(f"  Held-out: raw log-loss={test['raw_loss'].mean():.4f}, adjusted={test['adj_loss'].mean():.4f}")
    print(f"  Mean per-match log-loss improvement (raw - adjusted, >0 = adjustment better), "
          f"player-clustered: {observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    verdict = ("VALIDATED (positive, excludes zero)" if ci_excludes_zero and lo > 0
               else ("WORSE than raw Elo (excludes zero, wrong sign)" if hi < 0
                     else "NOT validated - CI straddles zero"))
    print(f"  VERDICT ({label}): train z={z:+.2f}; held-out {verdict}")

    # descriptive breakdown for interpretability: "meaningfully below peak" vs at/near peak, on
    # the held-out test rows only (never re-fit on this split)
    below = test["gap_below_peak"] >= GAP_THRESHOLD
    desc = pd.DataFrame({
        "population": ["at/near peak (<%.0f Elo below)" % GAP_THRESHOLD, "meaningfully below peak (>=%.0f Elo)" % GAP_THRESHOLD],
        "n": [len(test[~below]), len(test[below])],
        "assigned": [test.loc[~below, "pred_win"].mean(), test.loc[below, "pred_win"].mean()],
        "actual": [test.loc[~below, "actual_win"].mean(), test.loc[below, "actual_win"].mean()],
    })
    desc["gap_pp"] = desc["actual"] - desc["assigned"]
    print(f"\n  Descriptive: actual vs. Elo-assigned win rate, split at {GAP_THRESHOLD:.0f}-Elo gap-below-peak "
          f"threshold (test era only):")
    print(desc.to_string(index=False, formatters={
        "assigned": "{:.1%}".format, "actual": "{:.1%}".format, "gap_pp": "{:+.1%}".format,
    }))
    return beta, z, observed, lo, hi


def run(min_history_years):
    all_train, all_test = {"career_peak_prior": [], "peak_2y_prior": []}, {"career_peak_prior": [], "peak_2y_prior": []}
    coverage_rows = []

    for tour in ["ATP", "WTA"]:
        matches = load_matches_for_tour(tour)
        preds, editions = build_frozen_predictions(matches)
        preds = add_peak_features(preds)
        preds["tour"] = tour

        n_total = len(preds)
        eligible = preds[preds["years_of_history"] >= min_history_years]
        n_eligible = len(eligible)
        coverage_rows.append((tour, n_total, n_eligible, n_eligible / n_total if n_total else float("nan"),
                               eligible["player"].nunique()))

        split_idx = int(len(editions) * TRAIN_FRACTION)
        train_editions = set(editions["edition_id"].iloc[:split_idx])
        test_editions = set(editions["edition_id"].iloc[split_idx:])
        train = eligible[eligible["edition_id"].isin(train_editions)]
        test = eligible[eligible["edition_id"].isin(test_editions)]
        for col in ("career_peak_prior", "peak_2y_prior"):
            all_train[col].append(train)
            all_test[col].append(test)

    print(f"{'=' * 90}\nCOVERAGE - rows with >= {min_history_years:.1f} years of real tracked "
          f"history as of that edition (required for a meaningful peak estimate)\n{'=' * 90}")
    cov = pd.DataFrame(coverage_rows, columns=["tour", "n_total_rows", "n_eligible_rows", "eligible_share", "n_eligible_players"])
    print(cov.to_string(index=False, formatters={"eligible_share": "{:.1%}".format}))

    results = {}
    for col, label in [("career_peak_prior", "CAREER-HIGH PEAK (all-time, to-date)"),
                        ("peak_2y_prior", f"TRAILING {TRAILING_WINDOW_DAYS // 365}-YEAR PEAK")]:
        train = pd.concat(all_train[col], ignore_index=True)
        test = pd.concat(all_test[col], ignore_index=True)
        results[label] = _fit_and_validate(train, test, col, label)

    print(f"\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
    for label, (beta, z, observed, lo, hi) in results.items():
        ci_excludes_zero = lo > 0 or hi < 0
        print(f"  {label:<40} beta={beta:+.4f} (train z={z:+.2f}), held-out {observed:+.4f}, "
              f"95% CI [{lo:+.4f}, {hi:+.4f}] -> "
              f"{'VALIDATED' if ci_excludes_zero and lo > 0 else ('WORSE' if hi < 0 else 'NOT validated')}")
    return results


def report_player(name, tour, cutoff_str, min_history_years):
    """Real-world spot check for one named player: their OWN gap_below_peak (both definitions) as
    of a specific real cutoff date, using the exact same frozen walk-forward as the population
    test above (not elo_ratings.py's separately-windowed production Elo) - so this number is
    directly comparable to what the fitted beta above was estimated against."""
    matches = load_matches_for_tour(tour)
    preds, _editions = build_frozen_predictions(matches)
    preds = add_peak_features(preds)
    rows = preds[(preds["player"] == name) & (preds["date"] < pd.Timestamp(cutoff_str))]
    if rows.empty:
        print(f"\n{name} ({tour}): no rows found before {cutoff_str}")
        return
    last = rows.sort_values("date").iloc[-1]
    eligible = last["years_of_history"] >= min_history_years
    print(f"\n{name} ({tour}) - as of last tracked row before {cutoff_str} "
          f"({last['date'].date()}, edition {last['edition_id']}):")
    print(f"  current_elo (player_elo)  = {last['player_elo']:.1f}")
    print(f"  career_peak_prior         = {last['career_peak_prior']:.1f}  "
          f"(gap below = {max(last['career_peak_prior'] - last['player_elo'], 0):.1f})")
    print(f"  peak_2y_prior             = {last['peak_2y_prior']:.1f}  "
          f"(gap below = {max(last['peak_2y_prior'] - last['player_elo'], 0):.1f})")
    print(f"  years_of_history          = {last['years_of_history']:.2f} "
          f"({'meets' if eligible else 'BELOW'} the {min_history_years:.1f}-year coverage bar)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-history-years", type=float, default=MIN_HISTORY_YEARS_DEFAULT)
    parser.add_argument("--gap-threshold", type=float, default=GAP_THRESHOLD_DEFAULT)
    args = parser.parse_args()
    GAP_THRESHOLD = args.gap_threshold

    run(args.min_history_years)

    print(f"\n{'=' * 90}\nSPOT CHECKS - Zheng Qinwen and Naomi Osaka, real current situations\n{'=' * 90}")
    report_player("Zheng Q.", "WTA", "2026-08-30", args.min_history_years)
    report_player("Osaka N.", "WTA", "2026-08-30", args.min_history_years)
