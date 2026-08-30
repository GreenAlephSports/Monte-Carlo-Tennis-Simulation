"""Tests a genuinely new, narrow population: does a MAJOR CHAMPION (has won a Grand Slam final at
some point in their career, before this match) get systematically underrated by Elo for a real
WINDOW of time after returning from a long (90d+) absence - not just on the single "return match"
production's existing layoff bucket already prices, but for months afterward while their rating is
still climbing back from the K-factor scar of a rusty comeback stretch?

Why this is distinct from every layoff/veteran mechanism already tested:
  - layoff_test.py / win_probability.LAYOFF_BUCKET_EDGES_* (ALREADY IN PRODUCTION): a flat shift
    applied to days_since_last_match, which is frozen at whatever it was on THIS match's own date -
    it fires once, on the actual return match (and any other match that happens to follow a 90d+
    gap), and says nothing about matches 2, 5, or 20 real results later, once days_since_last_match
    has long since dropped back under 14. If a player's Elo is still lagging their true skill three
    months after coming back (because every match played while still rusty permanently moved their
    rating down by real K-factor points, and rebuilding it takes real wins), this correction cannot
    see that - it only ever looks at the CURRENT gap-since-last-match, not "how long ago was the
    last time this player had a long layoff."
  - layoff_within_tournament_decay_test.py: tests decay WITHIN a single tournament run (does the
    round-1-of-a-comeback penalty ease by round 3 of that SAME event) - a much shorter horizon
    (days/a week) than this test's real question (months of rating catch-up).
  - veteran_decline_test.py: tests the OPPOSITE direction and an unrelated population (old AND
    still-elite-rated players underperforming, not comeback-related at all).
  - thin_history_rank_blend_test.py / thin_history_platt_test.py: target players Elo barely knows
    ANYTHING about (<10 total career matches) - the opposite of a major champion, who Elo has a
    long, trustworthy history for; the mechanism here isn't "not enough data," it's "real data that
    is now stale/depressed relative to current true skill."
  - slam_venue_residual_test.py: tests a fixed, timeless per-venue skill gap, not anything tied to
    an absence/return event at all.

Population/control design:
  - MAJOR CHAMPION: has won at least one Grand Slam final (Round == "The Final" at one of the four
    Slams) at some point STRICTLY BEFORE this match's own date - the credential must predate the
    comeback, not be established by it.
  - COMEBACK EVENT: any real match where days_since_last_match >= COMEBACK_TRIGGER_DAYS (90, same
    threshold production's own 90d_plus layoff bucket uses) - this player's own most recent such
    event before the row's date, if any within COMEBACK_WINDOW_DAYS.
  - TREATMENT: rows where the player is a major champion AND falls inside their own most recent
    comeback window.
  - CONTROL: the mirror population - non-champions inside their OWN comeback window (same trigger,
    same window, different credential) - isolates whether championship pedigree specifically
    matters, not just "everyone recovers slowly from a long layoff" (already partially covered by
    the existing flat bucket shift, whatever residual is left after that).
  - Residual measured against BOTH raw (uncorrected) Elo and against production's existing flat
    90d_plus-shifted prediction (only applied to rows that are STILL within the flat shift's own
    single-match trigger window - most of a multi-month comeback window has no existing correction
    applied to it at all) - so this shows both the raw gap and how much (if any) of it the deployed
    correction already absorbs.
  - Same rigor as every other test this series: frozen per-tournament-edition Elo (elite_opponent_
    residual_test.build_frozen_predictions, with --max-editions quick-check support), player-
    clustered residual estimation (summarize_bucket) plus a two-sample player-clustered bootstrap
    CI on the champion-vs-non-champion gap (same two_sample_cluster_bootstrap pattern
    solid_venue_debut_test.py already validated for exactly this disjoint-population comparison
    shape).

Usage:
    python model/research/champion_comeback_test.py [--tour ATP|WTA] [--max-editions N] [--highlight "Osaka N."]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import build_frozen_predictions, logit, sigmoid  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from layoff_test import build_layoff_dataset  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci, summarize_bucket  # noqa: E402
from win_probability import LAYOFF_BUCKET_EDGES_ATP, LAYOFF_BUCKET_EDGES_WTA  # noqa: E402

SLAM_NAMES = {"Australian Open", "French Open", "US Open", "Wimbledon"}
COMEBACK_TRIGGER_DAYS = 90       # same threshold as production's own 90d_plus layoff bucket
COMEBACK_WINDOW_DAYS = 365       # how long "still recovering" is allowed to run after the comeback event


def two_sample_cluster_bootstrap(group_a, group_b, value_col, group_col="player", n_boot=5000, seed=42):
    """Player-clustered bootstrap CI for the difference in row-weighted mean residual between two
    DISJOINT populations - same pattern solid_venue_debut_test.py validated for this exact shape
    (unlike survivorship_upset_test.cluster_bootstrap_ci, which compares two columns on the SAME
    rows)."""
    def prep(g):
        codes, players = pd.factorize(g[group_col].values)
        sums = np.zeros(len(players))
        counts = np.zeros(len(players))
        np.add.at(sums, codes, g[value_col].values)
        np.add.at(counts, codes, 1)
        return sums, counts

    sums_a, counts_a = prep(group_a)
    sums_b, counts_b = prep(group_b)
    observed = group_a[value_col].mean() - group_b[value_col].mean()

    rng = np.random.default_rng(seed)
    n_a, n_b = len(sums_a), len(sums_b)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx_a = rng.integers(0, n_a, n_a)
        idx_b = rng.integers(0, n_b, n_b)
        boot[i] = sums_a[idx_a].sum() / counts_a[idx_a].sum() - sums_b[idx_b].sum() / counts_b[idx_b].sum()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return observed, lo, hi


def build_champion_flags(matches):
    """player -> sorted array of dates on which they won a Grand Slam final - used to check
    'already a major champion strictly before this match's date', not 'ever in the dataset'
    (avoids crediting a future title to a player's earlier, pre-championship comeback rows)."""
    finals = matches[(matches["Tournament"].isin(SLAM_NAMES)) & (matches["Round"] == "The Final")]
    titles = {}
    for row in finals.itertuples(index=False):
        titles.setdefault(row.Winner, []).append(row.Date)
    return {player: np.sort(np.array(dates, dtype="datetime64[ns]")) for player, dates in titles.items()}


def is_champion_before(player, date, champion_dates):
    dates = champion_dates.get(player)
    if dates is None or len(dates) == 0:
        return False
    return dates[0] < np.datetime64(date)


def build_comeback_windows(layoff_df):
    """player -> sorted array of comeback-event dates (rows where days_since_last >=
    COMEBACK_TRIGGER_DAYS) - used to find each row's own most recent comeback event, if any within
    COMEBACK_WINDOW_DAYS."""
    events = layoff_df[layoff_df["days_since_last"] >= COMEBACK_TRIGGER_DAYS]
    windows = {}
    for player, g in events.groupby("player"):
        windows[player] = np.sort(g["date"].values.astype("datetime64[ns]"))
    return windows


def days_since_comeback(player, date, windows):
    """Days since this player's own most recent comeback event strictly before `date`, or None if
    none exists within COMEBACK_WINDOW_DAYS (or ever). The trigger row ITSELF (days_since_last ==
    the huge gap) counts as day 0 of its own comeback window - a player's actual return match is
    the start of the recovery period being tested, not excluded from it."""
    dates = windows.get(player)
    if dates is None:
        return None
    idx = np.searchsorted(dates, np.datetime64(date), side="right") - 1
    if idx < 0:
        return None
    delta = (np.datetime64(date) - dates[idx]) / np.timedelta64(1, "D")
    return delta if 0 <= delta <= COMEBACK_WINDOW_DAYS else None


def run(tour, max_editions=None, highlight=None):
    if max_editions is not None:
        print(f"*** QUICK CHECK ON RECENT DATA ONLY (--max-editions {max_editions}) - NOT the "
              f"full-historical verdict. Major-champion-in-comeback-window rows are a narrow "
              f"population even at full scale - read the n's below before trusting any number "
              f"here. ***\n")
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches, max_editions=max_editions)
    layoff_df = build_layoff_dataset(matches, preds)
    print(f"{tour}: {len(editions)} tournament editions, {len(preds)} player-perspective rows\n")

    champion_dates = build_champion_flags(matches)
    print(f"{len(champion_dates)} distinct Grand Slam champions found in this window "
          f"({sum(len(v) for v in champion_dates.values())} total title-runs)")

    windows = build_comeback_windows(layoff_df)
    layoff_df = layoff_df.copy()
    layoff_df["days_since_comeback"] = [
        days_since_comeback(p, d, windows) for p, d in zip(layoff_df["player"], layoff_df["date"])
    ]
    layoff_df["is_champion"] = [
        is_champion_before(p, d, champion_dates) for p, d in zip(layoff_df["player"], layoff_df["date"])
    ]

    in_window = layoff_df[layoff_df["days_since_comeback"].notna()].copy()
    treatment = in_window[in_window["is_champion"]].copy()
    control = in_window[~in_window["is_champion"]].copy()
    print(f"\nRows inside a {COMEBACK_WINDOW_DAYS}-day post-comeback window (comeback = a real "
          f"{COMEBACK_TRIGGER_DAYS}+ day gap since last match): {len(in_window)} total")
    print(f"  TREATMENT (major champion): n={len(treatment)}, {treatment['player'].nunique()} distinct players")
    print(f"  CONTROL (non-champion)    : n={len(control)}, {control['player'].nunique()} distinct players")

    if len(treatment) < 30 or len(control) < 30:
        print("\nToo few treatment or control rows to say anything meaningful - stopping "
              "(population as defined is too small for this dataset/window).")
        return

    for name, g in [("TREATMENT (champion, in comeback window)", treatment),
                     ("CONTROL (non-champion, in comeback window)", control)]:
        s = summarize_bucket(name, g)
        print(f"\n  {name}: n={s['n']}  actual={s['actual_rate']:.1%}  pred(raw Elo)={s['pred_rate']:.1%}  "
              f"residual={s['residual']:+.1%}  95% CI[{s['residual_ci_lo']:+.1%},{s['residual_ci_hi']:+.1%}]  z={s['z']:.2f}")

    treatment["residual_row"] = treatment["actual_win"] - treatment["pred_win"]
    control["residual_row"] = control["actual_win"] - control["pred_win"]
    observed, lo, hi = two_sample_cluster_bootstrap(treatment, control, "residual_row")
    verdict = "champions underperform non-champions in their comeback window (CI excludes zero, <0)" if hi < 0 else (
        "champions OVER-perform non-champions in their comeback window (CI excludes zero, >0)" if lo > 0 else
        "NOT distinguishable from non-champions (CI straddles zero)")
    print(f"\nPlayer-clustered bootstrap: champion residual - non-champion residual = {observed:+.1%}, "
          f"95% CI [{lo:+.1%}, {hi:+.1%}]")
    print(f"  -> {verdict}")

    # how much of the raw gap does production's EXISTING flat 90d_plus shift already absorb? only
    # applies to rows whose OWN days_since_last (not days_since_comeback) is still >= 90 - i.e. the
    # single return match itself, not the following weeks/months this test is really asking about.
    bucket_edges = LAYOFF_BUCKET_EDGES_ATP if tour == "ATP" else LAYOFF_BUCKET_EDGES_WTA
    flat_shift = next(shift for name, _test, shift in bucket_edges if name == "90d_plus")
    treatment["already_corrected"] = treatment["days_since_last"] >= COMEBACK_TRIGGER_DAYS
    n_already = int(treatment["already_corrected"].sum())
    print(f"\nOf the {len(treatment)} champion comeback-window rows, {n_already} "
          f"({n_already / len(treatment):.1%}) are ALSO the immediate return match itself (already "
          f"gets production's flat {flat_shift:+.4f} logit shift) - the remaining "
          f"{len(treatment) - n_already} ({1 - n_already / len(treatment):.1%}) have NO existing "
          f"correction applied at all, regardless of what this test finds.")

    treatment["current_adjusted_pred"] = np.where(
        treatment["already_corrected"],
        treatment["pred_win"].apply(lambda p: sigmoid(logit(p) + flat_shift)),
        treatment["pred_win"],
    )
    residual_vs_current = (treatment["actual_win"] - treatment["current_adjusted_pred"]).mean()
    print(f"Champion comeback-window residual vs. what PRODUCTION ACTUALLY PREDICTS TODAY "
          f"(flat shift where it applies, raw Elo elsewhere): {residual_vs_current:+.1%} "
          f"(vs. {treatment['residual_row'].mean():+.1%} against fully raw, uncorrected Elo)")

    if highlight:
        print(f"\n{'=' * 90}\nNamed-case lookup: {highlight}\n{'=' * 90}")
        h_dates = champion_dates.get(highlight)
        print(f"  Major champion in this window: {'YES' if h_dates is not None and len(h_dates) else 'NO'}"
              + (f" (title date(s): {[str(pd.Timestamp(d).date()) for d in h_dates]})" if h_dates is not None and len(h_dates) else ""))
        h_rows = layoff_df[layoff_df["player"] == highlight].dropna(subset=["days_since_comeback"])
        if len(h_rows) == 0:
            print(f"  No rows for {highlight} currently fall inside a {COMEBACK_WINDOW_DAYS}-day "
                  f"post-comeback window in this dataset (no qualifying {COMEBACK_TRIGGER_DAYS}+ day "
                  f"gap found, or it's now outside the window).")
        else:
            s = summarize_bucket(highlight, h_rows)
            print(f"  Rows in comeback window: n={s['n']}  actual={s['actual_rate']:.1%}  "
                  f"pred(raw)={s['pred_rate']:.1%}  residual={s['residual']:+.1%}")
            most_recent = h_rows.sort_values("date").iloc[-1]
            print(f"  Most recent such row: {pd.Timestamp(most_recent['date']).date()}, "
                  f"{most_recent['days_since_comeback']:.0f} days into that comeback window")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="WTA", choices=["ATP", "WTA"])
    parser.add_argument("--max-editions", type=int, default=None,
                         help="quick-check mode: only score the most recent N tournament editions "
                              "(before any split), instead of the full lookback window")
    parser.add_argument("--highlight", default=None, help="ratings-csv-format player name to print a named-case lookup for")
    args = parser.parse_args()
    run(args.tour, max_editions=args.max_editions, highlight=args.highlight)
