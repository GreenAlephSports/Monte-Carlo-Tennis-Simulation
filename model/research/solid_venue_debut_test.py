"""Tests whether the R1 underperformance found in thin-history players (career-inexperienced,
<30 matches - see thin_history_rank_blend_test.py / thin_history_platt_test.py / historical_
bracket_calibration.py's round-depth breakdown) is really about CAREER length, or whether it's at
least partly a SITUATIONAL/unfamiliarity effect: does a SOLID player (>=30 career matches, a real,
trustworthy Elo - the same threshold every other test in this series uses to define "not thin")
underperform their own Elo in their very first R1 match at a SPECIFIC tournament they've never
played before, compared to a solid player playing R1 at a tournament they HAVE played before?

This is a genuinely different population from every prior thin-history test: every row here already
has a real, trustworthy Elo built from 30+ real matches - if there IS an effect, it can't be
explained by "the rating doesn't know this player yet" the way the thin-history population's could
be. If established players debuting at a new VENUE show the same kind of underperformance, that's
real evidence of a situational effect (new city/altitude/court speed/crowd/logistics - "first time at
THIS event", not "first time as a pro") distinct from career length. If they don't, that further
isolates the original effect to inexperienced players specifically, not first-timers-at-this-event.

Population/control design:
  - Reuses thin_history_rank_blend_test.build_dataset directly (frozen per-tournament-edition Elo,
    player_matches_before column) - not reimplemented.
  - "Tournament" (the venue/event identity) is recovered from edition_id by stripping the trailing
    " <year>" (edition_id is always "<Tournament> <year>" by construction) - groups an event across
    all its editions regardless of year, distinct from "edition" (one specific year's instance).
  - TREATMENT: round == "1st Round", player_matches_before >= SOLID_MATCHES, and this edition is the
    player's FIRST EVER appearance at this specific tournament anywhere in the loaded window.
  - CONTROL: round == "1st Round", player_matches_before >= SOLID_MATCHES, and this is NOT their
    first appearance at this specific tournament (a repeat visitor) - holds "solid career" and "R1"
    fixed, isolates the venue-familiarity variable specifically.
  - Guard against a real confound: if a tournament ITSELF is new to the loaded dataset window, every
    player's first row there looks like a "debut" for data-availability reasons, not because they
    genuinely never played it in real life. Treatment rows require the tournament to already have
    >= MIN_PRIOR_TOURNAMENT_EDITIONS real editions in the dataset BEFORE this player's debut, so a
    "debut" here means "this specific tournament was already an established, playable fixture and
    this player still hadn't shown up" - not a data-window artifact.
  - Disclosed, not fixable with this data: left-censoring at the start of the loaded window (2000
    ATP / 2007 WTA) - a solid player whose TRUE first visit to a tournament happened before the
    window starts would be miscounted as a "control" (repeat visitor) row at their first IN-WINDOW
    edition, or a genuine debut a few years in could still reflect a player who simply skipped the
    event early in a long career, not unfamiliarity in the "never been there" sense. This dilutes
    the treatment/control contrast somewhat (biases toward finding NO effect, not toward a spurious
    one) rather than invalidating it.
  - Player-clustered residual comparison (summarize_bucket, same per-bucket style veteran_decline_
    test.py/layoff_within_tournament_decay_test.py already use) plus a two-sample player-clustered
    bootstrap CI on the debut-vs-control residual GAP (a genuinely different comparison from this
    series' usual paired raw-vs-adjusted cluster_bootstrap_ci, since debut/control here are two
    disjoint player-row populations, not two predictions on the same rows).

Usage:
    python model/research/solid_venue_debut_test.py [--tour ATP|WTA] [--max-editions N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elo_ratings import load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci, summarize_bucket  # noqa: E402
from thin_history_rank_blend_test import SOLID_MATCHES, build_dataset  # noqa: E402

MIN_PRIOR_TOURNAMENT_EDITIONS = 3  # tournament must already be an established fixture, not new-to-data


def two_sample_cluster_bootstrap(group_a, group_b, group_col="player", n_boot=5000, seed=42):
    """Player-clustered bootstrap CI for the difference in row-weighted mean residual between two
    DISJOINT populations (unlike survivorship_upset_test.cluster_bootstrap_ci, which compares two
    columns on the SAME rows) - resamples players independently within each group, pools each
    resampled player's rows the same way the observed point estimate does."""
    def prep(g):
        codes, players = pd.factorize(g[group_col].values)
        sums = np.zeros(len(players))
        counts = np.zeros(len(players))
        np.add.at(sums, codes, g["residual_row"].values)
        np.add.at(counts, codes, 1)
        return sums, counts

    sums_a, counts_a = prep(group_a)
    sums_b, counts_b = prep(group_b)
    observed = group_a["residual_row"].mean() - group_b["residual_row"].mean()

    rng = np.random.default_rng(seed)
    n_a, n_b = len(sums_a), len(sums_b)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx_a = rng.integers(0, n_a, n_a)
        idx_b = rng.integers(0, n_b, n_b)
        mean_a = sums_a[idx_a].sum() / counts_a[idx_a].sum()
        mean_b = sums_b[idx_b].sum() / counts_b[idx_b].sum()
        boot[i] = mean_a - mean_b
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return observed, lo, hi


def run(tour, max_editions=None):
    if max_editions is not None:
        print(f"*** QUICK CHECK ON RECENT DATA ONLY (--max-editions {max_editions}) - NOT the "
              f"full-historical verdict. Debut-at-a-specific-venue rows are naturally rarer than "
              f"plain thin-history rows (needs a SOLID player who's somehow never played THIS event) "
              f"- read the n's below before trusting any number here. ***\n")
    matches = load_matches_for_tour(tour)
    preds, editions = build_dataset(matches, max_editions=max_editions)
    print(f"{tour}: {len(editions)} tournament editions, {len(preds)} player-perspective rows\n")

    df = preds.copy()
    df["tournament"] = df["edition_id"].str.rsplit(" ", n=1).str[0]

    # tournament's own first-seen date in this (possibly max_editions-truncated) window - used both
    # to find each player's debut edition and to guard against new-to-data tournaments below.
    tournament_first_date = df.groupby("tournament")["date"].min()
    tournament_edition_dates = (
        df[["tournament", "edition_id", "date"]].drop_duplicates(subset=["tournament", "edition_id"])
        .sort_values("date")
    )
    tournament_edition_dates["tournament_editions_seen_before"] = (
        tournament_edition_dates.groupby("tournament").cumcount()
    )
    edition_prior_count = tournament_edition_dates.set_index(["tournament", "edition_id"])[
        "tournament_editions_seen_before"]

    # each (player, tournament)'s debut edition = their earliest edition_id at that tournament
    player_tournament_dates = (
        df[["player", "tournament", "edition_id", "date"]]
        .drop_duplicates(subset=["player", "tournament", "edition_id"])
        .sort_values("date")
    )
    debut_edition = player_tournament_dates.groupby(["player", "tournament"])["edition_id"].first()
    df["debut_edition_id"] = df.set_index(["player", "tournament"]).index.map(debut_edition).values
    df["is_debut"] = df["edition_id"] == df["debut_edition_id"]
    df["tournament_editions_before_this"] = df.set_index(["tournament", "edition_id"]).index.map(
        edition_prior_count).values

    r1 = df[df["round"] == "1st Round"].copy()
    solid_r1 = r1[r1["player_matches_before"] >= SOLID_MATCHES]

    treatment = solid_r1[
        solid_r1["is_debut"] & (solid_r1["tournament_editions_before_this"] >= MIN_PRIOR_TOURNAMENT_EDITIONS)
    ].copy()
    control = solid_r1[~solid_r1["is_debut"]].copy()

    print(f"Solid-player (>= {SOLID_MATCHES} career matches) R1 rows: {len(solid_r1)} total")
    print(f"  TREATMENT (venue debut, tournament already established >= {MIN_PRIOR_TOURNAMENT_EDITIONS} "
          f"prior editions): n={len(treatment)}, {treatment['player'].nunique()} distinct players")
    print(f"  CONTROL (repeat visitor to this tournament): n={len(control)}, "
          f"{control['player'].nunique()} distinct players\n")

    if len(treatment) < 30 or len(control) < 30:
        print("Too few treatment or control rows to say anything meaningful - stopping "
              "(population as defined is too small for this dataset/window).")
        return

    for name, g in [("TREATMENT (venue debut)", treatment), ("CONTROL (repeat visitor)", control)]:
        s = summarize_bucket(name, g)
        print(f"  {name}: n={s['n']}  actual={s['actual_rate']:.1%}  pred={s['pred_rate']:.1%}  "
              f"residual={s['residual']:+.1%}  95% CI[{s['residual_ci_lo']:+.1%},{s['residual_ci_hi']:+.1%}]  "
              f"z={s['z']:.2f}")

    treatment["residual_row"] = treatment["actual_win"] - treatment["pred_win"]
    control["residual_row"] = control["actual_win"] - control["pred_win"]
    observed, lo, hi = two_sample_cluster_bootstrap(treatment, control)
    verdict = "debuting players underperform repeat visitors (CI excludes zero, <0)" if hi < 0 else (
        "debuting players OVER-perform repeat visitors (CI excludes zero, >0)" if lo > 0 else
        "NOT distinguishable from repeat visitors (CI straddles zero)")
    print(f"\nPlayer-clustered bootstrap: debut residual - control residual = {observed:+.1%}, "
          f"95% CI [{lo:+.1%}, {hi:+.1%}]")
    print(f"  -> {verdict}")

    if len(treatment) < 200 or len(control) < 200:
        print(f"\n  CAUTION: treatment n={len(treatment)} / control n={len(control)} is a small "
              f"sample - treat this as early/directional, not a settled verdict, regardless of "
              f"which way it points.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
    parser.add_argument("--max-editions", type=int, default=None,
                         help="quick-check mode: only score the most recent N tournament editions "
                              "(before any split), instead of the full lookback window")
    args = parser.parse_args()
    run(args.tour, max_editions=args.max_editions)
