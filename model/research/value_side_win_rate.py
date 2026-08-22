"""Direct value-side win-rate test: for every real, decided match with both a model and a market
probability, the "value side" is whichever player the MODEL favors more than the MARKET does -
regardless of whether that's the outright favorite or the underdog (that distinction is the whole
point: a model that's only "right" by agreeing with the favorite isn't finding anything the market
doesn't already know).

Reports one direct number: how often did the value side actually win, versus what the market's own
implied probability said they should win, on that same set of matches. A positive gap (value side's
real win rate above what the market priced them at) is evidence the model's disagreement with the
market finds real value; zero or negative is not - the model's picks perform no better (or worse)
than the market's own pricing already implied.

Two sources, combined:
  - kaggle_concluded (Montreal/Toronto): real historical bookmaker odds, via
    model_vs_market_calibration.collect_rows (not reimplemented here).
  - live_espn (Cincinnati): market_prob_a backfilled into calibration_log.csv from two places -
    (1) a one-time historical backfill from the single old bracket_export.json git snapshot that
    still had pregame Odds API prices for matches that are now decided, and (2) going forward,
    live_match_watcher.py now caches every unsettled matchup's pregame market_prob_a on every poll
    (see its update_market_price_cache) specifically so calibration_log.py can look it up once the
    match concludes - The Odds API drops an event entirely once it's Final, so this is the only way
    to ever recover a live match's pregame price after the fact. This sample starts small and grows
    on its own as live_match_watcher.py keeps running and calibration_log.py keeps getting re-run.

Usage:
    python model/research/value_side_win_rate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

import pandas as pd

from calibration_log import LOG_PATH  # noqa: E402
from model_vs_market_calibration import CONCLUDED_TOURNAMENTS, collect_rows  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402


def collect_kaggle_side():
    rows = []
    for bracket_path, _pretournament_csv, kaggle_tournament_name in CONCLUDED_TOURNAMENTS:
        print(f"Collecting {bracket_path.stem} (kaggle_concluded)...", file=sys.stderr)
        collected = collect_rows(bracket_path, kaggle_tournament_name)
        for r in collected:
            r["source"] = "kaggle_concluded"
        rows.extend(collected)
    return rows


def collect_live_side():
    """live_espn rows from calibration_log.csv that got a real market_prob_a (see module
    docstring) - model_prob_a is derived from the log's own favorite/favorite_prob columns rather
    than recomputed, so this reflects the model's prediction AS LOGGED at the time (favorite_prob
    is already relative to whichever side was favorite; player_a-relative here is
    favorite_prob if player_a was the favorite, else 1 - favorite_prob)."""
    if not LOG_PATH.exists():
        return []
    log = pd.read_csv(LOG_PATH)
    live = log[(log["source"] == "live_espn") & log["market_prob_a"].notna()]
    rows = []
    for r in live.itertuples():
        model_prob_a = r.favorite_prob if r.favorite == r.player_a else 1 - r.favorite_prob
        rows.append({
            "tour": r.tour, "tournament": r.tournament, "round": r.round_label,
            "player_a": r.player_a, "player_b": r.player_b,
            "actual_a_win": int(r.winner == r.player_a),
            "model_prob_a": model_prob_a, "market_prob_a": r.market_prob_a,
            "source": "live_espn",
        })
    return rows


def run():
    kaggle_rows = collect_kaggle_side()
    live_rows = collect_live_side()

    print(f"\nkaggle_concluded (Montreal/Toronto, real historical bookmaker odds): {len(kaggle_rows)} matches",
          file=sys.stderr)
    print(f"live_espn (Cincinnati, backfilled/cached pregame market prices): {len(live_rows)} matches",
          file=sys.stderr)

    df = pd.DataFrame(kaggle_rows + live_rows)
    if df.empty:
        print("\nNo matches with both a model and a market probability - nothing to test.")
        return

    # the value side: whichever player the MODEL favors MORE than the market does, regardless of
    # whether that's the favorite or the underdog on either side individually - a tie (model and
    # market agree exactly) has no value side and is excluded, not arbitrarily assigned to one side.
    diff = df["model_prob_a"] - df["market_prob_a"]
    df = df[diff != 0].copy()
    diff = diff[diff.index.isin(df.index)]

    df["value_side"] = df["player_a"].where(diff > 0, df["player_b"])
    df["value_side_won"] = df.apply(
        lambda r: int((r["value_side"] == r["player_a"]) == bool(r["actual_a_win"])), axis=1
    )
    # the market's own implied probability that the value side wins - i.e. what the market itself
    # said this exact outcome should have been, the direct comparison point for value_side_won.
    df["value_side_market_prob"] = df["market_prob_a"].where(
        df["value_side"] == df["player_a"], 1 - df["market_prob_a"]
    )

    n = len(df)
    real_win_rate = df["value_side_won"].mean()
    market_implied_rate = df["value_side_market_prob"].mean()
    gap = real_win_rate - market_implied_rate

    # raw_col first, adj_col second - cluster_bootstrap_ci computes mean(raw_col - adj_col), so
    # this order gives mean(real - market_implied) = the gap, not its negation.
    observed, lo, hi = cluster_bootstrap_ci(
        df.assign(_real=df["value_side_won"], _market=df["value_side_market_prob"]),
        "_real", "_market", group_col="value_side",
    )

    print(f"\n{'=' * 90}\nVALUE-SIDE WIN RATE TEST\n{'=' * 90}")
    print(f"n = {n} real, decided matches with both a model and a market probability "
          f"({len(df[df['source'] == 'kaggle_concluded'])} kaggle_concluded, "
          f"{len(df[df['source'] == 'live_espn'])} live_espn)")
    print(f"\nValue-side players actually won {real_win_rate:.1%} of the time.")
    print(f"The market's own implied probability said they should win {market_implied_rate:.1%} of the time.")
    print(f"\nGap (real - market-implied): {gap:+.1%}")
    print(f"Player-clustered 95% bootstrap CI on that gap: [{lo:+.1%}, {hi:+.1%}]")
    if gap > 0 and lo > 0:
        verdict = "POSITIVE and the CI clears zero - real evidence the model's disagreement with the market finds value."
    elif gap > 0:
        verdict = "positive but the CI straddles zero - not distinguishable from no edge at this sample size."
    else:
        verdict = "zero or negative - no evidence the model's disagreement with the market finds real value."
    print(f"\nVerdict: {verdict}")

    print(f"\n{'=' * 90}\nBy source\n{'=' * 90}")
    for source, g in df.groupby("source"):
        rr, mr = g["value_side_won"].mean(), g["value_side_market_prob"].mean()
        print(f"{source:<20} n={len(g):<4} value-side real win rate {rr:.1%}   "
              f"market-implied {mr:.1%}   gap {rr - mr:+.1%}")


if __name__ == "__main__":
    run()
