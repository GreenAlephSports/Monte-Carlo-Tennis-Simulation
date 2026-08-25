"""Real paper-trading backtest, scoped to Cincinnati only (ATP + WTA combined), over every real
match this project has an actual TRACKED market price for - not a re-simulation, not a synthetic
odds series. Two sources, both already real:
  - calibration_log.py's market_prob_a column - pregame de-vigged prices captured by
    live_match_watcher.py's price cache while each match was still genuinely pregame.
  - live_match_snapshot.py's own running log (output/live_match_snapshots.jsonl), if any on-demand
    live snapshots were ever taken for a Cincinnati match - each logged line's live_market_prob_1
    is a separate, later price point for the SAME match, kept as its own row (a different price
    at a different moment is a different real trading opportunity, not a duplicate).

For every such (match, price) row: the model's probability on both sides (win_probability(),
matching calibration_log.py's favorite_prob exactly) against the market's own de-vigged
probability on both sides. A "bet opportunity" is ANY side where the model's probability implies
positive EV against the market price - never pre-filtered to "the side we already expected to
disagree on".

Two sizing methods, side by side, both flagged as real assumptions:
  - Fractional Kelly at 0.25x and 0.5x (full Kelly is well known to be too aggressive for real
    bankroll variance - both fractions computed against a fixed, non-compounding 100-unit
    reference bankroll, since these are a scattered set of real-world-concurrent tournament bets,
    not a strict sequential series where compounding order would even mean anything).
  - Flat stake: the same fixed 1-unit stake on every opportunity, for direct comparison.

Settlement price ASSUMPTION, stated plainly: bets are settled at the market's DE-VIGGED implied
price (decimal odds = 1 / market_prob), not a real, vig-inclusive bookmaker price - the vig-free
price is the only one this project has ever computed and stored (see ev_comparison.implied_
probabilities), and it's also the correct reference price for "does my model see value" in the
first place. A REAL placed bet would be settled at a worse (vigged) price than this, so this
backtest's P&L is a genuine best case, not a claim about what a real sportsbook account would have
returned.

Usage:
    python model/cincinnati_paper_trading_backtest.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from calibration_log import LOG_PATH, load_existing_log  # noqa: E402

SNAPSHOT_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "output" / "live_match_snapshots.jsonl"

REFERENCE_BANKROLL = 100.0   # fixed, non-compounding unit for every Kelly-fraction stake below
FLAT_STAKE = 1.0             # fixed unit stake for the flat-sizing comparison
KELLY_FRACTIONS = [0.25, 0.5]


def kelly_fraction(model_prob, market_prob):
    """f* = (p - q) / (1 - q), the standard Kelly formula re-expressed directly in probability
    terms (p = model's win probability, q = market's de-vigged implied probability for the same
    side, decimal odds = 1/q) - algebraically identical to the textbook (p*b - (1-p))/b form with
    b = 1/q - 1, just without the intermediate odds conversion. Negative (no edge) is clamped to 0
    by the caller, never a real negative stake."""
    return (model_prob - market_prob) / (1 - market_prob)


def build_opportunity_rows():
    """One row per (match, price-observation, side) where the model implies positive EV against
    that side's de-vigged market price. Each real tracked price observation is checked on BOTH
    sides independently - a match can contribute 0, 1, or 2 opportunities."""
    log = load_existing_log()
    cinci = log[(log["tournament"] == "Cincinnati Open") & log["market_prob_a"].notna()].copy()

    observations = []
    for row in cinci.itertuples(index=False):
        observations.append({
            "source": "calibration_log (pregame)", "tour": row.tour, "round_label": row.round_label,
            "date": row.date, "player_a": row.player_a, "player_b": row.player_b,
            "winner": row.winner, "market_prob_a": row.market_prob_a,
            "model_prob_a": row.favorite_prob if row.favorite == row.player_a else 1 - row.favorite_prob,
        })

    n_snapshot = 0
    if SNAPSHOT_LOG_PATH.exists():
        for line in SNAPSHOT_LOG_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            snap = json.loads(line)
            if snap.get("tournament") != "Cincinnati Open":
                continue
            if snap.get("live_market_prob_1") is None or snap.get("live_model_prob_1") is None:
                continue
            observations.append({
                "source": "live_match_snapshot (in-play)", "tour": snap["tour"],
                "round_label": snap.get("round"), "date": snap.get("captured_at"),
                "player_a": snap["player_1"], "player_b": snap["player_2"],
                "winner": None,  # live_match_snapshot.py doesn't record the eventual winner itself
                "market_prob_a": snap["live_market_prob_1"], "model_prob_a": snap["live_model_prob_1"],
            })
            n_snapshot += 1
    print(f"Price observations: {len(cinci)} pregame (calibration_log.py) + {n_snapshot} in-play "
          f"(live_match_snapshot.py's own log) = {len(observations)} total")

    rows = []
    for obs in observations:
        for side, player, model_p, market_p, other_player in [
            ("a", obs["player_a"], obs["model_prob_a"], obs["market_prob_a"], obs["player_b"]),
            ("b", obs["player_b"], 1 - obs["model_prob_a"], 1 - obs["market_prob_a"], obs["player_a"]),
        ]:
            ev_per_unit = model_p / market_p - 1  # decimal odds = 1/market_p (de-vigged price)
            if ev_per_unit <= 0:
                continue
            winner = obs["winner"]
            rows.append({
                "source": obs["source"], "tour": obs["tour"], "round": obs["round_label"], "date": obs["date"],
                "bet_on": player, "opponent": other_player,
                "model_prob": model_p, "market_prob": market_p, "ev_per_unit": ev_per_unit,
                "decimal_odds": 1 / market_p, "winner": winner,
                "won": (winner == player) if winner is not None else None,
            })
    return pd.DataFrame(rows)


def size_and_settle(opps):
    opps = opps.copy()
    opps["kelly_f_raw"] = opps.apply(lambda r: kelly_fraction(r["model_prob"], r["market_prob"]), axis=1)
    opps["kelly_f_raw"] = opps["kelly_f_raw"].clip(lower=0)

    for frac in KELLY_FRACTIONS:
        stake_col = f"stake_kelly_{frac}"
        opps[stake_col] = opps["kelly_f_raw"] * frac * REFERENCE_BANKROLL
    opps["stake_flat"] = FLAT_STAKE

    for label in [f"kelly_{f}" for f in KELLY_FRACTIONS] + ["flat"]:
        stake_col = f"stake_{label}"
        opps[f"pnl_{label}"] = opps.apply(
            lambda r, sc=stake_col: (
                r[sc] * (r["decimal_odds"] - 1) if r["won"] is True
                else -r[sc] if r["won"] is False
                else float("nan")  # outcome unknown (in-play row with no recorded winner)
            ), axis=1,
        )
    return opps


def summarize(opps, label_col_prefix):
    settled = opps[opps["won"].notna()]
    out = {}
    for label in [f"kelly_{f}" for f in KELLY_FRACTIONS] + ["flat"]:
        staked = settled[f"stake_{label}"].sum()
        pnl = settled[f"pnl_{label}"].sum()
        out[label] = {
            "n_bets": len(settled), "n_wins": int(settled["won"].sum()),
            "win_rate": settled["won"].mean() if len(settled) else float("nan"),
            "total_staked": staked, "total_pnl": pnl,
            "roi_pct": (pnl / staked * 100) if staked > 0 else float("nan"),
        }
    return out


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    if not LOG_PATH.exists():
        sys.exit(f"ERROR: {LOG_PATH} doesn't exist - run calibration_log.py first.")

    opps = build_opportunity_rows()
    print(f"\nGenuine +EV opportunities found in Cincinnati's tracked-market-price matches: {len(opps)}")
    if len(opps) == 0:
        print("Zero +EV opportunities exist in the tracked data - nothing to paper-trade. Stopping "
              "here rather than reporting synthetic numbers.")
        return

    opps = size_and_settle(opps)
    unresolved = opps["won"].isna().sum()
    if unresolved:
        print(f"({unresolved} of {len(opps)} opportunities have no recorded real outcome yet - "
              f"an in-play snapshot for a match not yet finished - excluded from P&L below, shown "
              f"separately.)")

    print("\n--- Every +EV opportunity found ---")
    display = opps[["source", "tour", "round", "bet_on", "opponent", "model_prob", "market_prob",
                     "ev_per_unit", "decimal_odds", "won"]].copy()
    print(display.to_string(index=False, formatters={
        "model_prob": "{:.1%}".format, "market_prob": "{:.1%}".format, "ev_per_unit": "{:+.1%}".format,
        "decimal_odds": "{:.2f}".format,
    }))

    summary = summarize(opps, "")
    print(f"\n--- Results (n={summary['flat']['n_bets']} settled bets; reference bankroll = "
          f"{REFERENCE_BANKROLL:.0f} units for Kelly sizing, {FLAT_STAKE:.0f}-unit flat stake) ---")
    header = f"{'Method':<22} {'Bets':>5} {'Wins':>5} {'Win%':>7} {'Staked':>10} {'P&L':>10} {'ROI%':>8}"
    print(header)
    print("-" * len(header))
    labels = {**{f"kelly_{f}": f"Fractional Kelly {f}x" for f in KELLY_FRACTIONS}, "flat": "Flat stake"}
    for key, label in labels.items():
        s = summary[key]
        print(f"{label:<22} {s['n_bets']:>5} {s['n_wins']:>5} {s['win_rate']:>6.1%} "
              f"{s['total_staked']:>10.2f} {s['total_pnl']:>+10.2f} {s['roi_pct']:>+7.1f}%")

    # --- sensitivity: does the headline result depend on the single Nakashima B. bet? ---
    # (market had him at 18.6% implied, decimal odds 5.38, he won - by far the largest single
    # WINNING payout in the set; O Connell C.'s 7.30-odds bet is nominally larger but LOST, so
    # removing it would make every method look BETTER, not test the same "one lucky bet" concern)
    settled_all = opps[opps["won"].notna()]
    nakashima_rows = settled_all[settled_all["bet_on"] == "Nakashima B."]
    if len(nakashima_rows) != 1:
        print(f"\nWARNING: expected exactly 1 settled Nakashima B. bet, found {len(nakashima_rows)} - "
              f"skipping the sensitivity check.", file=sys.stderr)
    else:
        outlier_idx = nakashima_rows.index[0]
        outlier = settled_all.loc[outlier_idx]
        opps_excl = opps.drop(index=outlier_idx)
        summary_excl = summarize(opps_excl, "")

        print(f"\n--- Sensitivity: with vs. without the Nakashima B. bet "
              f"({outlier['bet_on']} over {outlier['opponent']}, decimal odds {outlier['decimal_odds']:.2f}, "
              f"{'won' if outlier['won'] else 'lost'}) ---")
        header2 = f"{'Method':<22} {'P&L (with)':>12} {'P&L (w/o)':>12} {'ROI% (with)':>13} {'ROI% (w/o)':>12}"
        print(header2)
        print("-" * len(header2))
        flips = []
        for key, label in labels.items():
            s_with, s_wo = summary[key], summary_excl[key]
            print(f"{label:<22} {s_with['total_pnl']:>+12.2f} {s_wo['total_pnl']:>+12.2f} "
                  f"{s_with['roi_pct']:>+12.1f}% {s_wo['roi_pct']:>+11.1f}%")
            if s_with["total_pnl"] > 0 and s_wo["total_pnl"] <= 0:
                flips.append(label)

        if flips:
            print(f"\nFLIPS PROFITABLE -> UNPROFITABLE without this one bet: {', '.join(flips)}. "
                  f"The headline result for {'these methods' if len(flips) > 1 else 'this method'} "
                  f"depends entirely on the single Nakashima B. outcome, not a broad-based edge "
                  f"across the sample.")
        else:
            print(f"\nStill profitable under every method with this bet removed (P&L: "
                  + ", ".join(f"{labels[k]} {summary_excl[k]['total_pnl']:+.2f}" for k in labels)
                  + f") - but with n={summary_excl['flat']['n_bets']} remaining bets and win rate down "
                  f"to {summary_excl['flat']['win_rate']:.1%}, this is still a thin sample, not proof "
                  f"the edge holds without this specific result.")

    print(f"\nASSUMPTIONS (stated plainly, not buried):")
    print(f"  - Fractional Kelly at 0.25x/0.5x of full Kelly - full Kelly is widely considered too "
          f"aggressive for real bankroll variance; these are real, deliberately conservative fractions.")
    print(f"  - Kelly stakes are sized against a FIXED {REFERENCE_BANKROLL:.0f}-unit reference "
          f"bankroll, not compounded sequentially - these opportunities span real-world-concurrent "
          f"matches (same tournament, overlapping days), not a strict single-bankroll sequence, so "
          f"compounding would just inject an arbitrary ordering effect.")
    print(f"  - Every bet is settled at the market's DE-VIGGED implied price (decimal odds = "
          f"1/market_prob) - the only price this project has ever computed/stored. A real "
          f"placed bet would face a worse, vig-inclusive price, so real P&L would run somewhat "
          f"below what's reported here - this is a best-case backtest, not a live-account claim.")
    print(f"  - Sample size is small ({summary['flat']['n_bets']} settled bets) because Cincinnati's "
          f"pregame market cache only ever captured 9 matches (all ATP, all Round 3/4) before this "
          f"backtest was run, and no live_match_snapshot.py in-play observations for Cincinnati "
          f"exist yet - the numbers above should be read as a real, small pilot, not a statistically "
          f"powered claim about the model's betting edge.")


if __name__ == "__main__":
    main()
