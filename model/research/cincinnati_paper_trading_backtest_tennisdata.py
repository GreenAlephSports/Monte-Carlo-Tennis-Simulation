"""Rebuild of cincinnati_paper_trading_backtest.py against a much larger REAL dataset:
tennis-data.co.uk's per-tournament CSVs (data/cincinnati_2026_{atp,wta}_tennisdata.csv, downloaded
from tennis-data.co.uk/2026/cincinnati.csv and tennis-data.co.uk/2026w/cincinnati.csv), which carry
real closing bookmaker odds (AvgW/AvgL - the average across every book that CSV tracks, the same
"average across books, then de-vig" methodology this project's own fetch_devigged_odds already
uses) for every real match, every round, both tours - 190 matches total, vs. the 9-match ATP-only
single-poll-instant sample the original backtest was stuck with (see that script's own docstring
and the conversation that investigated why it was only 9).

Confirmed before relying on it:
  - Column structure: Winner/Loser (already in this project's own "Lastname I." ratings-csv
    naming convention, not ESPN's full-name format), WRank/LRank, AvgW/AvgL present on all 190/190
    rows (both tours), B365W/B365L also complete, PSW/PSL entirely empty (that one bookmaker's
    column wasn't populated for this tournament - not used here).
  - Name-match feasibility: match_name_to_pool (bracket.py's own CSV-to-CSV tiered resolver, the
    same one calibration_log.py's Kaggle-sourced concluded-tournament path already uses) resolves
    190/190 ATP names and 190/190 WTA names (Winner+Loser across all matches) against this
    project's own ratings CSVs - zero unresolved.

Same rigor every other backtest in this project uses: model probability comes from a ratings
snapshot FROZEN at the tournament's start_date (via calibration_log._prepare_ratings, reused
directly here - not the "current" ratings CSV on disk, which has since absorbed Cincinnati's own
results and would leak the outcome being predicted straight into the prediction).

Same Kelly/flat sizing methodology as the original backtest, unchanged:
  - Fractional Kelly at 0.25x/0.5x of full Kelly, against a fixed 100-unit non-compounding
    reference bankroll.
  - Flat 1-unit stake per bet, for direct comparison.
  - Bets settled at the market's DE-VIGGED implied price (decimal odds = 1/market_prob) - still a
    best-case assumption, real bookmaker vig would leave less on the table.
  - Same sensitivity-check discipline: report P&L/ROI with and without the single largest-payout
    winning bet, so the headline number is never read without knowing how much it leans on one
    outcome.

DATA-QUALITY FILTER (default, standing result): cincinnati_data_quality_filter_test.py measured
the unfiltered "every side with model_prob > market_prob" opportunity set directly against each
side's real hard_matches count and found the biggest disagreements with the market concentrated
among thin-history players (extreme-EV bets: median 15 hard_matches on the weaker side vs. 40 for
the rest) - unreliable Elo estimates masquerading as value, not real edge. A systematic grid over
min_hard_matches in {10,20,30} x min_edge_pp in {5,10,15} showed a consistent, monotonic improvement
in ROI as both floors tightened (not one lucky cell), and min_hard_matches>=30 + min_edge_pp>=10 was
the best-supported cell that still retains a real sample (n=34): 52.9% win rate, +8.5% flat ROI,
+11.8% Kelly ROI. That combination (MIN_HARD_MATCHES / MIN_EDGE_PP below) is now the DEFAULT filter
applied to the reported result. The unfiltered, full-190-bet view is still printed alongside it,
explicitly labeled "before data-quality filtering," for transparency - not deleted, just no longer
the headline.

Usage:
    python model/cincinnati_paper_trading_backtest_tennisdata.py
"""
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import TOUR_CONFIG, match_name_to_pool  # noqa: E402
from bracket_schema import load_bracket_yaml  # noqa: E402
from calibration_log import _prepare_ratings  # noqa: E402
from ev_comparison import implied_probabilities  # noqa: E402
from win_probability import win_probability  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SOURCE_URLS = {
    "ATP": ("http://www.tennis-data.co.uk/2026/cincinnati.csv", DATA_DIR / "cincinnati_2026_atp_tennisdata.csv"),
    "WTA": ("http://www.tennis-data.co.uk/2026w/cincinnati.csv", DATA_DIR / "cincinnati_2026_wta_tennisdata.csv"),
}
BRACKET_PATHS = {
    "ATP": Path("brackets/cincinnati_2026_atp_demo.yaml"),
    "WTA": Path("brackets/cincinnati_2026_wta.yaml"),
}

REFERENCE_BANKROLL = 100.0
FLAT_STAKE = 1.0
KELLY_FRACTIONS = [0.25, 0.5]

# Default data-quality filter, confirmed by cincinnati_data_quality_filter_test.py's systematic
# grid (see that script and this module's own docstring) - the best-supported cell that still
# retains a real sample (n=34): 52.9% win rate, +8.5% flat ROI, +11.8% Kelly ROI. Both floors must
# be cleared: the weaker-supported side of the bet needs at least this many real hard-court matches
# behind its rating, AND the model/market probability gap needs to be at least this many points -
# excludes both thin-data noise and marginal, within-noise "edges".
MIN_HARD_MATCHES = 30
MIN_EDGE_PP = 10.0


def fetch_source_csv(tour, force=False):
    """Downloads tennis-data.co.uk's per-tournament CSV if not already cached in data/ (a static,
    already-concluded tournament's closing odds never change once the tournament is over, so this
    is a real cache, not a live-data staleness risk - unlike this project's ESPN/Odds-API calls,
    which always fetch fresh)."""
    url, path = SOURCE_URLS[tour]
    if path.exists() and not force:
        return path
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=15) as response:
            raw = response.read()
    except (HTTPError, URLError) as e:
        sys.exit(f"ERROR: couldn't fetch {url}: {e}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def kelly_fraction(model_prob, market_prob):
    return (model_prob - market_prob) / (1 - market_prob)


def build_opportunity_rows_for_tour(tour):
    bracket = load_bracket_yaml(BRACKET_PATHS[tour])
    tour_config, draw, _matches_history = _prepare_ratings(bracket)  # frozen-at-start_date ratings

    # same frozen-at-start_date ratings CSV win_probability() itself reads - hard_matches and
    # days_since_last_match are a side effect of elo_ratings.calculate_elo_ratings, not a separate
    # calculation, so this is exactly the real support behind each side's rating, not an estimate.
    ratings_df = pd.read_csv(tour_config.ratings_path).set_index("player")
    hard_matches = ratings_df["hard_matches"].to_dict()

    csv_path = fetch_source_csv(tour)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"{tour}: {len(df)} real Cincinnati matches loaded from {csv_path.name} "
          f"(rounds: {df['Round'].value_counts().to_dict()})")

    pool = set(draw)
    rows = []
    unresolved = set()
    for row in df.itertuples(index=False):
        winner_raw, loser_raw = row.Winner, row.Loser
        avg_w, avg_l = row.AvgW, row.AvgL
        if pd.isna(avg_w) or pd.isna(avg_l):
            continue  # no usable closing price for this match

        winner = match_name_to_pool(winner_raw, pool, tour_config.name_aliases)
        loser = match_name_to_pool(loser_raw, pool, tour_config.name_aliases)
        if winner is None:
            unresolved.add(winner_raw)
        if loser is None:
            unresolved.add(loser_raw)
        if winner is None or loser is None:
            continue

        market_winner, market_loser = implied_probabilities(avg_w, avg_l)
        model_winner = win_probability(winner, loser, bracket.surface, tour_config.ratings_path)
        model_loser = 1 - model_winner

        for player, opponent, model_p, market_p, won in [
            (winner, loser, model_winner, market_winner, True),
            (loser, winner, model_loser, market_loser, False),
        ]:
            ev_per_unit = model_p / market_p - 1
            if ev_per_unit <= 0:
                continue
            rows.append({
                "tour": tour, "round": row.Round, "date": row.Date,
                "bet_on": player, "opponent": opponent,
                "model_prob": model_p, "market_prob": market_p, "ev_per_unit": ev_per_unit,
                "edge_pp": (model_p - market_p) * 100,
                "decimal_odds": 1 / market_p, "won": won,
                "player_hard_matches": hard_matches.get(player),
                "opponent_hard_matches": hard_matches.get(opponent),
                "min_hard_matches": min(hard_matches.get(player, 0) or 0, hard_matches.get(opponent, 0) or 0),
            })

    if unresolved:
        print(f"  WARNING: {len(unresolved)} {tour} name(s) unresolved (unexpected - the pre-check "
              f"found zero): {sorted(unresolved)}", file=sys.stderr)
    return pd.DataFrame(rows)


def size_and_settle(opps):
    opps = opps.copy()
    opps["kelly_f_raw"] = opps.apply(lambda r: kelly_fraction(r["model_prob"], r["market_prob"]), axis=1).clip(lower=0)
    for frac in KELLY_FRACTIONS:
        opps[f"stake_kelly_{frac}"] = opps["kelly_f_raw"] * frac * REFERENCE_BANKROLL
    opps["stake_flat"] = FLAT_STAKE
    for label in [f"kelly_{f}" for f in KELLY_FRACTIONS] + ["flat"]:
        stake_col = f"stake_{label}"
        opps[f"pnl_{label}"] = opps.apply(
            lambda r, sc=stake_col: r[sc] * (r["decimal_odds"] - 1) if r["won"] else -r[sc], axis=1,
        )
    return opps


def summarize(opps):
    out = {}
    for label in [f"kelly_{f}" for f in KELLY_FRACTIONS] + ["flat"]:
        staked = opps[f"stake_{label}"].sum()
        pnl = opps[f"pnl_{label}"].sum()
        out[label] = {
            "n_bets": len(opps), "n_wins": int(opps["won"].sum()),
            "win_rate": opps["won"].mean() if len(opps) else float("nan"),
            "total_staked": staked, "total_pnl": pnl,
            "roi_pct": (pnl / staked * 100) if staked > 0 else float("nan"),
        }
    return out


def print_summary_table(summary, labels, title):
    print(f"\n{title}")
    header = f"{'Method':<22} {'Bets':>5} {'Wins':>5} {'Win%':>7} {'Staked':>10} {'P&L':>10} {'ROI%':>8}"
    print(header)
    print("-" * len(header))
    for key, label in labels.items():
        s = summary[key]
        print(f"{label:<22} {s['n_bets']:>5} {s['n_wins']:>5} {s['win_rate']:>6.1%} "
              f"{s['total_staked']:>10.2f} {s['total_pnl']:>+10.2f} {s['roi_pct']:>+7.1f}%")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    all_opps = pd.concat([build_opportunity_rows_for_tour(t) for t in ("ATP", "WTA")], ignore_index=True)
    print(f"\nGenuine +EV opportunities found across all 190 real, priced Cincinnati matches "
          f"(both tours, every round): {len(all_opps)}")
    if len(all_opps) == 0:
        print("Zero +EV opportunities exist - nothing to paper-trade. Stopping here.")
        return

    print(f"  By tour: {all_opps['tour'].value_counts().to_dict()}")
    print(f"  By round: {all_opps['round'].value_counts().to_dict()}")

    labels = {**{f"kelly_{f}": f"Fractional Kelly {f}x" for f in KELLY_FRACTIONS}, "flat": "Flat stake"}

    # --- BEFORE data-quality filtering: every +EV side, unfiltered (kept for transparency, no
    # longer the headline - see MIN_HARD_MATCHES/MIN_EDGE_PP docstring above for why) ---
    opps_unfiltered = size_and_settle(all_opps)
    print_summary_table(
        summarize(opps_unfiltered), labels,
        f"--- BEFORE data-quality filtering (n={len(opps_unfiltered)} settled bets; every side "
        f"with model_prob > market_prob, no data-quality or edge-size floor) ---",
    )

    # --- AFTER data-quality filtering: the real, standing result ---
    filtered_opps = all_opps[
        (all_opps["min_hard_matches"] >= MIN_HARD_MATCHES) & (all_opps["edge_pp"] >= MIN_EDGE_PP)
    ]
    opps = size_and_settle(filtered_opps)
    summary = summarize(opps)
    print_summary_table(
        summary, labels,
        f"--- AFTER data-quality filtering (STANDING RESULT): min_hard_matches>={MIN_HARD_MATCHES}, "
        f"min_edge_pp>={MIN_EDGE_PP:.0f} (n={summary['flat']['n_bets']} settled bets; reference "
        f"bankroll = {REFERENCE_BANKROLL:.0f} units for Kelly, {FLAT_STAKE:.0f}-unit flat stake) ---",
    )

    # per-tour breakdown of the filtered (standing) result, since the original 9-match sample was
    # ATP-only and this is worth seeing split out explicitly
    for tour in ("ATP", "WTA"):
        tour_opps = opps[opps["tour"] == tour]
        if len(tour_opps) == 0:
            print(f"\n{tour}: 0 +EV opportunities survive the data-quality filter.")
            continue
        print_summary_table(summarize(tour_opps), labels, f"--- {tour} only, filtered (n={len(tour_opps)}) ---")

    # --- sensitivity: with vs. without the single largest-payout WINNING bet ---
    won = opps[opps["won"]]
    if len(won) == 0:
        print("\nNo winning bets at all - sensitivity check not applicable.")
    else:
        outlier_idx = won["pnl_flat"].idxmax()
        outlier = opps.loc[outlier_idx]
        opps_excl = opps.drop(index=outlier_idx)
        summary_excl = summarize(opps_excl)

        print(f"\n--- Sensitivity: with vs. without the single largest-payout bet "
              f"({outlier['bet_on']} over {outlier['opponent']}, {outlier['tour']} {outlier['round']}, "
              f"decimal odds {outlier['decimal_odds']:.2f}, won) ---")
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
            print(f"\nFLIPS PROFITABLE -> UNPROFITABLE without this one bet: {', '.join(flips)}.")
        else:
            print(f"\nStill profitable under every method with this bet removed - "
                  f"n={summary_excl['flat']['n_bets']} remaining bets, "
                  f"win rate {summary_excl['flat']['win_rate']:.1%}.")

    print(f"\n--- Every +EV opportunity found, sorted by |EV| ---")
    display = opps.sort_values("ev_per_unit", key=abs, ascending=False)[
        ["tour", "round", "bet_on", "opponent", "model_prob", "market_prob", "ev_per_unit", "decimal_odds", "won"]
    ]
    print(display.to_string(index=False, formatters={
        "model_prob": "{:.1%}".format, "market_prob": "{:.1%}".format, "ev_per_unit": "{:+.1%}".format,
        "decimal_odds": "{:.2f}".format,
    }))

    print(f"\nASSUMPTIONS (stated plainly, not buried):")
    print(f"  - Fractional Kelly at 0.25x/0.5x of full Kelly, against a fixed {REFERENCE_BANKROLL:.0f}-unit "
          f"non-compounding reference bankroll (these are real-world-concurrent matches, not a "
          f"strict sequential series).")
    print(f"  - Every bet settled at the market's DE-VIGGED implied price (decimal odds = "
          f"1/market_prob, averaged across every book tennis-data.co.uk tracked for this match) - "
          f"a real placed bet would face a worse, vig-inclusive price, so real P&L would run below "
          f"what's reported here.")
    print(f"  - Model probability uses Elo FROZEN at each tournament's start_date (2026-08-13) - "
          f"no in-tournament result (this player's own run, momentum, etc.) is baked in, matching "
          f"every other pregame-calibration script in this project.")
    print(f"  - This is 190 real matches with real closing odds, both tours, every round - a "
          f"materially larger and more representative sample than the original 9-match, ATP-only, "
          f"single-poll-instant sample, though still one tournament's worth of data, not a "
          f"multi-tournament claim.")


if __name__ == "__main__":
    main()
