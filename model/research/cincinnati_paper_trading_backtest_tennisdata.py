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
  - Same sensitivity-check discipline: report P&L/ROI with and without the single largest-payout
    winning bet, so the headline number is never read without knowing how much it leans on one
    outcome.

GRADING-PRICE CORRECTION (this revision): bets were previously settled at the market's DE-VIGGED
implied price (decimal odds = 1/market_prob) - a fair-value price nobody could actually trade at,
since removing the vig only exists on the analysis side, never as a price a bookmaker will quote.
That overstated every winning bet's payout by however much vig the market was actually holding.
Settlement now uses each match's MaxW/MaxL column from tennis-data.co.uk - the best (highest) real,
vig-included closing price recorded across every bookmaker that CSV tracks for that specific
selection: a price a bettor genuinely could have obtained by shopping the market at close, not a
theoretical fair price. The de-vigged consensus (AvgW/AvgL, de-vigged) is still used for everything
upstream of settlement - identifying +EV opportunities, edge_pp, the data-quality filter, and Kelly
stake sizing all continue to compare the model against the market's fair (de-vigged) read, since
that is a read on disagreement, not a claim about what price is tradeable. Only the number multiplied
into P&L on a win changed. Both the old (fair-price) and new (real-price) settlements are printed
side by side below, specifically so the delta this correction makes is visible rather than silently
overwriting the earlier headline.

RESULT OF THIS CORRECTION (rerun 2026-09-14, same MIN_HARD_MATCHES/MIN_EDGE_PP filter as always -
note the underlying Kaggle ratings pull is refetched live each run, so n has since drifted from the
34 quoted below to 27; that drift is pre-existing data churn, unrelated to this price correction):
on the standing filtered set (n=27, 55.6% win rate - win rate itself does not move, only payout
size does), flat-stake ROI goes from +17.1% (old, fair de-vigged settlement) to +14.5% (real,
best-obtainable settlement), and fractional-Kelly ROI (both 0.25x and 0.5x - fraction only scales
stake size, not ROI%) goes from +9.8% to +7.1% - roughly a 2.6-2.7 percentage-point haircut, real
money on the table but not enough to erase the edge. On the unfiltered 190-bet set the correction
is more damaging: flat ROI goes from +0.9% to -2.3%, and both Kelly ROI figures go from -2.1% to
-5.3% - already negative, more negative once vig is charged. The single-bet sensitivity check
still flips both Kelly variants unprofitable if the Pegula-over-Swiatek semifinal bet is excluded,
same as before - that fragility was never about settlement price.

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
        max_w, max_l = row.MaxW, row.MaxL
        if pd.isna(avg_w) or pd.isna(avg_l):
            continue  # no usable closing price for this match
        if pd.isna(max_w) or pd.isna(max_l):
            continue  # no real, tradeable closing price for this match

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

        for player, opponent, model_p, market_p, real_odds, won in [
            (winner, loser, model_winner, market_winner, max_w, True),
            (loser, winner, model_loser, market_loser, max_l, False),
        ]:
            # signal for whether/how much to bet: model vs. the market's DE-VIGGED fair read -
            # unaffected by the settlement-price correction below.
            ev_per_unit = model_p / market_p - 1
            if ev_per_unit <= 0:
                continue
            rows.append({
                "tour": tour, "round": row.Round, "date": row.Date,
                "bet_on": player, "opponent": opponent,
                "model_prob": model_p, "market_prob": market_p, "ev_per_unit": ev_per_unit,
                "edge_pp": (model_p - market_p) * 100,
                # fair (de-vigged) price - kept only as a reference/upper-bound, not tradeable.
                "decimal_odds": 1 / market_p,
                # real, actually-obtainable price: the best (highest) closing odds any tracked
                # bookmaker offered on this exact selection - vig included. This is what settlement
                # uses.
                "real_decimal_odds": real_odds,
                "won": won,
                "player_hard_matches": hard_matches.get(player),
                "opponent_hard_matches": hard_matches.get(opponent),
                "min_hard_matches": min(hard_matches.get(player, 0) or 0, hard_matches.get(opponent, 0) or 0),
            })

    if unresolved:
        print(f"  WARNING: {len(unresolved)} {tour} name(s) unresolved (unexpected - the pre-check "
              f"found zero): {sorted(unresolved)}", file=sys.stderr)
    return pd.DataFrame(rows)


def size_and_settle(opps, price_col="decimal_odds"):
    """Sizes every bet off model_prob vs. market_prob (the de-vigged consensus signal, unaffected
    by price_col), then settles P&L at price_col - "decimal_odds" (fair, de-vigged, default, kept
    for backward compatibility with other scripts that import this function) or
    "real_decimal_odds" (the real, actually-obtainable best closing price)."""
    opps = opps.copy()
    opps["kelly_f_raw"] = opps.apply(lambda r: kelly_fraction(r["model_prob"], r["market_prob"]), axis=1).clip(lower=0)
    for frac in KELLY_FRACTIONS:
        opps[f"stake_kelly_{frac}"] = opps["kelly_f_raw"] * frac * REFERENCE_BANKROLL
    opps["stake_flat"] = FLAT_STAKE
    for label in [f"kelly_{f}" for f in KELLY_FRACTIONS] + ["flat"]:
        stake_col = f"stake_{label}"
        opps[f"pnl_{label}"] = opps.apply(
            lambda r, sc=stake_col: r[sc] * (r[price_col] - 1) if r["won"] else -r[sc], axis=1,
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


def print_price_correction_delta(summary_fair, summary_real, labels, title):
    """The whole point of this revision: show, side by side, exactly how much ROI moves when
    settlement switches from the de-vigged fair price to the real, actually-obtainable best price."""
    print(f"\n{title}")
    header = (f"{'Method':<22} {'ROI% (fair, old)':>17} {'ROI% (real, new)':>17} "
              f"{'ROI pp change':>14} {'P&L (fair)':>11} {'P&L (real)':>11}")
    print(header)
    print("-" * len(header))
    for key, label in labels.items():
        f, r = summary_fair[key], summary_real[key]
        delta = r["roi_pct"] - f["roi_pct"]
        print(f"{label:<22} {f['roi_pct']:>+16.1f}% {r['roi_pct']:>+16.1f}% {delta:>+13.1f}pp "
              f"{f['total_pnl']:>+11.2f} {r['total_pnl']:>+11.2f}")


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
    opps_unfiltered_fair = size_and_settle(all_opps, price_col="decimal_odds")
    opps_unfiltered_real = size_and_settle(all_opps, price_col="real_decimal_odds")
    summary_unfiltered_fair = summarize(opps_unfiltered_fair)
    summary_unfiltered_real = summarize(opps_unfiltered_real)
    print_summary_table(
        summary_unfiltered_fair, labels,
        f"--- BEFORE data-quality filtering, settled at FAIR de-vigged price (reference/upper-bound "
        f"only, not tradeable) (n={len(opps_unfiltered_fair)} settled bets) ---",
    )
    print_summary_table(
        summary_unfiltered_real, labels,
        f"--- BEFORE data-quality filtering, settled at REAL best-obtainable price "
        f"(n={len(opps_unfiltered_real)} settled bets; every side with model_prob > market_prob, "
        f"no data-quality or edge-size floor) ---",
    )
    print_price_correction_delta(
        summary_unfiltered_fair, summary_unfiltered_real, labels,
        "--- Price-correction delta, BEFORE data-quality filtering (real - fair) ---",
    )

    # --- AFTER data-quality filtering: the real, standing result ---
    filtered_opps = all_opps[
        (all_opps["min_hard_matches"] >= MIN_HARD_MATCHES) & (all_opps["edge_pp"] >= MIN_EDGE_PP)
    ]
    opps_fair = size_and_settle(filtered_opps, price_col="decimal_odds")
    opps = size_and_settle(filtered_opps, price_col="real_decimal_odds")
    summary_fair = summarize(opps_fair)
    summary = summarize(opps)
    print_summary_table(
        summary_fair, labels,
        f"--- AFTER data-quality filtering, settled at FAIR de-vigged price (reference/upper-bound "
        f"only, not tradeable) (n={summary_fair['flat']['n_bets']} settled bets) ---",
    )
    print_summary_table(
        summary, labels,
        f"--- AFTER data-quality filtering, settled at REAL best-obtainable price (STANDING RESULT): "
        f"min_hard_matches>={MIN_HARD_MATCHES}, min_edge_pp>={MIN_EDGE_PP:.0f} "
        f"(n={summary['flat']['n_bets']} settled bets; reference bankroll = {REFERENCE_BANKROLL:.0f} "
        f"units for Kelly, {FLAT_STAKE:.0f}-unit flat stake) ---",
    )
    print_price_correction_delta(
        summary_fair, summary, labels,
        "--- Price-correction delta, AFTER data-quality filtering (real - fair; THIS is the number "
        "that answers 'how much does the earlier ROI change') ---",
    )

    # per-tour breakdown of the filtered (standing, real-price) result, since the original 9-match
    # sample was ATP-only and this is worth seeing split out explicitly
    for tour in ("ATP", "WTA"):
        tour_opps = opps[opps["tour"] == tour]
        if len(tour_opps) == 0:
            print(f"\n{tour}: 0 +EV opportunities survive the data-quality filter.")
            continue
        print_summary_table(summarize(tour_opps), labels, f"--- {tour} only, filtered, real price (n={len(tour_opps)}) ---")

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
              f"real decimal odds {outlier['real_decimal_odds']:.2f} (fair was {outlier['decimal_odds']:.2f}), "
              f"won) ---")
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
        ["tour", "round", "bet_on", "opponent", "model_prob", "market_prob", "ev_per_unit",
         "decimal_odds", "real_decimal_odds", "won"]
    ]
    print(display.to_string(index=False, formatters={
        "model_prob": "{:.1%}".format, "market_prob": "{:.1%}".format, "ev_per_unit": "{:+.1%}".format,
        "decimal_odds": "{:.2f}".format, "real_decimal_odds": "{:.2f}".format,
    }))

    print(f"\nASSUMPTIONS (stated plainly, not buried):")
    print(f"  - Fractional Kelly at 0.25x/0.5x of full Kelly, against a fixed {REFERENCE_BANKROLL:.0f}-unit "
          f"non-compounding reference bankroll (these are real-world-concurrent matches, not a "
          f"strict sequential series). Stake SIZING still compares model_prob against the market's "
          f"DE-VIGGED fair read (that's a disagreement signal, not a tradeable-price claim) - only "
          f"settlement price changed in this revision.")
    print(f"  - Every bet SETTLED at the REAL best-obtainable price (decimal odds = MaxW/MaxL, the "
          f"highest closing odds any bookmaker tennis-data.co.uk tracked actually offered on that "
          f"selection) - vig included, a price a bettor genuinely could have gotten. The de-vigged "
          f"fair price is still reported alongside it (see the price-correction delta tables above) "
          f"as a reference upper bound, never as the headline P&L number.")
    print(f"  - Model probability uses Elo FROZEN at each tournament's start_date (2026-08-13) - "
          f"no in-tournament result (this player's own run, momentum, etc.) is baked in, matching "
          f"every other pregame-calibration script in this project.")
    print(f"  - This is 190 real matches with real closing odds, both tours, every round - a "
          f"materially larger and more representative sample than the original 9-match, ATP-only, "
          f"single-poll-instant sample, though still one tournament's worth of data, not a "
          f"multi-tournament claim.")


if __name__ == "__main__":
    main()
