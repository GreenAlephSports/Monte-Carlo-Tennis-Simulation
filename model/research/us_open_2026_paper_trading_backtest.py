"""US Open 2026 paper-trading backtest, run under the LOCKED methodology in
us_open_2026_backtest_methodology.md (commit 82e0211, locked 2026-09-14 01:51 UTC, before any USO
backtest/evaluation code existed). Nothing here re-derives or re-tunes the filter or stake rule:

  - Data-quality filter (MIN_HARD_MATCHES, MIN_EDGE_PP) and stake constants (FLAT_STAKE,
    KELLY_FRACTIONS, REFERENCE_BANKROLL) are imported directly from
    cincinnati_paper_trading_backtest_tennisdata.py - the same module canadian_open_paper_trading_
    backtest.py already reuses - so this script cannot silently drift from the tested min_edge_pp
    logic. edge_pp is computed against the de-vigged consensus market_prob (never against the real
    settlement price - that fair-price-vs-real-price separation is the Cincinnati grading-price
    correction) inside build_opportunity_rows_for_tour below, in the exact same shape as
    canadian_open_paper_trading_backtest.py's own build_opportunity_rows_for_tour (which is itself
    not imported from Cincinnati only because it hardcodes Cincinnati's bracket/CSV paths - same
    precedent followed here, not a new pattern).
  - The calibration machinery (Brier score, log-loss, player-clustered bootstrap CI, the primary
    metrics per Section 3 of the locked methodology) is imported directly from
    canadian_open_paper_trading_backtest.py, where it was first written and tested - not
    reimplemented here either.

Data: data/us_open_2026_{atp,wta}_filtered.csv - filtered out of the full-season
data/2026_{atp,wta}_raw.xlsx.xlsx files (Tournament == "US Open"). Verified before this script was
written: single Tournament string ("US Open", no naming variants in either file), single Location
("New York") for every row, full round coverage through The Final, 127 rows each tour, dates
2026-08-30 to 2026-09-13 (ATP) / 09-12 (WTA), zero null MaxW/MaxL. This replaces the earlier
data/atp_2026_usopen_tennisdata.csv cache, which was confirmed to actually contain Australian Open
matches under a misleading filename - not used here.

Model probability uses Elo FROZEN at start_date=2026-08-30 (brackets/us_open_2026_atp_real.yaml and
us_open_2026_wta_real.yaml), via calibration_log._prepare_ratings - the same frozen-ratings
discipline every other backtest in this project uses, so no result here leaks the tournament's own
outcomes back into its own prediction.

Per Section 4 of the locked methodology, this script reports, without exception: n (filtered and
unfiltered), Brier/log-loss model vs. market on both sets (primary), and ROI/win-rate flat + both
Kelly fractions (secondary) - in that order, with no adjustment to the filter or stake rule based on
how any of it comes out.

Usage:
    python model/research/us_open_2026_paper_trading_backtest.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import match_name_to_pool  # noqa: E402
from bracket_schema import load_bracket_yaml  # noqa: E402
from calibration_log import _prepare_ratings  # noqa: E402
from canadian_open_paper_trading_backtest import print_calibration_table  # noqa: E402
from cincinnati_paper_trading_backtest_tennisdata import (  # noqa: E402
    FLAT_STAKE, KELLY_FRACTIONS, MIN_EDGE_PP, MIN_HARD_MATCHES, REFERENCE_BANKROLL,
    print_summary_table, size_and_settle, summarize,
)
from ev_comparison import implied_probabilities  # noqa: E402
from win_probability import win_probability  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CSV_PATHS = {
    "ATP": DATA_DIR / "us_open_2026_atp_filtered.csv",
    "WTA": DATA_DIR / "us_open_2026_wta_filtered.csv",
}
BRACKET_PATHS = {
    "ATP": Path("brackets/us_open_2026_atp_real.yaml"),
    "WTA": Path("brackets/us_open_2026_wta_real.yaml"),
}


def build_opportunity_rows_for_tour(tour):
    """Identical shape to canadian_open_paper_trading_backtest.build_opportunity_rows_for_tour
    (itself identical in shape to cincinnati_paper_trading_backtest_tennisdata's version) - not
    imported directly because that function hardcodes the Canadian Open bracket/CSV paths, but
    every downstream field (model_prob, market_prob de-vigged, real_decimal_odds from MaxW/MaxL,
    edge_pp, hard_matches) is computed the exact same way, using the exact same shared
    kelly_fraction/size_and_settle/summarize functions at settlement time."""
    bracket = load_bracket_yaml(BRACKET_PATHS[tour])
    tour_config, draw, _matches_history = _prepare_ratings(bracket)  # frozen-at-start_date ratings

    ratings_df = pd.read_csv(tour_config.ratings_path).set_index("player")
    hard_matches = ratings_df["hard_matches"].to_dict()

    csv_path = CSV_PATHS[tour]
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"{tour}: {len(df)} real US Open matches loaded from {csv_path.name} "
          f"(rounds: {df['Round'].value_counts().to_dict()})")

    pool = set(draw)
    rows = []
    unresolved = set()
    for row in df.itertuples(index=False):
        winner_raw, loser_raw = row.Winner, row.Loser
        avg_w, avg_l = row.AvgW, row.AvgL
        max_w, max_l = row.MaxW, row.MaxL
        if pd.isna(avg_w) or pd.isna(avg_l) or pd.isna(max_w) or pd.isna(max_l):
            continue  # no usable real closing price for this match

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
            ev_per_unit = model_p / market_p - 1
            if ev_per_unit <= 0:
                continue
            rows.append({
                "tour": tour, "round": row.Round, "date": row.Date,
                "bet_on": player, "opponent": opponent,
                "model_prob": model_p, "market_prob": market_p, "ev_per_unit": ev_per_unit,
                # edge_pp computed against the de-vigged consensus market_prob - NEVER against
                # real_decimal_odds/real settlement price. This is the fix confirmed present here:
                # the same field, computed the same way, as the already-tested Cincinnati/Canadian
                # Open scripts - not reimplemented with different inputs.
                "edge_pp": (model_p - market_p) * 100,
                "decimal_odds": 1 / market_p,          # fair/de-vigged - reference only, never settled at
                "real_decimal_odds": real_odds,         # what settlement actually uses
                "won": won,
                "player_hard_matches": hard_matches.get(player),
                "opponent_hard_matches": hard_matches.get(opponent),
                "min_hard_matches": min(hard_matches.get(player, 0) or 0, hard_matches.get(opponent, 0) or 0),
            })

    if unresolved:
        print(f"  WARNING: {len(unresolved)} {tour} name(s) unresolved: {sorted(unresolved)}", file=sys.stderr)
    return pd.DataFrame(rows)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    all_opps = pd.concat([build_opportunity_rows_for_tour(t) for t in ("ATP", "WTA")], ignore_index=True)
    print(f"\nGenuine +EV opportunities found across all real, priced US Open matches "
          f"(both tours, every round): {len(all_opps)}")
    if len(all_opps) == 0:
        print("Zero +EV opportunities exist - nothing to paper-trade. Stopping here.")
        return

    print(f"  By tour: {all_opps['tour'].value_counts().to_dict()}")
    print(f"  By round: {all_opps['round'].value_counts().to_dict()}")

    filtered_opps = all_opps[
        (all_opps["min_hard_matches"] >= MIN_HARD_MATCHES) & (all_opps["edge_pp"] >= MIN_EDGE_PP)
    ]

    # --- PRIMARY (locked methodology Section 3.1-3.2): Brier score & log-loss, model vs. market ---
    print_calibration_table(all_opps, f"=== PRIMARY: calibration, UNFILTERED (n={len(all_opps)}) ===")
    print_calibration_table(
        filtered_opps,
        f"=== PRIMARY: calibration, FILTERED (min_hard_matches>={MIN_HARD_MATCHES}, "
        f"min_edge_pp>={MIN_EDGE_PP:.0f}, n={len(filtered_opps)}) ===",
    )
    print("\n  Read this the same way as the Cincinnati/Canadian Open baselines: this opportunity "
          "set is constructed to be exactly where the model disagrees with the market, so it is a "
          "biased sample for a calibration comparison by design - a confirmation of no gross "
          "miscalibration on the disagreement subset, not a claim of beating market calibration "
          "overall.")

    # --- SECONDARY (locked methodology Section 3.3): ROI / win rate, real-price settlement ---
    labels = {**{f"kelly_{f}": f"Fractional Kelly {f}x" for f in KELLY_FRACTIONS}, "flat": "Flat stake"}

    opps_unfiltered = size_and_settle(all_opps, price_col="real_decimal_odds")
    print_summary_table(
        summarize(opps_unfiltered), labels,
        f"--- SECONDARY: ROI, UNFILTERED, real-price settlement (n={len(opps_unfiltered)}) ---",
    )

    opps = size_and_settle(filtered_opps, price_col="real_decimal_odds")
    summary = summarize(opps)
    print_summary_table(
        summary, labels,
        f"--- SECONDARY: ROI, FILTERED, real-price settlement (n={summary['flat']['n_bets']}) ---",
    )

    for tour in ("ATP", "WTA"):
        tour_opps = opps[opps["tour"] == tour]
        if len(tour_opps) == 0:
            print(f"\n{tour}: 0 +EV opportunities survive the data-quality filter.")
            continue
        print_summary_table(summarize(tour_opps), labels, f"--- {tour} only, filtered (n={len(tour_opps)}) ---")

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
              f"real decimal odds {outlier['real_decimal_odds']:.2f}, won) ---")
        for key, label in labels.items():
            s_with, s_wo = summary[key], summary_excl[key]
            print(f"  {label:<22} P&L with {s_with['total_pnl']:>+8.2f}  w/o {s_wo['total_pnl']:>+8.2f}   "
                  f"ROI with {s_with['roi_pct']:>+7.1f}%  w/o {s_wo['roi_pct']:>+7.1f}%")

    print(f"\nASSUMPTIONS (stated plainly, not buried):")
    print(f"  - Filter (min_hard_matches>={MIN_HARD_MATCHES}, min_edge_pp>={MIN_EDGE_PP:.0f}) and "
          f"stake rule (flat {FLAT_STAKE:.0f}-unit, Kelly {'/'.join(f'{f}x' for f in KELLY_FRACTIONS)} "
          f"against a fixed {REFERENCE_BANKROLL:.0f}-unit non-compounding reference bankroll) are "
          f"imported unchanged from cincinnati_paper_trading_backtest_tennisdata.py per the LOCKED "
          f"methodology (us_open_2026_backtest_methodology.md, commit 82e0211) - not re-tuned on US "
          f"Open data.")
    print(f"  - Every bet settles at the real best-obtainable closing price (MaxW/MaxL) - the "
          f"de-vigged consensus is signal only (edge_pp, filter, Kelly sizing), never settlement.")
    print(f"  - Model probability uses Elo FROZEN at start_date=2026-08-30 for both tours (per the "
          f"real bracket YAMLs' own start_date field, matching the real first-round match dates).")
    print(f"  - Pointer: us_open_2026_backtest_methodology.md governs this analysis; any future "
          f"change to filter/stake/metrics must be recorded there as a dated amendment, never a "
          f"silent edit to this script.")


if __name__ == "__main__":
    main()
