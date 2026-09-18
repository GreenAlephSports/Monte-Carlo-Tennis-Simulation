"""Canadian Open (National Bank Open presented by Rogers) equivalent of
cincinnati_paper_trading_backtest_tennisdata.py - same corrected methodology from the start, not
retrofitted:

  - Bets SETTLE at the real, best-obtainable closing price (MaxW/MaxL from tennis-data.co.uk - the
    highest quoted odds across every bookmaker that CSV tracks for that selection, vig included).
    The de-vigged consensus (AvgW/AvgL, de-vigged) is still used, as elsewhere, only as the SIGNAL
    for identifying +EV opportunities and sizing Kelly stakes - never for settlement. See
    cincinnati_paper_trading_backtest_tennisdata.py's own docstring for the full reasoning; this
    script never had the earlier de-vigged-settlement version to correct, so there's no
    before/after delta table here the way that script has one.
  - Filter and stake rule are IMPORTED directly from cincinnati_paper_trading_backtest_tennisdata
    (MIN_HARD_MATCHES, MIN_EDGE_PP, REFERENCE_BANKROLL, FLAT_STAKE, KELLY_FRACTIONS, and the
    kelly_fraction/size_and_settle/summarize/print_summary_table functions themselves) rather than
    retyped - the same discipline model/research/us_open_2026_backtest_methodology.md requires for
    the US Open evaluation ("import the shared constants module rather than restating literals").
    Nothing here is re-tuned or re-gridded against Canadian Open data.
  - Brier score and log-loss (model vs. market, de-vigged consensus) are the PRIMARY reported
    metrics, with player-clustered bootstrap 95% CIs, computed on both the unfiltered and filtered
    sets. ROI/win-rate are reported SECOND and explicitly labeled secondary - same ordering as the
    Closing Line Report's Part 1 primacy statement, not an afterthought tacked onto an ROI-first
    script.

CUTOFF-DATE VERIFICATION (done BEFORE trusting any output from this script, not after):
  Both brackets/montreal_2026.yaml (ATP) and brackets/wta_toronto_2026.yaml (WTA) declare
  start_date: 2026-08-02. _prepare_ratings -> calculate_elo_ratings freezes ratings via
  apply_training_window (ATP) / _decay3_weighted_window (WTA, since WTA is in elo_ratings.
  DECAY3_TOURS) - both filter with the identical convention already confirmed for the
  elo_k_factor_test.py Canadian Open pilot: `df[df["Date"] < pd.Timestamp(cutoff_date)]`, strict
  less-than. Checked directly against the REAL 2026 Kaggle match dates for this tournament, not
  assumed:
    ATP (Tournament == "Canadian Open" in the ATP Kaggle set): 95 real matches, 2026-08-02 to
      2026-08-14. First real match date is 2026-08-02 - excluded by strict `<` against a
      2026-08-02 cutoff, exactly as it should be.
    WTA (Tournament == "Canadian Open" in the WTA Kaggle set): 88 real matches, 2026-08-02 to
      2026-08-13. Same result: first real match date == cutoff date, excluded by strict `<`.
  No leakage on either tour: the ratings used to predict this tournament are built only from
  matches strictly before it started.

UNCONFIRMED: the WTA tennis-data.co.uk slug. The ATP slug ("montreal") is confirmed - a cached,
real, 95-row CSV with AvgW/AvgL/MaxW/MaxL already exists at data/atp_2026_montreal_tennisdata.csv,
and "montreal" appears in pedigree_market_premium_test.py's own confirmed-live TOURNAMENTS list.
No WTA Canadian Open slug appears anywhere else in this project (checked: pedigree_market_premium_
test.py's confirmed list has no WTA Canadian Open entry at all; platt_atp_drift_tracker.py's
EXPANSION_CANDIDATES is ATP-only). WTA_SLUG_CANDIDATES below are tried in order and the first that
returns a real, non-empty, AvgW/AvgL-populated CSV wins - same "best-effort guesses, not assumed
correct" discipline this project already uses elsewhere (see platt_atp_drift_tracker.py's own
EXPANSION_CANDIDATES docstring note) for exactly this situation. Whichever slug actually resolves
gets printed loudly, not silently accepted.

NOT YET RUN END TO END: tennis-data.co.uk has been returning 503 Service Temporarily Unavailable
site-wide (confirmed from three independent networks - see the US Open blocker notes) for the
duration of this work, so the fetch step below has not been exercised against a live response, and
the WTA slug is therefore still unconfirmed. Everything above the fetch step (cutoff-date logic,
filter/stake reuse, Brier/log-loss machinery) was verified independently of the network call. Do
not trust this script's printed numbers until it has actually completed a real run - if it exits
with a fetch error, that means exactly what it says, not that the tournament has no data.

Usage:
    python model/research/canadian_open_paper_trading_backtest.py
"""
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import match_name_to_pool  # noqa: E402
from bracket_schema import load_bracket_yaml  # noqa: E402
from calibration_log import _prepare_ratings  # noqa: E402
from cincinnati_paper_trading_backtest_tennisdata import (  # noqa: E402
    FLAT_STAKE, KELLY_FRACTIONS, MIN_EDGE_PP, MIN_HARD_MATCHES, REFERENCE_BANKROLL,
    kelly_fraction, print_summary_table, size_and_settle, summarize,
)
from ev_comparison import implied_probabilities  # noqa: E402
from win_probability import win_probability  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
BRACKET_PATHS = {
    "ATP": Path("brackets/montreal_2026.yaml"),
    "WTA": Path("brackets/wta_toronto_2026.yaml"),
}
ATP_SLUG = "montreal"  # confirmed live - see module docstring
WTA_SLUG_CANDIDATES = ["canadianopen", "toronto", "canada"]  # unconfirmed - see module docstring

N_BOOT = 5000
BOOT_SEED = 42


def fetch_source_csv(tour, force=False):
    """ATP: fetches the confirmed slug directly. WTA: tries WTA_SLUG_CANDIDATES in order, keeps
    the first that returns a real CSV with usable AvgW/AvgL, and prints which slug actually
    worked (never silently assumes the first guess was right)."""
    if tour == "ATP":
        path = DATA_DIR / f"atp_2026_{ATP_SLUG}_tennisdata.csv"
        if path.exists() and not force:
            return path
        return _download(f"http://www.tennis-data.co.uk/2026/{ATP_SLUG}.csv", path)

    for slug in WTA_SLUG_CANDIDATES:
        path = DATA_DIR / f"wta_2026_{slug}_tennisdata.csv"
        if path.exists() and not force:
            print(f"  WTA: using cached slug '{slug}' ({path.name})")
            return path
        try:
            resolved = _download(f"http://www.tennis-data.co.uk/2026w/{slug}.csv", path, exit_on_fail=False)
        except RuntimeError as e:
            print(f"  WTA slug '{slug}' failed: {e}", file=sys.stderr)
            continue
        try:
            probe = pd.read_csv(resolved, encoding="utf-8-sig")
        except Exception as e:
            print(f"  WTA slug '{slug}' downloaded but didn't parse as a CSV: {e}", file=sys.stderr)
            resolved.unlink(missing_ok=True)
            continue
        if "AvgW" not in probe.columns or "AvgL" not in probe.columns or probe["AvgW"].isna().all():
            print(f"  WTA slug '{slug}' parsed but has no usable AvgW/AvgL - not this tournament", file=sys.stderr)
            resolved.unlink(missing_ok=True)
            continue
        print(f"  WTA: slug '{slug}' resolved to a real {len(probe)}-row CSV with usable odds - using it")
        return resolved

    sys.exit(
        f"ERROR: none of WTA_SLUG_CANDIDATES {WTA_SLUG_CANDIDATES} resolved to a real, priced "
        f"CSV. The WTA Canadian Open slug is still unconfirmed - add the real one once known, "
        f"don't guess further here."
    )


def _download(url, path, exit_on_fail=True):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=15) as response:
            raw = response.read()
    except (HTTPError, URLError) as e:
        msg = f"couldn't fetch {url}: {e}"
        if exit_on_fail:
            sys.exit(f"ERROR: {msg}")
        raise RuntimeError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def build_opportunity_rows_for_tour(tour):
    """Identical shape to cincinnati_paper_trading_backtest_tennisdata.build_opportunity_rows_for_
    tour - not imported directly because that function hardcodes the Cincinnati bracket/CSV paths,
    but every downstream field (model_prob, market_prob de-vigged, real_decimal_odds from MaxW/
    MaxL, edge_pp, hard_matches) is computed the exact same way."""
    bracket = load_bracket_yaml(BRACKET_PATHS[tour])
    tour_config, draw, _matches_history = _prepare_ratings(bracket)  # frozen-at-start_date ratings

    ratings_df = pd.read_csv(tour_config.ratings_path).set_index("player")
    hard_matches = ratings_df["hard_matches"].to_dict()

    csv_path = fetch_source_csv(tour)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"{tour}: {len(df)} real Canadian Open matches loaded from {csv_path.name} "
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


# ---------------------------------------------------------------------------
# Primary metrics: Brier score & log-loss, model vs. de-vigged market, player-clustered bootstrap
# ---------------------------------------------------------------------------

def brier_score(p, y):
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _cluster_indices(bet_on):
    codes, players = pd.factorize(bet_on)
    return codes, len(players)


def cluster_bootstrap_metric(df, prob_col, metric_fn, n_boot=N_BOOT, seed=BOOT_SEED):
    """Player-clustered bootstrap CI for a single metric (Brier or log-loss) on one probability
    column - resamples players (by bet_on) with replacement, same convention as survivorship_
    upset_test.cluster_bootstrap_ci used throughout this project's held-out validations."""
    y = df["won"].astype(float).to_numpy()
    p = df[prob_col].to_numpy()
    codes, n_players = _cluster_indices(df["bet_on"].to_numpy())
    idx_by_code = [np.where(codes == c)[0] for c in range(n_players)]

    observed = metric_fn(p, y)
    if n_players == 0:
        return observed, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.integers(0, n_players, size=n_players)
        rows = np.concatenate([idx_by_code[c] for c in chosen])
        boots[i] = metric_fn(p[rows], y[rows])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return observed, lo, hi


def cluster_bootstrap_diff(df, col_a, col_b, metric_fn, n_boot=N_BOOT, seed=BOOT_SEED + 1):
    """Same clustering, but for (metric(col_a) - metric(col_b)) directly, so the CI on the
    difference reflects shared resampling rather than two independently-bootstrapped CIs that
    can't be validly subtracted."""
    y = df["won"].astype(float).to_numpy()
    pa = df[col_a].to_numpy()
    pb = df[col_b].to_numpy()
    codes, n_players = _cluster_indices(df["bet_on"].to_numpy())
    idx_by_code = [np.where(codes == c)[0] for c in range(n_players)]

    observed = metric_fn(pa, y) - metric_fn(pb, y)
    if n_players == 0:
        return observed, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.integers(0, n_players, size=n_players)
        rows = np.concatenate([idx_by_code[c] for c in chosen])
        boots[i] = metric_fn(pa[rows], y[rows]) - metric_fn(pb[rows], y[rows])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return observed, lo, hi


def print_calibration_table(df, title):
    print(f"\n{title}")
    print(f"  {'':22s} {'Brier':>10s} {'95% CI':>22s} {'Log-loss':>10s} {'95% CI':>22s}")
    for label, col in [("Model", "model_prob"), ("Market (de-vigged)", "market_prob")]:
        b_obs, b_lo, b_hi = cluster_bootstrap_metric(df, col, brier_score)
        l_obs, l_lo, l_hi = cluster_bootstrap_metric(df, col, log_loss)
        print(f"  {label:22s} {b_obs:>10.4f} [{b_lo:>+.4f},{b_hi:>+.4f}] {l_obs:>10.4f} [{l_lo:>+.4f},{l_hi:>+.4f}]")
    b_diff, b_lo, b_hi = cluster_bootstrap_diff(df, "model_prob", "market_prob", brier_score)
    l_diff, l_lo, l_hi = cluster_bootstrap_diff(df, "model_prob", "market_prob", log_loss)
    print(f"  {'Model - Market':22s} Brier {b_diff:+.4f} [{b_lo:+.4f},{b_hi:+.4f}]   "
          f"LogLoss {l_diff:+.4f} [{l_lo:+.4f},{l_hi:+.4f}]  (negative = model beats market)")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    all_opps = pd.concat([build_opportunity_rows_for_tour(t) for t in ("ATP", "WTA")], ignore_index=True)
    print(f"\nGenuine +EV opportunities found across all real, priced Canadian Open matches "
          f"(both tours, every round): {len(all_opps)}")
    if len(all_opps) == 0:
        print("Zero +EV opportunities exist - nothing to paper-trade. Stopping here.")
        return

    print(f"  By tour: {all_opps['tour'].value_counts().to_dict()}")
    print(f"  By round: {all_opps['round'].value_counts().to_dict()}")

    filtered_opps = all_opps[
        (all_opps["min_hard_matches"] >= MIN_HARD_MATCHES) & (all_opps["edge_pp"] >= MIN_EDGE_PP)
    ]

    # --- PRIMARY: Brier score & log-loss, model vs. market ---
    print_calibration_table(all_opps, f"=== PRIMARY: calibration, UNFILTERED (n={len(all_opps)}) ===")
    print_calibration_table(
        filtered_opps,
        f"=== PRIMARY: calibration, FILTERED (min_hard_matches>={MIN_HARD_MATCHES}, "
        f"min_edge_pp>={MIN_EDGE_PP:.0f}, n={len(filtered_opps)}) ===",
    )
    print("\n  Read this the same way as the Cincinnati baseline: this opportunity set is "
          "constructed to be exactly where the model disagrees with the market, so it is a "
          "biased sample for a calibration comparison by design - a confirmation of no gross "
          "miscalibration on the disagreement subset, not a claim of beating market calibration "
          "overall.")

    # --- SECONDARY: ROI / win rate, real-price settlement ---
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
          f"imported unchanged from cincinnati_paper_trading_backtest_tennisdata.py - not re-tuned "
          f"on Canadian Open data.")
    print(f"  - Every bet settles at the real best-obtainable closing price (MaxW/MaxL) - the "
          f"de-vigged consensus is signal only, never settlement, from the start.")
    print(f"  - Model probability uses Elo FROZEN at start_date=2026-08-02 for both tours - cutoff "
          f"verified against real Kaggle match dates before this script was trusted (see module "
          f"docstring).")
    print(f"  - WTA source CSV slug is unconfirmed guesswork among WTA_SLUG_CANDIDATES until a "
          f"real run resolves one - check the printed 'WTA: slug ... resolved' line above before "
          f"citing any WTA number from this run.")


if __name__ == "__main__":
    main()
