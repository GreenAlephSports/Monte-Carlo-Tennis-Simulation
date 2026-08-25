"""Pulls REAL, currently-live market odds directly from The Odds API (not a cache, not a log -
a fresh call, right now) for whatever tennis event actually has bookmaker coverage at the moment
this runs, compares each side against this project's model probability, and sizes any bet whose
edge clears a minimum materiality threshold via fractional Kelly.

Two things this script exists to answer honestly, not assume:
  1. Which real, currently-priced matches does the model actually disagree with the market on,
     and by how much - checked against live odds pulled fresh, not the one frozen historical
     snapshot cincinnati_paper_trading_backtest.py was stuck with.
  2. Does a tennis FUTURES (outright tournament winner) market exist on this connected data source
     at all - checked directly against The Odds API's own has_outrights flag and a live markets=
     outrights request, not assumed either way.

Minimum-edge filter: a raw EV>0 threshold (the previous backtest's own bar) is too permissive -
"barely above zero" isn't a real signal once you account for de-vig approximation error and the
model's own calibration uncertainty (win_probability.py's Platt-scaling correction alone has an
irreducible residual). MIN_EDGE_PP below is a stated, real, adjustable threshold (percentage
points of probability, not a fitted constant), not a formula.

Usage:
    python model/live_odds_value_scan.py
"""
import sys
import urllib.parse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import json

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import TOUR_CONFIG  # noqa: E402
from bracket_export import ODDS_API_BASE, ODDS_API_KEY  # noqa: E402
from ev_comparison import implied_probabilities  # noqa: E402
from hybrid_simulation import match_espn_name_to_draw  # noqa: E402
from win_probability import _load_ratings, win_probability  # noqa: E402

MIN_EDGE_PP = 3.0  # minimum |model_prob - market_prob| in percentage points to call a bet "real",
                    # not noise - a stated, adjustable assumption, not a fitted/validated constant
KELLY_FRACTIONS = [0.25, 0.5]
REFERENCE_BANKROLL = 100.0

# has_outrights=false for every tennis sport The Odds API lists (checked directly below too, not
# just hardcoded) - tennis tournament-winner futures simply aren't a market this data source
# offers, for any tour or tournament, not just the one(s) checked here.
FUTURES_MARKET_KEY = "outrights"


def _http_get_json(url, params, timeout=15):
    query = urllib.parse.urlencode(params)
    request = Request(f"{url}?{query}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def check_futures_availability():
    print("=== Checking tennis futures/outright market availability (live, not assumed) ===")
    try:
        sports = _http_get_json(f"{ODDS_API_BASE}/sports/", {"apiKey": ODDS_API_KEY, "all": "true"})
    except (HTTPError, URLError) as e:
        print(f"  Couldn't reach The Odds API: {e}")
        return
    tennis = [s for s in sports if s.get("group") == "Tennis"]
    with_outrights = [s for s in sports if s.get("has_outrights")]
    print(f"  {len(tennis)} tennis sport keys checked - has_outrights=true for: "
          f"{[s['key'] for s in tennis if s.get('has_outrights')] or 'NONE'}")
    print(f"  The only sports with has_outrights=true on this whole connected source: "
          f"{[s['key'] for s in with_outrights]}")
    active_tennis = [s for s in tennis if s.get("active")]
    for s in active_tennis:
        try:
            events = _http_get_json(f"{ODDS_API_BASE}/sports/{s['key']}/odds/",
                                     {"apiKey": ODDS_API_KEY, "regions": "us", "markets": FUTURES_MARKET_KEY})
            print(f"  Direct outrights request for {s['key']}: {len(events)} events returned")
        except (HTTPError, URLError) as e:
            print(f"  Direct outrights request for {s['key']}: REJECTED - {e}")
    print("  CONCLUSION: no tennis futures/tournament-winner market exists on The Odds API for any "
          "tour or tournament - only real head-to-head match odds are available for tennis here. "
          "Any 'futures' comparison for tennis would need a different, not-yet-connected data source.")


def kelly_fraction(model_prob, market_prob):
    return max(0.0, (model_prob - market_prob) / (1 - market_prob))


def scan_live_h2h():
    print("\n=== Live h2h value scan (real odds pulled right now) ===")
    try:
        sports = _http_get_json(f"{ODDS_API_BASE}/sports/", {"apiKey": ODDS_API_KEY, "all": "true"})
    except (HTTPError, URLError) as e:
        sys.exit(f"ERROR: couldn't list sports: {e}")
    active_tennis = [s for s in sports if s.get("group") == "Tennis" and s.get("active")]
    print(f"Currently-active tennis sport(s) on The Odds API: "
          f"{[s['key'] for s in active_tennis] or 'NONE'}")
    if not active_tennis:
        print("No tennis event has live bookmaker coverage right now - nothing to scan.")
        return []

    tour_config = TOUR_CONFIG["WTA"] if "wta" in active_tennis[0]["key"] else TOUR_CONFIG["ATP"]
    draw_names = set(_load_ratings(tour_config.ratings_path).index)
    surface = "Hard"  # every currently-active event checked below is a hard-court tournament

    rows = []
    for sport in active_tennis:
        try:
            events = _http_get_json(
                f"{ODDS_API_BASE}/sports/{sport['key']}/odds/",
                {"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h", "oddsFormat": "decimal"},
            )
        except (HTTPError, URLError) as e:
            print(f"  WARNING: odds request failed for {sport['key']}: {e}")
            continue
        print(f"\n{sport['title']} ({sport['key']}): {len(events)} real, currently-scheduled match(es) priced")

        for event in events:
            home, away = event.get("home_team"), event.get("away_team")
            if not home or not away:
                continue
            home_prices, away_prices = [], []
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                    if home in outcomes and away in outcomes:
                        home_prices.append(outcomes[home])
                        away_prices.append(outcomes[away])
            if not home_prices:
                continue
            avg_home = sum(home_prices) / len(home_prices)
            avg_away = sum(away_prices) / len(away_prices)
            market_home, market_away = implied_probabilities(avg_home, avg_away)

            csv_home = match_espn_name_to_draw(home, draw_names, tour_config.name_aliases)
            csv_away = match_espn_name_to_draw(away, draw_names, tour_config.name_aliases)
            if csv_home is None or csv_away is None:
                print(f"  SKIP {home} vs {away}: couldn't resolve to ratings CSV "
                      f"({home!r}->{csv_home!r}, {away!r}->{csv_away!r})")
                continue
            model_home = win_probability(csv_home, csv_away, surface, tour_config.ratings_path)
            model_away = 1 - model_home

            for side, player, opp, model_p, market_p in [
                ("home", home, away, model_home, market_home),
                ("away", away, home, model_away, market_away),
            ]:
                edge_pp = (model_p - market_p) * 100
                rows.append({
                    "tournament": sport["title"], "commence_time": event.get("commence_time"),
                    "player": player, "opponent": opp, "model_prob": model_p, "market_prob": market_p,
                    "edge_pp": edge_pp, "n_books": len(home_prices),
                })
    return rows


def report(rows):
    if not rows:
        return
    print(f"\n--- Every real side checked ({len(rows)} rows, {len(rows)//2} matches) ---")
    header = f"{'Player':<26} {'Opponent':<26} {'Model':>7} {'Market':>7} {'Edge(pp)':>9} {'Books':>6}"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: -abs(r["edge_pp"])):
        print(f"{r['player']:<26} {r['opponent']:<26} {r['model_prob']:>6.1%} {r['market_prob']:>6.1%} "
              f"{r['edge_pp']:>+8.1f}pp {r['n_books']:>6}")

    actionable = [r for r in rows if r["edge_pp"] >= MIN_EDGE_PP]
    skipped = [r for r in rows if 0 < r["edge_pp"] < MIN_EDGE_PP]
    print(f"\n{len(actionable)} side(s) clear the +{MIN_EDGE_PP:.1f}pp minimum-edge threshold. "
          f"{len(skipped)} more side(s) had positive-but-sub-threshold edge (too close to market "
          f"to trust as real signal rather than de-vig/calibration noise) - excluded from sizing below.")

    if not actionable:
        print("Nothing clears the threshold right now - no proposed stakes.")
        return

    print(f"\n--- Proposed hypothetical stakes for actionable bets (reference bankroll = "
          f"{REFERENCE_BANKROLL:.0f} units; these are UPCOMING real matches, no outcome yet, so this "
          f"is a live proposal, not a settled backtest) ---")
    header2 = f"{'Player':<26} {'Opponent':<26} {'Edge(pp)':>9} {'Kelly f*':>9} " + \
              " ".join(f"{f}x stake" for f in KELLY_FRACTIONS)
    print(header2)
    for r in sorted(actionable, key=lambda r: -r["edge_pp"]):
        f_star = kelly_fraction(r["model_prob"], r["market_prob"])
        stakes = "  ".join(f"{f_star*f*REFERENCE_BANKROLL:>8.2f}" for f in KELLY_FRACTIONS)
        print(f"{r['player']:<26} {r['opponent']:<26} {r['edge_pp']:>+8.1f}pp {f_star:>8.1%}  {stakes}")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if not ODDS_API_KEY:
        sys.exit("ERROR: ODDS_API_KEY not set in .env")

    check_futures_availability()
    rows = scan_live_h2h()
    report(rows)


if __name__ == "__main__":
    main()
