"""One-shot (not a long-running loop - designed to be re-invoked periodically, e.g. by a scheduled
cron job or /loop) check for Osaka N.'s real US Open 2026 (WTA) matches: does a single poll of live
ESPN state to (a) cache any newly-available pregame market price for her NEXT match (reusing
live_match_watcher.update_market_price_cache's exact mechanism, so a price is never lost to The
Odds API dropping the event once it goes Final), then (b) runs calibration_log.py's own logging
pipeline to persist any of her matches that just concluded, then (c) prints every one of her logged
matches so far - model's pre-match probability vs. the market's, and which one (if either) called
the real outcome correctly.

This is a direct, live test of the "market is discounting a genuinely in-form player" read from the
earlier Osaka investigation: does she keep beating what the model assigns her, what the market
assigns her, both, or neither, as her real US Open games actually get decided? Reports honestly
whichever side it favors - this script does not have a preferred outcome.

Usage:
    python model/research/osaka_us_open_watch.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket_export import OUTPUT_DIR, export_bracket_json  # noqa: E402
from bracket_schema import load_bracket_yaml  # noqa: E402
from calibration_log import load_existing_log, run as run_calibration_log  # noqa: E402
from hybrid_simulation import TOUR_SINGLES_CATEGORY  # noqa: E402
from live_match_watcher import fetch_tournament_matches, update_market_price_cache  # noqa: E402
from live_scores import LiveScoresError  # noqa: E402

BRACKET_PATH = Path("brackets/us_open_2026_wta_real.yaml")
PLAYER_NAME = "Osaka N."


def refresh_market_cache():
    """One poll cycle - same mechanism live_match_watcher.watch() runs on every interval, just
    invoked once here instead of in an infinite loop, since this script itself is meant to be
    re-invoked periodically (cron/loop) rather than stay resident."""
    bracket = load_bracket_yaml(BRACKET_PATH)
    category = TOUR_SINGLES_CATEGORY[bracket.tour.lower()]
    try:
        _out_path, export_output = export_bracket_json(
            BRACKET_PATH, output_path=OUTPUT_DIR / f"{BRACKET_PATH.stem}_osaka_watch_baseline.json",
        )
        current_matches = fetch_tournament_matches(bracket.tour, bracket.tournament, category)
    except LiveScoresError as e:
        print(f"WARNING: couldn't fetch live ESPN state this cycle ({e}) - skipping market-cache "
              f"refresh, will retry next invocation.")
        return
    n_cached = update_market_price_cache(bracket, export_output, current_matches)
    if n_cached:
        print(f"Cached {n_cached} new pregame market price(s).")


def report_osaka(log, bracket):
    # calibration_log.py's log spans EVERY tournament this project tracks (Cincinnati, Montreal,
    # etc.), not just this one - a real bug caught while building this: filtering by player name
    # alone pulled in her already-concluded Cincinnati matches alongside the US Open ones. Must
    # also scope to this specific tournament + year.
    rows = log[
        ((log["player_a"] == PLAYER_NAME) | (log["player_b"] == PLAYER_NAME))
        & (log["tournament"] == bracket.tournament) & (log["year"] == bracket.year)
    ].copy()
    if len(rows) == 0:
        print(f"\nNo {PLAYER_NAME} {bracket.tournament} {bracket.year} matches logged yet "
              f"(tournament hasn't started, or her first match hasn't concluded).")
        return

    rows = rows.sort_values("round_num")
    print(f"\n{'=' * 100}\n{PLAYER_NAME} - real US Open 2026 matches logged so far ({len(rows)})\n{'=' * 100}")
    model_correct, market_correct, both, neither, n_with_market = 0, 0, 0, 0, 0
    for r in rows.itertuples(index=False):
        osaka_won = r.winner == PLAYER_NAME
        opponent = r.player_b if r.player_a == PLAYER_NAME else r.player_a
        model_prob_osaka = r.favorite_prob if r.favorite == PLAYER_NAME else 1 - r.favorite_prob
        model_favored_osaka = model_prob_osaka >= 0.5
        model_called_it = model_favored_osaka == osaka_won

        market_line = "no market price captured for this match"
        market_called_it = None
        if pd.notna(r.market_prob_a):
            market_prob_osaka = r.market_prob_a if r.player_a == PLAYER_NAME else 1 - r.market_prob_a
            market_favored_osaka = market_prob_osaka >= 0.5
            market_called_it = market_favored_osaka == osaka_won
            market_line = f"market gave her {market_prob_osaka:.1%} (called it: {'YES' if market_called_it else 'no'})"
            n_with_market += 1
            if model_called_it and market_called_it:
                both += 1
            elif model_called_it and not market_called_it:
                model_correct += 1
            elif market_called_it and not model_called_it:
                market_correct += 1
            else:
                neither += 1

        print(f"\n  R{r.round_num} ({r.round_label}) vs {opponent}, {r.date}: "
              f"Osaka {'WON' if osaka_won else 'lost'}")
        print(f"    model gave her {model_prob_osaka:.1%} (called it: {'YES' if model_called_it else 'no'})")
        print(f"    {market_line}")

    if n_with_market:
        print(f"\n{'-' * 100}\nHead-to-head across {n_with_market} match(es) with a captured market price:\n"
              f"  model right, market wrong: {model_correct}\n"
              f"  market right, model wrong: {market_correct}\n"
              f"  both right: {both}\n"
              f"  both wrong: {neither}")
        if model_correct > market_correct:
            print("  -> so far, the MODEL is calling her matches better than the market.")
        elif market_correct > model_correct:
            print("  -> so far, the MARKET is calling her matches better than the model - the "
                  "'market is discounting a genuinely in-form player' read is NOT holding up in "
                  "real results.")
        else:
            print("  -> so far, tied - not enough separation yet to say either side is winning "
                  "this real-time test.")
    else:
        print(f"\nNo market price was captured for any of her matches yet (either too early, or "
              f"live_match_watcher.py wasn't polling before her match(es) went live to catch a "
              f"pregame price - The Odds API drops an event once it's Final, so a missed pregame "
              f"snapshot can't be recovered after the fact).")


if __name__ == "__main__":
    # same fix live_match_watcher.main() uses: Windows' default console codepage (cp1252) can't
    # encode non-ASCII player names (e.g. diacritics), which crashes a plain print() of them -
    # reconfigure() is a no-op on streams that don't support it, so this is safe everywhere.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    print(f"Refreshing market price cache for {BRACKET_PATH}...")
    refresh_market_cache()

    print(f"\nRunning calibration_log.py to persist any newly-concluded matches...")
    run_calibration_log()

    log = load_existing_log()
    report_osaka(log, load_bracket_yaml(BRACKET_PATH))
