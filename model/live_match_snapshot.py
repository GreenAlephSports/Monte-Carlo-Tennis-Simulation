"""On-demand, single-shot snapshot for ONE live match - never polls, never runs on a timer. Each
run does exactly:
  1. one ESPN scoreboard fetch (find the match, read its real score/set state)
  2. at most one Odds API quota-consuming call (fetch that sport's current odds, including live
     in-play prices once a match has started - same endpoint bracket_export.py already uses)
  3. compute a live-adjusted model probability: this project's static pregame Elo probability
     (win_probability.py), shifted by a score-state heuristic (see score_adjusted_probability
     below) so being up/down a set or a break actually moves the number
  4. print pregame market / live market / live model side by side, and append one line to
     output/live_match_snapshots.jsonl so later runs against the same match can be compared over
     time - this script itself never loops or re-fetches on its own.

Usage:
    python model/live_match_snapshot.py --list-live --tour atp
    python model/live_match_snapshot.py --tour atp --match-id 181963 --surface Hard
    python model/live_match_snapshot.py --tour wta --player Arseneault --surface Hard
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket import TOUR_CONFIG  # noqa: E402
from bracket_export import ODDS_API_BASE, ODDS_API_KEY, _http_get_json, discover_odds_sport_key  # noqa: E402
from ev_comparison import implied_probabilities  # noqa: E402
from hybrid_simulation import match_espn_name_to_draw  # noqa: E402
from live_match_watcher import _market_price_cache_key, load_market_price_cache  # noqa: E402
from live_scores import LiveScoresError, extract_matches, fetch_scoreboard, filter_by_tour, format_match  # noqa: E402
from win_probability import _load_ratings, apply_logit_shift, win_probability  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
SNAPSHOT_LOG_PATH = OUTPUT_DIR / "live_match_snapshots.jsonl"

# ---- live score-state adjustment ----
# REASONED HEURISTIC, NOT BACKTESTED - unlike every constant in win_probability.py (each fit and
# held-out-validated against this project's own match history), there's no equivalent backtest
# here: that would need point-by-point historical score data this project doesn't have. These
# numbers come from widely-cited tennis-analytics figures for "P(win match | won set 1)" -
# roughly 78-85% in best-of-3, ~65-72% in best-of-5, for an otherwise even matchup - converted to
# a logit shift per completed set of lead: logit(0.80)-logit(0.5)=1.386 (bo3),
# logit(0.70)-logit(0.5)=0.847 (bo5). Treat this as directionally-correct, not calibrated.
LOGIT_SHIFT_PER_SET = {3: 1.386, 5: 0.847}
DEFAULT_LOGIT_SHIFT_PER_SET = 1.386


def score_state(m):
    """(sets_won_1, sets_won_2, current_set_games_1, current_set_games_2, best_of) from a
    live_scores.py match dict's raw games_1/games_2/current_period/best_of fields. A completed
    set is whichever of the first (current_period - 1) linescore entries has more games; the
    entry at index (current_period - 1), if present, is the in-progress set (0-0 if it hasn't
    been added to the feed yet, e.g. right at a new set's first point)."""
    period = m.get("current_period") or 1
    best_of = m.get("best_of") or 3
    games_1, games_2 = m.get("games_1") or [], m.get("games_2") or []
    n_completed = max(0, period - 1)
    completed_1, completed_2 = games_1[:n_completed], games_2[:n_completed]
    sets_1 = sum(1 for a, b in zip(completed_1, completed_2) if a > b)
    sets_2 = sum(1 for a, b in zip(completed_1, completed_2) if b > a)
    if len(games_1) > n_completed and len(games_2) > n_completed:
        current_1, current_2 = games_1[n_completed], games_2[n_completed]
    else:
        current_1, current_2 = 0, 0
    return sets_1, sets_2, current_1, current_2, best_of


def score_adjusted_probability(pregame_prob_1, sets_1, sets_2, current_games_1, current_games_2, best_of):
    """Shifts the static pregame probability in logit space by two components: a full per-set
    shift for every completed set of lead (see LOGIT_SHIFT_PER_SET's docstring), plus a partial
    shift for the CURRENT in-progress set's game differential (capped at +/-6 games, scaled
    linearly against the same per-set constant - a 6-game differential means the set is already
    effectively won, so it gets the full per-set shift; a 2-game differential, i.e. one break of
    serve with the rest level, gets a third of it)."""
    per_set = LOGIT_SHIFT_PER_SET.get(best_of, DEFAULT_LOGIT_SHIFT_PER_SET)
    set_shift = per_set * (sets_1 - sets_2)
    game_diff = max(-6, min(6, current_games_1 - current_games_2))
    game_shift = per_set * (game_diff / 6)
    return apply_logit_shift(pregame_prob_1, set_shift + game_shift)


def fetch_live_market_price(sport_key, api_key, player_1, player_2):
    """The one Odds API call this script ever makes (only reached when a real snapshot is being
    taken, never on --list-live) - fetches this sport's current odds (live in-play prices
    included once a match has started, per bracket_export.fetch_devigged_odds) and filters
    locally to just this one event, rather than reusing fetch_devigged_odds directly (which
    prints every bookmaker line for every match in the tournament - noisy for a single-match
    tool). Returns (prob_player_1, n_bookmakers); (None, 0) if this match has no bookmaker-quoted
    price right now (not covered by The Odds API at all, e.g. a lower-tier event, or in-play odds
    not posted yet)."""
    if not sport_key or not api_key:
        return None, 0
    try:
        events = _http_get_json(
            f"{ODDS_API_BASE}/sports/{sport_key}/odds/",
            {"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "decimal"},
        )
    except (HTTPError, URLError) as e:
        print(f"WARNING: The Odds API request failed ({e}) - no live market price available", file=sys.stderr)
        return None, 0

    target = frozenset((player_1, player_2))
    for event in events:
        home, away = event.get("home_team"), event.get("away_team")
        if not home or not away or frozenset((home, away)) != target:
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
            return None, 0
        avg_home = sum(home_prices) / len(home_prices)
        avg_away = sum(away_prices) / len(away_prices)
        prob_home, prob_away = implied_probabilities(avg_home, avg_away)
        return (prob_home if home == player_1 else prob_away), len(home_prices)
    return None, 0


def find_match(matches, match_id=None, player_substring=None):
    if match_id is not None:
        candidates = [m for m in matches if str(m["match_id"]) == str(match_id)]
        if not candidates:
            sys.exit(f"ERROR: no match found with match_id={match_id!r}")
        return candidates[0]

    needle = player_substring.lower()
    candidates = [
        m for m in matches
        if needle in m["player_1"].lower() or needle in m["player_2"].lower()
    ]
    if not candidates:
        sys.exit(f"ERROR: no match found with a player matching {player_substring!r}")
    if len(candidates) > 1:
        lines = "\n".join(f"  match_id={m['match_id']}: {format_match(m)}" for m in candidates)
        sys.exit(f"ERROR: {player_substring!r} matches {len(candidates)} matches - use --match-id "
                  f"instead:\n{lines}")
    return candidates[0]


def _fmt_pct(p):
    return f"{p:.1%}" if p is not None else "n/a"


def print_list_live(matches, tour):
    live = [m for m in matches if m["status_state"] == "in"]
    if not live:
        print(f"No {tour.upper()} matches currently in progress.")
        return
    print(f"{len(live)} {tour.upper()} match(es) currently in progress:\n")
    for m in live:
        print(f"  match_id={m['match_id']:<10} {format_match(m)}")


def build_snapshot(tour, match, surface, ratings_path, name_aliases, api_key):
    player_1, player_2 = match["player_1"], match["player_2"]
    tournament = match["tournament"]
    year = str(match["event_id"]).split("-")[-1] if match["event_id"] else None

    # --- pregame market price: read-only lookup in the existing cache, never written here ---
    pregame_prob_1 = None
    if year is not None:
        cache = load_market_price_cache()
        key = _market_price_cache_key(tour, tournament, year, player_1, player_2)
        entry = cache.get(key)
        if entry is not None:
            pregame_prob_1 = (
                entry["market_prob_a"] if entry["player_a"] == player_1 else 1 - entry["market_prob_a"]
            )

    # --- live market price: the one Odds API call this run spends ---
    sport_key = discover_odds_sport_key(tour, tournament, api_key) if api_key else None
    live_market_prob_1, n_books = fetch_live_market_price(sport_key, api_key, player_1, player_2)

    # --- live model probability: static pregame Elo, then score-state adjusted ---
    draw_names = set(_load_ratings(ratings_path).index)
    csv_1 = match_espn_name_to_draw(player_1, draw_names, name_aliases)
    csv_2 = match_espn_name_to_draw(player_2, draw_names, name_aliases)
    pregame_model_prob_1 = live_model_prob_1 = None
    sets_1 = sets_2 = current_1 = current_2 = best_of = None
    if csv_1 is None or csv_2 is None:
        print(f"WARNING: couldn't resolve one or both ESPN names to the ratings CSV "
              f"({player_1!r} -> {csv_1!r}, {player_2!r} -> {csv_2!r}) - no model probability "
              f"available this run.", file=sys.stderr)
    else:
        pregame_model_prob_1 = win_probability(csv_1, csv_2, surface, ratings_path)
        sets_1, sets_2, current_1, current_2, best_of = score_state(match)
        live_model_prob_1 = score_adjusted_probability(
            pregame_model_prob_1, sets_1, sets_2, current_1, current_2, best_of
        )

    return {
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tour": tour, "tournament": tournament, "round": match["round"], "match_id": match["match_id"],
        "player_1": player_1, "player_2": player_2,
        "status": match["status"], "status_state": match["status_state"],
        "sets_1": match["sets_1"], "sets_2": match["sets_2"],
        "sets_won_1": sets_1, "sets_won_2": sets_2,
        "current_set_games_1": current_1, "current_set_games_2": current_2, "best_of": best_of,
        "pregame_market_prob_1": pregame_prob_1,
        "live_market_prob_1": live_market_prob_1, "live_market_n_bookmakers": n_books,
        "pregame_model_prob_1": pregame_model_prob_1,
        "live_model_prob_1": live_model_prob_1,
    }


def print_snapshot(snap):
    print("=" * 78)
    print(f"{snap['tournament']} ({snap['tour']}) - {snap['round']} - match_id={snap['match_id']}")
    print(f"{snap['player_1']} vs {snap['player_2']}")
    print(f"status: {snap['status']}  |  sets: {' '.join(snap['sets_1']) or '-'} | "
          f"{' '.join(snap['sets_2']) or '-'}")
    if snap["status_state"] != "in":
        print(f"NOTE: status_state={snap['status_state']!r} - this match is not currently in "
              f"progress, so 'live' figures below are just this instant's snapshot, not an "
              f"in-play read.")
    if snap["sets_won_1"] is not None:
        print(f"score state used for the model adjustment: sets {snap['sets_won_1']}-{snap['sets_won_2']}, "
              f"current set games {snap['current_set_games_1']}-{snap['current_set_games_2']} "
              f"(best of {snap['best_of']})")
    print("-" * 78)
    print(f"{'':<28} {snap['player_1']:<22} {snap['player_2']:<22}")
    p1 = snap["pregame_market_prob_1"]
    p2 = 1 - p1 if p1 is not None else None
    print(f"{'Pregame market (cached)':<28} {_fmt_pct(p1):<22} {_fmt_pct(p2):<22}")
    m1 = snap["live_market_prob_1"]
    m2 = 1 - m1 if m1 is not None else None
    books_note = f" ({snap['live_market_n_bookmakers']} book(s))" if m1 is not None else ""
    print(f"{'Live market (Odds API)':<28} {_fmt_pct(m1):<22} {_fmt_pct(m2):<22}{books_note}")
    l1 = snap["live_model_prob_1"]
    l2 = 1 - l1 if l1 is not None else None
    print(f"{'Live model (score-adj.)':<28} {_fmt_pct(l1):<22} {_fmt_pct(l2):<22}")
    if snap["pregame_model_prob_1"] is not None:
        pm1 = snap["pregame_model_prob_1"]
        print(f"{'  (unadjusted pregame model)':<28} {_fmt_pct(pm1):<22} {_fmt_pct(1 - pm1):<22}")
    print("=" * 78)


def log_snapshot(snap):
    SNAPSHOT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(snap) + "\n")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tour", required=True, choices=["atp", "wta"])
    parser.add_argument("--dates", default=None, help="YYYYMMDD, passed to ESPN's ?dates=")
    parser.add_argument("--list-live", action="store_true",
                         help="print currently in-progress matches (with match_id) and exit - no "
                              "Odds API call, nothing logged")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--match-id", default=None, help="ESPN competition id, e.g. from --list-live")
    target.add_argument("--player", default=None, help="substring match on either player's name")
    parser.add_argument("--surface", default=None, choices=["Hard", "Clay", "Grass"],
                         help="required unless --list-live")
    parser.add_argument("--no-log", action="store_true", help="skip appending to " + str(SNAPSHOT_LOG_PATH))
    args = parser.parse_args()

    if not args.list_live:
        if not args.match_id and not args.player:
            parser.error("one of --match-id or --player is required (or use --list-live)")
        if not args.surface:
            parser.error("--surface is required unless --list-live")

    try:
        data = fetch_scoreboard(args.tour, dates=args.dates)
    except LiveScoresError as e:
        sys.exit(f"ERROR: {e}")
    matches, _stats = extract_matches(data)
    matches = filter_by_tour(matches, args.tour)

    if args.list_live:
        print_list_live(matches, args.tour)
        return

    match = find_match(matches, match_id=args.match_id, player_substring=args.player)

    tour_upper = args.tour.upper()
    tour_config = TOUR_CONFIG[tour_upper]
    snap = build_snapshot(
        tour_upper, match, args.surface, tour_config.ratings_path, tour_config.name_aliases, ODDS_API_KEY
    )
    print_snapshot(snap)

    if not args.no_log:
        log_snapshot(snap)
        print(f"\nLogged to {SNAPSHOT_LOG_PATH}")


if __name__ == "__main__":
    main()
