"""Pulls live/recent tennis results from ESPN's undocumented public scoreboard API:

    https://site.api.espn.com/apis/site/v2/sports/tennis/{atp|wta}/scoreboard
    optional ?dates=YYYYMMDD or ?dates=YYYYMMDD-YYYYMMDD


Usage:
    python model/live_scores.py                       # today's WTA scoreboard (women's singles/doubles)
    python model/live_scores.py --tour atp
    python model/live_scores.py --dates 20260812
    python model/live_scores.py --dates 20260810-20260812
    python model/live_scores.py --live-only
    python model/live_scores.py --event-id 718-2026     # one tournament only, ESPN event id
    python model/live_scores.py --tournament "Cincinnati Open"  # name substring match (less reliable)
"""
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard"
TOUR_CATEGORY_PREFIX = {"atp": "Men's", "wta": "Women's"}


class LiveScoresError(RuntimeError):
    """Raised when the ESPN response can't be parsed or doesn't have the expected shape."""


def fetch_scoreboard(tour="wta", dates=None, timeout=10):
    tour = tour.lower()
    if tour not in ("atp", "wta"):
        raise ValueError(f"tour must be 'atp' or 'wta', got {tour!r}")

    url = BASE_URL.format(tour=tour)
    if dates:
        url = f"{url}?dates={dates}"

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except (URLError, HTTPError) as e:
        raise LiveScoresError(f"Request to ESPN scoreboard failed ({url}): {e}") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LiveScoresError(
            f"ESPN scoreboard did not return valid JSON from {url} - this is an undocumented "
            f"API and can change without notice. Raw response started with: {raw[:200]!r}"
        ) from e

    if not isinstance(data, dict) or "events" not in data:
        found = list(data.keys()) if isinstance(data, dict) else type(data).__name__
        raise LiveScoresError(
            f"ESPN scoreboard response from {url} is missing the expected 'events' key - the "
            f"response shape may have changed. Top-level keys found: {found}"
        )

    return data


def _extract_sets(competitor):
    sets = []
    for linescore in competitor.get("linescores", []):
        value = linescore.get("value")
        if value is None:
            continue
        set_str = str(int(value))
        if linescore.get("tiebreak") is not None:
            set_str += f"({linescore['tiebreak']})"
        sets.append(set_str)
    return sets


def _is_doubles_shaped(competition):
    """Doubles competitors carry type='team' and a 'roster' of 2 athletes instead of a single
    'athlete' - a different, expected shape, not malformed data."""
    competitors = competition.get("competitors", [])
    return bool(competitors) and all(c.get("type") == "team" and "roster" in c for c in competitors)


def _extract_match(tournament_name, category_name, event_id, competition):
    competitors = competition.get("competitors", [])
    if len(competitors) != 2:
        raise ValueError(f"expected 2 competitors, got {len(competitors)}")

    by_order = sorted(competitors, key=lambda c: c.get("order", 0))
    names = []
    for competitor in by_order:
        name = (competitor.get("athlete") or {}).get("displayName")
        if not name:
            raise ValueError("competitor missing athlete displayName")
        names.append(name)

    status_type = (competition.get("status") or {}).get("type") or {}
    if "state" not in status_type:
        raise ValueError("competition status missing type.state")

    winner_index = next((i for i, c in enumerate(by_order) if c.get("winner") is True), None)

    return {
        "tournament": tournament_name,
        "event_id": event_id,
        "category": category_name,
        "round": (competition.get("round") or {}).get("displayName"),
        "status": status_type.get("description"),
        "status_state": status_type.get("state"),  # 'pre' | 'in' | 'post'
        "player_1": names[0],
        "player_2": names[1],
        "sets_1": _extract_sets(by_order[0]),
        "sets_2": _extract_sets(by_order[1]),
        "winner": names[winner_index] if winner_index is not None else None,
        "start_time": competition.get("date"),
        "match_id": competition.get("id"),
    }


def extract_matches(data):
    """Returns (matches, stats) where stats = {'doubles': n, 'malformed': n}. Doubles-shaped
    entries are counted and skipped quietly - they're an expected, different shape, not bad
    data. Only entries that don't match either the singles or doubles shape raise a loud
    WARNING, so real problems don't get buried under routine doubles-skip noise."""
    matches = []
    skipped_doubles = 0
    skipped_malformed = 0
    for event in data.get("events", []):
        tournament_name = event.get("name", "Unknown tournament")
        event_id = event.get("id")
        for grouping in event.get("groupings", []):
            category_name = (grouping.get("grouping") or {}).get("displayName", "Unknown category")
            for competition in grouping.get("competitions", []):
                if _is_doubles_shaped(competition):
                    skipped_doubles += 1
                    continue
                try:
                    matches.append(_extract_match(tournament_name, category_name, event_id, competition))
                except (ValueError, KeyError, TypeError) as e:
                    skipped_malformed += 1
                    match_id = competition.get("id", "?")
                    print(
                        f"WARNING: skipped malformed match (id={match_id}) in "
                        f"{tournament_name}/{category_name}: {e}",
                        file=sys.stderr,
                    )
    if skipped_doubles:
        print(f"Skipped {skipped_doubles} doubles matches (different shape, not extracted)", file=sys.stderr)
    return matches, {"doubles": skipped_doubles, "malformed": skipped_malformed}


def filter_by_tour(matches, tour):
    """Client-side tour filter, keyed on category name - the URL's {atp|wta} path segment
    doesn't actually narrow the response, both return the identical combined feed."""
    prefix = TOUR_CATEGORY_PREFIX[tour.lower()]
    return [m for m in matches if m["category"].startswith(prefix)]


def filter_by_event_id(matches, event_id):
    return [m for m in matches if m["event_id"] == event_id]


def filter_by_tournament_name(matches, name):
    """Case-insensitive substring match on tournament name - less reliable than event_id since
    ESPN's naming can be inconsistent (e.g. 'Cincinnati Open' vs 'Western & Southern Open')."""
    needle = name.lower()
    return [m for m in matches if needle in m["tournament"].lower()]


def format_match(m):
    score_1 = " ".join(m["sets_1"]) or "-"
    score_2 = " ".join(m["sets_2"]) or "-"
    return (
        f"[{m['tournament']} | {m['category']} | {m['round']}] "
        f"{m['player_1']} vs {m['player_2']} - {m['status']} "
        f"({score_1} | {score_2})"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tour", choices=["atp", "wta"], default="wta")
    parser.add_argument("--dates", default=None, help="YYYYMMDD or YYYYMMDD-YYYYMMDD")
    parser.add_argument("--live-only", action="store_true", help="only show in-progress matches")
    parser.add_argument("--limit", type=int, default=15, help="max matches to print")
    tournament_group = parser.add_mutually_exclusive_group()
    tournament_group.add_argument(
        "--event-id", default=None,
        help="restrict to one ESPN event id, e.g. 718-2026 (see espn_bracket.py) - more reliable than --tournament",
    )
    tournament_group.add_argument(
        "--tournament", default=None,
        help="restrict to tournaments whose name contains this substring (case-insensitive)",
    )
    args = parser.parse_args()

    try:
        data = fetch_scoreboard(args.tour, args.dates)
    except LiveScoresError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    matches, stats = extract_matches(data)
    matches = filter_by_tour(matches, args.tour)
    print(f"Fetched {len(matches)} {args.tour.upper()} matches "
          f"({stats['doubles']} doubles skipped, {stats['malformed']} malformed skipped)")

    if args.event_id:
        matches = filter_by_event_id(matches, args.event_id)
        print(f"{len(matches)} matches for event {args.event_id}")
    elif args.tournament:
        matches = filter_by_tournament_name(matches, args.tournament)
        print(f"{len(matches)} matches matching tournament {args.tournament!r}")

    if args.live_only:
        matches = [m for m in matches if m["status_state"] == "in"]
        print(f"{len(matches)} currently in progress")

    for m in matches[:args.limit]:
        print(format_match(m))
