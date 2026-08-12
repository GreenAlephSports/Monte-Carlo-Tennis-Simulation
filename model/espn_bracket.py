"""Builds a bracket YAML (the same schema parse_atp_draw.py produces) directly from ESPN's live
scoreboard data for a given tour + event ID, instead of parsing a draw-sheet PDF. run_tournament.py
works identically regardless of which of the two produced the file.

Structured JSON sidesteps the whole class of PDF-extraction bugs already hit and fixed in
parse_atp_draw.py (word-gap/x_tolerance guessing, regex collisions between a status code and a
surname's first letter, glued-together multi-word surnames) - names arrive whole, not
pre-truncated or OCR'd.

What ESPN's scoreboard actually gives us, confirmed by inspection (see investigation notes):
  - Every round, including ones that haven't been played - Round 1 through the Final - as
    'pre'-state competitions, not just completed/live matches.
  - Byes have no explicit flag: a player entering directly in Round 2 who never appears as a
    Round 1 competitor is a bye (inferred here, not provided).
  - Seed numbers (via competitor.curatedRank.current) are present only on players who actually
    carry a seed - which in this 96-draw format (32 seeds, all with byes) means only the Round 2
    bye entrants; nobody entering Round 1 is seeded, so its absence there is correct, not missing.
  - No bracket-slot/position numbers - draw order here is just the order matches came out in,
    which is fine: non-bye players are emitted as adjacent pairs in Round 1 order, byes appended
    after, matching the same list-order convention bracket.py's own byes handling already expects
    when no 'position' field is present.
  - No status codes (Q/WC/LL/PR) anywhere in this feed - out of scope per the request, ignored.
  - Some Round 1 slots can be 'TBD' - the draw was released before qualifying finished, so that
    slot's occupant genuinely isn't decided yet. Each gets its own uniquely-named placeholder
    entry (e.g. "TBD (Qualifier 1)") rather than being skipped or blocking the whole bracket -
    it never appears in any historical match data, so bracket.py's existing tier-3 fallback
    (STARTING_ELO placeholder for a player with no training-window history) picks it up
    automatically. No new fallback logic needed here for that part.

Usage:
    python model/espn_bracket.py --tour atp --event-id 718-2026 --surface Hard brackets/cincinnati_2026_atp.yaml
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket import TOUR_CONFIG  # noqa: E402
from elo_ratings import SURFACES  # noqa: E402
from hybrid_simulation import match_espn_name_to_draw  # noqa: E402
from live_scores import LiveScoresError, fetch_scoreboard  # noqa: E402
from parse_atp_draw import _truncation_length  # noqa: E402

TOUR_SINGLES_CATEGORY = {"atp": "Men's Singles", "wta": "Women's Singles"}


def _known_ratings_names(tour):
    """The existing ratings CSV, if one's already been generated - used as the candidate pool
    for match_espn_name_to_draw so an already-known player gets their already-correct truncated
    name reused directly, rather than re-derived (and re-risking the same Western/native
    name-order ambiguity ESPN's own data has - e.g. 'Wang Xiyu' vs 'Wang Xinyu', where ESPN's
    own shortName field gets the surname backwards for both)."""
    ratings_path = TOUR_CONFIG[tour.upper()].ratings_path
    if not ratings_path.exists():
        return []
    return pd.read_csv(ratings_path)["player"].tolist()


def _fallback_lastname_firstname(display_name, short_name):
    """For a player NOT already in the ratings CSV (a genuine newcomer - the only case that
    reaches this fallback, since anyone with real match history is caught by the CSV lookup
    above). ESPN's own shortName ('X. Lastname') reliably tells us how ESPN itself splits this
    exact display_name into initial + lastname (self-consistent, even on the rare display_name
    where that split doesn't match true tennis surname convention) - so trust it for this
    fallback rather than guessing independently."""
    match = re.match(r"^([A-Za-z])\.\s*(.+)$", (short_name or "").strip())
    if not match:
        words = display_name.split()
        return (words[-1], " ".join(words[:-1]) or words[-1]) if len(words) > 1 else (display_name, display_name)

    initial, lastname = match.group(1), match.group(2)
    lastname_word_count = len(lastname.split())
    display_words = display_name.split()
    firstname_words = display_words[:len(display_words) - lastname_word_count]
    firstname = " ".join(firstname_words) if firstname_words else initial
    return lastname, firstname


def _resolve_player_name(display_name, short_name, ratings_names):
    """Returns a final ratings-csv-style name ('Lastname X.') for a real (non-TBD) player.
    Tries the ratings-csv cross-reference first (exact, reuses match_espn_name_to_draw); falls
    back to a fresh best-effort truncation (single initial) only for players not already known -
    collision-widening against other same-lastname fallback players in this draw happens as a
    separate pass afterward, mirroring parse_atp_draw.py's own to_yaml_players logic."""
    matched = match_espn_name_to_draw(display_name, ratings_names) if ratings_names else None
    if matched:
        return matched, None  # None = "already final, skip collision-truncation pass"
    lastname, firstname = _fallback_lastname_firstname(display_name, short_name)
    return None, (lastname.title(), firstname)  # needs the collision-truncation pass


def _resolve_fallback_collisions(fallback_players):
    """Same collision-aware truncation parse_atp_draw.py uses for players sharing a lastname
    within one draw (e.g. two different 'Wang's) - reused directly (_truncation_length), not
    reimplemented, just applied to the fallback-only subset of players."""
    by_lastname = defaultdict(list)
    for index, (lastname, _firstname) in fallback_players.items():
        by_lastname[lastname].append(index)

    resolved = {}
    for lastname, indices in by_lastname.items():
        firstnames = [fallback_players[i][1] for i in indices]
        length = _truncation_length(firstnames)
        for index, firstname in zip(indices, firstnames):
            resolved[index] = f"{lastname} {firstname[:length]}."
    return resolved


def build_bracket_players(tour, event_id):
    """Returns a list of player dicts (seed/name/bye/status) in bracket order, plus the raw
    ESPN event dict (used by the caller for tournament/date metadata)."""
    tour = tour.lower()
    category = TOUR_SINGLES_CATEGORY[tour]
    data = fetch_scoreboard(tour)

    event = next((e for e in data.get("events", []) if e.get("id") == event_id), None)
    if event is None:
        available = [e.get("id") for e in data.get("events", [])]
        raise ValueError(f"No event with id {event_id!r} on the {tour.upper()} scoreboard. Available: {available}")

    grouping = next(
        (g for g in event.get("groupings", []) if (g.get("grouping") or {}).get("displayName") == category), None
    )
    if grouping is None:
        raise ValueError(f"No {category!r} grouping found for event {event_id}")

    competitions = grouping.get("competitions", [])
    round1 = [c for c in competitions if (c.get("round") or {}).get("displayName") == "Round 1"]
    round2 = [c for c in competitions if (c.get("round") or {}).get("displayName") == "Round 2"]
    if not round1:
        raise ValueError(f"No Round 1 matches found for event {event_id} / {category}")

    ratings_names = _known_ratings_names(tour)

    # pass 1: resolve every real (non-TBD) name to either an already-known ratings-csv name, or
    # a (lastname, firstname) pair still needing the collision-truncation pass
    raw_records = []  # each: dict(kind, seed, resolved_name_or_None, fallback_key_or_None)
    fallback_players = {}  # index into raw_records -> (lastname, firstname), for collision pass
    tbd_counter = 0

    def add_record(kind, competitor):
        nonlocal tbd_counter
        athlete = competitor.get("athlete") or {}
        display_name = athlete.get("displayName")
        seed = (competitor.get("curatedRank") or {}).get("current")
        if not display_name or display_name == "TBD":
            tbd_counter += 1
            raw_records.append({"kind": kind, "seed": None, "name": f"TBD (Qualifier {tbd_counter})"})
            return
        resolved, fallback_key = _resolve_player_name(display_name, athlete.get("shortName"), ratings_names)
        record = {"kind": kind, "seed": seed, "name": resolved}
        if fallback_key is not None:
            fallback_players[len(raw_records)] = fallback_key
        raw_records.append(record)

    for competition in round1:
        by_order = sorted(competition.get("competitors", []), key=lambda c: c.get("order", 0))
        for competitor in by_order:
            add_record("round1", competitor)

    round1_real_names = {r["name"] for r in raw_records if r["kind"] == "round1" and r["name"]}
    for competition in round2:
        for competitor in competition.get("competitors", []):
            athlete = competitor.get("athlete") or {}
            display_name = athlete.get("displayName")
            if not display_name or display_name == "TBD":
                continue  # the pending Round 1 winner - already represented by its Round 1 entry
            resolved, fallback_key = _resolve_player_name(display_name, athlete.get("shortName"), ratings_names)
            if resolved and resolved in round1_real_names:
                continue  # safety net - shouldn't happen, but never double-count a player
            record = {"kind": "round2_bye", "seed": (competitor.get("curatedRank") or {}).get("current"),
                       "name": resolved}
            if fallback_key is not None:
                fallback_players[len(raw_records)] = fallback_key
            raw_records.append(record)

    resolved_fallbacks = _resolve_fallback_collisions(fallback_players)
    for index, name in resolved_fallbacks.items():
        raw_records[index]["name"] = name

    players = []
    for record in raw_records:
        players.append({
            "seed": record["seed"],
            "name": record["name"],
            "status": None,
            "bye": record["kind"] == "round2_bye",
        })
    return players, event, round1


def _tournament_start_date(event, round1):
    dated = [c.get("date") for c in round1 if c.get("date")]
    date_str = min(dated) if dated else event.get("date")
    return date_str[:10] if date_str else None


def build_bracket_yaml(tour, event_id, surface):
    tour = tour.upper()
    if surface not in SURFACES:
        raise ValueError(f"surface must be one of {SURFACES}, got {surface!r}")

    players, event, round1 = build_bracket_players(tour, event_id)
    start_date = _tournament_start_date(event, round1)
    year = int(start_date[:4]) if start_date else None

    return {
        "tournament": event.get("name"),
        "year": year,
        "tour": tour,
        "surface": surface,
        "start_date": start_date,
        "draw_size": len(players),
        "players": players,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--tour", choices=["atp", "wta"], required=True)
    parser.add_argument("--event-id", required=True, help="ESPN event id, e.g. 718-2026")
    parser.add_argument("--surface", required=True, choices=SURFACES,
                         help="ESPN's scoreboard data has no surface field - must be supplied")
    args = parser.parse_args()

    try:
        bracket = build_bracket_yaml(args.tour, args.event_id, args.surface)
    except (LiveScoresError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    real_count = sum(1 for p in bracket["players"] if not str(p["name"]).startswith("TBD"))
    tbd_count = len(bracket["players"]) - real_count
    bye_count = sum(1 for p in bracket["players"] if p["bye"])
    print(f"Built bracket: {bracket['tournament']} {bracket['year']} ({bracket['tour']}, {bracket['surface']}) "
          f"- {len(bracket['players'])} players ({bye_count} byes, {tbd_count} TBD/qualifier placeholders)")

    with open(args.output_path, "w", encoding="utf-8") as f:
        yaml.dump(bracket, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote {args.output_path}")
