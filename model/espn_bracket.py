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
  - Some Round 1 slots can be 'TBD' - the draw was released before qualifying finished. Many of
    these aren't actually unknown, though: ESPN's feed carries the Qualifying Final matches for
    the same event, and if one has already gone final ('post' state, real score), its winner IS
    that TBD slot's occupant - so it's resolved to that winner's real name through the normal
    ratings-CSV pipeline, same as any other player, instead of a placeholder. Only a TBD slot
    with no decided Qualifying Final winner left to assign gets a genuine, uniquely-named
    placeholder entry (e.g. "TBD (Qualifier 1)") - it never appears in any historical match data,
    so bracket.py's existing tier-3 fallback (STARTING_ELO placeholder for a player with no
    training-window history) picks it up automatically.

Usage:
    python model/espn_bracket.py --tour atp --event-id 718-2026 --surface Hard brackets/cincinnati_2026_atp.yaml
"""
import argparse
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket import TOUR_CONFIG, _build_ratings_index  # noqa: E402
from elo_ratings import SURFACES  # noqa: E402
from hybrid_simulation import match_espn_name_to_draw  # noqa: E402
from live_scores import LiveScoresError, fetch_scoreboard  # noqa: E402
from parse_atp_draw import _truncation_length  # noqa: E402

TOUR_SINGLES_CATEGORY = {"atp": "Men's Singles", "wta": "Women's Singles"}


def _known_ratings_names(tour):

    ratings_path = TOUR_CONFIG[tour.upper()].ratings_path
    if not ratings_path.exists():
        return []
    ratings_df = pd.read_csv(ratings_path)
    return list(set(_build_ratings_index(ratings_df).values()))


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


def _resolve_player_name(display_name, short_name, ratings_names, name_aliases):
    """Returns a final ratings-csv-style name ('Lastname X.') for a real (non-TBD) player.
    Tries the ratings-csv cross-reference first (exact, reuses match_espn_name_to_draw,
    including the manual alias table for the handful of players whose ratings-csv name doesn't
    share an initial with their real first name at all, e.g. 'Osorio M.' for Camila Osorio);
    falls back to a fresh best-effort truncation (single initial) only for players not already
    known - collision-widening against other same-lastname fallback players in this draw
    happens as a separate pass afterward, mirroring parse_atp_draw.py's own to_yaml_players
    logic."""
    matched = match_espn_name_to_draw(display_name, ratings_names, name_aliases) if ratings_names else None
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


def _standard_seed_regions(num_seeds):
    """Standard tennis single-elimination bracket seeding order (the "S-curve"): returns a list
    of length num_seeds where element i (0-indexed, seed rank i+1) is that seed's 1-indexed
    REGION number out of num_seeds total regions - seed 1 -> region 1 (top of the draw), seed 2
    -> the last region (opposite half, so 1 and 2 can only meet in the final), seeds 3-4 -> the
    two remaining half-boundary regions (so either can only meet 1 or 2 in the semifinal), and so
    on recursively. This is the same publicly documented algorithm every seeded single-elim
    bracket (majors, Masters, NCAA, etc.) uses to keep top seeds apart for as long as possible.
    num_seeds must be a power of 2."""
    positions = [1]
    size = 1
    while size < num_seeds:
        size *= 2
        positions = [p for x in positions for p in (x, size + 1 - x)]
    return positions


def _reorder_by_standard_seeding(raw_records):
    """Fixes a real structural bug: ESPN's scoreboard gives real Round-1 pairings correctly but
    in NO reliable bracket order (confirmed by inspection - e.g. the 2026 US Open WTA draw had
    seed 2 landing inside seed 1's own half of the raw list, seeds 5/6/7 clustered a few matches
    apart, etc.) - just whatever order ESPN happened to list the matches in (looks schedule/
    court-driven, not bracket-position-driven). Left as-is, this can put two top seeds who the
    real published draw keeps apart until the final (e.g. a #3 and #8 seed) into the SAME quarter
    here instead, corrupting every downstream tournament_win_probability that depends on who can
    actually meet whom before the final.

    Reconstructs the correct region for every SEEDED player via the standard seeding algorithm
    (_standard_seed_regions) and moves their real, already-correct R1 pair into that region - so
    every seed ends up in its correct canonical quarter/eighth/etc, and every real Round-1 result stays
    completely unchanged (only the ORDER of pairs is corrected, never who plays whom). The
    unseeded-vs-unseeded pairs that share a region with a seed can't be placed with certainty this
    way (their exact position isn't recoverable from ESPN's data at all - only the official draw
    sheet has it) - they're distributed into the remaining region slots in their original relative
    ESPN order, a disclosed limitation. This does not affect the specific bug being fixed (top
    seeds ending up in the wrong quarter/section), since only one seed exists per region by
    construction and unseeded players essentially never contest the parts of tournament_win_
    probability this matters for.

    Only applies to a bye-free draw (every record is a real Round-1 pairing, e.g. a Grand Slam's
    128-draw) where the seed count is a power of two and every rank 1..num_seeds is present -
    returns raw_records unchanged (with a printed warning) if those preconditions don't hold,
    rather than silently producing a wrong reorder."""
    if any(r["kind"] != "round1" for r in raw_records) or len(raw_records) % 2 != 0:
        print("  (skipping standard-seeding reorder: draw has byes/non-round1 entries - untested "
              "for that shape, leaving ESPN's original order)")
        return raw_records

    pairs = [(raw_records[i], raw_records[i + 1]) for i in range(0, len(raw_records), 2)]
    seeded_pairs = {}
    unseeded_pairs = []
    for pair in pairs:
        seeds_in_pair = [r["seed"] for r in pair if r["seed"] is not None]
        if len(seeds_in_pair) == 1:
            seeded_pairs[seeds_in_pair[0]] = pair
        elif len(seeds_in_pair) == 0:
            unseeded_pairs.append(pair)
        else:
            print(f"  (skipping standard-seeding reorder: two seeded players drawn together in "
                  f"Round 1 - unexpected, leaving ESPN's original order)")
            return raw_records

    num_seeds = len(seeded_pairs)
    if num_seeds == 0 or (num_seeds & (num_seeds - 1)) != 0 or set(seeded_pairs) != set(range(1, num_seeds + 1)):
        print(f"  (skipping standard-seeding reorder: {num_seeds} seeded pairs found, not a clean "
              f"1..N power-of-two seed list - leaving ESPN's original order)")
        return raw_records
    if len(pairs) % num_seeds != 0:
        print(f"  (skipping standard-seeding reorder: {len(pairs)} Round-1 pairs doesn't divide "
              f"evenly by {num_seeds} seeds - leaving ESPN's original order)")
        return raw_records

    pairs_per_region = len(pairs) // num_seeds
    regions = _standard_seed_regions(num_seeds)  # regions[i] = region (1-indexed) for seed i+1
    region_pairs = {r: [] for r in range(1, num_seeds + 1)}
    for seed_rank, pair in seeded_pairs.items():
        region_pairs[regions[seed_rank - 1]].append(pair)

    unseeded_iter = iter(unseeded_pairs)
    ordered_pairs = []
    for region in range(1, num_seeds + 1):
        slots = region_pairs[region]
        while len(slots) < pairs_per_region:
            slots.append(next(unseeded_iter))
        ordered_pairs.extend(slots)

    return [record for pair in ordered_pairs for record in pair]


# Latin letters NFKD does NOT decompose into base+combining-mark (unlike most accented vowels,
# e.g. 'ú' -> 'u' + combining acute) - these are distinct code points with no ASCII decomposition
# at all, so unicodedata.normalize alone silently leaves them untouched. Confirmed by a real crash:
# 'Laslo Đere' (Serbian, stroke-D) stayed 'Đere' through _strip_diacritics, never matched the
# ratings pool's 'Dere L.', and fell all the way through to an unresolved player_2 that win_
# probability() then couldn't look up at all. Small, explicit map - not exhaustive, extend as new
# real cases surface (same "disclosed, not silently wrong" convention as everything else here).
_NON_DECOMPOSING_LETTERS = str.maketrans({
    "Đ": "D", "đ": "d", "Ł": "L", "ł": "l", "Ø": "O", "ø": "o", "Æ": "AE", "æ": "ae", "ß": "ss",
})


def _strip_diacritics(text):
    """The ratings-csv name pool is ASCII-normalized (e.g. 'Zarazua R.', 'Bondar A.' - no accents),
    but Wikipedia draw names carry real diacritics ('Renata Zarazúa', 'Anna Bondár') - normalize
    before matching so accented names aren't spuriously left unmatched."""
    text = text.translate(_NON_DECOMPOSING_LETTERS)
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _fetch_wikipedia_wikitext(title):
    response = None
    for attempt in range(4):
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "parse", "page": title, "prop": "wikitext", "format": "json"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        if response.status_code != 429:
            break
        wait = int(response.headers.get("Retry-After", 30)) * (attempt + 1)
        print(f"  (Wikipedia rate limit hit fetching {title!r}, waiting {wait}s before retry)")
        time.sleep(wait)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise ValueError(f"Wikipedia page {title!r} not found: {data['error']}")
    return data["parse"]["wikitext"]["*"]


def _extract_wiki_name(raw_value):
    value = raw_value.strip()
    link = re.search(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", value)
    name = link.group(1) if link else ""
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)  # strip Wikipedia disambiguators, e.g. "Ann Li (tennis)"
    return _strip_diacritics(name)


def _fetch_wikipedia_draw_order(title):
    """Returns the real, true bracket-position-ordered list of player display names (one per
    Round-1 slot, "" for a slot not yet named - an undecided qualifier/TBD) for a Slam's singles
    draw, read directly from its Wikipedia article's wikitext (e.g. "2026 US Open - Women's
    singles"). Confirmed by inspection: these articles' "Draw" section is built from a sequence
    of {{16TeamBracket-Compact-TennisN}} templates, one per 16-player section, each carrying
    'RD1-team01' through 'RD1-team16' in real, true Round-1 bracket-slot order (adjacent pairs
    are real Round-1 opponents, section-to-section order is the real left-to-right draw order) -
    this is the actual published draw, not schedule/court order like ESPN's own feed. The
    top-of-page "Finals" summary template uses single-digit 'RD1-team1'..'RD1-team8' keys (always
    blank placeholders) - the two-digit-only regex below skips it automatically, no special-casing
    needed.

    Bye-free only (every Slam/combined-draw article this is used for is a clean 128 with no
    byes) - see _fetch_wikipedia_draw_order_with_byes for the Masters-1000-shaped case, which
    needs real bye reconstruction this function deliberately doesn't attempt."""
    wikitext = _fetch_wikipedia_wikitext(title)
    return [_extract_wiki_name(m.group(1)) for m in re.finditer(r"RD1-team\d{2}=(.*)", wikitext)]


def _fetch_wikipedia_draw_order_with_byes(title):
    """Like _fetch_wikipedia_draw_order, but reconstructs real byes too - needed for Masters
    1000-shaped draws (56/96 players in a 64/128-slot bracket), which use a DIFFERENT Wikipedia
    template ('...-Byes' suffix, confirmed by inspection: '2022 Mutua Madrid Open' uses
    '16TeamBracket-Compact-Tennis3-Byes') with a real structural difference, not just a naming
    one: a bye player gets NO 'RD1-teamNN' entry at all (both slots of their would-be Round-1
    pair are simply absent from the wikitext) - they appear for the first time directly as an
    'RD2-teamNN' entry. A plain RD1-only scan (what _fetch_wikipedia_draw_order does) silently
    undercounts these draws by exactly the bye count, which is the real bug this fixes (confirmed
    directly: a 64-slot Masters draw with 16 byes came back as "48 non-bye players, 0 byes
    detected" - not a valid bracket size, since 24 Round-1 matchups is not a power of two on its
    own without the missing 16 byes accounted for).

    Reconstruction: for each RD2 slot k, its two corresponding RD1 slots are (2k-1, 2k). If BOTH
    are present in RD1, that's a normal real Round-1 match (both non-bye). If BOTH are absent,
    the RD2-team(k) entry IS the bye player (real player, no Round-1 opponent) - confirmed this
    is always a same-section-boundary numbering: sections restart their local slot count for
    each RD2 group, so RD1 slot numbers are read as (section_offset + local_slot) using the same
    16-slots-per-RD1-section grouping the plain templates use.

    Returns (names, is_bye) - two parallel lists in true bracket-adjacent order (byes occupy a
    single slot at their true position, non-bye Round-1 pairs occupy two adjacent slots each,
    same convention this project's own bracket schema already uses elsewhere).

    IMPORTANT: RD1/RD2 slot NUMBERS restart at 01 independently in EACH {{...Bracket...}}
    template on the page (one template per ~16-player section) - only the numbers are local, the
    real draw order is the sequence of templates themselves (document order). So gap-detection
    has to run separately per template block, never against one page-wide slot-number dict (that
    would silently collide/overwrite same-numbered slots from different sections)."""
    wikitext = _fetch_wikipedia_wikitext(title)
    block_starts = [m.start() for m in re.finditer(r"\{\{\d*TeamBracket", wikitext)]
    blocks = [
        wikitext[start: block_starts[i + 1] if i + 1 < len(block_starts) else len(wikitext)]
        for i, start in enumerate(block_starts)
    ]

    names, is_bye = [], []
    for block in blocks:
        rd1 = {int(m.group(1)): _extract_wiki_name(m.group(2))
               for m in re.finditer(r"RD1-team(\d{2})=(.*)", block)}
        rd2 = {int(m.group(1)): _extract_wiki_name(m.group(2))
               for m in re.finditer(r"RD2-team(\d{2})=(.*)", block)}
        if not rd2:
            continue  # a non-draw template (e.g. the top-of-page "Finals" summary) - skip
        for k in range(1, max(rd2) + 1):
            slot_a, slot_b = 2 * k - 1, 2 * k
            a_present, b_present = slot_a in rd1, slot_b in rd1
            if a_present and b_present:
                names.append(rd1[slot_a]); is_bye.append(False)
                names.append(rd1[slot_b]); is_bye.append(False)
            elif not a_present and not b_present:
                # neither Round-1 slot exists at all - the RD2 entry itself is the real bye
                # player, if this Wikipedia edition actually recorded one for this slot (an
                # as-yet-undecided qualifier slot can also be legitimately blank here - "" is
                # kept, not skipped, same "disclosed missing" convention the bye-free fetcher
                # already uses).
                names.append(rd2.get(k, "")); is_bye.append(True)
            else:
                # exactly one Round-1 slot present, the other absent - not a shape this
                # reconstruction handles with confidence (never observed in inspection); keep the
                # present player as non-bye and drop the absent slot rather than guessing at a
                # bye that may not be real.
                present_slot = slot_a if a_present else slot_b
                names.append(rd1[present_slot]); is_bye.append(False)
    return names, is_bye


def _reorder_by_wikipedia_draw(raw_records, wiki_title, name_aliases):
    """Reorders real Round-1 pairs into their true bracket position using the tournament's
    Wikipedia draw article as ground truth - strictly more accurate than
    _reorder_by_standard_seeding, since it recovers the correct position for EVERY player
    (seeded or not), not just the 32 seeds. Every real Round-1 pairing stays exactly as ESPN
    reported it; only the order of pairs is corrected.

    Matches each Wikipedia name to this draw's own already-resolved player-name pool via
    match_espn_name_to_draw (the same fuzzy matcher used for ESPN's own display names - both are
    "Firstname Lastname" strings being matched against "Lastname X." ratings-csv names). A pair
    is positioned as soon as ONE of its two players is uniquely matched - the other slot can stay
    blank on Wikipedia (an undecided qualifier) without blocking placement. Any pair that matches
    nothing (a name Wikipedia and ESPN's ratings-csv pool disagree on) is left unpositioned and
    appended, in original ESPN order, after every positioned pair - printed as a warning rather
    than silently mis-ordered.

    Raises the same preconditions as _reorder_by_standard_seeding (bye-free, even-length draw)
    since a Wikipedia Slam draw article always follows that shape."""
    if any(r["kind"] != "round1" for r in raw_records) or len(raw_records) % 2 != 0:
        print("  (skipping Wikipedia-draw reorder: draw has byes/non-round1 entries - untested "
              "for that shape, leaving ESPN's original order)")
        return raw_records

    wiki_names = _fetch_wikipedia_draw_order(wiki_title)
    if len(wiki_names) != len(raw_records):
        print(f"  (skipping Wikipedia-draw reorder: page has {len(wiki_names)} draw slots, "
              f"ESPN has {len(raw_records)} - draw-size mismatch, leaving ESPN's original order)")
        return raw_records

    pairs = [(raw_records[i], raw_records[i + 1]) for i in range(0, len(raw_records), 2)]
    resolved_pool = {r["name"] for r in raw_records if r["name"]}

    # resolved-name (ratings-csv "Lastname X." format) -> wiki pair index (0-based), first match wins
    wiki_pair_index_by_resolved_name = {}
    for position, wiki_name in enumerate(wiki_names):
        if not wiki_name:
            continue
        matched = match_espn_name_to_draw(wiki_name, resolved_pool, name_aliases)
        if matched is not None:
            wiki_pair_index_by_resolved_name.setdefault(matched, position // 2)

    slotted = {}  # wiki pair index -> our pair
    unmatched = []
    for pair in pairs:
        pair_index = next(
            (wiki_pair_index_by_resolved_name[r["name"]] for r in pair
             if r["name"] in wiki_pair_index_by_resolved_name),
            None,
        )
        if pair_index is None or pair_index in slotted:
            unmatched.append(pair)
        else:
            slotted[pair_index] = pair

    if unmatched:
        unmatched_names = [r["name"] or "?" for pair in unmatched for r in pair]
        print(f"  (Wikipedia-draw reorder: {len(unmatched)} of {len(pairs)} Round-1 pairs could not be "
              f"matched to a Wikipedia draw slot - left in original ESPN order, appended after every "
              f"matched pair: {unmatched_names})")

    ordered_pairs = [slotted[i] for i in sorted(slotted)] + unmatched
    return [record for pair in ordered_pairs for record in pair]


def _decided_qualifying_winners(qualifying_final):
    """Competitors who won a Qualifying Final ESPN already marked 'post' (final, real score) -
    real, already-decided players who belong in a Round 1 TBD slot, not placeholders. A
    Qualifying Final still 'pre' or 'in' hasn't concluded, so it contributes nothing - that's
    the genuinely-unknown case a TBD placeholder is still for."""
    winners = []
    for competition in qualifying_final:
        status_state = ((competition.get("status") or {}).get("type") or {}).get("state")
        if status_state != "post":
            continue
        winner = next((c for c in competition.get("competitors", []) if c.get("winner") is True), None)
        if winner is not None:
            winners.append(winner)
    return winners


def build_bracket_players(tour, event_id, dates=None, wiki_title=None):
    """Returns a list of player dicts (seed/name/bye/status) in bracket order, plus the raw
    ESPN event dict (used by the caller for tournament/date metadata).

    dates is passed straight through to fetch_scoreboard - ESPN's undated scoreboard defaults
    to "today" server-side, which only finds a tournament while it's still live/recent (see
    backtest_hard_court.py's own note on this). A single date anywhere inside the tournament's
    window returns its complete event regardless of how long ago it finished, so building a
    bracket for an already-concluded event needs one explicitly.

    wiki_title, if given (e.g. "2026 US Open - Women's singles"), uses that Wikipedia article's
    draw as the true bracket-order ground truth (_reorder_by_wikipedia_draw) - strictly more
    accurate than the standard-seeding fallback since it positions every player correctly, not
    just the 32 seeds. Falls back to _standard_seed_regions (seeds only) when omitted."""
    tour = tour.lower()
    category = TOUR_SINGLES_CATEGORY[tour]
    data = fetch_scoreboard(tour, dates=dates)

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
    qualifying_final = [c for c in competitions if (c.get("round") or {}).get("displayName") == "Qualifying Final"]
    if not round1:
        raise ValueError(f"No Round 1 matches found for event {event_id} / {category}")

    qualifier_winners = iter(_decided_qualifying_winners(qualifying_final))

    ratings_names = _known_ratings_names(tour)
    name_aliases = TOUR_CONFIG[tour.upper()].name_aliases

    # bye detection compares RAW ESPN display names, not resolved ratings-csv names - a
    # player's displayName is guaranteed identical between their Round 1 and Round 2 JSON
    # entries (same athlete object), whereas resolved names aren't a safe comparison key: two
    # calls can independently land on different results (name-matching ambiguity, or - for a
    # genuine newcomer with no ratings-csv entry - both stay unresolved until the later
    # collision-truncation pass). Comparing raw names first, before any resolution happens,
    # sidesteps that whole class of bug rather than depending on resolution being consistent.
    round1_display_names = set()
    for competition in round1:
        for competitor in competition.get("competitors", []):
            name = (competitor.get("athlete") or {}).get("displayName")
            if name and name != "TBD":
                round1_display_names.add(name)

    # pass 1: resolve every real (non-TBD) name to either an already-known ratings-csv name, or
    # a (lastname, firstname) pair still needing the collision-truncation pass
    raw_records = []  # each: dict(kind, seed, resolved_name_or_None, fallback_key_or_None)
    fallback_players = {}  # index into raw_records -> (lastname, firstname), for collision pass
    tbd_counter = 0
    qualifiers_resolved = 0

    def add_record(kind, competitor):
        nonlocal tbd_counter, qualifiers_resolved
        athlete = competitor.get("athlete") or {}
        display_name = athlete.get("displayName")

        if not display_name or display_name == "TBD":
            qualifier = next(qualifier_winners, None)
            if qualifier is None:
                tbd_counter += 1
                raw_records.append({"kind": kind, "seed": None, "name": f"TBD (Qualifier {tbd_counter})"})
                return
            qualifiers_resolved += 1
            competitor = qualifier
            athlete = competitor.get("athlete") or {}
            display_name = athlete.get("displayName")

        seed = (competitor.get("curatedRank") or {}).get("current")
        resolved, fallback_key = _resolve_player_name(display_name, athlete.get("shortName"), ratings_names, name_aliases)
        record = {"kind": kind, "seed": seed, "name": resolved}
        if fallback_key is not None:
            fallback_players[len(raw_records)] = fallback_key
        raw_records.append(record)

    for competition in round1:
        by_order = sorted(competition.get("competitors", []), key=lambda c: c.get("order", 0))
        for competitor in by_order:
            add_record("round1", competitor)

    for competition in round2:
        for competitor in competition.get("competitors", []):
            display_name = (competitor.get("athlete") or {}).get("displayName")
            if not display_name or display_name == "TBD":
                continue  # the pending Round 1 winner - already represented by its Round 1 entry
            if display_name in round1_display_names:
                continue  # the Round 1 winner, now confirmed and advancing - not a bye
            add_record("round2_bye", competitor)

    resolved_fallbacks = _resolve_fallback_collisions(fallback_players)
    for index, name in resolved_fallbacks.items():
        raw_records[index]["name"] = name

    if wiki_title:
        raw_records = _reorder_by_wikipedia_draw(raw_records, wiki_title, name_aliases)
    else:
        raw_records = _reorder_by_standard_seeding(raw_records)

    players = []
    for record in raw_records:
        players.append({
            "seed": record["seed"],
            "name": record["name"],
            "status": None,
            "bye": record["kind"] == "round2_bye",
        })
    qualifier_stats = {"resolved": qualifiers_resolved, "unresolved": tbd_counter}
    return players, event, round1, qualifier_stats


def build_historical_bracket_players(wiki_title, ratings_names, name_aliases):
    """Builds a bracket player list directly from a CONCLUDED tournament's Wikipedia draw
    article - no ESPN dependency at all, unlike build_bracket_players (the live path). A
    historical Wikipedia draw page already has the complete, final, correctly-ordered real
    bracket, so there's nothing left for ESPN to contribute - ESPN exists in the live path only
    to supply real names before Wikipedia's own page is filled in for a still-upcoming
    tournament. Real ordered names come from _fetch_wikipedia_draw_order, unchanged.

    Byes: two real, distinct shapes seen in practice, both handled here. (1) A literal 'Bye'
    opponent text in an otherwise-normal Round-1 pair (checked directly, case-insensitive). (2) A
    Masters-1000-shaped draw (a DIFFERENT Wikipedia template, '...-Byes' suffix), where a bye
    player gets NO Round-1 entry at all - not even a literal 'Bye' placeholder - and only
    appears for the first time in Round 2. That shape needs real reconstruction
    (_fetch_wikipedia_draw_order_with_byes) - a plain Round-1-only scan silently undercounts
    these draws by exactly the bye count (confirmed: this is what caused real "not a valid
    bracket size" failures on Masters 1000 editions before this was added). Always uses the
    byes-aware fetch now, not just as a fallback - confirmed (direct comparison against a known
    bye-free Slam draw) it returns identical output when there's nothing to reconstruct, so
    there's no separate bye-free code path to keep in sync.

    Name resolution mirrors _resolve_player_name/_resolve_fallback_collisions (no ESPN
    shortName available from Wikipedia, so the fallback splits on the raw display name alone).

    Returns (players, unmatched_names) - unmatched_names lists any real Wikipedia name that
    fell through to the fallback splitter without a matching prior ratings-csv entry (a genuine
    name-matching gap - printed by the caller, never silently dropped)."""
    # always the byes-aware fetch, never the plain one - confirmed (by direct comparison against
    # a known bye-free Slam draw) it returns IDENTICAL output when there are no real byes to
    # reconstruct, so there's no bye-free-case regression risk, and it avoids the wrong signal a
    # length-parity check on the PLAIN fetch would give: a byes draw's plain (Round-1-only) scan
    # can still come back an EVEN count (e.g. 48, 8 byes silently missing) since the byes just
    # never appear there at all rather than leaving a visible odd-length gap.
    wiki_names, wiki_is_bye = _fetch_wikipedia_draw_order_with_byes(wiki_title)
    if not wiki_names or len(wiki_names) % 2 != 0:
        raise ValueError(f"{wiki_title!r}: couldn't reconstruct a clean, even-length draw "
                          f"({len(wiki_names)} slots)")

    raw_records = []  # each: dict(name=resolved_or_None, bye=bool)
    fallback_players = {}  # index into raw_records -> (lastname, firstname), for collision pass
    unmatched = []

    def add(name, bye):
        matched, fallback_key = _resolve_player_name(name, None, ratings_names, name_aliases)
        record = {"name": matched, "bye": bye}
        if fallback_key is not None:
            fallback_players[len(raw_records)] = fallback_key
            unmatched.append(name)
        raw_records.append(record)

    i = 0
    while i < len(wiki_names):
        if wiki_is_bye[i]:
            # a real bye reconstructed by _fetch_wikipedia_draw_order_with_byes - occupies a
            # single slot on its own, no pairing with the next entry.
            name = wiki_names[i]
            if name:
                add(name, bye=True)
            i += 1
            continue
        a, b = wiki_names[i], wiki_names[i + 1]
        a_bye, b_bye = a.strip().lower() == "bye", b.strip().lower() == "bye"
        if a_bye and b_bye:
            i += 2
            continue  # no real player on either side (an undecided qualifier slot) - skip
        if a_bye or b_bye:
            real_name = b if a_bye else a
            if real_name:
                add(real_name, bye=True)
            i += 2
            continue
        for name in (a, b):
            if name:
                add(name, bye=False)
            else:
                raw_records.append({"name": None, "bye": False})  # undecided qualifier slot
        i += 2

    resolved_fallbacks = _resolve_fallback_collisions(fallback_players)
    for index, name in resolved_fallbacks.items():
        raw_records[index]["name"] = name

    players = [{"seed": None, "name": r["name"], "status": None, "bye": r["bye"]} for r in raw_records]
    return players, unmatched


def _tournament_start_date(event, round1):
    dated = [c.get("date") for c in round1 if c.get("date")]
    date_str = min(dated) if dated else event.get("date")
    return date_str[:10] if date_str else None


def build_bracket_yaml(tour, event_id, surface, dates=None, wiki_title=None):
    tour = tour.upper()
    if surface not in SURFACES:
        raise ValueError(f"surface must be one of {SURFACES}, got {surface!r}")

    players, event, round1, qualifier_stats = build_bracket_players(tour, event_id, dates=dates, wiki_title=wiki_title)
    start_date = _tournament_start_date(event, round1)
    year = int(start_date[:4]) if start_date else None

    bracket = {
        "tournament": event.get("name"),
        "year": year,
        "tour": tour,
        "surface": surface,
        "start_date": start_date,
        "draw_size": len(players),
        "players": players,
    }
    return bracket, qualifier_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--tour", choices=["atp", "wta"], required=True)
    parser.add_argument("--event-id", required=True, help="ESPN event id, e.g. 718-2026")
    parser.add_argument("--surface", required=True, choices=SURFACES,
                         help="ESPN's scoreboard data has no surface field - must be supplied")
    parser.add_argument("--dates", default=None,
                         help="YYYYMMDD, passed to ESPN's ?dates= - needed for an already-"
                              "concluded event, which the undated (\"today\") scoreboard can't find")
    parser.add_argument("--wiki-title", default=None,
                         help="Wikipedia article title for this draw (e.g. \"2026 US Open - "
                              "Women's singles\") - if given, corrects the real bracket position "
                              "of every player (seeded or not) using that article's draw as "
                              "ground truth, instead of the seeds-only standard-seeding fallback")
    args = parser.parse_args()

    try:
        bracket, qualifier_stats = build_bracket_yaml(args.tour, args.event_id, args.surface, dates=args.dates,
                                                        wiki_title=args.wiki_title)
    except (LiveScoresError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    real_count = sum(1 for p in bracket["players"] if not str(p["name"]).startswith("TBD"))
    tbd_count = len(bracket["players"]) - real_count
    bye_count = sum(1 for p in bracket["players"] if p["bye"])
    print(f"Built bracket: {bracket['tournament']} {bracket['year']} ({bracket['tour']}, {bracket['surface']}) "
          f"- {len(bracket['players'])} players ({bye_count} byes, {tbd_count} TBD/qualifier placeholders)")
    if qualifier_stats["resolved"] or qualifier_stats["unresolved"]:
        print(f"  Round 1 TBD slots: {qualifier_stats['resolved']} resolved to already-decided "
              f"qualifying winners, {qualifier_stats['unresolved']} still genuinely unknown "
              f"(qualifying not concluded)")

    with open(args.output_path, "w", encoding="utf-8") as f:
        yaml.dump(bracket, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote {args.output_path}")
