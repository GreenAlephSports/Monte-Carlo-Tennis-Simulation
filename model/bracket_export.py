"""Exports one JSON file per sim run, matching Daron's integration spec:
  - every player keyed by the exact ESPN competitor.athlete.displayName - no name matching on
    our side, byte-for-byte, everywhere in the output (our internal ratings-csv-style names,
    e.g. "Sinner J.", never appear here - see espn_to_draw/draw_to_espn below).
  - match IDs are half-prefixed (T-/B-) + Daron's round label (R1/R2/R3/R16/QF/SF/F) + a
    sequential index within that half/round, e.g. "T-QF-1"; the Final is just "Final" (the one
    cross-half match). Each match has slot_a/slot_b, "probability" always meaning P(slot_a wins).
  - player rows are quarter-based (Q1-Q4, per Daron's correction): p_champ, p_sf (wins the
    quarter = reaches the semifinal), p_final (wins the half = reaches the final).
  - probabilities prefer The Odds API's de-vigged match odds wherever a real matchup is known
    (even if not yet decided) and priced there; otherwise fall back to this project's own
    simulated/Elo-based probability.

Reuses hybrid_simulation.py's real-result and known-but-undecided-pairing extraction
(build_real_results_by_round / build_known_pairings_by_round / known_matchups_for_round /
replay_real_rounds) rather than re-deriving match state - this module only adds the Daron-shaped
output layer on top of that.

Quarter/half placement (see tag_halves_and_quarters()) is derived once, structurally, from the
draw's fixed Round 1 -> Round 2 bracket tree - never from which matches happen to be decided, so
every player's tag is stable from the moment the draw is set, not just once they reach Round 2.

Usage:
    python model/bracket_export.py brackets/cincinnati_2026_atp.yaml --simulations 5000
"""
import argparse
import difflib
import json
import os
import random
import re
import sys
import urllib.parse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket import (  # noqa: E402
    TOUR_CONFIG, get_matchups, match_draw_to_ratings, order_by_draw_position, split_byes,
    validate_bracket_structure, validate_draw,
)
from bracket_schema import BracketValidationError, load_bracket_yaml  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from ev_comparison import implied_probabilities  # noqa: E402
from hybrid_simulation import (  # noqa: E402
    TOUR_SINGLES_CATEGORY, build_known_pairings_by_round, build_real_results_by_round,
    known_matchups_for_round, match_espn_name_to_draw, reconstruct_leaves_by_round2_slot,
    replay_real_rounds, true_bracket_order,
)
from live_scores import LiveScoresError, extract_matches, fetch_scoreboard  # noqa: E402
from simulate import N_SIMULATIONS, run_simulations_tracking_milestones  # noqa: E402
from win_probability import win_probability  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
SEED = 42
ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def _load_env():
    """Tiny KEY=VALUE .env loader - avoids adding python-dotenv as a dependency for one key."""
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env()
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")


def _http_get_json(url, params, timeout=15):
    query = urllib.parse.urlencode(params)
    request = Request(f"{url}?{query}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


# ESPN's tournament name is sometimes a sponsor-branded title that shares no meaningful word with
# The Odds API's own (often older/historical) title for the same event - confirmed for the
# Canadian Open, which ESPN reports as "National Bank Open presented by Rogers" (current title
# sponsor) but The Odds API still lists as "WTA/ATP Canadian Open" (the tournament's long-standing
# name). Measured: fuzzy_similarity("National Bank Open presented by Rogers", "Canadian Open")
# scores below even unrelated tournaments like "Australian Open"/"Italian Open" (incidental
# "...ian Open" substring overlap outscores the real match) - a text-similarity match genuinely
# can't bridge a sponsor-branding swap like this, so it stays a hardcoded, logged last resort
# rather than something fuzzy matching should be tuned to catch. Checked only after
# _best_fuzzy_match finds nothing above FUZZY_MATCH_MIN_SCORE - real tournaments discovered by
# fuzzy matching stay untouched by this table.
TOURNAMENT_NAME_ALIASES = {
    "national bank open": "canadian open",
}

FUZZY_MATCH_MIN_SCORE = 0.6

# generic tournament-naming boilerplate that would otherwise inflate token overlap between two
# unrelated events (nearly every title contains "Open") - stripped before scoring so overlap
# reflects the tournament's actual identity (city/sponsor/name), not shared filler.
_TITLE_FILLER_WORDS = {
    "open", "championship", "championships", "cup", "masters", "the", "of", "and", "club",
    "presented", "by", "international", "internazionali", "invitational",
}


def _meaningful_tokens(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _TITLE_FILLER_WORDS}


def _fuzzy_similarity(name_a, name_b):
    """Best of two signals: Jaccard overlap of meaningful tokens (order-independent, catches
    reordered or partial-name matches, e.g. 'Miami Open' vs 'Miami Masters') and difflib's
    character-level ratio (catches close spelling variants token overlap alone would miss).
    Taking the max is deliberately generous - a real match strong on either axis should pass -
    but still leaves genuinely unrelated names (e.g. a sponsor-branded title with zero token
    overlap and only incidental character overlap) below FUZZY_MATCH_MIN_SCORE."""
    tokens_a, tokens_b = _meaningful_tokens(name_a), _meaningful_tokens(name_b)
    token_overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b) if (tokens_a or tokens_b) else 0.0
    char_ratio = difflib.SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio()
    return max(token_overlap, char_ratio)


def _best_fuzzy_match(query_name, candidates):
    """candidates: sport dicts already filtered to the right tour. Returns (chosen_sport, score) -
    the highest-scoring candidate if it clears FUZZY_MATCH_MIN_SCORE, else (None, best_score_seen)
    so a rejected best-guess can still be logged by the caller."""
    scored = []
    for s in candidates:
        # score against the title with the leading tour word ('ATP '/'WTA ') stripped, so that
        # shared prefix never inflates the similarity of an otherwise-unrelated tournament.
        title_body = s.get("title", "").split(" ", 1)[-1]
        scored.append((_fuzzy_similarity(query_name, title_body), s))
    if not scored:
        return None, 0.0
    scored.sort(key=lambda pair: (pair[0], pair[1].get("active", False)), reverse=True)
    best_score, best_sport = scored[0]
    if best_score >= FUZZY_MATCH_MIN_SCORE:
        return best_sport, best_score
    return None, best_score


def discover_odds_sport_key(tour, tournament_name, api_key):
    """The Odds API's tennis sport keys are per-tournament (e.g. 'tennis_atp_cincinnati_open')
    and only exist while that event is currently listed - discovered by fuzzy title match against
    /v4/sports rather than hardcoded, so this isn't wired to one specific tournament. Falls back to
    TOURNAMENT_NAME_ALIASES, logged loudly, only for the rare sponsor-branded title fuzzy matching
    can't bridge (see that table's docstring) - every other tournament is resolved by text
    similarity alone, with no hardcoded name."""
    try:
        sports = _http_get_json(f"{ODDS_API_BASE}/sports/", {"apiKey": api_key, "all": "true"})
    except (HTTPError, URLError) as e:
        print(f"WARNING: couldn't list The Odds API sports ({e}) - odds pricing unavailable", file=sys.stderr)
        return None

    tour_word = "ATP" if tour == "ATP" else "WTA"
    candidates = [
        s for s in sports if s.get("group") == "Tennis" and s.get("title", "").upper().startswith(tour_word)
    ]

    chosen, score = _best_fuzzy_match(tournament_name, candidates)
    if chosen is not None:
        return chosen["key"]

    alias_body = next(
        (alias for espn_prefix, alias in TOURNAMENT_NAME_ALIASES.items()
         if tournament_name.lower().startswith(espn_prefix)),
        None,
    )
    if alias_body is None:
        print(
            f"WARNING: no Odds API tennis sport matched {tournament_name!r} ({tour}) - best fuzzy "
            f"candidate scored {score:.2f}, below the {FUZZY_MATCH_MIN_SCORE} threshold, and no "
            f"manual alias is configured for it - odds pricing unavailable", file=sys.stderr,
        )
        return None

    print(
        f"WARNING: fuzzy matching found no confident Odds API sport for {tournament_name!r} "
        f"(best candidate scored {score:.2f}, below the {FUZZY_MATCH_MIN_SCORE} threshold) - "
        f"falling back to the manual alias table (matched alias {alias_body!r}). If this fires for "
        f"a tournament that ISN'T a known sponsor-branding case, the alias table may be masking a "
        f"real mismatch - check it.", file=sys.stderr,
    )
    aliased_chosen, _alias_score = _best_fuzzy_match(alias_body, candidates)
    if aliased_chosen is not None:
        return aliased_chosen["key"]

    print(
        f"WARNING: manual alias {alias_body!r} for {tournament_name!r} still matched no Odds API "
        f"tennis sport - odds pricing unavailable", file=sys.stderr,
    )
    return None


def fetch_devigged_odds(sport_key, api_key):
    """Returns odds_lookup: frozenset({espn_name_a, espn_name_b}) -> {espn_name: p_win}, for
    every event The Odds API currently has bookmaker h2h data for. Prints each bookmaker's raw
    decimal odds alongside its own de-vigged implied probability, then averages decimal odds
    across every bookmaker quoting both sides and de-vigs that average via
    ev_comparison.implied_probabilities - the same de-vig math already used elsewhere in this
    codebase, not reimplemented here. Cross-referencing against ESPN names is exact-string only
    (never fuzzy) - a name that doesn't match byte-for-byte just falls back to the model, per the
    spec's own fallback rule, rather than risking a wrong join."""
    if not sport_key or not api_key:
        return {}
    try:
        events = _http_get_json(
            f"{ODDS_API_BASE}/sports/{sport_key}/odds/",
            {"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "decimal"},
        )
    except (HTTPError, URLError) as e:
        print(f"WARNING: The Odds API request failed ({e}) - falling back to model probabilities "
              f"everywhere", file=sys.stderr)
        return {}

    odds_lookup = {}
    for event in events:
        home, away = event.get("home_team"), event.get("away_team")
        if not home or not away:
            continue
        home_prices, away_prices = [], []
        book_rows = []
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                if home in outcomes and away in outcomes:
                    home_odd, away_odd = outcomes[home], outcomes[away]
                    home_prices.append(home_odd)
                    away_prices.append(away_odd)
                    home_pct, away_pct = implied_probabilities(home_odd, away_odd)
                    book_rows.append((bookmaker.get("title", bookmaker.get("key", "?")),
                                       home_odd, away_odd, home_pct, away_pct))
        if not home_prices:
            continue
        print(f"{home} vs {away} - de-vigged odds by bookmaker:")
        for title, home_odd, away_odd, home_pct, away_pct in book_rows:
            print(f"  {title:<20} {home}: {home_odd:.2f} -> {home_pct:.1%}    "
                  f"{away}: {away_odd:.2f} -> {away_pct:.1%}")
        avg_home = sum(home_prices) / len(home_prices)
        avg_away = sum(away_prices) / len(away_prices)
        prob_home, prob_away = implied_probabilities(avg_home, avg_away)
        odds_lookup[frozenset((home, away))] = {home: prob_home, away: prob_away}
    return odds_lookup


def resolve_probability(espn_name_a, espn_name_b, draw_name_a, draw_name_b, odds_lookup, surface, ratings_path):
    """P(a beats b) - The Odds API's de-vigged odds when this exact pairing (matched by exact
    ESPN displayName) has live bookmaker data; otherwise this project's own model. Returns
    (probability, source) - source is for our own console reporting, not part of Daron's schema."""
    entry = odds_lookup.get(frozenset((espn_name_a, espn_name_b)))
    if entry is not None and espn_name_a in entry:
        return entry[espn_name_a], "odds_api"
    return win_probability(draw_name_a, draw_name_b, surface, ratings_path), "model"


def build_round_label_map(round_sequence):
    """Maps ESPN's own round labels (Round 1, Round 2, ..., Quarterfinal, Semifinal, Final) to
    Daron's fixed convention (R1, R2, R3, R16, QF, SF, F). The round immediately before
    Quarterfinal is always exactly 16 players by construction, regardless of what sequential
    number it would otherwise be - everything earlier is numbered sequentially from 1."""
    tail_map = {"Quarterfinal": "QF", "Semifinal": "SF", "Final": "F"}
    numbered = [s for s in round_sequence if s not in tail_map]
    label_map = {}
    for i, stage in enumerate(numbered):
        if i == len(numbered) - 1 and "Quarterfinal" in round_sequence:
            label_map[stage] = "R16"
        else:
            label_map[stage] = f"R{i + 1}"
    for stage, label in tail_map.items():
        if stage in round_sequence:
            label_map[stage] = label
    return label_map


def tag_halves_and_quarters(leaves_by_slot):
    """Static per-(ESPN name) half (Top/Bottom) + quarter (Q1-Q4) tag, derived from Round 2's
    slot index alone (i = 0..n2-1) - the same stable, structural index _reconstruct_leaves_by_
    round2_slot uses, just split 2 ways (half) or 4 ways (quarter) instead of kept per-slot. No
    live result or decided-status is consulted - a player's quarter is identical whether their
    Round 1 match is 'pre', 'in', or 'post'."""
    n2 = len(leaves_by_slot)

    def _half(i):
        return "Top" if i < n2 / 2 else "Bottom"

    def _quarter(i):
        return f"Q{int(i // (n2 / 4)) + 1}"

    half_by_name, quarter_by_name = {}, {}
    for i, slot_leaves in enumerate(leaves_by_slot):
        for name, _is_bye in slot_leaves:
            half_by_name[name] = _half(i)
            quarter_by_name[name] = _quarter(i)

    return half_by_name, quarter_by_name


# bracket YAMLs are built before qualifying wraps up, so a draw-size's worth of Round 1 slots
# start out as literal placeholders like this rather than a real player name.
QUALIFIER_PLACEHOLDER_RE = re.compile(r"^TBD \(Qualifier \d+\)$")


def resolve_qualifier_placeholders(players, byes, tournament_matches, ratings_df, name_aliases):
    """Replaces 'TBD (Qualifier N)' bracket-YAML placeholders with the real qualifier's ratings-
    csv name once qualifying has concluded and ESPN's Round 1 draw shows who actually won that
    slot. Without this, match_draw_to_ratings has no lastname/initials to work with for these
    slots and silently manufactures a fake 'TBD (Qualifier N)' player at STARTING_ELO (bracket.py
    tier 3) - the real qualifier can then never be matched against ESPN's live results and gets
    silently dropped from the whole export (see the NOTE printed near the end of
    export_bracket_json).

    Resolution is driven entirely by Round 1 bracket adjacency - each placeholder's Round 1
    opponent is a real, already-resolvable player (see get_matchups: Round 1 pairs up consecutive
    non-bye draw slots) - plus the exact same match_espn_name_to_draw lookup used everywhere else
    in this pipeline, just run against the full ratings table instead of the (still-incomplete)
    draw. That makes this generalize to any future tournament with unresolved qualifier slots,
    rather than requiring the bracket YAML to be hand-edited once qualifying finishes.

    A resolved candidate who is *also* recorded as the loser of a separate completed match (e.g.
    they lost their own qualifying final to someone else already seen elsewhere in the draw) is
    left unresolved rather than trusted - that pattern means ESPN's Round 1 bracket cell itself is
    stale (seeded before the qualifying final concluded, not yet refreshed), not that the player
    is actually alive.

    Returns (players_with_placeholders_resolved, warnings) - warnings is a list of human-readable
    strings for every placeholder slot that could NOT be resolved, meant to be surfaced in the
    exported JSON (not just logged) so an alive-but-unresolved player is never silently dropped.
    """
    non_bye, _bye_items = split_byes(players, byes)
    round1_matches = [m for m in tournament_matches if m["round"] == "Round 1"]
    ratings_names = list(ratings_df["player"])

    resolved_by_id = {}
    warnings = []
    for a, b in get_matchups(non_bye):
        if QUALIFIER_PLACEHOLDER_RE.match(a.name):
            placeholder, known = a, b
        elif QUALIFIER_PLACEHOLDER_RE.match(b.name):
            placeholder, known = b, a
        else:
            continue

        def _opponent_is_known(m, known=known):
            return (
                match_espn_name_to_draw(m["player_1"], [known.name], name_aliases) == known.name
                or match_espn_name_to_draw(m["player_2"], [known.name], name_aliases) == known.name
            )

        match = next((m for m in round1_matches if _opponent_is_known(m)), None)
        if match is None:
            warnings.append(
                f"{placeholder.name}: no Round 1 ESPN match found for its known opponent "
                f"{known.name!r} - left unresolved"
            )
            continue

        qualifier_espn_name = (
            match["player_2"]
            if match_espn_name_to_draw(match["player_1"], [known.name], name_aliases) == known.name
            else match["player_1"]
        )

        stale_loss = next(
            (m for m in tournament_matches
             if m is not match and m["status_state"] == "post" and m["winner"]
             and qualifier_espn_name in (m["player_1"], m["player_2"]) and m["winner"] != qualifier_espn_name),
            None,
        )
        if stale_loss is not None:
            warnings.append(
                f"{placeholder.name}: ESPN Round 1 still lists {qualifier_espn_name!r} opposite "
                f"{known.name!r}, but {qualifier_espn_name!r} already lost a completed "
                f"{stale_loss['round']!r} match to {stale_loss['winner']!r} - ESPN's bracket cell "
                f"looks stale; left unresolved rather than including an eliminated player"
            )
            continue

        resolved_csv_name = match_espn_name_to_draw(qualifier_espn_name, ratings_names, name_aliases)
        if resolved_csv_name is None:
            warnings.append(
                f"{placeholder.name}: ESPN shows {qualifier_espn_name!r} as the real qualifier, "
                f"but that name couldn't be matched to any player in the Elo ratings data - left "
                f"unresolved"
            )
            continue

        resolved_by_id[id(placeholder)] = resolved_csv_name

    if not resolved_by_id:
        return players, warnings

    updated_players = [
        replace(p, name=resolved_by_id[id(p)]) if id(p) in resolved_by_id else p for p in players
    ]
    return updated_players, warnings


def export_bracket_json(bracket_path, output_path=None, n_simulations=N_SIMULATIONS, seed=SEED):
    bracket = load_bracket_yaml(bracket_path)
    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    validate_bracket_structure(byes)

    tour_config = TOUR_CONFIG[bracket.tour]
    matches_history = load_matches_for_tour(bracket.tour)
    ratings_df = calculate_elo_ratings(matches_history, bracket.start_date)
    ratings_df = ratings_df.sort_values("overall_elo", ascending=False).reset_index(drop=True)

    espn_data = fetch_scoreboard(bracket.tour.lower())
    espn_matches, _ = extract_matches(espn_data)
    category = TOUR_SINGLES_CATEGORY[bracket.tour.lower()]
    tournament_matches = [
        m for m in espn_matches if m["tournament"] == bracket.tournament and m["category"] == category
    ]
    if not tournament_matches:
        raise RuntimeError(f"No live matches found for {bracket.tournament!r} / {category}")

    # must run before match_draw_to_ratings - see resolve_qualifier_placeholders' docstring for
    # why a literal 'TBD (Qualifier N)' placeholder can never be matched to ESPN's real name later.
    players, qualifier_warnings = resolve_qualifier_placeholders(
        players, byes, tournament_matches, ratings_df, tour_config.name_aliases
    )

    draw, resolutions, ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date
    )
    unmatched = [r for r in resolutions if r["tier"] is None]
    if unmatched:
        raise RuntimeError(f"Unmatched bracket names: {[r['name'] for r in unmatched]}")
    # win_probability() reads Elo from this file, not from ratings_df in memory.
    tour_config.ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(tour_config.ratings_path, index=False)

    validate_draw(draw)
    non_bye_players, bye_players = split_byes(draw, byes)

    results_by_round, round_sequence, unresolved_1 = build_real_results_by_round(
        tournament_matches, draw, tour_config.name_aliases
    )
    known_pairings_by_round, _, unresolved_2 = build_known_pairings_by_round(
        tournament_matches, draw, tour_config.name_aliases
    )
    unresolved_names = unresolved_1 | unresolved_2
    # a name that isn't a confirmed loser of any completed match is presumably still alive - never
    # drop those silently just because they couldn't be matched (see the module docstring's spec:
    # players should include everyone still alive).
    alive_unresolved = sorted(
        name for name in unresolved_names
        if not any(
            m["status_state"] == "post" and m["winner"] and name in (m["player_1"], m["player_2"])
            and m["winner"] != name
            for m in tournament_matches
        )
    )
    warnings = qualifier_warnings + [
        f"{name}: ESPN name could not be matched to the draw and doesn't appear to have lost a "
        f"completed match - likely still alive but excluded from this export" for name in alive_unresolved
    ]
    if unresolved_names:
        print(f"NOTE: {len(unresolved_names)} ESPN name(s) unresolved to the draw, excluded "
              f"from output: {sorted(unresolved_names)}", file=sys.stderr)
    if warnings:
        print(f"WARNING: {len(warnings)} issue(s) affecting export completeness - see output "
              f"JSON's 'warnings' field", file=sys.stderr)

    fields = replay_real_rounds(non_bye_players, bye_players, results_by_round, known_pairings_by_round)
    max_known_round = len(fields) - 1

    # exact ESPN displayName <-> our internal draw name, for every player actually seen live
    espn_to_draw, draw_to_espn = {}, {}
    for m in tournament_matches:
        for raw_name in (m["player_1"], m["player_2"]):
            if not raw_name or raw_name == "TBD" or raw_name in espn_to_draw:
                continue
            resolved = match_espn_name_to_draw(raw_name, draw, tour_config.name_aliases)
            if resolved is not None:
                espn_to_draw[raw_name] = resolved
                draw_to_espn.setdefault(resolved, raw_name)

    leaves_by_slot = reconstruct_leaves_by_round2_slot(tournament_matches, non_bye_players, bye_players)
    _half_by_draw, quarter_by_draw = tag_halves_and_quarters(leaves_by_slot)
    round_label_map = build_round_label_map(round_sequence)

    # alive = every player we have a real ESPN name for, minus anyone already lost in a decided match
    losers = set()
    for round_results in results_by_round.values():
        for pair, winner in round_results.items():
            losers.add(next(p for p in pair if p != winner))
    alive_draw_names = [p for p in draw if p in draw_to_espn and p not in losers]

    # --- odds ---
    sport_key = discover_odds_sport_key(bracket.tour, bracket.tournament, ODDS_API_KEY) if ODDS_API_KEY else None
    odds_lookup = fetch_devigged_odds(sport_key, ODDS_API_KEY) if sport_key else {}
    print(f"Odds API sport key: {sport_key!r} - {len(odds_lookup)} priced match(es) available")

    # --- matchups: every unsettled match, any round, where both sides are already real names ---
    matchups = {}
    odds_used = model_used = 0
    for round_label in round_sequence:
        round_num = round_sequence.index(round_label) + 1
        round_matches = [m for m in tournament_matches if m["round"] == round_label]
        n = len(round_matches)
        daron_round = round_label_map[round_label]
        decided = results_by_round.get(round_num, {})

        for i, m in enumerate(round_matches):
            p1_raw, p2_raw = m["player_1"], m["player_2"]
            if not p1_raw or not p2_raw or p1_raw == "TBD" or p2_raw == "TBD":
                continue
            draw_a, draw_b = espn_to_draw.get(p1_raw), espn_to_draw.get(p2_raw)
            if draw_a is None or draw_b is None:
                continue
            if frozenset((draw_a, draw_b)) in decided:
                continue  # already decided - matchups is unsettled matches only

            if daron_round == "F":
                match_id = "Final"
            else:
                half_prefix = "T" if i < n / 2 else "B"
                half_index = i if i < n / 2 else i - n // 2
                match_id = f"{half_prefix}-{daron_round}-{half_index + 1}"

            prob_a, source = resolve_probability(
                p1_raw, p2_raw, draw_a, draw_b, odds_lookup, bracket.surface, tour_config.ratings_path
            )
            odds_used += source == "odds_api"
            model_used += source == "model"
            prob_a = round(prob_a, 3)
            matchups[match_id] = {
                "slot_a": p1_raw, "slot_b": p2_raw,
                "p_slot_a": prob_a, "p_slot_b": round(1 - prob_a, 3),
            }

    print(f"matchups: {len(matchups)} unsettled ({odds_used} priced via The Odds API, "
          f"{model_used} via model)")

    # --- players: p_champ / p_sf / p_final via simulation from the current real state ---
    target_round = max_known_round + 1
    partial_field = fields[max_known_round]
    partial_matchups = known_matchups_for_round(target_round, partial_field, known_pairings_by_round)
    partial_known_results = {}
    if partial_matchups is not None:
        round_results = results_by_round.get(target_round, {})
        partial_known_results = {
            frozenset(pair): round_results[frozenset(pair)]
            for pair in partial_matchups if frozenset(pair) in round_results
        }
    # ordered_field/is_bye must reflect TRUE bracket adjacency (see run_simulations_tracking_
    # milestones's docstring) - reusing leaves_by_slot, the exact same reconstruction the quarter
    # tags above come from, guarantees the simulated bracket tree and the displayed quarters can
    # never disagree. fields[]'s own "round winners then byes appended" concatenation is only
    # good enough for pinning known results (frozenset-keyed, order-independent) - not for this.
    true_order = true_bracket_order(leaves_by_slot)
    if target_round == 1:
        ordered_field = [name for name, _is_bye in true_order]
        is_bye = [is_bye_flag for _name, is_bye_flag in true_order]
    else:
        leaf_position = {name: i for i, (name, _is_bye) in enumerate(true_order)}
        ordered_field = sorted(partial_field, key=lambda draw_name: leaf_position.get(draw_name, len(true_order)))
        is_bye = [False] * len(ordered_field)

    random.seed(seed)
    champ_counts, sf_counts, final_counts = run_simulations_tracking_milestones(
        ordered_field, is_bye, partial_known_results, bracket.surface, n_simulations, tour_config.ratings_path
    )

    players_out = []
    for draw_name in alive_draw_names:
        espn_name = draw_to_espn[draw_name]
        quarter = quarter_by_draw.get(draw_name)
        if quarter is None:
            continue  # not yet placeable in a quarter - see tag_halves_and_quarters
        players_out.append({
            "player": espn_name,
            "quarter": quarter,
            "p_champ": round(champ_counts.get(draw_name, 0) / n_simulations, 3),
            "p_sf": round(sf_counts.get(draw_name, 0) / n_simulations, 3),
            "p_final": round(final_counts.get(draw_name, 0) / n_simulations, 3),
        })
    players_out.sort(key=lambda r: -r["p_champ"])

    # --- head_to_head: every alive pair not already an unsettled "matchups" entry ---
    matchup_pairs = {frozenset((m["slot_a"], m["slot_b"])) for m in matchups.values()}
    alive_espn = sorted(
        draw_to_espn[d] for d in alive_draw_names if quarter_by_draw.get(d)
    )
    head_to_head = {}
    for idx, name_a in enumerate(alive_espn):
        for name_b in alive_espn[idx + 1:]:
            if frozenset((name_a, name_b)) in matchup_pairs:
                continue
            draw_a, draw_b = espn_to_draw[name_a], espn_to_draw[name_b]
            prob_a, _ = resolve_probability(
                name_a, name_b, draw_a, draw_b, odds_lookup, bracket.surface, tour_config.ratings_path
            )
            head_to_head[f"{name_a}|{name_b}"] = round(prob_a, 3)

    tour_word = "men" if bracket.tour == "ATP" else "women"
    tournament_slug = bracket.tournament.lower().replace(" ", "-")
    output = {
        "meta": {
            "tournament": f"{tournament_slug}-{tour_word}-{bracket.year}",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "iterations": n_simulations,
            "seed": seed,
        },
        "players": players_out,
        "matchups": matchups,
        "head_to_head": head_to_head,
        "warnings": warnings,
    }

    if output_path is None:
        output_path = OUTPUT_DIR / f"{Path(bracket_path).stem}_bracket_export.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output_path, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bracket_path", type=Path)
    parser.add_argument("--simulations", type=int, default=N_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        output_path, output = export_bracket_json(args.bracket_path, args.output, args.simulations, args.seed)
    except (BracketValidationError, LiveScoresError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nWrote {output_path}")
    print(f"players: {len(output['players'])} alive, matchups: {len(output['matchups'])}, "
          f"head_to_head: {len(output['head_to_head'])}, warnings: {len(output['warnings'])}")
