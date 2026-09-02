"""Hybrid simulation: treats real match results (from live_scores.py / ESPN) as known up through
a given round, then Monte Carlo-simulates everything after that - even rounds that have already
actually happened in reality, if they're past the cutoff.

Usage:
    python model/hybrid_simulation.py brackets/wta_toronto_2026.yaml --through-round 3
    python model/hybrid_simulation.py brackets/wta_toronto_2026.yaml --all-rounds
"""
import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket import (  # noqa: E402
    TOUR_CONFIG, get_matchups, match_draw_to_ratings, order_by_draw_position,
    split_byes, validate_bracket_structure, validate_draw,
)
from bracket_schema import BracketValidationError, load_bracket_yaml  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from live_scores import LiveScoresError, extract_matches, fetch_scoreboard  # noqa: E402
from simulate import N_SIMULATIONS, run_simulations_from_field, run_simulations_partial_round  # noqa: E402
from win_probability import win_probability  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

STANDARD_TAIL_ROUNDS = ["Quarterfinal", "Semifinal", "Final"]
NUMBERED_ROUND_RE = re.compile(r"^Round (\d+)$")
TOUR_SINGLES_CATEGORY = {"atp": "Men's Singles", "wta": "Women's Singles"}


def build_round_sequence(round_labels):
    """Orders ESPN round labels into the main-draw sequence: numeric 'Round N' labels first
    (ascending), then Quarterfinal/Semifinal/Final. Anything else (e.g. 'Qualifying Final') is
    pre-draw and excluded - those players already appear as regular entries in the main draw."""
    numbered = sorted(
        (label for label in round_labels if NUMBERED_ROUND_RE.match(label)),
        key=lambda label: int(NUMBERED_ROUND_RE.match(label).group(1)),
    )
    tail = [label for label in STANDARD_TAIL_ROUNDS if label in round_labels]
    return numbered + tail


def _lastname_and_suffix(csv_name):
    """Simpler split than bracket._split_csv_name: last whitespace token is always treated as
    the truncated-firstname suffix (whatever convention it uses - single dotted initial, e.g.
    'E.', multi-letter compound initials, e.g. 'E.G.', or a literal N-char prefix used to
    disambiguate a collision, e.g. 'Xiy.'), everything before it is the lastname. Good enough
    for approximate matching against messy real-world ESPN name strings; bracket.py's stricter
    splitter cares about exact tier-1/tier-2 lookup semantics, which isn't the goal here."""
    tokens = csv_name.split()
    if len(tokens) < 2:
        return csv_name.lower(), ""
    lastname = " ".join(tokens[:-1]).lower()
    suffix = re.sub(r"[.\-]", "", tokens[-1]).lower()
    return lastname, suffix


def _firstname_matches_suffix(firstname_part, suffix):
    """The suffix might be a literal prefix of a single-word firstname (e.g. 'xiy' matching
    'Xiyu' - a disambiguating truncation) or the per-word initials of a compound/hyphenated
    firstname (e.g. 'eg' matching 'Elena-Gabriela')."""
    if not suffix:
        return False
    firstname_part = firstname_part.lower()
    if firstname_part.startswith(suffix):
        return True
    initials = "".join(w[0] for w in re.split(r"[\s\-]+", firstname_part) if w)
    return initials == suffix or initials.startswith(suffix)


def match_espn_name_to_draw(espn_name, draw_csv_names, name_aliases=None):
    """Matches an ESPN full display name (e.g. 'Iga Swiatek') to one of the bracket's already-
    resolved ratings-csv-format names (e.g. 'Swiatek I.'). Tries the draw name's lastname
    against both the trailing AND leading word(s) of the ESPN name - most names are Western
    order (Firstname Lastname), but ESPN represents some players (seen here: Chinese players
    Wang Xiyu/Wang Xinyu/Zhang Shuai) in native order (Lastname Firstname) instead, and there's
    no reliable per-player signal for which. Checked BEFORE name-similarity matching (not merged
    into the same candidate set): the manual alias table exists precisely for players name-
    similarity gets wrong or can't reach at all - e.g. 'Osorio M.' for Camila Osorio (no shared
    initial at all), or 'SoonWoo Kwon' (ESPN glues a two-part Korean given name into one unspaced
    word, so the compound-initials check below can never recover 'S.W.' from it - it silently
    but confidently matches the wrong, sparse 'Kwon S.' csv entry instead of the real
    'Kwon S.W.' one, which has this player's actual match history). Checking the alias first and
    returning immediately avoids that: blending it into the same set would just make an
    already-wrong unique match ambiguous (two candidates) rather than actually correct.
    Returns None if zero or multiple name-similarity candidates match, rather than guess."""
    espn_words = espn_name.split()

    if name_aliases and len(espn_words) >= 2:
        alias_key = f"{espn_words[-1].title()} {espn_words[0][0].upper()}."
        alias_target = name_aliases.get(alias_key)
        if alias_target in draw_csv_names:
            return alias_target

    candidates = set()
    for csv_name in draw_csv_names:
        lastname, suffix = _lastname_and_suffix(csv_name)
        if not suffix:
            continue
        lastname_words = lastname.split()
        n = len(lastname_words)
        if n == 0 or len(espn_words) < n + 1:
            continue
        for lastname_at_tail in (True, False):
            candidate_lastname = [w.lower() for w in (espn_words[-n:] if lastname_at_tail else espn_words[:n])]
            if candidate_lastname != lastname_words:
                continue
            # " ".join, not "": _firstname_matches_suffix re-splits this on whitespace/hyphen to
            # recover compound initials (e.g. "Carol Young Suh" -> "cys") - "".join collapsed a
            # genuinely space-separated multi-word given name into one unsplittable token before
            # that regex ever ran, so a 3+-word given name could never match its own suffix
            # (confirmed: 'Carol Young Suh Lee' failed to resolve to draw name 'Lee C.Y.', silently
            # dropping her already-decided Round 1 result). A hyphenated compound (e.g.
            # "Elena-Gabriela") is unaffected either way - the hyphen is preserved inside a single
            # espn_words token, not lost at a join boundary.
            firstname_part = " ".join(espn_words[:-n] if lastname_at_tail else espn_words[n:])
            if _firstname_matches_suffix(firstname_part, suffix):
                candidates.add(csv_name)

    return next(iter(candidates)) if len(candidates) == 1 else None


def _round_index(espn_matches):
    """Shared by build_real_results_by_round and build_known_pairings_by_round so both agree on
    the same round numbering - both derive round_sequence from the identical input list (not
    filtered by match status first), so calling this once per (tournament, category) match list
    and reusing the result, or just calling both functions with that same list, keeps them
    consistent automatically."""
    round_labels = {m["round"] for m in espn_matches if m["round"]}
    round_sequence = build_round_sequence(round_labels)
    return round_sequence, {label: i + 1 for i, label in enumerate(round_sequence)}


def build_real_results_by_round(espn_matches, draw_csv_names, name_aliases=None):
    """Returns (results_by_round, round_sequence, unresolved_names). results_by_round[n] maps
    frozenset({player_a, player_b}) -> winner, for round n. Only completed matches with a clear
    winner and both players resolvable to the draw contribute a result - a match that hasn't
    been played yet, even with two real (non-'TBD') names, contributes nothing here; see
    build_known_pairings_by_round for that weaker, separate signal."""
    round_sequence, round_index = _round_index(espn_matches)

    results_by_round = defaultdict(dict)
    unresolved_names = set()
    for m in espn_matches:
        round_num = round_index.get(m["round"])
        if round_num is None:
            continue
        if m["status_state"] != "post" or not m["winner"]:
            continue

        p1 = match_espn_name_to_draw(m["player_1"], draw_csv_names, name_aliases)
        p2 = match_espn_name_to_draw(m["player_2"], draw_csv_names, name_aliases)
        if p1 is None:
            unresolved_names.add(m["player_1"])
        if p2 is None:
            unresolved_names.add(m["player_2"])
        if p1 is None or p2 is None:
            continue

        winner_csv = p1 if m["winner"] == m["player_1"] else p2
        results_by_round[round_num][frozenset((p1, p2))] = winner_csv

    return results_by_round, round_sequence, unresolved_names


def build_known_pairings_by_round(espn_matches, draw_csv_names, name_aliases=None):
    """Returns (known_pairings_by_round, round_sequence, unresolved_names). known_pairings_by_round[n]
    is a set of frozenset({player_a, player_b}) for round n - every match where BOTH players are
    already resolvable to the draw (i.e. neither side is still 'TBD'), regardless of whether the
    match has actually been played yet. A genuinely separate, weaker signal than
    build_real_results_by_round's "decided" results: ESPN returns every round's competitions
    upfront, using 'TBD' only for a slot still waiting on an earlier round's real outcome (see
    espn_bracket.py's own docstring) - so a round's matchup structure is often fully knowable well
    before any of its matches conclude."""
    round_sequence, round_index = _round_index(espn_matches)

    known_pairings_by_round = defaultdict(set)
    unresolved_names = set()
    for m in espn_matches:
        round_num = round_index.get(m["round"])
        if round_num is None:
            continue
        p1_name, p2_name = m["player_1"], m["player_2"]
        if not p1_name or not p2_name or p1_name == "TBD" or p2_name == "TBD":
            continue

        p1 = match_espn_name_to_draw(p1_name, draw_csv_names, name_aliases)
        p2 = match_espn_name_to_draw(p2_name, draw_csv_names, name_aliases)
        if p1 is None:
            unresolved_names.add(p1_name)
        if p2 is None:
            unresolved_names.add(p2_name)
        if p1 is None or p2 is None:
            continue

        known_pairings_by_round[round_num].add(frozenset((p1, p2)))

    return known_pairings_by_round, round_sequence, unresolved_names


def known_matchups_for_round(round_num, current_field, known_pairings_by_round):
    """Returns round_num's full matchup list if it's completely determined, else None.

    Round 1's matchups are always determined - get_matchups(current_field) reconstructs the real
    bracket pairing directly from the draw itself, independent of anything ESPN reports (a bye's
    phantom opponent never sits between two real non-bye players). Round 2+ has no such
    independent source: the "winners + byes" concatenation that forms a simulated field does NOT
    reconstruct true bracket-tree adjacency, so round 2+'s pairing can only come from ESPN's own
    known_pairings_by_round - and only counts as determined if those known pairings account for
    every player in current_field exactly once (anything less means some slot is still 'TBD',
    waiting on a match this function doesn't have visibility into)."""
    if round_num == 1:
        return get_matchups(current_field)

    known_pairs = known_pairings_by_round.get(round_num, set())
    players_in_known_pairs = [p for pair in known_pairs for p in pair]
    if set(players_in_known_pairs) != set(current_field) or len(players_in_known_pairs) != len(current_field):
        return None
    return [tuple(pair) for pair in known_pairs]


def upcoming_match_info(round_num, field_before_round, known_pairings_by_round, surface, ratings_path):
    """{player: (opponent, model-predicted P(that player wins their round_num match))}, for every
    player whose round_num pairing is already known (see known_matchups_for_round) - {} if it
    isn't. Reports each side of the matchup from win_probability() directly, so it means the same
    thing whether the match has already been decided (a real, already-known win/loss - e.g. a
    29.8% underdog who actually won still shows 29.8%, not retroactively adjusted toward 100%) or
    hasn't been played yet (a genuine forward-looking prediction for an already-known upcoming
    pairing) - this is always the pre-match prediction, never a retrospective one. Deliberately
    scoped to exactly ONE real match, unlike tournament_win_probability (which folds in every
    later round too) - the two together let a viewer separate "this player's own upcoming/just-
    played match odds" from "the aggregate number moved because the rest of the draw's difficulty
    became more/less known"."""
    pairing = known_matchups_for_round(round_num, field_before_round, known_pairings_by_round)
    if pairing is None:
        return {}
    info = {}
    for player_a, player_b in pairing:
        p_a = win_probability(player_a, player_b, surface, ratings_path)
        info[player_a] = (player_b, p_a)
        info[player_b] = (player_a, 1 - p_a)
    return info


def reconstruct_leaves_by_round2_slot(tournament_matches, non_bye_players, bye_players, results_by_round,
                                       name_aliases=None):
    """The single structural reconstruction the real bracket tree - the one known_matchups_for_
    round's own docstring says a plain 'winners + byes' concatenation can't reconstruct - is built
    from: for each of Round 2's draw_size/2 slots (i = 0..n2-1, ESPN's own stable list order), the
    ordered list of (draw_name, is_bye) leaves that feed it - either one bye or the two Round 1
    match participants that will produce its winner.

    Identifies which specific Round 1 match feeds a Round 2 slot by WHO ACTUALLY WON it
    (results_by_round[1], resolved the same way build_real_results_by_round already resolves
    everything else), not by position. An earlier version assumed Round 2's list order tracked
    non_bye_players' own Round 1 pairing order (i.e. "the next unconsumed Round 1 pair, in
    sequence") - confirmed FALSE by direct empirical trace against a real live draw: Round 1 match
    index 5's winner (Kecmanovic, beat Ugo Carabelli) fed Round 2 slot index 1, not index 5 - ESPN
    orders Round 2 by its own seeding-sheet layout, not by Round 1's left-to-right match number,
    so the two lists are NOT index-aligned the way the sequential-pointer approach required. That
    silently fed the wrong Round 1 pair into most slots after the first couple, which is how
    Djokovic (a bye) ended up with a simulated Round 2 opponent of Mensik J. (another bye) instead
    of his real opponent Tirante T.A.

    Falls back to the old positional-pointer behavior only for a slot whose Round 1 match hasn't
    been decided yet (still 'TBD', or a name that resolves to nothing in results_by_round[1] - e.g.
    ESPN listing a genuine walkover/substitute under a name that doesn't match any recorded Round 1
    winner) - undecided matches have no "who actually won" signal to key off yet, so the sequential
    guess is the best available placeholder until a real result exists, same limitation the
    original implementation always had for this case."""
    round1_names = set()
    for m in tournament_matches:
        if m["round"] != "Round 1":
            continue
        for name in (m["player_1"], m["player_2"]):
            if name and name != "TBD":
                round1_names.add(name)

    draw_csv_names = non_bye_players + bye_players
    # winner -> (loser, winner), keyed by the RESOLVED draw-csv name results_by_round already uses
    # - not the raw ESPN string - so it can be looked up directly against a Round 2 name resolved
    # the same way.
    winner_to_pair = {}
    for pair, winner in results_by_round.get(1, {}).items():
        loser = next(p for p in pair if p != winner)
        winner_to_pair[winner] = (loser, winner)

    round2 = [m for m in tournament_matches if m["round"] == "Round 2"]

    leaves_by_slot = []
    r1_pointer = 0  # positional fallback only - see docstring
    bye_pointer = 0  # next unconsumed bye_players entry
    for m in round2:
        slot_leaves = []
        for name in (m["player_1"], m["player_2"]):
            is_bye_slot = bool(name) and name != "TBD" and name not in round1_names
            if is_bye_slot:
                slot_leaves.append((bye_players[bye_pointer], True))
                bye_pointer += 1
                continue

            resolved = (
                match_espn_name_to_draw(name, draw_csv_names, name_aliases)
                if name and name != "TBD" else None
            )
            pair = winner_to_pair.get(resolved) if resolved is not None else None
            if pair is not None:
                loser, winner = pair
                slot_leaves.append((loser, False))
                slot_leaves.append((winner, False))
            else:
                slot_leaves.append((non_bye_players[2 * r1_pointer], False))
                slot_leaves.append((non_bye_players[2 * r1_pointer + 1], False))
            r1_pointer += 1
        leaves_by_slot.append(slot_leaves)
    return leaves_by_slot


def true_bracket_order(leaves_by_slot):
    """Flattens reconstruct_leaves_by_round2_slot's per-slot leaves into the single ordered
    (draw_name, is_bye) list real draw adjacency requires: consecutive non-bye entries are always
    an actual Round 1 match, and a bye always sits at its true position relative to the Round 1
    matches around it - not grouped separately the way a plain 'round winners + byes'
    concatenation would (see reconstruct_leaves_by_round2_slot's docstring)."""
    return [leaf for slot_leaves in leaves_by_slot for leaf in slot_leaves]


def replay_real_rounds(non_bye_players, bye_players, results_by_round, known_pairings_by_round=None):
    """Replays as many rounds as have a COMPLETE set of real results, starting at round 1 and
    stopping at the first round whose matchup structure isn't fully known (known_matchups_for_round
    returns None) or that has any unresolved matchup within a known structure. Returns a list
    `fields` where fields[n] is the real field entering round n+1 (fields[0] = pre-Round-1
    non-bye field); len(fields) - 1 is the highest round fully known (matchups AND results) from
    real results.

    known_pairings_by_round is optional (round 1 never needs it - see known_matchups_for_round);
    omitting it just means round 2+ can never be treated as known here, matching this function's
    original round-1-only behavior."""
    known_pairings_by_round = known_pairings_by_round or {}
    fields = [list(non_bye_players)]
    current_field = list(non_bye_players)
    round_num = 1
    while len(current_field) > 1:
        matchups = known_matchups_for_round(round_num, current_field, known_pairings_by_round)
        if matchups is None:
            return fields

        round_results = results_by_round.get(round_num, {})
        winners = []
        for pair in matchups:
            winner = round_results.get(frozenset(pair))
            if winner is None:
                return fields
            winners.append(winner)
        current_field = winners + list(bye_players) if round_num == 1 else winners
        fields.append(current_field)
        round_num += 1
    return fields


def _report(label, rows, output_path, top_n=10):
    """rows: compute_round_snapshot's own per-player dicts (player, win_count,
    tournament_win_probability, upcoming_opponent, upcoming_match_win_probability), already
    sorted - written to CSV as-is and the top_n printed to the console."""
    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)

    print(f"\n=== {label} === (saved to {output_path})")
    print(results.head(top_n).to_string(index=False))


def load_hybrid_state(bracket_path, dates=None):
    """Loads and prepares everything compute_round_snapshot needs to simulate any known (or the
    next partial) round of `bracket_path` - the setup previously duplicated inline at the top of
    main(), now split out so a caller that wants EVERY round (build_round_history) can do one load
    instead of reloading ratings/live-scores data per round.

    true_order/leaf_position come directly from `draw` - already true bracket-position order (see
    bracket_export.tag_halves_and_quarters' docstring for why) - NOT from
    reconstruct_leaves_by_round2_slot's ESPN-Round-2 reconstruction. That reconstruction has its
    own latent bug (confirmed: 14 missing + 14 duplicated leaves out of 128 on a real live draw
    with mixed decided/undecided Round 1 matches - see that function's own docstring) which every
    caller of this file's --all-rounds output was silently exposed to until now. This is the exact
    fix bracket_export.py's matchups loop already applies; porting it here fixes it at the source
    for every caller of compute_round_snapshot (the CLI and build_round_history alike), not just
    bracket_export.py's own separate copy of the same fix."""
    bracket = load_bracket_yaml(bracket_path)
    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    validate_bracket_structure(byes)

    tour_config = TOUR_CONFIG[bracket.tour]
    matches_history = load_matches_for_tour(bracket.tour)
    ratings_df = calculate_elo_ratings(matches_history, bracket.start_date)
    ratings_df = ratings_df.sort_values("overall_elo", ascending=False).reset_index(drop=True)

    draw, resolutions, ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date
    )
    unmatched = [r for r in resolutions if r["tier"] is None]
    if unmatched:
        raise RuntimeError(
            f"Unmatched bracket names, fix before running a hybrid simulation: "
            f"{[r['name'] for r in unmatched]}"
        )
    tour_config.ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(tour_config.ratings_path, index=False)
    validate_draw(draw)
    non_bye_players, bye_players = split_byes(draw, byes)

    espn_data = fetch_scoreboard(bracket.tour.lower(), dates=dates)
    espn_matches, _stats = extract_matches(espn_data)
    category = TOUR_SINGLES_CATEGORY[bracket.tour.lower()]
    tournament_matches = [
        m for m in espn_matches if m["tournament"] == bracket.tournament and m["category"] == category
    ]
    if not tournament_matches:
        raise RuntimeError(f"No live matches found for tournament {bracket.tournament!r} / {category}")

    results_by_round, round_sequence, unresolved_names = build_real_results_by_round(
        tournament_matches, draw, tour_config.name_aliases
    )
    known_pairings_by_round, _round_sequence_2, unresolved_names_2 = build_known_pairings_by_round(
        tournament_matches, draw, tour_config.name_aliases
    )
    unresolved_names = unresolved_names | unresolved_names_2

    # ESPN displayName <-> internal draw (ratings-csv-style) name - same mapping bracket_export.py
    # builds for its own output, needed here too so a consumer of build_round_history's output
    # sees the same player-naming convention as bracket_export.py's futures odds, instead of two
    # different name formats for the same person across one consolidated file (confirmed as a real
    # bug when an early draft of the consolidated-export work reused a name-mismatched baseline).
    espn_to_draw, draw_to_espn = {}, {}
    for m in tournament_matches:
        for raw_name in (m["player_1"], m["player_2"]):
            if not raw_name or raw_name == "TBD" or raw_name in espn_to_draw:
                continue
            resolved = match_espn_name_to_draw(raw_name, draw, tour_config.name_aliases)
            if resolved is not None:
                espn_to_draw[raw_name] = resolved
                draw_to_espn.setdefault(resolved, raw_name)

    fields = replay_real_rounds(non_bye_players, bye_players, results_by_round, known_pairings_by_round)
    max_known_round = len(fields) - 1

    true_order = list(zip(draw, byes))
    leaf_position = {name: i for i, name in enumerate(draw)}

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

    return SimpleNamespace(
        bracket=bracket, tour_config=tour_config, draw=draw, bye_players=bye_players,
        fields=fields, max_known_round=max_known_round, round_sequence=round_sequence,
        known_pairings_by_round=known_pairings_by_round, results_by_round=results_by_round,
        true_order=true_order, leaf_position=leaf_position, target_round=target_round,
        partial_field=partial_field, partial_matchups=partial_matchups,
        partial_known_results=partial_known_results, unresolved_names=unresolved_names,
        draw_to_espn=draw_to_espn, tournament_matches=tournament_matches,
    )


def _round_matches(state, n):
    """Round n's own real matchup pairing (via known_matchups_for_round, the same mechanism
    everything else in this module uses - not a positional reconstruction), each with the model's
    pregame win probability for player_a and the real result if round n is already decided. This
    is the MODEL-only half of a match-level model-vs-market comparison for round n specifically -
    a caller that also has market data (bracket_export.py's live Odds API fetch, for a still-
    unsettled match; or a persisted pregame-price cache, for an already-decided one, since The
    Odds API drops an event once it's Final) attaches that separately - see
    consolidated_export.build_match_checkpoints.

    state.fields[n - 1] is round n's real starting field for every n this is actually called with
    (both a fully-known round and the one partial round - state.partial_field IS
    state.fields[state.max_known_round] = state.fields[state.target_round - 1], so this same
    lookup is correct for n == state.target_round too, no special-casing needed)."""
    pairing = known_matchups_for_round(n, state.fields[n - 1], state.known_pairings_by_round)
    if pairing is None:
        return []
    decided = state.results_by_round.get(n, {})
    matches = []
    for player_a, player_b in pairing:
        model_prob_a = win_probability(player_a, player_b, state.bracket.surface, state.tour_config.ratings_path)
        winner_draw = decided.get(frozenset((player_a, player_b)))
        matches.append({
            "player_a": state.draw_to_espn.get(player_a, player_a),
            "player_b": state.draw_to_espn.get(player_b, player_b),
            "model_prob_a": round(model_prob_a, 3),
            "decided": winner_draw is not None,
            "winner": state.draw_to_espn.get(winner_draw, winner_draw) if winner_draw is not None else None,
        })
    return matches


def _true_order_sorted(leaf_position, field):
    return sorted(field, key=lambda name: leaf_position.get(name, len(leaf_position)))


def compute_round_snapshot(state, n, n_simulations, seed, use_upset_boost=True):
    """Runs round n's snapshot (or, if n == state.target_round and it's a partial/about-to-play
    round, that partial round) against a state already loaded by load_hybrid_state. Returns
    (upcoming_round_num, field_size, label, rows) - rows is a list of {"player", "win_count",
    "tournament_win_probability", "upcoming_opponent", "upcoming_match_win_probability"} dicts for
    EVERY player in state.draw (win_count 0 for anyone not alive at this checkpoint), sorted by
    tournament_win_probability descending - the same content main()'s CLI writes to a per-round
    CSV, just returned in memory so a caller (build_round_history) isn't forced to round-trip
    through disk for every round.

    Raises ValueError if n is out of range (0..state.max_known_round, or state.target_round when
    a partial round is available) - the caller decides how to report that (CLI exits, a library
    caller can let it propagate)."""
    partial = (
        n == state.target_round and state.partial_matchups is not None and state.partial_known_results
    )
    if not partial and not (0 <= n <= state.max_known_round):
        raise ValueError(f"round {n} is out of range (0..{state.max_known_round} known)")

    # reseeded per-round (not once for the whole run) so a standalone single-round call always
    # reproduces exactly the same snapshot as round N within a build_round_history/--all-rounds
    # run - neither draws from wherever the global random stream happened to be left by whatever
    # ran before it.
    random.seed(seed + n)
    tour_config = state.tour_config
    bracket = state.bracket
    leaf_position = state.leaf_position

    if partial:
        # byes only join in right after round 1 - for any later partial round they're already
        # folded into partial_field (state.fields[max_known_round]).
        extra_after = state.bye_players if n == 1 else []
        ordered_partial_field = _true_order_sorted(leaf_position, state.partial_field)
        champion_counts = run_simulations_partial_round(
            ordered_partial_field, extra_after, state.partial_known_results, bracket.surface,
            n_simulations, tour_config.ratings_path, use_upset_boost=use_upset_boost,
            matchups=state.partial_matchups,
        )
        upcoming = upcoming_match_info(
            n, state.partial_field, state.known_pairings_by_round, bracket.surface, tour_config.ratings_path)
        label = (
            f"Round {n} - {len(state.partial_field)} players about to play "
            f"({len(state.partial_known_results)}/{len(state.partial_matchups)} already decided, "
            f"rest simulated)"
        )
        upcoming_round_num, field_size = n, len(state.partial_field)
    else:
        starting_field = _true_order_sorted(leaf_position, state.fields[n])
        confirmed_real_pairing_for = {}
        for r in range(n + 1, state.max_known_round + 1):
            round_pairing = known_matchups_for_round(r, state.fields[r - 1], state.known_pairings_by_round)
            if round_pairing is not None:
                confirmed_real_pairing_for[frozenset(state.fields[r - 1])] = round_pairing

        def matchups_resolver(players, _real_pairing_for=confirmed_real_pairing_for):
            return _real_pairing_for.get(frozenset(players))

        champion_counts = run_simulations_from_field(
            starting_field, bracket.surface, n_simulations, tour_config.ratings_path,
            use_upset_boost=use_upset_boost, matchups_resolver=matchups_resolver,
        )
        upcoming_round = n + 1
        upcoming = upcoming_match_info(
            upcoming_round, state.fields[n], state.known_pairings_by_round, bracket.surface, tour_config.ratings_path)
        label = f"Round {upcoming_round} - {len(starting_field)} players about to play"
        upcoming_round_num, field_size = upcoming_round, len(starting_field)

    rows = []
    for player in state.draw:
        opp, prob = upcoming.get(player, (None, None))
        rows.append({
            "player": player,
            "win_count": champion_counts.get(player, 0),
            "tournament_win_probability": champion_counts.get(player, 0) / n_simulations,
            "upcoming_opponent": opp,
            "upcoming_match_win_probability": prob,
        })
    rows.sort(key=lambda r: -r["tournament_win_probability"])
    return upcoming_round_num, field_size, label, rows


def build_round_history(bracket_path, n_simulations=2000, seed=42, dates=None, use_upset_boost=True,
                         previous_history=None):
    """Every fully-known round (1..max_known_round) plus the next partial/about-to-play round (if
    its matchups and at least one result are already known) - the same per-round champion-
    probability computation --all-rounds already writes to CSV, just collected here as one ordered
    list in memory for a consolidated-export-style caller. Players are keyed by ESPN displayName
    (via state.draw_to_espn), matching bracket_export.py's own player-naming convention, and any
    player with a 0.0 tournament_win_probability at a given checkpoint is dropped from that
    checkpoint's list (already eliminated by then - keeping 100+ zero rows per round would bloat
    the output for no signal). Each entry also carries "matches" - round n's own real matchup
    pairing with the model's pregame probability and real result if decided (see _round_matches) -
    the model-only half of a match-level model-vs-market comparison scoped to that specific round;
    a caller with market data (consolidated_export.py) attaches that separately per round.

    n_simulations defaults lower than a futures export's own (2000 vs 10000) - this runs one full
    Monte Carlo simulation per historical round (up to 6-7 of them deep into a Slam), and
    round-by-round HISTORY doesn't need the same precision a single current snapshot does to still
    show the right shape of how a player's odds moved round to round.

    previous_history, if given, is a prior call's own return value (or the "round_history" field
    of a consolidated export built from one) - any of its entries for a round that is (a) not
    partial and (b) still <= the currently known round is reused as-is instead of rerun, since a
    round's own result never changes once it's fully decided (fields[n] is locked in by
    replay_real_rounds at that point - rerunning it would only spend simulation budget to
    reproduce the same distribution, not learn anything new). Only a round that just became fully
    known for the FIRST time this call, or the still-in-progress partial round (which by
    definition changes every time a match completes within it, so it is NEVER cached), triggers an
    actual new simulation - this is what lets a caller like live_match_watcher.py refresh the
    round history on every match completion without resimulating the whole tournament's history
    each time."""
    state = load_hybrid_state(bracket_path, dates=dates)
    round_numbers = list(range(1, state.max_known_round + 1))
    is_partial_available = state.partial_matchups is not None and bool(state.partial_known_results)
    if is_partial_available:
        round_numbers.append(state.target_round)

    # cached_by_n: n -> that round's own previously-computed, non-partial history entry, keyed by
    # the INPUT round n (not the output "round" field) - a non-partial compute_round_snapshot(state,
    # n, ...) always labels its output "round" as n+1 (see its own docstring), so entry["round"] - 1
    # recovers n. Filtering to non-partial entries first means a round that was still the partial
    # "about to play" checkpoint last time this was called is correctly excluded here and gets a
    # real (non-partial) computation below now that it's finished, instead of reusing its stale
    # partial snapshot.
    cached_by_n = {
        entry["round"] - 1: entry
        for entry in (previous_history or [])
        if not entry["partial"] and entry["round"] - 1 <= state.max_known_round
    }

    history = []
    for n in round_numbers:
        if n in cached_by_n:
            history.append(cached_by_n[n])
            continue
        upcoming_round_num, field_size, label, rows = compute_round_snapshot(
            state, n, n_simulations, seed, use_upset_boost=use_upset_boost)
        history.append({
            "round": upcoming_round_num,
            "players_about_to_play": field_size,
            "label": label,
            "partial": n == state.target_round and is_partial_available,
            "players": [
                {
                    "player": state.draw_to_espn.get(r["player"], r["player"]),
                    "tournament_win_probability": round(r["tournament_win_probability"], 4),
                    "upcoming_opponent": (
                        state.draw_to_espn.get(r["upcoming_opponent"], r["upcoming_opponent"])
                        if r["upcoming_opponent"] is not None else None
                    ),
                    "upcoming_match_win_probability": (
                        round(r["upcoming_match_win_probability"], 4)
                        if r["upcoming_match_win_probability"] is not None else None
                    ),
                }
                for r in rows if r["tournament_win_probability"] > 0
            ],
            "matches": _round_matches(state, n),
        })
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bracket_path", type=Path)
    parser.add_argument("--through-round", type=int, default=None,
                         help="rounds 1..N are treated as known (real); everything after is "
                              "simulated normally regardless of what actually happened")
    parser.add_argument("--all-rounds", action="store_true",
                         help="generate one snapshot per fully-known real round (1..latest), "
                              "each to its own labeled output file")
    parser.add_argument("--simulations", type=int, default=N_SIMULATIONS)
    parser.add_argument("--top-n", type=int, default=10,
                         help="how many players to print to the console per checkpoint (the CSV "
                              "always has every player - this only limits the printed preview)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible Monte Carlo results")
    parser.add_argument("--use-upset-boost", action=argparse.BooleanOptionalAction, default=True,
                         help="apply the fitted in-tournament momentum boost (see "
                              "UPSET_BOOST_LOGIT_SHIFT in win_probability.py) to a player's next "
                              "match whenever their most recent win THIS tournament beat someone "
                              ">100 Elo points higher than themselves. On by default, same as "
                              "the rank-gap/confidence-calibration adjustments in win_probability.py; "
                              "pass --no-use-upset-boost to disable for testing/comparison.")
    parser.add_argument("--dates", default=None,
                         help="YYYYMMDD, passed to ESPN's ?dates= - needed for an already-"
                              "concluded event, which the undated (\"today\") scoreboard can't find")
    args = parser.parse_args()

    if not args.all_rounds and args.through_round is None:
        parser.error("pass --through-round N or --all-rounds")

    try:
        state = load_hybrid_state(args.bracket_path, dates=args.dates)
    except (BracketValidationError, LiveScoresError, RuntimeError) as e:
        print(f"{args.bracket_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Round sequence observed: {state.round_sequence}")
    print(f"Real results fully known through round {state.max_known_round} of the main draw "
          f"(then simulated normally from there)")
    if state.unresolved_names:
        print(f"WARNING: {len(state.unresolved_names)} ESPN player name(s) could not be matched to "
              f"the draw (their matches are excluded from real results): {sorted(state.unresolved_names)}",
              file=sys.stderr)

    is_partial_available = state.partial_matchups is not None and bool(state.partial_known_results)
    if is_partial_available:
        print(f"Round {state.target_round} matchups are fully known ({len(state.partial_field)} players) - "
              f"{len(state.partial_known_results)}/{len(state.partial_matchups)} already final - "
              f"--through-round {state.target_round} will pin those and simulate the rest.")
    elif state.max_known_round == 0:
        print("No fully-known real round available, and the next round's matchups and/or "
              "results aren't usable yet either - nothing to replay yet.", file=sys.stderr)
        sys.exit(1)

    bracket_stem = args.bracket_path.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def run_snapshot(n):
        try:
            upcoming_round_num, field_size, label, rows = compute_round_snapshot(
                state, n, args.simulations, args.seed, use_upset_boost=args.use_upset_boost)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        output_path = args.output_dir / f"{bracket_stem}_round_{upcoming_round_num}_of_{field_size}_players.csv"
        _report(label, rows, output_path, top_n=args.top_n)

    if args.all_rounds:
        for n in range(1, state.max_known_round + 1):
            run_snapshot(n)
        if is_partial_available:
            run_snapshot(state.target_round)
    else:
        run_snapshot(args.through_round)


if __name__ == "__main__":
    main()
