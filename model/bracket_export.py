"""Exports one JSON file per sim run, matching Daron's integration spec:
  - every player keyed by the exact ESPN competitor.athlete.displayName - no name matching on
    our side, byte-for-byte, everywhere in the output (our internal ratings-csv-style names,
    e.g. "Sinner J.", never appear here - see espn_to_draw/draw_to_espn below).
  - match IDs are half-prefixed (T-/B-) + Daron's round label (R1/R2/R3/R16/QF/SF/F) + a
    sequential index within that half/round, e.g. "T-QF-1"; the Final is just "Final" (the one
    cross-half match). Each match has slot_a/slot_b, "probability" always meaning P(slot_a wins).
  - player rows are quarter-based (Q1-Q4, per Daron's correction): p_champ, p_sf (wins the
    quarter = reaches the semifinal), p_final (wins the half = reaches the final).
  - purely a model-output export: every probability in this file (matchups.p_slot_a/b,
    head_to_head.model_prob_a/b) is this project's own Elo-based win_probability(), with no
    market/odds-API blending anywhere in the pipeline - see this module's git history for the
    now-removed The Odds API integration.
Usage:
    python model/bracket_export.py brackets/cincinnati_2026_atp.yaml --simulations 10000
"""
import argparse
import json
import random
import re
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "research"))  # projected_draw_builder's overrides machinery

from bracket import (  # noqa: E402
    TOUR_CONFIG, get_matchups, match_draw_to_ratings, order_by_draw_position, split_byes,
    validate_bracket_structure, validate_draw,
)
from bracket_schema import BracketValidationError, load_bracket_yaml  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from hybrid_simulation import (  # noqa: E402
    TOUR_SINGLES_CATEGORY, build_known_pairings_by_round, build_real_results_by_round,
    known_matchups_for_round, match_espn_name_to_draw, replay_real_rounds,
)
from live_scores import LiveScoresError, extract_matches, fetch_scoreboard  # noqa: E402
from simulate import N_SIMULATIONS, run_simulations_tracking_milestones  # noqa: E402
from title_odds_movement import build_comparison, print_table, write_csv  # noqa: E402
from win_probability import win_probability  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
SEED = 42
# same path projected_draw_builder.DEFAULT_OVERRIDES_PATH resolves to - redefined here (rather
# than imported) because that module can't be imported at module load time, see the comment on
# the lazy import inside export_bracket_json below.
DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent.parent / "overrides.yaml"


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


def tag_halves_and_quarters(draw):
    """Static per-(draw name) half (Top/Bottom) + quarter (Q1-Q4) tag, derived directly from each
    player's index in `draw` - the original bracket-YAML draw-position order (order_by_draw_
    position / match_draw_to_ratings), fixed before any live result is known. A player's quarter
    is a pre-tournament fact that never changes as the draw plays out, so this needs no ESPN data
    and can never disagree with a player's real matches later. Deliberately NOT derived from
    reconstruct_leaves_by_round2_slot's ESPN-Round-2 traversal (see that function's own docstring
    for the adjacency problem it was built to solve) - that reconstruction turned out to have its
    own latent bug (confirmed empirically: with several Round 1 matches still undecided, its
    r1_pointer fallback silently produced 14 missing + 14 duplicated leaves out of 128 in a live
    US Open ATP draw, which would have dropped those live-match players from the exported title
    odds entirely, not just mislabeled a match id) - `draw`'s own order already IS true position,
    with no ESPN parsing needed to recover it."""
    draw_size = len(draw)

    def _half(i):
        return "Top" if i < draw_size / 2 else "Bottom"

    def _quarter(i):
        return f"Q{int(i // (draw_size / 4)) + 1}"

    half_by_name = {name: _half(i) for i, name in enumerate(draw)}
    quarter_by_name = {name: _quarter(i) for i, name in enumerate(draw)}
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
            # Real bug this branch used to hit (confirmed 2026-09-05, Coleman Wong at the 2026 US
            # Open): leaving the placeholder unresolved here means it's STILL literally
            # 'TBD (Qualifier N)' - a string that can never match this player's own real ESPN
            # results later (match_espn_name_to_draw has nothing to compare it against), so a real,
            # live, alive qualifier silently drops out of the whole export with no further trace
            # beyond this one warning - exactly the failure mode this function's own docstring
            # already anticipated ("the real qualifier can then never be matched against ESPN's
            # live results and gets silently dropped"). This was previously "documented but not
            # actually prevented".
            #
            # Fix: fall back to the raw ESPN name itself as the resolved name. This still gives a
            # synthetic STARTING_ELO rating (match_draw_to_ratings' existing tier-3 path - the raw
            # ESPN name won't match any ratings-csv lastname/initials pattern either, so it takes
            # the same fresh-placeholder branch a genuinely history-less debutant already does) -
            # a real accuracy cost (their true Elo history, which DOES exist under their csv-format
            # name, isn't used), but that's strictly better than the player vanishing from the
            # export entirely. Once real ESPN results start coming in for them, this name is now
            # the one draw_to_espn/espn_to_draw will actually see and match against - it survives.
            warnings.append(
                f"{placeholder.name}: ESPN shows {qualifier_espn_name!r} as the real qualifier, "
                f"but that name couldn't be matched to any player in the Elo ratings data - using "
                f"a fresh STARTING_ELO placeholder under their real ESPN name instead of leaving "
                f"them unresolved (their true Elo history, if any, is not reflected)"
            )
            resolved_by_id[id(placeholder)] = qualifier_espn_name
            continue

        resolved_by_id[id(placeholder)] = resolved_csv_name

    if not resolved_by_id:
        return players, warnings

    updated_players = [
        replace(p, name=resolved_by_id[id(p)]) if id(p) in resolved_by_id else p for p in players
    ]
    return updated_players, warnings


def pretournament_baseline_path(bracket_path, output_dir=OUTPUT_DIR):
    """Fixed, predictable filename for a bracket's locked pre-tournament reference snapshot -
    always under OUTPUT_DIR (not wherever --output happens to point for the regular/live export),
    so it can be found later without having to remember or pass around a path. Deliberately a
    different name from the regular '<stem>_bracket_export.json' - that filename gets overwritten
    on every run once real results exist, which is exactly the problem this file exists to avoid."""
    return Path(output_dir) / f"{Path(bracket_path).stem}_pretournament_baseline.json"


def ensure_pretournament_baseline(bracket_path, output, results_by_round, output_dir=OUTPUT_DIR):
    """Locks `output` in as the permanent 'before any real match happened' reference point, the
    first time (and only the first time) this bracket is ever exported with zero real results
    anywhere in results_by_round - i.e. genuinely pre-tournament, not just early-tournament. Once
    written, this file is never touched again by this function (or anything else in this module) -
    a later call, even from a run with a full live draw, always finds the file already there and
    leaves it alone, so --through-round/--all-rounds-style callers can diff against the exact same
    starting point no matter how far the tournament has since progressed.

    Deliberately does NOT try to retroactively reconstruct a baseline once real results already
    exist by the time this bracket is first exported (e.g. someone starts running this mid-Round-1)
    - there's no real 'before' state left to capture at that point; see the CLI's own warning for
    that case. Returns (baseline_path, was_just_created)."""
    baseline_path = pretournament_baseline_path(bracket_path, output_dir)
    if any(results_by_round.values()):
        return baseline_path, False
    if baseline_path.exists():
        return baseline_path, False
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return baseline_path, True


def export_bracket_json(
    bracket_path, output_path=None, n_simulations=N_SIMULATIONS, seed=SEED, dates=None,
    overrides_path=DEFAULT_OVERRIDES_PATH,
):
    # lazy import: projected_draw_builder.py -> calibration_log.py -> live_calibration_check.py
    # -> `from bracket_export import resolve_qualifier_placeholders` - a real circular import if
    # this were a top-level import in this module instead.
    from projected_draw_builder import apply_health_adjustments, health_check_top_seeds, load_overrides

    bracket = load_bracket_yaml(bracket_path)
    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    validate_bracket_structure(byes)

    tour_config = TOUR_CONFIG[bracket.tour]
    matches_history = load_matches_for_tour(bracket.tour)
    ratings_df = calculate_elo_ratings(matches_history, bracket.start_date, tour=bracket.tour)
    ratings_df = ratings_df.sort_values("overall_elo", ascending=False).reset_index(drop=True)

    # manual, disclosed judgment-call adjustments (e.g. a known injury Elo structurally can't see -
    # see load_overrides/apply_health_adjustments docstrings) - same mechanism
    # simulate_projected_draw already applies for a pre-tournament projection, wired in here too so
    # a REAL/live bracket export doesn't silently ignore it. Resolved through the same fuzzy
    # tiered matcher withdrawals use, errors loudly on an unresolved name rather than no-op'ing.
    overrides = load_overrides(overrides_path, bracket.tour)
    ratings_df, health_adjustments_applied = apply_health_adjustments(ratings_df, overrides["health_adjustments"])
    if health_adjustments_applied:
        print(f"\nHealth adjustment(s) applied ({len(health_adjustments_applied)}):")
        for adj in health_adjustments_applied:
            print(f"  {adj['player']}: {adj['elo_penalty']:+.0f} Elo - {adj['reason']}")

    # dates is passed straight through to fetch_scoreboard - ESPN's undated scoreboard defaults
    # to "today" server-side, which only finds a tournament while it's still live/recent (see
    # espn_bracket.py's build_bracket_players and backtest_hard_court.py's own note on this). A
    # single date anywhere inside the tournament's window returns its complete event regardless
    # of how long ago it finished, so exporting a checkpoint for an already-concluded event needs
    # one explicitly.
    espn_data = fetch_scoreboard(bracket.tour.lower(), dates=dates)
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

    # pre-draw health check: top 20 seeds, flagged if their most recent TRACKED match ended in a
    # retirement/walkover - visibility only (see health_check_top_seeds docstring), same check
    # simulate_projected_draw already runs for a projected bracket, now run here too so a real/live
    # bracket surfaces the same signal the moment it's built, not just a pre-tournament projection.
    seeded_players = [(p.seed, name) for p, name in zip(players, draw)]
    health_check_top_seeds(seeded_players, top_n=20)

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

    _half_by_draw, quarter_by_draw = tag_halves_and_quarters(draw)
    round_label_map = build_round_label_map(round_sequence)

    # true_order/leaf_position: TRUE draw-position order and each player's index in it, used by
    # the matchups loop below (match_id), simulation seeding, and (via tag_halves_and_quarters
    # above) quarter tags - all three must agree, and `draw` already IS true position order (see
    # tag_halves_and_quarters' docstring for why this replaced the older ESPN-Round-2-based
    # reconstruction), so no further reconstruction is needed here.
    true_order = list(zip(draw, byes))
    draw_size = len(draw)
    leaf_position = {name: i for i, name in enumerate(draw)}

    # alive = every player we have a real ESPN name for, minus anyone already lost in a decided match
    losers = set()
    for round_results in results_by_round.values():
        for pair, winner in round_results.items():
            losers.add(next(p for p in pair if p != winner))
    alive_draw_names = [p for p in draw if p in draw_to_espn and p not in losers]

    # --- matchups: every unsettled match, any round, where both sides are already real names ---
    matchups = {}
    for round_label in round_sequence:
        round_num = round_sequence.index(round_label) + 1
        round_matches = [m for m in tournament_matches if m["round"] == round_label]
        daron_round = round_label_map[round_label]
        decided = results_by_round.get(round_num, {})
        # total matches this round has by TRUE bracket structure (draw_size halves each round) -
        # deliberately NOT len(round_matches), which is just however many ESPN currently lists for
        # this round and carries no adjacency guarantee (see leaf_position's own comment above).
        matches_this_round = draw_size // (2 ** round_num)

        for m in round_matches:
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
                # true_index: this match's 0-based position within the round, derived from where
                # its players actually sit in the original draw (leaf_position), not from ESPN's
                # list order - the same fix already applied to quarters/simulation seeding above.
                true_index = min(leaf_position[draw_a], leaf_position[draw_b]) // (2 ** round_num)
                half_prefix = "T" if true_index < matches_this_round / 2 else "B"
                half_index = (
                    true_index if true_index < matches_this_round / 2
                    else true_index - matches_this_round // 2
                )
                match_id = f"{half_prefix}-{daron_round}-{half_index + 1}"

            # purely this project's own Elo-based model - no market/odds-API blending, see the
            # module docstring.
            prob_a = round(win_probability(draw_a, draw_b, bracket.surface, tour_config.ratings_path), 3)
            matchups[match_id] = {
                "slot_a": p1_raw, "slot_b": p2_raw,
                "p_slot_a": prob_a, "p_slot_b": round(1 - prob_a, 3),
            }

    print(f"matchups: {len(matchups)} unsettled")

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
    # milestones's docstring) - reusing true_order/leaf_position (computed above from
    # leaves_by_slot, the same reconstruction the quarter tags and matchups match_ids come from)
    # guarantees the simulated bracket tree, the displayed quarters, and the match ids can never
    # disagree. fields[]'s own "round winners then byes appended" concatenation is only good
    # enough for pinning known results (frozenset-keyed, order-independent) - not for this.
    if target_round == 1:
        ordered_field = [name for name, _is_bye in true_order]
        is_bye = [is_bye_flag for _name, is_bye_flag in true_order]
    else:
        ordered_field = sorted(partial_field, key=lambda draw_name: leaf_position.get(draw_name, len(true_order)))
        is_bye = [False] * len(ordered_field)

    random.seed(seed)
    champ_counts, sf_counts, final_counts = run_simulations_tracking_milestones(
        ordered_field, is_bye, partial_known_results, bracket.surface, n_simulations, tour_config.ratings_path
    )

    health_adjustment_by_player = {adj["player"]: adj for adj in health_adjustments_applied}
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
            # present ONLY for a player with an active manual override, so this is never silently
            # indistinguishable from an unadjusted model output - see apply_health_adjustments.
            **({"health_adjustment": {
                "elo_penalty": health_adjustment_by_player[draw_name]["elo_penalty"],
                "reason": health_adjustment_by_player[draw_name]["reason"],
            }} if draw_name in health_adjustment_by_player else {}),
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
            # model_prob_a: pure Elo-model probability, win_probability()'s own output every
            # time - structurally never blended with market data (there is none in this export
            # any more, see the module docstring). Named "model_prob_a" rather than a bare
            # "prob_a" to match matchups' field naming and stay unambiguous to any downstream
            # consumer, per Daron's request.
            model_prob_a = round(win_probability(draw_a, draw_b, bracket.surface, tour_config.ratings_path), 3)
            head_to_head[f"{name_a}|{name_b}"] = {
                "model_prob_a": model_prob_a,
                "model_prob_b": round(1 - model_prob_a, 3),
            }

    if health_adjustments_applied:
        warnings = warnings + [
            "One or more players carry a manual health_adjustment (see overrides.yaml) - a "
            "disclosed human judgment call, not a statistical model output. Affected players are "
            "flagged individually under players[].health_adjustment; see also the top-level "
            "health_adjustments list."
        ]

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
        "health_adjustments": health_adjustments_applied,
        "warnings": warnings,
    }

    if output_path is None:
        output_path = OUTPUT_DIR / f"{Path(bracket_path).stem}_bracket_export.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    baseline_path, baseline_created = ensure_pretournament_baseline(bracket_path, output, results_by_round)
    if baseline_created:
        print(f"Locked new pre-tournament baseline: {baseline_path} (zero real results existed yet - "
              f"this file is now permanent and will never be overwritten by a future run)")

    return output_path, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bracket_path", type=Path)
    parser.add_argument("--simulations", type=int, default=N_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dates", default=None,
                         help="YYYYMMDD, passed to ESPN's ?dates= - needed for an already-"
                              "concluded event, which the undated (\"today\") scoreboard can't find")
    parser.add_argument(
        "--overrides", type=Path, default=DEFAULT_OVERRIDES_PATH,
        help="overrides YAML for manual health_adjustments (withdrawals/seed/name overrides don't "
             "apply here - those are build-time only, see espn_bracket.py) - see load_overrides "
             "docstring; pass a nonexistent path to skip")
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="pre-tournament baseline JSON to diff this export against (default: this bracket's "
             "auto-locked <stem>_pretournament_baseline.json, if one exists - see "
             "ensure_pretournament_baseline)")
    parser.add_argument(
        "--no-compare", action="store_true",
        help="skip printing/writing the pre-tournament-vs-current comparison table")
    args = parser.parse_args()

    try:
        output_path, output = export_bracket_json(
            args.bracket_path, args.output, args.simulations, args.seed, dates=args.dates,
            overrides_path=args.overrides,
        )
    except (BracketValidationError, LiveScoresError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nWrote {output_path}")
    print(f"players: {len(output['players'])} alive, matchups: {len(output['matchups'])}, "
          f"head_to_head: {len(output['head_to_head'])}, warnings: {len(output['warnings'])}")

    if not args.no_compare:
        baseline_path = args.baseline or pretournament_baseline_path(args.bracket_path)
        if baseline_path.exists() and baseline_path.resolve() != output_path.resolve():
            rows, baseline_meta, current_meta = build_comparison(baseline_path, output_path)
            print(f"\n=== Pre-tournament vs current (baseline: {baseline_path}) ===")
            print_table(rows, baseline_meta, current_meta, top_n=len(rows))
            comparison_csv = OUTPUT_DIR / f"{args.bracket_path.stem}_vs_baseline.csv"
            write_csv(rows, comparison_csv)
            print(f"Wrote full pre-tournament-vs-current comparison ({len(rows)} players) to {comparison_csv}")
        elif not baseline_path.exists():
            print(f"\n(no pre-tournament baseline available at {baseline_path} - either this bracket "
                  f"already had real results the first time it was ever exported (nothing genuinely "
                  f"'pre-tournament' left to compare against), or this is that first, baseline-"
                  f"defining run itself - pass --baseline to point at a specific file instead)")
