"""Builds the single DATA blob consumed by the three-tab dashboard artifact (bracket tree /
round-by-round / market vs. model), by reading already-regenerated output files - never
recomputes a simulation itself. Run bracket_export.py and hybrid_simulation.py --all-rounds
first (this script assumes their output files, plus one fresh live ESPN pull, are current):

    python model/bracket_export.py brackets/cincinnati_2026_atp_demo.yaml
    python model/hybrid_simulation.py brackets/cincinnati_2026_atp_demo.yaml --all-rounds
    python model/dashboard_export.py brackets/cincinnati_2026_atp_demo.yaml

Writes output/<bracket stem>_dashboard.json.

--- Section 1 (bracket tree) schema notes, reverse-engineered from the last hand-built version of
this view ("Match Point Drift") so the same renderer can be reused unmodified ---
meta: {tournament, round_labels: {"1".."7": label}, max_known_round, current_round}
rounds: {"1".."7": [node, ...]}, node =
  bye:   {type:"bye", player, tourney_prob, half, row}
  match: {type:"match", round, a, b, winner, match_prob_a, match_prob_b, tourney_prob_a,
          tourney_prob_b, half, row, [child_a_idx, child_b_idx for round>=2]}
transitions: [{from, to, round, trigger_matches:[{winner,loser}], deltas:{player: delta}}]

Round 1's node list is built directly from ESPN's Round 1 matches + the draw's own bye list,
interleaved into true bracket order via hybrid_simulation.reconstruct_leaves_by_round2_slot's
same per-name walk (re-derived here at box granularity - that function returns leaves, not box
boundaries, so it can't be reused directly, but reuses every one of its lower-level building
blocks: results_by_round, match_espn_name_to_draw, the bye/non-bye player lists). Round r>=2's
node list is one node per ESPN-listed match for that round (in ESPN's own order, which is the
real bracket-sheet order - see reconstruct_leaves_by_round2_slot's own docstring) - child indices
are then a structural (2*i, 2*i+1) position, NOT a name lookup, because a still-undetermined
'TBD' slot has no name to look up yet. This positional relationship is what the isotonic
regression below also relies on for row placement, and was spot-checked against the previous
artifact's own embedded data (round 2 node 0 has child idx {0,1}, node 1 has {2,3}, etc.).

Row placement: round 1 gets its row from true bracket order position (0..N-1, split into two
halves). Round r>=2's naive row is the mean of its two children's rows; IsotonicRegression then
enforces non-decreasing rows across the round (by structural position) so connector lines never
cross - falls back to a manual pool-adjacent-violators averaging if sklearn isn't importable.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket import TOUR_CONFIG, order_by_draw_position, split_byes, validate_bracket_structure  # noqa: E402
from bracket_export import (  # noqa: E402
    ODDS_API_KEY, build_round_label_map, discover_odds_sport_key, fetch_devigged_odds,
)
from bracket_schema import load_bracket_yaml  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from hybrid_simulation import (  # noqa: E402
    TOUR_SINGLES_CATEGORY, build_real_results_by_round, build_round_sequence, match_espn_name_to_draw,
)
from live_scores import LiveScoresError, extract_matches, fetch_scoreboard, filter_by_tour  # noqa: E402
from win_probability import _load_ratings, win_probability  # noqa: E402

try:
    from sklearn.isotonic import IsotonicRegression
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _isotonic_nondecreasing(values):
    """Smallest non-decreasing sequence close (least-squares) to `values`, in index order - used
    to keep round r>=2's mean-of-children row placement from ever producing a visually-crossing
    (non-monotonic) box order. Falls back to a manual pool-adjacent-violators implementation
    (textbook PAVA - repeatedly average any adjacent decreasing pair until none remain) if
    scikit-learn isn't installed, so this dashboard doesn't gain a hard new dependency."""
    if _HAVE_SKLEARN:
        ir = IsotonicRegression(increasing=True)
        xs = list(range(len(values)))
        return list(ir.fit_transform(xs, values))

    pooled = [[v, 1] for v in values]  # [value, weight] blocks
    i = 0
    while i < len(pooled) - 1:
        if pooled[i][0] > pooled[i + 1][0]:
            v1, w1 = pooled[i]
            v2, w2 = pooled[i + 1]
            merged = [(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2]
            pooled[i:i + 2] = [merged]
            i = max(0, i - 1)
        else:
            i += 1
    out = []
    for v, w in pooled:
        out.extend([v] * w)
    return out


# ---------- Section 1: bracket tree ----------

def build_round1_nodes(tournament_matches, non_bye_players, bye_players, results_by_round, name_aliases):
    """One node per Round-1 'box' (a bye, or a real 2-player match), in true bracket order -
    reimplements reconstruct_leaves_by_round2_slot's per-name walk at box granularity (that
    function only returns flattened leaves, not box boundaries), reusing every one of its
    lower-level primitives (results_by_round[1], match_espn_name_to_draw + the tour's real name
    aliases, the already-ordered non_bye_players/bye_players lists) so it inherits the same
    name-resolution fixes (e.g. the Landaluce/Andaluce PDF-glitch alias in bracket.py)."""
    round1_matches = [m for m in tournament_matches if m["round"] == "Round 1"]
    round1_names = {n for m in round1_matches for n in (m["player_1"], m["player_2"]) if n and n != "TBD"}
    draw_csv_names = non_bye_players + bye_players

    winner_to_match = {}
    for m in round1_matches:
        p1 = match_espn_name_to_draw(m["player_1"], draw_csv_names, name_aliases)
        p2 = match_espn_name_to_draw(m["player_2"], draw_csv_names, name_aliases)
        if p1 is None or p2 is None:
            continue
        winner = results_by_round.get(1, {}).get(frozenset((p1, p2)))
        if winner is not None:
            winner_to_match[winner] = (m, p1, p2)

    round2_matches = [m for m in tournament_matches if m["round"] == "Round 2"]
    nodes = []
    used_ids = set()
    bye_pointer = 0
    r1_pointer = 0
    for m2 in round2_matches:
        for name in (m2["player_1"], m2["player_2"]):
            is_direct_bye = bool(name) and name != "TBD" and name not in round1_names
            if is_direct_bye:
                bye_name = bye_players[bye_pointer]
                bye_pointer += 1
                nodes.append({"type": "bye", "player": bye_name})
                continue

            resolved = match_espn_name_to_draw(name, draw_csv_names, name_aliases) if name and name != "TBD" else None
            hit = winner_to_match.get(resolved) if resolved is not None else None
            if hit is not None and id(hit[0]) not in used_ids:
                used_ids.add(id(hit[0]))
                _m, p1, p2 = hit
                winner = results_by_round.get(1, {}).get(frozenset((p1, p2)))
                nodes.append({"type": "match", "round": 1, "a": p1, "b": p2, "winner": winner})
            else:
                # fallback: still-undecided or unresolved slot - positional guess, same fallback
                # reconstruct_leaves_by_round2_slot itself uses for this case.
                if 2 * r1_pointer + 1 < len(non_bye_players):
                    a, b = non_bye_players[2 * r1_pointer], non_bye_players[2 * r1_pointer + 1]
                else:
                    a, b = None, None
                nodes.append({"type": "match", "round": 1, "a": a, "b": b, "winner": None})
                r1_pointer += 1
    return nodes


def build_later_round_nodes(round_num, round_label, tournament_matches, name_aliases, draw_csv_names,
                             prev_nodes):
    """One node per ESPN-listed match for this round, in ESPN's own order (the real bracket-sheet
    order). Child indices are found by NAME (which previous-round box's own winner - or bye
    player - equals this node's a/b), NOT positionally (2*i, 2*i+1) - confirmed empirically that
    positional indexing only holds for round 1 -> round 2 here, not beyond: cross-checking every
    round pair against real Cincinnati data found 33 mismatches from round 3 onward (e.g. round 3
    node 6 lists 'Fils A.' as a player, but position-based indices {12,13} into round 2 pointed at
    a completely different pair of names) - the same "fields[] concatenation order isn't true
    bracket adjacency past round 2" issue hybrid_simulation.py's own module comments describe for
    the real 'Fils A. vs Zverev A.' pairing bug. A child index is left None (skipped by the
    renderer's connector-drawing code) only when that name can't be found among the previous
    round's decided winners/byes at all - a still-undetermined 'TBD' slot with no real winner
    yet."""
    winner_index_by_name = {}
    for j, node in enumerate(prev_nodes):
        name = node["player"] if node["type"] == "bye" else node.get("winner")
        if name is not None:
            winner_index_by_name[name] = j

    round_matches = [m for m in tournament_matches if m["round"] == round_label]
    nodes = []
    for m in round_matches:
        a = match_espn_name_to_draw(m["player_1"], draw_csv_names, name_aliases) if m["player_1"] and m["player_1"] != "TBD" else None
        b = match_espn_name_to_draw(m["player_2"], draw_csv_names, name_aliases) if m["player_2"] and m["player_2"] != "TBD" else None
        winner = None
        if m["status_state"] == "post" and m["winner"]:
            winner = a if match_espn_name_to_draw(m["winner"], draw_csv_names, name_aliases) == a else (
                b if match_espn_name_to_draw(m["winner"], draw_csv_names, name_aliases) == b else None
            )
        nodes.append({
            "type": "match", "round": round_num, "a": a, "b": b, "winner": winner,
            "child_a_idx": winner_index_by_name.get(a) if a else None,
            "child_b_idx": winner_index_by_name.get(b) if b else None,
        })
    return nodes


def assign_rows_and_halves(rounds_by_num, max_round):
    """Round 1: row = position within its half (0..N/2-1), half = 'left' for the first half of the
    true-bracket-ordered list, 'right' for the second. Round r>=2 (not the Final): naive row =
    mean of the two children's rows, then isotonic-regression'd (see _isotonic_nondecreasing) to
    stay non-decreasing across the round so connector lines never visually cross; half is
    inherited from either child (they're always the same half). The Final gets half=None."""
    n1 = len(rounds_by_num[1])
    half_size = n1 // 2
    for i, node in enumerate(rounds_by_num[1]):
        node["half"] = "left" if i < half_size else "right"
        node["row"] = i if i < half_size else i - half_size

    for r in range(2, max_round + 1):
        nodes = rounds_by_num[r]
        prev = rounds_by_num[r - 1]
        is_final = (r == max_round)
        naive_rows, halves = [], []
        for node in nodes:
            ci_a, ci_b = node.get("child_a_idx"), node.get("child_b_idx")
            child_rows = [prev[ci]["row"] for ci in (ci_a, ci_b) if ci is not None and ci < len(prev)]
            naive_rows.append(sum(child_rows) / len(child_rows) if child_rows else 0.0)
            child_halves = [prev[ci]["half"] for ci in (ci_a, ci_b) if ci is not None and ci < len(prev)]
            halves.append(child_halves[0] if child_halves else None)
        if is_final:
            for node in nodes:
                node["half"] = None
            nodes[0]["row"] = naive_rows[0] if naive_rows else 0.0
            continue
        # isotonic within each half separately (left/right rows are independent numbering lanes)
        for half in ("left", "right"):
            idxs = [i for i, h in enumerate(halves) if h == half]
            if not idxs:
                continue
            smoothed = _isotonic_nondecreasing([naive_rows[i] for i in idxs])
            for i, row in zip(idxs, smoothed):
                nodes[i]["row"] = row
                nodes[i]["half"] = half


def attach_probabilities(rounds_by_num, max_round, checkpoints_by_round, surface, ratings_path):
    """match_prob_a/b: this project's static pregame Elo probability for the exact pairing
    (win_probability - the same function every other view in this dashboard uses). tourney_prob_a/b
    (bye nodes: tourney_prob): that player's tournament_win_probability from the round-N
    checkpoint CSV whose label matches this node's own round (i.e. the live title odds AS OF
    right before this round was played) - None if that player isn't in the checkpoint (name
    unresolved, or no checkpoint exists yet for a future round)."""
    for r in range(1, max_round + 1):
        cp = checkpoints_by_round.get(r, {})  # player -> tournament_win_probability
        for node in rounds_by_num[r]:
            if node["type"] == "bye":
                node["tourney_prob"] = cp.get(node["player"])
                continue
            a, b = node.get("a"), node.get("b")
            if a and b:
                try:
                    node["match_prob_a"] = win_probability(a, b, surface, ratings_path)
                    node["match_prob_b"] = 1 - node["match_prob_a"]
                except ValueError:
                    node["match_prob_a"] = node["match_prob_b"] = None
            else:
                node["match_prob_a"] = node["match_prob_b"] = None
            node["tourney_prob_a"] = cp.get(a) if a else None
            node["tourney_prob_b"] = cp.get(b) if b else None


def build_transitions(round_label_map, checkpoints_by_round, results_by_round, max_known_round):
    """One entry per round boundary through max_known_round: the real matches decided in that
    round (trigger_matches) and every player's tournament_win_probability delta between the two
    surrounding checkpoints - both computed directly from real, already-saved data (the round-N
    checkpoint CSVs and results_by_round), no re-simulation here."""
    transitions = []
    round_nums = sorted(checkpoints_by_round)
    # checkpoint round R's data is the state right before round R is played, i.e. "after round
    # R-1 decided" - checkpoint 1 is the one exception (the pre-tournament full-field baseline,
    # from run_tournament.py, with no real round decided yet).
    labels = {1: "Pre-tournament"}
    for r in round_nums:
        if r == 1:
            continue
        labels[r] = f"After {round_label_map.get(r - 1, f'Round {r - 1}')}"

    for idx in range(1, len(round_nums)):
        prev_r, cur_r = round_nums[idx - 1], round_nums[idx]
        if prev_r > max_known_round + 1:
            break
        prev_cp, cur_cp = checkpoints_by_round[prev_r], checkpoints_by_round[cur_r]
        deltas = {}
        for player in set(prev_cp) | set(cur_cp):
            if player in prev_cp and player in cur_cp:
                deltas[player] = round(cur_cp[player] - prev_cp[player], 4)
        trigger_matches = [
            {"winner": winner, "loser": next(p for p in pair if p != winner)}
            for pair, winner in results_by_round.get(prev_r, {}).items()
        ]
        transitions.append({
            "from": labels[prev_r], "to": labels[cur_r], "round": prev_r,
            "trigger_matches": trigger_matches, "deltas": deltas,
        })
    return transitions


# ---------- Section 3: market vs model (compare_match.py's exact logic) ----------

def build_market_vs_model(export_data, tour, surface, ratings_path, name_aliases, status_by_pair):
    """Reuses compare_match.py's exact fields/formulas (gap in percentage points, relative_change
    as a % of the market's own probability, PREGAME/LIVE/FINAL status labeling with the same
    '(stale)' self-label for a non-pregame market price) - scoped to data['matchups'] only (real
    scheduled pairings), never head_to_head (the full hypothetical set), per the dashboard spec."""
    draw_csv_names = set(_load_ratings(ratings_path).index)
    rows = []
    for match_id, info in export_data["matchups"].items():
        a, b = info["slot_a"], info["slot_b"]
        market_p = info["p_slot_a"]
        status_state = status_by_pair.get(frozenset((a, b)))
        is_pregame = status_state == "pre"
        status_label = {"pre": "PREGAME", "in": "LIVE (excluded)", "post": "FINAL (excluded)"}.get(
            status_state, "unknown (excluded)"
        )

        csv_a = match_espn_name_to_draw(a, draw_csv_names, name_aliases)
        csv_b = match_espn_name_to_draw(b, draw_csv_names, name_aliases)

        model_p = gap = rel_pct = None
        if csv_a is not None and csv_b is not None:
            try:
                model_p = win_probability(csv_a, csv_b, surface, ratings_path=ratings_path)
                if is_pregame:
                    gap = market_p - model_p
                    rel_pct = (model_p - market_p) / market_p * 100
            except ValueError:
                model_p = None

        rows.append({
            "match_id": match_id, "player_a": a, "player_b": b,
            "market_prob": round(market_p, 4), "market_is_stale": not is_pregame,
            "model_prob": round(model_p, 4) if model_p is not None else None,
            "gap_pp": round(gap * 100, 2) if gap is not None else None,
            "relative_change_pct": round(rel_pct, 1) if rel_pct is not None else None,
            "status": status_label, "is_pregame": is_pregame,
        })
    rows.sort(key=lambda row: abs(row["gap_pp"]) if row["gap_pp"] is not None else -1, reverse=True)
    return rows


# ---------- Section 2: round-by-round (direct pass-through of the checkpoint CSVs) ----------

def build_round_by_round(checkpoint_dfs, round_label_map):
    """checkpoint_dfs: {round_num: DataFrame} from the round_N_of_M_players.csv files -
    hybrid_simulation.py already computes exactly the four columns this section needs
    (player/tournament_win_probability/upcoming_opponent/upcoming_match_win_probability); this
    just reshapes them into the dashboard's per-round row list, unchanged."""
    out = {}
    for r, df in sorted(checkpoint_dfs.items()):
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "player": row["player"],
                "tournament_win_probability": round(float(row["tournament_win_probability"]), 4),
                "upcoming_opponent": row.get("upcoming_opponent") if pd_notna(row.get("upcoming_opponent")) else None,
                "upcoming_match_win_probability": (
                    round(float(row["upcoming_match_win_probability"]), 4)
                    if pd_notna(row.get("upcoming_match_win_probability")) else None
                ),
            })
        out[r] = {"label": round_label_map.get(r, f"Round {r}"), "players": rows}
    return out


def pd_notna(x):
    import pandas as pd
    return pd.notna(x)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bracket_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    bracket = load_bracket_yaml(args.bracket_path)
    tour_config = TOUR_CONFIG[bracket.tour]
    bracket_stem = args.bracket_path.stem

    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    validate_bracket_structure(byes)

    # match_draw_to_ratings needs the current ratings snapshot, same as bracket_export.py -
    # cheap here since we only need it for name resolution (draw csv names), not to resimulate.
    matches_history = load_matches_for_tour(bracket.tour)
    ratings_df = calculate_elo_ratings(matches_history, bracket.start_date)
    from bracket import match_draw_to_ratings
    draw, resolutions, ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date
    )
    non_bye_players, bye_players = split_byes(draw, byes)

    espn_data = fetch_scoreboard(bracket.tour.lower())
    espn_matches, _stats = extract_matches(espn_data)
    category = TOUR_SINGLES_CATEGORY[bracket.tour.lower()]
    tournament_matches = [
        m for m in espn_matches if m["tournament"] == bracket.tournament and m["category"] == category
    ]
    if not tournament_matches:
        sys.exit(f"ERROR: no live matches found for {bracket.tournament!r} / {category}")

    results_by_round, round_sequence, _unresolved = build_real_results_by_round(
        tournament_matches, draw, tour_config.name_aliases
    )
    round_label_map = build_round_label_map(round_sequence)
    label_to_num = {v: k for k, v in round_label_map.items()}
    # round_label_map maps ESPN's own label ('Round 1', 'Quarterfinal', ...) -> Daron's short code
    # (R1/QF/...); this dashboard wants the numeric round index instead - build that directly from
    # build_round_sequence's own ordering (main-draw rounds only, qualifying excluded).
    main_draw_sequence = build_round_sequence({m["round"] for m in tournament_matches if m["round"]})
    round_num_by_label = {label: i + 1 for i, label in enumerate(main_draw_sequence)}
    round_label_by_num = {i + 1: label for i, label in enumerate(main_draw_sequence)}
    max_round = len(main_draw_sequence)

    # --- load every already-regenerated round checkpoint CSV for this bracket ---
    # round 1's own checkpoint is the pre-tournament full-field baseline (run_tournament.py's
    # output - hybrid_simulation.py --all-rounds never produces a "Round 1" file itself, since its
    # first snapshot is already "Round 2 - X about to play", i.e. AFTER round 1's results are
    # folded in - see hybrid_simulation.main()'s run_snapshot: n=1 always labels its output
    # "Round {n+1}").
    checkpoint_dfs = {}
    baseline_path = OUTPUT_DIR / f"{bracket.tournament.lower().replace(' ', '_')}_{bracket.year}_simulation_results_{bracket.tour.lower()}.csv"
    if baseline_path.exists():
        checkpoint_dfs[1] = pd.read_csv(baseline_path)
    for csv_path in OUTPUT_DIR.glob(f"{bracket_stem}_round_*_of_*_players.csv"):
        # filename: <stem>_round_<N>_of_<M>_players.csv
        stem_parts = csv_path.stem.split("_round_")[-1].split("_of_")
        round_num = int(stem_parts[0])
        checkpoint_dfs[round_num] = pd.read_csv(csv_path)
    if not checkpoint_dfs:
        sys.exit(f"ERROR: no checkpoint files found in {OUTPUT_DIR} - run run_tournament.py and "
                  f"hybrid_simulation.py --all-rounds first.")
    # checkpoint N represents the state entering round N (rounds 1..N-1 already decided), so the
    # highest checkpoint number available is one past the last fully-decided round - e.g. a
    # "Round 7 (Final) about to play" checkpoint means round 6 (Semifinal) is the last one known,
    # not round 7 itself (the Final hasn't been played).
    current_round = max(checkpoint_dfs)
    max_known_round = current_round - 1
    checkpoints_by_round = {
        r: dict(zip(df["player"], df["tournament_win_probability"])) for r, df in checkpoint_dfs.items()
    }

    # --- Section 1: bracket tree ---
    round1_nodes = build_round1_nodes(
        tournament_matches, non_bye_players, bye_players, results_by_round, tour_config.name_aliases
    )
    rounds_by_num = {1: round1_nodes}
    draw_csv_names = set(draw)
    for r in range(2, max_round + 1):
        label = round_label_by_num[r]
        rounds_by_num[r] = build_later_round_nodes(
            r, label, tournament_matches, tour_config.name_aliases, draw_csv_names, rounds_by_num[r - 1]
        )
    assign_rows_and_halves(rounds_by_num, max_round)
    attach_probabilities(rounds_by_num, max_round, checkpoints_by_round, bracket.surface, tour_config.ratings_path)
    transitions = build_transitions(round_label_by_num, checkpoints_by_round, results_by_round, max_known_round)

    section1 = {
        "meta": {
            "tournament": bracket.tournament, "round_labels": {str(r): l for r, l in round_label_by_num.items()},
            "max_known_round": max_known_round, "current_round": current_round,
        },
        "rounds": {str(r): rounds_by_num[r] for r in rounds_by_num},
        "transitions": transitions,
    }

    # --- Section 2: round-by-round ---
    section2 = build_round_by_round(checkpoint_dfs, round_label_by_num)

    # --- Section 3: market vs model ---
    export_path = OUTPUT_DIR / f"{bracket_stem}_bracket_export.json"
    export_data = json.loads(export_path.read_text(encoding="utf-8"))
    status_by_pair = {
        frozenset((m["player_1"], m["player_2"])): m["status_state"] for m in tournament_matches
    }
    section3 = build_market_vs_model(
        export_data, bracket.tour, bracket.surface, tour_config.ratings_path, tour_config.name_aliases,
        status_by_pair,
    )

    dashboard = {"section1": section1, "section2": section2, "section3": section3}

    output_path = args.output or (OUTPUT_DIR / f"{bracket_stem}_dashboard.json")
    output_path.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"section1: {max_round} rounds, {len(transitions)} transitions, current_round={current_round}")
    print(f"section2: {len(section2)} round checkpoints")
    print(f"section3: {len(section3)} real scheduled matchup(s)")


if __name__ == "__main__":
    main()
