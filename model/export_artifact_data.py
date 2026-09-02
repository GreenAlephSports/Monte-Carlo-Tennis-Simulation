"""Builds one merged JSON per tour for the interactive bracket artifact: true draw-position order
(so Round 1 boxes sit next to their real opponent) plus every player's pretournament/current futures
odds and every real match (any round, decided or not) with model vs market, sourced from the
existing consolidated export - no new odds computation happens here.

Usage:
    python model/export_artifact_data.py brackets/us_open_2026_atp_real.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket_export import OUTPUT_DIR  # noqa: E402
from consolidated_export import consolidated_output_path  # noqa: E402
from hybrid_simulation import load_hybrid_state  # noqa: E402


def _live_status_by_pair(tournament_matches):
    """Keyed by frozenset({player_1, player_2}) - the model's win probability is frozen at whatever
    it was pre-match (see the artifact's own footer note: it isn't recomputed as a match plays out),
    while ESPN's status/score below is genuinely live - this lookup is what lets the artifact flag
    that gap instead of presenting a stale model number next to a live one silently."""
    by_pair = {}
    for m in tournament_matches:
        if not m["player_1"] or not m["player_2"]:
            continue
        sets = [f"{a}-{b}" for a, b in zip(m["sets_1"], m["sets_2"])]
        by_pair[frozenset((m["player_1"], m["player_2"]))] = {
            "live_state": m["status_state"],  # 'pre' | 'in' | 'post'
            "status_text": m["status"],
            "score": ", ".join(sets) if sets else None,
        }
    return by_pair


def build_artifact_data(bracket_path):
    state = load_hybrid_state(bracket_path)
    draw = state.draw
    seed_by_name = {p.name: p.seed for p in state.bracket.players}
    bye_set = set(state.bye_players)
    live_by_pair = _live_status_by_pair(state.tournament_matches)

    n = len(draw)
    quarter_size = n // 4
    slots = []
    for i, name in enumerate(draw):
        espn_name = state.draw_to_espn.get(name, name)
        slots.append({
            "position": i + 1,
            "quarter": f"Q{i // quarter_size + 1}",
            "seed": seed_by_name.get(name),
            "draw_name": name,
            "player": espn_name,
            "bye": name in bye_set,
        })

    consolidated_path = consolidated_output_path(bracket_path)
    with open(consolidated_path, encoding="utf-8") as f:
        consolidated = json.load(f)

    checkpoints = consolidated["round_history"]
    pretournament_by_player = {}
    for c in checkpoints:
        if c["checkpoint"] == "pretournament":
            pretournament_by_player = {p["player"]: p for p in c["players"]}
            break

    current = checkpoints[-1]
    current_by_player = {p["player"]: p for p in current["players"]}

    players_out = []
    for p in slots:
        name = p["player"]
        players_out.append({
            **p,
            "pretournament": pretournament_by_player.get(name),
            "current": current_by_player.get(name),
            "alive": name in current_by_player,
        })

    matches_out = []
    for c in checkpoints:
        if c["checkpoint"] == "pretournament":
            continue
        for m in c["matches"]:
            live = live_by_pair.get(frozenset((m["player_a"], m["player_b"])))
            matches_out.append({
                **m, "round": c["round"], "round_label": c["label"],
                "live_state": live["live_state"] if live else None,
                "status_text": live["status_text"] if live else None,
                "score": live["score"] if live else None,
            })

    return {
        "meta": consolidated["meta"],
        "draw_size": n,
        "players": players_out,
        "matches": matches_out,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bracket_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = build_artifact_data(args.bracket_path)
    output_path = args.output or (OUTPUT_DIR / f"{args.bracket_path.stem}_artifact_data.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {output_path}")
    print(f"players: {len(data['players'])}, matches: {len(data['matches'])}")
