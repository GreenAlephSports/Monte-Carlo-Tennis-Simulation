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


def _load_artifact_market(bracket_path, draw_to_espn):
    """Reads the artifact-only market file written by
    model/research/fetch_artifact_market_odds.py, if one exists - never generated or touched by
    the production pipeline itself. That file keys records by draw CSV name (e.g. 'Zverev A.'),
    but matches_out below (like every other consolidated match row) is keyed by ESPN display name
    (e.g. 'Alexander Zverev') - draw_to_espn is the same per-bracket mapping hybrid_simulation.py
    itself uses to bridge the two, so remap here rather than storing ESPN names in the market file.
    Missing file (script never run for this bracket, or no market data resolved) just means every
    match's market_* fields stay None below, not an error."""
    market_path = OUTPUT_DIR / f"{bracket_path.stem}_artifact_market.json"
    if not market_path.exists():
        return {}
    with open(market_path, encoding="utf-8") as f:
        records = json.load(f)
    by_pair = {}
    for r in records:
        espn_a = draw_to_espn.get(r["player_a"], r["player_a"])
        espn_b = draw_to_espn.get(r["player_b"], r["player_b"])
        by_pair[frozenset((espn_a, espn_b))] = {**r, "player_a": espn_a, "player_b": espn_b}
    return by_pair


def _market_comparison(model_prob_a, player_a, player_b, market_by_pair):
    """Both directions of the model-vs-market comparison for one real match, keyed off draw
    names, so a single lookup covers whichever side the artifact needs to render:
      - gap_pp_a/b: model minus market, in percentage points (e.g. +6.4 = model 6.4pp higher)
      - relative_pct_a/b: that same gap as a fraction of the market's own number (e.g. model 55%
        vs market 50% is +5pp but +10% relative) - reported separately because a fixed pp gap
        means something very different for a coin-flip match than for a 90/10 mismatch.
    None everywhere when no market file was loaded for this bracket, or this exact pairing never
    got a resolved market price (market not yet posted, or a name The Odds API used couldn't be
    matched back to the draw - see fetch_artifact_market_odds.py's own warning for that case)."""
    entry = market_by_pair.get(frozenset((player_a, player_b)))
    if entry is None or model_prob_a is None:
        return {
            "market_prob_a": None, "market_prob_b": None,
            "gap_pp_a": None, "gap_pp_b": None,
            "relative_pct_a": None, "relative_pct_b": None,
            "market_captured_at": None,
        }
    market_prob_a = entry["market_prob_a"] if entry["player_a"] == player_a else 1 - entry["market_prob_a"]
    market_prob_b = 1 - market_prob_a
    model_prob_b = 1 - model_prob_a
    gap_pp_a = (model_prob_a - market_prob_a) * 100
    gap_pp_b = (model_prob_b - market_prob_b) * 100
    relative_pct_a = ((model_prob_a - market_prob_a) / market_prob_a * 100) if market_prob_a else None
    relative_pct_b = ((model_prob_b - market_prob_b) / market_prob_b * 100) if market_prob_b else None
    return {
        "market_prob_a": market_prob_a, "market_prob_b": market_prob_b,
        "gap_pp_a": gap_pp_a, "gap_pp_b": gap_pp_b,
        "relative_pct_a": relative_pct_a, "relative_pct_b": relative_pct_b,
        "market_captured_at": entry["captured_at"],
    }


def build_artifact_data(bracket_path):
    state = load_hybrid_state(bracket_path)
    draw = state.draw
    seed_by_name = {p.name: p.seed for p in state.bracket.players}
    bye_set = set(state.bye_players)
    live_by_pair = _live_status_by_pair(state.tournament_matches)
    market_by_pair = _load_artifact_market(bracket_path, state.draw_to_espn)

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
            pair = frozenset((m["player_a"], m["player_b"]))
            live = live_by_pair.get(pair)
            market = _market_comparison(m["model_prob_a"], m["player_a"], m["player_b"], market_by_pair)
            matches_out.append({
                **m, "round": c["round"], "round_label": c["label"],
                "live_state": live["live_state"] if live else None,
                "status_text": live["status_text"] if live else None,
                "score": live["score"] if live else None,
                **market,
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
