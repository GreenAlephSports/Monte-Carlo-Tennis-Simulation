"""Builds one consolidated per-bracket JSON combining everything bracket_export.py and
hybrid_simulation.py already compute separately, so there's a single file to always check instead
of stitching several outputs together by hand.

The whole thing is one ordered "round_history" list of self-contained checkpoints - pretournament,
each historical round once it's fully decided, then the current/latest state - each carrying BOTH
halves of the picture for that exact moment:
  - "players": futures odds by model (who wins the whole tournament from here).
  - "matches": that checkpoint's own real matchups with this project's own model probability -
    pretournament has none yet (no real matches exist), a historical round has its own now-decided
    matches, and "current" has the live round's matches, decided and not-yet-decided alike. Purely
    a model-output export - no market/odds-API data anywhere in this pipeline.

A historical round's checkpoint, once written, never changes again (see
hybrid_simulation.build_round_history's own caching) - only the trailing "current" checkpoint (and
the freshly-decided round that just replaced what used to be "current") gets recomputed each
refresh.

Refreshed automatically by live_match_watcher.py the same way its own bracket_export.py export
already is - see watch()'s call to build_consolidated_export after every export_bracket_json run.

Usage:
    python model/consolidated_export.py brackets/us_open_2026_atp_real.yaml
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket_export import (  # noqa: E402
    DEFAULT_OVERRIDES_PATH, OUTPUT_DIR, export_bracket_json, pretournament_baseline_path,
)
from bracket_schema import BracketValidationError, load_bracket_yaml  # noqa: E402
from hybrid_simulation import build_round_history  # noqa: E402
from live_scores import LiveScoresError  # noqa: E402
from simulate import N_SIMULATIONS  # noqa: E402

HISTORY_SIMULATIONS_DEFAULT = 2000


def consolidated_output_path(bracket_path, output_dir=OUTPUT_DIR):
    return Path(output_dir) / f"{Path(bracket_path).stem}_consolidated.json"


def _match_row(player_a, player_b, model_prob_a, decided, winner):
    return {
        "player_a": player_a, "player_b": player_b,
        "model_prob_a": round(model_prob_a, 3) if model_prob_a is not None else None,
        "decided": decided,
        "winner": winner,
    }


def _enrich_historical_matches(model_only_matches):
    rows = [
        _match_row(m["player_a"], m["player_b"], m["model_prob_a"], m["decided"], m["winner"])
        for m in model_only_matches
    ]
    rows.sort(key=lambda r: r["player_a"])
    return rows


def _build_current_matches(current_export, partial_model_only_matches):
    """The 'current' checkpoint's matches: every real match in the round currently being played,
    decided or not - not just the unsettled ones. current_export['matchups'] only ever contains
    STILL-unsettled matches (bracket_export.py explicitly excludes anything already decided - see
    its own matchups-loop comment) - already-decided matches within the round still in progress
    would be silently missing from 'current' entirely without partial_model_only_matches
    (hybrid_simulation's own _round_matches for this exact round, which covers the whole round,
    decided or not, and is where 'decided'/'winner' come from here)."""
    live_by_pair = {
        frozenset((info["slot_a"], info["slot_b"])): info for info in current_export["matchups"].values()
    }
    rows, seen_pairs = [], set()
    for m in partial_model_only_matches:
        pair = frozenset((m["player_a"], m["player_b"]))
        seen_pairs.add(pair)
        live = live_by_pair.get(pair)
        model_prob_a = live["p_slot_a"] if live is not None else m["model_prob_a"]
        rows.append(_match_row(m["player_a"], m["player_b"], model_prob_a, m["decided"], m["winner"]))

    # any live unsettled matchup NOT already covered above (e.g. a later round whose pairing is
    # also already fully known) - still surfaced, just appended rather than dropped.
    for pair, info in live_by_pair.items():
        if pair in seen_pairs:
            continue
        rows.append(_match_row(info["slot_a"], info["slot_b"], info["p_slot_a"], False, None))

    rows.sort(key=lambda r: r["player_a"])
    return rows


def build_consolidated_export(
    bracket_path, n_simulations=N_SIMULATIONS, seed=42, dates=None,
    history_simulations=HISTORY_SIMULATIONS_DEFAULT, overrides_path=DEFAULT_OVERRIDES_PATH,
    output_path=None, current_export=None, current_export_path=None,
):
    """current_export/current_export_path let a caller that already just ran export_bracket_json
    this cycle (live_match_watcher.py, once per detected transition) pass that result straight in
    instead of triggering a second, redundant live simulation here - export_bracket_json is the
    expensive part (a full Monte Carlo run), and this function has no independent reason to run it
    a second time in the same refresh."""
    bracket = load_bracket_yaml(bracket_path)

    if current_export is None:
        current_export_path, current_export = export_bracket_json(
            bracket_path, current_export_path, n_simulations, seed, dates=dates,
            overrides_path=overrides_path,
        )

    baseline_path = pretournament_baseline_path(bracket_path)
    baseline = None
    if baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)

    resolved_output_path = Path(output_path) if output_path is not None else consolidated_output_path(bracket_path)

    # the PREVIOUS consolidated export at this same output path already has its own round-checkpoint
    # entries - sourcing build_round_history's cache from just those (never the pretournament/current
    # entries mixed in below - see the "checkpoint" tag filter) means a fully-decided round's
    # snapshot gets computed exactly once, ever, and every later refresh just carries it forward.
    # A corrupt/unreadable/missing previous file just means everything gets recomputed this time,
    # same as the very first run for this bracket.
    previous_round_checkpoints = None
    if resolved_output_path.exists():
        try:
            with open(resolved_output_path, encoding="utf-8") as f:
                previous_entries = json.load(f).get("round_history") or []
            previous_round_checkpoints = [e for e in previous_entries if e.get("checkpoint") == "round"]
        except (json.JSONDecodeError, OSError):
            previous_round_checkpoints = None

    raw_history = build_round_history(
        bracket_path, n_simulations=history_simulations, seed=seed, dates=dates,
        previous_history=previous_round_checkpoints,
    )
    partial_entry = next((e for e in raw_history if e["partial"]), None)

    checkpoints = []

    if baseline is not None:
        checkpoints.append({
            "checkpoint": "pretournament",
            "round": 0,
            "label": "Pre-tournament",
            "generated_at": baseline["meta"]["generated_at"],
            "players": baseline["players"],
            "matches": [],
        })

    for entry in raw_history:
        if entry["partial"]:
            continue  # superseded by the "current" checkpoint built below
        checkpoints.append({
            "checkpoint": "round",
            "round": entry["round"],
            "label": entry["label"],
            "players": entry["players"],
            "matches": _enrich_historical_matches(entry["matches"]),
        })

    current_round_num = partial_entry["round"] if partial_entry else (raw_history[-1]["round"] if raw_history else 1)
    checkpoints.append({
        "checkpoint": "current",
        "round": current_round_num,
        "label": f"Current - {len(current_export['players'])} players alive",
        "generated_at": current_export["meta"]["generated_at"],
        "players": current_export["players"],
        "matches": _build_current_matches(
            current_export, partial_entry["matches"] if partial_entry else []
        ),
    })

    warnings = list(current_export["warnings"])
    if baseline is None:
        warnings.append(
            "No locked pre-tournament baseline exists yet for this bracket (see "
            "bracket_export.ensure_pretournament_baseline) - no 'pretournament' checkpoint below. "
            "Either this bracket already had real results the first time it was ever exported, or "
            "this run IS that first, baseline-defining export."
        )

    output = {
        "meta": {
            "tournament": current_export["meta"]["tournament"],
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "current_iterations": current_export["meta"]["iterations"],
            "current_seed": current_export["meta"]["seed"],
            "round_history_iterations": history_simulations,
            "pretournament_baseline_path": str(baseline_path) if baseline is not None else None,
        },
        "round_history": checkpoints,
        "head_to_head": current_export["head_to_head"],
        "health_adjustments": current_export["health_adjustments"],
        "warnings": warnings,
    }

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return resolved_output_path, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bracket_path", type=Path)
    parser.add_argument("--simulations", type=int, default=N_SIMULATIONS)
    parser.add_argument("--history-simulations", type=int, default=HISTORY_SIMULATIONS_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dates", default=None,
                         help="YYYYMMDD, passed to ESPN's ?dates= - needed for an already-"
                              "concluded event, which the undated (\"today\") scoreboard can't find")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--overrides", type=Path, default=DEFAULT_OVERRIDES_PATH,
        help="overrides YAML for manual health_adjustments - see bracket_export.py's own flag; "
             "pass a nonexistent path to skip")
    args = parser.parse_args()

    try:
        output_path, output = build_consolidated_export(
            args.bracket_path, args.simulations, args.seed, args.dates,
            history_simulations=args.history_simulations, overrides_path=args.overrides,
            output_path=args.output,
        )
    except (BracketValidationError, LiveScoresError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nWrote {output_path}")
    checkpoint_summary = ", ".join(
        f"{c['checkpoint']}({len(c['players'])}p/{len(c['matches'])}m)" for c in output["round_history"]
    )
    print(f"round_history: {checkpoint_summary}")
