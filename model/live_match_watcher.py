"""Polls live_scores.py for a tournament on a fixed interval, detects when any individual match
transitions to Final (comparing each match's own last-seen status against its current status - not
just "did the feed change at all", which would also fire on score updates within an in-progress
match, a new round's matchups being published, etc.), and on a real transition automatically
reruns bracket_export.py's live simulation and reports which players' p_champ/p_sf/p_final moved
as a direct result, sorted by magnitude of change.

bracket_export.py (not hybrid_simulation.py) is the "appropriate simulation" here: it already
reads live ESPN state itself and produces exactly the p_champ/p_sf/p_final numbers this report
needs (via run_simulations_tracking_milestones), whereas hybrid_simulation.py's CLI only writes a
champion-probability CSV per --through-round checkpoint, not the sf/final breakdown. The polling
loop itself stays cheap (a single scoreboard fetch, no simulation) - a full export only runs when a
transition is actually detected, plus once up front to establish the pre-transition baseline.

Usage:
    python model/live_match_watcher.py brackets/cincinnati_2026_atp_demo.yaml
    python model/live_match_watcher.py brackets/cincinnati_2026_atp_demo.yaml --interval 30
    python model/live_match_watcher.py brackets/cincinnati_2026_atp_demo.yaml --exit-after 1
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket_export import OUTPUT_DIR, export_bracket_json  # noqa: E402
from bracket_schema import BracketValidationError, load_bracket_yaml  # noqa: E402
from hybrid_simulation import TOUR_SINGLES_CATEGORY  # noqa: E402
from live_scores import LiveScoresError, extract_matches, fetch_scoreboard, format_match  # noqa: E402
from simulate import N_SIMULATIONS  # noqa: E402

DEFAULT_INTERVAL_SECONDS = 60
MILESTONE_FIELDS = ["p_champ", "p_sf", "p_final"]


def fetch_tournament_matches(tour, tournament_name, category, dates=None):
    """Same tournament/category filter export_bracket_json applies internally, exposed standalone
    so the poll loop can check match status without paying for a full simulation run each time."""
    data = fetch_scoreboard(tour.lower(), dates=dates)
    matches, _stats = extract_matches(data)
    return [m for m in matches if m["tournament"] == tournament_name and m["category"] == category]


def snapshot_statuses(matches):
    """match_id -> status_state ('pre' | 'in' | 'post'), keyed on ESPN's own competition id - the
    one field stable across polls for the same real-world match (names alone aren't a safe key:
    ESPN reuses 'TBD' for multiple different not-yet-determined slots at once)."""
    return {m["match_id"]: m["status_state"] for m in matches if m["match_id"]}


def detect_new_finals(previous_statuses, current_matches):
    """Returns the list of match dicts that transitioned into Final THIS poll - status_state is
    now 'post' with a decided winner, and either this match_id wasn't seen before or it wasn't
    already 'post' last time. Deliberately does not fire on any other kind of change (a score
    update mid-match, a new round's matchups appearing, a match moving pre -> in) - only the
    specific pre/in -> post transition this watcher exists to catch."""
    newly_final = []
    for m in current_matches:
        match_id = m["match_id"]
        if not match_id or m["status_state"] != "post" or not m["winner"]:
            continue
        if previous_statuses.get(match_id) != "post":
            newly_final.append(m)
    return newly_final


def players_by_espn_name(export_output):
    return {row["player"]: row for row in export_output["players"]}


def print_delta_report(completed_matches, before_players, after_players):
    print("\n" + "=" * 78)
    print(f"MATCH COMPLETED - {len(completed_matches)} match(es) this cycle:")
    for m in completed_matches:
        print(f"  {format_match(m)}")
    print("-" * 78)

    deltas = []
    all_names = set(before_players) | set(after_players)
    for name in all_names:
        before = before_players.get(name)
        after = after_players.get(name)
        # a player who just lost drops out of alive_draw_names entirely (export_bracket_json only
        # reports currently-alive players) - report that as an explicit elimination line, not a
        # silently-missing row, and skip it from the numeric delta ranking (no "after" value to
        # rank by, and 0 -> 0 for every field isn't the story - "eliminated" is).
        if after is None:
            deltas.append((name, None, before, "ELIMINATED"))
            continue
        if before is None:
            deltas.append((name, None, after, "NEW"))
            continue
        magnitude = sum(abs(after[f] - before[f]) for f in MILESTONE_FIELDS)
        if magnitude < 1e-9:
            continue
        deltas.append((name, magnitude, (before, after), None))

    ranked = sorted((d for d in deltas if d[3] is None), key=lambda d: -d[1])
    special = [d for d in deltas if d[3] is not None]

    if not ranked and not special:
        print("No detectable change in any player's p_champ/p_sf/p_final.")
    for name, _magnitude, (before, after), _tag in ranked:
        print(f"  {name:<28} "
              f"p_champ {before['p_champ']:.3f} -> {after['p_champ']:.3f} ({after['p_champ']-before['p_champ']:+.3f})  "
              f"p_sf {before['p_sf']:.3f} -> {after['p_sf']:.3f} ({after['p_sf']-before['p_sf']:+.3f})  "
              f"p_final {before['p_final']:.3f} -> {after['p_final']:.3f} ({after['p_final']-before['p_final']:+.3f})")
    for name, _magnitude, row, tag in sorted(special, key=lambda d: d[0]):
        print(f"  {name:<28} {tag} (last known: p_champ {row['p_champ']:.3f}, "
              f"p_sf {row['p_sf']:.3f}, p_final {row['p_final']:.3f})")
    print("=" * 78)


def watch(bracket_path, interval, n_simulations, seed, dates, exit_after):
    bracket = load_bracket_yaml(bracket_path)
    tour = bracket.tour.lower()
    category = TOUR_SINGLES_CATEGORY[tour]

    print(f"Watching {bracket.tournament} ({bracket.tour}) via bracket {bracket_path} - "
          f"polling every {interval}s")

    print("Running initial simulation to establish baseline p_champ/p_sf/p_final...")
    _out_path, baseline_export = export_bracket_json(
        bracket_path, output_path=OUTPUT_DIR / f"{Path(bracket_path).stem}_watcher_baseline.json",
        n_simulations=n_simulations, seed=seed, dates=dates,
    )
    current_players = players_by_espn_name(baseline_export)
    print(f"Baseline established: {len(current_players)} players alive.")

    try:
        current_matches = fetch_tournament_matches(bracket.tour, bracket.tournament, category, dates=dates)
    except LiveScoresError as e:
        print(f"ERROR fetching initial scoreboard: {e}", file=sys.stderr)
        sys.exit(1)
    last_statuses = snapshot_statuses(current_matches)
    n_already_final = sum(1 for s in last_statuses.values() if s == "post")
    print(f"{len(last_statuses)} matches tracked, {n_already_final} already Final at startup "
          f"(not treated as new completions) - watching for the next transition.\n")

    transitions_seen = 0
    while True:
        time.sleep(interval)
        try:
            current_matches = fetch_tournament_matches(bracket.tour, bracket.tournament, category, dates=dates)
        except LiveScoresError as e:
            print(f"WARNING: scoreboard fetch failed this cycle ({e}) - will retry next interval", file=sys.stderr)
            continue

        current_statuses = snapshot_statuses(current_matches)
        newly_final = detect_new_finals(last_statuses, current_matches)
        last_statuses = current_statuses

        if not newly_final:
            print(f"[{time.strftime('%H:%M:%S')}] no new completions ({len(current_matches)} matches tracked)")
            continue

        print(f"[{time.strftime('%H:%M:%S')}] detected {len(newly_final)} new completion(s) - rerunning simulation...")
        _out_path, new_export = export_bracket_json(
            bracket_path, output_path=OUTPUT_DIR / f"{Path(bracket_path).stem}_watcher_baseline.json",
            n_simulations=n_simulations, seed=seed, dates=dates,
        )
        new_players = players_by_espn_name(new_export)
        print_delta_report(newly_final, current_players, new_players)
        current_players = new_players

        transitions_seen += 1
        if exit_after is not None and transitions_seen >= exit_after:
            print(f"\nReached --exit-after {exit_after} transition(s), stopping.")
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bracket_path", type=Path)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS,
                         help=f"seconds between polls (default {DEFAULT_INTERVAL_SECONDS})")
    parser.add_argument("--simulations", type=int, default=N_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dates", default=None,
                         help="YYYYMMDD, passed to ESPN's ?dates= - only needed for a tournament "
                              "that's not 'today' by ESPN's own default")
    parser.add_argument("--exit-after", type=int, default=None,
                         help="stop after this many detected transitions (default: run forever)")
    args = parser.parse_args()

    try:
        watch(args.bracket_path, args.interval, args.simulations, args.seed, args.dates, args.exit_after)
    except (BracketValidationError, LiveScoresError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
