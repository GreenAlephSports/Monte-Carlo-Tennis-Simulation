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
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket import DuplicatePlayerDrawError  # noqa: E402
from bracket_export import OUTPUT_DIR, QUALIFIER_PLACEHOLDER_RE, export_bracket_json  # noqa: E402
from bracket_schema import BracketValidationError, load_bracket_yaml  # noqa: E402
from hybrid_simulation import TOUR_SINGLES_CATEGORY  # noqa: E402
from live_scores import LiveScoresError, ended_by_retirement, extract_matches, fetch_scoreboard, format_match  # noqa: E402
from simulate import N_SIMULATIONS  # noqa: E402

DEFAULT_INTERVAL_SECONDS = 60
MILESTONE_FIELDS = ["p_champ", "p_sf", "p_final"]

# The Odds API drops an event entirely once it's Final (confirmed in compare_match.py's own
# investigation), and calibration_log.py only logs a match AFTER it's decided - so there is no way
# to fetch a pregame market price for a match after the fact. This cache is the fix: every poll
# here already runs export_bracket_json, which already fetches market_prob_a for every unsettled
# matchup as a side effect (see bracket_export.resolve_probability) - this just persists that
# price, keyed by ESPN name pair, while it's still available and still genuinely pregame, so
# calibration_log.py can look it up later once the match concludes. See update_market_price_cache.
MARKET_PRICE_CACHE_PATH = OUTPUT_DIR / "market_price_cache.json"


def _market_price_cache_key(tour, tournament, year, player_a, player_b):
    return f"{tour}|{tournament}|{year}|{'::'.join(sorted((player_a, player_b)))}"


def load_market_price_cache():
    if not MARKET_PRICE_CACHE_PATH.exists():
        return {}
    return json.loads(MARKET_PRICE_CACHE_PATH.read_text(encoding="utf-8"))


def update_market_price_cache(bracket, export_output, current_matches):
    """Snapshots market_prob_a for every currently-pregame matchup export_bracket_json just
    returned. Only caches a matchup confirmed 'pre' status_state at THIS poll (not 'in') - a price
    fetched while a match is already live/in-progress isn't a genuine pregame number (same
    exclusion logic compare_match.py already applies for the same reason). Skips anything already
    cached under the same key so an early, more-pregame snapshot is never overwritten by a later
    poll's (still valid, but no more informative) re-confirmation - the first pregame price seen is
    the one worth keeping. Returns how many new prices were cached this call."""
    status_by_pair = {
        frozenset((m["player_1"], m["player_2"])): m["status_state"] for m in current_matches
    }
    cache = load_market_price_cache()
    added = 0
    for info in export_output["matchups"].values():
        market_prob_a = info.get("market_prob_a")
        if market_prob_a is None:
            continue
        slot_a, slot_b = info["slot_a"], info["slot_b"]
        if status_by_pair.get(frozenset((slot_a, slot_b))) != "pre":
            continue
        key = _market_price_cache_key(bracket.tour, bracket.tournament, bracket.year, slot_a, slot_b)
        if key in cache:
            continue
        cache[key] = {
            "player_a": slot_a, "player_b": slot_b, "market_prob_a": market_prob_a,
            "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        added += 1
    if added:
        MARKET_PRICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        MARKET_PRICE_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return added


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


def check_no_tbd_placeholders(bracket, bracket_path):
    """Fails fast, before any polling starts, if the bracket YAML still has literal
    'TBD (Qualifier N)' entries - a bracket built before qualifying wrapped up (see
    espn_bracket.py's module docstring) that was never regenerated once real names were known.

    This exists because a stale placeholder doesn't fail HERE - bracket_export.py happily accepts
    it (tier-3 STARTING_ELO fallback in match_draw_to_ratings, or resolve_qualifier_placeholders
    leaving it unresolved) and the watcher can poll cleanly for a while, only for the placeholder
    to eventually blow up deep in simulation once something touches it (either validate_draw's
    duplicate check if the same real player also appears elsewhere in a stale draw, or a raw
    ValueError from win_probability.py's ratings-CSV lookup if the placeholder itself gets fed
    into a simulated matchup) - both hit in practice, on different bracket files, in the same
    week. Catching it at startup means a bad bracket file fails immediately and obviously, instead
    of hours into a poll."""
    placeholders = [p.name for p in bracket.players if QUALIFIER_PLACEHOLDER_RE.match(p.name)]
    if placeholders:
        raise RuntimeError(
            f"{bracket_path} still has {len(placeholders)} unresolved 'TBD (Qualifier N)' "
            f"placeholder(s) ({sorted(placeholders)}) - this bracket was likely built before "
            f"qualifying wrapped up and was never regenerated. Regenerate it against live ESPN "
            f"data first: python model/espn_bracket.py {bracket_path} --tour {bracket.tour.lower()} "
            f"--event-id <ESPN event id> --surface {bracket.surface}"
        )


def players_by_espn_name(export_output):
    return {row["player"]: row for row in export_output["players"]}


def print_delta_report(completed_matches, before_players, after_players):
    print("\n" + "=" * 78)
    print(f"MATCH COMPLETED - {len(completed_matches)} match(es) this cycle:")
    n_retirements = 0
    for m in completed_matches:
        if ended_by_retirement(m):
            n_retirements += 1
            print(f"  {format_match(m)}  [RETIREMENT/WALKOVER - flagged for calibration_log.py]")
        else:
            print(f"  {format_match(m)}")
    if n_retirements:
        print(f"  ({n_retirements} of {len(completed_matches)} this cycle ended via retirement/walkover - "
              f"run calibration_log.py to persist this alongside the calibration data)")
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
    check_no_tbd_placeholders(bracket, bracket_path)
    tour = bracket.tour.lower()
    category = TOUR_SINGLES_CATEGORY[tour]

    print(f"Watching {bracket.tournament} ({bracket.tour}) via bracket {bracket_path} - "
          f"polling every {interval}s")

    print("Running initial simulation to establish baseline p_champ/p_sf/p_final...")
    baseline_export = None
    while baseline_export is None:
        try:
            _out_path, baseline_export = export_bracket_json(
                bracket_path, output_path=OUTPUT_DIR / f"{Path(bracket_path).stem}_watcher_baseline.json",
                n_simulations=n_simulations, seed=seed, dates=dates,
            )
        except DuplicatePlayerDrawError as e:
            print(
                f"WARNING: {bracket.tournament} ({bracket.tour}) bracket has duplicate player(s) "
                f"{e.duplicate_names} - likely a stale bracket YAML (e.g. a qualifier slot that "
                f"resolved to a player who already has a separate literal entry elsewhere in the "
                f"draw). Can't establish a baseline until this is fixed (regenerate the bracket "
                f"YAML with espn_bracket.py). Retrying in {interval}s in case the underlying data "
                f"self-corrects.", file=sys.stderr,
            )
            time.sleep(interval)
    current_players = players_by_espn_name(baseline_export)
    print(f"Baseline established: {len(current_players)} players alive.")

    try:
        current_matches = fetch_tournament_matches(bracket.tour, bracket.tournament, category, dates=dates)
    except LiveScoresError as e:
        print(f"ERROR fetching initial scoreboard: {e}", file=sys.stderr)
        sys.exit(1)
    n_cached = update_market_price_cache(bracket, baseline_export, current_matches)
    if n_cached:
        print(f"Cached {n_cached} new pregame market price(s) to {MARKET_PRICE_CACHE_PATH} "
              f"for calibration_log.py to backfill once these matches conclude.")
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
        try:
            _out_path, new_export = export_bracket_json(
                bracket_path, output_path=OUTPUT_DIR / f"{Path(bracket_path).stem}_watcher_baseline.json",
                n_simulations=n_simulations, seed=seed, dates=dates,
            )
        except DuplicatePlayerDrawError as e:
            print(
                f"WARNING: {bracket.tournament} ({bracket.tour}) bracket has duplicate player(s) "
                f"{e.duplicate_names} - likely a stale bracket YAML. Skipping this cycle's delta "
                f"report (the {len(newly_final)} completion(s) just detected are NOT reflected in "
                f"current_players) and continuing to poll - fix the bracket YAML (regenerate with "
                f"espn_bracket.py) for the report to resume.", file=sys.stderr,
            )
            continue
        n_cached = update_market_price_cache(bracket, new_export, current_matches)
        if n_cached:
            print(f"Cached {n_cached} new pregame market price(s) to {MARKET_PRICE_CACHE_PATH}.")
        new_players = players_by_espn_name(new_export)
        print_delta_report(newly_final, current_players, new_players)
        current_players = new_players

        transitions_seen += 1
        if exit_after is not None and transitions_seen >= exit_after:
            print(f"\nReached --exit-after {exit_after} transition(s), stopping.")
            return


def main():
    # player/tournament names routinely carry non-ASCII characters (e.g. 'Krunić') and Windows'
    # default console codepage (cp1252) can't encode many of them - without this, a plain print()
    # of such a name crashes the whole watcher process. UTF-8 covers every name this pipeline
    # sees; reconfigure() is a no-op on streams that don't support it (e.g. when captured by a
    # test harness), so this is safe everywhere.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

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
