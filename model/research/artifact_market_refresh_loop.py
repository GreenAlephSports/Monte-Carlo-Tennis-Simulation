"""Standalone scheduled refresh loop for the "Model vs Market Draw" artifact's market data ONLY -
repeatedly calls fetch_artifact_market_odds.build_market_records() for one or more brackets on a
fixed interval and writes each bracket's output/{stem}_artifact_market.json.

Deliberately its own separate process, never imported by or wired into live_match_watcher.py: the
production automation path stays exactly as market-free as it's been since 2026-09-04 (see
bracket_export.py's module docstring), and this loop dying, crash-looping, or never being started
has zero effect on it. Its only consumer is model/export_artifact_data.py (via
fetch_artifact_market_odds's own output file), which already treats a missing/stale market file as
"no market data" rather than an error - and the artifact UI itself now surfaces that staleness
directly (the "market data last updated" indicator, driven by this file's own mtime), so a stalled
loop is visible on the page rather than silently serving quietly-rotting numbers.

Usage:
    python model/research/artifact_market_refresh_loop.py brackets/us_open_2026_atp_real.yaml brackets/us_open_2026_wta_real.yaml
    python model/research/artifact_market_refresh_loop.py brackets/us_open_2026_atp_real.yaml --interval 300
    python model/research/artifact_market_refresh_loop.py brackets/us_open_2026_atp_real.yaml --once   # single pass, no loop
"""
import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetch_artifact_market_odds import build_market_records  # noqa: E402
import json  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
CRASH_LOG_PATH = OUTPUT_DIR / "artifact_market_refresh_crashes.log"
DEFAULT_INTERVAL_SECONDS = 300  # 5 minutes - well within The Odds API's free-tier rate limit for a
# couple of brackets polled this infrequently, and frequent enough that "X minutes ago" on the
# artifact rarely reads much past the interval itself.


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_crash(message):
    print(f"ERROR: {message}", file=sys.stderr)
    try:
        CRASH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CRASH_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{_now_iso()}] {message}\n")
    except OSError:
        pass


def refresh_once(bracket_paths):
    """One pass over every bracket - each bracket's fetch/write is independent, so a failure (or a
    zero-pairing result) on one tour never blocks or clobbers another's file."""
    for bracket_path in bracket_paths:
        try:
            records = build_market_records(bracket_path)
            output_path = OUTPUT_DIR / f"{bracket_path.stem}_artifact_market.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            print(f"[{_now_iso()}] {bracket_path.stem}: wrote {output_path.name} - "
                  f"{len(records)} pairing(s) with a live market price")
        except Exception as e:
            _log_crash(
                f"refresh failed for {bracket_path} with {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bracket_paths", type=Path, nargs="+")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS,
                         help=f"seconds between refreshes (default {DEFAULT_INTERVAL_SECONDS})")
    parser.add_argument("--once", action="store_true", help="run a single pass and exit, no loop")
    args = parser.parse_args()

    if args.once:
        refresh_once(args.bracket_paths)
        sys.exit(0)

    print(f"Refreshing {[p.stem for p in args.bracket_paths]} every {args.interval}s "
          f"(Ctrl+C to stop) - independent of live_match_watcher.py, production pipeline untouched.")
    while True:
        refresh_once(args.bracket_paths)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
