
import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_file

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket_export import OUTPUT_DIR  # noqa: E402
from live_scores import LiveScoresError, fetch_scoreboard  # noqa: E402

app = Flask(__name__)

# unset (the default) = no auth, every request served as before. Set (env var or .env) = every
# request must carry a matching X-API-Key header. Read once at import time, not per-request - this
# server isn't meant to have its key rotated while running; restart to change it.
API_KEY_ENV_VAR = "API_SERVER_API_KEY"
CONFIGURED_API_KEY = os.environ.get(API_KEY_ENV_VAR) or None


@app.before_request
def _check_api_key():
    if CONFIGURED_API_KEY is None:
        return None  # auth disabled - default, unchanged behavior
    if request.headers.get("X-API-Key") != CONFIGURED_API_KEY:
        return jsonify({
            "error": "unauthorized",
            "message": "Missing or invalid X-API-Key header.",
        }), 401
    return None


# the two filename conventions this project's export-writing scripts actually use - see module
# docstring. Order matters only for readability; "latest" is always resolved by mtime, not by
# this list's order.
EXPORT_SUFFIXES = ["_bracket_export.json", "_watcher_baseline.json"]

# tournament_id feeds directly into a filesystem path below - restricting it to the same charset
# bracket YAML stems actually use (letters/digits/underscore/hyphen) blocks path traversal
# ('../../etc') without needing to canonicalize-and-compare paths for every request.
VALID_TOURNAMENT_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _candidate_paths(tournament_id):
    return [OUTPUT_DIR / f"{tournament_id}{suffix}" for suffix in EXPORT_SUFFIXES]


def _discover_tournament_ids():
    ids = set()
    if not OUTPUT_DIR.exists():
        return []
    for path in OUTPUT_DIR.glob("*.json"):
        for suffix in EXPORT_SUFFIXES:
            if path.name.endswith(suffix):
                ids.add(path.name[: -len(suffix)])
                break
    return sorted(ids)


def _latest_export_path(tournament_id):
    """The most-recently-modified existing candidate file for this tournament_id, or None if
    neither convention has a file on disk yet."""
    existing = [p for p in _candidate_paths(tournament_id) if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _not_found_response(tournament_id):
    known = _discover_tournament_ids()
    return jsonify({
        "error": "tournament_not_found",
        "message": (
            f"No export file exists yet for tournament_id={tournament_id!r} - looked for "
            f"{[p.name for p in _candidate_paths(tournament_id)]} under {OUTPUT_DIR}. This server "
            f"only serves files already written by bracket_export.py or live_match_watcher.py; "
            f"run one of those for this bracket first, then retry."
        ),
        "known_tournament_ids": known,
    }), 404


@app.get("/tournaments")
def list_tournaments():
    """Summary list so a consumer can discover what's available without already knowing the
    output/ naming convention - one entry per tournament_id, describing whichever of its
    candidate files is currently newest."""
    entries = []
    for tournament_id in _discover_tournament_ids():
        path = _latest_export_path(tournament_id)
        entry = {
            "tournament_id": tournament_id,
            "source_file": path.name,
            "last_modified": path.stat().st_mtime,
        }
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("meta", {})
            entry["tournament"] = meta.get("tournament")
            entry["generated_at"] = meta.get("generated_at")
            entry["n_players"] = len(data.get("players", []))
        except (OSError, json.JSONDecodeError) as e:
            # a file that exists but can't be read (mid-write, corrupted) still gets listed - a
            # consumer polling this endpoint should see that the tournament_id exists and that its
            # current file is unreadable, not have it silently vanish from the list.
            entry["error"] = f"export file exists but could not be read: {e}"
        entries.append(entry)
    return jsonify({"tournaments": entries})


@app.get("/tournament/<tournament_id>")
def get_tournament(tournament_id):
    if not VALID_TOURNAMENT_ID.match(tournament_id):
        return jsonify({
            "error": "invalid_tournament_id",
            "message": f"tournament_id must match {VALID_TOURNAMENT_ID.pattern!r}, got {tournament_id!r}",
        }), 400

    path = _latest_export_path(tournament_id)
    if path is None:
        return _not_found_response(tournament_id)

    # send_file streams the file's bytes back unchanged (Daron's spec is untouched) and, for a
    # real filesystem path, sets the Last-Modified header from the file's own mtime and honors an
    # incoming If-Modified-Since with a 304 - exactly the "tell freshness without re-fetching the
    # body every poll" behavior a polling consumer wants, for free.
    return send_file(path, mimetype="application/json", conditional=True)


ARTIFACT_DATA_SUFFIX = "_artifact_data.json"


@app.get("/artifact_data/<tournament_id>")
def get_artifact_data(tournament_id):
    """Serves {tournament_id}_artifact_data.json (model/export_artifact_data.py's output - draw
    position order, every player's title odds, and every real match with model vs artifact-only
    market comparison) for the standalone 'Model vs Market Draw' artifact's own polling loop.
    Distinct from /tournament/<id> above (which serves the production *_bracket_export.json /
    *_watcher_baseline.json, deliberately market-free per Daron's 2026-09-04 request) - this file
    is written by a separate, on-demand script and never touched by the live automation path.
    CORS-open and unauthenticated for the same reason /live/<tour>/scoreboard is: a browser-side
    fetch() from the artifact page can't attach a custom header without CORS-preflight
    complications, and nothing served here is sensitive (same data already public on usopen.org)."""
    if not VALID_TOURNAMENT_ID.match(tournament_id):
        return jsonify({
            "error": "invalid_tournament_id",
            "message": f"tournament_id must match {VALID_TOURNAMENT_ID.pattern!r}, got {tournament_id!r}",
        }), 400

    path = OUTPUT_DIR / f"{tournament_id}{ARTIFACT_DATA_SUFFIX}"
    if not path.exists():
        return jsonify({
            "error": "tournament_not_found",
            "message": (
                f"No artifact data file exists yet for tournament_id={tournament_id!r} - looked for "
                f"{path.name} under {OUTPUT_DIR}. Run model/export_artifact_data.py for this bracket first."
            ),
        }), 404

    response = send_file(path, mimetype="application/json", conditional=True)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


# dates gets interpolated straight into the ESPN URL by live_scores.fetch_scoreboard - unlike
# tournament_id (used only as a local filesystem path segment), this value would otherwise flow
# untouched into an outbound request URL, so it's validated against ESPN's own documented shape
# (YYYYMMDD or YYYYMMDD-YYYYMMDD) before ever reaching that call.
VALID_SCOREBOARD_DATES = re.compile(r"^\d{8}(-\d{8})?$")


@app.get("/live/<tour>/scoreboard")
def live_scoreboard_proxy(tour):
    """Browser-side CORS proxy for ESPN's own live scoreboard endpoint (see live_scores.py's
    module docstring for the upstream URL) - added for the standalone live-status HTML artifact,
    which polls this directly from the browser with no dependency on live_match_watcher.py or any
    export file. Exists because a plain browser fetch() straight to ESPN was found to be blocked
    (see the artifact's own in-page diagnostic banner, which reports live whether direct fetch
    would have worked) - server-to-server requests from this project's own machine work fine (confirmed
    all tournament by live_scores.py itself), so this just re-serves that same successful request
    with permissive CORS headers attached, rather than reimplementing scoreboard-fetching logic.

    Read-only, and deliberately NOT gated behind the X-API-Key check above (a browser fetch() from
    a third-party artifact page can't attach a custom header to a simple cross-origin GET without
    triggering its own CORS preflight complications, and there's nothing sensitive being proxied -
    this mirrors ESPN's own public scoreboard, nothing from this project's own output/ directory)."""
    tour = tour.lower()
    if tour not in ("atp", "wta"):
        return jsonify({"error": "invalid_tour", "message": "tour must be 'atp' or 'wta'"}), 400

    dates = request.args.get("dates")
    if dates is not None and not VALID_SCOREBOARD_DATES.match(dates):
        return jsonify({
            "error": "invalid_dates",
            "message": f"dates must match {VALID_SCOREBOARD_DATES.pattern!r} (YYYYMMDD or YYYYMMDD-YYYYMMDD)",
        }), 400

    try:
        data = fetch_scoreboard(tour, dates=dates)
    except LiveScoresError as e:
        return jsonify({"error": "upstream_fetch_failed", "message": str(e)}), 502

    response = jsonify(data)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.get("/")
def health():
    return jsonify({
        "service": "bracket export polling API",
        "read_only": True,
        "auth_required": CONFIGURED_API_KEY is not None,
        "endpoints": [
            "/tournaments", "/tournament/<tournament_id>", "/live/<atp|wta>/scoreboard",
            "/artifact_data/<tournament_id>",
        ],
        "output_dir": str(OUTPUT_DIR),
    })


def _local_lan_ip():
    """Best-effort LAN IP for the startup banner only - opens a UDP socket toward a public
    address without sending any packet (UDP connect() is just a routing-table lookup) purely to
    ask the OS which local interface/IP would be used, so this works without needing real
    internet connectivity and doesn't contact 8.8.8.8 at all."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0",
                         help="default 0.0.0.0 - reachable from other devices on the same LAN, not the public internet")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if CONFIGURED_API_KEY:
        print(f"API key check: ENABLED ({API_KEY_ENV_VAR} is set) - every request must carry a matching X-API-Key header")
    else:
        print(f"API key check: disabled (set {API_KEY_ENV_VAR} to enable) - all requests served with no auth")

    if args.host in ("0.0.0.0", "::"):
        lan_ip = _local_lan_ip()
        print(f"Bound to {args.host}:{args.port} - reachable on this machine's LAN "
              f"(e.g. http://{lan_ip or '<this-machine-LAN-IP>'}:{args.port}), NOT the public internet "
              f"(no port-forwarding/external hosting is set up by this script)")

    # Flask's dev server already isolates a per-request exception (returns 500, keeps serving) -
    # the real unattended-operation risk is the SERVER PROCESS itself dying (an OS-level socket
    # error, a crash inside Werkzeug's own request loop). For a read-only poller meant to stay up
    # for the rest of the tournament with no one watching a terminal, restart rather than exit.
    crash_log_path = OUTPUT_DIR / "api_server_crashes.log"
    backoff_seconds = 15
    attempt = 0
    while True:
        attempt += 1
        try:
            app.run(host=args.host, port=args.port)
            break  # app.run() returning at all (not raising) means a clean shutdown - don't restart
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            message = (
                f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] api_server crashed "
                f"(attempt {attempt}) with {type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
            print(f"ERROR: {message}\nRestarting in {backoff_seconds}s...", file=sys.stderr)
            try:
                crash_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(crash_log_path, "a", encoding="utf-8") as f:
                    f.write(message + "\n")
            except OSError:
                pass
            time.sleep(backoff_seconds)
