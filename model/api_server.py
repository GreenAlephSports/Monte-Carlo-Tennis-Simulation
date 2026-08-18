"""Lightweight, read-only polling API that serves whatever bracket_export.py (directly, or via
live_match_watcher.py's baseline re-export) most recently wrote to output/*.json - it never
triggers a simulation itself, just reads a file off disk on request. Regeneration stays a
separate process (run bracket_export.py or live_match_watcher.py, as today).

Endpoints:
    GET /tournaments              - list every tournament this server currently has an export for
    GET /tournament/<tournament_id> - the latest export JSON for that tournament, byte-identical
                                       to what bracket_export.py wrote (players/matchups/
                                       head_to_head/meta, Daron's spec, untouched) - freshness is
                                       already in that body via meta.generated_at, and also exposed
                                       as a standard HTTP Last-Modified header (from the file's own
                                       mtime) so a polling consumer can send If-Modified-Since and
                                       get a cheap 304 instead of re-downloading an unchanged export.

tournament_id is the bracket YAML's stem (e.g. 'cincinnati_2026_atp' for
brackets/cincinnati_2026_atp.yaml) - the same name bracket_export.py and live_match_watcher.py
already derive their output filenames from, so nothing new needs to be invented or configured.
A tournament can have up to two candidate files on disk (bracket_export.py's own
'{stem}_bracket_export.json' and live_match_watcher.py's '{stem}_watcher_baseline.json') - "latest"
means whichever of the two that exist has the newer filesystem mtime, i.e. whichever process ran
most recently for that tournament.

Binds to 0.0.0.0 by default - reachable from other devices on the same local network (useful for
testing from a phone/another machine), NOT the public internet. Nothing here does port-forwarding
or exposes this beyond the LAN; that would be a separate, deliberate decision, not a side effect of
this default.

Optional API key check, off by default: set API_SERVER_API_KEY (env var or .env, same loader
bracket_export.py already uses) to require every request to carry a matching 'X-API-Key' header.
Leave it unset for continued local/LAN testing with no auth friction - it exists now so a real key
can be turned on later, the moment this ever needs to be exposed more broadly, without a code
change at that point.

Usage:
    python model/api_server.py                    # http://0.0.0.0:8000 - reachable on the LAN
    python model/api_server.py --host 127.0.0.1    # revert to localhost-only
    python model/api_server.py --port 8080
    API_SERVER_API_KEY=some-secret python model/api_server.py   # require X-API-Key on every request
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_file

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bracket_export import OUTPUT_DIR  # noqa: E402

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


@app.get("/")
def health():
    return jsonify({
        "service": "bracket export polling API",
        "read_only": True,
        "auth_required": CONFIGURED_API_KEY is not None,
        "endpoints": ["/tournaments", "/tournament/<tournament_id>"],
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

    app.run(host=args.host, port=args.port)
