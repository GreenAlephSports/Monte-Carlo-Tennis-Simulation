import urllib.request
import json
import time
import sys
from datetime import datetime

EVENT_ID = "718-2026"  # Cincinnati
POLL_SECONDS = 15

def fetch():
    url = f"https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard"
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())

print(f"Watching Cincinnati (event {EVENT_ID}) - polling every {POLL_SECONDS}s. Ctrl+C to stop.\n")

last_seen = {}
while True:
    now = datetime.now().strftime("%H:%M:%S")
    try:
        data = fetch()
    except Exception as e:
        print(f"{now} - FETCH ERROR: {e}")
        time.sleep(POLL_SECONDS)
        continue

    found_any = False
    for e in data.get("events", []):
        if e.get("id") != EVENT_ID and EVENT_ID not in str(e.get("$ref", "")):
            continue
        for g in e.get("groupings", []):
            for c in g.get("competitions", []):
                state = c.get("status", {}).get("type", {}).get("state")
                if state == "pre":
                    continue  # too noisy to print every scheduled match every poll
                found_any = True
                names = [comp.get("athlete", {}).get("displayName", "?") for comp in c.get("competitors", [])]
                sets = [comp.get("linescores", []) for comp in c.get("competitors", [])]
                key = c.get("id")
                snapshot = (state, str(sets))
                changed = last_seen.get(key) != snapshot
                marker = " <-- CHANGED" if changed and key in last_seen else (" <-- NEW" if key not in last_seen else "")
                print(f"{now} [{state}] {' vs '.join(names)} - {sets}{marker}")
                last_seen[key] = snapshot

    if not found_any:
        print(f"{now} - nothing in-progress or finished yet")

    time.sleep(POLL_SECONDS)