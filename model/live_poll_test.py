import urllib.request
import json
import time
from datetime import datetime

last_score = None
while True:
    data = json.loads(urllib.request.urlopen("https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard").read())
    for e in data.get("events", []):
        for c in e.get("competitions", []):
            if c.get("status", {}).get("type", {}).get("state") == "in":
                sets = [comp.get("linescores", []) for comp in c.get("competitors", [])]
                score_snapshot = str(sets)
                if score_snapshot != last_score:
                    now = datetime.now().strftime("%H:%M:%S")
                    print(f"{now} - CHANGE: {score_snapshot}")
                    last_score = score_snapshot
    time.sleep(30)