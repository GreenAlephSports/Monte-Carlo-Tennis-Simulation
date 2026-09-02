import urllib.request, json
data = json.loads(urllib.request.urlopen("https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard").read())
seen = set()
for e in data.get("events", []):
    if e.get("id") not in seen:
        seen.add(e.get("id"))
        print(e.get("id"), "-", e.get("name"))
