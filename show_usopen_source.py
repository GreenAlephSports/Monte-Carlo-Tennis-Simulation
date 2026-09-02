import urllib.request, json
url = "https://www.usopen.org/en_US/scores/feeds/2026/draws/MS.json"
data = json.loads(urllib.request.urlopen(url).read())
print(f"Loaded {len(data.get('matches', data))} entries from official usopen.org feed")
print(json.dumps(data, indent=2)[:500])
