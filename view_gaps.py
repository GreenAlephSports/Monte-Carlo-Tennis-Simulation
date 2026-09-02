import json
data = json.load(open("output/us_open_2026_atp_real_consolidated.json"))
for m in data["match_model_vs_market"][:10]:
    print(m)
