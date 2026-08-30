import json
for tour, file in [("ATP", "us_open_2026_atp_real_bracket_export.json"), ("WTA", "us_open_2026_wta_real_bracket_export.json")]:
    data = json.load(open(f"output/{file}"))
    print(f"--- {tour} Round 1: Model vs Market (sorted by gap) ---")
    rows = []
    for match_id, info in data["matchups"].items():
        if not match_id.startswith(("T-R1", "B-R1")):
            continue
        a, b = info["slot_a"], info["slot_b"]
        model_p = info.get("model_prob_a")
        market_p = info.get("market_prob_a")
        if model_p is None or market_p is None:
            continue
        gap = (model_p - market_p) * 100
        rel_pct = ((model_p - market_p) / market_p * 100) if market_p else None
        rows.append((abs(gap), match_id, a, b, model_p, market_p, gap, rel_pct))
    rows.sort(key=lambda r: r[0], reverse=True)
    for _, match_id, a, b, model_p, market_p, gap, rel_pct in rows:
        rel_str = f"{rel_pct:+.1f}%" if rel_pct is not None else "N/A"
        print(f"{match_id}: {a} vs {b} | Model: {model_p*100:.1f}%  Market: {market_p*100:.1f}%  Gap: {gap:+.1f}pp  Rel: {rel_str}")
    print()
