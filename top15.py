import json
for tour, file in [('ATP', 'us_open_2026_atp_real_bracket_export.json'), ('WTA', 'us_open_2026_wta_real_bracket_export.json')]:
    data = json.load(open(f'output/{file}'))
    ranked = sorted(data['players'], key=lambda p: p['p_champ'], reverse=True)[:15]
    print(f'--- {tour} Top 15 ---')
    for i, p in enumerate(ranked, 1):
        print(f"{i}. {p['player']}: {p['p_champ']*100:.1f}%")
    print()
