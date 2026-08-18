import json
data = json.load(open('output/cincinnati_2026_atp_bracket_export.json'))
bad = [(k, v) for k, v in data['matchups'].items() if abs(v['p_slot_a'] + v['p_slot_b'] - 1.0) > 0.001]
print(f'{len(data["matchups"])} matchups checked, {len(bad)} violations:', bad)