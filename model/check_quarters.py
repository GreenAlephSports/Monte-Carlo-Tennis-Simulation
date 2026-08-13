import json
from collections import defaultdict
data = json.load(open('output/cincinnati_2026_atp_bracket_export.json'))
by_quarter = defaultdict(float)
for p in data['players']:
    by_quarter[p['quarter']] += p['p_sf']
for q, total in sorted(by_quarter.items()):
    print(f'{q}: {total:.4f}')