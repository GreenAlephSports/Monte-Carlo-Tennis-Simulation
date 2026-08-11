import pdfplumber
import re
import yaml
import sys
from collections import defaultdict
from datetime import datetime

HEADER_DATE_RE = re.compile(
    r'(?P<start_day>\d{1,2})(?P<start_month>[A-Za-z]+)—'
    r'(?P<end_day>\d{1,2})(?P<end_month>[A-Za-z]+)(?P<year>\d{4})'
    r'\|.*\|(?P<surface>\w+)'
)

def extract_header(first_page, top_cutoff=55):
    """Pulls tournament name, location, start date, and surface from the top
    of page 1 - e.g. 'Omnium Banque Nationale...' / 'Montreal,Canada' /
    '2August—13August2026|USD9415724|Hard'. Same row-grouping trick as the
    player list, just restricted to the header region instead of the left column."""
    words = [w for w in first_page.extract_words() if w['top'] < top_cutoff]
    rows = defaultdict(list)
    for w in words:
        rows[round(w['top'])].append((w['x0'], w['text']))
    lines = [' '.join(t for _, t in sorted(rows[top], key=lambda x: x[0]))
             for top in sorted(rows.keys())]

    tournament = lines[0] if len(lines) > 0 else None
    location = lines[1] if len(lines) > 1 else None
    start_date, surface = None, None
    for line in lines:
        m = HEADER_DATE_RE.search(line)
        if m:
            dt = datetime.strptime(
                f"{m.group('start_day')} {m.group('start_month')} {m.group('year')}", '%d %B %Y'
            )
            start_date = dt.strftime('%Y-%m-%d')
            surface = m.group('surface')
            break
    return tournament, location, start_date, surface

def extract_left_column(page, x_cutoff=150, y_tolerance=3):
    words = [w for w in page.extract_words() if w['x0'] < x_cutoff]
    words.sort(key=lambda w: w['top'])
    rows, current_row, current_top = [], [], None
    for w in words:
        if current_top is None or abs(w['top'] - current_top) <= y_tolerance:
            current_row.append(w)
            current_top = w['top'] if current_top is None else current_top
        else:
            rows.append(current_row)
            current_row = [w]
            current_top = w['top']
    if current_row:
        rows.append(current_row)
    return [' '.join(w['text'] for w in sorted(row, key=lambda w: w['x0'])) for row in rows]

LINE_RE = re.compile(
    r'^(?P<position>\d+)\s*'
    r'(?:(?P<seed>\d+)\s+|(?P<status>Q|WC|PR|L)\s*)?'
    r'(?:Bye|(?P<lastname>[A-Z][A-Za-z\-\.\'…]*(?:\s[A-Z][A-Za-z\-\.\'…]*)*),\s*'
    r'(?P<firstname>[A-Za-z\-\.…]+)\s*(?P<country>[A-Z]{3})?)\s*$'
)

def parse_draw_pdf(path, expected_size=None):
    with pdfplumber.open(path) as pdf:
        lines = []
        for page in pdf.pages:
            lines.extend(extract_left_column(page))

    merged, buffer = [], ""
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\d+$', stripped):
            buffer = stripped
        elif buffer:
            merged.append(f"{buffer} {stripped}")
            buffer = ""
        else:
            merged.append(stripped)

    by_position = {}
    unparsed = []
    for line in merged:
        m = LINE_RE.match(line)
        if not m:
            unparsed.append(line)
            continue
        pos = int(m.group('position'))
        # keep first match per position - sidebar noise that happens to reuse a
        # low position number (e.g. the seed table) appears AFTER the real entry,
        # so first-seen wins
        if pos in by_position:
            continue
        by_position[pos] = {
            'position': pos,
            'seed': int(m.group('seed')) if m.group('seed') else None,
            'status': m.group('status'),
            'lastname': m.group('lastname'),
            'firstname': m.group('firstname').strip() if m.group('firstname') else None,
            'country': m.group('country'),
            'bye': m.group('lastname') is None,
        }

    max_pos = expected_size or max(by_position.keys())
    entries = [by_position[p] for p in range(1, max_pos + 1) if p in by_position]
    missing = [p for p in range(1, max_pos + 1) if p not in by_position]
    return entries, missing, unparsed

def to_yaml_players(entries):
    """Positions pair up (1,2), (3,4), (5,6)... A literal 'Bye' placeholder line
    means the OTHER player in that pair has a real bye (auto-advances to Round 2)
    - it does not mean there's a 32nd 'ghost' player. This pairs them up and
    outputs only the 96 real players, each correctly flagged."""
    by_position = {e['position']: e for e in entries}
    players = []
    positions = sorted(by_position.keys())
    for i in range(0, len(positions), 2):
        pos_a, pos_b = positions[i], positions[i + 1]
        entry_a, entry_b = by_position[pos_a], by_position[pos_b]

        if entry_a['bye'] and entry_b['bye']:
            raise ValueError(f"Both slots in pair ({pos_a},{pos_b}) are Bye placeholders - unexpected")

        if entry_a['bye'] or entry_b['bye']:
            # one real player + one empty placeholder -> the real one has a bye
            real = entry_b if entry_a['bye'] else entry_a
            name = f"{real['lastname'].title()} {real['firstname'][0]}."
            players.append({
                'position': real['position'], 'seed': real['seed'], 'name': name,
                'bye': True, 'status': real['status'],
            })
        else:
            # both real players - a genuine Round 1 match, output both
            for entry in (entry_a, entry_b):
                name = f"{entry['lastname'].title()} {entry['firstname'][0]}."
                players.append({
                    'position': entry['position'], 'seed': entry['seed'], 'name': name,
                    'bye': False, 'status': entry['status'],
                })
    return players

if __name__ == "__main__":
    with pdfplumber.open(sys.argv[1]) as pdf:
        tournament, location, start_date, surface = extract_header(pdf.pages[0])
    print(f"Header found: {tournament!r} | {location!r} | starts {start_date!r} | surface {surface!r}")
    if not all([tournament, location, start_date, surface]):
        print("WARNING: header extraction incomplete - check/fill these fields manually in the output YAML")

    entries, missing, unparsed = parse_draw_pdf(sys.argv[1], expected_size=128)
    print(f"Parsed {len(entries)}/128 expected positions")
    if missing:
        print(f"MISSING positions: {missing}")
    players = to_yaml_players(entries)

    year = int(start_date[:4]) if start_date else 2026
    output = {
        'tournament': tournament or 'PLACEHOLDER', 'year': year, 'tour': 'ATP',
        'surface': surface or 'PLACEHOLDER', 'start_date': start_date or 'PLACEHOLDER',
        'draw_size': len(players), 'players': players,
    }
    out_path = sys.argv[2] if len(sys.argv) > 2 else "parsed_draw.yaml"
    with open(out_path, 'w') as f:
        yaml.dump(output, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote {out_path}")