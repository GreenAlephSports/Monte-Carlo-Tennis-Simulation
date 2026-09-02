"""Diffs two bracket_export.json snapshots' p_champ by player - "how has this player's title
equity changed since the tournament started", for the whole field at once rather than one player
at a time.

Both snapshots are already-produced bracket_export.py output (see that module's docstring for the
schema) - this script does no simulation of its own, it only reads two files and compares the
"players[].p_champ" each one already computed. Matching is by the exact ESPN displayName in
"player" (bracket_export.py's own contract: every player row is keyed by that name, byte-for-byte,
never our internal ratings-csv name) - so a name that resolves differently between the two runs
(rare, but possible if match_espn_name_to_draw's fuzzy tier picks a different candidate) would
silently fail to match; not observed in practice, but worth knowing if a player looks like they
vanished AND appeared under a slightly different string in the same report.

A player can appear in only one snapshot:
  - in baseline, not in current: they've been eliminated (bracket_export.py's players[] only ever
    lists players still alive) - current p_champ is correctly 0.0, not missing.
  - in current, not in baseline: not a real market mover, just something baseline literally
    couldn't see - most commonly a qualifier slot baseline still had as a 'TBD (Qualifier N)'
    placeholder (see baseline's own warnings[] for this), now resolved to the real player's name.
    baseline p_champ is treated as 0.0 for the same reason - there's nothing else it could mean -
    but pct_change is left as None (a 0 -> X% jump isn't a meaningful ratio) and flagged in notes
    rather than presented as an infinite percentage.

Usage:
    python model/research/title_odds_movement.py \\
        output/us_open_2026_atp_real_bracket_export.json \\
        output/us_open_2026_atp_real_bracket_export_current.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"


def _load_p_champ(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {p["player"]: p["p_champ"] for p in data["players"]}, data["meta"]


def build_comparison(baseline_path, current_path):
    baseline, baseline_meta = _load_p_champ(baseline_path)
    current, current_meta = _load_p_champ(current_path)

    rows = []
    for name in sorted(set(baseline) | set(current)):
        in_baseline, in_current = name in baseline, name in current
        baseline_p = baseline.get(name, 0.0)
        current_p = current.get(name, 0.0)
        diff = current_p - baseline_p

        if not in_current:
            note = "eliminated"
            pct = (diff / baseline_p * 100) if baseline_p else None
        elif not in_baseline:
            note = "not in baseline (unresolved/qualifier slot at snapshot time)"
            pct = None
        else:
            note = ""
            pct = (diff / baseline_p * 100) if baseline_p else None

        rows.append({
            "player": name,
            "p_champ_baseline": round(baseline_p, 4),
            "p_champ_current": round(current_p, 4),
            "diff": round(diff, 4),
            "pct_change": round(pct, 1) if pct is not None else None,
            "note": note,
        })

    rows.sort(key=lambda r: abs(r["diff"]), reverse=True)
    return rows, baseline_meta, current_meta


def _fmt_pct(v):
    return f"{v:+.1f}%" if v is not None else "n/a"


def print_table(rows, baseline_meta, current_meta, top_n):
    print(f"Baseline: {baseline_meta.get('generated_at')}  "
          f"({baseline_meta.get('iterations')} sims, seed={baseline_meta.get('seed')})")
    print(f"Current:  {current_meta.get('generated_at')}  "
          f"({current_meta.get('iterations')} sims, seed={current_meta.get('seed')})")
    print(f"{len(rows)} players total, showing top {min(top_n, len(rows))} movers by |diff|\n")

    header = f"{'player':<28}{'baseline':>10}{'current':>10}{'diff':>10}{'pct':>10}  note"
    print(header)
    print("-" * len(header))
    for r in rows[:top_n]:
        print(
            f"{r['player']:<28}{r['p_champ_baseline']:>10.1%}{r['p_champ_current']:>10.1%}"
            f"{r['diff']:>+10.1%}{_fmt_pct(r['pct_change']):>10}  {r['note']}"
        )


def write_csv(rows, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "player", "p_champ_baseline", "p_champ_current", "diff", "pct_change", "note",
        ])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path, help="earlier bracket_export.json (e.g. locked pre-tournament)")
    parser.add_argument("current", type=Path, help="later bracket_export.json (e.g. current live)")
    parser.add_argument("--top", type=int, default=25, help="how many biggest movers to print (default 25)")
    parser.add_argument("--output", type=Path, default=None, help="CSV output path (default: output/title_odds_movement.csv)")
    args = parser.parse_args()

    rows, baseline_meta, current_meta = build_comparison(args.baseline, args.current)
    print_table(rows, baseline_meta, current_meta, args.top)

    out_path = args.output or (OUTPUT_DIR / "title_odds_movement.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_path)
    print(f"\nWrote full comparison ({len(rows)} players) to {out_path}")
