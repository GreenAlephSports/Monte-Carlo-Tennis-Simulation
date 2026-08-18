"""Side-by-side comparison of bracket_export.py's market/blended probability against a
freshly-computed pure-Elo probability, for every matchup in an export JSON.

The export stores players under their ESPN display names (e.g. "Thiago Agustin Tirante"), but
win_probability() expects internal ratings-csv names (e.g. "Tirante T.A.") - so each slot name is
resolved via match_espn_name_to_draw (the same resolver bracket_export.py itself uses for live
ESPN matches), against the full pool of names in the tour's ratings csv, before calling
win_probability().

Usage:
    python model/compare_match.py [export.json] [--tour ATP|WTA] [--surface Hard|Clay|Grass] [--sort-by-gap]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import TOUR_CONFIG  # noqa: E402
from hybrid_simulation import match_espn_name_to_draw  # noqa: E402
from win_probability import _load_ratings, win_probability  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("export_path", nargs="?", default="output/cincinnati_2026_atp_demo_bracket_export.json")
parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
parser.add_argument("--surface", default="Hard", choices=["Hard", "Clay", "Grass"])
parser.add_argument("--sort-by-gap", action="store_true",
                     help="sort rows by |Market - Model|, largest disagreement first, instead of match order")
args = parser.parse_args()

tour_config = TOUR_CONFIG[args.tour]
draw_csv_names = set(_load_ratings(tour_config.ratings_path).index)

data = json.load(open(args.export_path))

rows = []
errors_shown = 0
for match_id, info in data["matchups"].items():
    a, b = info["slot_a"], info["slot_b"]
    market_p = info["p_slot_a"]

    csv_a = match_espn_name_to_draw(a, draw_csv_names, tour_config.name_aliases)
    csv_b = match_espn_name_to_draw(b, draw_csv_names, tour_config.name_aliases)

    gap = None
    try:
        if csv_a is None or csv_b is None:
            raise ValueError(f"unresolved name(s): {a if csv_a is None else ''} {b if csv_b is None else ''}".strip())
        model_p = win_probability(csv_a, csv_b, args.surface, ratings_path=tour_config.ratings_path)
        model_str = f"{model_p:.1%}"
        gap = market_p - model_p
        gap_str = f"{gap:+.1%}"
    except Exception as e:
        model_str = "N/A"
        gap_str = "N/A"
        if errors_shown < 3:
            print(f"  [ERROR on {a} vs {b}]: {e}")
            errors_shown += 1

    rows.append((match_id, a, b, market_p, model_str, gap, gap_str))

if args.sort_by_gap:
    rows.sort(key=lambda row: abs(row[5]) if row[5] is not None else -1, reverse=True)

print(f"{'Match ID':<10} {'Player A':<25} {'Player B':<25} {'Market/Blend':<15} {'Pure Model':<12} {'Gap':<8}")
print("-" * 103)
for match_id, a, b, market_p, model_str, gap, gap_str in rows:
    print(f"{match_id:<10} {a:<25} {b:<25} {market_p:.1%}{'':<10} {model_str:<12} {gap_str:<8}")
