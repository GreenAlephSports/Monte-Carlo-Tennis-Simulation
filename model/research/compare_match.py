"""Side-by-side comparison of bracket_export.py's market/blended probability against a
freshly-computed pure-Elo probability, for every matchup in an export JSON.

The export stores players under their ESPN display names (e.g. "Thiago Agustin Tirante"), but
win_probability() expects internal ratings-csv names (e.g. "Tirante T.A.") - so each slot name is
resolved via match_espn_name_to_draw (the same resolver bracket_export.py itself uses for live
ESPN matches), against the full pool of names in the tour's ratings csv, before calling
win_probability().

win_probability() is pregame-only - it has no awareness of an in-progress score. market_prob_a in
the export is whatever The Odds API returned at export time, which (confirmed empirically: see
live_scores.py cross-check) keeps updating with the live score once a match starts - it is NOT a
frozen pregame number. So a gap computed for a match that's already "In Progress" (or "Final") is
comparing a pregame model probability against a live/in-play or settled market price - not a
meaningful market/model disagreement, just the model being blind to the score. This script cross-
references each matchup against live_scores.py's real-time ESPN status (matched by exact ESPN
displayName, the same identity space the export already uses - no fuzzy matching needed) and
excludes any non-pregame match from the gap ranking, flagging it clearly instead of silently
dropping it.

Usage:
    python model/compare_match.py [export.json] [--tour ATP|WTA] [--surface Hard|Clay|Grass] [--sort-by-gap] [--dates YYYYMMDD]
    python model/compare_match.py --bracket brackets/wta_toronto_2026.yaml [--sort-by-gap] [--dates YYYYMMDD]

--bracket derives --tour/--surface (and, if export_path is omitted, the export path itself -
bracket_export.py's own default naming convention) straight from the bracket YAML, the same way
hybrid_simulation.py/bracket_export.py already take a bracket path instead of separate tour/surface
flags a caller has to keep in sync by hand - passing an ATP export alongside a left-at-default
--tour WTA (or vice versa) previously wasn't an error, just silently wrong ratings/name resolution.
--bracket wins over --tour/--surface whenever both are given, since it's the source of truth for a
real export; the plain flags still work standalone for an export with no bracket file handy.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import TOUR_CONFIG  # noqa: E402
from bracket_schema import load_bracket_yaml  # noqa: E402
from hybrid_simulation import match_espn_name_to_draw  # noqa: E402
from live_scores import LiveScoresError, extract_matches, fetch_scoreboard, filter_by_tour  # noqa: E402
from win_probability import _load_ratings, win_probability  # noqa: E402

DEFAULT_EXPORT_PATH = "output/cincinnati_2026_atp_demo_bracket_export.json"

parser = argparse.ArgumentParser()
parser.add_argument("export_path", nargs="?", default=None,
                     help=f"defaults to {DEFAULT_EXPORT_PATH!r}, or - if --bracket is given and this is "
                          f"omitted - output/<bracket stem>_bracket_export.json")
parser.add_argument("--bracket", type=Path, default=None,
                     help="a bracket YAML (any tour) - derives --tour/--surface from it, taking priority "
                          "over the flags below if both are given")
parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
parser.add_argument("--surface", default="Hard", choices=["Hard", "Clay", "Grass"])
parser.add_argument("--sort-by-gap", action="store_true",
                     help="sort rows by |Market - Model|, largest disagreement first, instead of match order "
                          "(non-pregame rows are always sorted last, regardless of this flag)")
parser.add_argument("--dates", default=None,
                     help="YYYYMMDD, passed to ESPN's ?dates= for the live-status lookup - needed if the "
                          "export is from a day other than today (matches espn_bracket.py's own convention)")
args = parser.parse_args()

if args.bracket is not None:
    bracket = load_bracket_yaml(args.bracket)
    args.tour, args.surface = bracket.tour, bracket.surface
    if args.export_path is None:
        args.export_path = f"output/{args.bracket.stem}_bracket_export.json"
    print(f"--bracket {args.bracket} -> tour={args.tour}, surface={args.surface}, "
          f"export_path={args.export_path}", file=sys.stderr)
elif args.export_path is None:
    args.export_path = DEFAULT_EXPORT_PATH

tour_config = TOUR_CONFIG[args.tour]
draw_csv_names = set(_load_ratings(tour_config.ratings_path).index)

data = json.load(open(args.export_path))

# --- live status lookup: frozenset({espn_name_a, espn_name_b}) -> 'pre' | 'in' | 'post' ---
# Same identity space the export's slot_a/slot_b already use (exact ESPN displayName), so this is
# a direct dict lookup, never fuzzy - matches bracket_export.py's own exact-match philosophy for
# cross-referencing ESPN data (see fetch_devigged_odds's docstring).
status_by_pair = {}
try:
    espn_data = fetch_scoreboard(args.tour.lower(), dates=args.dates)
    espn_matches, _stats = extract_matches(espn_data)
    for m in filter_by_tour(espn_matches, args.tour):
        status_by_pair[frozenset((m["player_1"], m["player_2"]))] = m["status_state"]
except LiveScoresError as e:
    print(f"WARNING: couldn't fetch live status ({e}) - all matches will be flagged 'unknown' and "
          f"excluded from the gap ranking rather than risk comparing a live price as if pregame", file=sys.stderr)

STATUS_LABEL = {"pre": "PREGAME", "in": "LIVE (excluded)", "post": "FINAL (excluded)", None: "unknown (excluded)"}

rows = []
errors_shown = 0
for match_id, info in data["matchups"].items():
    a, b = info["slot_a"], info["slot_b"]
    market_p = info["p_slot_a"]
    status_state = status_by_pair.get(frozenset((a, b)))
    is_pregame = status_state == "pre"
    status_label = STATUS_LABEL.get(status_state, "unknown (excluded)")

    csv_a = match_espn_name_to_draw(a, draw_csv_names, tour_config.name_aliases)
    csv_b = match_espn_name_to_draw(b, draw_csv_names, tour_config.name_aliases)

    gap = None
    try:
        if csv_a is None or csv_b is None:
            raise ValueError(f"unresolved name(s): {a if csv_a is None else ''} {b if csv_b is None else ''}".strip())
        model_p = win_probability(csv_a, csv_b, args.surface, ratings_path=tour_config.ratings_path)
        model_str = f"{model_p:.1%}"
        # gap is only meaningful pregame - a live/final/unknown-status match still gets model_p
        # shown (useful context), but its gap is never computed or eligible for the ranking, so a
        # score-contaminated market price can't masquerade as a model disagreement.
        if is_pregame:
            gap = market_p - model_p
            gap_str = f"{gap:+.1%}"
            # same formula as bracket_export.py's relative_change_pct: how far the model's
            # probability is from the market's, as a percentage OF the market's own probability -
            # not a percentage-point gap like Gap above. Pregame-only, same reasoning as Gap
            # itself: a live/final market price isn't a pregame number to be relative to.
            rel_pct = (model_p - market_p) / market_p * 100
            rel_pct_str = f"{rel_pct:+.1f}%"
        else:
            gap_str = "N/A"
            rel_pct_str = "N/A"
    except Exception as e:
        model_str = "N/A"
        gap_str = "N/A"
        rel_pct_str = "N/A"
        if errors_shown < 3:
            print(f"  [ERROR on {a} vs {b}]: {e}")
            errors_shown += 1

    # the raw market_p figure itself stays visible for every row (useful context, never hidden),
    # but for any non-pregame row it's whatever The Odds API returned at export time - possibly
    # long stale by the time this script runs (see module docstring). Self-label the figure, not
    # just the Status column, so it can't be misread as current on its own - applies to every
    # exclusion reason (LIVE, FINAL, unknown), not just one of them.
    market_str = f"{market_p:.1%}" if is_pregame else f"{market_p:.1%} (stale)"

    rows.append((match_id, a, b, market_str, model_str, gap, gap_str, rel_pct_str, status_label))

if args.sort_by_gap:
    # non-pregame rows (gap is None) always sort last, regardless of --sort-by-gap - excluding
    # them from the ranking, not just labeling them, is the point.
    rows.sort(key=lambda row: abs(row[5]) if row[5] is not None else -1, reverse=True)

excluded = sum(1 for row in rows if row[5] is None)
if excluded:
    print(f"NOTE: {excluded} of {len(rows)} matchup(s) excluded from the gap ranking (not pregame - "
          f"see Status column)\n")

print(f"{'Match ID':<10} {'Player A':<25} {'Player B':<25} {'Market/Blend':<17} {'Pure Model':<12} {'Gap':<8} {'Rel. %':<10} {'Status':<18}")
print("-" * 134)
for match_id, a, b, market_str, model_str, gap, gap_str, rel_pct_str, status_label in rows:
    print(f"{match_id:<10} {a:<25} {b:<25} {market_str:<17} {model_str:<12} {gap_str:<8} {rel_pct_str:<10} {status_label:<18}")
