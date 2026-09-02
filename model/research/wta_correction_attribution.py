"""Isolates which specific correction(s) in our stack (rank-gap, layoff, recent-form, Platt,
upset-boost) drive Muchova/Sabalenka DOWN and Gauff/Kostyuk UP when going from pure Sackmann hard
Elo to hard Elo + full methods (the wta_three_way_blend_comparison.py finding). Leave-one-out: start
from the full stack, remove ONE correction at a time, and see how far each player's p_champ moves
back toward the elo-only baseline - the correction whose removal moves a player the most is the one
responsible for that player's shift.

Uses REDUCED simulation count (5,000, not the usual 10,000) since this is 7 separate runs on top of
an already-loaded system tonight - a diagnostic sweep, not a final number. Same Sackmann hard-Elo
blend and Rybakina -165 override as the run this is following up on.

Usage:
    python model/research/wta_correction_attribution.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import TOUR_CONFIG, match_draw_to_ratings, order_by_draw_position, validate_bracket_structure, validate_draw  # noqa: E402
from bracket_schema import load_bracket_yaml  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from projected_draw_builder import DEFAULT_OVERRIDES_PATH, apply_health_adjustments, load_overrides  # noqa: E402
from simulate import run_simulations_tracking_all_rounds  # noqa: E402
from wta_sackmann_blend_comparison import calculate_elo_ratings_sackmann_blend  # noqa: E402

BRACKET_PATH = Path(__file__).resolve().parent.parent.parent / "brackets" / "us_open_2026_wta_real.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
N_SIM = 5000
RYBAKINA_OVERRIDE_ELO = -165.0
WATCH_PLAYERS = ["Muchova K.", "Sabalenka A.", "Gauff C.", "Kostyuk M.", "Osaka N.", "Andreeva M.", "Rybakina E."]

FULL = {"use_rank_adjustment": True, "use_confidence_calibration": True,
        "use_layoff_adjustment": True, "use_recent_form_adjustment": True}
ELO_ONLY = {"use_rank_adjustment": False, "use_confidence_calibration": False,
            "use_layoff_adjustment": False, "use_recent_form_adjustment": False}

VARIANTS = {
    "full (all methods, upset-boost on)": (FULL, True),
    "elo-only (no methods, upset-boost off)": (ELO_ONLY, False),
    "no rank-gap": ({**FULL, "use_rank_adjustment": False}, True),
    "no layoff": ({**FULL, "use_layoff_adjustment": False}, True),
    "no recent-form": ({**FULL, "use_recent_form_adjustment": False}, True),
    "no Platt calibration": ({**FULL, "use_confidence_calibration": False}, True),
    "no upset-boost": (FULL, False),
}


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    bracket = load_bracket_yaml(str(BRACKET_PATH))
    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    validate_bracket_structure(byes)
    tour_config = TOUR_CONFIG[bracket.tour]
    matches = load_matches_for_tour(bracket.tour)
    overrides = load_overrides(DEFAULT_OVERRIDES_PATH, bracket.tour)
    overrides["health_adjustments"] = dict(overrides["health_adjustments"])
    overrides["health_adjustments"]["Rybakina E."] = {
        **overrides["health_adjustments"]["Rybakina E."], "elo_penalty": RYBAKINA_OVERRIDE_ELO,
    }

    ratings_df = calculate_elo_ratings_sackmann_blend(matches, bracket.start_date, tour=bracket.tour)
    ratings_df, applied = apply_health_adjustments(ratings_df, overrides["health_adjustments"])
    print("Health adjustment(s): " + ", ".join(f"{a['player']} {a['elo_penalty']:+.0f}" for a in applied))

    draw, res, ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date)
    unmatched = [r for r in res if r["tier"] is None]
    if unmatched:
        raise RuntimeError(f"unmatched: {[r['name'] for r in unmatched]}")
    path = OUTPUT_DIR / "_correction_attribution_ratings.csv"
    ratings_df.to_csv(path, index=False)
    validate_draw(draw)

    results = {}
    for label, (kwargs, upset_boost) in VARIANTS.items():
        print(f"Running {N_SIM} sims: {label}...")
        depth_counts = run_simulations_tracking_all_rounds(
            draw, byes, bracket.surface, N_SIM, path, use_upset_boost=upset_boost, win_probability_kwargs=kwargs)
        champ = depth_counts[0]
        results[label] = {name: champ.get(name, 0) / N_SIM for name in draw}

    full_p = results["full (all methods, upset-boost on)"]
    elo_p = results["elo-only (no methods, upset-boost off)"]

    print(f"\n{'=' * 110}\nLeave-one-out attribution (p_champ, N={N_SIM})\n{'=' * 110}")
    header = f"{'player':<15}{'elo-only':>10}{'full':>10}"
    for label in VARIANTS:
        if label not in ("full (all methods, upset-boost on)", "elo-only (no methods, upset-boost off)"):
            header += f"{label:>22}"
    print(header)
    for player in WATCH_PLAYERS:
        row = f"{player:<15}{elo_p.get(player, 0):>10.1%}{full_p.get(player, 0):>10.1%}"
        for label in VARIANTS:
            if label in ("full (all methods, upset-boost on)", "elo-only (no methods, upset-boost off)"):
                continue
            row += f"{results[label].get(player, 0):>22.1%}"
        print(row)

    print(f"\n{'=' * 110}\nInterpretation: for each player, compare each 'no X' column to the 'full' column.\n"
          f"If removing X moves the player BACK toward their elo-only number, X is (part of) what's driving the shift.\n"
          f"If removing X barely changes anything, X isn't the cause.\n{'=' * 110}")
    for player in ["Muchova K.", "Sabalenka A.", "Gauff C.", "Kostyuk M."]:
        print(f"\n{player}: elo-only={elo_p.get(player, 0):.1%}  full={full_p.get(player, 0):.1%}  "
              f"(full-vs-elo shift = {full_p.get(player, 0) - elo_p.get(player, 0):+.1%})")
        for label in VARIANTS:
            if label in ("full (all methods, upset-boost on)", "elo-only (no methods, upset-boost off)"):
                continue
            v = results[label].get(player, 0)
            moved_back = v - full_p.get(player, 0)
            print(f"  {label:<25}: {v:.1%}  (removing this moved it {moved_back:+.1%} from full)")


if __name__ == "__main__":
    main()
