"""Shows exactly how surface_mismatch_damping fix (elo_ratings._damp_surface_mismatch, live by
default) is affecting the current real 2026 US Open WTA field: per-player before/after hard_elo for
the damping fix specifically, and the current WTA top 15 by title probability (p_champ) from the
real bracket (brackets/us_open_2026_wta_real.yaml), annotated with each player's overall_elo,
hard_elo, and whether/how much the damping fix is adjusting them.

"Before" (undamped) is computed by monkeypatching elo_ratings.SURFACE_MISMATCH_DAMP_POINTS to 0
for a second ratings pass - 0 damp points means the threshold-floor clamp in _damp_surface_mismatch
still fires but pulls nothing back (damped_abs = max(abs_mismatch - 0, threshold) = abs_mismatch
whenever abs_mismatch > threshold), i.e. an exact no-op, reproducing the pre-fix surface_elo exactly.

Usage:
    python model/research/wta_top15_damping_audit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

import elo_ratings  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from projected_draw_builder import simulate_projected_draw  # noqa: E402

BRACKET_PATH = Path(__file__).resolve().parent.parent.parent / "brackets" / "us_open_2026_wta_real.yaml"
NAMED_PLAYERS = ["Osaka N.", "Andreeva M."]


def undamped_ratings(matches_history, cutoff_date, tour):
    original = elo_ratings.SURFACE_MISMATCH_DAMP_POINTS
    elo_ratings.SURFACE_MISMATCH_DAMP_POINTS = 0.0
    try:
        return calculate_elo_ratings(matches_history, cutoff_date, tour=tour)
    finally:
        elo_ratings.SURFACE_MISMATCH_DAMP_POINTS = original


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    import yaml
    bracket_raw = yaml.safe_load(BRACKET_PATH.read_text(encoding="utf-8"))
    tour = bracket_raw["tour"]
    start_date = bracket_raw["start_date"]

    matches = load_matches_for_tour(tour)
    damped_df = calculate_elo_ratings(matches, start_date, tour=tour).set_index("player")
    undamped_df = undamped_ratings(matches, start_date, tour=tour).set_index("player")

    print(f"\n{'=' * 100}\nPART 1: named-player before/after (hard_elo, since US Open is Hard)\n{'=' * 100}")
    for name in NAMED_PLAYERS:
        if name not in damped_df.index:
            print(f"  {name}: not found in ratings")
            continue
        overall = damped_df.loc[name, "overall_elo"]
        raw_hard = undamped_df.loc[name, "hard_elo"]
        damped_hard = damped_df.loc[name, "hard_elo"]
        mismatch_raw = raw_hard - overall
        mismatch_damped = damped_hard - overall
        direction = "ABOVE overall (specialist direction)" if mismatch_raw > 0 else "BELOW overall (weakness direction)"
        print(f"\n  {name}")
        print(f"    overall_elo             : {overall:.1f}")
        print(f"    hard_elo BEFORE damping  : {raw_hard:.1f}   (mismatch = {mismatch_raw:+.1f}, {direction})")
        print(f"    hard_elo AFTER damping   : {damped_hard:.1f}   (mismatch = {mismatch_damped:+.1f})")
        if abs(mismatch_raw) > 50.0:
            print(f"    -> damping IS firing: pulled hard_elo by {damped_hard - raw_hard:+.1f} Elo points")
        else:
            print(f"    -> damping NOT firing: |mismatch|={abs(mismatch_raw):.1f} is under the 50-point threshold, no change")

    print(f"\n{'=' * 100}\nPART 2: current WTA top 15 by real title probability (US Open bracket)\n{'=' * 100}")
    _, result = simulate_projected_draw(str(BRACKET_PATH))
    players_sorted = sorted(result["players"], key=lambda r: -r["p_champ"])[:15]

    print(f"{'rank':<5}{'player':<20}{'p_champ':>9}{'overall_elo':>13}{'hard_elo(dmp)':>15}{'hard_elo(raw)':>15}{'damping shift':>15}{'direction':>28}")
    for i, row in enumerate(players_sorted, 1):
        name = row["player"]
        if name not in damped_df.index:
            print(f"{i:<5}{name:<20}{row['p_champ']:>9.1%}  (not in ratings - Tier-3 placeholder?)")
            continue
        overall = damped_df.loc[name, "overall_elo"]
        raw_hard = undamped_df.loc[name, "hard_elo"]
        damped_hard = damped_df.loc[name, "hard_elo"]
        mismatch_raw = raw_hard - overall
        shift = damped_hard - raw_hard
        if abs(mismatch_raw) <= 50.0:
            direction = "no damping (under threshold)"
        elif mismatch_raw > 0:
            direction = "specialist direction (pulled DOWN)"
        else:
            direction = "weakness direction (pulled UP)"
        print(f"{i:<5}{name:<20}{row['p_champ']:>9.1%}{overall:>13.1f}{damped_hard:>15.1f}{raw_hard:>15.1f}{shift:>+15.1f}{direction:>28}")


if __name__ == "__main__":
    main()
