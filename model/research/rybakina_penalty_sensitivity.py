"""How large would Rybakina's manual health Elo penalty need to be for her title probability to
land ~5 percentage points BELOW Osaka's and Andreeva's (the user's real-world read of current book
prices)? This is NOT a validated correction - it's a reverse-engineering exercise to see what
magnitude of penalty that market gap actually implies, using a reduced simulation count (3,000/run,
not the usual 10,000) since this is a rough calibration sweep across many candidate values, not a
final number.

Usage:
    python model/research/rybakina_penalty_sensitivity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import TOUR_CONFIG, match_draw_to_ratings, order_by_draw_position, validate_bracket_structure, validate_draw  # noqa: E402
from bracket_schema import load_bracket_yaml  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from simulate import run_simulations_tracking_all_rounds  # noqa: E402

BRACKET_PATH = Path(__file__).resolve().parent.parent.parent / "brackets" / "us_open_2026_wta_real.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
N_SIM = 3000
TARGET_PLAYER = "Rybakina E."
COMPARE_TO = ["Osaka N.", "Andreeva M."]
CANDIDATE_PENALTIES = [-100, -150, -200, -250, -300, -400, -500, -650, -800]


def run_one(penalty, players, byes, tour_config, bracket, matches):
    ratings_df = calculate_elo_ratings(matches, bracket.start_date, tour=bracket.tour)
    idx = ratings_df.index[ratings_df["player"] == TARGET_PLAYER]
    if len(idx) == 0:
        raise RuntimeError(f"{TARGET_PLAYER} not found in ratings")
    for col in ("overall_elo", "hard_elo", "clay_elo", "grass_elo"):
        ratings_df.loc[idx, col] += penalty

    draw, res, ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date)
    unmatched = [r for r in res if r["tier"] is None]
    if unmatched:
        raise RuntimeError(f"unmatched: {[r['name'] for r in unmatched]}")
    path = OUTPUT_DIR / "_rybakina_sensitivity_ratings.csv"
    ratings_df.to_csv(path, index=False)
    validate_draw(draw)

    depth_counts = run_simulations_tracking_all_rounds(draw, byes, bracket.surface, N_SIM, path, use_upset_boost=True)
    champ = depth_counts[0]
    return {name: champ.get(name, 0) / N_SIM for name in draw}


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

    print(f"{'penalty':>9}{'Rybakina':>10}" + "".join(f"{n:>14}" for n in COMPARE_TO) + f"{'gap-to-target':>16}")
    baseline = None
    for penalty in CANDIDATE_PENALTIES:
        p = run_one(penalty, players, byes, tour_config, bracket, matches)
        ryb = p.get(TARGET_PLAYER, 0.0)
        comps = [p.get(n, 0.0) for n in COMPARE_TO]
        avg_comp = sum(comps) / len(comps)
        gap = ryb - (avg_comp - 0.05)  # target: 5pp BELOW the Osaka/Andreeva average
        print(f"{penalty:>9}{ryb:>10.1%}" + "".join(f"{c:>14.1%}" for c in comps) + f"{gap:>+16.1%}")


if __name__ == "__main__":
    main()
