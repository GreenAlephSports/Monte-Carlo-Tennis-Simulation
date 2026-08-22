"""Real, decided-match report: every completed match for a live tournament so far, grouped by
round, with the model's pre-match probability for that exact pairing - pulled from
calibration_log.csv if it's already logged there, computed fresh via win_probability() only for
whatever's decided but not yet logged.

results_by_round (from hybrid_simulation.build_real_results_by_round) already has exactly this
data - every completed match with a resolved winner, by round - but until now it was only ever
consumed internally by replay_real_rounds for checkpoint replay. This just prints it.

Usage:
    python model/research/real_results_report.py [bracket_path]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

import pandas as pd

from bracket_schema import load_bracket_yaml  # noqa: E402
from calibration_log import LOG_PATH, _match_key, _prepare_ratings  # noqa: E402
from hybrid_simulation import TOUR_SINGLES_CATEGORY, build_real_results_by_round  # noqa: E402
from live_scores import LiveScoresError, extract_matches, fetch_scoreboard  # noqa: E402
from win_probability import win_probability  # noqa: E402

DEFAULT_BRACKET = Path(__file__).resolve().parent.parent.parent / "brackets" / "cincinnati_2026_atp_demo.yaml"


def _load_log_lookup():
    """match_key -> {favorite, favorite_prob}, or {} if the log doesn't exist yet - match_key is
    already order-independent (sorted player pair, see calibration_log._pair_key), so a lookup
    with (player_a, player_b) in either order hits the same row."""
    if not LOG_PATH.exists():
        return {}
    log = pd.read_csv(LOG_PATH)
    return log.set_index("match_key")[["favorite", "favorite_prob"]].to_dict("index")


def run():
    bracket_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BRACKET
    bracket = load_bracket_yaml(bracket_path)
    tour_config, draw, _matches_history = _prepare_ratings(bracket)

    try:
        espn_data = fetch_scoreboard(bracket.tour.lower())
    except LiveScoresError as e:
        print(f"ERROR fetching live results: {e}", file=sys.stderr)
        sys.exit(1)
    espn_matches, _stats = extract_matches(espn_data)
    category = TOUR_SINGLES_CATEGORY[bracket.tour.lower()]
    tournament_matches = [
        m for m in espn_matches if m["tournament"] == bracket.tournament and m["category"] == category
    ]
    if not tournament_matches:
        print(f"ERROR: no live matches found for {bracket.tournament!r} / {category}", file=sys.stderr)
        sys.exit(1)

    results_by_round, round_sequence, unresolved = build_real_results_by_round(
        tournament_matches, draw, tour_config.name_aliases
    )
    if unresolved:
        print(f"NOTE: {len(unresolved)} ESPN name(s) unresolved, excluded: {sorted(unresolved)}",
              file=sys.stderr)
    round_num_to_label = {i + 1: label for i, label in enumerate(round_sequence)}

    log_lookup = _load_log_lookup()

    rows = []
    n_from_log = n_computed = 0
    for round_num in sorted(results_by_round):
        round_label = round_num_to_label.get(round_num, f"Round {round_num}")
        for pair, winner in results_by_round[round_num].items():
            player_a, player_b = tuple(pair)

            key = _match_key(bracket.tour, bracket.tournament, bracket.year, round_num, player_a, player_b)
            logged = log_lookup.get(key)
            if logged is not None:
                favorite, favorite_prob, source = logged["favorite"], float(logged["favorite_prob"]), "log"
                n_from_log += 1
            else:
                prob_a = win_probability(player_a, player_b, bracket.surface, tour_config.ratings_path)
                favorite = player_a if prob_a >= 0.5 else player_b
                favorite_prob = round(max(prob_a, 1 - prob_a), 4)
                source = "computed"
                n_computed += 1

            rows.append({
                "round_num": round_num, "round_label": round_label,
                "player_a": player_a, "player_b": player_b, "winner": winner,
                "favorite": favorite, "favorite_prob": favorite_prob,
                "favorite_won": favorite == winner, "source": source,
            })

    if not rows:
        print("No real, decided matches found yet for this tournament.")
        return

    df = pd.DataFrame(rows).sort_values(["round_num", "player_a"]).reset_index(drop=True)

    print(f"{bracket.tournament} {bracket.year} ({bracket.tour}) - {len(df)} real, decided match(es) "
          f"so far ({n_from_log} pulled from calibration_log.csv, {n_computed} computed fresh via "
          f"win_probability() - decided but not yet logged)\n")

    for round_num in sorted(df["round_num"].unique()):
        round_df = df[df["round_num"] == round_num]
        label = round_df["round_label"].iloc[0]
        n_upsets = (~round_df["favorite_won"]).sum()
        print(f"=== {label} ({len(round_df)} matches, {n_upsets} upset(s) vs. the model's own "
              f"favorite) ===")
        print(f"{'Player A':<24} {'Player B':<24} {'Winner':<20} {'Model favorite':<18} "
              f"{'Fav. prob':>10} {'Result':>8} {'Src':>9}")
        print("-" * 118)
        for _, r in round_df.iterrows():
            result = "correct" if r["favorite_won"] else "UPSET"
            print(f"{r['player_a']:<24} {r['player_b']:<24} {r['winner']:<20} {r['favorite']:<18} "
                  f"{r['favorite_prob']:>9.1%} {result:>8} {r['source']:>9}")
        print()


if __name__ == "__main__":
    run()
