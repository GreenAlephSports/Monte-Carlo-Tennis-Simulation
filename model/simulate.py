import random
from collections import Counter
from pathlib import Path

import pandas as pd

from bracket import ATP_RATINGS_PATH, DRAW, ROUND_NAMES, SURFACE, WTA_DRAW, WTA_RATINGS_PATH, get_matchups, validate_draw
from win_probability import win_probability

N_SIMULATIONS = 1000
ATP_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output" / "wimbledon_2026_simulation_results.csv"
WTA_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output" / "wimbledon_2026_simulation_results_wta.csv"


# plays out one full random bracket: each round, rng by weighted by win_probability to pick the winner of each matchup, then the winners become next round's field
def simulate_tournament(draw, surface, ratings_path):
    players = draw
    for _ in ROUND_NAMES:
        winners = []
        for player_a, player_b in get_matchups(players):
            prob_a = win_probability(player_a, player_b, surface, ratings_path)
            winners.append(player_a if random.random() < prob_a else player_b)
        players = winners
    return players[0]


def run_simulations(draw, surface, n_simulations, ratings_path):
    champion_counts = Counter()
    for _ in range(n_simulations):
        champion_counts[simulate_tournament(draw, surface, ratings_path)] += 1
    return champion_counts


def simulate_and_report(tour_name, draw, surface, ratings_path, output_path):
    validate_draw(draw)

    champion_counts = run_simulations(draw, surface, N_SIMULATIONS, ratings_path)
    results = pd.DataFrame({
        "player": draw,
        "win_count": [champion_counts.get(player, 0) for player in draw],
    })
    results["tournament_win_probability"] = results["win_count"] / N_SIMULATIONS
    results = results.sort_values("tournament_win_probability", ascending=False).reset_index(drop=True)

    results.to_csv(output_path, index=False)
    print(f"Ran {N_SIMULATIONS} simulations for {tour_name}, saved results to {output_path}")

    print(f"\nTop 15 {tour_name} players by tournament-win probability:")
    print(results.head(15).to_string(index=False))


if __name__ == "__main__":
    simulate_and_report("ATP", DRAW, SURFACE, ATP_RATINGS_PATH, ATP_OUTPUT_PATH)
    print()
    simulate_and_report("WTA", WTA_DRAW, SURFACE, WTA_RATINGS_PATH, WTA_OUTPUT_PATH)
