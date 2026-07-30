import random
from collections import Counter
from pathlib import Path

import pandas as pd

from bracket import DRAW, ROUND_NAMES, SURFACE, get_matchups, validate_draw
from win_probability import win_probability

N_SIMULATIONS = 10000
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output" / "wimbledon_2026_simulation_results.csv"


# plays out one full random bracket: each round, rng by weighted by win_probability to pick the winner of each matchup, then the winners become next round's field
def simulate_tournament(draw, surface):
    players = draw
    for _ in ROUND_NAMES:
        winners = []
        for player_a, player_b in get_matchups(players):
            prob_a = win_probability(player_a, player_b, surface)
            winners.append(player_a if random.random() < prob_a else player_b)
        players = winners
    return players[0]


def run_simulations(draw, surface, n_simulations):
    champion_counts = Counter()
    for _ in range(n_simulations):
        champion_counts[simulate_tournament(draw, surface)] += 1
    return champion_counts


if __name__ == "__main__":
    validate_draw(DRAW)

    champion_counts = run_simulations(DRAW, SURFACE, N_SIMULATIONS)

    results = pd.DataFrame({
        "player": DRAW,
        "win_count": [champion_counts.get(player, 0) for player in DRAW],
    })
    results["tournament_win_probability"] = results["win_count"] / N_SIMULATIONS
    results = results.sort_values("tournament_win_probability", ascending=False).reset_index(drop=True)

    results.to_csv(OUTPUT_PATH, index=False)
    print(f"Ran {N_SIMULATIONS} simulations, saved results to {OUTPUT_PATH}")

    print("\nTop 15 players by tournament-win probability:")
    print(results.head(15).to_string(index=False))
