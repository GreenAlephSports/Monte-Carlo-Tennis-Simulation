import random
from collections import Counter

import pandas as pd

from bracket import get_matchups, validate_draw
from win_probability import win_probability

N_SIMULATIONS = 10000


def _play_round(players, surface, ratings_path):
    winners = []
    for player_a, player_b in get_matchups(players):
        prob_a = win_probability(player_a, player_b, surface, ratings_path)
        winners.append(player_a if random.random() < prob_a else player_b)
    return winners


# plays out one full random bracket. Round 1 only pairs up non-bye players (rng by weighted by
# win_probability to pick the winner of each matchup); those winners are then combined with the
# bye players - who skipped Round 1 entirely - to form the Round 2 field, and play continues the
# same way each round until one player remains. With no byes, non_bye_players is the whole draw
# and bye_players is empty, so this reduces to the plain single-elimination case.
def simulate_tournament(non_bye_players, bye_players, surface, ratings_path):
    players = _play_round(non_bye_players, surface, ratings_path) + bye_players
    while len(players) > 1:
        players = _play_round(players, surface, ratings_path)
    return players[0]


def run_simulations(non_bye_players, bye_players, surface, n_simulations, ratings_path):
    champion_counts = Counter()
    for _ in range(n_simulations):
        champion_counts[simulate_tournament(non_bye_players, bye_players, surface, ratings_path)] += 1
    return champion_counts


def simulate_and_report(tour_name, draw, non_bye_players, bye_players, surface, ratings_path, output_path,
                         n_simulations=N_SIMULATIONS):
    validate_draw(draw)

    champion_counts = run_simulations(non_bye_players, bye_players, surface, n_simulations, ratings_path)
    results = pd.DataFrame({
        "player": draw,
        "win_count": [champion_counts.get(player, 0) for player in draw],
    })
    results["tournament_win_probability"] = results["win_count"] / n_simulations
    results = results.sort_values("tournament_win_probability", ascending=False).reset_index(drop=True)

    results.to_csv(output_path, index=False)
    print(f"Ran {n_simulations} simulations for {tour_name}, saved results to {output_path}")

    print(f"\nTop 15 {tour_name} players by tournament-win probability:")
    print(results.head(15).to_string(index=False))
