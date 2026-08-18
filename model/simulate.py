import random
from collections import Counter

import pandas as pd

from bracket import get_matchups, validate_draw
from win_probability import (
    UPSET_BOOST_ELO_GAP_THRESHOLD, UPSET_BOOST_LOGIT_SHIFT, apply_logit_shift, get_surface_elo, win_probability,
)

N_SIMULATIONS = 10000


def _beat_a_big_favorite(player, player_elo, prior_beaten_elo):
    beaten_elo = prior_beaten_elo.get(player)
    return beaten_elo is not None and beaten_elo - player_elo > UPSET_BOOST_ELO_GAP_THRESHOLD


def _play_round(players, surface, ratings_path, known_results=None, prior_beaten_elo=None):
    """known_results, if given, maps frozenset({player_a, player_b}) -> real winner for
    pairings already decided in reality - that winner is used as-is (no randomness spent),
    while any pairing missing from known_results is Monte Carlo-simulated as usual. Lets a
    round mix real, already-known results with genuinely undecided matches instead of requiring
    the whole round to be one or the other.

    prior_beaten_elo, if given (a dict, possibly empty - not None), turns on the upset-boost
    adjustment: maps player -> the Elo of whoever they beat in the immediately preceding round of
    THIS SAME call chain (see UPSET_BOOST_ELO_GAP_THRESHOLD's docstring in win_probability.py for
    where the threshold/shift come from). Returns (winners, next_prior_beaten_elo) - the second
    element is None when prior_beaten_elo was None (feature off), otherwise a fresh dict mapping
    this round's winners to the Elo of whoever they just beat, ready to feed the next round."""
    track_upsets = prior_beaten_elo is not None
    winners = []
    next_beaten_elo = {} if track_upsets else None
    for player_a, player_b in get_matchups(players):
        elo_a = elo_b = None
        if track_upsets:
            elo_a = get_surface_elo(player_a, surface, ratings_path)
            elo_b = get_surface_elo(player_b, surface, ratings_path)

        known_winner = (known_results or {}).get(frozenset((player_a, player_b)))
        if known_winner is not None:
            winner = known_winner
        else:
            prob_a = win_probability(player_a, player_b, surface, ratings_path)
            if track_upsets:
                boost_a = UPSET_BOOST_LOGIT_SHIFT if _beat_a_big_favorite(player_a, elo_a, prior_beaten_elo) else 0.0
                boost_b = UPSET_BOOST_LOGIT_SHIFT if _beat_a_big_favorite(player_b, elo_b, prior_beaten_elo) else 0.0
                if boost_a != boost_b:
                    prob_a = apply_logit_shift(prob_a, boost_a - boost_b)
            winner = player_a if random.random() < prob_a else player_b
        winners.append(winner)
        if track_upsets:
            next_beaten_elo[winner] = elo_b if winner == player_a else elo_a
    return winners, next_beaten_elo


# plays out single-elimination from an arbitrary starting field until one player remains. Used
# both for a full from-scratch simulation (starting field = Round 1 winners + byes) and for a
# hybrid simulation that resumes from a field already partly decided by real results.
#
# use_upset_boost only ever affects rounds played out WITHIN this call - a field's first round
# here never has a "most recent win this tournament" to condition on (same structural gap a
# fatigue adjustment would have at Round 1), so it starts boost-inactive and only turns on for
# players once they've won a round inside this same replay.
def simulate_from_field(field, surface, ratings_path, use_upset_boost=True):
    players = list(field)
    prior_beaten_elo = {} if use_upset_boost else None
    while len(players) > 1:
        players, prior_beaten_elo = _play_round(players, surface, ratings_path, prior_beaten_elo=prior_beaten_elo)
    return players[0]


# plays out one full random bracket. Round 1 only pairs up non-bye players (rng by weighted by
# win_probability to pick the winner of each matchup); those winners are then combined with the
# bye players - who skipped Round 1 entirely - to form the Round 2 field, and play continues the
# same way each round until one player remains. With no byes, non_bye_players is the whole draw
# and bye_players is empty, so this reduces to the plain single-elimination case.
#
# This is always the pre-tournament baseline (no real results exist yet) - never takes an
# upset-boost flag, since the signal it needs (a real win already on the board) can't exist here.
def simulate_tournament(non_bye_players, bye_players, surface, ratings_path):
    winners, _ = _play_round(non_bye_players, surface, ratings_path)
    return simulate_from_field(winners + bye_players, surface, ratings_path)


def run_simulations_from_field(field, surface, n_simulations, ratings_path, use_upset_boost=True):
    champion_counts = Counter()
    for _ in range(n_simulations):
        champion_counts[simulate_from_field(field, surface, ratings_path, use_upset_boost=use_upset_boost)] += 1
    return champion_counts


# for a round that's only partly decided in reality (some matches final, others not yet played) -
# not just round 1, any round whose matchup structure is fully known (see
# hybrid_simulation.known_matchups_for_round). known_results pins whichever pairings already have
# a real winner, and Monte Carlo-decides the rest, same as any other simulated match. bye_players
# is only non-empty when starting_field is the pre-round-1 field (byes join in right after round
# 1); for any later round they're already folded into starting_field, so pass ().  Every round
# after this one is always fully simulated, same as run_simulations_from_field.
def run_simulations_partial_round(starting_field, bye_players, known_results, surface, n_simulations, ratings_path,
                                   use_upset_boost=True):
    champion_counts = Counter()
    for _ in range(n_simulations):
        prior_beaten_elo = {} if use_upset_boost else None
        round_winners, prior_beaten_elo = _play_round(
            starting_field, surface, ratings_path, known_results, prior_beaten_elo)
        field = round_winners + list(bye_players)
        champion_counts[simulate_from_field(field, surface, ratings_path, use_upset_boost=use_upset_boost)] += 1
    return champion_counts


# like run_simulations_partial_round, but also tracks which players reach the semifinal (last 4
# remaining) and final (last 2 remaining) in each trial, not just the eventual champion - needed
# for a p_sf/p_final breakdown alongside p_champ.
#
# ordered_field/is_bye must be in TRUE bracket order (real draw adjacency), not "plays now" and
# "joins after" concatenated in bulk - see known_matchups_for_round's docstring in
# hybrid_simulation.py for why naive concatenation of round winners + byes doesn't reconstruct the
# real bracket tree (it always pairs winner-vs-winner and bye-vs-bye in the following round instead
# of winner-vs-bye). A bye (is_bye[i] True) passes straight through to the next round unplayed;
# two consecutive non-bye entries are this round's real match. For a round with no byes joining
# (i.e. every round after byes have already been absorbed), pass is_bye as all False.
def run_simulations_tracking_milestones(ordered_field, is_bye, known_results, surface, n_simulations, ratings_path):
    champion_counts = Counter()
    semifinal_counts = Counter()
    final_counts = Counter()
    n = len(ordered_field)

    # a player already among the last <=4 (or <=2) real, live entrants has already reached the
    # semifinal (or final) as a plain fact of the current bracket state, not a simulated outcome -
    # e.g. once a live draw is down to its own Final, both finalists have certainly already won
    # their semifinal. The per-trial loop below only ever observes that transition happening
    # *during* a simulated round, so it can't record a milestone the real field already starts
    # past - without this, an already-decided finalist would wrongly show p_sf = p_final = 0.
    if not any(is_bye):
        if n <= 4:
            for p in ordered_field:
                semifinal_counts[p] = n_simulations
        if n <= 2:
            for p in ordered_field:
                final_counts[p] = n_simulations

    for _ in range(n_simulations):
        players = []
        i = 0
        while i < n:
            if is_bye[i]:
                players.append(ordered_field[i])
                i += 1
                continue
            player_a, player_b = ordered_field[i], ordered_field[i + 1]
            known_winner = (known_results or {}).get(frozenset((player_a, player_b)))
            if known_winner is not None:
                winner = known_winner
            else:
                prob_a = win_probability(player_a, player_b, surface, ratings_path)
                winner = player_a if random.random() < prob_a else player_b
            players.append(winner)
            i += 2
        while True:
            if len(players) == 4:
                for p in players:
                    semifinal_counts[p] += 1
            elif len(players) == 2:
                for p in players:
                    final_counts[p] += 1
            if len(players) == 1:
                champion_counts[players[0]] += 1
                break
            players, _ = _play_round(players, surface, ratings_path)
    return champion_counts, semifinal_counts, final_counts


def run_simulations(non_bye_players, bye_players, surface, n_simulations, ratings_path):
    champion_counts = Counter()
    for _ in range(n_simulations):
        champion_counts[simulate_tournament(non_bye_players, bye_players, surface, ratings_path)] += 1
    return champion_counts


def report_results(tour_name, draw, champion_counts, n_simulations, output_path):
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


def simulate_and_report(tour_name, draw, non_bye_players, bye_players, surface, ratings_path, output_path,
                         n_simulations=N_SIMULATIONS):
    validate_draw(draw)
    champion_counts = run_simulations(non_bye_players, bye_players, surface, n_simulations, ratings_path)
    report_results(tour_name, draw, champion_counts, n_simulations, output_path)
