"""Conditional equity diagnostic: formalizes the law-of-total-probability check for "player X
just won a real match - how much of the resulting tournament_win_probability change is
attributable to that win in isolation (the 'vacuum' branch) vs. every OTHER real result that
happened to resolve in the same round window?"

For a player/round: pulls the pre-match checkpoint (tournament_win_probability from the field
entering that round), the real own-match win probability, computes the vacuum conditional equity
(pre_equity / match_win_prob - what the checkpoint algebra says the win-branch alone is worth,
since the lose-branch is mechanically 0 once eliminated), pulls the real, fully-realized
post-round checkpoint, and reports the gap between the two as dispersion attributable to every
OTHER match that resolved in the same round. That dispersion is then attributed match-by-match:
each other real result in the round is isolated via the same forced-branch counterfactual
technique used for the Paul/Atmane-vs-Zverev analysis, holding this player's own match pinned to
its real winner and leaving everything else genuinely Monte Carlo.

Also runs a structural check, independent of any one player's story: for a real, decided match,
verify pre-match equity for EACH side equals (that side's real match win probability) times (that
side's forced-win-branch equity) - the law of total probability the whole checkpoint architecture
is supposed to satisfy by construction, since a forced-loss branch is mechanically 0 (an
eliminated player is simply absent from every field after their loss, not down-weighted).

Usage:
    python model/research/conditional_equity_report.py brackets/cincinnati_2026_atp_demo.yaml \
        --case "Paul T.:4" --case "Nakashima B.:3"
"""
import argparse
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bracket import (  # noqa: E402
    TOUR_CONFIG, match_draw_to_ratings, order_by_draw_position, split_byes,
    validate_bracket_structure, validate_draw,
)
from bracket_schema import load_bracket_yaml  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from hybrid_simulation import (  # noqa: E402
    TOUR_SINGLES_CATEGORY, build_known_pairings_by_round, build_real_results_by_round,
    known_matchups_for_round, reconstruct_leaves_by_round2_slot, replay_real_rounds, true_bracket_order,
)
from live_scores import extract_matches, fetch_scoreboard  # noqa: E402
from simulate import N_SIMULATIONS, run_simulations_from_field, run_simulations_partial_round  # noqa: E402
from win_probability import win_probability  # noqa: E402


def load_tournament_state(bracket_path, dates=None, ratings_cache=None, espn_cache=None):
    """ratings_cache/espn_cache, if given, freeze the two live data sources (Kaggle match
    history, ESPN scoreboard) to a local pickle file on first use and reuse it on every later
    invocation that passes the same path - needed to isolate a code change's actual effect from
    ordinary live-data drift (a live, in-progress tournament's Elo ratings and even its own
    already-final results can shift between two runs minutes apart, otherwise confounding any
    before/after comparison; confirmed concretely - Paul T.'s pre-match equity moved from 4.20%
    to 3.57% between two runs of this script ~20 minutes apart with no code change relevant to
    that number at all)."""
    bracket = load_bracket_yaml(bracket_path)
    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    validate_bracket_structure(byes)
    tour_config = TOUR_CONFIG[bracket.tour]

    if ratings_cache and ratings_cache.exists():
        with open(ratings_cache, "rb") as f:
            matches_history = pickle.load(f)
        print(f"Loaded cached Kaggle match history from {ratings_cache} (frozen snapshot, not re-fetched)")
    else:
        matches_history = load_matches_for_tour(bracket.tour)
        if ratings_cache:
            ratings_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(ratings_cache, "wb") as f:
                pickle.dump(matches_history, f)
            print(f"Fetched live Kaggle match history and cached it to {ratings_cache}")

    ratings_df = calculate_elo_ratings(matches_history, bracket.start_date)
    draw, _resolutions, ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date,
    )
    validate_draw(draw)
    non_bye_players, bye_players = split_byes(draw, byes)

    category = TOUR_SINGLES_CATEGORY[bracket.tour.lower()]
    if espn_cache and espn_cache.exists():
        with open(espn_cache, "rb") as f:
            tournament_matches = pickle.load(f)
        print(f"Loaded cached ESPN matches from {espn_cache} (frozen snapshot, not re-fetched)")
    else:
        espn_data = fetch_scoreboard(bracket.tour.lower(), dates=dates)
        espn_matches, _stats = extract_matches(espn_data)
        tournament_matches = [
            m for m in espn_matches if m["tournament"] == bracket.tournament and m["category"] == category
        ]
        if espn_cache:
            espn_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(espn_cache, "wb") as f:
                pickle.dump(tournament_matches, f)
            print(f"Fetched live ESPN scoreboard and cached it to {espn_cache}")

    results_by_round, _round_seq, _u1 = build_real_results_by_round(
        tournament_matches, draw, tour_config.name_aliases)
    known_pairings_by_round, _round_seq2, _u2 = build_known_pairings_by_round(
        tournament_matches, draw, tour_config.name_aliases)
    fields = replay_real_rounds(non_bye_players, bye_players, results_by_round, known_pairings_by_round)

    leaves_by_slot = reconstruct_leaves_by_round2_slot(
        tournament_matches, non_bye_players, bye_players, results_by_round, tour_config.name_aliases)
    true_order = true_bracket_order(leaves_by_slot)
    leaf_position = {name: i for i, (name, _is_bye) in enumerate(true_order)}

    return {
        "bracket": bracket,
        "tour_config": tour_config,
        "draw": draw,
        "results_by_round": results_by_round,
        "known_pairings_by_round": known_pairings_by_round,
        "fields": fields,
        "max_known_round": len(fields) - 1,
        "leaf_position": leaf_position,
    }


def true_order_sorted(state, field):
    leaf_position = state["leaf_position"]
    return sorted(field, key=lambda name: leaf_position.get(name, len(leaf_position)))


def confirmed_real_pairing_for(state, n):
    """Same construction as hybrid_simulation.run_snapshot: maps each later fully-known round's
    real starting field (frozenset) to that round's real pairing, so a full from-field simulation
    keeps using the real bracket structure for as long as simulated results keep matching reality."""
    fields = state["fields"]
    mapping = {}
    for r in range(n + 1, state["max_known_round"] + 1):
        round_pairing = known_matchups_for_round(r, fields[r - 1], state["known_pairings_by_round"])
        if round_pairing is not None:
            mapping[frozenset(fields[r - 1])] = round_pairing
    return mapping


def full_checkpoint_counts(state, n, n_simulations, seed):
    """Reproduces hybrid_simulation's through_round_{n}.csv checkpoint: {player: tournament_win_probability},
    for a round n that's fully real (n <= max_known_round)."""
    assert 0 <= n <= state["max_known_round"], f"round {n} is not fully resolved yet (known through {state['max_known_round']})"
    starting_field = true_order_sorted(state, state["fields"][n])
    pairing_map = confirmed_real_pairing_for(state, n)

    def matchups_resolver(players, _m=pairing_map):
        return _m.get(frozenset(players))

    random.seed(seed)
    counts = run_simulations_from_field(
        starting_field, state["bracket"].surface, n_simulations, state["tour_config"].ratings_path,
        use_upset_boost=True, matchups_resolver=matchups_resolver,
    )
    total = n_simulations
    return {p: counts.get(p, 0) / total for p in state["draw"]}


def partial_round_prob(state, round_num, known_results, n_simulations, seed, target_players=None,
                        use_resolver_fix=True):
    """Runs round_num from fields[round_num - 1] with known_results pinned (any pairing not in
    known_results is genuinely Monte Carlo), then simulates every round after normally - except
    any round beyond round_num that ALSO already happened for real (round_num < max_known_round)
    still gets its real bracket pairing threaded through via matchups_resolver, same as
    full_checkpoint_counts/confirmed_real_pairing_for do; a simulated trial only benefits from
    this for as long as its own simulated survivors keep landing on the real historical field for
    that round - the moment one diverges, it falls back to plain positional pairing like any
    genuinely-unknown round, exactly as confirmed_real_pairing_for already documents.

    use_resolver_fix=False reproduces the exact pre-fix behavior (matchups_resolver=None passed to
    run_simulations_partial_round, so every round after round_num - even an already-real one -
    falls back to plain positional pairing) - lets the fix's effect be isolated with a code-level
    toggle instead of literally reverting simulate.py, since run_simulations_partial_round's own
    behavior is unchanged when no resolver is passed.

    Returns {player: probability} for target_players (defaults to every player)."""
    field_before = state["fields"][round_num - 1]
    matchups = known_matchups_for_round(round_num, field_before, state["known_pairings_by_round"])
    assert matchups is not None, f"round {round_num}'s pairing isn't fully known yet"

    matchups_resolver = None
    if use_resolver_fix:
        pairing_map = confirmed_real_pairing_for(state, round_num)

        def matchups_resolver(players, _m=pairing_map):
            return _m.get(frozenset(players))

    random.seed(seed)
    counts = run_simulations_partial_round(
        field_before, [], known_results, state["bracket"].surface, n_simulations,
        state["tour_config"].ratings_path, use_upset_boost=True, matchups=matchups,
        matchups_resolver=matchups_resolver,
    )
    players = target_players if target_players is not None else state["draw"]
    return {p: counts.get(p, 0) / n_simulations for p in players}, matchups


def mc_stderr(p, n):
    return (p * (1 - p) / n) ** 0.5


def analyze_case(state, player, round_num, args):
    field_before = state["fields"][round_num - 1]
    matchups = known_matchups_for_round(round_num, field_before, state["known_pairings_by_round"])
    assert matchups is not None, f"round {round_num}'s pairing isn't fully known"
    player_pair = next((pair for pair in matchups if player in pair), None)
    assert player_pair is not None, f"{player} isn't in round {round_num}'s field"
    opponent = next(p for p in player_pair if p != player)
    fs_pair = frozenset(player_pair)
    real_winner = state["results_by_round"].get(round_num, {}).get(fs_pair)
    assert real_winner == player, f"{player} did not win the recorded round {round_num} result ({real_winner!r})"

    surface = state["bracket"].surface
    ratings_path = state["tour_config"].ratings_path
    n = args.simulations
    base_seed = args.seed

    # (1) pre-match checkpoint: field entering round_num, i.e. through_round_{round_num-1}
    pre_counts = full_checkpoint_counts(state, round_num - 1, n, base_seed + round_num - 1)
    pre_equity = pre_counts[player]
    opponent_pre_equity = pre_counts[opponent]

    # (2) real own-match win probability (unboosted - no prior-round signal folded in, same as
    # own_match_win_probabilities in hybrid_simulation.py)
    match_win_prob = win_probability(player, opponent, surface, ratings_path)

    # (3) vacuum conditional equity: force ONLY this match to its real result, leave every other
    # round_num match genuinely Monte Carlo. Algebraically this IS pre_equity / match_win_prob,
    # since the lose-branch is mechanically 0 - computed directly (not by division) to also get a
    # real MC estimate with its own sampling noise, rather than trusting the identity blindly.
    vacuum_counts, _ = partial_round_prob(
        state, round_num, {fs_pair: player}, n, base_seed + 500, target_players=[player],
        use_resolver_fix=args.use_resolver_fix,
    )
    vacuum_equity = vacuum_counts[player]
    vacuum_equity_via_division = pre_equity / match_win_prob if match_win_prob > 0 else float("nan")

    # (3b) forced-loss branch for the SAME match, as a direct empirical check that "eliminated"
    # really does mean probability exactly 0 by construction, not just assumed.
    lose_counts, _ = partial_round_prob(
        state, round_num, {fs_pair: opponent}, n, base_seed + 501, target_players=[player],
        use_resolver_fix=args.use_resolver_fix,
    )
    forced_lose_equity = lose_counts[player]

    # (4) real, fully-realized post-round checkpoint (every real round_num result applied, not
    # just this player's own match)
    post_round = round_num
    if post_round > state["max_known_round"]:
        post_equity = None
    else:
        post_counts = full_checkpoint_counts(state, post_round, n, base_seed + post_round)
        post_equity = post_counts[player]

    dispersion = None if post_equity is None else post_equity - vacuum_equity

    # (5) attribute dispersion across every OTHER real result in round_num: hold this player's
    # match pinned to its real winner, flip exactly one other match at a time (real winner vs the
    # other side), leave everything else genuinely random. Same seed for the real/flip pair of
    # runs (common random numbers) so most of the shared Monte Carlo noise cancels in the
    # difference.
    other_pairs = [pair for pair in matchups if frozenset(pair) != fs_pair]
    attributions = []
    a_n = args.attribution_simulations
    for idx, pair in enumerate(other_pairs):
        fs_m = frozenset(pair)
        m_real_winner = state["results_by_round"].get(round_num, {}).get(fs_m)
        if m_real_winner is None:
            continue  # this match hasn't actually been decided yet - nothing to attribute
        m_loser = next(p for p in pair if p != m_real_winner)
        seed_m = base_seed + 10000 + idx

        real_counts, _ = partial_round_prob(
            state, round_num, {fs_pair: player, fs_m: m_real_winner}, a_n, seed_m, target_players=[player],
            use_resolver_fix=args.use_resolver_fix,
        )
        flip_counts, _ = partial_round_prob(
            state, round_num, {fs_pair: player, fs_m: m_loser}, a_n, seed_m, target_players=[player],
            use_resolver_fix=args.use_resolver_fix,
        )
        p_real = real_counts[player]
        p_flip = flip_counts[player]
        attributions.append({
            "match": f"{m_real_winner} def. {m_loser}",
            "winner": m_real_winner,
            "loser": m_loser,
            "p_if_real_result": p_real,
            "p_if_flipped": p_flip,
            "contribution": p_real - p_flip,
            "stderr": (mc_stderr(p_real, a_n) ** 2 + mc_stderr(p_flip, a_n) ** 2) ** 0.5,
        })
    attributions.sort(key=lambda r: abs(r["contribution"]), reverse=True)

    return {
        "player": player,
        "opponent": opponent,
        "round": round_num,
        "pre_equity": pre_equity,
        "opponent_pre_equity": opponent_pre_equity,
        "match_win_prob": match_win_prob,
        "vacuum_equity": vacuum_equity,
        "vacuum_equity_via_division": vacuum_equity_via_division,
        "forced_lose_equity": forced_lose_equity,
        "post_equity": post_equity,
        "dispersion": dispersion,
        "attributions": attributions,
        "other_match_count": len(other_pairs),
    }


def structural_check(state, player, opponent, round_num, args):
    """Verifies, for BOTH sides of one real decided match, that pre-match equity = match_win_prob
    * forced-win-branch-equity (the lose-branch being mechanically 0), within a few Monte Carlo
    standard errors. Independent of whichever side is the "story" - checks both the winner and
    the loser of the same match, since the loser is the cleaner "genuinely eliminated" case."""
    field_before = state["fields"][round_num - 1]
    matchups = known_matchups_for_round(round_num, field_before, state["known_pairings_by_round"])
    pair = next(p for p in matchups if player in p and opponent in p)
    fs_pair = frozenset(pair)
    real_winner = state["results_by_round"][round_num][fs_pair]
    real_loser = next(p for p in pair if p != real_winner)

    n = args.simulations
    pre_counts = full_checkpoint_counts(state, round_num - 1, n, args.seed + round_num - 1)
    rows = []
    for side, other in ((real_winner, real_loser), (real_loser, real_winner)):
        p_win_match = win_probability(side, other, state["bracket"].surface, state["tour_config"].ratings_path)
        pre_equity_side = pre_counts[side]
        won_counts, _ = partial_round_prob(
            state, round_num, {fs_pair: side}, n, args.seed + 700, target_players=[side],
            use_resolver_fix=args.use_resolver_fix,
        )
        lost_counts, _ = partial_round_prob(
            state, round_num, {fs_pair: other}, n, args.seed + 701, target_players=[side],
            use_resolver_fix=args.use_resolver_fix,
        )
        forced_win_equity = won_counts[side]
        forced_lose_equity = lost_counts[side]
        predicted = p_win_match * forced_win_equity
        se = mc_stderr(forced_win_equity, n) * p_win_match
        rows.append({
            "side": side,
            "actually_won": side == real_winner,
            "pre_equity": pre_equity_side,
            "match_win_prob": p_win_match,
            "forced_win_branch_equity": forced_win_equity,
            "forced_lose_branch_equity": forced_lose_equity,
            "predicted_pre_equity": predicted,
            "abs_error": abs(pre_equity_side - predicted),
            "stderr_budget": se,
        })
    return rows


def print_case_report(result):
    p, opp, r = result["player"], result["opponent"], result["round"]
    print(f"\n{'=' * 70}\n{p} vs {opp} - Round {r}\n{'=' * 70}")
    print(f"  Pre-match tournament_win_probability ({p}): {result['pre_equity']:.4f} ({result['pre_equity']*100:.2f}%)")
    print(f"  Real own-match win probability vs {opp}:    {result['match_win_prob']:.4f} ({result['match_win_prob']*100:.2f}%)")
    print(f"  Vacuum conditional equity (forced-win only, nothing else pinned):")
    print(f"    direct MC estimate:      {result['vacuum_equity']:.4f} ({result['vacuum_equity']*100:.2f}%)")
    print(f"    via pre_equity/match_win_prob: {result['vacuum_equity_via_division']:.4f} ({result['vacuum_equity_via_division']*100:.2f}%)")
    print(f"  Forced-loss branch (sanity check, should be ~0.0000): {result['forced_lose_equity']:.4f}")
    if result["post_equity"] is not None:
        print(f"  Real, fully-realized post-round tournament_win_probability: {result['post_equity']:.4f} ({result['post_equity']*100:.2f}%)")
        print(f"  Dispersion attributable to the other {result['other_match_count']} real round-{r} results: "
              f"{result['dispersion']:+.4f} ({result['dispersion']*100:+.2f}pp)")
    else:
        print(f"  Round {r} isn't fully resolved yet - no post-round checkpoint available.")

    if result["attributions"]:
        print(f"\n  Per-match attribution (holding {p}'s own match at its real result, isolating one "
              f"other match at a time):")
        for row in result["attributions"]:
            print(f"    {row['match']:35s}  contribution={row['contribution']:+.4f} "
                  f"(real={row['p_if_real_result']:.4f}, flipped={row['p_if_flipped']:.4f}, "
                  f"stderr~{row['stderr']:.4f})")
        total_attributed = sum(row["contribution"] for row in result["attributions"])
        print(f"    sum of individual contributions: {total_attributed:+.4f} "
              f"(vs. total dispersion {result['dispersion']:+.4f} - won't match exactly, "
              f"interactions + MC noise)")


def print_structural_report(rows, round_num, tag):
    print(f"\n{'-' * 70}\nStructural check ({tag}, round {round_num}): "
          f"pre_equity == match_win_prob * forced_win_branch_equity ?\n{'-' * 70}")
    for row in rows:
        status = "within noise" if row["abs_error"] <= max(3 * row["stderr_budget"], 0.003) else "OUTSIDE budget"
        print(f"  {row['side']:20s} ({'won' if row['actually_won'] else 'lost'} the real match)")
        print(f"    pre_equity={row['pre_equity']:.4f}  match_win_prob={row['match_win_prob']:.4f}  "
              f"forced_win_branch={row['forced_win_branch_equity']:.4f}  forced_lose_branch={row['forced_lose_branch_equity']:.4f}")
        print(f"    predicted={row['predicted_pre_equity']:.4f}  actual={row['pre_equity']:.4f}  "
              f"abs_error={row['abs_error']:.4f}  (~{row['stderr_budget']:.4f} stderr budget)  -> {status}")


def parse_case(spec):
    if ":" in spec:
        name, round_str = spec.rsplit(":", 1)
        return name, int(round_str)
    return spec, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bracket_path", type=Path)
    parser.add_argument("--case", action="append", required=True,
                         help='"Player Name:Round" - e.g. --case "Paul T.:4". Round is required '
                              '(must be a round that already fully resolved for real).')
    parser.add_argument("--simulations", type=int, default=N_SIMULATIONS)
    parser.add_argument("--attribution-simulations", type=int, default=2000,
                         help="smaller simulation count for the per-adjacent-match attribution "
                              "loop, which runs 2 sims per other match in the round")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dates", default=None)
    parser.add_argument("--ratings-cache", type=Path, default=None,
                         help="pickle path for a frozen Kaggle match-history snapshot - fetched "
                              "and saved on first use, reused (never re-fetched) on every later "
                              "run that passes the same path")
    parser.add_argument("--espn-cache", type=Path, default=None,
                         help="pickle path for a frozen ESPN scoreboard snapshot, same freeze/reuse "
                              "semantics as --ratings-cache")
    parser.add_argument("--no-use-resolver-fix", dest="use_resolver_fix", action="store_false",
                         help="reproduce the pre-fix behavior (matchups_resolver=None passed to "
                              "run_simulations_partial_round, so any known round after the forced "
                              "one falls back to plain positional pairing) - for isolating the "
                              "fix's effect against an identical frozen data snapshot")
    parser.set_defaults(use_resolver_fix=True)
    args = parser.parse_args()

    state = load_tournament_state(
        args.bracket_path, dates=args.dates, ratings_cache=args.ratings_cache, espn_cache=args.espn_cache,
    )
    print(f"Real results fully known through round {state['max_known_round']} of {args.bracket_path.name}")
    print(f"matchups_resolver fix: {'ENABLED' if args.use_resolver_fix else 'DISABLED (pre-fix behavior)'}")

    for spec in args.case:
        player, round_num = parse_case(spec)
        assert round_num is not None, f"--case {spec!r} needs an explicit :ROUND"
        result = analyze_case(state, player, round_num, args)
        print_case_report(result)
        structural_rows = structural_check(state, player, result["opponent"], round_num, args)
        print_structural_report(structural_rows, round_num, f"{player} vs {result['opponent']}")


if __name__ == "__main__":
    main()
