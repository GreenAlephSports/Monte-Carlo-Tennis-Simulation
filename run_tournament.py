"""Single entry point for the pipeline: bracket YAML -> Elo ratings -> name matching -> simulation.

Usage:
    python run_tournament.py data/wimbledon_2026_atp.yaml
    python run_tournament.py data/wimbledon_2026_wta.yaml --simulations 5000
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))

from bracket import (  # noqa: E402
    TOUR_CONFIG, match_draw_to_ratings, order_by_draw_position, split_byes, validate_bracket_structure, validate_draw,
)
from bracket_schema import BracketValidationError, load_bracket_yaml  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from hybrid_simulation import (  # noqa: E402
    TOUR_SINGLES_CATEGORY, reconstruct_leaves_by_round2_slot, true_bracket_order,
)
from live_scores import LiveScoresError, extract_matches, fetch_scoreboard  # noqa: E402
from simulate import (  # noqa: E402
    N_SIMULATIONS, report_results, run_simulations_tracking_milestones, simulate_and_report,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def default_output_path(bracket):
    tournament_slug = bracket.tournament.lower().replace(" ", "_")
    tour_slug = bracket.tour.lower()
    return OUTPUT_DIR / f"{tournament_slug}_{bracket.year}_simulation_results_{tour_slug}.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bracket_path", type=Path, help="Path to a bracket YAML file")
    parser.add_argument("--simulations", type=int, default=N_SIMULATIONS, help="Number of Monte Carlo runs")
    parser.add_argument("--output", type=Path, default=None, help="Simulation results CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible Monte Carlo results")
    args = parser.parse_args()

    try:
        bracket = load_bracket_yaml(args.bracket_path)
    except BracketValidationError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    # a PDF-extracted bracket carries an explicit draw 'position' per player (some slots are
    # omitted at the source - a bye's would-be opponent has no row to extract) - reorder by it
    # when present so Round 1 pairing is correct regardless of the YAML's raw list order
    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    try:
        non_bye_count, bye_count = validate_bracket_structure(byes)
    except ValueError as e:
        print(f"{args.bracket_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded bracket: {bracket.tournament} {bracket.year} ({bracket.tour}, {bracket.surface}) "
          f"— {len(players)} players ({non_bye_count} play Round 1, {bye_count} bye(s)), "
          f"start date {bracket.start_date.date()}")

    tour_config = TOUR_CONFIG[bracket.tour]

    matches = load_matches_for_tour(bracket.tour)
    ratings_df = calculate_elo_ratings(matches, bracket.start_date)
    ratings_df = ratings_df.sort_values("overall_elo", ascending=False).reset_index(drop=True)
    print(f"Calculated Elo ratings for {len(ratings_df)} players as of {bracket.start_date.date()} "
          f"(cutoff excludes matches on/after this date)")

    draw, resolutions, ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date
    )

    tier_counts = Counter(r["tier"] for r in resolutions)
    print("Name matching:")
    print(f"  Tier 0 (manual alias override): {tier_counts.get(0, 0)}")
    print(f"  Tier 1 (exact lastname + full initials): {tier_counts.get(1, 0)}")
    print(f"  Tier 2 (lastname + first initial, unique candidate or compound-initials match): {tier_counts.get(2, 0)}")
    print(f"  Tier 3 (no training-window history, STARTING_ELO placeholder): {tier_counts.get(3, 0)}")
    print(f"  Unresolved: {tier_counts.get(None, 0)}")

    unmatched = [r for r in resolutions if r["tier"] is None]
    if unmatched:
        print("\nUnmatched names (check spelling/format against the Elo CSV, or add a manual alias):", file=sys.stderr)
        for entry in unmatched:
            seed = f"[{entry['seed']}]" if entry["seed"] else ""
            print(f"  {entry['name']} {seed}  (looked for key={entry['expected_key']})", file=sys.stderr)
        sys.exit(1)

    tour_config.ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(tour_config.ratings_path, index=False)
    print(f"Saved Elo ratings to {tour_config.ratings_path}")

    validate_draw(draw)
    non_bye_players, bye_players = split_byes(draw, byes)

    output_path = args.output or default_output_path(bracket)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print()
    random.seed(args.seed)
    tour_name = f"{bracket.tournament} {bracket.year} {bracket.tour}"

    if bye_count == 0:
        # no byes means "round winners + byes" concatenation and true bracket order are
        # identical (bye_players is empty) - a no-op case that never needs ESPN, so a bare
        # bracket YAML (e.g. a PDF-sourced Slam draw) keeps working standalone.
        simulate_and_report(
            tour_name, draw, non_bye_players, bye_players, bracket.surface,
            tour_config.ratings_path, output_path, n_simulations=args.simulations,
        )
    else:
        # with byes present, "round winners + byes" concatenation pairs bye-vs-bye and
        # winner-vs-winner in Round 2 instead of the bracket's real winner-vs-bye adjacency -
        # reconstructing true order (as bracket_export.py does) requires ESPN's live Round 1/
        # Round 2 structure, so a bare bracket YAML is no longer sufficient once byes exist.
        try:
            espn_data = fetch_scoreboard(bracket.tour.lower())
        except LiveScoresError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        espn_matches, _ = extract_matches(espn_data)
        category = TOUR_SINGLES_CATEGORY[bracket.tour.lower()]
        tournament_matches = [
            m for m in espn_matches if m["tournament"] == bracket.tournament and m["category"] == category
        ]
        if not tournament_matches:
            print(
                f"ERROR: bracket has {bye_count} bye(s), which requires ESPN's live Round 1/Round 2 "
                f"structure to interleave correctly, but no live ESPN match data was found for "
                f"{bracket.tournament!r} ({category}). A bare bracket YAML alone isn't enough once "
                f"byes are present.", file=sys.stderr,
            )
            sys.exit(1)

        leaves_by_slot = reconstruct_leaves_by_round2_slot(tournament_matches, non_bye_players, bye_players)
        true_order = true_bracket_order(leaves_by_slot)
        ordered_field = [name for name, _is_bye in true_order]
        is_bye = [is_bye_flag for _name, is_bye_flag in true_order]

        champion_counts, _sf_counts, _final_counts = run_simulations_tracking_milestones(
            ordered_field, is_bye, {}, bracket.surface, args.simulations, tour_config.ratings_path
        )
        report_results(tour_name, draw, champion_counts, args.simulations, output_path)


if __name__ == "__main__":
    main()
