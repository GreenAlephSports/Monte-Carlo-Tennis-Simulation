"""Single entry point for the pipeline: bracket YAML -> Elo ratings -> name matching -> simulation.

Usage:
    python run_tournament.py data/wimbledon_2026_atp.yaml
    python run_tournament.py data/wimbledon_2026_wta.yaml --simulations 5000
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))

from bracket import (  # noqa: E402
    TOUR_CONFIG, match_draw_to_ratings, order_by_draw_position, split_byes, validate_bracket_structure, validate_draw,
)
from bracket_schema import BracketValidationError, load_bracket_yaml  # noqa: E402
from data_loader import load_matches  # noqa: E402
from elo_ratings import calculate_elo_ratings  # noqa: E402
from simulate import N_SIMULATIONS, simulate_and_report  # noqa: E402

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

    matches = load_matches(tour_config.match_data_path)
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
    print(f"  Tier 2 (lastname + first initial, unique candidate): {tier_counts.get(2, 0)}")
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
    simulate_and_report(
        f"{bracket.tournament} {bracket.year} {bracket.tour}", draw, non_bye_players, bye_players, bracket.surface,
        tour_config.ratings_path, output_path, n_simulations=args.simulations,
    )


if __name__ == "__main__":
    main()
