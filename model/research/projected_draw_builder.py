"""Builds a projected bracket YAML (the same schema espn_bracket.py / parse_atp_draw.py produce)
from current rankings, for a tournament whose real draw hasn't been released yet. run_tournament.py
and bracket.py work identically regardless of which of the three produced the file.

There is no live-draw-ceremony data to parse here (that's the whole point - this runs BEFORE one
exists), so seeding/placement is necessarily a projection, not a scrape:

  - The top `--seeds` players by current ranking are seeded 1..N. "Current ranking" comes from
    elo_ratings.py's own current_rank column (whatever ATP/WTA rank each player carried in their
    most recent training-window match, per the live Kaggle pull) - the same no-lookahead,
    already-in-the-pipeline ranking signal used everywhere else, not a separate scrape.
  - Seeds are placed into bracket slots using the standard recursive seeding template every
    single-elimination draw (Slam or otherwise) is built from - seed 1 and 2 at opposite ends of
    the draw, seeds 3-4 split the two remaining quarters, 5-8 the four remaining eighths, and so
    on. See seed_bracket_positions - verified against the published 128-draw seeding chart
    (seed 1 -> slot 1, seed 2 -> slot 128, {3,4} -> {64,65}, {5-8} -> {32,33,96,97}, ...).
    Where the real draw ceremony randomly breaks a tie within a seed group (e.g. which of slots
    64/65 gets seed 3 vs 4), this instead breaks it deterministically by rank order - a
    reasonable stand-in, not a prediction of which the live draw will pick.
  - Everyone else (remaining direct-acceptance entrants by rank, plus placeholder slots for
    qualifiers/wildcards not yet decided) fills the remaining bracket slots via a seeded RNG -
    reproducible run to run, not meant to predict the real draw's random slotting either.

Manual overrides (a withdrawal, a forced seed, a wildcard/qualifier name once the tour announces
it before the full draw) go in a small YAML file - see load_overrides/apply_overrides.

Usage:
    python model/research/projected_draw_builder.py build output.yaml --tour atp \
        --tournament "US Open" --start-date 2026-08-31 --surface Hard \
        [--overrides overrides.yaml] [--draw-size 128] [--seeds 32] \
        [--qualifiers 16] [--wildcards 8] [--rng-seed 0]

    # once the real draw is out, check how the projection did:
    python model/research/projected_draw_builder.py compare projected.yaml real.yaml
"""
import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import (  # noqa: E402
    TOUR_CONFIG, _split_csv_name, match_draw_to_ratings, order_by_draw_position, validate_bracket_structure,
    validate_draw,
)
from bracket_schema import load_bracket_yaml  # noqa: E402
from elo_ratings import SURFACES, calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from simulate import N_SIMULATIONS, run_simulations_tracking_milestones  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"


def _is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def seed_bracket_positions(n):
    """0-indexed bracket slot for seed i+1 (1-indexed) in an n-slot draw, via the standard
    recursive seeding template (see module docstring). Only meaningful when every one of the n
    slots is itself seeded; build_projected_bracket instead computes this for the FULL draw size
    and takes just the first `--seeds` entries, since that's what correctly reproduces the real
    published seeding chart for e.g. 32 seeds in a 128 draw (the recursion for n=128 restricted
    to its first 32 entries IS the real 32-seed chart, not an approximation of it)."""
    if not _is_power_of_two(n):
        raise ValueError(f"n must be a power of two, got {n}")
    positions = [0]
    length = 1
    while length < n:
        positions = [x for p in positions for x in (p, length * 2 - 1 - p)]
        length *= 2
    return positions


def current_rankings(tour, cutoff_date):
    """Current-rank-ordered list of (rank, ratings-csv-name) as of cutoff_date, freshly computed
    via elo_ratings.py's own pipeline (live Kaggle pull, local-CSV fallback) rather than read off
    a possibly-stale output/ CSV. Players with no recorded current_rank (no Rank_1/Rank_2 ever
    seen for them in the training window) are excluded - there's no signal to seed/order them by.

    Deduped on (lastname, initials) via bracket.py's own name splitter - the source Kaggle data
    occasionally spells the same player two different ways across rows (e.g. 'Tirante T. A.' vs
    'Tirante T.A.'), which without this would seed the same real player twice under two names
    that bracket.py's own fuzzy matcher later collapses back into one, producing a duplicate-
    player draw. Keeps whichever spelling comes first in rank order (i.e. its better/lowest rank)."""
    matches = load_matches_for_tour(tour)
    ratings = calculate_elo_ratings(matches, cutoff_date)
    ranked = ratings.dropna(subset=["current_rank"]).copy()
    ranked = ranked.sort_values("current_rank", kind="stable")

    seen_keys = set()
    deduped = []
    for rank, name in zip(ranked["current_rank"].astype(int), ranked["player"]):
        key = _split_csv_name(name)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append((rank, name))
    return deduped


def load_overrides(path):
    """Override YAML schema (all keys optional):

        withdrawals:
          - "Nadal R."             # ratings-csv-style name. Dropped from the ranked pool
                                    # entirely before seeding/slotting, so the next-ranked player
                                    # (or alternate) automatically moves up to fill their slot -
                                    # same effect a real tour withdrawal + alternate has.
        seed_overrides:
          "Alcaraz C.": 1          # force this player into seed slot 1, bumping whoever the
                                    # rank-order would have put there back into the unseeded pool.
        name_overrides:
          "Qualifier 3": "Some Player X."   # rename any single entry (a generic qualifier/
                                    # wildcard placeholder, or a ranked entrant) once a real name
                                    # is known - matched by whatever name it currently carries.
    """
    if path is None:
        return {"withdrawals": [], "seed_overrides": {}, "name_overrides": {}}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {
        "withdrawals": list(data.get("withdrawals") or []),
        "seed_overrides": dict(data.get("seed_overrides") or {}),
        "name_overrides": dict(data.get("name_overrides") or {}),
    }


def _select_seeds(pool_names, seed_count, seed_overrides):
    """pool_names: ranked-order list of names (withdrawals already removed). Returns
    (seed_names, leftover_names) where seed_names[i] is the player seeded i+1, and leftover_names
    is what's left of pool_names in rank order, for unseeded direct-acceptance slots."""
    pinned = dict(seed_overrides)  # name -> seed_number, as given
    pinned_names = set(pinned.keys())
    pinned_by_seed = {seed_num: name for name, seed_num in pinned.items()}

    invalid = [s for s in pinned_by_seed if not (1 <= s <= seed_count)]
    if invalid:
        raise ValueError(f"seed_overrides has out-of-range seed number(s): {sorted(invalid)}")

    remaining = [n for n in pool_names if n not in pinned_names]
    remaining_iter = iter(remaining)

    seed_names = []
    for seed_num in range(1, seed_count + 1):
        if seed_num in pinned_by_seed:
            seed_names.append(pinned_by_seed[seed_num])
        else:
            seed_names.append(next(remaining_iter))
    leftover_names = list(remaining_iter)
    return seed_names, leftover_names


def build_projected_bracket(
    tour, tournament, start_date, surface, draw_size=128, seeds=32, qualifiers=16, wildcards=8,
    overrides_path=None, rng_seed=0,
):
    tour = tour.upper()
    if surface not in SURFACES:
        raise ValueError(f"surface must be one of {SURFACES}, got {surface!r}")
    if not _is_power_of_two(draw_size):
        raise ValueError(f"draw_size must be a power of two, got {draw_size}")
    if not _is_power_of_two(seeds):
        raise ValueError(f"seeds must be a power of two, got {seeds}")
    total_direct = draw_size - qualifiers - wildcards
    if total_direct < seeds:
        raise ValueError(
            f"draw_size ({draw_size}) minus qualifiers ({qualifiers}) and wildcards ({wildcards}) "
            f"leaves only {total_direct} direct-acceptance slots, fewer than seeds ({seeds})"
        )

    overrides = load_overrides(overrides_path)

    ranked = current_rankings(tour, start_date)
    pool_names = [name for _rank, name in ranked if name not in overrides["withdrawals"]]

    seed_names, leftover_names = _select_seeds(pool_names, seeds, overrides["seed_overrides"])

    unseeded_direct_count = total_direct - seeds
    unseeded_direct = leftover_names[:unseeded_direct_count]

    entries = [{"seed": i + 1, "name": name, "status": None} for i, name in enumerate(seed_names)]
    entries += [{"seed": None, "name": name, "status": None} for name in unseeded_direct]
    entries += [{"seed": None, "name": f"Qualifier {i}", "status": "Q"} for i in range(1, qualifiers + 1)]
    entries += [{"seed": None, "name": f"Wildcard {i}", "status": "WC"} for i in range(1, wildcards + 1)]

    name_overrides = overrides["name_overrides"]
    for entry in entries:
        if entry["name"] in name_overrides:
            entry["name"] = name_overrides[entry["name"]]

    seeded_entries = [e for e in entries if e["seed"] is not None]
    unseeded_entries = [e for e in entries if e["seed"] is None]

    rng = random.Random(rng_seed)
    rng.shuffle(unseeded_entries)

    full_positions = seed_bracket_positions(draw_size)
    slots = [None] * draw_size
    for entry in seeded_entries:
        slots[full_positions[entry["seed"] - 1]] = entry

    remaining_positions = [i for i in range(draw_size) if slots[i] is None]
    if len(remaining_positions) != len(unseeded_entries):
        raise ValueError(
            f"internal error: {len(remaining_positions)} open bracket slots but "
            f"{len(unseeded_entries)} unseeded entries"
        )
    for pos, entry in zip(remaining_positions, unseeded_entries):
        slots[pos] = entry

    players = [
        {"seed": e["seed"], "name": e["name"], "status": e["status"], "bye": False} for e in slots
    ]

    bracket = {
        "tournament": tournament,
        "year": int(str(start_date)[:4]),
        "tour": tour,
        "surface": surface,
        "start_date": str(start_date)[:10],
        "draw_size": draw_size,
        "players": players,
    }
    stats = {
        "seeded": len(seed_names),
        "unseeded_direct": len(unseeded_direct),
        "qualifiers": qualifiers,
        "wildcards": wildcards,
        "withdrawals_applied": len(overrides["withdrawals"]),
        "seed_overrides_applied": len(overrides["seed_overrides"]),
        "name_overrides_applied": len(name_overrides),
    }
    return bracket, stats


def simulate_projected_draw(bracket_path, n_simulations=N_SIMULATIONS, seed=None, output_path=None):
    """Runs a full pre-tournament Monte Carlo simulation over a projected bracket and exports the
    same players/p_champ/p_sf/p_final JSON shape bracket_export.py produces for a LIVE tournament -
    but without any of bracket_export.py's ESPN-live-result machinery, which is the wrong tool here
    and was silently producing a near-empty "alive" list when tried against a projected draw.

    Root cause of that collapse: bracket_export.py determines who's "alive" by cross-referencing
    ESPN's live scoreboard feed (draw_to_espn / alive_draw_names) - a player only appears in the
    export if ESPN's feed has already reported a real match involving them. For a projected draw
    built ahead of the real draw ceremony, ESPN has no matches for this event's main draw at all
    (or, worse, its undated "today" scoreboard can pick up an unrelated same-name event, e.g.
    qualifying) - so nearly the entire 128-player field silently fails that cross-reference and
    gets excluded, leaving only the handful of names that happened to collide with whatever ESPN
    data was found. That's a mismatch between the tool and the situation (no real matches exist to
    replay yet), not a name-resolution bug - match_draw_to_ratings' own Tier-3 STARTING_ELO
    fallback (bracket.py) already resolves every unmatched qualifier/wildcard placeholder correctly
    on its own, confirmed by bracket.py's "All N players matched" check.

    This function instead treats the WHOLE draw as pre-Round-1 (every player alive, nobody
    eliminated yet - true for any bracket that hasn't been played), and simulates from there via
    run_simulations_tracking_milestones (now upset-boost-aware - see that function's docstring for
    the separate bug fixed there). Quarters are derived directly from static bracket position
    (list order), since with no live results there's no ESPN-round-2-slot reconstruction to do -
    the YAML's own order already IS bracket adjacency. Only valid for a bye-free draw (a clean
    Grand-Slam-shape bracket, draw_size == 2**k with no byes) - a projected draw padded with byes
    would need real bracket-tree reconstruction to quarter-tag correctly, which this doesn't do.
    """
    bracket = load_bracket_yaml(bracket_path)
    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    non_bye_count, bye_count = validate_bracket_structure(byes)
    if bye_count:
        raise ValueError(
            f"simulate_projected_draw only supports a bye-free draw (got {bye_count} bye(s)) - "
            f"quarter tagging here assumes list order is bracket adjacency, which byes break"
        )

    tour_config = TOUR_CONFIG[bracket.tour]
    matches_history = load_matches_for_tour(bracket.tour)
    ratings_df = calculate_elo_ratings(matches_history, bracket.start_date)
    ratings_df = ratings_df.sort_values("overall_elo", ascending=False).reset_index(drop=True)

    draw, resolutions, ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date
    )
    unmatched = [r for r in resolutions if r["tier"] is None]
    if unmatched:
        raise RuntimeError(f"Unmatched bracket names: {[r['name'] for r in unmatched]}")
    # win_probability() reads Elo from this file, not from ratings_df in memory - same convention
    # export_bracket_json follows, so any freshly-created Tier-3 placeholder rows are visible to it.
    tour_config.ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(tour_config.ratings_path, index=False)

    validate_draw(draw)

    n = len(draw)
    quarter_size = n // 4
    quarter_by_name = {name: f"Q{i // quarter_size + 1}" for i, name in enumerate(draw)}

    # unseeded by default (Idan doesn't want reproducible/fixed-seed runs unless explicitly asked
    # for via --seed) - only fix the RNG when a seed is explicitly given.
    if seed is not None:
        random.seed(seed)
    champ_counts, sf_counts, final_counts = run_simulations_tracking_milestones(
        draw, byes, {}, bracket.surface, n_simulations, tour_config.ratings_path
    )

    players_out = [
        {
            "player": name,
            "quarter": quarter_by_name[name],
            "p_champ": round(champ_counts.get(name, 0) / n_simulations, 3),
            "p_sf": round(sf_counts.get(name, 0) / n_simulations, 3),
            "p_final": round(final_counts.get(name, 0) / n_simulations, 3),
        }
        for name in draw
    ]
    players_out.sort(key=lambda r: -r["p_champ"])

    tour_word = "men" if bracket.tour == "ATP" else "women"
    tournament_slug = bracket.tournament.lower().replace(" ", "-")
    output = {
        "meta": {
            "tournament": f"{tournament_slug}-{tour_word}-{bracket.year}",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "iterations": n_simulations,
            "seed": seed,
            "status": "projected",
        },
        "players": players_out,
        "matchups": {},
        "head_to_head": {},
        "warnings": [
            "Pre-tournament projection: no real draw/results exist yet, so 'player' is this "
            "project's internal ratings-csv name (not an ESPN displayName), and every entrant in "
            "the draw is included as alive.",
        ],
    }

    if output_path is None:
        output_path = OUTPUT_DIR / f"{Path(bracket_path).stem}_bracket_export.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output_path, output


def compare_brackets(projected_path, real_path):
    """Prints how a projected draw (built ahead of time) stacks up against the real one, once
    released: overall player overlap, how many seeds landed on the exact same name, and how many
    of those correctly-matched seeds also landed in the exact same bracket slot (list position).
    Both files are expected in the same schema, so this loads them with the normal bracket_schema
    loader rather than raw YAML."""
    projected = load_bracket_yaml(projected_path)
    real = load_bracket_yaml(real_path)

    proj_names = {p.name for p in projected.players}
    real_names = {p.name for p in real.players}
    overlap = proj_names & real_names

    proj_seed_by_name = {p.name: p.seed for p in projected.players if p.seed is not None}
    real_seed_by_name = {p.name: p.seed for p in real.players if p.seed is not None}
    seed_names_matched = set(proj_seed_by_name) & set(real_seed_by_name)
    seed_numbers_matched = {
        name for name in seed_names_matched if proj_seed_by_name[name] == real_seed_by_name[name]
    }

    proj_slot_by_name = {p.name: i for i, p in enumerate(projected.players)}
    real_slot_by_name = {p.name: i for i, p in enumerate(real.players)}
    exact_slot_matches = {
        name for name in seed_numbers_matched if proj_slot_by_name[name] == real_slot_by_name[name]
    }

    print(f"Projected: {projected.tournament} {projected.year} ({len(projected.players)} players)")
    print(f"Real:      {real.tournament} {real.year} ({len(real.players)} players)")
    print(f"\nPlayer overlap: {len(overlap)}/{len(real_names)} real-draw players were in the projection")
    print(
        f"Seeded in both: {len(seed_names_matched)}/{len(real_seed_by_name)} real seeds appeared "
        f"in the projected seed list"
    )
    print(f"  ...with the same seed number: {len(seed_numbers_matched)}")
    print(f"  ...with the same seed number AND exact bracket slot: {len(exact_slot_matches)}")

    seed_mismatches = [
        (name, proj_seed_by_name[name], real_seed_by_name[name])
        for name in seed_names_matched
        if proj_seed_by_name[name] != real_seed_by_name[name]
    ]
    if seed_mismatches:
        print("\nSeed number mismatches (name: projected -> real):")
        for name, proj_seed, real_seed in sorted(seed_mismatches, key=lambda x: x[2]):
            print(f"  {name}: {proj_seed} -> {real_seed}")

    missing_from_projection = sorted(real_names - proj_names)
    if missing_from_projection:
        print(f"\nIn the real draw but not the projection ({len(missing_from_projection)}):")
        for name in missing_from_projection:
            print(f"  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build a projected bracket YAML")
    build_parser.add_argument("output_path", type=Path)
    build_parser.add_argument("--tour", choices=["atp", "wta"], required=True)
    build_parser.add_argument("--tournament", required=True)
    build_parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    build_parser.add_argument("--surface", required=True, choices=SURFACES)
    build_parser.add_argument("--draw-size", type=int, default=128)
    build_parser.add_argument("--seeds", type=int, default=32)
    build_parser.add_argument("--qualifiers", type=int, default=16)
    build_parser.add_argument("--wildcards", type=int, default=8)
    build_parser.add_argument("--overrides", type=Path, default=None,
                               help="path to an overrides YAML - see load_overrides docstring")
    build_parser.add_argument("--rng-seed", type=int, default=0,
                               help="seeds the unseeded-slot shuffle, for reproducible builds")

    compare_parser = subparsers.add_parser("compare", help="compare a projected draw against the real one")
    compare_parser.add_argument("projected_path", type=Path)
    compare_parser.add_argument("real_path", type=Path)

    simulate_parser = subparsers.add_parser(
        "simulate", help="run a full pre-tournament Monte Carlo simulation over a projected bracket"
    )
    simulate_parser.add_argument("bracket_path", type=Path)
    simulate_parser.add_argument("--simulations", type=int, default=N_SIMULATIONS)
    simulate_parser.add_argument("--seed", type=int, default=None,
                                  help="fix the RNG for a reproducible run - unseeded (random) by default")
    simulate_parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    if args.command == "build":
        bracket, stats = build_projected_bracket(
            args.tour, args.tournament, args.start_date, args.surface,
            draw_size=args.draw_size, seeds=args.seeds, qualifiers=args.qualifiers,
            wildcards=args.wildcards, overrides_path=args.overrides, rng_seed=args.rng_seed,
        )
        print(
            f"Built projected bracket: {bracket['tournament']} {bracket['year']} "
            f"({bracket['tour']}, {bracket['surface']}) - {len(bracket['players'])} players "
            f"({stats['seeded']} seeded, {stats['unseeded_direct']} unseeded direct entries, "
            f"{stats['qualifiers']} qualifier slots, {stats['wildcards']} wildcard slots)"
        )
        if stats["withdrawals_applied"] or stats["seed_overrides_applied"] or stats["name_overrides_applied"]:
            print(
                f"  Overrides applied: {stats['withdrawals_applied']} withdrawal(s), "
                f"{stats['seed_overrides_applied']} seed override(s), "
                f"{stats['name_overrides_applied']} name override(s)"
            )
        with open(args.output_path, "w", encoding="utf-8") as f:
            yaml.dump(bracket, f, sort_keys=False, allow_unicode=True)
        print(f"Wrote {args.output_path}")
    elif args.command == "compare":
        compare_brackets(args.projected_path, args.real_path)
    else:
        output_path, output = simulate_projected_draw(
            args.bracket_path, n_simulations=args.simulations, seed=args.seed, output_path=args.output
        )
        print(f"Wrote {output_path}")
        print(f"players: {len(output['players'])} alive (pre-tournament projection, all entrants included)")
