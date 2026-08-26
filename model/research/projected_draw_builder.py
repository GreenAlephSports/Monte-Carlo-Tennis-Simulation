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

    # simulate a projected bracket (refuses to run if the bracket file predates overrides.yaml's
    # last edit - see check_bracket_not_stale - rebuild first if it does):
    python model/research/projected_draw_builder.py simulate projected.yaml [--overrides overrides.yaml]

    # flag top seeds whose most recent tracked match ended in retirement/walkover (visibility
    # only - also runs automatically as part of `simulate`, see health_check_top_seeds):
    python model/research/projected_draw_builder.py health-check projected.yaml [--top-n 20]
"""
import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import (  # noqa: E402
    TOUR_CONFIG, _split_csv_name, match_draw_to_ratings, match_name_to_pool, order_by_draw_position,
    validate_bracket_structure, validate_draw,
)
from bracket_schema import load_bracket_yaml  # noqa: E402
from calibration_log import load_existing_log  # noqa: E402
from elo_ratings import SURFACES, calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from live_scores import RETIREMENT_STATUS_NAMES  # noqa: E402
from simulate import N_SIMULATIONS, run_simulations_tracking_milestones  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent.parent.parent / "overrides.yaml"

# Seeding-specific activity gate - NOT a change to elo_ratings.LOOKBACK_YEARS (that 5-year window
# was tested and deliberately left alone tonight - see lookback_full_historical_test.py). A player
# can still sit inside that broader 5-year training window - which exists to keep their Elo/rating
# available for anyone who ever plays them - while being genuinely retired or long-inactive, since
# current_rank alone carries no "still on tour" concept. Confirmed via a real case: Ash Barty
# (retired 2022, last real match 2022-01-29) still carried current_rank=1 into a 2026 projected
# draw - seeded #1 - purely because that last match happened to fall just inside the 5-year
# window (1675 days before the 2026 cutoff). 548 days (~18 months) is generous enough to keep any
# player on a real injury/personal absence (the next-longest real gap found in either tour's real
# current top 20, Rune H. at 318 days, comfortably clears it) while excluding a multi-year-retired
# player like Barty.
ACTIVITY_CUTOFF_DAYS = 548


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


def current_rankings(tour, cutoff_date, activity_cutoff_days=ACTIVITY_CUTOFF_DAYS):
    """Current-rank-ordered list of (rank, ratings-csv-name) as of cutoff_date, freshly computed
    via elo_ratings.py's own pipeline (live Kaggle pull, local-CSV fallback) rather than read off
    a possibly-stale output/ CSV. Players with no recorded current_rank (no Rank_1/Rank_2 ever
    seen for them in the training window) are excluded - there's no signal to seed/order them by.

    ALSO excludes anyone whose most recent real match (days_since_last_match) is older than
    activity_cutoff_days - see ACTIVITY_CUTOFF_DAYS module comment for why this exists (a real,
    confirmed bug: Ash Barty, retired since 2022, was seeded #1 in a 2026 projected draw with no
    override needed to trigger it). Pass activity_cutoff_days=None to disable this gate entirely
    (e.g. for compare_brackets-style historical reproduction, where 'as it really would have
    looked' matters more than filtering staleness).

    Deduped on (lastname, initials) via bracket.py's own name splitter - the source Kaggle data
    occasionally spells the same player two different ways across rows (e.g. 'Tirante T. A.' vs
    'Tirante T.A.'), which without this would seed the same real player twice under two names
    that bracket.py's own fuzzy matcher later collapses back into one, producing a duplicate-
    player draw. Keeps whichever spelling comes first in rank order (i.e. its better/lowest rank)."""
    matches = load_matches_for_tour(tour)
    ratings = calculate_elo_ratings(matches, cutoff_date)
    ranked = ratings.dropna(subset=["current_rank"]).copy()

    if activity_cutoff_days is not None:
        stale = ranked[ranked["days_since_last_match"] > activity_cutoff_days]
        if len(stale):
            stale_desc = ", ".join(
                f"{row.player} (rank {int(row.current_rank)}, {int(row.days_since_last_match)}d since last match)"
                for row in stale.sort_values("current_rank").itertuples()
            )
            print(f"current_rankings: excluding {len(stale)} player(s) with a current_rank but no "
                  f"real match in > {activity_cutoff_days} days (likely retired/long-inactive): {stale_desc}")
        ranked = ranked[ranked["days_since_last_match"] <= activity_cutoff_days]

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


def load_overrides(path, tour):
    """Override YAML schema - top-level keyed by tour ("atp"/"wta", case-insensitive), each with
    the same three optional keys:

        atp:
          withdrawals:
            - "Nadal R."           # ratings-csv-style name. Dropped from the ranked pool
                                    # entirely before seeding/slotting, so the next-ranked player
                                    # (or alternate) automatically moves up to fill their slot -
                                    # same effect a real tour withdrawal + alternate has.
          seed_overrides:
            "Alcaraz C.": 1        # force this player into seed slot 1, bumping whoever the
                                    # rank-order would have put there back into the unseeded pool.
          name_overrides:
            "Qualifier 3": "Some Player X."   # rename any single entry (a generic qualifier/
                                    # wildcard placeholder, or a ranked entrant) once a real name
                                    # is known - matched by whatever name it currently carries.
          health_adjustments:
            "Rybakina E.":
              elo_penalty: -100     # subtracted from overall_elo AND every surface elo before
                                    # simulation - a manual, disclosed judgment call for a known
                                    # real-world fact the model structurally cannot see (an injury,
                                    # a retirement, anything current_rank/Elo has no way to reflect
                                    # on its own), NOT a statistical correction - no magnitude is
                                    # ever fit from data here, a person decided this number. Same
                                    # philosophy as a withdrawal, just a penalty instead of a
                                    # removal. `reason` is required and is carried through to the
                                    # simulation export so this is never silently indistinguishable
                                    # from a real, unadjusted model output - see
                                    # simulate_projected_draw/apply_health_adjustments.
              reason: "Retired mid-match at Cincinnati (2026-08-20), 3 weeks pre-tournament."
        wta:
          withdrawals: []
          ...

    Tour-scoped (rather than one flat list shared by both tours) because build_projected_bracket
    now resolves every withdrawal name through match_name_to_pool's fuzzy tiers and errors loudly
    on anything unresolved (see build_projected_bracket) - a single shared list would force every
    ATP-only withdrawal to also resolve against the WTA pool (and vice versa), which is not just
    noisy but unsafe: a real case, "Nava E." (Emilio Nava, ATP) fuzzy-matched "Navarro E." (Emma
    Navarro, WTA) via the glued-lastname-prefix tier, which would have silently withdrawn a real,
    active WTA player who never withdrew from anything.
    """
    empty = {"withdrawals": [], "seed_overrides": {}, "name_overrides": {}, "health_adjustments": {}}
    if path is None:
        return empty
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    tour_data = data.get(tour.lower()) or {}
    health_adjustments = dict(tour_data.get("health_adjustments") or {})
    for name, entry in health_adjustments.items():
        if not isinstance(entry, dict) or "elo_penalty" not in entry or "reason" not in entry:
            raise ValueError(
                f"health_adjustments entry for {name!r} must be a dict with both 'elo_penalty' "
                f"and 'reason' keys (reason is required - this is a disclosed judgment call, not "
                f"a silent adjustment), got {entry!r}"
            )
    return {
        "withdrawals": list(tour_data.get("withdrawals") or []),
        "seed_overrides": dict(tour_data.get("seed_overrides") or {}),
        "name_overrides": dict(tour_data.get("name_overrides") or {}),
        "health_adjustments": health_adjustments,
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
    overrides_path=None, rng_seed=0, activity_cutoff_days=ACTIVITY_CUTOFF_DAYS,
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

    overrides = load_overrides(overrides_path, tour)

    ranked = current_rankings(tour, start_date, activity_cutoff_days=activity_cutoff_days)
    all_names = [name for _rank, name in ranked]

    # withdrawal names come from a manually-typed overrides.yaml, which won't always share the
    # ratings csv's exact spelling (e.g. "Davidovich F." for "Davidovich Fokina A.") - resolve
    # each one through the same tiered fuzzy matcher bracket.py uses everywhere else instead of a
    # brittle exact-string check, and fail loudly (not silently) if one can't be resolved to
    # anyone in the current field, since a silently-unmatched withdrawal means that player stays
    # in the draw unremoved with no indication anything went wrong.
    resolved_withdrawals = {}
    unresolved_withdrawals = []
    for raw_name in overrides["withdrawals"]:
        match = match_name_to_pool(raw_name, all_names)
        if match is None:
            unresolved_withdrawals.append(raw_name)
        else:
            resolved_withdrawals[raw_name] = match
    if unresolved_withdrawals:
        raise ValueError(
            f"overrides.yaml withdrawal(s) could not be matched to anyone in the current {tour} "
            f"field: {unresolved_withdrawals} - check spelling/format against "
            f"output/player_elo_ratings_{tour.lower()}.csv"
        )

    pool_names = [name for name in all_names if name not in resolved_withdrawals.values()]

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


def check_bracket_not_stale(bracket_path, overrides_path=DEFAULT_OVERRIDES_PATH):
    """Refuses to let a projected bracket be simulated if it predates the overrides file it's
    supposed to reflect - overrides.yaml (withdrawals/seed/name overrides) is only ever applied at
    BUILD time (see build_projected_bracket), not at simulate time, so a bracket YAML built before
    a later edit to overrides.yaml (e.g. a new withdrawal added after the bracket was last built)
    would otherwise get silently simulated as if that edit didn't exist - real players who've
    actually withdrawn would still be in the field with no indication anything is wrong. Compares
    file modification times; a no-op if overrides_path doesn't exist (nothing for the bracket to
    be stale against) or is None (explicitly opted out)."""
    if overrides_path is None:
        return
    overrides_path = Path(overrides_path)
    if not overrides_path.exists():
        return
    bracket_mtime = Path(bracket_path).stat().st_mtime
    overrides_mtime = overrides_path.stat().st_mtime
    if bracket_mtime < overrides_mtime:
        bracket_dt = datetime.fromtimestamp(bracket_mtime).strftime("%Y-%m-%d %H:%M:%S")
        overrides_dt = datetime.fromtimestamp(overrides_mtime).strftime("%Y-%m-%d %H:%M:%S")
        raise RuntimeError(
            f"Refusing to simulate {bracket_path}: it was last built {bracket_dt}, which is OLDER "
            f"than {overrides_path} (last edited {overrides_dt}). This bracket predates that "
            f"override change and does not reflect it (e.g. a withdrawal added since the bracket "
            f"was built would still show that player as active). Rebuild it first with "
            f"`python model/research/projected_draw_builder.py build ... --overrides "
            f"{overrides_path}` before simulating."
        )


def health_check_top_seeds(seeded_players, top_n=20, log_df=None):
    """For each of the top `top_n` seeded players (seed number <= top_n), looks up their most
    recent real match in the persistent calibration log (output/calibration_log.csv, accumulated
    by calibration_log.py from both live ESPN and concluded-Kaggle sources) and flags any whose
    most recent LIVE-tracked match ended in a retirement or walkover - ESPN's own real
    status.type.name (STATUS_RETIRED/STATUS_WALKOVER), the same signal live_scores.py's
    ended_by_retirement already uses, not inferred from the score.

    Visibility only, same principle as the real Sinner withdrawal handled earlier: this never
    excludes the player, adjusts their Elo, or touches the bracket automatically - it just prints
    what a human would need to know to decide whether a manual override (load_overrides'
    withdrawals/seed_overrides) belongs in this bracket, same as that real withdrawal did.

    seeded_players: iterable of (seed, name) pairs, seed 1-indexed, name in the SAME format the
    calibration log itself uses (ratings-csv style, e.g. "Rybakina E."). Returns the list of flag
    dicts (empty if none), after printing the warning block.

    Coverage note: the calibration log's status_detail field only exists for live_espn-sourced
    rows - kaggle_concluded rows (the historical Kaggle dataset has no retirement/walkover marker
    at all - see calibration_log.py's own LOG_COLUMNS comment) carry status_detail=NaN, genuinely
    unknown rather than "confirmed normal finish". A top seed whose most recent tracked match is a
    kaggle_concluded row, or who has no log entry at all (never tracked live), is reported
    separately below the flag list rather than silently treated as healthy.
    """
    if log_df is None:
        log_df = load_existing_log()

    top_seeds = sorted(
        ((seed, name) for seed, name in seeded_players if seed is not None and seed <= top_n),
        key=lambda x: x[0],
    )

    flags, unknown_status, no_log_entry = [], [], []
    for seed, name in top_seeds:
        player_rows = log_df[(log_df["player_a"] == name) | (log_df["player_b"] == name)]
        if len(player_rows) == 0:
            no_log_entry.append((seed, name))
            continue
        most_recent = player_rows.sort_values("date").iloc[-1]
        status = most_recent["status_detail"]
        if pd.isna(status):
            unknown_status.append((seed, name, most_recent["date"], most_recent["source"]))
            continue
        if status in RETIREMENT_STATUS_NAMES:
            opponent = most_recent["player_b"] if most_recent["player_a"] == name else most_recent["player_a"]
            flags.append({
                "seed": seed, "player": name, "status_detail": status, "date": most_recent["date"],
                "tournament": most_recent["tournament"], "round_label": most_recent["round_label"],
                "opponent": opponent,
            })

    print(f"\n{'=' * 90}\nPRE-DRAW HEALTH CHECK - top {top_n} seeds' most recent tracked match\n{'=' * 90}")
    if flags:
        print(f"\n*** {len(flags)} player(s) flagged - most recent TRACKED match ended in retirement/walkover ***")
        for f in flags:
            print(f"  [seed {f['seed']}] {f['player']}: {f['status_detail']} vs {f['opponent']} in "
                  f"{f['tournament']} {f['round_label']} on {pd.Timestamp(f['date']).date()} - "
                  f"consider a manual override (see load_overrides docstring); NOT auto-applied here")
    else:
        print("No top-seed players flagged - no retirement/walkover found as anyone's most recent tracked match.")

    if unknown_status:
        print(f"\n{len(unknown_status)} top-seed player(s) have a tracked match but its outcome type is "
              f"unknown (kaggle_concluded source has no status_detail field):")
        for seed, name, date, source in unknown_status:
            print(f"  [seed {seed}] {name}: most recent tracked match {pd.Timestamp(date).date()} "
                  f"(source={source}), status unknown")

    if no_log_entry:
        print(f"\n{len(no_log_entry)} top-seed player(s) have NO match in the calibration log at all "
              f"(never tracked by calibration_log.py) - no information available:")
        for seed, name in no_log_entry:
            print(f"  [seed {seed}] {name}")

    return flags


def apply_health_adjustments(ratings_df, health_adjustments):
    """Applies each health_adjustments entry's elo_penalty to overall_elo AND every surface elo
    column (hard/clay/grass) for the matched player - a flat point subtraction, not a percentage or
    rank-conditional scaling, since (see load_overrides' docstring) this is a disclosed human
    judgment call with an explicitly chosen magnitude, not something fit from data. Player names are
    resolved via match_name_to_pool against ratings_df's own pool (the same fuzzy tiers withdrawals
    use), and an unresolved name errors loudly rather than silently no-op'ing - a health adjustment
    that silently fails to apply is worse than no adjustment at all, since nothing would ever flag
    that it didn't take effect.

    Returns (ratings_df, applied) where applied is a list of {"player": csv_name, "elo_penalty":
    ..., "reason": ...} dicts, for simulate_projected_draw's export JSON to flag explicitly - so an
    adjusted player's numbers are never silently indistinguishable from an unadjusted model output.
    """
    if not health_adjustments:
        return ratings_df, []
    ratings_df = ratings_df.copy()
    pool_names = list(ratings_df["player"])
    elo_columns = [c for c in ("overall_elo", "hard_elo", "clay_elo", "grass_elo") if c in ratings_df.columns]

    applied, unresolved = [], []
    for raw_name, entry in health_adjustments.items():
        match = match_name_to_pool(raw_name, pool_names)
        if match is None:
            unresolved.append(raw_name)
            continue
        penalty = entry["elo_penalty"]
        mask = ratings_df["player"] == match
        for col in elo_columns:
            ratings_df.loc[mask, col] = ratings_df.loc[mask, col] + penalty
        applied.append({"player": match, "elo_penalty": penalty, "reason": entry["reason"]})

    if unresolved:
        raise ValueError(
            f"overrides.yaml health_adjustments name(s) could not be matched to anyone in the "
            f"current field: {unresolved} - check spelling/format"
        )
    return ratings_df, applied


def simulate_projected_draw(
    bracket_path, n_simulations=N_SIMULATIONS, seed=None, output_path=None,
    overrides_path=DEFAULT_OVERRIDES_PATH,
):
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
    check_bracket_not_stale(bracket_path, overrides_path=overrides_path)

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

    overrides = load_overrides(overrides_path, bracket.tour)
    ratings_df, health_adjustments_applied = apply_health_adjustments(ratings_df, overrides["health_adjustments"])
    if health_adjustments_applied:
        print(f"\nHealth adjustment(s) applied ({len(health_adjustments_applied)}):")
        for adj in health_adjustments_applied:
            print(f"  {adj['player']}: {adj['elo_penalty']:+.0f} Elo - {adj['reason']}")

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

    # pre-draw health check: top 20 seeds, flagged if their most recent TRACKED match ended in a
    # retirement/walkover - visibility only (see health_check_top_seeds docstring), run against
    # the CANONICAL resolved names in `draw` (same order as `players`) so it matches whatever
    # naming the calibration log itself was written with, not the bracket file's raw spelling.
    seeded_players = [(p.seed, name) for p, name in zip(players, draw)]
    health_check_top_seeds(seeded_players, top_n=20)

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

    health_adjustment_by_player = {adj["player"]: adj for adj in health_adjustments_applied}
    players_out = [
        {
            "player": name,
            "quarter": quarter_by_name[name],
            "p_champ": round(champ_counts.get(name, 0) / n_simulations, 3),
            "p_sf": round(sf_counts.get(name, 0) / n_simulations, 3),
            "p_final": round(final_counts.get(name, 0) / n_simulations, 3),
            # present ONLY for a player with an active manual override, so this is never silently
            # indistinguishable from an unadjusted model output - see apply_health_adjustments.
            **({"health_adjustment": {
                "elo_penalty": health_adjustment_by_player[name]["elo_penalty"],
                "reason": health_adjustment_by_player[name]["reason"],
            }} if name in health_adjustment_by_player else {}),
        }
        for name in draw
    ]
    players_out.sort(key=lambda r: -r["p_champ"])

    tour_word = "men" if bracket.tour == "ATP" else "women"
    tournament_slug = bracket.tournament.lower().replace(" ", "-")
    warnings = [
        "Pre-tournament projection: no real draw/results exist yet, so 'player' is this "
        "project's internal ratings-csv name (not an ESPN displayName), and every entrant in "
        "the draw is included as alive.",
    ]
    if health_adjustments_applied:
        warnings.append(
            "One or more players carry a manual health_adjustment (see overrides.yaml) - a "
            "disclosed human judgment call, not a statistical model output. Affected players are "
            "flagged individually under players[].health_adjustment; see also the top-level "
            "health_adjustments list."
        )
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
        "health_adjustments": health_adjustments_applied,
        "warnings": warnings,
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
    build_parser.add_argument(
        "--activity-cutoff-days", type=int, default=ACTIVITY_CUTOFF_DAYS,
        help="exclude players from seeding/ranking whose most recent real match is older than "
             "this many days (catches retired/long-inactive players like Ash Barty carrying a "
             "stale current_rank - see ACTIVITY_CUTOFF_DAYS docstring); pass -1 to disable")

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
    simulate_parser.add_argument(
        "--overrides", type=Path, default=DEFAULT_OVERRIDES_PATH,
        help="overrides YAML this bracket is supposed to reflect - refuses to simulate if the "
             "bracket file is older than this file's last edit (see check_bracket_not_stale); "
             "pass a nonexistent/None-like path to skip the check entirely")

    health_parser = subparsers.add_parser(
        "health-check",
        help="flag top-seed players whose most recent tracked match ended in retirement/walkover "
             "(fast, standalone - doesn't run a simulation; `simulate` also runs this automatically)",
    )
    health_parser.add_argument("bracket_path", type=Path)
    health_parser.add_argument("--top-n", type=int, default=20)

    args = parser.parse_args()

    if args.command == "build":
        activity_cutoff_days = None if args.activity_cutoff_days < 0 else args.activity_cutoff_days
        bracket, stats = build_projected_bracket(
            args.tour, args.tournament, args.start_date, args.surface,
            draw_size=args.draw_size, seeds=args.seeds, qualifiers=args.qualifiers,
            wildcards=args.wildcards, overrides_path=args.overrides, rng_seed=args.rng_seed,
            activity_cutoff_days=activity_cutoff_days,
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
    elif args.command == "health-check":
        bracket = load_bracket_yaml(args.bracket_path)
        seeded_players = [(p.seed, p.name) for p in bracket.players if p.seed is not None]
        health_check_top_seeds(seeded_players, top_n=args.top_n)
    else:
        output_path, output = simulate_projected_draw(
            args.bracket_path, n_simulations=args.simulations, seed=args.seed, output_path=args.output,
            overrides_path=args.overrides,
        )
        print(f"Wrote {output_path}")
        print(f"players: {len(output['players'])} alive (pre-tournament projection, all entrants included)")
