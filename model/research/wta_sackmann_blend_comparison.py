"""Side-by-side WTA US Open title-probability comparison: our production surface-blend methodology
(match-count-weighted blend toward overall_elo, SURFACE_BLEND_K=7, plus tonight's surface-mismatch
damping) vs. Jeff Sackmann's PUBLISHED blend methodology (Tennis Abstract, "An Introduction to
Tennis Elo", https://www.tennisabstract.com/blog/2019/12/03/an-introduction-to-tennis-elo/) - a
flat, constant 50/50 blend of single-surface Elo and overall Elo, regardless of how many matches a
player has on that surface. Sackmann explicitly tested surface-specific/sample-size-dependent
weighting (he expected grass, with fewer matches, might need a different blend) and found 50/50
"worked for each surface" - i.e. his published choice is deliberately NOT match-count-weighted the
way ours is.

DISCLOSED LIMITATION: Sackmann's public writeup states the blend WEIGHT (50/50, flat) but does not
publish his exact K-factor or single-surface-Elo initialization/update mechanics. This script
reproduces the ONE thing that's actually documented and verified (the flat 50/50 blend weight,
replacing our match_count/(match_count+K) weight) while reusing our own K_FACTOR/STARTING_ELO/
online-update mechanics for the underlying single-surface Elo itself, since that part isn't publicly
specified and isn't the subject of this comparison. This is a reconstruction of his documented BLEND
CHOICE, not a byte-for-byte reproduction of his full system - treat it as "what does a flat,
sample-size-blind 50/50 blend do to our field," not "this is literally Sackmann's live rating."

No surface-mismatch damping is applied to the Sackmann variant - that fix targets a specific failure
mode of OUR match-count-weighted blend (a thin surface sample letting the blend swing too far); a
flat 50/50 blend doesn't have that same failure mode (it can never lean more than 50% on a
thin-sample surface rating), so applying our damping on top of a different blend philosophy would
conflate the two things this comparison is trying to separate.

Every other correction (rank-gap, layoff, recent-form, confidence-calibration, upset-boost) is
applied identically to both variants - only the surface_elo/hard_elo values feeding into
win_probability() differ.

Usage:
    python model/research/wta_sackmann_blend_comparison.py
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import TOUR_CONFIG, match_draw_to_ratings, order_by_draw_position, split_byes, validate_bracket_structure, validate_draw  # noqa: E402
from bracket_schema import load_bracket_yaml  # noqa: E402
from elo_ratings import K_FACTOR, STARTING_ELO, SURFACES, apply_training_window, calculate_elo_ratings, compute_recent_form_residuals, expected_score, load_matches_for_tour  # noqa: E402
from projected_draw_builder import DEFAULT_OVERRIDES_PATH, apply_health_adjustments, load_overrides  # noqa: E402
from simulate import N_SIMULATIONS, run_simulations_tracking_milestones  # noqa: E402

BRACKET_PATH = Path(__file__).resolve().parent.parent.parent / "brackets" / "us_open_2026_wta_real.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
NAMED_PLAYERS = ["Osaka N.", "Andreeva M."]
SACKMANN_BLEND_WEIGHT = 0.5  # flat, per his published testing - not match-count-dependent


def calculate_elo_ratings_sackmann_blend(df, cutoff_date, tour=None):
    """Identical to elo_ratings.calculate_elo_ratings EXCEPT the surface-elo blend weight is a flat
    0.5 (Sackmann's published 50/50) instead of match_count/(match_count+SURFACE_BLEND_K), and no
    surface-mismatch damping is applied (see module docstring). decay3/WTA windowing kept IDENTICAL
    to production so the only variable under test is the blend weight itself."""
    recent_form = compute_recent_form_residuals(df, cutoff_date)
    windowed = apply_training_window(df, cutoff_date)
    df = windowed.sort_values("Date", kind="stable")

    overall_elo = {}
    surface_elo = {surface: {} for surface in SURFACES}
    current_rank = {}
    has_rank_columns = {"Rank_1", "Rank_2"}.issubset(df.columns)
    last_match_date = {}
    cutoff_ts = pd.Timestamp(cutoff_date)

    for row in df.itertuples(index=False):
        p1, p2, winner, surface = row.Player_1, row.Player_2, row.Winner, row.Surface
        last_match_date[p1] = row.Date
        last_match_date[p2] = row.Date

        overall_elo.setdefault(p1, STARTING_ELO)
        overall_elo.setdefault(p2, STARTING_ELO)
        score_p1 = 1.0 if winner == p1 else 0.0
        expected_p1 = expected_score(overall_elo[p1], overall_elo[p2])
        overall_elo[p1] += K_FACTOR * (score_p1 - expected_p1)
        overall_elo[p2] += K_FACTOR * ((1 - score_p1) - (1 - expected_p1))

        if surface in SURFACES:
            ratings = surface_elo[surface]
            ratings.setdefault(p1, STARTING_ELO)
            ratings.setdefault(p2, STARTING_ELO)
            expected_p1_surface = expected_score(ratings[p1], ratings[p2])
            ratings[p1] += K_FACTOR * (score_p1 - expected_p1_surface)
            ratings[p2] += K_FACTOR * ((1 - score_p1) - (1 - expected_p1_surface))

        if has_rank_columns:
            if row.Rank_1 > 0:
                current_rank[p1] = row.Rank_1
            if row.Rank_2 > 0:
                current_rank[p2] = row.Rank_2

    players = sorted(overall_elo.keys())
    records = []
    for player in players:
        last_date = last_match_date.get(player)
        record = {
            "player": player,
            "overall_elo": overall_elo[player],
            "current_rank": current_rank.get(player),
            "days_since_last_match": (cutoff_ts - last_date).days if last_date is not None else None,
            "recent_form_residual": recent_form.get(player),
        }
        for surface in SURFACES:
            raw_elo = surface_elo[surface].get(player, STARTING_ELO)
            final_elo = SACKMANN_BLEND_WEIGHT * raw_elo + (1 - SACKMANN_BLEND_WEIGHT) * overall_elo[player]
            record[f"{surface.lower()}_elo"] = final_elo
            record[f"{surface.lower()}_matches"] = None
        records.append(record)

    columns = ["player", "hard_elo", "clay_elo", "grass_elo", "overall_elo",
               "hard_matches", "clay_matches", "grass_matches", "current_rank",
               "days_since_last_match", "recent_form_residual"]
    return pd.DataFrame.from_records(records, columns=columns)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    bracket = load_bracket_yaml(str(BRACKET_PATH))
    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    non_bye_count, bye_count = validate_bracket_structure(byes)
    print(f"Bracket: {len(players)} players, {bye_count} byes")

    tour_config = TOUR_CONFIG[bracket.tour]
    matches = load_matches_for_tour(bracket.tour)

    ours_df = calculate_elo_ratings(matches, bracket.start_date, tour=bracket.tour)
    sackmann_df = calculate_elo_ratings_sackmann_blend(matches, bracket.start_date, tour=bracket.tour)

    # apply the SAME disclosed manual health override(s) to both variants - this is a separate,
    # human-judgment correction (production behavior via export_bracket_json/simulate_projected_
    # draw), independent of surface-blend methodology, and must match Part 2's table exactly for
    # an apples-to-apples "ours" baseline.
    overrides = load_overrides(DEFAULT_OVERRIDES_PATH, bracket.tour)
    ours_df, ours_health_applied = apply_health_adjustments(ours_df, overrides["health_adjustments"])
    sackmann_df, _ = apply_health_adjustments(sackmann_df, overrides["health_adjustments"])
    if ours_health_applied:
        print(f"\nHealth adjustment(s) applied to both variants ({len(ours_health_applied)}):")
        for adj in ours_health_applied:
            print(f"  {adj['player']}: {adj['elo_penalty']:+.0f} Elo - {adj['reason']}")

    ours_draw, ours_res, ours_df = match_draw_to_ratings(
        players, ours_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date)
    sackmann_draw, sackmann_res, sackmann_df = match_draw_to_ratings(
        players, sackmann_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date)

    unmatched = [r for r in ours_res if r["tier"] is None]
    if unmatched:
        raise RuntimeError(f"Unmatched (ours): {[r['name'] for r in unmatched]}")
    if ours_draw != sackmann_draw:
        print("WARNING: resolved draw name lists differ between variants - comparison below aligns by name anyway")

    ours_path = OUTPUT_DIR / "_sackmann_compare_ours_ratings.csv"
    sackmann_path = OUTPUT_DIR / "_sackmann_compare_sackmann_ratings.csv"
    ours_df.to_csv(ours_path, index=False)
    sackmann_df.to_csv(sackmann_path, index=False)

    validate_draw(ours_draw)
    print(f"\nRunning {N_SIMULATIONS} simulations for OUR pipeline...")
    ours_champ, _, _ = run_simulations_tracking_milestones(ours_draw, byes, {}, bracket.surface, N_SIMULATIONS, ours_path)
    print(f"Running {N_SIMULATIONS} simulations for SACKMANN-blend variant...")
    sackmann_champ, _, _ = run_simulations_tracking_milestones(sackmann_draw, byes, {}, bracket.surface, N_SIMULATIONS, sackmann_path)

    ours_p = {name: ours_champ.get(name, 0) / N_SIMULATIONS for name in ours_draw}
    sackmann_p = {name: sackmann_champ.get(name, 0) / N_SIMULATIONS for name in sackmann_draw}

    ours_ranked = sorted(ours_p.items(), key=lambda kv: -kv[1])
    sackmann_ranked = sorted(sackmann_p.items(), key=lambda kv: -kv[1])
    ours_rank = {name: i + 1 for i, (name, _) in enumerate(ours_ranked)}
    sackmann_rank = {name: i + 1 for i, (name, _) in enumerate(sackmann_ranked)}

    top15_names = [name for name, _ in ours_ranked[:15]]
    for name, _ in sackmann_ranked[:15]:
        if name not in top15_names:
            top15_names.append(name)

    print(f"\n{'=' * 110}\nSIDE-BY-SIDE: OURS (full pipeline incl. damping) vs SACKMANN-BLEND (flat 50/50, no damping)\n{'=' * 110}")
    print(f"{'player':<20}{'ours rank':>10}{'ours p_champ':>14}{'sack rank':>10}{'sack p_champ':>14}{'rank shift':>12}{'p_champ shift':>15}")
    for name in top15_names:
        or_, op = ours_rank.get(name), ours_p.get(name, 0.0)
        sr, sp = sackmann_rank.get(name), sackmann_p.get(name, 0.0)
        shift = (sr - or_) if (or_ and sr) else None
        flag = "  <<<" if name in NAMED_PLAYERS else ""
        print(f"{name:<20}{or_:>10}{op:>14.1%}{sr:>10}{sp:>14.1%}{('%+d' % shift if shift is not None else 'n/a'):>12}{sp - op:>+15.1%}{flag}")

    print(f"\n{'=' * 110}\nNamed-player detail\n{'=' * 110}")
    for name in NAMED_PLAYERS:
        if name not in ours_p:
            print(f"  {name}: not in draw")
            continue
        oh = ours_df.set_index("player").loc[name, "hard_elo"] if name in ours_df["player"].values else None
        sh = sackmann_df.set_index("player").loc[name, "hard_elo"] if name in sackmann_df["player"].values else None
        print(f"  {name}: ours rank={ours_rank.get(name)} p_champ={ours_p.get(name, 0):.1%} hard_elo={oh:.1f}  |  "
              f"sackmann rank={sackmann_rank.get(name)} p_champ={sackmann_p.get(name, 0):.1%} hard_elo={sh:.1f}")


if __name__ == "__main__":
    main()
