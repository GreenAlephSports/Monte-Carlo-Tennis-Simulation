"""Three-way WTA US Open title-probability comparison:
  1. OURS       - our match-count-weighted surface blend (SURFACE_BLEND_K) + surface-mismatch
                  damping + the full correction stack (rank-gap, layoff, recent-form, Platt,
                  upset-boost).
  2. SACKMANN+  - Sackmann's published flat 50/50 blend (see wta_sackmann_blend_comparison.py's
                  docstring/citation) + the SAME full correction stack layered on top identically.
  3. SACKMANN-ELO-ONLY - the SAME Sackmann 50/50 blend, but NO other corrections at all: no
                  rank-gap, no layoff, no recent-form, no Platt calibration, no upset-boost - pure
                  expected_score() off Sackmann-blended surface Elo. This isolates what his blend
                  ALONE predicts, uncontaminated by any of our other empirically-fit corrections.

All three use the SAME disclosed manual health override (Rybakina E., -100 Elo, mid-match
retirement) - that's a separate human-judgment input, not part of "the correction stack" being
toggled here, and its absence would make any single-elo comparison misleading (see the real bug this
caught in the two-way version tonight).

Usage:
    python model/research/wta_three_way_blend_comparison.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import TOUR_CONFIG, match_draw_to_ratings, order_by_draw_position, validate_bracket_structure, validate_draw  # noqa: E402
from bracket_schema import load_bracket_yaml  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from projected_draw_builder import DEFAULT_OVERRIDES_PATH, apply_health_adjustments, load_overrides  # noqa: E402
from simulate import N_SIMULATIONS, run_simulations_tracking_all_rounds  # noqa: E402
from wta_sackmann_blend_comparison import calculate_elo_ratings_sackmann_blend  # noqa: E402

BRACKET_PATH = Path(__file__).resolve().parent.parent.parent / "brackets" / "us_open_2026_wta_real.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
NAMED_PLAYERS = ["Osaka N.", "Andreeva M.", "Rybakina E."]

# EXPERIMENTAL sensitivity run, not a production value: overrides.yaml's disclosed -100 Elo
# override wasn't fit from data (see injury_retirement_penalty_test.py's FINAL VERDICT - no
# validated basis for a larger number was found), but the user asked to see -150 to -175 as a
# what-if range; -165 (midpoint) is used here, applied identically to every variant.
RYBAKINA_OVERRIDE_ELO = -165.0

NO_CORRECTIONS_KWARGS = {
    "use_rank_adjustment": False, "use_confidence_calibration": False,
    "use_layoff_adjustment": False, "use_recent_form_adjustment": False,
}


def prep_variant(label, ratings_df, players, byes, tour_config, bracket, overrides, path):
    ratings_df, health_applied = apply_health_adjustments(ratings_df, overrides["health_adjustments"])
    if health_applied:
        print(f"  [{label}] health adjustment(s): " +
              ", ".join(f"{a['player']} {a['elo_penalty']:+.0f}" for a in health_applied))
    draw, res, ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date)
    unmatched = [r for r in res if r["tier"] is None]
    if unmatched:
        raise RuntimeError(f"[{label}] unmatched: {[r['name'] for r in unmatched]}")
    ratings_df.to_csv(path, index=False)
    validate_draw(draw)
    return draw, ratings_df, path


def champ_probs(draw, byes, surface, ratings_path, use_upset_boost, win_probability_kwargs):
    depth_counts = run_simulations_tracking_all_rounds(
        draw, byes, surface, N_SIMULATIONS, ratings_path,
        use_upset_boost=use_upset_boost, win_probability_kwargs=win_probability_kwargs)
    champ = depth_counts[0]
    return {name: champ.get(name, 0) / N_SIMULATIONS for name in draw}


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    bracket = load_bracket_yaml(str(BRACKET_PATH))
    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    validate_bracket_structure(byes)
    tour_config = TOUR_CONFIG[bracket.tour]
    matches = load_matches_for_tour(bracket.tour)
    overrides = load_overrides(DEFAULT_OVERRIDES_PATH, bracket.tour)
    overrides["health_adjustments"] = dict(overrides["health_adjustments"])
    overrides["health_adjustments"]["Rybakina E."] = {
        **overrides["health_adjustments"]["Rybakina E."],
        "elo_penalty": RYBAKINA_OVERRIDE_ELO,
    }
    print(f"Using EXPERIMENTAL Rybakina E. override: {RYBAKINA_OVERRIDE_ELO:+.0f} Elo "
          f"(production overrides.yaml has -100; this run only)")

    ours_raw = calculate_elo_ratings(matches, bracket.start_date, tour=bracket.tour)
    sackmann_raw = calculate_elo_ratings_sackmann_blend(matches, bracket.start_date, tour=bracket.tour)

    print("Preparing ratings + resolving draw for each variant...")
    ours_draw, ours_df, ours_path = prep_variant(
        "ours", ours_raw, players, byes, tour_config, bracket, overrides,
        OUTPUT_DIR / "_3way_ours_ratings.csv")
    sackmann_full_draw, sackmann_full_df, sackmann_full_path = prep_variant(
        "sackmann+full", sackmann_raw.copy(), players, byes, tour_config, bracket, overrides,
        OUTPUT_DIR / "_3way_sackmann_full_ratings.csv")
    sackmann_elo_draw, sackmann_elo_df, sackmann_elo_path = prep_variant(
        "sackmann-elo-only", sackmann_raw.copy(), players, byes, tour_config, bracket, overrides,
        OUTPUT_DIR / "_3way_sackmann_eloonly_ratings.csv")

    print(f"\nRunning {N_SIMULATIONS} sims x 3 variants...")
    ours_p = champ_probs(ours_draw, byes, bracket.surface, ours_path, True, None)
    sackmann_full_p = champ_probs(sackmann_full_draw, byes, bracket.surface, sackmann_full_path, True, None)
    sackmann_elo_p = champ_probs(sackmann_elo_draw, byes, bracket.surface, sackmann_elo_path, False, NO_CORRECTIONS_KWARGS)

    def ranked(d):
        r = sorted(d.items(), key=lambda kv: -kv[1])
        return r, {name: i + 1 for i, (name, _) in enumerate(r)}

    ours_ranked, ours_rank = ranked(ours_p)
    sf_ranked, sf_rank = ranked(sackmann_full_p)
    se_ranked, se_rank = ranked(sackmann_elo_p)

    top_names = [n for n, _ in ours_ranked[:15]]
    for n, _ in sf_ranked[:15]:
        if n not in top_names:
            top_names.append(n)
    for n, _ in se_ranked[:15]:
        if n not in top_names:
            top_names.append(n)

    print(f"\n{'=' * 130}\nTHREE-WAY: OURS  |  SACKMANN-BLEND + full corrections  |  SACKMANN-BLEND, elo only (no corrections)\n{'=' * 130}")
    print(f"{'player':<20}{'ours':>8}{'r':>4}{'sack+full':>11}{'r':>4}{'sack-elo':>10}{'r':>4}")
    for name in top_names:
        op, or_ = ours_p.get(name, 0.0), ours_rank.get(name)
        sfp, sfr = sackmann_full_p.get(name, 0.0), sf_rank.get(name)
        sep, ser = sackmann_elo_p.get(name, 0.0), se_rank.get(name)
        flag = "  <<<" if name in NAMED_PLAYERS else ""
        print(f"{name:<20}{op:>7.1%}{or_ or 0:>4}{sfp:>10.1%}{sfr or 0:>4}{sep:>9.1%}{ser or 0:>4}{flag}")

    print(f"\n{'=' * 130}\nNamed-player detail (overall_elo / hard_elo per variant)\n{'=' * 130}")
    for name in NAMED_PLAYERS:
        row_o = ours_df.set_index("player").loc[name] if name in ours_df["player"].values else None
        row_s = sackmann_full_df.set_index("player").loc[name] if name in sackmann_full_df["player"].values else None
        if row_o is None or row_s is None:
            print(f"  {name}: missing from one variant's draw")
            continue
        print(f"  {name}:")
        print(f"    ours       : overall={row_o['overall_elo']:.1f} hard_elo={row_o['hard_elo']:.1f}  "
              f"p_champ={ours_p.get(name, 0):.1%} (rank {ours_rank.get(name)})")
        print(f"    sackmann+  : overall={row_s['overall_elo']:.1f} hard_elo={row_s['hard_elo']:.1f}  "
              f"p_champ={sackmann_full_p.get(name, 0):.1%} (rank {sf_rank.get(name)})")
        print(f"    sack-elo   : overall={row_s['overall_elo']:.1f} hard_elo={row_s['hard_elo']:.1f}  "
              f"p_champ={sackmann_elo_p.get(name, 0):.1%} (rank {se_rank.get(name)})")


if __name__ == "__main__":
    main()
