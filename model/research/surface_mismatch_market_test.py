"""Tests whether the market systematically underweights surface-specific mismatches - i.e. whether
it prices players off general reputation/overall Elo rather than being properly surface-aware. This
is a mechanistically distinct claim from pedigree_market_premium_test.py's "market loves famous
names" hypothesis: a player can have zero pedigree and still be under- or over-priced purely because
the market isn't adjusting for "this player is much better/worse on THIS surface than their overall
level suggests."

Reuses the exact same 23-tournament, real tennis-data.co.uk closing-odds dataset (both tours, 2026
season) pedigree_market_premium_test.py built and validated - same fetch/cache, same neutral
alphabetical a/b framing (never winner-anchored), same OLS + player-clustered-bootstrap win-rate
methodology, so results are directly comparable in rigor and honesty standard.

Surface mismatch, per player per match: (that player's blended surface_elo for the match's own
surface) - (their overall_elo), both from the same frozen-at-event-start-date ratings snapshot. This
is a real, signed, checkable quantity already computed by production's own calculate_elo_ratings -
positive means the player rates ABOVE their general level on this surface (a specialist edge),
negative means BELOW it (a mismatch weakness). mismatch_diff = mismatch_a - mismatch_b.

Control variable is OVERALL Elo diff, not surface Elo diff (unlike the pedigree test's elo_diff).
This is deliberate, not an inconsistency: surface_elo_diff = overall_elo_diff + mismatch_diff by
construction (the blend formula in elo_ratings.py), so controlling for surface_elo_diff would make
mismatch_diff collinear with its own control and mechanically absorbed - a meaningless regression.
Controlling for overall_elo_diff instead isolates exactly the "surface fit beyond general skill"
question this test is asking.

Usage:
    python model/research/surface_mismatch_market_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import TOUR_CONFIG, match_name_to_pool  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from ev_comparison import implied_probabilities  # noqa: E402
from pedigree_market_premium_test import TOURNAMENTS, fetch_source_csv, ols  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from win_probability import win_probability  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"

# a match perspective counts as "surface-specialized" (either a specialist edge or a mismatch
# weakness) once |mismatch| clears this many Elo points. Chosen off the mismatch distribution's own
# spread (set below, after seeing the printed percentiles, but fixed BEFORE looking at any gap/
# win-rate results - same discipline as DECORATED_THRESHOLD in the pedigree test), not tuned to
# whatever makes the result look best.
SPECIALIST_ELO_THRESHOLD = 50.0


def build_match_rows(tour, slug, matches_history):
    csv_path = fetch_source_csv(tour, slug)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    if len(df) == 0:
        return pd.DataFrame(), f"{tour} {slug}: no rows with a parseable date"

    start_date = df["Date"].min()
    tournament_name = df["Tournament"].iloc[0] if "Tournament" in df.columns else slug

    ratings_df = calculate_elo_ratings(matches_history, start_date, tour=tour)
    ratings_path = OUTPUT_DIR / f"_surfmismatch_test_{tour.lower()}_{slug}_ratings.csv"
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(ratings_path, index=False)

    overall_elo = ratings_df.set_index("player")["overall_elo"].to_dict()
    surface_elo = {
        "Hard": ratings_df.set_index("player")["hard_elo"].to_dict(),
        "Clay": ratings_df.set_index("player")["clay_elo"].to_dict(),
        "Grass": ratings_df.set_index("player")["grass_elo"].to_dict(),
    }
    pool = set(ratings_df["player"])
    name_aliases = TOUR_CONFIG[tour].name_aliases

    rows, unresolved = [], set()
    for row in df.itertuples(index=False):
        avg_w, avg_l = getattr(row, "AvgW", None), getattr(row, "AvgL", None)
        if pd.isna(avg_w) or pd.isna(avg_l):
            continue
        surface = getattr(row, "Surface", None)
        if pd.isna(surface) or surface not in ("Hard", "Clay", "Grass"):
            continue
        winner = match_name_to_pool(row.Winner, pool, name_aliases)
        loser = match_name_to_pool(row.Loser, pool, name_aliases)
        if winner is None:
            unresolved.add(row.Winner)
        if loser is None:
            unresolved.add(row.Loser)
        if winner is None or loser is None:
            continue

        a, b = sorted((winner, loser))
        won_a = (winner == a)
        market_w, market_l = implied_probabilities(avg_w, avg_l)
        market_a = market_w if won_a else market_l
        try:
            model_a = win_probability(a, b, surface, ratings_path)
        except ValueError:
            continue

        oe_a, oe_b = overall_elo.get(a), overall_elo.get(b)
        se_a, se_b = surface_elo[surface].get(a), surface_elo[surface].get(b)
        if oe_a is None or oe_b is None or se_a is None or se_b is None:
            continue
        mismatch_a = se_a - oe_a
        mismatch_b = se_b - oe_b

        rows.append({
            "tour": tour, "tournament": tournament_name, "round": row.Round, "surface": surface,
            "player_a": a, "player_b": b,
            "model_prob_a": model_a, "market_prob_a": market_a,
            "gap": market_a - model_a,
            "overall_elo_diff": oe_a - oe_b,
            "mismatch_a": mismatch_a, "mismatch_b": mismatch_b,
            "mismatch_diff": mismatch_a - mismatch_b,
            "won_a": won_a,
        })

    warning = None
    if unresolved:
        warning = f"{tour} {tournament_name}: {len(unresolved)} name(s) unresolved: {sorted(unresolved)}"
    return pd.DataFrame(rows), warning


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    matches_by_tour = {tour: load_matches_for_tour(tour) for tour in ("ATP", "WTA")}

    all_frames, skipped = [], []
    for tour, slug in TOURNAMENTS:
        try:
            frame, warning = build_match_rows(tour, slug, matches_by_tour[tour])
        except RuntimeError as e:
            skipped.append(f"{tour} {slug}: {e}")
            continue
        if warning:
            print(f"  WARNING: {warning}", file=sys.stderr)
        if len(frame) == 0:
            skipped.append(f"{tour} {slug}: 0 usable rows")
            continue
        print(f"{tour} {slug} ({frame['tournament'].iloc[0]}): {len(frame)} usable matches")
        all_frames.append(frame)

    if skipped:
        print(f"\n{len(skipped)} tournament(s) skipped/failed: {skipped}")

    all_rows = pd.concat(all_frames, ignore_index=True)
    all_rows = all_rows.dropna(subset=["overall_elo_diff", "mismatch_diff"])
    print(f"\n{len(all_rows)} total real matches with a resolved model probability, market "
          f"probability, overall-Elo gap, and surface-mismatch gap, across {len(all_frames)} "
          f"tournaments (both tours, real 2026 closing odds) - this is the real sample size, and "
          f"unlike the pedigree test EVERY match qualifies (no decorated-only subsetting - surface "
          f"mismatch is defined for every player, every match).")

    both_mismatch = pd.concat([all_rows["mismatch_a"].abs(), all_rows["mismatch_b"].abs()])
    print(f"\n|mismatch| distribution across both players in every match (Elo points): "
          f"median={both_mismatch.median():.1f}  p75={both_mismatch.quantile(0.75):.1f}  "
          f"p90={both_mismatch.quantile(0.90):.1f}  max={both_mismatch.max():.1f}")
    print(f"Specialist threshold used below: |mismatch| >= {SPECIALIST_ELO_THRESHOLD:.0f} Elo pts")

    print(f"\n{'=' * 90}\nOLS: gap (market_prob_a - model_prob_a) ~ overall_elo_diff + mismatch_diff, "
          f"all {len(all_rows)} matches\n{'=' * 90}")
    y = all_rows["gap"].values
    X = all_rows[["overall_elo_diff", "mismatch_diff"]].values
    beta, se = ols(y, X)
    names = ["intercept", "overall_elo_diff", "mismatch_diff"]
    for name, b, s in zip(names, beta, se):
        z = b / s if s == s and s != 0 else float("nan")
        print(f"  {name:<17}: coef={b:+.6f}  SE={s:.6f}  z={z:+.2f}"
              + ("  (|z|>1.96, nominally significant)" if abs(z) > 1.96 else "  (not significant on its own)"))
    print(f"\n  Interpretation: a NEGATIVE mismatch_diff coefficient means that when player A rates "
          f"further above their overall level on this surface than player B does (mismatch_diff up), "
          f"the market gives A relatively LESS credit than the model does (gap down) - i.e. the "
          f"market underweights surface fit, exactly the hypothesis being tested. A coefficient near "
          f"zero/not significant means no detectable surface-blindness beyond what overall Elo "
          f"already captures.")

    # per-player-per-match perspectives, mirroring the pedigree test's win-rate framing exactly.
    persp = []
    for r in all_rows.itertuples(index=False):
        for player, opp, model_p, market_p, mismatch_self, won in [
            (r.player_a, r.player_b, r.model_prob_a, r.market_prob_a, r.mismatch_a, r.won_a),
            (r.player_b, r.player_a, 1 - r.model_prob_a, 1 - r.market_prob_a, r.mismatch_b, not r.won_a),
        ]:
            persp.append({
                "player": player, "opponent": opp, "surface": r.surface,
                "model_prob": model_p, "market_prob": market_p,
                "market_discount": model_p - market_p,  # positive = market underrates this player vs model
                "mismatch": mismatch_self, "won": won,
            })
    persp = pd.DataFrame(persp)

    def report_subset(label, subset, expect_note):
        print(f"\n{'=' * 90}\n{label} (n={len(subset)} match-perspectives)\n{'=' * 90}")
        if len(subset) < 10:
            print(f"  n={len(subset)} - too small to say anything real here.")
            return
        actual = subset["won"].mean()
        mkt = subset["market_prob"].mean()
        mdl = subset["model_prob"].mean()
        observed, lo, hi = cluster_bootstrap_ci(
            subset.assign(_a=subset["won"].astype(int), _s=subset["model_prob"]), "_a", "_s", group_col="player")
        print(f"  Real win rate: {actual:.1%}  |  market's average implied prob: {mkt:.1%}  |  "
              f"model's average prob: {mdl:.1%}")
        print(f"  Model calibration gap (actual - model), player-clustered: {observed:+.1%}, "
              f"95% CI [{lo:+.1%}, {hi:+.1%}]")
        print(f"  Market calibration gap (actual - market): {actual - mkt:+.1%}")
        print(f"  {expect_note}")

    specialists = persp[(persp["mismatch"] >= SPECIALIST_ELO_THRESHOLD) & (persp["market_discount"] > 0)]
    report_subset(
        "SPECIALIST rows: player rates >= threshold ABOVE their overall level on this surface, AND "
        "the market is less bullish on them than the model is (market_discount > 0)",
        specialists,
        "If the market truly underweights surface fit, actual win rate here should sit closer to "
        "the model's (higher) number than the market's (lower) one.",
    )

    mismatched = persp[(persp["mismatch"] <= -SPECIALIST_ELO_THRESHOLD) & (persp["market_discount"] < 0)]
    report_subset(
        "MISMATCH-WEAKNESS rows: player rates >= threshold BELOW their overall level on this surface, "
        "AND the market is MORE bullish on them than the model is (market_discount < 0, i.e. a "
        "market premium the model doesn't share)",
        mismatched,
        "If the market truly underweights surface fit here too, actual win rate should sit closer to "
        "the model's (lower) number than the market's (higher, overrated) one.",
    )

    print(f"\nAll specialist-edge rows (market underrating a surface specialist), for direct inspection:")
    print(specialists.sort_values("market_discount", ascending=False).to_string(index=False, formatters={
        "model_prob": "{:.1%}".format, "market_prob": "{:.1%}".format,
        "market_discount": "{:+.1%}".format, "mismatch": "{:+.1f}".format,
    }))


if __name__ == "__main__":
    main()
