"""Confirms (or rejects) the suspected mechanism behind surface_mismatch_market_test.py's finding:
is the surface-mismatch calibration error (both directions - the specialist-edge overconfidence AND
the mismatch-weakness underconfidence) concentrated among players with a THIN surface-specific match
sample, or spread evenly regardless of sample depth?

Mechanism under test: elo_ratings.calculate_elo_ratings blends surface_elo toward overall_elo with
weight = surface_matches / (surface_matches + SURFACE_BLEND_K), SURFACE_BLEND_K=7 - so a player with
few surface matches CAN still show a large |mismatch| (the blend never fully caps it), but that
large mismatch is more likely to be noise (a small, volatile sample) than a real, stable skill gap.
If the calibration error is concentrated in the thin-sample players, that's the mechanism. If it's
spread evenly (or worse among deep-sample players), the mismatch signal itself is the problem, not
sample depth.

Reuses the exact same 23-tournament dataset and mismatch/specialist definitions as
surface_mismatch_market_test.py (mismatch = surface_elo - overall_elo, SPECIALIST_ELO_THRESHOLD=50),
adding each player's own surface_matches count (hard_matches/clay_matches/grass_matches, whichever
matches the event's surface) at the same frozen-at-event-start-date snapshot.

Usage:
    python model/research/surface_mismatch_thin_sample_test.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import TOUR_CONFIG, match_name_to_pool  # noqa: E402
from elo_ratings import SURFACE_BLEND_K, calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from ev_comparison import implied_probabilities  # noqa: E402
from pedigree_market_premium_test import TOURNAMENTS, fetch_source_csv  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from win_probability import win_probability  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
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
    ratings_path = OUTPUT_DIR / f"_thinsample_test_{tour.lower()}_{slug}_ratings.csv"
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(ratings_path, index=False)

    idx = ratings_df.set_index("player")
    overall_elo = idx["overall_elo"].to_dict()
    surface_elo = {"Hard": idx["hard_elo"].to_dict(), "Clay": idx["clay_elo"].to_dict(), "Grass": idx["grass_elo"].to_dict()}
    surface_matches = {"Hard": idx["hard_matches"].to_dict(), "Clay": idx["clay_matches"].to_dict(), "Grass": idx["grass_matches"].to_dict()}
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

        rows.append({
            "tour": tour, "player_a": a, "player_b": b, "surface": surface,
            "model_prob_a": model_a, "market_prob_a": market_a, "won_a": won_a,
            "mismatch_a": se_a - oe_a, "mismatch_b": se_b - oe_b,
            "surface_matches_a": surface_matches[surface].get(a, 0),
            "surface_matches_b": surface_matches[surface].get(b, 0),
        })

    warning = None
    if unresolved:
        warning = f"{tour} {tournament_name}: {len(unresolved)} name(s) unresolved: {sorted(unresolved)}"
    return pd.DataFrame(rows), warning


def report(label, subset):
    if len(subset) < 10:
        print(f"  {label}: n={len(subset)} - too small for a real conclusion")
        return
    gap, lo, hi = cluster_bootstrap_ci(
        subset.assign(_a=subset["won"].astype(int), _s=subset["model_prob"]), "_a", "_s", group_col="player")
    print(f"  {label}: n={len(subset)}  median surface_matches={subset['surface_matches'].median():.0f}  "
          f"actual={subset['won'].mean():.1%}  model={subset['model_prob'].mean():.1%}  "
          f"gap={gap:+.1%} CI[{lo:+.1%},{hi:+.1%}]")


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
        all_frames.append(frame)
    if skipped:
        print(f"{len(skipped)} tournament(s) skipped/failed: {skipped}")

    all_rows = pd.concat(all_frames, ignore_index=True)
    print(f"{len(all_rows)} total real matches across {len(all_frames)} tournaments")

    persp = []
    for r in all_rows.itertuples(index=False):
        for player, model_p, market_p, won, mismatch, sm in [
            (r.player_a, r.model_prob_a, r.market_prob_a, r.won_a, r.mismatch_a, r.surface_matches_a),
            (r.player_b, 1 - r.model_prob_a, 1 - r.market_prob_a, not r.won_a, r.mismatch_b, r.surface_matches_b),
        ]:
            persp.append({
                "player": player, "surface": r.surface, "model_prob": model_p, "market_prob": market_p,
                "won": won, "mismatch": mismatch, "surface_matches": sm,
            })
    persp = pd.DataFrame(persp)

    specialists = persp[(persp["mismatch"] >= SPECIALIST_ELO_THRESHOLD) & ((persp["model_prob"] - persp["market_prob"]) > 0)]
    weakness = persp[(persp["mismatch"] <= -SPECIALIST_ELO_THRESHOLD) & ((persp["model_prob"] - persp["market_prob"]) < 0)]

    for label, group in [("SPECIALIST (model overconfident direction)", specialists),
                          ("MISMATCH-WEAKNESS (model underconfident direction)", weakness)]:
        print(f"\n{'=' * 90}\n{label}, n={len(group)}\n{'=' * 90}")
        if len(group) < 10:
            print(f"  n={len(group)} - too small to split further")
            continue
        median_sm = group["surface_matches"].median()
        print(f"  surface_matches distribution: median={median_sm:.0f}  p25={group['surface_matches'].quantile(0.25):.0f}  "
              f"p75={group['surface_matches'].quantile(0.75):.0f}  min={group['surface_matches'].min():.0f}  "
              f"max={group['surface_matches'].max():.0f}")

        print(f"\n  Split at SURFACE_BLEND_K={SURFACE_BLEND_K} (blend is <50% surface-weighted below this):")
        report("thin (<K matches on this surface)", group[group["surface_matches"] < SURFACE_BLEND_K])
        report("deep (>=K matches on this surface)", group[group["surface_matches"] >= SURFACE_BLEND_K])

        print(f"\n  Split at this group's own median ({median_sm:.0f} matches):")
        report("below median", group[group["surface_matches"] < median_sm])
        report("at/above median", group[group["surface_matches"] >= median_sm])


if __name__ == "__main__":
    main()
