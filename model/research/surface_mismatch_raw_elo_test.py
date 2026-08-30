"""Reruns surface_mismatch_market_test.py's specialist/mismatch-weakness calibration check with
EVERY live correction disabled - pure, unmodified surface-blended Elo only (elo_ratings.
expected_score on hard_elo/clay_elo/grass_elo straight from calculate_elo_ratings, with decay3
disabled too via tour=None). No rank-gap, no layoff, no recent-form, no confidence-calibration
(Platt), no decay3. Same 23-tournament dataset, same mismatch definition (surface_elo - overall_elo)
and same SPECIALIST_ELO_THRESHOLD=50 selection as the corrected-full-stack version, so the two
calibration gaps are directly comparable - this isolates whether the -14.2%/+9.5% miscalibration
found earlier is a property of raw Elo's surface blend itself, or something the correction stack
creates/amplifies on top of it.

Usage:
    python model/research/surface_mismatch_raw_elo_test.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import TOUR_CONFIG, match_name_to_pool  # noqa: E402
from elo_ratings import calculate_elo_ratings, expected_score, load_matches_for_tour  # noqa: E402
from ev_comparison import implied_probabilities  # noqa: E402
from pedigree_market_premium_test import TOURNAMENTS, fetch_source_csv  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

SPECIALIST_ELO_THRESHOLD = 50.0

# from correction_ablation_test.py / surface_mismatch_market_test.py's full-stack runs tonight
FULL_STACK_SPECIALIST = (-0.142, -0.252, -0.048, 71)   # gap, lo, hi, n
FULL_STACK_MISMATCH_WEAKNESS = (0.095, 0.062, 0.127, 695)


def build_match_rows(tour, slug, matches_history):
    csv_path = fetch_source_csv(tour, slug)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    if len(df) == 0:
        return pd.DataFrame(), f"{tour} {slug}: no rows with a parseable date"

    start_date = df["Date"].min()
    tournament_name = df["Tournament"].iloc[0] if "Tournament" in df.columns else slug

    # tour=None disables decay3 (its gate is `tour is not None and tour.upper() in DECAY3_TOURS`) -
    # otherwise identical to the corrected-full-stack ratings computation.
    ratings_df = calculate_elo_ratings(matches_history, start_date, tour=None)
    idx = ratings_df.set_index("player")
    overall_elo = idx["overall_elo"].to_dict()
    surface_elo = {"Hard": idx["hard_elo"].to_dict(), "Clay": idx["clay_elo"].to_dict(), "Grass": idx["grass_elo"].to_dict()}
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

        oe_a, oe_b = overall_elo.get(a), overall_elo.get(b)
        se_a, se_b = surface_elo[surface].get(a), surface_elo[surface].get(b)
        if oe_a is None or oe_b is None or se_a is None or se_b is None:
            continue

        raw_model_a = expected_score(se_a, se_b)  # pure surface-blended Elo, zero corrections

        rows.append({
            "tour": tour, "player_a": a, "player_b": b, "surface": surface, "won_a": won_a,
            "model_prob_a": raw_model_a, "market_prob_a": market_a,
            "mismatch_a": se_a - oe_a, "mismatch_b": se_b - oe_b,
        })

    warning = None
    if unresolved:
        warning = f"{tour} {tournament_name}: {len(unresolved)} name(s) unresolved: {sorted(unresolved)}"
    return pd.DataFrame(rows), warning


def report(label, subset, baseline):
    b_gap, b_lo, b_hi, b_n = baseline
    print(f"\n{'=' * 90}\n{label}\n{'=' * 90}")
    if len(subset) < 10:
        print(f"  n={len(subset)} - too small for a real conclusion")
        return
    actual = subset["won"].mean()
    mdl = subset["model_prob"].mean()
    gap, lo, hi = cluster_bootstrap_ci(
        subset.assign(_a=subset["won"].astype(int), _s=subset["model_prob"]), "_a", "_s", group_col="player")
    print(f"  RAW ELO ONLY: n={len(subset)}  actual={actual:.1%}  model={mdl:.1%}  gap={gap:+.1%} "
          f"CI[{lo:+.1%},{hi:+.1%}]")
    print(f"  FULL STACK  : n={b_n}  gap={b_gap:+.1%} CI[{b_lo:+.1%},{b_hi:+.1%}] (from earlier tonight)")

    overlap = not (hi < b_lo or lo > b_hi)
    same_direction = (gap < 0) == (b_gap < 0)
    print(f"\n  Same direction: {same_direction}  |  CIs overlap: {overlap}  |  "
          f"magnitude: raw={abs(gap):.1%} vs full-stack={abs(b_gap):.1%} "
          f"({'raw is LARGER' if abs(gap) > abs(b_gap) else 'raw is SMALLER or similar'})")
    if same_direction and overlap:
        print(f"  -> Roughly the same size with everything off: this is a raw-Elo/surface-blend "
              f"property, not something the correction stack creates or meaningfully amplifies.")
    elif same_direction and not overlap:
        print(f"  -> Same direction but CIs don't overlap: corrections are measurably changing the "
              f"MAGNITUDE of this effect, even though the underlying raw-Elo issue is real too.")
    else:
        print(f"  -> Different direction or no real relationship: the corrections are doing something "
              f"more than just scaling a pre-existing raw-Elo issue - worth understanding directly.")


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
    print(f"{len(all_rows)} total real matches across {len(all_frames)} tournaments (raw Elo only, "
          f"zero corrections, decay3 disabled)")

    persp = []
    for r in all_rows.itertuples(index=False):
        for player, model_p, market_p, won, mismatch in [
            (r.player_a, r.model_prob_a, r.market_prob_a, r.won_a, r.mismatch_a),
            (r.player_b, 1 - r.model_prob_a, 1 - r.market_prob_a, not r.won_a, r.mismatch_b),
        ]:
            persp.append({
                "player": player, "model_prob": model_p, "market_prob": market_p,
                "won": won, "mismatch": mismatch, "market_discount": model_p - market_p,
            })
    persp = pd.DataFrame(persp)

    specialists = persp[(persp["mismatch"] >= SPECIALIST_ELO_THRESHOLD) & (persp["market_discount"] > 0)]
    weakness = persp[(persp["mismatch"] <= -SPECIALIST_ELO_THRESHOLD) & (persp["market_discount"] < 0)]

    report("SPECIALIST rows (model overconfident direction)", specialists, FULL_STACK_SPECIALIST)
    report("MISMATCH-WEAKNESS rows (model underconfident direction)", weakness, FULL_STACK_MISMATCH_WEAKNESS)


if __name__ == "__main__":
    main()
