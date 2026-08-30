"""Leave-one-out ablation of every live win_probability.py correction (rank-gap, layoff,
recent-form, confidence-calibration/Platt, decay3-WTA), using the same real 1,749-match,
23-tournament, both-tour tennis-data.co.uk closing-odds dataset the pedigree and surface-mismatch
tests built. For each correction: full-stack-ON vs that-one-correction-OFF (everything else held
fixed), broken down by predicted-favorite confidence, not just overall log-loss - specifically
whether removing it helps or hurts calibration among HEAVY FAVORITES (top confidence decile),
mirroring the exact methodology that originally found the raw-Elo overconfidence-at-the-top problem
(recent_form_test.py / the Platt-scaling backtest cited in win_probability.PLATT_B's docstring).

upset-boost is EXCLUDED, not silently skipped: its own docstring in win_probability.py says it's
structurally in-tournament-only ("who did they just beat in THIS event" doesn't exist before Round
1) - it can never apply to a single closing-line match the way this dataset is shaped, only to a
round-by-round bracket replay. Testing it here would be a fake test, not a real null.

A real bug in the PRIOR two tests (pedigree_market_premium_test.py, surface_mismatch_market_test.py)
is fixed here: win_probability.py's own _layoff_bucket_edges_for() selects WTA-fit shifts only when
ratings_path is EXACTLY the canonical WTA_RATINGS_PATH constant - every per-event temp ratings path
those two scripts used (e.g. output/_pedigree_test_wta_cincinnati_ratings.csv) fails that equality
check and silently fell back to ATP's layoff shifts for WTA matches too. This script bypasses that
path-sniffing entirely and selects bucket edges from the real tour label directly, so the layoff
ablation specifically (and every other correction's WTA numbers) are computed correctly. This does
NOT retroactively fix the pedigree/surface-mismatch tests' own layoff-adjusted model_prob numbers -
their conclusions didn't depend on the layoff correction's precision, so they're not being rerun,
but it's disclosed here rather than silently carried forward.

Usage:
    python model/research/correction_ablation_test.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import TOUR_CONFIG, match_name_to_pool  # noqa: E402
from elo_ratings import calculate_elo_ratings, expected_score, load_matches_for_tour  # noqa: E402
from ev_comparison import implied_probabilities  # noqa: E402
from pedigree_market_premium_test import TOURNAMENTS, fetch_source_csv  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from win_probability import (  # noqa: E402
    LAYOFF_BUCKET_EDGES_ATP, LAYOFF_BUCKET_EDGES_WTA, RANK_ADJUSTMENT_ELO_WINDOW,
    _apply_confidence_calibration, _apply_layoff_adjustment, _apply_rank_adjustment,
    _apply_recent_form_adjustment,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
HEAVY_FAVORITE_DECILE = 0.90  # top 10% of full-stack favorite confidence, across the whole dataset


def compute_stack(elo_a, elo_b, rank_a, rank_b, days_a, days_b, resid_a, resid_b, bucket_edges, flags):
    prob = expected_score(elo_a, elo_b)
    if flags["rank"] and abs(elo_a - elo_b) <= RANK_ADJUSTMENT_ELO_WINDOW:
        prob = _apply_rank_adjustment(prob, rank_a, rank_b)
    if flags["layoff"]:
        prob = _apply_layoff_adjustment(prob, days_a, days_b, bucket_edges)
    if flags["recent_form"]:
        prob = _apply_recent_form_adjustment(prob, resid_a, resid_b)
    if flags["confidence_cal"]:
        prob = _apply_confidence_calibration(prob)
    return prob


FULL = {"rank": True, "layoff": True, "recent_form": True, "confidence_cal": True}
VARIANTS = {
    "no_rank_adjustment": {**FULL, "rank": False},
    "no_layoff_adjustment": {**FULL, "layoff": False},
    "no_recent_form_adjustment": {**FULL, "recent_form": False},
    "no_confidence_calibration": {**FULL, "confidence_cal": False},
}


def ratings_lookup(ratings_df):
    idx = ratings_df.set_index("player")
    return {
        "overall": idx["overall_elo"].to_dict(),
        "surface": {
            "Hard": idx["hard_elo"].to_dict(), "Clay": idx["clay_elo"].to_dict(), "Grass": idx["grass_elo"].to_dict(),
        },
        "rank": idx["current_rank"].to_dict(),
        "days": idx["days_since_last_match"].to_dict(),
        "resid": idx["recent_form_residual"].to_dict(),
    }


def g(d, k):
    v = d.get(k)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def build_match_rows(tour, slug, matches_history):
    csv_path = fetch_source_csv(tour, slug)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    if len(df) == 0:
        return pd.DataFrame(), f"{tour} {slug}: no rows with a parseable date"

    start_date = df["Date"].min()
    tournament_name = df["Tournament"].iloc[0] if "Tournament" in df.columns else slug
    bucket_edges = LAYOFF_BUCKET_EDGES_WTA if tour == "WTA" else LAYOFF_BUCKET_EDGES_ATP

    ratings_df = calculate_elo_ratings(matches_history, start_date, tour=tour)
    ratings_path = OUTPUT_DIR / f"_ablation_test_{tour.lower()}_{slug}_ratings.csv"
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(ratings_path, index=False)
    R = ratings_lookup(ratings_df)

    # decay3-off snapshot - WTA only, only computed when needed (tour=None disables decay3's
    # DECAY3_TOURS gate in calculate_elo_ratings, falling back to the plain 5yr hard-cutoff window;
    # nothing else about the computation differs).
    R_off = None
    if tour == "WTA":
        ratings_off_df = calculate_elo_ratings(matches_history, start_date, tour=None)
        R_off = ratings_lookup(ratings_off_df)

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

        se = R["surface"][surface]
        if g(se, a) is None or g(se, b) is None:
            continue
        elo_a, elo_b = se[a], se[b]
        rank_a, rank_b = g(R["rank"], a), g(R["rank"], b)
        days_a, days_b = g(R["days"], a), g(R["days"], b)
        resid_a, resid_b = g(R["resid"], a), g(R["resid"], b)
        oe_a, oe_b = g(R["overall"], a), g(R["overall"], b)
        if oe_a is None or oe_b is None:
            continue

        model_full = compute_stack(elo_a, elo_b, rank_a, rank_b, days_a, days_b, resid_a, resid_b, bucket_edges, FULL)
        record = {
            "tour": tour, "tournament": tournament_name, "surface": surface,
            "player_a": a, "player_b": b, "won_a": won_a,
            "market_prob_a": market_a,
            "model_prob_a_full": model_full,
            "mismatch_a": elo_a - oe_a, "mismatch_b": elo_b - oe_b,
        }
        for name, flags in VARIANTS.items():
            record[f"model_prob_a_{name}"] = compute_stack(
                elo_a, elo_b, rank_a, rank_b, days_a, days_b, resid_a, resid_b, bucket_edges, flags)

        if R_off is not None:
            se_off = R_off["surface"][surface]
            if g(se_off, a) is not None and g(se_off, b) is not None:
                record["model_prob_a_no_decay3"] = compute_stack(
                    se_off[a], se_off[b], g(R_off["rank"], a), g(R_off["rank"], b),
                    g(R_off["days"], a), g(R_off["days"], b), g(R_off["resid"], a), g(R_off["resid"], b),
                    bucket_edges, FULL)

        rows.append(record)

    warning = None
    if unresolved:
        warning = f"{tour} {tournament_name}: {len(unresolved)} name(s) unresolved: {sorted(unresolved)}"
    return pd.DataFrame(rows), warning


def favorite_frame(all_rows, full_col, variant_col=None):
    """One row per match (not per-perspective): the favorite per full_col, whether they actually
    won, full-stack confidence in them, and (if variant_col given) the ablated-variant confidence
    in that SAME favorite identity - so a calibration/log-loss comparison is always about the same
    matches and the same predicted side, only the correction differs."""
    favored_a = all_rows[full_col] >= 0.5
    player = np.where(favored_a, all_rows["player_a"], all_rows["player_b"])
    won = np.where(favored_a, all_rows["won_a"], ~all_rows["won_a"])
    conf_full = np.where(favored_a, all_rows[full_col], 1 - all_rows[full_col])
    out = pd.DataFrame({"player": player, "won": won.astype(bool), "conf_full": conf_full})
    if variant_col is not None:
        conf_var = np.where(favored_a, all_rows[variant_col], 1 - all_rows[variant_col])
        out["conf_variant"] = conf_var
    return out


def logloss(won, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.where(won, np.log(p), np.log(1 - p))


def report_ablation(label, all_rows, variant_col, heavy_cutoff, tour_filter=None):
    rows = all_rows if tour_filter is None else all_rows[all_rows["tour"] == tour_filter]
    rows = rows.dropna(subset=[variant_col])
    if len(rows) < 20:
        print(f"\n  [{label}{' - ' + tour_filter if tour_filter else ''}] n={len(rows)} - too small, skipping")
        return
    frame = favorite_frame(rows, "model_prob_a_full", variant_col)
    heavy = frame[frame["conf_full"] >= heavy_cutoff]
    print(f"\n  --- {label}{' (' + tour_filter + ' only)' if tour_filter else ' (both tours)'} ---")
    print(f"  All matches: n={len(frame)}  |  Heavy favorites (conf >= {heavy_cutoff:.1%}): n={len(heavy)}")
    if len(heavy) < 15:
        print(f"  Heavy-favorite subset too small (n={len(heavy)}) for a real conclusion.")
        return

    actual = heavy["won"].mean()
    full_gap, full_lo, full_hi = cluster_bootstrap_ci(
        heavy.assign(_a=heavy["won"].astype(int), _s=heavy["conf_full"]), "_a", "_s", group_col="player")
    var_gap, var_lo, var_hi = cluster_bootstrap_ci(
        heavy.assign(_a=heavy["won"].astype(int), _s=heavy["conf_variant"]), "_a", "_s", group_col="player")
    print(f"  Real win rate among heavy favorites: {actual:.1%}")
    print(f"  FULL-STACK  : mean conf={heavy['conf_full'].mean():.1%}  calibration gap (actual-conf)="
          f"{full_gap:+.1%} CI[{full_lo:+.1%},{full_hi:+.1%}]")
    print(f"  {label:<26}: mean conf={heavy['conf_variant'].mean():.1%}  calibration gap (actual-conf)="
          f"{var_gap:+.1%} CI[{var_lo:+.1%},{var_hi:+.1%}]")

    heavy = heavy.assign(
        loss_full=logloss(heavy["won"].values, heavy["conf_full"].values),
        loss_variant=logloss(heavy["won"].values, heavy["conf_variant"].values),
    )
    diff, dlo, dhi = cluster_bootstrap_ci(
        heavy.assign(_a=heavy["loss_full"], _s=heavy["loss_variant"]), "_a", "_s", group_col="player")
    sig = "excludes zero" if (dlo > 0 or dhi < 0) else "includes zero, not significant"
    print(f"  Log-loss (full - ablated), heavy favorites: {diff:+.4f} CI[{dlo:+.4f},{dhi:+.4f}] ({sig})")
    if dlo > 0:
        print(f"  -> Positive & significant: full-stack loss > ablated loss - REMOVING this correction "
              f"IMPROVES calibration among heavy favorites (it's net harmful here).")
    elif dhi < 0:
        print(f"  -> Negative & significant: full-stack loss < ablated loss - removing this correction "
              f"WORSENS calibration among heavy favorites (it's working as intended here).")
    else:
        print(f"  -> No significant difference among heavy favorites either way.")


def report_mismatch_concentration(all_rows, heavy_cutoff):
    print(f"\n{'=' * 90}\nIs the surface-mismatch overconfidence concentrated in heavy favorites?\n{'=' * 90}")
    persp = []
    for r in all_rows.itertuples(index=False):
        for player, model_p, won, mismatch in [
            (r.player_a, r.model_prob_a_full, r.won_a, r.mismatch_a),
            (r.player_b, 1 - r.model_prob_a_full, not r.won_a, r.mismatch_b),
        ]:
            persp.append({"player": player, "model_prob": model_p, "won": won, "mismatch": mismatch})
    persp = pd.DataFrame(persp)

    specialists = persp[(persp["mismatch"] >= 50) & (persp["model_prob"] >= 0.5)]
    heavy = specialists[specialists["model_prob"] >= heavy_cutoff]
    light = specialists[specialists["model_prob"] < heavy_cutoff]
    print(f"Specialist rows (mismatch>=+50, model favors them): n={len(specialists)}, split at "
          f"conf>={heavy_cutoff:.1%} into heavy n={len(heavy)} / non-heavy n={len(light)}")
    for name, g_ in [("heavy favorites", heavy), ("non-heavy (moderate confidence)", light)]:
        if len(g_) < 10:
            print(f"  {name}: n={len(g_)} - too small for a real conclusion")
            continue
        gap, lo, hi = cluster_bootstrap_ci(
            g_.assign(_a=g_["won"].astype(int), _s=g_["model_prob"]), "_a", "_s", group_col="player")
        print(f"  {name}: n={len(g_)}  actual={g_['won'].mean():.1%}  model={g_['model_prob'].mean():.1%}  "
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
        print(f"{tour} {slug}: {len(frame)} usable matches")
        all_frames.append(frame)

    if skipped:
        print(f"\n{len(skipped)} tournament(s) skipped/failed: {skipped}")

    all_rows = pd.concat(all_frames, ignore_index=True)
    print(f"\n{len(all_rows)} total real matches, {all_rows['tour'].eq('WTA').sum()} WTA / "
          f"{all_rows['tour'].eq('ATP').sum()} ATP")

    fav = favorite_frame(all_rows, "model_prob_a_full")
    heavy_cutoff = fav["conf_full"].quantile(HEAVY_FAVORITE_DECILE)
    print(f"\nHeavy-favorite cutoff (top {(1 - HEAVY_FAVORITE_DECILE):.0%} of full-stack favorite "
          f"confidence, across all {len(fav)} matches): conf >= {heavy_cutoff:.1%}")

    print(f"\n{'=' * 90}\nLEAVE-ONE-OUT ABLATION: full-stack vs each correction OFF, heavy favorites only\n{'=' * 90}")
    for name in VARIANTS:
        report_ablation(name, all_rows, f"model_prob_a_{name}", heavy_cutoff)
        report_ablation(name, all_rows, f"model_prob_a_{name}", heavy_cutoff, tour_filter="ATP")
        report_ablation(name, all_rows, f"model_prob_a_{name}", heavy_cutoff, tour_filter="WTA")

    print(f"\n  --- no_decay3 (WTA only - decay3 doesn't apply to ATP) ---")
    report_ablation("no_decay3", all_rows, "model_prob_a_no_decay3", heavy_cutoff, tour_filter="WTA")

    print(f"\n{'=' * 90}\nupset-boost: EXCLUDED from this ablation - structurally in-tournament-only "
          f"(depends on 'who did this player just beat in THIS event', which doesn't exist for a "
          f"single real closing-line match outside a round-by-round bracket replay). Not a null "
          f"result, not applicable to this dataset shape at all.\n{'=' * 90}")

    report_mismatch_concentration(all_rows, heavy_cutoff)


if __name__ == "__main__":
    main()
