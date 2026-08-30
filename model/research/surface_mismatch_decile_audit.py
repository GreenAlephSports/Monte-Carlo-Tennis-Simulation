"""Audits whether surface_mismatch_damping_fit.py's flat damping (DAMP_POINTS=90 past
SURFACE_MISMATCH_THRESHOLD=50, now live in elo_ratings._damp_surface_mismatch) is being applied too
broadly - specifically, whether the most EXTREME mismatches (150+, 200+ Elo points) represent real,
still-miscalibrated overconfidence, or plausibly-legitimate specialization that damping actively
hurts. The four-bucket validation in the fit script pooled everything >=50 together; this breaks the
same held-out test-era population into finer deciles, plus a direct top-15-by-|mismatch| row-level
inspection, both tours.

Reuses the EXACT same build (build_frozen_predictions_surface), same chronological 80/20 edition
split, same fitted DAMP_POINTS - this is a closer look at the same validated result, not a new fit.

Usage:
    python model/research/surface_mismatch_decile_audit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from surface_mismatch_damping_fit import (  # noqa: E402
    SURFACE_MISMATCH_THRESHOLD, add_damped_pred, build_tour, log_loss,
)
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

FITTED_DAMP_POINTS = 90.0
N_DECILES = 10
EXTREME_BUCKETS = [(50, 75), (75, 100), (100, 125), (125, 150), (150, 200), (200, float("inf"))]


def report_group(label, df, damp_points):
    if len(df) < 15:
        print(f"  {label:<28}: n={len(df):<5} - too small for a real conclusion")
        return None
    raw_loss = log_loss(df["actual_win"].values, df["pred_win"].values)
    damped_pred = add_damped_pred(df, damp_points)
    damped_loss = log_loss(df["actual_win"].values, damped_pred)
    d = df.assign(raw_loss=raw_loss, damped_loss=damped_loss, damped_pred=damped_pred)

    raw_gap, raw_lo, raw_hi = cluster_bootstrap_ci(
        d.assign(_a=d["actual_win"], _s=d["pred_win"]), "_a", "_s", group_col="player")
    damp_gap, damp_lo, damp_hi = cluster_bootstrap_ci(
        d.assign(_a=d["actual_win"], _s=d["damped_pred"]), "_a", "_s", group_col="player")
    ll_diff, ll_lo, ll_hi = cluster_bootstrap_ci(d, "raw_loss", "damped_loss", group_col="player")

    raw_sig = raw_lo > 0 or raw_hi < 0
    damp_sig = damp_lo > 0 or damp_hi < 0
    ll_sig = ll_lo > 0 or ll_hi < 0
    ll_worse = ll_hi < 0  # damping made log-loss WORSE, significantly

    print(f"  {label:<28}: n={len(d):<5} mean|mismatch|={d['mismatch'].abs().mean():6.1f}  "
          f"raw_gap={raw_gap:+.1%} CI[{raw_lo:+.1%},{raw_hi:+.1%}]{'*' if raw_sig else ' '}  "
          f"damped_gap={damp_gap:+.1%} CI[{damp_lo:+.1%},{damp_hi:+.1%}]{'*' if damp_sig else ' '}  "
          f"ll_diff={ll_diff:+.4f} CI[{ll_lo:+.4f},{ll_hi:+.4f}]{'*WORSE' if ll_worse else ('*better' if ll_sig else '')}")
    return dict(label=label, n=len(d), raw_gap=raw_gap, damp_gap=damp_gap, ll_diff=ll_diff, ll_lo=ll_lo, ll_hi=ll_hi)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    all_data = {tour: build_tour(tour) for tour in ("ATP", "WTA")}
    test = pd.concat([p[p["edition_id"].isin(tt)] for p, _, tt in all_data.values()], ignore_index=True)
    affected = test[test["mismatch"].abs() >= SURFACE_MISMATCH_THRESHOLD].copy()
    affected["abs_mismatch"] = affected["mismatch"].abs()
    print(f"\nHeld-out affected rows (|mismatch|>=50): n={len(affected)}, fitted DAMP_POINTS={FITTED_DAMP_POINTS:.0f}")
    print("(* = 95% CI excludes zero; on ll_diff, *WORSE means damping significantly HURT log-loss here)")

    print(f"\n{'=' * 100}\nFIXED MAGNITUDE BUCKETS (finer than the original 4-bucket validation)\n{'=' * 100}")
    for lo, hi in EXTREME_BUCKETS:
        label = f"[{lo:.0f},{hi:.0f})" if hi != float("inf") else f"[{lo:.0f}, inf)"
        bucket = affected[(affected["abs_mismatch"] >= lo) & (affected["abs_mismatch"] < hi)]
        report_group(label, bucket, FITTED_DAMP_POINTS)

    print(f"\n{'=' * 100}\nDECILES of |mismatch| within the affected population (n={len(affected)})\n{'=' * 100}")
    affected["decile"] = pd.qcut(affected["abs_mismatch"], N_DECILES, duplicates="drop")
    for interval, group in affected.groupby("decile", observed=True):
        report_group(f"decile [{interval.left:.0f},{interval.right:.0f}]", group, FITTED_DAMP_POINTS)

    print(f"\n{'=' * 100}\nSame deciles, EXTREME TAIL ONLY (>=150), split ATP vs WTA\n{'=' * 100}")
    extreme = affected[affected["abs_mismatch"] >= 150]
    for tour in ("ATP", "WTA"):
        report_group(f"{tour}, |mismatch|>=150", extreme[extreme["tour"] == tour], FITTED_DAMP_POINTS)

    print(f"\n{'=' * 100}\nTOP 15 MOST EXTREME |mismatch| ROWS, held-out test era, both tours\n{'=' * 100}")
    top = affected.sort_values("abs_mismatch", ascending=False).head(15).copy()
    top["damped_pred"] = add_damped_pred(top, FITTED_DAMP_POINTS)
    cols = ["tour", "edition_id", "player", "opponent", "surface", "mismatch", "pred_win", "damped_pred", "actual_win"]
    print(top[cols].to_string(index=False, formatters={
        "mismatch": "{:+.1f}".format, "pred_win": "{:.1%}".format, "damped_pred": "{:.1%}".format,
        "actual_win": lambda v: "WON" if v == 1 else "lost",
    }))
    print(f"\nFor each row: raw predicted this player at pred_win; if actual_win=WON and pred_win was "
          f"already high, that instance itself doesn't contradict raw Elo (though a single match "
          f"never confirms or denies calibration on its own - only the aggregated bucket/decile "
          f"gap above does that). Flag any row where the player is a well-known, plausible specialist "
          f"on that surface (e.g. a clay-only journeyman playing clay) for manual judgment - damping "
          f"still applies to them the same as anyone else since the correction has no player-specific "
          f"carve-out; whether that's appropriate is exactly what the bucket-level held-out numbers "
          f"above are meant to answer, not any single row.")


if __name__ == "__main__":
    main()
