"""Investigates the one soft spot from surface_mismatch_decile_audit.py: the [135,175) |mismatch|
decile showed a modest overcorrection after damping (-3.0% CI[-5.2%,-0.9%]), while its log-loss
change was NOT significant (CI straddled zero) - unlike every other decile, which was either neutral
or a clear log-loss improvement. Is this a real, distinct sub-effect (concentrated in a specific
tour/surface/player population, the way the original overconfidence finding was traced to specific
causes tonight), or is it noise given the log-loss result already says "not distinguishable from
zero" here?

Reuses the exact same build/split/fit as surface_mismatch_damping_fit.py and surface_mismatch_
decile_audit.py - no new methodology, just a finer breakdown of the same held-out rows.

Usage:
    python model/research/surface_mismatch_135_175_investigation.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from surface_mismatch_damping_fit import add_damped_pred, build_tour, log_loss  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

FITTED_DAMP_POINTS = 90.0
LO, HI = 135.0, 175.0


def report_group(label, df, damp_points):
    if len(df) < 10:
        print(f"  {label:<40}: n={len(df):<5} - too small for a real conclusion")
        return
    raw_loss = log_loss(df["actual_win"].values, df["pred_win"].values)
    damped_pred = add_damped_pred(df, damp_points)
    damped_loss = log_loss(df["actual_win"].values, damped_pred)
    d = df.assign(raw_loss=raw_loss, damped_loss=damped_loss, damped_pred=damped_pred)

    raw_gap, raw_lo, raw_hi = cluster_bootstrap_ci(
        d.assign(_a=d["actual_win"], _s=d["pred_win"]), "_a", "_s", group_col="player")
    damp_gap, damp_lo, damp_hi = cluster_bootstrap_ci(
        d.assign(_a=d["actual_win"], _s=d["damped_pred"]), "_a", "_s", group_col="player")
    ll_diff, ll_lo, ll_hi = cluster_bootstrap_ci(d, "raw_loss", "damped_loss", group_col="player")
    ll_sig = ll_lo > 0 or ll_hi < 0

    print(f"  {label:<40}: n={len(d):<5} raw_gap={raw_gap:+.1%} CI[{raw_lo:+.1%},{raw_hi:+.1%}]  "
          f"damped_gap={damp_gap:+.1%} CI[{damp_lo:+.1%},{damp_hi:+.1%}]  "
          f"ll_diff={ll_diff:+.4f} CI[{ll_lo:+.4f},{ll_hi:+.4f}]{'  *sig' if ll_sig else ''}")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    all_data = {tour: build_tour(tour) for tour in ("ATP", "WTA")}
    test = pd.concat([p[p["edition_id"].isin(tt)] for p, _, tt in all_data.values()], ignore_index=True)
    decile = test[(test["mismatch"].abs() >= LO) & (test["mismatch"].abs() < HI)].copy()
    print(f"\n[{LO:.0f},{HI:.0f}) decile: n={len(decile)}")

    print(f"\n{'=' * 100}\nBy tour\n{'=' * 100}")
    for tour in ("ATP", "WTA"):
        report_group(tour, decile[decile["tour"] == tour], FITTED_DAMP_POINTS)

    print(f"\n{'=' * 100}\nBy surface\n{'=' * 100}")
    for surface in ("Hard", "Clay", "Grass"):
        report_group(surface, decile[decile["surface"] == surface], FITTED_DAMP_POINTS)

    print(f"\n{'=' * 100}\nBy tour x surface\n{'=' * 100}")
    for tour in ("ATP", "WTA"):
        for surface in ("Hard", "Clay", "Grass"):
            sub = decile[(decile["tour"] == tour) & (decile["surface"] == surface)]
            report_group(f"{tour} {surface}", sub, FITTED_DAMP_POINTS)

    print(f"\n{'=' * 100}\nBy mismatch DIRECTION (specialist vs weakness)\n{'=' * 100}")
    report_group("specialist direction (mismatch>=+135)", decile[decile["mismatch"] > 0], FITTED_DAMP_POINTS)
    report_group("weakness direction (mismatch<=-135)", decile[decile["mismatch"] < 0], FITTED_DAMP_POINTS)

    print(f"\n{'=' * 100}\nPlayer concentration: top 15 players by row count in this decile\n{'=' * 100}")
    counts = decile.groupby("player").size().sort_values(ascending=False).head(15)
    for player, n in counts.items():
        sub = decile[decile["player"] == player]
        actual = sub["actual_win"].mean()
        raw = sub["pred_win"].mean()
        print(f"  {player:<20}: n={n:<4} actual={actual:.1%}  raw_pred={raw:.1%}  "
              f"tours={sorted(sub['tour'].unique())}  surfaces={sorted(sub['surface'].unique())}")
    top_share = counts.head(5).sum() / len(decile)
    print(f"\n  Top 5 players' share of this decile: {top_share:.1%} of n={len(decile)}")

    print(f"\n{'=' * 100}\nSame breakdown, ADJACENT deciles for comparison (are they just as noisy?)\n{'=' * 100}")
    for lo, hi in [(112, 135), (135, 175), (175, 250)]:
        sub = test[(test["mismatch"].abs() >= lo) & (test["mismatch"].abs() < hi)]
        report_group(f"[{lo},{hi})", sub, FITTED_DAMP_POINTS)


if __name__ == "__main__":
    main()
