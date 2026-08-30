"""Does the surface-mismatch miscalibration SCALE with the size of the surface_elo-vs-overall_elo
divergence, or is it roughly constant once a player clears the +/-50 threshold? Reuses the raw-Elo-
only dataset (surface_mismatch_raw_elo_test.py - zero corrections, decay3 disabled, so this is a
clean read on the underlying Elo-blend mechanism itself, not anything the correction stack touches).

Defines a single signed "overshoot" per match-perspective, pooling the specialist (mismatch>=+50)
and mismatch-weakness (mismatch<=-50) rows into one measure: overshoot = sign(mismatch) * (model_prob
- actual_win). Positive overshoot means raw Elo pushed the prediction TOO FAR in the direction the
mismatch itself points - whether that's "too confident in a specialist" (mismatch>0) or "too
pessimistic about a mismatched player" (mismatch<0), both get scored the same way, on the same scale,
so magnitude buckets combine both directions instead of needing two separate analyses.

If overshoot grows with |mismatch|, that's a real, fittable dose-response - pointing at a targeted
correction (damp large divergences specifically) rather than a blanket SURFACE_BLEND_K change that
would also flatten small, plausibly-real divergences. If overshoot is flat across magnitude, the
problem isn't "how big the split is" - something else is misfiring at the +/-50 threshold itself.

Usage:
    python model/research/surface_mismatch_magnitude_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elo_ratings import load_matches_for_tour  # noqa: E402
from pedigree_market_premium_test import TOURNAMENTS  # noqa: E402
from surface_mismatch_raw_elo_test import build_match_rows  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

MISMATCH_FLOOR = 50.0
BUCKETS = [(50, 75), (75, 100), (100, 150), (150, float("inf"))]


def ols_1d(y, x):
    X1 = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    n, k = X1.shape
    sigma2 = (resid @ resid) / (n - k)
    cov = sigma2 * np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    return beta, se


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
            continue
        all_frames.append(frame)
    if skipped:
        print(f"{len(skipped)} tournament(s) skipped/failed: {skipped}")

    all_rows = pd.concat(all_frames, ignore_index=True)
    print(f"{len(all_rows)} total real matches (raw Elo only)")

    persp = []
    for r in all_rows.itertuples(index=False):
        for player, model_p, market_p, won, mismatch in [
            (r.player_a, r.model_prob_a, r.market_prob_a, r.won_a, r.mismatch_a),
            (r.player_b, 1 - r.model_prob_a, 1 - r.market_prob_a, not r.won_a, r.mismatch_b),
        ]:
            market_discount = model_p - market_p
            persp.append({
                "player": player, "model_prob": model_p, "won": won,
                "mismatch": mismatch, "market_discount": market_discount,
            })
    persp = pd.DataFrame(persp)

    pool = persp[
        ((persp["mismatch"] >= MISMATCH_FLOOR) & (persp["market_discount"] > 0)) |
        ((persp["mismatch"] <= -MISMATCH_FLOOR) & (persp["market_discount"] < 0))
    ].copy()
    pool["abs_mismatch"] = pool["mismatch"].abs()
    pool["sign"] = np.sign(pool["mismatch"])
    pool["overshoot"] = pool["sign"] * (pool["model_prob"] - pool["won"].astype(int))
    print(f"\nPooled specialist + mismatch-weakness rows: n={len(pool)}")

    print(f"\n{'=' * 90}\nOLS: overshoot ~ |mismatch|\n{'=' * 90}")
    beta, se = ols_1d(pool["overshoot"].values, pool["abs_mismatch"].values)
    for name, b, s in zip(["intercept", "|mismatch|"], beta, se):
        z = b / s if s == s and s != 0 else float("nan")
        print(f"  {name:<12}: coef={b:+.6f}  SE={s:.6f}  z={z:+.2f}"
              + ("  (|z|>1.96, significant)" if abs(z) > 1.96 else "  (not significant)"))
    print(f"\n  Interpretation: a POSITIVE |mismatch| coefficient means overshoot grows as the "
          f"surface/overall divergence grows - a real dose-response, pointing at a targeted "
          f"'damp large divergences' fix. Near-zero/not-significant means the miscalibration is "
          f"roughly constant once a player clears the +/-{MISMATCH_FLOOR:.0f} threshold - a magnitude-"
          f"blind trigger effect instead.")

    print(f"\n{'=' * 90}\nBucketed by |mismatch| magnitude\n{'=' * 90}")
    for lo, hi in BUCKETS:
        label = f"[{lo:.0f},{hi:.0f})" if hi != float("inf") else f"[{lo:.0f}, inf)"
        bucket = pool[(pool["abs_mismatch"] >= lo) & (pool["abs_mismatch"] < hi)]
        if len(bucket) < 10:
            print(f"  {label:<14}: n={len(bucket)} - too small for a real conclusion")
            continue
        gap, gl, gh = cluster_bootstrap_ci(
            bucket.assign(_a=bucket["overshoot"], _s=pd.Series(0.0, index=bucket.index)),
            "_a", "_s", group_col="player")
        print(f"  {label:<14}: n={len(bucket):<4} mean|mismatch|={bucket['abs_mismatch'].mean():6.1f}  "
              f"overshoot={gap:+.1%} CI[{gl:+.1%},{gh:+.1%}]")


if __name__ == "__main__":
    main()
