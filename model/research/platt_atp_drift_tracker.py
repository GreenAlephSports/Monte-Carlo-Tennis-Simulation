"""Tracks whether correction_ablation_test.py's one real finding - that production's global Platt
confidence-calibration correction (win_probability.PLATT_B=0.9205) is net HARMFUL for ATP heavy
favorites (top confidence decile) in the current 2026 real-season dataset - holds up, strengthens,
weakens, or reverses as more real 2026 ATP matches accumulate.

BASELINE (recorded 2026-08-29, n=864 ATP matches across the 12 ATP tournaments tonight's tests used):
  Heavy favorites (conf>=83.3%), n=61: full-stack gap +7.1% CI[+2.0%,+13.4%]; log-loss (full-ablated)
  +0.0100 CI[+0.0007,+0.0199] - excludes zero, removing Platt IMPROVES calibration there.

Same fast-tooling discipline as the --max-editions quick-check pattern used earlier tonight: this is
a CHEAP, frequent check on whatever real data currently exists, explicitly NOT a substitute for the
full ablation - it only reruns the ATP-heavy-favorite Platt slice, and only actually recomputes once
the real ATP sample has roughly DOUBLED (>= 2x baseline = 1728 matches), same "don't over-trust a
small-sample reading" discipline as everything else tonight. Below that, it just reports progress and
does nothing else - no interim verdict is drawn from a still-small sample.

Tournament coverage note: the 23 tournaments in pedigree_market_premium_test.TOURNAMENTS are already
fixed/confirmed-live; EXPANSION_CANDIDATES below are best-effort GUESSES at tennis-data.co.uk slugs
for events later in the 2026 calendar (following the site's established naming convention, same as
the original 23 were confirmed), probed fresh each run - any that don't resolve are silently skipped
(disclosed via the skip count), not treated as evidence of no growth. If real growth doesn't show up
here, it may just mean these particular slug guesses are wrong, not that no new data exists - worth
a manual slug check if this stays flat for a long time despite real tournaments concluding.

Usage:
    python model/research/platt_atp_drift_tracker.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

import pandas as pd  # noqa: E402

from correction_ablation_test import HEAVY_FAVORITE_DECILE, build_match_rows, favorite_frame  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from pedigree_market_premium_test import TOURNAMENTS  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

BASELINE_ATP_MATCHES = 864
DOUBLE_THRESHOLD = 2 * BASELINE_ATP_MATCHES
ORIGINAL_GAP = (0.071, 0.020, 0.134)          # observed, lo, hi
ORIGINAL_LOGLOSS_DIFF = (0.0100, 0.0007, 0.0199)

# best-effort guesses only - see module docstring
EXPANSION_CANDIDATES = [
    ("ATP", "usopen"), ("ATP", "chengdu"), ("ATP", "tokyo"), ("ATP", "beijing"),
    ("ATP", "shanghai"), ("ATP", "vienna"), ("ATP", "basel"), ("ATP", "paris"),
]


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    atp_matches = load_matches_for_tour("ATP")
    all_slugs = [t for t in TOURNAMENTS if t[0] == "ATP"] + EXPANSION_CANDIDATES

    frames, skipped, used_slugs = [], [], []
    for tour, slug in all_slugs:
        try:
            frame, warning = build_match_rows(tour, slug, atp_matches)
        except RuntimeError as e:
            skipped.append(f"{slug}: {e}")
            continue
        if warning:
            print(f"  WARNING: {warning}", file=sys.stderr)
        if len(frame) == 0:
            skipped.append(f"{slug}: 0 usable rows")
            continue
        frames.append(frame)
        used_slugs.append(slug)

    all_rows = pd.concat(frames, ignore_index=True)
    n = len(all_rows)
    growth = n / BASELINE_ATP_MATCHES
    print(f"ATP tournaments used ({len(used_slugs)}): {used_slugs}")
    print(f"{len(skipped)} candidate slug(s) unavailable/unusable this run: {skipped}")
    print(f"\nCurrent real ATP match count: {n}  (baseline {BASELINE_ATP_MATCHES}, {growth:.2f}x growth, "
          f"target for rerun: >= {DOUBLE_THRESHOLD} = 2.00x)")

    if n < DOUBLE_THRESHOLD:
        print(f"\nBelow 2x threshold - no rerun yet. Checking again next scheduled tick.")
        return

    print(f"\n{'=' * 90}\nSample has (roughly) doubled - rerunning the ATP heavy-favorite Platt check\n{'=' * 90}")
    fav = favorite_frame(all_rows, "model_prob_a_full")
    heavy_cutoff = fav["conf_full"].quantile(HEAVY_FAVORITE_DECILE)
    frame2 = favorite_frame(all_rows, "model_prob_a_full", "model_prob_a_no_confidence_calibration")
    heavy = frame2[frame2["conf_full"] >= heavy_cutoff]
    print(f"Heavy-favorite cutoff (top {(1 - HEAVY_FAVORITE_DECILE):.0%}): conf >= {heavy_cutoff:.1%}, n={len(heavy)}")

    if len(heavy) < 15:
        print("Still too few heavy-favorite rows even after doubling the raw sample - reporting progress only.")
        return

    import numpy as np
    actual = heavy["won"].mean()
    full_gap, full_lo, full_hi = cluster_bootstrap_ci(
        heavy.assign(_a=heavy["won"].astype(int), _s=heavy["conf_full"]), "_a", "_s", group_col="player")
    heavy = heavy.assign(
        loss_full=-np.where(heavy["won"], np.log(np.clip(heavy["conf_full"], 1e-9, 1 - 1e-9)),
                             np.log(np.clip(1 - heavy["conf_full"], 1e-9, 1 - 1e-9))),
        loss_variant=-np.where(heavy["won"], np.log(np.clip(heavy["conf_variant"], 1e-9, 1 - 1e-9)),
                                np.log(np.clip(1 - heavy["conf_variant"], 1e-9, 1 - 1e-9))),
    )
    diff, dlo, dhi = cluster_bootstrap_ci(
        heavy.assign(_a=heavy["loss_full"], _s=heavy["loss_variant"]), "_a", "_s", group_col="player")

    print(f"\nReal win rate: {actual:.1%}  |  full-stack mean conf: {heavy['conf_full'].mean():.1%}")
    print(f"Full-stack calibration gap: {full_gap:+.1%} CI[{full_lo:+.1%},{full_hi:+.1%}]  "
          f"(original: +7.1% CI[+2.0%,+13.4%])")
    print(f"Log-loss (full - no_confidence_calibration): {diff:+.4f} CI[{dlo:+.4f},{dhi:+.4f}]  "
          f"(original: +0.0100 CI[+0.0007,+0.0199])")

    orig_sig = ORIGINAL_LOGLOSS_DIFF[1] > 0
    new_sig = dlo > 0
    if new_sig and diff >= ORIGINAL_LOGLOSS_DIFF[0]:
        verdict = "HELD AND STRENGTHENED - real evidence of a genuine distribution shift, worth acting on."
    elif new_sig:
        verdict = "HELD (still significant, similar or smaller magnitude) - real, not yet stronger."
    elif not new_sig and diff > 0:
        verdict = "WEAKENED (still same direction but no longer significant) - leaning small-sample artifact."
    else:
        verdict = "REVERSED OR VANISHED - confirms this was a small-sample artifact, consistent with several other patterns tonight."
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
