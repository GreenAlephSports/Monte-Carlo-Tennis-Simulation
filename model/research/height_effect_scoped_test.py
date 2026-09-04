"""Refits the height_diff -> outperformance-beyond-Elo effect on the ONE subpopulation that
height_effect_validation_test.py found consistently significant: prime-age (23-29 average),
best_rank <=150, and the 2015-2024 window (the only two 5-year decade buckets that individually
cleared significance). Every other axis validation_test checked (151+ rank, <23 and 29+ age,
2005-2014 and 2025-2029) is EXCLUDED here on purpose - this script asks whether the narrower,
precisely-scoped population clears significance on its own terms, not whether the broad population
does (that's already answered).

Same full rigor as every other test in this project:
  - full available history within the 2015-2024 restriction (both tours, no further truncation)
  - player-clustered bootstrap CI on the scoped population (same cluster_bootstrap_coef as
    height_serve_proxy_test.py / height_effect_validation_test.py)
  - a genuine held-out check: chronological (not random) train/test split within the scoped
    population - fit on the earlier matches, refit independently on the later matches, and report
    whether BOTH halves clear significance with the same sign. A coefficient that only lives in one
    half of its own scoped window isn't ship-worthy just because the pooled scoped number looks big.

Honest failure mode this script is built to catch: scoping this tightly could leave too few matches
to trust ANY verdict (a real risk flagged by the user) - MIN_BUCKET_N below is applied identically to
the pooled scoped result and to each half of the holdout split, and "too thin" is reported as its own
outcome, not glossed over.

Usage:
    python model/research/height_effect_scoped_test.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elo_ratings import load_matches_for_tour  # noqa: E402
from height_effect_validation_test import (  # noqa: E402
    build_dataset, fit_height_coef, load_birth_year_map, MIN_BUCKET_N,
)
from height_serve_proxy_test import load_height_map  # noqa: E402

# the two decade buckets validation_test found significant (2015-2019 and 2020-2024)
SCOPE_DECADES = (2015, 2020)
SCOPE_AGE_RANGE = (23, 29)     # avg age of both players, [lo, hi)
SCOPE_RANK_MAX = 150           # best (min) real rank of the two players


def report(label, df):
    if len(df) < MIN_BUCKET_N:
        print(f"  {label:<42} n={len(df):<7} TOO THIN (<{MIN_BUCKET_N}) - no verdict")
        return None
    coef, se, z, lo, hi = fit_height_coef(df)
    sig = abs(z) > 1.96 and (lo > 0 or hi < 0)
    tag = "SIGNIFICANT" if sig else "not significant"
    print(f"  {label:<42} n={len(df):<7} coef={coef:+.6f}  z={z:+.2f}  "
          f"boot CI=[{lo:+.6f},{hi:+.6f}]  {tag}")
    return coef, z, sig


def run():
    height = load_height_map()
    birth_year = load_birth_year_map()

    frames = []
    for tour in ("ATP", "WTA"):
        matches = load_matches_for_tour(tour)
        rows = build_dataset(tour, matches, height, birth_year)
        frames.append(rows)
    all_rows = pd.concat(frames, ignore_index=True)

    usable = all_rows.dropna(subset=["height_a", "height_b"]).copy()
    usable["height_diff"] = usable["height_a"] - usable["height_b"]

    # --- apply all three scope restrictions at once ---
    scoped = usable[usable["decade"].isin(SCOPE_DECADES)].copy()
    scoped = scoped.dropna(subset=["rank_a", "rank_b", "birth_a", "birth_b"])
    scoped["best_rank"] = scoped[["rank_a", "rank_b"]].min(axis=1)
    scoped["age_a"] = scoped["match_year"] - scoped["birth_a"]
    scoped["age_b"] = scoped["match_year"] - scoped["birth_b"]
    scoped["avg_age"] = (scoped["age_a"] + scoped["age_b"]) / 2
    lo_age, hi_age = SCOPE_AGE_RANGE
    scoped = scoped[
        (scoped["avg_age"] >= lo_age) & (scoped["avg_age"] < hi_age)
        & (scoped["best_rank"] <= SCOPE_RANK_MAX)
    ].sort_values("date").reset_index(drop=True)

    print(f"Scope: match_year in decade buckets {SCOPE_DECADES} (i.e. 2015-2024), avg age in "
          f"[{lo_age},{hi_age}), best_rank <= {SCOPE_RANK_MAX}.")
    print(f"{len(usable)} total height-known matches (both tours, full history) -> "
          f"{len(scoped)} matches survive this triple restriction "
          f"({len(scoped) / len(usable):.1%} of the height-known pool).")
    if len(scoped) > 0:
        print(f"Scoped date range: {scoped['date'].min().date()} to {scoped['date'].max().date()}")
        print(f"Scoped tour split: ATP={sum(scoped['tour'] == 'ATP')}, WTA={sum(scoped['tour'] == 'WTA')}")

    print(f"\n{'=' * 92}\nPOOLED SCOPED RESULT (in-sample - the number this whole exercise is "
          f"testing whether to trust)\n{'=' * 92}")
    pooled = report("Prime-age + rank<=150 + 2015-2024", scoped)

    print(f"\n{'=' * 92}\nHELD-OUT CHECK: chronological split within the scoped population "
          f"(fit independently on each half - a real effect should show up in both, not just one)\n"
          f"{'=' * 92}")
    if len(scoped) < 2 * MIN_BUCKET_N:
        print(f"  Scoped population too small to split into two independently-trustworthy halves "
              f"(need >= {2 * MIN_BUCKET_N}, have {len(scoped)}) - skipping the holdout check; "
              f"the pooled number above is all this population can support.")
        train_res = test_res = None
    else:
        split_idx = len(scoped) // 2
        train = scoped.iloc[:split_idx]
        test = scoped.iloc[split_idx:]
        print(f"Earlier half: {train['date'].min().date()} to {train['date'].max().date()} (n={len(train)})")
        train_res = report("Earlier half (train)", train)
        print(f"Later half:   {test['date'].min().date()} to {test['date'].max().date()} (n={len(test)})")
        test_res = report("Later half (test)", test)

    print(f"\n{'=' * 92}\nVERDICT\n{'=' * 92}")
    if pooled is None:
        print("Scoped population is too thin even for the pooled estimate - this restriction is not "
              "usable as a standalone correction. Stick with the broader finding's own caveats "
              "instead of trying to ship this narrower cut.")
    else:
        coef, z, sig = pooled
        print(f"Pooled scoped coefficient: coef={coef:+.6f}, z={z:+.2f}, "
              f"{'SIGNIFICANT' if sig else 'not significant'} on its own terms (n={len(scoped)}).")
        if train_res is None:
            print("Held-out split not possible (population too small) - this result rests entirely "
                  "on the single pooled scoped sample above, with no independent replication check. "
                  "Treat it as suggestive, not confirmed.")
        else:
            _, _, sig_train = train_res
            _, _, sig_test = test_res
            both_sig = sig_train and sig_test
            same_sign = (train_res[0] > 0) == (test_res[0] > 0)
            if both_sig and same_sign:
                print("Both chronological halves independently clear significance with the same "
                      "sign - this scoped correction replicates within its own population, not just "
                      "in the pooled number. This is the strongest form of support this dataset can "
                      "offer for a conditional correction this narrow.")
            elif same_sign and not both_sig:
                print("Same sign in both halves, but at least one half does not independently clear "
                      "significance - directionally consistent, but under-powered once split. "
                      "Read the pooled figure as suggestive rather than a ship-ready conditional "
                      "correction: it depends on pooling both halves to reach significance.")
            else:
                print("The two chronological halves do NOT agree in sign - the pooled scoped result "
                      "is not a stable population-level effect, it is being carried by one sub-period "
                      "within an already-narrow scope. This does NOT clear the bar for a ship-ready "
                      "conditional correction.")


if __name__ == "__main__":
    run()
