"""Definitive, full-historical validation of the adaptive K-factor idea prototyped in
elo_k_factor_test.py: K = 250/(5+n)^0.4 (n = a player's own career match count so far) vs. the
current production flat K_FACTOR=32, across the ENTIRE Kaggle match history for both tours (~228K
player-perspective rows, ~2,800 tournament editions, 2000-2026 ATP / 2006-2026 WTA) - not the
2-tournament Cincinnati + Canadian Open sample elo_k_factor_test.py used, which already showed a
sign flip between the two events and a CI straddling zero even at that small scale. This is the
same escalation lookback_full_historical_test.py applied to the lookback-window question after ITS
small-sample check looked promising (that one reversed at scale, worse in 4/6 eras) - here the
small-sample check didn't even look promising, so this is the same standard applied honestly in
the other direction: confirm the null result holds, or find that it was hiding a real effect the
2-tournament sample was too small to see.

Methodology - deliberately CHEAPER than lookback_full_historical_test.py's per-edition windowed
replay, and legitimately so: unlike the lookback-window question (where the window itself changes
which matches are even included, forcing a real per-edition rebuild), a K-factor variant only
changes how much a match MOVES a rating, not which matches are included - so a single continuously-
updated online Elo pass per tour per variant (same simplification elite_opponent_residual_test.py
and thin_history_rank_blend_test.py already use, and the same one elo_k_factor_test.py itself used
at the 2-tournament scale) is exactly equivalent to a from-scratch-at-every-edition replay here, at
a fraction of the cost. Predictions are frozen per tournament edition (every match in an edition
scored off the SAME pre-edition snapshot Elo, no in-tournament lookahead) exactly like every other
"same rigor" test in this series. Raw Elo win probability (expected_score), not the full
win_probability() pipeline - same convention as elite_opponent_residual_test.py and
lookback_full_historical_test.py, necessary at this scale.

Like the lookback-window full-historical test, this fits NO parameters on train-era data - flat and
adaptive are both fully mechanical Elo-update variants, so every edition's prediction is already
genuinely out-of-sample relative to that edition's own Elo by construction. The headline number
still uses the standard chronological 80/20 tournament-edition split (test = the most recent 20% of
editions), and the full-period decade breakdown is an additional stability check, not a second
independent held-out claim - same disclosure as lookback_full_historical_test.py.

Experience-group breakdown (the actually-relevant lens for a K-factor test, unlike the age lens
that mattered for the lookback-window question): each row is bucketed by the player's own career
match count AT the frozen snapshot point - thin (<20), developing (20-100), established (100+) -
since adaptive K's entire hypothesis is that thin/developing players' ratings should move faster
and established players' should move slower, not that the effect is age-correlated.

Usage:
    python model/research/elo_k_factor_full_historical_test.py

FINAL VERDICT (2026-08-26): REJECTED, decisively - not added to production. Unlike the
lookback-window question (where a promising 2-tournament result reversed to a modest, mixed loss at
full scale), the adaptive-K variant's 2-tournament result had already shown no signal at all
(sign-flipping, CI straddling zero) - and full scale doesn't rescue it, it reveals the small sample
was hiding a real, substantial, and completely unanimous loss. Combined held-out (both tours,
46778 rows, 561 editions): mean log-loss improvement -0.0093, 95% player-clustered bootstrap CI
[-0.0108, -0.0079] - clearly WORSE than flat K, not merely indistinguishable. Every single breakdown
agrees, with zero exceptions anywhere:
  - Per-tour: ATP worse (-0.0115, CI [-0.0137, -0.0096]), WTA worse (-0.0061, CI [-0.0080, -0.0042]).
  - Experience groups (the lens that actually matters for a K-factor hypothesis): thin (<20
    matches) is the WORST-hit bucket (-0.0229, CI [-0.0290, -0.0165]) - i.e. adaptive K hurts most
    exactly where it was designed to help most, moving thin players' ratings so aggressively off
    single results that it overshoots and actively degrades calibration. Developing (20-99 matches,
    -0.0102) and established (100+, -0.0062) are both worse too, just less severely.
  - Decade breakdown: WORSE than baseline in all 6 of 6 eras (2000-2004 through 2025-2029), every
    single one with a CI that excludes zero - no era where it helps, no era where it's even neutral.
Conclusion: the mechanism itself is unsound, not just unproven - moving thin-history players' Elo
this aggressively (K up to ~132 near a debut) makes their ratings noisier and worse-calibrated, not
better, contrary to the original intuition. Flat K_FACTOR=32 stays in production unchanged.
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import EPS, log_loss  # noqa: E402
from elo_k_factor_test import adaptive_k  # noqa: E402
from elo_ratings import K_FACTOR, STARTING_ELO, expected_score, load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

TRAIN_FRACTION = 0.8
VARIANTS = {
    "A. flat (K=32, production)": "flat",
    "B. adaptive (K=250/(5+n)^0.4)": "adaptive",
}
BASELINE_LABEL = "A. flat (K=32, production)"
EXPERIENCE_BUCKETS = [
    ("thin (<20 matches)", lambda n: n < 20),
    ("developing (20-99 matches)", lambda n: 20 <= n < 100),
    ("established (100+ matches)", lambda n: n >= 100),
]


def build_frozen_predictions(df, k_mode, tour_label):
    """Single continuously-updated online Elo pass, frozen per tournament edition (see module
    docstring for why this is equivalent to a full rebuild for a K-factor-only variant). Also
    carries each row's player_matches_before (career match count at the frozen snapshot point) for
    the experience-group breakdown."""
    df = df.sort_values("Date", kind="stable").copy()
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)
    edition_start = df.groupby("edition_id")["Date"].transform("min")
    editions = (
        df.assign(edition_start=edition_start)[["edition_id", "edition_start"]]
        .drop_duplicates().sort_values("edition_start").reset_index(drop=True)
    )

    elo, matches_played = {}, {}
    rows = []
    t0 = time.time()
    for idx, edition_id in enumerate(editions["edition_id"]):
        edition_matches = df[df["edition_id"] == edition_id]
        snap_elo, snap_count = dict(elo), dict(matches_played)

        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            e1, e2 = snap_elo.get(p1, STARTING_ELO), snap_elo.get(p2, STARTING_ELO)
            pred1 = expected_score(e1, e2)
            win1 = 1 if winner == p1 else 0
            n1, n2 = snap_count.get(p1, 0), snap_count.get(p2, 0)
            rows.append((edition_id, row.Date, p1, p2, pred1, win1, n1))
            rows.append((edition_id, row.Date, p2, p1, 1 - pred1, 1 - win1, n2))

        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            elo.setdefault(p1, STARTING_ELO)
            elo.setdefault(p2, STARTING_ELO)
            s1 = 1.0 if winner == p1 else 0.0
            e1 = expected_score(elo[p1], elo[p2])
            if k_mode == "flat":
                k1 = k2 = K_FACTOR
            else:
                k1 = adaptive_k(matches_played.get(p1, 0))
                k2 = adaptive_k(matches_played.get(p2, 0))
            elo[p1] += k1 * (s1 - e1)
            elo[p2] += k2 * ((1 - s1) - (1 - e1))
            matches_played[p1] = matches_played.get(p1, 0) + 1
            matches_played[p2] = matches_played.get(p2, 0) + 1

        if (idx + 1) % 300 == 0:
            print(f"    [{tour_label}, k={k_mode}] {idx + 1}/{len(editions)} editions replayed "
                  f"({time.time() - t0:.0f}s elapsed)")

    preds = pd.DataFrame(rows, columns=[
        "edition_id", "date", "player", "opponent", "pred_win", "actual_win", "player_matches_before",
    ])
    preds["loss"] = log_loss(preds["actual_win"].values, preds["pred_win"].values)
    print(f"    [{tour_label}, k={k_mode}] done: {len(editions)} editions, {len(preds)} "
          f"player-perspective rows, {time.time() - t0:.0f}s total")
    return preds, editions


def bootstrap_verdict(long_baseline, long_variant, merge_keys=("edition_id", "date", "player", "opponent")):
    merged = long_baseline[[*merge_keys, "loss"]].merge(
        long_variant[[*merge_keys, "loss"]], on=list(merge_keys), suffixes=("_baseline", "_variant"))
    observed, lo, hi = cluster_bootstrap_ci(merged, "loss_baseline", "loss_variant", group_col="player")
    verdict = "BEATS baseline (CI excludes zero, >0)" if lo > 0 else (
        "WORSE than baseline (CI excludes zero, <0)" if hi < 0 else "NOT distinguishable (CI straddles zero)")
    return merged, observed, lo, hi, verdict


def run():
    tours = ["ATP", "WTA"]
    all_preds = {}
    all_editions = {}

    for tour in tours:
        matches = load_matches_for_tour(tour)
        print(f"\n{'#' * 90}\n{tour}: {len(matches)} total matches, "
              f"{matches['Date'].min().date()} to {matches['Date'].max().date()}\n{'#' * 90}")
        for label, k_mode in VARIANTS.items():
            preds, editions = build_frozen_predictions(matches, k_mode, tour)
            all_preds[(tour, label)] = preds
            all_editions[tour] = editions

    test_edition_ids = {}
    for tour in tours:
        editions = all_editions[tour]
        split_idx = int(len(editions) * TRAIN_FRACTION)
        test_edition_ids[tour] = set(editions["edition_id"].iloc[split_idx:])
        print(f"\n{tour}: {len(editions)} editions total; held-out test era = most recent "
              f"{len(editions) - split_idx} editions, from "
              f"{editions['edition_start'].iloc[split_idx].date()} to "
              f"{editions['edition_start'].iloc[-1].date()}")

    # ============================================================================
    # HEADLINE: held-out (last 20% of editions), BOTH tours combined
    # ============================================================================
    print(f"\n{'=' * 90}\nHEADLINE - HELD-OUT TEST ERA (most recent 20% of tournament editions), "
          f"BOTH TOURS COMBINED\n{'=' * 90}")
    combined_test = {}
    for label in VARIANTS:
        parts = []
        for tour in tours:
            df = all_preds[(tour, label)]
            parts.append(df[df["edition_id"].isin(test_edition_ids[tour])].assign(tour=tour))
        combined_test[label] = pd.concat(parts, ignore_index=True)
        cdf = combined_test[label]
        print(f"\n{label}: {len(cdf)} held-out player-perspective rows ({cdf['edition_id'].nunique()} editions)")
        print(f"  mean log-loss = {cdf['loss'].mean():.4f}")
        print(f"  favorite calibration: assigned P(win) = {cdf['pred_win'].mean():.1%}, "
              f"actual win rate = {cdf['actual_win'].mean():.1%}")

    merge_keys = ("tour", "edition_id", "date", "player", "opponent")
    for label in VARIANTS:
        if label == BASELINE_LABEL:
            continue
        merged, observed, lo, hi, verdict = bootstrap_verdict(
            combined_test[BASELINE_LABEL], combined_test[label], merge_keys=merge_keys)
        print(f"\n{label} vs. baseline (HELD-OUT, COMBINED): {len(merged)} matched rows")
        print(f"  mean log-loss improvement (baseline - variant, >0 = variant better): "
              f"{observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  VERDICT: {verdict}")

    # ============================================================================
    # PER-TOUR breakdown, held-out era
    # ============================================================================
    print(f"\n{'=' * 90}\nPER-TOUR BREAKDOWN, held-out test era\n{'=' * 90}")
    for tour in tours:
        print(f"\n--- {tour} only ---")
        base_t = combined_test[BASELINE_LABEL][combined_test[BASELINE_LABEL]["tour"] == tour]
        for label in VARIANTS:
            if label == BASELINE_LABEL:
                continue
            var_t = combined_test[label][combined_test[label]["tour"] == tour]
            merged, observed, lo, hi, verdict = bootstrap_verdict(base_t, var_t, merge_keys=merge_keys)
            print(f"  {label}: {len(merged)} rows, improvement {observed:+.4f}, "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")

    # ============================================================================
    # EXPERIENCE-GROUP breakdown, held-out era, BOTH tours combined
    # ============================================================================
    print(f"\n{'=' * 90}\nEXPERIENCE-GROUP BREAKDOWN, held-out test era, both tours combined "
          f"(bucketed by career matches at the frozen snapshot point)\n{'=' * 90}")
    for group_label, cond in EXPERIENCE_BUCKETS:
        print(f"\n--- {group_label} ---")
        base_g = combined_test[BASELINE_LABEL][combined_test[BASELINE_LABEL]["player_matches_before"].apply(cond)]
        for label in VARIANTS:
            g = combined_test[label][combined_test[label]["player_matches_before"].apply(cond)]
            n = len(g)
            if n == 0:
                continue
            print(f"  {label}: n={n} | assigned P(win)={g['pred_win'].mean():.1%} | "
                  f"actual win rate={g['actual_win'].mean():.1%} | "
                  f"gap={g['pred_win'].mean() - g['actual_win'].mean():+.1%} | "
                  f"mean log-loss={g['loss'].mean():.4f}")
        for label in VARIANTS:
            if label == BASELINE_LABEL:
                continue
            var_g = combined_test[label][combined_test[label]["player_matches_before"].apply(cond)]
            merged = base_g[["tour", "edition_id", "date", "player", "opponent", "loss"]].merge(
                var_g[["tour", "edition_id", "date", "player", "opponent", "loss"]],
                on=["tour", "edition_id", "date", "player", "opponent"], suffixes=("_baseline", "_variant"))
            if len(merged) < 10:
                print(f"  {label} vs. baseline ({group_label}): only {len(merged)} rows - too few to bootstrap")
                continue
            observed, lo, hi = cluster_bootstrap_ci(merged, "loss_baseline", "loss_variant", group_col="player")
            verdict = "BEATS baseline" if lo > 0 else ("WORSE than baseline" if hi < 0 else "not distinguishable")
            print(f"  {label} vs. baseline ({group_label}): {len(merged)} rows, improvement "
                  f"{observed:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")

    # ============================================================================
    # ERA / DECADE breakdown, FULL period (both tours) - stability check across history, not a
    # second independent held-out claim (see module docstring - no parameters are fit on train data)
    # ============================================================================
    print(f"\n{'=' * 90}\nERA / DECADE BREAKDOWN, FULL PERIOD (both tours combined, all editions)"
          f"\n{'=' * 90}")
    full_by_label = {}
    for label in VARIANTS:
        parts = [all_preds[(tour, label)].assign(tour=tour) for tour in tours]
        full_by_label[label] = pd.concat(parts, ignore_index=True)
        full_by_label[label]["decade"] = (full_by_label[label]["date"].dt.year // 5) * 5

    decades = sorted(full_by_label[BASELINE_LABEL]["decade"].unique())
    for decade in decades:
        base_d = full_by_label[BASELINE_LABEL][full_by_label[BASELINE_LABEL]["decade"] == decade]
        if len(base_d) < 200:
            continue
        yr_range = f"{decade}-{decade + 4}"
        print(f"\n--- {yr_range} ---")
        for label in VARIANTS:
            g = full_by_label[label][full_by_label[label]["decade"] == decade]
            print(f"  {label}: n={len(g)} | mean log-loss={g['loss'].mean():.4f} | "
                  f"favorite calib: assigned={g['pred_win'].mean():.1%} vs actual={g['actual_win'].mean():.1%}")
        for label in VARIANTS:
            if label == BASELINE_LABEL:
                continue
            var_d = full_by_label[label][full_by_label[label]["decade"] == decade]
            merged = base_d[["tour", "edition_id", "date", "player", "opponent", "loss"]].merge(
                var_d[["tour", "edition_id", "date", "player", "opponent", "loss"]],
                on=["tour", "edition_id", "date", "player", "opponent"], suffixes=("_baseline", "_variant"))
            if len(merged) < 10:
                continue
            observed, lo, hi = cluster_bootstrap_ci(merged, "loss_baseline", "loss_variant", group_col="player")
            verdict = "BEATS baseline" if lo > 0 else ("WORSE than baseline" if hi < 0 else "not distinguishable")
            print(f"  {label} vs. baseline ({yr_range}): {len(merged)} rows, improvement "
                  f"{observed:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")


if __name__ == "__main__":
    run()
