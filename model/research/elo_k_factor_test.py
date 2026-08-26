"""Tests whether an adaptive, per-player K-factor - K = 250 / (5 + n)^0.4, where n is that
player's own career match count so far (within the training window, at the moment of the match
being updated) - calibrates better than the current production flat K_FACTOR=32
(elo_ratings.K_FACTOR), applied identically to every player regardless of how much real history
backs their rating.

Same two real, now-fully-concluded 2026 ATP hard-court tournaments elo_lookback_test.py used
(Cincinnati "Western & Southern Financial Group Masters", n=92, and the Canadian Open, n=91 -
see that file's data-reality note on why there's exactly one Canadian Open row, not two), same
frozen-per-edition-cutoff / held-out / player-clustered-bootstrap rigor, and this file reuses
elo_lookback_test.build_real_tournament_matches, calibrate_variant and _bootstrap_verdict
directly rather than re-implementing them - only calculate_elo_variant's K-factor mechanism is new
here; the lookback window itself is held at production's 5yr hard cutoff throughout (that question
was already asked and answered - see lookback_full_historical_test.py's verdict - this test isolates
K-factor alone as the one variable under test).

Adaptive-K mechanism: K = 250 / (5 + n)^0.4 gives a debut match (n=0) K = 250/5^0.4 ~= 131.6 (moves
FAST - a brand new player's rating should update aggressively off a single result, same intuition
production's Tier-3 STARTING_ELO placeholder already leans on informally), decaying to K ~= 32.5
(essentially flat-K parity) around n ~= 140 prior matches, and continuing to shrink below flat-K
for a long-tenured veteran with hundreds of matches on record (their rating should barely move off
any one result). n is tracked as a running per-player count of matches already processed earlier
in this SAME chronological within-window replay (0 for that player's first match in the window,
incrementing after each of their matches) - a no-lookahead-consistent proxy for "how much real
history already backs this rating," not a separate career-total lookup outside the window.

Usage:
    python model/research/elo_k_factor_test.py

FINAL VERDICT (2026-08-26): REJECTED - not a statistically validated improvement, no production
change made. Held out on the combined Cincinnati + Canadian Open 2026 real matches (n=183, 364
player-perspective rows), the adaptive-K variant shows no distinguishable effect: mean log-loss
improvement +0.0006, 95% player-clustered bootstrap CI [-0.0091, +0.0104] - straddles zero. Worse
than a simple null result, it isn't even directionally consistent: per-tournament, it's WORSE at
Cincinnati (-0.0092, CI [-0.0207, +0.0017]) and BETTER at Canadian Open (+0.0108, CI [-0.0024,
+0.0236]) - neither individually significant, and the sign flip between two real, same-surface,
same-week-adjacent tournaments is itself evidence this isn't a real, stable effect rather than
noise at this sample size. Applying the same standard that correctly rejected the 3-year lookback
window (small-sample results that look encouraging need a genuinely held-out check before trusting
them, not just a directional read) - here the small-sample check does not even clear that lower
bar, so there's no promising signal to justify the larger-scale full-historical validation
lookback_full_historical_test.py ran for the lookback-window question. Flat K_FACTOR=32 stays in
production; elo_ratings.calculate_elo_ratings is unchanged.

UPDATE (2026-08-26): the larger-scale check was run anyway (elo_k_factor_full_historical_test.py,
~228K rows both tours, full Kaggle history) despite this file's small-sample result giving no
promising signal to chase - and it reveals the 2-tournament sample wasn't just underpowered, it was
actively MASKING a real, substantial, unanimous negative effect. See that file's verdict: WORSE
than flat K at full scale, combined AND every single per-tour/experience-group/decade breakdown,
with no exceptions. Confirms REJECTED even more strongly than this file's own result suggested.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elo_lookback_test import (  # noqa: E402
    ELO_COLUMNS, SCRATCH_DIR, TOURNAMENTS, _bootstrap_verdict, build_real_tournament_matches,
    calibrate_variant,
)
from elo_ratings import K_FACTOR, LOOKBACK_YEARS, STARTING_ELO, SURFACES, SURFACE_BLEND_K, expected_score  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402


def adaptive_k(n):
    """K = 250 / (5 + n)^0.4 - see module docstring for the shape (fast for a thin-history
    player, converging toward and then below flat K_FACTOR=32 as n grows)."""
    return 250 / (5 + n) ** 0.4


def calculate_elo_variant(df, cutoff_date, k_mode, lookback_years=LOOKBACK_YEARS):
    """Same overall/surface Elo mechanism and surface-blend shrinkage as
    elo_ratings.calculate_elo_ratings, holding the lookback window at production's flat
    lookback_years (5yr hard cutoff by default - NOT the variable under test here, see module
    docstring). k_mode selects how K is chosen per player, per match:

      "flat"      - production's constant K_FACTOR=32 for every player, every match (both sides
                    of a match share the same K, exactly matching calculate_elo_ratings).
      "adaptive"  - each side of a match uses its OWN K = adaptive_k(n), n = that player's own
                    prior-match count so far in this replay (see module docstring) - two players
                    in the same match can move by different amounts off the same result.
    """
    if k_mode not in ("flat", "adaptive"):
        raise ValueError(f"k_mode must be 'flat' or 'adaptive', got {k_mode!r}")

    cutoff_ts = pd.Timestamp(cutoff_date)
    df = df[df["Date"] < cutoff_ts]
    max_date = df["Date"].max()
    if lookback_years is not None:
        lookback_start = max_date - pd.DateOffset(years=lookback_years)
        df = df[df["Date"] >= lookback_start]
    df = df.sort_values("Date", kind="stable")

    overall_elo, surface_elo, surface_matches = {}, {s: {} for s in SURFACES}, {s: {} for s in SURFACES}
    match_count, current_rank, last_match_date = {}, {}, {}
    has_rank_columns = {"Rank_1", "Rank_2"}.issubset(df.columns)

    for row in df.itertuples(index=False):
        p1, p2, winner, surface = row.Player_1, row.Player_2, row.Winner, row.Surface
        last_match_date[p1] = row.Date
        last_match_date[p2] = row.Date

        if k_mode == "flat":
            k_p1 = k_p2 = K_FACTOR
        else:
            k_p1 = adaptive_k(match_count.get(p1, 0))
            k_p2 = adaptive_k(match_count.get(p2, 0))

        overall_elo.setdefault(p1, STARTING_ELO)
        overall_elo.setdefault(p2, STARTING_ELO)
        score_p1 = 1.0 if winner == p1 else 0.0
        expected_p1 = expected_score(overall_elo[p1], overall_elo[p2])
        overall_elo[p1] += k_p1 * (score_p1 - expected_p1)
        overall_elo[p2] += k_p2 * ((1 - score_p1) - (1 - expected_p1))

        if surface in SURFACES:
            ratings, counts = surface_elo[surface], surface_matches[surface]
            ratings.setdefault(p1, STARTING_ELO)
            ratings.setdefault(p2, STARTING_ELO)
            expected_p1_s = expected_score(ratings[p1], ratings[p2])
            ratings[p1] += k_p1 * (score_p1 - expected_p1_s)
            ratings[p2] += k_p2 * ((1 - score_p1) - (1 - expected_p1_s))
            counts[p1] = counts.get(p1, 0) + 1
            counts[p2] = counts.get(p2, 0) + 1

        if has_rank_columns:
            if row.Rank_1 > 0:
                current_rank[p1] = row.Rank_1
            if row.Rank_2 > 0:
                current_rank[p2] = row.Rank_2

        match_count[p1] = match_count.get(p1, 0) + 1
        match_count[p2] = match_count.get(p2, 0) + 1

    records = []
    for player in sorted(overall_elo.keys()):
        last_date = last_match_date.get(player)
        record = {
            "player": player, "overall_elo": overall_elo[player], "current_rank": current_rank.get(player),
            "days_since_last_match": (cutoff_ts - last_date).days if last_date is not None else None,
        }
        for surface in SURFACES:
            mc = surface_matches[surface].get(player, 0)
            raw_elo = surface_elo[surface].get(player, STARTING_ELO)
            surface_weight = mc / (mc + SURFACE_BLEND_K)
            record[f"{surface.lower()}_elo"] = surface_weight * raw_elo + (1 - surface_weight) * overall_elo[player]
            record[f"{surface.lower()}_matches"] = mc
        records.append(record)
    return pd.DataFrame.from_records(records, columns=ELO_COLUMNS)


def run():
    matches = load_matches_for_tour("ATP")

    variant_specs = {
        "A. flat (K=32, production)": dict(k_mode="flat"),
        "B. adaptive (K=250/(5+n)^0.4)": dict(k_mode="adaptive"),
    }
    baseline_label = "A. flat (K=32, production)"

    per_tournament_longs = {label: [] for label in variant_specs}
    real_matches_by_tournament = {}
    for t_label, kaggle_name, cutoff in TOURNAMENTS:
        real_matches = build_real_tournament_matches(matches, kaggle_name, cutoff)
        real_matches_by_tournament[t_label] = real_matches
        print(f"\n{'#' * 90}\n{t_label} 2026 (ATP): {len(real_matches)} real matches found "
              f"(cutoff {cutoff.date()}, no-lookahead)\n{'#' * 90}")
        for i, (label, spec) in enumerate(variant_specs.items()):
            ratings_df = calculate_elo_variant(matches, cutoff, **spec)
            variant_path = SCRATCH_DIR / f"elo_k_factor_variant_{t_label.replace(' ', '_')}_{i}.csv"
            calib, long_df = calibrate_variant(f"{t_label} | {label}", ratings_df, real_matches, variant_path)
            long_df["tournament"] = t_label
            per_tournament_longs[label].append(long_df)

    longs = {label: pd.concat(dfs, ignore_index=True) for label, dfs in per_tournament_longs.items()}
    n_total_matches = sum(len(v) for v in real_matches_by_tournament.values())
    n_total_rows = len(longs[baseline_label])

    print(f"\n{'=' * 90}\nCOMBINED HELD-OUT RIGOR - {' + '.join(t for t, _, _ in TOURNAMENTS)}, "
          f"{n_total_matches} real matches, {n_total_rows} player-perspective rows\n{'=' * 90}")
    for label, long_df in longs.items():
        if label == baseline_label:
            continue
        merged, observed, lo, hi, verdict = _bootstrap_verdict(longs[baseline_label], long_df)
        print(f"\n{label} vs. baseline (COMBINED): {len(merged)} matched player-perspective rows")
        print(f"  mean log-loss improvement (baseline - variant, >0 = variant better): "
              f"{observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  VERDICT: {verdict}")

    print(f"\n{'=' * 90}\nPER-TOURNAMENT BREAKDOWN (same bootstrap check, one tournament at a time)"
          f"\n{'=' * 90}")
    for t_label, _, _ in TOURNAMENTS:
        print(f"\n--- {t_label} only ({len(real_matches_by_tournament[t_label])} matches) ---")
        for label, long_df in longs.items():
            if label == baseline_label:
                continue
            base_t = longs[baseline_label][longs[baseline_label]["tournament"] == t_label]
            var_t = long_df[long_df["tournament"] == t_label]
            merged, observed, lo, hi, verdict = _bootstrap_verdict(base_t, var_t)
            print(f"  {label} vs. baseline: {len(merged)} rows, improvement {observed:+.4f}, "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")


if __name__ == "__main__":
    run()
