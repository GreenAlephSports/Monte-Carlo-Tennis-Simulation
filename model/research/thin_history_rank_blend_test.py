"""Root-cause test (not another exclusion rule): for players with very few real matches behind
their Elo rating, does blending their rating more heavily toward a rank-implied Elo - instead of
leaning on overall_elo/STARTING_ELO the way the pipeline already does for thin SURFACE history -
produce genuinely better-calibrated predictions? Same question the Cincinnati data-quality filter
sidesteps by exclusion; this asks whether it can be fixed instead of just avoided.

Why this is a real candidate fix, not a patch: cincinnati_data_quality_filter_test.py found the
model's most extreme, wrongest-looking EV numbers (Fery A. at 4 matches rated 53.2% over a top-20
player it had almost no data on) concentrated among players with a handful of real matches. The
PIPELINE's existing thin-data handling (elo_ratings.SURFACE_BLEND_K) already blends a thin SURFACE
rating toward the player's OVERALL Elo - but for a player who is thin EVERYWHERE (career total, not
just on this surface), overall_elo is ALSO just STARTING_ELO=1500, which is a wildly optimistic
default for someone ranked outside the top 200. win_probability.py's own rank-gap correction
(_apply_rank_adjustment) exists and uses current_rank already - but it's deliberately gated to
|Elo diff| <= 50 (near-coin-flip matches only, the only regime it was ever validated in - see that
module's own docstring). A Fery-vs-elite-player matchup has a huge (wrong) Elo gap, so that existing
correction never even fires for exactly the cases this test is about.

Methodology (same rigor as every other correction test tonight):
  - Reuses elite_opponent_residual_test.py's own stated simplification: a single continuously-
    updated overall_elo (not the production pipeline's surface-specific, 5-year-windowed Elo) -
    tractable to compute across the full history, and the STARTING_ELO-default problem this test is
    about is identical in both versions. "matches_before" here means total career matches at the
    time of that match (continuous, not surface- or window-scoped) - a real simplification, stated
    plainly, not the pipeline's own hard_matches definition.
  - Chronological tournament-edition-level 80/20 train/test split, frozen ratings per edition - the
    same split boundary style as every other test in this series (never a random match-level split).
  - The rank-to-Elo mapping is FIT ONLY on train-era "solid" players (>=30 matches behind their
    rating - a real, trustworthy Elo) - never on the thin players it's meant to help, and never on
    test-era data.
  - Held-out validation, log-loss/Brier, on TEST-era rows only, restricted to the population this
    correction actually touches (at least one side thin, real current_rank known for that side) -
    the same "only score it on the population it's meant to fix" discipline elite_opponent_residual
    _test.py and layoff_test.py already use. Player-clustered bootstrap CI (reused directly from
    survivorship_upset_test.cluster_bootstrap_ci).
  - Blend weight uses the SAME functional form and the SAME K as the pipeline's existing
    SURFACE_BLEND_K (weight = matches / (matches + K)) - deliberately not fit/tuned fresh here, to
    avoid inventing a new free parameter on a limited dataset just to make this look better.

Usage:
    python model/research/thin_history_rank_blend_test.py [--tour ATP|WTA] [--thin-threshold N]

FINAL VERDICT (2026-08-26): REJECTED - not added to production. Held out on the full thin
population it targets (ATP, n=2998 test-era rows where at least one side has <10 career matches
and a known rank), the blend shows no significant overall effect: mean log-loss improvement
+0.0027, player-clustered 95% bootstrap CI [-0.0030, +0.0084] - straddles zero. Worse, it is
actively HARMFUL in the thinnest bucket it was specifically meant to help: for 0-2 career matches
(n=842, the Fery-A.-style case that motivated this test), log-loss gets WORSE under the blend
(0.6064 raw -> 0.6103 blended), and the share of predictions that are extreme (|pred-50%| > 30pp)
roughly DOUBLES (19.2% -> 32.3%) - the blend is making the model more confident in exactly the
population where it has the least real signal to be confident about. It only shows a (small,
untested-for-significance) improvement in the 3-9 match range, not the near-zero-match range the
original Fery-A. motivating case actually lives in. Conclusion: rank is not a safe stand-in for
missing career Elo history at the thinnest end - the fix this test hypothesized does not work for
the population it was built for. No production changes made. A related but distinct follow-up
question - whether the same rank/form-blend idea helps a different population, established players
whose current Elo lags a real recent trajectory shift rather than players with too little data -
was tested separately; see rank_trajectory_lag_test.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import TRAIN_FRACTION, log_loss  # noqa: E402
from elo_ratings import K_FACTOR, STARTING_ELO, SURFACE_BLEND_K, expected_score, load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

SOLID_MATCHES = 30       # >= this many career matches = "trustworthy" Elo, used to FIT the rank map
DEFAULT_THIN_THRESHOLD = 10  # matches the Cincinnati filter grid's original floor
BLEND_K = SURFACE_BLEND_K    # reuse the pipeline's own constant, not a freshly-tuned one


def build_dataset(df):
    """Same frozen-per-edition online-Elo construction as elite_opponent_residual_test.
    build_frozen_predictions, extended to also carry each side's own career match count so far
    (matches_before) and each side's own current_rank (not just the opponent's)."""
    df = df.copy()
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)
    edition_start = df.groupby("edition_id")["Date"].transform("min")
    editions = (
        df.assign(edition_start=edition_start)[["edition_id", "edition_start"]]
        .drop_duplicates().sort_values("edition_start").reset_index(drop=True)
    )

    overall_elo, current_rank, matches_played = {}, {}, {}
    rows = []
    for edition_id in editions["edition_id"]:
        edition_matches = df[df["edition_id"] == edition_id]
        snap_elo, snap_rank, snap_count = dict(overall_elo), dict(current_rank), dict(matches_played)

        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            elo_p1, elo_p2 = snap_elo.get(p1, STARTING_ELO), snap_elo.get(p2, STARTING_ELO)
            pred_p1 = expected_score(elo_p1, elo_p2)
            win1 = 1 if winner == p1 else 0
            n1, n2 = snap_count.get(p1, 0), snap_count.get(p2, 0)
            r1, r2 = snap_rank.get(p1), snap_rank.get(p2)
            rows.append((edition_id, row.Date, row.Round, p1, p2, elo_p1, elo_p2, pred_p1, win1, n1, n2, r1, r2))
            rows.append((edition_id, row.Date, row.Round, p2, p1, elo_p2, elo_p1, 1 - pred_p1, 1 - win1, n2, n1, r2, r1))

        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            overall_elo.setdefault(p1, STARTING_ELO)
            overall_elo.setdefault(p2, STARTING_ELO)
            score1 = 1.0 if winner == p1 else 0.0
            exp1 = expected_score(overall_elo[p1], overall_elo[p2])
            overall_elo[p1] += K_FACTOR * (score1 - exp1)
            overall_elo[p2] += K_FACTOR * ((1 - score1) - (1 - exp1))
            matches_played[p1] = matches_played.get(p1, 0) + 1
            matches_played[p2] = matches_played.get(p2, 0) + 1
            if pd.notna(row.Rank_1) and row.Rank_1 > 0:
                current_rank[p1] = row.Rank_1
            if pd.notna(row.Rank_2) and row.Rank_2 > 0:
                current_rank[p2] = row.Rank_2

    preds = pd.DataFrame(rows, columns=[
        "edition_id", "date", "round", "player", "opponent", "player_elo", "opponent_elo",
        "pred_win", "actual_win", "player_matches_before", "opponent_matches_before",
        "player_rank", "opponent_rank",
    ])
    return preds, editions


def fit_rank_to_elo(train_rows):
    """elo ~ a + b*log(rank), fit on train-era 'solid' players only (>=SOLID_MATCHES career
    matches, known rank) - deduped to one observation per (player, edition), since elo/rank are
    frozen constants within an edition and repeating every match-row would just over-weight
    players who played more rounds that edition, not add real information."""
    solid = train_rows[
        (train_rows["player_matches_before"] >= SOLID_MATCHES) & train_rows["player_rank"].notna()
    ].drop_duplicates(subset=["player", "edition_id"])
    log_rank = np.log(solid["player_rank"].astype(float))
    elo = solid["player_elo"].astype(float)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_rank, elo)
    print(f"Rank-to-Elo fit (train era, {len(solid)} solid-player-edition observations, "
          f">= {SOLID_MATCHES} career matches): elo = {intercept:.1f} + {slope:.1f} * ln(rank), "
          f"R^2 = {r_value**2:.3f}, p = {p_value:.2g}")

    def rank_elo(rank):
        return intercept + slope * np.log(rank)

    return rank_elo


def blend_elo(raw_elo, matches_before, rank, rank_elo_fn, threshold):
    if matches_before >= threshold or rank is None or pd.isna(rank):
        return raw_elo
    external = rank_elo_fn(rank)
    weight = matches_before / (matches_before + BLEND_K)
    return weight * raw_elo + (1 - weight) * external


def apply_blend(df, rank_elo_fn, threshold):
    df = df.copy()
    df["player_elo_blend"] = df.apply(
        lambda r: blend_elo(r["player_elo"], r["player_matches_before"], r["player_rank"], rank_elo_fn, threshold),
        axis=1,
    )
    df["opponent_elo_blend"] = df.apply(
        lambda r: blend_elo(r["opponent_elo"], r["opponent_matches_before"], r["opponent_rank"], rank_elo_fn, threshold),
        axis=1,
    )
    df["blended_pred"] = df.apply(
        lambda r: expected_score(r["player_elo_blend"], r["opponent_elo_blend"]), axis=1,
    )
    return df


def run(tour, thin_threshold):
    matches = load_matches_for_tour(tour)
    preds, editions = build_dataset(matches)

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    train = preds[preds["edition_id"].isin(train_editions)]
    test = preds[preds["edition_id"].isin(test_editions)]
    print(f"{tour}: {len(editions)} tournament editions, train = first {len(train_editions)} "
          f"(through {editions['edition_start'].iloc[split_idx - 1].date()}), "
          f"test = remaining {len(test_editions)} (from {editions['edition_start'].iloc[split_idx].date()})")
    print(f"{len(train)} train-era rows, {len(test)} test-era rows\n")

    rank_elo_fn = fit_rank_to_elo(train)

    # thin population this correction actually touches: at least one side under the threshold,
    # with a real rank known for that thin side (otherwise there's nothing to blend toward)
    def is_thin_row(r):
        player_thin = r["player_matches_before"] < thin_threshold and pd.notna(r["player_rank"])
        opp_thin = r["opponent_matches_before"] < thin_threshold and pd.notna(r["opponent_rank"])
        return player_thin or opp_thin

    test_blended = apply_blend(test, rank_elo_fn, thin_threshold)
    thin_mask = test_blended.apply(is_thin_row, axis=1)
    thin_test = test_blended[thin_mask].copy()

    print(f"Test-era rows where the blend actually changes something (>= 1 side < {thin_threshold} "
          f"career matches, rank known): {len(thin_test)} of {len(test_blended)} "
          f"({len(thin_test) / len(test_blended):.1%})\n")

    thin_test["raw_loss"] = log_loss(thin_test["actual_win"].values, thin_test["pred_win"].values)
    thin_test["blend_loss"] = log_loss(thin_test["actual_win"].values, thin_test["blended_pred"].values)
    thin_test["raw_brier"] = (thin_test["actual_win"] - thin_test["pred_win"]) ** 2
    thin_test["blend_brier"] = (thin_test["actual_win"] - thin_test["blended_pred"]) ** 2

    print(f"--- Held-out validation on the thin population only (n={len(thin_test)}) ---")
    print(f"  Raw Elo (current pipeline)  : log-loss = {thin_test['raw_loss'].mean():.4f}, "
          f"Brier = {thin_test['raw_brier'].mean():.4f}")
    print(f"  Rank-blended                : log-loss = {thin_test['blend_loss'].mean():.4f}, "
          f"Brier = {thin_test['blend_brier'].mean():.4f}")

    observed, lo, hi = cluster_bootstrap_ci(thin_test, "raw_loss", "blend_loss")
    print(f"  Mean per-row log-loss improvement (raw - blended, >0 = blend better), "
          f"player-clustered: {observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    verdict = "IMPROVES" if lo > 0 else ("HURTS" if hi < 0 else "NO SIGNIFICANT EFFECT (CI straddles zero)")
    print(f"  -> Rank-blending {verdict} calibration on the thin population, held out.")

    # does it actually tame the extreme, implausible predictions (the Fery-style cases), not just
    # move the average? bucket by how many career matches the THINNER side of each row had.
    print(f"\n--- Where along 0-{thin_threshold} the blend's effect actually lands ---")
    bucket_edges = [(f"0-2", lambda n: n <= 2), (f"3-5", lambda n: 3 <= n <= 5),
                    (f"6-{thin_threshold - 1}", lambda n: 6 <= n < thin_threshold)]
    thinner_side_matches = thin_test.apply(
        lambda r: min(
            r["player_matches_before"] if pd.notna(r["player_rank"]) else 10**9,
            r["opponent_matches_before"] if pd.notna(r["opponent_rank"]) else 10**9,
        ), axis=1,
    )
    for label, test_fn in bucket_edges:
        bucket = thin_test[thinner_side_matches.apply(test_fn)]
        if len(bucket) == 0:
            print(f"  {label:>6} career matches: n=0, skipped")
            continue
        extreme_raw = (bucket["pred_win"].sub(0.5).abs() > 0.3).mean()
        extreme_blend = (bucket["blended_pred"].sub(0.5).abs() > 0.3).mean()
        print(f"  {label:>6} career matches (n={len(bucket):>4}): raw log-loss={bucket['raw_loss'].mean():.4f}  "
              f"blended log-loss={bucket['blend_loss'].mean():.4f}  "
              f"share of |pred-50%|>30pp: raw={extreme_raw:.1%} -> blended={extreme_blend:.1%}")

    # the concrete claim from tonight's investigation: a thin player rated as a strong favorite/
    # live underdog against a much-better-ranked opponent purely because their Elo defaults near
    # STARTING_ELO. Show the raw-vs-blended prediction gap directly on the most extreme raw calls.
    print(f"\n--- Most extreme raw predictions in the thin test population (where blending should "
          f"matter most, if it works) ---")
    extreme_rows = thin_test.reindex(thin_test["pred_win"].sub(0.5).abs().sort_values(ascending=False).index).head(10)
    print(extreme_rows[["player", "opponent", "player_matches_before", "opponent_matches_before",
                         "player_rank", "opponent_rank", "pred_win", "blended_pred", "actual_win"]]
          .to_string(index=False, formatters={
              "pred_win": "{:.1%}".format, "blended_pred": "{:.1%}".format,
              "player_rank": lambda x: "" if pd.isna(x) else f"{x:.0f}",
              "opponent_rank": lambda x: "" if pd.isna(x) else f"{x:.0f}",
          }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
    parser.add_argument("--thin-threshold", type=int, default=DEFAULT_THIN_THRESHOLD)
    args = parser.parse_args()
    run(args.tour, args.thin_threshold)
