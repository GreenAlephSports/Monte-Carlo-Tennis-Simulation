"""Tests whether UPSET_BOOST_LOGIT_SHIFT's magnitude (win_probability.py) should scale with the
SIZE of the Elo gap overcome, rather than one flat shift for any gap > 100 - i.e. does beating a
250-Elo-point favorite carry more next-match carryover than beating someone barely over the 100
threshold?

Reuses survivorship_upset_test.py's exact machinery (which established that a >100 threshold is
real and beats both a no-adjustment baseline and a generic "won last round" framing): the same
frozen per-tournament-edition Elo predictions (elite_opponent_residual_test.build_frozen_predictions),
the same chronological tournament-level 80/20 split, the same player-clustered bootstrap. This
script only adds finer-grained bucketing and a continuous functional form ON TOP of that shared
machinery, restricted to the >100 population UPSET_BOOST_LOGIT_SHIFT already fires on - it does not
re-relitigate whether the 100-point threshold itself is real (survivorship_upset_test.py already
did, with train-era monotonicity + held-out validation).

Three candidates, all held-out validated against raw (unadjusted) Elo AND against each other:
  1. Flat (production): one logit shift for the whole gap>100 population - refit fresh on this
     script's own train split for an apples-to-apples comparison, not the hardcoded production
     constant (same precedent as survivorship_upset_test.py's own single-threshold candidate).
  2. Graduated buckets: 100-150 / 150-250 / 250-400 / 400+, one logit shift per bucket.
  3. Continuous: shift(gap) = BETA * ln(gap/100), a one-parameter log-linear form that is exactly
     0 at the gap=100 boundary by construction (not a fit accident) and grows with diminishing
     marginal effect for very large gaps - fit by 1D MLE (scipy) on train-era rows, offset =
     logit(pred_win), single covariate ln(gap/100).

Usage:
    python model/upset_boost_scaling_test.py [--tour ATP|WTA]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import EPS, TRAIN_FRACTION, build_frozen_predictions, log_loss, logit, sigmoid  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from survivorship_upset_test import ROUND_ORDER, cluster_bootstrap_ci  # noqa: E402
from win_probability import UPSET_BOOST_LOGIT_SHIFT  # noqa: E402

FINE_BUCKET_EDGES = [
    ("100_150", lambda gap: 100 < gap <= 150),
    ("150_250", lambda gap: 150 < gap <= 250),
    ("250_400", lambda gap: 250 < gap <= 400),
    ("400plus", lambda gap: gap > 400),
]
FINE_BUCKET_ORDER = [name for name, _ in FINE_BUCKET_EDGES]


def fine_bucket_for_gap(gap):
    for name, test in FINE_BUCKET_EDGES:
        if test(gap):
            return name
    raise ValueError(gap)  # unreachable for gap > 100, the only population this is called on


def build_gap_dataset(preds):
    """Same sequencing as survivorship_upset_test.build_upset_dataset (most recent WIN this
    tournament's Elo gap overcome, break on elimination, first_match/no_upset carried through
    unchanged) but keeps the raw numeric gap instead of collapsing straight to a bucket label, so
    this script can re-bucket it at a finer grain without re-deriving the sequencing logic."""
    df = preds[preds["round"].isin(ROUND_ORDER)].copy()
    df["round_order"] = df["round"].map(ROUND_ORDER)

    rows = []
    for (edition_id, player), g in df.sort_values("round_order").groupby(["edition_id", "player"], sort=False):
        prev_gap = None
        for row in g.itertuples(index=False):
            rows.append((edition_id, player, row.date, row.round, prev_gap,
                         row.pred_win, row.actual_win, row.player_elo, row.opponent_elo))
            if row.actual_win == 0:
                break
            prev_gap = row.opponent_elo - row.player_elo
    return pd.DataFrame(rows, columns=[
        "edition_id", "player", "date", "round", "prev_gap", "pred_win", "actual_win",
        "player_elo", "opponent_elo",
    ])


def summarize(name, g):
    n = len(g)
    actual_rate, pred_rate = g["actual_win"].mean(), g["pred_win"].mean()
    residual = actual_rate - pred_rate
    se = math.sqrt((g["pred_win"] * (1 - g["pred_win"])).sum()) / n if n else float("nan")
    z = residual / se if se > 0 else float("nan")
    return {"bucket": name, "n": n, "mean_gap": g["prev_gap"].mean(), "actual_rate": actual_rate,
            "pred_rate": pred_rate, "residual": residual, "z": z}


def fit_continuous_beta(train_upset):
    """MLE for BETA in P(win) = sigmoid(logit(pred_win) + BETA * ln(gap/100)) - a GLM with a fixed
    offset (logit(pred_win)) and one covariate, ln(prev_gap/100), which is exactly 0 at the gap=100
    boundary so BETA alone controls how fast the boost grows past that threshold."""
    x = np.log(train_upset["prev_gap"].values / 100.0)
    y = train_upset["actual_win"].values
    offset = np.array([logit(p) for p in train_upset["pred_win"].values])

    def neg_log_lik(beta):
        pred = 1 / (1 + np.exp(-(offset + beta * x)))
        pred = np.clip(pred, EPS, 1 - EPS)
        return -(y * np.log(pred) + (1 - y) * np.log(1 - pred)).sum()

    result = minimize_scalar(neg_log_lik, bounds=(-2.0, 2.0), method="bounded")
    return result.x


def apply_continuous(pred_win, gap, beta):
    return sigmoid(logit(pred_win) + beta * math.log(gap / 100.0))


def held_out_report(label, test_rows, adjusted_col_fn):
    """adjusted_col_fn(row) -> adjusted probability. Prints raw-vs-adjusted log-loss/Brier and a
    player-clustered bootstrap CI on the log-loss improvement, same statistic
    survivorship_upset_test.py itself reports."""
    t = test_rows.copy()
    t["adjusted_pred"] = t.apply(adjusted_col_fn, axis=1)
    t["raw_loss"] = log_loss(t["actual_win"].values, t["pred_win"].values)
    t["adj_loss"] = log_loss(t["actual_win"].values, t["adjusted_pred"].values)
    t["raw_brier"] = (t["actual_win"] - t["pred_win"]) ** 2
    t["adj_brier"] = (t["actual_win"] - t["adjusted_pred"]) ** 2
    observed, lo, hi = cluster_bootstrap_ci(t, "raw_loss", "adj_loss")
    print(f"\n{label}: {len(t)} test-era rows, {t['player'].nunique()} players")
    print(f"  Raw Elo        : log-loss = {t['raw_loss'].mean():.4f}, Brier = {t['raw_brier'].mean():.4f}")
    print(f"  Adjusted       : log-loss = {t['adj_loss'].mean():.4f}, Brier = {t['adj_brier'].mean():.4f}")
    print(f"  Mean per-match log-loss improvement (raw - adjusted, >0 = better), player-clustered: "
          f"{observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    return observed, lo, hi, t


def run(tour):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)
    gap_df = build_gap_dataset(preds)

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    train_all = gap_df[gap_df["edition_id"].isin(train_editions)]
    test_all = gap_df[gap_df["edition_id"].isin(test_editions)]

    train_upset = train_all[train_all["prev_gap"] > 100].copy()
    test_upset = test_all[test_all["prev_gap"] > 100].copy()
    print(f"{tour}: gap>100 population - {len(train_upset)} train-era rows "
          f"({train_upset['player'].nunique()} players), {len(test_upset)} test-era rows "
          f"({test_upset['player'].nunique()} players)")
    print(f"Current production UPSET_BOOST_LOGIT_SHIFT = {UPSET_BOOST_LOGIT_SHIFT:+.4f} (flat, ATP-fit, "
          f"reused for WTA) - for reference only; every candidate below is refit fresh on this "
          f"script's own train split for a fair comparison.")

    # --- train-era residual by fine bucket: does it actually get progressively bigger? ---
    train_upset["fine_bucket"] = train_upset["prev_gap"].map(fine_bucket_for_gap)
    fine_summary = pd.DataFrame(
        [summarize(b, g) for b, g in train_upset.groupby("fine_bucket") if len(g)]
    ).set_index("bucket").reindex(FINE_BUCKET_ORDER).reset_index()
    print("\n--- Train-era residual by gap-size sub-bucket (within the gap>100 population) ---")
    print(fine_summary.to_string(index=False, formatters={
        "mean_gap": "{:.0f}".format, "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format,
        "residual": "{:+.1%}".format, "z": "{:.2f}".format,
    }))
    present = fine_summary.dropna(subset=["residual"])
    is_monotonic = present["residual"].is_monotonic_increasing
    print(f"\nDoes train-era residual increase monotonically across sub-buckets, as the "
          f"'bigger gap overcome = more carryover' hypothesis predicts? {'YES' if is_monotonic else 'NO'} "
          f"({', '.join(f'{r:+.1%}' for r in present['residual'])})")

    # --- candidate 1: flat, refit on this train split ---
    flat_actual, flat_pred = train_upset["actual_win"].mean(), train_upset["pred_win"].mean()
    flat_shift = logit(flat_actual) - logit(flat_pred)
    print(f"\nCandidate 1 - flat shift refit on train (n={len(train_upset)}): {flat_shift:+.4f} logits "
          f"(actual {flat_actual:.1%} vs. Elo-predicted {flat_pred:.1%})")
    obs_flat, lo_flat, hi_flat, test_flat = held_out_report(
        "Candidate 1: flat shift (current production design)", test_upset,
        lambda r: sigmoid(logit(r["pred_win"]) + flat_shift),
    )

    # --- candidate 2: graduated buckets ---
    shift_by_fine_bucket = {}
    for name, g in train_upset.groupby("fine_bucket"):
        if len(g) == 0:
            continue
        shift_by_fine_bucket[name] = logit(g["actual_win"].mean()) - logit(g["pred_win"].mean())
    print("\nCandidate 2 - per-bucket shifts fit on train:")
    for name in FINE_BUCKET_ORDER:
        if name in shift_by_fine_bucket:
            print(f"  {name:<10} {shift_by_fine_bucket[name]:+.4f} logits "
                  f"(n={len(train_upset[train_upset['fine_bucket']==name])})")
    test_upset["fine_bucket"] = test_upset["prev_gap"].map(fine_bucket_for_gap)
    test_graduated = test_upset[test_upset["fine_bucket"].isin(shift_by_fine_bucket)].copy()
    obs_grad, lo_grad, hi_grad, test_grad = held_out_report(
        "Candidate 2: graduated buckets (100-150 / 150-250 / 250-400 / 400+)", test_graduated,
        lambda r: sigmoid(logit(r["pred_win"]) + shift_by_fine_bucket[r["fine_bucket"]]),
    )

    # --- candidate 3: continuous log-linear ---
    beta = fit_continuous_beta(train_upset)
    print(f"\nCandidate 3 - continuous fit: shift(gap) = {beta:+.4f} * ln(gap/100), fit by MLE on "
          f"train (n={len(train_upset)})")
    for gap_example in (100, 150, 250, 400, 600):
        print(f"    at gap={gap_example}: shift = {beta * math.log(gap_example/100):+.4f} logits")
    obs_cont, lo_cont, hi_cont, test_cont = held_out_report(
        "Candidate 3: continuous shift(gap) = BETA * ln(gap/100)", test_upset,
        lambda r: apply_continuous(r["pred_win"], r["prev_gap"], beta),
    )

    # --- head-to-head: does graduated/continuous actually beat flat on the SAME test rows? ---
    print("\n--- Head-to-head vs. flat, on the identical test-era row set (n={}) ---".format(len(test_upset)))
    common = test_upset.copy()
    common["flat_pred"] = common["pred_win"].apply(lambda p: sigmoid(logit(p) + flat_shift))
    common["fine_bucket"] = common["prev_gap"].map(fine_bucket_for_gap)
    common = common[common["fine_bucket"].isin(shift_by_fine_bucket)].copy()
    common["grad_pred"] = common.apply(lambda r: sigmoid(logit(r["pred_win"]) + shift_by_fine_bucket[r["fine_bucket"]]), axis=1)
    common["cont_pred"] = common.apply(lambda r: apply_continuous(r["pred_win"], r["prev_gap"], beta), axis=1)
    common["flat_loss"] = log_loss(common["actual_win"].values, common["flat_pred"].values)
    common["grad_loss"] = log_loss(common["actual_win"].values, common["grad_pred"].values)
    common["cont_loss"] = log_loss(common["actual_win"].values, common["cont_pred"].values)

    obs_gf, lo_gf, hi_gf = cluster_bootstrap_ci(common, "flat_loss", "grad_loss")
    obs_cf, lo_cf, hi_cf = cluster_bootstrap_ci(common, "flat_loss", "cont_loss")
    print(f"  Graduated vs. flat: mean log-loss improvement {obs_gf:+.4f}, 95% CI [{lo_gf:+.4f}, {hi_gf:+.4f}]")
    print(f"  Continuous vs. flat: mean log-loss improvement {obs_cf:+.4f}, 95% CI [{lo_cf:+.4f}, {hi_cf:+.4f}]")

    print("\n=== VERDICT ===")
    grad_clears = lo_grad > 0 and lo_gf > 0
    cont_clears = lo_cont > 0 and lo_cf > 0
    if not is_monotonic:
        print("Train-era residual does NOT increase monotonically with gap size - the core premise "
              "of a scaled boost doesn't even hold in-sample. Keep the flat design.")
    elif not (grad_clears or cont_clears):
        print("Train-era residual looked monotonic, but neither the graduated-bucket nor the "
              "continuous version clears held-out validation against BOTH raw Elo and the flat "
              "design (95% CI must exclude zero on both comparisons) - same pattern as the earlier "
              "2-bucket/3-bucket layoff collapse attempts that looked promising in-sample and failed "
              "out-of-sample. Keep the flat UPSET_BOOST_LOGIT_SHIFT design; do not ship a scaled "
              "version off this evidence.")
    else:
        winner = "graduated" if (grad_clears and obs_gf >= obs_cf) else "continuous"
        print(f"A scaled version clears held-out validation against both raw Elo and the flat design: "
              f"the {winner} form. See the fitted magnitudes/curve above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
    args = parser.parse_args()
    run(args.tour)
