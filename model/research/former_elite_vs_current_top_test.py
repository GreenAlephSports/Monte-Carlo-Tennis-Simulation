"""Tests a NARROW, specific scenario (Daron's framing), not the pooled peak-ceiling population
already tested and rejected in upset_boost_peak_ceiling_test.py: among matches where the WINNER is
a former-elite player currently well below their own career peak (gap_below_peak >=
FORMER_ELITE_GAP_THRESHOLD=150 Elo pts, career_peak_prior - player_elo), does that win carry MORE
next-match predictive weight specifically when the LOSER is a current top-ranked player (top-20 or
top-30 by real ATP/WTA rank at match time), compared to an otherwise-identical below-peak win over
a lower-ranked opponent?

Explicitly distinct from every prior test in this family, not a re-run under a new name:
  - upset_boost_peak_ceiling_test.py (REJECTED 2026-09-09) tested whether the SIZE of the
    beneficiary's own career-peak Elo scales the upset-boost shift, POOLED across the entire
    gap-overcome-by-Elo>100 population regardless of who the opponent specifically was (a fitted
    continuous PEAK_COEF over the whole population, wrong-signed and not held-out validated). That
    test's population is "any big Elo-gap upset by anyone", conditioned on the WINNER's peak.
  - THIS test's population is different and much narrower by construction: it is NOT gated on the
    Elo gap overcome at all (a below-peak former-elite player can easily be Elo-favored or
    Elo-underdog against a top-20 opponent depending on how far below peak they've fallen - gap is
    not the filter here). It is gated on the OPPONENT's real-world CURRENT RANK (top-20/30, the
    same no-lookahead Rank_1/Rank_2-derived opponent_rank column elite_opponent_residual_test.py's
    build_frozen_predictions and signature_win_boost_test.py already use for "who is a strong
    opponent" - not an Elo-based proxy for it), isolated as ITS OWN population and compared directly
    against a same-beneficiary-profile control group (below-peak wins over a NON-top opponent) -
    not pooled together and not diluted by wins over weak opponents the way a flat population-wide
    fit would be.
  - signature_win_boost_test.py (REJECTED 2026-08-26) tested a K-FACTOR multiplier on the Elo
    UPDATE itself, applied symmetrically to both players of a flagged match regardless of who the
    beneficiary was. This test never touches Elo updates - like upset-boost, it is a next-match
    win-probability question, evaluated only on rows that exist because the qualifying match was
    WON (see upset_boost_peak_ceiling_test.py's population-construction check: this family of tests
    cannot construct a symmetric loss-penalty by design).

Sample size discipline (the actual point of testing this scoped version): a "former-elite player
beats a current top-20/30 player" match is a real but RARE conjunction of two independently
uncommon conditions. This script reports the honest row/player count for this exact narrow
population BEFORE any modeling, and does not force a held-out train/test verdict if the resulting
test-era slice is too thin to support one (MIN_TEST_ROWS_FOR_HELD_OUT, checked explicitly) - in
that case only the full-sample descriptive comparison is reported, plainly labeled inconclusive
rather than dressed up as a validated result.

Population construction:
  1. Frozen per-tournament-edition Elo (elite_opponent_residual_test.build_frozen_predictions) -
     same convention as every other test in this project, opponent_rank already included (the
     opponent's real current_rank at match time, no lookahead).
  2. career_peak_prior / years_of_history via peak_elo_recovery_test.add_peak_features (unchanged,
     no-lookahead peak machinery already built and validated for coverage reporting).
  3. Within-tournament sequencing (same "most recent WIN this edition, broken on a loss" convention
     as survivorship_upset_test.build_upset_dataset / upset_boost_scaling_test.build_gap_dataset),
     extended to also carry prev_opponent_rank - the rank of the opponent BEATEN in that most
     recent win, not just the Elo gap overcome.
  4. Eligible row = a "next match" row (prev_opponent_rank is not null - there WAS a prior win this
     tournament) where the winner of that prior match had gap_below_peak >=
     FORMER_ELITE_GAP_THRESHOLD at the time (edition-level, frozen, so "at the time of that win" and
     "this edition" are the same value by this project's own frozen-per-edition convention) AND
     meets the same >= MIN_HISTORY_YEARS coverage bar peak_elo_recovery_test.py uses (a peak from a
     rookie's first two tournaments isn't a meaningful ceiling estimate).
  5. Group A (the actual scenario under test): prev_opponent_rank <= TOP_N.
     Group B (control, same below-peak profile): prev_opponent_rank > TOP_N (still real-ranked,
     just not top-tier) - NOT "no adjustment" or pooled-everyone, specifically the same
     below-peak-winner population with a weaker opponent, isolating the opponent-strength axis.
  Both TOP_N=20 and TOP_N=30 are reported (the user's own "top-20/30" framing), not just one.

Same rigor as the rest of this project where sample size allows: frozen-per-edition Elo, both
tours, chronological tournament-edition 80/20 split, player-clustered bootstrap CIs
(extended here to a two-group DIFFERENCE-in-residual statistic, not just raw-vs-adjusted, since the
actual claim is a comparison between Group A and Group B, not "does an adjustment beat raw Elo").

Usage:
    python model/research/former_elite_vs_current_top_test.py
    python model/research/former_elite_vs_current_top_test.py --gap-threshold 150 --min-history-years 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import (  # noqa: E402
    EPS, TRAIN_FRACTION, build_frozen_predictions, log_loss, logit, sigmoid,
)
from elo_ratings import load_matches_for_tour  # noqa: E402
from peak_elo_recovery_test import add_peak_features  # noqa: E402
from survivorship_upset_test import ROUND_ORDER  # noqa: E402

FORMER_ELITE_GAP_THRESHOLD_DEFAULT = 150.0
MIN_HISTORY_YEARS_DEFAULT = 2.0
TOP_N_CANDIDATES = [20, 30]
MIN_TEST_ROWS_FOR_HELD_OUT = 30  # per group - below this, held-out fit is not attempted, reported honestly


def build_narrow_dataset(preds):
    """Same within-tournament sequencing convention as survivorship_upset_test.build_upset_dataset
    / upset_boost_scaling_test.build_gap_dataset (most recent WIN this edition, loop breaks on a
    loss - so, as established in upset_boost_peak_ceiling_test.py's population-construction check,
    this can never construct a 'next match after a loss' row), extended to also carry
    prev_opponent_rank (the real current_rank of the opponent BEATEN in that most recent win) and
    the player's own career_peak_prior/years_of_history (edition-level, frozen, constant across the
    whole tournament run by this project's own frozen-per-edition convention - forwarded through
    unchanged rather than re-derived)."""
    df = preds[preds["round"].isin(ROUND_ORDER)].copy()
    df["round_order"] = df["round"].map(ROUND_ORDER)

    rows = []
    for (edition_id, player), g in df.sort_values("round_order").groupby(["edition_id", "player"], sort=False):
        prev_gap = None
        prev_opp_rank = None
        for row in g.itertuples(index=False):
            rows.append((edition_id, player, row.date, row.round, prev_gap, prev_opp_rank,
                         row.pred_win, row.actual_win, row.player_elo, row.opponent_elo,
                         row.career_peak_prior, row.years_of_history))
            if row.actual_win == 0:
                break
            prev_gap = row.opponent_elo - row.player_elo
            prev_opp_rank = row.opponent_rank
    return pd.DataFrame(rows, columns=[
        "edition_id", "player", "date", "round", "prev_gap", "prev_opponent_rank",
        "pred_win", "actual_win", "player_elo", "opponent_elo", "career_peak_prior", "years_of_history",
    ])


def cluster_bootstrap_group_diff(df, actual_col, pred_col, group_col, group_a, group_b,
                                  id_col="player", n_boot=5000, seed=42):
    """Player-clustered bootstrap CI for the DIFFERENCE in mean residual (actual - pred) between
    two groups, resampling PLAYERS jointly (not each group separately) since the same player can
    contribute rows to both groups - same row-weighted-sum convention as survivorship_upset_test.
    cluster_bootstrap_ci, extended from a raw-vs-adjusted diff to a group-vs-group diff. Bootstrap
    replicates where either group has zero resampled rows are dropped (possible with a genuinely
    small number of distinct players in one group) and the count of valid replicates is reported -
    an unstable/low-valid-replicate CI is itself part of the "sample too small" signal this test is
    required to report honestly."""
    codes, players = pd.factorize(df[id_col].values)
    n_players = len(players)
    is_a = (df[group_col] == group_a).values
    is_b = (df[group_col] == group_b).values
    actual, pred = df[actual_col].values, df[pred_col].values

    a_actual_sum = np.zeros(n_players); a_pred_sum = np.zeros(n_players); a_count = np.zeros(n_players)
    b_actual_sum = np.zeros(n_players); b_pred_sum = np.zeros(n_players); b_count = np.zeros(n_players)
    np.add.at(a_actual_sum, codes[is_a], actual[is_a]); np.add.at(a_pred_sum, codes[is_a], pred[is_a]); np.add.at(a_count, codes[is_a], 1)
    np.add.at(b_actual_sum, codes[is_b], actual[is_b]); np.add.at(b_pred_sum, codes[is_b], pred[is_b]); np.add.at(b_count, codes[is_b], 1)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_players, size=n_players)
        a_n, b_n = a_count[idx].sum(), b_count[idx].sum()
        if a_n == 0 or b_n == 0:
            continue
        a_res = (a_actual_sum[idx].sum() - a_pred_sum[idx].sum()) / a_n
        b_res = (b_actual_sum[idx].sum() - b_pred_sum[idx].sum()) / b_n
        boot.append(a_res - b_res)
    n_valid = len(boot)
    if n_valid < n_boot * 0.5:
        print(f"    WARNING: only {n_valid}/{n_boot} bootstrap replicates had both groups represented "
              f"- CI below is unstable, itself evidence this population is too thin for a firm verdict.")
    if n_valid < 20:
        return float("nan"), float("nan"), float("nan"), n_valid
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    observed = (a_actual_sum.sum() - a_pred_sum.sum()) / a_count.sum() - (b_actual_sum.sum() - b_pred_sum.sum()) / b_count.sum()
    return observed, lo, hi, n_valid


def describe_group(name, g):
    n = len(g)
    if n == 0:
        return {"group": name, "n": n, "n_players": 0, "pred_rate": float("nan"),
                "actual_rate": float("nan"), "residual": float("nan"), "z": float("nan")}
    pred_rate, actual_rate = g["pred_win"].mean(), g["actual_win"].mean()
    residual = actual_rate - pred_rate
    se = np.sqrt((g["pred_win"] * (1 - g["pred_win"])).sum()) / n
    z = residual / se if se > 0 else float("nan")
    return {"group": name, "n": n, "n_players": g["player"].nunique(), "pred_rate": pred_rate,
            "actual_rate": actual_rate, "residual": residual, "z": z}


def run_for_top_n(top_n, narrow_all, gap_threshold):
    print(f"\n{'=' * 90}\nTOP_N = {top_n} (loser must be ranked <= {top_n} at match time to count as "
          f"'current top player')\n{'=' * 90}")

    eligible = narrow_all[
        narrow_all["prev_opponent_rank"].notna()
        & (narrow_all["gap_below_peak"] >= gap_threshold)
    ].copy()
    eligible["group"] = np.where(eligible["prev_opponent_rank"] <= top_n, "A_beat_top", "B_beat_other")

    print(f"Full narrow population (former-elite winner, gap_below_peak >= {gap_threshold:.0f}, "
          f"prior win this edition exists): {len(eligible)} rows, {eligible['player'].nunique()} distinct players")
    counts = eligible.groupby("group").agg(n=("player", "size"), n_players=("player", "nunique"))
    print(counts.to_string())

    n_group_a = counts.loc["A_beat_top", "n"] if "A_beat_top" in counts.index else 0
    if n_group_a == 0:
        print("  Group A (beat a current top player) is EMPTY at this TOP_N - cannot test this scenario.")
        return None

    # ---- full-sample descriptive comparison (the honest headline, regardless of held-out power) ----
    desc = pd.DataFrame([
        describe_group(f"A: beat top-{top_n} opponent", eligible[eligible["group"] == "A_beat_top"]),
        describe_group(f"B: beat non-top opponent (control)", eligible[eligible["group"] == "B_beat_other"]),
    ])
    print("\nFull-sample descriptive (all eligible rows, not train/test-split):")
    print(desc.to_string(index=False, formatters={
        "pred_rate": "{:.1%}".format, "actual_rate": "{:.1%}".format,
        "residual": "{:+.1%}".format, "z": "{:.2f}".format,
    }))

    obs, lo, hi, n_valid = cluster_bootstrap_group_diff(
        eligible, "actual_win", "pred_win", "group", "A_beat_top", "B_beat_other")
    print(f"\nDifference in residual (Group A - Group B), player-clustered bootstrap "
          f"({n_valid}/5000 valid replicates):")
    if n_valid < 20:
        print("  Too few valid bootstrap replicates to report a CI - population has too few distinct "
              "players in one or both groups. INCONCLUSIVE at this TOP_N.")
    else:
        verdict = ("A carries MORE predictive weight (CI excludes zero, >0)" if lo > 0 else
                   ("A carries LESS predictive weight (CI excludes zero, <0)" if hi < 0 else
                    "NOT distinguishable (CI straddles zero)"))
        print(f"  observed {obs:+.1%}, 95% CI [{lo:+.1%}, {hi:+.1%}] -> {verdict}")

    # ---- held-out attempt, only if the test-era slice is large enough to be worth reporting ----
    # tour-specific edition split was already applied upstream when narrow_all was assembled
    # (train/test membership carried as a column) - just report counts and, if adequate, fit/validate.
    for split_label, split_col_val in [("train-era", "train"), ("test-era", "test")]:
        n_a = len(eligible[(eligible["group"] == "A_beat_top") & (eligible["split"] == split_col_val)])
        n_b = len(eligible[(eligible["group"] == "B_beat_other") & (eligible["split"] == split_col_val)])
        print(f"  {split_label}: Group A n={n_a}, Group B n={n_b}")

    test_a = eligible[(eligible["group"] == "A_beat_top") & (eligible["split"] == "test")]
    test_b = eligible[(eligible["group"] == "B_beat_other") & (eligible["split"] == "test")]
    if len(test_a) < MIN_TEST_ROWS_FOR_HELD_OUT or len(test_b) < MIN_TEST_ROWS_FOR_HELD_OUT:
        print(f"\nHeld-out train/test verdict: NOT ATTEMPTED - test-era Group A has {len(test_a)} rows, "
              f"Group B has {len(test_b)} rows; both need >= {MIN_TEST_ROWS_FOR_HELD_OUT} for a held-out "
              f"fit to mean anything. This population is genuinely too thin for that at TOP_N={top_n}. "
              f"The full-sample descriptive comparison above is the honest result at this scope.")
        return {"top_n": top_n, "n_a": len(eligible[eligible['group']=='A_beat_top']),
                "n_b": len(eligible[eligible['group']=='B_beat_other']),
                "held_out_attempted": False, "obs": obs, "lo": lo, "hi": hi, "n_valid": n_valid}

    train_a = eligible[(eligible["group"] == "A_beat_top") & (eligible["split"] == "train")]
    train_b = eligible[(eligible["group"] == "B_beat_other") & (eligible["split"] == "train")]
    shift_a = logit(train_a["actual_win"].mean()) - logit(train_a["pred_win"].mean())
    shift_b = logit(train_b["actual_win"].mean()) - logit(train_b["pred_win"].mean())
    print(f"\nHeld-out fit: train-era flat logit shift, Group A = {shift_a:+.4f}, Group B = {shift_b:+.4f}")

    def held_out_loss(test_df, shift):
        adj = test_df["pred_win"].apply(lambda p: sigmoid(logit(p) + shift))
        raw_loss = log_loss(test_df["actual_win"].values, test_df["pred_win"].values)
        adj_loss = log_loss(test_df["actual_win"].values, adj.values)
        return raw_loss.mean(), adj_loss.mean()

    raw_a, adj_a = held_out_loss(test_a, shift_a)
    raw_b, adj_b = held_out_loss(test_b, shift_b)
    print(f"  Group A test-era: raw log-loss={raw_a:.4f}, adjusted={adj_a:.4f}")
    print(f"  Group B test-era: raw log-loss={raw_b:.4f}, adjusted={adj_b:.4f}")

    return {"top_n": top_n, "n_a": len(eligible[eligible['group']=='A_beat_top']),
            "n_b": len(eligible[eligible['group']=='B_beat_other']),
            "held_out_attempted": True, "obs": obs, "lo": lo, "hi": hi, "n_valid": n_valid}


def run(gap_threshold, min_history_years):
    per_tour_narrow = []

    for tour in ["ATP", "WTA"]:
        matches = load_matches_for_tour(tour)
        preds, editions = build_frozen_predictions(matches)
        preds = add_peak_features(preds)

        narrow = build_narrow_dataset(preds)
        narrow["gap_below_peak"] = (narrow["career_peak_prior"] - narrow["player_elo"]).clip(lower=0)
        narrow = narrow[narrow["years_of_history"] >= min_history_years].copy()
        narrow["tour"] = tour

        split_idx = int(len(editions) * TRAIN_FRACTION)
        train_editions = set(editions["edition_id"].iloc[:split_idx])
        narrow["split"] = np.where(narrow["edition_id"].isin(train_editions), "train", "test")
        per_tour_narrow.append(narrow)

        n_any_prior_win = narrow["prev_opponent_rank"].notna().sum()
        print(f"{tour}: {len(narrow)} coverage-eligible sequenced rows, {n_any_prior_win} with a prior "
              f"win this edition (opponent_rank known)")

    narrow_all = pd.concat(per_tour_narrow, ignore_index=True)

    print(f"\n{'=' * 90}\nSCOPE CHECK: this is a NARROW, specific population by design - counts above "
          f"are BEFORE the gap_below_peak>={gap_threshold:.0f} and prior-win filters are applied. "
          f"Real eligible counts per TOP_N are reported below.\n{'=' * 90}")

    results = []
    for top_n in TOP_N_CANDIDATES:
        r = run_for_top_n(top_n, narrow_all, gap_threshold)
        if r is not None:
            results.append(r)

    print(f"\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
    for r in results:
        status = "held-out attempted" if r["held_out_attempted"] else "DESCRIPTIVE ONLY (too thin for held-out)"
        ci = f"[{r['lo']:+.1%}, {r['hi']:+.1%}]" if r["n_valid"] >= 20 else "N/A (too few valid bootstrap replicates)"
        print(f"  TOP_N={r['top_n']:<3} n_A={r['n_a']:<5} n_B={r['n_b']:<5} {status:<38} "
              f"diff-in-residual={r['obs']:+.1%} 95% CI {ci}")

    print(f"\n{'=' * 90}\nOVERALL VERDICT\n{'=' * 90}")
    any_real_signal = any(r["n_valid"] >= 20 and (r["lo"] > 0) for r in results)
    any_thin = any(not r["held_out_attempted"] for r in results)
    if any_real_signal:
        print("At least one TOP_N definition shows Group A (former-elite winner beating a current top "
              "player) with a significantly larger next-match residual than Group B (same profile, "
              "weaker opponent beaten) - see per-TOP_N breakdown above for magnitude and whether it "
              "survived a held-out split.")
    else:
        print("No TOP_N definition shows a statistically real difference between beating a current top "
              "player and beating a weaker one, among former-elite below-peak winners.")
    if any_thin:
        print("IMPORTANT: at least one TOP_N definition's population was too thin to support a held-out "
              "train/test verdict at all (see MIN_TEST_ROWS_FOR_HELD_OUT gate above) - this reflects "
              "genuine rarity of the exact scenario (former-elite player currently well below peak AND "
              "beating a current top-ranked opponent), not a bug. Any full-sample descriptive number "
              "reported for that TOP_N should be read as suggestive at most, not a validated correction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gap-threshold", type=float, default=FORMER_ELITE_GAP_THRESHOLD_DEFAULT)
    parser.add_argument("--min-history-years", type=float, default=MIN_HISTORY_YEARS_DEFAULT)
    args = parser.parse_args()
    run(args.gap_threshold, args.min_history_years)
