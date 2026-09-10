"""Two genuinely distinct hypotheses, neither a re-run of prior tests in this family under a new
name:

PART 1 - STREAK: does a former-elite player currently well below their own career peak
(gap_below_peak >= FORMER_ELITE_GAP_THRESHOLD, same threshold as former_elite_vs_current_top_test.py)
show a real "finding form" signal specifically over a SEQUENCE of consecutive real wins THIS
tournament (2, or 3+, in a row), beyond what a single win's own next-match residual already
captures? Every prior test in this family (upset_boost_scaling_test.py, upset_boost_peak_ceiling_
test.py, former_elite_vs_current_top_test.py) conditions on the MOST RECENT win only - none has ever
asked whether the LENGTH of an active win streak itself carries additional information. This is
that missing axis.

PART 2 - OPPONENT TRAJECTORY: does a former-elite below-peak player's win carry more next-match
predictive weight when the opponent BEATEN was themselves in current good form (their own
recent_form_residual > 0 - actually outperforming what Elo predicts for them lately, the same
rolling actual-vs-predicted metric already built and held-out validated in recent_form_test.py),
rather than a static rank threshold (already tested and rejected in former_elite_vs_current_top_
test.py - beating a top-20/30 RANKED opponent showed no signal)? Rank can lag real current form (a
top-20 player on a long losing streak, or an unranked/lower-ranked player on a hot streak) - this
tests the dynamic-form version of "quality opponent" instead of the static one already rejected.

Both parts share the same former-elite-below-peak population definition and machinery as
former_elite_vs_current_top_test.py (imported directly, not reimplemented): frozen per-tournament-
edition Elo (elite_opponent_residual_test.build_frozen_predictions), career_peak_prior/years_of_
history via peak_elo_recovery_test.add_peak_features (same >= MIN_HISTORY_YEARS coverage bar, same
gap_below_peak = clip(career_peak_prior - player_elo, 0) definition), chronological tournament-
edition 80/20 train/test split, player-clustered bootstrap CIs (cluster_bootstrap_group_diff,
reused directly from former_elite_vs_current_top_test.py rather than reimplemented), and the same
honesty discipline: MIN_TEST_ROWS_FOR_HELD_OUT gates whether a held-out verdict is even attempted,
and real sample sizes are reported before any modeling, both tours.

PART 1 sequencing detail: within a qualifying edition (gap_below_peak already edition-level frozen,
so the whole tournament run is either eligible or not), a "real competition" win is one where
opponent_rank is known (a genuine ranked tour opponent, not an unranked wildcard/qualifier with no
rank on record) - only THOSE wins increment the active streak counter. A LOSS to ANY opponent
(ranked or not) resets the streak to 0 - a loss is a loss regardless of who it was against. A win
against an UNRANKED opponent leaves the streak unchanged (neither counted as progress nor treated
as a break) - deliberately NOT filtering such rows out of the sequence entirely, which would let an
intervening real loss silently vanish and inflate the apparent streak length; this is a genuine data-
integrity choice, disclosed here rather than silently assumed. prev_streak_len is bucketed 0 (no_streak:
first match or immediately after a loss), 1, 2, 3+ - the row itself (the match being predicted) is
NOT required to be against a ranked opponent, only the wins that built the streak are.

PART 2 detail: opponent_recent_form_residual is the SAME player-level rolling metric recent_form_
test.add_recent_form computes for every row (that player's own actual-vs-Elo-predicted win rate
over their last PRIMARY_WINDOW=10 real matches, strictly prior, shift-protected against lookahead) -
just read off the OPPONENT's own row for the same real match via a self-join on
(edition_id, date, player, opponent) with player/opponent swapped, rather than recomputed. Rows
where the opponent didn't yet have 10 prior real matches (recent_form_residual undefined) are
excluded and the resulting coverage loss is reported honestly, same as every other coverage filter
in this project.

Usage:
    python model/research/former_elite_form_signal_test.py
    python model/research/former_elite_form_signal_test.py --gap-threshold 150 --min-history-years 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import (  # noqa: E402
    TRAIN_FRACTION, build_frozen_predictions, log_loss, logit, sigmoid,
)
from elo_ratings import load_matches_for_tour  # noqa: E402
from former_elite_vs_current_top_test import (  # noqa: E402
    FORMER_ELITE_GAP_THRESHOLD_DEFAULT, MIN_HISTORY_YEARS_DEFAULT, MIN_TEST_ROWS_FOR_HELD_OUT,
    cluster_bootstrap_group_diff, describe_group,
)
from peak_elo_recovery_test import add_peak_features  # noqa: E402
from recent_form_test import PRIMARY_WINDOW, add_recent_form  # noqa: E402
from survivorship_upset_test import ROUND_ORDER, cluster_bootstrap_ci  # noqa: E402

STREAK_BUCKET_ORDER = ["no_streak", "streak_1", "streak_2", "streak_3plus"]


# ============================================================================================
# PART 1 - STREAK
# ============================================================================================

def bucket_for_streak(n):
    if n <= 0:
        return "no_streak"
    if n == 1:
        return "streak_1"
    if n == 2:
        return "streak_2"
    return "streak_3plus"


def build_streak_dataset(preds):
    """See module docstring PART 1 sequencing detail for the streak-counting rules (ranked-opponent
    wins increment, any loss resets, unranked-opponent wins leave the streak unchanged)."""
    df = preds[preds["round"].isin(ROUND_ORDER)].copy()
    df["round_order"] = df["round"].map(ROUND_ORDER)

    rows = []
    for (edition_id, player), g in df.sort_values("round_order").groupby(["edition_id", "player"], sort=False):
        streak = 0
        for row in g.itertuples(index=False):
            rows.append((edition_id, player, row.date, row.round, streak, row.pred_win, row.actual_win,
                         row.player_elo, row.career_peak_prior, row.years_of_history))
            if row.actual_win == 0:
                streak = 0
            elif pd.notna(row.opponent_rank):
                streak += 1
            # else: won, but opponent unranked - streak carries forward unchanged
    return pd.DataFrame(rows, columns=[
        "edition_id", "player", "date", "round", "prev_streak_len", "pred_win", "actual_win",
        "player_elo", "career_peak_prior", "years_of_history",
    ])


def run_streak_part(streak_all, gap_threshold):
    print(f"\n{'#' * 90}\nPART 1 - STREAK: does a sequence of consecutive real-competition wins carry "
          f"more next-match predictive weight than a single win, among former-elite below-peak "
          f"players (gap_below_peak >= {gap_threshold:.0f})?\n{'#' * 90}")

    eligible = streak_all.copy()
    eligible["bucket"] = eligible["prev_streak_len"].apply(bucket_for_streak)
    print(f"\nFull eligible population: {len(eligible)} rows, {eligible['player'].nunique()} distinct players")
    counts = eligible.groupby("bucket").agg(n=("player", "size"), n_players=("player", "nunique")).reindex(STREAK_BUCKET_ORDER)
    print(counts.to_string())

    desc = pd.DataFrame([describe_group(b, eligible[eligible["bucket"] == b]) for b in STREAK_BUCKET_ORDER])
    print("\nFull-sample descriptive (all eligible rows, not train/test-split):")
    print(desc.to_string(index=False, formatters={
        "pred_rate": "{:.1%}".format, "actual_rate": "{:.1%}".format,
        "residual": "{:+.1%}".format, "z": "{:.2f}".format}))

    gradient = desc[desc["group"] != "no_streak"].reset_index(drop=True)
    is_monotonic = gradient["residual"].is_monotonic_increasing
    print(f"\nDoes residual increase monotonically streak_1 -> streak_2 -> streak_3plus, as the "
          f"'longer streak = more carryover' hypothesis specifically predicts? "
          f"{'YES' if is_monotonic else 'NO'} ({', '.join(f'{r:+.1%}' for r in gradient['residual'])})")

    # ---- the actual novel claim: streak>=2 vs streak==1, holding "just won" fixed ----
    core = eligible[eligible["bucket"].isin(["streak_1", "streak_2", "streak_3plus"])].copy()
    core["multi_group"] = np.where(core["bucket"] == "streak_1", "single_win", "multi_win_2plus")
    obs, lo, hi, n_valid = cluster_bootstrap_group_diff(
        core, "actual_win", "pred_win", "multi_group", "multi_win_2plus", "single_win")
    print(f"\nCore test: streak>=2 (n={len(core[core['multi_group']=='multi_win_2plus'])}) vs. "
          f"streak==1 (n={len(core[core['multi_group']=='single_win'])}) residual difference, "
          f"player-clustered bootstrap ({n_valid}/5000 valid replicates):")
    if n_valid < 20:
        print("  Too few valid bootstrap replicates - INCONCLUSIVE, population too thin at this split.")
    else:
        verdict = ("streak>=2 carries MORE predictive weight (CI excludes zero, >0)" if lo > 0 else
                   ("streak>=2 carries LESS predictive weight (CI excludes zero, <0)" if hi < 0 else
                    "NOT distinguishable (CI straddles zero)"))
        print(f"  observed {obs:+.1%}, 95% CI [{lo:+.1%}, {hi:+.1%}] -> {verdict}")

    # ---- held-out validation per bucket (skips a bucket entirely if too thin, reported honestly) ----
    print(f"\nHeld-out validation per bucket (flat logit shift fit on train-era rows of that bucket, "
          f"MIN_TEST_ROWS_FOR_HELD_OUT={MIN_TEST_ROWS_FOR_HELD_OUT}):")
    for b in STREAK_BUCKET_ORDER:
        if b == "no_streak":
            continue
        train_b = eligible[(eligible["bucket"] == b) & (eligible["split"] == "train")]
        test_b = eligible[(eligible["bucket"] == b) & (eligible["split"] == "test")]
        print(f"  {b}: train n={len(train_b)}, test n={len(test_b)}")
        if len(test_b) < MIN_TEST_ROWS_FOR_HELD_OUT or len(train_b) < MIN_TEST_ROWS_FOR_HELD_OUT:
            print(f"    NOT ATTEMPTED - too thin for a held-out fit at this bucket.")
            continue
        shift = logit(train_b["actual_win"].mean()) - logit(train_b["pred_win"].mean())
        t = test_b.copy()
        t["adjusted_pred"] = t["pred_win"].apply(lambda p: sigmoid(logit(p) + shift))
        t["raw_loss"] = log_loss(t["actual_win"].values, t["pred_win"].values)
        t["adj_loss"] = log_loss(t["actual_win"].values, t["adjusted_pred"].values)
        observed, lo_h, hi_h = cluster_bootstrap_ci(t, "raw_loss", "adj_loss")
        verdict = ("VALIDATED" if lo_h > 0 else ("WORSE" if hi_h < 0 else "not validated"))
        print(f"    train-fit shift={shift:+.4f}; held-out log-loss improvement {observed:+.4f}, "
              f"95% CI [{lo_h:+.4f}, {hi_h:+.4f}] -> {verdict}")

    return {"is_monotonic": is_monotonic, "obs": obs, "lo": lo, "hi": hi, "n_valid": n_valid}


# ============================================================================================
# PART 2 - OPPONENT TRAJECTORY (recent form, not static rank)
# ============================================================================================

def add_opponent_recent_form(preds):
    """Self-join preds onto itself, player/opponent swapped, to read the OPPONENT's own
    recent_form_residual (already computed per-row by recent_form_test.add_recent_form) as of the
    SAME real match - not recomputed, just read off the other perspective row of that match.
    Join key includes 'round' (not just edition_id/date/player/opponent) since the same two
    players can rarely meet twice at the same edition_id under the edition_id-collision artifact
    already documented in upset_boost_peak_ceiling_test.py's population-construction check (two
    distinct real tournaments sharing one Tournament+year label) - round narrows that down further.
    Any remaining duplicate join keys are deduplicated (keep first) rather than allowed to blow up
    the row count via a many-to-one match - this is the same negligible-frequency data artifact
    already disclosed there, not a new issue introduced here."""
    swapped = preds[["edition_id", "date", "round", "player", "opponent", "recent_form_residual"]].copy()
    swapped = swapped.rename(columns={
        "player": "opponent", "opponent": "player", "recent_form_residual": "opponent_recent_form_residual"})
    swapped = swapped.drop_duplicates(subset=["edition_id", "date", "round", "player", "opponent"])
    merged = preds.merge(swapped, on=["edition_id", "date", "round", "player", "opponent"], how="left")
    assert len(merged) == len(preds), "opponent-recent-form self-join changed row count"
    return merged


def build_opponent_form_dataset(preds):
    """Same 'most recent WIN this edition, break on loss' sequencing as former_elite_vs_current_
    top_test.build_narrow_dataset, carrying prev_opponent_recent_form_residual (the FORM, not rank,
    of the opponent beaten in that win) forward instead of prev_opponent_rank."""
    df = preds[preds["round"].isin(ROUND_ORDER)].copy()
    df["round_order"] = df["round"].map(ROUND_ORDER)

    rows = []
    for (edition_id, player), g in df.sort_values("round_order").groupby(["edition_id", "player"], sort=False):
        prev_opp_form = None
        for row in g.itertuples(index=False):
            rows.append((edition_id, player, row.date, row.round, prev_opp_form, row.pred_win,
                         row.actual_win, row.player_elo, row.career_peak_prior, row.years_of_history))
            if row.actual_win == 0:
                break
            prev_opp_form = row.opponent_recent_form_residual
    return pd.DataFrame(rows, columns=[
        "edition_id", "player", "date", "round", "prev_opponent_recent_form", "pred_win", "actual_win",
        "player_elo", "career_peak_prior", "years_of_history",
    ])


def run_opponent_form_part(form_all, gap_threshold):
    print(f"\n{'#' * 90}\nPART 2 - OPPONENT TRAJECTORY: does beating a CURRENTLY-hot opponent (their "
          f"own rolling recent_form_residual > 0) carry more next-match predictive weight than "
          f"beating a currently-cold one, among former-elite below-peak players "
          f"(gap_below_peak >= {gap_threshold:.0f})? - rank-based version already REJECTED in "
          f"former_elite_vs_current_top_test.py\n{'#' * 90}")

    n_no_prior_win = form_all["prev_opponent_recent_form"].isna().sum()
    eligible_any = form_all[form_all["prev_opponent_recent_form"].notna()].copy()
    n_no_form_coverage = len(eligible_any)
    print(f"\nRows with a prior win this edition: {n_no_form_coverage} (of {len(form_all)} total; "
          f"{n_no_prior_win} have no prior win and are excluded)")

    eligible = eligible_any.copy()
    eligible["group"] = np.where(eligible["prev_opponent_recent_form"] > 0, "A_beat_hot", "B_beat_cold")
    print(f"Eligible (opponent had a defined recent_form_residual - >= {PRIMARY_WINDOW} of their own "
          f"prior real matches): {len(eligible)} rows, {eligible['player'].nunique()} distinct players")
    counts = eligible.groupby("group").agg(n=("player", "size"), n_players=("player", "nunique"))
    print(counts.to_string())

    if "A_beat_hot" not in counts.index or counts.loc["A_beat_hot", "n"] == 0:
        print("  Group A (beat a currently-hot opponent) is EMPTY - cannot test this scenario.")
        return None

    desc = pd.DataFrame([
        describe_group("A: beat hot opponent (recent_form_residual > 0)", eligible[eligible["group"] == "A_beat_hot"]),
        describe_group("B: beat cold/neutral opponent (control)", eligible[eligible["group"] == "B_beat_cold"]),
    ])
    print("\nFull-sample descriptive (all eligible rows, not train/test-split):")
    print(desc.to_string(index=False, formatters={
        "pred_rate": "{:.1%}".format, "actual_rate": "{:.1%}".format,
        "residual": "{:+.1%}".format, "z": "{:.2f}".format}))

    obs, lo, hi, n_valid = cluster_bootstrap_group_diff(
        eligible, "actual_win", "pred_win", "group", "A_beat_hot", "B_beat_cold")
    print(f"\nDifference in residual (Group A - Group B), player-clustered bootstrap "
          f"({n_valid}/5000 valid replicates):")
    if n_valid < 20:
        print("  Too few valid bootstrap replicates - INCONCLUSIVE, population too thin.")
    else:
        verdict = ("A carries MORE predictive weight (CI excludes zero, >0)" if lo > 0 else
                   ("A carries LESS predictive weight (CI excludes zero, <0)" if hi < 0 else
                    "NOT distinguishable (CI straddles zero)"))
        print(f"  observed {obs:+.1%}, 95% CI [{lo:+.1%}, {hi:+.1%}] -> {verdict}")

    for split_label, split_val in [("train-era", "train"), ("test-era", "test")]:
        n_a = len(eligible[(eligible["group"] == "A_beat_hot") & (eligible["split"] == split_val)])
        n_b = len(eligible[(eligible["group"] == "B_beat_cold") & (eligible["split"] == split_val)])
        print(f"  {split_label}: Group A n={n_a}, Group B n={n_b}")

    test_a = eligible[(eligible["group"] == "A_beat_hot") & (eligible["split"] == "test")]
    test_b = eligible[(eligible["group"] == "B_beat_cold") & (eligible["split"] == "test")]
    held_out_attempted = len(test_a) >= MIN_TEST_ROWS_FOR_HELD_OUT and len(test_b) >= MIN_TEST_ROWS_FOR_HELD_OUT
    if not held_out_attempted:
        print(f"\nHeld-out train/test verdict: NOT ATTEMPTED - test-era Group A has {len(test_a)} rows, "
              f"Group B has {len(test_b)} rows; both need >= {MIN_TEST_ROWS_FOR_HELD_OUT}. This "
              f"population is too thin for that. The full-sample descriptive comparison above is the "
              f"honest result at this scope.")
    else:
        train_a = eligible[(eligible["group"] == "A_beat_hot") & (eligible["split"] == "train")]
        train_b = eligible[(eligible["group"] == "B_beat_cold") & (eligible["split"] == "train")]
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

    return {"n_a": len(eligible[eligible["group"] == "A_beat_hot"]),
            "n_b": len(eligible[eligible["group"] == "B_beat_cold"]),
            "held_out_attempted": held_out_attempted, "obs": obs, "lo": lo, "hi": hi, "n_valid": n_valid}


# ============================================================================================
def run(gap_threshold, min_history_years):
    streak_frames, form_frames = [], []

    for tour in ["ATP", "WTA"]:
        matches = load_matches_for_tour(tour)
        preds, editions = build_frozen_predictions(matches)
        preds = add_peak_features(preds)
        preds = add_recent_form(preds, PRIMARY_WINDOW)
        preds = add_opponent_recent_form(preds)
        preds["gap_below_peak"] = (preds["career_peak_prior"] - preds["player_elo"]).clip(lower=0)

        split_idx = int(len(editions) * TRAIN_FRACTION)
        train_editions = set(editions["edition_id"].iloc[:split_idx])

        former_elite = preds[
            (preds["years_of_history"] >= min_history_years) & (preds["gap_below_peak"] >= gap_threshold)
        ].copy()
        n_qualifying_rows = len(former_elite)
        n_qualifying_players = former_elite["player"].nunique()
        print(f"{tour}: {n_qualifying_rows} former-elite-below-peak rows "
              f"(gap_below_peak >= {gap_threshold:.0f}, years_of_history >= {min_history_years:.1f}), "
              f"{n_qualifying_players} distinct players")

        streak_df = build_streak_dataset(former_elite)
        streak_df["split"] = np.where(streak_df["edition_id"].isin(train_editions), "train", "test")
        streak_df["tour"] = tour
        streak_frames.append(streak_df)

        form_df = build_opponent_form_dataset(former_elite)
        form_df["split"] = np.where(form_df["edition_id"].isin(train_editions), "train", "test")
        form_df["tour"] = tour
        form_frames.append(form_df)

    streak_all = pd.concat(streak_frames, ignore_index=True)
    form_all = pd.concat(form_frames, ignore_index=True)

    streak_result = run_streak_part(streak_all, gap_threshold)
    form_result = run_opponent_form_part(form_all, gap_threshold)

    print(f"\n{'=' * 90}\nOVERALL VERDICT\n{'=' * 90}")
    if streak_result["n_valid"] >= 20 and streak_result["lo"] > 0:
        print("PART 1 (STREAK): real signal - streak>=2 shows a significantly larger next-match "
              "residual than a single win, among former-elite below-peak players.")
    else:
        print("PART 1 (STREAK): no statistically real difference between a single win and a "
              "2+-win streak, among former-elite below-peak players.")

    if form_result is None:
        print("PART 2 (OPPONENT TRAJECTORY): could not be tested - no rows in the 'beat a hot "
              "opponent' group.")
    elif form_result["n_valid"] >= 20 and form_result["lo"] > 0:
        print("PART 2 (OPPONENT TRAJECTORY): real signal - beating a currently-hot opponent shows a "
              "significantly larger next-match residual than beating a cold one.")
    else:
        print("PART 2 (OPPONENT TRAJECTORY): no statistically real difference between beating a "
              "currently-hot opponent and a cold one, among former-elite below-peak players.")
        if form_result is not None and not form_result["held_out_attempted"]:
            print("  (population too thin for a held-out verdict at this scope - see counts above)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gap-threshold", type=float, default=FORMER_ELITE_GAP_THRESHOLD_DEFAULT)
    parser.add_argument("--min-history-years", type=float, default=MIN_HISTORY_YEARS_DEFAULT)
    args = parser.parse_args()
    run(args.gap_threshold, args.min_history_years)
