"""Re-tests the within-tournament layoff-decay hypothesis (layoff_within_tournament_decay_test.py)
with a stricter, less circular performance signal, per direct request after that script's own
"performance-based decay" comparison was flagged as still potentially circular: it used "Elo gap
overcome in the player's most recent win" as the performance predictor, which - while not
literally the same actual-vs-predicted residual used elsewhere (a genuine bug that WAS caught and
fixed there: cumulative actual-minus-predicted is trivially positive for every survivor, since
reaching match N+1 REQUIRES having won match N) - is still a form of "did they win impressively,"
just measured through the opponent's rating rather than the scoreline.

This script isolates a more independent signal: MARGIN OF VICTORY in the match itself - how many
games they won by, and whether it took a deciding set - rather than anything about win/loss or
opponent strength. Two players who both win their 90d-layoff-starting 1st-round match (identical
actual_win=1, comparable opponent Elo) can have wildly different games_margin: 6-1 6-2 vs. 4-6 7-6
7-6. That's real information about how the player is striking the ball THAT DAY, unrelated to
whether Elo happened to already expect them to win.

Data availability, stated plainly up front: the Kaggle ATP/WTA datasets this codebase uses (see
data_loader_kaggle.py) carry only the set-by-set Score string (e.g. "6-3 6-7 4-6") - no serve/
return stats (aces, first-serve %, break points) exist anywhere in this pipeline's data. Score is
parsed into games_margin (games won - games lost, signed from the player's own perspective) and
straight_sets (won without dropping a set) as the two available margin-of-victory proxies. ~0.03%
of rows have a Score inconsistent with the recorded Winner (mid-set retirements scored oddly, from
a spot-check) - those rows' margin is dropped (score_consistent = False), not guessed at.

Population and methodology are otherwise identical to layoff_within_tournament_decay_test.py:
players whose FIRST tournament match falls in the 90d_plus layoff bucket, frozen per-edition Elo,
chronological tournament-level 80/20 split, held-out validation against raw Elo, player-clustered
bootstrap CIs. PRIMARY test: does match_1 margin of victory predict the size of the match_2
residual (the literal question asked - "does game 1 performance quality predict whether game 2
still shows the penalty"). SUPPLEMENTARY: the same test pooled across match_number >= 2, using each
match's immediately-preceding-match margin, for more power at the cost of mixing round-2/3/4+
together.

Usage:
    python model/research/layoff_margin_of_victory_test.py [--tour ATP|WTA]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import TRAIN_FRACTION, build_frozen_predictions, log_loss, logit, sigmoid  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from layoff_test import build_layoff_dataset  # noqa: E402
from layoff_within_tournament_decay_test import (  # noqa: E402
    MATCH_NUMBER_BUCKETS, build_within_tournament_sequences, fit_shifts, match_number_bucket, score,
)
from survivorship_upset_test import cluster_bootstrap_ci, summarize_bucket  # noqa: E402
from win_probability import LAYOFF_BUCKET_EDGES_ATP, LAYOFF_BUCKET_EDGES_WTA  # noqa: E402

MIN_SAMPLE_FOR_ANY_CONFIDENCE = 30

SCORE_SET_RE = re.compile(r"(\d+)-(\d+)(?:\(\d+\))?")


def parse_score(score_str):
    """(p1_sets, p2_sets, p1_games, p2_games) from a raw 'Score' string like '6-3 6-7 4-6', or None
    if any set token doesn't match the expected 'games-games(tiebreak)?' shape - unparseable rows
    are excluded from margin analysis entirely rather than guessed at."""
    p1_sets = p2_sets = p1_games = p2_games = 0
    tokens = str(score_str).split()
    if not tokens:
        return None
    for tok in tokens:
        m = SCORE_SET_RE.fullmatch(tok)
        if not m:
            return None
        a, b = int(m.group(1)), int(m.group(2))
        p1_games += a
        p2_games += b
        if a > b:
            p1_sets += 1
        elif b > a:
            p2_sets += 1
    return p1_sets, p2_sets, p1_games, p2_games


def build_margin_lookup(matches):
    """One row per (edition_id, date, round, player, opponent) - same key convention as
    layoff_test.compute_days_since_last_match - carrying that player's own games_margin
    (games_for - games_against, signed so positive always favors this row's 'player') and
    straight_sets (True if they didn't drop a set) for that specific match. score_consistent flags
    the rare (~0.03% in a spot check) row where the parsed set-winner majority disagrees with the
    recorded Winner column (mid-set retirement scored oddly) - those rows' margin should not be
    trusted and are left for the caller to filter out."""
    df = matches.copy()
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)
    parsed = df["Score"].apply(parse_score)
    ok = parsed.notna()
    df = df[ok].copy()
    p1_sets, p2_sets, p1_games, p2_games = zip(*parsed[ok])
    df["p1_sets"], df["p2_sets"] = p1_sets, p2_sets
    df["p1_games"], df["p2_games"] = p1_games, p2_games
    p1_won_majority = df["p1_sets"] > df["p2_sets"]
    p1_is_winner = df["Winner"] == df["Player_1"]
    df["score_consistent"] = p1_won_majority == p1_is_winner

    long = pd.concat([
        df.assign(player=df["Player_1"], opponent=df["Player_2"],
                   games_for=df["p1_games"], games_against=df["p2_games"], sets_against=df["p2_sets"]),
        df.assign(player=df["Player_2"], opponent=df["Player_1"],
                   games_for=df["p2_games"], games_against=df["p1_games"], sets_against=df["p1_sets"]),
    ], ignore_index=True)
    long["games_margin"] = long["games_for"] - long["games_against"]
    long["straight_sets"] = long["sets_against"] == 0
    long = long.rename(columns={"Date": "date", "Round": "round"})
    return long[[
        "edition_id", "date", "round", "player", "opponent", "games_margin", "straight_sets", "score_consistent",
    ]].drop_duplicates(subset=["edition_id", "date", "round", "player", "opponent"])


def attach_prior_margin(seq, margin_lookup):
    """Merges each row's OWN match margin in, then shifts it within (edition_id, player) so every
    row also carries prior_margin/prior_straight_sets/prior_score_consistent - the values from that
    player's IMMEDIATELY PRECEDING match in the same tournament (None for match_number == 1)."""
    seq = seq.merge(margin_lookup, on=["edition_id", "date", "round", "player", "opponent"], how="left")
    seq = seq.sort_values(["edition_id", "player", "match_number"])
    grouped = seq.groupby(["edition_id", "player"], sort=False)
    seq["prior_margin"] = grouped["games_margin"].shift(1)
    seq["prior_straight_sets"] = grouped["straight_sets"].shift(1)
    seq["prior_score_consistent"] = grouped["score_consistent"].shift(1)
    return seq


def run(tour):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)
    layoff_df = build_layoff_dataset(matches, preds)
    seq = build_within_tournament_sequences(layoff_df)
    margin_lookup = build_margin_lookup(matches)
    print(f"{tour}: {(~margin_lookup['score_consistent']).sum()} of {len(margin_lookup)} parsed "
          f"match rows have a Score disagreeing with the recorded Winner (dropped from margin "
          f"analysis, kept everywhere else) - {(~margin_lookup['score_consistent']).mean():.2%}")

    seq = attach_prior_margin(seq, margin_lookup)
    treatment = seq[seq["first_match_bucket"] == "90d_plus"].copy()
    treatment["mn_bucket"] = treatment["match_number"].apply(match_number_bucket)

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])

    # ================================================================================
    # PRIMARY: does match_1 margin of victory predict the match_2 residual specifically -
    # the literal question asked, not a pooled/broader proxy for it.
    # ================================================================================
    print(f"\n{'=' * 90}\nPRIMARY: match_1 margin of victory -> match_2 residual "
          f"(90d_plus-starting players only)\n{'=' * 90}")
    m2 = treatment[(treatment["match_number"] == 2) & (treatment["prior_score_consistent"] == True)].copy()  # noqa: E712
    m2["margin_bucket"] = np.where(m2["prior_straight_sets"], "dominant_win_straight_sets", "grinding_win_needed_decider")
    print(f"{len(treatment[treatment['match_number'] == 1])} players started this tournament in "
          f"the 90d_plus bucket; {len(treatment[treatment['match_number'] == 2])} of them won match "
          f"1 and reached match 2; {len(m2)} of those have a score-consistent match_1 margin "
          f"(margin_bucket split: {dict(m2['margin_bucket'].value_counts())})")

    if len(m2) < 20:
        print("\n  Too few match_2 rows with a usable match_1 margin to say anything at all - "
              "inconclusive by sample size. Treat the current flat-penalty design as the settled "
              "answer for now; this specific, stricter test simply doesn't have enough data to "
              "overturn it.")
    else:
        train_m2 = m2[m2["edition_id"].isin(train_editions)]
        test_m2 = m2[m2["edition_id"].isin(test_editions)]
        print(f"Same tournament-edition boundary as the rest of this series: {len(train_m2)} "
              f"train-era / {len(test_m2)} test-era match_2 rows")

        train_summary = pd.DataFrame([summarize_bucket(b, g) for b, g in train_m2.groupby("margin_bucket") if len(g)])
        print("\n--- Train-era match_2 residual (actual - raw-Elo-predicted) by match_1 margin ---")
        print(train_summary.to_string(index=False, formatters={
            "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format, "residual": "{:+.1%}".format,
            "residual_ci_lo": "{:+.1%}".format, "residual_ci_hi": "{:+.1%}".format, "z": "{:.2f}".format,
        }))
        avg_margin = m2.groupby("margin_bucket")["prior_margin"].mean()
        print(f"(mean match_1 games_margin per bucket, for context: "
              f"{dict(avg_margin.round(1))})")

        buckets = ["dominant_win_straight_sets", "grinding_win_needed_decider"]
        if len(train_m2) < MIN_SAMPLE_FOR_ANY_CONFIDENCE or any(
            len(train_m2[train_m2["margin_bucket"] == b]) == 0 for b in buckets
        ):
            print(f"\n  CAUTION: n={len(train_m2)} train-era rows is well below the "
                  f"~{MIN_SAMPLE_FOR_ANY_CONFIDENCE}-row floor this codebase's other calibration "
                  f"checks require - proceeding to held-out validation anyway for completeness, but "
                  f"this is directional at best.")

        shift_by_margin = fit_shifts(train_m2, "margin_bucket", buckets)
        for b, s in shift_by_margin.items():
            print(f"  Fitted train-era shift, {b}: {s:+.4f} logits (n={len(train_m2[train_m2['margin_bucket'] == b])})")

        bucket_edges = LAYOFF_BUCKET_EDGES_ATP if tour == "ATP" else LAYOFF_BUCKET_EDGES_WTA
        flat_shift = next(shift for name, _test, shift in bucket_edges if name == "90d_plus")
        test_scored = score(test_m2, "margin_bucket", shift_by_margin, flat_shift=flat_shift)

        if len(test_scored) == 0:
            print("\n  No test-era rows survived (both margin buckets empty in train, or none in "
                  "test) - inconclusive by sample size.")
        else:
            print(f"\n--- Held-out validation: {len(test_scored)} test-era match_2 rows, "
                  f"{test_scored['player'].nunique()} players ---")
            print(f"  Raw Elo (no layoff adj.)   : log-loss = {test_scored['raw_loss'].mean():.4f}, "
                  f"Brier = {test_scored['raw_brier'].mean():.4f}")
            print(f"  Current flat 90d_plus shift: log-loss = {test_scored['flat_loss'].mean():.4f}, "
                  f"Brier = {test_scored['flat_brier'].mean():.4f}")
            print(f"  Margin-of-victory adjusted : log-loss = {test_scored['adj_loss'].mean():.4f}, "
                  f"Brier = {test_scored['adj_brier'].mean():.4f}")

            obs_flat, lo_flat, hi_flat = cluster_bootstrap_ci(test_scored, "flat_loss", "adj_loss")
            print(f"  Margin-adjusted vs. CURRENT flat shift, player-clustered improvement "
                  f"(flat - margin_adj, >0 = margin-adjusted better): {obs_flat:+.4f}, "
                  f"95% CI [{lo_flat:+.4f}, {hi_flat:+.4f}]")
            obs_raw, lo_raw, hi_raw = cluster_bootstrap_ci(test_scored, "raw_loss", "adj_loss")
            print(f"  Margin-adjusted vs. raw Elo, player-clustered improvement "
                  f"(raw - margin_adj, >0 = margin-adjusted better): {obs_raw:+.4f}, "
                  f"95% CI [{lo_raw:+.4f}, {hi_raw:+.4f}]")

            if len(test_m2) < MIN_SAMPLE_FOR_ANY_CONFIDENCE:
                print(f"\n  CAUTION: n={len(test_m2)} test-era rows is well below the "
                      f"~{MIN_SAMPLE_FOR_ANY_CONFIDENCE}-row floor - this held-out read is "
                      f"directional only, not a settled verdict.")

    # ================================================================================
    # SUPPLEMENTARY: pooled across match_number >= 2, using each match's immediately-preceding
    # match's margin - more power, at the cost of mixing round 2/3/4+ together. Reported
    # separately and clearly labeled, not substituted for the primary (match_2-only) result above.
    # ================================================================================
    print(f"\n{'=' * 90}\nSUPPLEMENTARY (more power, less targeted): prior-match margin -> "
          f"residual, pooled across match_number >= 2\n{'=' * 90}")
    mn2plus = treatment[(treatment["match_number"] >= 2) & (treatment["prior_score_consistent"] == True)].copy()  # noqa: E712
    mn2plus["margin_bucket"] = np.where(mn2plus["prior_straight_sets"], "dominant_win_straight_sets", "grinding_win_needed_decider")
    print(f"{len(mn2plus)} match_number>=2 rows with a usable prior-match margin "
          f"(bucket split: {dict(mn2plus['margin_bucket'].value_counts())})")

    train2 = mn2plus[mn2plus["edition_id"].isin(train_editions)]
    test2 = mn2plus[mn2plus["edition_id"].isin(test_editions)]
    if len(train2) == 0 or len(test2) == 0 or any(
        len(train2[train2["margin_bucket"] == b]) == 0 for b in ["dominant_win_straight_sets", "grinding_win_needed_decider"]
    ):
        print("\n  Not enough data for the supplementary pooled comparison either - inconclusive "
              "by sample size, not by the hypothesis failing.")
        return

    print(f"{len(train2)} train-era / {len(test2)} test-era rows")
    shift_by_margin2 = fit_shifts(train2, "margin_bucket", ["dominant_win_straight_sets", "grinding_win_needed_decider"])
    for b, s in shift_by_margin2.items():
        print(f"  Fitted train-era shift, {b}: {s:+.4f} logits (n={len(train2[train2['margin_bucket'] == b])})")

    scored2 = score(test2, "margin_bucket", shift_by_margin2)
    obs2, lo2, hi2 = cluster_bootstrap_ci(scored2, "raw_loss", "adj_loss")
    print(f"\n--- Held-out validation: {len(scored2)} test-era rows, {scored2['player'].nunique()} players ---")
    print(f"  Raw Elo                  : log-loss = {scored2['raw_loss'].mean():.4f}, Brier = {scored2['raw_brier'].mean():.4f}")
    print(f"  Margin-of-victory adjusted: log-loss = {scored2['adj_loss'].mean():.4f}, Brier = {scored2['adj_brier'].mean():.4f}")
    print(f"  Margin-adjusted vs. raw Elo, player-clustered improvement: {obs2:+.4f}, "
          f"95% CI [{lo2:+.4f}, {hi2:+.4f}]")
    if len(test2) < MIN_SAMPLE_FOR_ANY_CONFIDENCE:
        print(f"\n  CAUTION: n={len(test2)} test-era rows is well below the "
              f"~{MIN_SAMPLE_FOR_ANY_CONFIDENCE}-row floor - directional only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tour", default="ATP", choices=["ATP", "WTA"])
    args = parser.parse_args()
    run(args.tour)
