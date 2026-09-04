"""Tests whether a lefty-vs-righty handedness MATCHUP predicts the real match outcome beyond what
the players' actual Elo gap already explains - the same controlled-regression isolation technique
pedigree_market_premium_test.py uses (OLS on elo_diff + the variable under test), not a raw
lefty-population-win-rate comparison, which would silently credit handedness for whatever is really
just "lefties happen to skew higher/lower Elo in this sample."

Dependent variable: the REAL match outcome (did the alphabetically-first player win), not a
market/model gap - this is a "does this variable carry real predictive signal" question, unlike the
pedigree test's "does the MARKET overpay for this beyond Elo" question, so the target is different,
but the isolate-the-control-first method is identical: elo_diff has to be in the regression before
the handedness-matchup coefficient means anything, exactly as pedigree_diff was only trustworthy
once elo_diff sat alongside it there.

Real historical outcomes + real frozen (no-lookahead) Elo come from
elite_opponent_residual_test.build_frozen_predictions - the same walk-forward machinery already
validated there (single continuously-updated overall_elo, exactly like that test's own documented
simplification; frozen per-edition, so no in-tournament or future-match leakage). That function
emits two player-perspective rows per real match (p1-vs-p2 and p2-vs-p1); this test collapses back
to ONE neutral row per match (player < opponent alphabetically, matching pedigree_market_premium_
test's own "neutral alphabetical a/b ordering, never anchored to the real winner" - anchoring to
the winner would trivially and spuriously correlate elo_diff with the outcome by construction).

Handedness comes from output/player_handedness.csv (this project's own Wikipedia-infobox scrape,
model/research/wikipedia_handedness_scrape.py) - only rows with a clean Right-handed/Left-handed
value on BOTH sides of a match are usable; Ambidextrous and unresolved/missing players are dropped
rather than guessed.

Usage:
    python model/research/handedness_matchup_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import build_frozen_predictions  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
HANDEDNESS_PATH = OUTPUT_DIR / "player_handedness.csv"


def load_hand_map():
    """{player_csv_name: 'Right-handed'|'Left-handed'} - Ambidextrous and anything without a clean
    resolved hand (unresolved_name, no_infobox, missing_plays_field, malformed_plays,
    hand_only_no_backhand doesn't apply here since hand itself IS known there - only backhand type
    is missing, so hand_only_no_backhand rows ARE usable for this hand-only test) are excluded."""
    df = pd.read_csv(HANDEDNESS_PATH, keep_default_na=False, dtype=str)
    usable = df[df["hand"].isin(["Right-handed", "Left-handed"])]
    return dict(zip(usable["player"], usable["hand"]))


def one_row_per_match(preds):
    """Collapses build_frozen_predictions' two-rows-per-match (player, opponent) + (opponent,
    player) format down to one neutral row per real match: keep only the row where `player` sorts
    alphabetically before `opponent` - exactly one of the two duplicate rows satisfies this for
    every match, and which one that is has nothing to do with who actually won, so this can't leak
    winner-anchoring bias into elo_diff the way keeping "the winner's row" would."""
    neutral = preds[preds["player"] < preds["opponent"]].copy()
    neutral = neutral.rename(columns={
        "player": "player_a", "opponent": "player_b",
        "player_elo": "elo_a", "opponent_elo": "elo_b", "actual_win": "won_a",
    })
    neutral["elo_diff"] = neutral["elo_a"] - neutral["elo_b"]
    return neutral[["edition_id", "date", "round", "player_a", "player_b", "elo_diff", "won_a"]]


def ols(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    n, k = X1.shape
    sigma2 = (resid @ resid) / (n - k)
    cov = sigma2 * np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    return beta, se


def run():
    hand = load_hand_map()
    print(f"Handedness known (Right-handed/Left-handed only) for {len(hand)} players "
          f"({sum(1 for h in hand.values() if h == 'Left-handed')} lefties, "
          f"{sum(1 for h in hand.values() if h == 'Right-handed')} righties)")

    frames = []
    for tour in ("ATP", "WTA"):
        matches = load_matches_for_tour(tour)
        preds, editions = build_frozen_predictions(matches)
        rows = one_row_per_match(preds)
        rows["tour"] = tour
        print(f"{tour}: {len(editions)} tournament editions, {len(rows)} real historical matches "
              f"with a frozen pre-match Elo gap")
        frames.append(rows)

    all_rows = pd.concat(frames, ignore_index=True)

    all_rows["hand_a"] = all_rows["player_a"].map(hand)
    all_rows["hand_b"] = all_rows["player_b"].map(hand)
    usable = all_rows.dropna(subset=["hand_a", "hand_b"]).copy()
    print(f"\n{len(all_rows)} total real matches (both tours) with a resolved Elo gap; "
          f"{len(usable)} of those have a known Right/Left hand for BOTH players "
          f"({len(usable) / len(all_rows):.1%} coverage) - this is the real sample the handedness "
          f"question has to work with, not the full {len(all_rows)}.")

    usable["mismatch"] = (usable["hand_a"] != usable["hand_b"]).astype(int)
    same_hand_pairs = usable["hand_a"] + " vs " + usable["hand_b"]
    print("\nMatchup composition:")
    print(same_hand_pairs.value_counts().to_string())

    # naive, unadjusted comparison - context for WHY a control is needed, not the answer itself
    naive_mismatch = usable.loc[usable["mismatch"] == 1, "won_a"]
    naive_same = usable.loc[usable["mismatch"] == 0, "won_a"]
    print(f"\nNaive (unadjusted) win rate for player_a: "
          f"lefty-vs-righty matches = {naive_mismatch.mean():.1%} (n={len(naive_mismatch)}), "
          f"same-handedness matches = {naive_same.mean():.1%} (n={len(naive_same)}) "
          f"- meaningless on its own since player_a is an arbitrary alphabetical label, shown only "
          f"to confirm the a/b split itself isn't secretly lopsided.")
    naive_elo_mismatch = usable.loc[usable["mismatch"] == 1, "elo_diff"].abs().mean()
    naive_elo_same = usable.loc[usable["mismatch"] == 0, "elo_diff"].abs().mean()
    print(f"Mean |elo_diff| in each group: lefty-vs-righty = {naive_elo_mismatch:.1f}, "
          f"same-handedness = {naive_elo_same:.1f} "
          f"(if these differ a lot, a raw win-rate-by-matchup-type comparison would be confounded "
          f"by skill gap - which is exactly why elo_diff has to be in the regression, not the SE).")

    print(f"\n{'=' * 90}\nOLS: won_a ~ elo_diff + mismatch, all {len(usable)} matches "
          f"(linear probability model, same technique as pedigree_market_premium_test.py)\n{'=' * 90}")
    y = usable["won_a"].values.astype(float)
    X = usable[["elo_diff", "mismatch"]].values.astype(float)
    beta, se = ols(y, X)
    names = ["intercept", "elo_diff", "mismatch"]
    for name, b, s in zip(names, beta, se):
        z = b / s if s == s and s != 0 else float("nan")
        print(f"  {name:<12}: coef={b:+.5f}  SE={s:.5f}  z={z:+.2f}"
              + ("  (|z|>1.96, nominally significant)" if abs(z) > 1.96 else "  (not significant)"))

    mismatch_z = beta[2] / se[2]
    print(f"\n{'=' * 90}")
    if abs(mismatch_z) > 1.96:
        print(f"Handedness-matchup coefficient IS significant after controlling for elo_diff "
              f"(z={mismatch_z:+.2f}): a lefty-vs-righty matchup shifts player_a's win probability "
              f"by {beta[2]:+.1%} beyond what the Elo gap alone predicts, holding elo_diff fixed.")
    else:
        print(f"Handedness-matchup coefficient is NOT significant after controlling for elo_diff "
              f"(z={mismatch_z:+.2f}, need |z|>1.96) - once real skill difference is accounted for, "
              f"lefty-vs-righty matchup shows no real edge for player_a in this sample. Same honest "
              f"null result pedigree_market_premium_test.py found for pedigree once Elo was controlled.")
    print("=" * 90)

    print("\nPer-tour breakdown (same regression, run separately):")
    for tour in ("ATP", "WTA"):
        sub = usable[usable["tour"] == tour]
        if len(sub) < 30:
            print(f"  {tour}: n={len(sub)} - too small to run separately with any confidence")
            continue
        y_t = sub["won_a"].values.astype(float)
        X_t = sub[["elo_diff", "mismatch"]].values.astype(float)
        beta_t, se_t = ols(y_t, X_t)
        z_t = beta_t[2] / se_t[2]
        print(f"  {tour} (n={len(sub)}): mismatch coef={beta_t[2]:+.5f}  SE={se_t[2]:.5f}  "
              f"z={z_t:+.2f}" + ("  (significant)" if abs(z_t) > 1.96 else "  (not significant)"))


if __name__ == "__main__":
    run()
