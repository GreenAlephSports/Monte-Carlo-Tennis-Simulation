"""Grid search over RANK_ADJUSTMENT_ELO_WINDOW (win_probability.py, currently 50) - is the
existing cutoff actually the best-performing gate for the rank-gap correction, or would a
narrower/wider window measurably beat it?

The rank-gap correction (RANK_ADJUSTMENT_C / RANK_ADJUSTMENT_D, fit once on ATP data - see
win_probability.py's own docstring) is held FIXED here; only the |Elo diff| <= W gate that decides
which matches ever get the correction applied is varied. That mirrors exactly what the
RANK_ADJUSTMENT_ELO_WINDOW constant controls in production (win_probability(), line ~385) - it is
not a re-fit of the shift formula itself, which is a separate question already answered by the
docstring's own sensitivity note ("held up across every Elo-window and rank-range sensitivity
check tried").

Same rigor as every other correction tested tonight: frozen per-tournament-edition Elo (reused
from elite_opponent_residual_test.build_frozen_predictions, extended here to also carry each
player's OWN current rank, not just their opponent's - the rank-gap correction needs both sides),
chronological tournament-level 80/20 train/test split, held-out log-loss, player-clustered
bootstrap CIs (survivorship_upset_test.cluster_bootstrap_ci).

Two views are reported per candidate cutoff W:
  1. CUMULATIVE: held-out log-loss improvement (raw Elo vs. rank-adjusted) over every test-era row
     with |Elo diff| <= W - i.e. "if production gated at W, how much does the correction help
     overall". Directly comparable across W since larger W is a strict superset of smaller W.
  2. MARGINAL bands (0-30, 30-40, 40-50, 50-60, 60-75, 75-100, 100+): the same improvement computed
     separately in each non-overlapping Elo-diff band, to see exactly where the signal decays -
     the real answer to "would a different cutoff be measurably better" lives here, not in the
     cumulative number alone.

ATP only - the correction's own docstring says it's fit on ATP data and "not validated for WTA";
running it on WTA here would test a different, unvalidated question.

Usage:
    python model/research/rank_adjustment_window_grid_search.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import TRAIN_FRACTION, log_loss  # noqa: E402
from elo_ratings import K_FACTOR, STARTING_ELO, expected_score, load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from win_probability import RANK_ADJUSTMENT_C, RANK_ADJUSTMENT_D, _apply_rank_adjustment  # noqa: E402

CANDIDATE_WINDOWS = [30, 40, 50, 60, 75]
MARGINAL_BANDS = [
    ("0_30", 0, 30), ("30_40", 30, 40), ("40_50", 40, 50), ("50_60", 50, 60),
    ("60_75", 60, 75), ("75_100", 75, 100), ("100_plus", 100, float("inf")),
]


def build_frozen_predictions_with_own_rank(df):
    """Same walk-forward as elite_opponent_residual_test.build_frozen_predictions (frozen
    per-tournament-edition overall Elo, single continuously-updated pass - same disclosed
    simplification every other correction in this series uses), extended to also snapshot each
    player's OWN current rank alongside their opponent's - build_frozen_predictions only returns
    opponent_rank, but the rank-gap correction needs both sides of the match."""
    df = df.copy()
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)
    edition_start = df.groupby("edition_id")["Date"].transform("min")
    editions = (
        df.assign(edition_start=edition_start)[["edition_id", "edition_start"]]
        .drop_duplicates()
        .sort_values("edition_start")
        .reset_index(drop=True)
    )

    overall_elo, current_rank = {}, {}
    rows = []
    for edition_id in editions["edition_id"]:
        edition_matches = df[df["edition_id"] == edition_id]

        snapshot_elo = dict(overall_elo)
        snapshot_rank = dict(current_rank)
        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            elo_p1 = snapshot_elo.get(p1, STARTING_ELO)
            elo_p2 = snapshot_elo.get(p2, STARTING_ELO)
            pred_p1 = expected_score(elo_p1, elo_p2)
            win1 = 1 if winner == p1 else 0
            rows.append((edition_id, row.Date, row.Round, p1, p2, elo_p1, elo_p2, pred_p1, win1,
                         snapshot_rank.get(p1), snapshot_rank.get(p2)))
            rows.append((edition_id, row.Date, row.Round, p2, p1, elo_p2, elo_p1, 1 - pred_p1, 1 - win1,
                         snapshot_rank.get(p2), snapshot_rank.get(p1)))

        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            overall_elo.setdefault(p1, STARTING_ELO)
            overall_elo.setdefault(p2, STARTING_ELO)
            score1 = 1.0 if winner == p1 else 0.0
            exp1 = expected_score(overall_elo[p1], overall_elo[p2])
            overall_elo[p1] += K_FACTOR * (score1 - exp1)
            overall_elo[p2] += K_FACTOR * ((1 - score1) - (1 - exp1))
            if pd.notna(row.Rank_1) and row.Rank_1 > 0:
                current_rank[p1] = row.Rank_1
            if pd.notna(row.Rank_2) and row.Rank_2 > 0:
                current_rank[p2] = row.Rank_2

    preds = pd.DataFrame(rows, columns=[
        "edition_id", "date", "round", "player", "opponent", "player_elo", "opponent_elo",
        "pred_win", "actual_win", "own_rank", "opponent_rank",
    ])
    return preds, editions


def report_window(label, rows):
    rows = rows.dropna(subset=["own_rank", "opponent_rank"])
    if len(rows) < 20:
        print(f"  {label:<10}: n={len(rows):<6} - too few rank-known rows to evaluate")
        return
    adjusted = rows.apply(
        lambda r: _apply_rank_adjustment(r["pred_win"], r["own_rank"], r["opponent_rank"]), axis=1)
    rows = rows.assign(
        raw_loss=log_loss(rows["actual_win"].values, rows["pred_win"].values),
        adj_loss=log_loss(rows["actual_win"].values, adjusted.values),
    )
    observed, lo, hi = cluster_bootstrap_ci(rows, "raw_loss", "adj_loss", group_col="player")
    sig = "  <- excludes zero" if (lo > 0 or hi < 0) else ""
    print(f"  {label:<10}: n={len(rows):<6} held-out log-loss improvement (raw-adj)={observed:+.5f} "
          f"CI[{lo:+.5f},{hi:+.5f}]{sig}")


def run():
    matches = load_matches_for_tour("ATP")
    preds, editions = build_frozen_predictions_with_own_rank(matches)
    preds["elo_diff"] = (preds["player_elo"] - preds["opponent_elo"]).abs()

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    test = preds[preds["edition_id"].isin(test_editions)]
    print(f"ATP: {len(editions)} tournament editions "
          f"({editions['edition_start'].min().date()} to {editions['edition_start'].max().date()}); "
          f"train = first {len(train_editions)} editions (through "
          f"{editions['edition_start'].iloc[split_idx - 1].date()}), "
          f"test = remaining {len(test_editions)} editions "
          f"(from {editions['edition_start'].iloc[split_idx].date()}), {len(test)} test-era rows")
    print(f"Fixed formula being gated: RANK_ADJUSTMENT_C={RANK_ADJUSTMENT_C}, "
          f"RANK_ADJUSTMENT_D={RANK_ADJUSTMENT_D} (unchanged - only the |Elo diff| gate varies)")

    print(f"\n{'=' * 100}\nCUMULATIVE: held-out improvement over ALL test rows with |Elo diff| <= W "
          f"(nested supersets - what production would see at each candidate gate)\n{'=' * 100}")
    for w in CANDIDATE_WINDOWS:
        report_window(f"W<={w}", test[test["elo_diff"] <= w])

    print(f"\n{'=' * 100}\nMARGINAL: held-out improvement inside each non-overlapping Elo-diff band "
          f"(where does the signal actually decay?)\n{'=' * 100}")
    for name, lo_edge, hi_edge in MARGINAL_BANDS:
        band = test[(test["elo_diff"] > lo_edge) & (test["elo_diff"] <= hi_edge)] if lo_edge > 0 \
            else test[test["elo_diff"] <= hi_edge]
        report_window(name, band)

    print(f"\n{'=' * 100}\nVERDICT\n{'=' * 100}")
    print("Compare the CUMULATIVE numbers above: the best-performing W is whichever has the "
          "largest positive held-out improvement with a CI excluding zero; if 30/40/50/60/75 all "
          "exclude zero and overlap heavily, they are not measurably different from each other and "
          "50 stands on parsimony grounds (it's the already-validated, already-shipped value). If "
          "a MARGINAL band beyond 50 (50_60, 60_75) also excludes zero on the positive side, that's "
          "direct evidence 50 is leaving real signal ungated and a wider cutoff would help; if a "
          "marginal band inside 50 (e.g. 40_50) does NOT exclude zero while 0_30/30_40 clearly do, "
          "that's evidence the correction's real operating range is narrower than 50 and a smaller "
          "cutoff would lose nothing.")


if __name__ == "__main__":
    run()
