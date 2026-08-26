"""Tests a specific, structured hypothesis: for players who are BOTH (a) old (age >= threshold,
tested at 33/35/37) AND (b) still carrying a historically-elite Elo rating, does their actual win
rate underperform what Elo predicts, beyond normal sampling variance? If real, this would be a
genuinely new sixth correction candidate - an age-decline adjustment distinct from the layoff
adjustment (which only captures absence, not a rating built up years ago no longer reflecting
current physical decline).

Data-availability note (read before trusting any number below): none of this project's Kaggle
match data (Tournament/Date/.../Rank_1/Rank_2/Pts_1/Pts_2/Odd_1/Odd_2/Score - checked directly,
no age/birthdate column exists) carries player age at all, unlike every other correction test in
this series, which derived their signal from columns already in the pipeline. Age here comes from
a ONE-OFF external join against Tennismylife/TML-Database's ATP_Database.csv (a real, public,
actively-maintained player-bio table with real birthdates - not this project's data source, not
wired into the production pipeline, ATP only: no comparable WTA source was found, Jeff Sackmann's
long-referenced tennis_atp/tennis_wta GitHub repos - the usual go-to for this - both now 404
across three independent fetch attempts, and his GitHub profile lists neither anymore). Matching
Kaggle's "Lastname I." player names to TML's "First Last" names reuses bracket.py's own
match_name_to_pool tiered fuzzy matcher (the same machinery this project already trusts to
reconcile cross-dataset name spelling differences elsewhere) - match coverage is reported below,
not assumed.

Same rigor as every other test tonight: frozen per-tournament-edition Elo and the chronological
80/20 train/test split reused directly from elite_opponent_residual_test.build_frozen_predictions,
held-out validation of a fitted logit-shift adjustment against raw Elo, and player-clustered
bootstrap confidence intervals (survivorship_upset_test.cluster_bootstrap_ci) so one long-lived
veteran's many matches don't count as many independent data points.

"Historically elite" is defined data-drivenly, not guessed: the top 10% of frozen player_elo
values across the whole train-era dataset (both ATP - the only tour with real age data here).

Usage:
    python model/research/veteran_decline_test.py [--elo-percentile 90]
"""
import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import match_name_to_pool  # noqa: E402
from elite_opponent_residual_test import TRAIN_FRACTION, build_frozen_predictions, log_loss, logit, sigmoid  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci, summarize_bucket  # noqa: E402

TML_BIRTHDATES_PATH = Path(
    r"C:\Users\idanh\AppData\Local\Temp\claude\x--idanh-Documents-VS-Code-Projects-Monte-Carlo-Simulation-Grand-Slam-Model-"
    r"\54b74d84-4643-4e25-9b66-35e4496bc57b\scratchpad\atp_players_birthdates.csv"
)
AGE_THRESHOLDS = [33, 35, 37]


def _to_csv_format(full_name):
    """'Novak Djokovic' -> 'Djokovic N.' - converts TML's 'First Last' shape into the same
    'Lastname I.' shape Kaggle's Player_1/Player_2 (and bracket.py's whole matching system) use,
    so match_name_to_pool's tiered matcher (exact lastname+initials, then first-initial, then
    glued-lastname prefix/suffix fuzzy match) can be reused as-is rather than reinvented."""
    tokens = full_name.split()
    if len(tokens) < 2:
        return None
    first, *rest = tokens
    return f"{' '.join(rest)} {first[0]}."


def load_birthdates():
    df = pd.read_csv(TML_BIRTHDATES_PATH, dtype=str, encoding="latin-1")
    df = df.dropna(subset=["player", "birthdate"])
    df = df[df["birthdate"].str.len() == 8]  # a real YYYYMMDD value, not blank/garbage
    df["csv_name"] = df["player"].apply(_to_csv_format)
    df = df.dropna(subset=["csv_name"])
    df["birthdate_parsed"] = pd.to_datetime(df["birthdate"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["birthdate_parsed"])
    # a handful of names collide onto the same converted csv_name (e.g. two different players
    # both reducing to "Smith J.") - keep the first, report the collision count so it's not silent
    n_before = len(df)
    df = df.drop_duplicates(subset="csv_name", keep="first")
    print(f"Loaded {n_before} TML player rows with a usable birthdate; "
          f"{n_before - len(df)} collided onto a duplicate converted name and were dropped, "
          f"leaving {len(df)} candidate pool names")
    return df.set_index("csv_name")["birthdate_parsed"]


def attach_age(preds, birthdate_by_name):
    pool_names = list(birthdate_by_name.index)
    unique_players = preds["player"].unique()
    resolved = {p: match_name_to_pool(p, pool_names) for p in unique_players}
    n_matched = sum(v is not None for v in resolved.values())
    print(f"Matched {n_matched}/{len(unique_players)} distinct players in the match dataset to a "
          f"TML birthdate ({n_matched / len(unique_players):.1%} coverage)")

    preds = preds.copy()
    preds["birthdate"] = preds["player"].map(resolved).map(birthdate_by_name)
    preds["age_years"] = (preds["date"] - preds["birthdate"]).dt.days / 365.25
    return preds


def run(elo_percentile):
    matches = load_matches_for_tour("ATP")
    preds, editions = build_frozen_predictions(matches)
    birthdate_by_name = load_birthdates()
    preds = attach_age(preds, birthdate_by_name)

    has_age = preds["age_years"].notna()
    print(f"{has_age.sum()}/{len(preds)} player-perspective rows have a resolved age "
          f"({has_age.mean():.1%} of the dataset)")

    split_idx = int(len(editions) * TRAIN_FRACTION)
    train_editions = set(editions["edition_id"].iloc[:split_idx])
    test_editions = set(editions["edition_id"].iloc[split_idx:])
    train_all = preds[preds["edition_id"].isin(train_editions)]

    elite_elo_threshold = train_all["player_elo"].quantile(elo_percentile / 100)
    print(f"\n'Historically elite' Elo threshold (train-era p{elo_percentile}, ATP): "
          f"{elite_elo_threshold:.0f}")

    for age_threshold in AGE_THRESHOLDS:
        print(f"\n{'=' * 90}\nAGE THRESHOLD >= {age_threshold}\n{'=' * 90}")

        df = preds[preds["age_years"].notna()].copy()
        df["bucket"] = np.where(
            (df["age_years"] >= age_threshold) & (df["player_elo"] >= elite_elo_threshold),
            "old_and_elite", "other",
        )
        train = df[df["edition_id"].isin(train_editions)]
        test = df[df["edition_id"].isin(test_editions)]

        n_old_elite_train = (train["bucket"] == "old_and_elite").sum()
        n_old_elite_players_train = train.loc[train["bucket"] == "old_and_elite", "player"].nunique()
        print(f"Train-era 'old (>= {age_threshold}) AND elite (Elo >= {elite_elo_threshold:.0f})' rows: "
              f"{n_old_elite_train} ({n_old_elite_players_train} distinct players)")
        if n_old_elite_players_train:
            names = sorted(train.loc[train["bucket"] == "old_and_elite", "player"].unique())
            print(f"  players: {names}")

        if n_old_elite_train < 20:
            print(f"  Too few train-era rows ({n_old_elite_train}) to estimate anything meaningful "
                  f"at this threshold - skipping.")
            continue

        summary = pd.DataFrame([summarize_bucket(b, g) for b, g in train.groupby("bucket") if len(g)]) \
            .set_index("bucket").reindex(["other", "old_and_elite"]).reset_index()
        print("\n--- Train-era residual (what Elo gets wrong, before any adjustment) ---")
        print(summary.to_string(index=False, formatters={
            "actual_rate": "{:.1%}".format, "pred_rate": "{:.1%}".format, "residual": "{:+.1%}".format,
            "residual_ci_lo": "{:+.1%}".format, "residual_ci_hi": "{:+.1%}".format, "z": "{:.2f}".format,
        }))

        # held-out validation: fit a logit shift on train-era old_and_elite rows, apply to the
        # matching test-era rows, compare log-loss/Brier to raw (unadjusted) Elo
        g_train = train[train["bucket"] == "old_and_elite"]
        actual_rate, pred_rate = g_train["actual_win"].mean(), g_train["pred_win"].mean()
        shift = logit(actual_rate) - logit(pred_rate)

        test_col = test[test["bucket"] == "old_and_elite"].copy()
        n_test_players = test_col["player"].nunique()
        print(f"\nHeld-out test-era 'old and elite' rows: {len(test_col)} ({n_test_players} players)")
        if len(test_col) < 10:
            print("  Too few held-out rows to validate - skipping held-out check for this threshold.")
            continue

        test_col["adjusted_pred"] = sigmoid_series = test_col["pred_win"].apply(
            lambda p: sigmoid(logit(p) + shift))
        test_col["raw_loss"] = log_loss(test_col["actual_win"].values, test_col["pred_win"].values)
        test_col["adj_loss"] = log_loss(test_col["actual_win"].values, test_col["adjusted_pred"].values)
        test_col["raw_brier"] = (test_col["actual_win"] - test_col["pred_win"]) ** 2
        test_col["adj_brier"] = (test_col["actual_win"] - test_col["adjusted_pred"]) ** 2

        observed, lo, hi = cluster_bootstrap_ci(test_col, "raw_loss", "adj_loss")
        print(f"  Fitted train-era logit shift: {shift:+.4f} (actual {actual_rate:.1%} vs. "
              f"Elo-predicted {pred_rate:.1%})")
        print(f"  Raw Elo        : log-loss = {test_col['raw_loss'].mean():.4f}, "
              f"Brier = {test_col['raw_brier'].mean():.4f}")
        print(f"  Age-adjusted   : log-loss = {test_col['adj_loss'].mean():.4f}, "
              f"Brier = {test_col['adj_brier'].mean():.4f}")
        print(f"  Mean per-match log-loss improvement (raw - adjusted, >0 = adjustment better), "
              f"player-clustered: {observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")

        ci_excludes_zero = lo > 0 or hi < 0
        train_z = summary.loc[summary["bucket"] == "old_and_elite", "z"].iloc[0]
        print(f"\n  VERDICT @ age >= {age_threshold}: train-era residual z = {train_z:.2f} "
              f"({'|z|>1.96, nominally significant' if abs(train_z) > 1.96 else 'not significant on its own'}); "
              f"held-out CI {'excludes zero - real, held-out-validated effect' if ci_excludes_zero else 'straddles zero - NOT validated out of sample'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--elo-percentile", type=float, default=90,
                         help="train-era player_elo percentile defining 'historically elite'")
    args = parser.parse_args()
    run(args.elo_percentile)
