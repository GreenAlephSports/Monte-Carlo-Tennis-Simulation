"""Two real gaps in former_elite_comeback_later_deep_run_test.py's "-18pp" finding, fixed here:

PART A - TIGHTER BASELINE: that script's baseline was "any below-peak player with no real recorded
layoff, for ANY reason" - undifferentiated. This part replaces it with two SPECIFICALLY-CAUSED
comparison groups instead, and reports whether -18pp survives against each:
  - AGE-BASED (ATP only - no WTA age source exists, exactly the same data-availability gap
    veteran_decline_test.py already disclosed): below-peak, no real layoff, age >= 33 - the same
    primary threshold veteran_decline_test.py itself found real (train z=-3.43, held-out CI
    [+0.0069,+0.0218]) via a one-off external join against Tennismylife/TML-Database's
    ATP_Database.csv (re-fetched fresh this session via WebFetch - the earlier session's cached
    copy no longer exists on disk - same bracket.match_name_to_pool tiered fuzzy matcher, match
    coverage reported below, not assumed).
  - GRADUAL-DECLINE (both tours): below-peak, no real layoff, AND the player was ALSO below-peak
    (gap_below_peak >= GRADUAL_DECLINE_GAP_THRESHOLD) in EACH of their previous
    GRADUAL_DECLINE_LOOKBACK real editions - i.e. a persistent, multi-edition erosion pattern, not
    a single isolated dip - the closest thing to "ranking/schedule gap" derivable directly from
    data already in this pipeline (current_rank/Elo trend) without a new external source.

PART B - REAL MARKET ODDS DURING THE COMEBACK EDITION ITSELF: Odd_1/Odd_2 are ALREADY present in
the same Kaggle dataset this entire project is built on (tennis-data.co.uk-sourced pregame
bookmaker prices, loaded by data_loader_kaggle.py, de-vigged the same way ev_comparison.
implied_probabilities/model_vs_market_calibration.py already do) - no new data source needed here,
unlike Part A. For every real match played DURING a comeback-pool instance's comeback edition (not
the whole 12-18mo forward window - the sharp question is what the market thought RIGHT THEN), this
reports real coverage honestly (overall and by decade - tennis-data.co.uk odds coverage is known to
thin out for older matches, and this is checked, not assumed) and, where coverage allows, compares
market log-loss vs. model log-loss on the SAME real outcomes: did the market discount these players
appropriately during the comeback, was it caught off guard while the model saw the risk, or the
reverse? Reported for both the single FIRST match of the comeback run (the sharpest "did the market
know" cut) and all matches within that comeback edition (more power, less precise a cut).

Same rigor as every prior test in this family: frozen per-tournament-edition Elo, both tours where
data allows, chronological tournament-edition 80/20 stability check, player-clustered bootstrap
CIs, real population/coverage sizes reported honestly before any conclusion, no held-out claim
forced on a sample too thin to support one.

Usage:
    python model/research/former_elite_comeback_refined_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import match_name_to_pool  # noqa: E402
from elite_opponent_residual_test import TRAIN_FRACTION, build_frozen_predictions, log_loss  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402
from ev_comparison import implied_probabilities  # noqa: E402
from former_elite_comeback_later_deep_run_test import (  # noqa: E402
    BASELINE_BUCKETS, COMEBACK_BUCKETS, MIN_INSTANCES_FOR_STATS,
    WINDOW_DAYS_PRIMARY, WINDOW_DAYS_SENSITIVITY, build_player_index, score_pool,
)
from former_elite_vs_current_top_test import (  # noqa: E402
    FORMER_ELITE_GAP_THRESHOLD_DEFAULT, MIN_HISTORY_YEARS_DEFAULT, cluster_bootstrap_group_diff,
)
from layoff_test import build_layoff_dataset  # noqa: E402
from layoff_within_tournament_decay_test import build_within_tournament_sequences  # noqa: E402
from peak_elo_recovery_test import add_peak_features  # noqa: E402
from survivorship_upset_test import ROUND_ORDER, cluster_bootstrap_ci  # noqa: E402
from veteran_decline_test import _to_csv_format  # noqa: E402

TML_BIRTHDATES_PATH = Path(
    r"C:\Users\idanh\AppData\Local\Temp\claude\x--idanh-Documents-VS-Code-Projects-Monte-Carlo-Simulation-Grand-Slam-Model-"
    r"\86d6ff86-cc77-4b5f-9274-8d6dc667b8b0\scratchpad\atp_players_birthdates.csv"
)
AGE_THRESHOLD = 33.0                      # same primary threshold veteran_decline_test.py validated
GRADUAL_DECLINE_GAP_THRESHOLD = 100.0
GRADUAL_DECLINE_LOOKBACK = 2
MIN_MARKET_ROWS_FOR_STATS = 20


def load_birthdates(path):
    df = pd.read_csv(path, dtype=str, encoding="latin-1")
    df = df.dropna(subset=["player", "birthdate"])
    df = df[df["birthdate"].str.len() == 8]
    df["csv_name"] = df["player"].apply(_to_csv_format)
    df = df.dropna(subset=["csv_name"])
    df["birthdate_parsed"] = pd.to_datetime(df["birthdate"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["birthdate_parsed"])
    n_before = len(df)
    df = df.drop_duplicates(subset="csv_name", keep="first")
    print(f"Loaded {n_before} TML player rows with a usable birthdate; "
          f"{n_before - len(df)} collided onto a duplicate converted name, leaving {len(df)} candidates")
    return df.set_index("csv_name")["birthdate_parsed"]


def build_run_level_and_seq(tour, gap_threshold, min_history_years):
    matches = load_matches_for_tour(tour)
    preds, editions = build_frozen_predictions(matches)
    preds = add_peak_features(preds)
    preds["gap_below_peak"] = (preds["career_peak_prior"] - preds["player_elo"]).clip(lower=0)

    layoff_df = build_layoff_dataset(matches, preds)
    seq = build_within_tournament_sequences(layoff_df)
    seq["round_order"] = seq["round"].map(ROUND_ORDER)

    peak_lvl = preds[["player", "edition_id", "gap_below_peak", "years_of_history"]].drop_duplicates(
        subset=["player", "edition_id"])
    seq = seq.merge(peak_lvl, on=["player", "edition_id"], how="left", validate="many_to_one")
    seq["tour"] = tour

    edition_start = editions.set_index("edition_id")["edition_start"]
    run_level = seq.groupby(["edition_id", "player"], sort=False).agg(
        deepest_round_order=("round_order", "max"),
        first_match_bucket=("first_match_bucket", "first"),
        gap_below_peak=("gap_below_peak", "first"),
        years_of_history=("years_of_history", "first"),
    ).reset_index()
    run_level["edition_start"] = run_level["edition_id"].map(edition_start)
    run_level["tour"] = tour
    data_last_date = matches["Date"].max()
    return run_level, seq, editions, data_last_date, matches


def add_gradual_decline_flag(run_level):
    run_level = run_level.sort_values(["player", "edition_start"]).reset_index(drop=True)
    g = run_level.groupby("player")["gap_below_peak"]
    run_level["prior1_gap"] = g.shift(1)
    run_level["prior2_gap"] = g.shift(2)
    run_level["is_gradual_decline"] = (
        (run_level["prior1_gap"] >= GRADUAL_DECLINE_GAP_THRESHOLD)
        & (run_level["prior2_gap"] >= GRADUAL_DECLINE_GAP_THRESHOLD)
    )
    return run_level


def add_age(run_level, birthdate_by_name):
    pool_names = list(birthdate_by_name.index)
    atp_mask = run_level["tour"] == "ATP"
    unique_players = run_level.loc[atp_mask, "player"].unique()
    resolved = {p: match_name_to_pool(p, pool_names) for p in unique_players}
    n_matched = sum(v is not None for v in resolved.values())
    print(f"Matched {n_matched}/{len(unique_players)} distinct ATP players to a TML birthdate "
          f"({n_matched / len(unique_players):.1%} coverage)")
    birthdate_series = pd.Series(pd.NaT, index=run_level.index, dtype="datetime64[ns]")
    birthdate_series.loc[atp_mask] = run_level.loc[atp_mask, "player"].map(resolved).map(birthdate_by_name).values
    run_level["birthdate"] = birthdate_series
    run_level["age_years"] = (run_level["edition_start"] - run_level["birthdate"]).dt.days / 365.25
    return run_level


def compare_pools(label, comeback_pool, tightened_pool, idx, data_last_date_by_tour, editions_by_tour):
    print(f"\n{'=' * 90}\n{label}\n{'=' * 90}")
    print(f"Comeback pool: {len(comeback_pool)} instances, {comeback_pool['player'].nunique()} players")
    print(f"Tightened baseline pool: {len(tightened_pool)} instances, {tightened_pool['player'].nunique()} players")
    if len(tightened_pool) < MIN_INSTANCES_FOR_STATS or len(comeback_pool) < MIN_INSTANCES_FOR_STATS:
        print("  Too few instances in one or both pools - not computing a comparison.")
        return

    for window_days, wlabel in [(WINDOW_DAYS_PRIMARY, "12mo"), (WINDOW_DAYS_SENSITIVITY, "18mo")]:
        print(f"\n  --- window = {wlabel} ({window_days}d) ---")
        comeback_scored = pd.concat([
            score_pool(comeback_pool[comeback_pool["tour"] == t], idx, data_last_date_by_tour[t], window_days, f"    comeback [{t}]")
            for t in comeback_pool["tour"].unique()
        ], ignore_index=True)
        tightened_scored = pd.concat([
            score_pool(tightened_pool[tightened_pool["tour"] == t], idx, data_last_date_by_tour[t], window_days, f"    tightened [{t}]")
            for t in tightened_pool["tour"].unique()
        ], ignore_index=True)
        if len(comeback_scored) < MIN_INSTANCES_FOR_STATS or len(tightened_scored) < MIN_INSTANCES_FOR_STATS:
            print("    Too few SCORED (post-censoring) instances - skipping this window.")
            continue
        comeback_scored["pool"], tightened_scored["pool"] = "comeback", "tightened"
        combined = pd.concat([comeback_scored, tightened_scored], ignore_index=True)
        combined["zero"] = 0.0
        obs, lo, hi, n_valid = cluster_bootstrap_group_diff(combined, "hit", "zero", "pool", "comeback", "tightened")
        if n_valid < 20:
            print("    Too few valid bootstrap replicates - inconclusive.")
            continue
        verdict = ("comeback pool LOWER hit rate (CI excludes zero, <0)" if hi < 0 else
                   ("comeback pool HIGHER hit rate (CI excludes zero, >0)" if lo > 0 else "NOT distinguishable"))
        print(f"    Hit-rate diff (comeback - tightened baseline): {obs:+.1%}, 95% CI "
              f"[{lo:+.1%}, {hi:+.1%}] -> {verdict}")

        for tour in set(comeback_pool["tour"]) & set(tightened_pool["tour"]):
            editions = editions_by_tour[tour]
            split_idx = int(len(editions) * TRAIN_FRACTION)
            train_editions = set(editions["edition_id"].iloc[:split_idx])
            tour_combined = combined[combined["tour"] == tour]
            for split_label, mask in [("train", tour_combined["edition_id"].isin(train_editions)),
                                       ("test", ~tour_combined["edition_id"].isin(train_editions))]:
                sub = tour_combined[mask]
                n_c, n_t = (sub["pool"] == "comeback").sum(), (sub["pool"] == "tightened").sum()
                if n_c < MIN_INSTANCES_FOR_STATS or n_t < MIN_INSTANCES_FOR_STATS:
                    print(f"      {tour} {split_label}-era: n_comeback={n_c}, n_tightened={n_t} - too few")
                    continue
                obs_s, lo_s, hi_s, nv_s = cluster_bootstrap_group_diff(sub, "hit", "zero", "pool", "comeback", "tightened")
                v = "REAL <0" if nv_s >= 20 and hi_s < 0 else ("REAL >0" if nv_s >= 20 and lo_s > 0 else "not distinguishable")
                print(f"      {tour} {split_label}-era: diff {obs_s:+.1%} (n_c={n_c}, n_t={n_t}) -> {v}")


def build_odds_lookup(matches):
    df = matches.copy()
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)
    long = pd.concat([
        df.assign(player=df["Player_1"], opponent=df["Player_2"], player_odd=df["Odd_1"], opponent_odd=df["Odd_2"]),
        df.assign(player=df["Player_2"], opponent=df["Player_1"], player_odd=df["Odd_2"], opponent_odd=df["Odd_1"]),
    ], ignore_index=True)
    long = long.rename(columns={"Date": "date", "Round": "round"})
    long = long[["edition_id", "date", "round", "player", "opponent", "player_odd", "opponent_odd"]]
    # defensive dedup on the join key - same rare edition_id-collision artifact documented in
    # former_elite_form_signal_test.add_opponent_recent_form (two distinct real tournaments sharing
    # one Tournament+year label); guards the merge below from a row-count blowup instead of crashing.
    return long.drop_duplicates(subset=["edition_id", "date", "round", "player", "opponent"])


def market_report(subset_label, df):
    n_total = len(df)
    valid = df["player_odd"].notna() & df["opponent_odd"].notna() & (df["player_odd"] > 1) & (df["opponent_odd"] > 1)
    n_valid = int(valid.sum())
    print(f"\n  {subset_label}: {n_total} real matches, {n_valid} with usable market odds "
          f"({n_valid / n_total:.1%} coverage)" if n_total else f"\n  {subset_label}: 0 matches")
    if n_total == 0:
        return
    by_decade = df.assign(decade=(df["date"].dt.year // 5) * 5, has_odds=valid)
    cov = by_decade.groupby("decade")["has_odds"].agg(["size", "mean"])
    print("  Coverage by 5-year period:")
    print(cov.rename(columns={"size": "n", "mean": "coverage"}).to_string(formatters={"coverage": "{:.1%}".format}))

    if n_valid < MIN_MARKET_ROWS_FOR_STATS:
        print(f"  Only {n_valid} priced matches - below the {MIN_MARKET_ROWS_FOR_STATS}-row floor for a "
              f"model-vs-market comparison. Reporting coverage only, no comparison forced.")
        return

    m = df[valid].copy()
    mp, _ = implied_probabilities(m["player_odd"].values, m["opponent_odd"].values)
    m["market_prob"] = mp
    m["market_loss"] = log_loss(m["actual_win"].values, m["market_prob"].values)
    m["model_loss"] = log_loss(m["actual_win"].values, m["pred_win"].values)
    print(f"  Mean market-implied win prob for the comeback player: {m['market_prob'].mean():.1%}")
    print(f"  Mean MODEL win prob for the comeback player:          {m['pred_win'].mean():.1%}")
    print(f"  Actual win rate:                                      {m['actual_win'].mean():.1%}")
    print(f"  Market log-loss: {m['market_loss'].mean():.4f}   Model log-loss: {m['model_loss'].mean():.4f}")
    obs, lo, hi = cluster_bootstrap_ci(m, "market_loss", "model_loss")
    verdict = ("model beats market here (CI excludes zero, >0)" if lo > 0 else
               ("market beats model here (CI excludes zero, <0)" if hi < 0 else "not distinguishable"))
    print(f"  Player-clustered bootstrap, mean(market_loss - model_loss): {obs:+.4f}, "
          f"95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")


def run():
    gap_threshold, min_history_years = FORMER_ELITE_GAP_THRESHOLD_DEFAULT, MIN_HISTORY_YEARS_DEFAULT
    run_levels, seqs, editions_by_tour, data_last_date_by_tour, matches_by_tour = [], [], {}, {}, {}
    for tour in ["ATP", "WTA"]:
        rl, seq, editions, last_date, matches = build_run_level_and_seq(tour, gap_threshold, min_history_years)
        editions_by_tour[tour] = editions
        data_last_date_by_tour[tour] = last_date
        matches_by_tour[tour] = matches
        print(f"{tour}: {len(rl)} total (player, edition) runs, data through {last_date.date()}")
        run_levels.append(rl)
        seqs.append(seq)
    all_run_level = pd.concat(run_levels, ignore_index=True)
    all_seq = pd.concat(seqs, ignore_index=True)
    idx = build_player_index(all_run_level)

    all_run_level = add_gradual_decline_flag(all_run_level)
    birthdate_by_name = load_birthdates(TML_BIRTHDATES_PATH)
    all_run_level = add_age(all_run_level, birthdate_by_name)

    base = all_run_level[(all_run_level["gap_below_peak"] >= gap_threshold)
                          & (all_run_level["years_of_history"] >= min_history_years)]
    comeback_pool = base[base["first_match_bucket"].isin(COMEBACK_BUCKETS)].copy()
    baseline_all = base[base["first_match_bucket"].isin(BASELINE_BUCKETS)].copy()
    baseline_age = baseline_all[(baseline_all["tour"] == "ATP") & (baseline_all["age_years"] >= AGE_THRESHOLD)].copy()
    baseline_gradual = baseline_all[baseline_all["is_gradual_decline"] == True].copy()  # noqa: E712

    print(f"\n{'=' * 90}\nPART A - TIGHTENED BASELINES\n{'=' * 90}")
    print(f"Original undifferentiated baseline (for reference): {len(baseline_all)} instances")
    compare_pools("A1: comeback vs. AGE >= 33 baseline (ATP only)",
                  comeback_pool[comeback_pool["tour"] == "ATP"], baseline_age, idx,
                  data_last_date_by_tour, editions_by_tour)
    compare_pools("A2: comeback vs. GRADUAL-DECLINE baseline (both tours, prior 2 editions also below-peak)",
                  comeback_pool, baseline_gradual, idx, data_last_date_by_tour, editions_by_tour)

    print(f"\n{'=' * 90}\nPART B - REAL MARKET ODDS DURING THE COMEBACK EDITION ITSELF\n{'=' * 90}")
    odds_by_tour = {t: build_odds_lookup(matches_by_tour[t]) for t in ["ATP", "WTA"]}
    for tour in ["ATP", "WTA"]:
        print(f"\n--- {tour} ---")
        tour_comeback_pool = comeback_pool[comeback_pool["tour"] == tour]
        tour_seq = all_seq[all_seq["tour"] == tour]
        comeback_matches = tour_seq.merge(tour_comeback_pool[["edition_id", "player"]],
                                           on=["edition_id", "player"], how="inner")
        n_before_odds_merge = len(comeback_matches)
        comeback_matches = comeback_matches.merge(
            odds_by_tour[tour], on=["edition_id", "date", "round", "player", "opponent"], how="left")
        assert len(comeback_matches) == n_before_odds_merge, "odds merge changed row count"
        market_report("ALL matches within the comeback edition", comeback_matches)
        market_report("FIRST match only (the actual return match)",
                       comeback_matches[comeback_matches["match_number"] == 1])


if __name__ == "__main__":
    run()
