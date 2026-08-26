"""Tests whether a shorter or recency-weighted Elo lookback window calibrates better than the
current production 5-year hard cutoff (elo_ratings.LOOKBACK_YEARS), across MULTIPLE real,
now-fully-concluded 2026 ATP hard-court tournaments, not just one - Cincinnati ("Western &
Southern Financial Group Masters") and the Canadian Open.

Data-reality note: the live ATP Kaggle feed carries exactly ONE "Canadian Open" row per year, not
separate "Montreal" and "Toronto" entries - the men's and women's draws alternate host city each
year but there is only ever one ATP Canadian Open per season. So this is a genuine two-tournament
combined check (Cincinnati n=92 + Canadian Open n=91 real matches), not three - reported honestly
rather than silently padding this out or inventing a third field.

Each tournament's three Elo variants are frozen at THAT tournament's own real start date (its
no-lookahead cutoff - Cincinnati 2026-08-13, Canadian Open 2026-08-02), never at a shared date,
since Canadian Open predates Cincinnati and using Cincinnati's cutoff for it would leak Canadian
Open's own results into its own pre-match Elo. All three variants feed the SAME downstream
win_probability() pipeline (rank-adjustment, layoff-adjustment, confidence-calibration all left
exactly as in production) - only the Elo table itself differs, isolating the lookback-window
choice as the one variable under test:

  A. baseline    - production elo_ratings.calculate_elo_ratings unchanged (5yr hard cutoff)
  B. hard3       - identical mechanism, LOOKBACK_YEARS=3 instead of 5
  C. decay3      - no hard cutoff at all (full available match history, back to the dataset's
                   start): each match's K-factor is scaled by a recency weight - 1.0 (full
                   weight) for anything within 3 years of the training window's own most recent
                   match, decaying with a 2-year half-life beyond that. Weight is computed once
                   per match relative to that fixed reference date, not updated match-by-match,
                   so it stays a well-defined static snapshot like A and B are.

Calibration methodology mirrors backtest_hard_court.py exactly: for every real, decided Cincinnati
match, does the model's pre-match favorite (via the SAME win_probability() every simulated match
uses) actually win it - reported as favorite-win-rate vs. average-assigned-probability, plus
log-loss/Brier (backtest_hard_court.py doesn't compute these, but every other test tonight does,
and a single tournament's ~90 matches is small enough that win-rate alone is noisy - log-loss adds
a sharper, per-match-weighted read of the same question).

Veteran-specific check: does variant C (recency-weighted) score real veteran-elite players'
matches more accurately than baseline WITHOUT an explicit age variable - i.e. does it recover the
age-decline signal from veteran_decline_test.py implicitly, just by weighting old form less?
"Veteran" here is determined fresh, from real birthdates (same TML-Database join as
veteran_decline_test.py) applied to whoever actually played Cincinnati 2026, not the stale
2000-2025 training-era veteran list from that earlier test (which is dominated by long-retired
names - Agassi, Federer, Nadal - who obviously aren't in a real 2026 draw).

Usage:
    python model/research/elo_lookback_test.py
"""
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import match_name_to_pool  # noqa: E402
from elite_opponent_residual_test import EPS, log_loss  # noqa: E402
from elo_ratings import K_FACTOR, STARTING_ELO, SURFACES, SURFACE_BLEND_K, expected_score, load_matches_for_tour  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from veteran_decline_test import load_birthdates  # noqa: E402
from win_probability import win_probability  # noqa: E402

SCRATCH_DIR = Path(
    r"C:\Users\idanh\AppData\Local\Temp\claude\x--idanh-Documents-VS-Code-Projects-Monte-Carlo-Simulation-Grand-Slam-Model-"
    r"\54b74d84-4643-4e25-9b66-35e4496bc57b\scratchpad"
)
AGE_THRESHOLD = 33
# (display_label, Kaggle Tournament name, real 2026 start date used as the no-lookahead cutoff)
TOURNAMENTS = [
    ("Cincinnati", "Western & Southern Financial Group Masters", pd.Timestamp("2026-08-13")),
    ("Canadian Open", "Canadian Open", pd.Timestamp("2026-08-02")),
]
ELO_COLUMNS = [
    "player", "hard_elo", "clay_elo", "grass_elo", "overall_elo",
    "hard_matches", "clay_matches", "grass_matches", "current_rank", "days_since_last_match",
]


def calculate_elo_variant(df, cutoff_date, lookback_years, decay_half_life_years=None, full_weight_years=3.0):
    """Same mechanism as elo_ratings.calculate_elo_ratings (surface-Elo shrinkage blending
    unchanged), parameterized by lookback_years (a hard cutoff, matching the production
    apply_training_window shape when decay_half_life_years is None) or, when decay_half_life_years
    is given, a per-match recency weight (full weight inside full_weight_years, exponential decay
    with that half-life beyond it) applied to K_FACTOR instead of a hard exclusion - lookback_years
    is then just how far back rows are even loaded before weighting (None = the whole dataset)."""
    cutoff_ts = pd.Timestamp(cutoff_date)
    df = df[df["Date"] < cutoff_ts]
    max_date = df["Date"].max()

    if lookback_years is not None:
        lookback_start = max_date - pd.DateOffset(years=lookback_years)
        df = df[df["Date"] >= lookback_start]

    df = df.sort_values("Date", kind="stable")

    if decay_half_life_years is not None:
        decay_rate = math.log(2) / decay_half_life_years
        age_years = (max_date - df["Date"]).dt.days / 365.25
        weights = age_years.apply(lambda a: 1.0 if a <= full_weight_years else math.exp(-decay_rate * (a - full_weight_years)))
    else:
        weights = pd.Series(1.0, index=df.index)

    overall_elo, surface_elo, surface_matches, current_rank, last_match_date = {}, {s: {} for s in SURFACES}, {s: {} for s in SURFACES}, {}, {}
    has_rank_columns = {"Rank_1", "Rank_2"}.issubset(df.columns)

    for row, w in zip(df.itertuples(index=False), weights):
        p1, p2, winner, surface = row.Player_1, row.Player_2, row.Winner, row.Surface
        last_match_date[p1] = row.Date
        last_match_date[p2] = row.Date
        k_eff = K_FACTOR * w

        overall_elo.setdefault(p1, STARTING_ELO)
        overall_elo.setdefault(p2, STARTING_ELO)
        score_p1 = 1.0 if winner == p1 else 0.0
        expected_p1 = expected_score(overall_elo[p1], overall_elo[p2])
        overall_elo[p1] += k_eff * (score_p1 - expected_p1)
        overall_elo[p2] += k_eff * ((1 - score_p1) - (1 - expected_p1))

        if surface in SURFACES:
            ratings, counts = surface_elo[surface], surface_matches[surface]
            ratings.setdefault(p1, STARTING_ELO)
            ratings.setdefault(p2, STARTING_ELO)
            expected_p1_s = expected_score(ratings[p1], ratings[p2])
            ratings[p1] += k_eff * (score_p1 - expected_p1_s)
            ratings[p2] += k_eff * ((1 - score_p1) - (1 - expected_p1_s))
            counts[p1] = counts.get(p1, 0) + 1
            counts[p2] = counts.get(p2, 0) + 1

        if has_rank_columns:
            if row.Rank_1 > 0:
                current_rank[p1] = row.Rank_1
            if row.Rank_2 > 0:
                current_rank[p2] = row.Rank_2

    records = []
    for player in sorted(overall_elo.keys()):
        last_date = last_match_date.get(player)
        record = {
            "player": player, "overall_elo": overall_elo[player], "current_rank": current_rank.get(player),
            "days_since_last_match": (cutoff_ts - last_date).days if last_date is not None else None,
        }
        for surface in SURFACES:
            match_count = surface_matches[surface].get(player, 0)
            raw_elo = surface_elo[surface].get(player, STARTING_ELO)
            surface_weight = match_count / (match_count + SURFACE_BLEND_K)
            record[f"{surface.lower()}_elo"] = surface_weight * raw_elo + (1 - surface_weight) * overall_elo[player]
            record[f"{surface.lower()}_matches"] = match_count
        records.append(record)
    return pd.DataFrame.from_records(records, columns=ELO_COLUMNS)


def build_real_tournament_matches(matches_df, kaggle_name, cutoff):
    window = matches_df[
        (matches_df["Tournament"] == kaggle_name)
        & (matches_df["Date"] >= cutoff - pd.Timedelta(days=2))
        & (matches_df["Date"] < cutoff + pd.Timedelta(days=21))
    ]
    return window[["Date", "Round", "Player_1", "Player_2", "Winner"]].copy()


def calibrate_variant(label, ratings_df, real_matches, ratings_path, surface="Hard"):
    # win_probability._load_ratings is @lru_cache'd on ratings_path - writing a new CSV to a path
    # already read once would silently keep serving the FIRST variant's ratings forever (confirmed:
    # all three variants scored bit-identical until this was caught). Each variant gets its own
    # path instead of clearing a shared production-module cache mid-run.
    ratings_df.to_csv(ratings_path, index=False)
    rows = []
    n_skipped = 0
    for row in real_matches.itertuples(index=False):
        a, b, winner = row.Player_1, row.Player_2, row.Winner
        try:
            prob_a = win_probability(a, b, surface, ratings_path)
        except ValueError:
            n_skipped += 1  # player not in this variant's ratings table at all (e.g. hasn't
            continue        # played within a shortened window) - skip and report, don't guess
        favorite = a if prob_a >= 0.5 else b
        favorite_prob = max(prob_a, 1 - prob_a)
        rows.append({
            "player_a": a, "player_b": b, "winner": winner, "favorite": favorite,
            "favorite_prob": favorite_prob, "favorite_won": favorite == winner,
            "prob_winner": prob_a if winner == a else 1 - prob_a,
        })
    calib = pd.DataFrame(rows)
    calib["prob_a"] = calib.apply(
        lambda r: win_probability(r["player_a"], r["player_b"], surface, ratings_path), axis=1)
    calib["log_loss"] = log_loss((calib["winner"] == calib["player_a"]).astype(int).values, calib["prob_a"].values)
    calib["brier"] = (calib["prob_winner"].apply(lambda p: 1 - p)) ** 2  # (1 - P(actual winner))^2, symmetric Brier
    skip_note = f" ({n_skipped} skipped - a player absent from this variant's ratings table entirely)" if n_skipped else ""
    print(f"\n{label}: {len(calib)} real Cincinnati matches scored{skip_note}")
    print(f"  favorite win rate      : {calib['favorite_won'].mean():.1%}")
    print(f"  avg favorite prob      : {calib['favorite_prob'].mean():.1%}")
    print(f"  mean log-loss          : {calib['log_loss'].mean():.4f}")
    print(f"  mean Brier             : {calib['brier'].mean():.4f}")

    # long (player-perspective) format - one row per (match, player), same shape
    # elite_opponent_residual_test.build_frozen_predictions uses - lets cluster_bootstrap_ci
    # (already player-clustered, exactly the held-out rigor every other test tonight used) be
    # reused as-is instead of reinvented for this per-match table.
    calib = calib.reset_index(drop=True)
    calib["match_id"] = calib.index  # stable across variants (same real_matches, same iteration order) - lets a
                                      # combined multi-tournament merge line up the SAME real match across variants
                                      # instead of colliding on repeated player-name pairs across different events
    long_df = pd.concat([
        calib[["match_id", "player_a", "player_b"]].rename(columns={"player_a": "player", "player_b": "opponent"})
        .assign(pred_win=calib["prob_a"], actual_win=(calib["winner"] == calib["player_a"]).astype(int)),
        calib[["match_id", "player_b", "player_a"]].rename(columns={"player_b": "player", "player_a": "opponent"})
        .assign(pred_win=1 - calib["prob_a"], actual_win=(calib["winner"] == calib["player_b"]).astype(int)),
    ], ignore_index=True)
    long_df["loss"] = log_loss(long_df["actual_win"].values, long_df["pred_win"].values)
    return calib, long_df


def _bootstrap_verdict(long_baseline, long_variant, merge_keys=("tournament", "match_id", "player")):
    merged = long_baseline[[*merge_keys, "loss"]].merge(
        long_variant[[*merge_keys, "loss"]], on=list(merge_keys), suffixes=("_baseline", "_variant"))
    observed, lo, hi = cluster_bootstrap_ci(merged, "loss_baseline", "loss_variant", group_col="player")
    verdict = "BEATS baseline (CI excludes zero, >0)" if lo > 0 else (
        "WORSE than baseline (CI excludes zero, <0)" if hi < 0 else "NOT distinguishable from baseline (CI straddles zero)")
    return merged, observed, lo, hi, verdict


def run():
    matches = load_matches_for_tour("ATP")

    variant_specs = {
        "A. baseline (5yr hard cutoff, production)": dict(lookback_years=5),
        "B. hard3 (3yr hard cutoff)": dict(lookback_years=3),
        "C. decay3 (full weight <=3yr, 2yr half-life decay beyond, no hard cutoff)":
            dict(lookback_years=None, decay_half_life_years=2.0, full_weight_years=3.0),
    }
    baseline_label = "A. baseline (5yr hard cutoff, production)"

    per_tournament_longs = {label: [] for label in variant_specs}
    real_matches_by_tournament = {}
    for t_label, kaggle_name, cutoff in TOURNAMENTS:
        real_matches = build_real_tournament_matches(matches, kaggle_name, cutoff)
        real_matches_by_tournament[t_label] = real_matches
        print(f"\n{'#' * 90}\n{t_label} 2026 (ATP): {len(real_matches)} real matches found "
              f"(cutoff {cutoff.date()}, no-lookahead)\n{'#' * 90}")
        for i, (label, spec) in enumerate(variant_specs.items()):
            ratings_df = calculate_elo_variant(matches, cutoff, **spec)
            variant_path = SCRATCH_DIR / f"elo_lookback_variant_{t_label.replace(' ', '_')}_{i}.csv"
            calib, long_df = calibrate_variant(f"{t_label} | {label}", ratings_df, real_matches, variant_path)
            long_df["tournament"] = t_label
            per_tournament_longs[label].append(long_df)

    longs = {label: pd.concat(dfs, ignore_index=True) for label, dfs in per_tournament_longs.items()}
    n_total_matches = sum(len(v) for v in real_matches_by_tournament.values())
    n_total_rows = len(longs[baseline_label])

    # --- combined, held-out rigor: player-clustered bootstrap CI across BOTH tournaments pooled,
    # merged on (tournament, match_id, player) so real matches from different tournaments never
    # collide even when the same two player names happen to recur.
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

    # --- per-tournament breakdown: does the combined result hold in EACH tournament separately,
    # or is it dragged one way by a single event? (the user's specific ask: "or whether it was
    # specific to Cincinnati's particular field")
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

    # --- age check: real players actually in EITHER draw, ages determined fresh from real
    # birthdates (not the stale training-era list from the earlier age test)
    print(f"\n{'=' * 90}\nAGE BREAKDOWN (age >= {AGE_THRESHOLD} = veteran, else prime), pooled real "
          f"participants across {' + '.join(t for t, _, _ in TOURNAMENTS)}\n{'=' * 90}")
    birthdate_by_name = load_birthdates()
    pool_names = list(birthdate_by_name.index)
    participants = sorted(set().union(*[
        set(rm["Player_1"]) | set(rm["Player_2"]) for rm in real_matches_by_tournament.values()
    ]))
    # each tournament uses its own cutoff for age-as-of-that-event, not a single shared date
    cutoff_by_tournament = {t_label: cutoff for t_label, _, cutoff in TOURNAMENTS}
    resolved = {p: match_name_to_pool(p, pool_names) for p in participants}
    # age is computed as-of the LATER tournament's cutoff (Cincinnati) when a player appears in
    # both events - the ~11-day gap between the two cutoffs never flips a real player across the
    # age>=33 boundary, so a single shared reference date is fine here rather than tracking age
    # separately per tournament per player.
    age_reference_cutoff = max(cutoff for _, _, cutoff in TOURNAMENTS)
    age_by_player = {}
    for p, resolved_name in resolved.items():
        if resolved_name is None:
            continue
        bd = birthdate_by_name.get(resolved_name)
        if pd.isna(bd):
            continue
        age_by_player[p] = (age_reference_cutoff - bd).days / 365.25

    veterans = sorted(p for p, age in age_by_player.items() if age >= AGE_THRESHOLD)
    prime = sorted(p for p, age in age_by_player.items() if age < AGE_THRESHOLD)
    print(f"Real participants (pooled, both tournaments) resolved to an age: {len(age_by_player)}/{len(participants)}")
    print(f"Veterans (age >= {AGE_THRESHOLD}, n={len(veterans)}): "
          f"{[(p, round(age_by_player[p], 1)) for p in veterans]}")
    print(f"Prime-age (age < {AGE_THRESHOLD}, n={len(prime)}): sample = "
          f"{[(p, round(age_by_player[p], 1)) for p in prime[:8]]} ...")

    if not veterans:
        print("No veteran-age players found in this draw - cannot run the age breakdown.")
        return

    for group_label, group in [("VETERAN (age >= 33)", veterans), ("PRIME-AGE (age < 33)", prime)]:
        print(f"\n--- {group_label} ---")
        for label, long_df in longs.items():
            group_rows = long_df[long_df["player"].isin(group)]
            if len(group_rows) == 0:
                continue
            n = len(group_rows)
            assigned = group_rows["pred_win"].mean()
            actual = group_rows["actual_win"].mean()
            mean_loss = group_rows["loss"].mean()
            print(f"  {label}: n={n} player-perspective rows | assigned P(win)={assigned:.1%} | "
                  f"actual win rate={actual:.1%} | gap={assigned - actual:+.1%} | mean log-loss={mean_loss:.4f}")

        # bootstrap CI within this age group specifically, same method as the overall check above
        for label, long_df in longs.items():
            if label == baseline_label:
                continue
            base_group = longs[baseline_label][longs[baseline_label]["player"].isin(group)]
            var_group = long_df[long_df["player"].isin(group)]
            merged = base_group[["tournament", "match_id", "player", "loss"]].merge(
                var_group[["tournament", "match_id", "player", "loss"]],
                on=["tournament", "match_id", "player"], suffixes=("_baseline", "_variant"))
            if len(merged) < 10:
                print(f"  {label} vs. baseline ({group_label}): only {len(merged)} rows - too few to bootstrap")
                continue
            observed, lo, hi = cluster_bootstrap_ci(merged, "loss_baseline", "loss_variant", group_col="player")
            verdict = "BEATS baseline" if lo > 0 else ("WORSE than baseline" if hi < 0 else "not distinguishable")
            print(f"  {label} vs. baseline ({group_label}): {len(merged)} rows, improvement "
                  f"{observed:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")


if __name__ == "__main__":
    run()
