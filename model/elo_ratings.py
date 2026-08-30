import math
import sys
from collections import deque
from pathlib import Path

import pandas as pd

from data_loader import load_matches
from data_loader_kaggle import load_matches as load_matches_kaggle

ATP_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "atp_tennis.csv"
WTA_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "wta_tennis.csv"
ATP_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output" / "player_elo_ratings_atp.csv"
WTA_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output" / "player_elo_ratings_wta.csv"
TOUR_LOCAL_PATH = {"ATP": ATP_DATA_PATH, "WTA": WTA_DATA_PATH}


def load_matches_for_tour(tour: str) -> pd.DataFrame:
    """Live-by-default match history: pulls the current Kaggle dataset first (auto-updating,
    no manual re-download needed). If that fails for any reason - no internet, Kaggle auth
    expired, the API having an issue - falls back to the local CSV snapshot in data/ instead of
    crashing, since a live-data hiccup shouldn't take down a pipeline run mid-tournament."""
    tour = tour.upper()
    try:
        df = load_matches_kaggle(tour)
        print(f"Loaded {len(df)} live {tour} rows from Kaggle "
              f"({df['Date'].min().date()} to {df['Date'].max().date()})")
        return df
    except Exception as e:
        local_path = TOUR_LOCAL_PATH[tour]
        print(f"WARNING: live Kaggle fetch failed for {tour} ({type(e).__name__}: {e}) - "
              f"falling back to local snapshot {local_path}", file=sys.stderr)
        return load_matches(local_path)

SURFACES = ["Hard", "Clay", "Grass"]
STARTING_ELO = 1500
K_FACTOR = 32
# surface_elo is blended toward overall_elo when surface_matches is small - weight on surface_elo
# is surface_matches / (surface_matches + SURFACE_BLEND_K), so it's 0 at 0 matches, 50% at
# SURFACE_BLEND_K matches, and approaches (but never fully reaches) 100% as matches grow.
SURFACE_BLEND_K = 7
LOOKBACK_YEARS = 5
# window size for compute_recent_form_residuals - see win_probability.RECENT_FORM_BETA's docstring
# for the held-out validation this was fit against (model/research/recent_form_test.py). The
# window=15 robustness check did NOT hold up out-of-sample and is not used here.
RECENT_FORM_WINDOW = 10


# standard logistic Elo formula - 400 pt gap = ~91% win chance for the higher rated player to winn
def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def compute_recent_form_residuals(df: pd.DataFrame, cutoff_date, window: int = RECENT_FORM_WINDOW):
    """Per-player recent_form_residual as of cutoff_date: that player's own (actual win rate minus
    Elo-predicted win rate) averaged over their most recent `window` real matches strictly before
    cutoff_date. Absent from the returned dict until a player has at least `window` such matches -
    same "missing means no adjustment" convention win_probability.py already uses for current_rank/
    days_since_last_match.

    Deliberately mirrors model/research/recent_form_test.py's validated methodology exactly: a
    single continuously-updated, UNWINDOWED, overall-Elo-only replay across the FULL match history
    (df here must be the raw, unwindowed history - the caller passes it before apply_training_window
    runs - not production's windowed/surface-blended rating). The validated beta (win_probability.
    RECENT_FORM_BETA) was fit against this exact convention, not the 5yr-windowed one calculate_
    elo_ratings otherwise uses, so reusing calculate_elo_ratings' own windowed overall_elo here
    would silently apply a coefficient to a differently-distributed input than the one it was
    validated on."""
    cutoff_ts = pd.Timestamp(cutoff_date)
    df = df[df["Date"] < cutoff_ts].sort_values("Date", kind="stable")

    elo = {}
    history = {}  # player -> deque[(actual_win, pred_win)], maxlen=window
    for row in df.itertuples(index=False):
        p1, p2, winner = row.Player_1, row.Player_2, row.Winner
        e1 = elo.setdefault(p1, STARTING_ELO)
        e2 = elo.setdefault(p2, STARTING_ELO)
        pred1 = expected_score(e1, e2)
        win1 = 1.0 if winner == p1 else 0.0

        history.setdefault(p1, deque(maxlen=window)).append((win1, pred1))
        history.setdefault(p2, deque(maxlen=window)).append((1 - win1, 1 - pred1))

        elo[p1] = e1 + K_FACTOR * (win1 - pred1)
        elo[p2] = e2 + K_FACTOR * ((1 - win1) - (1 - pred1))

    residuals = {}
    for player, h in history.items():
        if len(h) < window:
            continue
        actual_mean = sum(a for a, _ in h) / window
        pred_mean = sum(p for _, p in h) / window
        residuals[player] = actual_mean - pred_mean
    return residuals


# track current form instead of dragging in ancient match history
# cutoff_date excludes matches on/after it, so ratings never peek into the future relative to the
# tournament being predicted - it comes from the bracket file's start_date, not a hardcoded constant
def apply_training_window(df: pd.DataFrame, cutoff_date) -> pd.DataFrame:
    cutoff_date = pd.Timestamp(cutoff_date)
    df = df[df["Date"] < cutoff_date]
    lookback_start = df["Date"].max() - pd.DateOffset(years=LOOKBACK_YEARS)
    return df[df["Date"] >= lookback_start]


# Empirically-fit recency-decay lookback variant ("decay3") - see model/research/
# decay3_full_historical_test.py's FINAL VERDICT: at full historical scale (both tours, ~2,800
# tournament editions), replacing the hard 5-year cutoff above with NO cutoff at all - full weight
# on any match within DECAY3_FULL_WEIGHT_YEARS of the training data's own most recent match,
# exponentially decaying with a DECAY3_HALF_LIFE_YEARS half-life beyond that - beats the hard-cutoff
# baseline: +0.0002 combined held-out log-loss improvement, 95% player-clustered bootstrap CI
# [+0.0000,+0.0003], no decade where it's worse.
#
# WTA-ONLY: the same full-scale test's per-tour-decade breakdown found this benefit is not uniform.
# ATP shows a real, individually-significant effect in 2005-2019 (as large as +0.0006 in a single
# decade) but is flat in 2020-2029 - and the held-out test window (most recent tournament editions)
# falls almost entirely in that flat recent era, so decay3 is NOT currently demonstrated for the
# ATP population any live prediction actually draws from. WTA's effect is the mirror image: flat
# through 2014, significant from 2015 on (+0.0008 in 2020-2024, the single largest decade effect in
# either tour) - i.e. current, not historical. Same tour-specific-gating precedent as win_
# probability.LAYOFF_BUCKET_EDGES_ATP/WTA (separately fit constants, applied only to their own
# tour) - here the gate is coarser (on/off, not separately-fit constants) since ATP's own
# equivalent constants are simply "unchanged production behavior," not a second fitted variant.
DECAY3_TOURS = {"WTA"}
DECAY3_FULL_WEIGHT_YEARS = 3.0
DECAY3_HALF_LIFE_YEARS = 2.0


def _decay3_weighted_window(df: pd.DataFrame, cutoff_date):
    """decay3's mechanism, ported unchanged from model/research/elo_lookback_test.py's validated
    calculate_elo_variant (lookback_years=None, decay_half_life_years=DECAY3_HALF_LIFE_YEARS,
    full_weight_years=DECAY3_FULL_WEIGHT_YEARS) - no hard cutoff, every match kept, weighted by
    recency instead. Returns (windowed_df, per-row weight Series aligned to it) rather than a
    DataFrame alone, since the weight multiplies K_FACTOR per-match rather than excluding rows."""
    cutoff_ts = pd.Timestamp(cutoff_date)
    windowed = df[df["Date"] < cutoff_ts]
    max_date = windowed["Date"].max()
    decay_rate = math.log(2) / DECAY3_HALF_LIFE_YEARS
    age_years = (max_date - windowed["Date"]).dt.days / 365.25
    weights = age_years.apply(
        lambda a: 1.0 if a <= DECAY3_FULL_WEIGHT_YEARS else math.exp(-decay_rate * (a - DECAY3_FULL_WEIGHT_YEARS))
    )
    return windowed, weights


def calculate_elo_ratings(df: pd.DataFrame, cutoff_date, tour: str = None):
    # computed on the RAW (unwindowed) df, before windowing below overwrites it - see
    # compute_recent_form_residuals' own docstring for why it needs the unwindowed history.
    recent_form = compute_recent_form_residuals(df, cutoff_date)

    if tour is not None and tour.upper() in DECAY3_TOURS:
        windowed, weights = _decay3_weighted_window(df, cutoff_date)
    else:
        windowed = apply_training_window(df, cutoff_date)
        weights = pd.Series(1.0, index=windowed.index)

    df = windowed.sort_values("Date", kind="stable")  # elo is path dependent, gotta process in date order
    weights = weights.reindex(df.index)

    overall_elo = {}
    surface_elo = {surface: {} for surface in SURFACES}
    surface_matches = {surface: {} for surface in SURFACES}
    # current ATP/WTA ranking as of the cutoff date, i.e. whatever each player's rank was in their
    # most recent training-window match - same no-lookahead rule as Elo itself. Only the live
    # Kaggle pull carries Rank_1/Rank_2 (the local fallback snapshot in data/ doesn't have both
    # columns for either tour - see load_matches_for_tour's docstring on why that fallback exists
    # at all); current_rank is just left empty (NaN for every player) when they're absent, so
    # anything reading it degrades to "no rank data" rather than erroring.
    current_rank = {}
    has_rank_columns = {"Rank_1", "Rank_2"}.issubset(df.columns)
    # last real match date per player, frozen at cutoff the same no-lookahead way current_rank is -
    # feeds win_probability.py's layoff adjustment (days since a player's last recorded match,
    # anywhere, before this tournament started). df is already sorted by Date above, so the last
    # write for a player during the loop below is their true most recent pre-cutoff match.
    last_match_date = {}
    cutoff_ts = pd.Timestamp(cutoff_date)

    for row, weight in zip(df.itertuples(index=False), weights):
        p1, p2, winner, surface = row.Player_1, row.Player_2, row.Winner, row.Surface
        last_match_date[p1] = row.Date
        last_match_date[p2] = row.Date
        k_eff = K_FACTOR * weight

        overall_elo.setdefault(p1, STARTING_ELO)
        overall_elo.setdefault(p2, STARTING_ELO)
        score_p1 = 1.0 if winner == p1 else 0.0
        expected_p1 = expected_score(overall_elo[p1], overall_elo[p2])
        overall_elo[p1] += k_eff * (score_p1 - expected_p1)
        overall_elo[p2] += k_eff * ((1 - score_p1) - (1 - expected_p1))

        if surface in SURFACES:
            ratings = surface_elo[surface]
            counts = surface_matches[surface]
            ratings.setdefault(p1, STARTING_ELO)
            ratings.setdefault(p2, STARTING_ELO)
            expected_p1_surface = expected_score(ratings[p1], ratings[p2])
            ratings[p1] += k_eff * (score_p1 - expected_p1_surface)
            ratings[p2] += k_eff * ((1 - score_p1) - (1 - expected_p1_surface))
            counts[p1] = counts.get(p1, 0) + 1
            counts[p2] = counts.get(p2, 0) + 1

        if has_rank_columns:
            if row.Rank_1 > 0:
                current_rank[p1] = row.Rank_1
            if row.Rank_2 > 0:
                current_rank[p2] = row.Rank_2

    players = sorted(overall_elo.keys())
    records = []
    for player in players:
        last_date = last_match_date.get(player)
        record = {
            "player": player,
            "overall_elo": overall_elo[player],
            "current_rank": current_rank.get(player),
            "days_since_last_match": (cutoff_ts - last_date).days if last_date is not None else None,
            "recent_form_residual": recent_form.get(player),
        }
        # blend surface_elo toward overall_elo, weighted by how much surface-specific sample size
        # backs it up - a player with few surface matches leans on the more-data-backed overall
        # rating; one with a deep surface history gets (most of) their own surface rating instead
        # of a hard cutoff snapping between the two.
        for surface in SURFACES:
            match_count = surface_matches[surface].get(player, 0)
            raw_elo = surface_elo[surface].get(player, STARTING_ELO)
            surface_weight = match_count / (match_count + SURFACE_BLEND_K)
            final_elo = surface_weight * raw_elo + (1 - surface_weight) * overall_elo[player]
            record[f"{surface.lower()}_elo"] = final_elo
            record[f"{surface.lower()}_matches"] = match_count
        records.append(record)

    columns = [
        "player",
        "hard_elo",
        "clay_elo",
        "grass_elo",
        "overall_elo",
        "hard_matches",
        "clay_matches",
        "grass_matches",
        "current_rank",
        "days_since_last_match",
        "recent_form_residual",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Recalculate Elo ratings from match history up to a cutoff date."
    )
    parser.add_argument(
        "--cutoff-date", required=True,
        help="Exclude matches on/after this date (YYYY-MM-DD). Normally the target tournament's start_date.",
    )
    args = parser.parse_args()
    cutoff_date = pd.Timestamp(args.cutoff_date)

    for tour, output_path in [("ATP", ATP_OUTPUT_PATH), ("WTA", WTA_OUTPUT_PATH)]:
        matches = load_matches_for_tour(tour)
        ratings = calculate_elo_ratings(matches, cutoff_date, tour=tour)
        ratings = ratings.sort_values("overall_elo", ascending=False).reset_index(drop=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ratings.to_csv(output_path, index=False)
        print(f"Saved {len(ratings)} player ratings to {output_path}")

        print("\nTop 15 players by overall Elo:")
        print(ratings.head(15).to_string(index=False))

