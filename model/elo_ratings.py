import sys
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


# standard logistic Elo formula - 400 pt gap = ~91% win chance for the higher rated player to winn
def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


# track current form instead of dragging in ancient match history
# cutoff_date excludes matches on/after it, so ratings never peek into the future relative to the
# tournament being predicted - it comes from the bracket file's start_date, not a hardcoded constant
def apply_training_window(df: pd.DataFrame, cutoff_date) -> pd.DataFrame:
    cutoff_date = pd.Timestamp(cutoff_date)
    df = df[df["Date"] < cutoff_date]
    lookback_start = df["Date"].max() - pd.DateOffset(years=LOOKBACK_YEARS)
    return df[df["Date"] >= lookback_start]


def calculate_elo_ratings(df: pd.DataFrame, cutoff_date):
    df = apply_training_window(df, cutoff_date)
    df = df.sort_values("Date", kind="stable")  # elo is path dependent, gotta process in date order

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

    for row in df.itertuples(index=False):
        p1, p2, winner, surface = row.Player_1, row.Player_2, row.Winner, row.Surface
        last_match_date[p1] = row.Date
        last_match_date[p2] = row.Date

        overall_elo.setdefault(p1, STARTING_ELO)
        overall_elo.setdefault(p2, STARTING_ELO)
        score_p1 = 1.0 if winner == p1 else 0.0
        expected_p1 = expected_score(overall_elo[p1], overall_elo[p2])
        overall_elo[p1] += K_FACTOR * (score_p1 - expected_p1)
        overall_elo[p2] += K_FACTOR * ((1 - score_p1) - (1 - expected_p1))

        if surface in SURFACES:
            ratings = surface_elo[surface]
            counts = surface_matches[surface]
            ratings.setdefault(p1, STARTING_ELO)
            ratings.setdefault(p2, STARTING_ELO)
            expected_p1_surface = expected_score(ratings[p1], ratings[p2])
            ratings[p1] += K_FACTOR * (score_p1 - expected_p1_surface)
            ratings[p2] += K_FACTOR * ((1 - score_p1) - (1 - expected_p1_surface))
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
        ratings = calculate_elo_ratings(matches, cutoff_date)
        ratings = ratings.sort_values("overall_elo", ascending=False).reset_index(drop=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ratings.to_csv(output_path, index=False)
        print(f"Saved {len(ratings)} player ratings to {output_path}")

        print("\nTop 15 players by overall Elo:")
        print(ratings.head(15).to_string(index=False))

