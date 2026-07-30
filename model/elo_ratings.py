from pathlib import Path

import pandas as pd

from data_loader import load_matches

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output" / "player_elo_ratings.csv"

SURFACES = ["Hard", "Clay", "Grass"]
STARTING_ELO = 1500
K_FACTOR = 32
SURFACE_FALLBACK_THRESHOLD = 10
LOOKBACK_YEARS = 5
#Doeasnt include ratings past the wimbeldon start date. Doesnt allow for peeking into the future when calculating ratings
PREDICTION_CUTOFF = pd.Timestamp("2026-06-29")


# standard logistic Elo formula - 400 pt gap = ~91% win chance for the higher rated player to winn
def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


#track current form instead of dragging in ancient match history
def apply_training_window(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Date"] < PREDICTION_CUTOFF]
    lookback_start = df["Date"].max() - pd.DateOffset(years=LOOKBACK_YEARS)
    return df[df["Date"] >= lookback_start]


def calculate_elo_ratings(df: pd.DataFrame):
    df = apply_training_window(df)
    df = df.sort_values("Date", kind="stable")  # elo is path dependent, gotta process in date order

    overall_elo = {}
    surface_elo = {surface: {} for surface in SURFACES}
    surface_matches = {surface: {} for surface in SURFACES}

    for row in df.itertuples(index=False):
        p1, p2, winner, surface = row.Player_1, row.Player_2, row.Winner, row.Surface

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

    players = sorted(overall_elo.keys())
    records = []
    for player in players:
        record = {
            "player": player,
            "overall_elo": overall_elo[player],
        }
        # if a player hasn't played enough matches on a surface, the elo calculated for that surface isnt used
        # resorts to just ovr elo for greater sample size
        for surface in SURFACES:
            match_count = surface_matches[surface].get(player, 0)
            raw_elo = surface_elo[surface].get(player, STARTING_ELO)
            final_elo = raw_elo if match_count >= SURFACE_FALLBACK_THRESHOLD else overall_elo[player]
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
    ]
    return pd.DataFrame.from_records(records, columns=columns)


if __name__ == "__main__":
    matches = load_matches()
    ratings = calculate_elo_ratings(matches)
    ratings = ratings.sort_values("overall_elo", ascending=False).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(ratings)} player ratings to {OUTPUT_PATH}")

    print("\nTop 15 players by overall Elo:")
    print(ratings.head(15).to_string(index=False))
