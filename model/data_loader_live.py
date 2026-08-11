from pathlib import Path
from datetime import datetime

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
LOOKBACK_BUFFER_YEARS = 6  # a bit more than the 5yr window so nothing gets cut short
LOCAL_ODDS_PATH = Path(__file__).resolve().parent.parent / "data" / "atp_tennis.csv"


def _fetch_year(year: int) -> pd.DataFrame:
    url = f"{BASE_URL}/atp_matches_{year}.csv"
    return pd.read_csv(url)


def _rename_to_pipeline_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Sackmann's format is winner/loser based (winner_name, loser_name, tourney_date
    as YYYYMMDD int, tourney_name, surface, round) - remaps to this pipeline's
    Player_1/Player_2/Winner/Date/Tournament/Surface/Round schema."""
    df = df.rename(columns={
        "winner_name": "Player_1",
        "loser_name": "Player_2",
        "tourney_name": "Tournament",
        "surface": "Surface",
        "round": "Round",
        "winner_rank": "Rank_1",
        "loser_rank": "Rank_2",
    })
    df["Winner"] = df["Player_1"]  # Player_1 is always the winner in this source's raw format
    df["Date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    # Sackmann's data has no betting odds - these live matches won't have real
    # odds until/unless merged with a separate odds source (see ev_comparison.py note)
    df["Odd_1"] = -1.0
    df["Odd_2"] = -1.0
    return df[["Date", "Tournament", "Surface", "Round", "Player_1", "Player_2",
               "Winner", "Rank_1", "Rank_2", "Odd_1", "Odd_2"]]


def load_matches_live(lookback_years: int = LOOKBACK_BUFFER_YEARS) -> pd.DataFrame:
    """Pulls fresh match data directly from Sackmann's GitHub repo every call -
    always current, no manual re-download needed. Falls back to the local Kaggle
    snapshot (data/atp_tennis.csv) if the live fetch fails (e.g. no internet)."""
    current_year = datetime.now().year
    years = range(current_year - lookback_years, current_year + 1)

    frames = []
    failed_years = []
    for year in years:
        try:
            frames.append(_fetch_year(year))
        except Exception as e:
            failed_years.append((year, str(e)))

    if not frames:
        print("WARNING: live fetch failed for all years, falling back to local snapshot")
        return pd.read_csv(LOCAL_ODDS_PATH, parse_dates=["Date"])

    if failed_years:
        print(f"WARNING: could not fetch {len(failed_years)} year(s) live: "
              f"{[y for y, _ in failed_years]} - continuing with available years")

    combined = pd.concat(frames, ignore_index=True)
    return _rename_to_pipeline_schema(combined)


if __name__ == "__main__":
    df = load_matches_live()
    print(f"Loaded {len(df)} live rows spanning {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(df.head())