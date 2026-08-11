from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "atp_tennis.csv"


# parse_dates so date is real timestamp
# date cutoff and loockback window later use this as date
def load_matches(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df



if __name__ == "__main__":
    df = load_matches()
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
