from pathlib import Path

import kagglehub
import pandas as pd

ATP_DATASET = "dissfya/atp-tennis-2000-2023daily-pull"
WTA_DATASET = "dissfya/wta-tennis-2007-2023-daily-update"


def load_matches(tour: str = "ATP") -> pd.DataFrame:
    """Downloads the latest version of the Kaggle dataset (cached locally after
    first run - kagglehub only re-downloads when a newer version is published,
    so this stays current without manual re-downloading every time)."""
    dataset_id = ATP_DATASET if tour.upper() == "ATP" else WTA_DATASET
    download_path = Path(kagglehub.dataset_download(dataset_id))

    csv_files = list(download_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in downloaded dataset at {download_path}")

    # Odd_1/Odd_2 aren't cleanly numeric in the raw CSV (e.g. WTA has stray values like "5..5"
    # and "-"), which makes the C parser flip-flop between dtypes across chunks and raise a
    # DtypeWarning. Read them as strings (skips that inference entirely) and coerce to numeric
    # afterward so genuine garbage becomes NaN instead of silently staying an unusable string.
    df = pd.read_csv(csv_files[0], parse_dates=["Date"], dtype={"Odd_1": str, "Odd_2": str})
    df["Odd_1"] = pd.to_numeric(df["Odd_1"], errors="coerce")
    df["Odd_2"] = pd.to_numeric(df["Odd_2"], errors="coerce")
    return df


if __name__ == "__main__":
    df = load_matches("ATP")
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")