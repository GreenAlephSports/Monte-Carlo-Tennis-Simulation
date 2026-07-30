from functools import lru_cache
from pathlib import Path

import pandas as pd

from elo_ratings import expected_score

SURFACE_COLUMNS = {
    "Hard": "hard_elo",
    "Clay": "clay_elo",
    "Grass": "grass_elo",
}

ATP_RATINGS_PATH = Path(__file__).resolve().parent.parent / "output" / "player_elo_ratings_atp.csv"
WTA_RATINGS_PATH = Path(__file__).resolve().parent.parent / "output" / "player_elo_ratings_wta.csv"

# cache so we're not re-reading the csv on every single matchup lookup during a sim run
@lru_cache(maxsize=None)
def _load_ratings(ratings_path: Path) -> pd.DataFrame:
    return pd.read_csv(ratings_path).set_index("player")


def get_surface_elo(player: str, surface: str, ratings_path: Path) -> float:
    if surface not in SURFACE_COLUMNS:
        raise ValueError(f"Unsupported surface: {surface!r}. Expected one of {list(SURFACE_COLUMNS)}")

    ratings = _load_ratings(ratings_path)
    if player not in ratings.index:
        raise ValueError(f"Unknown player: {player!r}")

    return ratings.loc[player, SURFACE_COLUMNS[surface]]


# just pulls each player's surface-specific elo and updates ratings
def win_probability(player_a: str, player_b: str, surface: str, ratings_path: Path = ATP_RATINGS_PATH) -> float:
    elo_a = get_surface_elo(player_a, surface, ratings_path)
    elo_b = get_surface_elo(player_b, surface, ratings_path)
    return expected_score(elo_a, elo_b)


if __name__ == "__main__":
    p_a, p_b, surface = "Sinner J.", "Alcaraz C.", "Hard"
    prob = win_probability(p_a, p_b, surface)
    print(f"P({p_a} beats {p_b} on {surface}) = {prob:.3f}")
