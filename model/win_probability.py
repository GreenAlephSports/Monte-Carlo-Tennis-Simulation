import math
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

# Empirically-fit ranking-gap calibration adjustment - see the backtest that derived it: among
# 17,955 historical ATP matches with |Elo diff| <= 50 (i.e. Elo calls them a near-coin-flip),
# current ATP ranking predicted a real, monotonic excess win rate for the better-ranked player that
# Elo alone missed (up to +15.7pp for a 100+ rank-gap bucket, holding up across every Elo-window and
# rank-range sensitivity check tried). This shifted-log form (vs. plain log-linear or a saturating
# sigmoid) was chosen because it's the only one of the three that (a) is exactly zero at rank_gap=0
# by construction, not as a fit accident - two equally-ranked players must get zero adjustment - and
# (b) had the best held-out log-likelihood on a tournament-level 80/20 split. Fitted on ATP data
# only; not validated for WTA.
RANK_ADJUSTMENT_C = 1.0629
RANK_ADJUSTMENT_D = 260.72
# The adjustment is ONLY validated for matches Elo itself calls a near-coin-flip - applying it
# outside that window is extrapolation with zero empirical backing. Confirmed this isn't a
# theoretical nitpick: applying it unconditionally to real Montreal/Toronto matches (most of which
# have |Elo diff| well over 50 - median ~80-120 points there) made calibration measurably WORSE, not
# better, because it piled extra confidence onto favorites Elo was already correctly picking, without
# improving accuracy at all (identical favorite-win-rate, higher assigned confidence). Gating to the
# validated window is what fixed that.
RANK_ADJUSTMENT_ELO_WINDOW = 50

# Empirically-fit confidence-calibration correction (Platt scaling) - see the backtest that derived
# it: among 133,552 historical ATP player-match observations (2000-2026, frozen-per-tournament-
# edition Elo - the same dataset the rank-gap adjustment above was fit on), the model's most
# confident picks (raw predicted >= 0.78) beat their actual outcome by ~3.2pp on held-out
# tournaments (z=-5.20, p<0.0001) - real overconfidence, not noise. This one-parameter shrinkage
# (calibrated = sigmoid(PLATT_B * logit(raw))) cut that held-out gap to ~1.5pp (z=-2.32, p=0.02).
# The intercept is fixed at exactly 0, not fit loosely near it - the training data's mirrored-
# observation structure (every match contributes both (p, y) and (1-p, 1-y)) makes a=0 a
# mathematical guarantee, not an empirical accident: it's the only way a 50% prediction (no
# information to correct) stays at 50% after calibration.
#
# Composes with the rank-gap adjustment in the validated order - rank-gap FIRST, this SECOND:
# confirmed empirically that this correction's effect inside the rank-gap's own |Elo diff|<=50
# operating window is negligible (~30-60x smaller than the rank-gap shift itself there), so the two
# don't fight each other; applying this after rank-gap leaves the rank-gap correction's own
# calibration on the near-coin-flip population intact (still ~0.1pp off actual, statistically
# indistinguishable from zero) while still fixing the separate, unrelated overconfidence-in-the-
# tails problem.
PLATT_B = 0.9205


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


def get_current_rank(player: str, ratings_path: Path):
    """Current ranking as of the ratings file's cutoff date, or None if unknown - either because
    this player never appeared with a valid Rank_1/Rank_2 in the training window (e.g. a brand-new
    tier-3 placeholder - see bracket.match_draw_to_ratings), or because the ratings file was built
    from the local Kaggle fallback snapshot, which doesn't carry ranking columns at all (see
    calculate_elo_ratings' current_rank comment)."""
    ratings = _load_ratings(ratings_path)
    if player not in ratings.index or "current_rank" not in ratings.columns:
        return None
    rank = ratings.loc[player, "current_rank"]
    return None if pd.isna(rank) else rank


def _apply_rank_adjustment(prob_a: float, rank_a, rank_b) -> float:
    """Additive correction in logit space: adjusted_logit = logit(prob_a) + sign * C *
    ln(1 + rank_gap/D), favoring whichever player has the better (numerically lower) current
    ranking. A no-op whenever either player's current rank is unknown, or they're tied."""
    if rank_a is None or rank_b is None or pd.isna(rank_a) or pd.isna(rank_b) or rank_a == rank_b:
        return prob_a

    gap = abs(rank_a - rank_b)
    shift = RANK_ADJUSTMENT_C * math.log1p(gap / RANK_ADJUSTMENT_D)
    sign = 1.0 if rank_a < rank_b else -1.0  # lower rank number = better ranked = favored

    logit_p = math.log(prob_a / (1 - prob_a))
    return 1 / (1 + math.exp(-(logit_p + sign * shift)))


def _apply_confidence_calibration(prob_a: float) -> float:
    """Platt-scaling shrinkage toward 50/50, proportional to how confident the (possibly already
    rank-adjusted) prediction already is. See PLATT_B's docstring above for where this comes from."""
    logit_p = math.log(prob_a / (1 - prob_a))
    return 1 / (1 + math.exp(-(PLATT_B * logit_p)))


# just pulls each player's surface-specific elo and updates ratings
def win_probability(
    player_a: str, player_b: str, surface: str, ratings_path: Path = ATP_RATINGS_PATH,
    use_rank_adjustment: bool = False, use_confidence_calibration: bool = False,
) -> float:
    elo_a = get_surface_elo(player_a, surface, ratings_path)
    elo_b = get_surface_elo(player_b, surface, ratings_path)
    prob_a = expected_score(elo_a, elo_b)

    if use_rank_adjustment and abs(elo_a - elo_b) <= RANK_ADJUSTMENT_ELO_WINDOW:
        rank_a = get_current_rank(player_a, ratings_path)
        rank_b = get_current_rank(player_b, ratings_path)
        prob_a = _apply_rank_adjustment(prob_a, rank_a, rank_b)

    if use_confidence_calibration:
        prob_a = _apply_confidence_calibration(prob_a)

    return prob_a


if __name__ == "__main__":
    p_a, p_b, surface = "Sinner J.", "Alcaraz C.", "Hard"
    prob = win_probability(p_a, p_b, surface)
    print(f"P({p_a} beats {p_b} on {surface}) = {prob:.3f}")
