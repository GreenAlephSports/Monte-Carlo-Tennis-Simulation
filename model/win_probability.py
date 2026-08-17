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


def apply_logit_shift(prob: float, shift: float) -> float:
    """Additive adjustment in logit space - same pattern as _apply_rank_adjustment/_apply_
    confidence_calibration above, exposed for callers that need to shift an already-computed
    probability without recomputing it from raw Elo (e.g. simulate.py's in-tournament upset
    boost, which needs to combine two independent per-player shifts into one match)."""
    logit_p = math.log(prob / (1 - prob))
    return 1 / (1 + math.exp(-(logit_p + shift)))


# Empirically-fit in-tournament "beat a big favorite" momentum signal - see the backtest that
# derived it: model/survivorship_upset_test.py, run on ATP 2000-2026 with the same frozen-per-
# tournament-edition Elo and chronological 80/20 tournament split as the rank-gap/Platt fits
# above. Players whose most recent win THIS TOURNAMENT came against an opponent >100 Elo points
# higher than themselves beat Elo's own prediction in their NEXT match by +3.1pp on average
# (train era, n=7,696, z=6.32) - real and monotonic in gap size (no_upset -0.3% [n.s.], 0-50
# +0.6% [n.s.], 50-100 +2.9% [z=4.30], 100+ +3.1%), but the middle tier looked significant in
# training and didn't hold up out-of-sample (held-out log-loss improvement -0.0001, indistinguishable
# from zero) - so a single 100+ threshold was kept instead of the full graded scheme. That single
# threshold captures +0.0006 held-out log-loss improvement (95% player-clustered bootstrap CI
# [+0.0004, +0.0008], n=27,501 test-era rows) - nearly all of the full 4-bucket model's +0.0007,
# with one parameter instead of three, and clearly beats a generic "won last round at all" framing
# (+0.0002, CI barely excluding zero) - this is not the same effect as generic survivorship.
#
# Structurally an in-tournament-only signal - "who did they just beat in THIS event" doesn't exist
# before Round 1 - so unlike the rank-gap/Platt adjustments above, this can never apply to a
# pre-tournament baseline prediction, only a round-by-round replay that already knows real
# results. See simulate.py's use_upset_boost plumbing (wired into hybrid_simulation.py's round
# replay only, same constraint a fatigue adjustment would have).
UPSET_BOOST_ELO_GAP_THRESHOLD = 100
UPSET_BOOST_LOGIT_SHIFT = 0.1383


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
