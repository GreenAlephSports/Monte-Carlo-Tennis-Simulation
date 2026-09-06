import math
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from elo_ratings import expected_score

SURFACE_COLUMNS = {
    "Hard": "hard_elo",
    "Clay": "clay_elo",
    "Grass": "grass_elo",
}
SURFACE_COLUMNS_UNDAMPED = {
    "Hard": "hard_elo_undamped",
    "Clay": "clay_elo_undamped",
    "Grass": "grass_elo_undamped",
}

ATP_RATINGS_PATH = Path(__file__).resolve().parent.parent / "output" / "player_elo_ratings_atp.csv"
WTA_RATINGS_PATH = Path(__file__).resolve().parent.parent / "output" / "player_elo_ratings_wta.csv"
HEIGHT_METADATA_PATH = Path(__file__).resolve().parent.parent / "output" / "player_handedness.csv"

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

# DISABLED as of 2026-09-05 (see win_probability(), use_rank_adjustment default below) - the fixed
# constants above have gone stale. model/research/rank_gap_original_reproduction.py confirmed the
# original train-era pattern still reproduces (harness is trustworthy) but the EXISTING fixed formula
# shows no significant held-out benefit on the (mostly-recent) test-era split of full history.
# model/research/rank_adjustment_recency_refit.py then asked the direct follow-up: is that just
# stale constants, or has the underlying signal itself disappeared? Refit C, D via MLE on ONLY
# 2021-2026 ATP data (train-era half of a chronological split WITHIN that window: C=0.2185, D=117.10,
# vs. the fixed C=1.0629/D=260.72 above), then validated BOTH formulas on the SAME 2021-2026 test-era
# half, same |Elo diff|<=50 gate, player-clustered bootstrap CI: existing fixed formula
# -0.00125 CI[-0.00900,+0.00621] (negative point estimate, not significant), recency-refit formula
# +0.00226 CI[-0.00036,+0.00488] (closer, still doesn't clear zero). A same-era refit does NOT
# restore a real, significant benefit - this isn't stale constants, the rank-vs-Elo divergence this
# correction was built on appears to have genuinely weakened/disappeared in the current game. Left
# live (use_rank_adjustment=True) would mean shipping a correction with zero present-day evidence
# behind it; disabled by default until a real, held-out-validated replacement is found. The constants
# above are kept (not deleted) so a future refit attempt - e.g. once more 2024-2026+ editions
# accumulate - has the old baseline to compare against.

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


# cache so we're not re-reading the csv on every single matchup lookup during a sim run. Keyed on
# (path, mtime) - not path alone - because a real bug surfaced here: every normal invocation of
# this project writes a ratings CSV exactly once per process (bracket_export.py, simulate_
# projected_draw, etc.), so a path-only cache never had a chance to go stale. But
# historical_bracket_calibration.py's sweep runs many editions in one long-lived process, each
# overwriting the SAME shared per-tour ratings_path with a freshly-recomputed snapshot - a
# path-only cache would keep serving the FIRST edition's frozen ratings forever, silently
# simulating every later edition against the wrong (stale) Elo. Including mtime makes a fresh
# write bust the cache automatically, with zero behavior change for the single-write-per-process
# case every other caller already is.
@lru_cache(maxsize=None)
def _load_ratings_cached(ratings_path: Path, _mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(ratings_path).set_index("player")


def _load_ratings(ratings_path: Path) -> pd.DataFrame:
    ratings_path = Path(ratings_path)
    return _load_ratings_cached(ratings_path, ratings_path.stat().st_mtime_ns)


def load_ratings(ratings_path: Path) -> pd.DataFrame:
    """Public accessor for _load_ratings - lets a caller that's about to make MANY win_probability
    (or get_surface_elo/get_current_rank/etc.) calls against the same ratings_path load it ONCE
    and pass the result as those functions' _ratings= param, instead of paying _load_ratings' own
    cache-freshness check (a real stat() syscall - profiled 2026-08-31 at ~70-90us here, plausibly
    antivirus real-time scanning on this drive) on every single call. See simulate.py's callers for
    the pattern: load once per top-level entry point (not once per simulation, and never once per
    match), thread the result down through every _play_round/win_probability call underneath."""
    return _load_ratings(ratings_path)


def get_surface_elo(player: str, surface: str, ratings_path: Path, use_surface_mismatch_damping: bool = True,
                     _ratings: pd.DataFrame = None) -> float:
    """use_surface_mismatch_damping=False reads the pre-damping blended column
    (see elo_ratings._damp_surface_mismatch) instead of the damped one - a real column lookup,
    same cost as the default path, not a per-call recomputation. Falls back to the damped column
    if an older ratings CSV (written before this flag existed) doesn't have the undamped columns
    yet, rather than raising.

    _ratings, if given, is an already-loaded ratings DataFrame (from _load_ratings) - lets a caller
    that needs several of this module's get_* lookups for the same ratings_path (win_probability()
    itself, chiefly) pay _load_ratings' cache-freshness check (a real stat() syscall - see
    _load_ratings' docstring) once per match instead of once per lookup. Profiled 2026-08-31: this
    stat() call was 38% of total win_probability() time on this machine (cProfile, 20,000 calls,
    ~71us/stat - elevated, plausibly antivirus real-time scanning on this drive) despite every
    lookup only ever reading an in-process-cached, immutable-for-the-run DataFrame - not a real
    per-call cost, just an avoidable repeated freshness check. External callers are unaffected:
    omitting _ratings (the default) loads normally, identical to before this existed."""
    if surface not in SURFACE_COLUMNS:
        raise ValueError(f"Unsupported surface: {surface!r}. Expected one of {list(SURFACE_COLUMNS)}")

    ratings = _ratings if _ratings is not None else _load_ratings(ratings_path)
    if player not in ratings.index:
        raise ValueError(f"Unknown player: {player!r}")

    column = SURFACE_COLUMNS[surface]
    if not use_surface_mismatch_damping:
        undamped_column = SURFACE_COLUMNS_UNDAMPED[surface]
        if undamped_column in ratings.columns:
            column = undamped_column
    return ratings.loc[player, column]


def get_current_rank(player: str, ratings_path: Path, _ratings: pd.DataFrame = None):
    """Current ranking as of the ratings file's cutoff date, or None if unknown - either because
    this player never appeared with a valid Rank_1/Rank_2 in the training window (e.g. a brand-new
    tier-3 placeholder - see bracket.match_draw_to_ratings), or because the ratings file was built
    from the local Kaggle fallback snapshot, which doesn't carry ranking columns at all (see
    calculate_elo_ratings' current_rank comment).

    _ratings: see get_surface_elo's docstring - same optional preloaded-DataFrame passthrough."""
    ratings = _ratings if _ratings is not None else _load_ratings(ratings_path)
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


# Empirically-fit layoff/rust adjustment - see model/layoff_test.py, run separately on ATP and WTA
# 2000/2006-2026 with the same frozen-per-tournament-edition Elo and chronological 80/20 tournament
# split as every other correction above. Bucketed by days since a player's own last recorded match
# (in EITHER tour history, any earlier tournament - not just this one, so it's known before a draw
# is even set, unlike upset-boost above): actual win rate diverges from Elo's prediction by bucket,
# and that divergence held up on a held-out split in BOTH tours - ATP +0.0007 log-loss improvement
# (95% CI [+0.0001, +0.0013]), WTA +0.0011 (95% CI [+0.0005, +0.0019]).
#
# A single-threshold collapse (model/layoff_two_bucket_test.py, mirroring the upset-boost precedent
# above) was tried first and rejected: collapsing everything under 90 days to "no adjustment"
# dropped the overall held-out CI back to straddling zero in both tours ([-0.0001,+0.0004] ATP,
# [-0.0004,+0.0005] WTA), because the 30_60d bucket's own real, individually-significant
# contribution (ATP test-era log-loss improvement +0.0034 - the largest single bucket after
# 90d_plus) got thrown away along with the noisier 14_30d/60_90d buckets. Moving the boundary to 60
# days instead (model/layoff_three_bucket_test.py) didn't rescue it either (60_90d's own
# contribution is tiny: +0.0011 ATP/+0.0009 WTA), and neither did a targeted 30_60d+90d_plus-only
# 2-parameter version (ATP CI [-0.0000,+0.0006] - borderline; WTA CI [-0.0004,+0.0005] - still
# straddling zero) - confirming 30_60d needs to compose with the OTHER buckets, not just 90d_plus,
# to clear the bar reliably. The full 5-bucket gradient below is what actually does that, even
# though it isn't strictly monotonic (both tours show 60_90d's residual recovering partway before
# 90d_plus drops again - a real, tour-consistent bump, not noise, but not explained by the
# bucketing here).
#
# Shifts are the full-dataset refit (train+test combined, same convention as RANK_ADJUSTMENT/
# PLATT_B/UPSET_BOOST_LOGIT_SHIFT above - the held-out split was for validating the effect exists,
# not for holding back data from the final production fit).
#
# Fit SEPARATELY per tour, unlike RANK_ADJUSTMENT/PLATT_B/UPSET_BOOST (all ATP-only, reused for
# WTA): the two tours' bucket shifts diverge enough to matter here, not just differ by sampling
# noise around a shared shape - WTA's 14_30d shift is essentially zero (+0.0043) where ATP's is a
# real penalty (-0.0841), and WTA's 30_60d penalty (-0.0694) is under half of ATP's (-0.1579).
# Applying ATP's numbers to WTA would have meant a wrong-signed adjustment for one bucket, not just
# an imprecise magnitude.
#
# no_prior_match (a player's first-ever recorded match in the whole dataset - zero career history,
# not a known player returning from a break) is deliberately excluded in both tours: it's a
# different phenomenon, already partly handled by STARTING_ELO's neutral fallback for unrated
# players, not the layoff/rust mechanism this adjustment targets - layoff_test.py itself treats it
# as outside the gradient hypothesis. A player with no computable days-since-last-match (either
# reason) gets zero adjustment.
# KNOWN LIMITATION, confirmed 2026-08-31 (see model/research/layoff_under14_cross_tournament_
# refit.py): the under_14d bucket's fitted shift below is a POOLED average over two populations
# with genuinely different effect sizes that the original fit never separated - ~64-65% of the
# validated under_14d rows are a player who just won an earlier ROUND OF THE SAME TOURNAMENT (a
# near-universal, mostly uninformative "still competing" signal), not a player arriving fresh
# within 2 weeks of a NEW tournament (the ~35% remainder, and the only case that's actually
# relevant to a pre-tournament baseline prediction like this file computes). Isolating the two:
# ATP's genuine cross-tournament effect is +0.0184 (same-tournament alone is +0.0731, LARGER than
# the pooled +0.0489 below) - WTA's is +0.0711 (same-tournament alone is +0.0181, smaller than the
# pooled +0.0346) - i.e. the two tours are biased in OPPOSITE directions by the same pooling
# artifact. Neither isolated cross-tournament estimate reached held-out significance on its own
# (splitting the bucket roughly halves the per-side sample), so the pooled numbers below are left
# unchanged rather than swapped for an under-powered replacement - but treat any single
# pre-tournament under_14d comparison (e.g. two players with a 6-day vs 8-day gap) as resting on a
# softer empirical footing than the other buckets, which don't have this same-tournament
# contamination (a same-tournament round gap essentially never exceeds 14 days in single-
# elimination play).
LAYOFF_BUCKET_EDGES_ATP = [
    ("under_14d", lambda d: d < 14, 0.0489),
    ("14_30d", lambda d: 14 <= d < 30, -0.0841),
    ("30_60d", lambda d: 30 <= d < 60, -0.1579),
    ("60_90d", lambda d: 60 <= d < 90, -0.0820),
    ("90d_plus", lambda d: d >= 90, -0.2145),
]
LAYOFF_BUCKET_EDGES_WTA = [
    ("under_14d", lambda d: d < 14, 0.0346),
    ("14_30d", lambda d: 14 <= d < 30, 0.0043),
    ("30_60d", lambda d: 30 <= d < 60, -0.0694),
    ("60_90d", lambda d: 60 <= d < 90, -0.0530),
    ("90d_plus", lambda d: d >= 90, -0.2500),
]


# Empirically-fit recent-form residual correction - see model/research/recent_form_test.py: run at
# full historical scale (both tours, ~2,800 tournament editions, 46778+186893 combined train+test
# player-perspective rows), a player's own (actual win rate - Elo-predicted win rate) over their
# most recent 10 real matches (elo_ratings.RECENT_FORM_WINDOW) carries real, held-out-validated
# signal beyond a stable Elo rating alone - a short-term hot/cold wobble Elo itself doesn't react to.
# Single continuous logistic coefficient, same "fit on train, validate held-out" shape as
# RANK_ADJUSTMENT_C/D and PLATT_B above: train-era beta +0.2021 (SE=0.0364, z=+5.55). Held out:
# +0.0002 log-loss improvement, 95% player-clustered bootstrap CI [+0.0000, +0.0003] - excludes
# zero, a real but small effect. The window=15 robustness check did NOT hold up out-of-sample (CI
# straddled zero) and was not used - only the window=10 result is live here.
#
# Fit against RAW Elo predictions (elo_ratings.expected_score, no windowing, no surface split,
# no rank/layoff/confidence-calibration already applied) - same convention every other correction
# in this file was originally fit against, so it composes the same way: applied here, after layoff
# and before confidence-calibration (which must stay last, since Platt-scaling corrects whatever
# overconfidence remains after every other shift).
RECENT_FORM_BETA = 0.2021


# Empirically-fit height correction - see model/research/height_effect_scoped_test.py: a linear-
# probability-model coefficient (won_a ~ elo_diff + height_diff, OLS, NOT a logit-space fit like
# every other correction above) for height_diff (height_a - height_b, in cm) among ATP+WTA matches
# restricted to the ONE population height_effect_validation_test.py's full-history robustness audit
# found the effect actually stable in: prime-age (avg age 23-29), best_rank <= 150, 2015-2024. Pooled
# scoped coefficient +0.001621 (z=+4.51, 95% player-clustered bootstrap CI [+0.001086,+0.002179]),
# and - unlike a pooled number taken at face value - a chronological split WITHIN that scoped
# population independently replicates in BOTH halves with the same sign (earlier half +0.001346
# z=+2.79, later half +0.001864 z=+3.45): the strongest form of support this dataset can offer for a
# correction this narrow. Every broader/adjacent population validation_test checked (151+ rank,
# <23 or 29+ avg age, pre-2015/post-2024) did NOT hold up and is deliberately excluded here - this is
# NOT "height matters, scaled down everywhere else", it's "height matters in this one population and
# nowhere else checked".
#
# Applied as a DIRECT probability-space additive shift (prob_a + HEIGHT_COEF * height_diff), not
# routed through apply_logit_shift like the other corrections - because that's the scale it was
# actually fit on (a linear probability model against raw elo_diff, not a logistic fit against
# logit-space odds). Composed in the same downstream slot as recent-form (after layoff, before
# confidence calibration): fit against raw Elo like recent-form, same "applied where the codebase's
# existing convention already stacks corrections" pragmatic reasoning documented at RECENT_FORM_BETA
# above, not because a joint refit was done.
HEIGHT_COEF = 0.001621
HEIGHT_AGE_RANGE = (23, 29)  # avg age of both players, inclusive-exclusive [lo, hi)
HEIGHT_RANK_MAX = 150        # best (numerically lowest) current rank of the two players


@lru_cache(maxsize=None)
def _load_height_metadata_cached(_mtime_ns: int) -> pd.DataFrame:
    df = pd.read_csv(HEIGHT_METADATA_PATH, keep_default_na=False, dtype=str)
    df = df.set_index("player")
    df["height_cm"] = pd.to_numeric(df["height_cm"], errors="coerce")
    df["birth_year"] = pd.to_numeric(df["birth_year"], errors="coerce")
    return df


def _load_height_metadata() -> pd.DataFrame:
    return _load_height_metadata_cached(HEIGHT_METADATA_PATH.stat().st_mtime_ns)


def get_height(player: str):
    """Height in cm, or None if unknown (player not in player_handedness.csv, or their height_cm
    was never cleanly resolved - see wikipedia_handedness_scrape.py's height_status column)."""
    metadata = _load_height_metadata()
    if player not in metadata.index:
        return None
    height = metadata.loc[player, "height_cm"]
    return None if pd.isna(height) else float(height)


def get_birth_year(player: str):
    """Birth year, or None if unknown - same resolution caveat as get_height."""
    metadata = _load_height_metadata()
    if player not in metadata.index:
        return None
    birth_year = metadata.loc[player, "birth_year"]
    return None if pd.isna(birth_year) else int(birth_year)


def _apply_height_adjustment(prob_a: float, height_a, height_b, birth_a, birth_b, rank_a, rank_b) -> float:
    """A no-op unless BOTH players' height and birth year are known, both current ranks are known,
    and the pair falls inside the one validated population (HEIGHT_AGE_RANGE avg age, HEIGHT_RANK_MAX
    best rank) - this correction has zero empirical backing outside that population (see HEIGHT_COEF's
    docstring above) and is deliberately gated to it, same discipline as RANK_ADJUSTMENT_ELO_WINDOW."""
    if height_a is None or height_b is None or birth_a is None or birth_b is None:
        return prob_a
    if rank_a is None or rank_b is None or pd.isna(rank_a) or pd.isna(rank_b):
        return prob_a

    best_rank = min(rank_a, rank_b)
    if best_rank > HEIGHT_RANK_MAX:
        return prob_a

    current_year = date.today().year
    avg_age = ((current_year - birth_a) + (current_year - birth_b)) / 2
    lo_age, hi_age = HEIGHT_AGE_RANGE
    if not (lo_age <= avg_age < hi_age):
        return prob_a

    height_diff = height_a - height_b
    adjusted = prob_a + HEIGHT_COEF * height_diff
    return min(max(adjusted, 1e-6), 1 - 1e-6)


def get_recent_form_residual(player: str, ratings_path: Path, _ratings: pd.DataFrame = None):
    """Recent-form residual as of the ratings file's cutoff date - see elo_ratings.
    compute_recent_form_residuals. None if unknown: the player isn't in the ratings file, the
    column is missing (an older ratings file built before this correction existed), or they had
    fewer than elo_ratings.RECENT_FORM_WINDOW real matches in the full match history (a brand-new
    or very sparse player) - same "unknown means no-op" convention as current_rank/days_since_
    last_match.

    _ratings: see get_surface_elo's docstring - same optional preloaded-DataFrame passthrough."""
    ratings = _ratings if _ratings is not None else _load_ratings(ratings_path)
    if player not in ratings.index or "recent_form_residual" not in ratings.columns:
        return None
    residual = ratings.loc[player, "recent_form_residual"]
    return None if pd.isna(residual) else residual


def _apply_recent_form_adjustment(prob_a: float, residual_a, residual_b) -> float:
    """Additive correction in logit space, one shift per player based on their OWN recent-form
    residual - same per-player-then-combine pattern as _apply_layoff_adjustment. A no-op whenever
    either player's residual is unknown, or the two shifts are equal (including both zero)."""
    if residual_a is None or residual_b is None:
        return prob_a
    shift_a = RECENT_FORM_BETA * residual_a
    shift_b = RECENT_FORM_BETA * residual_b
    if shift_a == shift_b:
        return prob_a
    return apply_logit_shift(prob_a, shift_a - shift_b)


def get_days_since_last_match(player: str, ratings_path: Path, _ratings: pd.DataFrame = None):
    """Days between the tournament's cutoff date and this player's last recorded match, frozen the
    same way current_rank is - None if unknown (player not in the ratings file, or their first-ever
    recorded match is this tournament itself, i.e. days_since_last_match was NaN when the ratings
    file was built).

    _ratings: see get_surface_elo's docstring - same optional preloaded-DataFrame passthrough."""
    ratings = _ratings if _ratings is not None else _load_ratings(ratings_path)
    if player not in ratings.index or "days_since_last_match" not in ratings.columns:
        return None
    days = ratings.loc[player, "days_since_last_match"]
    return None if pd.isna(days) else days


def _layoff_bucket_edges_for(ratings_path: Path):
    """Selects the WTA-fit shifts only for the actual WTA ratings file; every other ratings_path
    (the ATP file, or any custom/backtest path - which in practice is always one tour's real
    ratings file reused under a different name, e.g. backtest_hard_court.py's per-bracket copies of
    tour_config.ratings_path) defaults to the ATP fit, same as before this split existed."""
    return LAYOFF_BUCKET_EDGES_WTA if Path(ratings_path) == WTA_RATINGS_PATH else LAYOFF_BUCKET_EDGES_ATP


def _layoff_shift_for_days(days, bucket_edges) -> float:
    if days is None:
        return 0.0
    for _, test, shift in bucket_edges:
        if test(days):
            return shift
    return 0.0  # unreachable - edges are exhaustive over [0, inf)


def _apply_layoff_adjustment(prob_a: float, days_a, days_b, bucket_edges) -> float:
    """Additive correction in logit space, one shift per player based on their OWN days-since-last-
    match bucket - same per-player-then-combine pattern as the upset-boost shift in simulate.py.
    A no-op whenever both players land in the same bucket (most commonly: neither has a layoff at
    all), including when both are unknown."""
    shift_a = _layoff_shift_for_days(days_a, bucket_edges)
    shift_b = _layoff_shift_for_days(days_b, bucket_edges)
    if shift_a == shift_b:
        return prob_a
    return apply_logit_shift(prob_a, shift_a - shift_b)


# just pulls each player's surface-specific elo and updates ratings
def win_probability(
    player_a: str, player_b: str, surface: str, ratings_path: Path = ATP_RATINGS_PATH,
    use_rank_adjustment: bool = False, use_confidence_calibration: bool = True,
    use_layoff_adjustment: bool = True, use_recent_form_adjustment: bool = True,
    use_height_adjustment: bool = True,
    use_surface_mismatch_damping: bool = True, _ratings: pd.DataFrame = None,
) -> float:
    # loaded once and threaded through every get_* call below instead of each one reloading
    # independently - see get_surface_elo's _ratings docstring for why (a real, profiled cost:
    # _load_ratings' cache-freshness stat() call was 38% of this function's total time before this
    # change, from up to 8 redundant stat()s per match instead of 1).
    #
    # _ratings, if given by the caller (see load_ratings' docstring), skips this function's own
    # load too - added 2026-08-31 after profiling a LIVE run found simulate.py's _play_round was
    # calling get_surface_elo directly (for its own upset-boost tracking) on every match BEFORE
    # ever reaching win_probability, so this function's own single _load_ratings call was only
    # ever removing HALF the redundancy - _play_round's direct calls were still paying a fresh
    # stat() apiece. Passing _ratings all the way down from a single top-level load closes the
    # other half.
    ratings = _ratings if _ratings is not None else _load_ratings(ratings_path)

    elo_a = get_surface_elo(player_a, surface, ratings_path, use_surface_mismatch_damping, _ratings=ratings)
    elo_b = get_surface_elo(player_b, surface, ratings_path, use_surface_mismatch_damping, _ratings=ratings)
    prob_a = expected_score(elo_a, elo_b)

    if use_rank_adjustment and abs(elo_a - elo_b) <= RANK_ADJUSTMENT_ELO_WINDOW:
        rank_a = get_current_rank(player_a, ratings_path, _ratings=ratings)
        rank_b = get_current_rank(player_b, ratings_path, _ratings=ratings)
        prob_a = _apply_rank_adjustment(prob_a, rank_a, rank_b)

    if use_layoff_adjustment:
        days_a = get_days_since_last_match(player_a, ratings_path, _ratings=ratings)
        days_b = get_days_since_last_match(player_b, ratings_path, _ratings=ratings)
        prob_a = _apply_layoff_adjustment(prob_a, days_a, days_b, _layoff_bucket_edges_for(ratings_path))

    if use_recent_form_adjustment:
        residual_a = get_recent_form_residual(player_a, ratings_path, _ratings=ratings)
        residual_b = get_recent_form_residual(player_b, ratings_path, _ratings=ratings)
        prob_a = _apply_recent_form_adjustment(prob_a, residual_a, residual_b)

    if use_height_adjustment:
        height_a = get_height(player_a)
        height_b = get_height(player_b)
        birth_a = get_birth_year(player_a)
        birth_b = get_birth_year(player_b)
        rank_a = get_current_rank(player_a, ratings_path, _ratings=ratings)
        rank_b = get_current_rank(player_b, ratings_path, _ratings=ratings)
        prob_a = _apply_height_adjustment(prob_a, height_a, height_b, birth_a, birth_b, rank_a, rank_b)

    if use_confidence_calibration:
        prob_a = _apply_confidence_calibration(prob_a)

    return prob_a


if __name__ == "__main__":
    p_a, p_b, surface = "Sinner J.", "Alcaraz C.", "Hard"
    prob = win_probability(p_a, p_b, surface)
    print(f"P({p_a} beats {p_b} on {surface}) = {prob:.3f}")
