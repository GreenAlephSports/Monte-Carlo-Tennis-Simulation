"""Tests whether the market gives decorated/high-pedigree players a probability premium beyond
what their current Elo already justifies - using real tennis-data.co.uk closing-odds CSVs (real
bookmaker AvgW/AvgL, both tours) across every 2026 tournament confirmed available on that site, not
just the original single-tournament (Cincinnati, n=30 decorated-player matches) version of this
test. Same site, same per-tournament CSV convention cincinnati_paper_trading_backtest_tennisdata.py
already validated - just pulled for every event that site has published so far this season, to
attack the small-sample problem the first version of this test was honestly limited by.

Confirmed available (probed directly against the live site before writing this):
  ATP: French Open, Australian Open, Canadian Open (Montreal), Internazionali BNL d'Italia (Rome),
       BNP Paribas Open (Indian Wells), Queen's Club, Qatar Open (Doha), Stuttgart Open,
       Hamburg Open, Citi Open (Washington), Cincinnati, Madrid.
  WTA: Australian Open, Mutua Madrid Open, Internazionali BNL d'Italia (Rome), BNP Paribas Open
       (Indian Wells), Dubai, Qatar Open (Doha), Citi Open (Washington), Cincinnati, French Open,
       Miami, Stuttgart.
A handful of other slugs (WTA Queen's Club - not a real WTA event; a few transient fetch timeouts
retried once) are excluded/disclosed rather than silently dropped - see TOURNAMENTS below for the
exact final list actually used, and any that failed to fetch print a warning instead of crashing
the run.

Unlike the original Cincinnati-only version, this does NOT depend on a pre-built bracket YAML for
each event (most of these tournaments don't have one) - ratings are computed directly via
elo_ratings.calculate_elo_ratings frozen at each event's own start date (the CSV's own earliest
match date), the same bracket-free mechanism historical_bracket_calibration.py uses for historical
reconstruction, and each event gets its own temp ratings snapshot so tournaments don't clobber each
other's frozen ratings file mid-run.

Pedigree score: real, static, checkable facts, independent of current form/Elo - career Grand Slam
singles titles won (PEDIGREE_TITLES below), for every player who has ever won at least one. Best-
available real-world knowledge as of this project's own knowledge cutoff, hand-curated - see this
module's own follow-up (a verified-source pass) for the disclosed uncertainty here.

Method: identical to the original single-tournament version - per match, gap = market_prob_a -
model_prob_a (neutral alphabetical a/b ordering, never anchored to the real winner), OLS gap ~
elo_diff + pedigree_diff, plus the direct win-rate/exploitability check on decorated-player rows
where the market is more bullish than the model.

Usage:
    python model/research/pedigree_market_premium_test.py
"""
import io
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import TOUR_CONFIG, match_name_to_pool  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from ev_comparison import implied_probabilities  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402
from win_probability import win_probability  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"

# (tour, tennis-data.co.uk slug) - confirmed live and structurally valid (real Winner/Loser/AvgW/
# AvgL columns, >= 10 rows with a real closing price) by direct probe before being added here.
TOURNAMENTS = [
    ("ATP", "cincinnati"), ("ATP", "frenchopen"), ("ATP", "ausopen"), ("ATP", "montreal"),
    ("ATP", "madrid"), ("ATP", "rome"), ("ATP", "indianwells"), ("ATP", "queens"),
    ("ATP", "doha"), ("ATP", "stuttgart"), ("ATP", "hamburg"), ("ATP", "washington"),
    ("WTA", "cincinnati"), ("WTA", "frenchopen"), ("WTA", "ausopen"), ("WTA", "madrid"),
    ("WTA", "rome"), ("WTA", "indianwells"), ("WTA", "miami"), ("WTA", "dubai"),
    ("WTA", "doha"), ("WTA", "stuttgart"), ("WTA", "washington"),
]

# Real career Grand Slam (AO/FO/Wimbledon/USO) singles titles - see this module's docstring on
# provenance/uncertainty.
PEDIGREE_TITLES = {
    "Djokovic N.": 24, "Medvedev D.": 1, "Cilic M.": 1, "Alcaraz C.": 4, "Sinner J.": 3,
    "Swiatek I.": 5, "Osaka N.": 4, "Sabalenka A.": 3, "Rybakina E.": 1, "Gauff C.": 1,
    "Krejcikova B.": 2, "Ostapenko J.": 1, "Kenin S.": 1, "Keys M.": 1, "Williams V.": 7,
}
DECORATED_THRESHOLD = 1


def pedigree(player):
    return PEDIGREE_TITLES.get(player, 0)


def fetch_source_csv(tour, slug, force=False):
    path = DATA_DIR / f"{tour.lower()}_2026_{slug}_tennisdata.csv"
    if path.exists() and not force:
        return path
    base = "http://www.tennis-data.co.uk/2026/" if tour == "ATP" else "http://www.tennis-data.co.uk/2026w/"
    url = base + slug + ".csv"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=25) as response:
            raw = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as e:
        raise RuntimeError(f"couldn't fetch {url}: {e}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def build_match_rows(tour, slug, matches_history):
    csv_path = fetch_source_csv(tour, slug)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    if len(df) == 0:
        return pd.DataFrame(), f"{tour} {slug}: no rows with a parseable date"

    start_date = df["Date"].min()
    tournament_name = df["Tournament"].iloc[0] if "Tournament" in df.columns else slug

    # frozen ratings snapshot as of this event's own start date - bracket-free, same mechanism
    # historical_bracket_calibration.py uses. Written to a per-event temp path (not the shared
    # production ratings_path) so concurrent/sequential tournaments never clobber each other's
    # snapshot mid-run - the same shared-mutable-file race that hit the Osaka watch script earlier
    # this session.
    ratings_df = calculate_elo_ratings(matches_history, start_date, tour=tour)
    ratings_path = OUTPUT_DIR / f"_pedigree_test_{tour.lower()}_{slug}_ratings.csv"
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(ratings_path, index=False)
    # surface-specific, not always hard_elo - a real bug in the first version of this generalized
    # script: several of these events are on clay (French Open, Rome, Madrid) or grass, and using
    # hard_elo there is comparing the wrong number entirely, diluting elo_diff's ability to absorb
    # "the market already knows this skill gap" and corrupting the pedigree_diff coefficient it's
    # supposed to be isolated from.
    surface_elo = {
        "Hard": ratings_df.set_index("player")["hard_elo"].to_dict(),
        "Clay": ratings_df.set_index("player")["clay_elo"].to_dict(),
        "Grass": ratings_df.set_index("player")["grass_elo"].to_dict(),
    }
    pool = set(ratings_df["player"])
    name_aliases = TOUR_CONFIG[tour].name_aliases

    rows, unresolved = [], set()
    for row in df.itertuples(index=False):
        avg_w, avg_l = getattr(row, "AvgW", None), getattr(row, "AvgL", None)
        if pd.isna(avg_w) or pd.isna(avg_l):
            continue
        surface = getattr(row, "Surface", None)
        if pd.isna(surface) or surface not in ("Hard", "Clay", "Grass"):
            continue
        winner = match_name_to_pool(row.Winner, pool, name_aliases)
        loser = match_name_to_pool(row.Loser, pool, name_aliases)
        if winner is None:
            unresolved.add(row.Winner)
        if loser is None:
            unresolved.add(row.Loser)
        if winner is None or loser is None:
            continue

        a, b = sorted((winner, loser))
        won_a = (winner == a)
        market_w, market_l = implied_probabilities(avg_w, avg_l)
        market_a = market_w if won_a else market_l
        try:
            model_a = win_probability(a, b, surface, ratings_path)
        except ValueError:
            continue  # a resolved name with no ratings-csv row at all - skip, don't crash the sweep

        rows.append({
            "tour": tour, "tournament": tournament_name, "round": row.Round,
            "player_a": a, "player_b": b,
            "model_prob_a": model_a, "market_prob_a": market_a,
            "gap": market_a - model_a,
            "elo_diff": surface_elo[surface].get(a, np.nan) - surface_elo[surface].get(b, np.nan),
            "pedigree_a": pedigree(a), "pedigree_b": pedigree(b),
            "pedigree_diff": pedigree(a) - pedigree(b),
            "won_a": won_a,
        })

    warning = None
    if unresolved:
        warning = f"{tour} {tournament_name}: {len(unresolved)} name(s) unresolved: {sorted(unresolved)}"
    return pd.DataFrame(rows), warning


def ols(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    n, k = X1.shape
    if n <= k:
        return beta, np.full_like(beta, np.nan)
    sigma2 = (resid @ resid) / (n - k)
    cov = sigma2 * np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    return beta, se


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    matches_by_tour = {tour: load_matches_for_tour(tour) for tour in ("ATP", "WTA")}

    all_frames, skipped = [], []
    for tour, slug in TOURNAMENTS:
        try:
            frame, warning = build_match_rows(tour, slug, matches_by_tour[tour])
        except RuntimeError as e:
            skipped.append(f"{tour} {slug}: {e}")
            continue
        if warning:
            print(f"  WARNING: {warning}", file=sys.stderr)
        if len(frame) == 0:
            skipped.append(f"{tour} {slug}: 0 usable rows")
            continue
        print(f"{tour} {slug} ({frame['tournament'].iloc[0]}): {len(frame)} usable matches")
        all_frames.append(frame)

    if skipped:
        print(f"\n{len(skipped)} tournament(s) skipped/failed: {skipped}")

    all_rows = pd.concat(all_frames, ignore_index=True)
    all_rows = all_rows.dropna(subset=["elo_diff"])
    print(f"\n{len(all_rows)} total real matches with a resolved model probability, market "
          f"probability, and Elo gap, across {len(all_frames)} tournaments (both tours, real 2026 "
          f"closing odds)")

    n_involving_decorated = int((all_rows[["pedigree_a", "pedigree_b"]].max(axis=1) >= DECORATED_THRESHOLD).sum())
    print(f"Of those, {n_involving_decorated} matches involve at least one player with "
          f">= {DECORATED_THRESHOLD} real career Grand Slam title(s) - THIS is the real sample size "
          f"the pedigree question actually has to work with, not the full {len(all_rows)}.")

    print(f"\n{'=' * 90}\nOLS: gap (market_prob_a - model_prob_a) ~ elo_diff + pedigree_diff, all {len(all_rows)} matches\n{'=' * 90}")
    y = all_rows["gap"].values
    X = all_rows[["elo_diff", "pedigree_diff"]].values
    beta, se = ols(y, X)
    names = ["intercept", "elo_diff", "pedigree_diff"]
    for name, b, s in zip(names, beta, se):
        z = b / s if s == s and s != 0 else float("nan")
        print(f"  {name:<15}: coef={b:+.5f}  SE={s:.5f}  z={z:+.2f}"
              + ("  (|z|>1.96, nominally significant)" if abs(z) > 1.96 else "  (not significant on its own)"))

    decorated = all_rows[all_rows[["pedigree_a", "pedigree_b"]].max(axis=1) >= DECORATED_THRESHOLD].copy()
    if len(decorated) < 10:
        print(f"\nToo few matches involving a decorated player (n={len(decorated)}) to run the "
              f"win-rate/premium-exploitability check with any real confidence.")
        return

    persp = []
    for r in decorated.itertuples(index=False):
        for player, opp, model_p, market_p, ped_self, won in [
            (r.player_a, r.player_b, r.model_prob_a, r.market_prob_a, r.pedigree_a, r.won_a),
            (r.player_b, r.player_a, 1 - r.model_prob_a, 1 - r.market_prob_a, r.pedigree_b, not r.won_a),
        ]:
            if ped_self < DECORATED_THRESHOLD:
                continue
            persp.append({
                "player": player, "opponent": opp, "model_prob": model_p, "market_prob": market_p,
                "market_premium": market_p - model_p, "pedigree": ped_self, "won": won,
            })
    persp = pd.DataFrame(persp)

    premium = persp[persp["market_premium"] > 0]
    print(f"\n{'=' * 90}\nDecorated-player rows where the market is MORE bullish than the model "
          f"(n={len(premium)} of {len(persp)} decorated-player match-perspectives)\n{'=' * 90}")
    if len(premium) >= 10:
        actual = premium["won"].mean()
        mkt = premium["market_prob"].mean()
        mdl = premium["model_prob"].mean()
        observed, lo, hi = cluster_bootstrap_ci(
            premium.assign(_a=premium["won"].astype(int), _s=premium["model_prob"]), "_a", "_s", group_col="player")
        print(f"  Real win rate: {actual:.1%}  |  market's average implied prob: {mkt:.1%}  |  "
              f"model's average prob: {mdl:.1%}")
        print(f"  Model calibration gap (actual - model), player-clustered: {observed:+.1%}, "
              f"95% CI [{lo:+.1%}, {hi:+.1%}]")
        print(f"  Market calibration gap (actual - market): {actual - mkt:+.1%}")
        if actual < mkt and actual >= mdl - 0.05:
            print(f"\n  -> Real win rate ({actual:.1%}) sits closer to the MODEL's number ({mdl:.1%}) than "
                  f"the market's ({mkt:.1%}) - consistent with a real, exploitable market premium.")
        elif actual >= mkt:
            print(f"\n  -> Real win rate ({actual:.1%}) meets or exceeds the market's number ({mkt:.1%}) - "
                  f"the premium is earned; the model looks miscalibrated here, not the market.")
        else:
            print(f"\n  -> Real win rate ({actual:.1%}) sits between the two - not a clean call either way.")
    else:
        print(f"  n={len(premium)} - too small to say anything real about actual win rate here.")

    print(f"\nAll decorated-player match-perspectives, for direct inspection:")
    print(persp.sort_values("market_premium", ascending=False).to_string(index=False, formatters={
        "model_prob": "{:.1%}".format, "market_prob": "{:.1%}".format, "market_premium": "{:+.1%}".format,
    }))


if __name__ == "__main__":
    main()
