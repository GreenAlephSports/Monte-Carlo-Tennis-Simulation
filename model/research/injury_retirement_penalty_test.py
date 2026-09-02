"""Fits a real, held-out-validated Elo/logit-space penalty for a player returning from a match that
ended in THEIR OWN retirement or walkover - the actual mechanism behind Rybakina's situation
tonight (retired mid-match at Cincinnati, 3 weeks before the US Open). Same rigor as every other
production correction: frozen-per-tournament-edition Elo, chronological 80/20 train/test split,
player-clustered bootstrap held-out validation.

DATA SOURCE, and why this couldn't be built from the datasets already in use tonight: the project's
main historical dataset (Kaggle, via elo_ratings.load_matches_for_tour) has NO retirement/walkover
field at all - confirmed by direct inspection, its Score column is fully normalized to clean set
scores with zero "RET"/"W/O" markers across 68k+ ATP and 45k+ WTA rows. tennis-data.co.uk (already
used tonight for the pedigree/surface-mismatch tests, but only for 2026-season per-tournament files)
DOES publish this: every full SEASON archive (http://www.tennis-data.co.uk/{year}/{year}.xlsx for
ATP, .../{year}w/{year}.xlsx for WTA) carries a real Comment column ("Completed"/"Retired"/
"Walkover"). This script pulls those season archives across many years - a real, previously-untapped
source - specifically to get this trigger signal at historical scale.

Trigger: for each match, "days_since_own_retirement" = days since this player's own most recent
match (in EITHER tour history) that ended with THEM as the Loser and Comment in
{"Retired","Walkover"} - None/no-signal if no such match in the last RETIREMENT_LOOKBACK_DAYS, same
"missing means no adjustment" convention as every other correction in win_probability.py. Bucketed
the same shape as the existing layoff correction (win_probability.LAYOFF_BUCKET_EDGES_ATP/WTA) since
that's the established, already-validated convention for a "days since an event" trigger - but keyed
on retirement/walkover specifically, not just any last match.

Usage:
    python model/research/injury_retirement_penalty_test.py
"""
import io
import math
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elo_ratings import K_FACTOR, STARTING_ELO, expected_score  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "tennis_data_seasons"
YEARS = list(range(2010, 2027))
TRAIN_FRACTION = 0.8
RETIREMENT_LOOKBACK_DAYS = 180
BUCKET_EDGES = [
    ("under_14d", lambda d: d < 14),
    ("14_30d", lambda d: 14 <= d < 30),
    ("30_60d", lambda d: 30 <= d < 60),
    ("60_90d", lambda d: 60 <= d < 90),
    ("90_180d", lambda d: 90 <= d < 180),
]
EPS = 1e-3


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def log_loss(actual, pred):
    pred = np.clip(pred, EPS, 1 - EPS)
    return -(actual * np.log(pred) + (1 - actual) * np.log(1 - pred))


def fetch_season(tour, year, force=False):
    path = DATA_DIR / f"{tour.lower()}_{year}.xlsx"
    if path.exists() and not force:
        return path
    base = "http://www.tennis-data.co.uk/" if tour == "ATP" else "http://www.tennis-data.co.uk/"
    sub = f"{year}/" if tour == "ATP" else f"{year}w/"
    url = f"{base}{sub}{year}.xlsx"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as e:
        raise RuntimeError(f"couldn't fetch {url}: {e}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def load_all(tour):
    frames = []
    for year in YEARS:
        try:
            path = fetch_season(tour, year)
        except RuntimeError as e:
            print(f"  {tour} {year}: SKIP - {e}", file=sys.stderr)
            continue
        try:
            df = pd.read_excel(path)
        except Exception as e:
            print(f"  {tour} {year}: SKIP - unreadable ({e})", file=sys.stderr)
            continue
        keep = ["Tournament", "Date", "Surface", "Round", "Winner", "Loser", "WRank", "LRank", "Comment"]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            print(f"  {tour} {year}: SKIP - missing columns {missing}", file=sys.stderr)
            continue
        df = df[keep].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Winner", "Loser"])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["Tournament", "Date", "Surface", "Round", "Winner", "Loser", "WRank", "LRank", "Comment"])
    return pd.concat(frames, ignore_index=True)


def build_frozen_predictions(df, tour):
    df = df.copy()
    df["edition_id"] = df["Tournament"].astype(str) + " " + df["Date"].dt.year.astype(str)
    edition_start = df.groupby("edition_id")["Date"].transform("min")
    editions = (
        df.assign(edition_start=edition_start)[["edition_id", "edition_start"]]
        .drop_duplicates().sort_values("edition_start").reset_index(drop=True)
    )

    overall_elo = {}
    last_retirement_date = {}
    rows = []

    for edition_id in editions["edition_id"]:
        edition_matches = df[df["edition_id"] == edition_id]
        snap_elo = dict(overall_elo)
        snap_ret = dict(last_retirement_date)

        for row in edition_matches.itertuples(index=False):
            w, l = row.Winner, row.Loser
            elo_w = snap_elo.get(w, STARTING_ELO)
            elo_l = snap_elo.get(l, STARTING_ELO)
            pred_w = expected_score(elo_w, elo_l)

            for player, opponent, pred, win in [(w, l, pred_w, 1), (l, w, 1 - pred_w, 0)]:
                ret_date = snap_ret.get(player)
                days_since = (row.Date - ret_date).days if ret_date is not None else None
                rows.append((edition_id, row.Date, tour, player, opponent, pred, win, days_since))

        for row in edition_matches.itertuples(index=False):
            w, l = row.Winner, row.Loser
            overall_elo.setdefault(w, STARTING_ELO)
            overall_elo.setdefault(l, STARTING_ELO)
            exp_w = expected_score(overall_elo[w], overall_elo[l])
            overall_elo[w] += K_FACTOR * (1 - exp_w)
            overall_elo[l] += K_FACTOR * (0 - (1 - exp_w))
            if str(row.Comment).strip() in ("Retired", "Walkover"):
                last_retirement_date[l] = row.Date  # the LOSER is the one who retired/couldn't play

    preds = pd.DataFrame(rows, columns=[
        "edition_id", "date", "tour", "player", "opponent", "pred_win", "actual_win", "days_since_retirement",
    ])
    return preds, editions


def bucket_of(days):
    if days is None or days != days or days > RETIREMENT_LOOKBACK_DAYS:
        return None
    for name, test in BUCKET_EDGES:
        if test(days):
            return name
    return None


def fit_intercept_newton(offset, y, iters=100, tol=1e-10):
    shift = 0.0
    for _ in range(iters):
        z = offset + shift
        p = 1 / (1 + np.exp(-np.clip(z, -35, 35)))
        grad = np.sum(y - p)
        hess = -np.sum(p * (1 - p))
        if hess == 0:
            break
        step = grad / hess
        shift -= step
        if abs(step) < tol:
            break
    z = offset + shift
    p = 1 / (1 + np.exp(-np.clip(z, -35, 35)))
    hess = -np.sum(p * (1 - p))
    se = math.sqrt(-1 / hess) if hess < 0 else float("nan")
    return shift, se


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    all_preds, all_editions = [], {}
    for tour in ("ATP", "WTA"):
        print(f"Fetching {tour} season archives {YEARS[0]}-{YEARS[-1]}...")
        raw = load_all(tour)
        print(f"  {tour}: {len(raw)} real matches loaded, "
              f"{(raw['Comment'].astype(str).str.strip().isin(['Retired', 'Walkover'])).sum()} retirement/walkover-ended")
        preds, editions = build_frozen_predictions(raw, tour)
        all_preds.append(preds)
        all_editions[tour] = editions

    preds = pd.concat(all_preds, ignore_index=True)
    preds["bucket"] = preds["days_since_retirement"].apply(bucket_of)
    triggered = preds[preds["bucket"].notna()]
    print(f"\nTotal player-perspective rows: {len(preds)}; rows with an active retirement-recency "
          f"trigger (<= {RETIREMENT_LOOKBACK_DAYS}d since their own Retired/Walkover loss): {len(triggered)}")
    print(triggered["bucket"].value_counts())

    # chronological 80/20 split per tour
    train_editions, test_editions = set(), set()
    for tour, editions in all_editions.items():
        split_idx = int(len(editions) * TRAIN_FRACTION)
        train_editions |= set(editions["edition_id"].iloc[:split_idx])
        test_editions |= set(editions["edition_id"].iloc[split_idx:])

    train = preds[preds["edition_id"].isin(train_editions)]
    test = preds[preds["edition_id"].isin(test_editions)]
    print(f"\nTrain-era rows: {len(train)}, test-era rows: {len(test)}")

    print(f"\n{'=' * 100}\nPer-bucket fit (train era) + held-out validation (test era)\n{'=' * 100}")
    fitted_shifts = {}
    for name, _ in BUCKET_EDGES:
        b_train = train[train["bucket"] == name]
        if len(b_train) < 20:
            print(f"  {name:<10}: n_train={len(b_train):<5} - too few to fit")
            continue
        offset = b_train["pred_win"].apply(logit).values
        y = b_train["actual_win"].values
        shift, se = fit_intercept_newton(offset, y)
        z = shift / se if se == se and se != 0 else float("nan")
        fitted_shifts[name] = shift

        b_test = test[test["bucket"] == name]
        if len(b_test) < 10:
            print(f"  {name:<10}: n_train={len(b_train):<5} shift={shift:+.4f} (z={z:+.2f}) - "
                  f"n_test={len(b_test)} too few to validate")
            continue
        raw_loss = log_loss(b_test["actual_win"].values, b_test["pred_win"].values)
        adj_pred = b_test.apply(lambda r: sigmoid(logit(r["pred_win"]) + shift), axis=1)
        adj_loss = log_loss(b_test["actual_win"].values, adj_pred.values)
        d = b_test.assign(raw_loss=raw_loss, adj_loss=adj_loss)
        diff, lo, hi = cluster_bootstrap_ci(d, "raw_loss", "adj_loss", group_col="player")
        actual_rate = b_test["actual_win"].mean()
        pred_rate = b_test["pred_win"].mean()
        print(f"  {name:<10}: n_train={len(b_train):<5} n_test={len(b_test):<5} fitted_shift={shift:+.4f} "
              f"(train z={z:+.2f})  test actual={actual_rate:.1%} vs raw_pred={pred_rate:.1%}  "
              f"log-loss improvement={diff:+.4f} CI[{lo:+.4f},{hi:+.4f}]" +
              ("  <- excludes zero" if (lo > 0 or hi < 0) else ""))

    print(f"\n{'=' * 100}\nTranslating the fitted shift to an approximate Elo-point penalty\n{'=' * 100}")
    print("(logit shift -s at a ~50/50 baseline translates to roughly -s*400/ln(10) Elo points "
          "of equivalent handicap against an even opponent - a rough, illustrative conversion, "
          "not exact away from 50/50)")
    for name, shift in fitted_shifts.items():
        elo_equiv = shift * 400 / math.log(10)
        print(f"  {name:<10}: logit shift={shift:+.4f}  ~= {elo_equiv:+.1f} Elo points")

    print(f"\n{'=' * 100}\nRybakina's actual bucket: retired 2026-08-20, US Open starts ~2026-08-30 -> ~10 days\n"
          f"{'=' * 100}")
    print("That's UNDER 14 days, not the 14-30d bucket the current -100 flat override might suggest - "
          "see the under_14d row above for the actual fitted number, if it validated.")


if __name__ == "__main__":
    main()
