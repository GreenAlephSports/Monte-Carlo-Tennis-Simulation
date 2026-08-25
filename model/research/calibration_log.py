"""Persistent, growing calibration log - replaces re-deriving real-match calibration from scratch
every time (backtest_hard_court.py for concluded tournaments, live_calibration_check.py for
in-progress ones) with one running dataset on disk that both accumulate INTO. Every real, decided
match this project has ever checked - Montreal, Toronto, Cincinnati, and whatever gets added later
- lives here as one row: which player the model favored, the probability it assigned (using
win_probability()'s full production correction stack - rank adjustment, confidence calibration,
layoff adjustment - exactly what bracket_export.py itself uses, not a research-only raw-Elo
variant), and whether that favorite actually won.

Safe to run repeatedly against the same tournament, live or concluded: every row gets a stable
match_key (tour|tournament|year|round|sorted player pair) and existing keys are read back from the
log before appending, so a match already logged is never duplicated - rerunning against an
in-progress tournament just appends whatever newly-decided matches weren't there last time.

Two sources, matching this project's two existing calibration scripts:
  - CONCLUDED_TOURNAMENTS (Kaggle-sourced, backtest_hard_court.py's TOURNAMENTS list) - fully
    finished events where every round's real result already sits in the auto-updating Kaggle
    dataset. Date comes from Kaggle's own per-row Date.
  - LIVE_TOURNAMENTS (ESPN-sourced, live_calibration_check.py's TOURNAMENTS list) - still-running
    events; only matches ESPN has already marked 'post' (decided) get logged. Date comes from
    ESPN's competition start_time.

Storage: output/calibration_log.csv - plain CSV (not SQLite) so it's readable, diffable, and
git-trackable like every other file in output/, and the row count here will stay in the thousands
at most, nowhere near where SQLite's advantages over a CSV would start to matter.

Usage:
    python model/research/calibration_log.py                 # backfill/update everything tracked
    python model/research/calibration_log.py --report-only    # skip fetching, just print the log's current calibration read
"""
import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from backtest_hard_court import KAGGLE_ROUND_LABELS  # noqa: E402
from backtest_hard_court import TOURNAMENTS as CONCLUDED_TOURNAMENTS  # noqa: E402
from bracket import (  # noqa: E402
    TOUR_CONFIG, match_draw_to_ratings, match_name_to_pool, order_by_draw_position,
    validate_bracket_structure, validate_draw,
)
from bracket_schema import BracketValidationError, load_bracket_yaml  # noqa: E402
from elo_ratings import calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from hybrid_simulation import TOUR_SINGLES_CATEGORY, build_round_sequence, match_espn_name_to_draw  # noqa: E402
from live_calibration_check import TOURNAMENTS as LIVE_TOURNAMENTS  # noqa: E402
from live_calibration_check import _wilson_ci  # noqa: E402
from live_match_watcher import _market_price_cache_key, load_market_price_cache  # noqa: E402
from live_scores import RETIREMENT_STATUS_NAMES, LiveScoresError, extract_matches, fetch_scoreboard  # noqa: E402
from win_probability import win_probability  # noqa: E402

LOG_PATH = Path(__file__).resolve().parent.parent.parent / "output" / "calibration_log.csv"

# status_detail/ended_by_retirement are pure instrumentation - accumulating which real, decided
# matches ended via retirement/walkover (ESPN's own status.type.name, not inferred from the
# score), so that once enough occurrences build up a real held-out test can eventually be run the
# same way as every other correction in this project (see win_probability.py's fitted constants).
# No fitted adjustment exists yet - there isn't enough data to validate a magnitude against.
# live_espn-sourced rows get the real value; kaggle_concluded rows get None/NA (that historical
# dataset's Score field carries no retirement/walkover marker at all - confirmed empirically, zero
# RET/W-O tokens in either tour's Kaggle CSV), not False - "unknown" and "confirmed normal finish"
# must stay distinguishable rather than silently defaulting every historical row to "no".
LOG_COLUMNS = [
    "match_key", "source", "tour", "tournament", "year", "round_label", "round_num", "date",
    "player_a", "player_b", "favorite", "favorite_prob", "winner", "favorite_won",
    "status_detail", "ended_by_retirement", "market_prob_a", "logged_at",
]


def _parse_nullable_bool(value):
    """Handles all three shapes ended_by_retirement can arrive in: a real Python bool (freshly
    built row, pre-CSV), a CSV-round-tripped 'True'/'False' string, or NaN (kaggle_concluded rows,
    or an old log written before this column existed) - each maps to True/False/pd.NA respectively,
    never silently defaulting NA to False."""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        return value == "True"
    return bool(value)


def _pair_key(a, b):
    return "::".join(sorted((a, b)))


def _match_key(tour, tournament, year, round_num, player_a, player_b):
    return f"{tour}|{tournament}|{year}|R{round_num}|{_pair_key(player_a, player_b)}"


def load_existing_log():
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    # dates aren't parsed here via parse_dates - concluded-source rows carry a tz-naive Kaggle
    # date, live-source rows a tz-aware ESPN one, and pandas' CSV date parser can't infer a single
    # format across both; normalize_dtypes below handles the mixed-format parsing instead.
    df = pd.read_csv(LOG_PATH)
    # backward compat: a log written before status_detail/ended_by_retirement existed won't have
    # these columns - every pre-existing row's retirement status is genuinely unknown (not "no"),
    # so backfill with NA rather than erroring or defaulting to False.
    for col in ("status_detail", "ended_by_retirement", "market_prob_a"):
        if col not in df.columns:
            df[col] = pd.NA
    return normalize_dtypes(df)


def normalize_dtypes(df):
    """pd.concat'ing the empty (dtype-less, first-run) log with a freshly-built rows DataFrame
    silently downcasts every column to object - which doesn't corrupt the VALUES (favorite_prob is
    still really a float) but does break groupby().agg() + to_string(formatters=...) below, which
    pandas silently no-ops on object-dtype columns instead of applying the formatter or raising.
    Restoring real dtypes here, once, right after any concat, keeps every aggregation and report
    downstream working the same whether this is the very first row ever logged or the ten
    thousandth."""
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["round_num"] = df["round_num"].astype(int)
    df["favorite_prob"] = df["favorite_prob"].astype(float)
    df["favorite_won"] = df["favorite_won"].astype(bool)
    # nullable boolean, not plain bool: kaggle_concluded rows have no retirement data at all (see
    # LOG_COLUMNS comment above), and must round-trip through CSV as real NA, not get coerced to
    # False (a silent, wrong "confirmed not a retirement") or True (bool(nan) is truthy in Python,
    # which plain .astype(bool) would do here).
    df["ended_by_retirement"] = df["ended_by_retirement"].map(_parse_nullable_bool).astype("boolean")
    # market_prob_a: plain nullable float - NaN round-trips through CSV fine on its own (unlike
    # bool, NaN is never mistakenly truthy here), so no custom parser is needed. NaN for every
    # kaggle_concluded row (that source's own market data - Odd_1/Odd_2 - is handled separately, by
    # model_vs_market_calibration.py, not folded into this log) and for any live_espn row whose
    # pregame price was never captured by live_match_watcher.py's cache (see that module's
    # update_market_price_cache) before the match concluded.
    df["market_prob_a"] = df["market_prob_a"].astype(float)
    # format="mixed" - concluded-source rows have a plain tz-naive Kaggle Date, live-source rows
    # have a tz-aware ESPN start_time; a single strptime format can't parse both, so each row's
    # format is inferred individually instead.
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True)
    df["logged_at"] = pd.to_datetime(df["logged_at"], format="mixed", utc=True)
    return df


def _prepare_ratings(bracket):
    """Same setup every calibration path in this project does: recompute Elo up to the bracket's
    start_date, match the draw against it, and write the ratings CSV win_probability() reads from
    - so favorite_prob below is computed exactly the way a live bracket_export.py run would."""
    tour_config = TOUR_CONFIG[bracket.tour]
    players = order_by_draw_position(bracket.players)
    byes = [p.bye for p in players]
    validate_bracket_structure(byes)

    matches_history = load_matches_for_tour(bracket.tour)
    ratings_df = calculate_elo_ratings(matches_history, bracket.start_date)
    ratings_df = ratings_df.sort_values("overall_elo", ascending=False).reset_index(drop=True)

    draw, resolutions, ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, bracket.start_date
    )
    unmatched = [r for r in resolutions if r["tier"] is None]
    if unmatched:
        raise RuntimeError(f"Unmatched bracket names for {bracket.tournament}: {[r['name'] for r in unmatched]}")

    tour_config.ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(tour_config.ratings_path, index=False)

    validate_draw(draw)
    return tour_config, draw, matches_history


def _row_for_match(tour, tournament, year, round_label, round_num, date, player_a, player_b, winner,
                    surface, ratings_path, status_detail=None, market_prob_a=None):
    prob_a = win_probability(player_a, player_b, surface, ratings_path)
    favorite = player_a if prob_a >= 0.5 else player_b
    return {
        "match_key": _match_key(tour, tournament, year, round_num, player_a, player_b),
        "source": None,  # filled in by caller
        "tour": tour, "tournament": tournament, "year": year,
        "round_label": round_label, "round_num": round_num, "date": date,
        "player_a": player_a, "player_b": player_b,
        "favorite": favorite, "favorite_prob": round(max(prob_a, 1 - prob_a), 4),
        "winner": winner, "favorite_won": favorite == winner,
        # status_detail is ESPN's raw status.type.name ('STATUS_FINAL'/'STATUS_RETIRED'/
        # 'STATUS_WALKOVER') for live_espn rows, None for kaggle_concluded rows (no such field
        # exists in that historical source - see LOG_COLUMNS comment).
        "status_detail": status_detail,
        "ended_by_retirement": (
            status_detail in RETIREMENT_STATUS_NAMES if status_detail is not None else pd.NA
        ),
        # relative to player_a, same convention as favorite_prob is relative to whichever side is
        # "favorite" - None whenever no pregame price was ever cached for this pairing (see
        # live_match_watcher.update_market_price_cache) or for a kaggle_concluded row (that
        # source's market data is handled separately - see market_prob_a's own dtype comment).
        "market_prob_a": market_prob_a,
        "logged_at": datetime.now(timezone.utc),
    }


def collect_concluded_rows(bracket_path, kaggle_tournament_name, existing_keys):
    """Kaggle-sourced: window matches by tournament name + date range around start_date, same
    disambiguation build_real_results_from_kaggle in backtest_hard_court.py uses - reimplemented
    (not imported) only so each row keeps its own real Date, which that function's dict-keyed
    (pair -> winner) return throws away."""
    bracket = load_bracket_yaml(bracket_path)
    tour_config, draw, matches_history = _prepare_ratings(bracket)

    window = matches_history[
        (matches_history["Tournament"] == kaggle_tournament_name)
        & (matches_history["Date"] >= bracket.start_date - pd.Timedelta(days=2))
        & (matches_history["Date"] < bracket.start_date + pd.Timedelta(days=21))
    ]
    round_labels = {KAGGLE_ROUND_LABELS[r] for r in window["Round"].unique() if r in KAGGLE_ROUND_LABELS}
    round_sequence = build_round_sequence(round_labels)
    round_index = {label: i + 1 for i, label in enumerate(round_sequence)}

    resolved_cache = {}

    def resolve(name):
        if name not in resolved_cache:
            resolved_cache[name] = match_name_to_pool(name, draw, tour_config.name_aliases)
        return resolved_cache[name]

    rows, unresolved = [], set()
    for row in window.itertuples():
        round_label = KAGGLE_ROUND_LABELS.get(row.Round)
        round_num = round_index.get(round_label)
        if round_num is None:
            continue
        p1, p2, winner = resolve(row.Player_1), resolve(row.Player_2), resolve(row.Winner)
        if p1 is None:
            unresolved.add(row.Player_1)
        if p2 is None:
            unresolved.add(row.Player_2)
        if p1 is None or p2 is None or winner not in (p1, p2):
            continue
        key = _match_key(bracket.tour, bracket.tournament, bracket.year, round_num, p1, p2)
        if key in existing_keys:
            continue
        out = _row_for_match(
            bracket.tour, bracket.tournament, bracket.year, round_label, round_num, row.Date,
            p1, p2, winner, bracket.surface, tour_config.ratings_path,
        )
        out["source"] = "kaggle_concluded"
        rows.append(out)

    if unresolved:
        print(f"WARNING: {len(unresolved)} Kaggle player name(s) unresolved for "
              f"{bracket.tournament} - excluded from log: {sorted(unresolved)}", file=sys.stderr)
    return rows


def collect_live_rows(bracket_path, existing_keys, dates=None):
    """ESPN-sourced: only matches already status_state == 'post' with a real winner - identical
    filter to live_calibration_check.analyze_live_tournament, but keeps each match's own
    start_time instead of collapsing straight to a pair -> winner dict.

    dates: YYYYMMDD passed straight through to fetch_scoreboard - ESPN's undated ("today")
    scoreboard only finds a tournament while it's still live/recent (same caveat bracket_export.py
    and hybrid_simulation.py's own --dates flags exist for); a single date anywhere inside the
    tournament's window returns its complete event regardless of how long ago it actually
    finished - needed once a "live" tournament in LIVE_TOURNAMENTS above has since concluded."""
    bracket = load_bracket_yaml(bracket_path)
    tour_config, draw, _matches_history = _prepare_ratings(bracket)

    espn_data = fetch_scoreboard(bracket.tour.lower(), dates=dates)
    espn_matches, _ = extract_matches(espn_data)
    category = TOUR_SINGLES_CATEGORY[bracket.tour.lower()]
    tournament_matches = [
        m for m in espn_matches if m["tournament"] == bracket.tournament and m["category"] == category
    ]
    if not tournament_matches:
        raise RuntimeError(f"No live matches found for {bracket.tournament!r} / {category}")

    round_labels = {m["round"] for m in tournament_matches if m["round"]}
    round_sequence = build_round_sequence(round_labels)
    round_index = {label: i + 1 for i, label in enumerate(round_sequence)}

    price_cache = load_market_price_cache()
    n_market_backfilled = 0

    rows, unresolved = [], set()
    for m in tournament_matches:
        round_num = round_index.get(m["round"])
        if round_num is None or m["status_state"] != "post" or not m["winner"]:
            continue
        p1 = match_espn_name_to_draw(m["player_1"], draw, tour_config.name_aliases)
        p2 = match_espn_name_to_draw(m["player_2"], draw, tour_config.name_aliases)
        if p1 is None:
            unresolved.add(m["player_1"])
        if p2 is None:
            unresolved.add(m["player_2"])
        if p1 is None or p2 is None:
            continue
        winner = p1 if m["winner"] == m["player_1"] else p2
        key = _match_key(bracket.tour, bracket.tournament, bracket.year, round_num, p1, p2)
        if key in existing_keys:
            continue
        date = pd.to_datetime(m["start_time"]) if m["start_time"] else pd.NaT

        # market_prob_a, relative to p1 (raw ESPN player_1) - looked up by ESPN name, not by the
        # resolved draw name, since that's what live_match_watcher.py cached it under. The cache
        # entry itself may have been captured under either raw ESPN ordering (player_1/player_2
        # can differ poll to poll for the same real match in rare cases), so orient explicitly by
        # comparing to the cached player_a rather than assuming it matches m["player_1"].
        cache_entry = price_cache.get(
            _market_price_cache_key(bracket.tour, bracket.tournament, bracket.year, m["player_1"], m["player_2"])
        )
        market_prob_a = None
        if cache_entry is not None:
            market_prob_a = (
                cache_entry["market_prob_a"] if cache_entry["player_a"] == m["player_1"]
                else 1 - cache_entry["market_prob_a"]
            )
            n_market_backfilled += 1

        out = _row_for_match(
            bracket.tour, bracket.tournament, bracket.year, m["round"], round_num, date,
            p1, p2, winner, bracket.surface, tour_config.ratings_path,
            status_detail=m.get("status_detail"), market_prob_a=market_prob_a,
        )
        out["source"] = "live_espn"
        rows.append(out)

    if unresolved:
        print(f"WARNING: {len(unresolved)} ESPN player name(s) unresolved for "
              f"{bracket.tournament} - excluded from log: {sorted(unresolved)}", file=sys.stderr)
    if n_market_backfilled:
        print(f"NOTE: {n_market_backfilled} of {len(rows)} new {bracket.tournament} row(s) got a "
              f"pregame market_prob_a from live_match_watcher.py's price cache.", file=sys.stderr)
    return rows


def print_calibration_read(log):
    if len(log) == 0:
        print("Log is empty - nothing to report yet.")
        return
    print(f"\n{'=' * 90}\nCalibration log: {len(log)} total real, decided matches "
          f"({log['source'].value_counts().to_dict()})\n{'=' * 90}")

    by_tournament = log.groupby(["tournament", "year", "tour"]).agg(
        matches=("favorite_won", "size"),
        avg_favorite_prob=("favorite_prob", "mean"),
        favorite_win_rate=("favorite_won", "mean"),
    ).reset_index()
    print(by_tournament.to_string(index=False, formatters={
        "avg_favorite_prob": "{:.1%}".format, "favorite_win_rate": "{:.1%}".format,
    }))

    n = len(log)
    n_won = int(log["favorite_won"].sum())
    win_rate = n_won / n
    avg_prob = log["favorite_prob"].mean()
    lo, hi = _wilson_ci(n_won, n)
    print(f"\nOverall (all logged tournaments combined): model's favorite actually won "
          f"{win_rate:.1%} of {n} real matches (model's average assigned favorite probability: "
          f"{avg_prob:.1%})")
    print(f"95% Wilson CI on the actual favorite-win-rate: [{lo:.1%}, {hi:.1%}]")

    # Pure instrumentation, no adjustment yet - accumulating this until there's enough of it
    # (months, likely) to run a real held-out test the same way as every other correction in this
    # project. Only live_espn-sourced rows carry a real value; kaggle_concluded rows are excluded
    # here rather than silently counted as "not a retirement".
    live_rows = log[log["source"] == "live_espn"]
    if len(live_rows):
        n_retirement = int(live_rows["ended_by_retirement"].fillna(False).sum())
        print(f"\nRetirement/walkover-ended matches accumulated so far (live/ESPN source only - "
              f"the historical Kaggle data has no equivalent field, so it's excluded from this "
              f"count): {n_retirement} of {len(live_rows)} live-sourced matches "
              f"({n_retirement / len(live_rows):.1%}) - instrumentation only, no fitted "
              f"adjustment yet")


def run(report_only=False, dates=None):
    existing = load_existing_log()
    existing_keys = set(existing["match_key"])
    new_rows = []

    if not report_only:
        for bracket_path, _pretournament_csv, kaggle_tournament_name in CONCLUDED_TOURNAMENTS:
            try:
                rows = collect_concluded_rows(bracket_path, kaggle_tournament_name, existing_keys)
            except (BracketValidationError, RuntimeError, FileNotFoundError) as e:
                print(f"ERROR logging {bracket_path} (concluded/Kaggle): {e}", file=sys.stderr)
                continue
            print(f"{bracket_path.stem}: {len(rows)} new match(es) logged (concluded/Kaggle source)")
            new_rows.extend(rows)
            existing_keys.update(r["match_key"] for r in rows)

        for bracket_path in LIVE_TOURNAMENTS:
            try:
                rows = collect_live_rows(bracket_path, existing_keys, dates=dates)
            except (BracketValidationError, RuntimeError, LiveScoresError) as e:
                print(f"ERROR logging {bracket_path} (live/ESPN): {e}", file=sys.stderr)
                continue
            print(f"{bracket_path.stem}: {len(rows)} new match(es) logged (live/ESPN source)")
            new_rows.extend(rows)
            existing_keys.update(r["match_key"] for r in rows)

    if new_rows:
        combined = pd.concat([existing, pd.DataFrame(new_rows)[LOG_COLUMNS]], ignore_index=True)
        combined = normalize_dtypes(combined)
        combined = combined.sort_values(["tour", "tournament", "year", "round_num", "player_a"]).reset_index(drop=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(LOG_PATH, index=False)
        print(f"\nAppended {len(new_rows)} new match(es) to {LOG_PATH}")
    else:
        combined = existing
        if not report_only:
            print("\nNo new matches to log - every already-checked result was already present.")

    print_calibration_read(combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true",
                         help="skip fetching/appending - just print the current log's calibration read")
    parser.add_argument("--dates", default=None,
                         help="YYYYMMDD, passed to ESPN's ?dates= for every LIVE_TOURNAMENTS entry - "
                              "needed once a 'live' tournament has since concluded and dropped off "
                              "ESPN's undated ('today') scoreboard")
    args = parser.parse_args()
    run(report_only=args.report_only, dates=args.dates)
