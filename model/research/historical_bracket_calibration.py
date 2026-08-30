"""Full-pipeline calibration harness, historical scale: reconstructs real Slam + Masters/1000
draws from the last LOOKBACK_YEARS years, runs the actual production pre-tournament Monte Carlo
simulation against each with a frozen cutoff (the tournament's own real start date - no
lookahead), and compares simulated round-reach probabilities (p_semifinal/p_final/p_champion) to
what actually happened. This validates a layer nothing else in this project's test suite touches:
every research/*_test.py file checks match-level Elo calibration at huge scale; the only thing
that ever validated the FULL deployed pipeline - real bracket construction, health adjustments,
7-round Monte Carlo compounding - was calibration_log.py, at just 4 recently-concluded
tournaments (380 matches). This gives that same question (is the deployed system, not just raw
Elo, well-calibrated?) real statistical power: ~130 editions, ~11,400 matches, both tours.

Draw source: Wikipedia only, both tours - a CONCLUDED tournament's draw article already has the
complete, final, correctly-ordered bracket, so (unlike the live pipeline in espn_bracket.py,
which needs ESPN to supply real names before Wikipedia's page for a future event is filled in)
there's nothing left for ESPN to contribute here. See espn_bracket.build_historical_bracket_
players. Article titles are resolved via MediaWiki's own search API rather than a hand-built
guess table - title conventions (hyphen vs en-dash, "Singles" vs "Men's Singles" on old combined-
draw articles) vary enough across ~15 years that guessing would silently miss real pages.

Match-result source, disclosed asymmetry: WTA stays on the live Kaggle pull (load_matches_for_
tour) - no independent alternative exists (Jeff Sackmann's tennis_wta repo is confirmed gone,
404). ATP ALSO stays on Kaggle in this first version, even though a direct cross-check (2023 US
Open: TML-Database showed 127 real matches with 6 retirement-marked scores; Kaggle showed only
120 matches, zero retirement markers) found Kaggle silently missing rows. Switching ATP to TML
requires a real name-normalization layer (TML's "Firstname Lastname" -> this project's "Lastname
X." convention, collision-safe across the FULL 1968-2026 history the way _resolve_fallback_
collisions is for a single draw) - a genuine sub-project, not a small addition, so it is
deliberately NOT bundled into this file. Flagged here, not silently deferred.

Usage:
    python model/research/historical_bracket_calibration.py --report-only   # read existing log only
    python model/research/historical_bracket_calibration.py                 # build/simulate/log
    python model/research/historical_bracket_calibration.py --limit 5       # smoke-test a few editions
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import (  # noqa: E402
    TOUR_CONFIG, DuplicatePlayerDrawError, match_draw_to_ratings, order_by_draw_position,
    split_byes, validate_bracket_structure, validate_draw,
)
from bracket_schema import PlayerEntry  # noqa: E402
from elo_ratings import LOOKBACK_YEARS, calculate_elo_ratings, load_matches_for_tour  # noqa: E402
from espn_bracket import build_historical_bracket_players  # noqa: E402
from signature_win_boost_test import ATP_TOP_TIER_SERIES, WTA_TOP_TIER_NAMES  # noqa: E402
from simulate import N_SIMULATIONS, run_simulations_tracking_all_rounds, run_simulations_tracking_milestones  # noqa: E402
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

LOG_PATH = Path(__file__).resolve().parent.parent.parent / "output" / "historical_bracket_calibration_log.csv"
LOG_COLUMNS = [
    "match_key", "tour", "tournament", "year", "player", "milestone",
    "sim_prob", "actual_reached", "logged_at",
]
MILESTONES = ["semifinal", "final", "champion"]

# a real full round-by-round replay costs the same as the existing milestone-only one (both play
# every round to completion regardless of how many checkpoints get recorded - see
# run_simulations_tracking_all_rounds' docstring), but running it TWICE (corrections on and off)
# doubles total sweep time, so this uses a smaller n than the milestone-only harness's default -
# disclosed here plainly, not silently: CI width is still dominated by the number of real editions/
# players (player-clustered bootstrap), not by per-edition simulation count, so this trade is safe.
BY_ROUND_N_SIMULATIONS = 500
BY_ROUND_LOG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "output" / "historical_bracket_calibration_by_round_log.csv"
)
BY_ROUND_LOG_COLUMNS = [
    "match_key", "tour", "tournament", "year", "player", "corrections", "depth", "round_label",
    "sim_prob", "actual_reached", "logged_at",
]
# depth 0 = champion, 1 = final, 2 = semifinal, 3 = quarterfinal - fixed, draw-size-independent
# labels for the tail; anything deeper is named by the real population size at that depth (a
# depth-4 field of 16 survivors is universally "Round of 16" regardless of the draw's total size).
ROUND_LABEL_BY_DEPTH = {0: "Champion", 1: "Final", 2: "Semifinal", 3: "Quarterfinal"}

# Kaggle's raw Round strings, ranked by how deep they are - used to derive each player's real
# furthest round reached in an edition. Only the top of the ladder matters here (semifinal and
# up); earlier labels are listed for completeness/robustness across draw sizes (56/64-draw events
# start later in this ladder than a 128-draw Slam, but the SAME string set applies either way).
ROUND_RANK = {
    "1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4, "5th Round": 5,
    "Quarterfinals": 6, "Semifinals": 7, "The Final": 8,
}


def enumerate_editions(tour, cutoff_years=LOOKBACK_YEARS):
    """Real Slam + Masters/1000 (ATP) / WTA-1000-equivalent (WTA) editions in the last
    cutoff_years, reusing the exact tier constants signature_win_boost_test.py already
    hand-curated (ATP_TOP_TIER_SERIES / WTA_TOP_TIER_NAMES) rather than re-deriving them."""
    matches = load_matches_for_tour(tour)
    window_start = matches["Date"].max() - pd.DateOffset(years=cutoff_years)
    matches = matches[matches["Date"] >= window_start]
    top = matches[matches["Series"].isin(ATP_TOP_TIER_SERIES)] if tour == "ATP" \
        else matches[matches["Tournament"].isin(WTA_TOP_TIER_NAMES)]

    editions = []
    for (tournament, year), g in top.groupby([top["Tournament"], top["Date"].dt.year]):
        editions.append({
            "tour": tour, "tournament": tournament, "year": int(year),
            "start_date": g["Date"].min(), "matches": g,
        })
    return sorted(editions, key=lambda e: e["start_date"]), matches


# Generic words that appear in almost every tournament name and carry no identifying signal -
# excluded when checking whether a candidate Wikipedia title is actually ABOUT the tournament
# being searched for, not just a coincidental year+category+"singles" match. Real bug this fixes,
# caught mid-sweep: "2023 Shanghai Masters" and "2023 Paris Masters" (Kaggle's ATP tournament
# names) both resolved to "2023 Western & Southern Open - Men's singles" - a DIFFERENT real
# event - because the old validation only checked year/category/"singles" were present anywhere
# in the title, never that the tournament's own distinguishing name was.
_GENERIC_TOURNAMENT_WORDS = {
    "open", "masters", "championships", "championship", "tennis", "international", "cup",
    "financial", "group", "presented", "by", "the", "of", "tour", "series", "1000", "premier",
    "mandatory", "women's", "men's", "wta", "atp", "bnp", "paribas",
}


def _distinguishing_words(tournament):
    words = [w.strip(".,'’").lower() for w in tournament.split()]
    return {w for w in words if w and w not in _GENERIC_TOURNAMENT_WORDS and len(w) >= 4}


def _wikipedia_search(query, srlimit=5, max_retries=4):
    """MediaWiki search, with retry/backoff on 429 (rate limit) - a real failure hit mid-sweep:
    Wikipedia throttled this project's IP after ~85 requests across roughly an hour, which isn't
    an unusually high rate - worth being defensive about even under light, spaced-out use."""
    for attempt in range(max_retries):
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": srlimit},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=15,
        )
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 30)) * (attempt + 1)
            print(f"    (Wikipedia rate limit hit, waiting {wait}s before retry {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json().get("query", {}).get("search", [])
    raise RuntimeError(f"Wikipedia search still rate-limited after {max_retries} retries: {query!r}")


# Kaggle's Tournament string is occasionally a sponsor/legacy name with ZERO lexical overlap
# with Wikipedia's real (geographic) title - no text-similarity heuristic can bridge that, so
# these are a small, explicit, disclosed override rather than a guessed pattern. Confirmed real
# cases: "BNP Paribas Masters" and "BNP Paribas Open" are literally the SAME sponsor name for two
# DIFFERENT events (Paris vs Indian Wells) in Kaggle's ATP data; "Internazionali BNL d'Italia" is
# Rome's Italian-language name, Wikipedia titles it "Italian Open". None=deliberately
# unreconstructible (ATP Finals/"Masters Cup" is round-robin, no traditional bracket exists).
KNOWN_TOURNAMENT_NAME_OVERRIDES = {
    ("ATP", "BNP Paribas Masters"): "Paris Masters",
    ("ATP", "Internazionali BNL d'Italia"): "Italian Open",
    ("WTA", "Internazionali BNL d'Italia"): "Italian Open",
    ("ATP", "Masters Cup"): None,
}


def resolve_wikipedia_title(tournament, year, tour):
    """Finds the real draw-article title via MediaWiki's own search API instead of guessing an
    exact string - title conventions (hyphen vs en-dash, 'Singles' vs 'Men's Singles' on old
    combined-draw articles) vary too much across years to hardcode reliably. Returns the best
    candidate title, or None if nothing plausible turned up (including a deliberate
    KNOWN_TOURNAMENT_NAME_OVERRIDES(...)=None entry, e.g. the ATP Finals).

    Tries three query shapes in order: with the tour's gender-category word (Slams and combined
    ATP+WTA 1000 events always split into separate 'Men's singles'/'Women's singles' articles),
    without it (ATP-only Masters 1000/Finals events are men's-only and sometimes keep the whole
    draw on the tournament's own bare page instead of a separate '- Singles' subpage), then
    without requiring 'Singles' in the title at all (some older Masters editions never split out
    a singles-only page - _fetch_wikipedia_draw_order's own RD1-template check is the real
    authority on whether a candidate page actually has a usable draw). Every candidate must
    contain a real DISTINGUISHING word from the tournament's own name (not just year+'singles')
    - see _distinguishing_words' docstring for the bug this fixes."""
    if (tour, tournament) in KNOWN_TOURNAMENT_NAME_OVERRIDES:
        override = KNOWN_TOURNAMENT_NAME_OVERRIDES[(tour, tournament)]
        if override is None:
            return None
        tournament = override

    dist_words = _distinguishing_words(tournament)
    category_word = "Men's" if tour == "ATP" else "Women's"

    queries = (
        f"{year} {tournament} {category_word} Singles",
        f"{year} {tournament} Singles",
        f"{year} {tournament}",
    )
    for require_singles, query in zip((True, True, False), queries):
        hits = _wikipedia_search(query)
        time.sleep(1)  # stay well under any reasonable rate limit across ~130 editions x 2 tours
        for hit in hits:
            title = hit["title"]
            title_lower = title.lower()
            if str(year) not in title:
                continue
            if require_singles and "singles" not in title_lower:
                continue
            if dist_words and not any(w in title_lower for w in dist_words):
                continue
            return title
    return None


def _build_edition_draw(edition):
    """Shared reconstruction pipeline (Wikipedia draw -> resolved names -> validated draw order)
    used by both the milestone-only harness and the round-depth/corrections-ablation harness, so
    the two can never silently diverge in how a draw gets built. Returns (draw, byes, surface,
    tour_config, warnings) on success, or (None, None, None, None, warnings) if this edition had
    to be skipped (every skip reason is disclosed in warnings, never silently dropped)."""
    tour, tournament, year = edition["tour"], edition["tournament"], edition["year"]
    warnings = []

    wiki_title = resolve_wikipedia_title(tournament, year, tour)
    if wiki_title is None:
        return None, None, None, None, [f"no Wikipedia draw article found for {tour} {tournament} {year}"]

    matches_history = load_matches_for_tour(tour)
    ratings_df = calculate_elo_ratings(matches_history, edition["start_date"], tour=tour)
    ratings_names = set(ratings_df["player"])

    tour_config = TOUR_CONFIG[tour]
    try:
        raw_players, unmatched = build_historical_bracket_players(
            wiki_title, ratings_names, tour_config.name_aliases
        )
    except ValueError as e:
        return None, None, None, None, [f"{tour} {tournament} {year} ({wiki_title!r}): {e}"]

    if unmatched:
        warnings.append(f"{tour} {tournament} {year}: {len(unmatched)} Wikipedia name(s) fell "
                         f"through to the fallback name-splitter (not necessarily wrong, but "
                         f"unverified against a prior ratings-csv entry): {unmatched}")

    players = [
        PlayerEntry(seed=None, name=p["name"], status=None, bye=p["bye"], position=None)
        for p in raw_players
    ]
    byes = [p.bye for p in players]
    try:
        non_bye_count, bye_count = validate_bracket_structure(byes)
    except ValueError as e:
        return None, None, None, None, [f"{tour} {tournament} {year} ({wiki_title!r}): {e}"]

    draw, resolutions, ratings_df = match_draw_to_ratings(
        players, ratings_df, tour_config.name_aliases, tour_config.match_data_path, edition["start_date"],
    )
    unmatched_tier = [r for r in resolutions if r["tier"] is None]
    if unmatched_tier:
        return None, None, None, None, [f"{tour} {tournament} {year}: unresolved bracket name(s): "
                                         f"{[r['name'] for r in unmatched_tier]}"]

    # belt-and-suspenders: every resolved draw name must actually have a ratings_df row, or
    # win_probability() crashes deep inside the simulation loop instead of failing cleanly here
    # (a real crash hit during development: a diacritic gap left a resolved name with no matching
    # row at all). Skip-and-disclose, same as every other guard in this function, rather than
    # letting one bad name take down the whole sweep.
    ratings_pool = set(ratings_df["player"])
    missing_from_ratings = [name for name in draw if name not in ratings_pool]
    if missing_from_ratings:
        return None, None, None, None, [f"{tour} {tournament} {year}: resolved name(s) with no ratings "
                                         f"row at all (likely a name-matching gap): {missing_from_ratings}"]

    tour_config.ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(tour_config.ratings_path, index=False)
    try:
        validate_draw(draw)
    except DuplicatePlayerDrawError as e:
        # a real, disclosed raw-data gap (confirmed case: the Kaggle WTA feed has 5 different
        # truncation variants for the Pliskova twins scattered across different rows - "Pliskova
        # Kar."/"Pliskova Kri."/"Pliskova Ka."/"Pliskova Kr."/"Pliskova K." - fragmenting them into
        # multiple ratings-pool identities and occasionally letting two different real players
        # collapse onto the same resolved name here. Fixing this needs a global, corpus-wide name
        # canonicalization pass (same truncation-widening logic already used per-draw, applied
        # across the whole historical dataset) - out of scope for this harness; skip and disclose
        # rather than crash the whole sweep over one bad edition.
        return None, None, None, None, [f"{tour} {tournament} {year} ({wiki_title!r}): "
                                         f"duplicate-name collision - {e}"]

    surface = edition["matches"]["Surface"].mode().iat[0]
    return draw, byes, surface, tour_config, warnings


def _real_deepest_rank(edition_matches):
    """Per-player: the highest ROUND_RANK value among their own real matches in this edition
    (the deepest round they actually played, win or lose), plus the real champion's name."""
    deepest_rank = {}
    champion_name = None
    for row in edition_matches.itertuples(index=False):
        rank = ROUND_RANK.get(row.Round)
        if rank is None:
            continue
        deepest_rank[row.Player_1] = max(deepest_rank.get(row.Player_1, 0), rank)
        deepest_rank[row.Player_2] = max(deepest_rank.get(row.Player_2, 0), rank)
        if row.Round == "The Final":
            champion_name = row.Winner
    return deepest_rank, champion_name


def _round_depth_map(edition_matches):
    """Maps each ROUND_RANK value actually observed in this edition to a "depth from final" that
    matches simulate.run_simulations_tracking_all_rounds' own depth convention exactly (depth 1 =
    reached the final, depth 2 = reached the semifinal, ...) - draw-size-independent, unlike
    ROUND_RANK's raw values, which conflate two different things: the tail rounds (QF/SF/F) are
    always a FIXED distance from the final regardless of draw size, but ROUND_RANK's absolute
    numbers assume a "5th Round" that real tour draws never actually use, silently shifting QF/SF/
    F by one relative to a true, gap-free round count.

    Fix: only use the round labels ACTUALLY PRESENT in this specific edition (real ranks
    observed), sort them, and assign sequential positions - this "compacts away" any unused gap
    (like the phantom 5th Round, or a smaller Masters draw's missing early rounds), so QF/SF/F
    always land at their correct, universal distance from the final no matter the draw size.
    Verified directly against simulate.run_simulations_tracking_all_rounds' own depth output on a
    real draw before relying on this."""
    ranks_present = sorted({ROUND_RANK[r] for r in edition_matches["Round"].unique() if r in ROUND_RANK})
    total = len(ranks_present)
    return {rank: total - i + 1 for i, rank in enumerate(ranks_present, start=1)}


def build_and_simulate_edition(edition, n_simulations=N_SIMULATIONS):
    """Returns (players_out, warnings) where players_out is a list of dicts with player/
    milestone/sim_prob/actual_reached, or (None, warnings) if this edition had to be skipped
    (disclosed via warnings, never silently dropped)."""
    tour, tournament, year = edition["tour"], edition["tournament"], edition["year"]
    draw, byes, surface, tour_config, warnings = _build_edition_draw(edition)
    if draw is None:
        return None, warnings

    champ_counts, sf_counts, final_counts = run_simulations_tracking_milestones(
        draw, byes, {}, surface, n_simulations, tour_config.ratings_path,
    )
    deepest_rank, champion_name = _real_deepest_rank(edition["matches"])

    players_out = []
    for name in draw:
        actual = {
            "semifinal": deepest_rank.get(name, 0) >= ROUND_RANK["Semifinals"],
            "final": deepest_rank.get(name, 0) >= ROUND_RANK["The Final"],
            "champion": name == champion_name,
        }
        sim = {
            "semifinal": sf_counts.get(name, 0) / n_simulations,
            "final": final_counts.get(name, 0) / n_simulations,
            "champion": champ_counts.get(name, 0) / n_simulations,
        }
        for milestone in MILESTONES:
            players_out.append({
                "tour": tour, "tournament": tournament, "year": year, "player": name,
                "milestone": milestone, "sim_prob": sim[milestone], "actual_reached": int(actual[milestone]),
            })
    return players_out, warnings


# corrections OFF = raw Elo only, matching win_probability()'s own kwarg names exactly so this
# can never silently drift from what the flags actually mean in production.
CORRECTIONS_OFF_KWARGS = {
    "use_rank_adjustment": False, "use_layoff_adjustment": False,
    "use_recent_form_adjustment": False, "use_confidence_calibration": False,
}


def build_and_simulate_edition_by_round(edition, corrections, n_simulations=BY_ROUND_N_SIMULATIONS):
    """Like build_and_simulate_edition, but tracks EVERY round depth (not just semifinal/final/
    champion) via simulate.run_simulations_tracking_all_rounds, and can run with production's
    win_probability() corrections forced off (corrections="off") to test whether tournament-level
    calibration actually depends on them, not just match-level calibration (a genuinely different
    question - see this module's ablation docstring). corrections must be "on" or "off"."""
    tour, tournament, year = edition["tour"], edition["tournament"], edition["year"]
    draw, byes, surface, tour_config, warnings = _build_edition_draw(edition)
    if draw is None:
        return None, warnings

    win_probability_kwargs = CORRECTIONS_OFF_KWARGS if corrections == "off" else None
    depth_counts = run_simulations_tracking_all_rounds(
        draw, byes, surface, n_simulations, tour_config.ratings_path,
        win_probability_kwargs=win_probability_kwargs,
    )
    deepest_rank, champion_name = _real_deepest_rank(edition["matches"])
    depth_map = _round_depth_map(edition["matches"])
    max_depth = max(depth_counts) if depth_counts else 0

    players_out = []
    for name in draw:
        player_depth = depth_map.get(deepest_rank.get(name, 0))  # smallest depth this player reached
        for depth in range(max_depth + 1):
            if depth == 0:
                actual = int(name == champion_name)
            else:
                actual = int(player_depth is not None and player_depth <= depth)
            sim_prob = depth_counts.get(depth, {}).get(name, 0) / n_simulations
            players_out.append({
                "tour": tour, "tournament": tournament, "year": year, "player": name,
                "corrections": corrections, "depth": depth,
                "round_label": ROUND_LABEL_BY_DEPTH.get(depth, f"Round of {2 ** depth}"),
                "sim_prob": sim_prob, "actual_reached": actual,
            })
    return players_out, warnings


def load_existing_log():
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(LOG_PATH)


def run(limit=None, n_simulations=N_SIMULATIONS):
    existing = load_existing_log()
    existing_keys = set(existing["match_key"]) if len(existing) else set()

    atp_editions, _ = enumerate_editions("ATP")
    wta_editions, _ = enumerate_editions("WTA")
    all_editions = atp_editions + wta_editions
    all_editions.sort(key=lambda e: e["start_date"])
    print(f"{len(atp_editions)} ATP + {len(wta_editions)} WTA = {len(all_editions)} candidate "
          f"editions in the last {LOOKBACK_YEARS} years")

    todo = [
        e for e in all_editions
        if f"{e['tour']}|{e['tournament']}|{e['year']}" not in existing_keys
    ]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(todo)} edition(s) not yet logged, processing now" +
          (f" (--limit {limit})" if limit is not None else ""))

    # written to disk after EVERY edition, not batched to the end - a 130-edition sweep is a
    # long-running process (each edition rebuilds Elo from ~5 years of history + runs a real
    # Monte Carlo sim), and losing every bit of progress to an interruption partway through would
    # waste real, expensive work. Safe to re-run/resume any time: existing_keys (loaded fresh at
    # the top of run()) already skips anything this file has logged, so a resumed run picks up
    # exactly where an interrupted one left off, same dedup convention calibration_log.py uses.
    combined = existing
    total_new_rows, skipped = 0, []
    t0 = time.time()
    for i, edition in enumerate(todo):
        key = f"{edition['tour']}|{edition['tournament']}|{edition['year']}"
        players_out, warnings = build_and_simulate_edition(edition, n_simulations=n_simulations)
        for w in warnings:
            print(f"  WARNING: {w}")
        if players_out is None:
            skipped.append(key)
            continue
        now = datetime.now(timezone.utc).isoformat()
        for row in players_out:
            row["match_key"] = key
            row["logged_at"] = now
        new_df = pd.DataFrame(players_out)[LOG_COLUMNS]
        combined = pd.concat([combined, new_df], ignore_index=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(LOG_PATH, index=False)
        total_new_rows += len(new_df)
        print(f"  [{i + 1}/{len(todo)}] {key}: {len(players_out) // len(MILESTONES)} players logged "
              f"({time.time() - t0:.0f}s elapsed)")

    if total_new_rows:
        print(f"\nAppended {total_new_rows} new row(s) ({total_new_rows // len(MILESTONES)} player-editions) "
              f"to {LOG_PATH}")
    else:
        print("\nNo new editions logged.")

    print(f"\n{len(skipped)} edition(s) skipped: {skipped}")
    print_report(combined)


def print_report(log):
    if len(log) == 0:
        print("Log is empty - nothing to report yet.")
        return
    print(f"\n{'=' * 90}\nHistorical bracket calibration: {log['match_key'].nunique()} editions, "
          f"{len(log) // len(MILESTONES)} player-edition rows\n{'=' * 90}")
    for milestone in MILESTONES:
        m = log[log["milestone"] == milestone]
        if len(m) < 10:
            continue
        assigned = m["sim_prob"].mean()
        actual = m["actual_reached"].mean()
        observed, lo, hi = cluster_bootstrap_ci(
            m.assign(_actual=m["actual_reached"], _sim=m["sim_prob"]), "_actual", "_sim", group_col="player",
        )
        print(f"\n{milestone:>10}: n={len(m)}  assigned={assigned:.1%}  actual={actual:.1%}  "
              f"gap(actual-assigned)={observed:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]")
        for tour in ("ATP", "WTA"):
            mt = m[m["tour"] == tour]
            if len(mt) < 10:
                continue
            print(f"    {tour}: n={len(mt)}  assigned={mt['sim_prob'].mean():.1%}  "
                  f"actual={mt['actual_reached'].mean():.1%}")


def load_existing_by_round_log():
    if not BY_ROUND_LOG_PATH.exists():
        return pd.DataFrame(columns=BY_ROUND_LOG_COLUMNS)
    return pd.read_csv(BY_ROUND_LOG_PATH)


def run_by_round(corrections_modes=("on", "off"), limit=None, n_simulations=BY_ROUND_N_SIMULATIONS):
    """Same edition set/dedup/incremental-write discipline as run() above, but for the round-
    depth x corrections-ablation harness. The dedup key includes `corrections` (unlike run()'s
    plain tour|tournament|year) since the SAME edition is deliberately logged once per condition -
    otherwise a corrections="off" pass would see the corrections="on" pass's key already present
    and wrongly skip it as "done"."""
    existing = load_existing_by_round_log()
    existing_keys = set(existing["match_key"]) if len(existing) else set()

    atp_editions, _ = enumerate_editions("ATP")
    wta_editions, _ = enumerate_editions("WTA")
    all_editions = sorted(atp_editions + wta_editions, key=lambda e: e["start_date"])
    print(f"{len(atp_editions)} ATP + {len(wta_editions)} WTA = {len(all_editions)} candidate "
          f"editions in the last {LOOKBACK_YEARS} years, x {len(corrections_modes)} corrections "
          f"mode(s) {corrections_modes}, n_simulations={n_simulations}")

    todo = [
        (edition, corrections)
        for corrections in corrections_modes
        for edition in all_editions
        if f"{edition['tour']}|{edition['tournament']}|{edition['year']}|{corrections}" not in existing_keys
    ]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(todo)} (edition, corrections) pair(s) not yet logged, processing now" +
          (f" (--limit {limit})" if limit is not None else ""))

    combined = existing
    total_new_rows, skipped = 0, []
    t0 = time.time()
    for i, (edition, corrections) in enumerate(todo):
        key = f"{edition['tour']}|{edition['tournament']}|{edition['year']}|{corrections}"
        players_out, warnings = build_and_simulate_edition_by_round(edition, corrections, n_simulations=n_simulations)
        for w in warnings:
            print(f"  WARNING: {w}")
        if players_out is None:
            skipped.append(key)
            continue
        now = datetime.now(timezone.utc).isoformat()
        n_depths = len({r["depth"] for r in players_out})
        for row in players_out:
            row["match_key"] = key
            row["logged_at"] = now
        new_df = pd.DataFrame(players_out)[BY_ROUND_LOG_COLUMNS]
        combined = pd.concat([combined, new_df], ignore_index=True)
        BY_ROUND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(BY_ROUND_LOG_PATH, index=False)
        total_new_rows += len(new_df)
        print(f"  [{i + 1}/{len(todo)}] {key}: {len(players_out) // n_depths} players x {n_depths} "
              f"round depths logged ({time.time() - t0:.0f}s elapsed)")

    if total_new_rows:
        print(f"\nAppended {total_new_rows} new row(s) to {BY_ROUND_LOG_PATH}")
    else:
        print("\nNo new (edition, corrections) pairs logged.")

    print(f"\n{len(skipped)} pair(s) skipped: {skipped}")
    print_report_by_round(combined)


def print_report_by_round(log):
    if len(log) == 0:
        print("Log is empty - nothing to report yet.")
        return
    print(f"\n{'=' * 100}\nROUND-DEPTH CALIBRATION: {log['match_key'].nunique()} (edition, corrections) "
          f"pairs, {len(log)} rows\n{'=' * 100}")

    for corrections in sorted(log["corrections"].unique()):
        c = log[log["corrections"] == corrections]
        print(f"\n{'-' * 100}\ncorrections = {corrections.upper()}\n{'-' * 100}")
        for depth in sorted(c["depth"].unique()):
            d = c[c["depth"] == depth]
            if len(d) < 10:
                continue
            label = d["round_label"].iloc[0]
            observed, lo, hi = cluster_bootstrap_ci(
                d.assign(_actual=d["actual_reached"], _sim=d["sim_prob"]), "_actual", "_sim", group_col="player",
            )
            print(f"  depth={depth:>2} ({label:<14}): n={len(d):>6}  assigned={d['sim_prob'].mean():.1%}  "
                  f"actual={d['actual_reached'].mean():.1%}  gap={observed:+.2%}  CI[{lo:+.2%},{hi:+.2%}]")

    # direct on-vs-off comparison, matched on (match_key without the corrections suffix, player,
    # depth) so the same real player/edition/round-depth is compared apples-to-apples between the
    # two conditions.
    if set(log["corrections"].unique()) >= {"on", "off"}:
        print(f"\n{'=' * 100}\nCORRECTIONS ON vs OFF - direct comparison per round depth\n{'=' * 100}")
        on = log[log["corrections"] == "on"].copy()
        off = log[log["corrections"] == "off"].copy()
        on["edition_key"] = on["match_key"].str.rsplit("|", n=1).str[0]
        off["edition_key"] = off["match_key"].str.rsplit("|", n=1).str[0]
        merged = on.merge(off, on=["edition_key", "player", "depth"], suffixes=("_on", "_off"))
        for depth in sorted(merged["depth"].unique()):
            d = merged[merged["depth"] == depth]
            if len(d) < 10:
                continue
            label = d["round_label_on"].iloc[0]
            gap_on, _, _ = cluster_bootstrap_ci(
                d.assign(_a=d["actual_reached_on"], _s=d["sim_prob_on"]), "_a", "_s", group_col="player")
            gap_off, _, _ = cluster_bootstrap_ci(
                d.assign(_a=d["actual_reached_off"], _s=d["sim_prob_off"]), "_a", "_s", group_col="player")
            # which condition's calibration is closer to zero (better) at this depth
            better = "ON" if abs(gap_on) < abs(gap_off) else "OFF"
            print(f"  depth={depth:>2} ({label:<14}): n={len(d):>6}  "
                  f"gap_ON={gap_on:+.2%}  gap_OFF={gap_off:+.2%}  closer-to-zero: {better}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="process at most N new editions (smoke test)")
    parser.add_argument("--simulations", type=int, default=N_SIMULATIONS)
    parser.add_argument("--by-round", action="store_true",
                         help="run the round-depth x corrections-ablation harness instead of the "
                              "milestone-only (semifinal/final/champion) one")
    parser.add_argument("--corrections", choices=["on", "off", "both"], default="both",
                         help="--by-round only: which corrections condition(s) to run")
    args = parser.parse_args()

    if args.by_round:
        modes = ("on", "off") if args.corrections == "both" else (args.corrections,)
        if args.report_only:
            print_report_by_round(load_existing_by_round_log())
        else:
            run_by_round(corrections_modes=modes, limit=args.limit,
                         n_simulations=args.simulations if args.simulations != N_SIMULATIONS else BY_ROUND_N_SIMULATIONS)
    elif args.report_only:
        print_report(load_existing_log())
    else:
        run(limit=args.limit, n_simulations=args.simulations)
