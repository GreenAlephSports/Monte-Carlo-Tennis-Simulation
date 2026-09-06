"""Fetches current pregame market odds from The Odds API for one bracket, for the "Model vs Market
Draw" artifact ONLY - kept entirely separate from the production pipeline (bracket_export.py /
consolidated_export.py / live_match_watcher.py), which had all market/odds-API blending removed on
2026-09-04 per Daron's request (see model/bracket_export.py's module docstring). Nothing here is
imported by, or writes into, any file the production pipeline reads.

This reuses the exact same de-vig math and Odds API client logic bracket_export.py used before that
removal (commit 751450e^), just relocated here and pointed at a dedicated output file so the two
paths can never collide:

    output/{bracket_stem}_artifact_market.json

Usage:
    python model/research/fetch_artifact_market_odds.py brackets/us_open_2026_atp_real.yaml
    python model/research/fetch_artifact_market_odds.py brackets/us_open_2026_wta_real.yaml
"""
import argparse
import difflib
import json
import os
import re
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bracket_schema import load_bracket_yaml  # noqa: E402
from ev_comparison import implied_probabilities  # noqa: E402
from hybrid_simulation import match_espn_name_to_draw  # noqa: E402
from bracket import TOUR_CONFIG  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def _load_env():
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env()
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")

TOURNAMENT_NAME_ALIASES = {"national bank open": "canadian open"}
FUZZY_MATCH_MIN_SCORE = 0.6
_TITLE_FILLER_WORDS = {
    "open", "championship", "championships", "cup", "masters", "the", "of", "and", "club",
    "presented", "by", "international", "internazionali", "invitational",
}


def _http_get_json(url, params, timeout=15):
    query = urllib.parse.urlencode(params)
    request = Request(f"{url}?{query}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _meaningful_tokens(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _TITLE_FILLER_WORDS}


def _fuzzy_similarity(name_a, name_b):
    tokens_a, tokens_b = _meaningful_tokens(name_a), _meaningful_tokens(name_b)
    token_overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b) if (tokens_a or tokens_b) else 0.0
    char_ratio = difflib.SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio()
    return max(token_overlap, char_ratio)


def _best_fuzzy_match(query_name, candidates):
    scored = []
    for s in candidates:
        title_body = s.get("title", "").split(" ", 1)[-1]
        scored.append((_fuzzy_similarity(query_name, title_body), s))
    if not scored:
        return None, 0.0
    scored.sort(key=lambda pair: (pair[0], pair[1].get("active", False)), reverse=True)
    best_score, best_sport = scored[0]
    if best_score >= FUZZY_MATCH_MIN_SCORE:
        return best_sport, best_score
    return None, best_score


def discover_odds_sport_key(tour, tournament_name, api_key):
    try:
        sports = _http_get_json(f"{ODDS_API_BASE}/sports/", {"apiKey": api_key, "all": "true"})
    except (HTTPError, URLError) as e:
        print(f"WARNING: couldn't list The Odds API sports ({e}) - odds pricing unavailable", file=sys.stderr)
        return None

    tour_word = "ATP" if tour.upper() == "ATP" else "WTA"
    candidates = [
        s for s in sports if s.get("group") == "Tennis" and s.get("title", "").upper().startswith(tour_word)
    ]
    chosen, score = _best_fuzzy_match(tournament_name, candidates)
    if chosen is not None:
        return chosen["key"]

    alias_body = next(
        (alias for espn_prefix, alias in TOURNAMENT_NAME_ALIASES.items()
         if tournament_name.lower().startswith(espn_prefix)),
        None,
    )
    if alias_body is None:
        print(
            f"WARNING: no Odds API tennis sport matched {tournament_name!r} ({tour}) - best fuzzy "
            f"candidate scored {score:.2f}, below {FUZZY_MATCH_MIN_SCORE} - odds unavailable",
            file=sys.stderr,
        )
        return None
    aliased_chosen, _ = _best_fuzzy_match(alias_body, candidates)
    return aliased_chosen["key"] if aliased_chosen is not None else None


def fetch_devigged_odds(sport_key, api_key):
    """Returns odds_lookup: frozenset({espn_name_a, espn_name_b}) -> {espn_name: p_win}."""
    if not sport_key or not api_key:
        return {}
    try:
        events = _http_get_json(
            f"{ODDS_API_BASE}/sports/{sport_key}/odds/",
            {"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "decimal"},
        )
    except (HTTPError, URLError) as e:
        print(f"WARNING: The Odds API request failed ({e}) - no market odds this run", file=sys.stderr)
        return {}

    odds_lookup = {}
    for event in events:
        home, away = event.get("home_team"), event.get("away_team")
        if not home or not away:
            continue
        home_prices, away_prices = [], []
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                if home in outcomes and away in outcomes:
                    home_prices.append(outcomes[home])
                    away_prices.append(outcomes[away])
        if not home_prices:
            continue
        avg_home = sum(home_prices) / len(home_prices)
        avg_away = sum(away_prices) / len(away_prices)
        prob_home, prob_away = implied_probabilities(avg_home, avg_away)
        odds_lookup[frozenset((home, away))] = {home: prob_home, away: prob_away}
    return odds_lookup


def build_market_records(bracket_path):
    """Resolves every ESPN name in the fetched odds board back to this bracket's own draw names
    (same resolver bracket_export.py used, model/hybrid_simulation.match_espn_name_to_draw) so the
    artifact-merge step in export_artifact_data.py can key strictly off draw names, never ESPN's
    raw strings - a real pairing with an unresolvable name on either side is dropped with a
    warning rather than guessed at."""
    bracket = load_bracket_yaml(bracket_path)
    tour_config = TOUR_CONFIG[bracket.tour]
    draw_names = {p.name for p in bracket.players}

    if not ODDS_API_KEY:
        print("WARNING: ODDS_API_KEY not set (.env) - no market odds fetched", file=sys.stderr)
        return []

    sport_key = discover_odds_sport_key(bracket.tour, bracket.tournament, ODDS_API_KEY)
    if not sport_key:
        return []
    odds_lookup = fetch_devigged_odds(sport_key, ODDS_API_KEY)

    records = []
    captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for pair, probs in odds_lookup.items():
        espn_a, espn_b = tuple(pair)
        draw_a = match_espn_name_to_draw(espn_a, draw_names, tour_config.name_aliases)
        draw_b = match_espn_name_to_draw(espn_b, draw_names, tour_config.name_aliases)
        if draw_a is None or draw_b is None:
            print(f"WARNING: couldn't resolve market pairing {espn_a!r} vs {espn_b!r} to draw "
                  f"names - skipped", file=sys.stderr)
            continue
        records.append({
            "player_a": draw_a,
            "player_b": draw_b,
            "market_prob_a": probs[espn_a],
            "captured_at": captured_at,
        })
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bracket_path", type=Path)
    args = parser.parse_args()

    records = build_market_records(args.bracket_path)
    output_path = OUTPUT_DIR / f"{args.bracket_path.stem}_artifact_market.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {output_path} - {len(records)} pairing(s) with a live market price")
