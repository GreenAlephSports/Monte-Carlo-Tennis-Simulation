"""Scrapes Wikipedia's `{{Infobox tennis biography}}` for playing hand + backhand type
(handedness never changes mid-career, so this is a one-time/rarely-refreshed lookup, not a live
feed) via a single combined MediaWiki API call per player - action=query with
generator=search+prop=revisions/rvsection=0 - which returns both the search hit titles AND each
hit's lead-section wikitext in one round trip (confirmed: a plain two-call search-then-parse
sequence tripped Wikimedia's per-IP rate limit hard from this environment's shared IP, see the
runtime note below - halving requests was the fix, not just throttling harder) to pull the
infobox's `plays` field, e.g.:

    | plays = Right-handed (two-handed backhand)

No API key required: https://en.wikipedia.org/w/api.php

Wikipedia article titles are full names ("Jannik Sinner"); this project's Elo ratings pool
(output/player_elo_ratings_{atp,wta}.csv) uses "Lastname I." csv form, and the csv format only
carries an initial - never a full first name - so a page URL can't just be guessed from the csv
row. Resolved instead with hybrid_simulation.match_espn_name_to_draw, the same tiered fuzzy
matcher already proven against exactly this shape of problem (ESPN's live-scoreboard full names ->
this project's draw/ratings csv names) - not a new one-off resolver - fed by a MediaWiki search
restricted to that player's lastname rather than a guessed title.

Incremental + cache-first, meant to be rerun as routine maintenance (e.g. before every Slam, as
new players enter the pool): results land in output/player_handedness.csv (player, tour, status,
hand, backhand, wikipedia_title, raw_plays), flushed to disk every FLUSH_EVERY players so an
interrupted run doesn't lose progress already paid for in API calls, and a rerun only fetches
players not already present in that file (use --retry-failed to also re-attempt rows that didn't
resolve last time, e.g. because the article didn't exist yet).

Never guesses: a player search that resolves to zero or >1 candidate csv names is left as
status=unresolved_name, an article with no {{Infobox tennis biography}} is no_infobox, an infobox
missing the `plays` field is missing_plays_field, and a `plays` value that doesn't cleanly parse
into a known hand/backhand pattern is malformed_plays (raw text kept in raw_plays for manual
review) - all of these need manual curation, none are inferred.

Usage:
    python model/research/wikipedia_handedness_scrape.py                # full ATP+WTA pool
    python model/research/wikipedia_handedness_scrape.py --tour atp
    python model/research/wikipedia_handedness_scrape.py --limit 20     # smoke test
    python model/research/wikipedia_handedness_scrape.py --retry-failed # also re-attempt misses
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

# player/article names routinely carry non-ASCII characters (Kučová, Cerúndolo, ...); Windows'
# default console codepage isn't UTF-8 and raises UnicodeEncodeError the first time one reaches
# print() - reconfigure rather than let a mid-run crash lose an otherwise-successful fetch batch
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from bracket import TOUR_CONFIG, _split_csv_name  # noqa: E402
from hybrid_simulation import match_espn_name_to_draw  # noqa: E402

API_URL = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks bot/script traffic to identify itself; no personal contact info attached here on
# purpose (see project guidance against sending user-identifying info to unrelated services) - this
# string alone is enough to satisfy the API's UA requirement.
USER_AGENT = "MonteCarloGrandSlamModel-HandednessScraper/1.0 (personal research project)"
THROTTLE_SECONDS = 2.0
FLUSH_EVERY = 25
SEARCH_LIMIT = 10

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
CACHE_PATH = OUTPUT_DIR / "player_handedness.csv"
CACHE_COLUMNS = ["player", "tour", "status", "hand", "backhand", "wikipedia_title", "raw_plays"]

# statuses that legitimately end the pipeline for a player without a hand/backhand answer - all
# require manual curation, never guessed
TERMINAL_FAILURE_STATUSES = {
    "unresolved_name", "no_infobox", "missing_plays_field", "malformed_plays", "hand_only_no_backhand",
}

HAND_PATTERNS = [
    # "-handed" is often dropped ("Right (two-handed backhand)" is at least as common on
    # real infoboxes as "Right-handed (...)") - anchor to the start of the value instead of
    # requiring the suffix, since hand is always stated first
    ("Right-handed", re.compile(r"^right(-handed)?\b", re.IGNORECASE)),
    ("Left-handed", re.compile(r"^left(-handed)?\b", re.IGNORECASE)),
    ("Ambidextrous", re.compile(r"^ambidextrous\b", re.IGNORECASE)),
]
BACKHAND_PATTERNS = [
    # both gaps flexible ([\s-]?): Wikipedia isn't consistent about which side of "handed" the
    # hyphen lands on - confirmed on Taylor Fritz's own infobox, "two handed-backhand" (hyphen
    # before backhand, not after two) - a fixed "handed backhand" would silently miss it.
    # "backhand(?:ed)?": confirmed on Daniel Blanch's infobox, "two-handed backhanded" - \bbackhand\b
    # alone can't match inside "backhanded" since there's no word boundary before its trailing "ed".
    # "(?:two|double)": confirmed on Toma Kostovic's infobox, "double-handed backhand" - a real,
    # if less common, Wikipedia synonym for the same two-handed grip
    ("two-handed", re.compile(r"\b(?:two|double)[\s-]?handed[\s-]?backhand(?:ed)?\b", re.IGNORECASE)),
    ("one-handed", re.compile(r"\bone[\s-]?handed[\s-]?backhand(?:ed)?\b", re.IGNORECASE)),
    # confirmed on Varvara Diatchenko/Kristina Kucova/Saki Hosogi's infoboxes: "two-handed both
    # sides" describes a two-handed grip on forehand AND backhand - the backhand half of that is
    # still a two-handed backhand, just phrased without the word "backhand" at all
    ("two-handed", re.compile(r"\btwo[\s-]?handed\s+both\s+sides\b", re.IGNORECASE)),
]

# [ \t]*, not \s* (which also matches newlines): a blank `| plays = ` value with no \s* guard would
# let a greedy \s* swallow the newline and leading pipe of the NEXT infobox line, silently
# capturing e.g. "|careerprizemoney = $220,596" as this player's "plays" value instead of correctly
# recognizing plays as blank - confirmed on Berrettini J. and Samrej K.'s cached raw_plays
PLAYS_FIELD_RE = re.compile(r"^[ \t]*\|[ \t]*plays[ \t]*=[ \t]*(.*?)[ \t]*$", re.MULTILINE)
INFOBOX_RE = re.compile(r"\{\{\s*Infobox tennis biography", re.IGNORECASE)
DISAMBIGUATOR_RE = re.compile(r"\s*\([^()]*\)\s*$")


def _strip_diacritics(text):
    """This project's ratings csvs come from an already-ASCII-normalized source, but Wikipedia
    article titles keep native diacritics (confirmed: "Jiří Lehečka" for csv "Lehecka J.",
    "Rafael Jódar" for "Jodar R.") - without stripping these, both the Western-order prefilter and
    match_espn_name_to_draw's own lastname comparison fail a literal character match and the
    player falls through to unresolved_name even though the right page was right there in the
    search hits."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


class WikipediaFetchError(RuntimeError):
    pass


MAX_RETRIES = 4


def _api_get(params, timeout=15):
    url = f"{API_URL}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(MAX_RETRIES):
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
            break
        except HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                # Wikipedia's Retry-After is usually short (a few seconds); back off harder each
                # retry rather than trusting a single fixed wait, since bursts of 429s tend to
                # cluster when the throttle is briefly too aggressive
                wait = float(e.headers.get("Retry-After", 5)) * (attempt + 1)
                print(f"  429 rate-limited, backing off {wait:.0f}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise WikipediaFetchError(f"request failed ({url}): {e}") from e
        except URLError as e:
            raise WikipediaFetchError(f"request failed ({url}): {e}") from e
    else:
        raise WikipediaFetchError(f"request failed ({url}): exceeded {MAX_RETRIES} retries on 429")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise WikipediaFetchError(f"non-JSON response from {url}: {raw[:200]!r}") from e


def _search_with_wikitext(query, limit=SEARCH_LIMIT):
    """One API call: search hits for `query` PLUS each hit's section-0 (lead + infobox) wikitext,
    via generator=search - avoids a separate search-then-parse round trip per candidate title.
    Returns {title: wikitext_or_None}, in the search engine's own relevance order (dict insertion
    order, Python 3.7+) since pages come back keyed by pageid with no guaranteed title order."""
    data = _api_get({
        "action": "query", "generator": "search", "gsrsearch": query, "gsrnamespace": 0,
        "gsrlimit": limit, "prop": "revisions", "rvprop": "content", "rvslots": "main",
        "rvsection": 0, "redirects": 1, "format": "json", "formatversion": 2,
        # redirects=1: without it, a bare-name redirect page (e.g. "João Fonseca" pointing at
        # "João Fonseca (tennis)") comes back as ITS OWN separate search hit whose "content" is
        # just the literal "#REDIRECT [[...]]" wikitext, not the target's infobox - confirmed this
        # produced real no_infobox false negatives for João Fonseca, Ann Li, Nuno Borges, Alexander
        # Shevchenko, and Jordan Thompson, all of whom the earlier dedup-by-clean-name fix had
        # (wrongly) resolved by picking that shorter redirect title over its disambiguated twin.
        # With redirects=1 the API resolves the redirect before prop=revisions runs, so the
        # redirect's title never shows up as a separate hit in the first place.
    })
    pages = data.get("query", {}).get("pages", [])
    # generator=search does not preserve search-relevance order in the pages list - re-sort by it
    order = {r["title"]: i for i, r in enumerate(data.get("query", {}).get("search", []) or pages)}
    pages = sorted(pages, key=lambda p: order.get(p["title"], len(pages)))
    out = {}
    for p in pages:
        revisions = p.get("revisions")
        content = revisions[0]["slots"]["main"]["content"] if revisions else None
        out[p["title"]] = content
    return out


def _fetch_wikitext_by_title(title):
    """Direct action=query&titles=... fetch (follows redirects) - used for alias-resolved players
    instead of a lastname search, since a lastname search can genuinely miss the right page:
    confirmed on Viktória Kužmová and Kateryna Kozlova, both since renamed on Wikipedia to a married
    surname ("Viktória Hrunčáková", "Kateryna Baindl") that shares no word with the ratings csv's
    lastname at all, so no lastname-based search query would ever surface it."""
    data = _api_get({
        "action": "query", "titles": title, "redirects": 1, "prop": "revisions", "rvprop": "content",
        "rvslots": "main", "rvsection": 0, "format": "json", "formatversion": 2,
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages or "missing" in pages[0]:
        return None, None
    page = pages[0]
    revisions = page.get("revisions")
    content = revisions[0]["slots"]["main"]["content"] if revisions else None
    return page["title"], content


def _resolve_wikipedia_page(csv_name, name_aliases, request_count):
    """Returns (title, wikitext, request_count) - wikitext is None if no page resolved. Tries a
    lastname+'tennis' search first (tight, usually a one-shot hit - see the Zapata Miralles/
    Kučová spot-checks this approach was validated against), then falls back to a bare-lastname
    search with a wider net if that finds nothing. A candidate title only counts if
    match_espn_name_to_draw (stripped of any disambiguator suffix, e.g. "Alex Bolt (tennis)" ->
    "Alex Bolt") maps it back to this exact csv_name - if that's true for more than one distinct
    title, or none, this is left unresolved rather than guessed.

    Tier 0, ahead of both searches: a direct-title fetch for any ATP_NAME_ALIASES/WTA_NAME_ALIASES
    entry whose VALUE is this csv_name, keyed by the exact Wikipedia article title (see
    _fetch_wikitext_by_title's docstring for why this has to bypass search rather than just
    filtering search hits)."""
    lastname, _initials = _split_csv_name(csv_name)
    lastname_words = lastname.split()

    for alias_title, target in (name_aliases or {}).items():
        if target != csv_name:
            continue
        # the codebase's other alias keys are all "Lastname I."-shaped (bracket-YAML form or
        # match_espn_name_to_draw's own ESPN-alias form) and so always split with a non-empty
        # initials part; a real Wikipedia title never does (_split_csv_name only strips a trailing
        # token when it looks like a dotted initial) - this is what tells the two key shapes apart
        # in a dict that intentionally mixes both, without a separate lookup table
        _, alias_initials = _split_csv_name(alias_title)
        if alias_initials:
            continue
        title, wikitext = _fetch_wikitext_by_title(alias_title)
        request_count += 1
        if title is not None:
            return title, wikitext, request_count

    def is_western_order(clean_title):
        """Wikipedia biography titles are essentially always Western order (Firstname ...
        Lastname); match_espn_name_to_draw also tries the reversed native-order reading (needed
        for ESPN's own feed, which does sometimes use it) - on arbitrary Wikipedia titles that
        reversed branch produces real false positives (confirmed: "Paul the Apostle" matched
        "Paul T." by reading "Paul" as the lastname and "the" as a first-initial "T" stand-in).
        This prefilter restricts candidates to the Western-order reading only, before the shared
        matcher's own (stricter, alias-aware) check runs."""
        words = [w.lower() for w in re.split(r"[\s\-]+", clean_title.strip()) if w]
        n = len(lastname_words)
        return len(words) > n and words[-n:] == lastname_words

    def matching_titles(hits):
        matched = []
        for title in hits:
            clean = _strip_diacritics(DISAMBIGUATOR_RE.sub("", title))
            if not is_western_order(clean):
                continue
            if match_espn_name_to_draw(clean, [csv_name], name_aliases) == csv_name:
                matched.append((title, clean))
        # the same real person's name can still come back as two distinct raw titles that both
        # match format even after _search_with_wikitext's redirects=1 resolves true #REDIRECT
        # pages - confirmed on "Ann Li (tennis)" + "Ann Li", "Nuno Borges (tennis)" + "Nuno
        # Borges", "João Fonseca (tennis)" + "João Fonseca". Collapsing by the post-disambiguator
        # clean name (not the raw title) treats those as the single candidate they actually are,
        # BUT must prefer the DISAMBIGUATED (longer) title, not the bare one: a bare title that
        # survives redirects=1 and still independently matches is often a genuine disambiguation
        # page (confirmed: "João Fonseca" is a 3-way dab page, not a redirect or duplicate -
        # picking it produced a real no_infobox false negative before this fix), not a duplicate
        # of the real biography.
        by_clean = {}
        for title, clean in matched:
            if clean not in by_clean or len(title) > len(by_clean[clean]):
                by_clean[clean] = title
        return list(by_clean.values())

    hits = _search_with_wikitext(f"{lastname} tennis")
    request_count += 1
    candidates = matching_titles(hits)
    if len(candidates) == 1:
        return candidates[0], hits[candidates[0]], request_count

    hits = _search_with_wikitext(lastname, limit=15)
    request_count += 1
    candidates = matching_titles(hits)
    if len(candidates) == 1:
        return candidates[0], hits[candidates[0]], request_count

    return None, None, request_count


def _classify_plays_value(raw_value):
    """Returns (hand, backhand, status) for one already-extracted `plays` field value string.
    Pulled out of _parse_plays_field so a cache reparse (see --reparse-cache) can re-run this same
    classification against already-stored raw_plays text with zero network calls, whenever only
    this function's own patterns change (as opposed to a title-resolution fix, which needs a real
    re-fetch). status is 'resolved' only when both hand and backhand parse cleanly; anything else
    is 'malformed_plays' with the raw text kept for manual review - never guessed."""
    if not raw_value.strip():
        return None, None, "missing_plays_field"

    # strip wiki markup/refs so e.g. "Right-handed<ref>...</ref> (two-handed backhand)" still
    # parses, and normalize the mojibake replacement character some older Wikipedia edits carry in
    # place of a hyphen (confirmed on Giustino L./Gunneswaran P./Travaglia S.'s cached raw_plays:
    # "two�handed backhand") back to one, rather than leaving it unparseable
    clean_value = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>|\[\[|\]\]|\{\{[^}]*\}\}", "", raw_value)
    clean_value = clean_value.replace("�", "-")

    hand = next((label for label, pat in HAND_PATTERNS if pat.search(clean_value)), None)
    backhand = next((label for label, pat in BACKHAND_PATTERNS if pat.search(clean_value)), None)

    if hand and backhand:
        return hand, backhand, "resolved"
    if hand and not backhand:
        # genuinely common on lower-profile players' infoboxes (confirmed: dozens of cases, e.g.
        # Fanselow S., Kraus S. - Wikipedia's own article simply never states a backhand type) -
        # distinct from malformed_plays, which means the text IS there but didn't parse; this is
        # "nothing to parse", a real data gap that still needs manual/other-source backhand lookup
        return hand, None, "hand_only_no_backhand"
    return hand, backhand, "malformed_plays"


def _parse_plays_field(wikitext):
    """Returns (hand, backhand, raw_value, status) by locating the `plays` field in `wikitext`
    and classifying it via _classify_plays_value."""
    match = PLAYS_FIELD_RE.search(wikitext)
    if match is None:
        return None, None, None, "missing_plays_field"
    raw_value = match.group(1)
    hand, backhand, status = _classify_plays_value(raw_value)
    return hand, backhand, raw_value, status


def _resolve_one_player(csv_name, name_aliases, request_count):
    """Returns (row_dict, request_count)."""
    title, wikitext, request_count = _resolve_wikipedia_page(csv_name, name_aliases, request_count)
    if title is None:
        return {"status": "unresolved_name", "hand": "", "backhand": "",
                "wikipedia_title": "", "raw_plays": ""}, request_count

    if not wikitext or not INFOBOX_RE.search(wikitext):
        return {"status": "no_infobox", "hand": "", "backhand": "",
                "wikipedia_title": title, "raw_plays": ""}, request_count

    hand, backhand, raw_value, status = _parse_plays_field(wikitext)
    return {
        "status": status, "hand": hand or "", "backhand": backhand or "",
        "wikipedia_title": title, "raw_plays": raw_value or "",
    }, request_count


def _load_cache():
    if CACHE_PATH.exists():
        df = pd.read_csv(CACHE_PATH, keep_default_na=False, dtype=str)
        return {(row["tour"], row["player"]): row.to_dict() for _, row in df.iterrows()}
    return {}


def _save_cache(cache):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = sorted(cache.values(), key=lambda r: (r["tour"], r["player"]))
    pd.DataFrame(rows, columns=CACHE_COLUMNS).to_csv(CACHE_PATH, index=False)


def _load_pool(tour):
    path = TOUR_CONFIG[tour].ratings_path
    return pd.read_csv(path)["player"].tolist()


def run(tours, limit=None, retry_failed=False):
    cache = _load_cache()

    to_fetch = []
    for tour in tours:
        for csv_name in _load_pool(tour):
            key = (tour, csv_name)
            already_done = key in cache and not (retry_failed and cache[key]["status"] in TERMINAL_FAILURE_STATUSES)
            if not already_done:
                to_fetch.append((tour, csv_name))

    total_pool = sum(len(_load_pool(t)) for t in tours)
    eligible_to_fetch = len(to_fetch)
    if limit is not None:
        to_fetch = to_fetch[:limit]

    print(f"Pool: {total_pool} players across {tours}. "
          f"{total_pool - eligible_to_fetch} already cached, {eligible_to_fetch} eligible to fetch"
          + (f", limited to {len(to_fetch)} this run." if limit is not None else "."))

    request_count = 0
    start = time.monotonic()
    for i, (tour, csv_name) in enumerate(to_fetch):
        name_aliases = TOUR_CONFIG[tour].name_aliases
        try:
            row, request_count = _resolve_one_player(csv_name, name_aliases, request_count)
        except WikipediaFetchError as e:
            print(f"WARNING: fetch failed for {tour} {csv_name!r}: {e} - leaving unresolved this run",
                  file=sys.stderr)
            row = {"status": "fetch_error", "hand": "", "backhand": "", "wikipedia_title": "", "raw_plays": ""}

        cache[(tour, csv_name)] = {"player": csv_name, "tour": tour, **row}

        if (i + 1) % FLUSH_EVERY == 0 or i + 1 == len(to_fetch):
            _save_cache(cache)
            elapsed = time.monotonic() - start
            print(f"  {i + 1}/{len(to_fetch)} fetched ({request_count} API requests, "
                  f"{elapsed:.0f}s elapsed)...")

        time.sleep(THROTTLE_SECONDS)

    elapsed = time.monotonic() - start
    _save_cache(cache)

    fetched_this_run = [cache[k] for k in cache if k in set(to_fetch)]
    _report(cache, tours, fetched_this_run, request_count, elapsed)


def reparse_cache():
    """Re-runs _classify_plays_value against every already-cached malformed_plays row's stored
    raw_plays text, with zero network calls - for when a HAND_PATTERNS/BACKHAND_PATTERNS fix (like
    the bare "Right"/"Left" and "backhanded" cases found in the first full-pool run) can reclassify
    already-captured text without needing to re-fetch the page. A row whose raw_plays itself was
    wrong (starts with "|" - the newline-crossing regex bug's signature, fixed in PLAYS_FIELD_RE
    but not retroactively for already-cached rows) can't be fixed this way and needs a real
    --retry-failed re-fetch instead; left untouched here rather than silently miscounted as fixed."""
    cache = _load_cache()
    before = Counter(r["status"] for r in cache.values())

    changed = 0
    still_corrupted_raw = 0
    for key, row in cache.items():
        if row["status"] != "malformed_plays":
            continue
        if row["raw_plays"].lstrip().startswith("|"):
            still_corrupted_raw += 1
            continue
        hand, backhand, status = _classify_plays_value(row["raw_plays"])
        if status != row["status"] or hand != (row["hand"] or None) or backhand != (row["backhand"] or None):
            changed += 1
        row["hand"], row["backhand"], row["status"] = hand or "", backhand or "", status

    _save_cache(cache)
    after = Counter(r["status"] for r in cache.values())
    print(f"Reparsed {sum(before.values())} cached rows, no network calls.")
    print(f"  malformed_plays: {before.get('malformed_plays', 0)} -> {after.get('malformed_plays', 0)}")
    print(f"  resolved:        {before.get('resolved', 0)} -> {after.get('resolved', 0)}")
    print(f"  {changed} rows reclassified; {still_corrupted_raw} malformed_plays rows have "
          f"corrupted raw_plays text (start with '|') and need --retry-failed, not a reparse.")


def _report(cache, tours, fetched_this_run, request_count, elapsed):
    pool_rows = [cache[k] for k in cache if k[0] in tours]
    counts = Counter(r["status"] for r in pool_rows)
    total = len(pool_rows)

    print(f"\n=== Coverage over full pool ({total} players, {tours}) ===")
    for status in ["resolved", "hand_only_no_backhand", "unresolved_name", "no_infobox",
                   "missing_plays_field", "malformed_plays", "fetch_error"]:
        if counts.get(status):
            print(f"  {status:20s} {counts[status]:4d}  ({counts[status] / total:.1%})")
    resolved = counts.get("resolved", 0)
    print(f"  {'TOTAL resolved':20s} {resolved:4d}  ({resolved / total:.1%})")

    if fetched_this_run:
        print(f"\n=== This run ===")
        print(f"  players fetched: {len(fetched_this_run)}")
        print(f"  API requests:    {request_count}")
        print(f"  wall time:       {elapsed:.0f}s ({elapsed / 60:.1f} min)")
        if fetched_this_run:
            print(f"  avg per player:  {elapsed / len(fetched_this_run):.2f}s")

    manual_needed = [r["player"] for r in pool_rows if r["status"] in TERMINAL_FAILURE_STATUSES]
    if manual_needed:
        print(f"\n{len(manual_needed)} players need manual curation (see status column in "
              f"{CACHE_PATH.relative_to(CACHE_PATH.parent.parent)}):")
        for name in manual_needed[:30]:
            print(f"  - {name}")
        if len(manual_needed) > 30:
            print(f"  ... and {len(manual_needed) - 30} more")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tour", choices=["atp", "wta", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="only fetch the first N new players (smoke test)")
    parser.add_argument("--retry-failed", action="store_true",
                         help="also re-attempt players whose cached status is a terminal failure")
    parser.add_argument("--reparse-cache", action="store_true",
                         help="reclassify already-cached malformed_plays rows against the current "
                              "hand/backhand patterns with zero network calls, then exit")
    args = parser.parse_args()

    if args.reparse_cache:
        reparse_cache()
        return

    tours = ["ATP", "WTA"] if args.tour == "both" else [args.tour.upper()]
    run(tours, limit=args.limit, retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()
