"""Pulls daily max temperature and mean relative humidity from Open-Meteo's free historical
weather archive (no API key required):

    https://archive-api.open-meteo.com/v1/archive
    ?latitude=..&longitude=..&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
    &daily=temperature_2m_max,relative_humidity_2m_mean&timezone=auto

One call per (location, date-range) - a tournament edition's whole date span is fetched in a
single request, not one request per match date, since Open-Meteo's daily endpoint already returns
one row per day in the range. Results are cached to disk (output/weather_cache.json) keyed on
(lat, lon rounded to 2dp, start_date, end_date), so a second run of the hypothesis test - or a
--tour BOTH run that revisits the same location/date-range for an ATP and WTA event sharing a
city and week - costs zero additional API calls.
"""
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CACHE_PATH = Path(__file__).resolve().parent.parent / "output" / "weather_cache.json"
DAILY_VARS = "temperature_2m_max,relative_humidity_2m_mean"


class WeatherFetchError(RuntimeError):
    pass


def _load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _cache_key(lat, lon, start_date, end_date):
    return f"{round(lat, 2)},{round(lon, 2)},{start_date},{end_date}"


def fetch_daily_weather_batch(requests_needed, throttle_seconds=0.05, timeout=15):
    """requests_needed: iterable of (lat, lon, start_date, end_date) as 'YYYY-MM-DD' strings.
    Returns {(lat, lon, start_date, end_date): {date_str: {'tmax': float, 'humidity_mean': float}}}.
    Cache-first: only the (lat, lon, range) combinations not already on disk hit the network, and
    the cache is written incrementally (after every new fetch, not just at the end) so an
    interrupted run doesn't lose progress already paid for in API calls."""
    cache = _load_cache()
    out = {}
    to_fetch = []
    for lat, lon, start_date, end_date in requests_needed:
        key = _cache_key(lat, lon, start_date, end_date)
        if key in cache:
            out[(lat, lon, start_date, end_date)] = cache[key]
        else:
            to_fetch.append((lat, lon, start_date, end_date))

    print(f"Weather: {len(out)} location/date-range combos already cached, "
          f"{len(to_fetch)} need a fresh Open-Meteo call")

    for i, (lat, lon, start_date, end_date) in enumerate(to_fetch):
        url = (f"{ARCHIVE_URL}?latitude={lat}&longitude={lon}&start_date={start_date}"
               f"&end_date={end_date}&daily={DAILY_VARS}&timezone=auto")
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
            data = json.loads(raw)
            daily = data["daily"]
            by_date = {
                d: {"tmax": t, "humidity_mean": h}
                for d, t, h in zip(daily["time"], daily["temperature_2m_max"], daily["relative_humidity_2m_mean"])
                if t is not None and h is not None
            }
        except (URLError, HTTPError, KeyError, json.JSONDecodeError) as e:
            print(f"WARNING: weather fetch failed for ({lat},{lon},{start_date}..{end_date}): {e} - "
                  f"skipping this location/range, its matches will be dropped from the weather test",
                  file=sys.stderr)
            by_date = {}

        key = _cache_key(lat, lon, start_date, end_date)
        cache[key] = by_date
        out[(lat, lon, start_date, end_date)] = by_date
        if (i + 1) % 50 == 0 or i + 1 == len(to_fetch):
            _save_cache(cache)
            print(f"  fetched {i + 1}/{len(to_fetch)}...")
        if throttle_seconds:
            time.sleep(throttle_seconds)

    _save_cache(cache)
    return out
