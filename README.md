# Monte-Carlo-Simulation-Grand-Slam-Model

GreenAleph Sports take-home project. Builds surface-adjusted Elo ratings from historical ATP/WTA match data, feeds them into a Monte Carlo simulation of a 128-player Grand Slam draw to estimate each player's title odds, and compares the model's round-1 win probabilities against real sportsbook odds to see where they disagree. Has since grown a live layer on top: pulling real ESPN bracket state, pricing unsettled matches against The Odds API alongside the model, auto-rerunning on match completions, and serving the result over HTTP.

## Integration points, for an external consumer

If you're integrating with this system from outside (a bot, a dashboard, another service) rather than reading the code, there are exactly two things to look at:

- **`model/bracket_export.py`** — writes one JSON file per tournament (`output/<bracket>_bracket_export.json`) with the actual data contract: every alive player's `p_champ`/`p_sf`/`p_final`, every unsettled matchup's blended probability (`p_slot_a`/`p_slot_b`, plus the raw `market_prob_a`/`model_prob_a` components and `relative_change_pct` behind it), and pairwise `head_to_head` odds for every remaining pair. Regenerating this file is the only way results change — either run it directly, or let `live_match_watcher.py` (below) trigger it automatically. `players`/`matchups`/`head_to_head` field shapes are the stable spec; treat everything else as informational.
- **`model/api_server.py`** — a lightweight, **read-only** Flask server that serves whatever `bracket_export.py` (or `live_match_watcher.py`) most recently wrote to disk. `GET /tournaments` lists what's available; `GET /tournament/<tournament_id>` returns that tournament's latest export JSON, byte-identical to the file on disk, with a `Last-Modified` header (and `304` support) so a polling consumer doesn't need to re-fetch an unchanged body. It never triggers a simulation itself — regeneration is always a separate step. See the module docstring for the optional `API_SERVER_API_KEY` auth toggle and LAN-binding notes.

Everything else in `model/` is what produces the file those two endpoints serve, or is historical research (see `model/research/` below) that doesn't run in the live path at all.

## Folder structure

- **`data/`** — raw inputs
  - `atp_tennis.csv` / `wta_tennis.csv` — historical match results (players, surface, date, round, bookmaker odds, etc.)
  - `wimbledon_2026_atp.yaml` / `wimbledon_2026_wta.yaml` — clean 128-draw bracket files, no byes (see "Bracket YAML format" below)
- **`brackets/`** — bracket files for events smaller than a clean 128-draw (Masters 1000s etc.), which pad out to 128 slots with byes, or built live from ESPN via `espn_bracket.py`
  - `montreal_2026.yaml` — a 96-player draw (64 play Round 1, 32 byes), extracted from the official PDF via `model/parse_atp_draw.py`
- **`model/`** — the live production pipeline. Every file here is imported (directly or transitively) by `bracket_export.py` and/or `api_server.py`, or is a standalone entry point in the same live path:
  - `data_loader.py` / `data_loader_kaggle.py` — load match-history CSVs (local file, or pulled fresh via Kaggle) into a DataFrame
  - `elo_ratings.py` — computes overall + per-surface Elo ratings from match history, up to a cutoff date
  - `win_probability.py` — looks up two players' surface Elo and returns a win probability
  - `bracket_schema.py` — YAML bracket schema validation and loading
  - `bracket.py` — matches bracket player names to Elo ratings (tier-based fallback matching), byes/draw-order helpers, draw validation (including the duplicate-player check)
  - `simulate.py` — runs the Monte Carlo tournament simulations over a resolved draw, byes-aware
  - `ev_comparison.py` — de-vig math (`implied_probabilities`) shared by `bracket_export.py` and the standalone market-comparison CLI below
  - `parse_atp_draw.py` — extracts a bracket YAML from an official draw-sheet PDF (position/seed/status/bye per player); also backs `espn_bracket.py`'s name-truncation logic
  - `espn_bracket.py` — builds a bracket YAML directly from ESPN's live scoreboard for a given tour + event ID, instead of parsing a PDF; also the fix for a bracket YAML gone stale (unresolved `TBD (Qualifier N)` slots) once qualifying has concluded
  - `live_scores.py` — thin client for ESPN's public (undocumented) live scoreboard API
  - `hybrid_simulation.py` — replays real ESPN results round-by-round and simulates the rest from the current live state; the shared machinery `bracket_export.py` and the backtest scripts in `research/` both build on
  - `bracket_export.py` — **integration point**, see above
  - `live_match_watcher.py` — polls `live_scores.py` for one tournament, detects a real match-completion transition (not just any feed change), and reruns `bracket_export.py` automatically, reporting which players' odds moved and by how much. Fails fast at startup if the bracket YAML still has unresolved `TBD (Qualifier N)` placeholders, rather than polling for a while and crashing once simulation touches the bad data.
  - `api_server.py` — **integration point**, see above
  - `research/` — historical validation/backtest work, **not part of the live pipeline** — see `model/research/README.md`
- **`run_tournament.py`** — original single entry point: takes a bracket YAML path and runs the full pipeline (Elo calculation, name matching, simulation) end to end, writing a CSV. Still works for a static/offline run; `bracket_export.py` is the one that also reads live ESPN/odds state and writes the JSON the live consumers use.
- **`output/`** — generated results (created by running the scripts below)
  - `player_elo_ratings_atp.csv` / `player_elo_ratings_wta.csv` — Elo ratings per player, from `run_tournament.py` / `elo_ratings.py`
  - `<tournament>_<year>_simulation_results_<tour>.csv` — tournament-win probabilities per player, from `run_tournament.py`
  - `<bracket>_bracket_export.json` / `<bracket>_watcher_baseline.json` — the live export, from `bracket_export.py` / `live_match_watcher.py` — this is what `api_server.py` serves
  - `wimbledon_2026_ev_comparison.csv` — model vs. market probabilities, from `ev_comparison.py`

## Bracket YAML format

A bracket file fully describes one draw — no more hand-parsed markdown. Example:

```yaml
tournament: Wimbledon
year: 2026
tour: ATP              # ATP or WTA — selects the match-history dataset, ratings file, and name-alias table
surface: Grass         # Hard, Clay, or Grass — used for both Elo lookups and simulation
start_date: 2026-06-29 # Elo ratings only use matches strictly before this date, so the model never peeks into the future
players:
  - seed: 1             # null if unseeded
    name: "Sinner J."   # written in ratings-csv format (Lastname Initials.) — see matching below
    status: null         # null, or 1-3 uppercase letters: Q = qualifier, WC = wildcard, L = lucky loser, PR = protected ranking, etc.
    bye: false           # true if this player skips Round 1 and advances straight to Round 2 (see "Byes" below); omit or false for a clean draw
  - seed: null
    name: "Kecmanovic M."
    status: null
  # ... in bracket order (or tag each entry with position: N — see "Byes" below)
```

Loading a bracket runs schema validation first and fails with a clear, itemized error if required fields are missing, `surface`/`tour` aren't recognized, `start_date` isn't parseable, `status` isn't a valid code, or the player list is empty. Try it directly:

```
python model/bracket_schema.py   # (import-only; see run_tournament.py or bracket.py for CLI usage)
```

or just run a broken file through `run_tournament.py` — it prints the same validation errors and exits non-zero.

### Byes

Events smaller than a clean 128-draw (Masters 1000s, 96-player events, etc.) are still described as a single bracket file — the players who don't fit evenly into the round just get `bye: true` and skip Round 1 entirely. `run_tournament.py`:

1. Validates the bracket's *shape* right after loading it: the number of non-bye players must be even (they need to pair up), and `bye_count + non_bye_count / 2` — the field size once Round 1 is done — must be a power of two, so Round 2 onward is a normal single-elimination bracket. A malformed count (odd non-bye players, or a Round-2 field that isn't a power of two) fails with a clear error before any Elo work happens.
2. Each Monte Carlo trial then pairs up **only** the non-bye players for Round 1 (in draw order), simulates that round, and combines the winners with the bye players to form the Round 2 field — which plays out with the same round-by-round logic as any other bracket.
3. Draw order is normally just the YAML list order. A PDF-extracted bracket (see `model/parse_atp_draw.py`) instead tags every player with an explicit `position` (their slot 1..N in the full bracket) because the source PDF omits the "phantom" opponent row for a bye entirely — when every player in the file has a `position`, that's used as the authoritative draw order instead of trusting the list order.

This is a no-op for a clean draw: zero byes means every player plays Round 1, and the logic reduces to plain single-elimination (verified against `data/wimbledon_2026_atp.yaml` / `..._wta.yaml`, which carry no `bye`/`position` fields at all).

### Name matching

Player names in the YAML are expected in the same format as the `player` column in the ratings CSVs (`Lastname Initials.`, e.g. `Van De Zandschulp B.`), so matching mostly resolves on **tier 1** (exact lastname + full initials) with no fuzzy logic involved. The tier-based fallback from the old markdown parser is kept for robustness:

- **Tier 0** — manual alias override (`ATP_NAME_ALIASES` / `WTA_NAME_ALIASES` in `model/bracket.py`) for the handful of players whose bracket-file name doesn't share a common lastname/initials shape with the ratings CSV (extra surname word, dropped given name, etc.)
- **Tier 1** — exact lastname + full initials match
- **Tier 2** — lastname + first-initial match, only used when it's unambiguous (single candidate)
- **Tier 3** — no rows matched, and the player has no match history in the training window at all → seeded with a fresh `STARTING_ELO` placeholder row
- **Unresolved** — none of the above hit; `run_tournament.py` prints the offending names and exits non-zero rather than simulating with missing players

## How to run

Requires `pandas`, `pyyaml`, and `flask` (see `requirements.txt`). Run the full pipeline for one bracket from the project root:

```
python run_tournament.py data/wimbledon_2026_atp.yaml
python run_tournament.py data/wimbledon_2026_wta.yaml --simulations 5000 --output output/custom_results.csv
python run_tournament.py brackets/montreal_2026.yaml   # 96-player draw with byes
```

This recalculates Elo ratings up to the bracket's `start_date`, matches every player name to a rating, writes the ratings CSV, and runs the Monte Carlo simulation — writing results to `output/`.

Other scripts remain runnable standalone for diagnostics or ad-hoc use:

```
python model/elo_ratings.py --cutoff-date 2026-06-29   # regenerate both tours' ratings CSVs directly
python model/bracket.py data/wimbledon_2026_atp.yaml    # print name-matching tier stats for a bracket, without simulating
python model/ev_comparison.py                           # compares model probabilities to sportsbook odds
```

`data_loader.py` and `win_probability.py` are shared helpers imported by the scripts above, not meant to be run directly.

### Live pipeline

For a bracket driven by live ESPN state (rather than a static, offline bracket YAML):

```
# 1. Build (or refresh) a bracket YAML from ESPN's live scoreboard - do this whenever qualifying
#    has just concluded, or a duplicate-player/unresolved-placeholder error points back here.
python model/espn_bracket.py brackets/cincinnati_2026_atp.yaml --tour atp --event-id 718-2026 --surface Hard

# 2. Write one export JSON snapshot - the file the two integration points above read.
python model/bracket_export.py brackets/cincinnati_2026_atp.yaml

# 3. Or, keep it live: poll ESPN, auto-rerun step 2 on every real match completion, and print a
#    delta report of which players' odds moved. Runs indefinitely (Ctrl-C to stop, or --exit-after N).
python model/live_match_watcher.py brackets/cincinnati_2026_atp.yaml

# 4. Serve whatever's currently on disk over HTTP, read-only, for an external consumer.
python model/api_server.py --port 8000
```

Steps 3 and 4 are independent, long-running processes meant to run side by side (separate terminals) — the watcher only writes files, the API server only reads them.

The Odds API key (`ODDS_API_KEY` in `.env`) is optional — `bracket_export.py` falls back to the pure Elo model for any matchup it can't price, and reports which source (`odds_api` vs `model`) it used for each.
