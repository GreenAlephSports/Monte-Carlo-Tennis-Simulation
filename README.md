# Monte-Carlo-Simulation-Grand-Slam-Model

GreenAleph Sports take-home project. Builds surface-adjusted Elo ratings from historical ATP/WTA match data, feeds them into a Monte Carlo simulation of a 128-player Grand Slam draw to estimate each player's title odds, and compares the model's round-1 win probabilities against real sportsbook odds to see where they disagree.

## Folder structure

- **`data/`** — raw inputs
  - `atp_tennis.csv` / `wta_tennis.csv` — historical match results (players, surface, date, round, bookmaker odds, etc.)
  - `wimbledon_2026_atp.yaml` / `wimbledon_2026_wta.yaml` — bracket files (see "Bracket YAML format" below)
- **`model/`** — the pipeline library, driven by `run_tournament.py`
  - `data_loader.py` — loads match-history CSVs into a DataFrame
  - `elo_ratings.py` — computes overall + per-surface Elo ratings from match history, up to a cutoff date
  - `win_probability.py` — looks up two players' surface Elo and returns a win probability
  - `bracket_schema.py` — YAML bracket schema validation and loading
  - `bracket.py` — matches bracket player names to Elo ratings (tier-based fallback matching)
  - `simulate.py` — runs the Monte Carlo tournament simulations over a resolved draw
  - `ev_comparison.py` — compares model round-1 win probabilities to de-vigged sportsbook odds
  - `explore_data.py` — scratch script for poking at the raw data, not part of the pipeline
- **`run_tournament.py`** — single entry point: takes a bracket YAML path and runs the full pipeline (Elo calculation, name matching, simulation) end to end
- **`output/`** — generated results (created by running the scripts below)
  - `player_elo_ratings_atp.csv` / `player_elo_ratings_wta.csv` — Elo ratings per player, from `run_tournament.py` / `elo_ratings.py`
  - `wimbledon_2026_simulation_results_atp.csv` / `..._wta.csv` — tournament-win probabilities per player, from `run_tournament.py`
  - `wimbledon_2026_ev_comparison.csv` — model vs. market probabilities, from `ev_comparison.py`

## Bracket YAML format

A bracket file fully describes one 128-player draw — no more hand-parsed markdown. Example:

```yaml
tournament: Wimbledon
year: 2026
tour: ATP              # ATP or WTA — selects the match-history dataset, ratings file, and name-alias table
surface: Grass         # Hard, Clay, or Grass — used for both Elo lookups and simulation
start_date: 2026-06-29 # Elo ratings only use matches strictly before this date, so the model never peeks into the future
players:
  - seed: 1             # null if unseeded
    name: "Sinner J."   # written in ratings-csv format (Lastname Initials.) — see matching below
    status: null         # null, or a single uppercase letter: Q = qualifier, W = wildcard, L = lucky loser
  - seed: null
    name: "Kecmanovic M."
    status: null
  # ... exactly 128 entries, in bracket order
```

Loading a bracket runs schema validation first and fails with a clear, itemized error if required fields are missing, `surface`/`tour` aren't recognized, `start_date` isn't parseable, or the player list isn't exactly 128 entries. Try it directly:

```
python model/bracket_schema.py   # (import-only; see run_tournament.py or bracket.py for CLI usage)
```

or just run a broken file through `run_tournament.py` — it prints the same validation errors and exits non-zero.

### Name matching

Player names in the YAML are expected in the same format as the `player` column in the ratings CSVs (`Lastname Initials.`, e.g. `Van De Zandschulp B.`), so matching mostly resolves on **tier 1** (exact lastname + full initials) with no fuzzy logic involved. The tier-based fallback from the old markdown parser is kept for robustness:

- **Tier 0** — manual alias override (`ATP_NAME_ALIASES` / `WTA_NAME_ALIASES` in `model/bracket.py`) for the handful of players whose bracket-file name doesn't share a common lastname/initials shape with the ratings CSV (extra surname word, dropped given name, etc.)
- **Tier 1** — exact lastname + full initials match
- **Tier 2** — lastname + first-initial match, only used when it's unambiguous (single candidate)
- **Tier 3** — no rows matched, and the player has no match history in the training window at all → seeded with a fresh `STARTING_ELO` placeholder row
- **Unresolved** — none of the above hit; `run_tournament.py` prints the offending names and exits non-zero rather than simulating with missing players

## How to run

Requires `pandas` and `pyyaml`. Run the full pipeline for one bracket from the project root:

```
python run_tournament.py data/wimbledon_2026_atp.yaml
python run_tournament.py data/wimbledon_2026_wta.yaml --simulations 5000 --output output/custom_results.csv
```

This recalculates Elo ratings up to the bracket's `start_date`, matches every player name to a rating, writes the ratings CSV, and runs the Monte Carlo simulation — writing results to `output/`.

Other scripts remain runnable standalone for diagnostics or ad-hoc use:

```
python model/elo_ratings.py --cutoff-date 2026-06-29   # regenerate both tours' ratings CSVs directly
python model/bracket.py data/wimbledon_2026_atp.yaml    # print name-matching tier stats for a bracket, without simulating
python model/ev_comparison.py                           # compares model probabilities to sportsbook odds
```

`data_loader.py` and `win_probability.py` are shared helpers imported by the scripts above, not meant to be run directly.
