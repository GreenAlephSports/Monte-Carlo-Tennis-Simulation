# Monte-Carlo-Simulation-Grand-Slam-Model

GreenAleph Sports take-home project. Builds surface-adjusted Elo ratings from historical ATP match data, feeds them into a Monte Carlo simulation of the Wimbledon 2026 draw to estimate each player's title odds, and compares the model's round-1 win probabilities against real sportsbook odds to see where they disagree.

## Folder structure

- **`data/`** — raw inputs
  - `atp_tennis.csv` — historical ATP match results (players, surface, date, round, bookmaker odds, etc.)
  - `wimbledon_2026_draw.md` — the 128-player Wimbledon 2026 draw sheet in bracket order
- **`model/`** — the pipeline scripts (see "How to run" below for order)
  - `data_loader.py` — loads `atp_tennis.csv` into a DataFrame
  - `elo_ratings.py` — computes overall + per-surface Elo ratings from match history
  - `win_probability.py` — looks up two players' surface Elo and returns a win probability
  - `bracket.py` — parses the draw markdown and matches each name to its Elo rating
  - `simulate.py` — runs the Monte Carlo tournament simulations over the draw
  - `ev_comparison.py` — compares model round-1 win probabilities to de-vigged sportsbook odds
  - `explore_data.py` — scratch script for poking at the raw data, not part of the pipeline
- **`output/`** — generated results (created by running the scripts below)
  - `player_elo_ratings.csv` — Elo ratings per player, from `elo_ratings.py`
  - `wimbledon_2026_simulation_results.csv` — tournament-win probabilities per player, from `simulate.py`
  - `wimbledon_2026_ev_comparison.csv` — model vs. market probabilities, from `ev_comparison.py`

## How to run

Run from the project root (requires `pandas`). Each script writes its output to `output/` and needs to run after the one before it:

```
python model/elo_ratings.py       # builds player_elo_ratings.csv from match history
python model/bracket.py           # matches the draw to Elo ratings, sanity-checks name matching
python model/simulate.py          # runs the Monte Carlo simulation, writes tournament win odds
python model/ev_comparison.py     # compares model probabilities to sportsbook odds
```

`data_loader.py` and `win_probability.py` are shared helpers imported by the scripts above, not meant to be run directly.
