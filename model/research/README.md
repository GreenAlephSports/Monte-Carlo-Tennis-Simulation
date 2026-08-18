# research/

**Historical validation and one-off research work — not part of the live pipeline.**

Everything in this folder was built to answer a specific question about the model (does it
under/over-react to a layoff, an elite opponent, a heatwave, a player's own quirks?) using
backtests against historical match data. None of it is imported by, or required to run, the
production pipeline in `model/` — `bracket_export.py` and `api_server.py` (the actual integration
points for a live consumer) have no dependency on anything in this folder.

These scripts DO import production modules from `model/` (Elo ratings, `win_probability`, the
frozen-prediction/backtest machinery in `hybrid_simulation`, etc.) — that dependency only runs one
direction. If you're editing something in `model/`, you don't need to touch this folder for the
live pipeline to keep working; if you're changing training/backtest methodology, some of these
scripts are worth rerunning to see whether a past finding still holds.

## What's here

- **Hypothesis/backtest scripts** (`*_test.py`, `backtest_hard_court.py`): each is a self-contained
  study - frozen per-tournament-edition Elo, a chronological train/test split, held-out validation,
  and honest reporting either way. Read a given file's own module docstring for its specific
  methodology and findings; several (`layoff_*`, `elite_opponent_residual_test.py`) document a
  finding that was later found to be an artifact and superseded by a follow-up script - the
  docstrings explain which.
- **Diagnostic one-offs** (`compare_match.py`, `check_matchups.py`, `check_quarters.py`,
  `live_watch.py`, `live_poll_test.py`): small scripts written to inspect one specific export file
  or live-data question by hand, not meant to be run on a schedule.
- **Research-only supporting modules** (`tournament_locations.py`, `weather_fetch.py`): the
  geographic/weather lookup layer built specifically for `weather_upset_test.py` and
  `player_heat_heterogeneity_test.py` - nothing in the live pipeline uses these.
- **`data_loader_live.py`**: an alternate (Sackmann GitHub CSV, not Kaggle) data-loading path.
  Currently unused by anything, production or research - kept here rather than deleted since it's
  not wired into the active pipeline.

## Running something in here

These scripts still `sys.path.insert` both this folder (for sibling research imports) and its
parent `model/` (for production imports), so they run the same way they always did:

    python model/research/weather_upset_test.py --tour ATP
    python model/research/compare_match.py output/cincinnati_2026_atp_bracket_export.json
