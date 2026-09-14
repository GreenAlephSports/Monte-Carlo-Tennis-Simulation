# US Open 2026 backtest methodology (LOCKED)

**Locked:** 2026-09-14 01:51 UTC, commit-timestamped below. **Nothing in this document may change
after US Open results are seen or after any USO backtest/evaluation code has been run.** If a
change is genuinely needed later, it must be recorded as a dated amendment at the bottom of this
file with the reason - never as a silent edit to the sections above.

This is the direct fix for the filter-snooping critique raised on the Cincinnati backtest: the
data-quality filter and stake rule below were derived entirely from Cincinnati data
(`cincinnati_data_quality_filter_test.py`, `cincinnati_paper_trading_backtest_tennisdata.py`) and
are carried over unchanged. They are not being re-tuned, re-gridded, or re-selected on US Open
data. No grid search over filter thresholds runs against USO data at all.

## 1. Data-quality filter (unchanged from Cincinnati)

A betting opportunity is included in the primary analysis set only if **both**:

- `min_hard_matches >= 30` - both the player and the opponent must each have at least 30 "hard"
  (i.e. sufficiently well-attested) historical matches on record. Computed as
  `min(player_hard_matches, opponent_hard_matches) >= 30`.
- `min_edge_pp >= 10` - the absolute gap between model probability and market-implied probability
  must be at least 10 percentage points.

Both floors must hold simultaneously (AND, not OR). This is the exact combination selected by the
Cincinnati grid search (`MIN_HARD_MATCHES_GRID = [0,3,5,8,10,20,30]` x
`MIN_EDGE_PP_GRID = [5.0,10.0,15.0]`), reused here as a fixed constant, not re-derived.

All opportunities below this filter are still recorded and reported (see Section 4), just not
used for the headline result.

## 2. Stake rule

Two sizing methods are reported side by side, both carried over unchanged from the Cincinnati
scripts:

- **Flat stake:** 1.0 unit per bet (`FLAT_STAKE = 1.0`).
- **Fractional Kelly:** `f* = (p - q) / (1 - q)` (model prob `p`, market-implied prob `q`),
  clipped at 0 (no negative stakes, no shorting the market's side). Reported at **0.25x** and
  **0.5x** of full Kelly. Full Kelly itself is not used or reported - it is well known to be too
  aggressive for real bankroll management.
- Kelly stakes are sized against a **fixed, non-compounding 100-unit reference bankroll**
  (`REFERENCE_BANKROLL = 100.0`), not a compounding running bankroll. This isolates sizing-rule
  comparison from path-dependent bankroll effects.

## 3. Primary evaluation metrics

Evaluation order of primacy, fixed in advance:

1. **Brier score vs. market** - mean squared error of model probability vs. outcome, compared
   against the same statistic for the market-implied probability, on the *same* match set.
2. **Log-loss vs. market** - mean negative log-likelihood of model probability vs. outcome,
   compared against market-implied probability, on the same match set.
3. **ROI and win-rate** - reported for context and for continuity with the Cincinnati backtests,
   but explicitly **secondary**. Headline claims about the US Open backtest are made in terms of
   (1) and (2). ROI/win-rate numbers must not be used to justify the filter or stake rule after
   the fact - those are fixed by Sections 1-2 above, independent of how ROI/win-rate come out.

Brier and log-loss are computed both on the filtered (Section 1) set and on the full unfiltered
opportunity set, so a reader can see whether the filter is doing anything other than selecting for
favorable ROI.

## 4. Reporting requirements

Any script or output that reports US Open backtest results must show, without exception:

- Sample size (`n`) for both the filtered and unfiltered sets.
- Brier score and log-loss, model vs. market, on both sets.
- ROI and win-rate, flat and both Kelly fractions, labeled as secondary.
- The filter and stake constants exactly as fixed in Sections 1-2, either inline or by importing
  the shared constants module rather than restating literals.
- A pointer back to this file.

## 5. Scope

This document governs the **US Open 2026 backtest/evaluation** work only (analysis of settled USO
matches against the model, using the live export data already being collected in
`output/us_open_2026_*_real_*.json`). It does not govern the live-export/watcher pipeline itself
(`osaka_us_open_watch.py`, `live_watch.py`, `fetch_artifact_market_odds.py`, etc.), which continues
to run as-is to collect data - only the *analysis* of that data once matches settle is scoped by
this document.

## Amendments

*(none yet - any future change must be appended here with a date and reason, never edited into
Sections 1-4 above)*
