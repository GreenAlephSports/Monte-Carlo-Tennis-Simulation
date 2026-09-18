# US Open + Canadian Open 2026 backtest: results, bucket analysis, and the Rybakina sensitivity check

Standalone summary of the full sequence run under the locked methodology
(`us_open_2026_backtest_methodology.md`, commit 82e0211). Every number below is read directly off a
run already made — nothing here is re-estimated, re-fit, or smoothed. Primary metrics (Brier,
log-loss vs. market) reported before secondary (ROI, win rate), per Section 3 of the locked
methodology.

## 1. The two backtests, as run

**Data**: filtered out of the full-season `data/2026_{atp,wta}_raw.xlsx.xlsx` files, verified before
use — single clean `Tournament` string ("US Open" / "Canadian Open", no naming variants), single
`Location` per event (New York / Montreal / Toronto), full round coverage through the Final, zero
null `MaxW`/`MaxL`. This replaced an earlier cached file (`data/atp_2026_usopen_tennisdata.csv`)
that was confirmed to actually contain Australian Open matches under a misleading filename.

**Methodology, confirmed at import time before running, not just by reading source**:
`MIN_HARD_MATCHES=30`, `MIN_EDGE_PP=10.0`, `FLAT_STAKE=1.0`, `KELLY_FRACTIONS=[0.25, 0.5]`,
`REFERENCE_BANKROLL=100.0` — imported unchanged from `cincinnati_paper_trading_backtest_tennisdata.py`,
matching the locked doc exactly. `edge_pp` is computed against the de-vigged consensus market
probability and kept fully separate from `real_decimal_odds` (MaxW/MaxL), which is used only at
settlement — the grading-price fix is genuinely present in the code that ran, verified by source
inspection of `build_opportunity_rows_for_tour`.

### US Open 2026 (both tours, every round)

Primary — Brier / log-loss, model vs. market:

| | Unfiltered (n=252) | Filtered (n=30) |
|---|---|---|
| Brier: Model / Market | 0.1777 / 0.1672 | 0.2079 / 0.1891 |
| Log-loss: Model / Market | 0.5311 / 0.5104 | 0.5981 / 0.5610 |
| Model − Market (Brier), 95% CI | +0.0105 [−0.0002, +0.0225] | +0.0188 [−0.0232, +0.0633] |

Secondary — ROI, real MaxW/MaxL settlement:

| | Unfiltered (n=252) | Filtered (n=30) |
|---|---|---|
| Win rate | 44.4% (112/252) | 46.7% (14/30) |
| Flat ROI | −5.5% | −1.2% |
| Kelly 0.25x ROI | −1.3% | −1.5% |

### Canadian Open 2026 (Montreal ATP + Toronto WTA)

Primary:

| | Unfiltered (n=190) | Filtered (n=30) |
|---|---|---|
| Brier: Model / Market | 0.2178 / 0.2094 | 0.2946 / 0.2856 |
| Log-loss: Model / Market | 0.6266 / 0.6039 | 0.8102 / 0.7724 |
| Model − Market (Brier), 95% CI | +0.0084 [−0.0042, +0.0223] | +0.0090 [−0.0393, +0.0652] |

Secondary:

| | Unfiltered (n=190) | Filtered (n=30) |
|---|---|---|
| Win rate | 48.9% (93/190) | 50.0% (15/30) |
| Flat ROI | −3.8% | +15.7% |
| Kelly 0.25x ROI | −2.2% | +6.6% |

**Bottom line on both events, unadjusted**: model calibration is worse than the market's on every
cut (positive Model − Market gap throughout), though most CIs include zero at this sample size.
Filtered ROI is positive for Canadian Open, roughly flat/negative for US Open — noisy, small-sample
secondary numbers, reported as-is with no filter/stake changes made in response.

## 2. Bucket analysis: the top-conviction tier is overconfident

Pooling both events' filtered sets (n=60) and bucketing into thirds by Kelly 0.25x stake size
(a direct proxy for model conviction):

| Bucket | n | Stake range | Win rate | Mean predicted P(win) | Gap | Kelly ROI |
|---|---|---|---|---|---|---|
| Top third | 20 | 8.68–20.20 | 55.0% (11/20) | 79.8% | **+24.8pp** | −12.3% |
| Middle third | 20 | 5.11–8.66 | 55.0% (11/20) | 60.1% | +5.1pp | +25.4% |
| Bottom third | 20 | 3.10–5.07 | 35.0% (7/20) | 42.2% | +7.2pp | +7.9% |

The top third does not win more often than the middle third — both sit at 55.0%. Point-biserial
correlations: Kelly stake vs. win, r=+0.271 (p=0.036, real but weak); `edge_pp` vs. win, r=+0.039
(p=0.769, not significant); `model_prob` vs. win, r=+0.258 (p=0.046). The 8 largest individual
losses by Kelly P&L (−9.2 to −15.2 units each) are all in the top third; 9 of its 20 bets lost
outright.

**Diagnosis, stated precisely**: not "the model has no edge" (`model_prob` does correlate with
winning) — the specific problem is that **the model's most confident bets are its most
overconfident ones**, and `edge_pp` inherits that miscalibration since it's computed directly off
`model_prob`, sizing bets up largest exactly where the model's certainty is least trustworthy.

Separately verified and ruled out as a cause: the live-export pipeline's known slot-ordering bug
(`_build_current_matches`, fixed 2026-09-07, commit `d08e3f5`) does not apply here — this backtest's
`build_opportunity_rows_for_tour` never imports `consolidated_export.py` or touches ESPN
slot_a/slot_b data at all; it reads `Winner`/`Loser` directly from the historical CSV. Confirmed
empirically, not just by inspection: all three Khachanov matchups in the top-conviction bucket
(Bonzi, Auger-Aliassime, Tien) were recomputed fresh via independent `win_probability()` calls in
both slot orders — each pair sums to exactly 1.0 and matches the stored `model_prob` to full float
precision.

## 3. Rybakina health-adjustment sensitivity — and why it wasn't applied

`overrides.yaml` carries a disclosed `-100` Elo penalty for Rybakina E. ("Retired mid-match at
Cincinnati (2026-08-20), 3 weeks pre-tournament"). Confirmed empirically that this override has **no
effect on the backtest as run**: removing the entry and recomputing `win_probability('Rybakina E.',
'Osaka N.', ...)` returned 0.7619677973660435 — identical to the stored value to 11 decimal places.
The mechanism (`apply_health_adjustments`, a flat subtraction from `overall_elo` and every surface
Elo column) only runs inside the forward-looking bracket-simulation path
(`bracket_export.py`/`consolidated_export.py`), never in the historical `win_probability()` path
these backtest scripts use.

A sensitivity sweep was run anyway — via `win_probability`'s `_ratings=` override, no file or code
changes — across a range of possible penalty magnitudes, since −100 was itself never validated,
only disclosed as a judgment call:

| Penalty | Filtered n (both events) | Win rate | Flat ROI | Kelly 0.25x ROI |
|---|---|---|---|---|
| 0 (standing result) | 60 | 48.3% | +7.2% | +2.6% |
| −50 | 58 | 46.6% | +6.7% | **−0.4%** |
| −75 | 58 | 46.6% | +6.7% | −1.5% |
| −100 (the disclosed value) | 54 | 44.4% | +3.0% | −5.1% |
| −125 | 52 | 42.3% | +2.2% | −7.1% |
| −150 | 53 | 41.5% | +0.2% | −8.8% |
| −175 | 54 | 40.7% | −1.6% | −10.7% |

Kelly ROI flips from positive to negative already at −50 — less than half the disclosed magnitude —
and keeps degrading roughly monotonically from there. By −125, all 8 of Rybakina's filtered bets
have dropped out of the 10pp edge filter entirely, several flipping to negative EV outright.

**Not applied to the standing result, for two reasons.** First, this project's own code treats a
health adjustment as "a disclosed human judgment call... NOT a statistical correction — no
magnitude is ever fit from data here" (`apply_health_adjustments` docstring), deliberately separate
from the model being evaluated; threading it into `win_probability()` now, after already knowing
which specific bets it would affect, is exactly the kind of after-the-fact model-definition change
the locked methodology exists to prevent. Second, Rybakina went **7-for-8** in these actual bets —
removing them doesn't correct a model error, it deletes real, correct calls because an unvalidated
number says they shouldn't count.

## 4. Final finding: the top bucket's real overconfidence gap is +46.3pp, not +24.8pp

All 8 of Rybakina's filtered bets sit inside the top-conviction bucket (n=20) — not a handful of
them, all of them. Removing her bets from that bucket and recomputing directly:

| | Top third, pooled (Section 2) | Top third, Rybakina excluded |
|---|---|---|
| n | 20 | **12** |
| Wins | 11 | **4** |
| Win rate | 55.0% | **33.3%** |
| Mean predicted P(win) | 79.8% | 79.6% |
| Overconfidence gap | +24.8pp | **+46.3pp** |

Three of the highest-edge calls in the entire 60-bet dataset lost outright once she's excluded:
Bonzi over Khachanov (27.2pp edge), Altmaier over Svajda (22.5pp edge), Bucsa over Uchijima (18.5pp
edge). n=12 is small enough that +46.3pp shouldn't be treated as a precise estimate — but the
direction is unambiguous.

**This is the real headline, not the pooled +24.8pp figure reported in Section 2.** The pooled
number was not wrong, but it was flattering: most of what made the top-conviction bucket look
reasonably calibrated was one player's real, strong performance, not broad accuracy at high
confidence. With that performance excluded, the model's highest-conviction predictions win at
roughly one-third the rate implied by their own stated confidence — a considerably more serious
overconfidence problem than the pooled result on its own would suggest.
