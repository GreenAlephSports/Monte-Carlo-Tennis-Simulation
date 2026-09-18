# Inconclusive hypothesis tracker

Standing reference for "is it worth re-testing yet" after each new tournament's data comes in —
built once, updated in place, not re-derived from scratch each time. Reflects the corrected
classification as of 2026-09-18 (six entries reclassified from Rejected after a full CI-audit pass
against the Closing Line Report's own CI-must-exclude-zero rule; see "Reclassified" column). The
Closing Line Report artifact itself still shows the pre-correction (2026-09-14) classification and
needs to be republished from this file — noted in the final report.

**How "n needed" is estimated**: rough scaling only, not a power calculation — assumes the bootstrap
CI half-width shrinks as ~1/sqrt(n) and the true point estimate stays exactly where it is now.
`n_needed ≈ n_current × (half_width / |point_estimate|)²`. Where the point estimate is near zero
relative to the CI width, this blows up to an unusable number or the CI is already
near-symmetric-around-zero — in both cases this is flagged as "not a power problem" rather than given
a fake precise number: more data is unlikely to resolve these, since there's no real effect trending
either direction to sharpen.

## Table

| Hypothesis | Script | n | CI / effect | Reclassified? | Re-test priority |
|---|---|---|---|---|---|
| Recent-form residual, 10-match window | `recent_form_test.py` | 43,420 | [+0.0000,+0.0003] | no | Low — near-zero point estimate, not a power problem. Already live in production regardless of status. |
| Smooth exponential recency decay (decay3) | `decay3_full_historical_test.py` | 46,778 | [+0.0000,+0.0003] | no (was ACCEPTED pre-Correction-3, now correctly Inconclusive) | Low — same boundary-artifact pattern as recent-form; not shipped, low urgency. |
| Recent-form residual, 15-match window | `recent_form_test.py` | 42,249 | [−0.0001,+0.0001] | no | Low — CI symmetric around ~0, no effect to sharpen. |
| Peak-Elo recovery, trailing-2yr variant | `peak_elo_recovery_test.py` | 40,926 | [−0.0001,+0.0001] | **yes — split out of the bundled "Peak-Elo recovery: rejected" row; the career-high-peak variant correctly stays Rejected (CI [−0.0004,−0.0000], excludes zero, wrong sign)** | Low — same near-zero pattern. |
| Rank/trajectory-lag Elo blend | `rank_trajectory_lag_test.py` | 4,813 | +0.0020 [−0.0004,+0.0049] | no | **Medium** — real positive point estimate, CI ~1.4x current n from excluding zero. Cheapest re-test on this list; worth another tournament's data. |
| Within-tournament layoff decay | `layoff_within_tournament_decay_test.py` | 1,649 | [−0.0009,+0.0019] | no | Medium — thin sample by nature (needs a real in-tournament layoff, rare); ~7.8x n needed. Revisit after several more Slam-length events accumulate. |
| Pedigree market premium | `pedigree_market_premium_test.py` | 136 | +1.6% [−3.3,+6.5] | no | Medium — ~9.4x n needed, but base population (Slam-title holders in a rematch-eligible spot) grows slowly; low natural data velocity. |
| Solid-player venue-debut underperformance | `solid_venue_debut_test.py` | 31,104 | −0.8% [−2.0,+0.5] | no | Low-medium — ~2.8x n needed but effect is small; not a priority use of a re-test slot. |
| Handedness matchup | `handedness_matchup_test.py` | 42,590 | +0.00655 [−0.0045,+0.0176] (z=+1.16) | **yes — moved from Rejected; CI contains zero** | Low — ~2.8x n needed, but point estimate is small and non-obviously trending; low expected value from more data alone. |
| One-hander vs. topspin-proxy | `onehander_topspin_test.py` | 23,490 | [−0.11389,+0.05777] (z=−0.48) | **yes — moved from Rejected; CI contains zero** | Low — ~9.4x n needed; also proxy-limited (clay-vs-hard win-rate is not a real spin measurement), so more data doesn't fix the proxy's own ceiling. |
| Height × serve-proxy interaction | `height_serve_proxy_test.py` | 31,648 | [−0.001734,+0.003780] (z=+0.48) | **yes — moved from Rejected; CI contains zero** | Low — ~7.3x n needed. |
| Layoff 2-bucket collapse, ATP | `layoff_two_bucket_test.py` | 27,553 test | [−0.0001,+0.0004] | **yes — moved from Rejected; register's own text ("reverts to straddling 0") already said this** | Low — ~2.8x n needed, small effect. |
| Layoff 2-bucket collapse, WTA | `layoff_two_bucket_test.py` | 18,565 test | [−0.0004,+0.0005] | **yes — moved from Rejected; same as ATP row** | Very low — ~81x n needed, effectively no signal to chase. |
| Points-trajectory momentum | `points_trajectory_test.py` | 45,827 | [−0.0002,+0.0003] | **yes — moved from Rejected; register's own text ("held-out [-0.0002,+0.0003]") already said this** | Very low — ~25x n needed. |
| Correction-stack ablation, heavy favorites | `correction_ablation_test.py` | 175 | most CI straddle 0 (no single reportable number) | no | Cannot estimate — needs a re-run to pull a real numeric CI before this row is even trackable quantitatively. |
| Surface-mismatch [135,175) decile soft spot | `surface_mismatch_135_175_investigation.py` | held-out decile (n not captured) | log-loss straddles 0 | no | Cannot estimate — same issue, re-run needed for a real n/CI. |
| Former-elite comeback family (4 sub-hypotheses) | `former_elite_comeback_*.py` | thin by design | not persisted numerically | no | Cannot estimate — by construction this population (former-elite player, deep comeback run) grows very slowly; check back opportunistically, not on a schedule. |

## Not on this list (needs separate handling, flagged by the repo audit, not yet resolved here)

- **"Thin-history calibration fixes (3 mechanisms)"** — currently one bundled Rejected row citing
  `thin_history_*.py`. Real component check done tonight: `thin_history_rank_blend_test.py`'s own CI
  ([−0.0030,+0.0084]) contains zero (rejection there is justified by descriptive harm in the thinnest
  bucket, not a CI exclusion), while `thin_history_platt_test.py` reports "-15.4% gap, CI excludes
  zero" for the same thinnest bucket — a genuinely different, stronger result. These should NOT share
  one row. Needs disaggregation into 2-3 separate register rows before this tracker can carry them
  correctly — flagged, not done here (scope of tonight's audit was the CI-exclusion pass, not a full
  re-derivation of this bundle).
- **WTA numbers from `pedigree_market_premium_test.py` and `surface_mismatch_market_test.py`** — the
  repo audit surfaced a disclosed, not-retroactively-fixed bug in both
  (`_layoff_bucket_edges_for()` silently used ATP layoff shifts for WTA rows). The Pedigree market
  premium row above cites `pedigree_market_premium_test.py`; its WTA-relevant component should be
  treated as suspect until that script is re-run post-fix, not re-added to this tracker's numbers as-is.

## Update cadence

Re-open this file after each new tournament's real data lands. For each row, recompute n and the
CI with the new data added, not just eyeball whether it "feels" resolved — re-run the underlying
script. Move a row to Accepted/Rejected only when the new CI actually excludes zero on a real,
held-out re-run — not on point-estimate movement alone.
