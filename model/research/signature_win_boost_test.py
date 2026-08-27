"""Tests a "signature win" Elo boost: a larger K-factor applied to a SPECIFIC match, gated on that
match's own significance (beating a top-5-ranked opponent, or winning the final of a Slam/Masters-
tier event), not on the player's career experience. Explicitly distinct from the already-rejected
uniform experience-based adaptive K (elo_k_factor_full_historical_test.py, K=250/(5+n)^0.4,
REJECTED - worse everywhere, worst for thin-history players specifically):
  1. Gated on the RESULT's significance, not career match count - so an established player like
     Djokovic can also qualify (beating a top-5 opponent, or winning a Slam), unlike the uniform
     version which only moved fast for players with few career matches.
  2. Doesn't touch every match a thin-history player plays - only the specific match(es) that meet
     the signature bar - so it can't reproduce the uniform version's overshoot failure mode (which
     hurt thin-history players on EVERY match, signature or not, by moving their whole rating series
     too fast).

Signature-match criterion (a match, not a player - a boosted K applies symmetrically to BOTH
players' updates for that match, same convention flat K_FACTOR already uses):
  - "beats a top-5 opponent": either player's real ATP/WTA rank (Rank_1/Rank_2, the same
    already-in-the-pipeline no-lookahead rank column elo_ratings.py itself uses for current_rank)
    is <= 5 at the time of the match. Symmetric by construction (a match a top-5 player is IN is
    "signature" for both sides), not conditioned on which side wins.
  - "wins a Slam/Masters title": Round == "The Final" AND the tournament is top-tier. ATP has a
    real Series column (Grand Slam / Masters 1000 / Masters Cup) - used directly. WTA has NO
    equivalent tier column in this dataset, so WTA top-tier status uses a hand-curated set of
    Grand Slam + WTA 1000/Premier Mandatory/Premier 5 tournament name variants (see
    WTA_TOP_TIER_NAMES) - approximate by necessity, disclosed here rather than silently assumed
    complete; a handful of renamed/relocated editions across 2006-2026 may be missed.

Boost magnitude (K_FACTOR * boost_mult for a signature match, flat K_FACTOR otherwise) is a real,
new free parameter - grid-searched on TRAIN-era signature-match rows only (minimizing train log-
loss), same discipline rank_trajectory_lag_test.py used for its own blend weight, then validated
held-out on test-era rows the chosen multiplier never touched.

Same full rigor as elo_k_factor_full_historical_test.py: both tours, full Kaggle history (~228K
player-perspective rows, ~2,800 editions), frozen-per-edition single continuous online Elo pass
(equivalent to a full rebuild for a K-only variant - no window/inclusion changes), chronological
80/20 held-out split, player-clustered bootstrap CIs. Also directly answers the concrete question
that motivated this test: replays the CURRENT production 5yr training window (cutoff 2026-08-31,
the live US Open bracket's own cutoff) under the winning boost variant, and reports Osaka's and
Andreeva's actual pre-tournament Elo compared to the current flat-K value - not just an aggregate
calibration number.

Usage:
    python model/research/signature_win_boost_test.py

FINAL VERDICT (2026-08-26): REJECTED - not added to production, and the concrete case reveals a
real mechanistic reason this idea doesn't work, not just a lack of statistical power. Grid search on
train-era signature rows shows log-loss getting MONOTONICALLY WORSE as boost_mult increases
(1.25x: 0.4690 -> 3.0x: 0.4800) - even the smallest, most conservative candidate tested is already
past the point where boosting helps, so boost_mult=1.25 (K=40) was selected only because it's the
least-bad option, not because it looked good. Held out at that already-most-favorable setting, it's
still WORSE than flat K everywhere: combined (-0.0003, CI [-0.0004,-0.0002]), on the signature-match
population specifically that this correction targets (-0.0010, CI [-0.0016,-0.0003] - the population
it was BUILT for is where it hurts hardest), both tours separately, and 3 of 6 decades (the other 3
not distinguishable, none better).

Root cause, confirmed directly on the two motivating cases: the boost is symmetric by construction
(K_FACTOR*mult applies to BOTH players in a signature match, exactly like flat K already does) - a
match against a top-5 opponent is "signature" whether you win OR lose it. Replaying the current
production window (cutoff 2026-08-31): Osaka's boosted-vs-flat pre-tournament Elo difference is
-0.7 pts (her signature wins - Wimbledon over Sabalenka +30.6, 2025 US Open R4 +27.1 - are almost
exactly offset by boosted LOSSES in other signature matches, e.g. Canadian Open 2025 Final -24.8,
Cincinnati QF 2026 -16.6). Andreeva's is worse: -15.2 pts, net WORSE than flat K, not better - her
real signature wins (French Open QF over Sabalenka +31.9, Indian Wells Final +24.3, Dubai Final
+15.8) are outweighed by a larger volume of boosted signature LOSSES (Bad Homburg -34.9, Canadian
Open 3rd round -32.9, China Open 4th round -31.6, Wuhan -31.2, Wimbledon 2nd round -28.3) - players
who face elite opponents often, including rising players building a real career against the top of
the game, lose to them more often than they beat them, so a symmetric boost mechanically amplifies
that losing record MORE than it rewards the signature wins mixed in. This is the opposite of what
the hypothesis needed: crediting Osaka's Sabalenka win or Andreeva's Roland Garros title with a
result-gated boost necessarily also over-punishes every hard-fought loss to elite competition either
of them has had along the way, and empirically the losses win. A one-sided version (boost applied
only to the WINNER's gain, not the loser's loss) would break Elo's zero-sum invariant and wasn't
tested here - a genuinely different, more invasive change outside this test's scope, not something
to build without discussing the implications with Idan first. No production change made.
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import EPS, log_loss  # noqa: E402
from elo_ratings import (  # noqa: E402
    K_FACTOR, STARTING_ELO, apply_training_window, expected_score, load_matches_for_tour,
)
from survivorship_upset_test import cluster_bootstrap_ci  # noqa: E402

TRAIN_FRACTION = 0.8
BOOST_CANDIDATES = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
CURRENT_CUTOFF = pd.Timestamp("2026-08-31")  # the live US Open bracket's own start_date/cutoff

ATP_TOP_TIER_SERIES = {"Grand Slam", "Masters 1000", "Masters Cup"}
WTA_TOP_TIER_NAMES = {
    # Grand Slams
    "Australian Open", "French Open", "US Open", "Wimbledon",
    # WTA 1000 / Premier Mandatory / Premier 5 - name variants seen across 2006-2026 in this dataset
    "BNP Paribas Open", "Pacific Life Open",                              # Indian Wells
    "Miami Open", "Sony Ericsson Open", "e-Boks Sony Ericsson Open",      # Miami
    "Mutua Madrid Open", "Mutua Madrileña Madrid Open",              # Madrid
    "Internazionali BNL d'Italia",                                       # Rome
    "Canadian Open",                                                     # Montreal/Toronto (WTA 1000)
    "Western & Southern Financial Group Women's Open",                   # Cincinnati
    "China Open", "Wuhan Open",                                          # Beijing / Wuhan
    "Qatar Total Open", "Qatar Ladies Open",                              # Doha
    "Barclays Dubai Tennis Championships", "Dubai Duty Free Tennis Championships",
    "Dubai Duty Free Women's Open",                                      # Dubai
}


def is_top_tier_tournament(tournament, tour, series=None):
    if tour == "ATP":
        return series in ATP_TOP_TIER_SERIES
    return tournament in WTA_TOP_TIER_NAMES


def is_signature_match(row, tour):
    rank1, rank2 = getattr(row, "Rank_1", None), getattr(row, "Rank_2", None)
    top5 = (pd.notna(rank1) and rank1 > 0 and rank1 <= 5) or (pd.notna(rank2) and rank2 > 0 and rank2 <= 5)
    if top5:
        return True
    if row.Round != "The Final":
        return False
    series = getattr(row, "Series", None) if tour == "ATP" else None
    return is_top_tier_tournament(row.Tournament, tour, series)


def build_frozen_predictions(df, tour, k_mode, boost_mult=None, tour_label=""):
    """Single continuously-updated online Elo pass, frozen per tournament edition (same
    equivalence argument as elo_k_factor_full_historical_test.py: a K-only variant doesn't change
    which matches are included, so this is exactly equivalent to a from-scratch-per-edition
    rebuild, at a fraction of the cost). k_mode 'flat' always uses K_FACTOR; 'boost' uses
    K_FACTOR*boost_mult for any match is_signature_match flags, flat K_FACTOR otherwise. Carries a
    'signature' flag per player-perspective row for the population breakdown."""
    df = df.sort_values("Date", kind="stable").copy()
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)
    edition_start = df.groupby("edition_id")["Date"].transform("min")
    editions = (
        df.assign(edition_start=edition_start)[["edition_id", "edition_start"]]
        .drop_duplicates().sort_values("edition_start").reset_index(drop=True)
    )

    elo = {}
    rows = []
    t0 = time.time()
    for idx, edition_id in enumerate(editions["edition_id"]):
        edition_matches = df[df["edition_id"] == edition_id]
        snap_elo = dict(elo)

        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            e1, e2 = snap_elo.get(p1, STARTING_ELO), snap_elo.get(p2, STARTING_ELO)
            pred1 = expected_score(e1, e2)
            win1 = 1 if winner == p1 else 0
            sig = is_signature_match(row, tour)
            rows.append((edition_id, row.Date, p1, p2, pred1, win1, sig))
            rows.append((edition_id, row.Date, p2, p1, 1 - pred1, 1 - win1, sig))

        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            elo.setdefault(p1, STARTING_ELO)
            elo.setdefault(p2, STARTING_ELO)
            s1 = 1.0 if winner == p1 else 0.0
            e1 = expected_score(elo[p1], elo[p2])
            if k_mode == "boost" and is_signature_match(row, tour):
                k_eff = K_FACTOR * boost_mult
            else:
                k_eff = K_FACTOR
            elo[p1] += k_eff * (s1 - e1)
            elo[p2] += k_eff * ((1 - s1) - (1 - e1))

        if (idx + 1) % 300 == 0:
            print(f"    [{tour_label}, k={k_mode}{f'x{boost_mult}' if boost_mult else ''}] "
                  f"{idx + 1}/{len(editions)} editions replayed ({time.time() - t0:.0f}s elapsed)")

    preds = pd.DataFrame(rows, columns=["edition_id", "date", "player", "opponent", "pred_win", "actual_win", "signature"])
    preds["loss"] = log_loss(preds["actual_win"].values, preds["pred_win"].values)
    print(f"    [{tour_label}, k={k_mode}{f'x{boost_mult}' if boost_mult else ''}] done: "
          f"{len(editions)} editions, {len(preds)} rows, {preds['signature'].sum()} signature rows, "
          f"{time.time() - t0:.0f}s total")
    return preds, editions


def bootstrap_verdict(long_baseline, long_variant, merge_keys=("edition_id", "date", "player", "opponent")):
    merged = long_baseline[[*merge_keys, "loss"]].merge(
        long_variant[[*merge_keys, "loss"]], on=list(merge_keys), suffixes=("_baseline", "_variant"))
    observed, lo, hi = cluster_bootstrap_ci(merged, "loss_baseline", "loss_variant", group_col="player")
    verdict = "BEATS baseline (CI excludes zero, >0)" if lo > 0 else (
        "WORSE than baseline (CI excludes zero, <0)" if hi < 0 else "NOT distinguishable (CI straddles zero)")
    return merged, observed, lo, hi, verdict


def run():
    tours = ["ATP", "WTA"]
    baseline_preds, boost_preds_by_mult = {}, {mult: {} for mult in BOOST_CANDIDATES}
    editions_by_tour = {}
    raw_matches = {}

    for tour in tours:
        matches = load_matches_for_tour(tour)
        raw_matches[tour] = matches
        print(f"\n{'#' * 90}\n{tour}: {len(matches)} total matches\n{'#' * 90}")
        preds, editions = build_frozen_predictions(matches, tour, "flat", tour_label=tour)
        baseline_preds[tour] = preds
        editions_by_tour[tour] = editions
        for mult in BOOST_CANDIDATES:
            preds, _ = build_frozen_predictions(matches, tour, "boost", boost_mult=mult, tour_label=tour)
            boost_preds_by_mult[mult][tour] = preds

    test_edition_ids = {}
    for tour in tours:
        editions = editions_by_tour[tour]
        split_idx = int(len(editions) * TRAIN_FRACTION)
        test_edition_ids[tour] = set(editions["edition_id"].iloc[split_idx:])
        print(f"\n{tour}: {len(editions)} editions; held-out test era = most recent "
              f"{len(editions) - split_idx} editions, from {editions['edition_start'].iloc[split_idx].date()}")

    # ============================================================================
    # GRID SEARCH boost_mult on TRAIN-era signature rows only, both tours combined
    # ============================================================================
    print(f"\n{'=' * 90}\nGRID SEARCH (train-era signature-match rows only, both tours combined)\n{'=' * 90}")
    best_mult, best_train_loss = None, float("inf")
    for mult in BOOST_CANDIDATES:
        parts = []
        for tour in tours:
            df = boost_preds_by_mult[mult][tour]
            train_df = df[~df["edition_id"].isin(test_edition_ids[tour])]
            parts.append(train_df[train_df["signature"]])
        train_sig = pd.concat(parts, ignore_index=True)
        loss = train_sig["loss"].mean()
        print(f"  boost_mult={mult:.2f} (K={K_FACTOR * mult:.1f}): train signature-row log-loss = "
              f"{loss:.4f} (n={len(train_sig)})")
        if loss < best_train_loss:
            best_train_loss, best_mult = loss, mult
    print(f"  -> selected boost_mult={best_mult:.2f} (K={K_FACTOR * best_mult:.1f} for a signature match)\n")

    boost_preds = boost_preds_by_mult[best_mult]

    # ============================================================================
    # HEADLINE: held-out (last 20% of editions), BOTH tours combined, ALL rows
    # ============================================================================
    print(f"{'=' * 90}\nHEADLINE - HELD-OUT TEST ERA, BOTH TOURS COMBINED, ALL ROWS "
          f"(boost_mult={best_mult:.2f})\n{'=' * 90}")
    combined_test = {}
    for label, preds_by_tour in [("A. flat (K=32, production)", baseline_preds), ("B. signature-boost", boost_preds)]:
        parts = [preds_by_tour[tour][preds_by_tour[tour]["edition_id"].isin(test_edition_ids[tour])].assign(tour=tour)
                 for tour in tours]
        combined_test[label] = pd.concat(parts, ignore_index=True)
        cdf = combined_test[label]
        print(f"\n{label}: {len(cdf)} held-out rows | mean log-loss = {cdf['loss'].mean():.4f} | "
              f"signature rows = {cdf['signature'].sum()}")

    merge_keys = ("tour", "edition_id", "date", "player", "opponent")
    merged, observed, lo, hi, verdict = bootstrap_verdict(
        combined_test["A. flat (K=32, production)"], combined_test["B. signature-boost"], merge_keys=merge_keys)
    print(f"\nB. signature-boost vs. baseline (ALL held-out rows): {len(merged)} matched rows")
    print(f"  mean log-loss improvement (baseline - variant, >0 = variant better): "
          f"{observed:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  VERDICT: {verdict}")

    # ============================================================================
    # SIGNATURE-ONLY population breakdown, held-out - the population this actually targets
    # ============================================================================
    print(f"\n{'=' * 90}\nSIGNATURE-MATCH-ONLY BREAKDOWN, held-out test era (the population this "
          f"correction actually touches)\n{'=' * 90}")
    base_sig = combined_test["A. flat (K=32, production)"][combined_test["A. flat (K=32, production)"]["signature"]]
    var_sig = combined_test["B. signature-boost"][combined_test["B. signature-boost"]["signature"]]
    print(f"Signature rows held out: {len(base_sig)}")
    if len(base_sig) >= 10:
        merged, observed, lo, hi, verdict = bootstrap_verdict(base_sig, var_sig, merge_keys=merge_keys)
        print(f"  mean log-loss improvement: {observed:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")
    else:
        print("  too few rows to bootstrap")

    # ============================================================================
    # PER-TOUR breakdown, held-out, all rows
    # ============================================================================
    print(f"\n{'=' * 90}\nPER-TOUR BREAKDOWN, held-out test era, all rows\n{'=' * 90}")
    for tour in tours:
        base_t = combined_test["A. flat (K=32, production)"][combined_test["A. flat (K=32, production)"]["tour"] == tour]
        var_t = combined_test["B. signature-boost"][combined_test["B. signature-boost"]["tour"] == tour]
        merged, observed, lo, hi, verdict = bootstrap_verdict(base_t, var_t, merge_keys=merge_keys)
        print(f"  {tour}: {len(merged)} rows, improvement {observed:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")

    # ============================================================================
    # DECADE stability breakdown, FULL period (both tours) - no train-fit params here either
    # (boost_mult IS fit on train, disclosed above - this is a stability check on the CHOSEN
    # multiplier across history, not a second independent held-out claim)
    # ============================================================================
    print(f"\n{'=' * 90}\nDECADE STABILITY BREAKDOWN, FULL PERIOD, both tours combined "
          f"(boost_mult={best_mult:.2f} fixed, chosen on train era above)\n{'=' * 90}")
    full_base = pd.concat([baseline_preds[t].assign(tour=t) for t in tours], ignore_index=True)
    full_boost = pd.concat([boost_preds[t].assign(tour=t) for t in tours], ignore_index=True)
    full_base["decade"] = (full_base["date"].dt.year // 5) * 5
    full_boost["decade"] = (full_boost["date"].dt.year // 5) * 5
    for decade in sorted(full_base["decade"].unique()):
        base_d = full_base[full_base["decade"] == decade]
        if len(base_d) < 200:
            continue
        var_d = full_boost[full_boost["decade"] == decade]
        merged = base_d[["tour", "edition_id", "date", "player", "opponent", "loss"]].merge(
            var_d[["tour", "edition_id", "date", "player", "opponent", "loss"]],
            on=["tour", "edition_id", "date", "player", "opponent"], suffixes=("_baseline", "_variant"))
        if len(merged) < 10:
            continue
        observed, lo, hi = cluster_bootstrap_ci(merged, "loss_baseline", "loss_variant", group_col="player")
        verdict = "BEATS baseline" if lo > 0 else ("WORSE than baseline" if hi < 0 else "not distinguishable")
        yr_range = f"{decade}-{decade + 4}"
        print(f"  {yr_range}: n={len(merged)}, improvement {observed:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")

    # ============================================================================
    # CONCRETE CASE: replay the CURRENT production 5yr window (cutoff 2026-08-31) under the
    # winning boost_mult, report Osaka's and Andreeva's actual pre-tournament Elo vs. current flat-K
    # ============================================================================
    print(f"\n{'=' * 90}\nCONCRETE CASE - current US Open bracket cutoff ({CURRENT_CUTOFF.date()}), "
          f"WTA, boost_mult={best_mult:.2f}\n{'=' * 90}")
    wta_matches = raw_matches["WTA"]
    windowed = apply_training_window(wta_matches, CURRENT_CUTOFF).sort_values("Date", kind="stable")

    def replay(df, k_mode, boost_mult):
        elo = {}
        deltas = {"Osaka N.": [], "Andreeva M.": []}
        for row in df.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            elo.setdefault(p1, STARTING_ELO)
            elo.setdefault(p2, STARTING_ELO)
            s1 = 1.0 if winner == p1 else 0.0
            e1 = expected_score(elo[p1], elo[p2])
            sig = is_signature_match(row, "WTA")
            k_eff = K_FACTOR * boost_mult if (k_mode == "boost" and sig) else K_FACTOR
            d1, d2 = k_eff * (s1 - e1), k_eff * ((1 - s1) - (1 - e1))
            elo[p1] += d1
            elo[p2] += d2
            for p, d in [(p1, d1), (p2, d2)]:
                if p in deltas:
                    deltas[p].append((row.Date, row.Tournament, row.Round, sig, d))
        return elo, deltas

    elo_flat, deltas_flat = replay(windowed, "flat", 1.0)
    elo_boost, deltas_boost = replay(windowed, "boost", best_mult)

    for player in ["Osaka N.", "Andreeva M."]:
        print(f"\n{player}: current-window final Elo (as of {CURRENT_CUTOFF.date()})")
        print(f"  flat K=32           : {elo_flat.get(player, STARTING_ELO):.1f}")
        print(f"  signature-boost K={K_FACTOR * best_mult:.0f} on flagged matches: {elo_boost.get(player, STARTING_ELO):.1f}")
        print(f"  difference: {elo_boost.get(player, STARTING_ELO) - elo_flat.get(player, STARTING_ELO):+.1f} pts")
        sig_matches = [d for d in deltas_boost[player] if d[3]]
        print(f"  signature matches in this window ({len(sig_matches)}):")
        for date, tourney, rnd, sig, d in sig_matches:
            print(f"    {date.date()}  {tourney:<25} {rnd:<14} boosted delta={d:+.2f}")


if __name__ == "__main__":
    run()
