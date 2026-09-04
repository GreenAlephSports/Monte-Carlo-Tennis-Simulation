"""Tests whether player height predicts outperformance beyond Elo, and specifically whether
height's effect concentrates in serve-dependent, low-rally situations - using each player's OWN
tiebreak-win-rate-minus-overall-win-rate as a real, defensible proxy for serve strength: a player
who wins tiebreaks more often than their overall record would predict is winning disproportionately
many serve-dependent, low-rally points, since a tiebreak is close to pure serve-hold pressure with
almost no time for a longer rally-based game plan to take over. This is a PROXY, not a measured
ace count or serve speed - stated explicitly here and in every printed section, same disclosure
standard as onehander_topspin_test.py's clay-minus-hard topspin proxy. Ace/serve-speed data doesn't
exist anywhere in this project's data (same gap already confirmed for spin rate and reused here).

Same isolation discipline as handedness_matchup_test.py and onehander_topspin_test.py:
  - Real frozen (no-lookahead) pre-match Elo, from elite_opponent_residual_test.build_frozen_
    predictions - same walk-forward, same single continuously-updated overall_elo simplification.
  - A NEW walk-forward (build_tiebreak_track_record, below) computes each player's cumulative
    tiebreak win-loss record AND overall match win-loss record, frozen at every tournament
    EDITION's start (never using that edition's own results) - mirrors onehander_topspin_test.py's
    build_surface_track_record exactly, just tracking tiebreaks-vs-overall instead of clay-vs-hard.
    Tiebreak sets are identified directly from the real Score string (a set token of "7-6" or "6-7",
    following the same SCORE_SET_RE-based parsing layoff_margin_of_victory_test.py already uses for
    this dataset's only available shot-level-adjacent signal).
  - One neutral row per real match (player_a < player_b alphabetically, never anchored to the
    actual winner) - same anti-leakage a/b construction as the other two tests tonight.
  - Because the hypothesis is directional but symmetric-in-labeling (does MY height interact with
    MY OWN serve-proxy), every regressor is the pairwise difference (a's value minus b's value) of
    the underlying per-player quantity - same technique onehander_topspin_test.py uses to fold a
    directional effect into a neutral, order-independent row.
  - Player-clustered bootstrap CI on the interaction coefficient (real height data is concentrated
    in a non-i.i.d. way just like the one-hander population was - some players appear in far more
    rows than others).

Usage:
    python model/research/height_serve_proxy_test.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

from elite_opponent_residual_test import build_frozen_predictions  # noqa: E402
from elo_ratings import load_matches_for_tour  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
HANDEDNESS_PATH = OUTPUT_DIR / "player_handedness.csv"
MIN_TIEBREAKS = 10       # real tiebreaks played, before the serve-proxy is trusted at all
MIN_MATCHES_FOR_RATE = 15  # real matches played, before the "overall win rate" baseline is trusted

SCORE_SET_RE = re.compile(r"(\d+)-(\d+)(?:\(\d+\))?")


def load_height_map():
    """{player_csv_name: height_cm (float)} - only rows with a cleanly-parsed height_cm (see
    wikipedia_handedness_scrape.py's height_status column); anything blank/unparsed is excluded
    rather than guessed."""
    df = pd.read_csv(HANDEDNESS_PATH, keep_default_na=False, dtype=str)
    usable = df[(df["height_cm"] != "") & df["height_cm"].notna()]
    return {row.player: float(row.height_cm) for row in usable.itertuples(index=False)}


def build_tiebreak_track_record(matches):
    """Mirrors onehander_topspin_test.build_surface_track_record's edition-chronological,
    frozen-at-edition-start walk-forward exactly, but tracks (tiebreak wins, tiebreak losses,
    overall match wins, overall match losses) per player instead of clay/hard match records.
    Returns {edition_id: snapshot}, snapshot = (tb_w, tb_l, match_w, match_l) dicts holding counts
    from STRICTLY EARLIER editions only."""
    df = matches.copy()
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)
    edition_start = df.groupby("edition_id")["Date"].transform("min")
    editions = (
        df.assign(edition_start=edition_start)[["edition_id", "edition_start"]]
        .drop_duplicates().sort_values("edition_start").reset_index(drop=True)
    )

    tb_w, tb_l, match_w, match_l = {}, {}, {}, {}
    snapshots = {}
    for edition_id in editions["edition_id"]:
        snapshots[edition_id] = (dict(tb_w), dict(tb_l), dict(match_w), dict(match_l))
        edition_matches = df[df["edition_id"] == edition_id]
        for row in edition_matches.itertuples(index=False):
            p1, p2, winner = row.Player_1, row.Player_2, row.Winner
            loser = p2 if winner == p1 else p1
            match_w[winner] = match_w.get(winner, 0) + 1
            match_l[loser] = match_l.get(loser, 0) + 1

            for tok in str(row.Score).split():
                m = SCORE_SET_RE.fullmatch(tok)
                if not m:
                    continue
                a, b = int(m.group(1)), int(m.group(2))
                if {a, b} != {6, 7}:
                    continue  # only the standard 7-6/6-7 tiebreak-set signature counts
                # SCORE_SET_RE tokens are Player_1-Player_2 per set (same convention confirmed and
                # consistency-checked against Winner in layoff_margin_of_victory_test.py)
                set_winner, set_loser = (p1, p2) if a > b else (p2, p1)
                tb_w[set_winner] = tb_w.get(set_winner, 0) + 1
                tb_l[set_loser] = tb_l.get(set_loser, 0) + 1
    return snapshots


def tiebreak_proxy(snapshot, player):
    """tiebreak_win_rate - overall_match_win_rate, using ONLY strictly-prior-edition history; NaN
    (unmeasurable, not zero) below MIN_TIEBREAKS real tiebreaks or MIN_MATCHES_FOR_RATE real
    matches."""
    tb_w, tb_l, match_w, match_l = snapshot
    tw, tl = tb_w.get(player, 0), tb_l.get(player, 0)
    mw, ml = match_w.get(player, 0), match_l.get(player, 0)
    if tw + tl < MIN_TIEBREAKS or mw + ml < MIN_MATCHES_FOR_RATE:
        return np.nan
    return tw / (tw + tl) - mw / (mw + ml)


def build_match_rows(tour, matches, height):
    preds, editions = build_frozen_predictions(matches)
    tb_snapshots = build_tiebreak_track_record(matches)

    neutral = preds[preds["player"] < preds["opponent"]].copy()
    neutral = neutral.rename(columns={
        "player": "player_a", "opponent": "player_b",
        "player_elo": "elo_a", "opponent_elo": "elo_b", "actual_win": "won_a",
    })

    height_a = neutral["player_a"].map(height)
    height_b = neutral["player_b"].map(height)

    tb_a, tb_b = [], []
    for row in neutral.itertuples(index=False):
        snap = tb_snapshots[row.edition_id]
        tb_a.append(tiebreak_proxy(snap, row.player_a))
        tb_b.append(tiebreak_proxy(snap, row.player_b))

    return pd.DataFrame({
        "tour": tour, "edition_id": neutral["edition_id"].values, "date": neutral["date"].values,
        "player_a": neutral["player_a"].values, "player_b": neutral["player_b"].values,
        "elo_diff": (neutral["elo_a"] - neutral["elo_b"]).values,
        "won_a": neutral["won_a"].values,
        "height_a": height_a.values, "height_b": height_b.values,
        "tiebreak_a": tb_a, "tiebreak_b": tb_b,
    })


def ols(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    n, k = X1.shape
    sigma2 = (resid @ resid) / (n - k)
    cov = sigma2 * np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    return beta, se


def cluster_bootstrap_coef(df, y_col, x_cols, coef_index, group_cols, n_boot=3000, seed=42):
    """Player-clustered bootstrap CI for one OLS coefficient - same technique as
    onehander_topspin_test.cluster_bootstrap_coef, clustering on the union of both players in a
    row so a player's real match history counts once regardless of which alphabetical slot they
    happened to land in for a given match."""
    all_players = pd.unique(df[group_cols].values.ravel())
    rng = np.random.default_rng(seed)
    y = df[y_col].values.astype(float)
    X = df[x_cols].values.astype(float)

    rows_by_player = {}
    for i, (a, b) in enumerate(zip(df[group_cols[0]].values, df[group_cols[1]].values)):
        rows_by_player.setdefault(a, []).append(i)
        rows_by_player.setdefault(b, []).append(i)

    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled_players = rng.choice(all_players, size=len(all_players), replace=True)
        idx = np.concatenate([rows_by_player[p] for p in sampled_players if p in rows_by_player])
        beta, _ = ols(y[idx], X[idx])
        boot[i] = beta[coef_index + 1]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return lo, hi


def run():
    height = load_height_map()
    print(f"Height known (cleanly parsed from Wikipedia infobox) for {len(height)} players "
          f"(range {min(height.values()):.0f}-{max(height.values()):.0f} cm)")

    frames = []
    for tour in ("ATP", "WTA"):
        matches = load_matches_for_tour(tour)
        rows = build_match_rows(tour, matches, height)
        print(f"{tour}: {len(rows)} real historical matches with a frozen pre-match Elo gap")
        frames.append(rows)
    all_rows = pd.concat(frames, ignore_index=True)

    # --- coverage, reported in stages, honestly, before any result ---
    print(f"\n{len(all_rows)} total real matches (both tours) with a resolved Elo gap.")

    both_heights = all_rows["height_a"].notna() & all_rows["height_b"].notna()
    print(f"{both_heights.sum()} of those have a known height for BOTH players "
          f"({both_heights.mean():.1%} coverage) - the population the height-outperformance "
          f"question has to work with.")

    usable = all_rows[both_heights].dropna(subset=["tiebreak_a", "tiebreak_b"]).copy()
    print(f"The interaction test additionally needs BOTH players' tiebreak-proxy measurable "
          f"(>= {MIN_TIEBREAKS} prior tiebreaks and >= {MIN_MATCHES_FOR_RATE} prior matches each): "
          f"{len(usable)} matches qualify - this is the real sample the serve-dependent-interaction "
          f"question is actually about.")

    if len(usable) < 100:
        print(f"\n{'!' * 90}\nSample too thin (n={len(usable)}) to run a meaningful regression - "
              f"same standard applied to every other test tonight. Not reporting a coefficient/"
              f"significance verdict at this sample size.\n{'!' * 90}")
        return

    usable["height_diff"] = usable["height_a"] - usable["height_b"]
    usable["tiebreak_diff"] = usable["tiebreak_a"] - usable["tiebreak_b"]
    usable["interaction_diff"] = usable["height_diff"] * usable["tiebreak_diff"]

    # --- confound check, same standard as the other two tests tonight ---
    high_interaction = usable["interaction_diff"].abs() > usable["interaction_diff"].abs().median()
    print(f"\nConfound check: mean |elo_diff| for matches with an above-median height x "
          f"tiebreak-proxy interaction magnitude = "
          f"{usable.loc[high_interaction, 'elo_diff'].abs().mean():.1f}, vs. "
          f"{usable.loc[~high_interaction, 'elo_diff'].abs().mean():.1f} for the rest "
          f"(if these differ a lot, a raw comparison would be confounded by skill gap, not just "
          f"height/serve-proxy exposure - this is why elo_diff has to be in the regression).")

    print(f"\n{'=' * 92}\nOLS: won_a ~ elo_diff + height_diff + tiebreak_diff + interaction_diff\n"
          f"n={len(usable)} matches, both tours pooled\n{'=' * 92}")
    y = usable["won_a"].values.astype(float)
    x_cols = ["elo_diff", "height_diff", "tiebreak_diff", "interaction_diff"]
    X = usable[x_cols].values.astype(float)
    beta, se = ols(y, X)
    names = ["intercept"] + x_cols
    for name, b, s in zip(names, beta, se):
        z = b / s if s == s and s != 0 else float("nan")
        print(f"  {name:<15}: coef={b:+.6f}  SE={s:.6f}  z={z:+.2f}"
              + ("  (|z|>1.96, nominally significant)" if abs(z) > 1.96 else "  (not significant)"))

    height_idx = x_cols.index("height_diff")
    interaction_idx = x_cols.index("interaction_diff")
    height_z = beta[height_idx + 1] / se[height_idx + 1]
    interaction_z = beta[interaction_idx + 1] / se[interaction_idx + 1]

    lo_h, hi_h = cluster_bootstrap_coef(usable, "won_a", x_cols, height_idx, ["player_a", "player_b"])
    lo_i, hi_i = cluster_bootstrap_coef(usable, "won_a", x_cols, interaction_idx, ["player_a", "player_b"])
    print(f"\nPlayer-clustered bootstrap 95% CI on height_diff: [{lo_h:+.6f}, {hi_h:+.6f}] "
          f"(closed-form point estimate {beta[height_idx + 1]:+.6f})")
    print(f"Player-clustered bootstrap 95% CI on interaction_diff: [{lo_i:+.6f}, {hi_i:+.6f}] "
          f"(closed-form point estimate {beta[interaction_idx + 1]:+.6f})")

    print(f"\n{'=' * 92}")
    height_real = abs(height_z) > 1.96 and (lo_h > 0 or hi_h < 0)
    interaction_real = abs(interaction_z) > 1.96 and (lo_i > 0 or hi_i < 0)

    if height_real:
        direction = "outperform" if beta[height_idx + 1] > 0 else "underperform"
        print(f"height_diff IS significant after controlling for elo_diff (z={height_z:+.2f}, "
              f"bootstrap CI excludes zero): taller players {direction} their Elo in this sample.")
    else:
        print(f"height_diff is NOT significant after controlling for elo_diff (z={height_z:+.2f}, "
              f"bootstrap CI [{lo_h:+.6f}, {hi_h:+.6f}]) - no real height-outperformance edge shows "
              f"up beyond skill in this sample.")

    if interaction_real:
        direction = "amplifies" if beta[interaction_idx + 1] > 0 else "suppresses"
        print(f"interaction_diff IS significant (z={interaction_z:+.2f}, bootstrap CI excludes "
              f"zero): a serve-proxy gap {direction} height's effect on winning beyond Elo.")
    else:
        print(f"interaction_diff is NOT significant (z={interaction_z:+.2f}, bootstrap CI "
              f"[{lo_i:+.6f}, {hi_i:+.6f}]) - no real height x serve-dependent-situation interaction "
              f"shows up beyond skill in this sample. Same honest null-result standard "
              f"onehander_topspin_test.py and handedness_matchup_test.py apply.")
    print("REMINDER: tiebreak-win-rate-minus-overall-win-rate is a PROXY for serve strength in "
          "serve-dependent situations, not a measured ace count or serve speed - a real or null "
          "result here is a statement about this proxy, not a direct physical measurement.")
    print("=" * 92)


if __name__ == "__main__":
    run()
