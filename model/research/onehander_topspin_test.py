"""Tests whether one-handed-backhand players underperform their Elo specifically against a
heavy-topspin playing style, using each opponent's OWN clay-win-rate-minus-hard-win-rate as a real,
defensible proxy for how topspin-heavy their game is - heavy topspin disproportionately rewards
clay-court play (a real, established tennis pattern: high, heavy-bouncing balls sit up above a
one-handed backhand's preferred low-shoulder contact point, and clay's slower, higher-bouncing
surface is exactly where that shot-tolerance gap shows up most). This is a PROXY, not a measured
spin rate - stated explicitly here and in every printed section, not just this docstring, since a
clay-vs-hard win-rate gap can in principle reflect other surface-specific skills (movement, patience,
serve neutralization) that have nothing to do with topspin. It is used because no direct spin-rate
data exists for historical matches; it is the same kind of defensible-but-imperfect proxy the
project's own pedigree/handedness tests use real proxies for elsewhere, not an invented one.

Same isolation discipline as handedness_matchup_test.py:
  - Real frozen (no-lookahead) pre-match Elo, from elite_opponent_residual_test.build_frozen_
    predictions - same walk-forward, same single continuously-updated overall_elo simplification.
  - A NEW walk-forward (build_surface_track_record, below) computes each player's cumulative
    clay/hard win-loss record frozen at every tournament EDITION's start (never using that edition's
    own results, exactly like the Elo walk-forward) - the topspin proxy is only ever computed from a
    player's history strictly BEFORE the match being evaluated, never their career-long final
    record, which would leak future surface performance into predicting a past match.
  - One neutral row per real match (player_a < player_b alphabetically, never anchored to the actual
    winner) - same anti-leakage a/b construction as handedness_matchup_test.py and
    pedigree_market_premium_test.py before it.
  - Because the hypothesis itself is directional (a ONE-HANDER facing a TOPSPIN opponent, not a
    symmetric "matchup type" like lefty-vs-righty), each neutral-row regressor is built as the
    symmetric pairwise DIFFERENCE of the underlying per-player quantity (a's value minus b's value)
    - this is the standard way to fold a directional two-sided effect into a neutral, order-
    independent row without anchoring to the winner. See build_match_rows for the exact terms.
  - Player-clustered bootstrap CI on the interaction coefficient, since real one-handers are a
    concentrated, non-i.i.d. group (a handful of one-handers appear across many rows) and the
    closed-form OLS standard error would understate that.

Usage:
    python model/research/onehander_topspin_test.py
"""
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
MIN_SURFACE_MATCHES = 15  # per surface (clay AND hard), before the topspin proxy is trusted at all


def load_backhand_map():
    """{player_csv_name: 'one-handed'|'two-handed'} - only these two clean values; blank/other
    statuses (unresolved_name, no_infobox, missing_plays_field, malformed_plays,
    hand_only_no_backhand - hand known but backhand itself wasn't stated on Wikipedia) are excluded
    rather than guessed, since backhand type is exactly what this test needs."""
    df = pd.read_csv(HANDEDNESS_PATH, keep_default_na=False, dtype=str)
    usable = df[df["backhand"].isin(["one-handed", "two-handed"])]
    return dict(zip(usable["player"], usable["backhand"]))


def build_surface_track_record(matches):
    """Mirrors build_frozen_predictions' own edition-chronological walk-forward exactly (same
    edition_id construction, same frozen-at-edition-start discipline), but accumulates each
    player's clay/hard win-loss counts instead of Elo. Returns {edition_id: snapshot}, where
    snapshot is (clay_w, clay_l, hard_w, hard_l) dicts holding every count from STRICTLY EARLIER
    editions only - never the edition itself, exactly like the Elo walk-forward never uses an
    edition's own results to predict that same edition's matches. Grass matches are counted in
    neither total - the topspin proxy is specifically clay-vs-hard, and grass is neither."""
    df = matches.copy()
    df["edition_id"] = df["Tournament"] + " " + df["Date"].dt.year.astype(str)
    edition_start = df.groupby("edition_id")["Date"].transform("min")
    editions = (
        df.assign(edition_start=edition_start)[["edition_id", "edition_start"]]
        .drop_duplicates().sort_values("edition_start").reset_index(drop=True)
    )

    clay_w, clay_l, hard_w, hard_l = {}, {}, {}, {}
    snapshots = {}
    for edition_id in editions["edition_id"]:
        snapshots[edition_id] = (dict(clay_w), dict(clay_l), dict(hard_w), dict(hard_l))
        edition_matches = df[df["edition_id"] == edition_id]
        for row in edition_matches.itertuples(index=False):
            if row.Surface not in ("Clay", "Hard"):
                continue
            winner, loser = (row.Player_1, row.Player_2) if row.Winner == row.Player_1 else (row.Player_2, row.Player_1)
            w_book, l_book = (clay_w, clay_l) if row.Surface == "Clay" else (hard_w, hard_l)
            w_book[winner] = w_book.get(winner, 0) + 1
            l_book[loser] = l_book.get(loser, 0) + 1
    return snapshots


def topspin_proxy(snapshot, player):
    """clay_win_rate - hard_win_rate, using ONLY strictly-prior-edition history; NaN (unmeasurable,
    not zero) if the player has fewer than MIN_SURFACE_MATCHES on either surface yet."""
    clay_w, clay_l, hard_w, hard_l = snapshot
    cw, cl = clay_w.get(player, 0), clay_l.get(player, 0)
    hw, hl = hard_w.get(player, 0), hard_l.get(player, 0)
    if cw + cl < MIN_SURFACE_MATCHES or hw + hl < MIN_SURFACE_MATCHES:
        return np.nan
    return cw / (cw + cl) - hw / (hw + hl)


def build_match_rows(tour, matches, backhand):
    preds, editions = build_frozen_predictions(matches)
    surface_snapshots = build_surface_track_record(matches)

    # one row per real match, per-player perspective rows collapsed to the neutral a<b ordering -
    # same technique handedness_matchup_test.py uses, never anchored to who actually won
    neutral = preds[preds["player"] < preds["opponent"]].copy()
    neutral = neutral.rename(columns={
        "player": "player_a", "opponent": "player_b",
        "player_elo": "elo_a", "opponent_elo": "elo_b", "actual_win": "won_a",
    })

    onehander_a = neutral["player_a"].map(backhand).map({"one-handed": 1, "two-handed": 0})
    onehander_b = neutral["player_b"].map(backhand).map({"one-handed": 1, "two-handed": 0})

    topspin_a, topspin_b = [], []
    for row in neutral.itertuples(index=False):
        snap = surface_snapshots[row.edition_id]
        topspin_a.append(topspin_proxy(snap, row.player_a))
        topspin_b.append(topspin_proxy(snap, row.player_b))

    out = pd.DataFrame({
        "tour": tour, "edition_id": neutral["edition_id"].values, "date": neutral["date"].values,
        "player_a": neutral["player_a"].values, "player_b": neutral["player_b"].values,
        "elo_diff": (neutral["elo_a"] - neutral["elo_b"]).values,
        "won_a": neutral["won_a"].values,
        "onehander_a": onehander_a.values, "onehander_b": onehander_b.values,
        "topspin_a": topspin_a, "topspin_b": topspin_b,
    })
    return out


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
    """Player-clustered bootstrap CI for one OLS coefficient (coef_index into x_cols, after the
    intercept). Clusters on the UNION of both players in a row (group_cols=['player_a','player_b'])
    - a real one-hander's matches show up in this dataset regardless of which alphabetical slot
    they landed in, so clustering on player_a alone would under-count how concentrated the real
    one-hander population is. Resamples that player-id space with replacement, refits OLS on every
    row belonging to a resampled player (rows with a resampled player on EITHER side are included,
    matching how a player's own real-world match history isn't limited to one slot)."""
    all_players = pd.unique(df[group_cols].values.ravel())
    rng = np.random.default_rng(seed)
    y = df[y_col].values.astype(float)
    X = df[x_cols].values.astype(float)

    # precompute a fast per-player row-index lookup instead of re-filtering the dataframe 3000x
    rows_by_player = {}
    for i, (a, b) in enumerate(zip(df[group_cols[0]].values, df[group_cols[1]].values)):
        rows_by_player.setdefault(a, []).append(i)
        rows_by_player.setdefault(b, []).append(i)

    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled_players = rng.choice(all_players, size=len(all_players), replace=True)
        idx = np.concatenate([rows_by_player[p] for p in sampled_players if p in rows_by_player])
        beta, _ = ols(y[idx], X[idx])
        boot[i] = beta[coef_index + 1]  # +1 for intercept
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return lo, hi


def run():
    backhand = load_backhand_map()
    n_one = sum(1 for v in backhand.values() if v == "one-handed")
    n_two = sum(1 for v in backhand.values() if v == "two-handed")
    print(f"Backhand type known for {len(backhand)} players ({n_one} one-handed, {n_two} "
          f"two-handed) - one-handed backhand is a real minority style in the modern game, "
          f"this is a genuinely small population, not a data gap.")

    frames = []
    for tour in ("ATP", "WTA"):
        matches = load_matches_for_tour(tour)
        rows = build_match_rows(tour, matches, backhand)
        print(f"{tour}: {len(rows)} real historical matches with a frozen pre-match Elo gap")
        frames.append(rows)
    all_rows = pd.concat(frames, ignore_index=True)

    # --- coverage, reported in stages, honestly, before any result ---
    print(f"\n{len(all_rows)} total real matches (both tours) with a resolved Elo gap.")

    has_backhand_either_side = all_rows["onehander_a"].notna() | all_rows["onehander_b"].notna()
    print(f"{has_backhand_either_side.sum()} of those have a known one-handed/two-handed backhand "
          f"type for AT LEAST ONE player.")

    onehander_present = (all_rows["onehander_a"] == 1) | (all_rows["onehander_b"] == 1)
    onehander_vs_measurable_opp = onehander_present & (
        ((all_rows["onehander_a"] == 1) & all_rows["topspin_b"].notna())
        | ((all_rows["onehander_b"] == 1) & all_rows["topspin_a"].notna())
    )
    print(f"{onehander_present.sum()} involve at least one player with a KNOWN one-handed backhand "
          f"specifically. Of those, {onehander_vs_measurable_opp.sum()} also have a measurable "
          f"topspin proxy (>= {MIN_SURFACE_MATCHES} prior clay AND hard matches) for that "
          f"one-hander's opponent - this is the real population the underperformance question is "
          f"actually about.")

    # the symmetric neutral-row regression needs BOTH sides fully known (both backhand type AND
    # both topspin proxies) to compute a valid pairwise difference - a strictly narrower cut than
    # the one-directional coverage numbers above, stated explicitly rather than blurred together
    usable = all_rows.dropna(subset=["onehander_a", "onehander_b", "topspin_a", "topspin_b"]).copy()
    print(f"\nThe controlled regression itself needs BOTH players' backhand type AND BOTH players' "
          f"topspin proxy known (to build the neutral pairwise-difference regressors below): "
          f"{len(usable)} matches qualify.")

    if len(usable) < 100:
        print(f"\n{'!' * 90}\nSample too thin (n={len(usable)}) to run a meaningful regression - "
              f"same standard applied to every other test tonight. Reporting composition only, "
              f"NOT a coefficient/significance verdict, since a real answer isn't available at "
              f"this sample size.\n{'!' * 90}")
        one_rows = all_rows[onehander_present]
        print(f"\nFor reference, the {len(one_rows)} known-one-hander-involved matches break down as:")
        print(f"  one-hander vs opponent with measurable topspin proxy: {onehander_vs_measurable_opp.sum()}")
        return

    usable["backhand_diff"] = usable["onehander_a"] - usable["onehander_b"]
    usable["opp_topspin_diff"] = usable["topspin_b"] - usable["topspin_a"]
    usable["interaction_diff"] = (
        usable["onehander_a"] * usable["topspin_b"] - usable["onehander_b"] * usable["topspin_a"]
    )

    # --- confound check, same standard as handedness_matchup_test.py ---
    high_topspin_matchup = usable["interaction_diff"].abs() > usable["interaction_diff"].abs().median()
    print(f"\nConfound check: mean |elo_diff| for matches where a one-hander faces a higher-topspin-"
          f"proxy opponent (interaction_diff magnitude above median) = "
          f"{usable.loc[high_topspin_matchup, 'elo_diff'].abs().mean():.1f}, vs. "
          f"{usable.loc[~high_topspin_matchup, 'elo_diff'].abs().mean():.1f} for the rest "
          f"(if these differ a lot, a raw comparison would be confounded by skill gap, not just "
          f"topspin exposure - this is exactly why elo_diff has to be in the regression).")

    print(f"\n{'=' * 92}\nOLS: won_a ~ elo_diff + backhand_diff + opp_topspin_diff + interaction_diff\n"
          f"n={len(usable)} matches, both tours pooled\n{'=' * 92}")
    y = usable["won_a"].values.astype(float)
    x_cols = ["elo_diff", "backhand_diff", "opp_topspin_diff", "interaction_diff"]
    X = usable[x_cols].values.astype(float)
    beta, se = ols(y, X)
    names = ["intercept"] + x_cols
    for name, b, s in zip(names, beta, se):
        z = b / s if s == s and s != 0 else float("nan")
        print(f"  {name:<18}: coef={b:+.5f}  SE={s:.5f}  z={z:+.2f}"
              + ("  (|z|>1.96, nominally significant)" if abs(z) > 1.96 else "  (not significant)"))

    interaction_idx = x_cols.index("interaction_diff")
    lo, hi = cluster_bootstrap_coef(usable, "won_a", x_cols, interaction_idx, ["player_a", "player_b"])
    interaction_coef = beta[interaction_idx + 1]
    interaction_z = interaction_coef / se[interaction_idx + 1]
    print(f"\nPlayer-clustered bootstrap 95% CI on interaction_diff: [{lo:+.5f}, {hi:+.5f}] "
          f"(closed-form point estimate {interaction_coef:+.5f})")

    print(f"\n{'=' * 92}")
    bootstrap_excludes_zero = lo > 0 or hi < 0
    if abs(interaction_z) > 1.96 and bootstrap_excludes_zero:
        direction = "underperform" if interaction_coef < 0 else "OUTPERFORM"
        print(f"interaction_diff IS significant after controlling for elo_diff and both main "
              f"effects (z={interaction_z:+.2f}, bootstrap CI excludes zero): one-handers "
              f"{direction} their Elo specifically against higher clay-win-rate-vs-hard-win-rate "
              f"(topspin-PROXY) opponents in this sample.")
    else:
        print(f"interaction_diff is NOT significant after controlling for elo_diff and both main "
              f"effects (z={interaction_z:+.2f}, bootstrap CI [{lo:+.5f}, {hi:+.5f}]"
              f"{' straddles zero' if bootstrap_excludes_zero is False else ''}) - once real skill "
              f"difference and the main effects are accounted for, no real one-hander-vs-topspin-"
              f"proxy edge shows up in this sample. Same honest null-result standard "
              f"handedness_matchup_test.py and pedigree_market_premium_test.py apply.")
    print("REMINDER: clay-minus-hard win rate is a PROXY for topspin-heaviness, not a measured "
          "spin rate - a real or null result here is a statement about this proxy, not a direct "
          "physical measurement.")
    print("=" * 92)


if __name__ == "__main__":
    run()
