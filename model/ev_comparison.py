from pathlib import Path

import pandas as pd

from data_loader import load_matches
from win_probability import win_probability

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output" / "wimbledon_2026_ev_comparison.csv"
SURFACE = "Grass"
WIMBLEDON_YEAR = 2026


def load_wimbledon_matches():
    df = load_matches()
    return df[
        df["Tournament"].str.contains("Wimbledon", case=False, na=False)
        & (df["Date"].dt.year == WIMBLEDON_YEAR)
        & (df["Round"] == "1st Round")
    ]


def implied_probabilities(odd_1, odd_2):
    """De-vig bookmaker odds into a pair of probabilities that sum to 1."""
    raw_1 = 1 / odd_1
    raw_2 = 1 / odd_2
    total = raw_1 + raw_2
    return raw_1 / total, raw_2 / total


def build_comparison(matches):
    rows = []
    skipped = []
    for row in matches.itertuples(index=False):
        p1, p2 = row.Player_1, row.Player_2
        try:
            model_prob_p1 = win_probability(p1, p2, SURFACE)
        except ValueError as e:
            skipped.append((p1, p2, str(e)))
            continue

        market_prob_p1, market_prob_p2 = implied_probabilities(row.Odd_1, row.Odd_2)
        model_prob_p2 = 1 - model_prob_p1

        rows.append({"player": p1, "opponent": p2,
                      "market_probability": market_prob_p1, "model_probability": model_prob_p1,
                      "difference": model_prob_p1 - market_prob_p1})
        rows.append({"player": p2, "opponent": p1,
                      "market_probability": market_prob_p2, "model_probability": model_prob_p2,
                      "difference": model_prob_p2 - market_prob_p2})

    return pd.DataFrame(rows), skipped


if __name__ == "__main__":
    matches = load_wimbledon_matches()
    print(f"Loaded {len(matches)} Wimbledon 2026 Round 1 matches")

    comparison, skipped = build_comparison(matches)
    if skipped:
        print(f"\nSkipped {len(skipped)} match(es) — player(s) not found in Elo ratings:")
        for p1, p2, err in skipped:
            print(f"  {p1} vs {p2}: {err}")

    comparison["match_id"] = comparison.apply(lambda r: frozenset((r["player"], r["opponent"])), axis=1)
    comparison["abs_difference"] = comparison["difference"].abs()
    comparison = comparison.sort_values("abs_difference", ascending=False).reset_index(drop=True)

    output_columns = ["player", "opponent", "market_probability", "model_probability", "difference"]
    comparison[output_columns].to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(comparison)} player-rows ({len(comparison) // 2} matches) to {OUTPUT_PATH}")

    # One row per match (not both mirrored perspectives) for the printed top 10.
    biggest_disagreements = comparison.drop_duplicates(subset="match_id").head(10)
    print("\nTop 10 biggest model-vs-market disagreements:")
    print(biggest_disagreements[output_columns].to_string(index=False))
