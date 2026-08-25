"""Futures/tournament-winner section of the paper-trading analysis.

Two real questions, checked directly, not assumed:

  1. Does The Odds API carry a real "outrights" (tournament-winner) market for tennis, separate
     from the h2h match market every other backtest in this project already uses? Tested directly:
     every tennis sport key's has_outrights flag, plus a live direct outrights request against
     whatever tennis event is currently priced (tennis_wta_monterrey_open at the time this ran) -
     same infrastructure Cincinnati's own odds would come from, since has_outrights is a property
     of the market TYPE per sport, not of one specific tournament.
  2. If no real futures market exists (it doesn't - see REPORTED FINDING below), the model's own
     tournament_win_probability (p_champ) for the real eventual champion can still be shown
     DESCRIPTIVELY - how the model's own belief evolved round to round for the player who actually
     won - just without anything to compare it against.

REPORTED FINDING (checked 2026-08-25, live against The Odds API):
  44 tennis sport keys checked - has_outrights=true for: NONE. The only sports with
  has_outrights=true anywhere on this connected source are NFL/NCAAF/MLB/NBA/NCAAB/NHL
  championship-winner markets, 4 golf majors, and 2 non-sports (politics/World Cup). A direct
  outrights request for the one currently-active tennis event (tennis_wta_monterrey_open) was
  REJECTED with HTTP 422 Unprocessable Entity. Conclusion stands for Cincinnati too, since this is
  a market-type limitation of the data source, not anything specific to one tournament: there is no
  futures market to compare the model against for tennis, on this connected source, at all.

p_champ trajectory: for each completed tournament (Montreal ATP, Toronto WTA, Cincinnati ATP,
Cincinnati WTA), replays the REAL results round by round (same replay_real_rounds/
reconstruct_leaves_by_round2_slot machinery hybrid_simulation.py and conditional_equity_report.py
already use - not reimplemented) and, at each fully-resolved checkpoint, runs a genuine Monte Carlo
tournament simulation (win_probability + upset-boost, same as bracket_export.py's real
tournament_win_probability column) from that real field forward to get the eventual real champion's
title probability at that point in the draw.

SAMPLE SIZE, STATED PLAINLY: this is at most 4 real draws (2 tournaments x up to 2 tours each,
though Montreal/Toronto only fielded one tour apiece in this project's bracket data) - nowhere near
enough to be a powered result. It's illustrative and directional: does the model's belief in the
real eventual champion trend the way you'd want (generally upward, with real upsets survived along
the way), not a statistically validated claim.

Usage:
    python model/research/tournament_futures_analysis.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conditional_equity_report import full_checkpoint_counts, load_tournament_state  # noqa: E402

N_SIMULATIONS = 5000
SEED = 42

DRAWS = [
    ("Montreal (ATP)", Path("brackets/montreal_2026.yaml"), "20260802-20260814"),
    ("Toronto (WTA)", Path("brackets/wta_toronto_2026.yaml"), "20260802-20260814"),
    ("Cincinnati (ATP)", Path("brackets/cincinnati_2026_atp_demo.yaml"), "20260813-20260825"),
    ("Cincinnati (WTA)", Path("brackets/cincinnati_2026_wta.yaml"), "20260813-20260825"),
]


def real_champion(state):
    """The real winner of the final = whoever won the last fully-resolved round."""
    final_round = state["max_known_round"]
    final_results = state["results_by_round"].get(final_round, {})
    assert len(final_results) == 1, f"expected exactly one final result, got {final_results}"
    return next(iter(final_results.values()))


def trajectory_for_draw(label, bracket_path, dates):
    print(f"\n=== {label}: {bracket_path.name} ===")
    state = load_tournament_state(bracket_path, dates=dates)
    max_round = state["max_known_round"]
    print(f"  Real results known through round {max_round} (final = round {max_round})")

    champion = real_champion(state)
    print(f"  Real champion: {champion}")

    trajectory = []
    for n in range(0, max_round + 1):
        counts = full_checkpoint_counts(state, n, N_SIMULATIONS, SEED + n)
        p_champ = counts.get(champion)
        trajectory.append((n, p_champ))
        label_n = "pre-tournament field" if n == 0 else f"through round {n}"
        print(f"    checkpoint {n:>2} ({label_n:<22}): p_champ({champion}) = {p_champ:.1%}")
    return {"label": label, "champion": champion, "max_round": max_round, "trajectory": trajectory}


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    print("=== Part 1: does a real tennis futures/outrights market exist on The Odds API? ===")
    print("(see this module's own REPORTED FINDING docstring for the full, dated check - re-run "
          "model/research/live_odds_value_scan.py directly to reproduce it live)")
    print("  ANSWER: No. 44/44 tennis sport keys have has_outrights=false; a direct outrights "
          "request was rejected (HTTP 422). No market comparison is possible for tennis futures "
          "on this connected source - noted plainly, not assumed away.")

    print("\n=== Part 2: descriptive p_champ trajectory for each real eventual champion ===")
    results = []
    for label, path, dates in DRAWS:
        results.append(trajectory_for_draw(label, path, dates))

    print("\n=== Summary ===")
    for r in results:
        pre, post = r["trajectory"][0][1], r["trajectory"][-1][1]
        print(f"  {r['label']:<18} champion={r['champion']:<18} "
              f"p_champ: pre-tournament {pre:.1%} -> final checkpoint {post:.1%}")


if __name__ == "__main__":
    main()
