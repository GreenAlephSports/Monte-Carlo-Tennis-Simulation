"""Before/after comparison: what changed in the model's live output since Daron last saw it.

Daron's last look predates all four empirically-fit corrections in win_probability.py
(rank_adjustment, confidence_calibration, layoff_adjustment) and simulate.py (upset_boost) - this
runs bracket_export.py's real export pipeline twice against the current live bracket, once with
every correction forced off (raw Elo only - matching what Daron saw) and once with the current
default (every correction on), and diffs the real p_champ / pregame-matchup output side by side.

Important honesty note on upset_boost, confirmed by reading the actual call graph, not assumed:
bracket_export.py's live p_champ simulation goes through simulate.run_simulations_tracking_
milestones, whose own inner round-replay loop calls simulate._play_round(players, surface,
ratings_path) with NO prior_beaten_elo argument - so track_upsets is always False on that path,
regardless of use_upset_boost anywhere else. upset_boost is wired into hybrid_simulation.py's
round-by-round REPLAY only (see win_probability.UPSET_BOOST_LOGIT_SHIFT's own docstring: "this can
never apply to a pre-tournament baseline prediction, only a round-by-round replay"), which
bracket_export.py's simulation step doesn't use for the forward Monte Carlo. So toggling
upset_boost cannot change bracket_export.py's p_champ output as currently wired - this script
still "forces it off" for correctness/documentation purposes (via simulate.UPSET_BOOST_LOGIT_SHIFT),
but reports this structural fact plainly rather than fabricating a difference that doesn't exist.

Mechanism: win_probability() and _play_round()/run_simulations_tracking_milestones() are called
positionally with no override kwargs anywhere in bracket_export.py/simulate.py, so there's no
existing flag to pass through export_bracket_json(). Rather than editing production code to add
one, this script monkeypatches the three real knobs at the constant level for the "off" run
(RANK_ADJUSTMENT_C, PLATT_B forced to no-op values, LAYOFF shifts zeroed) - the same constants
every correction's own logic already reads from - then restores them before the "on" run.

Usage:
    python model/research/before_after_comparison.py [bracket_path] [--simulations N]
"""
import contextlib
import copy
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # production modules in model/

import bracket_export  # noqa: E402
import hybrid_simulation  # noqa: E402
import win_probability  # noqa: E402
from bracket_export import export_bracket_json  # noqa: E402

DEFAULT_BRACKET = Path(__file__).resolve().parent.parent.parent / "brackets" / "cincinnati_2026_atp_demo.yaml"
N_SIMULATIONS = 10000
SEED = 42

# real constants read live are cached via _load_ratings/win_probability's own module-level
# references - snapshot every one of them so the "off" run can zero them out and the "on" run can
# restore the exact real fitted values afterward, not hardcoded guesses of what they were.
CORRECTION_CONSTANTS = [
    "RANK_ADJUSTMENT_C", "PLATT_B",
    "LAYOFF_BUCKET_EDGES_ATP", "LAYOFF_BUCKET_EDGES_WTA",
]


def _zero_out_layoff_edges(edges):
    """Same bucket structure/test functions, shift forced to 0.0 - keeps _layoff_shift_for_days'
    bucket-selection logic exercised identically, just contributing nothing."""
    return [(name, test, 0.0) for name, test, shift in edges]


def _force_corrections_off():
    win_probability.RANK_ADJUSTMENT_C = 0.0
    win_probability.PLATT_B = 1.0  # sigmoid(1.0 * logit(p)) == p - exact identity, not an approximation
    win_probability.LAYOFF_BUCKET_EDGES_ATP = _zero_out_layoff_edges(win_probability.LAYOFF_BUCKET_EDGES_ATP)
    win_probability.LAYOFF_BUCKET_EDGES_WTA = _zero_out_layoff_edges(win_probability.LAYOFF_BUCKET_EDGES_WTA)
    win_probability._load_ratings.cache_clear()  # ratings themselves are untouched, but be safe


def _restore_corrections(saved):
    for name, value in saved.items():
        setattr(win_probability, name, value)
    win_probability._load_ratings.cache_clear()


@contextlib.contextmanager
def freeze_live_data():
    """Pulls live Kaggle match history and the live ESPN scoreboard exactly ONCE per comparison
    run, then patches both entry points (in every module that imports them - bracket_export.py and
    hybrid_simulation.py each did `from elo_ratings import load_matches_for_tour` /
    `from live_scores import fetch_scoreboard`, so each holds its own module-level reference that
    needs patching separately) to return that one frozen pull for every subsequent call inside the
    `with` block, instead of re-hitting the live sources per condition/round.

    This exists because run_export/run_hybrid_all_rounds below invoke export_bracket_json()/
    hybrid_simulation.main() twice each (once per corrections OFF/ON condition) - each call pulls
    Kaggle + ESPN live on its own. Between those two calls, real time passes and live state can
    change (e.g. a match goes from in-progress to final), so a "corrections OFF vs ON" comparison
    could actually be silently comparing two different snapshots of reality - the mechanism behind
    a real confirmed bug where Tommy Paul's p_champ appeared to DROP across a checkpoint that
    included a real win, because the two checkpoints weren't run against the same underlying data.
    Freezing is scoped to a single comparison run (this context manager), not a change to how the
    live model refreshes day to day outside of these research scripts.
    """
    load_cache = {}
    scoreboard_cache = {}

    real_load_matches_for_tour = hybrid_simulation.load_matches_for_tour
    real_fetch_scoreboard = hybrid_simulation.fetch_scoreboard

    def frozen_load_matches_for_tour(tour):
        tour = tour.upper()
        if tour not in load_cache:
            load_cache[tour] = real_load_matches_for_tour(tour)
        return load_cache[tour]

    def frozen_fetch_scoreboard(tour, dates=None):
        key = (tour, dates)
        if key not in scoreboard_cache:
            scoreboard_cache[key] = real_fetch_scoreboard(tour, dates=dates)
        return scoreboard_cache[key]

    patch_targets = [hybrid_simulation, bracket_export]
    saved = [(mod, mod.load_matches_for_tour, mod.fetch_scoreboard) for mod in patch_targets]
    try:
        for mod in patch_targets:
            mod.load_matches_for_tour = frozen_load_matches_for_tour
            mod.fetch_scoreboard = frozen_fetch_scoreboard
        yield
    finally:
        for mod, orig_load, orig_fetch in saved:
            mod.load_matches_for_tour = orig_load
            mod.fetch_scoreboard = orig_fetch


SCRATCH_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "_before_after_scratch"


def run_export(bracket_path, n_simulations, label):
    print(f"\n--- Running export ({label}) ---", file=sys.stderr)
    _output_path, output = export_bracket_json(
        bracket_path, output_path=SCRATCH_DIR / f"_before_after_{label}.json",
        n_simulations=n_simulations, seed=SEED,
    )
    return output


def run():
    bracket_path = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else DEFAULT_BRACKET
    n_simulations = N_SIMULATIONS
    for i, arg in enumerate(sys.argv):
        if arg == "--simulations" and i + 1 < len(sys.argv):
            n_simulations = int(sys.argv[i + 1])

    saved = {name: copy.deepcopy(getattr(win_probability, name)) for name in CORRECTION_CONSTANTS}

    with freeze_live_data():
        _force_corrections_off()
        off_output = run_export(bracket_path, n_simulations, "corrections_off")

        _restore_corrections(saved)
        on_output = run_export(bracket_path, n_simulations, "corrections_on")

    off_champ = {p["player"]: p["p_champ"] for p in off_output["players"]}
    on_champ = {p["player"]: p["p_champ"] for p in on_output["players"]}
    common_players = set(off_champ) & set(on_champ)

    rows = []
    for player in common_players:
        before, after = off_champ[player], on_champ[player]
        rows.append((player, before, after, after - before))
    rows.sort(key=lambda r: -abs(r[3]))

    print(f"\n{'=' * 90}\nP_CHAMP: corrections OFF (raw Elo, matches what Daron last saw) vs ON "
          f"(current default)\n{len(common_players)} players alive in both runs, "
          f"{n_simulations} simulations each, same seed ({SEED})\n{'=' * 90}")
    print(f"{'Player':<28} {'p_champ OFF':>12} {'p_champ ON':>12} {'Change':>10} {'Rel. %':>10}")
    print("-" * 74)
    for player, before, after, delta in rows[:15]:
        rel_pct = f"{(after - before) / before * 100:+.1f}%" if before > 0 else "n/a (0->" + f"{after:.1%})"
        print(f"{player:<28} {before:>11.1%} {after:>11.1%} {delta:>+9.1%} {rel_pct:>10}")

    # --- pregame matchups: same before/after, for a handful of real, currently-unsettled matches ---
    off_matchups = off_output["matchups"]
    on_matchups = on_output["matchups"]
    common_matches = sorted(set(off_matchups) & set(on_matchups))[:8]

    # model_prob_a, not p_slot_a: p_slot_a is the blended "official" value (market when The Odds
    # API has a price for the pairing, model only as fallback), so it's flat between OFF/ON
    # whenever a market price exists - it's not the number the four corrections actually touch.
    # model_prob_a is the model's own view every time, corrections included, so it's the real
    # before/after signal here.
    print(f"\n{'=' * 90}\nPregame matchup - MODEL's own P(slot_a wins): corrections OFF vs ON, same "
          f"real live matchups (p_slot_a/blended is omitted here - it's market-priced for all of "
          f"these via The Odds API, so it wouldn't move regardless of these corrections)\n{'=' * 90}")
    print(f"{'Match ID':<10} {'Slot A':<24} {'Slot B':<24} {'Model P(A) OFF':>15} {'Model P(A) ON':>15} {'Change':>10}")
    print("-" * 100)
    for match_id in common_matches:
        off_m, on_m = off_matchups[match_id], on_matchups[match_id]
        before_p, after_p = off_m["model_prob_a"], on_m["model_prob_a"]
        print(f"{match_id:<10} {off_m['slot_a']:<24} {off_m['slot_b']:<24} "
              f"{before_p:>14.1%} {after_p:>14.1%} {after_p - before_p:>+9.1%}")

    print(f"\n{'=' * 90}\nNOTE on upset_boost: confirmed via the call graph (see this script's module "
          f"docstring) that bracket_export.py's live p_champ simulation never wires up prior_beaten_elo, "
          f"so upset_boost cannot affect this output as currently built - it's only active in "
          f"hybrid_simulation.py's round-by-round replay, a different code path from this export. "
          f"Toggling it here would be cosmetic, not real, so it's reported as structurally inert "
          f"rather than faked into the diff above.\n{'=' * 90}")


# --- hybrid_simulation.py --all-rounds: full round-by-round progression, not just one snapshot ---

ROUND_FILE_RE = re.compile(r"_through_round_(\d+)(_partial)?\.csv$")


def run_hybrid_all_rounds(bracket_path, n_simulations, corrections_on, out_dir):
    """Runs hybrid_simulation.py's own main() in-process (same monkeypatch approach as
    run_export above - main() reads sys.argv itself, so it's set here rather than calling
    argparse-internal pieces directly) with --all-rounds, once per condition. Corrections-off
    disables upset_boost via the real --no-use-upset-boost flag (the only one of the four
    hybrid_simulation.py itself exposes) on top of the same rank/confidence/layoff monkeypatch
    run_export's corrections_off/on already use - so all four are actually off together, not just
    three. Returns {round_num: DataFrame} read back from whatever CSVs main() wrote to out_dir -
    reading its real output files rather than reimplementing its round-selection logic here."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    argv = [
        "hybrid_simulation.py", str(bracket_path), "--all-rounds",
        "--seed", str(SEED), "--simulations", str(n_simulations), "--output-dir", str(out_dir),
        "--use-upset-boost" if corrections_on else "--no-use-upset-boost",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        hybrid_simulation.main()
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(f"hybrid_simulation.py exited with code {e.code} for {argv}")
    finally:
        sys.argv = old_argv

    rounds = {}
    for csv_path in out_dir.glob("*_through_round_*.csv"):
        m = ROUND_FILE_RE.search(csv_path.name)
        if m is None:
            continue
        round_num = int(m.group(1))
        is_partial = m.group(2) is not None
        # a partial-round file coexists with a same-numbered full-round file only when that round
        # IS the boundary case (last fully-known round's own partial continuation) - main() only
        # ever writes one or the other per round number in a real --all-rounds run, so this can't
        # collide; kept explicit (not just "last write wins") so a real collision would be obvious
        # from the printed round number being silently sourced from the partial file instead.
        rounds[round_num] = pd.read_csv(csv_path).set_index("player")["win_count"]
        rounds.setdefault("_partial_rounds", set())
        if is_partial:
            rounds["_partial_rounds"].add(round_num)
    return rounds


def report_hybrid_round(round_num, off_counts, on_counts, n_simulations, is_partial):
    common = off_counts.index.intersection(on_counts.index)
    df = pd.DataFrame({
        "off": off_counts.reindex(common, fill_value=0),
        "on": on_counts.reindex(common, fill_value=0),
    })
    df["delta"] = df["on"] - df["off"]
    df["p_off"] = df["off"] / n_simulations
    df["p_on"] = df["on"] / n_simulations
    df = df.reindex(df["delta"].abs().sort_values(ascending=False).index)

    label = f"Round {round_num}" + (" (PARTIAL)" if is_partial else "")
    print(f"\n--- {label}: win_count, corrections OFF vs ON (top 15 by |change|) ---")
    print(f"{'Player':<24} {'win_count OFF':>14} {'win_count ON':>14} {'p_champ OFF':>12} {'p_champ ON':>12} {'Delta':>8}")
    print("-" * 90)
    for player, row in df.head(15).iterrows():
        print(f"{player:<24} {int(row['off']):>14} {int(row['on']):>14} "
              f"{row['p_off']:>11.1%} {row['p_on']:>11.1%} {int(row['delta']):>+8}")


def run_hybrid_comparison(bracket_path, n_simulations):
    print(f"\n{'=' * 90}\nHYBRID_SIMULATION.PY --all-rounds: full round-by-round progression, "
          f"corrections OFF (raw Elo + no upset-boost, matching what Daron saw) vs ON (current "
          f"default)\n{'=' * 90}")

    saved = {name: copy.deepcopy(getattr(win_probability, name)) for name in CORRECTION_CONSTANTS}

    with freeze_live_data():
        _force_corrections_off()
        off_rounds = run_hybrid_all_rounds(bracket_path, n_simulations, corrections_on=False,
                                            out_dir=SCRATCH_DIR / "hybrid_off")

        _restore_corrections(saved)
        on_rounds = run_hybrid_all_rounds(bracket_path, n_simulations, corrections_on=True,
                                           out_dir=SCRATCH_DIR / "hybrid_on")

    off_partial = off_rounds.pop("_partial_rounds", set())
    on_partial = on_rounds.pop("_partial_rounds", set())
    common_round_nums = sorted(set(off_rounds) & set(on_rounds))
    if not common_round_nums:
        print("No round checkpoints in common between the OFF and ON runs - nothing to compare.")
        return

    for round_num in common_round_nums:
        is_partial = round_num in off_partial or round_num in on_partial
        report_hybrid_round(round_num, off_rounds[round_num], on_rounds[round_num], n_simulations, is_partial)


if __name__ == "__main__":
    run()
    bracket_path = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else DEFAULT_BRACKET
    n_simulations = N_SIMULATIONS
    for i, arg in enumerate(sys.argv):
        if arg == "--simulations" and i + 1 < len(sys.argv):
            n_simulations = int(sys.argv[i + 1])
    run_hybrid_comparison(bracket_path, n_simulations)
