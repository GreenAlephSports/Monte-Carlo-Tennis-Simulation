"""Synthetic test for live_match_watcher.py - same rigor as the earlier synthetic zero-bye test:
no waiting on a real live match, every poll result is a fabricated/mocked snapshot, and the
watcher's own real code (detect_new_finals, snapshot_statuses, print_delta_report, watch) is
exercised end to end, not reimplemented.

Two layers:

1. Unit-level: detect_new_finals against hand-built (previous_statuses, current_matches) pairs -
   confirms it fires on a genuine pre/in -> post transition and stays silent on every other kind
   of change (score update mid-match, pre -> in, post but no winner yet, already-post at startup).

2. End-to-end: watch() is run against a real bracket file (so bracket loading/validation is real),
   with fetch_tournament_matches and export_bracket_json monkeypatched to return three fabricated
   scoreboard polls and two fabricated simulation exports:
     poll 1 (startup baseline)        - two matches in progress, one already Final (pre-existing)
     poll 2 (score update, still "in")- the live match's score changes, status_state unchanged
     poll 3 (genuine transition)      - the live match finishes; a real rerun should fire
   This confirms: no rerun and no delta report on poll 2; exactly one rerun and an accurate delta
   report (magnitude-ranked movers, an ELIMINATED line for the loser, unchanged players omitted)
   on poll 3; and the pre-existing Final at startup never itself triggers a report.

Usage:
    python model/live_match_watcher_test.py
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_match_watcher import detect_new_finals, snapshot_statuses, watch  # noqa: E402

BRACKET_PATH = Path(__file__).resolve().parent.parent / "brackets" / "cincinnati_2026_atp_demo.yaml"

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def match(match_id, status_state, p1, p2, winner=None, sets_1=None, sets_2=None, round_="Round 1"):
    return {
        "tournament": "Cincinnati Open", "category": "Men's Singles", "round": round_,
        "status": {"post": "Final", "in": "In Progress", "pre": "Scheduled"}[status_state],
        "status_state": status_state, "player_1": p1, "player_2": p2,
        "sets_1": sets_1 or [], "sets_2": sets_2 or [], "winner": winner,
        "start_time": "2026-08-13T17:00Z", "match_id": match_id,
    }


# ---------------------------------------------------------------------------
# Layer 1: detect_new_finals unit checks
# ---------------------------------------------------------------------------

def test_detect_new_finals():
    print("\n--- Layer 1: detect_new_finals unit checks ---")

    zverev_in = match("m1", "in", "Alexander Zverev", "Daniil Medvedev", sets_1=["4"], sets_2=["2"])
    zverev_in_later = match("m1", "in", "Alexander Zverev", "Daniil Medvedev", sets_1=["6", "3"], sets_2=["3", "1"])
    zverev_final = match("m1", "post", "Alexander Zverev", "Daniil Medvedev", winner="Alexander Zverev",
                          sets_1=["6", "6"], sets_2=["3", "2"])
    zverev_post_no_winner = match("m1", "post", "Alexander Zverev", "Daniil Medvedev", winner=None)
    other_pre = match("m2", "pre", "Tommy Paul", "Learner Tien")
    other_in = match("m2", "in", "Tommy Paul", "Learner Tien", sets_1=["2"], sets_2=["4"])
    already_final = match("m3", "post", "Taylor Fritz", "Felix Auger-Aliassime", winner="Taylor Fritz",
                           sets_1=["6", "6"], sets_2=["2", "4"])

    prev = snapshot_statuses([zverev_in, other_pre, already_final])

    score_update_only = detect_new_finals(prev, [zverev_in_later, other_pre, already_final])
    check("score update mid-match (still 'in') does not fire", score_update_only == [],
          f"got {[m['match_id'] for m in score_update_only]}")

    round_started = detect_new_finals(prev, [zverev_in, other_in, already_final])
    check("pre -> in transition does not fire", round_started == [],
          f"got {[m['match_id'] for m in round_started]}")

    already_final_again = detect_new_finals(prev, [zverev_in, other_pre, already_final])
    check("match already Final in both snapshots does not re-fire", already_final_again == [],
          f"got {[m['match_id'] for m in already_final_again]}")

    post_no_winner = detect_new_finals(prev, [zverev_post_no_winner, other_pre, already_final])
    check("status 'post' without a winner does not fire (malformed-data guard)", post_no_winner == [],
          f"got {[m['match_id'] for m in post_no_winner]}")

    genuine = detect_new_finals(prev, [zverev_final, other_pre, already_final])
    check("genuine pre/in -> post transition with a winner fires",
          [m["match_id"] for m in genuine] == ["m1"], f"got {[m['match_id'] for m in genuine]}")


# ---------------------------------------------------------------------------
# Layer 2: end-to-end watch() against fabricated polls + fabricated exports
# ---------------------------------------------------------------------------

BASELINE_EXPORT = {
    "meta": {"tournament": "cincinnati-open-men-2026", "iterations": 200, "seed": 42},
    "players": [
        {"player": "Alexander Zverev", "quarter": "Q2", "p_champ": 0.160, "p_sf": 0.320, "p_final": 0.255},
        {"player": "Daniil Medvedev", "quarter": "Q2", "p_champ": 0.075, "p_sf": 0.210, "p_final": 0.150},
        {"player": "Tommy Paul", "quarter": "Q3", "p_champ": 0.065, "p_sf": 0.260, "p_final": 0.160},
        {"player": "Learner Tien", "quarter": "Q3", "p_champ": 0.045, "p_sf": 0.180, "p_final": 0.110},
        {"player": "Taylor Fritz", "quarter": "Q4", "p_champ": 0.080, "p_sf": 0.170, "p_final": 0.115},
    ],
}

# after Zverev beats Medvedev: Zverev's own numbers rise (he's one match closer, plus he no longer
# has to get past Medvedev later), Medvedev drops out of alive_draw_names entirely (eliminated),
# Tommy Paul/Learner Tien are in the other half of the draw and genuinely unaffected, Fritz gets a
# small bump from the field thinning slightly.
AFTER_EXPORT = {
    "meta": {"tournament": "cincinnati-open-men-2026", "iterations": 200, "seed": 42},
    "players": [
        {"player": "Alexander Zverev", "quarter": "Q2", "p_champ": 0.205, "p_sf": 0.400, "p_final": 0.320},
        {"player": "Tommy Paul", "quarter": "Q3", "p_champ": 0.065, "p_sf": 0.260, "p_final": 0.160},
        {"player": "Learner Tien", "quarter": "Q3", "p_champ": 0.045, "p_sf": 0.180, "p_final": 0.110},
        {"player": "Taylor Fritz", "quarter": "Q4", "p_champ": 0.086, "p_sf": 0.178, "p_final": 0.121},
    ],
}


def test_end_to_end():
    print("\n--- Layer 2: end-to-end watch() against fabricated polls ---")

    zverev_v1 = match("z1", "in", "Alexander Zverev", "Daniil Medvedev", sets_1=["4"], sets_2=["2"], round_="Round 2")
    zverev_v2_score_update = match("z1", "in", "Alexander Zverev", "Daniil Medvedev",
                                    sets_1=["6", "3"], sets_2=["3", "1"], round_="Round 2")
    zverev_v3_final = match("z1", "post", "Alexander Zverev", "Daniil Medvedev", winner="Alexander Zverev",
                             sets_1=["6", "6"], sets_2=["3", "2"], round_="Round 2")
    already_final = match("f1", "post", "Taylor Fritz", "Felix Auger-Aliassime", winner="Taylor Fritz",
                           sets_1=["6", "6"], sets_2=["2", "4"], round_="Round 2")

    poll_1 = [zverev_v1, already_final]          # startup baseline
    poll_2 = [zverev_v2_score_update, already_final]  # score update only - must NOT fire
    poll_3 = [zverev_v3_final, already_final]     # genuine transition - must fire

    export_calls = []

    def fake_export(bracket_path, output_path=None, n_simulations=None, seed=None, dates=None):
        export_calls.append(n_simulations)
        export = BASELINE_EXPORT if len(export_calls) == 1 else AFTER_EXPORT
        return output_path, export

    fetch_calls = []

    def fake_fetch(tour, tournament_name, category, dates=None):
        fetch_calls.append(1)
        return [poll_1, poll_2, poll_3][len(fetch_calls) - 1]

    buf = io.StringIO()
    with patch("live_match_watcher.export_bracket_json", side_effect=fake_export), \
         patch("live_match_watcher.fetch_tournament_matches", side_effect=fake_fetch), \
         patch("live_match_watcher.time.sleep", return_value=None):
        with redirect_stdout(buf):
            watch(BRACKET_PATH, interval=1, n_simulations=200, seed=42, dates=None, exit_after=1)

    output = buf.getvalue()
    print(output)

    check("baseline export ran once at startup, and again exactly once after the real transition "
          "(not on the score-update-only poll)", len(export_calls) == 2, f"export ran {len(export_calls)} times")
    check("all 3 fabricated polls were consumed (startup + 2 loop iterations)", len(fetch_calls) == 3,
          f"fetch ran {len(fetch_calls)} times")

    lines = output.splitlines()
    no_completion_lines = [l for l in lines if "no new completions" in l]
    check("score-update-only poll logged as 'no new completions', no delta report", len(no_completion_lines) == 1,
          f"found {len(no_completion_lines)} such lines")

    check("exactly one 'MATCH COMPLETED' report was printed (not on the score-update poll, not "
          "for the pre-existing Final)", output.count("MATCH COMPLETED") == 1,
          f"found {output.count('MATCH COMPLETED')}")

    check("delta report names the actual completed match (Zverev d. Medvedev)",
          "Alexander Zverev vs Daniil Medvedev" in output and "Final" in output.split("MATCH COMPLETED")[1][:400])

    zverev_line = next((l for l in lines if l.strip().startswith("Alexander Zverev")), "")
    check("Zverev's p_champ delta matches the fabricated before/after exactly (0.160 -> 0.205, +0.045)",
          "0.160 -> 0.205" in zverev_line and "+0.045" in zverev_line, zverev_line)
    check("Zverev's p_sf delta matches (0.320 -> 0.400, +0.080)",
          "0.320 -> 0.400" in zverev_line and "+0.080" in zverev_line, zverev_line)
    check("Zverev's p_final delta matches (0.255 -> 0.320, +0.065)",
          "0.255 -> 0.320" in zverev_line and "+0.065" in zverev_line, zverev_line)

    medvedev_line = next((l for l in lines if l.strip().startswith("Daniil Medvedev")), "")
    check("Medvedev (the loser, dropped from alive_draw_names) is reported ELIMINATED, not silently omitted",
          "ELIMINATED" in medvedev_line, medvedev_line)

    fritz_lines = [l for l in lines if "Taylor Fritz" in l and ("p_champ" in l or "ELIMINATED" in l)]
    check("Fritz's small genuine movement (0.080 -> 0.086) is still reported, not rounded away as noise",
          any("0.080 -> 0.086" in l for l in fritz_lines), fritz_lines)

    check("Tommy Paul (genuinely unchanged before/after) is omitted from the ranked delta list "
          "rather than printed as a zero-magnitude row",
          not any(l.strip().startswith("Tommy Paul") and "p_champ" in l for l in lines))
    check("Learner Tien (genuinely unchanged before/after) is likewise omitted",
          not any(l.strip().startswith("Learner Tien") and "p_champ" in l for l in lines))

    mover_names = []
    for l in lines:
        stripped = l.strip()
        if stripped.startswith(("Alexander Zverev", "Taylor Fritz")) and "p_champ" in stripped:
            mover_names.append(stripped.split()[0] + " " + stripped.split()[1])
    check("movers are ranked by magnitude of change, largest first (Zverev's combined delta "
          "dwarfs Fritz's)", mover_names == ["Alexander Zverev", "Taylor Fritz"], mover_names)


def main():
    test_detect_new_finals()
    test_end_to_end()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
