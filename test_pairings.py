"""Tests for pairings.py — multi-rotation, skill-balanced, doubles+singles."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from pairings import (
    PairingPlan,
    append_to_history,
    compute_display_names,
    make_plan,
    recent_pairs,
    swap_players_in_plan,
    swap_rotations_in_plan,
)

# Fake names (deterministic) for size-sensitive tests.
FAKE_NAMES = [f"Player{n:02d}" for n in range(25)]


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@pytest.fixture
def fake_roster(tmp_path):
    players = {
        name: {"gender": "?", "rating": "?", "notes": ""} for name in FAKE_NAMES
    }
    players_path = tmp_path / "players.json"
    history_path = tmp_path / "history.json"
    _write(players_path, players)
    _write(history_path, [])
    return players_path, history_path


# ---------- integrity helpers --------------------------------------------


def _assert_plan_integrity(plan: PairingPlan):
    """Core invariants for every non-degenerate plan."""
    for rot in plan.rotations:
        on_court = [p for c in rot.courts for p in c.players]
        assert len(on_court) == len(set(on_court)), (
            f"rotation {rot.rotation_num}: duplicate on-court players"
        )
        assert set(on_court) | set(rot.sit_outs) == set(plan.attendees), (
            f"rotation {rot.rotation_num}: on-court + sit-outs != attendees"
        )
        for c in rot.courts:
            if c.mode == "doubles":
                assert len(c.players) == 4
                assert len(c.pairs) == 2
                pair_players = {p for pair in c.pairs for p in pair}
                assert pair_players == set(c.players)
            elif c.mode == "singles":
                assert len(c.players) == 2
                assert len(c.pairs) == 1
                assert set(c.pairs[0]) == set(c.players)
            else:
                raise AssertionError(f"unexpected court mode: {c.mode}")


# ---------- capacity & sit-out policy -----------------------------------


def test_16_on_4_courts_all_doubles_no_sitouts(fake_roster):
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:16], players, hist, num_courts=4, num_rotations=3, seed=1
    )
    assert len(plan.rotations) == 3
    for rot in plan.rotations:
        assert len(rot.courts) == 4
        assert all(c.mode == "doubles" for c in rot.courts)
        assert rot.sit_outs == []
    # Default 3-rotation durations are [45, 40, 35] (club standard).
    assert [r.start_time for r in plan.rotations] == ["19:30", "20:15", "20:55"]
    assert [r.end_time for r in plan.rotations] == ["20:15", "20:55", "21:30"]
    _assert_plan_integrity(plan)


def test_14_on_4_courts_yields_3_doubles_plus_1_singles(fake_roster):
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:14], players, hist, num_courts=4, num_rotations=3, seed=1
    )
    for rot in plan.rotations:
        modes = [c.mode for c in rot.courts]
        assert modes.count("doubles") == 3
        assert modes.count("singles") == 1
        assert rot.sit_outs == []  # no sit-outs for even count within capacity
    _assert_plan_integrity(plan)


def test_12_on_4_courts_yields_2_doubles_plus_2_singles(fake_roster):
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:12], players, hist, num_courts=4, num_rotations=3, seed=1
    )
    for rot in plan.rotations:
        modes = [c.mode for c in rot.courts]
        assert modes.count("doubles") == 2
        assert modes.count("singles") == 2
        assert rot.sit_outs == []
    _assert_plan_integrity(plan)


def test_15_odd_on_4_courts_one_sitout_rotating(fake_roster):
    """15 is odd: 1 sits out each rotation, remaining 14 → 3D + 1S."""
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:15], players, hist, num_courts=4, num_rotations=3, seed=2
    )
    sit_counts: Counter[str] = Counter()
    for rot in plan.rotations:
        assert len(rot.sit_outs) == 1
        sit_counts.update(rot.sit_outs)
        modes = [c.mode for c in rot.courts]
        assert modes.count("doubles") == 3
        assert modes.count("singles") == 1
    # Each player sits out at most once in 3 rotations (15 slots ÷ 3 = 1)
    assert max(sit_counts.values(), default=0) <= 1
    _assert_plan_integrity(plan)


def test_18_on_4_courts_rejects_over_capacity(fake_roster):
    players, hist = fake_roster
    with pytest.raises(ValueError, match="exceeds capacity"):
        make_plan(
            FAKE_NAMES[:18], players, hist, num_courts=4, num_rotations=3, seed=1
        )


def test_too_few_players_degenerate(fake_roster):
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:3], players, hist, num_courts=4, num_rotations=3, seed=1
    )
    for rot in plan.rotations:
        assert rot.courts == []
        assert rot.sit_outs == FAKE_NAMES[:3]
    assert "Only 3 attendees" in plan.notes


def test_custom_court_labels_are_used(fake_roster):
    """Admin gives specific court numbers; the plan surfaces those labels."""
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:16],
        players,
        hist,
        court_labels=[7, 8, 9, 10],
        num_rotations=3,
        seed=1,
    )
    assert plan.court_labels == ["7", "8", "9", "10"]
    labels_on_court = {c.court_label for rot in plan.rotations for c in rot.courts}
    assert labels_on_court == {"7", "8", "9", "10"}


def test_singles_goes_on_highest_labelled_courts(fake_roster):
    """With 14 on [4,5,6,7], singles should be on court '7'."""
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:14],
        players,
        hist,
        court_labels=[4, 5, 6, 7],
        num_rotations=3,
        seed=1,
    )
    for rot in plan.rotations:
        doubles_labels = {c.court_label for c in rot.courts if c.mode == "doubles"}
        singles_labels = {c.court_label for c in rot.courts if c.mode == "singles"}
        assert doubles_labels == {"4", "5", "6"}
        assert singles_labels == {"7"}


# ---------- skill balance & singles selection ----------------------------


def test_singles_picks_strongest_players(tmp_path):
    """Strong (rating 1-2) players should preferentially play singles."""
    # Ratings: 6 strong, 8 middling → 14 attendees, 4 courts → 1 singles court.
    strong = [f"Strong{i:02d}" for i in range(6)]
    middling = [f"Mid{i:02d}" for i in range(8)]
    all_players = strong + middling
    players_path = tmp_path / "players.json"
    hist_path = tmp_path / "history.json"
    roster = {}
    for n in strong:
        roster[n] = {"gender": "?", "rating": 2, "notes": ""}
    for n in middling:
        roster[n] = {"gender": "?", "rating": 4, "notes": ""}
    _write(players_path, roster)
    _write(hist_path, [])

    plan = make_plan(
        all_players,
        players_path,
        hist_path,
        num_courts=4,
        num_rotations=3,
        seed=1,
    )
    # Every singles slot (2 × 3 rotations = 6) must be filled by a strong player.
    for rot in plan.rotations:
        for c in rot.courts:
            if c.mode == "singles":
                for player in c.players:
                    assert player in strong, (
                        f"singles player {player} is not one of the strong players"
                    )


def test_skill_balance_prefers_balanced_pairs(tmp_path):
    """Given ratings [1, 1, 5, 5] × 4, the balanced layout pairs a 1 with a 5."""
    # 4 strong and 4 weak → 8 players, 2 courts, all doubles.
    strong = ["A", "B", "C", "D"]
    weak = ["W", "X", "Y", "Z"]
    roster = {n: {"gender": "?", "rating": 1, "notes": ""} for n in strong}
    roster.update({n: {"gender": "?", "rating": 5, "notes": ""} for n in weak})
    players_path = tmp_path / "players.json"
    hist_path = tmp_path / "history.json"
    _write(players_path, roster)
    _write(hist_path, [])

    plan = make_plan(
        strong + weak,
        players_path,
        hist_path,
        num_courts=2,
        num_rotations=1,
        seed=1,
    )
    # For each doubles court, the two pairs should have equal rating sums
    # (i.e. each pair is one strong + one weak).
    for rot in plan.rotations:
        for c in rot.courts:
            pair_sums = []
            for pair in c.pairs:
                s = sum(
                    1 if p in strong else 5 for p in pair
                )
                pair_sums.append(s)
            assert pair_sums[0] == pair_sums[1], (
                f"court {c.court_label}: pair sums {pair_sums} — not balanced"
            )


# ---------- partner diversity & history -----------------------------------


def test_sixteen_no_partner_repeats_within_evening(fake_roster):
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:16], players, hist, num_courts=4, num_rotations=3, seed=7
    )
    seen_pairs: set[frozenset] = set()
    for rot in plan.rotations:
        for c in rot.courts:
            for pair in c.pairs:
                fs = frozenset(pair)
                assert fs not in seen_pairs, f"pair {pair} repeated"
                seen_pairs.add(fs)


def test_weekly_repeat_is_avoided_where_possible(fake_roster):
    players, hist = fake_roster
    last_pair = [FAKE_NAMES[0], FAKE_NAMES[1]]
    _write(
        hist,
        [
            {
                "date": "2026-04-16",
                "attendees": FAKE_NAMES[:16],
                "rotations": [
                    {
                        "courts": [
                            {"pairs": [last_pair, [FAKE_NAMES[2], FAKE_NAMES[3]]]}
                        ]
                    }
                ],
            }
        ],
    )
    banned = frozenset(last_pair)
    repeats = 0
    trials = 20
    for seed in range(trials):
        plan = make_plan(
            FAKE_NAMES[:16], players, hist, num_courts=4, num_rotations=3, seed=seed
        )
        pairs = {
            frozenset(pair)
            for rot in plan.rotations
            for c in rot.courts
            for pair in c.pairs
        }
        if banned in pairs:
            repeats += 1
    assert repeats == 0


# ---------- metadata & I/O ------------------------------------------------


def test_unknown_attendees_flagged(fake_roster):
    players, hist = fake_roster
    unknown = "Stranger Danger"
    plan = make_plan(
        FAKE_NAMES[:15] + [unknown],
        players,
        hist,
        num_courts=4,
        num_rotations=3,
        seed=1,
    )
    assert unknown in plan.unknown_attendees


def test_plan_is_json_serialisable(fake_roster):
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:16], players, hist, num_courts=4, num_rotations=3, seed=1
    )
    s = json.dumps(plan.to_dict())
    rt = json.loads(s)
    assert rt["court_labels"] == ["1", "2", "3", "4"]
    assert len(rt["rotations"]) == 3
    first = rt["rotations"][0]["courts"][0]
    assert first["mode"] in ("doubles", "singles")
    assert isinstance(first["pairs"][0], list)


def test_append_to_history_roundtrip(fake_roster):
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:16], players, hist, num_courts=4, num_rotations=3, seed=1
    )
    append_to_history(plan, hist)
    data = json.loads(Path(hist).read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["date"] == plan.date


def test_append_to_history_accepts_dict(fake_roster):
    # admin_bot path: the draft has been persisted as a dict in session
    # state (possibly edited). append_to_history must accept it directly.
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:16], players, hist, num_courts=4, num_rotations=3, seed=2
    )
    append_to_history(plan.to_dict(), hist)
    data = json.loads(Path(hist).read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["date"] == plan.date


# ---------- plan editing (swap_players / swap_rotations) ----------------


def _names_in_rotation(rot):
    on_court = {p for c in rot["courts"] for p in c["players"]}
    return on_court | set(rot.get("sit_outs", []))


def test_swap_players_swaps_across_all_rotations(fake_roster):
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:16], players, hist, num_courts=4, num_rotations=3, seed=3
    ).to_dict()
    # Capture each player's rotation footprint before the swap so we can
    # verify they really swapped places, not just got renamed somewhere.
    a, b = "Player00", "Player01"
    before = {a: [], b: []}
    for rot in plan["rotations"]:
        for c in rot["courts"]:
            if a in c["players"]:
                before[a].append((rot["rotation_num"], c["court_label"]))
            if b in c["players"]:
                before[b].append((rot["rotation_num"], c["court_label"]))

    swapped = swap_players_in_plan(plan, a, b)
    assert swapped == [1, 2, 3]

    after = {a: [], b: []}
    for rot in plan["rotations"]:
        for c in rot["courts"]:
            if a in c["players"]:
                after[a].append((rot["rotation_num"], c["court_label"]))
            if b in c["players"]:
                after[b].append((rot["rotation_num"], c["court_label"]))
    # After the swap, A is where B was and vice versa.
    assert after[a] == before[b]
    assert after[b] == before[a]


def test_swap_players_one_rotation(fake_roster):
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:16], players, hist, num_courts=4, num_rotations=3, seed=4
    ).to_dict()
    rot1_names_before = _names_in_rotation(plan["rotations"][0])
    rot2_names_before = _names_in_rotation(plan["rotations"][1])

    swapped = swap_players_in_plan(plan, "Player00", "Player01", rotation_num=1)
    assert swapped == [1]

    # Rotation 2 should be untouched in terms of pair structure (its
    # players sit in the same slots).
    assert _names_in_rotation(plan["rotations"][1]) == rot2_names_before
    # Rotation 1 still has the same set of names — just rearranged.
    assert _names_in_rotation(plan["rotations"][0]) == rot1_names_before


def test_swap_players_raises_when_no_rotation_has_both(fake_roster):
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:8], players, hist, num_courts=2, num_rotations=1, seed=5
    ).to_dict()
    with pytest.raises(KeyError):
        swap_players_in_plan(plan, "Player00", "NotInPlan")


def test_swap_rotations_swaps_payload_keeps_times(fake_roster):
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:16], players, hist, num_courts=4, num_rotations=3, seed=6
    ).to_dict()
    rot1_courts_before = plan["rotations"][0]["courts"]
    rot2_courts_before = plan["rotations"][1]["courts"]
    rot1_start_before = plan["rotations"][0]["start_time"]
    rot1_end_before = plan["rotations"][0]["end_time"]

    swap_rotations_in_plan(plan, 1, 2)

    # Times stayed put.
    assert plan["rotations"][0]["start_time"] == rot1_start_before
    assert plan["rotations"][0]["end_time"] == rot1_end_before
    # Payloads moved.
    assert plan["rotations"][0]["courts"] == rot2_courts_before
    assert plan["rotations"][1]["courts"] == rot1_courts_before


def test_swap_rotations_rejects_out_of_range(fake_roster):
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:8], players, hist, num_courts=2, num_rotations=2, seed=7
    ).to_dict()
    with pytest.raises(ValueError):
        swap_rotations_in_plan(plan, 1, 99)


def test_recent_pairs_parses_multi_rotation_history():
    history = [
        {
            "rotations": [
                {"courts": [{"pairs": [["A", "B"], ["C", "D"]]}]},
                {"courts": [{"pairs": [["E", "F"]]}]},
            ]
        }
    ]
    pairs = recent_pairs(history, lookback=1)
    assert frozenset(["A", "B"]) in pairs
    assert frozenset(["E", "F"]) in pairs
    assert len(pairs) == 3


# ---------- display names -------------------------------------------------


def test_display_names_basic():
    # Unique first names → first name only (no surname initial).
    d = compute_display_names(["Geoff Chapman", "Silvia Musso", "Hannah Banana"])
    assert d == {
        "Geoff Chapman": "Geoff",
        "Silvia Musso": "Silvia",
        "Hannah Banana": "Hannah",
    }


def test_display_names_shared_first_lengthens_prefix():
    d = compute_display_names(["Paul Vickers", "Paul Vincent", "Paul Abbott"])
    assert d["Paul Abbott"] == "Paul A"
    assert d["Paul Vickers"] == "Paul Vic"
    assert d["Paul Vincent"] == "Paul Vin"


def test_display_names_single_token():
    d = compute_display_names(["Jasmine", "Geoff Chapman"])
    assert d == {"Jasmine": "Jasmine", "Geoff Chapman": "Geoff"}


def test_display_names_paul_without_surname_coexists_with_paul_vickers():
    d = compute_display_names(["Paul", "Paul Vickers"])
    assert d["Paul"] == "Paul"
    assert d["Paul Vickers"] == "Paul V"


# ---------- gender composition rules ------------------------------------


def _gendered_roster(tmp_path, gender_for: dict[str, str]):
    """Roster files where each name's gender is set explicitly (rating '?')."""
    players = {
        n: {"gender": gender_for[n], "rating": "?", "notes": ""}
        for n in gender_for
    }
    players_path = tmp_path / "players.json"
    history_path = tmp_path / "history.json"
    _write(players_path, players)
    _write(history_path, [])
    return players_path, history_path


def _has_3F1M_court(plan):
    for rot in plan.rotations:
        for c in rot.courts:
            if c.mode != "doubles":
                continue
            g = [plan.attendees and None for _ in c.players]
            # We don't have genders on the plan itself; count via name lookup
            # in the caller's mapping is simpler — see specific tests below.
            del g
    return False  # placeholder; tests do their own gender lookup


def test_gender_avoids_3F1M_court(tmp_path):
    # 5 women + 3 men → 8 players → 2 courts. The forbidden split is
    # 3F+1M / 2F+2M; the algorithm should prefer 4F / 1F+3M.
    names = [f"F{i}" for i in range(5)] + [f"M{i}" for i in range(3)]
    gender_for = {n: ("F" if n.startswith("F") else "M") for n in names}
    players_path, history_path = _gendered_roster(tmp_path, gender_for)
    plan = make_plan(
        names, players_path, history_path,
        num_courts=2, num_rotations=1, seed=7,
    )
    for rot in plan.rotations:
        for c in rot.courts:
            if c.mode != "doubles":
                continue
            f = sum(1 for p in c.players if gender_for[p] == "F")
            m = sum(1 for p in c.players if gender_for[p] == "M")
            assert not (f == 3 and m == 1), (
                f"forbidden 3F+1M on court {c.court_label}: {c.players}"
            )


def test_gender_avoids_MM_vs_FF_on_mixed_court(tmp_path):
    # 4M + 4F on 2 courts: each court ends up 2M+2F. Forbidden pairing is
    # MM-vs-FF; only MF-vs-MF is allowed within those courts.
    names = [f"F{i}" for i in range(4)] + [f"M{i}" for i in range(4)]
    gender_for = {n: ("F" if n.startswith("F") else "M") for n in names}
    players_path, history_path = _gendered_roster(tmp_path, gender_for)
    plan = make_plan(
        names, players_path, history_path,
        num_courts=2, num_rotations=1, seed=11,
    )
    for rot in plan.rotations:
        for c in rot.courts:
            if c.mode != "doubles":
                continue
            f = sum(1 for p in c.players if gender_for[p] == "F")
            m = sum(1 for p in c.players if gender_for[p] == "M")
            if f == 2 and m == 2:
                pa, pb = c.pairs
                gen_a = sorted(gender_for[p] for p in pa)
                gen_b = sorted(gender_for[p] for p in pb)
                assert {tuple(gen_a), tuple(gen_b)} != {("F", "F"), ("M", "M")}, (
                    f"forbidden MM-vs-FF on {c.court_label}: {c.pairs}"
                )


# ---------- singles-preference rule -------------------------------------


def _singles_pref_roster(tmp_path, prefs: dict[str, str], rating: int = 3):
    players = {
        n: {"gender": "?", "rating": rating, "notes": "", "singles": pref}
        for n, pref in prefs.items()
    }
    players_path = tmp_path / "players.json"
    history_path = tmp_path / "history.json"
    _write(players_path, players)
    _write(history_path, [])
    return players_path, history_path


def test_singles_avoids_avoid_pref(tmp_path):
    # 6 players + 2 courts → 1 doubles + 1 singles (2 singles slots).
    # Two players opt out of singles; four are neutral. The singles court
    # should fill from the neutrals only.
    avoid = {"AvoidA", "AvoidB"}
    names = list(avoid) + ["NeutralA", "NeutralB", "NeutralC", "NeutralD"]
    prefs = {n: ("avoid" if n in avoid else "") for n in names}
    players_path, history_path = _singles_pref_roster(tmp_path, prefs)
    plan = make_plan(
        names, players_path, history_path,
        num_courts=2, num_rotations=1, seed=4,
    )
    rot = plan.rotations[0]
    singles_courts = [c for c in rot.courts if c.mode == "singles"]
    assert len(singles_courts) == 1
    on_singles = set(singles_courts[0].players)
    assert avoid.isdisjoint(on_singles), (
        f"avoid-singles player ended up on singles: {on_singles & avoid}"
    )


def test_singles_exclude_overrides_roster_pref(tmp_path):
    # Roster says 'prefer' but singles_exclude rules them out for this run.
    prefer = "PreferA"
    names = [prefer] + [f"Neutral{i}" for i in range(5)]
    prefs = {n: ("prefer" if n == prefer else "") for n in names}
    players_path, history_path = _singles_pref_roster(tmp_path, prefs)
    plan = make_plan(
        names, players_path, history_path,
        num_courts=2, num_rotations=1, seed=9,
        singles_exclude=[prefer],
    )
    rot = plan.rotations[0]
    singles = [c for c in rot.courts if c.mode == "singles"]
    on_singles = {p for c in singles for p in c.players}
    assert prefer not in on_singles


def test_singles_include_overrides_roster_pref(tmp_path):
    # Roster says 'avoid' but singles_include forces them onto singles
    # for this run only.
    boosted = "AvoidA"
    names = [boosted] + [f"Neutral{i}" for i in range(5)]
    prefs = {n: ("avoid" if n == boosted else "") for n in names}
    players_path, history_path = _singles_pref_roster(tmp_path, prefs)
    plan = make_plan(
        names, players_path, history_path,
        num_courts=2, num_rotations=1, seed=9,
        singles_include=[boosted],
    )
    rot = plan.rotations[0]
    singles = [c for c in rot.courts if c.mode == "singles"]
    on_singles = {p for c in singles for p in c.players}
    assert boosted in on_singles


def test_singles_no_repeat_when_enough_fresh_candidates(tmp_path):
    # 14 attendees + 4 courts: capacity 16, so 1 singles court per
    # rotation → 2 singles slots × 3 rotations = 6 singles slots. With
    # 14 candidates and the 1-per-evening cap, no player should repeat.
    names = [f"P{i}" for i in range(14)]
    prefs = {n: "" for n in names}
    players_path, history_path = _singles_pref_roster(tmp_path, prefs)
    plan = make_plan(
        names, players_path, history_path,
        num_courts=4, num_rotations=3, seed=12,
    )
    appearances: dict[str, int] = {n: 0 for n in names}
    for rot in plan.rotations:
        for c in rot.courts:
            if c.mode != "singles":
                continue
            for p in c.players:
                appearances[p] += 1
    assert all(v <= 1 for v in appearances.values()), appearances


def test_singles_repeats_only_when_forced(tmp_path):
    # 8 attendees + 3 courts: capacity 12, so 2 singles courts per
    # rotation → 4 singles slots × 3 rotations = 12 singles slots. With
    # only 8 candidates we must repeat — but the cap rule should mean
    # each player appears at most ceil(12/8) = 2 times, never 3.
    names = [f"P{i}" for i in range(8)]
    prefs = {n: "" for n in names}
    players_path, history_path = _singles_pref_roster(tmp_path, prefs)
    plan = make_plan(
        names, players_path, history_path,
        num_courts=3, num_rotations=3, seed=8,
    )
    appearances: dict[str, int] = {n: 0 for n in names}
    for rot in plan.rotations:
        for c in rot.courts:
            if c.mode != "singles":
                continue
            for p in c.players:
                appearances[p] += 1
    assert max(appearances.values()) <= 2, appearances


def test_singles_picks_prefer_first(tmp_path):
    # 6 players + 2 courts → 1 doubles + 1 singles. Two prefer singles,
    # all others neutral. Both singles slots must go to the preferers.
    prefer = {"PreferA", "PreferB"}
    names = list(prefer) + ["NeutralA", "NeutralB", "NeutralC", "NeutralD"]
    prefs = {n: ("prefer" if n in prefer else "") for n in names}
    players_path, history_path = _singles_pref_roster(tmp_path, prefs)
    plan = make_plan(
        names, players_path, history_path,
        num_courts=2, num_rotations=1, seed=5,
    )
    rot = plan.rotations[0]
    singles_courts = [c for c in rot.courts if c.mode == "singles"]
    assert len(singles_courts) == 1
    assert set(singles_courts[0].players) == prefer


def test_gender_concentrates_two_women_into_one_court(tmp_path):
    # 6M + 2F across 2 courts. 3M+1F twice is a "comparable" split; the
    # soft preference should collapse to 4M / 2M+2F so only one court has
    # a woman (no court is 3M+1F).
    names = [f"M{i}" for i in range(6)] + [f"F{i}" for i in range(2)]
    gender_for = {n: ("F" if n.startswith("F") else "M") for n in names}
    players_path, history_path = _gendered_roster(tmp_path, gender_for)
    plan = make_plan(
        names, players_path, history_path,
        num_courts=2, num_rotations=1, seed=3,
    )
    rot = plan.rotations[0]
    isolated = 0
    for c in rot.courts:
        if c.mode != "doubles":
            continue
        f = sum(1 for p in c.players if gender_for[p] == "F")
        m = sum(1 for p in c.players if gender_for[p] == "M")
        if m == 3 and f == 1:
            isolated += 1
    assert isolated == 0, "soft rule failed — both courts ended up 3M+1F"
