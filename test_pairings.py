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
    # Start times progress by 40 minutes
    assert [r.start_time for r in plan.rotations] == ["19:30", "20:10", "20:50"]
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
    d = compute_display_names(["Geoff Chapman", "Silvia Musso", "Hannah Banana"])
    assert d == {
        "Geoff Chapman": "Geoff C",
        "Silvia Musso": "Silvia M",
        "Hannah Banana": "Hannah B",
    }


def test_display_names_shared_first_lengthens_prefix():
    d = compute_display_names(["Paul Vickers", "Paul Vincent", "Paul Abbott"])
    assert d["Paul Abbott"] == "Paul A"
    assert d["Paul Vickers"] == "Paul Vic"
    assert d["Paul Vincent"] == "Paul Vin"


def test_display_names_single_token():
    d = compute_display_names(["Jasmine", "Geoff Chapman"])
    assert d == {"Jasmine": "Jasmine", "Geoff Chapman": "Geoff C"}


def test_display_names_paul_without_surname_coexists_with_paul_vickers():
    d = compute_display_names(["Paul", "Paul Vickers"])
    assert d["Paul"] == "Paul"
    assert d["Paul Vickers"] == "Paul V"
