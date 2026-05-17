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
    recent_pair_weights,
    recent_pairs,
    swap_courts_in_plan,
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
        roster[n] = {"gender": "?", "rating": 4, "notes": ""}
    for n in middling:
        roster[n] = {"gender": "?", "rating": 8, "notes": ""}
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
    """Given ratings [2, 2, 10, 10] × 4, the balanced layout pairs a 2 with a 10."""
    # 4 strong and 4 weak → 8 players, 2 courts, all doubles.
    strong = ["A", "B", "C", "D"]
    weak = ["W", "X", "Y", "Z"]
    roster = {n: {"gender": "?", "rating": 2, "notes": ""} for n in strong}
    roster.update({n: {"gender": "?", "rating": 10, "notes": ""} for n in weak})
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


def test_bracket_values_match_actual_sums(tmp_path):
    # to_dict() must precompute bracket_values that exactly equal the
    # individual-rating sums on each doubles court (and the individual
    # ratings on each singles court). Boris reads these verbatim, so a
    # mismatch here is a rendering bug.
    names = [f"P{i}" for i in range(8)]
    ratings_map = {f"P{i}": (i % 4) + 1 for i in range(8)}  # mix of 1..4
    players = {n: {"gender": "?", "rating": ratings_map[n], "notes": ""} for n in names}
    players_path = tmp_path / "players.json"
    history_path = tmp_path / "history.json"
    _write(players_path, players)
    _write(history_path, [])
    plan_dict = make_plan(
        names, players_path, history_path,
        num_courts=3, num_rotations=2, seed=99,
    ).to_dict()
    for rot in plan_dict["rotations"]:
        for c in rot["courts"]:
            assert "bracket_values" in c, c
            if c["mode"] == "doubles":
                pa, pb = c["pairs"]
                expected = [
                    ratings_map[pa[0]] + ratings_map[pa[1]],
                    ratings_map[pb[0]] + ratings_map[pb[1]],
                ]
                assert c["bracket_values"] == expected, (c["pairs"], c["bracket_values"])
            else:
                expected = [ratings_map[p] for p in c["players"]]
                assert c["bracket_values"] == expected


def test_swap_players_refreshes_bracket_values(fake_roster):
    # After swap_players_in_plan, bracket_values must reflect the new
    # player arrangement, not the pre-swap one.
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:8], players, hist, num_courts=2, num_rotations=1, seed=33
    ).to_dict()
    # All FAKE_NAMES have rating "?" → counts as 5, so any swap leaves
    # bracket_values at [6, 6]. Use a richer setup to actually validate.
    pass  # validated in the next test with explicit ratings


def test_swap_players_with_explicit_ratings_updates_bracket(tmp_path):
    names = ["P1", "P2", "P3", "P4"]
    rmap = {"P1": 1, "P2": 4, "P3": 2, "P4": 3}
    players = {n: {"gender": "?", "rating": rmap[n], "notes": ""} for n in names}
    players_path = tmp_path / "players.json"
    history_path = tmp_path / "history.json"
    _write(players_path, players)
    _write(history_path, [])
    plan_dict = make_plan(
        names, players_path, history_path, num_courts=1, num_rotations=1, seed=4
    ).to_dict()
    court = plan_dict["rotations"][0]["courts"][0]
    # Ratings 1,2,3,4 → optimal split (1+4)v(2+3) → bracket [5, 5]
    assert sorted(court["bracket_values"]) == [5, 5]
    # Swap P1 (rating 1) with P3 (rating 2). Some pair sums change.
    swap_players_in_plan(plan_dict, "P1", "P3")
    court = plan_dict["rotations"][0]["courts"][0]
    pa, pb = court["pairs"]
    expected = [rmap[pa[0]] + rmap[pa[1]], rmap[pb[0]] + rmap[pb[1]]]
    assert court["bracket_values"] == expected


def test_no_repeat_opponent_matchups_when_feasible(fake_roster):
    # 16 players / 4 courts / 3 rotations leaves plenty of room to avoid
    # opponent repeats. Across all rotations, no two players should face
    # each other on opposite sides of a doubles net more than once.
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:16], players, hist,
        num_courts=4, num_rotations=3, seed=51,
    )
    seen: set[frozenset] = set()
    for rot in plan.rotations:
        for c in rot.courts:
            if c.mode != "doubles":
                continue
            pa, pb = c.pairs
            opponents = [
                frozenset([pa[0], pb[0]]), frozenset([pa[0], pb[1]]),
                frozenset([pa[1], pb[0]]), frozenset([pa[1], pb[1]]),
            ]
            for op in opponents:
                assert op not in seen, (
                    f"opponents {sorted(op)} repeat across rotations"
                )
                seen.add(op)


def test_no_repeat_opponents_includes_singles(tmp_path):
    # 6 players / 2 courts (1 doubles + 1 singles) / 3 rotations. Each
    # singles match generates one opponent pair; that pair shouldn't
    # then be opponents in a later doubles court either.
    names = [f"P{i}" for i in range(6)]
    players = {n: {"gender": "?", "rating": "?", "notes": ""} for n in names}
    players_path = tmp_path / "players.json"
    history_path = tmp_path / "history.json"
    _write(players_path, players)
    _write(history_path, [])
    plan = make_plan(
        names, players_path, history_path,
        num_courts=2, num_rotations=3, seed=53,
    )
    seen: set[frozenset] = set()
    for rot in plan.rotations:
        for c in rot.courts:
            if c.mode == "doubles":
                pa, pb = c.pairs
                pairs_to_check = [
                    frozenset([pa[0], pb[0]]), frozenset([pa[0], pb[1]]),
                    frozenset([pa[1], pb[0]]), frozenset([pa[1], pb[1]]),
                ]
            else:
                pairs_to_check = [frozenset(c.pairs[0])]
            for op in pairs_to_check:
                assert op not in seen, sorted(op)
                seen.add(op)


def test_same_court_successive_rule_is_soft(fake_roster):
    # The soft rule discourages two players sharing a court in
    # consecutive rotations but doesn't forbid it. With 8 players on 2
    # courts (impossible to avoid some overlap in 3 rotations) the rule
    # shouldn't produce a no-feasible-plan; we just check the algorithm
    # ran and produced a valid plan.
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:8], players, hist,
        num_courts=2, num_rotations=3, seed=57,
    )
    _assert_plan_integrity(plan)


def test_within_court_pairing_is_locally_optimal(tmp_path):
    # 4 players, ratings 1/2/3/4 → only the (1+4) v (2+3) split is
    # perfectly balanced (5 v 5). The other two splits give imbalance
    # 4 and 2 respectively. Once the shuffle hands these 4 to a court,
    # _build_best_doubles_court should always pick the balanced split,
    # regardless of seed — the prior implementation got it right only
    # by luck of the shuffle.
    names = ["P1", "P2", "P3", "P4"]
    ratings = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
    players = {n: {"gender": "?", "rating": ratings[n], "notes": ""} for n in names}
    players_path = tmp_path / "players.json"
    history_path = tmp_path / "history.json"
    _write(players_path, players)
    _write(history_path, [])

    for seed in range(8):
        plan = make_plan(
            names, players_path, history_path,
            num_courts=1, num_rotations=1, seed=seed,
        )
        court = plan.rotations[0].courts[0]
        sums = sorted(
            ratings[p1] + ratings[p2] for p1, p2 in court.pairs
        )
        assert sums == [5, 5], f"seed={seed}: pairs={court.pairs}, sums={sums}"


def test_swap_courts_moves_matchups_keeps_labels(fake_roster):
    # 14 players + 4 courts → 1 singles court (Ct '4'). Move singles
    # to Ct '1' via swap_courts.
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:14], players, hist, num_courts=4, num_rotations=1, seed=20
    ).to_dict()
    # Find which court_label currently hosts singles.
    rot = plan["rotations"][0]
    singles_court = next(c for c in rot["courts"] if c["mode"] == "singles")
    singles_label = singles_court["court_label"]
    target_label = "1" if singles_label != "1" else "2"
    target_before = next(
        c for c in rot["courts"] if c["court_label"] == target_label
    )
    target_mode_before = target_before["mode"]
    target_players_before = list(target_before["players"])

    swap_courts_in_plan(plan, singles_label, target_label)

    # After the swap, the court labelled `target_label` holds singles
    # (i.e. the matchups have moved); the court labelled `singles_label`
    # holds whatever was previously on `target_label`.
    new_singles = next(
        c for c in plan["rotations"][0]["courts"]
        if c["court_label"] == target_label
    )
    assert new_singles["mode"] == "singles"
    other = next(
        c for c in plan["rotations"][0]["courts"]
        if c["court_label"] == singles_label
    )
    assert other["mode"] == target_mode_before
    assert other["players"] == target_players_before


def test_swap_courts_unknown_label_raises(fake_roster):
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:8], players, hist, num_courts=2, num_rotations=1, seed=21
    ).to_dict()
    with pytest.raises(ValueError):
        swap_courts_in_plan(plan, "1", "999")


def test_swap_courts_same_label_is_noop(fake_roster):
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:8], players, hist, num_courts=2, num_rotations=1, seed=22
    ).to_dict()
    snapshot = json.dumps(plan, sort_keys=True)
    swap_courts_in_plan(plan, "1", "1")
    assert json.dumps(plan, sort_keys=True) == snapshot


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


def _wk(*pairs):
    """Tiny helper: build a one-rotation week dict from a list of pairs."""
    return {"rotations": [{"courts": [{"pairs": [list(p) for p in pairs]}]}]}


def test_recent_pair_weights_uses_default_2_week_decay():
    # 3-week history; only the last 2 weeks are weighted now ([10, 5]):
    #   Week-3 (oldest): A-B   → unweighted (beyond the 2-week window)
    #   Week-2:          A-B, C-D
    #   Week-1 (newest): A-B, E-F
    history = [_wk(("A", "B")),
               _wk(("A", "B"), ("C", "D")),
               _wk(("A", "B"), ("E", "F"))]
    w = recent_pair_weights(history)  # default [10, 5]
    # A-B in the two weighted weeks → 10 + 5 = 15 (the oldest week
    # is outside the 2-week window and contributes nothing).
    assert w[frozenset(["A", "B"])] == 15
    # C-D only 2 weeks ago → 5
    assert w[frozenset(["C", "D"])] == 5
    # E-F is the most recent → 10
    assert w[frozenset(["E", "F"])] == 10


def test_recent_pair_weights_handles_short_history():
    # Only 1 week of history → only the [0]=10 weight applies.
    history = [_wk(("A", "B"))]
    w = recent_pair_weights(history)
    assert w == {frozenset(["A", "B"]): 10}


def test_recent_pair_weights_custom_weights():
    history = [_wk(("A", "B")), _wk(("A", "B"))]
    w = recent_pair_weights(history, weights=[20, 1])
    # Most recent week → 20, week before → 1, sum = 21.
    assert w[frozenset(["A", "B"])] == 21


def test_recent_pair_weights_empty_weights_returns_empty():
    assert recent_pair_weights([_wk(("A", "B"))], weights=[]) == {}


def test_polish_never_makes_score_worse(tmp_path):
    """polish_plan only accepts strictly improving moves, so the
    polished plan's total must be <= baseline."""
    from pairings import make_plan, polish_plan

    # 16 players, 4 courts, 3 rotations, with a wide rating spread
    # likely to produce non-trivial baseline scores.
    names = [f"P{i}" for i in range(16)]
    ratings = [2, 2, 4, 4, 6, 6, 6, 6, 8, 8, 8, 8, 10, 10, 10, 10]
    players = {
        n: {"gender": "?", "rating": ratings[i], "notes": ""}
        for i, n in enumerate(names)
    }
    players_path = tmp_path / "players.json"
    history_path = tmp_path / "history.json"
    _write(players_path, players)
    _write(history_path, [])
    baseline = make_plan(
        names, players_path, history_path,
        num_courts=4, num_rotations=3, seed=42, polish=False,
    )
    polished = make_plan(
        names, players_path, history_path,
        num_courts=4, num_rotations=3, seed=42, polish=True,
    )
    base_total = sum(int(r.get("best_score") or 0) for r in baseline.metrics["rotations"])
    polish_total = sum(int(r.get("best_score") or 0) for r in polished.metrics["rotations"])
    assert polish_total <= base_total, (
        f"polish made score worse: baseline={base_total} polished={polish_total}"
    )
    # Polish metadata should be present on the polished plan.
    assert "polish" in polished.metrics
    assert polished.metrics["polish"]["baseline_total"] == base_total
    assert polished.metrics["polish"]["final_total"] == polish_total


def test_polish_preserves_attendees_and_courts(tmp_path):
    """Polish swaps player slots but the SET of players in each
    rotation must remain the full attendee set, and court labels /
    modes / sit-outs must stay identical to the baseline."""
    from pairings import make_plan

    names = [f"P{i}" for i in range(16)]
    ratings = [2, 2, 4, 4, 6, 6, 6, 6, 8, 8, 8, 8, 10, 10, 10, 10]
    players = {
        n: {"gender": "?", "rating": ratings[i], "notes": ""}
        for i, n in enumerate(names)
    }
    players_path = tmp_path / "players.json"
    history_path = tmp_path / "history.json"
    _write(players_path, players)
    _write(history_path, [])
    polished = make_plan(
        names, players_path, history_path,
        num_courts=4, num_rotations=3, seed=42, polish=True,
    )
    for rot in polished.rotations:
        rot_players = sorted(p for c in rot.courts for p in c.players)
        # All 16 players must appear exactly once per rotation
        # (no sit-outs in this 4-court setup).
        assert rot_players == sorted(names), (
            f"rotation {rot.rotation_num} missing or duplicate players: "
            f"{rot_players}"
        )
        # Court labels stable.
        assert [c.court_label for c in rot.courts] == ["1", "2", "3", "4"]
        # All doubles for this 16-player / 4-court setup.
        for c in rot.courts:
            assert c.mode == "doubles"
            assert len(c.players) == 4
            assert len(c.pairs) == 2


def test_polish_meta_records_skipped_iff_baseline_zero(tmp_path):
    """polish_meta.skipped tracks whether polish bailed out early
    (baseline already zero)."""
    from pairings import make_plan

    # 16 same-rating players, 4 courts — gives enough room for the
    # greedy pass to land at zero.
    names = [f"P{i}" for i in range(16)]
    players = {n: {"gender": "?", "rating": 6, "notes": ""} for n in names}
    players_path = tmp_path / "players.json"
    history_path = tmp_path / "history.json"
    _write(players_path, players)
    _write(history_path, [])
    plan = make_plan(
        names, players_path, history_path,
        num_courts=4, num_rotations=3, seed=1, polish=True,
    )
    total = sum(int(r.get("best_score") or 0) for r in plan.metrics["rotations"])
    polish_meta = plan.metrics["polish"]
    if total == 0:
        assert polish_meta["skipped"] is True, polish_meta
    else:
        assert polish_meta["skipped"] is False, polish_meta
    assert polish_meta["baseline_total"] >= polish_meta["final_total"]


def test_extended_search_triggers_and_surfaces_blocking_rules(tmp_path):
    # 4 players locked onto a single doubles court for 3 rotations —
    # after R1 every pair has been both partners and opponents, so
    # R2/R3 can't avoid opponent repeats no matter the seed. Multi-seed
    # should extend up to MAX_SEED_ATTEMPTS and surface the unavoidable
    # rule violations in metrics.multi_seed.blocking_rules.
    names = [f"P{i}" for i in range(4)]
    players = {n: {"gender": "?", "rating": "?", "notes": ""} for n in names}
    players_path = tmp_path / "players.json"
    history_path = tmp_path / "history.json"
    _write(players_path, players)
    _write(history_path, [])
    plan = make_plan(
        names, players_path, history_path,
        num_courts=1, num_rotations=3, seed=1,
    )
    import pairings as _p
    ms = plan.metrics["multi_seed"]
    assert ms["extended_search"] is True
    # The 4-on-1-court scenario can never reach TARGET_SCORE so the
    # search exhausts the MAX_SEED_ATTEMPTS cap.
    assert len(ms["seeds_tried"]) == _p.MAX_SEED_ATTEMPTS
    assert ms["blocking_rules"], (
        f"expected blocking_rules with chosen_total={ms['chosen_total']}: {ms}"
    )
    for entry in ms["blocking_rules"]:
        assert entry["rule"] == "opponent_repeat"
        assert entry["rotation_num"] in (1, 2, 3)
        assert entry["penalty"] >= 500


def test_make_plan_emits_per_rotation_metrics(fake_roster):
    players, hist = fake_roster
    plan = make_plan(
        FAKE_NAMES[:16], players, hist,
        num_courts=4, num_rotations=3, seed=1,
    )
    metrics = plan.metrics
    assert metrics["max_attempts_cap"] == 1000
    assert metrics["total_seconds"] >= 0
    assert len(metrics["rotations"]) == 3
    for rot_m in metrics["rotations"]:
        assert rot_m["rotation_num"] in (1, 2, 3)
        assert isinstance(rot_m["attempts_made"], int)
        assert 1 <= rot_m["attempts_made"] <= 1000
        assert isinstance(rot_m["best_score"], int)
        assert rot_m["best_score"] >= 0
        # Breakdown should sum back to best_score.
        assert sum(rot_m["breakdown"].values()) == rot_m["best_score"]
    # multi_seed metadata is present when num_seed_attempts > 1 (default).
    assert "multi_seed" in metrics
    import pairings as _p
    seeds = metrics["multi_seed"]["seeds_tried"]
    # At minimum the initial DEFAULT_SEED_ATTEMPTS were tried; up to
    # MAX_SEED_ATTEMPTS if the extended-search trigger fired.
    assert _p.DEFAULT_SEED_ATTEMPTS <= len(seeds) <= _p.MAX_SEED_ATTEMPTS
    assert metrics["multi_seed"]["chosen_total"] == sum(
        r["best_score"] for r in metrics["rotations"]
    )


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


def test_pinned_singles_places_pair_in_specified_rotation(tmp_path):
    # 14 players + 4 courts: 1 singles court per rotation. Pin two
    # specific players to rotation 2's singles slot — they must appear
    # there and nowhere else's singles.
    names = [f"P{i}" for i in range(14)]
    prefs = {n: "" for n in names}
    players_path, history_path = _singles_pref_roster(tmp_path, prefs)
    plan = make_plan(
        names, players_path, history_path,
        num_courts=4, num_rotations=3, seed=11, num_seed_attempts=1,
        pinned_singles=[{"rotation_num": 2, "players": ["P0", "P1"]}],
    )
    rot2_singles = next(
        c for c in plan.rotations[1].courts if c.mode == "singles"
    )
    assert set(rot2_singles.players) == {"P0", "P1"}
    # And neither P0 nor P1 appears on a singles court in any other rotation
    # (cap rule still holds — pin counts as one appearance).
    for r_idx, rot in enumerate(plan.rotations):
        if r_idx == 1:
            continue
        for c in rot.courts:
            if c.mode != "singles":
                continue
            assert "P0" not in c.players and "P1" not in c.players


def test_pinned_singles_honours_court_label(tmp_path):
    # 12 players + 4 courts: capacity 16, so 2 singles courts per rotation
    # (1 doubles + 3 singles courts? no, (16-12)/2 = 2 singles, 2 doubles).
    # Pin a pair to a specific singles court_label.
    names = [f"P{i}" for i in range(12)]
    prefs = {n: "" for n in names}
    players_path, history_path = _singles_pref_roster(tmp_path, prefs)
    # Custom court labels — singles will live on the trailing two ('3','4').
    plan = make_plan(
        names, players_path, history_path,
        court_labels=["1", "2", "3", "4"], num_rotations=1, seed=13,
        pinned_singles=[{
            "rotation_num": 1,
            "players": ["P5", "P6"],
            "court_label": "4",
        }],
    )
    ct4 = next(c for c in plan.rotations[0].courts if c.court_label == "4")
    assert ct4.mode == "singles"
    assert set(ct4.players) == {"P5", "P6"}


def test_pinned_singles_rejects_duplicate_player_across_rotations(tmp_path):
    names = [f"P{i}" for i in range(14)]
    prefs = {n: "" for n in names}
    players_path, history_path = _singles_pref_roster(tmp_path, prefs)
    with pytest.raises(ValueError, match="cap"):
        make_plan(
            names, players_path, history_path,
            num_courts=4, num_rotations=3, seed=14,
            pinned_singles=[
                {"rotation_num": 1, "players": ["P0", "P1"]},
                {"rotation_num": 2, "players": ["P0", "P2"]},
            ],
        )


def test_pinned_singles_rejects_unknown_player(tmp_path):
    names = [f"P{i}" for i in range(14)]
    prefs = {n: "" for n in names}
    players_path, history_path = _singles_pref_roster(tmp_path, prefs)
    with pytest.raises(ValueError, match="not in attendees"):
        make_plan(
            names, players_path, history_path,
            num_courts=4, num_rotations=3, seed=15,
            pinned_singles=[{"rotation_num": 1, "players": ["P0", "Stranger"]}],
        )


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


def test_3m1f_court_carries_no_gender_penalty():
    """3M+1F is allowed and is NOT penalised. Only 3F+1M is hard-blocked."""
    from pairings import (
        Court, _gender_court_penalty,
    )

    court_3m1f = Court(
        court_label="1", mode="doubles",
        players=["M1", "M2", "M3", "F1"],
        pairs=[("M1", "F1"), ("M2", "M3")],
    )
    genders = {"M1": "M", "M2": "M", "M3": "M", "F1": "F"}
    assert _gender_court_penalty(court_3m1f, genders) == 0


def test_3f1m_court_carries_soft_gender_penalty():
    """3F+1M is a soft preference now (used to be hard)."""
    from pairings import (
        Court, GENDER_3F1M_PENALTY, _gender_court_penalty,
    )

    court_3f1m = Court(
        court_label="1", mode="doubles",
        players=["F1", "F2", "F3", "M1"],
        pairs=[("F1", "M1"), ("F2", "F3")],
    )
    genders = {"F1": "F", "F2": "F", "F3": "F", "M1": "M"}
    assert _gender_court_penalty(court_3f1m, genders) == GENDER_3F1M_PENALTY


def test_mm_vs_ff_pairing_is_soft():
    """The 2M+2F segregated MM-vs-FF pairing is now a soft 50-point
    penalty (was a hard 1000)."""
    from pairings import (
        Court, GENDER_MM_VS_FF_PENALTY, _gender_court_penalty,
    )

    court_mmff = Court(
        court_label="1", mode="doubles",
        players=["M1", "M2", "F1", "F2"],
        pairs=[("M1", "M2"), ("F1", "F2")],
    )
    genders = {"M1": "M", "M2": "M", "F1": "F", "F2": "F"}
    assert _gender_court_penalty(court_mmff, genders) == GENDER_MM_VS_FF_PENALTY
    assert GENDER_MM_VS_FF_PENALTY == 50


def test_rating_gap_band_classification():
    """Bands on the court's min↔max gap: 0-3 balanced, 4-5 unbalanced,
    6-7 very, 8-9 extremely."""
    from pairings import _rating_gap_band

    assert _rating_gap_band(0) == ("balanced", 0)
    assert _rating_gap_band(3) == ("balanced", 0)
    assert _rating_gap_band(4) == ("unbalanced", 20)
    assert _rating_gap_band(5) == ("unbalanced", 20)
    assert _rating_gap_band(6) == ("very_unbalanced", 50)
    assert _rating_gap_band(7) == ("very_unbalanced", 50)
    assert _rating_gap_band(8) == ("extremely_unbalanced", 100)
    assert _rating_gap_band(9) == ("extremely_unbalanced", 100)


def test_rating_gap_penalty_scales_per_player_history():
    """Penalty = base × Σ over players of 3 ** prior_non_balanced."""
    from pairings import _rating_gap_penalty

    players = ["a", "b", "c", "d"]
    court = _doubles_court(players, pairs=[("a", "c"), ("b", "d")])
    # gap = |1 .. 10| = 9 → extremely_unbalanced, base 100.
    ratings = {"a": 1, "b": 4, "c": 7, "d": 10}

    # No prior non-balanced rotations → factor 1+1+1+1 = 4 → 400.
    band, pts = _rating_gap_penalty(court, ratings, {})
    assert band == "extremely_unbalanced"
    assert pts == 100 * 4

    # One player already has 2 non-balanced rotations, rest have 0:
    # factor = 3**2 + 3**0 + 3**0 + 3**0 = 9 + 1 + 1 + 1 = 12 → 1200.
    band, pts = _rating_gap_penalty(court, ratings, {"a": 2})
    assert pts == 100 * 12

    # Balanced court (gap 3) always 0 regardless of history.
    band, pts = _rating_gap_penalty(
        _doubles_court(players), {"a": 3, "b": 4, "c": 5, "d": 6},
        {"a": 2, "b": 2, "c": 2, "d": 2},
    )
    assert band == "balanced" and pts == 0


def test_rating_gap_penalty_applies_to_singles():
    """Singles courts now get the gap penalty too (the old
    extreme-mix rule that protected singles is gone)."""
    from pairings import Court, _score_singles_courts

    singles = Court(
        court_label="9", mode="singles",
        players=["s", "w"], pairs=[("s", "w")],
    )
    ratings = {"s": 1, "w": 10}     # gap 9 → extremely_unbalanced
    # 2 players, both fresh → base 100 × (3**0 + 3**0) = 200.
    score = _score_singles_courts(
        [singles], intra_opponents=set(), prev_court_pairs=set(),
        ratings=ratings, unbalanced_count={},
    )
    assert score == 100 * 2


def _doubles_court(players, _unused_ratings=None, *, pairs=None):
    """Build a Court for tests. Defaults to splitting players as
    (1st, 2nd) vs (3rd, 4th) — pass ``pairs`` to override (e.g. to
    pre-balance for tests asserting non-pair-related penalties)."""
    from pairings import Court

    if pairs is None:
        pairs = [(players[0], players[1]), (players[2], players[3])]
    return Court(
        court_label="1", mode="doubles",
        players=list(players),
        pairs=list(pairs),
    )


def test_classify_balance_thresholds():
    from pairings import _classify_balance

    # 1-10 scale: 0-3 balanced, 4-5 unbalanced, 6-7 very, 8-9 extreme.
    assert _classify_balance(0) == "balanced"
    assert _classify_balance(3) == "balanced"
    assert _classify_balance(4) == "unbalanced"
    assert _classify_balance(5) == "unbalanced"
    assert _classify_balance(6) == "very_unbalanced"
    assert _classify_balance(7) == "very_unbalanced"
    assert _classify_balance(8) == "extremely_unbalanced"
    assert _classify_balance(9) == "extremely_unbalanced"


def test_court_max_rating_diff_handles_unknowns():
    from pairings import _court_max_rating_diff, UNKNOWN_RATING

    court = _doubles_court(["a", "b", "c", "d"])
    ratings = {"a": 2, "b": 2, "c": 4, "d": 8}
    assert _court_max_rating_diff(court, ratings) == 6
    # Unknown ratings → UNKNOWN_RATING (6) — so a (?) + (1) court is gap 5.
    ratings_q = {"a": 1, "b": UNKNOWN_RATING, "c": UNKNOWN_RATING, "d": UNKNOWN_RATING}
    assert _court_max_rating_diff(court, ratings_q) == 5
    # Now also works for singles (was hard-coded to 0 for non-doubles).
    from pairings import Court
    singles = Court(
        court_label="9", mode="singles",
        players=["x", "y"], pairs=[("x", "y")],
    )
    assert _court_max_rating_diff(singles, {"x": 2, "y": 9}) == 7


def test_very_unbalanced_court_band_penalty():
    from pairings import _score_doubles_court

    players = ["a", "b", "c", "d"]
    # Pre-balance the pair split so PAIR_IMBALANCE_WEIGHT contributes 0
    # (pairs 2+8 vs 2+8 = 10 each).
    court = _doubles_court(players, pairs=[("a", "c"), ("b", "d")])
    ratings = {"a": 2, "b": 2, "c": 8, "d": 8}    # gap = 6 → very (50)
    genders = {p: "?" for p in players}
    score = _score_doubles_court(
        court, weekly_pair_penalties={}, intra_partners=set(),
        intra_opponents=set(), prev_court_pairs=set(),
        ratings=ratings, genders=genders, unbalanced_count={},
    )
    # Only contribution: very_unbalanced base 50 × Σ 3**0 (4 fresh) = 200.
    assert score == 50 * 4


def test_unbalanced_court_costs_base_times_fresh_players():
    """A gap-4 'unbalanced' court with all-fresh players costs
    base(20) × 4 (no escalation yet) + the pair-sum imbalance."""
    from pairings import _score_doubles_court, PAIR_IMBALANCE_WEIGHT

    players = ["a", "b", "c", "d"]
    court = _doubles_court(players)               # split (a,b) vs (c,d)
    ratings = {"a": 2, "b": 4, "c": 4, "d": 6}    # gap = 4 → unbalanced
    genders = {p: "?" for p in players}
    score = _score_doubles_court(
        court, weekly_pair_penalties={}, intra_partners=set(),
        intra_opponents=set(), prev_court_pairs=set(),
        ratings=ratings, genders=genders, unbalanced_count={},
    )
    # imbalance = |2+4 - 4+6| = 4 → 4 × weight(1) = 4.
    # gap-band = unbalanced base 20 × (3**0 × 4) = 80.
    assert score == 4 * PAIR_IMBALANCE_WEIGHT + 20 * 4


def test_rating_gap_escalates_with_prior_non_balanced():
    from pairings import _score_doubles_court, PAIR_IMBALANCE_WEIGHT

    players = ["a", "b", "c", "d"]
    court = _doubles_court(players)
    ratings = {"a": 2, "b": 4, "c": 4, "d": 6}    # gap 4 → unbalanced (20)
    genders = {p: "?" for p in players}
    imb = 4 * PAIR_IMBALANCE_WEIGHT               # |6 - 10| = 4

    # Every player already has 1 non-balanced rotation:
    # factor = Σ 3**1 = 4 × 3 = 12 → 20 × 12 = 240.
    score_1 = _score_doubles_court(
        court, weekly_pair_penalties={}, intra_partners=set(),
        intra_opponents=set(), prev_court_pairs=set(),
        ratings=ratings, genders=genders,
        unbalanced_count={p: 1 for p in players},
    )
    assert score_1 == imb + 20 * 12

    # Every player already has 2 → factor = 4 × 3**2 = 36 → 720.
    score_2 = _score_doubles_court(
        court, weekly_pair_penalties={}, intra_partners=set(),
        intra_opponents=set(), prev_court_pairs=set(),
        ratings=ratings, genders=genders,
        unbalanced_count={p: 2 for p in players},
    )
    assert score_2 == imb + 20 * 36

    # Mixed: one player at 2, rest fresh → 3**2 + 3 × 3**0 = 12.
    score_mix = _score_doubles_court(
        court, weekly_pair_penalties={}, intra_partners=set(),
        intra_opponents=set(), prev_court_pairs=set(),
        ratings=ratings, genders=genders,
        unbalanced_count={"a": 2},
    )
    assert score_mix == imb + 20 * 12


def test_unbalanced_count_propagates_across_rotations(tmp_path):
    """End-to-end: with a roster where balanced layouts ARE feasible,
    the algorithm should distribute unbalanced rotations rather than
    repeatedly stranding the same players on unbalanced courts.

    12 players, 3 courts × 3 rotations. A single rating-3 player among
    rating-1/2 partners produces unbalanced (gap-2) courts only when
    paired with a rating-2 player. Across 3 rotations the rating-6
    player can play with rating-4 partners every time (balanced) by
    rotation, so no player needs > 1 unbalanced rotation.
    """
    from pairings import (
        _classify_balance, _court_max_rating_diff,
        make_plan,
    )

    names = [f"S{i}" for i in range(4)] + [f"M{i}" for i in range(7)] + ["X1"]
    rating_for: dict[str, int] = {}
    for n in names:
        if n.startswith("S"):
            rating_for[n] = 2
        elif n.startswith("M"):
            rating_for[n] = 4
        else:
            rating_for[n] = 6
    players_path = tmp_path / "players.json"
    history_path = tmp_path / "history.json"
    players_path.write_text(json.dumps({
        n: {"gender": "?", "rating": r, "notes": ""}
        for n, r in rating_for.items()
    }))
    history_path.write_text(json.dumps([]))

    plan = make_plan(
        names, players_path, history_path,
        num_courts=3, num_rotations=3, seed=1,
    )
    counts = {n: 0 for n in names}
    for rot in plan.rotations:
        for c in rot.courts:
            if c.mode != "doubles":
                continue
            diff = _court_max_rating_diff(c, plan.ratings)
            if _classify_balance(diff) != "balanced":
                for p in c.players:
                    counts[p] += 1
    assert max(counts.values()) <= 1, (
        f"a player ended up with 2+ unbalanced rotations when 1 was "
        f"feasible: {counts}"
    )


# ---------- late court (court available only from a later rotation) -----


def test_late_court_pinned_4_sit_out_before_first_rotation(fake_roster):
    """20 players, 5 courts where court '5' is only available from R2.

    Before R2: 4 pinned players sit out (along with whoever else the
    odd-count rules dictate); the late court doesn't appear on court.
    In R2: the 4 pinned land on the late court as a doubles court.
    In R3: the late court is in the full pool, no pinning.
    """
    players, hist = fake_roster
    attendees = FAKE_NAMES[:20]
    pinned = attendees[:4]
    plan = make_plan(
        attendees, players, hist,
        court_labels=[1, 2, 3, 4, 5],
        num_rotations=3,
        seed=1,
        late_court={"label": "5", "first_rotation": 2, "pinned_players": pinned},
    )
    assert plan.num_rotations == 3

    # R1: late court excluded; pinned 4 sit out (20 - 4 = 16 active, 4
    # courts × 4 = 16 → all doubles, no extra sit-outs).
    r1 = plan.rotations[0]
    r1_labels = {c.court_label for c in r1.courts}
    assert "5" not in r1_labels
    assert len(r1.courts) == 4
    for p in pinned:
        assert p in r1.sit_outs

    # R2: late court is in the plan AND is the doubles court holding the
    # 4 pinned players.
    r2 = plan.rotations[1]
    r2_labels = {c.court_label for c in r2.courts}
    assert "5" in r2_labels
    late_court_r2 = next(c for c in r2.courts if c.court_label == "5")
    assert late_court_r2.mode == "doubles"
    assert set(late_court_r2.players) == set(pinned)
    for p in pinned:
        assert p not in r2.sit_outs

    # R3: late court in pool. Pinned players are no longer required to
    # be on it; they're just regular players.
    r3 = plan.rotations[2]
    r3_labels = {c.court_label for c in r3.courts}
    assert "5" in r3_labels

    _assert_plan_integrity(plan)


def test_late_court_with_odd_count_keeps_pinned_off_sitout(fake_roster):
    """19 players (odd) + late court of label '5' from R2.

    In R1: 4 pinned + 1 odd-count sit-out = 5 sit-outs. None of the 4
    pinned can be the odd-count sit-out (so total sit-outs from the
    pinned group = exactly 4 in R1).
    """
    players, hist = fake_roster
    attendees = FAKE_NAMES[:19]
    pinned = attendees[:4]
    plan = make_plan(
        attendees, players, hist,
        court_labels=[1, 2, 3, 4, 5],
        num_rotations=3,
        seed=2,
        late_court={"label": "5", "first_rotation": 2, "pinned_players": pinned},
    )
    r1 = plan.rotations[0]
    sit_out_set = set(r1.sit_outs)
    assert set(pinned).issubset(sit_out_set)
    assert len(sit_out_set) == 5
    _assert_plan_integrity(plan)


def test_late_court_first_rotation_3_only(fake_roster):
    """Late court only joins for R3 (final rotation). Pinned sit out
    R1 + R2, then play together on the late court in R3."""
    players, hist = fake_roster
    attendees = FAKE_NAMES[:20]
    pinned = attendees[:4]
    plan = make_plan(
        attendees, players, hist,
        court_labels=[1, 2, 3, 4, 5],
        num_rotations=3,
        seed=3,
        late_court={"label": "5", "first_rotation": 3, "pinned_players": pinned},
    )
    for r_idx in (0, 1):
        rot = plan.rotations[r_idx]
        labels = {c.court_label for c in rot.courts}
        assert "5" not in labels
        for p in pinned:
            assert p in rot.sit_outs

    r3 = plan.rotations[2]
    late = next(c for c in r3.courts if c.court_label == "5")
    assert late.mode == "doubles"
    assert set(late.players) == set(pinned)
    _assert_plan_integrity(plan)


def test_late_court_rejects_non_attendee_pin(fake_roster):
    players, hist = fake_roster
    with pytest.raises(ValueError, match="not in attendees"):
        make_plan(
            FAKE_NAMES[:20], players, hist,
            court_labels=[1, 2, 3, 4, 5],
            num_rotations=3,
            seed=1,
            late_court={
                "label": "5",
                "first_rotation": 2,
                "pinned_players": ["Stranger1", "Stranger2", "Stranger3", "Stranger4"],
            },
        )


def test_late_court_requires_label_in_court_labels(fake_roster):
    players, hist = fake_roster
    with pytest.raises(ValueError, match="not in court_labels"):
        make_plan(
            FAKE_NAMES[:20], players, hist,
            court_labels=[1, 2, 3, 4],  # no '5'
            num_rotations=3,
            seed=1,
            late_court={
                "label": "5",
                "first_rotation": 2,
                "pinned_players": FAKE_NAMES[:4],
            },
        )
