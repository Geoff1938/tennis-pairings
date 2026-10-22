"""Pins (singles court placement, polish lock, late-court timing),
shared name matching, and pin clean-up when a player is removed."""
from __future__ import annotations

import json

import pytest

from name_match import match_names
from pairings import make_plan, polish_plan


def _roster(tmp_path, names, rating=3):
    players = {
        n: {"gender": "?", "rating": rating + (i % 5), "notes": "",
            "singles": ""}
        for i, n in enumerate(names)
    }
    pp, hp = tmp_path / "players.json", tmp_path / "history.json"
    pp.write_text(json.dumps(players))
    hp.write_text("[]")
    return pp, hp


# ---------- singles pin on a non-trailing court -------------------------


def test_singles_pin_on_first_court_label(tmp_path):
    # 14 players, 4 courts -> 1 singles court per rotation, normally the
    # last label. Pinning singles to court "1" used to crash.
    names = [f"P{i}" for i in range(14)]
    pp, hp = _roster(tmp_path, names)
    plan = make_plan(
        names, pp, hp, court_labels=["1", "2", "3", "4"],
        num_rotations=2, seed=3, num_seed_attempts=1,
        pinned_singles=[{"rotation_num": 1, "players": ["P0", "P1"],
                         "court_label": "1"}],
    )
    ct1 = next(c for c in plan.rotations[0].courts if c.court_label == "1")
    assert ct1.mode == "singles"
    assert set(ct1.players) == {"P0", "P1"}


# ---------- polish keeps pinned singles --------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_polish_never_breaks_pinned_singles(tmp_path, seed):
    # 12 players on 4 courts -> 2 singles courts per rotation, and a
    # lopsided pinned pair (ratings 1 v 9): without the lock, polish
    # swaps one of them with the other singles court every time.
    names = [f"P{i}" for i in range(12)]
    pp, hp = _roster(tmp_path, names)
    players = json.loads(pp.read_text())
    players["P0"]["rating"], players["P1"]["rating"] = 1, 9
    pp.write_text(json.dumps(players))
    pins = [{"rotation_num": 2, "players": ["P0", "P1"]}]
    plan = make_plan(
        names, pp, hp, num_courts=4, num_rotations=3, seed=seed,
        num_seed_attempts=1, polish=False, pinned_singles=pins,
    )
    polished = polish_plan(
        plan, seed=seed, verbose=False, max_iterations=3000,
        max_no_improvement=3000, pinned_singles=pins,
    )
    singles = [c for c in polished.rotations[1].courts if c.mode == "singles"]
    assert any(set(c.players) == {"P0", "P1"} for c in singles)


# ---------- late-court players can't be pinned before it opens ----------


LATE = {"label": "5", "first_rotation": 2,
        "pinned_players": ["P0", "P1", "P2", "P3"]}


def test_late_court_player_pinned_to_singles_before_opening(tmp_path):
    names = [f"P{i}" for i in range(18)]
    pp, hp = _roster(tmp_path, names)
    with pytest.raises(ValueError, match="waiting for late court"):
        make_plan(
            names, pp, hp, court_labels=["1", "2", "3", "4", "5"],
            num_rotations=3, seed=1, num_seed_attempts=1,
            late_court=LATE,
            pinned_singles=[{"rotation_num": 1, "players": ["P0", "P9"]}],
        )


def test_late_court_player_pinned_to_doubles_before_opening(tmp_path):
    names = [f"P{i}" for i in range(20)]
    pp, hp = _roster(tmp_path, names)
    with pytest.raises(ValueError, match="late court"):
        make_plan(
            names, pp, hp, court_labels=["1", "2", "3", "4", "5"],
            num_rotations=3, seed=1, num_seed_attempts=1,
            late_court=LATE,
            pinned_doubles=[{
                "rotation_num": 1,
                "players": ["P0", "P8", "P9", "P10"],
                "pairs": [["P0", "P8"], ["P9", "P10"]],
            }],
        )


def test_late_court_player_can_be_pinned_after_it_opens(tmp_path):
    names = [f"P{i}" for i in range(20)]
    pp, hp = _roster(tmp_path, names)
    plan = make_plan(
        names, pp, hp, court_labels=["1", "2", "3", "4", "5"],
        num_rotations=3, seed=1, num_seed_attempts=1,
        late_court=LATE,
        pinned_doubles=[{
            "rotation_num": 3,
            "players": ["P0", "P8", "P9", "P10"],
            "pairs": [["P0", "P8"], ["P9", "P10"]],
        }],
    )
    r3 = plan.rotations[2]
    assert any(set(c.players) == {"P0", "P8", "P9", "P10"} for c in r3.courts)


# ---------- name matching ----------------------------------------------


ROSTER = ["Tom Smith", "Tom Smithson", "Sally Jones", "Alan Brown",
          "Louise Clark"]


def test_exact_name_beats_longer_name():
    assert match_names("tom smith", ROSTER) == ["Tom Smith"]
    assert match_names("Tom  Smith", ROSTER) == ["Tom Smith"]


def test_whole_word_beats_substring():
    assert match_names("Smith", ROSTER) == ["Tom Smith"]


def test_short_query_matches_word_starts_only():
    assert match_names("Al", ROSTER) == ["Alan Brown"]
    assert match_names("ally", ["Sally Jones"]) == ["Sally Jones"]
    assert match_names("ly", ["Sally Jones"]) == []


def test_ambiguity_still_reported():
    assert match_names("Tom", ROSTER) == ["Tom Smith", "Tom Smithson"]


def test_partial_false_never_picks_a_longer_name():
    options = ["Sam Smithson (Member)", "Jo Bloggs"]
    assert match_names("Sam Smith", options, partial=False) == []
    options.append("Sam Smith (Member)")
    assert match_names("Sam Smith", options, partial=False) == [
        "Sam Smith (Member)"]


# ---------- removing a player clears their pins ------------------------


@pytest.fixture
def ss(tmp_path, monkeypatch):
    import session_state as s

    monkeypatch.setattr(s, "SESSION_STATE_PATH", tmp_path / "ss.json")
    return s


def test_remove_player_drops_their_pins(ss, monkeypatch):
    import admin_bot

    ss.start_tonight([f"P{i}" for i in range(12)])
    st = ss._load()
    st.pinned_doubles = [
        {"rotation_num": 1, "players": ["P0", "P1", "P2", "P3"],
         "pairs": [["P0", "P1"], ["P2", "P3"]], "court_label": None},
        {"rotation_num": 2, "players": ["P4", "P5", "P6", "P7"],
         "pairs": [["P4", "P5"], ["P6", "P7"]], "court_label": None},
    ]
    st.late_court_label = "5"
    st.late_court_first_rotation = 2
    st.late_court_pinned_players = ["P0", "P8", "P9", "P10"]
    ss._save(st)

    res = admin_bot.tool_remove_from_tonight("P0")
    assert res["ok"] and res["removed"] == "P0"
    st = ss._load()
    assert [p["players"][0] for p in st.pinned_doubles] == ["P4"]
    assert st.late_court_pinned_players == ["P8", "P9", "P10"]
    assert "pins_message" in res and "late_court_message" in res
