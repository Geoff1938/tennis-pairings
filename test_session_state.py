"""Tests for session_state.py — phase transitions and persistence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Point SESSION_STATE_PATH at a tmp file so tests don't touch the
    real one. Returns the imported module so tests can call its API.
    """
    import session_state as ss

    target = tmp_path / "session_state.json"
    monkeypatch.setattr(ss, "SESSION_STATE_PATH", target)
    return ss


def test_phase_defaults_to_empty_for_fresh_state(isolated_state):
    ss = isolated_state
    assert ss.get_phase() == ""
    state = ss.get_tonight()
    assert state.phase == ""


def test_session_type_round_trips_through_start_tonight(isolated_state):
    ss = isolated_state
    state = ss.start_tonight(
        ["Geoff Chapman"], session_type="saturday", date="2026-05-30",
    )
    assert state.session_type == "saturday"
    # Reload from disk to confirm persistence.
    reloaded = ss.get_tonight()
    assert reloaded.session_type == "saturday"
    raw = json.loads(Path(ss.SESSION_STATE_PATH).read_text())
    assert raw["session_type"] == "saturday"


def test_session_type_defaults_to_empty(isolated_state):
    ss = isolated_state
    ss.start_tonight(["Geoff Chapman"])
    assert ss.get_tonight().session_type == ""


def test_set_and_get_phase_round_trip(isolated_state):
    ss = isolated_state
    ss.set_phase("awaiting_extras")
    assert ss.get_phase() == "awaiting_extras"
    # Should persist to file too.
    raw = json.loads(Path(ss.SESSION_STATE_PATH).read_text())
    assert raw["phase"] == "awaiting_extras"


def test_set_phase_through_lifecycle(isolated_state):
    ss = isolated_state
    for phase in [
        "awaiting_extras",
        "ready_to_generate",
        "draft_ready",
        "finalised",
        "",  # back to no in-flight session
    ]:
        ss.set_phase(phase)
        assert ss.get_phase() == phase


def test_set_phase_rejects_unknown_value(isolated_state):
    ss = isolated_state
    with pytest.raises(ValueError, match="unknown phase"):
        ss.set_phase("nonsense")


def test_phase_survives_other_state_writes(isolated_state):
    ss = isolated_state
    ss.set_phase("draft_ready")
    ss.add_to_tonight("Geoff Chapman")
    assert ss.get_phase() == "draft_ready"
    state = ss.get_tonight()
    assert "Geoff Chapman" in state.attendees
    assert state.phase == "draft_ready"


def test_clear_tonight_resets_phase(isolated_state):
    ss = isolated_state
    ss.set_phase("draft_ready")
    ss.clear_tonight()
    assert ss.get_phase() == ""


def test_load_normalises_invalid_phase_to_empty(isolated_state, tmp_path):
    ss = isolated_state
    # Write a corrupted state with an invalid phase value.
    Path(ss.SESSION_STATE_PATH).write_text(
        json.dumps({"phase": "not_a_real_phase", "attendees": []})
    )
    assert ss.get_phase() == ""


def test_test_mode_defaults_to_false(isolated_state):
    ss = isolated_state
    assert ss.get_tonight().test_mode is False


def test_start_tonight_persists_test_mode(isolated_state):
    ss = isolated_state
    ss.start_tonight(["Alice", "Bob"], test_mode=True)
    state = ss.get_tonight()
    assert state.test_mode is True
    assert state.attendees == ["Alice", "Bob"]
    raw = json.loads(Path(ss.SESSION_STATE_PATH).read_text())
    assert raw["test_mode"] is True


def test_clear_tonight_resets_test_mode(isolated_state):
    ss = isolated_state
    ss.start_tonight(["Alice"], test_mode=True)
    assert ss.get_tonight().test_mode is True
    ss.clear_tonight()
    assert ss.get_tonight().test_mode is False


def test_load_handles_missing_test_mode_field(isolated_state):
    ss = isolated_state
    # Pre-existing state files (written before this field existed) must
    # still load cleanly with test_mode defaulting to False.
    Path(ss.SESSION_STATE_PATH).write_text(
        json.dumps({"phase": "draft_ready", "attendees": ["X"]})
    )
    state = ss.get_tonight()
    assert state.test_mode is False
    assert state.phase == "draft_ready"


# ---------- pinned_doubles -------------------------------------------------


def _seed_attendees(ss, *names: str) -> None:
    ss.start_tonight(list(names))
    ss.set_courts_for_tonight(["5", "6", "9"])


def test_pinned_doubles_default_empty(isolated_state):
    ss = isolated_state
    assert ss.get_tonight().pinned_doubles == []


def test_add_pinned_doubles_happy_path(isolated_state):
    ss = isolated_state
    _seed_attendees(ss, "Alan", "Penny", "Peter", "Ben", "X", "Y", "Z", "W")
    state = ss.add_pinned_doubles(
        ["Alan", "Penny", "Peter", "Ben"],
        [["Alan", "Penny"], ["Peter", "Ben"]],
        rotation_num=2,
        court_label="5",
    )
    assert len(state.pinned_doubles) == 1
    pin = state.pinned_doubles[0]
    assert pin["players"] == ["Alan", "Penny", "Peter", "Ben"]
    assert pin["pairs"] == [["Alan", "Penny"], ["Peter", "Ben"]]
    assert pin["rotation_num"] == 2
    assert pin["court_label"] == "5"


def test_add_pinned_doubles_any_rotation(isolated_state):
    ss = isolated_state
    _seed_attendees(ss, "Alan", "Penny", "Peter", "Ben")
    state = ss.add_pinned_doubles(
        ["Alan", "Penny", "Peter", "Ben"],
        [["Alan", "Penny"], ["Peter", "Ben"]],
    )
    assert state.pinned_doubles[0]["rotation_num"] is None
    assert state.pinned_doubles[0]["court_label"] is None


def test_add_pinned_doubles_rejects_wrong_count(isolated_state):
    ss = isolated_state
    _seed_attendees(ss, "Alan", "Penny", "Peter", "Ben")
    with pytest.raises(ValueError, match="exactly 4"):
        ss.add_pinned_doubles(
            ["Alan", "Penny", "Peter"],
            [["Alan", "Penny"], ["Peter", "Ben"]],
        )


def test_add_pinned_doubles_rejects_pair_player_mismatch(isolated_state):
    ss = isolated_state
    _seed_attendees(ss, "Alan", "Penny", "Peter", "Ben", "Quinn")
    with pytest.raises(ValueError, match="partition"):
        ss.add_pinned_doubles(
            ["Alan", "Penny", "Peter", "Ben"],
            [["Alan", "Penny"], ["Quinn", "Ben"]],
        )


def test_add_pinned_doubles_rejects_non_attendee(isolated_state):
    ss = isolated_state
    _seed_attendees(ss, "Alan", "Penny", "Peter", "Ben")
    with pytest.raises(ValueError, match="attendees"):
        ss.add_pinned_doubles(
            ["Alan", "Penny", "Peter", "Stranger"],
            [["Alan", "Penny"], ["Peter", "Stranger"]],
        )


def test_add_pinned_doubles_rejects_overlap_same_rotation(isolated_state):
    ss = isolated_state
    _seed_attendees(ss, "Alan", "Penny", "Peter", "Ben", "X", "Y", "Z", "W")
    ss.add_pinned_doubles(
        ["Alan", "Penny", "Peter", "Ben"],
        [["Alan", "Penny"], ["Peter", "Ben"]],
        rotation_num=1,
    )
    with pytest.raises(ValueError, match="already pinned"):
        ss.add_pinned_doubles(
            ["Alan", "X", "Y", "Z"],
            [["Alan", "X"], ["Y", "Z"]],
            rotation_num=1,
        )


def test_add_pinned_doubles_allows_overlap_different_rotation(isolated_state):
    # A player can be in two pinned-doubles courts if they're in
    # different rotations.
    ss = isolated_state
    _seed_attendees(ss, "Alan", "Penny", "Peter", "Ben", "X", "Y", "Z", "W")
    ss.add_pinned_doubles(
        ["Alan", "Penny", "Peter", "Ben"],
        [["Alan", "Penny"], ["Peter", "Ben"]],
        rotation_num=1,
    )
    state = ss.add_pinned_doubles(
        ["Alan", "X", "Y", "Z"],
        [["Alan", "X"], ["Y", "Z"]],
        rotation_num=2,
    )
    assert len(state.pinned_doubles) == 2


def test_clear_pinned_doubles(isolated_state):
    ss = isolated_state
    _seed_attendees(ss, "Alan", "Penny", "Peter", "Ben")
    ss.add_pinned_doubles(
        ["Alan", "Penny", "Peter", "Ben"],
        [["Alan", "Penny"], ["Peter", "Ben"]],
        rotation_num=1,
    )
    state = ss.clear_pinned_doubles()
    assert state.pinned_doubles == []


def test_pinned_doubles_round_trip_through_disk(isolated_state):
    ss = isolated_state
    _seed_attendees(ss, "Alan", "Penny", "Peter", "Ben")
    ss.add_pinned_doubles(
        ["Alan", "Penny", "Peter", "Ben"],
        [["Alan", "Penny"], ["Peter", "Ben"]],
        rotation_num=2,
        court_label="5",
    )
    # Re-load from disk via a fresh _load() to confirm persistence.
    state = ss.get_tonight()
    assert state.pinned_doubles[0]["players"] == [
        "Alan", "Penny", "Peter", "Ben"
    ]
    assert state.pinned_doubles[0]["rotation_num"] == 2


def test_clear_tonight_wipes_pinned_doubles(isolated_state):
    ss = isolated_state
    _seed_attendees(ss, "Alan", "Penny", "Peter", "Ben")
    ss.add_pinned_doubles(
        ["Alan", "Penny", "Peter", "Ben"],
        [["Alan", "Penny"], ["Peter", "Ben"]],
        rotation_num=1,
    )
    ss.clear_tonight()
    assert ss.get_tonight().pinned_doubles == []
