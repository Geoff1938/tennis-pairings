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
