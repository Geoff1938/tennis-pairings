"""Tests for the in-process Thursday-kickoff trigger logic in admin_bot."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest


@pytest.fixture
def isolated_kickoff_state(tmp_path, monkeypatch):
    """Point KICKOFF_STATE_PATH at a tmp file + clear env overrides."""
    import admin_bot

    monkeypatch.setattr(
        admin_bot, "KICKOFF_STATE_PATH", tmp_path / ".kickoff_state.json"
    )
    monkeypatch.delenv("BORIS_KICKOFF_TIME_OVERRIDE", raising=False)
    monkeypatch.delenv("BORIS_KICKOFF_ANY_DAY", raising=False)
    return admin_bot


# ---------- _kickoff_target_time ----------------------------------------


def test_default_kickoff_time(isolated_kickoff_state):
    ab = isolated_kickoff_state
    assert ab._kickoff_target_time() == (9, 35)


def test_kickoff_time_env_override(isolated_kickoff_state, monkeypatch):
    monkeypatch.setenv("BORIS_KICKOFF_TIME_OVERRIDE", "14:00")
    assert isolated_kickoff_state._kickoff_target_time() == (14, 0)


def test_kickoff_time_invalid_override_falls_back(isolated_kickoff_state, monkeypatch):
    monkeypatch.setenv("BORIS_KICKOFF_TIME_OVERRIDE", "garbage")
    assert isolated_kickoff_state._kickoff_target_time() == (9, 35)


# ---------- _should_fire_thursday_kickoff -------------------------------


def _thu(h, m):
    # Thu 7 May 2026 — weekday() == 3.
    return datetime(2026, 5, 7, h, m, 0)


def _wed(h, m):
    # Wed 6 May 2026 — weekday() == 2.
    return datetime(2026, 5, 6, h, m, 0)


def test_does_not_fire_on_non_thursday(isolated_kickoff_state):
    assert isolated_kickoff_state._should_fire_thursday_kickoff(_wed(9, 35)) is False


def test_does_not_fire_before_target_time(isolated_kickoff_state):
    assert isolated_kickoff_state._should_fire_thursday_kickoff(_thu(9, 34)) is False


def test_fires_at_or_after_target_time(isolated_kickoff_state):
    ab = isolated_kickoff_state
    assert ab._should_fire_thursday_kickoff(_thu(9, 35)) is True
    assert ab._should_fire_thursday_kickoff(_thu(11, 0)) is True


def test_does_not_refire_after_recording_attempt(isolated_kickoff_state):
    ab = isolated_kickoff_state
    now = _thu(9, 36)
    assert ab._should_fire_thursday_kickoff(now) is True
    ab._record_kickoff_attempt(now.date().isoformat())
    assert ab._should_fire_thursday_kickoff(now) is False
    # And again at a later time the same day.
    assert ab._should_fire_thursday_kickoff(_thu(15, 0)) is False


def test_can_fire_again_on_a_different_thursday(isolated_kickoff_state):
    ab = isolated_kickoff_state
    ab._record_kickoff_attempt(_thu(9, 35).date().isoformat())
    next_thu = datetime(2026, 5, 14, 9, 35)  # Thu 14 May 2026
    assert ab._should_fire_thursday_kickoff(next_thu) is True


def test_any_day_env_override(isolated_kickoff_state, monkeypatch):
    monkeypatch.setenv("BORIS_KICKOFF_ANY_DAY", "1")
    assert isolated_kickoff_state._should_fire_thursday_kickoff(_wed(9, 35)) is True


# ---------- persistence -------------------------------------------------


def test_record_and_read_attempt_round_trip(isolated_kickoff_state):
    ab = isolated_kickoff_state
    assert ab._last_kickoff_attempt_date() == ""
    ab._record_kickoff_attempt("2026-05-07")
    assert ab._last_kickoff_attempt_date() == "2026-05-07"


def test_corrupt_state_file_treated_as_empty(isolated_kickoff_state):
    ab = isolated_kickoff_state
    Path(ab.KICKOFF_STATE_PATH).write_text("not json", encoding="utf-8")
    assert ab._last_kickoff_attempt_date() == ""
