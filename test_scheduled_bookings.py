"""Tests for scheduled_bookings.py — persistence, timing, lifecycle."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest


@pytest.fixture
def sb(tmp_path, monkeypatch):
    import scheduled_bookings as s

    target = tmp_path / "scheduled_bookings.json"
    monkeypatch.setattr(s, "SCHEDULED_PATH", target)
    monkeypatch.delenv("BORIS_BOOKING_OVERRIDE_OPENS_AT", raising=False)
    return s


def _make(s, **overrides) -> "scheduled_bookings.ScheduledBooking":
    base = dict(
        scheduled_by_phone="447000",
        scheduled_by_account_key="geoff",
        channel_jid="abc@g.us",
        play_date="2026-05-14",
        start_time_hhmm="0800",
        duration_minutes=90,
        partner_name="Louise Clark",
        court_label="6",
        court_type=None,
    )
    base.update(overrides)
    return s.add_pending(**base)


def test_window_opens_at_six_days_before_at_eight(sb):
    s = sb
    opens = s.compute_window_opens_at("2026-05-14")
    assert opens.year == 2026
    assert opens.month == 5
    assert opens.day == 8         # 14 minus 6
    assert opens.hour == 8
    assert opens.minute == 0
    assert opens.tzinfo is not None


def test_add_pending_assigns_ids_and_persists(sb):
    s = sb
    a = _make(s)
    b = _make(s, partner_name="Maggie Cochrane")
    assert a.id == 1
    assert b.id == 2
    raw = json.loads(Path(s.SCHEDULED_PATH).read_text())
    assert raw["next_id"] == 3
    assert len(raw["pending"]) == 2
    assert raw["pending"][1]["partner_name"] == "Maggie Cochrane"


def test_list_pending_filters_by_account(sb):
    s = sb
    _make(s, scheduled_by_account_key="geoff")
    _make(s, scheduled_by_account_key="shirley")
    geoff = s.list_pending(account_key="geoff")
    shirley = s.list_pending(account_key="shirley")
    everyone = s.list_pending()
    assert [b.scheduled_by_account_key for b in geoff] == ["geoff"]
    assert [b.scheduled_by_account_key for b in shirley] == ["shirley"]
    assert len(everyone) == 2


def test_cancel_moves_to_history(sb):
    s = sb
    a = _make(s)
    cancelled, entry = s.cancel_pending(a.id, by_account_key="geoff")
    assert cancelled is True
    assert entry.state == s.STATE_CANCELLED
    assert s.list_pending() == []
    history = s.list_history()
    assert [b.id for b in history] == [a.id]


def test_cancel_enforces_ownership(sb):
    s = sb
    a = _make(s, scheduled_by_account_key="geoff")
    cancelled, _ = s.cancel_pending(a.id, by_account_key="shirley")
    assert cancelled is False
    assert len(s.list_pending()) == 1


def test_due_now_returns_only_open_windows(sb):
    s = sb
    # Window for 2026-05-14 opens 2026-05-08 08:00 BST.
    a = _make(s, play_date="2026-05-14")
    before_open = datetime(2026, 5, 8, 7, 59, 59, tzinfo=s.LOCAL_TZ)
    after_open = datetime(2026, 5, 8, 8, 0, 1, tzinfo=s.LOCAL_TZ)
    assert s.due_now(now=before_open) == []
    due = s.due_now(now=after_open)
    assert [b.id for b in due] == [a.id]


def test_due_now_paces_retries(sb):
    s = sb
    a = _make(s, play_date="2026-05-14")
    # Mark a recent attempt — should not be considered due immediately.
    fired_at = datetime(2026, 5, 8, 8, 0, 5, tzinfo=s.LOCAL_TZ)
    s.mark_attempt(a.id, succeeded=False, error="too_early", now=fired_at)
    just_after = datetime(2026, 5, 8, 8, 0, 8, tzinfo=s.LOCAL_TZ)  # 3s later
    later = datetime(2026, 5, 8, 8, 0, 20, tzinfo=s.LOCAL_TZ)      # 15s later
    assert s.due_now(now=just_after) == []
    assert [b.id for b in s.due_now(now=later)] == [a.id]


def test_mark_attempt_promotes_to_history_on_success(sb):
    s = sb
    a = _make(s)
    s.mark_attempt(a.id, succeeded=True, result={"reservation_id": "R1"})
    assert s.list_pending() == []
    history = s.list_history()
    assert len(history) == 1
    assert history[0].state == s.STATE_SUCCEEDED
    assert history[0].result == {"reservation_id": "R1"}


def test_mark_attempt_fails_after_max_attempts(sb):
    s = sb
    a = _make(s)
    for i in range(s.MAX_FIRE_ATTEMPTS - 1):
        s.mark_attempt(a.id, succeeded=False, error="too_early")
        assert s.list_pending()[0].state == s.STATE_SCHEDULED
    s.mark_attempt(a.id, succeeded=False, error="taken")
    assert s.list_pending() == []
    history = s.list_history()
    assert history[0].state == s.STATE_FAILED
    assert history[0].fire_attempts == s.MAX_FIRE_ATTEMPTS
    assert history[0].last_error == "taken"


def test_override_env_forces_due_now(sb, monkeypatch):
    s = sb
    a = _make(s, play_date="2026-12-31")    # window opens 2026-12-25
    well_before = datetime(2026, 5, 6, 12, 0, 0, tzinfo=s.LOCAL_TZ)
    assert s.due_now(now=well_before) == []
    monkeypatch.setenv(
        "BORIS_BOOKING_OVERRIDE_OPENS_AT", "2027-01-01T00:00:00+00:00"
    )
    due = s.due_now(now=well_before)
    assert [b.id for b in due] == [a.id]


def test_round_trip_preserves_all_fields(sb):
    s = sb
    a = _make(s, court_label="9", court_type="clay", notes="test")
    s.mark_attempt(a.id, succeeded=False, error="too_early")
    # Re-load from disk via fresh _load — checks every field round-trips.
    pending = s.list_pending()
    assert len(pending) == 1
    e = pending[0]
    assert e.court_label == "9"
    assert e.court_type == "clay"
    assert e.notes == "test"
    assert e.fire_attempts == 1
    assert e.last_error == "too_early"
