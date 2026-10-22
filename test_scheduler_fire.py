"""Tests for bot_scheduler's scheduled-booking firing: no double bookings
on teardown errors / restarts, no pointless retries, honest replies."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

import bot_scheduler
import bot_tools_booking


@pytest.fixture
def env(tmp_path, monkeypatch):
    import accounts
    import scheduled_bookings as sb

    monkeypatch.setattr(sb, "SCHEDULED_PATH", tmp_path / "sched.json")
    monkeypatch.delenv("BORIS_BOOKING_OVERRIDE_OPENS_AT", raising=False)
    acct = SimpleNamespace(
        display_name="Geoff", court_preference_list=lambda: ["6"],
    )
    monkeypatch.setattr(accounts, "account_for_key", lambda k: acct)
    sent: list[str] = []
    monkeypatch.setattr(bot_scheduler, "send_to_group",
                        lambda jid, text: sent.append(text))
    calls = {"prepare": 0, "submit": 0}
    behaviour: dict = {}

    class Prepared:
        def submit(self):
            calls["submit"] += 1
            if behaviour.get("submit_raises"):
                raise RuntimeError("page crashed after click")
            return {"ok": True, "status": "booked", "court_label": "6",
                    "verified": behaviour.get("verified", True)}

    class FakeCR:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            if behaviour.get("teardown_raises"):
                raise RuntimeError("browser close failed")

        def prepare_court_booking(self, **kw):
            calls["prepare"] += 1
            if "prep_result" in behaviour:
                return behaviour["prep_result"]
            return Prepared()

    monkeypatch.setattr(accounts, "cr_client", lambda a: FakeCR())
    entry = sb.add_pending(
        scheduled_by_phone="447", scheduled_by_account_key="geoff",
        channel_jid="g@g.us", play_date="2026-05-14",
        start_time_hhmm="08:00", duration_minutes=90,
        partner_name="Louise Clark",
    )
    return sb, entry, calls, behaviour, sent


def _fire(sb, entry_id):
    entry = next(e for e in sb.list_pending() if e.id == entry_id)
    bot_scheduler._fire_scheduled_booking(entry)


def test_teardown_error_after_booking_counts_as_booked(env):
    # A raising __exit__ on the client (the real one now swallows
    # teardown errors, but the scheduler mustn't depend on that).
    sb, entry, calls, behaviour, sent = env
    behaviour["teardown_raises"] = True
    _fire(sb, entry.id)
    assert calls["submit"] == 1
    assert sb.list_pending() == []
    assert sb.list_history()[-1].state == sb.STATE_SUCCEEDED
    assert sent and "Court booked" in sent[0]


def test_error_after_submit_is_uncertain_and_not_retried(env):
    sb, entry, calls, behaviour, sent = env
    behaviour["submit_raises"] = True
    _fire(sb, entry.id)
    assert sb.list_pending() == []
    last = sb.list_history()[-1]
    assert last.state == sb.STATE_FAILED and last.last_error == "uncertain"
    assert "may or may not have booked" in sent[0]


def test_partner_not_found_fails_once_without_retries(env):
    sb, entry, calls, behaviour, sent = env
    behaviour["prep_result"] = {"ok": False, "status": "partner_not_found"}
    _fire(sb, entry.id)
    assert sb.list_pending() == []
    assert sb.list_history()[-1].fire_attempts == 1
    assert "partner_not_found" in sent[0]


def test_transient_failure_still_retries(env):
    sb, entry, calls, behaviour, sent = env
    behaviour["prep_result"] = {"ok": False, "status": "no_court_available"}
    _fire(sb, entry.id)
    pending = sb.list_pending()
    assert len(pending) == 1 and pending[0].in_flight_since is None
    assert sent == []  # silent until the final attempt


def test_unverified_booking_asks_admin_to_check(env):
    sb, entry, calls, behaviour, sent = env
    behaviour["verified"] = False
    _fire(sb, entry.id)
    assert "please check CourtReserve" in sent[0]


def test_restart_mid_fire_is_reported_not_refired(env):
    sb, entry, calls, behaviour, sent = env
    sb.mark_in_flight(entry.id)  # previous process died here
    bot_scheduler._maybe_fire_scheduled_bookings(
        datetime(2026, 5, 7, 8, 5, tzinfo=sb.LOCAL_TZ)
    )
    assert calls["prepare"] == 0
    assert sb.list_pending() == []
    assert "interrupted by a restart" in sent[0]


def test_cr_client_exit_swallows_teardown_errors():
    from courtreserve import CourtReserveClient

    cr = object.__new__(CourtReserveClient)
    stopped = []

    def bad_close():
        raise RuntimeError("boom")

    cr._ctx = SimpleNamespace(close=bad_close)
    cr._pw = SimpleNamespace(stop=lambda: stopped.append(True))
    cr.__exit__(None, None, None)  # must not raise
    assert stopped == [True]       # and still stops playwright


# ---------- cancel verification ----------------------------------------


@pytest.fixture
def cancel_env(monkeypatch):
    import accounts

    state = {"present": True, "cancelled": []}

    class FakeCR:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def find_my_court_booking(self, date, hhmm):
            if state["present"]:
                return {"reservation_id": "123", "court_label": "Court #6"}
            return None

        def cancel_court_reservation(self, rid):
            state["cancelled"].append(rid)
            if state.get("cancel_works"):
                state["present"] = False
            return {"ok": True, "status": "cancelled", "reservation_id": rid}

    monkeypatch.setattr(accounts, "cr_client", lambda a: FakeCR())
    monkeypatch.setattr(bot_tools_booking, "_resolve_booking_account",
                        lambda b: (SimpleNamespace(), None))
    return state


def test_cancel_confirmed_when_booking_gone(cancel_env):
    state = cancel_env
    state["cancel_works"] = True
    res = bot_tools_booking.tool_cancel_court_booking(
        date="2026-05-14", start_time_hhmm="08:00")
    assert res["ok"] is True and res["verified"] is True
    assert state["cancelled"] == ["123"]


def test_cancel_reported_when_booking_still_there(cancel_env):
    state = cancel_env
    res = bot_tools_booking.tool_cancel_court_booking(
        date="2026-05-14", start_time_hhmm="08:00")
    assert res["ok"] is False
    assert res["status"] == "cancel_not_confirmed"


def test_cancel_rejects_non_numeric_reservation_id():
    from courtreserve import CourtReserveClient

    cr = object.__new__(CourtReserveClient)
    cr._page = object()
    res = cr.cancel_court_reservation("12; DROP")
    assert res == {"ok": False, "status": "invalid_reservation_id",
                   "reservation_id": "12; DROP"}
