"""Tests for tool_confirm_provisional_ratings — the bulk-clear path.

Mocks Roster + session_state so we exercise the dispatching /
classification logic without touching Google Sheets or the live
session file."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _FakeRoster:
    """In-memory Roster stand-in. ``data`` maps name -> entry."""

    def __init__(self, data: dict[str, dict]):
        self._data = {k: dict(v) for k, v in data.items()}
        self.cleared: list[str] = []
        self.errors_for: set[str] = set()  # raise from clear_provisional

    def get(self, name):
        entry = self._data.get(name)
        return dict(entry) if entry is not None else None

    def clear_provisional(self, name):
        if name in self.errors_for:
            raise RuntimeError("quota")
        if name not in self._data:
            raise KeyError(name)
        self._data[name]["provisional"] = False
        self.cleared.append(name)
        return dict(self._data[name])


@pytest.fixture
def bot(monkeypatch):
    """Patch admin_bot.Roster + session_state.get_tonight so the tool
    can run without network or session files."""
    import admin_bot
    import session_state

    state = SimpleNamespace(attendees=[], phase="awaiting_extras")
    fake_roster = _FakeRoster({})

    def _set_roster(d):
        nonlocal fake_roster
        fake_roster = _FakeRoster(d)
        monkeypatch.setattr(admin_bot, "Roster", lambda: fake_roster)
        return fake_roster

    def _set_attendees(names):
        state.attendees = list(names)

    monkeypatch.setattr(admin_bot, "Roster", lambda: fake_roster)
    monkeypatch.setattr(session_state, "get_tonight", lambda: state)
    return SimpleNamespace(
        admin_bot=admin_bot,
        state=state,
        set_roster=_set_roster,
        set_attendees=_set_attendees,
        roster=lambda: fake_roster,
    )


# ---------- default-scope (session attendees) ---------------------------


def test_default_scope_clears_provisional_for_session_attendees(bot):
    bot.set_roster({
        "Alice": {"provisional": True, "rating": 5},
        "Bob": {"provisional": True, "rating": 6},
        "Carol": {"provisional": False, "rating": 4},
        "Dan": {"provisional": True, "rating": 3},  # not on tonight's list
    })
    bot.set_attendees(["Alice", "Bob", "Carol"])

    res = bot.admin_bot.tool_confirm_provisional_ratings()
    assert res["ok"] is True
    assert res["scope_source"] == "session"
    assert sorted(res["cleared"]) == ["Alice", "Bob"]
    assert res["already_confirmed"] == ["Carol"]
    assert res["not_found"] == []
    assert res["cleared_count"] == 2
    # Dan was NOT in tonight's list, so left untouched.
    assert bot.roster()._data["Dan"]["provisional"] is True


def test_no_attendees_returns_error(bot):
    bot.set_attendees([])
    res = bot.admin_bot.tool_confirm_provisional_ratings()
    assert res["ok"] is False
    assert res["error"] == "no_attendees"


def test_unknown_name_reported_not_failed(bot):
    bot.set_roster({"Alice": {"provisional": True, "rating": 5}})
    bot.set_attendees(["Alice", "Mystery"])
    res = bot.admin_bot.tool_confirm_provisional_ratings()
    assert res["ok"] is True
    assert res["cleared"] == ["Alice"]
    assert res["not_found"] == ["Mystery"]
    assert res["failed"] == []


# ---------- explicit-scope ---------------------------------------------


def test_explicit_attendees_overrides_session(bot):
    bot.set_roster({
        "Alice": {"provisional": True, "rating": 5},
        "Bob": {"provisional": True, "rating": 6},
    })
    bot.set_attendees(["Alice"])  # session-scope would only pick Alice
    res = bot.admin_bot.tool_confirm_provisional_ratings(
        attendees=["Alice", "Bob"],
    )
    assert res["ok"] is True
    assert res["scope_source"] == "explicit"
    assert sorted(res["cleared"]) == ["Alice", "Bob"]


def test_clear_failure_captured_in_failed_list(bot):
    """A per-name network blip from clear_provisional shouldn't abort
    the whole batch — the rest still go through, and the failure is
    reported."""
    r = bot.set_roster({
        "Alice": {"provisional": True, "rating": 5},
        "Bob": {"provisional": True, "rating": 6},
    })
    r.errors_for = {"Bob"}
    bot.set_attendees(["Alice", "Bob"])
    res = bot.admin_bot.tool_confirm_provisional_ratings()
    assert res["cleared"] == ["Alice"]
    assert len(res["failed"]) == 1
    assert res["failed"][0]["name"] == "Bob"
