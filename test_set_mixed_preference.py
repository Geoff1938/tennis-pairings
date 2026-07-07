"""Tests for tool_set_mixed_preference — the WhatsApp mixed opt-in.

Mocks admin_bot.Roster so we exercise the fuzzy-match / validation /
dispatch logic without touching Google Sheets.
"""
from __future__ import annotations

import pytest


class _FakeRoster:
    def __init__(self, data: dict[str, dict]):
        self._data = {k: dict(v) for k, v in data.items()}
        self.sets: list[tuple] = []

    def get(self, name):
        entry = self._data.get(name)
        return dict(entry) if entry is not None else None

    def find_by_fuzzy(self, query):
        q = query.strip().lower()
        return [n for n in self._data if q in n.lower()]

    def set_mixed(self, name, value):
        if name not in self._data:
            raise KeyError(name)
        self._data[name]["mixed"] = value
        self.sets.append((name, value))
        return dict(self._data[name])


@pytest.fixture
def patch_roster(monkeypatch):
    import admin_bot

    def _install(data):
        fake = _FakeRoster(data)
        monkeypatch.setattr(admin_bot, "Roster", lambda: fake)
        return fake

    return _install


def _roster():
    return {
        "Luke O'Mahoney": {"gender": "M", "rating": 1, "mixed": ""},
        "Luke Skywalker": {"gender": "M", "rating": 4, "mixed": ""},
        "Priya Patel": {"gender": "F", "rating": 5, "mixed": ""},
    }


def test_opt_in_prefer(patch_roster):
    import admin_bot
    fake = patch_roster({"Priya Patel": {"gender": "F", "rating": 5,
                                         "mixed": ""}})
    res = admin_bot.tool_set_mixed_preference("Priya Patel", "prefer")
    assert res["ok"] is True
    assert res["entry"]["mixed"] == "prefer"
    assert fake.sets == [("Priya Patel", "prefer")]


def test_neutral_clears(patch_roster):
    import admin_bot
    fake = patch_roster({"Priya Patel": {"gender": "F", "rating": 5,
                                         "mixed": "prefer"}})
    res = admin_bot.tool_set_mixed_preference("Priya Patel", "neutral")
    assert res["ok"] is True
    assert res["entry"]["mixed"] == ""
    assert fake.sets == [("Priya Patel", "")]


def test_invalid_preference_rejected(patch_roster):
    import admin_bot
    patch_roster({"Priya Patel": {"gender": "F", "rating": 5, "mixed": ""}})
    res = admin_bot.tool_set_mixed_preference("Priya Patel", "avoid")
    assert res["ok"] is False
    assert res["error"] == "invalid_preference"


def test_fuzzy_partial_name(patch_roster):
    import admin_bot
    fake = patch_roster({"Priya Patel": {"gender": "F", "rating": 5,
                                         "mixed": ""}})
    res = admin_bot.tool_set_mixed_preference("priya", "prefer")
    assert res["ok"] is True
    assert res["name"] == "Priya Patel"
    assert fake.sets == [("Priya Patel", "prefer")]


def test_ambiguous_returns_candidates(patch_roster):
    import admin_bot
    patch_roster(_roster())
    res = admin_bot.tool_set_mixed_preference("luke", "prefer")
    assert res["ok"] is False
    assert res["error"] == "ambiguous"
    assert set(res["candidates"]) == {"Luke O'Mahoney", "Luke Skywalker"}


def test_not_found(patch_roster):
    import admin_bot
    patch_roster(_roster())
    res = admin_bot.tool_set_mixed_preference("Nonexistent", "prefer")
    assert res["ok"] is False
    assert res["error"] == "not_found"


def test_tool_registered_in_impls_and_schemas():
    import admin_bot
    assert "set_mixed_preference" in admin_bot.TOOL_IMPLS
    names = {t["name"] for t in admin_bot.TOOL_SCHEMAS}
    assert "set_mixed_preference" in names
