"""Tests for the mixed-doubles opt-in column on the roster.

Same fake-worksheet approach as test_roster_provisional.py — no network.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from roster import COL_MIXED, Roster


class _FakeWS:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.calls: list[tuple] = []
        self._row_for_name = {
            r.get("Name"): i + 2 for i, r in enumerate(rows)
        }

    def get_all_records(self, **_kwargs):
        return list(self.rows)

    def find(self, value, in_column=None):
        row = self._row_for_name.get(value)
        return SimpleNamespace(row=row) if row else None

    def update_cell(self, row, col, value):
        self.calls.append(("update_cell", row, col, value))

    def append_row(self, row, value_input_option=None):
        self.calls.append(("append_row", list(row)))


def _roster_with(rows: list[dict]) -> Roster:
    r = Roster.__new__(Roster)
    r._ws = _FakeWS(rows)
    r._data = {}
    r.load()
    return r


def test_load_picks_up_mixed_prefer():
    r = _roster_with([
        {"Name": "Luke", "Gender": "M", "Rating": 1, "Notes": "",
         "Phone": "", "Singles": "", "Provisional": "", "Mixed": "prefer"},
        {"Name": "Bob", "Gender": "M", "Rating": 3, "Notes": "",
         "Phone": "", "Singles": "", "Provisional": "", "Mixed": ""},
    ])
    assert r.get("Luke")["mixed"] == "prefer"
    assert r.get("Bob")["mixed"] == ""


def test_load_missing_mixed_column_defaults_blank():
    # Older sheet with no Mixed header at all.
    r = _roster_with([
        {"Name": "Ada", "Gender": "F", "Rating": 5, "Notes": "",
         "Phone": "", "Singles": "", "Provisional": ""},
    ])
    assert r.get("Ada")["mixed"] == ""


def test_load_invalid_mixed_value_coerced_blank():
    r = _roster_with([
        {"Name": "Sam", "Gender": "M", "Rating": 4, "Notes": "",
         "Phone": "", "Singles": "", "Provisional": "", "Mixed": "yes"},
    ])
    assert r.get("Sam")["mixed"] == ""


def test_set_mixed_writes_column_and_cache():
    r = _roster_with([
        {"Name": "Luke", "Gender": "M", "Rating": 1, "Notes": "",
         "Phone": "", "Singles": "", "Provisional": "", "Mixed": ""},
    ])
    out = r.set_mixed("Luke", "prefer")
    assert out["mixed"] == "prefer"
    assert r.get("Luke")["mixed"] == "prefer"
    writes = [c for c in r._ws.calls if c[0] == "update_cell"
              and c[2] == COL_MIXED]
    assert writes == [("update_cell", 2, COL_MIXED, "prefer")]


def test_set_mixed_rejects_bad_value():
    r = _roster_with([
        {"Name": "Luke", "Gender": "M", "Rating": 1, "Notes": "",
         "Phone": "", "Singles": "", "Provisional": "", "Mixed": ""},
    ])
    with pytest.raises(ValueError):
        r.set_mixed("Luke", "sometimes")


def test_set_mixed_unknown_player():
    r = _roster_with([])
    with pytest.raises(KeyError):
        r.set_mixed("Nobody", "prefer")


def test_add_player_writes_mixed_cell():
    r = _roster_with([])
    r.add("New Person", gender="F", rating=5, mixed="prefer")
    appended = [c for c in r._ws.calls if c[0] == "append_row"][0][1]
    assert appended[COL_MIXED - 1] == "prefer"
    assert r.get("New Person")["mixed"] == "prefer"
