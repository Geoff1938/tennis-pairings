"""Tests for the mixed-doubles opt-in column on the roster.

Same temp-file approach as test_roster_provisional.py.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from roster import Roster


def _roster_with(rows: list[dict]) -> Roster:
    """Build a Roster on a temp players.json seeded from sheet-style
    rows ({"Name": ..., "Gender": ..., ...})."""
    path = Path(tempfile.mkdtemp()) / "players.json"
    data = {
        r["Name"]: {k.lower(): v for k, v in r.items() if k != "Name"}
        for r in rows
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return Roster(path)


def _stored(r: Roster) -> dict:
    """What is actually on disk."""
    return json.loads(r.path.read_text(encoding="utf-8"))


def _raw(r: Roster) -> str:
    return r.path.read_text(encoding="utf-8")


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


def test_set_mixed_saves_and_updates_cache():
    r = _roster_with([
        {"Name": "Luke", "Gender": "M", "Rating": 1, "Notes": "",
         "Phone": "", "Singles": "", "Provisional": "", "Mixed": ""},
    ])
    out = r.set_mixed("Luke", "prefer")
    assert out["mixed"] == "prefer"
    assert r.get("Luke")["mixed"] == "prefer"
    assert _stored(r)["Luke"]["mixed"] == "prefer"


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


def test_add_player_saves_mixed():
    r = _roster_with([])
    r.add("New Person", gender="F", rating=5, mixed="prefer")
    assert _stored(r)["New Person"]["mixed"] == "prefer"
    assert r.get("New Person")["mixed"] == "prefer"
