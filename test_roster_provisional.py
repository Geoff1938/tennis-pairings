"""Tests for the provisional-rating field on the roster.

Each test builds a Roster on its own temporary players.json and
checks both the in-memory view and what was saved to disk."""

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


# ---------- load + read -------------------------------------------------


def test_load_picks_up_provisional_Y():
    r = _roster_with([
        {"Name": "Alice", "Gender": "F", "Rating": 5,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": "Y"},
        {"Name": "Bob", "Gender": "M", "Rating": 3,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": ""},
    ])
    assert r.get("Alice")["provisional"] is True
    assert r.get("Bob")["provisional"] is False


def test_load_tolerates_missing_provisional_column():
    """Old sheet (no column G) → everyone is non-provisional."""
    r = _roster_with([
        {"Name": "Alice", "Gender": "F", "Rating": 5,
         "Notes": "", "Phone": "", "Singles": ""},
    ])
    assert r.get("Alice")["provisional"] is False


def test_load_treats_lowercase_y_as_truthy():
    r = _roster_with([
        {"Name": "Alice", "Gender": "F", "Rating": 5,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": "y"},
    ])
    assert r.get("Alice")["provisional"] is True


# ---------- add ---------------------------------------------------------


def test_add_with_provisional_saves_flag():
    r = _roster_with([])
    r.add("Carol", gender="F", rating=4, provisional=True)
    # Last call should be the append_row.
    assert _stored(r)["Carol"]["provisional"] is True
    assert r.get("Carol")["provisional"] is True


def test_add_without_provisional_saves_false():
    r = _roster_with([])
    r.add("Dan", gender="M", rating=5)
    assert _stored(r)["Dan"]["provisional"] is False
    assert r.get("Dan")["provisional"] is False


# ---------- set_rating clears provisional --------------------------------


def test_set_rating_on_provisional_player_clears_the_flag():
    """Admin's explicit rating call IS the team's confirmation of the
    rating — drop the (P) marker. Also writes the cleared flag back
    to disk so it sticks."""
    r = _roster_with([
        {"Name": "Eve", "Gender": "F", "Rating": 6,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": "Y"},
    ])
    assert r.get("Eve")["provisional"] is True
    r.set_rating("Eve", 7)
    assert r.get("Eve")["rating"] == 7
    assert r.get("Eve")["provisional"] is False
    assert _stored(r)["Eve"]["rating"] == 7
    assert _stored(r)["Eve"]["provisional"] is False


def test_set_rating_on_non_provisional_player_keeps_flag_off():
    r = _roster_with([
        {"Name": "Frank", "Gender": "M", "Rating": 5,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": ""},
    ])
    r.set_rating("Frank", 6)
    assert _stored(r)["Frank"]["rating"] == 6
    assert _stored(r)["Frank"]["provisional"] is False


def test_set_rating_on_unknown_player_raises():
    r = _roster_with([])
    with pytest.raises(KeyError):
        r.set_rating("Ghost", 5)


# ---------- clear_provisional -------------------------------------------


def test_clear_provisional_changes_only_the_flag():
    r = _roster_with([
        {"Name": "Hank", "Gender": "M", "Rating": 5,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": "Y"},
    ])
    r.clear_provisional("Hank")
    assert r.get("Hank")["provisional"] is False
    assert _stored(r)["Hank"]["provisional"] is False
    assert _stored(r)["Hank"]["rating"] == 5


def test_clear_provisional_is_idempotent_when_not_provisional():
    """No write when the flag wasn't set."""
    r = _roster_with([
        {"Name": "Ivy", "Gender": "F", "Rating": 5,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": ""},
    ])
    before = _raw(r)
    r.clear_provisional("Ivy")
    assert r.get("Ivy")["provisional"] is False
    assert _raw(r) == before


def test_clear_provisional_unknown_name_raises():
    r = _roster_with([])
    with pytest.raises(KeyError):
        r.clear_provisional("Ghost")


# ---------- whitespace normalisation on add -----------------------------


def test_add_collapses_internal_double_space():
    """A double-space in the input (typical of pasted/imported data)
    is collapsed before storage so we don't end up with two near-
    duplicate rows."""
    r = _roster_with([])
    r.add("Jack  Fenner", gender="M", rating=4)
    # Stored under the canonical single-space spelling.
    assert r.get("Jack Fenner") is not None
    assert r.get("Jack  Fenner") is None  # the bad spelling isn't a key
    # And the clean name is what was saved.
    assert list(_stored(r)) == ["Jack Fenner"]


def test_add_treats_whitespace_variant_as_duplicate():
    """If 'Jack Fenner' already exists, adding 'Jack  Fenner' is a
    no-op (same person) rather than a second row."""
    r = _roster_with([
        {"Name": "Jack Fenner", "Gender": "M", "Rating": 4,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": ""},
    ])
    before = _raw(r)
    r.add("Jack  Fenner", gender="M", rating=5)
    assert _raw(r) == before  # nothing written
    # Original entry unchanged.
    assert r.get("Jack Fenner")["rating"] == 4


def test_add_strips_leading_and_trailing_whitespace():
    r = _roster_with([])
    r.add("  Ada Lovelace ", gender="F", rating=5)
    assert r.get("Ada Lovelace") is not None
    assert r.get("  Ada Lovelace ") is None


def test_add_empty_name_after_normalisation_raises():
    r = _roster_with([])
    with pytest.raises(ValueError):
        r.add("   ", gender="F", rating=5)


def test_add_many_from_cr_normalises_and_dedupes():
    """CR sometimes delivers slight whitespace inconsistencies; add_many
    should treat them as one player. Also dedupes within the same call."""
    r = _roster_with([
        {"Name": "Jack Fenner", "Gender": "M", "Rating": 4,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": ""},
    ])
    added = r.add_many_from_cr([
        "Jack  Fenner",     # existing, with extra space — skip
        "  Ada Lovelace ",  # new, with trim whitespace
        "Ada Lovelace",     # same as above after normalising — skip on 2nd
    ])
    added_names = [e["name"] for e in added]
    assert added_names == ["Ada Lovelace"]


# ---------- rename ------------------------------------------------------


def test_rename_updates_key_on_disk_and_cache():
    r = _roster_with([
        {"Name": "Jack  Fenner", "Gender": "M", "Rating": 5,
         "Notes": "", "Phone": "+447", "Singles": "", "Provisional": ""},
    ])
    r.rename("Jack  Fenner", "Jack Fenner")
    assert r.get("Jack Fenner") is not None
    assert r.get("Jack  Fenner") is None
    # The other field values survive the rename.
    assert r.get("Jack Fenner")["phone"] == "+447"
    # Saved under the new name only.
    assert list(_stored(r)) == ["Jack Fenner"]


def test_rename_normalises_target():
    """Common case: admin asks to rename 'James  Elliott' to
    'James Elliott'. Also works if they pass extra padding."""
    r = _roster_with([
        {"Name": "James  Elliott", "Gender": "M", "Rating": 6,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": ""},
    ])
    r.rename("James  Elliott", "  James Elliott  ")
    assert r.get("James Elliott") is not None


def test_rename_noop_when_normalised_target_equals_old():
    """Admin says 'rename X to X' (or X to a value that normalises
    back to X) — just return the existing entry without writing."""
    r = _roster_with([
        {"Name": "Geoff Chapman", "Gender": "M", "Rating": 4,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": ""},
    ])
    before = _raw(r)
    r.rename("Geoff Chapman", "  Geoff Chapman  ")
    assert _raw(r) == before  # nothing written


def test_rename_rejects_collision_with_existing():
    """Rename to a name that's already in the roster — would silently
    merge fields and is dangerous. Refuse; tell the caller to use
    merge_and_delete instead."""
    r = _roster_with([
        {"Name": "Jack  Fenner", "Gender": "M", "Rating": 5,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": ""},
        {"Name": "Jack Fenner", "Gender": "M", "Rating": 4,
         "Notes": "", "Phone": "+447", "Singles": "", "Provisional": ""},
    ])
    with pytest.raises(ValueError, match="already in the roster"):
        r.rename("Jack  Fenner", "Jack Fenner")


def test_rename_unknown_old_name_raises():
    r = _roster_with([])
    with pytest.raises(KeyError):
        r.rename("Ghost", "Other Ghost")


def test_rename_to_empty_raises():
    r = _roster_with([
        {"Name": "X", "Gender": "?", "Rating": 5,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": ""},
    ])
    with pytest.raises(ValueError, match="non-empty"):
        r.rename("X", "   ")
