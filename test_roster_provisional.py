"""Tests for the provisional-rating field on the roster.

Roster talks to a Google Sheet via gspread; these tests bypass the
network by constructing a Roster without ``__init__`` and attaching
a fake worksheet that records ``update_cell`` / ``append_row`` /
``find`` calls. That's enough to exercise the bits of
``load``/``add``/``set_rating`` that handle the new column."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from roster import (
    COL_PROVISIONAL,
    COL_RATING,
    PROVISIONAL_FALSE,
    PROVISIONAL_TRUE,
    Roster,
)


class _FakeWS:
    """Minimal gspread Worksheet stand-in. ``rows`` are the dicts the
    real ``get_all_records`` returns; ``calls`` records writes for
    assertions."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.calls: list[tuple] = []
        # Map name → 1-based row index (header = row 1).
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
    """Build a Roster without going to the network; seed its cache by
    running ``load`` against the fake worksheet."""
    r = Roster.__new__(Roster)
    r._ws = _FakeWS(rows)
    r._data = {}
    r.load()
    return r


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


def test_add_with_provisional_writes_Y_in_column_G():
    r = _roster_with([])
    r.add("Carol", gender="F", rating=4, provisional=True)
    # Last call should be the append_row.
    last = r._ws.calls[-1]
    assert last[0] == "append_row"
    # Columns are: Name, Gender, Rating, Notes, Phone, Singles, Provisional
    assert last[1][COL_PROVISIONAL - 1] == PROVISIONAL_TRUE
    assert r.get("Carol")["provisional"] is True


def test_add_without_provisional_writes_blank_in_column_G():
    r = _roster_with([])
    r.add("Dan", gender="M", rating=5)
    last = r._ws.calls[-1]
    assert last[1][COL_PROVISIONAL - 1] == PROVISIONAL_FALSE
    assert r.get("Dan")["provisional"] is False


# ---------- set_rating clears provisional --------------------------------


def test_set_rating_on_provisional_player_clears_the_flag():
    """Admin's explicit rating call IS the team's confirmation of the
    rating — drop the (P) marker. Also writes the cleared flag back
    to the sheet (column G) so it sticks."""
    r = _roster_with([
        {"Name": "Eve", "Gender": "F", "Rating": 6,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": "Y"},
    ])
    assert r.get("Eve")["provisional"] is True
    r.set_rating("Eve", 7)
    assert r.get("Eve")["rating"] == 7
    assert r.get("Eve")["provisional"] is False
    # Two writes happened: the rating cell + the provisional cell.
    rating_writes = [
        c for c in r._ws.calls
        if c[0] == "update_cell" and c[2] == COL_RATING
    ]
    prov_writes = [
        c for c in r._ws.calls
        if c[0] == "update_cell" and c[2] == COL_PROVISIONAL
    ]
    assert rating_writes == [("update_cell", 2, COL_RATING, "7")]
    assert prov_writes == [
        ("update_cell", 2, COL_PROVISIONAL, PROVISIONAL_FALSE),
    ]


def test_set_rating_on_non_provisional_player_skips_the_extra_write():
    """No-op for the flag if it wasn't set — don't burn a sheet API
    call writing blank into a cell that's already blank."""
    r = _roster_with([
        {"Name": "Frank", "Gender": "M", "Rating": 5,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": ""},
    ])
    r.set_rating("Frank", 6)
    prov_writes = [
        c for c in r._ws.calls
        if c[0] == "update_cell" and c[2] == COL_PROVISIONAL
    ]
    assert prov_writes == []  # never touched column G


def test_set_rating_on_unknown_player_raises():
    r = _roster_with([])
    with pytest.raises(KeyError):
        r.set_rating("Ghost", 5)


# ---------- clear_provisional -------------------------------------------


def test_clear_provisional_writes_only_column_G():
    """Touches column G only — doesn't bother re-writing the rating."""
    r = _roster_with([
        {"Name": "Hank", "Gender": "M", "Rating": 5,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": "Y"},
    ])
    r.clear_provisional("Hank")
    assert r.get("Hank")["provisional"] is False
    # Only one write call: column G.
    writes = [c for c in r._ws.calls if c[0] == "update_cell"]
    assert writes == [("update_cell", 2, COL_PROVISIONAL, PROVISIONAL_FALSE)]


def test_clear_provisional_is_idempotent_when_not_provisional():
    """No write when the flag wasn't set — saves a sheet API call."""
    r = _roster_with([
        {"Name": "Ivy", "Gender": "F", "Rating": 5,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": ""},
    ])
    r.clear_provisional("Ivy")
    assert r.get("Ivy")["provisional"] is False
    writes = [c for c in r._ws.calls if c[0] == "update_cell"]
    assert writes == []


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
    # And the append_row call wrote the clean name to the sheet.
    appended = [c for c in r._ws.calls if c[0] == "append_row"][0][1]
    assert appended[0] == "Jack Fenner"


def test_add_treats_whitespace_variant_as_duplicate():
    """If 'Jack Fenner' already exists, adding 'Jack  Fenner' is a
    no-op (same person) rather than a second row."""
    r = _roster_with([
        {"Name": "Jack Fenner", "Gender": "M", "Rating": 4,
         "Notes": "", "Phone": "", "Singles": "", "Provisional": ""},
    ])
    # Snapshot the calls list before the add — should not grow.
    calls_before = len(r._ws.calls)
    r.add("Jack  Fenner", gender="M", rating=5)
    assert len(r._ws.calls) == calls_before  # no append_row happened
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


def test_rename_updates_name_cell_and_cache_key():
    r = _roster_with([
        {"Name": "Jack  Fenner", "Gender": "M", "Rating": 5,
         "Notes": "", "Phone": "+447", "Singles": "", "Provisional": ""},
    ])
    r.rename("Jack  Fenner", "Jack Fenner")
    assert r.get("Jack Fenner") is not None
    assert r.get("Jack  Fenner") is None
    # The other field values survive the rename.
    assert r.get("Jack Fenner")["phone"] == "+447"
    # Sheet was updated: one update_cell at the Name column.
    name_writes = [
        c for c in r._ws.calls
        if c[0] == "update_cell" and c[2] == 1  # COL_NAME = 1
    ]
    assert name_writes == [("update_cell", 2, 1, "Jack Fenner")]


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
    calls_before = len(r._ws.calls)
    r.rename("Geoff Chapman", "  Geoff Chapman  ")
    # No sheet write.
    assert len(r._ws.calls) == calls_before


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
