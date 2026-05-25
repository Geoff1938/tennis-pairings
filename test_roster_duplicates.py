"""Tests for the roster duplicate-detection helpers (apostrophe
variants, nickname expansion, whitespace differences)."""

from __future__ import annotations

from roster import (
    _canonical_key,
    _expand_first_name,
    _normalise_apostrophes,
    _normalise_whitespace,
    find_duplicates,
)


# ---------- low-level helpers -------------------------------------------


def test_normalise_apostrophes_curly_to_straight():
    # U+2019 (right single quotation mark) → U+0027 (apostrophe).
    assert _normalise_apostrophes("Luke O’Mahoney") == "Luke O'Mahoney"


def test_normalise_apostrophes_leaves_straight_unchanged():
    assert _normalise_apostrophes("Luke O'Mahoney") == "Luke O'Mahoney"


def test_normalise_whitespace_collapses_runs():
    assert _normalise_whitespace("  Geoff   Chapman  ") == "Geoff Chapman"


def test_expand_first_name_known_nicknames():
    assert _expand_first_name("Ben") == "benjamin"
    assert _expand_first_name("Mike") == "michael"
    assert _expand_first_name("Tom") == "thomas"


def test_expand_first_name_unknown_passes_through_lower():
    assert _expand_first_name("Aurangzeb") == "aurangzeb"


# ---------- canonical keys ----------------------------------------------


def test_canonical_key_collapses_apostrophe_variants():
    assert _canonical_key("Luke O’Mahoney") == _canonical_key("Luke O'Mahoney")


def test_canonical_key_collapses_nicknames():
    assert _canonical_key("Ben Hodgson") == _canonical_key("Benjamin Hodgson")
    assert _canonical_key("Mike Webber") == _canonical_key("Michael Webber")


def test_canonical_key_collapses_whitespace_and_case():
    assert _canonical_key("  geoff   chapman ") == _canonical_key("Geoff Chapman")


def test_canonical_key_distinguishes_different_surnames():
    # Same first name, different last → different keys.
    assert _canonical_key("Ben Hodgson") != _canonical_key("Ben Durrant")


def test_canonical_key_single_name():
    assert _canonical_key("Cher") == "cher"


# ---------- find_duplicates ---------------------------------------------


def _roster(*names: str) -> dict[str, dict]:
    """Tiny fixture: a roster cache with just names (no metadata)."""
    return {n: {"gender": "?", "rating": "?", "notes": ""} for n in names}


def test_find_duplicates_flags_apostrophe_pair():
    data = _roster("Luke O’Mahoney", "Luke O'Mahoney", "Geoff Chapman")
    groups = find_duplicates(data)
    assert len(groups) == 1
    g = groups[0]
    assert set(g["names"]) == {"Luke O’Mahoney", "Luke O'Mahoney"}
    assert g["hint"] == "apostrophe variant"


def test_find_duplicates_flags_nickname_pair():
    data = _roster("Ben Hodgson", "Benjamin Hodgson", "Geoff Chapman")
    groups = find_duplicates(data)
    assert len(groups) == 1
    assert set(groups[0]["names"]) == {"Ben Hodgson", "Benjamin Hodgson"}
    assert groups[0]["hint"] == "nickname/whitespace variant"


def test_find_duplicates_returns_empty_when_no_dupes():
    data = _roster("Geoff Chapman", "Hannah Han", "Aurangzeb Khan")
    assert find_duplicates(data) == []


def test_find_duplicates_ignores_singletons_only():
    # Three different people whose first names happen to be nicknames
    # of each other but surnames differ — must NOT be grouped.
    data = _roster("Ben Smith", "Ben Hodgson", "Benjamin Jones")
    groups = find_duplicates(data)
    # Only the "Ben"/"Benjamin" pair with matching surname would group,
    # and there isn't one here.
    assert groups == []


def test_find_duplicates_finds_multiple_independent_groups():
    data = _roster(
        "Luke O’Mahoney", "Luke O'Mahoney",
        "Ben Hodgson", "Benjamin Hodgson",
        "Geoff Chapman",
    )
    groups = find_duplicates(data)
    assert len(groups) == 2
    keys = {g["key"] for g in groups}
    # Both keys should be lowercased, expanded.
    assert any("o'mahoney" in k for k in keys)
    assert any("benjamin hodgson" in k for k in keys)


def test_find_duplicates_deterministic_ordering():
    # Same input should produce the same group order across calls.
    data = _roster(
        "Ben Hodgson", "Benjamin Hodgson",
        "Luke O’Mahoney", "Luke O'Mahoney",
    )
    g1 = find_duplicates(data)
    g2 = find_duplicates(data)
    assert g1 == g2
