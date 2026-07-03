"""Tests for the kickoff message formatter.

Specifically the bits that change rendering based on the
``provisional`` flag — the rest of the formatter is straightforward
string building and hard to break."""

from __future__ import annotations

from session_types import SESSION_TYPES
from thursday_kickoff import format_kickoff_message


def _data(*registrants: dict, waitlist: list[dict] | None = None) -> dict:
    """Minimal data dict matching what ``_collect_session_data`` builds."""
    return {
        "date_str": "Sat 30 May, 14:00",
        "registrants": list(registrants),
        "waitlist": list(waitlist or []),
        "cr_courts": ["Court #1", "Court #5"],
        "new_player_names": [],
    }


def _entry(name: str, rating, *, is_new: bool = False,
           provisional: bool = False) -> dict:
    return {
        "name": name, "rating": rating,
        "is_new": is_new, "provisional": provisional,
    }


def test_provisional_player_renders_with_P_suffix():
    sat = SESSION_TYPES["saturday"]
    data = _data(
        _entry("Geoff", 2, provisional=False),
        _entry("Silvia", 6, provisional=True),
    )
    msg = format_kickoff_message(data, sat)
    assert "Geoff (rating 2)" in msg
    assert "Silvia (rating 6P)" in msg
    # The legend explaining "P" appears once we have at least one.
    assert "are provisional" in msg


def test_no_provisional_means_no_legend():
    sat = SESSION_TYPES["saturday"]
    data = _data(
        _entry("Geoff", 2),
        _entry("Silvia", 6),
    )
    msg = format_kickoff_message(data, sat)
    # Plain rendering, no P, no legend.
    assert "Silvia (rating 6)" in msg
    assert "6P" not in msg
    assert "are provisional" not in msg


def test_provisional_with_unknown_rating_does_not_get_P():
    """Edge case: a provisional flag combined with rating "?" — the
    "?" should stay clean. Provisional is meaningful only when a
    numeric rating exists to be confirmed."""
    sat = SESSION_TYPES["saturday"]
    data = _data(
        _entry("Mystery", "?", provisional=True),
    )
    msg = format_kickoff_message(data, sat)
    assert "Mystery (rating ?)" in msg  # no "P"
    # And the provisional legend doesn't trigger (we filter on
    # rating != "?" so this player doesn't count).
    assert "are provisional" not in msg


def test_provisional_player_on_waitlist_also_marked():
    sat = SESSION_TYPES["saturday"]
    data = _data(
        _entry("Geoff", 2),
        waitlist=[_entry("Hannah", 4, provisional=True)],
    )
    msg = format_kickoff_message(data, sat)
    assert "Hannah (rating 4P)" in msg


def test_provisional_count_in_legend_matches_marker_count():
    sat = SESSION_TYPES["saturday"]
    data = _data(
        _entry("A", 5, provisional=True),
        _entry("B", 6, provisional=True),
        _entry("C", 7, provisional=True),
        _entry("D", 4, provisional=False),
    )
    msg = format_kickoff_message(data, sat)
    assert "(3) are provisional" in msg
