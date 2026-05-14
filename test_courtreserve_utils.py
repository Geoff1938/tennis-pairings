"""Tests for the pure helpers in courtreserve.py — normalize_hhmm,
_compute_blocked_courts. Anything that doesn't need Playwright.
"""
from __future__ import annotations

import pytest

from courtreserve import _compute_blocked_courts, normalize_hhmm


# ---------- normalize_hhmm ------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("13:00", "13:00"),
        ("1300", "13:00"),
        ("09:30", "09:30"),
        ("0930", "09:30"),
        ("9:30", "09:30"),
        ("930", "09:30"),
        ("1:00", "01:00"),
        ("100", "01:00"),
        ("13", "13:00"),
        ("9", "09:00"),
        ("00:00", "00:00"),
        ("0000", "00:00"),
        ("23:59", "23:59"),
        ("2359", "23:59"),
        ("  13:00  ", "13:00"),
    ],
)
def test_normalize_hhmm_accepts_common_shapes(raw, expected):
    assert normalize_hhmm(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "24:00",       # out of range hour
        "13:60",       # out of range minute
        "2400",        # out of range hour, 4-digit form
        "abc",
        "13:00:00",    # too many colons
        "13:",         # empty minute part
        ":30",         # empty hour part
        "12345",       # too long
        "1pm",         # mixed
        "13.00",       # wrong separator
    ],
)
def test_normalize_hhmm_rejects_garbage(raw):
    with pytest.raises(ValueError):
        normalize_hhmm(raw)


def test_normalize_hhmm_rejects_none():
    with pytest.raises(ValueError):
        normalize_hhmm(None)  # type: ignore[arg-type]


# ---------- _compute_blocked_courts ---------------------------------------


def _reservation(court_number: str, start_iso: str, end_iso: str,
                 title: str = ""):
    return {
        "court_number": court_number,
        "start": start_iso,
        "end": end_iso,
        "title": title,
    }


def test_blocked_courts_overlap_detection():
    # Target: 21 May 2026 13:00–14:30 BST = 12:00–13:30 UTC (May is BST).
    reservations = [
        # Court 5: 12:30–13:30 UTC — overlaps target.
        _reservation("5", "2026-05-21T12:30:00.000Z",
                     "2026-05-21T13:30:00.000Z", "Group Lesson"),
        # Court 6: 14:00–15:00 UTC — too late, no overlap (target end is 13:30 UTC).
        _reservation("6", "2026-05-21T14:00:00.000Z",
                     "2026-05-21T15:00:00.000Z"),
        # Court 7: 11:00–12:00 UTC — abuts target start, no overlap.
        _reservation("7", "2026-05-21T11:00:00.000Z",
                     "2026-05-21T12:00:00.000Z"),
        # Court 8: 11:30–13:30 UTC — straddles target start, overlap.
        _reservation("8", "2026-05-21T11:30:00.000Z",
                     "2026-05-21T13:30:00.000Z", "Coaching"),
    ]
    blocked = _compute_blocked_courts(
        reservations, "2026-05-21", "13:00", 90,
    )
    assert blocked == {"5", "8"}


def test_blocked_courts_accepts_hhmm_without_colon():
    """Regression: scheduled bookings stored start_time as '1300'.
    _compute_blocked_courts must normalise that internally so callers
    can't be hurt by historical persisted formats.
    """
    reservations = [
        _reservation("5", "2026-05-21T12:30:00.000Z",
                     "2026-05-21T13:30:00.000Z"),
    ]
    blocked = _compute_blocked_courts(
        reservations, "2026-05-21", "1300", 90,
    )
    assert blocked == {"5"}


def test_blocked_courts_ignores_other_dates():
    reservations = [
        # Same court+time but a day later — no overlap.
        _reservation("5", "2026-05-22T12:30:00.000Z",
                     "2026-05-22T13:30:00.000Z"),
    ]
    assert _compute_blocked_courts(
        reservations, "2026-05-21", "13:00", 90,
    ) == set()


def test_blocked_courts_skips_malformed_entries():
    reservations = [
        {"court_number": "5", "start": None, "end": None},
        {"court_number": "6", "start": "garbage", "end": "garbage"},
        _reservation("7", "2026-05-21T12:30:00.000Z",
                     "2026-05-21T13:30:00.000Z"),
    ]
    assert _compute_blocked_courts(
        reservations, "2026-05-21", "13:00", 90,
    ) == {"7"}


def test_blocked_courts_winter_offset_handling():
    # January is GMT (UTC+0) — target 13:00 local = 13:00 UTC.
    reservations = [
        # Court 5: 13:30–14:30 UTC — overlaps target end (14:30 local = 14:30 UTC).
        _reservation("5", "2026-01-15T13:30:00.000Z",
                     "2026-01-15T14:30:00.000Z"),
    ]
    assert _compute_blocked_courts(
        reservations, "2026-01-15", "13:00", 90,
    ) == {"5"}
