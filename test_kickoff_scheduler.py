"""Tests for the session-kickoff plumbing: next-session resolution and
the session_type carrying through start_tonight + tool_generate_pairings'
defaulting logic.

The original auto-kickoff scheduler (Thursday 09:35) was removed —
sessions now only fire when an organiser says so. Resolution of "which
session do you mean?" defaults to the next scheduled day by weekday.
"""
from __future__ import annotations

from datetime import date

import pytest


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Point session_state's storage at a tmp file."""
    import session_state as ss

    monkeypatch.setattr(ss, "SESSION_STATE_PATH", tmp_path / "session_state.json")
    return ss


# ---------- resolve_next_session ----------------------------------------


def test_resolve_next_session_monday_picks_tuesday():
    from session_types import resolve_next_session

    # 2026-05-25 is a Monday.
    assert resolve_next_session(date(2026, 5, 25)).key == "tuesday"


def test_resolve_next_session_tuesday_picks_tuesday():
    from session_types import resolve_next_session

    # 2026-05-26 is a Tuesday — same-day session wins (offset 0).
    assert resolve_next_session(date(2026, 5, 26)).key == "tuesday"


def test_resolve_next_session_wednesday_picks_thursday():
    from session_types import resolve_next_session

    assert resolve_next_session(date(2026, 5, 27)).key == "thursday"


def test_resolve_next_session_thursday_picks_thursday():
    from session_types import resolve_next_session

    assert resolve_next_session(date(2026, 5, 28)).key == "thursday"


def test_resolve_next_session_friday_picks_saturday():
    from session_types import resolve_next_session

    assert resolve_next_session(date(2026, 5, 22)).key == "saturday"


def test_resolve_next_session_saturday_picks_saturday():
    from session_types import resolve_next_session

    assert resolve_next_session(date(2026, 5, 23)).key == "saturday"


def test_resolve_next_session_sunday_picks_tuesday():
    from session_types import resolve_next_session

    # 2026-05-24 is a Sunday — wraps around to next Tuesday.
    assert resolve_next_session(date(2026, 5, 24)).key == "tuesday"


# ---------- session-type registry sanity --------------------------------


def test_registry_has_three_sessions():
    from session_types import SESSION_TYPES

    assert set(SESSION_TYPES.keys()) == {"tuesday", "thursday", "saturday"}


def test_saturday_is_afternoon():
    from session_types import SESSION_TYPES

    assert SESSION_TYPES["saturday"].start_time_hhmm == "14:00"


def test_tuesday_and_thursday_share_evening_start():
    from session_types import SESSION_TYPES

    assert SESSION_TYPES["tuesday"].start_time_hhmm == "19:30"
    assert SESSION_TYPES["thursday"].start_time_hhmm == "19:30"


def test_tuesday_and_saturday_share_admin_group():
    from session_types import SESSION_TYPES

    assert (
        SESSION_TYPES["tuesday"].admin_group_name
        == SESSION_TYPES["saturday"].admin_group_name
        == "Westside social tennis organisers"
    )
    assert (
        SESSION_TYPES["thursday"].admin_group_name == "Thursday Tennis Organisers"
    )


def test_tuesday_and_saturday_share_docx_template():
    from session_types import SESSION_TYPES

    assert (
        SESSION_TYPES["tuesday"].docx_template_relpath
        == SESSION_TYPES["saturday"].docx_template_relpath
    )
    assert (
        SESSION_TYPES["thursday"].docx_template_relpath
        != SESSION_TYPES["saturday"].docx_template_relpath
    )


# ---------- session_type plumbing through admin_bot helpers -------------


def test_docx_template_resolves_by_session_type():
    from admin_bot import _docx_template_for

    assert _docx_template_for("thursday").name == "Thursday Social Tennis.docx"
    assert _docx_template_for("tuesday").name == "Westside Social Tennis.docx"
    assert _docx_template_for("saturday").name == "Westside Social Tennis.docx"
    # Empty / unknown falls back to Thursday template.
    assert _docx_template_for("").name == "Thursday Social Tennis.docx"


def test_docx_basename_resolves_by_session_type():
    from admin_bot import _docx_basename_for

    assert _docx_basename_for("thursday") == "Thursday Social Tennis"
    assert _docx_basename_for("tuesday") == "Tuesday Social Tennis"
    assert _docx_basename_for("saturday") == "Saturday Social Tennis"
    assert _docx_basename_for("") == "Thursday Social Tennis"
