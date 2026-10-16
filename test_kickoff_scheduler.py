"""Tests for the session-kickoff plumbing: next-session resolution and
the session_type carrying through start_tonight + tool_generate_pairings'
defaulting logic.

The original auto-kickoff scheduler (Thursday 09:35) was removed —
sessions now only fire when an organiser says so. Resolution of "which
session do you mean?" defaults to the next scheduled day by weekday.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

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


def test_registry_has_five_sessions():
    """Four weekly entries (incl. the Jun 2026 Thursday 18-29 youth
    session) plus the occasional Friday Tennis & Tapas social."""
    from session_types import SESSION_TYPES

    assert set(SESSION_TYPES.keys()) == {
        "tuesday", "thursday", "thursday_1829", "saturday", "friday_tapas",
    }


def test_friday_tapas_is_occasional_four_rotations():
    from session_types import SESSION_TYPES

    st = SESSION_TYPES["friday_tapas"]
    assert st.num_rotations == 4
    assert len(st.rotation_durations) == 4
    assert st.start_time_hhmm == "19:00"
    assert sum(st.rotation_durations) == 90  # 19:00-20:30
    assert st.rotation_durations == [25, 25, 20, 20]
    assert st.weekly is False


def test_resolve_next_session_friday_without_variant_skips_tapas():
    """The occasional Friday event must be asked for explicitly — a
    bare kickoff on a Friday still rolls forward to Saturday."""
    from session_types import resolve_next_session

    # 2026-10-16 is a Friday.
    assert resolve_next_session(date(2026, 10, 16)).key == "saturday"
    assert resolve_next_session(
        date(2026, 10, 15), variant="tapas",
    ).key == "friday_tapas"


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
        == "Westside organisers of social tennis"
    )
    assert (
        SESSION_TYPES["thursday"].admin_group_name == "Thursday Tennis Organisers"
    )


def test_all_session_types_share_docx_template():
    """All three session types now use the shared Westside template —
    Thursday was moved off its bespoke (QR-code) template Jun 2026.
    The renderer rewrites the page-header banner per session, so
    each day's doc still says the right day at the top."""
    from session_types import SESSION_TYPES

    templates = {
        st.docx_template_relpath for st in SESSION_TYPES.values()
    }
    assert templates == {"tmp/Westside Social Tennis.docx"}


# ---------- session_type plumbing through admin_bot helpers -------------


def test_docx_template_resolves_by_session_type():
    from admin_bot import _docx_template_for

    # All three days now point at the shared Westside template
    # (Thursday's bespoke QR-code template was retired Jun 2026).
    assert _docx_template_for("thursday").name == "Westside Social Tennis.docx"
    assert _docx_template_for("tuesday").name == "Westside Social Tennis.docx"
    assert _docx_template_for("saturday").name == "Westside Social Tennis.docx"
    # Empty / unknown still falls back to the Thursday entry — which
    # is the shared Westside template too.
    assert _docx_template_for("").name == "Westside Social Tennis.docx"


def test_docx_basename_resolves_by_session_type():
    from admin_bot import _docx_basename_for

    assert _docx_basename_for("thursday") == "Thursday Social Tennis"
    assert _docx_basename_for("tuesday") == "Tuesday Social Tennis"
    assert _docx_basename_for("saturday") == "Saturday Social Tennis"
    assert _docx_basename_for("") == "Thursday Social Tennis"


# ---------- generate_pairings rotation defaults --------------------------


def _capture_generate(monkeypatch, tmp_path, session_type):
    """Run tool_generate_pairings with a stubbed engine; return the
    kwargs it would have passed to _generate_parallel."""
    import admin_bot
    import session_state as ss
    from pairings import make_plan

    monkeypatch.setattr(ss, "SESSION_STATE_PATH", tmp_path / "session_state.json")
    monkeypatch.setattr(admin_bot, "Roster", lambda: SimpleNamespace(all=lambda: {}))
    monkeypatch.setattr(admin_bot, "_CURRENT_GROUP_JID", SimpleNamespace(get=lambda d=None: None))
    captured: list[dict] = []

    def fake_generate(**kw):
        captured.append(kw)
        hist = tmp_path / "h.json"
        hist.write_text("[]")
        names = kw["attendees"]
        return make_plan(
            names, {n: {"rating": 5, "gender": "?"} for n in names}, hist,
            court_labels=kw["court_labels"], num_rotations=kw["num_rotations"],
            start_time_hhmm=kw["start_time_hhmm"],
            rotation_durations=kw["rotation_durations"], seed=1,
        )

    monkeypatch.setattr(admin_bot, "_generate_parallel", fake_generate)
    ss.start_tonight(
        [f"P{i}" for i in range(8)], session_type=session_type,
        court_labels=["1", "2"],
    )
    return admin_bot, captured


def test_generate_uses_tapas_defaults_and_reroll_keeps_times(monkeypatch, tmp_path):
    admin_bot, captured = _capture_generate(monkeypatch, tmp_path, "friday_tapas")

    assert admin_bot.tool_generate_pairings().get("ok") is not False
    assert captured[-1]["num_rotations"] == 4
    assert captured[-1]["rotation_durations"] == [25, 25, 20, 20]
    assert captured[-1]["start_time_hhmm"] == "19:00"

    # Re-time the draft, then re-roll: the new times survive.
    res = admin_bot.tool_set_rotation_times([30, 20, 20, 20], "18:45")
    assert res["ok"] and res["times"][0] == "18:45-19:15"
    admin_bot.tool_generate_pairings()
    assert captured[-1]["num_rotations"] == 4
    assert captured[-1]["rotation_durations"] == [30, 20, 20, 20]
    assert captured[-1]["start_time_hhmm"] == "18:45"

    # Different rotation count with no durations → usual 90 min span split.
    admin_bot.tool_generate_pairings(num_rotations=3)
    assert captured[-1]["rotation_durations"] == [30, 30, 30]
    assert captured[-1]["start_time_hhmm"] == "19:00"


def test_generate_weekly_session_keeps_three_rotations(monkeypatch, tmp_path):
    admin_bot, captured = _capture_generate(monkeypatch, tmp_path, "thursday")
    admin_bot.tool_generate_pairings()
    assert captured[-1]["num_rotations"] == 3
    assert captured[-1]["rotation_durations"] == [45, 40, 35]
    admin_bot.tool_generate_pairings(num_rotations=4)
    assert captured[-1]["rotation_durations"] == [30, 30, 30, 30]
