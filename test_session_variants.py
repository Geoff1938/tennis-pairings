"""Tests for the variant-based session resolution introduced when
the 18-29 Social Doubles Night was added to Thursdays alongside the
existing regular Thursday social.

Covers:
  * resolve_next_session(variant=...) routing
  * _normalise_variant alias handling (incl. en-dash, case)
  * tool_kickoff_session disambiguation when CR lists both events
  * session_in_progress error names the loaded session
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ---------- resolve_next_session(variant=...) ---------------------------


def test_regular_variant_picks_next_regular_session():
    """On a Wednesday the next regular session is Thursday."""
    from session_types import resolve_next_session

    st = resolve_next_session(date(2026, 6, 24), variant="regular")
    assert st.key == "thursday"


def test_18_29_variant_picks_next_thursday_1829_from_anywhere_in_week():
    """The 18-29 session always resolves to the next Thursday's
    youth event, regardless of which day of the week the admin
    typed the command on."""
    from session_types import resolve_next_session

    # Tuesday → next 18-29 is two days later on Thursday
    assert resolve_next_session(
        date(2026, 6, 23), variant="18-29",
    ).key == "thursday_1829"
    # Thursday itself → today's session
    assert resolve_next_session(
        date(2026, 6, 25), variant="18-29",
    ).key == "thursday_1829"
    # Saturday → next is the following Thursday (5-day wrap-around)
    assert resolve_next_session(
        date(2026, 6, 27), variant="18-29",
    ).key == "thursday_1829"


def test_regular_variant_skips_18_29_on_thursday():
    """On a Thursday, 'regular' must NOT return the youth session
    just because it shares the weekday — variant filter wins."""
    from session_types import resolve_next_session

    st = resolve_next_session(date(2026, 6, 25), variant="regular")
    assert st.key == "thursday"


def test_unknown_variant_raises_lookuperror():
    """Typo / unknown variant surfaces a LookupError the caller can
    map to a user-friendly error."""
    from session_types import resolve_next_session

    with pytest.raises(LookupError, match="variant"):
        resolve_next_session(date(2026, 6, 23), variant="bogus")


def test_resolve_without_variant_keeps_legacy_behaviour():
    """Backwards compat: no variant arg → same picks as before
    (Mon/Tue → Tue, Wed/Thu → Thu, etc.). When two SessionTypes
    share the weekday the result is one of them (we don't care
    which since the no-variant call site is the legacy fallback)."""
    from session_types import resolve_next_session

    assert resolve_next_session(date(2026, 6, 23)).key == "tuesday"
    # On Thursday, weekday matches BOTH regular and 18-29; either is
    # acceptable for the legacy fallback path (the kickoff caller
    # prefers the CR-probe path when both args are blank anyway).
    assert resolve_next_session(date(2026, 6, 25)).key in {
        "thursday", "thursday_1829",
    }


# ---------- _normalise_variant alias handling ---------------------------


@pytest.mark.parametrize(
    "phrase,expected", [
        ("regular", "regular"),
        ("Regular", "regular"),
        ("REG", "regular"),
        ("intermediate+", "regular"),
        ("Intermediate Plus", "regular"),
        ("int+", "regular"),
        ("18-29", "18-29"),
        ("18–29", "18-29"),       # en-dash
        ("18—29", "18-29"),       # em-dash
        ("18-30", "18-29"),
        ("18 to 29", "18-29"),
        ("Youth", "18-29"),
        ("young", "18-29"),
        ("  18-29  ", "18-29"),   # trim whitespace
    ],
)
def test_normalise_variant_aliases(phrase, expected):
    from admin_bot import _normalise_variant

    assert _normalise_variant(phrase) == expected


def test_normalise_variant_none_returns_none():
    """Empty / None input stays None so the caller can route to the
    'no qualifier — let the kickoff path probe CR' branch."""
    from admin_bot import _normalise_variant

    assert _normalise_variant(None) is None
    assert _normalise_variant("") is None


def test_normalise_variant_unknown_raises_valueerror():
    from admin_bot import _normalise_variant

    with pytest.raises(ValueError, match="unknown variant"):
        _normalise_variant("bogus-day")


# ---------- tool_kickoff_session disambiguation -------------------------


def test_tool_returns_unknown_variant_error_on_typo():
    """Typed variant the bot can't normalise → friendly error rather
    than crashing or falling through to the wrong session."""
    import admin_bot

    result = admin_bot.tool_kickoff_session(variant="bogus")
    assert result["ok"] is False
    assert result["error"] == "unknown_variant"


def test_tool_dispatches_canonical_variant_to_kickoff(monkeypatch):
    """Admin says '18-30', tool normalises to '18-29' and passes
    THAT to kickoff_session — not the raw alias."""
    import admin_bot
    import thursday_kickoff

    captured: dict = {}
    def fake_kickoff(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "session_type": "thursday_1829"}

    monkeypatch.setattr(thursday_kickoff, "kickoff_session", fake_kickoff)
    admin_bot.tool_kickoff_session(variant="18-30")
    assert captured["variant"] == "18-29"


def test_tool_dispatches_session_type_through(monkeypatch):
    """Explicit session_type wins; variant can be omitted."""
    import admin_bot
    import thursday_kickoff

    captured: dict = {}
    def fake_kickoff(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(thursday_kickoff, "kickoff_session", fake_kickoff)
    admin_bot.tool_kickoff_session(session_type="thursday_1829")
    assert captured["session_key"] == "thursday_1829"
    assert captured["variant"] is None


# ---------- kickoff_session ambiguity & lock -----------------------------


class _FakeState:
    def __init__(self, phase: str = "", session_type: str = ""):
        self.phase = phase
        self.session_type = session_type


def test_kickoff_session_in_progress_message_identifies_loaded_session(
    monkeypatch,
):
    """When a 18-29 kickoff is rejected because a regular Thursday
    session is already loaded, the message says so — not just
    'phase=awaiting_extras'."""
    import thursday_kickoff
    import session_state

    monkeypatch.setattr(
        session_state, "get_tonight",
        lambda: _FakeState(
            phase="awaiting_extras", session_type="thursday",
        ),
    )
    # Stop CR scraping mid-flow by making variant resolution succeed
    # but then the in-progress check fires before any CR call.
    result = thursday_kickoff.kickoff_session(variant="18-29")
    assert result["ok"] is False
    assert result["error"] == "session_in_progress"
    assert result["in_progress_session_type"] == "thursday"
    # Friendly message names the loaded session by display name.
    assert "Thursday Evening Club Social" in result["message"]


def test_kickoff_session_in_progress_message_for_unset_session_type(
    monkeypatch,
):
    """Legacy sessions started before the session_type field existed
    still get a sane (if generic) error message rather than a
    KeyError or empty display name."""
    import thursday_kickoff
    import session_state

    monkeypatch.setattr(
        session_state, "get_tonight",
        lambda: _FakeState(phase="awaiting_extras", session_type=""),
    )
    result = thursday_kickoff.kickoff_session(variant="regular")
    assert result["ok"] is False
    assert "in flight" in result["message"]


def test_resolve_kickoff_session_needs_disambiguation_when_both_listed(
    monkeypatch,
):
    """The crux of the feature: both events listed on CR for today
    → tool returns needs_disambiguation so the bot can ask which."""
    import thursday_kickoff
    from session_types import SESSION_TYPES

    # Force the candidate-probe to find BOTH Thursday SessionTypes.
    monkeypatch.setattr(
        thursday_kickoff, "_candidate_sessions_for_today",
        lambda today=None: [
            SESSION_TYPES["thursday"], SESSION_TYPES["thursday_1829"],
        ],
    )
    session, err = thursday_kickoff._resolve_kickoff_session(
        session_key=None, variant=None, today=date(2026, 6, 25),
    )
    assert session is None
    assert err is not None
    assert err["error"] == "needs_disambiguation"
    variants = {opt["variant"] for opt in err["options"]}
    assert variants == {"regular", "18-29"}


def test_resolve_kickoff_session_auto_picks_when_one_listed(monkeypatch):
    """Only the regular event listed → no prompt, just use it."""
    import thursday_kickoff
    from session_types import SESSION_TYPES

    monkeypatch.setattr(
        thursday_kickoff, "_candidate_sessions_for_today",
        lambda today=None: [SESSION_TYPES["thursday"]],
    )
    session, err = thursday_kickoff._resolve_kickoff_session(
        session_key=None, variant=None, today=date(2026, 6, 25),
    )
    assert err is None
    assert session.key == "thursday"


def test_resolve_kickoff_session_falls_back_when_cr_unreachable(
    monkeypatch,
):
    """CR scrape blows up → we don't abort the kickoff, we fall back
    to the weekday resolver. The downstream _collect_session_data
    will surface a real scrape error if CR is genuinely down."""
    import thursday_kickoff

    def _boom(today=None):
        raise ConnectionError("dns failure")

    monkeypatch.setattr(
        thursday_kickoff, "_candidate_sessions_for_today", _boom,
    )
    session, err = thursday_kickoff._resolve_kickoff_session(
        session_key=None, variant=None, today=date(2026, 6, 25),
    )
    assert err is None
    assert session is not None
    # Either Thursday variant is acceptable here; we just need
    # SOMETHING so kickoff can proceed.
    assert session.weekday == 3


def test_unknown_session_key_returns_friendly_error():
    """Bad session_type from the LLM → clean error, not a crash."""
    import thursday_kickoff

    session, err = thursday_kickoff._resolve_kickoff_session(
        session_key="weekend_extravaganza", variant=None,
    )
    assert session is None
    assert err is not None
    assert err["error"] == "unknown_session_type"
