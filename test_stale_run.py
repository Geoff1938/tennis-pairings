"""Tests for the stale-run reminder: session_state activity tracking
and admin_bot._maybe_remind_stale_run decision logic.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest


@pytest.fixture
def ss(tmp_path, monkeypatch):
    import session_state as s
    monkeypatch.setattr(s, "SESSION_STATE_PATH", tmp_path / "session_state.json")
    return s


# ---------- session_state activity tracking -------------------------------


def test_start_tonight_stamps_started_and_activity(ss):
    state = ss.start_tonight(["A", "B"], date="2026-05-21")
    assert state.started_at
    assert state.last_activity_at
    assert state.started_at == state.last_activity_at
    assert state.idle_reminder_sent is False


def test_note_activity_noop_without_run(ss):
    # No phase set → no in-flight run → note_activity must not stamp.
    out = ss.note_activity(started_by="447x", channel_jid="jid")
    assert out.started_by == ""
    assert out.last_activity_at == ""


def test_note_activity_records_owner_once(ss):
    ss.start_tonight(["A"], date="2026-05-21")
    ss.set_phase("awaiting_extras")
    ss.note_activity(started_by="447111", channel_jid="grpA")
    s1 = ss.get_tonight()
    assert s1.started_by == "447111"
    assert s1.channel_jid == "grpA"
    # A later interaction by a different sender must NOT overwrite the
    # original owner / channel.
    ss.note_activity(started_by="447999", channel_jid="grpB")
    s2 = ss.get_tonight()
    assert s2.started_by == "447111"
    assert s2.channel_jid == "grpA"


def test_note_activity_resets_reminder_gate(ss):
    ss.start_tonight(["A"], date="2026-05-21")
    ss.set_phase("draft_ready")
    ss.mark_idle_reminder_sent()
    assert ss.get_tonight().idle_reminder_sent is True
    ss.note_activity(started_by="447111", channel_jid="grpA")
    assert ss.get_tonight().idle_reminder_sent is False


def test_clear_tonight_wipes_tracking(ss):
    ss.start_tonight(["A"], date="2026-05-21")
    ss.set_phase("draft_ready")
    ss.note_activity(started_by="447111", channel_jid="grpA")
    ss.clear_tonight()
    s = ss.get_tonight()
    assert s.phase == ""
    assert s.started_by == ""
    assert s.last_activity_at == ""
    assert s.idle_reminder_sent is False


# ---------- _maybe_remind_stale_run ---------------------------------------


@pytest.fixture
def bot(tmp_path, monkeypatch):
    import admin_bot
    import session_state as s
    monkeypatch.setattr(s, "SESSION_STATE_PATH", tmp_path / "session_state.json")
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        admin_bot, "send_to_group",
        lambda jid, text: sent.append((jid, text)) or True,
    )
    return admin_bot, s, sent


def test_no_reminder_without_run(bot):
    admin_bot, s, sent = bot
    admin_bot._maybe_remind_stale_run(datetime.now(), fallback_jid="grp")
    assert sent == []


def test_no_reminder_before_threshold(bot):
    admin_bot, s, sent = bot
    s.start_tonight(["A"], date="2026-05-21")
    s.set_phase("awaiting_extras")
    s.note_activity(started_by="447111", channel_jid="grpA")
    # 30 min later — under the 60-min threshold.
    later = datetime.now().astimezone() + timedelta(minutes=30)
    admin_bot._maybe_remind_stale_run(later, fallback_jid="fallback")
    assert sent == []


def test_reminder_fires_after_threshold(bot, monkeypatch):
    admin_bot, s, sent = bot
    s.start_tonight(["A"], date="2026-05-21")
    s.set_phase("awaiting_extras")
    s.note_activity(started_by="447111", channel_jid="grpA")
    later = datetime.now().astimezone() + timedelta(
        minutes=admin_bot.STALE_RUN_REMINDER_MINUTES + 5
    )
    admin_bot._maybe_remind_stale_run(later, fallback_jid="fallback")
    assert len(sent) == 1
    jid, text = sent[0]
    assert jid == "grpA"                       # posts to the start channel
    assert "session in progress" in text       # neutral wording (no test/dry)
    assert "boris clear run" in text
    assert "won't clear it unless you ask" in text
    # Gate is now set — a second tick must NOT repeat.
    admin_bot._maybe_remind_stale_run(
        later + timedelta(minutes=10), fallback_jid="fallback"
    )
    assert len(sent) == 1


def test_reminder_resets_after_activity(bot):
    admin_bot, s, sent = bot
    s.start_tonight(["A"], date="2026-05-21")
    s.set_phase("draft_ready")
    s.note_activity(started_by="447111", channel_jid="grpA")
    later = datetime.now().astimezone() + timedelta(
        minutes=admin_bot.STALE_RUN_REMINDER_MINUTES + 1
    )
    admin_bot._maybe_remind_stale_run(later, fallback_jid="fb")
    assert len(sent) == 1
    # Admin interacts again → gate resets, fresh idle period can renudge.
    s.note_activity(started_by="447111", channel_jid="grpA")
    assert s.get_tonight().idle_reminder_sent is False
    later2 = later + timedelta(
        minutes=admin_bot.STALE_RUN_REMINDER_MINUTES + 1
    )
    admin_bot._maybe_remind_stale_run(later2, fallback_jid="fb")
    assert len(sent) == 2


def test_reminder_uses_fallback_jid_when_no_channel(bot):
    admin_bot, s, sent = bot
    # Simulate an auto-kickoff: run started but no channel recorded.
    s.start_tonight(["A"], date="2026-05-21")
    s.set_phase("awaiting_extras")
    later = datetime.now().astimezone() + timedelta(
        minutes=admin_bot.STALE_RUN_REMINDER_MINUTES + 2
    )
    admin_bot._maybe_remind_stale_run(later, fallback_jid="adminGrp")
    assert len(sent) == 1
    assert sent[0][0] == "adminGrp"
