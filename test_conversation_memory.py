"""Tests for _recent_conversation — the multi-turn memory
reconstruction that feeds prior Boris<->admin turns into run_agent.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pytest


@pytest.fixture
def bridged(tmp_path, monkeypatch):
    """A synthetic bridge messages.db + admin_bot pointed at it."""
    import admin_bot

    db = tmp_path / "messages.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE messages (id TEXT, chat_jid TEXT, sender TEXT, "
        "content TEXT, timestamp TEXT)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(admin_bot, "BRIDGE_DB", db)
    return admin_bot, db


def _insert(db, chat_jid, rows):
    conn = sqlite3.connect(db)
    for i, (content, ts) in enumerate(rows):
        conn.execute(
            "INSERT INTO messages VALUES (?,?,?,?,?)",
            (f"m{i}", chat_jid, "447798854809", content, ts),
        )
    conn.commit()
    conn.close()


def test_reconstructs_thread_in_order(bridged):
    admin_bot, db = bridged
    jid = "grp@g.us"
    base = datetime(2026, 5, 15, 9, 18, 0)
    _insert(db, jid, [
        ("boris can you cancel the court booking you made earlier today",
         base.isoformat()),
        (admin_bot.BOT_REPLY_PREFIX + admin_bot.WORKING_ON_IT_TEXT,
         (base + timedelta(seconds=2)).isoformat()),
        (admin_bot.BOT_REPLY_PREFIX + "I don't have it in context. "
         "Give me court + date/time.",
         (base + timedelta(seconds=30)).isoformat()),
    ])
    now = (base + timedelta(minutes=1)).isoformat()
    convo = admin_bot._recent_conversation(jid, now)
    assert convo == [
        {"role": "user",
         "content": "can you cancel the court booking you made earlier today"},
        {"role": "assistant",
         "content": "I don't have it in context. Give me court + date/time."},
    ]


def test_drops_working_on_it_filler(bridged):
    admin_bot, db = bridged
    jid = "grp@g.us"
    base = datetime(2026, 5, 15, 9, 0, 0)
    _insert(db, jid, [
        ("boris hello", base.isoformat()),
        (admin_bot.BOT_REPLY_PREFIX + admin_bot.WORKING_ON_IT_TEXT,
         (base + timedelta(seconds=1)).isoformat()),
        (admin_bot.BOT_REPLY_PREFIX + "Hi! How can I help?",
         (base + timedelta(seconds=5)).isoformat()),
    ])
    convo = admin_bot._recent_conversation(
        jid, (base + timedelta(minutes=1)).isoformat()
    )
    assert {"role": "assistant", "content": admin_bot.WORKING_ON_IT_TEXT} \
        not in convo
    assert convo[-1] == {"role": "assistant", "content": "Hi! How can I help?"}


def test_excludes_untriggered_chatter(bridged):
    admin_bot, db = bridged
    jid = "grp@g.us"
    base = datetime(2026, 5, 15, 9, 0, 0)
    _insert(db, jid, [
        ("boris what's on tonight", base.isoformat()),
        (admin_bot.BOT_REPLY_PREFIX + "Social tennis at 19:30.",
         (base + timedelta(seconds=4)).isoformat()),
        ("see you all there!", (base + timedelta(seconds=10)).isoformat()),
        ("anyone bringing balls?", (base + timedelta(seconds=20)).isoformat()),
    ])
    convo = admin_bot._recent_conversation(
        jid, (base + timedelta(minutes=1)).isoformat()
    )
    assert convo == [
        {"role": "user", "content": "what's on tonight"},
        {"role": "assistant", "content": "Social tennis at 19:30."},
    ]


def test_window_excludes_stale_thread(bridged):
    admin_bot, db = bridged
    jid = "grp@g.us"
    old = datetime(2026, 5, 15, 8, 0, 0)
    _insert(db, jid, [
        ("boris old request", old.isoformat()),
        (admin_bot.BOT_REPLY_PREFIX + "old reply",
         (old + timedelta(seconds=5)).isoformat()),
    ])
    # 30 min later — outside the 15-min window.
    now = (old + timedelta(minutes=30)).isoformat()
    assert admin_bot._recent_conversation(jid, now) == []


def test_starts_with_user_turn(bridged):
    admin_bot, db = bridged
    jid = "grp@g.us"
    base = datetime(2026, 5, 15, 9, 0, 0)
    # Leading assistant message (e.g. a proactive kickoff post) must be
    # dropped so the list starts with a user turn.
    _insert(db, jid, [
        (admin_bot.BOT_REPLY_PREFIX + "Kickoff! Reply boris go ahead.",
         base.isoformat()),
        ("boris go ahead", (base + timedelta(seconds=30)).isoformat()),
    ])
    convo = admin_bot._recent_conversation(
        jid, (base + timedelta(minutes=1)).isoformat()
    )
    assert convo and convo[0]["role"] == "user"
    assert convo[0]["content"] == "go ahead"


def test_coalesces_consecutive_same_role(bridged):
    admin_bot, db = bridged
    jid = "grp@g.us"
    base = datetime(2026, 5, 15, 9, 0, 0)
    _insert(db, jid, [
        ("boris first part", base.isoformat()),
        ("boris and the rest", (base + timedelta(seconds=5)).isoformat()),
        (admin_bot.BOT_REPLY_PREFIX + "got it",
         (base + timedelta(seconds=10)).isoformat()),
    ])
    convo = admin_bot._recent_conversation(
        jid, (base + timedelta(minutes=1)).isoformat()
    )
    assert convo == [
        {"role": "user", "content": "first part\n\nand the rest"},
        {"role": "assistant", "content": "got it"},
    ]


def test_empty_when_no_history(bridged):
    admin_bot, db = bridged
    assert admin_bot._recent_conversation(
        "grp@g.us", datetime(2026, 5, 15, 9, 0, 0).isoformat()
    ) == []
