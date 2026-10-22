"""The delayed 'working on it…' placeholder must be suppressed once
real output has already been delivered to the channel for the current
command (e.g. a tool that posts the rules PDF mid-run).
"""
from __future__ import annotations

import threading

import pytest

import bot_config


@pytest.fixture
def bot(monkeypatch):
    import bot_bridge

    class _Resp:
        status_code = 200
        text = "ok"

    monkeypatch.setattr(
        bot_bridge.requests, "post", lambda *a, **k: _Resp()
    )
    yield bot_bridge
    bot_config._ACTIVE_DONE["ev"] = None


def test_signal_noop_when_no_active_command(bot):
    bot_config._ACTIVE_DONE["ev"] = None
    bot_config._signal_output_delivered()  # must not raise


def test_send_to_group_suppresses_placeholder(bot):
    ev = threading.Event()
    bot_config._ACTIVE_DONE["ev"] = ev
    assert not ev.is_set()
    assert bot.send_to_group("grp@g.us", "hello") is True
    # A tool delivering output flips the in-flight done event, so the
    # working-on-it timer's `if done.is_set(): return` guard fires.
    assert ev.is_set()


def test_send_doc_suppresses_placeholder(bot):
    ev = threading.Event()
    bot_config._ACTIVE_DONE["ev"] = ev
    assert bot.send_doc_to_group("grp@g.us", "/tmp/x.pdf", "cap") is True
    assert ev.is_set()


def test_failed_send_does_not_suppress(bot, monkeypatch):
    class _Bad:
        status_code = 500
        text = "boom"

    monkeypatch.setattr(bot.requests, "post", lambda *a, **k: _Bad())
    ev = threading.Event()
    bot_config._ACTIVE_DONE["ev"] = ev
    assert bot.send_to_group("grp@g.us", "hi") is False
    # Send failed → no real output delivered → placeholder still wanted.
    assert not ev.is_set()
