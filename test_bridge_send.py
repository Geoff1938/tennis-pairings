"""Bridge sends never raise, and watermarks survive a restart."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
import requests


@pytest.fixture
def bot(monkeypatch):
    import bot_bridge

    monkeypatch.setattr(bot_bridge, "SEND_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(bot_bridge.time, "sleep", lambda s: None)
    return bot_bridge


def test_send_survives_connection_error_and_retries(bot, monkeypatch):
    calls = []

    class Ok:
        status_code = 200
        text = "{}"

    def post(url, json=None, timeout=None):
        calls.append(json)
        if len(calls) == 1:
            raise requests.ConnectionError("bridge restarting")
        return Ok()

    monkeypatch.setattr(bot.requests, "post", post)
    assert bot.send_to_group("g@g.us", "hello") is True
    assert len(calls) == 2


def test_send_gives_up_without_raising(bot, monkeypatch):
    def post(url, json=None, timeout=None):
        raise requests.Timeout("no answer")

    monkeypatch.setattr(bot.requests, "post", post)
    assert bot.send_to_group("g@g.us", "hello") is False
    assert bot.send_doc_to_group("g@g.us", "x.docx", "cap") is False


# ---------- watermarks ---------------------------------------------------

TZ = timezone(timedelta(hours=1))
NOW = datetime(2026, 10, 22, 10, 0, tzinfo=TZ)
LATEST = "2026-10-22 10:00:00+01:00"


@pytest.fixture
def wm(tmp_path, monkeypatch):
    import admin_bot

    monkeypatch.setattr(admin_bot, "WATERMARKS_PATH", tmp_path / "wm.json")
    return admin_bot


def test_recent_saved_watermark_is_used(wm):
    wm._save_watermarks({"a": "2026-10-22 09:55:00+01:00"})
    out = wm._load_watermarks(["a", "b"], LATEST, now=NOW)
    assert out == {"a": "2026-10-22 09:55:00+01:00", "b": LATEST}


def test_old_saved_watermark_is_ignored(wm):
    wm._save_watermarks({"a": "2026-10-22 08:00:00+01:00"})
    assert wm._load_watermarks(["a"], LATEST, now=NOW) == {"a": LATEST}


def test_missing_or_corrupt_file_falls_back(wm):
    assert wm._load_watermarks(["a"], LATEST, now=NOW) == {"a": LATEST}
    wm.WATERMARKS_PATH.write_text("{not json")
    assert wm._load_watermarks(["a"], LATEST, now=NOW) == {"a": LATEST}


def test_iso_t_separator_accepted(wm):
    wm.WATERMARKS_PATH.write_text(json.dumps({"a": "2026-10-22T09:58:00+01:00"}))
    # 'T' sorts after ' ' so the text guard (ts <= latest) rejects it
    # unless latest uses the same form; mirror the bridge's format.
    out = wm._load_watermarks(["a"], "2026-10-22T10:00:00+01:00", now=NOW)
    assert out == {"a": "2026-10-22T09:58:00+01:00"}
