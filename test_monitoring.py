"""Canary retries before alerting; the bot's heartbeat for the
watchdog."""
from __future__ import annotations

import importlib.util
import pathlib
import urllib.error

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "boris_daily_canary",
    pathlib.Path(__file__).parent / "scripts" / "boris_daily_canary.py",
)
canary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(canary)


@pytest.fixture
def fake_clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(canary.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        canary.time, "sleep", lambda s: now.__setitem__(0, now[0] + s))
    alerts = []
    monkeypatch.setattr(canary, "_alert_on_failure",
                        lambda **k: alerts.append(k))
    return alerts


def test_canary_retries_until_bridge_is_back(monkeypatch, fake_clock):
    # Just after boot: bridge unreachable, then not yet connected, then OK.
    results = iter([
        urllib.error.URLError("refused"),
        (500, {"success": False, "message": "not connected"}),
        (200, {"success": True}),
    ])

    def send():
        r = next(results)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(canary, "send_marker", send)
    assert canary.main() == 0
    assert fake_clock == []


def test_canary_alerts_after_retry_window(monkeypatch, fake_clock):
    calls = []

    def send():
        calls.append(1)
        return 500, {"success": False}

    monkeypatch.setattr(canary, "send_marker", send)
    assert canary.main() == 1
    assert len(fake_clock) == 1
    # Attempts at t = 0, 30, ..., 300: the whole window is used.
    assert len(calls) == 1 + canary.RETRY_WINDOW_SECONDS // \
        canary.RETRY_INTERVAL_SECONDS


def test_heartbeat_touch_never_raises(tmp_path, monkeypatch):
    import admin_bot

    hb = tmp_path / "hb"
    monkeypatch.setattr(admin_bot, "HEARTBEAT_PATH", hb)
    admin_bot._touch_heartbeat()
    assert hb.exists()
    monkeypatch.setattr(admin_bot, "HEARTBEAT_PATH",
                        tmp_path / "missing-dir" / "hb")
    admin_bot._touch_heartbeat()  # logs, doesn't raise
