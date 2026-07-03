"""Tests for the error-handling additions in admin_bot:
_format_bot_error, _alert_throttled, _create_with_model_fallback.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------- _format_bot_error -------------------------------------------


def test_format_bot_error_anthropic_500_returns_friendly_message():
    """5xx from Anthropic gets the "transient, try again" text — no
    raw stack details leaked to WhatsApp."""
    import admin_bot
    from anthropic import APIStatusError

    e = APIStatusError(
        "boom",
        response=MagicMock(status_code=500),
        body={"type": "error"},
    )
    reply, key = admin_bot._format_bot_error(e)
    assert "transient" in reply.lower()
    assert "try again" in reply.lower()
    # Sanity: the raw "Internal server error" / "api_error" / req_id
    # MUST NOT appear in the reply.
    assert "api_error" not in reply
    assert "request_id" not in reply.lower()
    assert key == "anthropic_500"


def test_format_bot_error_anthropic_429_returns_rate_limit_message():
    """429 gets a separate, rate-limit-flavoured message."""
    import admin_bot
    from anthropic import APIStatusError

    e = APIStatusError(
        "rate limited",
        response=MagicMock(status_code=429),
        body={"type": "error"},
    )
    reply, key = admin_bot._format_bot_error(e)
    assert "rate limit" in reply.lower()
    assert key == "anthropic_429"


def test_format_bot_error_unknown_keeps_raw_detail():
    """A non-API exception keeps the raw text — admin needs the detail
    to diagnose from WhatsApp without pulling logs."""
    import admin_bot

    e = ValueError("roster column missing")
    reply, key = admin_bot._format_bot_error(e)
    assert "ValueError" in reply
    assert "roster column missing" in reply
    assert key.startswith("unknown_")


# ---------- _alert_throttled --------------------------------------------


def test_alert_throttled_first_fires_second_throttles(monkeypatch):
    """First alert with a given key goes through; immediate second
    with the same key is suppressed."""
    import admin_bot

    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        admin_bot, "_send_alert_email",
        lambda subject, body: sent.append((subject, body)),
    )
    # Clear any state from prior tests.
    admin_bot._alert_last_sent.clear()

    admin_bot._alert_throttled("test_key", "subj1", "body1")
    admin_bot._alert_throttled("test_key", "subj2", "body2")

    assert len(sent) == 1
    assert sent[0] == ("subj1", "body1")


def test_alert_throttled_different_keys_both_fire(monkeypatch):
    """Different keys are independent buckets."""
    import admin_bot

    sent: list[str] = []
    monkeypatch.setattr(
        admin_bot, "_send_alert_email",
        lambda subject, body: sent.append(subject),
    )
    admin_bot._alert_last_sent.clear()

    admin_bot._alert_throttled("key_a", "subj-a", "body")
    admin_bot._alert_throttled("key_b", "subj-b", "body")

    assert sent == ["subj-a", "subj-b"]


def test_alert_throttled_fires_again_after_window(monkeypatch):
    """After the rate-limit window expires, the same key fires again."""
    import admin_bot

    sent: list[str] = []
    monkeypatch.setattr(
        admin_bot, "_send_alert_email",
        lambda subject, body: sent.append(subject),
    )
    monkeypatch.setattr(admin_bot, "ALERT_RATE_LIMIT_SECONDS", 60.0)
    admin_bot._alert_last_sent.clear()

    t = {"now": 1000.0}
    monkeypatch.setattr(admin_bot.time, "monotonic", lambda: t["now"])

    admin_bot._alert_throttled("recurring", "first", "body")
    t["now"] += 30  # within window
    admin_bot._alert_throttled("recurring", "throttled", "body")
    t["now"] += 31  # now past 60s
    admin_bot._alert_throttled("recurring", "second", "body")

    assert sent == ["first", "second"]


# ---------- _send_alert_email -------------------------------------------


def test_send_alert_email_noop_without_smtp_config(monkeypatch):
    """No SMTP env vars set → silent no-op, no exception."""
    import admin_bot

    monkeypatch.setattr(admin_bot, "SMTP_USER", "")
    monkeypatch.setattr(admin_bot, "SMTP_PASSWORD", "")
    monkeypatch.setattr(admin_bot, "ALERT_EMAIL", "")
    # If this tried to connect to smtp.gmail.com it'd hang the test;
    # a no-op just returns.
    admin_bot._send_alert_email("anything", "body")


# ---------- _create_with_model_fallback ---------------------------------


def test_fallback_returns_primary_on_success():
    """No fallback when the primary returns cleanly."""
    import admin_bot

    client = MagicMock()
    client.messages.create.return_value = "primary-response"

    out = admin_bot._create_with_model_fallback(
        client, admin_bot.MODEL_SONNET, foo="bar",
    )
    assert out == "primary-response"
    assert client.messages.create.call_count == 1
    assert client.messages.create.call_args.kwargs["model"] \
        == admin_bot.MODEL_SONNET


def test_fallback_swaps_model_on_500():
    """Sonnet 500 → retry with Haiku, return Haiku's response."""
    import admin_bot
    from anthropic import APIStatusError

    client = MagicMock()
    err = APIStatusError(
        "down", response=MagicMock(status_code=500),
        body={"type": "error"},
    )
    # First call raises, second returns.
    client.messages.create.side_effect = [err, "haiku-response"]

    out = admin_bot._create_with_model_fallback(
        client, admin_bot.MODEL_SONNET, foo="bar",
    )
    assert out == "haiku-response"
    assert client.messages.create.call_count == 2
    assert client.messages.create.call_args_list[0].kwargs["model"] \
        == admin_bot.MODEL_SONNET
    assert client.messages.create.call_args_list[1].kwargs["model"] \
        == admin_bot.MODEL_HAIKU


def test_fallback_swaps_haiku_to_sonnet():
    """Same retry path the other direction — Haiku 503 → Sonnet."""
    import admin_bot
    from anthropic import APIStatusError

    client = MagicMock()
    err = APIStatusError(
        "down", response=MagicMock(status_code=503),
        body={"type": "error"},
    )
    client.messages.create.side_effect = [err, "sonnet-response"]

    admin_bot._create_with_model_fallback(
        client, admin_bot.MODEL_HAIKU, foo="bar",
    )
    assert client.messages.create.call_args_list[1].kwargs["model"] \
        == admin_bot.MODEL_SONNET


def test_fallback_does_not_retry_on_4xx():
    """4xx is our bug, not theirs — no point burning a retry that'll
    also fail. Re-raise immediately."""
    import admin_bot
    from anthropic import APIStatusError

    client = MagicMock()
    err = APIStatusError(
        "bad request", response=MagicMock(status_code=400),
        body={"type": "error"},
    )
    client.messages.create.side_effect = [err]

    with pytest.raises(APIStatusError):
        admin_bot._create_with_model_fallback(
            client, admin_bot.MODEL_SONNET,
        )
    assert client.messages.create.call_count == 1
