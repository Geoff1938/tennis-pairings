"""Email alerts for Boris errors, rate-limited per error kind.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import bot_config  # noqa: F401  (loads .env before SMTP_* are read)


# Email-alert config. All optional — without these env vars set, bot
# errors get logged to journal only. Gmail SMTP requires an "app
# password" (myaccount.google.com → Security → 2-Step Verification →
# App passwords); plain account passwords have been disallowed since
# 2022. ALERT_RATE_LIMIT_SECONDS bounds per-key alert frequency so a
# sustained Anthropic outage pages once and stays quiet for the
# window instead of mailing every retry.
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
ALERT_EMAIL = os.environ.get("ALERT_EMAIL", "")
ALERT_RATE_LIMIT_SECONDS = float(
    os.environ.get("ALERT_RATE_LIMIT_SECONDS", "300")
)
_alert_last_sent: dict[str, float] = {}
_alert_lock = threading.Lock()


def _send_alert_email(subject: str, body: str) -> None:
    """Best-effort SMTP send. No-op when SMTP isn't configured; any
    send failure logs to stderr and swallows — we're already on an
    error path and shouldn't mask the real issue."""
    if not (SMTP_USER and SMTP_PASSWORD and ALERT_EMAIL):
        return
    import smtplib
    from email.message import EmailMessage
    try:
        msg = EmailMessage()
        msg["From"] = SMTP_USER
        msg["To"] = ALERT_EMAIL
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASSWORD)
            s.send_message(msg)
    except Exception as exc:
        print(f"[alert] email send failed: {exc!r}", file=sys.stderr)


def _alert_throttled(key: str, subject: str, body: str) -> None:
    """Send at most one alert per ``key`` per ``ALERT_RATE_LIMIT_SECONDS``.
    Lets a sustained Anthropic incident mail us once and stay quiet."""
    now = time.monotonic()
    with _alert_lock:
        last = _alert_last_sent.get(key, 0.0)
        if now - last < ALERT_RATE_LIMIT_SECONDS:
            print(f"[alert] throttled ({key})", file=sys.stderr)
            return
        _alert_last_sent[key] = now
    print(f"[alert] sending: {subject}", file=sys.stderr)
    _send_alert_email(subject, body)
