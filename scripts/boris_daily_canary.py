#!/usr/bin/env python3
"""Daily end-to-end canary for the Boris pipeline.

Posts a known marker ("Boris daily test msg") to the Boris admin
group via the bridge's ``/api/send`` endpoint, then inspects the
response.

The bridge's ``sendWhatsAppMessage`` calls whatsmeow's
``client.SendMessage(...)`` which returns ONLY after WhatsApp's
servers have acknowledged receipt over the websocket. The bridge
then responds with ``{"success": true, ...}`` on HTTP 200, or
``{"success": false, "message": "Error sending message: ..."}``
on HTTP 500. So a 200 + ``success: true`` confirms:

  * bot's network path to the bridge HTTP server is intact
  * bridge process is healthy
  * bridge has a live websocket to WhatsApp (the case that broke
    in Jun 2026 — "Client outdated 405" — would surface here as
    a 500 because whatsmeow's SendMessage fails immediately when
    the socket is closed)
  * WhatsApp acknowledged the message for delivery

That covers the 8-day silent-failure mode that prompted this
canary. (Polling messages.db for an echo doesn't work — WhatsApp
doesn't echo own-originated messages back to the originator
device.)

Run by ``boris-canary.service`` via ``boris-canary.timer`` once a
day. Manual invocation:
    ~/projects/tennis-pairings/.venv/bin/python \
        ~/projects/tennis-pairings/scripts/boris_daily_canary.py

The bot recognises the exact marker string and silently drops it —
no LLM call, no chat-visible reply. See ``CANARY_MARKER`` in
admin_bot.py.

On failure the canary ALSO tries to send an alert email (if
``SMTP_USER`` / ``SMTP_PASSWORD`` / ``ALERT_EMAIL`` are configured
in the environment — typically via the systemd unit's
``EnvironmentFile=...env`` line). Email send is best-effort: a
failure to email is logged to stderr but does NOT change the
canary's exit code (the systemctl-failed state is the primary
signal; email is a convenience nudge).

Exit codes:
  0  send acknowledged by WhatsApp
  1  send call failed (bridge unreachable, HTTP error, or
     whatsmeow reported an error from WhatsApp)
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request

BRIDGE_URL = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8080")
BORIS_GROUP_JID = os.environ.get(
    "BORIS_GROUP_JID", "120363408518957244@g.us"
)
CANARY_MARKER = "Boris daily test msg"
# whatsmeow's SendMessage typically returns in a couple of seconds
# on a healthy link. 15s gives headroom for the occasional sluggish
# minute without dragging a real failure out.
SEND_TIMEOUT_SECONDS = float(os.environ.get("SEND_TIMEOUT_SECONDS", "15"))

# Email-alert config. Names match the admin_bot SMTP block so a
# single block of env vars in .env serves both. All optional —
# without these set, the canary just exits non-zero and surfaces
# via `systemctl --user --failed`.
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
ALERT_EMAIL = os.environ.get("ALERT_EMAIL", "")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(msg: str) -> None:
    print(f"{_now_iso()} [canary] {msg}", flush=True)


def send_marker() -> tuple[int, dict]:
    """POST the marker. Returns ``(http_status, parsed_body)``.

    Non-200 still returns a status + body so the caller can log the
    bridge's error string (e.g. "Error sending message: client not
    connected") rather than just a generic HTTPError.
    """
    payload = json.dumps({
        "recipient": BORIS_GROUP_JID,
        "message": CANARY_MARKER,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{BRIDGE_URL}/api/send",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            req, timeout=SEND_TIMEOUT_SECONDS,
        ) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        # Bridge gives 500 + JSON body on whatsmeow failures.
        try:
            body = json.loads(e.read() or b"{}")
        except (ValueError, OSError):
            body = {}
        return e.code, body


def send_alert_email(subject: str, body: str) -> None:
    """Best-effort SMTP alert. Silent no-op when SMTP isn't
    configured. Any send failure is logged to stderr but does NOT
    propagate — the canary's exit code is the primary failure
    signal (systemctl-failed), email is a nudge on top."""
    if not (SMTP_USER and SMTP_PASSWORD and ALERT_EMAIL):
        log("email alert skipped (SMTP not configured)")
        return
    # Local imports keep the script start-up cheap on the happy
    # path where no email is sent.
    import smtplib
    from email.message import EmailMessage
    try:
        msg = EmailMessage()
        msg["From"] = SMTP_USER
        msg["To"] = ALERT_EMAIL
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASSWORD)
            s.send_message(msg)
        log(f"alert email sent to {ALERT_EMAIL}")
    except Exception as e:
        log(f"alert email send FAILED: {type(e).__name__}: {e}")


def _alert_on_failure(headline: str, detail: str) -> None:
    """Build a short, human-readable alert and try to email it.
    Body carries the headline + detail + host + UTC timestamp so
    a forwarded copy is self-describing."""
    host = socket.gethostname()
    body = (
        f"Boris daily canary FAILED on {host} at {_now_iso()}.\n\n"
        f"{headline}\n\n"
        f"Detail:\n{detail}\n\n"
        f"What this means: the daily round-trip test that confirms "
        "the bridge can reach WhatsApp's servers did not succeed. "
        "Check the Pi:\n"
        "  ssh pi-1\n"
        "  journalctl --user-unit=boris-canary -n 30 --no-pager\n"
        "  journalctl --user-unit=boris-bridge -n 30 --no-pager\n"
        "and see RESTART.txt 'Auto-recovery story' for the typical "
        "fix path (most common cause: whatsmeow client outdated, "
        "needs rebuild — see watchdog 'bridge LOCKED OUT' section)."
    )
    send_alert_email(
        subject="[Boris] daily canary FAILED on pi-1", body=body,
    )


def main() -> int:
    start = time.monotonic()
    try:
        status, body = send_marker()
    except (urllib.error.URLError, OSError) as e:
        log(f"send FAILED at HTTP layer: {e}")
        _alert_on_failure(
            headline=f"Bridge HTTP layer unreachable: {e}",
            detail=(
                f"Tried to POST to {BRIDGE_URL}/api/send and got "
                f"{type(e).__name__}. This usually means the bridge "
                "process is down or HTTP isn't responding."
            ),
        )
        return 1
    elapsed = int(time.monotonic() - start)
    if status == 200 and body.get("success") is True:
        log(f"OK — WhatsApp ACK'd in {elapsed}s "
            f"({body.get('message', 'sent')})")
        return 0
    log(f"FAIL — HTTP {status}, body={body!r} (after {elapsed}s)")
    _alert_on_failure(
        headline=f"Bridge replied HTTP {status} (success=false)",
        detail=(
            f"Body: {body!r}\n"
            f"Elapsed: {elapsed}s.\n"
            "Bridge accepted the request but whatsmeow's SendMessage "
            "did not succeed. Most often this means the WhatsApp "
            "websocket is closed (client outdated / rejected by "
            "server / lost network)."
        ),
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
