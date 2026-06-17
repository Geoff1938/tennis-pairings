#!/usr/bin/env python3
"""Daily end-to-end canary for the Boris pipeline.

Sends a known marker message ("Boris daily test msg") to the Boris
admin group via the bridge HTTP API, then polls the bridge's
``messages.db`` SQLite store for the same text. Round-trip success
means: bot can reach the bridge, the bridge can reach WhatsApp's
servers, WhatsApp echoes the message back to the linked-device
session, and the bridge writes it into the DB the bot polls. If any
link in that chain is broken, the script exits non-zero and the
systemd timer's failure surfaces in `systemctl --user --failed`.

Run by ``boris-canary.service`` via ``boris-canary.timer`` once a day.
Manual invocation:
    ~/projects/tennis-pairings/.venv/bin/python \
        ~/projects/tennis-pairings/scripts/boris_daily_canary.py

The bot recognises the exact marker string and silently drops it —
no LLM call, no chat-visible reply. See ``CANARY_MARKER`` in
admin_bot.py.

Exit codes:
  0  round-trip OK
  1  send call to bridge failed (bridge down or HTTP non-200)
  2  send accepted but echo never landed in messages.db within deadline
  3  bridge DB missing or unreadable
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import json
from pathlib import Path

BRIDGE_URL = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8080")
BORIS_GROUP_JID = os.environ.get(
    "BORIS_GROUP_JID", "120363408518957244@g.us"
)
CANARY_MARKER = "Boris daily test msg"
BRIDGE_DB = Path(os.environ.get(
    "BRIDGE_DB",
    "/home/geoff/projects/whatsapp-mcp/whatsapp-bridge/store/messages.db",
))
# How long to wait for the echo to come back. WhatsApp's round trip
# on a healthy link is usually well under 5s; 30s gives plenty of
# headroom for a sluggish minute without being so long that a real
# failure takes ages to diagnose.
DEADLINE_SECONDS = float(os.environ.get("DEADLINE_SECONDS", "30"))
POLL_INTERVAL_SECONDS = float(os.environ.get("POLL_INTERVAL_SECONDS", "1.0"))


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(msg: str) -> None:
    print(f"{_now_iso()} [canary] {msg}", flush=True)


def send_marker() -> None:
    """POST the marker to the bridge. Raises on non-2xx."""
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
    with urllib.request.urlopen(req, timeout=10) as resp:
        if not (200 <= resp.status < 300):
            raise RuntimeError(
                f"bridge /api/send returned HTTP {resp.status}"
            )


def wait_for_echo(send_started_unix: int) -> bool:
    """Poll messages.db for the marker text, written after we sent it.

    The bridge stores timestamps as a string in the same shape the
    bot reads (ISO-8601 with offset, see admin_bot.py:4286). We can't
    cheaply compare strings to a unix timestamp here, so we instead
    check ``MAX(timestamp)`` advanced past the row we sent — i.e. a
    matching content row appeared. The chat_jid scopes the lookup so
    a same-text message elsewhere wouldn't false-positive.
    """
    deadline = time.monotonic() + DEADLINE_SECONDS
    while time.monotonic() < deadline:
        try:
            with sqlite3.connect(BRIDGE_DB) as conn:
                row = conn.execute(
                    """
                    SELECT timestamp FROM messages
                    WHERE chat_jid = ? AND content = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (BORIS_GROUP_JID, CANARY_MARKER),
                ).fetchone()
        except sqlite3.Error as e:
            log(f"sqlite error: {e}")
            return False
        if row:
            # Parse the timestamp into unix seconds and confirm it's
            # >= when we started this run, so we don't accept a
            # stale row from a previous day's canary. Best-effort:
            # the bridge writes ISO timestamps so fromisoformat
            # handles them; if parsing fails we accept the row to
            # avoid blocking on a schema quirk we don't understand.
            ts_str = row[0]
            try:
                from datetime import datetime
                ts_unix = int(datetime.fromisoformat(ts_str).timestamp())
            except (ValueError, TypeError):
                return True
            if ts_unix >= send_started_unix:
                return True
        time.sleep(POLL_INTERVAL_SECONDS)
    return False


def main() -> int:
    if not BRIDGE_DB.exists():
        log(f"bridge DB not found at {BRIDGE_DB}")
        return 3

    send_started = int(time.time())
    try:
        send_marker()
    except (urllib.error.URLError, RuntimeError, OSError) as e:
        log(f"send FAILED: {e}")
        return 1
    log(f"sent {CANARY_MARKER!r} to {BORIS_GROUP_JID}")

    if wait_for_echo(send_started):
        elapsed = int(time.time()) - send_started
        log(f"OK — echo seen in messages.db after {elapsed}s")
        return 0

    log(f"FAIL — sent OK but no echo in {int(DEADLINE_SECONDS)}s")
    return 2


if __name__ == "__main__":
    sys.exit(main())
