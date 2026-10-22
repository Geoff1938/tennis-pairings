"""Talking to the WhatsApp bridge: sending messages and files,
reading new commands and the recent conversation from its
SQLite store.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from bot_config import (
    ADMIN_GROUP_NAMES,
    BOT_REPLY_PREFIX,
    BOT_TRIGGER_PATTERN,
    BRIDGE_DB,
    BRIDGE_URL,
    CANARY_MARKER,
    HISTORY_MAX_MESSAGES,
    HISTORY_WINDOW_MINUTES,
    WORKING_ON_IT_TEXT,
    _signal_output_delivered,
)


def find_admin_group_jids() -> dict[str, str]:
    """Resolve every name in ``ADMIN_GROUP_NAMES`` to its WhatsApp JID.

    Returns a dict ``{name: jid}``. Names that aren't found yet (the group
    has been created but no message has reached the bridge) are skipped
    with a warning — the bot will still serve the others.
    """
    if not BRIDGE_DB.exists():
        raise SystemExit(f"Bridge DB not found at {BRIDGE_DB}. Is the bridge running?")
    found: dict[str, str] = {}
    with sqlite3.connect(BRIDGE_DB) as conn:
        for name in ADMIN_GROUP_NAMES:
            row = conn.execute(
                "SELECT jid FROM chats WHERE LOWER(name) = LOWER(?) LIMIT 1",
                (name,),
            ).fetchone()
            if row:
                found[name] = row[0]
            else:
                print(
                    f"[warn] admin group {name!r} not found in bridge DB — "
                    "send any message in it from your phone to register it."
                )
    if not found:
        raise SystemExit(
            f"None of {ADMIN_GROUP_NAMES} found. Create at least one of "
            "them on WhatsApp, send a message in it, and restart."
        )
    return found




SEND_RETRY_DELAY_SECONDS = 3


def _post_to_bridge(payload: dict, timeout: int, label: str) -> bool:
    """POST ``payload`` to the bridge's /send, retrying once on a
    connection error or timeout. Never raises: a bridge outage must
    not crash the poll loop (which would lose queued commands) or
    abort a tool half-way (e.g. commit after the history write)."""
    for attempt in (1, 2):
        try:
            r = requests.post(
                f"{BRIDGE_URL}/send", json=payload, timeout=timeout,
            )
        except requests.RequestException as e:
            print(f"{label} error (attempt {attempt}): {e}", file=sys.stderr)
            if attempt == 1:
                time.sleep(SEND_RETRY_DELAY_SECONDS)
            continue
        if r.status_code != 200:
            print(f"{label} failed: {r.status_code} {r.text}", file=sys.stderr)
            return False
        _signal_output_delivered()
        return True
    return False


def send_to_group(group_jid: str, text: str) -> bool:
    """Send ``text`` to the WhatsApp group ``group_jid`` via the bridge."""
    if not text:
        text = "(no reply from bot)"
    return _post_to_bridge(
        {"recipient": group_jid, "message": text}, 15, "send",
    )


def send_doc_to_group(
    group_jid: str, file_path: str | Path, caption: str = "",
) -> bool:
    """Send a file (e.g. the final-pairings .docx) to ``group_jid``.

    The bridge reads ``file_path`` from disk so it must be an absolute
    path on the machine the bridge is running on (the Pi).
    """
    fp = str(Path(file_path).resolve())
    return _post_to_bridge(
        {"recipient": group_jid, "message": caption, "media_path": fp},
        60, "send_doc",
    )




def fetch_triggered_messages(
    admin_jid: str, since_iso: str
) -> list[tuple[str, str, str, str]]:
    """Return (id, timestamp_iso, sender, command) for trigger-word messages
    after ``since_iso``. Boris's own replies (prefixed) are filtered out so
    they don't loop back via the bridge.
    """
    with sqlite3.connect(BRIDGE_DB) as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, sender, content
            FROM messages
            WHERE chat_jid = ? AND timestamp > ?
            ORDER BY timestamp ASC
            """,
            (admin_jid, since_iso),
        ).fetchall()
    out = []
    for msg_id, ts, sender, content in rows:
        if not isinstance(content, str) or not content.strip():
            continue
        if content.startswith(BOT_REPLY_PREFIX):
            continue
        if content.strip() == CANARY_MARKER:
            # Daily end-to-end canary — confirms the round trip
            # but should never reach the agent.
            continue
        m = BOT_TRIGGER_PATTERN.match(content)
        if not m:
            continue
        command = content[m.end():].strip()
        out.append((msg_id, ts, sender or "", command))
    return out


def _recent_conversation(group_jid: str, before_ts: str) -> list[dict]:
    """Reconstruct the recent Boris<->admin exchange for ``group_jid``
    as an Anthropic ``messages`` list (text-only, role-tagged), so the
    agent has multi-turn context instead of seeing each message in
    isolation.

    Includes only the Boris thread:
      * trigger-word commands addressed to Boris  → role ``user``
      * Boris's own substantive replies           → role ``assistant``
    Untriggered group chatter and the "Working on it…" filler are
    excluded. Bounded to the last ``HISTORY_MAX_MESSAGES`` turns within
    ``HISTORY_WINDOW_MINUTES`` before ``before_ts``; returns ``[]`` if
    nothing recent (a fresh conversation, no context to carry).

    The returned list is guaranteed to start with a ``user`` turn and
    to have no two consecutive same-role turns (Anthropic requires
    strict alternation; consecutive same-role turns are coalesced).
    """
    # The bridge stores timestamps as "YYYY-MM-DD HH:MM:SS+01:00" (space,
    # not "T") and the query compares strings, so the cutoff must use
    # the same separator — a "T" cutoff sorts after every same-day row.
    sep = "T" if "T" in str(before_ts) else " "
    try:
        cutoff = (
            datetime.fromisoformat(before_ts)
            - timedelta(minutes=HISTORY_WINDOW_MINUTES)
        ).isoformat(sep=sep)
    except (ValueError, TypeError):
        return []
    try:
        with sqlite3.connect(BRIDGE_DB) as conn:
            rows = conn.execute(
                """
                SELECT timestamp, content
                FROM messages
                WHERE chat_jid = ? AND timestamp < ? AND timestamp >= ?
                ORDER BY timestamp ASC
                """,
                (group_jid, before_ts, cutoff),
            ).fetchall()
    except Exception:
        return []
    turns: list[dict] = []
    for _ts, content in rows:
        if not isinstance(content, str) or not content.strip():
            continue
        if content.strip() == CANARY_MARKER:
            continue  # daily canary — not a real turn
        if content.startswith(BOT_REPLY_PREFIX):
            body = content[len(BOT_REPLY_PREFIX):].strip()
            if not body or body == WORKING_ON_IT_TEXT:
                continue  # filler — not a real turn
            turns.append({"role": "assistant", "content": body})
            continue
        m = BOT_TRIGGER_PATTERN.match(content)
        if not m:
            continue  # untriggered chatter — not part of the thread
        cmd = content[m.end():].strip()
        if cmd:
            turns.append({"role": "user", "content": cmd})
    turns = turns[-HISTORY_MAX_MESSAGES:]
    # Must start with a user turn.
    while turns and turns[0]["role"] == "assistant":
        turns.pop(0)
    # Coalesce consecutive same-role turns.
    coalesced: list[dict] = []
    for t in turns:
        if coalesced and coalesced[-1]["role"] == t["role"]:
            coalesced[-1]["content"] += "\n\n" + t["content"]
        else:
            coalesced.append(dict(t))
    return coalesced
