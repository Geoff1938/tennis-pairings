"""Append human-readable mirrors of pairing plans to two CSV files.

The authoritative history is still ``history.json`` — it's what
``pairings.py`` reads to avoid repeat pairings. The CSVs are a record for
people to browse (they open in Excel): one row per session in
``session_log.csv`` and one row per court per rotation in ``pair_log.csv``.
Both live in the repo root (git-ignored) and are in the nightly backup.

Until Nov 2026 these were the "Session log" / "Pair log" tabs of a
Google Sheet; ``scripts/migrate_sheet_to_local.py`` copied them across.
"""

from __future__ import annotations

import csv
import io
import threading
from pathlib import Path
from typing import Any

from atomic_io import atomic_write_text

PROJECT_ROOT = Path(__file__).parent
SESSION_LOG_PATH = PROJECT_ROOT / "session_log.csv"
PAIR_LOG_PATH = PROJECT_ROOT / "pair_log.csv"

_LOCK = threading.Lock()

SESSION_LOG_HEADERS = [
    "Date",
    "Session type",
    "Attendees",
    "Courts",
    "Court labels",
    "Rotations",
    "Start time",
    "Attendee names",
    "Strategy",
    "Notes",
]
PAIR_LOG_HEADERS = [
    "Date",
    "Rotation",
    "Start time",
    "Court",
    "Mode",
    "Pair A",
    "Pair B",
    "Sit-outs",
]


def _read_rows(path: Path) -> list[list[str]]:
    """All rows of a CSV, header included ([] if the file is missing)."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [row for row in csv.reader(f)]


def _write_rows(path: Path, rows: list[list[Any]]) -> None:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\r\n").writerows(rows)
    # BOM so Excel reads names with accents / curly apostrophes correctly.
    atomic_write_text(path, "﻿" + buf.getvalue())


def append_rows(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    """Append ``rows`` to the CSV at ``path``, writing ``headers`` first
    if the file is new or empty."""
    with _LOCK:
        if not path.exists() or path.stat().st_size == 0:
            _write_rows(path, [list(headers)])
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\r\n").writerows(rows)


def _display(plan: dict, full_name: str) -> str:
    """Return the plan's short display name for a full name (fallback: full)."""
    return plan.get("display_names", {}).get(full_name, full_name)


def log_plan(plan: dict) -> dict:
    """Append a Session-log row and one Pair-log row per court-per-rotation.

    Accepts the ``plan.to_dict()`` form produced by ``pairings.make_plan``.
    Returns ``{"session_rows_appended": 1, "pair_rows_appended": N}``.
    """
    date = plan.get("date", "")
    session_type = plan.get("session_type", "") or ""
    attendees = plan.get("attendees", []) or []
    court_labels = plan.get("court_labels") or []
    num_rotations = plan.get("num_rotations", "")
    rotations = plan.get("rotations", []) or []
    strategy = plan.get("strategy", "")
    notes = plan.get("notes", "")

    start_time = rotations[0].get("start_time", "") if rotations else ""
    display_names = plan.get("display_names", {}) or {}
    attendee_display = ", ".join(display_names.get(a, a) for a in attendees)

    append_rows(
        SESSION_LOG_PATH,
        SESSION_LOG_HEADERS,
        [[
            date,
            session_type,
            len(attendees),
            len(court_labels),
            ", ".join(str(x) for x in court_labels),
            num_rotations,
            start_time,
            attendee_display,
            strategy,
            notes,
        ]],
    )

    pair_rows: list[list[Any]] = []
    for rot in rotations:
        rot_num = rot.get("rotation_num", "")
        rot_start = rot.get("start_time", "")
        sit_outs = rot.get("sit_outs", []) or []
        sit_out_str = ", ".join(display_names.get(p, p) for p in sit_outs)
        for court in rot.get("courts", []):
            mode = court.get("mode", "doubles")
            pairs = court.get("pairs", []) or []
            if mode == "doubles":
                pair_a = pairs[0] if len(pairs) > 0 else ["", ""]
                pair_b = pairs[1] if len(pairs) > 1 else ["", ""]
                pair_a_str = f"{_display(plan, pair_a[0])} & {_display(plan, pair_a[1])}"
                pair_b_str = f"{_display(plan, pair_b[0])} & {_display(plan, pair_b[1])}"
            else:
                # Singles: single matchup in pairs[0], format as "A v B"
                match = pairs[0] if pairs else ["", ""]
                pair_a_str = (
                    f"{_display(plan, match[0])} v {_display(plan, match[1])}"
                )
                pair_b_str = ""
            pair_rows.append(
                [
                    date,
                    rot_num,
                    rot_start,
                    court.get("court_label", ""),
                    mode,
                    pair_a_str,
                    pair_b_str,
                    sit_out_str,
                ]
            )

    if pair_rows:
        append_rows(PAIR_LOG_PATH, PAIR_LOG_HEADERS, pair_rows)

    return {
        "session_rows_appended": 1,
        "pair_rows_appended": len(pair_rows),
    }


def unlog_plan(
    date: str, session_rows: int, pair_rows: int,
) -> dict:
    """Best-effort reversal of a ``log_plan`` append (undo-commit).

    Deletes the *trailing* Session-log row and the trailing
    ``pair_rows`` Pair-log rows — but ONLY if their Date column
    matches ``date``. This is safe because ``log_plan`` always
    appends at the end and an undo happens right after an inadvertent
    commit (nothing else appended in between). If the trailing rows
    don't match (something else was appended, or counts are
    off), it leaves them intact and reports a warning rather than
    risk eating unrelated rows.

    Returns ``{"session_rows_deleted", "pair_rows_deleted",
    "warnings": [...]}``.
    """
    result: dict[str, Any] = {
        "session_rows_deleted": 0,
        "pair_rows_deleted": 0,
        "warnings": [],
    }
    with _LOCK:
        if session_rows:
            rows = _read_rows(SESSION_LOG_PATH)
            if len(rows) >= 2 and rows[-1] and rows[-1][0] == date:
                _write_rows(SESSION_LOG_PATH, rows[:-1])
                result["session_rows_deleted"] = 1
            else:
                result["warnings"].append(
                    "Session-log trailing row date did not match; "
                    "left it intact."
                )
        if pair_rows:
            rows = _read_rows(PAIR_LOG_PATH)
            tail = rows[-pair_rows:]
            if len(rows) - pair_rows >= 1 and all(
                r and r[0] == date for r in tail
            ):
                _write_rows(PAIR_LOG_PATH, rows[:-pair_rows])
                result["pair_rows_deleted"] = pair_rows
            else:
                result["warnings"].append(
                    "Pair-log trailing rows did not all match the date; "
                    "left them intact."
                )
    return result
