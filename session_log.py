"""Append human-readable mirrors of pairing plans into the Google Sheet.

The authoritative history is still ``history.json`` — it's what
``pairings.py`` reads to avoid repeat pairings. The sheet mirrors are for
admins to browse: one row per session in ``Session log`` and one row per
court per rotation in ``Pair log``.

Headers are written on first append so the tabs don't need pre-populating.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

PROJECT_ROOT = Path(__file__).parent
DEFAULT_CREDENTIALS_PATH = PROJECT_ROOT / "gcp_service_account.json"

SESSION_LOG_TAB = "Session log"
PAIR_LOG_TAB = "Pair log"

SESSION_LOG_HEADERS = [
    "Date",
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


def _open_tab(tab_name: str):
    """Open (creds + sheet + worksheet) for the given tab. Ensures header row."""
    import gspread

    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    if not sheet_id:
        raise RuntimeError("GOOGLE_SHEET_ID missing from .env")
    gc = gspread.service_account(filename=str(DEFAULT_CREDENTIALS_PATH))
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(tab_name)
    # If row 1 is empty, write headers.
    headers_for = {
        SESSION_LOG_TAB: SESSION_LOG_HEADERS,
        PAIR_LOG_TAB: PAIR_LOG_HEADERS,
    }
    if not ws.row_values(1):
        ws.update("A1", [headers_for[tab_name]], value_input_option="USER_ENTERED")
    return ws


def _display(plan: dict, full_name: str) -> str:
    """Return the plan's short display name for a full name (fallback: full)."""
    return plan.get("display_names", {}).get(full_name, full_name)


def log_plan(plan: dict) -> dict:
    """Append a Session-log row and one Pair-log row per court-per-rotation.

    Accepts the ``plan.to_dict()`` form produced by ``pairings.make_plan``.
    Returns ``{"session_rows_appended": 1, "pair_rows_appended": N}``.
    """
    date = plan.get("date", "")
    attendees = plan.get("attendees", []) or []
    court_labels = plan.get("court_labels") or []
    num_rotations = plan.get("num_rotations", "")
    rotations = plan.get("rotations", []) or []
    strategy = plan.get("strategy", "")
    notes = plan.get("notes", "")

    start_time = rotations[0].get("start_time", "") if rotations else ""
    display_names = plan.get("display_names", {}) or {}
    attendee_display = ", ".join(display_names.get(a, a) for a in attendees)

    session_ws = _open_tab(SESSION_LOG_TAB)
    session_ws.append_row(
        [
            date,
            len(attendees),
            len(court_labels),
            ", ".join(str(x) for x in court_labels),
            num_rotations,
            start_time,
            attendee_display,
            strategy,
            notes,
        ],
        value_input_option="USER_ENTERED",
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
        pair_ws = _open_tab(PAIR_LOG_TAB)
        pair_ws.append_rows(pair_rows, value_input_option="USER_ENTERED")

    return {
        "session_rows_appended": 1,
        "pair_rows_appended": len(pair_rows),
    }
