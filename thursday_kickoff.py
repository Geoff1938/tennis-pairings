"""Session kickoff for the Boris pairing workflow.

Drives Tuesday, Thursday and Saturday sessions via the
``session_types.SESSION_TYPES`` registry. The main entry point is
:func:`kickoff_session`, which:

1. Picks a SessionType (explicit or defaulted to "next scheduled by
   weekday").
2. Looks up the matching upcoming event in CourtReserve.
3. Auto-adds any unseen names to the roster (rating ``?``).
4. Calls ``start_tonight`` with the attendees / waitlist / courts and
   sets ``session_state.session_type`` and ``phase = "awaiting_extras"``.
5. Posts the structured "today's lineup + reply with extras" message
   to the session type's admin WhatsApp group.

``kickoff_thursday`` is a thin back-compat shim for out-of-tree
callers / scripts. ``replay_past_session_plan`` is a Python-only
helper for internal A/B testing — it returns a fresh PairingPlan
against a past session's attendees + courts without touching
session_state, history.json, the Sheet, or WhatsApp.

Failures (CR down, no event, bridge unreachable) are surfaced via the
return value AND a best-effort fallback message in the admin group;
the function never raises so the poll loop / bot tool can rely on it.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from session_types import SESSION_TYPES, SessionType, get as _session_get, resolve_next_session

# Bridge endpoint & well-known group names. Kept here (rather than
# imported from admin_bot) so this module stays independently runnable.
BRIDGE_URL = "http://localhost:8080/api"
TEST_GROUP_NAME = "Boris the tennis bot"
BOT_REPLY_PREFIX = "From Boris the tennis bot: "


# ---------- bridge sender ----------------------------------------------


def _resolve_group_jid(group_name: str) -> str | None:
    """Look up a group's JID via the bridge's SQLite chat table.

    Returns None if the bridge hasn't seen the group yet (e.g. no
    message ever reached it). Mirrors admin_bot's startup resolution.
    """
    import sqlite3

    bridge_db = (
        Path(__file__).parent
        / "whatsapp-mcp" / "whatsapp-bridge" / "store" / "messages.db"
    )
    if not bridge_db.exists():
        return None
    with sqlite3.connect(str(bridge_db)) as conn:
        row = conn.execute(
            "SELECT jid FROM chats WHERE name = ? LIMIT 1",
            (group_name,),
        ).fetchone()
    return row[0] if row else None


def _send_to_admin_group(
    text: str,
    *,
    admin_group_name: str,
    prefer_test_channel: bool = False,
    target_jid: str | None = None,
) -> bool:
    """Send a message to the session's admin group via the bridge.

    If ``target_jid`` is supplied it wins (used to keep dry runs in
    the channel that triggered them). Otherwise the session's
    ``admin_group_name`` is resolved to a JID via the bridge DB, with
    a fall-through to the test channel when ``prefer_test_channel`` is
    True or when the configured admin group hasn't been seen.
    """
    if target_jid:
        jid: str | None = target_jid
    else:
        primary = TEST_GROUP_NAME if prefer_test_channel else admin_group_name
        jid = _resolve_group_jid(primary)
        if not jid:
            other = admin_group_name if prefer_test_channel else TEST_GROUP_NAME
            jid = _resolve_group_jid(other)
    if not jid:
        return False
    try:
        r = requests.post(
            f"{BRIDGE_URL}/send",
            json={"recipient": jid, "message": BOT_REPLY_PREFIX + text},
            timeout=15,
        )
        return r.status_code == 200
    except Exception:
        return False


# ---------- formatting --------------------------------------------------


def format_kickoff_message(data: dict, session: SessionType) -> str:
    """Render the structured kickoff message from the data dict.

    ``data`` shape (produced by ``_collect_session_data``):
      * date_str: human-readable date
      * registrants: list[{name, rating, is_new}]
      * waitlist:   list[{name, rating, is_new}]
      * cr_courts:  list[str] — courts as listed by CourtReserve
      * new_player_names: list[str] (subset with rating="?")

    The ``session`` argument is used to title the message ("Today's
    Saturday lineup" etc.) and tailor the rotation-time hint.
    """
    day_word = session.key.capitalize()
    lines: list[str] = []
    lines.append(f"{day_word}'s lineup ({data['date_str']}) is currently:")
    lines.append("")

    def _rating_str(r: dict) -> str:
        # Append "P" to the rating value when provisional (matches
        # the pairings draft format "Geoff(6P)"). Only meaningful for
        # numeric ratings — "?" stays as "?".
        rating = r["rating"]
        if r.get("provisional") and rating != "?":
            return f"{rating}P"
        return rating

    lines.append(f"Registered ({len(data['registrants'])}):")
    for r in data["registrants"]:
        marker = " [NEW]" if r["is_new"] else ""
        lines.append(f"  • {r['name']} (rating {_rating_str(r)}){marker}")
    lines.append("")
    lines.append(f"Waitlist ({len(data['waitlist'])}):")
    if not data["waitlist"]:
        lines.append("  (none)")
    else:
        for r in data["waitlist"]:
            marker = " [NEW]" if r["is_new"] else ""
            lines.append(f"  • {r['name']} (rating {_rating_str(r)}){marker}")
    lines.append("")
    lines.append(
        f"Courts on CourtReserve: {', '.join(data['cr_courts']) or '(none)'}"
    )

    unrated = [r for r in data["registrants"] if str(r["rating"]) == "?"]
    if unrated:
        lines.append("")
        lines.append(
            f"⚠ Players with no rating ({len(unrated)}) — "
            "will be treated as rating 6 (mid-strength) for balancing "
            "unless you update them:"
        )
        for r in unrated:
            marker = " [NEW]" if r["is_new"] else ""
            lines.append(f"  • {r['name']}{marker}")

    provisional = [
        r for r in data["registrants"]
        if r.get("provisional") and str(r["rating"]) != "?"
    ]
    if provisional:
        lines.append("")
        lines.append(
            f"ⓘ Ratings with 'P' ({len(provisional)}) are provisional "
            "(bulk-imported from history, not yet confirmed). "
            "Say 'boris rate <name> <N>' to confirm or correct."
        )

    lines.append("")
    lines.append("Before I generate pairings, please reply with:")
    lines.append("  • Any EXTRA courts available beyond CourtReserve")
    lines.append("    (e.g. 'add courts 3 and 5')")
    if unrated:
        lines.append(
            "  • Ratings for any of the unrated players above "
            "(default treatment is 6 — only override if you actually "
            "know the rating)."
        )
    lines.append(
        "  • Any singles matchups to pin (e.g. "
        "'rotation 1 singles: Amir vs Patrick')"
    )
    lines.append("  • Anything else you want considered")
    lines.append("")
    lines.append(
        "When you're ready, say 'boris go ahead' or 'boris generate pairings'."
    )
    return "\n".join(lines)


# ---------- core --------------------------------------------------------


def _format_rating(rating: Any) -> str:
    return "?" if not isinstance(rating, int) else str(rating)


def _collect_session_data(session: SessionType) -> dict | None:
    """Fetch the upcoming session matching ``session`` from CourtReserve.

    Returns None when no matching event can be found. Raises on outright
    scrape errors (caller catches).
    """
    from courtreserve import CourtReserveClient
    from roster import Roster

    # day_of_week is matched by 3-letter prefix of date string — the
    # session_type weekday gives us that prefix.
    day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][session.weekday]

    with CourtReserveClient() as cr:
        events = cr.list_events(day_of_week=day_name)
        now = datetime.now()
        needle = session.cr_name_contains.lower()
        time_needle = session.cr_time_fragment  # empty string = no constraint
        upcoming = [
            e for e in events
            if (e.start_dt is None or e.start_dt >= now)
            and needle in e.name.lower()
            and (not time_needle or time_needle in e.date_str)
        ]
        if not upcoming:
            return None
        target = upcoming[0]
        reg = cr.get_event_registrants(target.reservation_number)

    registered_names = [r.name for r in reg.registrants]
    waitlist_names = [r.name for r in reg.waitlist]

    roster = Roster()
    pre_existing = set(roster.names())
    new_added = roster.add_many_from_cr(registered_names + waitlist_names)
    new_player_names = [e["name"] for e in new_added]

    roster_dict = roster.all()

    def _entry(name: str) -> dict:
        info = roster_dict.get(name, {})
        return {
            "name": name,
            "rating": _format_rating(info.get("rating", "?")),
            "is_new": name not in pre_existing,
            "provisional": bool(info.get("provisional", False)),
        }

    return {
        "reservation_number": target.reservation_number,
        "date_str": target.date_str,
        "registrants": [_entry(n) for n in registered_names],
        "waitlist": [_entry(n) for n in waitlist_names],
        "cr_courts": list(reg.courts or []),
        "new_player_names": new_player_names,
    }


def kickoff_session(
    session_key: str | None = None,
    *,
    prefer_test_channel: bool = False,
    target_jid: str | None = None,
) -> dict:
    """Run the kickoff for one of the three weekly sessions.

    ``session_key`` is one of the keys in
    :data:`session_types.SESSION_TYPES` (``"tuesday"`` / ``"thursday"`` /
    ``"saturday"``). When ``None`` the next session by weekday is
    picked — Mon/Tue → Tuesday, Wed/Thu → Thursday, Fri/Sat → Saturday,
    Sun → Tuesday.

    On success: posts the structured message, calls ``start_tonight``
    with ``session_type=<key>``, sets ``phase = "awaiting_extras"``,
    and returns ``{ok: True, ...}``.

    On any failure (CR down, no event, bridge unreachable): posts a
    fallback error message into the admin group (best-effort) and
    returns ``{ok: False, error, message}``. Never raises.
    """
    from session_state import get_tonight, set_phase, start_tonight

    if session_key is None:
        session = resolve_next_session()
    else:
        try:
            session = _session_get(session_key)
        except KeyError as e:
            return {
                "ok": False,
                "error": "unknown_session_type",
                "message": str(e),
            }

    # Refuse if there's already an in-flight session.
    state = get_tonight()
    if state.phase:
        return {
            "ok": False,
            "error": "session_in_progress",
            "phase": state.phase,
            "message": (
                f"A session is already in flight (phase={state.phase!r}). "
                "Clear it with 'boris clear run' before running another "
                "kickoff."
            ),
        }

    try:
        data = _collect_session_data(session)
    except Exception as e:
        msg = (
            f"⚠ Kickoff failed: couldn't fetch the upcoming "
            f"{session.display_name} session "
            f"({type(e).__name__}: {e}). Please run 'boris kickoff' "
            "manually once it's sorted."
        )
        _send_to_admin_group(
            msg,
            admin_group_name=session.admin_group_name,
            prefer_test_channel=prefer_test_channel,
            target_jid=target_jid,
        )
        return {"ok": False, "error": "scrape_failed", "message": str(e)}

    if data is None:
        msg = (
            f"⚠ Kickoff: I couldn't find a {session.display_name} "
            "event in CourtReserve for today. If there's no session "
            "this week, you can ignore this. Otherwise check CR and "
            "run 'boris kickoff' once the event is visible."
        )
        _send_to_admin_group(
            msg,
            admin_group_name=session.admin_group_name,
            prefer_test_channel=prefer_test_channel,
            target_jid=target_jid,
        )
        return {"ok": False, "error": "no_event_found"}

    attendee_names = [r["name"] for r in data["registrants"]]
    waitlist_names = [r["name"] for r in data["waitlist"]]
    start_tonight(
        attendees=attendee_names,
        date=data["date_str"],
        source=f"courtreserve:{data['reservation_number']}",
        session_type=session.key,
        court_labels=data["cr_courts"],
        waitlist=waitlist_names,
    )
    set_phase("awaiting_extras")

    text = format_kickoff_message(data, session)
    sent = _send_to_admin_group(
        text,
        admin_group_name=session.admin_group_name,
        prefer_test_channel=prefer_test_channel,
        target_jid=target_jid,
    )
    return {
        "ok": True,
        "posted": sent,
        "session_type": session.key,
        "reservation_number": data["reservation_number"],
        "date_str": data["date_str"],
        "registrants_count": len(data["registrants"]),
        "waitlist_count": len(data["waitlist"]),
        "new_player_names": data["new_player_names"],
        "cr_courts": data["cr_courts"],
        "message": text,
    }


def kickoff_thursday(
    *,
    allow_non_thursday: bool = False,  # noqa: ARG001 — accepted for back-compat
    prefer_test_channel: bool = False,
    target_jid: str | None = None,
) -> dict:
    """Back-compat shim: equivalent to ``kickoff_session("thursday")``.

    The legacy ``allow_non_thursday`` flag is accepted (and ignored) so
    older callers continue to work — the modern entry point lets the
    organiser explicitly pick the session, so the weekday guard is no
    longer needed.
    """
    return kickoff_session(
        "thursday",
        prefer_test_channel=prefer_test_channel,
        target_jid=target_jid,
    )


# ---------- internal replay helper (no session_state mutation) ----------


def replay_past_session_plan(date: str | None = None):
    """Return a fresh PairingPlan generated against a past session's
    attendees + court labels, scored with the CURRENT roster ratings
    + rules. Intended for internal A/B testing of rule or rating
    changes — does NOT touch session_state, history.json, the Sheet,
    or WhatsApp.

    ``date`` is an ISO YYYY-MM-DD that picks a specific history entry;
    omit to replay the most recent one. Raises ``ValueError`` if the
    history is missing or the requested date isn't present.
    """
    from pairings import make_plan
    from roster import Roster

    history_path = Path(__file__).parent / "history.json"
    if not history_path.exists():
        raise ValueError("history.json doesn't exist yet")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        raise ValueError("history is empty (no past sessions)")
    if date:
        matches = [h for h in history if h.get("date") == date]
        if not matches:
            available = sorted({h.get("date", "?") for h in history})
            raise ValueError(
                f"no history entry for {date!r}; available dates: "
                f"{', '.join(available[-10:])}"
            )
        entry = matches[-1]
    else:
        entry = history[-1]

    attendees = list(entry.get("attendees") or [])
    court_labels = list(entry.get("court_labels") or [])
    if not attendees or not court_labels:
        raise ValueError(
            f"history entry for {entry.get('date')!r} is missing "
            "attendees or court_labels"
        )

    return make_plan(
        attendees,
        players_path=Roster().all(),
        history_path=str(history_path),
        court_labels=court_labels,
        num_rotations=int(entry.get("num_rotations") or 3),
    )


# ---------- CLI ---------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a session kickoff.")
    parser.add_argument(
        "--session",
        default=None,
        choices=sorted(SESSION_TYPES.keys()),
        help=(
            "Which weekly session to kick off (tuesday/thursday/saturday). "
            "Defaults to the next scheduled session by weekday."
        ),
    )
    parser.add_argument(
        "--test-channel",
        action="store_true",
        help="Post to Boris the tennis bot instead of the live admin group.",
    )
    args = parser.parse_args()

    result = kickoff_session(
        session_key=args.session,
        prefer_test_channel=args.test_channel,
    )
    json.dump(
        {k: v for k, v in result.items() if k != "message"},
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    sys.exit(0 if result.get("ok") else 2)
