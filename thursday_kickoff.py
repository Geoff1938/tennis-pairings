"""Session kickoff for the Boris pairing workflow.

Originally Thursday-only — now drives Tuesday, Thursday and Saturday
sessions via the ``session_types.SESSION_TYPES`` registry.

The main entry point is :func:`kickoff_session`. It:

1. Picks a SessionType (explicit or defaulted to "next scheduled by
   weekday").
2. Looks up the matching upcoming event in CourtReserve.
3. Auto-adds any unseen names to the roster (rating ``?``).
4. Calls ``start_tonight`` with the attendees / waitlist / courts and
   sets ``session_state.session_type`` and ``phase = "awaiting_extras"``.
5. Posts the structured "today's lineup + reply with extras" message
   to the session type's admin WhatsApp group (or to a passed
   ``target_jid`` for dry runs).

``kickoff_thursday`` is preserved as a thin wrapper for any
out-of-tree callers / scripts. ``kickoff_from_history`` is still here
unchanged in behaviour — it replays a past committed plan as a test run.

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
    lines.append(f"Today's {day_word} lineup ({data['date_str']}):")
    lines.append("")
    lines.append(f"Registered ({len(data['registrants'])}):")
    for r in data["registrants"]:
        marker = " [NEW]" if r["is_new"] else ""
        lines.append(f"  • {r['name']} (rating {r['rating']}){marker}")
    lines.append("")
    lines.append(f"Waitlist ({len(data['waitlist'])}):")
    if not data["waitlist"]:
        lines.append("  (none)")
    else:
        for r in data["waitlist"]:
            marker = " [NEW]" if r["is_new"] else ""
            lines.append(f"  • {r['name']} (rating {r['rating']}){marker}")
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
        }

    return {
        "reservation_number": target.reservation_number,
        "date_str": target.date_str,
        "registrants": [_entry(n) for n in registered_names],
        "waitlist": [_entry(n) for n in waitlist_names],
        "cr_courts": list(reg.courts or []),
        "new_player_names": new_player_names,
    }


TEST_RUN_BANNER = (
    "🧪 TEST RUN — these pairings won't be saved in pairings history. "
    "Rating updates will still be saved.\n\n"
)


def kickoff_session(
    session_key: str | None = None,
    *,
    prefer_test_channel: bool = False,
    target_jid: str | None = None,
    test_mode: bool = False,
) -> dict:
    """Run the morning kickoff for one of the three weekly sessions.

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
    if state.phase and state.phase not in ("", "finalised"):
        return {
            "ok": False,
            "error": "session_in_progress",
            "phase": state.phase,
            "message": (
                f"A session is already in flight (phase={state.phase!r}). "
                "Clear it with 'boris start over' / clear_tonight before "
                "running another kickoff."
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
        test_mode=test_mode,
    )
    set_phase("awaiting_extras")

    text = format_kickoff_message(data, session)
    if test_mode:
        text = TEST_RUN_BANNER + text
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
        "test_mode": test_mode,
        "message": text,
    }


def kickoff_thursday(
    *,
    allow_non_thursday: bool = False,  # noqa: ARG001 — accepted for back-compat
    prefer_test_channel: bool = False,
    target_jid: str | None = None,
    test_mode: bool = False,
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
        test_mode=test_mode,
    )


# ---------- replay-from-history ----------------------------------------


def kickoff_from_history(
    *,
    date: str | None = None,
    target_jid: str | None = None,
    prefer_test_channel: bool = False,
) -> dict:
    """Set up a TEST RUN session whose attendees and court labels come
    from a past committed plan in history.json — so the admin can
    iterate on ratings/extras and see how the new ratings change the
    pairings, against a known real-world roster rather than whatever
    is currently signed up on CourtReserve.

    Always test_mode=True. Always allowed off-day. ``date`` (ISO
    YYYY-MM-DD) picks a specific history entry; defaults to the most
    recent. The session_type field is carried over from the history
    entry if present, otherwise defaults to ``"thursday"`` (the only
    kind of session we wrote before this generalisation).

    Posts a kickoff-style message into the calling channel either way;
    never raises.
    """
    from session_state import get_tonight, set_phase, start_tonight

    state = get_tonight()
    if state.phase and state.phase not in ("", "finalised"):
        return {
            "ok": False,
            "error": "session_in_progress",
            "phase": state.phase,
            "message": (
                f"A session is already in flight (phase={state.phase!r}). "
                "Clear it with 'boris clear this run' before replaying."
            ),
        }

    history_path = Path(__file__).parent / "history.json"
    # Pick a reasonable admin-group for the error fallback when we
    # don't yet know which session_type the replay will end up being.
    fallback_admin_group = SESSION_TYPES["thursday"].admin_group_name

    if not history_path.exists():
        msg = "⚠ Replay failed: history.json doesn't exist yet."
        _send_to_admin_group(
            msg,
            admin_group_name=fallback_admin_group,
            prefer_test_channel=prefer_test_channel,
            target_jid=target_jid,
        )
        return {"ok": False, "error": "no_history_file", "message": msg}
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception as e:
        msg = f"⚠ Replay failed: couldn't read history.json ({e!r})."
        _send_to_admin_group(
            msg,
            admin_group_name=fallback_admin_group,
            prefer_test_channel=prefer_test_channel,
            target_jid=target_jid,
        )
        return {"ok": False, "error": "history_unreadable", "message": str(e)}
    if not history:
        msg = "⚠ Replay failed: history is empty (no past sessions to replay)."
        _send_to_admin_group(
            msg,
            admin_group_name=fallback_admin_group,
            prefer_test_channel=prefer_test_channel,
            target_jid=target_jid,
        )
        return {"ok": False, "error": "history_empty", "message": msg}

    if date:
        matches = [h for h in history if h.get("date") == date]
        if not matches:
            available = sorted({h.get("date", "?") for h in history})
            msg = (
                f"⚠ Replay failed: no history entry for {date!r}. "
                f"Available dates: {', '.join(available[-5:]) or '(none)'}."
            )
            _send_to_admin_group(
                msg,
                admin_group_name=fallback_admin_group,
                prefer_test_channel=prefer_test_channel,
                target_jid=target_jid,
            )
            return {"ok": False, "error": "date_not_found", "message": msg}
        entry = matches[-1]
    else:
        entry = history[-1]

    attendees = list(entry.get("attendees") or [])
    court_labels = list(entry.get("court_labels") or [])
    entry_date = entry.get("date", "(unknown date)")
    entry_session_type = str(entry.get("session_type") or "thursday")
    if not attendees or not court_labels:
        msg = (
            f"⚠ Replay failed: history entry for {entry_date!r} is missing "
            "attendees or court_labels."
        )
        _send_to_admin_group(
            msg,
            admin_group_name=fallback_admin_group,
            prefer_test_channel=prefer_test_channel,
            target_jid=target_jid,
        )
        return {"ok": False, "error": "incomplete_entry", "message": msg}

    start_tonight(
        attendees=attendees,
        date=entry_date,
        source=f"history:{entry_date}",
        session_type=entry_session_type,
        court_labels=court_labels,
        waitlist=[],
        test_mode=True,
    )
    set_phase("awaiting_extras")

    from roster import Roster

    try:
        roster = Roster().all()
    except Exception:
        roster = {}

    def _r(n: str) -> str:
        info = roster.get(n) or {}
        v = info.get("rating", "?")
        return str(v) if isinstance(v, int) else "?"

    lines: list[str] = []
    lines.append(TEST_RUN_BANNER.rstrip())
    lines.append("")
    lines.append(
        f"Replaying the session from {entry_date} — same {len(attendees)} "
        f"players and same {len(court_labels)} courts. "
        "Current roster ratings shown alongside (any changes you make "
        "to ratings will persist; the pairings won't be saved)."
    )
    lines.append("")
    lines.append(f"Players ({len(attendees)}):")
    unrated_count = 0
    for n in attendees:
        rating = _r(n)
        if rating == "?":
            unrated_count += 1
        lines.append(f"  • {n} (rating {rating})")
    lines.append("")
    lines.append(f"Courts ({len(court_labels)}): {', '.join(court_labels)}")
    if unrated_count:
        lines.append("")
        lines.append(
            f"⚠ {unrated_count} player(s) still unrated — they'll be treated "
            "as rating 5 unless you update them."
        )
    lines.append("")
    lines.append("Before I generate pairings, please reply with:")
    lines.append("  • Any rating changes you'd like to apply (e.g. 'Tomoki = 1')")
    lines.append("  • Any singles matchups to pin")
    lines.append("  • 'boris go ahead' / 'boris generate pairings' when ready")
    text = "\n".join(lines)

    # Pick the admin group of the session_type the replay is mimicking,
    # so a replay of a Saturday session lands in the Westside group.
    admin_group_name = SESSION_TYPES.get(
        entry_session_type, SESSION_TYPES["thursday"]
    ).admin_group_name

    sent = _send_to_admin_group(
        text,
        admin_group_name=admin_group_name,
        prefer_test_channel=prefer_test_channel,
        target_jid=target_jid,
    )
    return {
        "ok": True,
        "posted": sent,
        "replayed_date": entry_date,
        "session_type": entry_session_type,
        "attendees_count": len(attendees),
        "court_labels": court_labels,
        "test_mode": True,
        "message": text,
    }


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
