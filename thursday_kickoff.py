"""Thursday-morning kickoff for the Boris pairing workflow.

Fetches the next ``Thursday Social Tennis Evening`` event from
CourtReserve, auto-adds any new names to the roster, calls
``start_tonight`` with the full attendee + waitlist + court list, sets
``session_state.phase = "awaiting_extras"``, and posts a structured
message to the ``Thursday Tennis Organisers`` group asking for the
additional info needed to generate pairings.

Two entry points:

1. ``kickoff_thursday()`` — programmatic; called by both the
   admin_bot's scheduled-trigger check and the WhatsApp bot tool.
2. ``python thursday_kickoff.py`` — CLI shim for ad-hoc manual runs.

If the CourtReserve scrape fails or no Thursday-evening event is found,
the function posts a fallback error message to the admin group instead
of raising. The caller can rely on best-effort behaviour: it never
crashes the polling loop.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# These constants are duplicated from admin_bot rather than imported to
# keep this module independently runnable (and to avoid a circular
# import once admin_bot calls into here).
BRIDGE_URL = "http://localhost:8080/api"
ADMIN_GROUP_NAME = "Thursday Tennis Organisers"
TEST_GROUP_NAME = "Boris the tennis bot"
BOT_REPLY_PREFIX = "From Boris the tennis bot: "

# Used when looking up the upcoming session — match the prior code's
# selector for the Thursday-evening 19:30-21:30 social.
EVENT_NAME_FRAGMENT = "social"
EVENT_DATE_FRAGMENT = "19:30 - 21:30"


# ---------- bridge sender (no admin_bot dep) ----------------------------


def _resolve_admin_group_jid(group_name: str) -> str | None:
    """Look up a group's JID via the bridge's chat list.

    Returns None if the group hasn't been seen by the bridge yet (e.g.
    no message has reached it). Mirrors the resolution path admin_bot
    does at startup.
    """
    import sqlite3
    from pathlib import Path

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
    prefer_test_channel: bool = False,
    target_jid: str | None = None,
) -> bool:
    """Send a message to the admin group via the bridge.

    If ``target_jid`` is supplied it wins — used when the kickoff was
    triggered from a specific WhatsApp group and the post should go
    back there (e.g. a dry run from Boris the tennis bot).

    Otherwise defaults to ``Thursday Tennis Organisers``, falling back to the
    test channel when ``prefer_test_channel`` is True.
    """
    if target_jid:
        jid: str | None = target_jid
    else:
        primary = TEST_GROUP_NAME if prefer_test_channel else ADMIN_GROUP_NAME
        jid = _resolve_admin_group_jid(primary)
        if not jid:
            other = ADMIN_GROUP_NAME if prefer_test_channel else TEST_GROUP_NAME
            jid = _resolve_admin_group_jid(other)
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


def format_kickoff_message(data: dict) -> str:
    """Render the structured kickoff message from the data dict.

    ``data`` shape (produced by ``_collect_session_data``):
      * date_str: human-readable date
      * registrants: list[{name, rating, is_new}]
      * waitlist:   list[{name, rating, is_new}]
      * cr_courts:  list[str] — courts as listed by CourtReserve
      * new_player_names: list[str] (subset with rating="?")
    """
    lines: list[str] = []
    lines.append(f"Today's Thursday lineup ({data['date_str']}):")
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

    # Highlight any registered players (new or existing) whose rating is
    # still "?" — they'll be treated as 5 unless the admin updates them.
    unrated = [r for r in data["registrants"] if str(r["rating"]) == "?"]
    if unrated:
        lines.append("")
        lines.append(
            f"⚠ Players with no rating ({len(unrated)}) — "
            "will be treated as rating 5 (mid-strength) for balancing "
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
            "  • Ratings for any of the unrated players above — "
            "e.g. 'Tomoki = 2'. (Default treatment is 5 — only override "
            "if you actually know the rating.)"
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


def _collect_session_data() -> dict | None:
    """Fetch the upcoming Thursday-evening session and roster context.

    Returns None when no matching CourtReserve event can be found.
    Raises on outright scrape errors (caller catches).
    """
    from courtreserve import CourtReserveClient
    from roster import Roster

    with CourtReserveClient() as cr:
        events = cr.list_events(day_of_week="Thursday")
        now = datetime.now()
        upcoming = [
            e for e in events
            if (e.start_dt is None or e.start_dt >= now)
            and EVENT_NAME_FRAGMENT in e.name.lower()
            and EVENT_DATE_FRAGMENT in e.date_str
        ]
        if not upcoming:
            return None
        target = upcoming[0]
        reg = cr.get_event_registrants(target.reservation_number)

    registered_names = [r.name for r in reg.registrants]
    waitlist_names = [r.name for r in reg.waitlist]

    # Auto-add any unseen names to the roster (rating="?" by default).
    roster = Roster()
    pre_existing = set(roster.names())
    new_added = roster.add_many_from_cr(registered_names + waitlist_names)
    new_player_names = [e["name"] for e in new_added]

    # Refresh the cache after any adds.
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


def kickoff_thursday(
    *,
    allow_non_thursday: bool = False,
    prefer_test_channel: bool = False,
    target_jid: str | None = None,
    test_mode: bool = False,
) -> dict:
    """Run the Thursday-morning kickoff.

    On success: posts the structured message to the admin group, calls
    ``start_tonight``, sets ``session_state.phase = "awaiting_extras"``,
    and returns ``{ok: True, ...}``.

    When ``test_mode`` is True the session is flagged as a dry run —
    commit_plan / log_pairings_to_sheet refuse — and the kickoff post
    is prefixed with a TEST RUN banner. ``target_jid`` overrides the
    default channel destination (used to keep dry-runs in the channel
    that triggered them).

    On any failure (CR down, no event found, bridge unreachable):
    posts a fallback error message to the admin group (best-effort)
    and returns ``{ok: False, error: ..., message: ...}``. Never
    raises — the caller (poll loop or bot tool) can rely on this.
    """
    from session_state import (
        SessionState, get_tonight, set_phase, start_tonight,
    )

    if not allow_non_thursday:
        weekday = datetime.now().weekday()
        if weekday != 3:  # Mon=0, Thu=3
            return {
                "ok": False,
                "error": "not_thursday",
                "message": (
                    f"Today is weekday={weekday}; kickoff is for Thursdays "
                    "only. Pass allow_non_thursday=True to override."
                ),
            }

    # Refuse if there's already an in-flight session that hasn't been
    # finalised — the admin should explicitly clear it first.
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
        data = _collect_session_data()
    except Exception as e:
        msg = (
            "⚠ Kickoff failed: couldn't fetch tonight's CourtReserve "
            f"session ({type(e).__name__}: {e}). Please run 'boris kickoff' "
            "manually once it's sorted."
        )
        _send_to_admin_group(
            msg,
            prefer_test_channel=prefer_test_channel,
            target_jid=target_jid,
        )
        return {"ok": False, "error": "scrape_failed", "message": str(e)}

    if data is None:
        msg = (
            "⚠ Kickoff: I couldn't find a Thursday Evening Club Social "
            "event in CourtReserve for today. If there's no session this "
            "week, you can ignore this. Otherwise check CR and run "
            "'boris kickoff' once the event is visible."
        )
        _send_to_admin_group(
            msg,
            prefer_test_channel=prefer_test_channel,
            target_jid=target_jid,
        )
        return {"ok": False, "error": "no_event_found"}

    # Persist session state.
    attendee_names = [r["name"] for r in data["registrants"]]
    waitlist_names = [r["name"] for r in data["waitlist"]]
    start_tonight(
        attendees=attendee_names,
        date=data["date_str"],
        source=f"courtreserve:{data['reservation_number']}",
        court_labels=data["cr_courts"],
        waitlist=waitlist_names,
        test_mode=test_mode,
    )
    set_phase("awaiting_extras")

    # Post the kickoff message.
    text = format_kickoff_message(data)
    if test_mode:
        text = TEST_RUN_BANNER + text
    sent = _send_to_admin_group(
        text,
        prefer_test_channel=prefer_test_channel,
        target_jid=target_jid,
    )
    return {
        "ok": True,
        "posted": sent,
        "reservation_number": data["reservation_number"],
        "date_str": data["date_str"],
        "registrants_count": len(data["registrants"]),
        "waitlist_count": len(data["waitlist"]),
        "new_player_names": data["new_player_names"],
        "cr_courts": data["cr_courts"],
        "test_mode": test_mode,
        "message": text,
    }


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

    Always test_mode=True (commit_plan / log_pairings_to_sheet refuse).
    Always allow_non_thursday — replay is inherently off-day.

    ``date`` (ISO YYYY-MM-DD) picks a specific history entry; defaults
    to the most recent one when None.

    Returns ``{ok: True, ...}`` on success or
    ``{ok: False, error, message}`` on a non-fatal failure (no
    history, requested date not found, in-flight session). Posts a
    structured message into the calling channel either way.
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
    if not history_path.exists():
        msg = "⚠ Replay failed: history.json doesn't exist yet."
        _send_to_admin_group(
            msg, prefer_test_channel=prefer_test_channel,
            target_jid=target_jid,
        )
        return {"ok": False, "error": "no_history_file", "message": msg}
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception as e:
        msg = f"⚠ Replay failed: couldn't read history.json ({e!r})."
        _send_to_admin_group(
            msg, prefer_test_channel=prefer_test_channel,
            target_jid=target_jid,
        )
        return {"ok": False, "error": "history_unreadable", "message": str(e)}
    if not history:
        msg = "⚠ Replay failed: history is empty (no past sessions to replay)."
        _send_to_admin_group(
            msg, prefer_test_channel=prefer_test_channel,
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
                msg, prefer_test_channel=prefer_test_channel,
                target_jid=target_jid,
            )
            return {"ok": False, "error": "date_not_found", "message": msg}
        entry = matches[-1]
    else:
        entry = history[-1]

    attendees = list(entry.get("attendees") or [])
    court_labels = list(entry.get("court_labels") or [])
    entry_date = entry.get("date", "(unknown date)")
    if not attendees or not court_labels:
        msg = (
            f"⚠ Replay failed: history entry for {entry_date!r} is missing "
            "attendees or court_labels."
        )
        _send_to_admin_group(
            msg, prefer_test_channel=prefer_test_channel,
            target_jid=target_jid,
        )
        return {"ok": False, "error": "incomplete_entry", "message": msg}

    # Persist as session_state — flagged as a test run.
    start_tonight(
        attendees=attendees,
        date=entry_date,
        source=f"history:{entry_date}",
        court_labels=court_labels,
        waitlist=[],
        test_mode=True,
    )
    set_phase("awaiting_extras")

    # Render the structured kickoff message — pull current ratings
    # from the live roster so the admin sees what's IN PLAY now (not
    # the ratings as they were when this plan was committed).
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

    sent = _send_to_admin_group(
        text, prefer_test_channel=prefer_test_channel,
        target_jid=target_jid,
    )
    return {
        "ok": True,
        "posted": sent,
        "replayed_date": entry_date,
        "attendees_count": len(attendees),
        "court_labels": court_labels,
        "test_mode": True,
        "message": text,
    }


# ---------- CLI ---------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the Thursday kickoff.")
    parser.add_argument(
        "--allow-non-thursday",
        action="store_true",
        help="Bypass the Thursday-only check (testing).",
    )
    parser.add_argument(
        "--test-channel",
        action="store_true",
        help="Post to Boris the tennis bot instead of the live admin group.",
    )
    args = parser.parse_args()

    result = kickoff_thursday(
        allow_non_thursday=args.allow_non_thursday,
        prefer_test_channel=args.test_channel,
    )
    json.dump(
        {k: v for k, v in result.items() if k != "message"},
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    sys.exit(0 if result.get("ok") else 2)
