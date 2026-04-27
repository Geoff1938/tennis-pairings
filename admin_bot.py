"""WhatsApp admin bot for the Thursday Tennis group.

Watches a WhatsApp group called "Thursday tennis Admin" (or whatever name
is configured in ADMIN_GROUP_NAME). When a message in that group starts
with ``boris`` or ``bot`` (either word, case-insensitive; an optional
trailing colon is accepted), the rest of the message is sent to Claude
along with a small toolbox (look up polls, players, history, generate
pairings, fetch CourtReserve registrants). Claude's final text reply is
posted back into the same admin group, prefixed with
"From Boris the tennis bot: ". If the agent takes more than ~5 seconds,
a short "working on it" update is sent first.

For safety the bot only ever sends replies into the admin group — no tool
exists for writing to the main tennis group. That stays a manual step.

Run:
    set ANTHROPIC_API_KEY=sk-ant-...        (once, in your shell)
    py -3 admin_bot.py
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
from anthropic import Anthropic
from dotenv import load_dotenv

from roster import Roster

# Load ANTHROPIC_API_KEY (and anything else) from .env in the project root.
load_dotenv(Path(__file__).parent / ".env")

# ---------- config --------------------------------------------------------

ROOT = Path(__file__).parent
HISTORY_PATH = ROOT / "history.json"
BRIDGE_DB = ROOT / "whatsapp-mcp" / "whatsapp-bridge" / "store" / "messages.db"
BRIDGE_URL = "http://localhost:8080/api"

ADMIN_GROUP_NAMES = [
    "Thursday tennis Admin",
    "Boris test channel",
]
TENNIS_GROUP_JID = "120363408685115680@g.us"  # Thursday Social Tennis Evening

POLL_INTERVAL_SECONDS = 1
# Matches either "boris" or "bot" as a leading word (case-insensitive),
# followed by any combination of whitespace and simple punctuation that a
# human might type before their actual question: `:`, `?`, `!`, `,`, `.`.
# "bottle", "both", "borised" etc. won't trigger (word-boundary required).
BOT_TRIGGER_PATTERN = re.compile(r"^\s*(?:boris|bot)\b[\s:?!,.]*", re.IGNORECASE)
BOT_REPLY_PREFIX = "From Boris the tennis bot: "
WORKING_ON_IT_DELAY_SECONDS = 5.0
WORKING_ON_IT_TEXT = "Request received. Working on it…"
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048
AGENT_LOOP_MAX_TURNS = 8

SYSTEM_PROMPT = """\
You are "Boris the tennis bot", the admin assistant for the Westside
Thursday Social Tennis evenings (and any other club session the admin
asks about).

You receive commands from the admin chat. Every command starts with the
trigger word `boris` or `bot` (optionally followed by `:` / `?` / `!`).
Messages without the trigger never reach you — they're filtered out
upstream. So treat every message you do see as a real bot request.

Reply with the BODY of a WhatsApp message — plain text only, no markdown,
no headers, no tables, no code blocks. Use emoji sparingly.

Do NOT include any "From Boris the tennis bot:" prefix yourself — the
wrapper around you adds that automatically. Just write the message body.

Context
-------
- Today's date is injected into each user message.
- Player roster is a Google Sheet ("Players" tab), keyed by full name,
  with fields gender (M/F/?), rating (1-5 or "?"), notes.
  **Rating scale: 1 = strongest, 5 = weakest, ? = unknown.** Don't mix
  this up — the singles-selection and skill-balance logic depend on it.
- Historical pairings are kept locally (history.json) for repeat-pair
  avoidance. Session log + Pair log tabs in the sheet are human-browse
  mirrors.
- Registration source of truth is CourtReserve (Westside Racquets), in
  the "Social & Club Sessions" category.

Capacity policy (important)
---------------------------
With `n` attendees and `c` courts reserved, each court seats 4 doubles
or 2 singles (total capacity `4c`):
- `n > 4c` → tool returns error; ask admin to drop someone or add a court.
- `n == 4c` → all doubles, no sit-outs.
- `n < 4c` and even → `(4c - n)/2` singles courts + the rest as doubles.
  No sit-outs. Singles goes on the highest-numbered reserved courts.
  Singles players are picked from the strongest (lowest-rated) attendees.
- `n < 4c` and odd → 1 player sits out each rotation (rotated fairly);
  remaining 14 handled as the even case.

Available tools
---------------
SESSION STATE (what the admin is building up for tonight):
- start_tonight: initialise tonight. Pass reservation_number (fetches
  from CourtReserve, INCLUDING the waitlist) OR attendee_names (manual).
  Optionally court_labels.
- get_tonight: show current state (attendees, waitlist, court_labels,
  date).
- add_to_tonight: append one attendee name (creates roster entry if new).
- remove_from_tonight: remove one attendee (fuzzy name match).
- set_courts_for_tonight: set/overwrite court labels (e.g. [7,8,9,10]
  — these can include courts that aren't in CourtReserve).
- promote_from_waitlist: move a waitlisted player into attendees (fuzzy
  name match against the current waitlist).
- clear_tonight: wipe session state.

COURT-RESERVE READ:
- list_club_sessions: upcoming CR sessions, optionally filtered.
- get_session_registrants: registrants + waitlist + metadata for a given
  reservation_number; auto-adds unseen names from BOTH lists.

ROSTER:
- read_players_roster: full map of name → {gender, rating, notes}.
- set_player_rating: update 1-5 rating (fuzzy name). '?' resets.

HISTORY + PAIRINGS:
- read_pairings_history: past weeks' plans.
- generate_pairings: when called with no args, uses the session state.
  Refuses if the session is missing attendees or court_labels.
  `log_to_sheet=true` ONLY when the admin has confirmed the plan.
- log_pairings_to_sheet: post-hoc log a previously-generated plan.

Typical Thursday flow
---------------------
1. "boris get tonight's attendees from courtreserve" →
   list_club_sessions(day_of_week="Thursday") to find tonight's session,
   then start_tonight(reservation_number=...). The session state will
   carry both `attendees` (registered) and `waitlist` (priority order).
   Reply with the registered list AND the waitlist, and explicitly ask
   the admin:
     a) Which extra courts (beyond what CourtReserve shows) are
        available tonight? CourtReserve's courts are in the response's
        `event.courts_from_courtreserve` field — repeat them so the
        admin can confirm.
     b) Given the extra courts, which waitlisted players will play?
2. Admin replies with extra courts and waitlist promotions, e.g.
   "we also have courts 1, 2, 3 — promote the first 4 waitlisted".
   - Call set_courts_for_tonight with the COMBINED list (CR courts +
     extra courts).
   - Call promote_from_waitlist for each name they confirm. For "first
     N waitlisted", iterate through state.waitlist[:N] and promote each.
3. Admin may also do ad-hoc "boris remove X" / "boris add Y" — use
   remove_from_tonight / add_to_tonight.
4. ONLY when courts AND attendees are confirmed, "boris generate
   pairings" → generate_pairings() with no args (uses state). DO NOT
   call generate_pairings until you have:
     • attendees populated, AND
     • court_labels set covering enough capacity for those attendees.
   If the admin asks for pairings before either is set, ask them first
   rather than guessing.
5. Format pairings output with one line per court per rotation using
   the `display_names` map from the response:
     Rotation 1 (19:30)
     Court 4: Geoff C & Silvia M  v  Paul V & Hannah B  (doubles)
     ...
     Court 6: David G v Jack M  (singles)
   Include sit-outs if any, and mention the plan's `notes`.
6. When the admin confirms ("use those" / "save"), re-run
   generate_pairings with log_to_sheet=true, or call
   log_pairings_to_sheet with the plan dict.

Handling day-only references ("who's signed up for Thursday?")
--------------------------------------------------------------
Use list_club_sessions(day_of_week=<day>) and take the earliest upcoming
match. If multiple distinct sessions match on that day, ask the admin to
clarify.

Rating updates ("set rating for X to N")
----------------------------------------
Call set_player_rating directly; if the tool returns 'ambiguous', show
the candidates and ask.

If unsure, ask a short clarifying question rather than guessing.
"""


# ---------- admin group lookup -------------------------------------------


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


# ---------- tool implementations -----------------------------------------


def tool_list_club_sessions(
    day_of_week: Optional[str] = None,
    days_ahead: int = 14,
    name_contains: Optional[str] = None,
    category: str = "Social & Club Sessions",
) -> dict:
    """List upcoming CourtReserve sessions matching optional filters."""
    # Local import so a Playwright import error doesn't crash bot startup.
    from courtreserve import CourtReserveClient

    with CourtReserveClient() as cr:
        events = cr.list_events(
            category=category,
            day_of_week=day_of_week,
            days_ahead=days_ahead,
            name_contains=name_contains,
        )
    return {
        "count": len(events),
        "events": [
            {
                "name": e.name,
                "category": e.category,
                "date": e.date_str,
                "reservation_number": e.reservation_number,
                "res_id": e.res_id,
                "detail_url": e.detail_url,
            }
            for e in events
        ],
    }


def tool_get_session_registrants(reservation_number: str) -> dict:
    """Fetch the registrant + waitlist for a specific CourtReserve session.

    Automatically adds any previously-unseen names (from BOTH lists) to the
    local roster with a guessed gender and rating "?". The response carries
    the registrants and the waitlist (in CourtReserve priority order)
    separately so Boris can ask the admin which waitlisted players will
    actually play given any extra courts available.
    """
    from courtreserve import CourtReserveClient

    with CourtReserveClient() as cr:
        reg = cr.get_event_registrants(reservation_number)
    registered = [r.name for r in reg.registrants]
    waitlist = [r.name for r in reg.waitlist]

    roster = Roster()
    new_players = roster.add_many_from_cr(registered + waitlist)

    return {
        "event_name": reg.event_name,
        "category": reg.category,
        "date": reg.date_str,
        "is_full": reg.is_full,
        "spots_remaining": reg.spots_remaining,
        "courts": reg.courts,
        "age_restriction": reg.age_restriction,
        "registrants": registered,
        "waitlist": waitlist,
        "new_players_added": new_players,
        "detail_url": reg.detail_url,
    }


def tool_read_players_roster() -> dict:
    """Return the full roster mapping name -> {gender, rating, notes}."""
    return Roster().all()


def tool_set_player_rating(name: str, rating: Any) -> dict:
    """Update a player's 1-5 rating. Fuzzy-matches the name; ambiguous
    matches return an error with the candidates so the admin can pick.
    """
    roster = Roster()
    # Exact match first.
    if roster.get(name) is None:
        matches = roster.find_by_fuzzy(name)
        if not matches:
            return {"ok": False, "error": "not_found", "query": name, "candidates": []}
        if len(matches) > 1:
            return {
                "ok": False,
                "error": "ambiguous",
                "query": name,
                "candidates": matches,
            }
        name = matches[0]

    try:
        entry = roster.set_rating(name, rating)
    except ValueError as e:
        return {"ok": False, "error": "invalid_rating", "message": str(e)}
    return {"ok": True, "name": name, "entry": entry}


def tool_read_pairings_history(lookback: int = 4) -> list:
    if not HISTORY_PATH.exists():
        return []
    with HISTORY_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return data[-lookback:] if lookback else data


def tool_generate_pairings(
    attendee_names: Optional[list[str]] = None,
    court_labels: Optional[list] = None,
    num_courts: Optional[int] = None,
    num_rotations: int = 3,
    seed: Optional[int] = None,
    start_time_hhmm: str = "19:30",
    rotation_minutes: int = 40,
    log_to_sheet: bool = False,
) -> dict:
    """Build a pairing plan; optionally mirror it into the Session/Pair-log tabs.

    If ``attendee_names`` is None, uses the current session state (what the
    admin has set via start_tonight / add / remove). Same for ``court_labels``.
    """
    from pairings import make_plan
    from session_state import get_tonight

    state = get_tonight()
    names = list(attendee_names) if attendee_names is not None else list(state.attendees)
    labels = court_labels if court_labels is not None else (state.court_labels or None)
    if not names:
        return {
            "ok": False,
            "error": "no_attendees",
            "message": "No attendees supplied and session state is empty. "
            "Run start_tonight first, or pass attendee_names.",
        }
    if not labels and num_courts is None:
        return {
            "ok": False,
            "error": "no_courts",
            "message": "No court_labels in session state and num_courts not given. "
            "Set courts via set_courts_for_tonight or pass court_labels.",
        }

    try:
        plan = make_plan(
            names,
            players_path=Roster().all(),
            history_path=str(HISTORY_PATH),
            num_courts=num_courts,
            court_labels=labels,
            num_rotations=num_rotations,
            start_time_hhmm=start_time_hhmm,
            rotation_minutes=rotation_minutes,
            seed=seed,
        )
    except ValueError as e:
        return {"ok": False, "error": "over_capacity_or_bad_input", "message": str(e)}
    result = plan.to_dict()
    if log_to_sheet:
        try:
            from session_log import log_plan

            result["sheet_log"] = log_plan(result)
        except Exception as e:
            result["sheet_log_error"] = str(e)
    return result


def tool_log_pairings_to_sheet(plan: dict) -> dict:
    """Append a previously-generated plan to the Session/Pair-log tabs."""
    from session_log import log_plan

    return log_plan(plan)


# ---------- session-state tools -----------------------------------------


def tool_start_tonight(
    reservation_number: Optional[str] = None,
    attendee_names: Optional[list[str]] = None,
    date: Optional[str] = None,
    court_labels: Optional[list] = None,
) -> dict:
    """Begin a new session for tonight (or a future date).

    Supply EITHER a CourtReserve ``reservation_number`` (we'll fetch the
    registrants), OR an explicit ``attendee_names`` list. Overwrites any
    previous session state.
    """
    from session_state import start_tonight

    if reservation_number and attendee_names:
        return {
            "ok": False,
            "error": "ambiguous",
            "message": "Pass either reservation_number or attendee_names, not both.",
        }
    source = ""
    event_meta: dict = {}
    new_players: list = []
    waitlist: list[str] = []
    if reservation_number:
        from courtreserve import CourtReserveClient

        with CourtReserveClient() as cr:
            reg = cr.get_event_registrants(reservation_number)
        names = [r.name for r in reg.registrants]
        waitlist = [r.name for r in reg.waitlist]
        new_players = Roster().add_many_from_cr(names + waitlist)
        source = f"courtreserve:{reservation_number}"
        event_meta = {
            "event_name": reg.event_name,
            "date": reg.date_str,
            "courts_from_courtreserve": reg.courts,
            "is_full": reg.is_full,
            "spots_remaining": reg.spots_remaining,
        }
        if not date:
            date = reg.date_str
    elif attendee_names:
        names = list(attendee_names)
        new_players = Roster().add_many_from_cr(names)
        source = "manual"
    else:
        return {
            "ok": False,
            "error": "missing_input",
            "message": "Supply reservation_number or attendee_names.",
        }

    state = start_tonight(
        names,
        date=date or "",
        source=source,
        court_labels=court_labels,
        waitlist=waitlist,
    )
    return {
        "ok": True,
        "state": state.to_dict(),
        "event": event_meta,
        "new_players_added": new_players,
    }


def tool_promote_from_waitlist(name: str) -> dict:
    """Move a CourtReserve-waitlisted player into tonight's attendee list.

    Use this when the admin says e.g. "promote Alice" or "add the first 3
    waitlisted". Fuzzy-matches the name against the waitlist; if more than
    one match the response carries 'ambiguous' + candidates.
    """
    from session_state import find_attendee_fuzzy, get_tonight, promote_from_waitlist

    state = get_tonight()
    if not state.waitlist:
        return {
            "ok": False,
            "error": "no_waitlist",
            "message": "No waitlist in the current session state.",
        }
    q = name.strip().lower()
    matches = [w for w in state.waitlist if q in w.lower()]
    if not matches:
        return {"ok": False, "error": "not_found", "query": name}
    if len(matches) > 1:
        return {
            "ok": False,
            "error": "ambiguous",
            "query": name,
            "candidates": matches,
        }
    state, promoted = promote_from_waitlist(name)
    return {
        "ok": True,
        "promoted": promoted,
        "state": state.to_dict(),
    }


def tool_get_tonight() -> dict:
    """Return the current session state."""
    from session_state import get_tonight

    return get_tonight().to_dict()


def tool_add_to_tonight(name: str) -> dict:
    """Add a player to tonight's attendee list. Creates roster entry if new."""
    from session_state import add_to_tonight, get_tonight

    state = get_tonight()
    if not state.attendees and not state.court_labels:
        return {
            "ok": False,
            "error": "no_session",
            "message": "No session in flight — call start_tonight first.",
        }
    roster = Roster()
    roster_new: list = []
    if roster.get(name) is None:
        roster_new = roster.add_many_from_cr([name])
    state = add_to_tonight(name)
    return {"ok": True, "state": state.to_dict(), "new_players_added": roster_new}


def tool_remove_from_tonight(name: str) -> dict:
    """Remove a player from tonight's attendee list. Fuzzy-matches the name."""
    from session_state import find_attendee_fuzzy, get_tonight, remove_from_tonight

    state = get_tonight()
    if name in state.attendees:
        match = name
    else:
        candidates = find_attendee_fuzzy(name)
        if not candidates:
            return {"ok": False, "error": "not_found", "query": name}
        if len(candidates) > 1:
            return {
                "ok": False,
                "error": "ambiguous",
                "query": name,
                "candidates": candidates,
            }
        match = candidates[0]
    state, removed = remove_from_tonight(match)
    return {"ok": True, "removed": removed, "state": state.to_dict()}


def tool_set_courts_for_tonight(court_labels: list) -> dict:
    """Set the court labels (e.g. [7, 8, 9, 10]) Boris should use tonight."""
    from session_state import set_courts_for_tonight

    state = set_courts_for_tonight(court_labels)
    return {"ok": True, "state": state.to_dict()}


def tool_clear_tonight() -> dict:
    """Clear the current session state."""
    from session_state import clear_tonight

    state = clear_tonight()
    return {"ok": True, "state": state.to_dict()}


TOOL_IMPLS: dict[str, Any] = {
    "list_club_sessions": tool_list_club_sessions,
    "get_session_registrants": tool_get_session_registrants,
    "read_players_roster": tool_read_players_roster,
    "set_player_rating": tool_set_player_rating,
    "read_pairings_history": tool_read_pairings_history,
    "generate_pairings": tool_generate_pairings,
    "log_pairings_to_sheet": tool_log_pairings_to_sheet,
    "start_tonight": tool_start_tonight,
    "get_tonight": tool_get_tonight,
    "add_to_tonight": tool_add_to_tonight,
    "remove_from_tonight": tool_remove_from_tonight,
    "set_courts_for_tonight": tool_set_courts_for_tonight,
    "promote_from_waitlist": tool_promote_from_waitlist,
    "clear_tonight": tool_clear_tonight,
}


TOOL_SCHEMAS: list[dict] = [
    {
        "name": "list_club_sessions",
        "description": "List upcoming CourtReserve sessions (category defaults to "
        "'Social & Club Sessions'). Filter by day_of_week (e.g. 'Thursday'), "
        "days_ahead (default 14), or a name substring. Each result includes a "
        "reservation_number — pass that to get_session_registrants to see who's "
        "signed up. Takes ~5–8 s (browser navigation).",
        "input_schema": {
            "type": "object",
            "properties": {
                "day_of_week": {
                    "type": "string",
                    "description": "Day-of-week filter, e.g. 'Thursday', 'Tuesday'. "
                    "If the admin gives only a day (no date), use this.",
                },
                "days_ahead": {
                    "type": "integer",
                    "description": "Max days from today. Default 14.",
                    "default": 14,
                },
                "name_contains": {
                    "type": "string",
                    "description": "Optional substring match against event name.",
                },
                "category": {
                    "type": "string",
                    "description": "Category substring. Default 'Social & Club Sessions'. "
                    "Pass an empty string to include all categories.",
                    "default": "Social & Club Sessions",
                },
            },
        },
    },
    {
        "name": "get_session_registrants",
        "description": "Fetch the registrant list AND the waitlist (in CourtReserve "
        "priority order) for a session, plus event metadata (date, courts, spots, "
        "is_full). Auto-adds any previously-unseen names — from BOTH lists — to the "
        "roster (rating '?', gender guessed). Use the waitlist to discuss with the "
        "admin which players will be promoted given any extra courts available. "
        "Takes ~5–8 s.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reservation_number": {
                    "type": "string",
                    "description": "The reservation_number from list_club_sessions, "
                    "e.g. 'V1PCW1S2146898'.",
                }
            },
            "required": ["reservation_number"],
        },
    },
    {
        "name": "read_players_roster",
        "description": "Return the full player roster: a mapping of full name -> "
        "{gender, rating, notes}. Rating is an integer 1-5 or the string '?'.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_player_rating",
        "description": "Set a player's 1-5 rating. The name can be partial; the tool "
        "fuzzy-matches — if the query matches multiple players the response's "
        "`error` will be 'ambiguous' with a list of candidates for you to clarify "
        "with the admin. Pass the string '?' to reset a rating to unknown.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Player name (full or partial). Case-insensitive.",
                },
                "rating": {
                    "description": "Integer 1-5, or the string '?'.",
                },
            },
            "required": ["name", "rating"],
        },
    },
    {
        "name": "read_pairings_history",
        "description": "Return the last N weeks of historical pairing plans from "
        "history.json, most recent last.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lookback": {
                    "type": "integer",
                    "description": "How many past weeks to include. Default 4.",
                    "default": 4,
                }
            },
        },
    },
    {
        "name": "generate_pairings",
        "description": "Build a pairing plan. When called with no attendee_names / "
        "court_labels arguments, uses the current session state (set via "
        "start_tonight + set_courts_for_tonight). Returns the structured plan "
        "including rotations, courts (doubles or singles), pairs, sit-outs, and a "
        "display_names map — use those short forms when writing WhatsApp replies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "attendee_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional — overrides session state.",
                },
                "court_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific court labels "
                    "(e.g. ['4','5','6','7']) to override session state.",
                },
                "num_courts": {
                    "type": "integer",
                    "description": "Legacy count of courts — produces labels '1','2',...",
                },
                "num_rotations": {
                    "type": "integer",
                    "description": "Number of rotation blocks (default 3).",
                    "default": 3,
                },
                "seed": {
                    "type": "integer",
                    "description": "Optional RNG seed for reproducibility.",
                },
                "start_time_hhmm": {
                    "type": "string",
                    "description": "First rotation start, HH:MM. Default 19:30.",
                    "default": "19:30",
                },
                "rotation_minutes": {
                    "type": "integer",
                    "description": "Minutes per rotation. Default 40.",
                    "default": 40,
                },
                "log_to_sheet": {
                    "type": "boolean",
                    "description": "If true, also append to the 'Session log' and "
                    "'Pair log' tabs. Default false — only set true when the admin "
                    "has confirmed the pairings are final.",
                    "default": False,
                },
            },
        },
    },
    {
        "name": "start_tonight",
        "description": "Initialise (or overwrite) tonight's session state. Use "
        "reservation_number to pull the attendee list from CourtReserve, OR pass "
        "attendee_names for a manual list. Optionally set court_labels at the same "
        "time (can also be done later with set_courts_for_tonight). Any unseen "
        "names are auto-added to the roster.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reservation_number": {
                    "type": "string",
                    "description": "CourtReserve reservation_number from list_club_sessions.",
                },
                "attendee_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Full names — alternative to reservation_number.",
                },
                "date": {
                    "type": "string",
                    "description": "Optional date string for bookkeeping.",
                },
                "court_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional court labels to seed the state.",
                },
            },
        },
    },
    {
        "name": "get_tonight",
        "description": "Return the current session state: date, attendees, "
        "court_labels. Empty lists if no session in flight.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "add_to_tonight",
        "description": "Append a player to tonight's attendee list. If the name is "
        "new to the roster, adds them (rating '?', gender guessed).",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full player name."}
            },
            "required": ["name"],
        },
    },
    {
        "name": "remove_from_tonight",
        "description": "Remove a player from tonight's attendee list. Fuzzy-matches "
        "— returns error 'ambiguous' with candidates if the query matches multiple.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name or partial name."}
            },
            "required": ["name"],
        },
    },
    {
        "name": "promote_from_waitlist",
        "description": "Move a CourtReserve-waitlisted player into tonight's "
        "attendee list. Use after the admin has confirmed extra courts are "
        "available beyond what CourtReserve knows about and has decided which "
        "waitlisted players will play. Fuzzy-matches the name; ambiguous "
        "matches return error 'ambiguous' with candidates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name (full or partial). Matched against the "
                    "current session_state.waitlist.",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "set_courts_for_tonight",
        "description": "Set the list of court labels (e.g. [7, 8, 9, 10]) that will "
        "be used tonight. Overwrites any previously-set courts for this session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "court_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Court labels in the order given by the admin.",
                }
            },
            "required": ["court_labels"],
        },
    },
    {
        "name": "clear_tonight",
        "description": "Wipe the current session state. Use when the admin wants to "
        "start from scratch mid-session.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "log_pairings_to_sheet",
        "description": "Append a previously-generated plan to the Session/Pair-log "
        "tabs in Google Sheets. Use this if the admin approves a plan AFTER it was "
        "generated with log_to_sheet=false.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "object",
                    "description": "The full plan dict returned by generate_pairings.",
                }
            },
            "required": ["plan"],
        },
    },
]


# ---------- Claude agent loop -------------------------------------------


def run_agent(client: Anthropic, user_text: str) -> tuple[str, dict]:
    """Run a Claude tool-use loop. Returns ``(final_text, usage)`` where
    ``usage`` is a dict of accumulated token counts across all turns.
    """
    today = datetime.now().strftime("%A %Y-%m-%d")
    messages: list[dict] = [
        {
            "role": "user",
            "content": f"Today is {today}.\nAdmin command: {user_text}",
        }
    ]
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    for _ in range(AGENT_LOOP_MAX_TURNS):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        # Accumulate token usage across all turns of the loop.
        u = resp.usage
        usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
        usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
        usage["cache_read_input_tokens"] += (
            getattr(u, "cache_read_input_tokens", 0) or 0
        )
        usage["cache_creation_input_tokens"] += (
            getattr(u, "cache_creation_input_tokens", 0) or 0
        )

        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "end_turn":
            text = "\n".join(
                block.text for block in resp.content if block.type == "text"
            ).strip()
            return text, usage
        if resp.stop_reason != "tool_use":
            return f"[unexpected stop_reason: {resp.stop_reason}]", usage
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            try:
                fn = TOOL_IMPLS[block.name]
                result = fn(**(block.input or {}))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )
            except Exception as e:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"error": str(e)}),
                        "is_error": True,
                    }
                )
        messages.append({"role": "user", "content": tool_results})
    return "[agent loop exceeded max turns]", usage


# ---------- WhatsApp send ------------------------------------------------


def send_to_group(group_jid: str, text: str) -> bool:
    """Send ``text`` to the WhatsApp group ``group_jid`` via the bridge."""
    if not text:
        text = "(no reply from bot)"
    r = requests.post(
        f"{BRIDGE_URL}/send",
        json={"recipient": group_jid, "message": text},
        timeout=15,
    )
    if r.status_code != 200:
        print(f"send failed: {r.status_code} {r.text}", file=sys.stderr)
        return False
    return True


# ---------- main polling loop -------------------------------------------


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
        m = BOT_TRIGGER_PATTERN.match(content)
        if not m:
            continue
        command = content[m.end():].strip()
        out.append((msg_id, ts, sender or "", command))
    return out


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    group_jids = find_admin_group_jids()  # {name: jid}
    print("Watching admin groups:")
    for name, jid in group_jids.items():
        print(f"  - {name!r}  ({jid})")
    print(
        f"Poll interval: {POLL_INTERVAL_SECONDS}s. "
        'Trigger: messages starting with "boris" or "bot" '
        "(case-insensitive, colon/?/! tolerated)."
    )

    # One watermark per group — start at the latest timestamp seen in the
    # bridge so historical messages aren't re-processed.
    with sqlite3.connect(BRIDGE_DB) as conn:
        row = conn.execute("SELECT MAX(timestamp) FROM messages").fetchone()
    initial_watermark = row[0] if row and row[0] else datetime.now().isoformat()
    watermarks: dict[str, str] = {jid: initial_watermark for jid in group_jids.values()}
    print(f"Initial watermark: {initial_watermark}\n")

    client = Anthropic()

    while True:
        for group_name, group_jid in group_jids.items():
            try:
                new_msgs = fetch_triggered_messages(group_jid, watermarks[group_jid])
            except Exception as e:
                print(f"[{group_name}] poll error: {e}", file=sys.stderr)
                continue

            for msg_id, ts, sender, command in new_msgs:
                watermarks[group_jid] = ts  # advance even on errors
                print(f"[{ts}] [{group_name}] {sender}: {command!r}")

                # Optional "Working on it" after 5 s of quiet.
                done = threading.Event()

                def _send_working_on_it(jid=group_jid) -> None:
                    if done.is_set():
                        return
                    send_to_group(jid, BOT_REPLY_PREFIX + WORKING_ON_IT_TEXT)

                timer = threading.Timer(
                    WORKING_ON_IT_DELAY_SECONDS, _send_working_on_it
                )
                timer.daemon = True
                timer.start()
                usage: dict = {}
                try:
                    if command:
                        reply_body, usage = run_agent(client, command)
                    else:
                        reply_body = "(empty command — say e.g. 'boris help')"
                except Exception as e:
                    reply_body = f"(bot error: {e})"
                finally:
                    done.set()
                    timer.cancel()

                reply = BOT_REPLY_PREFIX + reply_body
                print(f"  -> reply: {reply[:200]}{'…' if len(reply) > 200 else ''}")
                send_to_group(group_jid, reply)

                # Per-command usage logging (best-effort; never crashes the loop).
                if usage:
                    try:
                        from usage_log import log_usage

                        cost = log_usage(
                            group=group_name,
                            sender=sender,
                            command=command,
                            usage=usage,
                            model=MODEL,
                        )
                        print(f"  -> usage: in={usage['input_tokens']} "
                              f"cache={usage['cache_read_input_tokens']} "
                              f"out={usage['output_tokens']}  ${cost:.4f}")
                    except Exception as e:
                        print(f"  -> usage log error: {e}", file=sys.stderr)

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
