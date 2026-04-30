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
- set_singles_preference: set 'avoid' / 'prefer' / 'neutral' (fuzzy name).

HISTORY + PAIRINGS:
- read_pairings_history: past weeks' plans.
- generate_pairings: when called with no args, uses the session state.
  Refuses if the session is missing attendees or court_labels. Accepts
  per-session `singles_exclude` / `singles_include` lists when the
  admin attaches ad-hoc instructions like "don't put Geoff in singles
  tonight" — these don't change the roster. The result is saved as
  the session's draft plan; subsequent swap_players / swap_rotations /
  commit_plan all act on that draft.
- swap_players(name1, name2, rotation_num?): edit the draft — swap
  two players' slots. Omit rotation_num to swap their whole evening.
- swap_rotations(a, b): edit the draft — swap two rotations' contents
  (times stay tied to position).
- swap_courts(label_a, label_b): edit the draft — swap the matchups on
  two courts across every rotation (labels stay put). Use for "put
  singles on Ct N" / "move courts X and Y".
- commit_plan: finalise the draft → appends to history.json AND mirrors
  to the Sheet log tabs, then clears the draft. Use this when the
  admin approves ("use those" / "save" / "log it" / "final").
- log_pairings_to_sheet: escape hatch — log an arbitrary plan dict.
  Prefer commit_plan for the normal flow.

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
5. Format pairings for mobile WhatsApp (narrow screens). Always start
   the message with this exact two-line preamble, with the date drawn
   from the plan's `date` field formatted as "Thursday Dth Month" (e.g.
   "Thursday 30th April"). The second sentence MUST be on its own line:

     Here are the pairings for Thursday 30th April.
     At the end of each rotation please finish your game within a minute or two, if need be using a "next point wins" option.

   Then for each rotation, render a header line with the full time
   range `(start-end)` from the rotation's `start_time` and `end_time`
   fields, immediately followed (NO blank line) by one line per court:
   `Ct N: A & B v C & D` (doubles) or `Ct N: A v B` (singles). Do NOT
   append `(doubles)` / `(singles)` tags. One court per line, no blank
   lines between consecutive courts. Use `display_names` verbatim —
   most players appear as just a first name; only those needing
   disambiguation get a surname initial.

   Default = NO ratings. The plan's `ratings` map is for your reference
   only — do NOT include rating numbers in the output unless the admin
   explicitly asks ("with ratings", "show ratings", "include ratings",
   "draft with ratings", etc.). Default output looks like:

     Rotation 1 (19:30-20:15)
     Ct 4: Geoff & Silvia v Paul V & Hannah
     Ct 5: ...
     Ct 6: David v Jack

   When the admin explicitly asks for ratings, insert the pair-rating
   sums in `[a v b]` form between `Ct N` and the colon, using the
   `ratings` map (rating "?" counts as 3). For singles, the bracket
   holds the two individual ratings:

     Rotation 1 (19:30-20:15)
     Ct 4 [5 v 6]: Geoff & Silvia v Paul V & Hannah
     Ct 5 [4 v 5]: ...
     Ct 6 [3 v 2]: David v Jack

   Separate consecutive rotations with a blank line. Include sit-outs
   (if any) and the plan's `notes` after the last rotation.
6. Mid-iteration: if the admin asks to tweak the draft ("swap Patrick
   and Geoff", "switch rotations 1 and 2"), call swap_players or
   swap_rotations — these mutate the saved draft in session state and
   return the updated plan. Render the updated plan to the admin in the
   same DRAFT format as before. Do NOT call generate_pairings again
   unless the admin explicitly asks for a fresh re-roll.
7. When the admin confirms ("use those" / "save" / "log it" / "final"),
   call commit_plan (no args). It appends to history.json AND mirrors
   to the Sheet log tabs and clears the draft. Confirmation does NOT
   change the rendering — keep ratings out unless they were explicitly
   asked for in this conversation.

Handling day-only references ("who's signed up for Thursday?")
--------------------------------------------------------------
Use list_club_sessions(day_of_week=<day>) and take the earliest upcoming
match. If multiple distinct sessions match on that day, ask the admin to
clarify.

Rating updates ("set rating for X to N")
----------------------------------------
Call set_player_rating directly; if the tool returns 'ambiguous', show
the candidates and ask.

Singles-preference updates
--------------------------
"X doesn't want singles" / "X prefers singles" / "reset X to neutral on
singles" → call set_singles_preference with preference 'avoid', 'prefer',
or 'neutral'. Same fuzzy-match-and-disambiguate flow as ratings.

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


def tool_set_singles_preference(name: str, preference: str) -> dict:
    """Update a player's singles-court preference. Fuzzy-matches the name.

    ``preference`` must be ``"avoid"`` (don't pick for singles unless
    forced), ``"prefer"`` (pick for singles first), or ``""`` / ``"neutral"``
    (default — clear any existing preference).
    """
    pref = (preference or "").strip().lower()
    if pref == "neutral":
        pref = ""
    if pref not in {"", "avoid", "prefer"}:
        return {
            "ok": False,
            "error": "invalid_preference",
            "message": "preference must be 'avoid', 'prefer', or 'neutral'",
        }
    roster = Roster()
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
    entry = roster.set_singles(name, pref)
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
    rotation_durations: Optional[list[int]] = None,
    singles_exclude: Optional[list[str]] = None,
    singles_include: Optional[list[str]] = None,
) -> dict:
    """Build a pairing plan and stash it as the session's draft.

    If ``attendee_names`` is None, uses the current session state (what
    the admin has set via start_tonight / add / remove). Same for
    ``court_labels``. The plan is saved to ``session_state.draft_plan``
    so the admin can iterate (swap_players / swap_rotations) before
    finalising via commit_plan.
    """
    from pairings import make_plan
    from session_state import get_tonight, set_draft_plan

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
            rotation_durations=rotation_durations,
            singles_exclude=singles_exclude,
            singles_include=singles_include,
            seed=seed,
        )
    except ValueError as e:
        return {"ok": False, "error": "over_capacity_or_bad_input", "message": str(e)}
    result = plan.to_dict()
    set_draft_plan(result)
    return result


def tool_log_pairings_to_sheet(plan: dict) -> dict:
    """Escape hatch: append a plan dict to the Session/Pair-log tabs.

    Prefer ``commit_plan`` for the normal admin flow — it appends to
    ``history.json`` AND logs to the Sheet, then clears the draft.
    """
    from session_log import log_plan

    return log_plan(plan)


def tool_swap_players(
    name1: str,
    name2: str,
    rotation_num: Optional[int] = None,
) -> dict:
    """Swap the schedule slots of two players in the current draft plan.

    If ``rotation_num`` is omitted, the swap is applied in every rotation
    where both players appear (typical "swap their evening" intent).
    """
    from pairings import swap_players_in_plan
    from session_state import get_draft_plan, set_draft_plan

    plan = get_draft_plan()
    if not plan:
        return {
            "ok": False,
            "error": "no_draft",
            "message": "No draft plan in session state — run generate_pairings first.",
        }
    try:
        swapped = swap_players_in_plan(plan, name1, name2, rotation_num)
    except (KeyError, ValueError) as e:
        return {"ok": False, "error": "swap_failed", "message": str(e)}
    set_draft_plan(plan)
    return {"ok": True, "rotations_changed": swapped, "plan": plan}


def tool_swap_courts(label_a: str, label_b: str) -> dict:
    """Swap the contents of two courts in the current draft plan.

    Use for admin requests like 'put singles on court 5' (then swap the
    current singles court with court 5) or 'move courts 7 and 9'. Court
    labels stay put — only the matchups move — so pinned slots elsewhere
    in the plan are unaffected.
    """
    from pairings import swap_courts_in_plan
    from session_state import get_draft_plan, set_draft_plan

    plan = get_draft_plan()
    if not plan:
        return {
            "ok": False,
            "error": "no_draft",
            "message": "No draft plan in session state — run generate_pairings first.",
        }
    try:
        swap_courts_in_plan(plan, label_a, label_b)
    except ValueError as e:
        return {"ok": False, "error": "swap_failed", "message": str(e)}
    set_draft_plan(plan)
    return {"ok": True, "swapped": [str(label_a), str(label_b)], "plan": plan}


def tool_swap_rotations(a: int, b: int) -> dict:
    """Swap the contents of two rotations in the current draft plan.

    Times stay tied to position — rotation 1 always runs at the first
    time slot, etc.
    """
    from pairings import swap_rotations_in_plan
    from session_state import get_draft_plan, set_draft_plan

    plan = get_draft_plan()
    if not plan:
        return {
            "ok": False,
            "error": "no_draft",
            "message": "No draft plan in session state — run generate_pairings first.",
        }
    try:
        swap_rotations_in_plan(plan, a, b)
    except ValueError as e:
        return {"ok": False, "error": "swap_failed", "message": str(e)}
    set_draft_plan(plan)
    return {"ok": True, "swapped": [a, b], "plan": plan}


def tool_commit_plan() -> dict:
    """Finalise the current draft plan.

    Appends to ``history.json`` (so the next session's pairing engine
    avoids repeating tonight's pairs) AND mirrors to the Google Sheet
    ``Session log`` / ``Pair log`` tabs. Clears the draft from session
    state on success. Call this when the admin says "use those" /
    "save" / "log it" / "final".
    """
    from pairings import append_to_history
    from session_state import clear_draft_plan, get_draft_plan
    from session_log import log_plan

    plan = get_draft_plan()
    if not plan:
        return {
            "ok": False,
            "error": "no_draft",
            "message": "No draft plan to commit — run generate_pairings first.",
        }
    try:
        append_to_history(plan, str(HISTORY_PATH))
    except Exception as e:
        return {
            "ok": False,
            "error": "history_write_failed",
            "message": str(e),
        }
    sheet_log = None
    sheet_error = None
    try:
        sheet_log = log_plan(plan)
    except Exception as e:
        sheet_error = str(e)
    clear_draft_plan()
    return {
        "ok": True,
        "history_appended": True,
        "sheet_log": sheet_log,
        "sheet_error": sheet_error,
    }


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
    "set_singles_preference": tool_set_singles_preference,
    "read_pairings_history": tool_read_pairings_history,
    "generate_pairings": tool_generate_pairings,
    "swap_players": tool_swap_players,
    "swap_rotations": tool_swap_rotations,
    "swap_courts": tool_swap_courts,
    "commit_plan": tool_commit_plan,
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
        "name": "set_singles_preference",
        "description": "Set a player's singles-court preference. The name can "
        "be partial; the tool fuzzy-matches — ambiguous matches return an "
        "error with candidates so you can clarify with the admin. Use this "
        "for messages like 'Tim doesn't want singles' / 'Fernando prefers "
        "singles' / 'reset Geoff to neutral on singles'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Player name (full or partial).",
                },
                "preference": {
                    "type": "string",
                    "enum": ["avoid", "prefer", "neutral", ""],
                    "description": "'avoid' = don't pick for singles unless "
                    "forced; 'prefer' = pick for singles first; 'neutral' (or "
                    "empty string) = clear any preference.",
                },
            },
            "required": ["name", "preference"],
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
                "rotation_durations": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Per-rotation lengths in minutes, e.g. [45,40,35]. "
                    "Length must equal num_rotations. If omitted, defaults to "
                    "[45,40,35] for 3 rotations (the club standard) else 40 each.",
                },
                "singles_exclude": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names to keep OFF the singles courts for this "
                    "session only (no roster change). Use for ad-hoc admin "
                    "instructions like 'don't put Geoff in singles tonight'. "
                    "Treated as if their roster preference were 'avoid'.",
                },
                "singles_include": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names to bias TOWARDS singles courts for this "
                    "session only (no roster change). Use for ad-hoc admin "
                    "instructions like 'try to put Tim on singles tonight'. "
                    "Treated as if their roster preference were 'prefer'.",
                },
            },
        },
    },
    {
        "name": "swap_players",
        "description": "Edit the current draft plan: swap the schedule slots "
        "of two players. If rotation_num is omitted, applies to every "
        "rotation where both players appear. Use for admin requests like "
        "'swap Patrick and Geoff' or 'swap Tim with Hannah in rotation 2'. "
        "Requires generate_pairings to have been run first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name1": {"type": "string", "description": "First player full name."},
                "name2": {"type": "string", "description": "Second player full name."},
                "rotation_num": {
                    "type": "integer",
                    "description": "Optional 1-indexed rotation. Omit to swap "
                    "across the whole evening.",
                },
            },
            "required": ["name1", "name2"],
        },
    },
    {
        "name": "swap_rotations",
        "description": "Edit the current draft plan: swap the courts and "
        "sit-outs of two rotations. Times stay tied to position — "
        "rotation 1 still runs at the first time slot. Use for admin "
        "requests like 'swap rotation 1 with rotation 2'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First rotation (1-indexed)."},
                "b": {"type": "integer", "description": "Second rotation (1-indexed)."},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "swap_courts",
        "description": "Edit the current draft plan: swap the matchups on "
        "two courts across every rotation. Court labels stay put — only "
        "the players/pairs/mode move — so other pinned slots are "
        "unaffected. Use for admin requests like 'put singles on Ct 5' "
        "(swap the current singles court with court 5) or 'move courts "
        "7 and 9'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "label_a": {
                    "type": "string",
                    "description": "First court label (e.g. '5').",
                },
                "label_b": {
                    "type": "string",
                    "description": "Second court label (e.g. '11').",
                },
            },
            "required": ["label_a", "label_b"],
        },
    },
    {
        "name": "commit_plan",
        "description": "Finalise the current draft plan when the admin "
        "approves ('use those' / 'save' / 'log it' / 'final'). Appends "
        "to history.json AND the Sheet's Session/Pair log tabs, then "
        "clears the draft. Takes no arguments — uses the draft saved "
        "by the last generate_pairings call (and any subsequent "
        "swap_players / swap_rotations edits).",
        "input_schema": {"type": "object", "properties": {}},
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

    # Running totals for the lifetime of this admin_bot process.
    session_totals = {
        "commands": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cost": 0.0,
    }

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
                        session_totals["commands"] += 1
                        for k in ("input_tokens", "output_tokens",
                                  "cache_read_input_tokens",
                                  "cache_creation_input_tokens"):
                            session_totals[k] += usage[k]
                        session_totals["cost"] += cost
                        print(f"  -> usage: in={usage['input_tokens']} "
                              f"cache={usage['cache_read_input_tokens']} "
                              f"out={usage['output_tokens']}  ${cost:.4f}")
                        print(f"  -> session total ({session_totals['commands']} cmds): "
                              f"in={session_totals['input_tokens']} "
                              f"cache={session_totals['cache_read_input_tokens']} "
                              f"out={session_totals['output_tokens']}  "
                              f"${session_totals['cost']:.4f}")
                    except Exception as e:
                        print(f"  -> usage log error: {e}", file=sys.stderr)

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
