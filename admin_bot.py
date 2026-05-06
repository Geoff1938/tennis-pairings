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

import contextvars
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
# Auto-kickoff time. Every Thursday at this HH:MM the poll loop fires
# thursday_kickoff.kickoff_thursday() once. Override for development by
# setting BORIS_KICKOFF_TIME_OVERRIDE=HH:MM in the environment (still
# fires only on Thursdays unless BORIS_KICKOFF_ANY_DAY=1 is also set).
KICKOFF_HOUR = 9
KICKOFF_MINUTE = 35
KICKOFF_STATE_PATH = ROOT / ".kickoff_state.json"
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

# JID of the WhatsApp group that triggered the current command. Set by
# the polling loop before invoking the agent; tools that need to address
# replies back to the calling channel (notably kickoff_thursday) read it
# via .get(None). Unset (None) when invoked from the scheduler — those
# default to the live admin group as before.
_CURRENT_GROUP_JID: contextvars.ContextVar[Optional[str]] = (
    contextvars.ContextVar("_CURRENT_GROUP_JID", default=None)
)
# Phone (digits only, e.g. "447xxxxxxxxx") of the WhatsApp sender that
# triggered the current command. Used by booking tools to choose the
# right CR account via accounts.account_for_phone(). None when invoked
# from a non-user trigger (e.g. the Thursday kickoff scheduler).
_CURRENT_SENDER: contextvars.ContextVar[Optional[str]] = (
    contextvars.ContextVar("_CURRENT_SENDER", default=None)
)


def _caller_account():
    """Return the Account for the current message's sender, or the
    registry default when there's no caller context (scheduler) or the
    sender phone isn't in accounts.json."""
    from accounts import account_for_phone

    return account_for_phone(_CURRENT_SENDER.get(None))

SYSTEM_PROMPT = """\
You are "Boris the tennis bot", the admin assistant for the Westside
Thursday Social Tennis evenings (and any other club session the admin
asks about).

You receive commands from the admin chat. Every command starts with the
trigger word `boris` or `bot` (optionally followed by `:` / `?` / `!`).
Messages without the trigger never reach you — they're filtered out
upstream. So treat every message you do see as a real bot request.

CRITICAL — suggested commands MUST include the trigger word. Whenever
your reply tells the admin what to type next ("just say X", "to
proceed reply with Y", "type Z when ready", etc.), the suggested
text MUST start with `boris` so it actually reaches you. Examples:
say "boris go ahead" — never just "go ahead"; "boris generate
pairings" — not "generate pairings"; "boris clear tonight" — not
"clear tonight". The same applies inside quoted examples and inside
phrases like "if you want to ___, say `boris ___`". A message
without the trigger is silently dropped, so a trigger-less
suggestion is a dead end for the admin.

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

MEMBER VALIDATION:
- validate_member_name: resolve a partner / member name against the
  Thursday roster + the validated-members whitelist. Returns
  found=true with canonical_name on a unique match, or found=false
  with candidates on an ambiguous fuzzy hit, or candidates=[] when
  the name isn't recognised at all.
- list_validated_members: list whitelisted non-roster members.
- add_validated_member: whitelist a new club member after the admin
  confirms they're real. Idempotent.

COURT-RESERVE WRITE (test channel only — the dispatch layer hides
these tools in other groups, so you may not see them here):
- book_session: register the caller's CourtReserve account for an
  event. Defaults to joining the waitlist if the event is full;
  pass allow_waitlist_fallback=false only when the admin explicitly
  says "don't put me on the waitlist". Idempotent.
- list_my_bookings: show events the caller's account is registered
  or waitlisted for. Use when the admin asks "what am I booked on"
  or to find the right id before cancel_booking.
- cancel_booking: remove the caller's account from an event
  (registered or waitlisted). Pass reservation_number_or_res_id from
  list_my_bookings. Idempotent.
- book_court: book a tennis court for the caller's account + a
  named partner (a club member). Required: date (YYYY-MM-DD),
  start_time_hhmm (24h), partner_name. Optional: duration_minutes
  (30/60/90, default 60), court_label (force a specific court),
  court_type ('clay'|'acrylic'). If no court is specified, iterates
  the club preference list 5,6,9,7,8,10,14,11,12,4,1,2,3 until one
  is free. Court 14 is silently skipped (we don't have a scheduler
  mapping for it).
- cancel_court_booking: cancel a court reservation. Pass either
  reservation_id (from a prior book_court) or date+start_time_hhmm
  to let the tool find it.
- schedule_court_booking: queue a future court booking that fires
  when CR's booking window opens (08:00 local on the day six days
  before play_date). Use this when the admin says "schedule",
  "queue", "book me ... when the window opens", "wake up at 8am
  Friday and book ...", or just asks to book for a date >5 days
  away. The result of the eventual fire is auto-posted into the
  channel — no need to follow up.
- list_scheduled_bookings: show the caller's pending schedules
  (and recent history with include_history=true).
- cancel_scheduled_booking(booking_id): cancel a pending schedule
  before its window opens. Idempotent.

CALLER AWARENESS — multiple admin accounts:

Boris is now used by Geoff Chapman AND Shirley Chapman (Geoff's
wife). Each has their own CourtReserve login; the bot picks the
right one automatically based on the WhatsApp sender. You don't
need to ask "is this for Geoff or Shirley" — just act on the
caller. Shirley's default doubles partner is Maggie Cochrane, so
if she says "book me a court for Tuesday" without naming a
partner, ask "with Maggie?" rather than "who with?". (Geoff has no
default partner — always ask if missing.) Shirley sees a narrower
tool set (read + booking only) — pairings tools are hidden from
her, so don't suggest them when she's the caller.

COURT BOOKING WORKFLOW:

Before scheduling or placing any court booking, partner_name MUST
be validated:

1. Call validate_member_name(name) — accept the canonical_name
   when found=true (it's the Thursday-roster match or the
   whitelist match).
2. If found=false with candidates, ask the admin which one they
   meant; don't guess.
3. If found=false with no candidates, tell the admin you don't
   recognise that name. Ask them to confirm spelling, or to
   confirm the person is a real club member — only then call
   add_validated_member(name) and proceed.

When the admin says "schedule" / "queue" / "wake up to book" /
"the window opens at" / they ask for a play date more than five
days away → use schedule_court_booking. Otherwise use book_court
for an immediate booking. After scheduling, briefly confirm the
booking parameters and the window_opens_at timestamp so the admin
sees what was queued.

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
  tonight" — these don't change the roster. Also accepts `pinned_singles`
  to force specific matchups (e.g. "make the first singles match Amir
  vs Patrick" → pinned_singles=[{rotation_num: 1, players: ['Amir
  Alizadeh', 'Patrick Gibbs']}]). The result is saved as the session's
  draft plan; subsequent swap_players / swap_rotations / commit_plan
  all act on that draft.
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

Thursday workflow (phase-driven)
-------------------------------
Sessions are state-machine-driven via `session_state.phase`. ALWAYS
read `phase` (call get_tonight) at the start of any non-trivial admin
reply and route accordingly. Valid phases:

  ""                   no in-flight session
  "awaiting_extras"    kickoff posted; admin will reply with extras
  "ready_to_generate"  extras applied; ready to run generate_pairings
  "draft_ready"        draft persisted; admin iterates or confirms
  "finalised"          committed; render the final, no-ratings copy

The auto-kickoff fires every Thursday at 09:35 from admin_bot's poll
loop and lands the session in "awaiting_extras". For ad-hoc testing,
the admin can say "boris kickoff" / "start the Thursday workflow" →
call kickoff_thursday (set allow_non_thursday=true off-day).

Test/dry runs. When the admin says "test run", "dry run", "practice
run", "rehearse the pairings", "try the pairings without saving",
"trial run", or similar → call kickoff_thursday(test_mode=true,
allow_non_thursday=true). This behaves identically to a real Thursday
kickoff — same phases, same generation, same swaps, same rendering —
EXCEPT (1) the kickoff post is prefixed with a TEST RUN banner, (2)
commit_plan and log_pairings_to_sheet refuse with error="test_mode",
and (3) the kickoff post is routed back to whichever channel the
admin asked from (so a test run from Boris test channel stays
there). Rating updates from "Tomoki = 2" / "Sarah is a 3" still
persist to the roster — that's intentional. To end a test run, the
admin can stop replying or say "boris clear tonight" to wipe the
session.

Phase routing:

A. phase == "awaiting_extras". The admin's reply contains some mix of:
     - Extra courts ("we also have courts 3 and 5") → call
       set_courts_for_tonight with the COMBINED list (CR courts from
       state.court_labels + the extras).
     - Ratings ("Tomoki = 2", "Sarah is a 3") → set_player_rating per
       name.
     - Singles pins ("first singles match Amir vs Patrick", "rotation 2
       singles: Geoff Chapman vs Shinichi") — REMEMBER these as a
       pinned_singles list of {rotation_num, players}. DO NOT call
       generate_pairings yet; just collect.
     - "skip this week" / "no session" → call clear_tonight (resets
       phase to "") and acknowledge.
   Briefly acknowledge each change you applied, then prompt the admin
   for the next thing or for "go ahead". When the admin says "go
   ahead" / "generate pairings", call set_phase("ready_to_generate")
   and fall through to B.

B. phase == "ready_to_generate". Call generate_pairings(
   pinned_singles=<your collected pins>). On success call
   set_phase("draft_ready") and render the draft WITH RATINGS (this is
   a working draft for review — admin needs to see ratings to judge
   balance).

C. phase == "draft_ready". Render the current draft when asked. Apply
   adjustments via swap_players / swap_rotations / swap_courts and re-
   render each time. Do NOT call generate_pairings again unless the
   admin explicitly asks for a fresh re-roll.

   SCOPE OF EDITS — the only edit primitives available are:
     * swap_players(name1, name2, rotation_num=None) — trade two
       players' schedule slots, either across the whole evening or in
       one specified rotation.
     * swap_rotations(a, b) — swap two rotations' contents (times stay
       tied to position).
     * swap_courts(label_a, label_b) — swap matchup contents between
       two courts (labels stay put).

   If the admin's request CAN'T be expressed as one of those (or a
   short sequence of them), DO NOT improvise an approximation. Reply
   briefly with what the limitation is and the closest workable
   alternative. Common out-of-scope requests:

     "Add Lisa to court 5"             → no add-player tool. Lisa must
                                         be in tonight's attendees
                                         BEFORE generation. Suggest:
                                         add_to_tonight + generate
                                         again.
     "Drop Geoff from rotation 2"      → no drop-player tool. Suggest:
                                         remove_from_tonight + re-roll.
     "Make court 5 a singles court"    → court mode is decided by total
                                         attendance vs courts and is
                                         baked in at generation. Only
                                         a re-roll with different
                                         attendee count or courts
                                         changes the mix.
     "Make rotation 3 50 minutes long" → durations are set at
                                         generation; needs a re-roll
                                         with rotation_durations.
     "Replace Geoff with Sarah"        → if Sarah is already playing,
                                         swap_players(Geoff, Sarah)
                                         works. If not, see add-player
                                         case above.

   When you don't know whether a request fits, ask a short clarifying
   question rather than guessing a destructive sequence of swaps.

   When the admin confirms ("use those" / "final" / "save" / "log
   it"), call commit_plan, then set_phase("finalised") and proceed to
   D.

D. phase == "finalised". Render the plan WITHOUT RATINGS (see Pairing
   rendering below) and end with:

     "Copy + paste this into the Thursday Social Tennis Evening
     group when ready — I won't post there myself."

   On the admin's next message, treat as a fresh start (phase has
   already been finalised; clear via clear_tonight if appropriate).

To start over mid-flow (any phase): "boris start over" / "kickoff
fresh" → clear_tonight, then run kickoff_thursday again if asked.

Pairing rendering
-----------------
Format pairings for mobile WhatsApp (narrow screens). Always start
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

   When the admin explicitly asks for ratings, two things change at
   once: (a) insert the bracket `[a v b]` between `Ct N` and the colon,
   using the court's `bracket_values` field VERBATIM (do NOT recompute
   from `ratings` — those numbers are already correct); and (b) append
   each player's individual rating in parentheses immediately after
   their display name, taken from the `ratings` map. Show rating "?"
   as the literal `?` after the name. For singles courts, the bracket
   holds the two individual ratings (which match the per-name ones).

     Rotation 1 (19:30-20:15)
     Ct 4 [5 v 6]: Geoff(2) & Silvia(3) v Paul V(2) & Hannah(4)
     Ct 5 [4 v 5]: ...
     Ct 6 [3 v 2]: David(3) v Jack(2)

   Separate consecutive rotations with a blank line. Include sit-outs
   (if any) and the plan's `notes` after the last rotation.

   QUALITY WARNING: if the plan dict has
   `metrics.multi_seed.blocking_rules` (a non-empty list), the
   algorithm couldn't find a clean layout even after the extended
   multi-seed search — append a brief warning at the very end:

     ⚠ Note: best total score was {chosen_total}; couldn't fully
     avoid {rule list} (e.g. "an opponent repeat in rotation 3", "a
     3F+1M court in rotation 2"). Consider tweaking attendees or
     swapping a player.

   Translate the rule keys into plain English: opponent_repeat → "an
   opponent repeat"; gender_hard_3F1M → "a 3-women + 1-man court";
   gender_hard_MM_vs_FF → "a men-vs-women segregated pairing". Pull
   the rotation number from the entry's `rotation_num`.
(Edits / commit / final-render guidance is covered in phase routing
above — sections C and D.)

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


def tool_book_court(
    date: str,
    start_time_hhmm: str,
    partner_name: str,
    duration_minutes: int = 60,
    court_label: Optional[str] = None,
    court_type: Optional[str] = None,
    court_preference: Optional[list[str]] = None,
) -> dict:
    """Book a court for the caller's CR account + ``partner_name``.

    Iterates the preference list (default: club's standard order) until
    one court is available at the requested time. ``court_label``
    (e.g. '5') overrides preference. ``court_type`` is 'clay' / 'acrylic'
    (synonym 'hard'); narrows the candidates if no specific court given.
    The CR account used is whichever one is mapped to the WhatsApp
    sender via accounts.json (Geoff by default; Shirley if she sends).
    """
    from accounts import cr_client

    with cr_client(_caller_account()) as cr:
        result = cr.book_court(
            date=date,
            start_time_hhmm=start_time_hhmm,
            partner_name=partner_name,
            duration_minutes=duration_minutes,
            court_label=court_label,
            court_type=court_type,
            court_preference=court_preference,
        )
    return result


def tool_cancel_court_booking(
    reservation_id: Optional[str] = None,
    date: Optional[str] = None,
    start_time_hhmm: Optional[str] = None,
) -> dict:
    """Cancel a court reservation owned by the caller's CR account.

    Either pass ``reservation_id`` directly, or provide ``date`` +
    ``start_time_hhmm`` and the tool will find the caller's booking at
    that slot before cancelling.
    """
    from accounts import cr_client

    with cr_client(_caller_account()) as cr:
        if not reservation_id:
            if not (date and start_time_hhmm):
                return {
                    "ok": False,
                    "status": "missing_args",
                    "message": "Pass reservation_id, OR both date and start_time_hhmm.",
                }
            found = cr.find_my_court_booking(date, start_time_hhmm)
            if not found:
                return {
                    "ok": False,
                    "status": "no_booking_found",
                    "date": date,
                    "start_time_hhmm": start_time_hhmm,
                }
            reservation_id = found["reservation_id"]
        return cr.cancel_court_reservation(reservation_id)


def tool_list_my_bookings() -> dict:
    """List CourtReserve events the caller's account is registered or
    waitlisted for. Read-only — same channel restriction as the booking
    tools, since it leaks the user's schedule.
    """
    from accounts import cr_client

    with cr_client(_caller_account()) as cr:
        bookings = cr.list_my_bookings()
    return {"ok": True, "count": len(bookings), "bookings": bookings}


def tool_schedule_court_booking(
    play_date: str,
    start_time_hhmm: str,
    partner_name: str,
    duration_minutes: int = 60,
    court_label: Optional[str] = None,
    court_type: Optional[str] = None,
    notes: str = "",
) -> dict:
    """Queue a future court booking that fires when CourtReserve's
    booking window opens (08:00 local, 6 days before ``play_date``).

    Validates ``partner_name`` against the Thursday roster + validated-
    members whitelist before persisting. Returns the saved entry's id
    + window_opens_at on success, or an error explaining what to fix.

    Uses the caller's CR account (Geoff or Shirley). The result of
    the eventual booking attempt is posted back to the channel this
    schedule was created in.
    """
    import scheduled_bookings as sb
    from validated_members import is_known_member

    account = _caller_account()

    # 1. Validate the partner name. Roster + whitelist.
    lookup = is_known_member(partner_name, roster_names=Roster().names())
    if not lookup.found:
        return {
            "ok": False,
            "error": "unknown_partner",
            "partner_name": partner_name,
            "candidates": list(lookup.candidates),
            "message": (
                "I don't recognise that partner name. "
                "Pick one of the candidates if any look right, or — if "
                "they really are a club member — call add_validated_member "
                "first to whitelist them."
            ),
        }
    partner_canonical = lookup.canonical_name or partner_name

    # 2. Persist the schedule.
    entry = sb.add_pending(
        scheduled_by_phone=_CURRENT_SENDER.get(None),
        scheduled_by_account_key=account.key,
        channel_jid=_CURRENT_GROUP_JID.get(None),
        play_date=play_date,
        start_time_hhmm=start_time_hhmm,
        duration_minutes=duration_minutes,
        partner_name=partner_canonical,
        court_label=court_label,
        court_type=court_type,
        notes=notes,
    )

    return {
        "ok": True,
        "id": entry.id,
        "play_date": entry.play_date,
        "start_time_hhmm": entry.start_time_hhmm,
        "duration_minutes": entry.duration_minutes,
        "partner_name": entry.partner_name,
        "court_label": entry.court_label,
        "court_type": entry.court_type,
        "window_opens_at": entry.window_opens_at,
        "account": account.display_name,
    }


def tool_list_scheduled_bookings(include_history: bool = False) -> dict:
    """List the caller's pending scheduled bookings (and recent history
    when ``include_history`` is true). Each entry includes its id so it
    can be cancelled with ``cancel_scheduled_booking``.
    """
    import scheduled_bookings as sb

    account = _caller_account()
    pending = [b.to_dict() for b in sb.list_pending(account_key=account.key)]
    out: dict = {"ok": True, "pending": pending, "count": len(pending)}
    if include_history:
        out["history"] = [
            b.to_dict() for b in sb.list_history(account_key=account.key)
        ]
    return out


def tool_cancel_scheduled_booking(booking_id: int) -> dict:
    """Cancel a pending scheduled booking owned by the caller's account
    before its window opens. Idempotent."""
    import scheduled_bookings as sb

    account = _caller_account()
    cancelled, entry = sb.cancel_pending(
        int(booking_id), by_account_key=account.key
    )
    if not cancelled:
        return {
            "ok": False,
            "error": "not_found_or_not_owned",
            "id": booking_id,
            "message": (
                "No pending booking with that id belongs to you. "
                "Run list_scheduled_bookings to confirm the id."
            ),
        }
    return {
        "ok": True,
        "id": entry.id,
        "state": entry.state,
        "play_date": entry.play_date,
    }


def tool_cancel_booking(reservation_number_or_res_id: str) -> dict:
    """Remove the caller's account from an event (registered or
    waitlisted).

    Accepts either the alphanumeric reservation_number (preferred) or the
    numeric res_id you'd typically have from list_my_bookings.
    Idempotent.
    """
    from accounts import cr_client

    with cr_client(_caller_account()) as cr:
        result = cr.cancel_event_registration(reservation_number_or_res_id)
    return {"ok": result.get("status") == "cancelled" or result.get("status") == "not_registered", **result}


def tool_kickoff_thursday(
    allow_non_thursday: bool = False,
    test_mode: bool = False,
) -> dict:
    """Run the Thursday-morning kickoff workflow on demand.

    Same code path as the scheduled trigger: fetches the next Thursday
    Social Tennis Evening event from CourtReserve, auto-adds any new
    names to the roster, calls start_tonight, sets
    session_state.phase = "awaiting_extras", and posts the structured
    "today's lineup + please reply with extras" message to the
    Thursday tennis Admin group.

    By default refuses to run on non-Thursdays. Pass
    ``allow_non_thursday=True`` for testing. Pass ``test_mode=True`` for
    a dry run — the kickoff post is marked as a test, commit_plan and
    log_pairings_to_sheet are blocked, and rating updates still persist.
    """
    from thursday_kickoff import kickoff_thursday

    # Route the kickoff post back to whichever admin group asked for it,
    # so a "boris test run" from Boris test channel doesn't spam the
    # live group. Falls back to the live admin group when invoked from
    # the scheduler (no caller context set).
    target_jid = _CURRENT_GROUP_JID.get(None)

    # Don't echo the kickoff message back as the bot's reply — it's
    # already been posted to the admin group by kickoff_thursday().
    result = kickoff_thursday(
        allow_non_thursday=allow_non_thursday,
        test_mode=test_mode,
        target_jid=target_jid,
    )
    # Strip the (long) message text from the tool result — it's gone to
    # WhatsApp, no need to send it twice.
    return {k: v for k, v in result.items() if k != "message"}


def tool_book_session(
    reservation_number: str,
    allow_waitlist_fallback: bool = True,
) -> dict:
    """Register the bot's CourtReserve account for an event.

    By default, if the event is full and a waitlist is open, the
    waitlist is joined automatically. Pass
    ``allow_waitlist_fallback=False`` to bail out instead.

    Returns ``{ok, status, ...}``. ``status`` is one of:
    ``registered`` / ``waitlisted`` / ``already_registered`` /
    ``already_waitlisted`` / ``event_full_no_fallback`` /
    ``no_action_available`` / ``verification_failed`` /
    ``modal_submit_not_found``.

    Uses the caller's CR account (Geoff or Shirley, depending on
    sender). Guarded at the dispatch layer: only available when the
    trigger came from the "Boris test channel" group.
    """
    from accounts import cr_client

    with cr_client(_caller_account()) as cr:
        result = cr.register_for_event(
            reservation_number,
            allow_waitlist_fallback=allow_waitlist_fallback,
        )
    return {"ok": result.get("status") not in {
        "no_action_available", "verification_failed", "modal_submit_not_found",
    }, **result}


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


def tool_validate_member_name(name: str) -> dict:
    """Resolve ``name`` against the Thursday roster + validated-members
    whitelist. Returns ``{ok: True, found, canonical_name, source,
    candidates}``. ``source`` is "roster" / "validated_members" / null.
    Ambiguous fuzzy matches return ``found=false`` with the close
    candidates, so the bot can ask the admin to disambiguate.
    """
    from validated_members import is_known_member

    res = is_known_member(name, roster_names=Roster().names())
    return {
        "ok": True,
        "found": res.found,
        "canonical_name": res.canonical_name,
        "source": res.source,
        "candidates": list(res.candidates),
    }


def tool_list_validated_members() -> dict:
    """List the validated-member whitelist (newest entries last)."""
    from validated_members import list_members

    members = list_members()
    return {"ok": True, "count": len(members), "members": members}


def tool_add_validated_member(name: str) -> dict:
    """Whitelist a club member's name so it can be used as a partner
    in court bookings even though they're not in the Thursday roster.
    Idempotent — duplicates are no-ops.
    """
    from validated_members import add_member

    return add_member(name)


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
    pinned_singles: Optional[list[dict]] = None,
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
            pinned_singles=pinned_singles,
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
    from session_state import get_tonight

    if get_tonight().test_mode:
        return {
            "ok": False,
            "error": "test_mode",
            "message": (
                "Test run — sheet logging is disabled. Say "
                "'boris clear tonight' to wipe the dry run, or stop here."
            ),
        }
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
    from session_state import clear_draft_plan, get_draft_plan, get_tonight
    from session_log import log_plan

    if get_tonight().test_mode:
        return {
            "ok": False,
            "error": "test_mode",
            "message": (
                "Test run — final commit is disabled. The draft is fine "
                "to keep iterating on, but it won't be written to "
                "history.json or the Sheet. Say 'boris clear tonight' to "
                "wipe the dry run, or stop here."
            ),
        }
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
    "kickoff_thursday": tool_kickoff_thursday,
    "book_session": tool_book_session,
    "list_my_bookings": tool_list_my_bookings,
    "cancel_booking": tool_cancel_booking,
    "book_court": tool_book_court,
    "cancel_court_booking": tool_cancel_court_booking,
    "schedule_court_booking": tool_schedule_court_booking,
    "list_scheduled_bookings": tool_list_scheduled_bookings,
    "cancel_scheduled_booking": tool_cancel_scheduled_booking,
    "read_players_roster": tool_read_players_roster,
    "validate_member_name": tool_validate_member_name,
    "list_validated_members": tool_list_validated_members,
    "add_validated_member": tool_add_validated_member,
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
        "name": "kickoff_thursday",
        "description": "Run the Thursday-morning kickoff workflow on "
        "demand: fetch the next Thursday Social Tennis Evening event "
        "from CourtReserve, auto-add new names to the roster, call "
        "start_tonight, set the workflow phase to 'awaiting_extras', "
        "and POST the structured 'today's lineup + please reply with "
        "extras' message to the admin group. Use this when the admin "
        "says 'boris kickoff' / 'start the Thursday workflow'. The "
        "same code path runs automatically every Thursday at 09:35 "
        "from admin_bot's poll loop. Set allow_non_thursday=true when "
        "the admin is testing off-day. Set test_mode=true when the "
        "admin says 'test run' / 'dry run' / 'practice run' / "
        "'rehearse the pairings' / 'try the pairings without saving' — "
        "the kickoff post is marked TEST RUN and final commit is "
        "blocked, but ratings still persist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "allow_non_thursday": {
                    "type": "boolean",
                    "default": False,
                    "description": "Override the Thursday-only check "
                    "(use when testing).",
                },
                "test_mode": {
                    "type": "boolean",
                    "default": False,
                    "description": "Run a dry run — kickoff post is "
                    "marked, commit_plan / log_pairings_to_sheet refuse, "
                    "rating updates still persist. Set when the admin "
                    "asks for a test/dry/practice run.",
                },
            },
        },
    },
    {
        "name": "book_session",
        "description": "Register the bot's CourtReserve account (currently "
        "Geoff Chapman) for an event. If the event is full and a waitlist "
        "is open, the waitlist is joined automatically unless "
        "allow_waitlist_fallback is set to false (use that when the admin "
        "explicitly says 'don't put me on the waitlist'). Idempotent — "
        "returns 'already_registered' / 'already_waitlisted' if the "
        "account is already on either list. ONLY available in the 'Boris "
        "test channel' admin group; calling from anywhere else returns an "
        "error from the dispatch layer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reservation_number": {
                    "type": "string",
                    "description": "Reservation number from list_club_sessions.",
                },
                "allow_waitlist_fallback": {
                    "type": "boolean",
                    "description": "Default true. Set false to refuse "
                    "waitlist signup when the event is full.",
                    "default": True,
                },
            },
            "required": ["reservation_number"],
        },
    },
    {
        "name": "list_my_bookings",
        "description": "List the CourtReserve events the bot's account is "
        "currently registered for OR waitlisted for. Returns each event's "
        "name, date string, status (registered/waitlisted), res_id and a "
        "detail_url. Use this when the admin asks 'what am I booked on' "
        "or to pick the right event before cancel_booking. ONLY available "
        "in the 'Boris test channel' admin group.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "cancel_booking",
        "description": "Remove the bot's account from an event (whether "
        "currently registered or waitlisted). Idempotent — returns "
        "'not_registered' if there's nothing to cancel. Use list_my_bookings "
        "first to find the right reservation_number_or_res_id. ONLY "
        "available in the 'Boris test channel' admin group.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reservation_number_or_res_id": {
                    "type": "string",
                    "description": "Either the alphanumeric reservation_number "
                    "(e.g. 'AEK3RNY2146914') or the numeric res_id from "
                    "list_my_bookings (e.g. '52824526').",
                },
            },
            "required": ["reservation_number_or_res_id"],
        },
    },
    {
        "name": "book_court",
        "description": "Book a tennis court for the bot's account + a "
        "named partner. The partner must be an existing club member "
        "(autocomplete will search their name). Iterates the court "
        "preference list (default: 5,6,9,7,8,10,14,11,12,4,1,2,3) until "
        "one is free at the requested slot. Pass court_label (e.g. '5') "
        "to force a specific court, or court_type ('clay'|'acrylic') to "
        "narrow the candidates. Duration: 30, 60 (default), or 90 min. "
        "ONLY available in the 'Boris test channel' admin group.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "ISO date YYYY-MM-DD.",
                },
                "start_time_hhmm": {
                    "type": "string",
                    "description": "Start time HH:MM (24h, e.g. '13:00').",
                },
                "partner_name": {
                    "type": "string",
                    "description": "Partner's full name (autocompleted "
                    "against the club roster). REQUIRED — the admin must "
                    "supply this in the request.",
                },
                "duration_minutes": {
                    "type": "integer",
                    "enum": [30, 60, 90],
                    "default": 60,
                    "description": "30 / 60 / 90 minutes — maps to the "
                    "matching reservation type ('30 min hit' / '60 min "
                    "hit' / '1 hour 30 min hit').",
                },
                "court_label": {
                    "type": "string",
                    "description": "Specific court number to force, e.g. '5'.",
                },
                "court_type": {
                    "type": "string",
                    "enum": ["clay", "acrylic", "hard"],
                    "description": "Constrain candidates to a surface type "
                    "if no specific court_label was given.",
                },
                "court_preference": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Override the default preference list.",
                },
            },
            "required": ["date", "start_time_hhmm", "partner_name"],
        },
    },
    {
        "name": "cancel_court_booking",
        "description": "Cancel a court reservation. Pass either "
        "reservation_id (numeric, from a prior book_court response), or "
        "date + start_time_hhmm and the tool will find the booking. "
        "ONLY available in the 'Boris test channel' admin group.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reservation_id": {"type": "string"},
                "date": {"type": "string", "description": "ISO date YYYY-MM-DD."},
                "start_time_hhmm": {"type": "string"},
            },
        },
    },
    {
        "name": "schedule_court_booking",
        "description": "Queue a future court booking that fires when the "
        "CourtReserve booking window opens (08:00 local on the day six "
        "days before play_date — so a Thursday 14th booking fires Friday "
        "8th at 08:00). Use this when the admin says things like "
        "'schedule to book me a court next Thursday', 'book me a court "
        "for Tuesday afternoon and queue it up', 'book the 8am slot for "
        "Thursday 14th when the window opens'. Validates partner_name "
        "against the roster + validated-members whitelist before queuing "
        "— if validation fails, ask the admin to confirm the spelling "
        "or call add_validated_member first. Returns the booking id and "
        "the window_opens_at timestamp so the bot can confirm to the "
        "admin. The eventual booking attempt's success/failure message "
        "is posted automatically into this channel when it fires.",
        "input_schema": {
            "type": "object",
            "properties": {
                "play_date": {
                    "type": "string",
                    "description": "ISO date YYYY-MM-DD when the court "
                    "should be played.",
                },
                "start_time_hhmm": {
                    "type": "string",
                    "description": "Start time of play in 24h HHMM "
                    "(e.g. '0800').",
                },
                "partner_name": {
                    "type": "string",
                    "description": "Full club-member name. Must be a "
                    "Thursday-roster player or on the validated-members "
                    "whitelist; otherwise the tool returns "
                    "error='unknown_partner' with candidates.",
                },
                "duration_minutes": {
                    "type": "integer",
                    "default": 60,
                    "description": "30 / 60 / 90 / 120. Default 60.",
                },
                "court_label": {
                    "type": "string",
                    "description": "Force a specific court (e.g. '6'). "
                    "Without this the booking iterates the club's "
                    "preference list at fire time.",
                },
                "court_type": {
                    "type": "string",
                    "description": "'clay' or 'acrylic' (synonym 'hard').",
                },
                "notes": {
                    "type": "string",
                    "description": "Free-text note attached to the entry.",
                },
            },
            "required": ["play_date", "start_time_hhmm", "partner_name"],
        },
    },
    {
        "name": "list_scheduled_bookings",
        "description": "List the caller's pending scheduled bookings (and "
        "recent history when include_history=true). Each pending entry "
        "has an id usable with cancel_scheduled_booking.",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_history": {
                    "type": "boolean",
                    "default": False,
                    "description": "Also include recent succeeded / "
                    "failed / cancelled entries.",
                }
            },
        },
    },
    {
        "name": "cancel_scheduled_booking",
        "description": "Cancel a pending scheduled booking by id (must be "
        "owned by the caller's account). Idempotent. The booking moves "
        "to history with state='cancelled' so it shows up in "
        "list_scheduled_bookings(include_history=true).",
        "input_schema": {
            "type": "object",
            "properties": {
                "booking_id": {
                    "type": "integer",
                    "description": "The id from list_scheduled_bookings.",
                }
            },
            "required": ["booking_id"],
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
        "name": "validate_member_name",
        "description": "Check whether a name belongs to a valid club member. "
        "Looks first in the Thursday roster (read_players_roster), then in "
        "the validated-members whitelist. Returns found=true with "
        "canonical_name on a unique match; found=false with candidates on "
        "an ambiguous fuzzy match; found=false with empty candidates if "
        "the name isn't recognised at all. Use this BEFORE scheduling a "
        "court booking against a partner you haven't seen before. If "
        "found=false and the admin confirms the person is a real club "
        "member, call add_validated_member to whitelist them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The partner / member name to look up.",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_validated_members",
        "description": "List club members on the validated-members whitelist "
        "(non-Thursday-roster members already approved as booking partners).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "add_validated_member",
        "description": "Add a name to the validated-members whitelist. Use this "
        "when the admin confirms an unknown partner is a real club member. "
        "Idempotent — duplicates are no-ops. The Thursday roster is checked "
        "first by validate_member_name, so don't add roster members here.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Full name as it should appear on bookings.",
                }
            },
            "required": ["name"],
        },
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
                "pinned_singles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rotation_num": {
                                "type": "integer",
                                "description": "1-indexed rotation.",
                            },
                            "players": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 2,
                                "description": "Exactly two full names.",
                            },
                            "court_label": {
                                "type": "string",
                                "description": "Optional court label; default = the "
                                "first singles court of that rotation.",
                            },
                        },
                        "required": ["rotation_num", "players"],
                    },
                    "description": "Force specific singles matchups before generating. "
                    "Use for admin instructions like 'make the first singles match Amir "
                    "vs Patrick' (rotation_num=1, players=['Amir Alizadeh', 'Patrick "
                    "Gibbs']). Each pinned player counts as their one singles "
                    "appearance under the per-evening cap, so the same player can't "
                    "be pinned to multiple rotations. Doubles balance is then "
                    "optimised around the pin.",
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

# Tools that mutate state in CourtReserve (i.e. book on the user's behalf)
# are restricted to the test channel — they're filtered out of the schema
# AND the dispatch table for any other admin group, so the LLM can't even
# see them. Hard guarantee at the dispatch layer rather than a soft "please
# don't" in the prompt.
PROTECTED_TOOLS: set[str] = {
    "book_session", "cancel_booking", "list_my_bookings",
    "book_court", "cancel_court_booking",
    "schedule_court_booking", "list_scheduled_bookings",
    "cancel_scheduled_booking",
}
TEST_CHANNEL_NAME = "Boris test channel"


def _tools_for_caller(
    group_name: str, account: Any
) -> tuple[list[dict], dict[str, Any]]:
    """Return the (schemas, impls) pair the LLM should see for this
    caller. Two filters compose:

    1. Channel filter — booking tools (PROTECTED_TOOLS) are only
       visible in the test channel, since they mutate CR state.
    2. Per-account scope — Account.is_tool_allowed (driven by
       accounts.json's ``tool_scope`` field) further narrows which
       tools the caller can see. ``full`` scope is a no-op; tighter
       scopes hide pairings/admin tools.
    """
    if group_name == TEST_CHANNEL_NAME:
        candidates = list(TOOL_IMPLS.keys())
    else:
        candidates = [n for n in TOOL_IMPLS if n not in PROTECTED_TOOLS]

    allowed = [n for n in candidates if account.is_tool_allowed(n)]
    schemas = [s for s in TOOL_SCHEMAS if s["name"] in allowed]
    impls = {n: TOOL_IMPLS[n] for n in allowed}
    return schemas, impls


def run_agent(
    client: Anthropic, user_text: str, group_name: str = ""
) -> tuple[str, dict]:
    """Run a Claude tool-use loop. Returns ``(final_text, usage)`` where
    ``usage`` is a dict of accumulated token counts across all turns.

    Two filters control which tools are exposed: the channel-based
    PROTECTED_TOOLS gate (booking tools only visible in the test
    channel) and the caller's account scope (full / read_and_book /
    booking_only / read_only — see accounts.py). The caller's account
    is resolved from the _CURRENT_SENDER contextvar set by the poll
    loop.
    """
    account = _caller_account()
    tool_schemas, tool_impls = _tools_for_caller(group_name, account)
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
            tools=tool_schemas,
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
                if block.name not in tool_impls:
                    # Could be channel-restricted, scope-restricted, or
                    # just unknown. The composed filter above already
                    # narrowed schemas; if Claude calls something outside
                    # that, reject without calling the underlying impl.
                    if not account.is_tool_allowed(block.name):
                        raise PermissionError(
                            f"Tool {block.name!r} is not available to "
                            f"account {account.key!r} (scope="
                            f"{account.tool_scope!r})."
                        )
                    raise KeyError(
                        f"Tool {block.name!r} is not available in this "
                        f"channel ({group_name!r})."
                    )
                fn = tool_impls[block.name]
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


def _kickoff_target_time() -> tuple[int, int]:
    """Return (hour, minute) for the kickoff. Honours an env override."""
    override = os.environ.get("BORIS_KICKOFF_TIME_OVERRIDE", "").strip()
    if override and ":" in override:
        try:
            h_s, m_s = override.split(":", 1)
            return int(h_s), int(m_s)
        except ValueError:
            pass
    return KICKOFF_HOUR, KICKOFF_MINUTE


def _last_kickoff_attempt_date() -> str:
    """ISO date of the last kickoff attempt, or '' if never."""
    if not KICKOFF_STATE_PATH.exists():
        return ""
    try:
        data = json.loads(KICKOFF_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return data.get("last_attempt_date", "")


def _record_kickoff_attempt(date_iso: str) -> None:
    KICKOFF_STATE_PATH.write_text(
        json.dumps({"last_attempt_date": date_iso}),
        encoding="utf-8",
    )


def _should_fire_thursday_kickoff(now: datetime) -> bool:
    """Decide whether the poll loop should fire the kickoff this tick.

    Conditions: it's Thursday (Mon=0…Thu=3) — unless BORIS_KICKOFF_ANY_DAY=1
    in env (testing); the current local time is at or past the trigger
    time (default 09:35); and we haven't already fired today.
    """
    any_day = os.environ.get("BORIS_KICKOFF_ANY_DAY", "").strip() == "1"
    if not any_day and now.weekday() != 3:
        return False
    h, m = _kickoff_target_time()
    if (now.hour, now.minute) < (h, m):
        return False
    return _last_kickoff_attempt_date() != now.date().isoformat()


def _maybe_fire_thursday_kickoff(now: datetime) -> None:
    """If conditions are right, run the kickoff. Best-effort, never raises.

    The attempt date is recorded BEFORE the work runs, so a slow / failing
    kickoff doesn't get retried every second for the rest of the day.
    """
    if not _should_fire_thursday_kickoff(now):
        return
    _record_kickoff_attempt(now.date().isoformat())
    print(
        f"[scheduler] firing Thursday kickoff at {now.isoformat(timespec='seconds')}"
    )
    try:
        from thursday_kickoff import kickoff_thursday
        result = kickoff_thursday()
        print(f"[scheduler] kickoff result: {result.get('ok')} "
              f"({result.get('error') or 'posted'})")
    except Exception as e:
        print(f"[scheduler] kickoff crashed: {e!r}", file=sys.stderr)


def _fire_scheduled_booking(entry) -> None:
    """Attempt one scheduled booking. Posts the result back to the
    channel that scheduled it, then records succeeded / failed in
    scheduled_bookings.json. Synchronous: blocks the poll loop for
    the duration of the CR call (~10-30 s)."""
    import scheduled_bookings as sb
    from accounts import account_for_key, cr_client

    print(
        f"[scheduler] firing scheduled booking #{entry.id} "
        f"({entry.scheduled_by_account_key} → court "
        f"{entry.court_label or '<any>'} on {entry.play_date} at "
        f"{entry.start_time_hhmm}, partner={entry.partner_name}, "
        f"attempt {entry.fire_attempts + 1}/{sb.MAX_FIRE_ATTEMPTS})"
    )

    account = account_for_key(entry.scheduled_by_account_key)
    if account is None:
        sb.mark_attempt(
            entry.id, succeeded=False,
            error=f"account_unknown:{entry.scheduled_by_account_key}",
        )
        return

    try:
        with cr_client(account) as cr:
            result = cr.book_court(
                date=entry.play_date,
                start_time_hhmm=entry.start_time_hhmm,
                partner_name=entry.partner_name,
                duration_minutes=entry.duration_minutes,
                court_label=entry.court_label,
                court_type=entry.court_type,
            )
    except Exception as e:
        result = {"ok": False, "status": "exception", "error": repr(e)}

    succeeded = bool(result.get("ok"))
    transient = (not succeeded) and result.get("status") in {
        "too_early", "no_court_available", "all_taken", "exception",
    }

    final_attempt = (
        entry.fire_attempts + 1 >= sb.MAX_FIRE_ATTEMPTS
    ) or not transient

    sb.mark_attempt(
        entry.id,
        succeeded=succeeded,
        result=result,
        error=(None if succeeded else result.get("status") or result.get("error")),
    )

    # Post a result message back to the channel only on success or
    # final failure — silent retries avoid noise during the 30 s window.
    if entry.channel_jid and (succeeded or final_attempt):
        if succeeded:
            text = (
                f"✓ Court booked for {account.display_name}: "
                f"{result.get('court_label') or entry.court_label or '?'} "
                f"on {entry.play_date} at {entry.start_time_hhmm} "
                f"({entry.duration_minutes} min) with {entry.partner_name}."
            )
        else:
            text = (
                f"✗ Scheduled booking #{entry.id} failed for "
                f"{account.display_name} ({entry.play_date} "
                f"{entry.start_time_hhmm}, partner {entry.partner_name}): "
                f"{result.get('status') or result.get('error') or 'unknown'}."
            )
        try:
            send_to_group(entry.channel_jid, BOT_REPLY_PREFIX + text)
        except Exception as e:
            print(f"[scheduler] post-back failed: {e!r}", file=sys.stderr)


def _maybe_fire_scheduled_bookings(now: datetime) -> None:
    """Fire any pending bookings whose window has opened. Best-effort."""
    import scheduled_bookings as sb

    try:
        due = sb.due_now(now=now.astimezone(sb.LOCAL_TZ) if now.tzinfo else now)
    except Exception as e:
        print(f"[scheduler] due_now error: {e!r}", file=sys.stderr)
        return
    for entry in due:
        try:
            _fire_scheduled_booking(entry)
        except Exception as e:
            print(
                f"[scheduler] booking #{entry.id} fire crashed: {e!r}",
                file=sys.stderr,
            )


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
    h, m = _kickoff_target_time()
    last = _last_kickoff_attempt_date() or "never"
    print(
        f"Auto-kickoff: Thursdays at {h:02d}:{m:02d} (last attempt: {last})"
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
        # Auto-kickoff check — fires once per Thursday at the configured
        # trigger time (default 09:35). Best-effort; never blocks the
        # message-polling that follows.
        _maybe_fire_thursday_kickoff(datetime.now())

        # Scheduled court-booking check — fires any pending entries
        # whose 6-day-ahead window has opened (08:00 local). Each fire
        # is a 10-30 s blocking CR call; the watermark catches up on
        # the next iteration.
        _maybe_fire_scheduled_bookings(datetime.now())

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
                # Tools that need to address replies back to the calling
                # group (e.g. kickoff_thursday) read group_jid via .get().
                # Booking tools resolve the caller's CR account via
                # _CURRENT_SENDER → accounts.account_for_phone(...).
                jid_token = _CURRENT_GROUP_JID.set(group_jid)
                sender_token = _CURRENT_SENDER.set(sender)
                try:
                    if command:
                        reply_body, usage = run_agent(
                            client, command, group_name=group_name
                        )
                    else:
                        reply_body = "(empty command — say e.g. 'boris help')"
                except Exception as e:
                    reply_body = f"(bot error: {e})"
                finally:
                    done.set()
                    timer.cancel()
                    _CURRENT_GROUP_JID.reset(jid_token)
                    _CURRENT_SENDER.reset(sender_token)

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
