"""WhatsApp admin bot for the Thursday Tennis group.

Watches a WhatsApp group called "Thursday Tennis Organisers" (or whatever name
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
from datetime import datetime, timedelta
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
FINAL_DOCX_OUTPUT_DIR = ROOT / "output_files"


def _docx_template_for(session_type: str) -> Path:
    """Resolve the DOCX template path for a session type. Falls back to
    Thursday's template when the type is unknown or empty (legacy /
    manual starts)."""
    from session_types import SESSION_TYPES

    st = SESSION_TYPES.get(session_type) if session_type else None
    relpath = (
        st.docx_template_relpath
        if st is not None
        else SESSION_TYPES["thursday"].docx_template_relpath
    )
    return ROOT / relpath


def _docx_preamble_count_for(session_type: str) -> int:
    """How many template paragraphs to keep verbatim before the date
    heading. Thursday's signup-blurb + QR template = 2; the Westside
    template (Tuesday + Saturday) opens straight on the heading = 0."""
    from session_types import SESSION_TYPES

    st = SESSION_TYPES.get(session_type) if session_type else None
    return (
        st.docx_preamble_paragraph_count
        if st is not None
        else SESSION_TYPES["thursday"].docx_preamble_paragraph_count
    )


def _plan_total_score(plan) -> int:
    """Sum of per-rotation best_score from a PairingPlan's metrics.
    Accepts the live ``PairingPlan`` object — mirrors ``pairings._plan_total``
    but kept here so the retry helper doesn't reach into pairings'
    private module."""
    return sum(
        int(r.get("best_score") or 0)
        for r in (plan.metrics.get("rotations") or [])
    )


def _merge_loser_work_into_winner(winner, loser) -> None:
    """Roll the LOSER's wall time + permutations into the WINNER's
    metrics so the rendered-to-WhatsApp footer ("tried N permutations
    in Xs") reflects the total effort across both attempts of the
    best-of-N flow — not just whichever run came out on top."""
    wm = winner.metrics
    lm = loser.metrics

    wm["wall_seconds"] = round(
        float(wm.get("wall_seconds") or 0)
        + float(lm.get("wall_seconds") or 0),
        2,
    )

    w_ms = wm.setdefault("multi_seed", {})
    l_ms = lm.get("multi_seed") or {}
    w_ms["total_permutations_tried"] = (
        int(w_ms.get("total_permutations_tried") or 0)
        + int(l_ms.get("total_permutations_tried") or 0)
    )
    w_ms["wall_seconds"] = round(
        float(w_ms.get("wall_seconds") or 0)
        + float(l_ms.get("wall_seconds") or 0),
        2,
    )


def _generate_with_retry(
    *,
    threshold: int | None = None,
    notice_callback=None,
    make_plan_fn=None,
    **make_plan_kwargs,
):
    """Run ``make_plan`` once; if the total score is >= ``threshold``
    invoke ``notice_callback`` (best-effort) and re-roll once more
    with a different seed, returning the lower-scoring plan with the
    loser's work folded into the winner's metrics.

    ``make_plan_fn`` is the function to call. Defaults to
    ``pairings.make_plan``; tests override it to return controlled
    plans without running the real optimiser. ``threshold`` defaults
    to ``GENERATE_RETRY_THRESHOLD`` at call time (resolved lazily so
    the constant can live in the conventional constants block lower
    down the module).
    """
    if make_plan_fn is None:
        from pairings import make_plan as _real_make_plan
        make_plan_fn = _real_make_plan
    if threshold is None:
        threshold = GENERATE_RETRY_THRESHOLD

    plan = make_plan_fn(**make_plan_kwargs)
    if _plan_total_score(plan) < threshold:
        return plan

    # First run came in above threshold — notify the admin (best
    # effort: a failed notice doesn't block the retry) and re-roll.
    if notice_callback is not None:
        try:
            notice_callback()
        except Exception:
            pass

    retry_kwargs = dict(make_plan_kwargs)
    s = retry_kwargs.get("seed")
    retry_kwargs["seed"] = (s + 1) if s is not None else None
    plan_2 = make_plan_fn(**retry_kwargs)

    if _plan_total_score(plan_2) < _plan_total_score(plan):
        _merge_loser_work_into_winner(plan_2, plan)
        return plan_2
    _merge_loser_work_into_winner(plan, plan_2)
    return plan


def _docx_header_text_for(session_type: str) -> str:
    """Page-header banner per session type. Falls back to Thursday."""
    from session_types import SESSION_TYPES

    st = SESSION_TYPES.get(session_type) if session_type else None
    return (
        st.docx_header_text
        if st is not None
        else SESSION_TYPES["thursday"].docx_header_text
    )


def _docx_basename_for(session_type: str) -> str:
    """Filename prefix for the rendered final doc (date is appended).
    Sessions sharing a template (Tue+Sat) still get distinct output
    filenames so the file says what kind of session it was for."""
    from session_types import SESSION_TYPES

    st = SESSION_TYPES.get(session_type) if session_type else None
    if st is None:
        return "Thursday Social Tennis"
    if st.key == "thursday":
        return "Thursday Social Tennis"
    if st.key == "tuesday":
        return "Tuesday Social Tennis"
    if st.key == "saturday":
        return "Saturday Social Tennis"
    return "Westside Social Tennis"
RULES_PDF_PATH = ROOT / "output_files" / "Pairing rules and weights.pdf"
RULES_PDF_CAPTION = "The pairing rules and weights are described in this PDF file."

ADMIN_GROUP_NAMES = [
    "Thursday Tennis Organisers",
    "Westside organisers of social tennis",
    "Boris the tennis bot",
]
TENNIS_GROUP_JID = "120363408685115680@g.us"  # Thursday Social Tennis Evening

POLL_INTERVAL_SECONDS = 0.3
# Best-of-N for generate_pairings. If the first run's total score is
# >= GENERATE_RETRY_THRESHOLD we suspect we landed in a sub-optimal
# local minimum (polish is stochastic, so different runs of the same
# inputs can land 50-100 points apart), and we re-roll once more with
# a different seed. The lower-scoring plan wins. A second run adds
# ~30s; with the threshold at 80 we only pay it when polish hasn't
# already found a near-optimal layout.
GENERATE_RETRY_THRESHOLD = 80
GENERATE_RETRY_NOTICE = (
    f"score for pairing above threshold of {GENERATE_RETRY_THRESHOLD} "
    "so having another go, about another 30 seconds"
)
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
# Multi-turn memory: how far back to reconstruct the Boris<->admin
# exchange so a follow-up ("Friday 22nd May, court 5") is understood
# as answering Boris's own prior question rather than a fresh request.
# Bounded by recency AND count to keep token cost down and stop a
# stale, unrelated thread from bleeding into a new command.
HISTORY_WINDOW_MINUTES = 15
HISTORY_MAX_MESSAGES = 12
# A loaded run (dry or real) untouched for this many minutes triggers
# a one-off "continue or clear?" nudge in the channel it was started
# in. Reset whenever the admin interacts again.
STALE_RUN_REMINDER_MINUTES = 120

# JID of the WhatsApp group that triggered the current command. Set by
# the polling loop before invoking the agent; tools that need to address
# replies back to the calling channel (notably kickoff_session) read it
# via .get(None). Unset (None) when invoked from a context with no
# user-triggering message — those fall back to the session's
# configured admin group.
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


# Holds the in-flight command's "done" Event while a command is being
# handled. Outbound sends (a tool posting a PDF / its own message)
# call _signal_output_delivered() so the delayed "working on it…"
# placeholder is suppressed once real output has already reached the
# channel — otherwise it arrives AFTER the answer, out of sync.
_ACTIVE_DONE: dict = {"ev": None}


def _signal_output_delivered() -> None:
    ev = _ACTIVE_DONE.get("ev")
    if ev is not None:
        ev.set()


def _resolve_booking_account(book_as: Optional[str]):
    """Resolve which CR account a booking should run under.

    ``book_as`` lets an admin book under a *different* club account's
    CourtReserve login (e.g. Geoff messaging "book it in Shirley's
    name using her login"). Without it, bookings use the caller's own
    account, exactly as before.

    Matching is by account key (``"shirley"``) or display name
    (``"Shirley Chapman"`` / first name ``"Shirley"``), case-
    insensitive. Authorisation: a caller may always book as
    themselves; booking as a *different* account requires the caller's
    own account to have ``full`` tool scope (the admin/Geoff account).
    This stops a narrow-scope account from booking with someone
    else's credentials.

    Returns ``(account, None)`` on success or ``(None, error_dict)``
    where ``error_dict`` is a ready-to-return tool result.
    """
    caller = _caller_account()
    if not book_as or not str(book_as).strip():
        return caller, None

    from accounts import get_registry

    q = str(book_as).strip().lower()
    reg = get_registry()
    target = None
    for a in reg.accounts:
        names = {
            a.key.lower(),
            (a.display_name or "").strip().lower(),
            (a.display_name or "").strip().lower().split(" ")[0],
        }
        if q in names:
            target = a
            break
    if target is None:
        return None, {
            "ok": False,
            "error": "unknown_account",
            "book_as": book_as,
            "message": (
                f"I don't recognise a club account called {book_as!r}. "
                f"Known: "
                + ", ".join(sorted(x.display_name for x in reg.accounts))
                + "."
            ),
        }
    if target.key == caller.key:
        return target, None
    # Booking as someone else — caller must be an admin (full scope).
    if caller.tool_scope != "full":
        return None, {
            "ok": False,
            "error": "account_override_denied",
            "book_as": target.display_name,
            "message": (
                "Only the admin account can place a booking under a "
                "different member's CourtReserve login."
            ),
        }
    return target, None

SYSTEM_PROMPT = """\
You are "Boris the tennis bot", the admin assistant for the Westside
Thursday Social Tennis evenings (and any other club session the admin
asks about).

You receive commands from the admin chat. Every command starts with the
trigger word `boris` or `bot` (optionally followed by `:` / `?` / `!`).
Messages without the trigger never reach you — they're filtered out
upstream. So treat every message you do see as a real bot request.

The recent back-and-forth from this chat (last ~15 min) is included
as prior turns. Use it: if your previous turn asked the admin for
specific details (a court number, a date/time, a partner name, which
item to cancel, etc.) and their new message supplies them, treat it
as the answer to THAT question and continue that task — do not
reinterpret a bare detail-reply as a brand-new request. E.g. if you
just asked "which booking — give me court + date/time" and they say
"Friday 22nd May 13:00, court 5", that is the cancel target, not a
new booking.

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
  with fields gender (M/F/?), rating (1-10 or "?"), notes.
  **Rating scale: 1 = strongest, 10 = weakest, ? = unknown (treated as 6).**
  Don't mix this up — the singles-selection and skill-balance logic
  depend on it.
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
- list_my_bookings: show CLUB EVENTS (socials, lessons, etc) the
  caller's account is registered or waitlisted for. Does NOT return
  ad-hoc court bookings (e.g. a Court 5 reservation placed via
  book_court / schedule_court_booking) — those have a separate
  cancel path (see cancel_court_booking). Use this when the admin
  asks "what am I booked on" specifically for events, or to find the
  right id before cancel_booking.
- cancel_booking: remove the caller's account from a CLUB EVENT
  (registered or waitlisted). Pass reservation_number_or_res_id from
  list_my_bookings. Idempotent. This does NOT cancel court
  reservations — for those use cancel_court_booking.
- book_court: book a tennis court for a club account + a named
  partner (a club member). Required: date (YYYY-MM-DD),
  start_time_hhmm (24h), partner_name. Optional: duration_minutes
  (30/60/90/120, default 90 — see Booking-type names below),
  court_label (force a specific court), court_type
  ('clay'|'acrylic'), book_as (place it under another member's
  login — see CALLER AWARENESS). If no court is specified, iterates
  the booking account's preference (or the club default
  5,6,9,7,8,10,11,12,4,1,2,3) until one is free.
  The success result includes reservation_id and booked_under —
  you MUST echo both (plus date/time/court/partner) in your
  confirmation message text, because tool results don't persist
  across turns; only your reply text does. See CONFIRMATION
  MESSAGE under COURT BOOKING WORKFLOW.
- cancel_court_booking: cancel an AD-HOC court reservation (placed
  via book_court / schedule_court_booking). Pass either
  reservation_id (numeric, ideally from the prior book_court
  response) OR date+start_time_hhmm to let the tool find it. Use
  this — NOT cancel_booking — when the admin says "cancel that
  court" / "cancel the booking we just made" / refers to a court
  number + date+time. If the admin just made a booking in this
  conversation, you already have the date+time+reservation_id; do
  NOT call list_my_bookings first (it won't show the court
  reservation).
- schedule_court_booking: queue a future court booking that fires
  when CR's booking window opens (08:00 local on the day seven days
  before play_date). Use this when the admin says "schedule",
  "queue", "book me ... when the window opens", "wake up at 8am
  Friday and book ...", or just asks to book for a date >6 days
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

BOOKING UNDER THE OTHER ACCOUNT: when the admin (Geoff) asks to
book "in Shirley's name", "using Shirley's login", "with her
credentials", or similar, you CAN do this — pass book_as="shirley"
to book_court / schedule_court_booking / cancel_court_booking. Do
NOT claim you can only use the sender's account; that limitation no
longer applies. The booking then runs under Shirley's CourtReserve
login and automatically uses HER saved court preference (not the
club default). Only the Geoff/admin account may book as another
member; if Shirley asks to book as Geoff the tool will refuse.
When you confirm such a booking, state whose login it will use and
that her court preference applies — don't echo a court-preference
list unless asked (and never present the club default as if it
were the booking account's).

COURT BOOKING WORKFLOW:

Booking-type names (translate the admin's natural-language phrasing
into the right duration_minutes value):

  "30 minute hit"                   → duration_minutes=30
  "60 minute hit" / "1 hour hit"    → duration_minutes=60
  "1 hour 30 minute hit" / "90 min  → duration_minutes=90 (DEFAULT
   hit" / "hit" / "knock-up"           when the admin doesn't say)
  "ladder match" / "ladder game"    → duration_minutes=120

If the admin doesn't specify a type or duration, default to 90
("1 hour 30 minute hit"). Never use 60 by default unless explicitly
requested. CourtReserve only supports those four reservation types
— anything else (e.g. "45 min", "doubles tournament") returns
status="unsupported_duration"; tell the admin which durations are
allowed.

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

CONFIRMATION MESSAGE — CRITICAL. Tool results are NOT remembered
across turns; only your own reply text is. So after a SUCCESSFUL
book_court, your confirmation MUST explicitly restate, in the
message body: the date, start time, duration, court
(court_label), partner, the account it's under (booked_under),
AND the reservation_id from the result. If you omit these, a
later "cancel that" is unrecoverable without asking the admin
again. Example: "Booked Court 11 on Tue 19 May at 13:00 (60 min)
under Shirley's login, partner Geoff Chapman. reservation_id
54109945."

CANCELLING A COURT BOOKING: when the admin says "cancel that" /
"cancel the court" referring to a booking from earlier in this
conversation, read the reservation_id (and whose login it was
under) from your own earlier confirmation text and call
cancel_court_booking(reservation_id=..., book_as=<that account if
it wasn't the caller's own>). If the confirmation details aren't
in the visible history, ask for date + start time + whose account
— and pass book_as to the cancel so it searches the right login
(a booking made with book_as="shirley" can ONLY be found/cancelled
with book_as="shirley").

ROSTER:
- read_players_roster: full map of name → {gender, rating, notes}.
- set_player_rating: update 1-10 rating (fuzzy name). '?' resets.
- set_player_gender: update M / F / ? (fuzzy name) — use this for
  messages like "Longjie Jia is male" / "Sam is F" / "reset Pat's
  gender to unknown". Don't tell the admin to edit the sheet by hand.
- set_singles_preference: set 'avoid' / 'prefer' / 'neutral' (fuzzy name).
- find_roster_duplicates: scan for likely-duplicate rows
  (apostrophe variants, nickname variants like Ben/Benjamin). Flags
  which spelling CourtReserve currently uses (KEEP that one or the
  next scrape silently re-creates the duplicate). Read-only.
- merge_and_delete_player: fix a duplicate found above. ALWAYS call
  with confirm=false first to get a preview; show the admin what
  would change and get explicit approval before re-calling with
  confirm=true. Merges populated fields from delete→keep then drops
  the redundant row.

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
- pin_doubles(players=[4 names], pairs=[[A,B],[C,D]], rotation_num?,
  court_label?): pin a specific 4-player doubles match-up before
  generation. Used when the admin wants e.g. stronger players to play
  one rotation with weaker players for coaching/support. The pinned
  court isn't scored on its own, but the rotation still counts toward
  each player's whole-evening per-player rules and cross-rotation
  tallies. Pair structure is REQUIRED — never guess; ASK if the admin
  only names 4 players without saying who partners whom.
- clear_pinned_doubles: drop every pinned-doubles entry for tonight.
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
- undo_commit: reverse the most recent commit_plan ("undo the
  commit", "I committed by mistake", "unfinalise", "revert that",
  or wanting to edit a just-finalised plan). Removes the history.json
  entry + Sheet log rows, restores the draft, sets phase back to
  draft_ready so editing/regenerate work again. Only the latest
  commit, only before clear_tonight. After undo, treat the session
  as draft_ready (re-render the DRAFT view, not the final).
- log_pairings_to_sheet: escape hatch — log an arbitrary plan dict.
  Prefer commit_plan for the normal flow.
- send_rules_pdf: when the admin asks about the pairing rules,
  weights, scoring rules, "how does the algorithm score", "what
  weightings do you use", "give me the current rules", or similar,
  call this tool. It regenerates and sends a one-page PDF summary
  of every rule + weight as a WhatsApp attachment (with its own
  caption). Emit an empty assistant reply afterwards — the
  attachment carries the message.

Session workflow (phase-driven)
-------------------------------
Boris organises THREE weekly sessions at Westside: Tuesday Evening
Club Social for Intermediate+ (19:30-21:30), Thursday Social Tennis
Evening (19:30-21:30), and Saturday Social Doubles (14:00-16:00).
Each follows the same phase machine via `session_state.phase`. ALWAYS
read `phase` (call get_tonight) at the start of any non-trivial admin
reply and route accordingly. Valid phases:

  ""                   no in-flight session
  "awaiting_extras"    kickoff posted; admin will reply with extras
  "ready_to_generate"  extras applied; ready to run generate_pairings
  "draft_ready"        draft persisted; admin iterates or confirms
  "finalised"          committed; render the final, no-ratings copy

There is NO auto-kickoff — sessions only start when an organiser
asks. When they say "boris kickoff", "start the workflow", "let's
do tonight", "kick off tomorrow's session" etc., call
kickoff_session. Omit session_type unless they explicitly name a
day — it defaults to the next scheduled session by weekday (Mon/Tue
→ Tuesday, Wed/Thu → Thursday, Fri/Sat → Saturday, Sun → Tuesday).
If they DO name a day ("boris kickoff Saturday's session"), pass
session_type="saturday" (etc.) so they get the right one even if
today's weekday would point elsewhere.

Test/dry runs. When the admin says "test run", "dry run", "practice
run", "rehearse the pairings", "try the pairings without saving",
"trial run", or similar → call kickoff_session(test_mode=true).
Behaves identically — same phases, same generation, same swaps,
same rendering — EXCEPT (1) the kickoff post is prefixed with a
TEST RUN banner, (2) commit_plan and log_pairings_to_sheet refuse
with error="test_mode", and (3) the kickoff post is routed back to
whichever channel the admin asked from (so a test from Boris the
tennis bot stays there). Rating updates from "Tomoki = 2" still
persist — that's intentional. To end a test run, the admin can stop
replying or say "boris clear tonight".

REPLAY a past session — when the admin wants to test against the
ACTUAL players/courts from a previous session rather than fresh
CourtReserve data → call kickoff_from_history(date=...). Trigger
phrasings: "test run with last night's players", "replay last
week", "redo last session", "use last session's roster",
"rerun last Thursday with the new ratings", "replay 2026-04-30",
etc. Pass date=ISO when the admin names a specific date; omit it
to replay the most recent committed session. This is ALWAYS a
test run (commit refuses, no history is written) so don't also
pass test_mode anywhere — just call the tool. The replay loads
the past session's attendees + court_labels from history.json and
posts a structured TEST RUN message with the CURRENT roster
ratings (so changes since the original session are visible).
Useful for A/B testing rating changes ("does Tomoki at rating 1
vs rating 2 produce different pairings on the same roster?").

CRITICAL — both kickoff_session AND kickoff_from_history post
their own message into the channel. When either returns ok=true,
DO NOT write any reply text of your own. Output an empty
assistant turn (no characters at all). The structured kickoff
message has already reached the admin; an extra paraphrased
"Fresh test run is live 🎾" follow-up is duplicate noise. Only
write a reply when the tool returned ok=false — in which case
explain the error briefly so the admin knows what failed.

Phase routing:

A. phase == "awaiting_extras". The admin's reply contains some mix of:
     - Extra courts ("we also have courts 3 and 5") → call
       set_courts_for_tonight with the COMBINED list (CR courts from
       state.court_labels + the extras).
     - Late-arriving extra court ("court 5 is only available from R2
       and X, Y, Z, W are on it"; "extra court but booked until 8pm,
       4 names for that court") → call set_late_court(label,
       first_rotation, players=[4 names]). Use first_rotation=2 if
       the admin says "from 8:15 onwards" with a 19:30 start, or
       first_rotation=3 if they say "only the last rotation". The
       court label is auto-added to tonight's court_labels by the
       tool, so do NOT also call set_courts_for_tonight with it.
       Confirm to the admin: "Pinned <4 names> to court <X> from
       R<n> onwards — they'll sit out earlier rotations." If only 3
       names are supplied, ask for the 4th — the algorithm needs
       exactly 4.
     - Ratings ("Tomoki = 2", "Sarah is a 3") → set_player_rating per
       name.
     - Singles pins ("first singles match Amir vs Patrick", "rotation 2
       singles: Geoff Chapman vs Shinichi") — REMEMBER these as a
       pinned_singles list of {rotation_num, players}. DO NOT call
       generate_pairings yet; just collect.
     - Doubles pins ("Alan and Penny to play Peter and Ben in rotation
       2", "Alan and Penny vs Peter and Ben in one of the rotations",
       "have A+B partner against C+D in R1 on court 5") → call
       pin_doubles(players=[4 names], pairs=[[A,B],[C,D]],
       rotation_num=N or null for any-rotation, court_label=optional).
       The phrasing "X and Y to play P and Q" names the two
       partnerships explicitly — pairs=[['X','Y'],['P','Q']]. If the
       admin gives 4 names without saying who partners whom, ASK
       which pairs they want before calling pin_doubles. Confirm the
       pin to the admin: "Pinned <A>+<B> vs <C>+<D> in rotation <N>
       (court <X>)" — and remind them it won't be scored on its own
       but still counts as one of those players' rotations for the
       per-player rules.
     - "skip this week" / "no session" → call clear_tonight (resets
       phase to "") and acknowledge.
   Briefly acknowledge each change you applied, then prompt the admin
   for the next thing or for "go ahead". When the admin says "go
   ahead" / "generate pairings", call set_phase("ready_to_generate")
   and fall through to B.

B. phase == "ready_to_generate". Call generate_pairings(
   pinned_singles=<your collected pins>). On success call
   set_phase("draft_ready") and render the draft using the DRAFT
   format (see Pairing rendering below): simple "Here are the draft
   pairings." header, WITH ratings, with the score footer.

C. phase == "draft_ready". Re-render the current draft (DRAFT
   format) when asked. Apply adjustments via swap_players /
   swap_rotations / swap_courts and re-render each time. Do NOT
   call generate_pairings again unless the admin explicitly asks
   for a fresh re-roll.

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
   it"), call commit_plan, then call send_final_docx(pairings_text=
   "<FULL FINAL render>") — pass the full FINAL-format text as
   pairings_text. The tool posts that text first AND THEN posts the
   Word doc as an attachment, so the model MUST emit an empty
   assistant reply after this tool call (do not duplicate the text).
   Then set_phase("finalised") and proceed to D.

   IF the session is a test run (session_state.test_mode is True),
   commit_plan will refuse with error="test_mode". Don't propagate
   that as a failure — the test run is meant to preview a finalise
   without persisting. Reply with this short message verbatim — do
   NOT render the full FINAL preview by default (it's expensive
   to generate and the admin already saw the draft in this same
   chat):

     🧪 This is a test run — nothing has been written to history.
     • Say "boris show final preview" to see the members-facing copy + Word doc.
     • Say "boris clear this run" to wipe and start fresh.

   IF the admin then asks "show final preview" / "members
   preview" / "what would have been posted" / similar, render the
   FINAL format text (full preamble, NO ratings, copy-paste hint
   at the bottom) and pass it to send_final_docx(pairings_text=
   "<FULL FINAL render>"). The tool posts the text and the doc; the
   model MUST emit an empty assistant reply afterwards. Otherwise
   stay short.

   Don't call set_phase — leave the session in draft_ready so the
   admin can keep iterating or clear.

D. phase == "finalised". send_final_docx has already posted the
   FINAL pairings text + Word-doc attachment to this channel
   (during the commit step). Emit an empty assistant reply — DO
   NOT re-render the FINAL text or add any further commentary.
   The admin can copy the just-posted text and forward the doc to
   the players' WhatsApp group themselves.

   On the admin's next message, treat as a fresh start (phase has
   already been finalised; clear via clear_tonight if appropriate).

To start over mid-flow (any phase): "boris start over" / "kickoff
fresh" / "boris clear this run" / "boris clear this test run" /
"boris wipe this run" → clear_tonight, then run kickoff_session
again if asked.

Pairings rendering
------------------
Format for mobile WhatsApp (narrow screens). Two distinct formats:
DRAFT (used while iterating, phase=draft_ready) and FINAL (used
after commit, phase=finalised). They are intentionally different —
drafts are admin-facing review with ratings + score; finals are
exactly what the admin will copy-paste to members.

TEST RUN BANNER. When session_state.test_mode is True, prepend
this single line at the very top of EVERY render (draft, final
preview, every re-render in between), then a blank line, then the
rest of the message:

     🧪 TEST RUN — these pairings won't be saved in pairings history. Rating updates will still be saved.

ROTATION BLOCK (shared by both formats — DIFFERS in whether ratings
are shown). Header line with the full time range `(start-end)` from
each rotation's `start_time` / `end_time`, immediately followed (NO
blank line) by one court per line. Use `display_names` verbatim —
most players appear as just a first name; only those needing
disambiguation get a surname initial. Do NOT append `(doubles)` /
`(singles)` tags. Separate consecutive rotations with a single
blank line.

DRAFT FORMAT (phase=draft_ready). Header is exactly:

     Here are the draft pairings.

then a blank line, then the rotation block WITH RATINGS:
(a) bracket `[a v b]` between `Ct N` and the colon, using the
court's `bracket_values` field VERBATIM (do NOT recompute from
`ratings`); (b) append each player's rating in parens immediately
after their display name, from the `ratings` map. Show rating "?"
as the literal `?`. For singles, the bracket holds the two
individual ratings.

     Here are the draft pairings.

     Rotation 1 (19:30-20:15)
     Ct 4 [5 v 6]: Geoff(2) & Silvia(3) v Paul V(2) & Hannah(4)
     Ct 5 [4 v 5]: ...
     Ct 6 [3 v 2]: David(3) v Jack(2)

     Rotation 2 (20:15-20:55)
     ...

After the last rotation, include sit-outs (if any) and the plan's
`notes`, then add the SCORE FOOTER + COMMANDS HELP block. Format
exactly (substituting the live values for {score}, {permutations},
{seconds}):

     Residual score: {score} (lower is better; 0 = perfect fit to all rules). Tried {permutations} permutations, took {seconds}s. Say "boris score detail" for a breakdown of where the residual score comes from.

     If you'd like to make changes before finalising:
       • boris swap <name1> and <name2>          — swap two players for the whole evening
       • boris swap <name1> and <name2> in rotation <N>   — swap them in one rotation only
       • boris swap rotations <A> and <B>        — swap the contents of two whole rotations
       • boris swap courts <X> and <Y>           — swap matchups between two courts (across all rotations)
       • boris with ratings                      — re-show the draft with each player's rating
       • boris re-roll                           — generate a fresh draft from scratch
       • boris commit                            — finalise the draft and save to history

Pull the values from the plan dict:
  * {score}        = sum of each rotation's `metrics.rotations[*].best_score`
  * {permutations} = `metrics.multi_seed.total_permutations_tried` (an integer).
                     ALWAYS format with thousands separators in the
                     output — e.g. 75000 → "75,000", 1234567 →
                     "1,234,567". Never render the bare integer.
  * {seconds}      = `metrics.wall_seconds` (rounded to 1 dp is fine)

These come from the FRESHLY GENERATED plan only. After a swap or
re-render, the layout is the same but seconds/permutations are the
original generation's. Show them once after generate_pairings;
subsequent re-renders (after swaps) should just say "Residual
score: {score}. Say 'boris score detail' for a breakdown of where
the residual score comes from." — drop the permutations/seconds
line so it doesn't read as if the swap took a minute.

Render the COMMANDS HELP block VERBATIM as written — same bullets,
same wording. The admins copy commands from it. Do not add or
remove items, do not paraphrase, do not "be more conversational"
about it. The only adaptation is using literal court labels /
rotation numbers / player names if you want to give an example
that matches tonight's plan.

FINAL FORMAT (phase=finalised, OR test-run preview when commit
refuses). Header is the full members-facing preamble — date drawn
from the plan's `date` field, formatted as "Thursday Dth Month"
(e.g. "Thursday 30th April"). The second sentence MUST be on its
own line:

     Pairings for Thursday 30th April.
     At the end of each rotation please finish your game within a minute or two, if need be using a "next point wins" option.

then a blank line, then the rotation block WITHOUT ratings — no
bracket, no per-player parens:

     Rotation 1 (19:30-20:15)
     Ct 4: Geoff & Silvia v Paul V & Hannah
     Ct 5: Mei & Tomoki v Andy P & Sarah
     Ct 6: David v Jack

Include sit-outs and `notes` after the last rotation. NO score
footer in the final.

SCORE DRILL-DOWN (draft-phase only). The score footer appears in
drafts. Two ways the admin can dig deeper:
* "boris score" / "what's the score" → re-emit the one-line total.
* "boris score detail" / "breakdown" / "explain the score" / "more
  detail" → see SCORE DETAIL block below.

SCORE DETAIL (drill-down). When the admin asks "score detail",
"breakdown", "why is the score not zero", "explain the score",
"more detail" (in the context of a draft), use
`metrics.rotations[*].breakdown_items` — a list of attributed
contributions, each with ``rule``, ``points``, and optional
``court``, ``pair``, ``player`` fields. List one bullet per item,
named the involved player or court so the admin sees who/what is
driving the score.

Group by rotation. Skip rotations whose total best_score is 0.
Format:

     Score breakdown:

     Rotation 2: 32
       • a 2nd unbalanced rotation for Geoff C (Ct 5): 30
       • doubles pair-sum imbalance on Ct 5 (gap 1): 2
     Rotation 3: 30
       • a 3rd+ unbalanced rotation for Sarah F (Ct 4): 30

   Translate rule keys to plain English; include the attribution
   from the item:
     opponent_repeat            → "an opponent repeat between {pair} on Ct {court}"
     intra_partner              → "a partner repeat ({pair}) on Ct {court}"
     weekly_history             → "recent-week history pair ({pair}) on Ct {court}"
     same_court_successive      → "{pair} sharing a court again on Ct {court}"
     imbalance                  → "doubles pair-sum imbalance on Ct {court}"
     gender_3F1M                → "a 3F+1M court (Ct {court})"
     gender_MM_vs_FF            → "a men-vs-women segregated pairing on Ct {court}"
     rating_gap_unbalanced      → "an unbalanced court, rating gap 4-5 (Ct {court})"
     rating_gap_very_unbalanced → "a very unbalanced court, rating gap 6-7 (Ct {court})"
     rating_gap_extremely_unbalanced → "an extremely unbalanced court, rating gap 8-9 (Ct {court})"
     standard_too_low           → "{player} was among weaker company every rotation (even their best rotation was materially weaker than them)"
     top_player_no_strong_rotation → "{player} (a top player) never got a rotation with all strong company"
     hard_court_repeat          → "{player} played {hard_rotations}× on a hard court (clay preferred)"

   Use display_names (just the first name in most cases — the same
   short form used in the rendered pairings) when naming a player or
   pair so the admin can match them up. Sum each rotation's bullets
   and show the total at the rotation header. Aggregating items
   with the same rule + same attribution into one bullet is OK if
   it's tidier (e.g. several `weekly_history` hits on the same
   court).

QUALITY WARNING (DRAFT only). If `metrics.multi_seed.blocking_rules`
is non-empty, append a brief warning at the very end of the draft
(below the score footer):

     ⚠ Note: best total score was {chosen_total}; the optimizer
     tried {total_permutations_tried} different permutations and
     this was the best it could find. Couldn't fully avoid
     {rule list} (e.g. "an opponent repeat in rotation 3", "a
     rating-1 / rating-10 mix on the same court in rotation 2").
     Consider tweaking attendees or swapping a player.

   `total_permutations_tried` lives at
   `metrics.multi_seed.total_permutations_tried`. NEVER use the
   word "seeds" in the message — that's an internal implementation
   detail, meaningless to admins. Use "permutations",
   "combinations", or "layouts". Use the plain-English rule
   translations above for the rule names.
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
    duration_minutes: int = 90,
    court_label: Optional[str] = None,
    court_type: Optional[str] = None,
    court_preference: Optional[list[str]] = None,
    book_as: Optional[str] = None,
) -> dict:
    """Book a court for a CR account + ``partner_name``.

    Iterates the preference list (default: the booking account's
    configured order, else the club's standard order) until one court
    is available at the requested time. ``court_label`` (e.g. '5')
    overrides preference. ``court_type`` is 'clay' / 'acrylic'
    (synonym 'hard'); narrows the candidates if no specific court
    given. By default the CR account is whichever one is mapped to the
    WhatsApp sender; ``book_as`` (e.g. 'shirley') runs the booking
    under that club account's login instead — admin-only.
    """
    from accounts import cr_client

    acct, err = _resolve_booking_account(book_as)
    if err is not None:
        return err
    # No explicit order and no specific court/type → use the booking
    # account's configured preference if it has one (else the club
    # default, applied downstream in _build_court_candidates).
    if court_preference is None and court_label is None and court_type is None:
        court_preference = acct.court_preference_list()
    with cr_client(acct) as cr:
        result = cr.book_court(
            date=date,
            start_time_hhmm=start_time_hhmm,
            partner_name=partner_name,
            duration_minutes=duration_minutes,
            court_label=court_label,
            court_type=court_type,
            court_preference=court_preference,
        )
    if isinstance(result, dict):
        result.setdefault("booked_under", acct.display_name)
    return result


def tool_cancel_court_booking(
    reservation_id: Optional[str] = None,
    date: Optional[str] = None,
    start_time_hhmm: Optional[str] = None,
    book_as: Optional[str] = None,
) -> dict:
    """Cancel a court reservation.

    Either pass ``reservation_id`` directly, or provide ``date`` +
    ``start_time_hhmm`` and the tool will find the booking at that
    slot. The reservation lives in whichever account placed it, so to
    cancel a booking made under another member's login pass the same
    ``book_as`` used to create it (admin-only).
    """
    from accounts import cr_client

    acct, err = _resolve_booking_account(book_as)
    if err is not None:
        return err
    with cr_client(acct) as cr:
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
    duration_minutes: int = 90,
    court_label: Optional[str] = None,
    court_type: Optional[str] = None,
    notes: str = "",
    book_as: Optional[str] = None,
) -> dict:
    """Queue a future court booking that fires when CourtReserve's
    booking window opens (08:00 local, 7 days before ``play_date``).

    Validates ``partner_name`` against the Thursday roster + validated-
    members whitelist before persisting. Returns the saved entry's id
    + window_opens_at on success, or an error explaining what to fix.

    By default uses the caller's CR account; ``book_as`` (e.g.
    'shirley') queues it under that club account instead (admin-only)
    — the fire path resolves the account by key, so the account's
    saved court preference also applies when it eventually books.
    """
    import scheduled_bookings as sb
    from courtreserve import normalize_hhmm
    from validated_members import is_known_member

    account, _acct_err = _resolve_booking_account(book_as)
    if _acct_err is not None:
        return _acct_err

    # Canonicalise the time before persisting so every entry on disk
    # is in the single "HH:MM" shape that downstream consumers expect.
    try:
        start_time_hhmm = normalize_hhmm(start_time_hhmm)
    except ValueError as exc:
        return {
            "ok": False,
            "error": "invalid_start_time",
            "start_time_hhmm": start_time_hhmm,
            "message": (
                f"Couldn't parse start time: {exc}. "
                "Use 24h HH:MM, e.g. '13:00'."
            ),
        }

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


def tool_kickoff_session(
    session_type: Optional[str] = None,
    test_mode: bool = False,
) -> dict:
    """Run the kickoff workflow for one of the weekly tennis sessions.

    ``session_type`` is one of ``"tuesday"`` / ``"thursday"`` /
    ``"saturday"`` (the keys in session_types.SESSION_TYPES). When
    omitted (or None) the next session by weekday is picked — Mon/Tue
    → Tuesday, Wed/Thu → Thursday, Fri/Sat → Saturday, Sun → Tuesday.

    Behaviour: fetches the matching upcoming event from CourtReserve,
    auto-adds any new names to the roster, calls start_tonight
    (carrying the session_type through), sets phase = "awaiting_extras",
    and posts the structured "today's lineup + please reply with extras"
    message to the session's admin WhatsApp group.

    Pass ``test_mode=True`` for a dry run — the kickoff post is marked
    as a test, commit_plan and log_pairings_to_sheet are blocked, and
    rating updates still persist.
    """
    from thursday_kickoff import kickoff_session

    # Route the kickoff post back to whichever admin group asked for
    # it, so a "boris test run" from Boris the tennis bot doesn't spam
    # the live group.
    target_jid = _CURRENT_GROUP_JID.get(None)
    result = kickoff_session(
        session_key=session_type,
        test_mode=test_mode,
        target_jid=target_jid,
    )
    # Strip the (long) message text — it's already been posted.
    return {k: v for k, v in result.items() if k != "message"}


def tool_kickoff_from_history(date: Optional[str] = None) -> dict:
    """Replay a past committed session as a TEST RUN.

    Loads attendees + court_labels from history.json (most recent
    entry, or a specific date if given), starts a session_state
    flagged test_mode=True at phase=awaiting_extras, and posts the
    replay-style kickoff message to the calling channel. Always a
    test run — commit_plan refuses, no history is written.

    Use this when the admin says things like "test run with last
    night's players", "replay last week", "replay {date} with the
    new ratings", "use last session's roster" — anywhere they want
    to A/B test the algorithm against a known real-world roster
    rather than pull a fresh CourtReserve event.
    """
    from thursday_kickoff import kickoff_from_history

    target_jid = _CURRENT_GROUP_JID.get(None)
    result = kickoff_from_history(date=date, target_jid=target_jid)
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
    trigger came from the "Boris the tennis bot" group.
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
    """Update a player's 1-10 rating. Fuzzy-matches the name; ambiguous
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


def tool_set_player_gender(name: str, gender: str) -> dict:
    """Update a player's gender (M / F / ?). Fuzzy-matches the name;
    ambiguous matches return an error with the candidates so the admin
    can pick.

    Used for messages like 'Longjie Jia is male' / 'Sam Fairhurst is F'
    / 'reset Pat's gender to unknown'. The gender-guesser usually
    nails it on roster-add but occasionally misclassifies an unfamiliar
    first name — this tool is for the manual override.
    """
    g = (gender or "").strip().upper()
    if g in {"MALE", "MAN"}:
        g = "M"
    elif g in {"FEMALE", "WOMAN"}:
        g = "F"
    elif g in {"UNKNOWN", "?", ""}:
        g = "?"
    if g not in {"M", "F", "?"}:
        return {
            "ok": False,
            "error": "invalid_gender",
            "message": "gender must be M, F, or ? (also accepts "
            "male/female/man/woman/unknown).",
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
    try:
        entry = roster.set_gender(name, g)
    except ValueError as e:
        return {"ok": False, "error": "invalid_gender", "message": str(e)}
    return {"ok": True, "name": name, "entry": entry}


def _names_seen_in_courtreserve(lookback_history: int = 4) -> set[str]:
    """Names CourtReserve has used recently — drawn from the current
    in-flight session state (attendees + waitlist) plus the last
    ``lookback_history`` history.json entries' attendees lists.

    Used by the duplicate-finder to flag which of a pair of variants
    is the one CR currently writes — important because if we keep the
    wrong spelling, the next CR scrape will silently re-create the
    duplicate.
    """
    seen: set[str] = set()
    try:
        from session_state import get_tonight
        st = get_tonight()
        seen.update(st.attendees or [])
        seen.update(st.waitlist or [])
    except Exception:
        pass
    if HISTORY_PATH.exists():
        try:
            with HISTORY_PATH.open(encoding="utf-8") as f:
                history = json.load(f)
            for entry in history[-max(0, lookback_history):]:
                seen.update(entry.get("attendees") or [])
        except Exception:
            pass
    return seen


def tool_find_roster_duplicates() -> dict:
    """Scan the roster for likely-duplicate player rows.

    Catches the common causes — curly-vs-straight apostrophe, nickname
    variants (Ben/Benjamin, Mike/Michael...), whitespace/case differences.
    For each duplicate group, marks which variant is the one CourtReserve
    is currently writing (drawn from the in-flight session's attendees +
    the last few history entries) so the admin keeps the canonical
    spelling — otherwise the next CR scrape will re-create the duplicate.

    Read-only — does not delete anything. The follow-up tool
    ``merge_and_delete_player`` handles the destructive step (under
    confirmation).

    Returns ``{ok: True, groups: [...]}`` where each group is:

        {"key": canonical-key,
         "hint": "apostrophe variant" | "nickname/whitespace variant",
         "names": [
            {"name": ..., "rating": ..., "gender": ...,
             "in_courtreserve": bool},
            ...
         ],
         "suggested_keep": <name from group> | None,
         "suggested_delete": [<name>, ...]}
    """
    from roster import find_duplicates

    roster = Roster()
    data = roster.all()
    groups_raw = find_duplicates(data)
    cr_seen = _names_seen_in_courtreserve()

    enriched: list[dict] = []
    for g in groups_raw:
        names_info = []
        for n in g["names"]:
            entry = data.get(n, {})
            names_info.append({
                "name": n,
                "rating": entry.get("rating", "?"),
                "gender": entry.get("gender", "?"),
                "singles": entry.get("singles", ""),
                "in_courtreserve": n in cr_seen,
            })
        cr_names = [i["name"] for i in names_info if i["in_courtreserve"]]
        # Suggest keeping the CR-canonical spelling so the next scrape
        # doesn't re-create the duplicate. If multiple CR-seen variants
        # exist (rare — would mean CR changed its mind) or none do,
        # leave the choice to the admin.
        suggested_keep = cr_names[0] if len(cr_names) == 1 else None
        suggested_delete = (
            [n for n in g["names"] if n != suggested_keep]
            if suggested_keep else []
        )
        enriched.append({
            "key": g["key"],
            "hint": g["hint"],
            "names": names_info,
            "suggested_keep": suggested_keep,
            "suggested_delete": suggested_delete,
        })

    return {"ok": True, "groups": enriched, "group_count": len(enriched)}


def _merged_entry_preview(keep_entry: dict, delete_entry: dict) -> dict:
    """For each roster field, return the value that would land on
    ``keep`` after a merge. The rule: only copy from ``delete`` when
    ``keep`` is at its default / empty value and ``delete`` has real
    data. Never overwrites a populated keep-side field.
    """
    merged = dict(keep_entry)
    # Rating: copy delete→keep only if keep is "?".
    if keep_entry.get("rating", "?") == "?":
        d_rating = delete_entry.get("rating", "?")
        if d_rating != "?":
            merged["rating"] = d_rating
    # Gender: copy only if keep is "?".
    if keep_entry.get("gender", "?") == "?":
        d_gender = delete_entry.get("gender", "?")
        if d_gender != "?":
            merged["gender"] = d_gender
    # Phone / singles / notes: copy only if keep is empty.
    for field in ("phone", "singles", "notes"):
        if not (keep_entry.get(field) or "").strip():
            d_val = (delete_entry.get(field) or "").strip()
            if d_val:
                merged[field] = d_val
    return merged


def tool_merge_and_delete_player(
    keep_name: str,
    delete_name: str,
    confirm: bool = False,
) -> dict:
    """Resolve a duplicate by merging fields from ``delete_name`` into
    ``keep_name`` (only filling default/empty slots — never overwriting
    populated data on the keep side), then deleting the redundant row.

    DESTRUCTIVE — pass ``confirm=True`` to actually delete. Without
    confirm the tool returns a *preview* showing exactly what the
    merged keep-side entry would look like and what would be deleted,
    so the admin can verify before committing.

    Both names are exact-match against the roster (no fuzzy fallback —
    you wouldn't want a typo here to delete a different player).
    Call ``find_roster_duplicates`` first to get the exact names.
    """
    roster = Roster()
    keep = roster.get(keep_name)
    delete = roster.get(delete_name)
    if keep is None:
        return {"ok": False, "error": "keep_not_found", "name": keep_name}
    if delete is None:
        return {"ok": False, "error": "delete_not_found", "name": delete_name}
    if keep_name == delete_name:
        return {
            "ok": False,
            "error": "same_name",
            "message": "keep_name and delete_name are the same row.",
        }

    merged_preview = _merged_entry_preview(keep, delete)
    diff = {
        f: {"before": keep.get(f), "after": merged_preview.get(f)}
        for f in ("rating", "gender", "phone", "singles", "notes")
        if keep.get(f) != merged_preview.get(f)
    }

    if not confirm:
        return {
            "ok": False,
            "error": "confirmation_required",
            "preview": True,
            "keep_name": keep_name,
            "delete_name": delete_name,
            "keep_before": keep,
            "delete_entry": delete,
            "keep_after_merge": merged_preview,
            "fields_changing_on_keep": diff,
            "message": (
                f"Preview only. Will copy {len(diff)} field(s) from "
                f"{delete_name!r} into {keep_name!r}, then delete "
                f"{delete_name!r}. Re-call with confirm=true to proceed."
            ),
        }

    # Apply each field change first; if any of those fail we haven't
    # deleted anything yet and the admin can retry. Only AFTER the
    # merge succeeds do we drop the redundant row.
    applied: dict = {}
    for field, change in diff.items():
        new_val = change["after"]
        try:
            if field == "rating":
                roster.set_rating(keep_name, new_val)
            elif field == "gender":
                roster.set_gender(keep_name, new_val)
            elif field == "phone":
                roster.set_phone(keep_name, new_val)
            elif field == "singles":
                roster.set_singles(keep_name, new_val)
            # Notes column has no setter today — skip with a flag so
            # the admin knows. Rare in practice (notes are usually
            # blank), worth flagging if it ever bites.
            elif field == "notes":
                applied[field] = "skipped (no notes-setter)"
                continue
            applied[field] = new_val
        except Exception as e:
            return {
                "ok": False,
                "error": "merge_failed",
                "field": field,
                "applied_before_failure": applied,
                "message": (
                    f"Failed to copy {field} → {keep_name!r}: {e!r}. "
                    f"Redundant row {delete_name!r} NOT deleted. "
                    "Investigate, then retry."
                ),
            }

    try:
        deleted_entry = roster.delete(delete_name)
    except Exception as e:
        return {
            "ok": False,
            "error": "delete_failed",
            "merged_fields": applied,
            "message": (
                f"Merge succeeded but delete of {delete_name!r} failed: "
                f"{e!r}. Roster now has data on {keep_name!r} but the "
                "duplicate row is still present."
            ),
        }
    return {
        "ok": True,
        "kept": keep_name,
        "deleted": delete_name,
        "merged_fields": applied,
        "deleted_entry": deleted_entry,
        "keep_after_merge": merged_preview,
        "message": (
            f"Deleted {delete_name!r} and copied "
            f"{len(applied)} field(s) into {keep_name!r}."
        ),
    }


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
    start_time_hhmm: Optional[str] = None,
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

    ``start_time_hhmm`` / ``rotation_durations`` default to the
    in-flight session_type's values (Tue/Thu evening 19:30 + 45/40/35;
    Saturday afternoon 14:00 + 45/40/35) when not explicitly passed.
    """
    from pairings import make_plan
    from session_state import get_tonight, set_draft_plan
    from session_types import SESSION_TYPES

    state = get_tonight()
    names = list(attendee_names) if attendee_names is not None else list(state.attendees)
    labels = court_labels if court_labels is not None else (state.court_labels or None)
    # Resolve session-type defaults for start time + rotation durations
    # when the caller didn't pin them explicitly.
    st = SESSION_TYPES.get(state.session_type) if state.session_type else None
    if start_time_hhmm is None:
        start_time_hhmm = st.start_time_hhmm if st is not None else "19:30"
    if rotation_durations is None and st is not None:
        # Only inherit from the registry when the rotation count
        # matches — otherwise the algorithm's own fallback handles it.
        if len(st.rotation_durations) == num_rotations:
            rotation_durations = list(st.rotation_durations)
    if not names:
        return {
            "ok": False,
            "error": "no_attendees",
            "message": "No attendees supplied and session state is empty. "
            "Run start_tonight first, or pass attendee_names.",
        }

    # Heads-up: pairings generation can take up to a minute (multi-seed
    # greedy + hill-climb polish). Post a notice into the calling
    # channel before kicking off so the admin doesn't think the bot
    # has stalled. Best-effort — if we can't resolve a channel jid,
    # carry on without the notice.
    notice_jid = _CURRENT_GROUP_JID.get(None)
    if notice_jid:
        try:
            send_to_group(
                notice_jid,
                BOT_REPLY_PREFIX
                + "Generating pairings — this may take up to a minute…",
            )
        except Exception:
            pass
    if not labels and num_courts is None:
        return {
            "ok": False,
            "error": "no_courts",
            "message": "No court_labels in session state and num_courts not given. "
            "Set courts via set_courts_for_tonight or pass court_labels.",
        }

    # Pick up the late-court config (if any) from session state so it
    # auto-applies to the run — the admin sets it via set_late_court
    # and we honour it here without needing it on every call.
    late_court_arg: dict | None = None
    if state.late_court_label and state.late_court_first_rotation >= 1:
        late_court_arg = {
            "label": state.late_court_label,
            "first_rotation": state.late_court_first_rotation,
            "pinned_players": list(state.late_court_pinned_players),
        }
    # Same for any admin-pinned doubles match-ups (pin_doubles tool).
    pinned_doubles_arg: list[dict] | None = (
        list(state.pinned_doubles) if state.pinned_doubles else None
    )

    # Best-of-2 wrapper: run make_plan; if first score >= threshold,
    # post the retry notice and run a second time, then keep the
    # lower-scoring plan with the loser's work folded into its metrics.
    def _post_retry_notice() -> None:
        retry_jid = _CURRENT_GROUP_JID.get(None)
        if retry_jid:
            send_to_group(retry_jid, BOT_REPLY_PREFIX + GENERATE_RETRY_NOTICE)

    try:
        plan = _generate_with_retry(
            notice_callback=_post_retry_notice,
            make_plan_fn=make_plan,
            attendees=names,
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
            pinned_doubles=pinned_doubles_arg,
            late_court=late_court_arg,
            seed=seed,
        )
    except ValueError as e:
        return {"ok": False, "error": "over_capacity_or_bad_input", "message": str(e)}
    result = plan.to_dict()
    # Tag the plan with the in-flight session type so it persists into
    # history.json on commit (and downstream — DOCX template choice,
    # replay-from-history, sheet log column).
    if state.session_type:
        result["session_type"] = state.session_type
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


def tool_swap_courts(
    label_a: str,
    label_b: str,
    rotation_nums: Optional[list[int]] = None,
) -> dict:
    """Swap the contents of two courts in the current draft plan.

    Use for admin requests like 'put singles on court 5' (then swap the
    current singles court with court 5) or 'move courts 7 and 9'. Court
    labels stay put — only the matchups move — so pinned slots elsewhere
    in the plan are unaffected.

    ``rotation_nums`` scopes the swap to specific rotations (1-based).
    Omit for an all-rotations swap (the default). Use the list form
    for messages like 'swap courts 1 and 5 for rotations 2 and 3'
    when only some rotations should change (e.g. the same group is on
    the less-preferred hard court for those rotations and should move
    to the clay court).
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
        swapped = swap_courts_in_plan(
            plan, label_a, label_b, rotation_nums=rotation_nums,
        )
    except ValueError as e:
        return {"ok": False, "error": "swap_failed", "message": str(e)}
    set_draft_plan(plan)
    return {
        "ok": True,
        "swapped": [str(label_a), str(label_b)],
        "rotations_swapped": swapped,
        "plan": plan,
    }


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
    from session_state import (
        clear_draft_plan, get_draft_plan, get_tonight, record_commit,
    )
    from session_log import log_plan

    if get_tonight().test_mode:
        return {
            "ok": False,
            "error": "test_mode",
            "message": (
                "Test run — these pairings won't be saved in pairings "
                "history. The draft is fine to keep iterating on, but "
                "it won't be written to history.json or the Sheet. Say "
                "'boris clear tonight' to wipe the dry run, or stop here."
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
    # Remember what we just committed so an inadvertent commit can be
    # undone (undo_commit). Stash before clearing the draft.
    record_commit(
        plan,
        sheet_session_rows=(
            (sheet_log or {}).get("session_rows_appended", 0)
        ),
        sheet_pair_rows=(sheet_log or {}).get("pair_rows_appended", 0),
    )
    clear_draft_plan()
    return {
        "ok": True,
        "history_appended": True,
        "sheet_log": sheet_log,
        "sheet_error": sheet_error,
        "undo_hint": "Say 'boris undo commit' to reverse this.",
    }


def tool_undo_commit() -> dict:
    """Reverse the most recent ``commit_plan``.

    Removes the just-appended ``history.json`` entry, deletes the
    matching Session/Pair-log rows from the Sheet (best-effort),
    restores the committed plan as the live draft and sets the
    workflow phase back to ``draft_ready`` so the admin can keep
    editing. Only the latest commit, and only before
    ``clear_tonight`` wipes session state. Idempotent-ish: a second
    undo with nothing to reverse returns ``error="nothing_to_undo"``.
    """
    import json
    from session_state import (
        clear_last_commit, get_last_commit, set_draft_plan, set_phase,
    )

    lc = get_last_commit()
    if not lc:
        return {
            "ok": False,
            "error": "nothing_to_undo",
            "message": (
                "No recent commit to undo (already undone, or the "
                "session was cleared)."
            ),
        }
    plan = lc.get("plan") or {}
    date = lc.get("date", "") or plan.get("date", "")

    # 1) Remove the committed entry from history.json — the LAST entry
    #    matching this date (the one commit_plan appended).
    history_removed = False
    try:
        if HISTORY_PATH.exists():
            with HISTORY_PATH.open(encoding="utf-8") as f:
                hist = json.load(f)
            if isinstance(hist, list):
                for i in range(len(hist) - 1, -1, -1):
                    rec = hist[i]
                    if isinstance(rec, dict) and rec.get("date") == date:
                        hist.pop(i)
                        history_removed = True
                        break
                if history_removed:
                    HISTORY_PATH.write_text(
                        json.dumps(hist, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
    except Exception as e:
        return {
            "ok": False,
            "error": "history_rewrite_failed",
            "message": str(e),
        }

    # 2) Delete the mirrored Sheet rows (best-effort — sheet may be
    #    down; the authoritative undo is the history.json rewrite).
    sheet_result = None
    sheet_error = None
    s_rows = int(lc.get("sheet_session_rows", 0) or 0)
    p_rows = int(lc.get("sheet_pair_rows", 0) or 0)
    if s_rows or p_rows:
        try:
            from session_log import unlog_plan
            sheet_result = unlog_plan(date, s_rows, p_rows)
        except Exception as e:
            sheet_error = str(e)

    # 3) Restore the draft and reopen the workflow for more edits.
    set_draft_plan(plan)
    try:
        set_phase("draft_ready")
    except Exception:
        pass
    clear_last_commit()

    return {
        "ok": True,
        "date": date,
        "history_entry_removed": history_removed,
        "sheet_result": sheet_result,
        "sheet_error": sheet_error,
        "message": (
            "Commit undone — the draft is restored and editable again. "
            + ("" if history_removed else
               "(No matching history.json entry was found to remove.) ")
            + "Re-commit with 'boris commit' when ready."
        ),
    }


DOCX_ATTACHMENT_CAPTION = (
    "The attached word doc contains these pairings in a format "
    "suitable for printing"
)


def tool_send_rules_pdf() -> dict:
    """Generate the pairing-rules PDF and send it to this channel.

    Always re-renders from the live constants in pairings.RULE_DOCS,
    so the attachment is up-to-date the moment any weight changes
    (no separate sync step needed). Sends as a WhatsApp attachment
    with a short covering caption.
    """
    from rules_pdf import render_rules_pdf

    jid = _CURRENT_GROUP_JID.get(None)
    if not jid:
        return {
            "ok": False,
            "error": "no_channel",
            "message": "no calling-channel JID available to send to.",
        }
    try:
        render_rules_pdf(RULES_PDF_PATH)
    except Exception as e:
        return {"ok": False, "error": "render_failed", "message": str(e)}
    ok = send_doc_to_group(
        jid, RULES_PDF_PATH,
        caption=BOT_REPLY_PREFIX + RULES_PDF_CAPTION,
    )
    return {
        "ok": ok,
        "path": str(RULES_PDF_PATH),
        "sent_to": jid,
        "message": (
            "Sent the rules-and-weights PDF to this channel."
            if ok else
            f"PDF generated but the bridge refused to send it; file is at {RULES_PDF_PATH}."
        ),
    }


def tool_send_final_docx(pairings_text: str) -> dict:
    """Post the final pairings text, then the Word doc, to this channel.

    Sends two separate WhatsApp messages, in this order:
      1. ``pairings_text`` (the rendered FINAL pairings block, exactly
         as the admin would copy-paste to members). Boris's standard
         "From Boris the tennis bot: " prefix is added.
      2. The Thursday Social Tennis Word doc as an attachment, with
         the caption "The attached word doc contains these pairings
         in a format suitable for printing".

    The plan source (preference order): the current draft_plan
    (preview path) or the latest history.json entry (post-commit
    path). The doc is rendered from the session's DOCX template (one
    per session type — Thursday has its own; Tuesday and Saturday
    share a Westside-branded template) and saved to
    ``output_files/<session display name> - <date>.docx``.

    Because the tool posts the text itself, the model should emit an
    empty assistant reply after calling this tool — same convention as
    kickoff_session.
    """
    from pairings_docx import render_final_docx
    from session_state import get_draft_plan, get_tonight

    plan = get_draft_plan()
    if not plan:
        if HISTORY_PATH.exists():
            try:
                with HISTORY_PATH.open(encoding="utf-8") as f:
                    history = json.load(f)
                if history:
                    plan = history[-1]
            except Exception:
                plan = None
    if not plan:
        return {
            "ok": False,
            "error": "no_plan",
            "message": "No draft plan in session state and no history "
            "entry to render — run generate_pairings first.",
        }

    # Pick the template + filename basename from the in-flight
    # session's type. Falls back to Thursday if the plan was generated
    # before session_type plumbing existed.
    session_type = (
        plan.get("session_type") if isinstance(plan, dict) else ""
    ) or get_tonight().session_type
    template_path = _docx_template_for(session_type)
    if not template_path.exists():
        return {
            "ok": False,
            "error": "template_missing",
            "message": f"Template not found at {template_path}.",
        }

    if not isinstance(pairings_text, str) or not pairings_text.strip():
        return {
            "ok": False,
            "error": "no_text",
            "message": "pairings_text is required — render the FINAL "
            "pairings block and pass it as pairings_text.",
        }

    jid = _CURRENT_GROUP_JID.get(None)
    if not jid:
        return {
            "ok": False,
            "error": "no_channel",
            "message": "no calling-channel JID available to send to.",
        }

    safe_date = (plan.get("date") or "session").replace("/", "-")
    basename = _docx_basename_for(session_type)
    out_path = FINAL_DOCX_OUTPUT_DIR / f"{basename} - {safe_date}.docx"
    try:
        render_final_docx(
            plan, template_path, out_path,
            preamble_paragraph_count=_docx_preamble_count_for(session_type),
            header_text=_docx_header_text_for(session_type),
        )
    except Exception as e:
        return {"ok": False, "error": "render_failed", "message": str(e)}

    # 1) Text message — full FINAL pairings, with Boris's prefix.
    text_ok = send_to_group(jid, BOT_REPLY_PREFIX + pairings_text)
    # 2) Doc attachment — with explanatory caption (also prefixed).
    doc_ok = send_doc_to_group(
        jid, out_path,
        caption=BOT_REPLY_PREFIX + DOCX_ATTACHMENT_CAPTION,
    )
    return {
        "ok": text_ok and doc_ok,
        "text_sent": text_ok,
        "doc_sent": doc_ok,
        "path": str(out_path),
        "sent_to": jid,
        "message": (
            "Posted FINAL pairings text + Word-doc attachment."
            if (text_ok and doc_ok) else
            f"Partial send: text_ok={text_ok}, doc_ok={doc_ok}. "
            f"File is at {out_path}."
        ),
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


def tool_set_late_court(
    label: str,
    first_rotation: int,
    players: list[str],
) -> dict:
    """Configure tonight's late-arriving extra court.

    A court that's booked by someone else until ``first_rotation``
    starts. The 4 named players are pinned to that court in
    ``first_rotation`` (Boris picks the 2v2 split); they sit out
    earlier rotations; the court is added to the regular pool from
    ``first_rotation`` onwards.

    Names are fuzzy-matched against tonight's attendees so the admin
    can supply partial names ("Geoff", "Maggie"). Returns an error if
    fewer/more than 4 unique attendees match.
    """
    from session_state import find_attendee_fuzzy, set_late_court

    if not isinstance(players, list) or len(players) != 4:
        return {
            "ok": False,
            "error": "bad_players",
            "message": "late_court needs exactly 4 player names (got "
            f"{players!r}).",
        }

    resolved: list[str] = []
    for raw in players:
        matches = find_attendee_fuzzy(str(raw))
        if len(matches) == 0:
            return {
                "ok": False,
                "error": "no_match",
                "message": f"No attendee matches {raw!r} — check the name.",
            }
        if len(matches) > 1:
            return {
                "ok": False,
                "error": "ambiguous",
                "message": (
                    f"{raw!r} matches multiple attendees: {matches}. "
                    "Use a more specific name."
                ),
            }
        resolved.append(matches[0])
    if len(set(resolved)) != 4:
        return {
            "ok": False,
            "error": "duplicates",
            "message": f"Pinned players must be distinct (resolved: {resolved}).",
        }
    try:
        state = set_late_court(str(label), int(first_rotation), resolved)
    except ValueError as e:
        return {"ok": False, "error": "invalid", "message": str(e)}
    return {
        "ok": True,
        "label": state.late_court_label,
        "first_rotation": state.late_court_first_rotation,
        "players": list(state.late_court_pinned_players),
        "state": state.to_dict(),
    }


def tool_clear_late_court() -> dict:
    """Remove any configured late-court pinning for tonight."""
    from session_state import clear_late_court

    state = clear_late_court()
    return {"ok": True, "state": state.to_dict()}


def tool_pin_doubles(
    players: list[str],
    pairs: list[list[str]],
    rotation_num: Optional[int] = None,
    court_label: Optional[str] = None,
) -> dict:
    """Pin a 4-player doubles match-up for tonight.

    ``players`` is the 4-name list, ``pairs`` is the partnership split
    (two 2-name lists whose union equals ``players``). All names are
    fuzzy-matched against tonight's attendees, so partial names are
    fine. ``rotation_num`` is 1-based; omit it for "any rotation, the
    optimiser picks". ``court_label`` is optional; omit it to let the
    optimiser pick a free doubles court in the chosen rotation.

    The pinned court is excluded from per-court scoring (admin's
    choice), but still feeds the cross-rotation tallies: a partner
    repeat with the pinned pair elsewhere will be penalised, the
    pinned 4 players' ``unbalanced_count`` ticks up if the court is
    non-balanced, and the whole-evening per-player rules (e.g.
    ``standard_too_low``, ``top_player_no_strong_rotation``) treat the
    pinned rotation as one of the player's rotations.
    """
    from session_state import add_pinned_doubles, find_attendee_fuzzy

    if not isinstance(players, list) or len(players) != 4:
        return {
            "ok": False,
            "error": "bad_players",
            "message": "pin_doubles needs exactly 4 player names "
            f"(got {players!r}).",
        }
    if (
        not isinstance(pairs, list)
        or len(pairs) != 2
        or any(not isinstance(p, list) or len(p) != 2 for p in pairs)
    ):
        return {
            "ok": False,
            "error": "bad_pairs",
            "message": "pairs must be two 2-name lists, e.g. "
            "[['Alan','Penny'],['Peter','Ben']] "
            f"(got {pairs!r}).",
        }

    def _resolve(name: str) -> dict | str:
        matches = find_attendee_fuzzy(str(name))
        if not matches:
            return {
                "ok": False,
                "error": "no_match",
                "message": f"No attendee matches {name!r} — check the name.",
            }
        if len(matches) > 1:
            return {
                "ok": False,
                "error": "ambiguous",
                "message": (
                    f"{name!r} matches multiple attendees: {matches}. "
                    "Use a more specific name."
                ),
            }
        return matches[0]

    resolved_players: list[str] = []
    for raw in players:
        r = _resolve(raw)
        if isinstance(r, dict):
            return r
        resolved_players.append(r)
    if len(set(resolved_players)) != 4:
        return {
            "ok": False,
            "error": "duplicates",
            "message": (
                f"Pinned players must be distinct (resolved: {resolved_players})."
            ),
        }
    resolved_pairs: list[list[str]] = []
    for pair in pairs:
        resolved_pair: list[str] = []
        for raw in pair:
            r = _resolve(raw)
            if isinstance(r, dict):
                return r
            resolved_pair.append(r)
        resolved_pairs.append(resolved_pair)
    # Pairs must partition the players exactly.
    flat = [p for pair in resolved_pairs for p in pair]
    if sorted(flat) != sorted(resolved_players):
        return {
            "ok": False,
            "error": "pairs_mismatch",
            "message": (
                "The pair structure doesn't match the 4 named players. "
                f"Players resolved to {resolved_players}; "
                f"pairs resolved to {resolved_pairs}."
            ),
        }

    try:
        state = add_pinned_doubles(
            resolved_players,
            resolved_pairs,
            rotation_num=rotation_num,
            court_label=(str(court_label) if court_label is not None else None),
        )
    except ValueError as e:
        return {"ok": False, "error": "invalid", "message": str(e)}
    return {
        "ok": True,
        "players": resolved_players,
        "pairs": resolved_pairs,
        "rotation_num": rotation_num,
        "court_label": court_label,
        "pinned_doubles": list(state.pinned_doubles),
    }


def tool_clear_pinned_doubles() -> dict:
    """Remove every pinned-doubles entry for tonight."""
    from session_state import clear_pinned_doubles

    state = clear_pinned_doubles()
    return {"ok": True, "state": state.to_dict()}


TOOL_IMPLS: dict[str, Any] = {
    "list_club_sessions": tool_list_club_sessions,
    "get_session_registrants": tool_get_session_registrants,
    "kickoff_session": tool_kickoff_session,
    "kickoff_from_history": tool_kickoff_from_history,
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
    "set_player_gender": tool_set_player_gender,
    "find_roster_duplicates": tool_find_roster_duplicates,
    "merge_and_delete_player": tool_merge_and_delete_player,
    "set_singles_preference": tool_set_singles_preference,
    "read_pairings_history": tool_read_pairings_history,
    "generate_pairings": tool_generate_pairings,
    "swap_players": tool_swap_players,
    "swap_rotations": tool_swap_rotations,
    "swap_courts": tool_swap_courts,
    "commit_plan": tool_commit_plan,
    "undo_commit": tool_undo_commit,
    "send_final_docx": tool_send_final_docx,
    "send_rules_pdf": tool_send_rules_pdf,
    "log_pairings_to_sheet": tool_log_pairings_to_sheet,
    "start_tonight": tool_start_tonight,
    "get_tonight": tool_get_tonight,
    "add_to_tonight": tool_add_to_tonight,
    "remove_from_tonight": tool_remove_from_tonight,
    "set_courts_for_tonight": tool_set_courts_for_tonight,
    "promote_from_waitlist": tool_promote_from_waitlist,
    "clear_tonight": tool_clear_tonight,
    "set_late_court": tool_set_late_court,
    "clear_late_court": tool_clear_late_court,
    "pin_doubles": tool_pin_doubles,
    "clear_pinned_doubles": tool_clear_pinned_doubles,
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
        "name": "kickoff_session",
        "description": "Run the kickoff workflow for one of the weekly "
        "Westside social tennis sessions (Tuesday evening, Thursday "
        "evening, or Saturday afternoon). Fetches the upcoming "
        "matching event from CourtReserve, auto-adds new names to the "
        "roster, calls start_tonight (carrying the session_type), sets "
        "the workflow phase to 'awaiting_extras', and POSTS the "
        "structured 'today's lineup + please reply with extras' "
        "message to the session's admin group. Use this when an "
        "organiser says 'boris kickoff', 'start the workflow', "
        "'kick off the Saturday session' etc. If they don't specify a "
        "day, omit session_type — it defaults to the NEXT scheduled "
        "session by weekday (Mon/Tue → Tuesday, Wed/Thu → Thursday, "
        "Fri/Sat → Saturday, Sun → Tuesday). Set test_mode=true when "
        "they ask for a test/dry/practice/rehearse run — the post is "
        "marked TEST RUN, commit_plan and log_pairings_to_sheet refuse "
        "to write, rating updates still persist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_type": {
                    "type": "string",
                    "enum": ["tuesday", "thursday", "saturday"],
                    "description": "Which weekly session to kick off. "
                    "Omit (or pass null) to default to the next "
                    "scheduled session by today's weekday.",
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
        "name": "kickoff_from_history",
        "description": "Replay a past committed session as a TEST RUN — "
        "load attendees + court_labels from history.json (most recent "
        "entry by default, or a specific date) and start a session in "
        "test_mode=True at phase=awaiting_extras. Use this when the "
        "admin wants to A/B test the algorithm or a rating change "
        "against a known real-world roster rather than whatever's "
        "currently signed up on CourtReserve. Trigger phrasings: "
        "'test run with last night's players', 'replay last week', "
        "'redo last session with the new ratings', 'replay 2026-04-30', "
        "'use last session's roster for a test run', etc. Always a "
        "test run — commit refuses, no history is written.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Optional ISO date YYYY-MM-DD picking "
                    "a specific past session. Omit to replay the most "
                    "recent committed session.",
                }
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
        "in the 'Boris the tennis bot' admin group.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "cancel_booking",
        "description": "Remove the bot's account from an event (whether "
        "currently registered or waitlisted). Idempotent — returns "
        "'not_registered' if there's nothing to cancel. Use list_my_bookings "
        "first to find the right reservation_number_or_res_id. ONLY "
        "available in the 'Boris the tennis bot' admin group.",
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
        "preference list (default: 5,6,9,7,8,10,11,12,4,1,2,3) until "
        "one is free at the requested slot. Pass court_label (e.g. '5') "
        "to force a specific court, or court_type ('clay'|'acrylic') to "
        "narrow the candidates. Duration: 30, 60 (default), or 90 min. "
        "ONLY available in the 'Boris the tennis bot' admin group.",
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
                    "enum": [30, 60, 90, 120],
                    "default": 90,
                    "description": "30 / 60 / 90 / 120 minutes — maps "
                    "to the matching CourtReserve reservation type: "
                    "30='30 min hit', 60='60 min hit', "
                    "90='1 hour 30 min hit' (default), "
                    "120='ladder match'.",
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
                "book_as": {
                    "type": "string",
                    "description": "Optional: place the booking under a "
                    "DIFFERENT club account's CourtReserve login, e.g. "
                    "'shirley' or 'Shirley Chapman'. Use this whenever "
                    "the admin says things like 'book in Shirley's name "
                    "/ using her login / her credentials'. Admin "
                    "(Geoff) account only. The chosen account's saved "
                    "court preference automatically applies.",
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
        "ONLY available in the 'Boris the tennis bot' admin group.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reservation_id": {"type": "string"},
                "date": {"type": "string", "description": "ISO date YYYY-MM-DD."},
                "start_time_hhmm": {"type": "string"},
                "book_as": {
                    "type": "string",
                    "description": "If the booking was placed under "
                    "another member's login (e.g. 'shirley'), pass the "
                    "same value here so it's cancelled in the right "
                    "account. Admin only.",
                },
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
                    "description": "Start time of play, 24h HH:MM "
                    "(e.g. '08:00', '13:00').",
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
                    "enum": [30, 60, 90, 120],
                    "default": 90,
                    "description": "30 / 60 / 90 / 120 — maps to the "
                    "matching CourtReserve reservation type: "
                    "30='30 min hit', 60='60 min hit', "
                    "90='1 hour 30 min hit' (default), "
                    "120='ladder match'.",
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
                "book_as": {
                    "type": "string",
                    "description": "Optional: queue the booking under a "
                    "DIFFERENT club account's CourtReserve login, e.g. "
                    "'shirley'. Use when the admin says 'schedule it in "
                    "Shirley's name / using her login'. Admin (Geoff) "
                    "account only. That account's saved court "
                    "preference applies when it fires.",
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
        "{gender, rating, notes}. Rating is an integer 1-10 or the string '?'.",
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
        "description": "Set a player's 1-10 rating. The name can be partial; the tool "
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
                    "description": "Integer 1-10, or the string '?'.",
                },
            },
            "required": ["name", "rating"],
        },
    },
    {
        "name": "set_player_gender",
        "description": "Set a player's gender (M / F / ?). The name can be "
        "partial; the tool fuzzy-matches — ambiguous matches return an "
        "error with candidates so you can clarify with the admin. Use this "
        "when the gender-guesser misclassified someone on roster-add or "
        "the admin says e.g. 'Longjie Jia is male' / 'Sam is F' / 'reset "
        "Pat's gender to unknown'. Accepts M/F/? as well as "
        "male/female/man/woman/unknown (case-insensitive).",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Player name (full or partial). Case-insensitive.",
                },
                "gender": {
                    "type": "string",
                    "description": "M / F / ? (also accepts "
                    "male/female/man/woman/unknown).",
                },
            },
            "required": ["name", "gender"],
        },
    },
    {
        "name": "find_roster_duplicates",
        "description": "Scan the roster for likely-duplicate player rows "
        "(curly-vs-straight apostrophe variants like \"Luke O’Mahoney\" "
        "vs \"Luke O'Mahoney\"; nickname variants like Ben/Benjamin, "
        "Mike/Michael; whitespace/case differences). For each duplicate "
        "group the response marks which spelling CourtReserve is currently "
        "writing — that's the one to KEEP (otherwise the next CR scrape "
        "will silently re-create the duplicate). Read-only — does not "
        "delete anything. Use this when the admin asks 'are there any "
        "duplicates in the roster?' or you spot a duplicate while "
        "rendering attendees. Follow up with merge_and_delete_player to "
        "actually fix one.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "merge_and_delete_player",
        "description": "Resolve a duplicate-pair surfaced by "
        "find_roster_duplicates. Copies any populated fields (rating, "
        "gender, phone, singles preference) from the delete-side row "
        "into the keep-side row WHERE the keep-side is still default "
        "('?' / empty), then deletes the delete-side row. Never "
        "overwrites real data on the keep side. DESTRUCTIVE — first "
        "call with confirm=false (the default) to get a preview of "
        "exactly what would happen, then re-call with confirm=true to "
        "actually do it. Always show the preview to the admin and get "
        "their explicit go-ahead before passing confirm=true. Use the "
        "EXACT names from the roster (no fuzzy matching — typos here "
        "could delete the wrong person).",
        "input_schema": {
            "type": "object",
            "properties": {
                "keep_name": {
                    "type": "string",
                    "description": "Exact roster name to KEEP — should be "
                    "the spelling CourtReserve currently uses.",
                },
                "delete_name": {
                    "type": "string",
                    "description": "Exact roster name to DELETE (the "
                    "redundant duplicate).",
                },
                "confirm": {
                    "type": "boolean",
                    "default": False,
                    "description": "Set true ONLY after the admin has "
                    "seen and approved the preview. Without it the tool "
                    "returns a preview and refuses to delete.",
                },
            },
            "required": ["keep_name", "delete_name"],
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
        "two courts. Court labels stay put — only the players/pairs/mode "
        "move — so other pinned slots are unaffected. By default the "
        "swap applies across every rotation. Pass rotation_nums to "
        "scope to specific rotations only — e.g. for 'swap courts 1 "
        "and 5 for rotation 2 and 3' pass label_a='1', label_b='5', "
        "rotation_nums=[2, 3]. Use the no-rotation_nums form for "
        "blanket swaps like 'put singles on Ct 5' (swap the current "
        "singles court with court 5).",
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
                "rotation_nums": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional list of 1-based rotation "
                    "numbers to swap (e.g. [2, 3]). Omit / null = all "
                    "rotations.",
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
        "name": "undo_commit",
        "description": "Reverse the MOST RECENT commit_plan when the "
        "admin says 'undo the commit', 'I committed by mistake', "
        "'unfinalise', 'revert that commit', or wants to edit a plan "
        "they just finalised. Removes the just-added history.json "
        "entry, deletes the mirrored Sheet Session/Pair-log rows, "
        "restores the committed plan as the live draft and sets the "
        "phase back to draft_ready so swap_players / swap_rotations / "
        "swap_courts / regenerate work again. Only the latest commit, "
        "only before clear_tonight. Takes no arguments. Returns "
        "error='nothing_to_undo' if there's nothing to reverse.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "send_rules_pdf",
        "description": (
            "Send the current pairing-rules-and-weights PDF as an "
            "attachment to the calling WhatsApp channel, with a "
            "covering caption \"The pairing rules and weights are "
            "described in this PDF file.\". The PDF is regenerated "
            "from the live constants in pairings.py on every call, "
            "so it always reflects the current weights. Use this "
            "when the admin asks for the pairing rules, the "
            "weightings, the scoring rules, how the algorithm "
            "scores, or anything similar. No arguments. The model "
            "MUST emit an empty assistant reply after calling this "
            "tool (the attachment carries its own caption)."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "send_final_docx",
        "description": (
            "Post the FINAL pairings to this WhatsApp channel as TWO "
            "separate messages, in order: (1) the rendered FINAL "
            "pairings text passed in as `pairings_text`, then (2) the "
            "Thursday Social Tennis Word-doc attachment with caption "
            "\"The attached word doc contains these pairings in a "
            "format suitable for printing\". Call this immediately "
            "after commit_plan succeeds, AND when the admin asks for "
            "'final preview' / 'members preview' / 'show final "
            "preview'. The model MUST emit an empty assistant reply "
            "after calling this tool (the tool posts the text itself, "
            "so the model's own reply would duplicate it)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pairings_text": {
                    "type": "string",
                    "description": (
                        "The fully-rendered FINAL pairings block — "
                        "starts with 'Pairings for Thursday Dth Month.' "
                        "followed by the second-sentence reminder, a "
                        "blank line, and each rotation. No ratings, no "
                        "score footer. This is the text that will be "
                        "posted as the first message before the doc."
                    ),
                },
            },
            "required": ["pairings_text"],
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
        "name": "set_late_court",
        "description": (
            "Configure tonight's late-arriving extra court. Use when the admin "
            "tells you a court is only available from a later rotation onwards "
            "(typically R2 onwards because someone else has it until 8pm). "
            "The 4 named players are pinned to that court in its first "
            "available rotation; they sit out earlier rotations; from that "
            "rotation onwards the court is in the general pool. The court "
            "label is auto-added to tonight's court_labels if missing. "
            "Players are fuzzy-matched against tonight's attendees — partial "
            "names like 'Geoff' or 'Maggie' are fine."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Court label, e.g. '5' or 'C5'.",
                },
                "first_rotation": {
                    "type": "integer",
                    "description": "1-based rotation index from which the court "
                    "is available (so '2' means it joins from the 8:15 "
                    "rotation onwards if the evening starts at 19:30).",
                },
                "players": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exactly 4 player names (fuzzy-matched).",
                },
            },
            "required": ["label", "first_rotation", "players"],
        },
    },
    {
        "name": "clear_late_court",
        "description": "Remove any configured late-court pinning for tonight. "
        "Use if the admin changes their mind or the late court falls through.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "pin_doubles",
        "description": (
            "Pin a specific 4-player doubles match-up for tonight. Use "
            "when the admin says things like \"Alan and Penny to play "
            "Peter and Ben in rotation 2\" (rotation_num=2) or "
            "\"Alan and Penny vs Peter and Ben in one of the "
            "rotations\" (rotation_num omitted — optimiser picks). "
            "The phrasing names the two partnerships explicitly: "
            "'A and B to play C and D' means pairs=[['A','B'],"
            "['C','D']]. Pair structure is REQUIRED — never guess; if "
            "the admin only names 4 players without specifying who "
            "partners whom, ask first. Names are fuzzy-matched against "
            "tonight's attendees. court_label is optional; omit to let "
            "the optimiser pick a free doubles court in the chosen "
            "rotation. Multiple pin_doubles calls are allowed (one per "
            "match-up). The pinned court is NOT scored on its own "
            "(admin chose it) but still feeds the cross-rotation "
            "tallies — partner/opponent repeats with the pinned pairs "
            "elsewhere are still penalised, and the pinned rotation "
            "counts toward each player's whole-evening rules "
            "(standard_too_low, top_player_no_strong_rotation, "
            "unbalanced-count escalation). Returns the updated "
            "pinned_doubles list. If the admin wants to change a "
            "pin, call clear_pinned_doubles first and re-add."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "players": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Exactly four player names "
                    "(fuzzy-matched against tonight's attendees).",
                },
                "pairs": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "The two partnerships, e.g. "
                    "[['Alan Smith','Penny Jones'],"
                    "['Peter Brown','Ben Black']]. Must partition the "
                    "4 players exactly.",
                },
                "rotation_num": {
                    "type": "integer",
                    "description": "1-based rotation. Omit (or pass "
                    "null) to mean \"any rotation, optimiser picks\".",
                },
                "court_label": {
                    "type": "string",
                    "description": "Optional specific doubles court "
                    "label. Omit to let the optimiser pick.",
                },
            },
            "required": ["players", "pairs"],
        },
    },
    {
        "name": "clear_pinned_doubles",
        "description": (
            "Remove every pinned-doubles entry for tonight. Use when "
            "the admin says \"forget the pinned match-up\" / \"cancel "
            "the pin\" / wants to start the doubles pinning over."
        ),
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
TEST_CHANNEL_NAME = "Boris the tennis bot"


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
    client: Anthropic,
    user_text: str,
    group_name: str = "",
    history: list[dict] | None = None,
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
    # Prepend the recent Boris<->admin exchange (if any) so a terse
    # follow-up is understood in context. _recent_conversation
    # guarantees the history starts with a user turn and alternates;
    # appending the current command can produce two consecutive user
    # turns (history ended on a user turn that never got a reply), so
    # coalesce that boundary into one user turn.
    messages: list[dict] = list(history or [])
    current = f"Today is {today}.\nAdmin command: {user_text}"
    if messages and messages[-1]["role"] == "user":
        messages[-1] = {
            "role": "user",
            "content": messages[-1]["content"] + "\n\n" + current,
        }
    else:
        messages.append({"role": "user", "content": current})
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    for _ in range(AGENT_LOOP_MAX_TURNS):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            # Anthropic prompt caching: marking the system block with
            # cache_control caches the entire static prefix (system +
            # tools) for ~5 minutes, refreshed on each use. Cache reads
            # are billed at ~10% of normal rate AND are several times
            # faster on input processing — the dominant cost for a
            # large prompt + tool-schemas like ours. First call of a
            # session pays full freight; subsequent calls within ~5min
            # mostly hit the cache.
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
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
    _signal_output_delivered()
    return True


def send_doc_to_group(
    group_jid: str, file_path: str | Path, caption: str = "",
) -> bool:
    """Send a file (e.g. the final-pairings .docx) to ``group_jid``.

    The bridge reads ``file_path`` from disk so it must be an absolute
    path on the machine the bridge is running on (the Pi).
    """
    fp = str(Path(file_path).resolve())
    r = requests.post(
        f"{BRIDGE_URL}/send",
        json={
            "recipient": group_jid,
            "message": caption,
            "media_path": fp,
        },
        timeout=60,
    )
    if r.status_code != 200:
        print(f"send_doc failed: {r.status_code} {r.text}", file=sys.stderr)
        return False
    _signal_output_delivered()
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
    try:
        cutoff = (
            datetime.fromisoformat(before_ts)
            - timedelta(minutes=HISTORY_WINDOW_MINUTES)
        ).isoformat()
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


def _fire_scheduled_booking(entry) -> None:
    """Attempt one scheduled booking. Posts the result back to the
    channel that scheduled it, then records succeeded / failed in
    scheduled_bookings.json. Synchronous: blocks the poll loop for
    the duration of the CR call.

    Uses a pre-warm strategy when the entry's window_opens_at is
    still in the future (within IMMINENT_WINDOW_SECONDS): boot
    Chromium, log in, navigate to the booking form and fill the
    modal BEFORE the window opens, then click submit at T+0 so the
    booking lands within ~1s of the window going live. Falls back
    to a single-shot path when the window has already passed.
    """
    import time as _t
    from datetime import datetime as _dt
    import scheduled_bookings as sb
    from accounts import account_for_key, cr_client

    fire_started_at = _dt.now(sb.LOCAL_TZ)
    opens = sb.parse_iso(entry.window_opens_at)
    seconds_until_open = (
        (opens - fire_started_at).total_seconds() if opens else 0.0
    )
    pre_warm = opens is not None and seconds_until_open > 0.5

    print(
        f"[scheduler] firing scheduled booking #{entry.id} "
        f"at {fire_started_at.isoformat(timespec='milliseconds')} "
        f"({entry.scheduled_by_account_key} → court "
        f"{entry.court_label or '<any>'} on {entry.play_date} at "
        f"{entry.start_time_hhmm}, partner={entry.partner_name}, "
        f"attempt {entry.fire_attempts + 1}/{sb.MAX_FIRE_ATTEMPTS}, "
        f"window_opens_at={entry.window_opens_at}, "
        f"seconds_until_open={seconds_until_open:.2f}, "
        f"pre_warm={pre_warm})"
    )

    account = account_for_key(entry.scheduled_by_account_key)
    if account is None:
        sb.mark_attempt(
            entry.id, succeeded=False,
            error=f"account_unknown:{entry.scheduled_by_account_key}",
        )
        return

    # Honour the booking account's configured court order. court_label
    # / court_type still take precedence (handled in
    # _build_court_candidates); this only sets the fallback iteration.
    acct_pref = account.court_preference_list()
    try:
        with cr_client(account) as cr:
            if pre_warm:
                # Phase 1: prep — boot, log in, navigate, fill modal.
                # This typically takes 10-15s; we fire at T-30s so prep
                # finishes well before the window opens.
                t_prep_start = _t.perf_counter()
                prepared = cr.prepare_court_booking(
                    date=entry.play_date,
                    start_time_hhmm=entry.start_time_hhmm,
                    partner_name=entry.partner_name,
                    duration_minutes=entry.duration_minutes,
                    court_label=entry.court_label,
                    court_type=entry.court_type,
                    court_preference=acct_pref,
                )
                prep_secs = _t.perf_counter() - t_prep_start

                # Stale-grid recovery: CR doesn't publish post-window
                # slots until window_opens_at, so a pre-warm prep that
                # targets an afternoon slot will see only the pre-window
                # morning slots and bail with 'no_court_available'. If
                # we're still pre-window, wait for the window then
                # re-prep — that second navigation lands on a fresh
                # server response with the now-published slots.
                if (
                    isinstance(prepared, dict)
                    and prepared.get("status") == "no_court_available"
                    and opens is not None
                ):
                    seconds_left = (opens - _dt.now(sb.LOCAL_TZ)).total_seconds()
                    if seconds_left > -0.5:
                        wait = max(0.0, seconds_left + 0.3)
                        print(
                            f"[scheduler] booking #{entry.id} pre-warm "
                            f"saw stale grid (no slot for "
                            f"{entry.start_time_hhmm}); waiting "
                            f"{wait:.2f}s for window then re-prepping"
                        )
                        if wait > 0:
                            _t.sleep(wait)
                        t_prep_start = _t.perf_counter()
                        prepared = cr.prepare_court_booking(
                            date=entry.play_date,
                            start_time_hhmm=entry.start_time_hhmm,
                            partner_name=entry.partner_name,
                            duration_minutes=entry.duration_minutes,
                            court_label=entry.court_label,
                            court_type=entry.court_type,
                            court_preference=acct_pref,
                        )
                        prep_secs = _t.perf_counter() - t_prep_start

                if isinstance(prepared, dict):
                    # Prep failed (no court visible / partner not
                    # found). Surface as a single-shot result.
                    print(
                        f"[scheduler] booking #{entry.id} prep failed in "
                        f"{prep_secs:.2f}s: status={prepared.get('status')!r}"
                    )
                    result = prepared
                else:
                    # Phase 2: hold the modal until just before the
                    # window opens, then submit. Sleep until T-0.3s
                    # so the click lands within ~0.5s of T+0.
                    now = _dt.now(sb.LOCAL_TZ)
                    wait = (opens - now).total_seconds() - 0.3
                    print(
                        f"[scheduler] booking #{entry.id} prep complete in "
                        f"{prep_secs:.2f}s on court "
                        f"{prepared.prep['court_label']!r}; "
                        f"holding modal, sleeping {max(0, wait):.2f}s "
                        f"until window opens"
                    )
                    if wait > 0:
                        _t.sleep(wait)
                    print(
                        f"[scheduler] booking #{entry.id} submitting at "
                        f"{_dt.now(sb.LOCAL_TZ).isoformat(timespec='milliseconds')}"
                    )
                    result = prepared.submit()
                    print(
                        f"[scheduler] booking #{entry.id} submit returned: "
                        f"ok={result.get('ok')} status={result.get('status')!r} "
                        f"submit_attempts={result.get('submit_attempt') or result.get('submit_attempts')}"
                    )
            else:
                # Already past the window — single-shot legacy path.
                print(
                    f"[scheduler] booking #{entry.id} window already "
                    f"open ({-seconds_until_open:.1f}s ago); single-shot"
                )
                result = cr.book_court(
                    date=entry.play_date,
                    start_time_hhmm=entry.start_time_hhmm,
                    partner_name=entry.partner_name,
                    duration_minutes=entry.duration_minutes,
                    court_label=entry.court_label,
                    court_type=entry.court_type,
                    court_preference=acct_pref,
                )
    except Exception as e:
        import traceback as _tb
        print(
            f"[scheduler] booking #{entry.id} crashed: {e!r}\n"
            + _tb.format_exc(),
            file=sys.stderr,
        )
        result = {"ok": False, "status": "exception", "error": repr(e)}

    succeeded = bool(result.get("ok"))
    transient = (not succeeded) and result.get("status") in {
        "too_early", "no_court_available", "all_taken", "exception",
        "submit_rejected", "submit_court_taken",
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


def _maybe_remind_stale_run(now: datetime, fallback_jid: str = "") -> None:
    """If a run (dry or real) has been loaded but untouched for
    ``STALE_RUN_REMINDER_MINUTES``, post a single nudge asking whether
    to continue or clear it. Never clears anything — the admin must
    ask explicitly. Best-effort; never raises.

    Fires once per idle period: ``session_state.note_activity`` resets
    the gate whenever the admin interacts again, so a fresh hour of
    silence can trigger a new reminder.
    """
    import session_state as ss
    from accounts import account_for_phone

    try:
        state = ss.get_tonight()
    except Exception as e:
        print(f"[stale-run] load error: {e!r}", file=sys.stderr)
        return
    if not state.phase:
        return  # no run loaded
    if state.idle_reminder_sent:
        return  # already nudged for this idle period
    last = state.last_activity_at or state.started_at
    if not last:
        # Pre-existing session from before this feature — stamp now so
        # the timer starts from here rather than firing immediately.
        try:
            ss.note_activity()
        except Exception:
            pass
        return
    try:
        last_dt = datetime.fromisoformat(last)
    except (ValueError, TypeError):
        return
    now_aware = now if now.tzinfo else now.astimezone()
    if last_dt.tzinfo is None:
        last_dt = last_dt.astimezone()
    idle_minutes = (now_aware - last_dt).total_seconds() / 60.0
    if idle_minutes < STALE_RUN_REMINDER_MINUTES:
        return

    # Resolve a friendly name + start time for the message.
    who = "someone"
    if state.started_by:
        try:
            who = account_for_phone(state.started_by).display_name
        except Exception:
            who = "someone"
    elif state.started_at:
        who = "the scheduler"
    when = "?"
    if state.started_at:
        try:
            when = datetime.fromisoformat(
                state.started_at
            ).strftime("%H:%M")
        except (ValueError, TypeError):
            when = "?"
    run_kind = "dry run" if state.test_mode else "run"
    target_jid = state.channel_jid or fallback_jid
    if not target_jid:
        print(
            "[stale-run] no channel to remind in; skipping",
            file=sys.stderr,
        )
        return
    msg = (
        f"Reminder: there's a {run_kind} in progress, started by "
        f"{who} at {when}, with no activity for over "
        f"{int(idle_minutes)} min. Reply \"boris continue\" to keep "
        f"working on it, or \"boris clear this run\" to wipe it. "
        f"I won't clear it unless you ask."
    )
    try:
        send_to_group(target_jid, BOT_REPLY_PREFIX + msg)
        ss.mark_idle_reminder_sent()
        print(
            f"[stale-run] reminder posted (idle {int(idle_minutes)} min, "
            f"phase={state.phase!r}, test_mode={state.test_mode})"
        )
    except Exception as e:
        print(f"[stale-run] post failed: {e!r}", file=sys.stderr)


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
    print(
        "Kickoff: organiser-triggered only "
        '(say "boris kickoff" in an admin group; defaults to the next '
        "scheduled session by weekday)."
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
        # Scheduled court-booking check — fires any pending entries
        # whose 7-day-ahead window has opened (08:00 local). Each fire
        # is a 10-30 s blocking CR call; the watermark catches up on
        # the next iteration.
        _maybe_fire_scheduled_bookings(datetime.now())

        # Stale-run nudge — one reminder if a loaded run goes quiet for
        # an hour. Fallback channel = first admin group (used only when
        # the run has no recorded channel, e.g. auto-kickoff).
        _maybe_remind_stale_run(
            datetime.now(),
            fallback_jid=next(iter(group_jids.values()), ""),
        )

        for group_name, group_jid in group_jids.items():
            try:
                new_msgs = fetch_triggered_messages(group_jid, watermarks[group_jid])
            except Exception as e:
                print(f"[{group_name}] poll error: {e}", file=sys.stderr)
                continue

            for msg_id, ts, sender, command in new_msgs:
                watermarks[group_jid] = ts  # advance even on errors
                print(f"[{ts}] [{group_name}] {sender}: {command!r}")

                # Optional "Working on it" after a few s of quiet —
                # suppressed if the run finished OR a tool already
                # delivered output to the channel (e.g. the rules PDF),
                # which would otherwise make the placeholder arrive
                # out of sync, after the answer.
                done = threading.Event()
                _ACTIVE_DONE["ev"] = done

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
                # group (e.g. kickoff_session) read group_jid via .get().
                # Booking tools resolve the caller's CR account via
                # _CURRENT_SENDER → accounts.account_for_phone(...).
                jid_token = _CURRENT_GROUP_JID.set(group_jid)
                sender_token = _CURRENT_SENDER.set(sender)
                try:
                    if command:
                        history = _recent_conversation(group_jid, ts)
                        reply_body, usage = run_agent(
                            client, command, group_name=group_name,
                            history=history,
                        )
                    else:
                        reply_body = "(empty command — say e.g. 'boris help')"
                except Exception as e:
                    reply_body = f"(bot error: {e})"
                finally:
                    done.set()
                    timer.cancel()
                    _ACTIVE_DONE["ev"] = None
                    _CURRENT_GROUP_JID.reset(jid_token)
                    _CURRENT_SENDER.reset(sender_token)

                # Any command issued while a run is loaded counts as
                # activity — resets the stale-run idle timer and (first
                # time only) records who started it + where. No-op when
                # no run is in flight.
                try:
                    import session_state as _ss
                    _ss.note_activity(
                        started_by=sender or "", channel_jid=group_jid,
                    )
                except Exception as e:
                    print(
                        f"[stale-run] note_activity failed: {e!r}",
                        file=sys.stderr,
                    )

                # Some tools (notably kickoff_session) post their own
                # structured message into the channel and the bot's
                # follow-up reply is redundant. The bot is told to
                # output an empty turn in those cases — treat that as
                # "nothing more to say" and skip the WhatsApp send.
                if reply_body.strip():
                    reply = BOT_REPLY_PREFIX + reply_body
                    print(f"  -> reply: {reply[:200]}{'…' if len(reply) > 200 else ''}")
                    send_to_group(group_jid, reply)
                else:
                    print("  -> (no reply — tool posted its own message)")

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
