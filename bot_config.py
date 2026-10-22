"""Shared configuration for Boris: paths, group IDs, model
names, limits, and the per-command context (who sent it,
which group).
"""

from __future__ import annotations

import contextvars
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY (and anything else) from .env in the
# project root before anything reads the environment.
load_dotenv(Path(__file__).parent / ".env")


ROOT = Path(__file__).parent
HISTORY_PATH = ROOT / "history.json"
BRIDGE_DB = ROOT / "whatsapp-mcp" / "whatsapp-bridge" / "store" / "messages.db"
BRIDGE_URL = "http://localhost:8080/api"
FINAL_DOCX_OUTPUT_DIR = ROOT / "output_files"
RULES_PDF_PATH = ROOT / "output_files" / "Pairing rules and weights.pdf"
RULES_PDF_CAPTION = "The pairing rules and weights are described in this PDF file."

ADMIN_GROUP_NAMES = [
    "Thursday Tennis Organisers",
    "Westside organisers of social tennis",
    "Boris the tennis bot",
]
TENNIS_GROUP_JID = "120363408685115680@g.us"  # Thursday Social Tennis Evening

POLL_INTERVAL_SECONDS = 0.3
# Parallel best-of-N for generate_pairings. Each worker runs a full
# independent make_plan with a different starting seed; the lowest-
# scoring plan wins. Wall time is roughly that of ONE make_plan call
# (workers run on separate cores). On the Pi 5 we have 4 cores, so 4
# workers max them out without contention. This replaces the older
# sequential best-of-2-with-threshold retry — strictly more
# exploration in roughly the same time.
GENERATE_PARALLEL_WORKERS = 4
# Matches either "boris" or "bot" as a leading word (case-insensitive),
# followed by any combination of whitespace and simple punctuation that a
# human might type before their actual question: `:`, `?`, `!`, `,`, `.`.
# "bottle", "both", "borised" etc. won't trigger (word-boundary required).
BOT_TRIGGER_PATTERN = re.compile(r"^\s*(?:boris|bot)\b[\s:?!,.]*", re.IGNORECASE)
BOT_REPLY_PREFIX = "From Boris the tennis bot: "
# Marker text the daily end-to-end canary posts. The canary sends this
# via the bridge to confirm the full round trip (bot → bridge →
# WhatsApp → bridge SQLite). The bot must NOT LLM-process it: that
# would burn an API call, spam the group with a reply, and (worse)
# the LLM might misinterpret "test msg" and do something. Skipped at
# the message-fetch layer so it never reaches the agent loop.
CANARY_MARKER = "Boris daily test msg"
WORKING_ON_IT_DELAY_SECONDS = 5.0
WORKING_ON_IT_TEXT = "Request received. Working on it…"
# Two-model setup. The classifier in _classify_command decides
# per-command which one to use. Sonnet handles complex commands
# (pairings algorithm, draft renders, multi-turn / workflow); Haiku
# handles simple one-shots (add/remove player, rate, lookup, help).
# Haiku is ~3× cheaper on input, ~3× cheaper on output, so routing
# even 25% of traffic to it materially cuts the bill — the
# classifier itself costs basically nothing (small fixed system
# prompt, single short answer).
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_HAIKU = "claude-haiku-4-5-20251001"
# Room for a long answer (a full score breakdown, a list of bookings)
# without being cut off; most replies are far shorter. Kept well under
# the size that would need streaming.
MAX_TOKENS = 8192
# Model calls per command. A multi-step change (e.g. remove a player,
# re-pin, re-roll, show the draft) needs several tool rounds.
AGENT_LOOP_MAX_TURNS = 10
# Multi-turn memory: how far back to reconstruct the Boris<->admin
# exchange so a follow-up ("Friday 22nd May, court 5") is understood
# as answering Boris's own prior question rather than a fresh request.
# Bounded by recency AND count to keep token cost down and stop a
# stale, unrelated thread from bleeding into a new command.
HISTORY_WINDOW_MINUTES = 15
HISTORY_MAX_MESSAGES = 6
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


DOCX_ATTACHMENT_CAPTION = (
    "The attached word doc contains these pairings in a format "
    "suitable for printing"
)



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
TEST_CHANNEL_NAME = "Boris the tennis bot"  # display only; see BOOKING_GROUP_JID
# The booking channel is identified by its WhatsApp JID, not its name:
# any member can rename a group, so a name check could be satisfied by
# renaming another admin group to "Boris the tennis bot".
BOOKING_GROUP_JID = "120363408518957244@g.us"
