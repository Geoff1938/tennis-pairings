"""Current-session state: attendees + court labels for the evening about to be played.

Boris maintains this between WhatsApp commands. Typical flow:

    boris get tonight's attendees           # calls start_tonight(...)
    boris tonight we have 3 courts: 4,5,6   # set_courts_for_tonight([4,5,6])
    boris remove Fred C                     # remove_from_tonight("Fred C")
    boris add Joe Graham                    # add_to_tonight("Joe Graham")
    boris generate pairings                 # reads from session state

Single-session model — there's at most one session in flight. Storage is a
single JSON file at the project root (``session_state.json``). If a second
session is started, the previous one is overwritten; use ``get_tonight()``
first if in doubt.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SESSION_STATE_PATH = Path(__file__).parent / "session_state.json"


@dataclass
class SessionState:
    date: str = ""                   # ISO date the session is for
    source: str = ""                 # e.g. "courtreserve:V8WE4BB2146425" or "manual"
    attendees: list[str] = field(default_factory=list)
    court_labels: list[str] = field(default_factory=list)
    # CourtReserve waitlist captured at start_tonight time, in priority order.
    # Names that the admin promotes to playing should be moved into
    # `attendees` (typically via add_to_tonight or promote_from_waitlist).
    waitlist: list[str] = field(default_factory=list)
    notes: str = ""
    # Most recent plan returned by generate_pairings, kept here so admins
    # can iterate (swap players / rotations) before committing. Cleared by
    # commit_plan once written to history.json + the Sheet log tabs.
    draft_plan: dict | None = None
    # Workflow phase the bot/admin are in for this session. Empty when
    # there's no in-flight session. The Thursday kickoff sets this to
    # "awaiting_extras"; subsequent admin actions transition through
    # "ready_to_generate" → "draft_ready" → "finalised".
    phase: str = ""
    # When True, commit_plan and log_pairings_to_sheet refuse to run —
    # the admin is doing a dry run. Set by kickoff_thursday(test_mode=True)
    # and cleared by clear_tonight. Rating writes etc. are unaffected.
    test_mode: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


VALID_PHASES = {
    "",                  # no in-flight session
    "awaiting_extras",   # kickoff posted, waiting for admin to supply extras
    "ready_to_generate", # admin said "go ahead"; not yet generated
    "draft_ready",       # draft_plan persisted; awaiting tweaks or confirm
    "finalised",         # commit_plan succeeded; bot can render final
}


def _load() -> SessionState:
    if not SESSION_STATE_PATH.exists():
        return SessionState()
    with SESSION_STATE_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)
    phase = str(raw.get("phase", "") or "")
    if phase not in VALID_PHASES:
        phase = ""
    return SessionState(
        date=raw.get("date", ""),
        source=raw.get("source", ""),
        attendees=list(raw.get("attendees") or []),
        court_labels=[str(x) for x in (raw.get("court_labels") or [])],
        waitlist=list(raw.get("waitlist") or []),
        notes=raw.get("notes", ""),
        draft_plan=raw.get("draft_plan") or None,
        phase=phase,
        test_mode=bool(raw.get("test_mode", False)),
    )


def _save(state: SessionState) -> None:
    SESSION_STATE_PATH.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------- public API ---------------------------------------------------


def get_tonight() -> SessionState:
    """Return the current session state (may be empty)."""
    return _load()


def clear_tonight() -> SessionState:
    """Reset the session state to empty."""
    state = SessionState()
    _save(state)
    return state


def start_tonight(
    attendees: list[str],
    *,
    date: str = "",
    source: str = "",
    court_labels: list | None = None,
    waitlist: list[str] | None = None,
    notes: str = "",
    test_mode: bool = False,
) -> SessionState:
    """Begin (or replace) tonight's session.

    Attendees come from CourtReserve (or a manual list). ``waitlist`` is
    populated when the CR event is full; admin then decides which (if any)
    to promote into ``attendees`` after considering extra courts. Court
    labels are usually set in a follow-up message.
    """
    state = SessionState(
        date=date,
        source=source,
        attendees=list(attendees),
        court_labels=[str(x) for x in (court_labels or [])],
        waitlist=list(waitlist or []),
        notes=notes,
        test_mode=test_mode,
    )
    _save(state)
    return state


def promote_from_waitlist(name: str) -> tuple[SessionState, str | None]:
    """Move a name from the waitlist to attendees. Fuzzy match.

    Returns ``(state, promoted_name)``. ``promoted_name`` is ``None`` if no
    waitlist match was found.
    """
    state = _load()
    q = name.strip().lower()
    if not q:
        return state, None
    matches = [w for w in state.waitlist if q in w.lower()]
    if len(matches) != 1:
        # Caller surfaces ambiguous / not-found errors.
        return state, None
    promoted = matches[0]
    state.waitlist.remove(promoted)
    if promoted not in state.attendees:
        state.attendees.append(promoted)
    _save(state)
    return state, promoted


def add_to_tonight(name: str) -> SessionState:
    """Append ``name`` to tonight's attendee list (no-op if already present)."""
    state = _load()
    if name in state.attendees:
        return state
    state.attendees.append(name)
    _save(state)
    return state


def remove_from_tonight(name: str) -> tuple[SessionState, str | None]:
    """Remove a name (exact match) from tonight's attendees.

    Returns ``(state, removed_name)`` where ``removed_name`` is ``None`` if no
    match was found. Caller is responsible for fuzzy matching before calling.
    """
    state = _load()
    if name not in state.attendees:
        return state, None
    state.attendees.remove(name)
    _save(state)
    return state, name


def set_courts_for_tonight(court_labels: list) -> SessionState:
    """Overwrite tonight's court labels."""
    state = _load()
    state.court_labels = [str(x) for x in court_labels]
    _save(state)
    return state


def set_draft_plan(plan: dict) -> SessionState:
    """Stash a freshly-generated plan dict as the current draft."""
    state = _load()
    state.draft_plan = plan
    _save(state)
    return state


def get_draft_plan() -> dict | None:
    return _load().draft_plan


def clear_draft_plan() -> SessionState:
    state = _load()
    state.draft_plan = None
    _save(state)
    return state


def set_phase(phase: str) -> SessionState:
    """Set the workflow phase. Raises ``ValueError`` for unknown phases."""
    if phase not in VALID_PHASES:
        raise ValueError(
            f"unknown phase {phase!r} — valid: {sorted(VALID_PHASES)}"
        )
    state = _load()
    state.phase = phase
    _save(state)
    return state


def get_phase() -> str:
    return _load().phase


def find_attendee_fuzzy(query: str) -> list[str]:
    """Return attendee names containing ``query`` case-insensitively."""
    state = _load()
    q = query.strip().lower()
    if not q:
        return []
    return [a for a in state.attendees if q in a.lower()]
