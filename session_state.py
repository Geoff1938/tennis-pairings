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
    # Late-arriving extra court: a court that's only available from
    # rotation ``late_court_first_rotation`` onwards (because it's booked
    # by someone else earlier in the evening). The four named players
    # are pinned to that court in its first available rotation — Boris
    # picks the best 2-pair split among them. In earlier rotations
    # those four sit out (and the court is excluded from the available
    # set). In later rotations the court is fully in the pool. Set via
    # set_late_court(); cleared by clear_late_court() or clear_tonight.
    # ``late_court_first_rotation`` of 0 means "no late court configured".
    late_court_label: str = ""
    late_court_first_rotation: int = 0
    late_court_pinned_players: list[str] = field(default_factory=list)

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
        late_court_label=str(raw.get("late_court_label", "") or ""),
        late_court_first_rotation=int(raw.get("late_court_first_rotation", 0) or 0),
        late_court_pinned_players=list(raw.get("late_court_pinned_players") or []),
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


def set_late_court(
    label: str,
    first_rotation: int,
    players: list[str],
) -> SessionState:
    """Configure the evening's late-arriving extra court.

    Adds ``label`` to ``court_labels`` if not already there. Stores the
    four pinned players (exact-name match required — caller resolves
    fuzzy names against the roster/attendees BEFORE invoking this).
    ``first_rotation`` is 1-based (so for "available from R2 onwards",
    pass 2). Raises ``ValueError`` for bad inputs.
    """
    label = (label or "").strip()
    if not label:
        raise ValueError("late_court_label must not be empty")
    if first_rotation < 1:
        raise ValueError(
            f"late_court_first_rotation must be >= 1 (got {first_rotation})"
        )
    if len(players) != 4:
        raise ValueError(
            f"late court needs exactly 4 pinned players "
            f"(got {len(players)}: {players!r})"
        )
    if len(set(players)) != 4:
        raise ValueError(f"late court players must be distinct (got {players!r})")

    state = _load()
    seen_court_labels = {str(x).strip() for x in state.court_labels}
    if label not in seen_court_labels:
        state.court_labels.append(label)
    for p in players:
        if p not in state.attendees:
            raise ValueError(
                f"late court pin: player {p!r} is not in tonight's attendees"
            )
    state.late_court_label = label
    state.late_court_first_rotation = int(first_rotation)
    state.late_court_pinned_players = list(players)
    _save(state)
    return state


def clear_late_court() -> SessionState:
    """Wipe any configured late court (does NOT remove the label from
    ``court_labels`` — admin may still want the court generally
    available). Just clears the pinning."""
    state = _load()
    state.late_court_label = ""
    state.late_court_first_rotation = 0
    state.late_court_pinned_players = []
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
