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
from datetime import datetime
from pathlib import Path

SESSION_STATE_PATH = Path(__file__).parent / "session_state.json"


def _now_iso() -> str:
    """Local-aware ISO timestamp (matches the rest of the codebase's
    local-time convention; tz-aware so deltas are unambiguous)."""
    return datetime.now().astimezone().isoformat()


@dataclass
class SessionState:
    date: str = ""                   # ISO date the session is for
    source: str = ""                 # e.g. "courtreserve:V8WE4BB2146425" or "manual"
    # Which weekly session this run is for. One of the keys in
    # session_types.SESSION_TYPES ("tuesday" / "thursday" / "saturday"),
    # or "" if the kickoff didn't set it (legacy / manual starts).
    # Drives default start time, rotation durations, kickoff message
    # styling, and which DOCX template the final doc uses.
    session_type: str = ""
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
    # there's no in-flight session. The kickoff sets this to
    # "awaiting_extras"; subsequent admin actions transition through
    # "ready_to_generate" → "draft_ready". commit_plan clears the
    # session back to "" so there's no separate "finalised" state.
    phase: str = ""
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
    # Admin-pinned doubles match-ups for the evening. Each entry pins
    # a specific 4-player + 2-pair structure to a rotation (or to "any
    # rotation" when ``rotation_num`` is ``None`` — the optimiser picks
    # which one when generating). The pinned court contributes 0 to its
    # own per-court score (the admin asked for it; we don't second-guess
    # their pair choice) but still feeds the cross-rotation tallies
    # (partner/opponent repeats, unbalanced-count escalation, the
    # whole-evening per-player rules). Each pin is a dict::
    #
    #     {"rotation_num": int | None,
    #      "players": [str, str, str, str],     # 4 distinct names
    #      "pairs": [[str, str], [str, str]],   # partnership split
    #      "court_label": str | None}           # specific court (optional)
    pinned_doubles: list[dict] = field(default_factory=list)
    # Stale-run reminder bookkeeping. ``started_by`` is the WhatsApp
    # sender id of whoever kicked the run off; ``channel_jid`` is where
    # the kickoff was issued (so a reminder posts back there).
    # ``last_activity_at`` is bumped on every bot command while a run
    # is loaded; ``idle_reminder_sent`` gates the reminder to once per
    # idle period (reset whenever activity resumes).
    started_by: str = ""
    started_at: str = ""
    last_activity_at: str = ""
    channel_jid: str = ""
    idle_reminder_sent: bool = False
    # The most recent commit_plan, kept so an inadvertent commit can
    # be undone (history entry removed, Sheet rows deleted, draft
    # restored). Shape: {"plan": <plan dict>, "date": str,
    # "sheet_session_rows": int, "sheet_pair_rows": int,
    # "committed_at": iso}. Cleared by a successful undo or by
    # clear_tonight (fresh SessionState).
    last_commit: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


VALID_PHASES = {
    "",                  # no in-flight session (also the post-commit state)
    "awaiting_extras",   # kickoff posted, waiting for admin to supply extras
    "ready_to_generate", # admin said "go ahead"; not yet generated
    "draft_ready",       # draft_plan persisted; awaiting tweaks or commit
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
        session_type=str(raw.get("session_type", "") or ""),
        attendees=list(raw.get("attendees") or []),
        court_labels=[str(x) for x in (raw.get("court_labels") or [])],
        waitlist=list(raw.get("waitlist") or []),
        notes=raw.get("notes", ""),
        draft_plan=raw.get("draft_plan") or None,
        phase=phase,
        late_court_label=str(raw.get("late_court_label", "") or ""),
        late_court_first_rotation=int(raw.get("late_court_first_rotation", 0) or 0),
        late_court_pinned_players=list(raw.get("late_court_pinned_players") or []),
        pinned_doubles=[
            dict(p) for p in (raw.get("pinned_doubles") or [])
            if isinstance(p, dict)
        ],
        started_by=str(raw.get("started_by", "") or ""),
        started_at=str(raw.get("started_at", "") or ""),
        last_activity_at=str(raw.get("last_activity_at", "") or ""),
        channel_jid=str(raw.get("channel_jid", "") or ""),
        idle_reminder_sent=bool(raw.get("idle_reminder_sent", False)),
        last_commit=(
            raw.get("last_commit")
            if isinstance(raw.get("last_commit"), dict)
            else None
        ),
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
    session_type: str = "",
    court_labels: list | None = None,
    waitlist: list[str] | None = None,
    notes: str = "",
) -> SessionState:
    """Begin (or replace) tonight's session.

    Attendees come from CourtReserve (or a manual list). ``waitlist`` is
    populated when the CR event is full; admin then decides which (if any)
    to promote into ``attendees`` after considering extra courts. Court
    labels are usually set in a follow-up message. ``session_type`` is
    one of ``"tuesday" / "thursday" / "saturday"`` (or ``""`` if the
    caller doesn't know — back-compat with manual starts).
    """
    now = _now_iso()
    state = SessionState(
        date=date,
        source=source,
        session_type=str(session_type or ""),
        attendees=list(attendees),
        court_labels=[str(x) for x in (court_labels or [])],
        waitlist=list(waitlist or []),
        notes=notes,
        started_at=now,
        last_activity_at=now,
    )
    _save(state)
    return state


def note_activity(
    *, started_by: str = "", channel_jid: str = ""
) -> SessionState:
    """Record that the admin just interacted with the bot while a run
    is loaded. No-op when there is no in-flight session (``phase``
    empty) so unrelated commands don't create churn.

    Bumps ``last_activity_at`` and clears ``idle_reminder_sent`` (so a
    fresh idle period can trigger a new reminder). ``started_by`` /
    ``channel_jid`` are recorded only the first time (they identify who
    kicked the run off and where), never overwritten on later activity.
    """
    state = _load()
    if not state.phase:
        return state  # no run in flight — nothing to track
    state.last_activity_at = _now_iso()
    state.idle_reminder_sent = False
    if started_by and not state.started_by:
        state.started_by = started_by
    if channel_jid and not state.channel_jid:
        state.channel_jid = channel_jid
    _save(state)
    return state


def mark_idle_reminder_sent() -> SessionState:
    """Flag that the stale-run reminder has been posted for the current
    idle period (so it isn't repeated until activity resumes)."""
    state = _load()
    state.idle_reminder_sent = True
    _save(state)
    return state


def record_commit(
    plan: dict,
    *,
    sheet_session_rows: int = 0,
    sheet_pair_rows: int = 0,
) -> SessionState:
    """Remember the just-committed plan so it can be undone."""
    from datetime import datetime

    state = _load()
    state.last_commit = {
        "plan": plan,
        "date": plan.get("date", ""),
        "sheet_session_rows": int(sheet_session_rows),
        "sheet_pair_rows": int(sheet_pair_rows),
        "committed_at": datetime.now().astimezone().isoformat(),
    }
    _save(state)
    return state


def get_last_commit() -> dict | None:
    return _load().last_commit


def clear_last_commit() -> SessionState:
    state = _load()
    state.last_commit = None
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


def add_pinned_doubles(
    players: list[str],
    pairs: list[list[str]],
    *,
    rotation_num: int | None = None,
    court_label: str | None = None,
) -> SessionState:
    """Append a pinned-doubles entry for tonight.

    ``players`` must be 4 distinct names already in ``attendees``.
    ``pairs`` is the partnership split — two lists of two names whose
    union equals ``players``. ``rotation_num`` is 1-based; ``None``
    means "the optimiser picks which rotation". ``court_label`` is
    optional; when omitted any available doubles court is fine.
    Raises ``ValueError`` for bad input.
    """
    if len(players) != 4:
        raise ValueError(
            f"pinned_doubles needs exactly 4 players (got {len(players)}: "
            f"{players!r})"
        )
    if len(set(players)) != 4:
        raise ValueError(
            f"pinned_doubles players must be distinct (got {players!r})"
        )
    if (
        not isinstance(pairs, list)
        or len(pairs) != 2
        or any(not isinstance(p, list) or len(p) != 2 for p in pairs)
    ):
        raise ValueError(
            f"pinned_doubles.pairs must be 2 lists of 2 names (got {pairs!r})"
        )
    flat = [p for pair in pairs for p in pair]
    if sorted(flat) != sorted(players):
        raise ValueError(
            "pinned_doubles.pairs must partition players exactly "
            f"(players={players!r}, pairs={pairs!r})"
        )
    if rotation_num is not None and rotation_num < 1:
        raise ValueError(
            f"pinned_doubles.rotation_num must be >= 1 or None "
            f"(got {rotation_num!r})"
        )

    state = _load()
    missing = [p for p in players if p not in state.attendees]
    if missing:
        raise ValueError(
            f"pinned_doubles: player(s) not in tonight's attendees: "
            f"{missing!r}"
        )
    if court_label is not None:
        seen = {str(x).strip() for x in state.court_labels}
        if str(court_label) not in seen:
            raise ValueError(
                f"pinned_doubles.court_label {court_label!r} not in "
                f"court_labels {state.court_labels!r}"
            )
    # Reject overlap with any existing pinned_doubles in the same
    # rotation (or globally when rotation_num is None — pinning the
    # same person to two free-rotation entries would force them to
    # play two matches in the same rotation by elimination).
    existing_players_same_rot: set[str] = set()
    for pin in state.pinned_doubles:
        if (
            rotation_num is None
            or pin.get("rotation_num") is None
            or pin.get("rotation_num") == rotation_num
        ):
            for p in pin.get("players") or []:
                existing_players_same_rot.add(p)
    clash = [p for p in players if p in existing_players_same_rot]
    if clash:
        raise ValueError(
            f"pinned_doubles: player(s) already pinned in an overlapping "
            f"rotation: {clash!r}"
        )

    state.pinned_doubles.append({
        "rotation_num": rotation_num,
        "players": list(players),
        "pairs": [list(p) for p in pairs],
        "court_label": str(court_label) if court_label is not None else None,
    })
    _save(state)
    return state


def clear_pinned_doubles() -> SessionState:
    """Remove every pinned-doubles entry from tonight's session."""
    state = _load()
    state.pinned_doubles = []
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
