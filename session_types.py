"""Registry of the weekly tennis-session kinds Boris organises.

There are three:
  * ``tuesday``  — Tuesday Evening Club Social for Intermediate+, 19:30-21:30
  * ``thursday`` — Thursday Social Tennis Evening, 19:30-21:30
  * ``saturday`` — Saturday Social Doubles, 14:00-16:00

Every entry carries the bits the kickoff + pairings pipeline needs:
the weekday (0=Mon … 6=Sun), the start time and rotation durations
the pairing algorithm should use, the CourtReserve event-name fragment
to filter on, the WhatsApp admin-group name that should receive the
kickoff post, the DOCX template the final document is rendered from,
and a human-readable display string.

This module is intentionally side-effect-free — it just defines
the registry and a few small helpers. Callers consume it via
``SESSION_TYPES["thursday"]`` or via ``resolve_next_session(today)``
which picks the upcoming session by weekday.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta


@dataclass(frozen=True)
class SessionType:
    """One weekly session kind. See module docstring for what each field is for."""

    key: str
    weekday: int                # Mon=0 .. Sun=6
    start_time_hhmm: str        # e.g. "19:30" / "14:00"
    rotation_durations: list[int]
    display_name: str           # human label, e.g. "Thursday Social Tennis Evening"
    # CourtReserve filters. ``cr_name_contains`` is the case-insensitive
    # substring used to spot the right event in the listing; the legacy
    # Thursday code matched on a deliberately broad ``"social"`` so
    # we keep that for Thursday and use the precise titles for the
    # other two days.
    cr_name_contains: str
    # Optional: a CR date-string fragment to disambiguate when two
    # events on the same day might both contain ``cr_name_contains``
    # (e.g. a morning and evening "social" — Thursday's evening
    # listing carries ``"19:30 - 21:30"``).
    cr_time_fragment: str = ""
    # WhatsApp group the kickoff message lands in.
    admin_group_name: str = ""
    # DOCX template (relative to repo root). The final-doc generator
    # substitutes the title text per session type.
    docx_template_relpath: str = ""


# The registry. Order matters only when displayed (and for tests); the
# scheduler picks by weekday.
SESSION_TYPES: dict[str, SessionType] = {
    "tuesday": SessionType(
        key="tuesday",
        weekday=1,
        start_time_hhmm="19:30",
        rotation_durations=[45, 40, 35],
        display_name="Tuesday Evening Club Social for Intermediate+",
        cr_name_contains="Tuesday Evening Club Social",
        cr_time_fragment="19:30 - 21:30",
        admin_group_name="Westside organisers of social tennis",
        docx_template_relpath="tmp/Westside Social Tennis.docx",
    ),
    "thursday": SessionType(
        key="thursday",
        weekday=3,
        start_time_hhmm="19:30",
        rotation_durations=[45, 40, 35],
        display_name="Thursday Social Tennis Evening",
        cr_name_contains="social",
        cr_time_fragment="19:30 - 21:30",
        admin_group_name="Thursday Tennis Organisers",
        docx_template_relpath="tmp/Thursday Social Tennis.docx",
    ),
    "saturday": SessionType(
        key="saturday",
        weekday=5,
        start_time_hhmm="14:00",
        rotation_durations=[45, 40, 35],
        display_name="Saturday Social Doubles",
        cr_name_contains="Saturday Social Doubles",
        cr_time_fragment="",  # CR may not list times in the date string for Sat
        admin_group_name="Westside organisers of social tennis",
        docx_template_relpath="tmp/Westside Social Tennis.docx",
    ),
}


def get(session_key: str) -> SessionType:
    """Look up a session type by key; raise KeyError with a helpful message."""
    try:
        return SESSION_TYPES[session_key]
    except KeyError:
        raise KeyError(
            f"unknown session_type {session_key!r}; valid keys: "
            f"{sorted(SESSION_TYPES)}"
        )


def resolve_next_session(today: date | None = None) -> SessionType:
    """Return the SessionType whose weekday falls on/after today, wrapping.

    Used when an admin types ``boris kickoff`` with no day specified —
    they almost always mean "the next scheduled session". Mon/Tue →
    Tuesday, Wed/Thu → Thursday, Fri/Sat → Saturday, Sun → Tuesday.
    """
    ref = today or date.today()
    today_wd = ref.weekday()
    # Sort by smallest forward offset (with wrap-around).
    by_offset = sorted(
        SESSION_TYPES.values(),
        key=lambda st: (st.weekday - today_wd) % 7,
    )
    return by_offset[0]
