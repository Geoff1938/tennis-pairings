"""Registry of the weekly tennis-session kinds Boris organises.

There are four:
  * ``tuesday``       — Tuesday Evening Club Social for Intermediate+, 19:30-21:30
  * ``thursday``      — Thursday Evening Club Social for Intermediate+, 19:30-21:30
  * ``thursday_1829`` — Thursday 18-29 Social Doubles Night, 19:30-21:30
  * ``saturday``      — Saturday Social Doubles, 14:00-16:00

Two of those land on the same weekday (Thursday). The ``variant``
field groups them: "regular" (Tue / Thu / Sat) vs "18-29" (the new
Thursday youth event). ``resolve_next_session(today, variant=...)``
filters by variant so ``boris kickoff regular`` and ``boris kickoff
18-29`` each pick the right next event regardless of what day the
command is typed on.

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
    # How many "preamble" paragraphs the template opens with that the
    # renderer should leave untouched (the next paragraph is rewritten
    # as the "Pairings for …" date heading). All three current session
    # types use the Westside template, which has no preamble — kept
    # parameterised so a future bespoke template (signup blurb, QR
    # code, sponsorship banner, etc.) can opt into the preserve-N path.
    docx_preamble_paragraph_count: int = 0
    # Text shown in the page-header banner across the top of every
    # page. The Westside template (shared Tue + Sat) stores
    # "Thursday Social Tennis" verbatim because it was derived from
    # the Thursday template — the renderer rewrites this on each run
    # so Tue/Sat docs don't silently say the wrong day.
    docx_header_text: str = ""
    # Disambiguation group for ``resolve_next_session`` and the
    # ``boris kickoff <variant>`` command. "regular" covers the three
    # historical weekday sessions (Tue / Thu / Sat); "18-29" is the
    # new Thursday-evening youth session. Two SessionTypes with the
    # SAME weekday must have DIFFERENT variants — that's the whole
    # reason this field exists.
    variant: str = "regular"


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
        docx_preamble_paragraph_count=0,
        docx_header_text="Tuesday Social Tennis",
    ),
    "thursday": SessionType(
        key="thursday",
        weekday=3,
        start_time_hhmm="19:30",
        rotation_durations=[45, 40, 35],
        display_name="Thursday Evening Club Social for Intermediate+",
        # Tightened from the historic "social" — that used to be
        # uniquely identifying when only one Thursday event existed,
        # but the 18-29 event below also contains "social" in its
        # title. The full event title is the safest non-overlapping
        # substring.
        cr_name_contains="Thursday Evening Club Social",
        cr_time_fragment="19:30 - 21:30",
        admin_group_name="Thursday Tennis Organisers",
        docx_template_relpath="tmp/Westside Social Tennis.docx",
        docx_preamble_paragraph_count=0,
        docx_header_text="Thursday Social Tennis",
        variant="regular",
    ),
    "thursday_1829": SessionType(
        key="thursday_1829",
        weekday=3,
        start_time_hhmm="19:30",
        rotation_durations=[45, 40, 35],
        display_name="Thursday 18-29 Social Doubles Night",
        # CR title uses an en-dash (U+2013), not a hyphen. The full
        # event name is the safest filter — guaranteed unique.
        cr_name_contains="18–29 Social Doubles Night",
        cr_time_fragment="19:30 - 21:30",
        admin_group_name="Thursday Tennis Organisers",
        docx_template_relpath="tmp/Westside Social Tennis.docx",
        docx_preamble_paragraph_count=0,
        docx_header_text="18-29 Social Doubles Night",
        variant="18-29",
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
        docx_preamble_paragraph_count=0,
        docx_header_text="Saturday Social Tennis",
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


def resolve_next_session(
    today: date | None = None,
    variant: str | None = None,
) -> SessionType:
    """Return the SessionType whose weekday falls on/after today, wrapping.

    Used when an admin types ``boris kickoff`` with no day specified —
    they almost always mean "the next scheduled session". Mon/Tue →
    Tuesday, Wed/Thu → Thursday, Fri/Sat → Saturday, Sun → Tuesday.

    When ``variant`` is supplied, only SessionTypes with that variant
    are considered. So ``resolve_next_session(variant="18-29")`` on a
    Tuesday returns the Thursday 18-29 session (next future Thursday),
    not the Tuesday regular session. Raises ``LookupError`` if no
    session matches the variant filter (typo / unknown variant).
    """
    ref = today or date.today()
    today_wd = ref.weekday()
    candidates = list(SESSION_TYPES.values())
    if variant is not None:
        candidates = [st for st in candidates if st.variant == variant]
        if not candidates:
            raise LookupError(
                f"no SessionType matches variant={variant!r}; "
                f"available: {sorted({st.variant for st in SESSION_TYPES.values()})}"
            )
    # Sort by smallest forward offset (with wrap-around). When two
    # candidates share a weekday (the regular-vs-18-29 case), the
    # variant filter has already narrowed to one, so ties don't
    # arise here.
    by_offset = sorted(
        candidates,
        key=lambda st: (st.weekday - today_wd) % 7,
    )
    return by_offset[0]
