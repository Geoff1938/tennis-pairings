"""Boris tools for tonight's run: kickoff, attendees,
waitlist, courts, late court, coaching group, pins.
"""

from __future__ import annotations

from typing import Optional

from roster import Roster
from bot_config import _CURRENT_GROUP_JID


_VARIANT_ALIASES: dict[str, str] = {
    # Regular (the historic Tue/Thu/Sat sessions).
    "regular": "regular",
    "reg": "regular",
    "intermediate+": "regular",
    "intermediate plus": "regular",
    "intermediate": "regular",
    "int+": "regular",
    # 18-29 youth session (Thursday evening; CR title uses an en-dash).
    "18-29": "18-29",
    "18-30": "18-29",
    "18 to 29": "18-29",
    "18 to 30": "18-29",
    "youth": "18-29",
    "young": "18-29",
    # Friday Tennis & Tapas (occasional social, 4 rotations).
    "tapas": "tapas",
    "tennis & tapas": "tapas",
    "tennis and tapas": "tapas",
    "friday tennis & tapas": "tapas",
    "friday tennis and tapas": "tapas",
}


def _normalise_variant(raw: str | None) -> str | None:
    """Map a user-typed variant phrase to the canonical SessionType
    variant. Returns ``None`` for empty input, or the canonical
    string ("regular" / "18-29"); raises ValueError on an unknown
    phrase so the caller can ask for clarification."""
    if not raw:
        return None
    # Collapse en-dash/em-dash to plain hyphen, lowercase, trim.
    key = (
        raw.strip()
        .lower()
        .replace("–", "-")
        .replace("—", "-")
    )
    if key in _VARIANT_ALIASES:
        return _VARIANT_ALIASES[key]
    raise ValueError(
        f"unknown variant {raw!r}; valid: "
        f"{sorted(set(_VARIANT_ALIASES.values()))}"
    )


def tool_kickoff_session(
    session_type: Optional[str] = None,
    variant: Optional[str] = None,
) -> dict:
    """Run the kickoff workflow for one of the weekly tennis sessions.

    ``session_type`` is the explicit registry key — ``"tuesday"`` /
    ``"thursday"`` / ``"thursday_1829"`` / ``"saturday"``. Use this
    when the admin named a specific day.

    ``variant`` is the higher-level group label — ``"regular"`` or
    ``"18-29"``. Use this when the admin typed ``boris kickoff
    regular`` or ``boris kickoff 18-29``: the kickoff finds the next
    future event of that variant. User-typed aliases are normalised
    here (``"intermediate+"`` / ``"18-30"`` / ``"youth"`` etc.).

    When BOTH are omitted, the bot probes CourtReserve for today's
    events. If exactly one matches, it kicks off automatically. If
    two match (a Thursday with both the regular and 18-29 events
    listed), it returns ``needs_disambiguation`` so YOU can ask the
    admin which one — present the ``options`` to the user and re-call
    with the chosen ``variant``. If none match, the historic
    weekday fallback runs.

    Behaviour on success: fetches the matching CR event, auto-adds
    any new names to the roster, calls start_tonight (carrying the
    session_type through), sets phase = "awaiting_extras", posts the
    structured "today's lineup" message back to the calling channel.

    Behaviour when another session is already in flight: returns
    ``session_in_progress`` with a message identifying the loaded
    session by display name, so the admin can decide whether to
    finish it or 'boris clear run' before retrying.
    """
    from thursday_kickoff import kickoff_session

    try:
        canonical_variant = _normalise_variant(variant)
    except ValueError as e:
        return {"ok": False, "error": "unknown_variant", "message": str(e)}

    # Route the kickoff post back to whichever admin group asked for
    # it (so a kickoff fired from Boris the tennis bot stays there).
    target_jid = _CURRENT_GROUP_JID.get(None)
    result = kickoff_session(
        session_key=session_type or None,
        variant=canonical_variant,
        target_jid=target_jid,
    )
    # Strip the (long) message text — it's already been posted.
    return {k: v for k, v in result.items() if k != "message"}




def _iso_date_or_blank(value: Optional[str]) -> str:
    """``value`` as YYYY-MM-DD if it already is an ISO date, else ""."""
    from datetime import date as _date

    try:
        return _date.fromisoformat(str(value or "").strip()).isoformat()
    except ValueError:
        return ""


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
        session_date=_iso_date_or_blank(date),
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
    from name_match import match_names

    matches = match_names(name, state.waitlist)
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
    pins_before = state.pinned_doubles
    on_late_court = match in state.late_court_pinned_players
    state, removed = remove_from_tonight(match)
    result: dict = {"ok": True, "removed": removed, "state": state.to_dict()}
    dropped = [p for p in pins_before if p not in state.pinned_doubles]
    if dropped:
        result["pins_removed"] = dropped
        result["pins_message"] = (
            f"Also removed {len(dropped)} pinned doubles match"
            f"{'es' if len(dropped) > 1 else ''} that included {match} "
            "— tell the admin and offer to re-pin with a replacement."
        )
    if on_late_court:
        result["late_court_message"] = (
            f"{match} was one of the 4 players for late court "
            f"{state.late_court_label}; it now has "
            f"{len(state.late_court_pinned_players)}. Ask the admin who "
            "should replace them (set_late_court) before generating."
        )
    return result


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


def tool_set_coaching_group(
    players: list[str],
    courts: list,
    rotations: int = 1,
) -> dict:
    """Set (replace) tonight's coaching group: players + courts with
    the coach for the first ``rotations`` rotations, joining the social
    afterwards. Names are fuzzy-matched against tonight's attendees."""
    from session_state import find_attendee_fuzzy, set_coaching_group

    if not isinstance(players, list) or not players:
        return {
            "ok": False,
            "error": "bad_players",
            "message": "coaching group needs a list of player names.",
        }
    resolved: list[str] = []
    for raw in players:
        matches = find_attendee_fuzzy(str(raw))
        if len(matches) == 0:
            return {
                "ok": False,
                "error": "no_match",
                "message": f"No attendee matches {raw!r} — add them first "
                "with add_to_tonight, or check the name.",
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
        if matches[0] not in resolved:
            resolved.append(matches[0])
    try:
        state = set_coaching_group(
            resolved, list(courts or []), int(rotations or 1),
        )
    except ValueError as e:
        return {"ok": False, "error": "invalid", "message": str(e)}
    return {
        "ok": True,
        "players": list(state.coaching_players),
        "courts": list(state.coaching_courts),
        "rotations": state.coaching_rotations,
        "state": state.to_dict(),
    }


def tool_clear_coaching_group() -> dict:
    """Drop tonight's coaching split (everyone plays every rotation)."""
    from session_state import clear_coaching_group

    state = clear_coaching_group()
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
