"""Boris tools for CourtReserve: club sessions, court
bookings, scheduled bookings, registrants and the
validated-member list.
"""

from __future__ import annotations

from typing import Optional

from roster import Roster
from bot_config import _CURRENT_GROUP_JID, _CURRENT_SENDER, _caller_account


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
        result = cr.cancel_court_reservation(reservation_id)
        if not result.get("ok"):
            return result
        if not (date and start_time_hhmm):
            # Cancelled by id only — no slot to re-check against.
            result["verified"] = False
            result["message"] = (
                "Cancel submitted; give the date and start time next "
                "time and I can confirm it's gone."
            )
            return result
        # Confirm the reservation is really gone (the click alone
        # doesn't prove CR accepted it).
        try:
            still = cr.find_my_court_booking(date, start_time_hhmm)
        except Exception as e:
            result["verified"] = False
            result["message"] = f"Cancel submitted but couldn't re-check: {e!r}"
            return result
        if still and str(still.get("reservation_id")) == str(reservation_id):
            return {
                "ok": False,
                "status": "cancel_not_confirmed",
                "reservation_id": reservation_id,
                "message": (
                    "Clicked cancel but the booking is still on "
                    "CourtReserve — please check and cancel it there."
                ),
            }
        result["verified"] = True
        return result


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
