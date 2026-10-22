"""Background jobs run from the poll loop: scheduled court
bookings and the stale-run reminder.
"""

from __future__ import annotations

import sys
from datetime import datetime

from bot_bridge import send_to_group
from bot_config import BOT_REPLY_PREFIX, STALE_RUN_REMINDER_MINUTES


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
            entry.id, succeeded=False, final=True,
            error=f"account_unknown:{entry.scheduled_by_account_key}",
        )
        return

    # Persist "firing now" before touching CR: if the bot dies mid-fire
    # (crash, deploy restart) the next loop sees the flag and reports
    # it instead of firing again — the first fire may have booked.
    sb.mark_in_flight(entry.id)
    # Set once a submit click may have reached CR. An exception after
    # that point is ambiguous (booked or not?), so it must not retry.
    submit_started = False
    result: dict | None = None

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
                    submit_started = True
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
                # Same as cr.book_court, split so an exception before
                # the submit click is retryable and one after isn't.
                prepared = cr.prepare_court_booking(
                    date=entry.play_date,
                    start_time_hhmm=entry.start_time_hhmm,
                    partner_name=entry.partner_name,
                    duration_minutes=entry.duration_minutes,
                    court_label=entry.court_label,
                    court_type=entry.court_type,
                    court_preference=acct_pref,
                )
                if isinstance(prepared, dict):
                    result = prepared
                else:
                    submit_started = True
                    result = prepared.submit()
    except Exception as e:
        import traceback as _tb
        print(
            f"[scheduler] booking #{entry.id} crashed: {e!r}\n"
            + _tb.format_exc(),
            file=sys.stderr,
        )
        if result is not None:
            pass  # submit already returned — keep its outcome
        elif submit_started:
            result = {"ok": False, "status": "uncertain", "error": repr(e)}
        else:
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
        final=final_attempt,
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
            if result.get("verified") is False:
                text += (
                    " I couldn't see it on the court grid afterwards, "
                    "so please check CourtReserve."
                )
        elif result.get("status") == "uncertain":
            text = (
                f"? Scheduled booking #{entry.id} for "
                f"{account.display_name} ({entry.play_date} "
                f"{entry.start_time_hhmm}) hit an error after submitting, "
                "so it may or may not have booked. Please check "
                "CourtReserve — I won't retry it."
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
    target_jid = state.channel_jid or fallback_jid
    if not target_jid:
        print(
            "[stale-run] no channel to remind in; skipping",
            file=sys.stderr,
        )
        return
    msg = (
        f"Reminder: there's a session in progress, started by "
        f"{who} at {when}, with no activity for over "
        f"{int(idle_minutes)} min. Reply \"boris continue\" to keep "
        f"working on it, or \"boris clear run\" to wipe it. "
        f"I won't clear it unless you ask."
    )
    try:
        send_to_group(target_jid, BOT_REPLY_PREFIX + msg)
        ss.mark_idle_reminder_sent()
        print(
            f"[stale-run] reminder posted (idle {int(idle_minutes)} min, "
            f"phase={state.phase!r})"
        )
    except Exception as e:
        print(f"[stale-run] post failed: {e!r}", file=sys.stderr)


def _maybe_fire_scheduled_bookings(now: datetime) -> None:
    """Fire any pending bookings whose window has opened. Best-effort."""
    import scheduled_bookings as sb

    try:
        interrupted = sb.take_interrupted()
    except Exception as e:
        print(f"[scheduler] take_interrupted error: {e!r}", file=sys.stderr)
        interrupted = []
    for entry in interrupted:
        print(
            f"[scheduler] booking #{entry.id} was interrupted mid-fire "
            "(bot restarted?) — not retrying",
            file=sys.stderr,
        )
        if entry.channel_jid:
            try:
                send_to_group(
                    entry.channel_jid,
                    BOT_REPLY_PREFIX + (
                        f"? Scheduled booking #{entry.id} "
                        f"({entry.play_date} {entry.start_time_hhmm}, "
                        f"partner {entry.partner_name}) was interrupted "
                        "by a restart while booking, so it may or may "
                        "not have gone through. Please check "
                        "CourtReserve — I won't retry it."
                    ),
                )
            except Exception as e:
                print(f"[scheduler] post-back failed: {e!r}", file=sys.stderr)
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
