"""Future-dated court bookings that fire when the CR booking window
opens.

CourtReserve opens court bookings 6 days in advance at 08:00 local
time (so Thursday's slots open the previous Friday at 08:00). Boris
lets the admin queue a booking ahead of that — e.g. on a Wednesday,
say "schedule a court for next Thursday" — and the bot wakes up at
the right moment to place it.

Persistence: ``scheduled_bookings.json`` at the repo root, gitignored.
Schema::

    {
      "next_id": 4,
      "pending": [ <ScheduledBooking dict>, ... ],
      "history": [ <ScheduledBooking dict>, ... ]
    }

Window timing: ``compute_window_opens_at(play_date)`` returns the
wall-clock moment the bot should attempt the booking — 08:00 local on
``play_date - 6 days``. The poll loop polls due_now() every tick;
when one or more entries qualify, they get fired in turn (synchronously,
so the call sites block the loop for the duration of the booking
flow — typically 10-30 s — but that's acceptable because firings are
rare and the watermark catches up).

Testing override: setting ``BORIS_BOOKING_OVERRIDE_OPENS_AT`` to an
ISO8601 timestamp causes due_now() to consider every pending entry's
window to be that timestamp instead. Useful for forcing an immediate
fire from the test channel.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

ROOT = Path(__file__).parent
SCHEDULED_PATH = ROOT / "scheduled_bookings.json"
LOCAL_TZ = ZoneInfo("Europe/London")

# Booking window: 6 days ahead at this local time.
WINDOW_OPEN_HOUR = 8
WINDOW_OPEN_MINUTE = 0
DAYS_AHEAD = 6

# Retry policy when the first fire returns "too early" / "taken".
MAX_FIRE_ATTEMPTS = 3
RETRY_INTERVAL_SECONDS = 10

# State strings.
STATE_SCHEDULED = "scheduled"
STATE_SUCCEEDED = "succeeded"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"


@dataclass
class ScheduledBooking:
    id: int
    scheduled_at: str
    scheduled_by_phone: Optional[str]
    scheduled_by_account_key: str
    channel_jid: Optional[str]
    play_date: str
    start_time_hhmm: str
    duration_minutes: int
    court_label: Optional[str]
    court_type: Optional[str]
    partner_name: str
    window_opens_at: str
    state: str = STATE_SCHEDULED
    fire_attempts: int = 0
    last_attempt_at: Optional[str] = None
    last_error: Optional[str] = None
    result: Optional[dict] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduledBooking":
        # Tolerate older shapes that lack later fields.
        return cls(
            id=int(d["id"]),
            scheduled_at=str(d.get("scheduled_at", "")),
            scheduled_by_phone=d.get("scheduled_by_phone"),
            scheduled_by_account_key=str(d.get("scheduled_by_account_key", "")),
            channel_jid=d.get("channel_jid"),
            play_date=str(d.get("play_date", "")),
            start_time_hhmm=str(d.get("start_time_hhmm", "")),
            duration_minutes=int(d.get("duration_minutes", 60)),
            court_label=d.get("court_label"),
            court_type=d.get("court_type"),
            partner_name=str(d.get("partner_name", "")),
            window_opens_at=str(d.get("window_opens_at", "")),
            state=str(d.get("state", STATE_SCHEDULED)),
            fire_attempts=int(d.get("fire_attempts", 0)),
            last_attempt_at=d.get("last_attempt_at"),
            last_error=d.get("last_error"),
            result=d.get("result"),
            notes=str(d.get("notes", "")),
        )


# ---- timing -------------------------------------------------------------


def compute_window_opens_at(
    play_date: str,
    *,
    tz: ZoneInfo = LOCAL_TZ,
) -> datetime:
    """Return the booking window's open moment for a given play date.

    ``play_date`` is an ISO date string (YYYY-MM-DD). The window opens
    at 08:00 local on ``play_date - DAYS_AHEAD`` days.
    """
    play = datetime.strptime(play_date, "%Y-%m-%d").date()
    open_date = play - timedelta(days=DAYS_AHEAD)
    return datetime.combine(
        open_date,
        time(WINDOW_OPEN_HOUR, WINDOW_OPEN_MINUTE),
        tzinfo=tz,
    )


def parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _override_now() -> Optional[datetime]:
    raw = os.environ.get("BORIS_BOOKING_OVERRIDE_OPENS_AT")
    if not raw:
        return None
    return parse_iso(raw)


# ---- persistence --------------------------------------------------------


@dataclass
class _Store:
    next_id: int = 1
    pending: list[ScheduledBooking] = field(default_factory=list)
    history: list[ScheduledBooking] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "next_id": self.next_id,
            "pending": [b.to_dict() for b in self.pending],
            "history": [b.to_dict() for b in self.history],
        }


def _load(path: Path | None = None) -> _Store:
    target = path or SCHEDULED_PATH
    if not target.exists():
        return _Store()
    raw = json.loads(target.read_text(encoding="utf-8"))
    return _Store(
        next_id=int(raw.get("next_id", 1)),
        pending=[ScheduledBooking.from_dict(d) for d in raw.get("pending", [])],
        history=[ScheduledBooking.from_dict(d) for d in raw.get("history", [])],
    )


def _save(store: _Store, path: Path | None = None) -> None:
    target = path or SCHEDULED_PATH
    target.write_text(
        json.dumps(store.to_json(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ---- public API ---------------------------------------------------------


def add_pending(
    *,
    scheduled_by_phone: Optional[str],
    scheduled_by_account_key: str,
    channel_jid: Optional[str],
    play_date: str,
    start_time_hhmm: str,
    duration_minutes: int,
    partner_name: str,
    court_label: Optional[str] = None,
    court_type: Optional[str] = None,
    notes: str = "",
    now: Optional[datetime] = None,
    path: Path | None = None,
) -> ScheduledBooking:
    """Append a new pending booking and return it (with id assigned)."""
    store = _load(path)
    now = now or datetime.now(LOCAL_TZ)
    window = compute_window_opens_at(play_date)
    entry = ScheduledBooking(
        id=store.next_id,
        scheduled_at=now.isoformat(),
        scheduled_by_phone=scheduled_by_phone,
        scheduled_by_account_key=scheduled_by_account_key,
        channel_jid=channel_jid,
        play_date=play_date,
        start_time_hhmm=start_time_hhmm,
        duration_minutes=int(duration_minutes),
        court_label=court_label,
        court_type=court_type,
        partner_name=partner_name,
        window_opens_at=window.isoformat(),
        notes=notes,
    )
    store.pending.append(entry)
    store.next_id += 1
    _save(store, path)
    return entry


def list_pending(
    *, account_key: Optional[str] = None, path: Path | None = None
) -> list[ScheduledBooking]:
    """Return pending bookings, optionally filtered to one account."""
    store = _load(path)
    if account_key is None:
        return list(store.pending)
    return [b for b in store.pending if b.scheduled_by_account_key == account_key]


def list_history(
    *,
    account_key: Optional[str] = None,
    limit: int = 20,
    path: Path | None = None,
) -> list[ScheduledBooking]:
    store = _load(path)
    items = (
        store.history
        if account_key is None
        else [b for b in store.history if b.scheduled_by_account_key == account_key]
    )
    return items[-limit:]


def cancel_pending(
    booking_id: int,
    *,
    by_account_key: Optional[str] = None,
    now: Optional[datetime] = None,
    path: Path | None = None,
) -> tuple[bool, Optional[ScheduledBooking]]:
    """Move a pending entry to history with state=cancelled.

    Returns (cancelled?, entry). ``by_account_key`` (when set) is
    enforced — a non-matching account can't cancel another's entry.
    """
    store = _load(path)
    for i, entry in enumerate(store.pending):
        if entry.id != booking_id:
            continue
        if (
            by_account_key
            and entry.scheduled_by_account_key != by_account_key
        ):
            return False, None
        entry.state = STATE_CANCELLED
        entry.last_attempt_at = (now or datetime.now(LOCAL_TZ)).isoformat()
        store.pending.pop(i)
        store.history.append(entry)
        _save(store, path)
        return True, entry
    return False, None


def due_now(
    now: Optional[datetime] = None, *, path: Path | None = None
) -> list[ScheduledBooking]:
    """Return pending entries whose window has opened.

    Honours BORIS_BOOKING_OVERRIDE_OPENS_AT — when set, the override
    is treated as the effective ``now`` (firing every pending entry
    whose window has opened relative to the override).
    """
    store = _load(path)
    effective = _override_now() or now or datetime.now(LOCAL_TZ)
    out: list[ScheduledBooking] = []
    for entry in store.pending:
        opens = parse_iso(entry.window_opens_at)
        if opens is None:
            continue
        if opens <= effective:
            # Apply retry pacing: skip if last attempt was too recent.
            if entry.last_attempt_at:
                last = parse_iso(entry.last_attempt_at)
                if (
                    last is not None
                    and (effective - last).total_seconds() < RETRY_INTERVAL_SECONDS
                ):
                    continue
            out.append(entry)
    return out


def mark_attempt(
    booking_id: int,
    *,
    succeeded: bool,
    result: Optional[dict] = None,
    error: Optional[str] = None,
    now: Optional[datetime] = None,
    path: Path | None = None,
) -> Optional[ScheduledBooking]:
    """Record an attempt against a pending booking.

    On ``succeeded=True`` or after MAX_FIRE_ATTEMPTS failures the
    entry is moved into history with the final state. Otherwise it
    stays in pending with an updated last_attempt_at / fire_attempts /
    last_error so retries can pace themselves.
    """
    store = _load(path)
    for i, entry in enumerate(store.pending):
        if entry.id != booking_id:
            continue
        entry.fire_attempts += 1
        entry.last_attempt_at = (now or datetime.now(LOCAL_TZ)).isoformat()
        entry.last_error = error
        entry.result = result
        if succeeded:
            entry.state = STATE_SUCCEEDED
            store.pending.pop(i)
            store.history.append(entry)
        elif entry.fire_attempts >= MAX_FIRE_ATTEMPTS:
            entry.state = STATE_FAILED
            store.pending.pop(i)
            store.history.append(entry)
        _save(store, path)
        return entry
    return None
