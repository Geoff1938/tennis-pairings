"""CourtReserve scraper for the Westside members' portal.

Used by the tennis-pairings admin bot to answer questions like "who's signed up
for the next Tuesday evening social?". We scrape because:

  * CourtReserve's "public" API endpoints (events.courtreserve.com/…/ApiLoadEvents)
    require a session-bound encrypted `requestData` token that's embedded in the
    rendered HTML, and
  * those endpoints return HTML fragments wrapped in JSON anyway.

So we drive a real browser via Playwright with a persistent user-data directory
(login happens once; cookies live on disk). Cloudflare is friendlier to headed
browsers for initial logins — the client falls back from headless to headed if
CF blocks us.

Usage:
    with CourtReserveClient() as cr:
        events = cr.list_events(category="Social & Club Sessions",
                                day_of_week="Thu", days_ahead=14)
        for e in events:
            print(e.name, e.date_str)
            reg = cr.get_event_registrants(e.reservation_number)
            print(f"  {len(reg['registrants'])} registered")

Everything lives on a single browser page — methods are not thread-safe; open a
fresh client per workflow.
"""

from __future__ import annotations

import os
import re
import time as _t
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from dateutil import parser as date_parser
from dotenv import load_dotenv
from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    sync_playwright,
    TimeoutError as PlaywrightTimeout,
)

# ---------- config -------------------------------------------------------

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

CR_PORTAL_URL = os.environ.get(
    "COURTRESERVE_URL", "https://app.courtreserve.com/Online/Portal/Index/2146"
)
CR_USERNAME = os.environ.get("COURTRESERVE_USERNAME")
CR_PASSWORD = os.environ.get("COURTRESERVE_PASSWORD")
CR_STATE_DIR = ROOT / ".cr_state"
EVENTS_LIST_URL = "https://app.courtreserve.com/Online/Events/List/2146"
DETAIL_URL_TMPL = (
    "https://app.courtreserve.com/Online/Events/Details/2146/{rn}?resId={res_id}"
)
DETAIL_BY_RESID_TMPL = (
    "https://app.courtreserve.com/Online/Events/Details/2146?resId={res_id}"
)
MY_BOOKINGS_URL_TMPL = (
    "https://app.courtreserve.com/Online/Bookings/List/2146?type={type}"
)
# CourtReserve "type" query-string values for the My Bookings page.
MY_BOOKINGS_TYPE_REGISTERED = "4"
MY_BOOKINGS_TYPE_WAITLISTED = "5"
# Court bookings — schedulers + court-number → court-type mapping.
COURT_BOOKINGS_URL_TMPL = (
    "https://app.courtreserve.com/Online/Reservations/Bookings/2146?sId={sid}"
)
_LOCAL_TZ = ZoneInfo("Europe/London")
SCHEDULER_ID_ACRYLIC = "17"
SCHEDULER_ID_CLAY = "18"
COURT_NUMBER_TO_SCHEDULER_ID: dict[str, str] = {
    "1": SCHEDULER_ID_ACRYLIC, "2": SCHEDULER_ID_ACRYLIC,
    "3": SCHEDULER_ID_ACRYLIC, "4": SCHEDULER_ID_ACRYLIC,
    "5": SCHEDULER_ID_CLAY,    "6": SCHEDULER_ID_CLAY,
    "7": SCHEDULER_ID_CLAY,    "8": SCHEDULER_ID_CLAY,
    "9": SCHEDULER_ID_CLAY,    "10": SCHEDULER_ID_CLAY,
    "11": SCHEDULER_ID_CLAY,   "12": SCHEDULER_ID_CLAY,
}
COURT_TYPE_TO_NUMBERS: dict[str, list[str]] = {
    "clay":    ["5", "6", "7", "8", "9", "10", "11", "12"],
    "acrylic": ["1", "2", "3", "4"],
    "hard":    ["1", "2", "3", "4"],
}
DEFAULT_COURT_PREFERENCE = [
    "5", "6", "9", "7", "8", "10", "11", "12", "4", "1", "2", "3",
]
DURATION_TO_RES_TYPE: dict[int, str] = {
    30: "30 min hit",
    60: "60 min hit",
    90: "1 hour 30 min hit",
    120: "ladder match",
}
COURT_CANCEL_URL_TMPL = (
    "https://app.courtreserve.com/Online/MyProfile/CancelReservation/2146"
    "?reservationId={reservation_id}"
)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)
VIEWPORT = {"width": 1280, "height": 900}
DEFAULT_TIMEOUT_MS = 30_000


# ---------- data classes -------------------------------------------------


@dataclass
class EventInfo:
    """A single event card from the Events list page."""

    name: str
    category: str
    date_str: str  # raw, e.g. "Tue, Apr 28th, 19:30 - 21:30"
    reservation_number: str
    res_id: str
    detail_url: str
    start_dt: datetime | None = None  # best-effort parsed start datetime


@dataclass
class Registrant:
    name: str


@dataclass
class EventRegistration:
    event_name: str
    category: str
    date_str: str
    detail_url: str
    spots_remaining: str | None  # e.g. "17 of 28" or None if not present
    courts: list[str] = field(default_factory=list)
    age_restriction: str | None = None
    registrants: list[Registrant] = field(default_factory=list)
    # Waitlist appears when the event is at capacity. Order is the priority
    # order the club uses to promote players when slots open up.
    waitlist: list[Registrant] = field(default_factory=list)
    is_full: bool = False


# ---------- HTML parsing helpers -----------------------------------------


_CARD_START_RE = re.compile(
    r'<div class="fj_post style2 jlv5" data-testid="event-card">'
)
_NAME_RE = re.compile(
    r'<h4 data-testid="event-name">\s*(.+?)\s*</h4>', re.DOTALL
)
_DATE_RE = re.compile(
    r'<a data-testid="date-time-section">\s*(.+?)\s*</a>', re.DOTALL
)
_CATEGORY_RE = re.compile(
    r'<h5[^>]*data-testid="category-name">\s*(.+?)\s*<i', re.DOTALL
)
_DETAIL_RE = re.compile(
    r'/Online/Events/Details/2146/(\w+)\?resId=(\d+)'
)

_REGISTRANTS_NAME_RE = re.compile(
    r'<tbody data-testid="registrants-table-body">(.*?)</tbody>', re.DOTALL
)
_REGISTRANT_ROW_RE = re.compile(
    r'data-testid="name"[^>]*>\s*(.+?)\s*</th>', re.DOTALL
)
_WAITLIST_BODY_RE = re.compile(
    r'<tbody data-testid="waitlisted-table-body">(.*?)</tbody>', re.DOTALL
)
# Waitlist rows lack a data-testid; pattern is `<th scope="col">N</th>` then
# `<th scope="row" ...>Name</th>` inside each <tr>.
_WAITLIST_ROW_RE = re.compile(
    r'<th[^>]*scope="row"[^>]*>\s*(.+?)\s*</th>', re.DOTALL
)
_FULL_BADGE_RE = re.compile(r'\bFULL\b', re.IGNORECASE)
_SPOTS_RE = re.compile(r'(\d+\s+of\s+\d+)\s+spots?\s+remaining', re.IGNORECASE)
_COURTS_RE = re.compile(
    r'<p[^>]*data-testid="court-details"[^>]*>\s*(.+?)\s*</p>', re.DOTALL
)
_AGE_RE = re.compile(
    r'<p[^>]*data-testid="min-age"[^>]*>\s*(.+?)\s*</p>', re.DOTALL
)


def _clean(text: str) -> str:
    """Collapse whitespace and decode a couple of common HTML entities."""
    text = re.sub(r"\s+", " ", text).strip()
    return (
        text.replace("&amp;", "&")
        .replace("&nbsp;", " ")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
    )


def _strip_tags(text: str) -> str:
    return _clean(re.sub(r"<[^>]+>", " ", text))


def _parse_event_cards(html: str) -> list[EventInfo]:
    """Pull EventInfo out of a /Online/Events/List/... HTML page."""
    starts = [m.start() for m in _CARD_START_RE.finditer(html)]
    if not starts:
        return []
    starts.append(len(html))
    out: list[EventInfo] = []
    for i in range(len(starts) - 1):
        card = html[starts[i] : starts[i + 1]]
        name_m = _NAME_RE.search(card)
        ids_m = _DETAIL_RE.search(card)
        if not name_m or not ids_m:
            continue
        name = _strip_tags(name_m.group(1))
        date_s = _clean(_DATE_RE.search(card).group(1)) if _DATE_RE.search(card) else ""
        cat_s = _clean(_CATEGORY_RE.search(card).group(1)) if _CATEGORY_RE.search(card) else ""
        rn, res_id = ids_m.group(1), ids_m.group(2)
        detail_url = DETAIL_URL_TMPL.format(rn=rn, res_id=res_id)
        # Best-effort parse of the start datetime (just the first date + time).
        start_dt: datetime | None = None
        try:
            # Extract the first "Weekday, Month Dayth" and the first HH:MM in the string.
            m_date = re.search(r"([A-Za-z]{3},\s+[A-Za-z]{3}\s+\d{1,2}(?:st|nd|rd|th)?)", date_s)
            m_time = re.search(r"(\d{1,2}:\d{2})", date_s)
            if m_date and m_time:
                start_dt = date_parser.parse(f"{m_date.group(1)} {m_time.group(1)}")
        except Exception:
            start_dt = None
        out.append(
            EventInfo(
                name=name,
                category=cat_s,
                date_str=date_s,
                reservation_number=rn,
                res_id=res_id,
                detail_url=detail_url,
                start_dt=start_dt,
            )
        )
    return out


def _parse_event_detail(html: str) -> EventRegistration:
    """Pull registrants and metadata from a /Online/Events/Details/... HTML page."""
    # Title: detail page uses <h4 … data-testid="event-name">Title</h4>.
    title = ""
    m = re.search(
        r'<h[1-4][^>]*data-testid="event-name"[^>]*>\s*(.+?)\s*</h[1-4]>',
        html,
        re.DOTALL,
    )
    if m:
        title = _strip_tags(m.group(1))

    # Category: detail page uses <span … data-testid="event-type">Social & Club Sessions</span>.
    # (List page uses data-testid="category-name" inside an <h5>.)
    category = ""
    m_type = re.search(
        r'data-testid="event-type"[^>]*>\s*(.+?)\s*</span>', html, re.DOTALL
    )
    if m_type:
        category = _clean(m_type.group(1))
    else:
        m_cat = _CATEGORY_RE.search(html)
        if m_cat:
            category = _clean(m_cat.group(1))

    # Detail page splits date and times into separate spans (not one <a> like the list page).
    date_str = ""
    date_part = re.search(
        r'<span[^>]*data-testid="date"[^>]*>\s*(.+?)\s*</span>', html, re.DOTALL
    )
    times_part = re.search(
        r'<span[^>]*data-testid="times"[^>]*>\s*(.+?)\s*</span>', html, re.DOTALL
    )
    if date_part and times_part:
        date_str = f"{_clean(date_part.group(1))}, {_clean(times_part.group(1))}"
    elif date_part:
        date_str = _clean(date_part.group(1))
    else:
        m_date = _DATE_RE.search(html)
        if m_date:
            date_str = _clean(m_date.group(1))

    spots: str | None = None
    m_spots = _SPOTS_RE.search(html)
    if m_spots:
        spots = m_spots.group(1)

    courts: list[str] = []
    m_courts = re.search(
        r'<p[^>]*data-testid="courts"[^>]*>\s*(.+?)\s*</p>', html, re.DOTALL
    )
    if m_courts:
        raw = _strip_tags(m_courts.group(1))
        courts = [c.strip() for c in raw.split(",") if c.strip()]

    age = None
    m_age = re.search(
        r'data-testid="min-age"[^>]*>\s*(.+?)\s*</[a-z]+>', html, re.DOTALL
    )
    if m_age:
        age = _strip_tags(m_age.group(1))

    registrants: list[Registrant] = []
    m_body = _REGISTRANTS_NAME_RE.search(html)
    if m_body:
        for m in _REGISTRANT_ROW_RE.finditer(m_body.group(1)):
            nm = _strip_tags(m.group(1))
            if nm:
                registrants.append(Registrant(name=nm))

    waitlist: list[Registrant] = []
    m_wbody = _WAITLIST_BODY_RE.search(html)
    if m_wbody:
        for m in _WAITLIST_ROW_RE.finditer(m_wbody.group(1)):
            nm = _strip_tags(m.group(1))
            if nm:
                waitlist.append(Registrant(name=nm))

    # The presence of a waitlist tbody is the most reliable signal that
    # the event is at (or beyond) capacity. The visible badge ("Full") is
    # rendered via a CSS uppercase class so its text is mixed-case in HTML.
    is_full = bool(m_wbody) or bool(re.search(r'>\s*Full\s*<', html))

    return EventRegistration(
        event_name=title,
        category=category,
        date_str=date_str,
        detail_url="",  # filled in by caller
        spots_remaining=spots,
        courts=courts,
        age_restriction=age,
        registrants=registrants,
        waitlist=waitlist,
        is_full=is_full,
    )


# ---------- filter helpers -----------------------------------------------


def _match_day_of_week(event: EventInfo, day: str | None) -> bool:
    if not day:
        return True
    token = day[:3].lower()
    return event.date_str.lower().startswith(token)


def _match_days_ahead(event: EventInfo, days_ahead: int | None) -> bool:
    if not days_ahead:
        return True
    if not event.start_dt:
        return True  # can't parse → don't filter out
    cutoff = datetime.now() + timedelta(days=days_ahead)
    return event.start_dt <= cutoff and event.start_dt >= datetime.now() - timedelta(hours=1)


def _match_name_contains(event: EventInfo, needle: str | None) -> bool:
    if not needle:
        return True
    return needle.lower() in event.name.lower()


def _match_category(event: EventInfo, needle: str | None) -> bool:
    if not needle:
        return True
    return needle.lower() in event.category.lower()


# ---------- client -------------------------------------------------------


class CloudflareBlocked(RuntimeError):
    """Raised when CourtReserve's Cloudflare bot check blocks our browser."""


@dataclass
class PreparedBooking:
    """A CourtReserveClient sitting at a state where the booking modal
    is open and form-filled — call ``submit()`` to finalise.

    Used by the scheduler so the bot can prepare a booking BEFORE the
    CR window opens (handling Chromium boot, login, navigation, slot
    click, partner pick) and then submit at the right moment with
    minimum latency. ``submit()`` is idempotent against transient
    too-early errors — it retries the click for up to ~12s on
    'too early' / 'available at' / similar messages.
    """
    client: "CourtReserveClient"
    prep: dict
    attempted: list[str] = field(default_factory=list)
    skipped_prebooked: list[str] = field(default_factory=list)
    # Fields needed for in-flight iteration when CR rejects the first
    # court at submit-time with "no available courts for the time
    # requested" (SwAl2). On rejection, submit() advances to the next
    # entry in ``remaining_candidates`` (same scheduler page — we don't
    # switch sIds mid-flight; the scheduler's 10s retry covers that).
    start_time_hhmm: str = ""
    partner_name: str = ""
    duration_minutes: int = 90
    remaining_candidates: list[str] = field(default_factory=list)

    def submit(self) -> dict:
        """Click submit on the current prep. If CR rejects with
        "court taken at submit-time", dismiss the SwAl2, escape the
        modal, and try the next candidate on the same scheduler page
        (re-prep + re-submit) until either we book successfully, the
        candidates are exhausted, or we hit a non-court-specific
        error (e.g. partner_not_found).
        """
        result = self.client._finalise_booking(self.prep)
        # Loop only on the specific "this court got taken under us"
        # signal. Non-recoverable failures (partner_not_found,
        # submit_rejected with other messages, hard validation) break
        # the loop on the very next iteration.
        while (
            result.get("status") == "submit_court_taken"
            and self.remaining_candidates
        ):
            next_court = self.remaining_candidates.pop(0)
            print(
                f"[cr/submit] court {self.prep.get('court_label')!r} taken "
                f"at submit — trying next candidate {next_court!r}"
            )
            self.attempted.append(next_court)
            new_prep = self.client._prepare_one_court(
                next_court,
                self.start_time_hhmm,
                self.partner_name,
                self.duration_minutes,
            )
            if new_prep is None:
                # Slot button no longer in the DOM — CR has now
                # removed it because the other booking just landed.
                # Try the next candidate.
                continue
            if new_prep.get("status") != "ready":
                # Non-fatal but not retryable across courts (e.g.
                # partner_not_found). Surface it as the final result.
                result = {"ok": False, **new_prep}
                break
            self.prep = new_prep
            result = self.client._finalise_booking(new_prep)
        result["attempted"] = list(self.attempted)
        result["skipped_prebooked"] = list(self.skipped_prebooked)
        return result

    def abandon(self) -> None:
        """Close the modal cleanly without booking. Lets the same
        client be reused for a later attempt."""
        page = self.client._page
        if page is None:
            return
        try:
            page.keyboard.press("Escape")
            page.wait_for_timeout(500)
        except Exception:
            pass


def normalize_hhmm(s: str) -> str:
    """Canonicalise a 24-hour clock-time string to zero-padded ``HH:MM``.

    Accepts the messy shapes produced by different callers / LLM tool
    schemas over the bot's lifetime:

      * ``"13:00"`` / ``"9:30"`` — already-colon, possibly unpadded
      * ``"1300"`` / ``"0930"`` — 4-digit no-colon (legacy schedule path)
      * ``"930"``  / ``"100"``  — 3-digit no-colon (early morning hours)
      * ``"9"``    / ``"13"``   — bare hour (assumed :00)

    Returns canonical ``"HH:MM"``. Raises ``ValueError`` for anything
    not in the 00:00–23:59 range or that can't be parsed at all.

    Normalising at the public boundary lets every downstream consumer
    rely on a single shape — in particular ``_find_slot_button``'s
    substring trick (``f"{hhmm}:00"`` matched against the CR slot
    ``start`` attribute) only works for ``"HH:MM"``.
    """
    if s is None:
        raise ValueError("start_time_hhmm is required (got None)")
    raw = str(s).strip()
    if not raw:
        raise ValueError("start_time_hhmm is required (got empty string)")
    if ":" in raw:
        parts = raw.split(":")
        if len(parts) != 2:
            raise ValueError(f"invalid time {s!r}: expected HH:MM")
        hh_str, mm_str = parts[0], parts[1]
    else:
        # 4-digit "1300", 3-digit "930", 2-digit "13" (hour-only),
        # 1-digit "9" (hour-only). All other lengths are invalid.
        if not raw.isdigit():
            raise ValueError(f"invalid time {s!r}: non-numeric")
        if len(raw) == 4:
            hh_str, mm_str = raw[:2], raw[2:]
        elif len(raw) == 3:
            hh_str, mm_str = raw[:1], raw[1:]
        elif len(raw) in (1, 2):
            hh_str, mm_str = raw, "00"
        else:
            raise ValueError(f"invalid time {s!r}: unexpected length")
    try:
        hh, mm = int(hh_str), int(mm_str)
    except ValueError as exc:
        raise ValueError(f"invalid time {s!r}: not integer") from exc
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"invalid time {s!r}: out of range")
    return f"{hh:02d}:{mm:02d}"


def _compute_blocked_courts(
    reservations: list[dict],
    play_date: str,
    start_time_hhmm: str,
    duration_minutes: int,
) -> set[str]:
    """Return court numbers (as strings, e.g. {"5", "9"}) whose
    target slot is blocked by any reservation.

    ``reservations`` is the list produced by
    ``CourtReserveClient._read_scheduled_reservations`` — each entry
    has ISO-UTC ``start`` and ``end`` (Kendo dataSource shape) and a
    parsed ``court_number``. Comparisons happen in UTC; the target
    time is interpreted in Europe/London (LOCAL_TZ).
    """
    hhmm = normalize_hhmm(start_time_hhmm)
    target_start_local = datetime.strptime(
        f"{play_date} {hhmm}", "%Y-%m-%d %H:%M"
    ).replace(tzinfo=_LOCAL_TZ)
    target_end_local = target_start_local + timedelta(minutes=duration_minutes)
    target_start_utc = target_start_local.astimezone(timezone.utc)
    target_end_utc = target_end_local.astimezone(timezone.utc)
    blocked: set[str] = set()
    for r in reservations:
        s, e = r.get("start"), r.get("end")
        if not s or not e:
            continue
        try:
            r_start = datetime.fromisoformat(s.replace("Z", "+00:00"))
            r_end = datetime.fromisoformat(e.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        # [r_start, r_end) overlaps [target_start, target_end) iff
        # r_start < target_end AND r_end > target_start.
        if r_start < target_end_utc and r_end > target_start_utc:
            cn = r.get("court_number")
            if cn:
                blocked.add(cn)
    return blocked


class CourtReserveClient:
    """Playwright-driven client against app.courtreserve.com.

    Used as a context manager. First entry may show a visible browser window
    (headed) if the headless session trips Cloudflare; subsequent entries
    reuse the persistent user-data directory and typically stay headless.
    """

    def __init__(
        self,
        user_data_dir: Path | None = None,
        headless: bool = True,
        portal_url: str = CR_PORTAL_URL,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.user_data_dir = user_data_dir or CR_STATE_DIR
        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.portal_url = portal_url
        self.username = username or CR_USERNAME
        self.password = password or CR_PASSWORD
        if not self.username or not self.password:
            raise RuntimeError(
                "COURTRESERVE_USERNAME / COURTRESERVE_PASSWORD not set in .env"
            )
        self._pw: Playwright | None = None
        self._ctx: BrowserContext | None = None
        self._page: Page | None = None
        self._member_full_name: str | None = None
        self._current_user_id: str | None = None

    def __enter__(self) -> "CourtReserveClient":
        self._pw = sync_playwright().start()
        self._ctx = self._launch(self.headless)
        self._page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        self._page.set_default_timeout(DEFAULT_TIMEOUT_MS)
        return self

    def __exit__(self, *exc) -> None:
        try:
            if self._ctx is not None:
                self._ctx.close()
        finally:
            if self._pw is not None:
                self._pw.stop()

    def _launch(self, headless: bool) -> BrowserContext:
        return self._pw.chromium.launch_persistent_context(  # type: ignore[union-attr]
            user_data_dir=str(self.user_data_dir),
            headless=headless,
            user_agent=UA,
            viewport=VIEWPORT,
            locale="en-GB",
            timezone_id="Europe/London",
        )

    # --- auth --------------------------------------------------------------

    def _looks_like_cloudflare(self, page: Page) -> bool:
        try:
            body = (page.content() or "").lower()
        except Exception:
            return False
        return (
            "attention required" in body
            and "cloudflare" in body
            and "/cdn-cgi/" in body
        )

    def ensure_logged_in(self) -> None:
        """Navigate to the portal; log in if the session has expired.

        If Cloudflare blocks the headless session, re-launch headed and retry
        the login (one-off, usually only needed once). Cookies persist in the
        user-data dir thereafter.
        """
        page = self._page
        assert page is not None
        try:
            page.goto(self.portal_url, wait_until="domcontentloaded", timeout=DEFAULT_TIMEOUT_MS)
        except PlaywrightTimeout:
            pass
        if self._looks_like_cloudflare(page) and self.headless:
            # Retry once headed — CF is more lenient on real browsers.
            self._ctx.close()  # type: ignore[union-attr]
            self._ctx = self._launch(headless=False)
            self._page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
            self._page.set_default_timeout(DEFAULT_TIMEOUT_MS)
            self.headless = False
            page = self._page
            page.goto(self.portal_url, wait_until="domcontentloaded")
            if self._looks_like_cloudflare(page):
                raise CloudflareBlocked(
                    "Cloudflare still blocking us in headed mode. "
                    "Log in manually once via a visible browser to refresh cookies."
                )

        # If a "Log In" link is present we're on the public landing — click it.
        login_link = page.locator("text=Log In").first
        if login_link.count() > 0:
            login_link.click()
            page.wait_for_load_state("domcontentloaded")

        email_input = page.locator("input[name='email']").first
        if email_input.count() == 0:
            # Already authenticated — no form present.
            return
        email_input.fill(self.username)  # type: ignore[arg-type]
        page.fill("input[name='password']", self.password)  # type: ignore[arg-type]
        page.locator(
            "button[type='submit'], button:has-text('Log In')"
        ).first.click()
        page.wait_for_load_state("networkidle", timeout=DEFAULT_TIMEOUT_MS)

    # --- listing -----------------------------------------------------------

    def list_events(
        self,
        *,
        category: str | None = None,
        day_of_week: str | None = None,
        days_ahead: int | None = 14,
        name_contains: str | None = None,
    ) -> list[EventInfo]:
        """Return filtered events from the unfiltered Events list.

        Filters are applied locally (the list page already returns everything).

        Args:
            category: substring match against the event's category label
                (e.g. ``"Social & Club Sessions"``).
            day_of_week: matches the first three chars of the date string
                (``"Thursday"`` → ``"Thu"``).
            days_ahead: exclude events starting more than this many days from
                now, and exclude anything already finished.
            name_contains: case-insensitive substring match against the event
                name.
        """
        assert self._page is not None
        self.ensure_logged_in()
        self._page.goto(EVENTS_LIST_URL, wait_until="networkidle", timeout=DEFAULT_TIMEOUT_MS)
        html = self._page.content()
        events = _parse_event_cards(html)
        events = [
            e
            for e in events
            if _match_category(e, category)
            and _match_day_of_week(e, day_of_week)
            and _match_days_ahead(e, days_ahead)
            and _match_name_contains(e, name_contains)
        ]
        # Sort by parsed start_dt when we have it, else leave in page order.
        events.sort(key=lambda e: (e.start_dt or datetime.max))
        return events

    # --- registrants -------------------------------------------------------

    def get_event_registrants(
        self, reservation_number_or_url: str, res_id: str | None = None
    ) -> EventRegistration:
        """Fetch an event's detail page and parse the registrants section."""
        assert self._page is not None
        self.ensure_logged_in()

        if reservation_number_or_url.startswith("http"):
            url = reservation_number_or_url
        elif reservation_number_or_url.startswith("/"):
            url = "https://app.courtreserve.com" + reservation_number_or_url
        else:
            if not res_id:
                # No res_id supplied; try to resolve via the events list.
                candidates = self.list_events(days_ahead=180)
                match = next(
                    (e for e in candidates if e.reservation_number == reservation_number_or_url),
                    None,
                )
                if not match:
                    raise ValueError(
                        f"Could not resolve reservation_number {reservation_number_or_url!r} "
                        "to a detail URL — pass the full detail_url or res_id."
                    )
                url = match.detail_url
            else:
                url = DETAIL_URL_TMPL.format(rn=reservation_number_or_url, res_id=res_id)

        self._page.goto(url, wait_until="networkidle", timeout=DEFAULT_TIMEOUT_MS)
        # The Kendo tabstrip that holds the registrants table is rendered by
        # JS after the initial HTML arrives — networkidle alone isn't enough
        # on a cold browser. Wait for the table body to appear; fall back to
        # whatever's there after the timeout.
        try:
            self._page.wait_for_selector(
                "tbody[data-testid='registrants-table-body']",
                state="attached",
                timeout=10_000,
            )
        except PlaywrightTimeout:
            pass
        html = self._page.content()
        reg = _parse_event_detail(html)
        reg.detail_url = url
        return reg

    # --- registration -----------------------------------------------------

    def get_member_full_name(self) -> str:
        """Logged-in member's full name, e.g. ``"Geoff Chapman"``.

        Cached after first lookup. Sources the name from the user-dropdown
        item in the portal nav (a ``parent-header-link`` link inside the
        ``float-right`` nav list item).
        """
        if self._member_full_name:
            return self._member_full_name
        self.ensure_logged_in()
        page = self._page
        assert page is not None
        page.goto(self.portal_url, wait_until="networkidle", timeout=DEFAULT_TIMEOUT_MS)
        # The user dropdown <li> floats to the right of the nav. Inside
        # it sits a parent-header-link <a> whose <span> contains the full
        # name. The "Pay Now" link is also a parent-header-link but is
        # not float-right, so this scoping is enough.
        candidates = page.locator(
            "li.float-right a.parent-header-link span"
        ).all()
        for loc in candidates:
            try:
                txt = loc.inner_text().strip()
            except Exception:
                continue
            tokens = txt.split()
            if 2 <= len(tokens) <= 4 and all(t[:1].isupper() for t in tokens):
                self._member_full_name = txt
                return txt
        raise RuntimeError(
            "Could not auto-detect member name from the portal nav."
        )

    def get_current_user_id(self) -> str | None:
        """Logged-in member's CR ``UserId`` as a string (e.g. ``"92963"``),
        or ``None`` if it can't be extracted.

        Cached after the first lookup. CR doesn't expose this via a
        documented API on portal pages, so we scrape it from inline
        JavaScript: the first ``userId : <N>`` or similar token in the
        page HTML is reliably the user-dropdown's own id (the dropdown
        renders near the top of the document, before any other member's
        id appears). Used by ``find_my_court_booking`` to identify
        which Kendo events are owned by this account vs other members.
        """
        if self._current_user_id is not None:
            return self._current_user_id or None
        self.ensure_logged_in()
        page = self._page
        assert page is not None
        # Use the current page (whatever it is) — every CR portal page
        # carries the user-dropdown markup with this token.
        try:
            uid = page.evaluate(
                """() => {
                    const html = document.documentElement.outerHTML;
                    const m = html.match(
                        /(?:["']?userId["']?|MemberId)\\s*[:=]\\s*["']?(\\d{3,10})/i
                    );
                    return m ? m[1] : null;
                }"""
            )
        except Exception:
            uid = None
        self._current_user_id = uid or ""
        return uid

    def register_for_event(
        self,
        reservation_number: str,
        *,
        allow_waitlist_fallback: bool = True,
    ) -> dict:
        """Register the logged-in member for an event.

        Idempotent: returns ``status="already_registered"`` /
        ``"already_waitlisted"`` if the member is already on either list.
        Otherwise looks for the action button on the event detail page:

          * a "Register" / "Reserve" type button → click it, end status
            ``"registered"``;
          * a "Join Waitlist" button (event full) → click it (if
            ``allow_waitlist_fallback`` is True), end status
            ``"waitlisted"``; otherwise return
            ``"event_full_no_fallback"`` without acting;
          * neither → ``"no_action_available"``.

        Verifies the result by re-fetching the registrant list and
        checking the member's name appears in the expected place.
        """
        assert self._page is not None
        self.ensure_logged_in()
        me = self.get_member_full_name().strip().lower()

        reg = self.get_event_registrants(reservation_number)
        if any(r.name.strip().lower() == me for r in reg.registrants):
            return {
                "status": "already_registered",
                "name": self._member_full_name,
                "reservation_number": reservation_number,
            }
        if any(r.name.strip().lower() == me for r in reg.waitlist):
            return {
                "status": "already_waitlisted",
                "name": self._member_full_name,
                "reservation_number": reservation_number,
            }

        page = self._page
        page.goto(reg.detail_url, wait_until="networkidle", timeout=DEFAULT_TIMEOUT_MS)
        # Let any client-side rendering of action buttons settle.
        page.wait_for_timeout(2000)

        # Find a modal-launching action button. Inspect data-href to
        # classify: SignUpToWaitingList = waitlist, anything else = direct
        # register/reserve.
        modal_btns = page.locator("button.btn-modal[data-href]").all()
        register_btn = None
        register_href: str | None = None
        waitlist_btn = None
        waitlist_href: str | None = None
        for b in modal_btns:
            try:
                href = b.get_attribute("data-href") or ""
            except Exception:
                continue
            if "SignUpToWaitingList" in href:
                waitlist_btn = b
                waitlist_href = href
            elif "WaitingListPullOut" in href:
                # Already on waitlist (rare — we'd have caught it above,
                # but covers a stale-cache case).
                return {
                    "status": "already_waitlisted",
                    "name": self._member_full_name,
                    "reservation_number": reservation_number,
                }
            elif any(k in href for k in ("Reserve", "Register", "SignUp")):
                # Catch the various direct-register endpoints.
                register_btn = b
                register_href = href

        if register_btn is None and waitlist_btn is None:
            return {
                "status": "no_action_available",
                "name": self._member_full_name,
                "reservation_number": reservation_number,
                "message": "No Register / Join Waitlist button on the event page.",
            }

        if register_btn is None and not allow_waitlist_fallback:
            return {
                "status": "event_full_no_fallback",
                "name": self._member_full_name,
                "reservation_number": reservation_number,
            }

        target_href = register_href or waitlist_href
        target_status = "registered" if register_btn is not None else "waitlisted"
        assert target_href is not None
        modal_url = (
            target_href
            if target_href.startswith("http")
            else "https://app.courtreserve.com" + target_href
        )
        page.goto(modal_url, wait_until="networkidle", timeout=DEFAULT_TIMEOUT_MS)
        submit = page.locator("button.btn-submit").first
        if submit.count() == 0:
            return {
                "status": "modal_submit_not_found",
                "name": self._member_full_name,
                "reservation_number": reservation_number,
                "modal_url": modal_url,
            }
        with page.expect_navigation(wait_until="networkidle", timeout=DEFAULT_TIMEOUT_MS):
            submit.click()

        # Verify by re-reading the registrant list.
        reg2 = self.get_event_registrants(reservation_number)
        on_register = any(r.name.strip().lower() == me for r in reg2.registrants)
        on_waitlist = any(r.name.strip().lower() == me for r in reg2.waitlist)
        if target_status == "registered" and on_register:
            return {"status": "registered", "name": self._member_full_name,
                    "reservation_number": reservation_number}
        if target_status == "waitlisted" and on_waitlist:
            position = next(
                (i + 1 for i, r in enumerate(reg2.waitlist)
                 if r.name.strip().lower() == me),
                None,
            )
            return {"status": "waitlisted", "name": self._member_full_name,
                    "reservation_number": reservation_number,
                    "waitlist_position": position}
        # Submit went through but the list didn't update — surface as failure.
        return {
            "status": "verification_failed",
            "name": self._member_full_name,
            "reservation_number": reservation_number,
            "attempted": target_status,
        }

    def list_my_bookings(self) -> list[dict]:
        """Events the logged-in member is registered for and waitlisted for.

        Each entry: ``{status, event_name, date_str, res_id, detail_url}``.
        ``status`` is ``"registered"`` or ``"waitlisted"``. Two pages are
        scraped (My Events + My Waitlisted Events), results combined.
        """
        assert self._page is not None
        self.ensure_logged_in()
        out: list[dict] = []
        for type_value, status in (
            (MY_BOOKINGS_TYPE_REGISTERED, "registered"),
            (MY_BOOKINGS_TYPE_WAITLISTED, "waitlisted"),
        ):
            self._page.goto(
                MY_BOOKINGS_URL_TMPL.format(type=type_value),
                wait_until="networkidle",
                timeout=DEFAULT_TIMEOUT_MS,
            )
            # Cards are rendered client-side; give them a moment.
            self._page.wait_for_timeout(3000)
            cards = self._page.locator('[data-testid="booking-card"]').all()
            for card in cards:
                # Event name is in a span styled with bold-ish weight 500.
                name_loc = card.locator('span[style*="font-weight: 500"]').first
                event_name = (
                    name_loc.inner_text().strip() if name_loc.count() else ""
                )
                # Date row has a dedicated test-id.
                date_loc = card.locator(
                    '[data-testid="row-date-and-times"]'
                ).first
                date_str = (
                    date_loc.inner_text().strip() if date_loc.count() else ""
                )
                # Pull res_id from the Edit link / first details href.
                detail_link = card.locator('a[href*="resId="]').first
                href = (
                    detail_link.get_attribute("href") if detail_link.count() else ""
                ) or ""
                m = re.search(r"resId=(\d+)", href)
                res_id = m.group(1) if m else ""
                out.append({
                    "status": status,
                    "event_name": event_name,
                    "date_str": date_str,
                    "res_id": res_id,
                    "detail_url": (
                        ("https://app.courtreserve.com" + href)
                        if href.startswith("/")
                        else href
                    ),
                })
        return out

    def cancel_event_registration(
        self, reservation_number_or_res_id: str
    ) -> dict:
        """Remove the logged-in member from an event (registered or waitlisted).

        Accepts either the alphanumeric ``reservation_number`` (e.g.
        ``"AEK3RNY2146914"``) or the numeric ``res_id`` from the booking
        cards (e.g. ``"52824526"``). Idempotent — returns
        ``status="not_registered"`` if you weren't on the event in the
        first place.

        Other statuses: ``"cancelled"``, ``"no_action_available"``,
        ``"modal_submit_not_found"``, ``"verification_failed"``.
        """
        assert self._page is not None
        self.ensure_logged_in()
        me = self.get_member_full_name().strip().lower()
        ident = reservation_number_or_res_id.strip()
        is_res_id = ident.isdigit()

        # Are they actually on the event right now?
        if is_res_id:
            url = DETAIL_BY_RESID_TMPL.format(res_id=ident)
            reg = self.get_event_registrants(url)
        else:
            reg = self.get_event_registrants(ident)
        on_register = any(r.name.strip().lower() == me for r in reg.registrants)
        on_waitlist = any(r.name.strip().lower() == me for r in reg.waitlist)
        if not on_register and not on_waitlist:
            return {
                "status": "not_registered",
                "name": self._member_full_name,
                "ident": ident,
            }

        page = self._page
        page.goto(reg.detail_url, wait_until="networkidle", timeout=DEFAULT_TIMEOUT_MS)
        page.wait_for_timeout(2000)

        # Find the cancel/unsubscribe button — modal endpoint contains
        # "PullOut" (waitlist) or "Cancel" / "Unregister" (registered).
        btns = page.locator("button.btn-modal[data-href]").all()
        cancel_btn = None
        cancel_href: str | None = None
        for b in btns:
            href = (b.get_attribute("data-href") or "")
            if any(k in href for k in (
                "WaitingListPullOut", "CancelRegistration",
                "UnregisterFromEvent", "Unregister", "Cancel",
            )):
                cancel_btn = b
                cancel_href = href
                break
        if cancel_btn is None or cancel_href is None:
            return {
                "status": "no_action_available",
                "name": self._member_full_name,
                "ident": ident,
                "message": "No cancel/unsubscribe button found on the event page.",
            }

        modal_url = (
            cancel_href
            if cancel_href.startswith("http")
            else "https://app.courtreserve.com" + cancel_href
        )
        page.goto(modal_url, wait_until="networkidle", timeout=DEFAULT_TIMEOUT_MS)
        submit = page.locator("button.btn-submit").first
        if submit.count() == 0:
            return {
                "status": "modal_submit_not_found",
                "name": self._member_full_name,
                "ident": ident,
                "modal_url": modal_url,
            }
        with page.expect_navigation(wait_until="networkidle", timeout=DEFAULT_TIMEOUT_MS):
            submit.click()

        # Verify by re-fetching registrants.
        if is_res_id:
            reg2 = self.get_event_registrants(
                DETAIL_BY_RESID_TMPL.format(res_id=ident)
            )
        else:
            reg2 = self.get_event_registrants(ident)
        still_on_register = any(r.name.strip().lower() == me for r in reg2.registrants)
        still_on_waitlist = any(r.name.strip().lower() == me for r in reg2.waitlist)
        if not still_on_register and not still_on_waitlist:
            return {
                "status": "cancelled",
                "name": self._member_full_name,
                "ident": ident,
                "previously": "registered" if on_register else "waitlisted",
                "event_name": reg.event_name,
                "date_str": reg.date_str,
            }
        return {
            "status": "verification_failed",
            "name": self._member_full_name,
            "ident": ident,
        }


    # --- court bookings ---------------------------------------------------

    def _open_court_scheduler(self, sid: str, target_date) -> None:
        """Navigate to the court-booking page for ``sid`` and pick the
        target date by driving the Kendo scheduler widget directly.

        The previous approach (click ``.k-nav-current`` → set
        ``kendoCalendar.value()`` in the popup) stopped working after a
        CR UI update — the calendar's value change no longer cascaded
        into the scheduler view, so the page silently stayed on
        today's date. We now call ``scheduler.date(new Date(...))``
        which the scheduler handles natively, and verify the
        navigation actually took effect by reading the rendered
        header date back out.
        """
        page = self._page
        assert page is not None
        page.goto(
            COURT_BOOKINGS_URL_TMPL.format(sid=sid),
            wait_until="networkidle",
            timeout=DEFAULT_TIMEOUT_MS,
        )
        page.wait_for_timeout(2500)

        ymd = (target_date.year, target_date.month - 1, target_date.day)
        nav_result = page.evaluate(
            f"""() => {{
                const el = document.querySelector('[data-role=scheduler]');
                if (!el) return {{ ok: false, reason: 'no_scheduler_element' }};
                const w = jQuery(el).data('kendoScheduler');
                if (!w) return {{ ok: false, reason: 'no_scheduler_widget' }};
                w.date(new Date({ymd[0]}, {ymd[1]}, {ymd[2]}));
                return {{ ok: true, date: w.date().toString() }};
            }}"""
        )
        if not nav_result or not nav_result.get("ok"):
            reason = (nav_result or {}).get("reason", "unknown")
            raise RuntimeError(
                f"Failed to navigate court scheduler (sid={sid}) to "
                f"{target_date}: {reason!r}"
            )

        # ACTIVELY WAIT for the day-grid to re-render with the target
        # date. scheduler.date() updates the widget instantly, but the
        # kendoScheduler then re-renders its button grid asynchronously.
        # During that transient, the DOM still has the PREVIOUS day's
        # buttons — which look fine on a quick count but mean any
        # slot lookup picks the wrong day's availability. Check that
        # at least one button[start] has a Date attribute matching the
        # target year/month/day, not just that "some buttons exist".
        # This was the root cause of 2026-05-11 and 2026-05-12 fires:
        # the grid existed but with today's slots instead of next
        # week's, and we just happened not to find a 13:00 match.
        wait_start = _t.perf_counter()
        deadline = wait_start + 20.0
        target_y, target_m, target_d = (
            target_date.year, target_date.month, target_date.day
        )
        last_total = 0
        last_matching = 0
        while _t.perf_counter() < deadline:
            try:
                state = page.evaluate(
                    """([y, m, d]) => {
                        const btns = Array.from(document.querySelectorAll('button[start]'));
                        let matching = 0;
                        for (const b of btns) {
                            const t = new Date(b.getAttribute('start'));
                            if (t && t.getFullYear() === y &&
                                (t.getMonth() + 1) === m && t.getDate() === d) {
                                matching++;
                            }
                        }
                        return { total: btns.length, matching };
                    }""",
                    [target_y, target_m, target_d],
                )
                last_total = state.get("total", 0)
                last_matching = state.get("matching", 0)
            except Exception:
                pass
            if last_matching > 0:
                break
            page.wait_for_timeout(500)
        wait_secs = _t.perf_counter() - wait_start

        try:
            header = (
                page.locator(".k-nav-current").first.text_content() or ""
            )
        except Exception:
            header = ""

        if last_matching == 0:
            # Grid never re-rendered to the target date OR never
            # populated. Capture a screenshot for forensics before
            # raising, so the next diagnostic session can see what
            # CR actually returned.
            shot_path = (
                ROOT / "output_files"
                / f"cr-empty-grid-{datetime.now():%Y%m%d-%H%M%S}-sid{sid}.png"
            )
            try:
                shot_path.parent.mkdir(exist_ok=True)
                page.screenshot(path=str(shot_path), full_page=True)
            except Exception as e:
                print(f"[cr/nav] screenshot save failed: {e!r}")
                shot_path = None
            raise RuntimeError(
                f"Court scheduler never re-rendered to {target_date} "
                f"(sid={sid}, header={header!r}, waited {wait_secs:.1f}s, "
                f"button[start]={last_total} but 0 had the target date). "
                f"Screenshot: {shot_path}"
            )

        print(
            f"[cr/nav] sid={sid} -> {target_date}: header={header!r}, "
            f"button[start]={last_total} ({last_matching} for target date, "
            f"waited {wait_secs:.1f}s)"
        )

        expected_day = str(target_date.day)
        expected_year = str(target_date.year)
        if expected_day not in header or expected_year not in header:
            print(
                f"[cr/nav] WARNING: scheduler header {header!r} doesn't "
                f"contain day {expected_day} / year {expected_year} — "
                f"scheduler.date() may not have cascaded to the view. "
                f"Continuing anyway, but slot lookup may miss."
            )

    def _find_slot_button(
        self, court_label_full: str, start_iso_local: str
    ):
        """Return the slot button for the given court+start, or None."""
        # ``start_iso_local`` is "HH:MM:SS" (24h) — we match against the
        # button's `start` attribute by substring.
        page = self._page
        assert page is not None
        btns = page.locator(
            f'button[start][courtlabel="{court_label_full}"]'
        ).all()
        for b in btns:
            if start_iso_local in (b.get_attribute("start") or ""):
                return b
        return None

    def _lookup_just_booked_reservation_id(
        self,
        court_label_full: str | None,
        start_time_hhmm: str | None,
    ) -> str | None:
        """After a successful submit, scan the current scheduler page's
        Kendo dataSource for the reservation we just created and return
        its numeric ``ReservationId`` as a string.

        Matching: same ``CourtLabel`` AND start time. We're already on
        the correct date's grid (the prep + submit just happened
        there), so a date prefix on the start ISO isn't strictly
        needed — but we still convert the local-time HH:MM into the
        UTC ``HH:MM`` prefix that Kendo emits so we don't match an
        unrelated reservation on the same court at a different hour.
        Returns ``None`` (best-effort) on any mismatch; the booking
        already succeeded so a missing id isn't fatal.
        """
        if not court_label_full or not start_time_hhmm:
            return None
        try:
            self._page.evaluate(
                """() => {
                    try {
                        const w = jQuery('#CourtsScheduler')
                            .data('kendoScheduler');
                        if (w && w.dataSource) w.dataSource.read();
                    } catch (err) {}
                }"""
            )
            self._page.wait_for_timeout(800)
            # The grid header carries the date the user navigated to —
            # cheap to read for the UTC-conversion. Falls back to today
            # if not present (the date prefix is just a tighter filter).
            reservations = self._read_scheduled_reservations()
            try:
                hh, mm = (int(x) for x in start_time_hhmm.split(":"))
            except (ValueError, AttributeError):
                return None
            # Determine the play date from the first reservation's
            # start, since they're all on the same day on this grid.
            date_str = ""
            for r in reservations:
                s = r.get("start") or ""
                if s:
                    date_str = s[:10]
                    break
            if not date_str:
                return None
            local_start = datetime.strptime(
                f"{date_str} {start_time_hhmm}", "%Y-%m-%d %H:%M"
            ).replace(tzinfo=_LOCAL_TZ)
            utc_prefix = local_start.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M"
            )
            for r in reservations:
                if r.get("court_label") != court_label_full:
                    continue
                if not (r.get("start") or "").startswith(utc_prefix):
                    continue
                rid = r.get("reservation_id")
                if rid:
                    return str(rid)
            return None
        except Exception as e:
            print(f"[cr/submit] reservation-id lookup failed: {e!r}")
            return None

    def _read_scheduled_reservations(self, target_date=None) -> list[dict]:
        """Read all reservations on the currently-loaded scheduler
        page via Kendo's dataSource API. Returns a list of dicts:
        ``court_label``, ``court_number``, ``start`` (ISO UTC),
        ``end`` (ISO UTC), ``title``, ``reservation_id``, ``event_id``.

        When ``target_date`` is provided (a ``date`` object), the
        reservation dataSource is force-refreshed and we wait briefly
        for events whose start date matches the target before reading.
        Without this, the dataSource can stay stale after a date-nav
        on the 2nd-and-later scheduler page in a session (the slot
        button grid re-renders but the event data isn't always refetched
        by Kendo on its own).

        Returns ``[]`` if the widget isn't reachable. Callers should
        be prepared to fall back to "try every court" in that case.
        """
        assert self._page is not None
        if target_date is not None:
            # Trigger a dataSource refresh; poll until events for the
            # target date appear or a short deadline elapses. Catches
            # the "stale Kendo dataSource on 2nd nav" failure mode.
            self._page.evaluate(
                """() => {
                    try {
                        const w = jQuery('#CourtsScheduler')
                            .data('kendoScheduler');
                        if (w && w.dataSource) w.dataSource.read();
                    } catch (err) {}
                }"""
            )
            deadline = _t.perf_counter() + 4.0
            target_iso = target_date.isoformat()
            while _t.perf_counter() < deadline:
                got = self._page.evaluate(
                    """(targetIso) => {
                        try {
                            const w = jQuery('#CourtsScheduler')
                                .data('kendoScheduler');
                            if (!w || !w.dataSource) return false;
                            return w.dataSource.data().some(e => {
                                if (!e.start || !e.start.toISOString) return false;
                                return e.start.toISOString().startsWith(targetIso);
                            });
                        } catch (err) { return false; }
                    }""",
                    target_iso,
                )
                if got:
                    break
                self._page.wait_for_timeout(250)
        raw = self._page.evaluate(
            """() => {
                try {
                    const $w = window.jQuery && jQuery('#CourtsScheduler');
                    if (!$w || !$w.length) return null;
                    const w = $w.data('kendoScheduler');
                    if (!w || !w.dataSource) return null;
                    return w.dataSource.data().map(e => {
                        // MemberIds is a Kendo ObservableArray —
                        // flatten to a plain integer list so it survives
                        // serialisation across the Playwright bridge.
                        const mids = [];
                        if (e.MemberIds && e.MemberIds.length != null) {
                            for (let i = 0; i < e.MemberIds.length; i++) {
                                const v = e.MemberIds[i];
                                if (v != null) mids.push(v);
                            }
                        }
                        return {
                            court_label: e.CourtLabel || '',
                            start: e.start && e.start.toISOString
                                    ? e.start.toISOString() : null,
                            end: e.end && e.end.toISOString
                                    ? e.end.toISOString() : null,
                            title: e.title || '',
                            reservation_id: e.ReservationId || 0,
                            event_id: e.EventId || null,
                            user_id: e.UserId || null,
                            member_ids: mids,
                        };
                    });
                } catch (err) {
                    return null;
                }
            }"""
        )
        if not raw:
            return []
        out: list[dict] = []
        for r in raw:
            cl = (r.get("court_label") or "")
            m = re.match(r"Court #(\d+)", cl)
            r["court_number"] = m.group(1) if m else ""
            out.append(r)
        return out

    def _prepare_one_court(
        self,
        court_number: str,
        start_time_hhmm: str,
        partner_name: str,
        duration_minutes: int,
    ) -> dict | None:
        """Open the booking modal for one court and fill EVERY field
        except the final submit click. Assumes the right scheduler
        page + date are already loaded.

        Returns:
          * ``None`` — slot not visible / unavailable. Caller may try
            the next court.
          * ``{"status": "ready", "_handle_state": {...}}`` — the modal
            is open and form-filled; caller must click submit (or
            abandon).
          * ``{"status": "..."}`` — non-fatal error (partner not
            found, unsupported duration etc). Modal already closed by
            this method so the next attempt is clean.
        """
        page = self._page
        assert page is not None
        t0 = _t.perf_counter()
        court_label_full = f"Court #{court_number} - Floodlit"
        print(
            f"[cr/prep] trying court {court_number} ({court_label_full}) "
            f"at {start_time_hhmm} for {duration_minutes}min ..."
        )
        slot = self._find_slot_button(
            court_label_full, f"{start_time_hhmm}:00"
        )
        if slot is None:
            # Diagnose: how many buttons does the page have, and what's
            # in their start attributes? Lets us tell "page didn't load"
            # apart from "this court has no slots at this time" apart
            # from "selector / attribute changed under us". The label
            # is passed as a positional arg to evaluate() so Python
            # repr doesn't conflict with the JS string quoting.
            try:
                total = page.locator("button[start]").count()
                this_court = page.locator(
                    f'button[start][courtlabel="{court_label_full}"]'
                ).count()
                sample_starts = page.evaluate(
                    "(label) => Array.from(document.querySelectorAll("
                    "'button[start][courtlabel=\"' + label + '\"]'"
                    ")).slice(0, 6).map(b => b.getAttribute('start'))",
                    court_label_full,
                )
            except Exception as e:
                total = this_court = -1
                sample_starts = [f"diag-error:{e!r}"]
            print(
                f"[cr/prep] court {court_number}: slot button not found "
                f"(page has {total} button[start] total, "
                f"{this_court} for {court_label_full!r}; "
                f"available starts on this court: {sample_starts})"
            )
            return None  # slot unavailable
        print(
            f"[cr/prep] court {court_number}: slot found, clicking "
            f"(t+{_t.perf_counter() - t0:.2f}s)"
        )
        slot.evaluate("el => el.click()")
        page.wait_for_timeout(5000)

        target_res_type = DURATION_TO_RES_TYPE.get(duration_minutes)
        if target_res_type is None:
            page.keyboard.press("Escape")
            return {
                "status": "unsupported_duration",
                "court_label": court_number,
                "duration_minutes": duration_minutes,
            }
        set_id = page.evaluate(
            f"""() => {{
                const dd = jQuery('#ReservationTypeId').data('kendoDropDownList');
                if (!dd) return null;
                const item = dd.dataSource.data().find(x => x.Name === {target_res_type!r});
                if (!item) return null;
                dd.value(String(item.Id));
                dd.trigger('change');
                return item.Id;
            }}"""
        )
        if set_id is None:
            page.keyboard.press("Escape")
            return {
                "status": "reservation_type_unavailable",
                "court_label": court_number,
                "wanted_type": target_res_type,
            }
        print(
            f"[cr/prep] court {court_number}: reservation type set "
            f"({target_res_type!r}, id={set_id}, "
            f"t+{_t.perf_counter() - t0:.2f}s)"
        )
        page.wait_for_timeout(2000)

        # Partner pick
        page.locator('input[name="OwnersDropdown_input"]').first.click()
        page.keyboard.type(partner_name, delay=80)
        page.wait_for_timeout(2500)
        match = page.locator(
            f'#OwnersDropdown_listbox li:has-text({partner_name!r})'
        ).first
        if match.count() == 0:
            page.keyboard.press("Escape")
            page.wait_for_timeout(1000)
            return {
                "status": "partner_not_found",
                "court_label": court_number,
                "partner_name": partner_name,
            }
        match.click()
        page.wait_for_timeout(2000)

        members = page.evaluate(
            """() => Array.from(document.querySelectorAll(
                'input[name^="SelectedMembers"][name$=".FirstName"]'
            )).map(i => i.value)"""
        )
        if len(members) < 2:
            page.keyboard.press("Escape")
            return {
                "status": "partner_not_added",
                "court_label": court_number,
            }
        print(
            f"[cr/prep] court {court_number}: partner '{partner_name}' added "
            f"({len(members)} members, t+{_t.perf_counter() - t0:.2f}s)"
        )

        return {
            "status": "ready",
            "court_label": court_number,
            "court_label_full": court_label_full,
            "duration_minutes": duration_minutes,
            "partner_name": partner_name,
            "start_time_hhmm": start_time_hhmm,
        }

    def _finalise_booking(self, prep: dict) -> dict:
        """Click the submit button on a prepared modal and wait for the
        outcome. ``prep`` must be the dict returned by
        ``_prepare_one_court`` with status='ready'.

        Detects 'too early' / validation errors and retries the click
        for a few seconds when the failure looks transient.
        """
        page = self._page
        assert page is not None
        court_number = prep["court_label"]

        # Snapshot the page text BEFORE clicking so we can diff what
        # appears (error banners, validation messages) for diagnosis.
        before_text = ""
        try:
            before_text = page.locator("body").inner_text(timeout=2000)
        except Exception:
            pass

        # Tight retry on transient post-submit errors. Each click +
        # response cycle is ~1-2s; retry up to ~10s past the moment we
        # first try. CourtReserve seems to operate to the second so the
        # window-opens-at delta with our local clock is what matters.
        deadline = _t.perf_counter() + 12.0
        attempt = 0
        last_err: dict | None = None
        while _t.perf_counter() < deadline:
            attempt += 1
            click_ts = datetime.now().isoformat(timespec="milliseconds")
            print(
                f"[cr/submit] court {court_number}: clicking submit "
                f"(attempt {attempt}, {click_ts})"
            )
            try:
                # Short-timeout click so a SwAl2 overlay intercepting
                # pointer events doesn't burn the whole retry budget
                # on a single attempt. The default 30s would otherwise
                # wedge us against the overlay's pointer-blocking divs.
                page.locator("button.btn-submit").first.click(timeout=3000)
            except Exception as e:
                # Modal might have closed unexpectedly, OR a SwAl2
                # overlay is intercepting clicks (server-side rejection
                # popup). Fall through to the post-click checks so the
                # SwAl2 handler below can read its text and dismiss it.
                click_err = repr(e)
                print(
                    f"[cr/submit] court {court_number}: submit click "
                    f"intercepted/failed on attempt {attempt}: "
                    f"{click_err[:200]}"
                )
            page.wait_for_timeout(2500)

            # CR sometimes responds with a SweetAlert2 modal instead
            # of an inline error banner (e.g. "Reservation Notice /
            # Sorry, no available courts for the time requested"). It
            # overlays the booking modal and intercepts pointer events,
            # so left undetected it wedges the retry loop until the
            # 12s deadline. Read it first, dismiss it, and surface a
            # clean status.
            swal_text = ""
            try:
                swal = page.locator(
                    "div.swal2-container div.swal2-html-container"
                ).first
                if swal.is_visible(timeout=400):
                    swal_text = swal.inner_text(timeout=800).strip()
            except Exception:
                swal_text = ""
            if swal_text:
                swal_lower = swal_text.lower()
                print(
                    f"[cr/submit] court {court_number}: SwAl2 popup "
                    f"intercepted submit (attempt {attempt}): "
                    f"{swal_text[:200]!r}"
                )
                # Click the SwAl2 confirm button so the overlay clears
                # and the underlying booking modal becomes reachable
                # again (for a clean Escape or a subsequent retry).
                try:
                    page.locator(
                        "div.swal2-container button.swal2-confirm"
                    ).first.click(timeout=2000)
                    page.wait_for_timeout(400)
                except Exception:
                    pass
                court_taken = (
                    "no available" in swal_lower
                    or "no court" in swal_lower
                    or "not available" in swal_lower
                    or "already booked" in swal_lower
                )
                # Try to close the now-unblocked booking modal so the
                # browser context is reusable for the next attempt.
                try:
                    page.keyboard.press("Escape")
                    page.wait_for_timeout(300)
                except Exception:
                    pass
                return {
                    "ok": False,
                    "status": "submit_court_taken" if court_taken else "submit_rejected",
                    "court_label": court_number,
                    "court_label_full": prep["court_label_full"],
                    "duration_minutes": prep["duration_minutes"],
                    "submit_attempts": attempt,
                    "last_error": {
                        "swal_text": swal_text[:500],
                        "attempt": attempt,
                    },
                }

            # Read the page state. If the modal has closed AND no
            # error banner is visible, we treat it as booked — the
            # caller's downstream verification (or list_my_bookings)
            # can confirm.
            modal_open = False
            try:
                modal_open = page.locator(
                    "div.modal.in, div.modal.show"
                ).first.is_visible(timeout=500)
            except Exception:
                pass

            error_text = ""
            try:
                # Common CR error spots: validation banner inside the
                # modal, or a top-of-page alert. Capture any visible
                # error-flavoured text for diagnostics.
                error_text = page.locator(
                    ".validation-summary-errors, .field-validation-error, "
                    ".alert-danger, .toast-error"
                ).first.inner_text(timeout=800)
            except Exception:
                error_text = ""

            if not modal_open and not error_text.strip():
                page.wait_for_timeout(2500)  # let the page settle
                print(
                    f"[cr/submit] court {court_number}: modal closed without "
                    f"error — treating as booked (attempt {attempt})"
                )
                # Try to capture the new reservation's id from the Kendo
                # dataSource. The page is still on the booking grid for
                # the target date so any new k-event matching our
                # court_label + start_time is the one we just created.
                reservation_id = self._lookup_just_booked_reservation_id(
                    prep.get("court_label_full"),
                    prep.get("start_time_hhmm"),
                )
                return {
                    "ok": True,
                    "status": "booked",
                    "court_label": court_number,
                    "court_label_full": prep["court_label_full"],
                    "duration_minutes": prep["duration_minutes"],
                    "submit_attempt": attempt,
                    "reservation_id": reservation_id,
                }

            err_lower = (error_text or "").lower()
            transient = (
                "too early" in err_lower
                or "not yet" in err_lower
                or "not open" in err_lower
                or "available at" in err_lower
                or (modal_open and not error_text.strip())
            )
            print(
                f"[cr/submit] court {court_number}: submit attempt {attempt} "
                f"did not complete — modal_open={modal_open!r}, "
                f"error={error_text.strip()[:200]!r}, "
                f"transient={transient}"
            )
            last_err = {
                "modal_open": modal_open,
                "error_text": error_text.strip()[:500],
                "attempt": attempt,
            }
            if not transient:
                # Non-recoverable validation failure. Stop.
                break
            # Wait a beat, then retry — we may be racing the window
            # opening on the server side.
            page.wait_for_timeout(700)

        # Capture a chunk of the body text on the way out for forensic
        # diagnosis of unexpected page states.
        try:
            after_text = page.locator("body").inner_text(timeout=2000)
        except Exception:
            after_text = ""
        diff_excerpt = after_text[: 800]

        # Try to close cleanly so the browser context is reusable.
        try:
            page.keyboard.press("Escape")
        except Exception:
            pass

        return {
            "ok": False,
            "status": "submit_rejected",
            "court_label": court_number,
            "court_label_full": prep["court_label_full"],
            "duration_minutes": prep["duration_minutes"],
            "submit_attempts": attempt,
            "last_error": last_err,
            "page_excerpt": diff_excerpt,
        }

    def _book_one_court(
        self,
        court_number: str,
        start_time_hhmm: str,
        partner_name: str,
        duration_minutes: int,
    ) -> dict | None:
        """Legacy single-shot path: prep + finalise back-to-back.

        Returns:
          * ``None`` — slot unavailable (try next court).
          * ``{ok, status, ...}`` — final outcome.
        """
        prep = self._prepare_one_court(
            court_number, start_time_hhmm, partner_name, duration_minutes
        )
        if prep is None:
            return None
        if prep.get("status") != "ready":
            return prep  # non-fatal error; caller decides whether to keep iterating
        return self._finalise_booking(prep)

    def prepare_court_booking(
        self,
        date: str,
        start_time_hhmm: str,
        partner_name: str,
        *,
        duration_minutes: int = 90,
        court_label: str | None = None,
        court_type: str | None = None,
        court_preference: list[str] | None = None,
    ) -> "PreparedBooking | dict":
        """Drive the booking flow up to the 'modal filled, ready to
        submit' state and return a ``PreparedBooking`` whose
        ``submit()`` can be called later.

        Used by the scheduler to pre-warm Chromium, log in, and
        navigate to the booking form BEFORE the CR window opens, so
        the final submit click can fire the moment the window is
        live. Iterates the candidate court list the same way
        ``book_court`` does — the first court that gets to 'ready'
        wins.

        Returns ``PreparedBooking`` on success. Returns a result dict
        (``{ok: False, status: ..., ...}``) when no court could be
        prepared. The dict shape mirrors what ``book_court`` returns
        for the same failure modes.
        """
        from datetime import date as _date_cls

        assert self._page is not None
        prep_t0 = _t.perf_counter()
        try:
            start_time_hhmm = normalize_hhmm(start_time_hhmm)
        except ValueError as exc:
            return {
                "ok": False,
                "status": "invalid_start_time",
                "error": str(exc),
            }
        login_t0 = _t.perf_counter()
        self.ensure_logged_in()
        login_secs = _t.perf_counter() - login_t0
        target_date = _date_cls.fromisoformat(date)
        print(
            f"[cr/prep] login ok ({login_secs:.2f}s); preparing booking for "
            f"{date} {start_time_hhmm} {duration_minutes}min "
            f"partner={partner_name!r}"
        )

        candidates = self._build_court_candidates(
            court_label, court_type, court_preference
        )
        if isinstance(candidates, dict):
            return candidates  # error result

        attempted: list[str] = []
        skipped_prebooked: list[str] = []
        last_error: dict | None = None
        for sid in (SCHEDULER_ID_CLAY, SCHEDULER_ID_ACRYLIC):
            type_courts = [c for c in candidates
                           if COURT_NUMBER_TO_SCHEDULER_ID.get(c) == sid]
            if not type_courts:
                continue
            self._open_court_scheduler(sid, target_date)
            # Read what's already reserved on this scheduler so we
            # can skip courts whose target slot is blocked by a club
            # booking, court-maintenance closure, group lesson, etc.
            # Saves time at 08:00 (no wasted attempts on courts that
            # can't possibly take this booking).
            reservations = self._read_scheduled_reservations(
                target_date=target_date,
            )
            blocked = _compute_blocked_courts(
                reservations, date, start_time_hhmm, duration_minutes,
            )
            blocked_in_scope = sorted(
                (c for c in type_courts if c in blocked), key=int,
            )
            if blocked_in_scope:
                # Show a one-line summary of the overlapping reservations
                # so the log explains *why* each court was skipped.
                reasons = []
                for r in reservations:
                    if r.get("court_number") in blocked_in_scope:
                        title = (r.get("title") or "").strip() or "(no title)"
                        reasons.append(
                            f"{r['court_number']}={title[:40]}"
                        )
                print(
                    f"[cr/prep] sid={sid}: pre-booked at {start_time_hhmm} "
                    f"({duration_minutes}min) — skipping courts "
                    f"{blocked_in_scope} ({'; '.join(reasons[:8])})"
                )
                skipped_prebooked.extend(blocked_in_scope)
            type_courts = [c for c in type_courts if c not in blocked]
            if not type_courts:
                continue
            for idx, court_num in enumerate(type_courts):
                attempted.append(court_num)
                prep = self._prepare_one_court(
                    court_num, start_time_hhmm, partner_name, duration_minutes,
                )
                if prep is None:
                    continue  # slot not visible — try next court
                if prep.get("status") == "ready":
                    print(
                        f"[cr/prep] READY on court {court_num} "
                        f"(total prep time: {_t.perf_counter() - prep_t0:.2f}s)"
                    )
                    # Courts still untried on this sid — fed to
                    # ``PreparedBooking.submit`` so it can advance
                    # in-flight if CR rejects with "court taken".
                    remaining_same_sid = list(type_courts[idx + 1:])
                    return PreparedBooking(
                        client=self,
                        prep=prep,
                        attempted=list(attempted),
                        skipped_prebooked=list(skipped_prebooked),
                        start_time_hhmm=start_time_hhmm,
                        partner_name=partner_name,
                        duration_minutes=duration_minutes,
                        remaining_candidates=remaining_same_sid,
                    )
                # Non-fatal but not retryable across courts (e.g. partner
                # not found) — return the error.
                last_error = prep
                return {
                    "ok": False,
                    **prep,
                    "attempted": attempted,
                    "skipped_prebooked": skipped_prebooked,
                }

        skipped = [c for c in candidates if c not in COURT_NUMBER_TO_SCHEDULER_ID]
        return {
            "ok": False,
            "status": "no_court_available",
            "attempted": attempted,
            "skipped_prebooked": skipped_prebooked,
            "skipped_unmapped": skipped,
            **(last_error or {}),
        }

    def _build_court_candidates(
        self,
        court_label: str | None,
        court_type: str | None,
        court_preference: list[str] | None,
    ) -> list[str] | dict:
        """Resolve the {court_label, court_type, court_preference}
        triple into an ordered list of court numbers. Returns an error
        dict on bad input."""
        if court_label is not None:
            num = str(court_label).lstrip("#").strip()
            if num not in COURT_NUMBER_TO_SCHEDULER_ID:
                return {
                    "ok": False,
                    "status": "unknown_court",
                    "court_label": court_label,
                }
            return [num]
        if court_type is not None:
            ct = court_type.strip().lower()
            if ct not in COURT_TYPE_TO_NUMBERS:
                return {
                    "ok": False,
                    "status": "unknown_court_type",
                    "court_type": court_type,
                }
            return list(COURT_TYPE_TO_NUMBERS[ct])
        return list(court_preference or DEFAULT_COURT_PREFERENCE)

    def book_court(
        self,
        date: str,
        start_time_hhmm: str,
        partner_name: str,
        *,
        duration_minutes: int = 90,
        court_label: str | None = None,
        court_type: str | None = None,
        court_preference: list[str] | None = None,
    ) -> dict:
        """Book a court immediately — single-shot prep + submit.

        ``date`` is ISO ``YYYY-MM-DD``. ``start_time_hhmm`` is 24h ``HH:MM``.
        Court selection: explicit ``court_label`` wins; else ``court_type``
        narrows to clay/acrylic; else iterates ``court_preference``
        (default: club preference list).

        For deferred submission (e.g. waiting for a booking window to
        open) call ``prepare_court_booking`` then ``submit`` on the
        returned ``PreparedBooking`` instead.
        """
        prepared = self.prepare_court_booking(
            date, start_time_hhmm, partner_name,
            duration_minutes=duration_minutes,
            court_label=court_label,
            court_type=court_type,
            court_preference=court_preference,
        )
        if isinstance(prepared, dict):
            return prepared  # error during prep
        return prepared.submit()

    def find_my_court_booking(
        self, date: str, start_time_hhmm: str
    ) -> dict | None:
        """Find the bot's ad-hoc court booking on ``date`` starting at
        ``start_time_hhmm`` (local time).

        Returns ``{"reservation_id", "court_label", "sid", "summary"}``
        or ``None`` if no matching booking is found.

        Uses CR's Kendo scheduler dataSource (the same source the
        booking grid is rendered from) rather than scraping event
        innerText. A reservation is "the bot's" when:
          * its start time matches the target (local-time → UTC), AND
          * ``EventId`` is null (excludes club events / socials), AND
          * the logged-in user's ``UserId`` appears in ``MemberIds``
            (or matches the event's lone ``UserId`` for older shapes).

        The text-scraping approach this replaces never worked for
        ad-hoc bookings because CR renders them with empty
        ``innerText`` — the booker/partner names are on a tooltip
        that only appears on hover.
        """
        from datetime import date as _date_cls, time as _time_cls

        assert self._page is not None
        start_time_hhmm = normalize_hhmm(start_time_hhmm)
        self.ensure_logged_in()
        target_date = _date_cls.fromisoformat(date)
        hh, mm = (int(x) for x in start_time_hhmm.split(":"))
        target_start_local = datetime.combine(
            target_date, _time_cls(hh, mm), tzinfo=_LOCAL_TZ,
        )
        target_start_utc = target_start_local.astimezone(timezone.utc)
        target_iso_prefix = target_start_utc.strftime("%Y-%m-%dT%H:%M")
        my_uid_str = self.get_current_user_id()
        try:
            my_uid = int(my_uid_str) if my_uid_str else None
        except (TypeError, ValueError):
            my_uid = None
        if my_uid is None:
            print(
                "[cr/find] warning: could not resolve current user id; "
                "find_my_court_booking will return None"
            )
            return None
        for sid in (SCHEDULER_ID_CLAY, SCHEDULER_ID_ACRYLIC):
            self._open_court_scheduler(sid, target_date)
            reservations = self._read_scheduled_reservations(
                target_date=target_date,
            )
            for r in reservations:
                if r.get("event_id") is not None:
                    continue  # club event, not ad-hoc
                start_iso = r.get("start") or ""
                if not start_iso.startswith(target_iso_prefix):
                    continue
                owner_ids = set(r.get("member_ids") or [])
                if r.get("user_id") is not None:
                    owner_ids.add(int(r["user_id"]))
                if my_uid not in owner_ids:
                    continue
                rid = r.get("reservation_id")
                if not rid:
                    continue
                return {
                    "reservation_id": str(rid),
                    "court_label": r.get("court_label", ""),
                    "sid": sid,
                    "summary": (
                        f"{r.get('court_label', '?')} "
                        f"{target_start_local.strftime('%Y-%m-%d %H:%M')} "
                        f"(reservation_id={rid})"
                    ),
                }
        return None

    def cancel_court_reservation(self, reservation_id: str) -> dict:
        """Cancel a court reservation by its numeric reservation_id."""
        assert self._page is not None
        self.ensure_logged_in()
        page = self._page
        page.goto(
            COURT_CANCEL_URL_TMPL.format(reservation_id=reservation_id),
            wait_until="networkidle",
            timeout=DEFAULT_TIMEOUT_MS,
        )
        page.wait_for_timeout(2500)
        submit = page.locator(
            'form[action*="CancelReservation"] button[type="submit"]'
        ).first
        if submit.count() == 0:
            return {
                "ok": False,
                "status": "modal_submit_not_found",
                "reservation_id": reservation_id,
            }
        submit.click()
        page.wait_for_timeout(6000)
        return {
            "ok": True,
            "status": "cancelled",
            "reservation_id": reservation_id,
        }


# ---------- CLI smoke test ------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="CourtReserve CLI smoke test")
    parser.add_argument("--headed", action="store_true", help="Run visible browser")
    parser.add_argument("--category", default="Social & Club Sessions")
    parser.add_argument("--day", default=None, help="Day of week filter (e.g. Thursday)")
    parser.add_argument("--days-ahead", type=int, default=14)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--registrants-for",
        metavar="reservationNumber",
        help="Fetch registrants for the given reservation_number",
    )
    args = parser.parse_args()

    with CourtReserveClient(headless=not args.headed) as cr:
        if args.registrants_for:
            reg = cr.get_event_registrants(args.registrants_for)
            print(json.dumps({
                "event": reg.event_name,
                "category": reg.category,
                "date": reg.date_str,
                "spots_remaining": reg.spots_remaining,
                "courts": reg.courts,
                "age_restriction": reg.age_restriction,
                "registrants": [r.name for r in reg.registrants],
                "detail_url": reg.detail_url,
            }, indent=2, ensure_ascii=False))
        else:
            events = cr.list_events(
                category=args.category,
                day_of_week=args.day,
                days_ahead=args.days_ahead,
                name_contains=args.name,
            )
            print(f"{len(events)} events found:\n")
            for e in events:
                print(f"  {e.name:<45} {e.date_str:<40}  rn={e.reservation_number}")
