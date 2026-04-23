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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

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

    return EventRegistration(
        event_name=title,
        category=category,
        date_str=date_str,
        detail_url="",  # filled in by caller
        spots_remaining=spots,
        courts=courts,
        age_restriction=age,
        registrants=registrants,
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
