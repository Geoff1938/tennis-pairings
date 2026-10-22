"""CourtReserve reading: year-less dates across New Year, and unreadable
pages raising instead of looking like "nobody registered"."""
from __future__ import annotations

from datetime import datetime

import pytest

import courtreserve as cr


# ---------- year inference ----------------------------------------------


def test_january_event_seen_in_december_is_next_year():
    # 8 Jan 2027 is a Friday.
    dt = cr._parse_yearless_datetime(
        "Fri, Jan 8th", "19:30", now=datetime(2026, 12, 20, 9, 0))
    assert dt == datetime(2027, 1, 8, 19, 30)


def test_december_event_seen_in_january_is_last_year():
    # 31 Dec 2026 is a Thursday.
    dt = cr._parse_yearless_datetime(
        "Thu, Dec 31st", "19:30", now=datetime(2027, 1, 2, 9, 0))
    assert dt == datetime(2026, 12, 31, 19, 30)


def test_ordinary_date_stays_in_current_year():
    dt = cr._parse_yearless_datetime(
        "Thu, Oct 22nd", "19:30", now=datetime(2026, 10, 20))
    assert dt == datetime(2026, 10, 22, 19, 30)


def test_event_cards_use_the_inferred_year():
    html = (
        '<div class="fj_post style2 jlv5" data-testid="event-card">'
        '<h4 data-testid="event-name">Thursday Club Night</h4>'
        '<a data-testid="date-time-section">Thu, Jan 7th, 7:30 PM - 9:30 PM</a>'
        '<a href="/Online/Events/Details/2146/ABC?resId=123">x</a></div>'
    )
    events = cr._parse_event_cards(html, now=datetime(2026, 12, 28))
    assert events[0].start_dt.date() == datetime(2027, 1, 7).date()


# ---------- unreadable pages --------------------------------------------


class _Loc:
    def __init__(self, visible):
        self.visible = visible
        self.first = self

    def is_visible(self):
        return self.visible

    def count(self):
        return 1 if self.visible else 0


class _Page:
    def __init__(self, html, login_form=False):
        self.html = html
        self.login_form = login_form

    def goto(self, *a, **k):
        pass

    def content(self):
        return self.html

    def locator(self, sel):
        return _Loc(self.login_form and "password" in sel)

    def wait_for_selector(self, *a, **k):
        pass


def _client(page):
    c = object.__new__(cr.CourtReserveClient)
    c._page = page
    c.ensure_logged_in = lambda: None
    return c


def test_empty_events_page_raises():
    c = _client(_Page("<html><body>Something new</body></html>"))
    with pytest.raises(cr.CourtReservePageError):
        c.list_events()


def test_login_form_instead_of_event_page_raises():
    c = _client(_Page("<html>Log in</html>", login_form=True))
    with pytest.raises(cr.CourtReservePageError):
        c.get_event_registrants("https://app.courtreserve.com/x")


def test_cloudflare_instead_of_event_page_raises():
    c = _client(_Page(
        "<html>Attention Required | Cloudflare /cdn-cgi/ </html>"))
    with pytest.raises(cr.CloudflareBlocked):
        c.get_event_registrants("https://app.courtreserve.com/x")


def test_unrecognised_event_page_raises():
    c = _client(_Page("<html><body>new layout</body></html>"))
    with pytest.raises(cr.CourtReservePageError):
        c.get_event_registrants("https://app.courtreserve.com/x")


def test_event_with_title_and_no_registrants_is_fine():
    c = _client(_Page(
        '<h4 data-testid="event-name">Thursday Club Night</h4>'
        '<tbody data-testid="registrants-table-body"></tbody>'))
    reg = c.get_event_registrants("https://app.courtreserve.com/x")
    assert reg.event_name == "Thursday Club Night"
    assert reg.registrants == []
