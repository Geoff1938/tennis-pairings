"""CourtReserve reading: the events list and event detail JSON the
pages fetch, and unreadable pages raising instead of looking like
"nobody registered"."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime

import pytest

import courtreserve as cr


def _list_item(name="Thursday Evening Club Social for Intermediate+",
               start="2027-01-07 19:30:00", end="2027-01-07 21:30:00",
               rn="MUMFCQF2146684", res_id=57952826):
    return {
        "EventName": name,
        "CategoryName": "Social &amp; Club Sessions",
        "Next_EventStartDateTimeStringDisplay": start,
        "Next_EventEndDateTimeStringDisplay": end,
        "DateRangeDisplay": "Thu, Jan 7th 2027, 7:30p - 9:30p",
        "NextReservationNumber": rn,
        "NextReservationId": res_id,
    }


def _detail(**over):
    e = {
        "EventName": "Thursday Evening Club Social for Intermediate+",
        "EventType": "Social & Club Sessions",
        "StartStringDisplay": "2026-10-29 19:30:00",
        "EndStringDisplay": "2026-10-29 21:30:00",
        "MaxPlayers": 20,
        "PlacesAvailable": 0,
        "IsFull": True,
        "MinAgeRestriction": 14,
        "CourtDisplay": "Court #7 - Floodlit, Court #8 - Floodlit",
        "AllRegisteredMembers": [
            {"FullName": "Ann Smith", "IsCancelled": None},
            {"FullName": "Bob Jones", "IsCancelled": True},
            {"FullName": "Cy Brown"},
        ],
        "ReservationWaitListMembers": [
            {"FirstName": "Tim", "LastName": "Hodson"},
        ],
    }
    e.update(over)
    return {"Data": {"EventData": e}, "IsValid": True}


# ---------- events list -------------------------------------------------


def test_events_json_gives_old_style_date_and_detail_url():
    [e] = cr._parse_events_json({"Data": {"List": [_list_item()]}})
    assert e.name == "Thursday Evening Club Social for Intermediate+"
    assert e.category == "Social & Club Sessions"
    # Day-of-week filtering and SessionType.cr_time_fragment use this form.
    assert e.date_str == "Thu, Jan 7th, 19:30 - 21:30"
    assert e.start_dt == datetime(2027, 1, 7, 19, 30)
    assert e.reservation_number == "MUMFCQF2146684"
    assert e.res_id == "57952826"
    assert e.detail_url.endswith("/MUMFCQF2146684?resId=57952826")
    assert cr._match_day_of_week(e, "Thursday")


@pytest.mark.parametrize("day,expected", [
    (1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"),
    (12, "12th"), (13, "13th"), (21, "21st"), (22, "22nd"), (31, "31st"),
])
def test_ordinal(day, expected):
    assert cr._ordinal(day) == expected


def test_events_json_skips_items_without_a_reservation():
    item = _list_item()
    item["NextReservationNumber"] = None
    assert cr._parse_events_json({"Data": {"List": [item]}}) == []


def test_events_json_falls_back_to_display_text_without_a_start():
    [e] = cr._parse_events_json(
        {"Data": {"List": [_list_item(start=None)]}})
    assert e.start_dt is None
    assert e.date_str == "Thu, Jan 7th 2027, 7:30p - 9:30p"


# ---------- event detail ------------------------------------------------


def test_detail_json():
    reg = cr._parse_event_detail_json(_detail())
    assert reg.event_name == "Thursday Evening Club Social for Intermediate+"
    assert reg.category == "Social & Club Sessions"
    assert reg.date_str == "Thu, Oct 29th, 19:30 - 21:30"
    assert reg.spots_remaining == "0 of 20"
    assert reg.is_full
    assert reg.age_restriction == "Min Age 14"
    assert reg.courts == ["Court #7 - Floodlit", "Court #8 - Floodlit"]
    # Cancelled registrants are left out; waitlist rows have no FullName.
    assert [r.name for r in reg.registrants] == ["Ann Smith", "Cy Brown"]
    assert [r.name for r in reg.waitlist] == ["Tim Hodson"]


def test_detail_json_with_no_registrants_or_limits():
    reg = cr._parse_event_detail_json(_detail(
        AllRegisteredMembers=[], ReservationWaitListMembers=None,
        MaxPlayers=None, MinAgeRestriction=None, IsFull=False))
    assert reg.registrants == [] and reg.waitlist == []
    assert reg.spots_remaining is None
    assert reg.age_restriction is None
    assert not reg.is_full


# ---------- unreadable pages --------------------------------------------


class _Loc:
    def __init__(self, visible):
        self.visible = visible
        self.first = self

    def is_visible(self):
        return self.visible

    def count(self):
        return 1 if self.visible else 0


class _Resp:
    def __init__(self, data, status=200):
        self.data = data
        self.status = status
        self.ok = status < 400

    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data


class _Info:
    def __init__(self, value):
        self.value = value


class _Page:
    """Fake page: ``api`` is the JSON its data call returns, or None
    when the page never makes the call."""

    def __init__(self, html="", api=None, login_form=False, status=200):
        self.html = html
        self.api = api
        self.login_form = login_form
        self.status = status

    @contextmanager
    def expect_response(self, predicate, timeout=None):
        if self.api is None:
            raise cr.PlaywrightTimeout("no data call")
        yield _Info(_Resp(self.api, self.status))

    def goto(self, *a, **k):
        pass

    def content(self):
        return self.html

    def locator(self, sel):
        return _Loc(self.login_form and "password" in sel)


def _client(page):
    c = object.__new__(cr.CourtReserveClient)
    c._page = page
    c.ensure_logged_in = lambda: None
    return c


def test_events_list_reads_the_api_json():
    c = _client(_Page(api={"Data": {"List": [
        _list_item(start="2099-01-01 19:30:00", end="2099-01-01 21:30:00",
                   name="Later"),
        _list_item(start="2098-01-01 19:30:00", end="2098-01-01 21:30:00",
                   name="Sooner"),
    ]}, "IsValid": True}))
    assert [e.name for e in c.list_events(days_ahead=None)] == [
        "Sooner", "Later"]


def test_empty_events_list_raises():
    c = _client(_Page(api={"Data": {"List": []}, "IsValid": True}))
    with pytest.raises(cr.CourtReservePageError):
        c.list_events()


def test_events_page_without_its_data_call_raises():
    c = _client(_Page("<html><body>Something new</body></html>"))
    with pytest.raises(cr.CourtReservePageError):
        c.list_events()


def test_logged_out_events_page_raises():
    c = _client(_Page(
        "<html>You need to be a member of this organization to access "
        "this module!</html>"))
    with pytest.raises(cr.CourtReservePageError, match="logged out"):
        c.list_events()


@pytest.mark.parametrize("api,status", [
    (ValueError("not json"), 200),
    ({"IsValid": False}, 200),
    ({"Data": {}}, 500),
])
def test_bad_api_response_raises(api, status):
    c = _client(_Page(api=api, status=status))
    with pytest.raises(cr.CourtReservePageError, match="unreadable"):
        c.get_event_registrants("https://app.courtreserve.com/x")


def test_login_form_instead_of_event_page_raises():
    c = _client(_Page("<html>Log in</html>", login_form=True))
    with pytest.raises(cr.CourtReservePageError):
        c.get_event_registrants("https://app.courtreserve.com/x")


def test_cloudflare_instead_of_event_page_raises():
    c = _client(_Page(
        "<html>Attention Required | Cloudflare /cdn-cgi/ </html>"))
    with pytest.raises(cr.CloudflareBlocked):
        c.get_event_registrants("https://app.courtreserve.com/x")


def test_unrecognised_event_data_raises():
    c = _client(_Page(api={"Data": {"EventData": {}}, "IsValid": True}))
    with pytest.raises(cr.CourtReservePageError):
        c.get_event_registrants("https://app.courtreserve.com/x")


def test_event_with_title_and_no_registrants_is_fine():
    c = _client(_Page(api=_detail(AllRegisteredMembers=[])))
    reg = c.get_event_registrants("https://app.courtreserve.com/x")
    assert reg.event_name == "Thursday Evening Club Social for Intermediate+"
    assert reg.registrants == []
    assert reg.detail_url == "https://app.courtreserve.com/x"
