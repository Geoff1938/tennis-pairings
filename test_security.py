"""Booking tools need the booking group (by JID) AND a sender listed in
accounts.json; CourtReserve session folders are owner-only."""
from __future__ import annotations

import json
import os
import stat

import pytest

import admin_bot
from admin_bot import BOOKING_GROUP_JID, PROTECTED_TOOLS, _tools_for_caller


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    import accounts as a

    target = tmp_path / "accounts.json"
    target.write_text(json.dumps({
        "default_account_key": "geoff",
        "accounts": [
            {"key": "geoff", "phone": "447000000001", "tool_scope": "full"},
            {"key": "shirley", "phone": "447000000002",
             "tool_scope": "read_and_book"},
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(a, "ACCOUNTS_PATH", target)
    a.reset_registry()
    yield a
    a.reset_registry()


def _names(group_jid, account, enrolled):
    _, impls = _tools_for_caller(group_jid, account, enrolled)
    return set(impls)


def test_enrolment(accounts):
    assert accounts.is_enrolled("447000000002")
    assert not accounts.is_enrolled("447999999999")
    assert not accounts.is_enrolled(None)


def test_legacy_mode_counts_everyone_as_enrolled(tmp_path, monkeypatch):
    import accounts as a

    monkeypatch.setattr(a, "ACCOUNTS_PATH", tmp_path / "missing.json")
    a.reset_registry()
    try:
        assert a.is_enrolled("447999999999")
    finally:
        a.reset_registry()


def test_booking_tools_only_for_enrolled_sender_in_booking_group(accounts):
    geoff = accounts.account_for_key("geoff")
    assert PROTECTED_TOOLS <= _names(BOOKING_GROUP_JID, geoff, True)
    # Unknown sender in the booking group: default account, no booking.
    stranger = accounts.account_for_phone("447999999999")
    assert stranger.key == "geoff"
    assert not PROTECTED_TOOLS & _names(BOOKING_GROUP_JID, stranger, False)
    # Everyday tools are still there.
    assert "get_tonight" in _names(BOOKING_GROUP_JID, stranger, False)


def test_group_renamed_to_booking_name_gets_no_booking_tools(accounts):
    geoff = accounts.account_for_key("geoff")
    assert not PROTECTED_TOOLS & _names("120363425872453317@g.us", geoff,
                                        True)


def test_run_agent_withholds_booking_from_unknown_sender(accounts,
                                                         monkeypatch):
    seen = {}

    def fake_tools(group_jid, account, enrolled=True):
        seen.update(jid=group_jid, enrolled=enrolled)
        raise RuntimeError("stop")

    monkeypatch.setattr(admin_bot, "_tools_for_caller", fake_tools)
    jt = admin_bot._CURRENT_GROUP_JID.set(BOOKING_GROUP_JID)
    st = admin_bot._CURRENT_SENDER.set("447999999999")
    try:
        with pytest.raises(RuntimeError, match="stop"):
            admin_bot.run_agent(None, "book court 5 friday")
    finally:
        admin_bot._CURRENT_GROUP_JID.reset(jt)
        admin_bot._CURRENT_SENDER.reset(st)
    assert seen == {"jid": BOOKING_GROUP_JID, "enrolled": False}


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_cr_state_dirs_are_owner_only(tmp_path, monkeypatch):
    import courtreserve as cr

    root = tmp_path / ".cr_state"
    monkeypatch.setattr(cr, "CR_STATE_DIR", root)
    cr._private_dir(root / "geoff")
    for d in (root, root / "geoff"):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700
