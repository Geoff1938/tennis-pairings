"""Tests for _resolve_booking_account — the book_as account override."""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def bot(tmp_path, monkeypatch):
    import accounts
    import admin_bot

    accs = {
        "default_account_key": "geoff",
        "accounts": [
            {
                "key": "geoff", "phone": "447111",
                "display_name": "Geoff Chapman",
                "cr_state_subdir": "geoff",
                "cr_username_env": "U", "cr_password_env": "P",
                "tool_scope": "full",
            },
            {
                "key": "shirley", "phone": "447222",
                "display_name": "Shirley Chapman",
                "cr_state_subdir": "shirley",
                "cr_username_env": "US", "cr_password_env": "PS",
                "tool_scope": "read_and_book",
                "court_preference": ["9", "12", "7"],
            },
        ],
    }
    p = tmp_path / "accounts.json"
    p.write_text(json.dumps(accs), encoding="utf-8")
    monkeypatch.setattr(accounts, "ACCOUNTS_PATH", p)
    accounts.reset_registry()

    def set_sender(phone):
        return admin_bot._CURRENT_SENDER.set(phone)

    return admin_bot, set_sender


def test_no_book_as_uses_caller(bot):
    admin_bot, set_sender = bot
    tok = set_sender("447111")  # Geoff
    try:
        acct, err = admin_bot._resolve_booking_account(None)
        assert err is None
        assert acct.key == "geoff"
    finally:
        admin_bot._CURRENT_SENDER.reset(tok)


def test_admin_can_book_as_other_by_key(bot):
    admin_bot, set_sender = bot
    tok = set_sender("447111")  # Geoff (full scope)
    try:
        acct, err = admin_bot._resolve_booking_account("shirley")
        assert err is None
        assert acct.key == "shirley"
        # Shirley's saved preference travels with the resolved account.
        assert acct.court_preference_list() == ["9", "12", "7"]
    finally:
        admin_bot._CURRENT_SENDER.reset(tok)


@pytest.mark.parametrize("q", ["Shirley", "Shirley Chapman",
                               "SHIRLEY", "shirley chapman"])
def test_book_as_matches_display_and_first_name(bot, q):
    admin_bot, set_sender = bot
    tok = set_sender("447111")
    try:
        acct, err = admin_bot._resolve_booking_account(q)
        assert err is None and acct.key == "shirley"
    finally:
        admin_bot._CURRENT_SENDER.reset(tok)


def test_book_as_self_allowed_for_narrow_scope(bot):
    admin_bot, set_sender = bot
    tok = set_sender("447222")  # Shirley (read_and_book)
    try:
        acct, err = admin_bot._resolve_booking_account("shirley")
        assert err is None and acct.key == "shirley"
    finally:
        admin_bot._CURRENT_SENDER.reset(tok)


def test_narrow_scope_cannot_book_as_other(bot):
    admin_bot, set_sender = bot
    tok = set_sender("447222")  # Shirley → cannot book as Geoff
    try:
        acct, err = admin_bot._resolve_booking_account("geoff")
        assert acct is None
        assert err["error"] == "account_override_denied"
    finally:
        admin_bot._CURRENT_SENDER.reset(tok)


def test_unknown_account_rejected(bot):
    admin_bot, set_sender = bot
    tok = set_sender("447111")
    try:
        acct, err = admin_bot._resolve_booking_account("nigel")
        assert acct is None
        assert err["error"] == "unknown_account"
    finally:
        admin_bot._CURRENT_SENDER.reset(tok)
