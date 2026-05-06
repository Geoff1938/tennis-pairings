"""Tests for accounts.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def fresh_accounts(tmp_path, monkeypatch):
    """Point accounts.ACCOUNTS_PATH at a tmp file and reset the cached
    registry so each test gets a clean load.
    """
    import accounts

    target = tmp_path / "accounts.json"
    monkeypatch.setattr(accounts, "ACCOUNTS_PATH", target)
    accounts.reset_registry()
    return accounts


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_legacy_when_file_missing(fresh_accounts):
    a = fresh_accounts
    reg = a.load_registry()
    assert reg.multi_account is False
    assert reg.default().key == "legacy"
    # Unknown phone routes to legacy default.
    assert a.account_for_phone("447xxxxxxxxx").key == "legacy"


def test_loads_multi_account(fresh_accounts):
    a = fresh_accounts
    _write(a.ACCOUNTS_PATH, {
        "default_account_key": "geoff",
        "accounts": [
            {
                "key": "geoff", "phone": "447111", "display_name": "Geoff",
                "cr_state_subdir": "geoff",
                "cr_username_env": "COURTRESERVE_USERNAME",
                "cr_password_env": "COURTRESERVE_PASSWORD",
                "tool_scope": "full",
            },
            {
                "key": "shirley", "phone": "447222",
                "display_name": "Shirley", "default_partner": "Maggie",
                "cr_state_subdir": "shirley",
                "cr_username_env": "COURTRESERVE_USERNAME_SHIRLEY",
                "cr_password_env": "COURTRESERVE_PASSWORD_SHIRLEY",
                "tool_scope": "read_and_book",
            },
        ],
    })
    reg = a.load_registry()
    assert reg.multi_account is True
    assert reg.default().key == "geoff"
    assert reg.by_phone("447111").key == "geoff"
    assert reg.by_phone("447222").key == "shirley"
    # Unknown phone falls back to default.
    assert reg.by_phone("447999").key == "geoff"


def test_full_scope_allows_everything(fresh_accounts):
    a = fresh_accounts
    geoff = a.Account(
        key="geoff", phone=None, display_name="g",
        default_partner=None, cr_state_subdir="geoff",
        cr_username_env="X", cr_password_env="Y", tool_scope="full",
    )
    assert geoff.is_tool_allowed("kickoff_thursday") is True
    assert geoff.is_tool_allowed("commit_plan") is True
    assert geoff.is_tool_allowed("book_court") is True


def test_read_and_book_scope(fresh_accounts):
    a = fresh_accounts
    shirley = a.Account(
        key="shirley", phone=None, display_name="s",
        default_partner=None, cr_state_subdir="shirley",
        cr_username_env="X", cr_password_env="Y",
        tool_scope="read_and_book",
    )
    # Reads + bookings allowed.
    assert shirley.is_tool_allowed("read_players_roster") is True
    assert shirley.is_tool_allowed("list_club_sessions") is True
    assert shirley.is_tool_allowed("book_court") is True
    assert shirley.is_tool_allowed("schedule_court_booking") is True
    assert shirley.is_tool_allowed("validate_member_name") is True
    # Pairings flow / writes blocked.
    assert shirley.is_tool_allowed("kickoff_thursday") is False
    assert shirley.is_tool_allowed("generate_pairings") is False
    assert shirley.is_tool_allowed("commit_plan") is False
    assert shirley.is_tool_allowed("set_player_rating") is False
    assert shirley.is_tool_allowed("clear_tonight") is False


def test_read_only_scope_blocks_booking(fresh_accounts):
    a = fresh_accounts
    visitor = a.Account(
        key="visitor", phone=None, display_name="v",
        default_partner=None, cr_state_subdir="visitor",
        cr_username_env="X", cr_password_env="Y", tool_scope="read_only",
    )
    assert visitor.is_tool_allowed("read_players_roster") is True
    assert visitor.is_tool_allowed("book_court") is False


def test_unknown_scope_rejected(fresh_accounts):
    a = fresh_accounts
    _write(a.ACCOUNTS_PATH, {
        "accounts": [{
            "key": "x", "phone": None, "display_name": "x",
            "cr_state_subdir": "x",
            "cr_username_env": "X", "cr_password_env": "Y",
            "tool_scope": "free_for_all",
        }]
    })
    with pytest.raises(ValueError, match="unknown tool_scope"):
        a.load_registry()


def test_duplicate_key_rejected(fresh_accounts):
    a = fresh_accounts
    _write(a.ACCOUNTS_PATH, {
        "accounts": [
            {"key": "g", "phone": None, "display_name": "g",
             "cr_state_subdir": "g", "cr_username_env": "X",
             "cr_password_env": "Y", "tool_scope": "full"},
            {"key": "g", "phone": None, "display_name": "g2",
             "cr_state_subdir": "g2", "cr_username_env": "X",
             "cr_password_env": "Y", "tool_scope": "full"},
        ]
    })
    with pytest.raises(ValueError, match="duplicate key"):
        a.load_registry()


def test_bad_default_key_rejected(fresh_accounts):
    a = fresh_accounts
    _write(a.ACCOUNTS_PATH, {
        "default_account_key": "nope",
        "accounts": [{
            "key": "g", "phone": None, "display_name": "g",
            "cr_state_subdir": "g", "cr_username_env": "X",
            "cr_password_env": "Y", "tool_scope": "full",
        }]
    })
    with pytest.raises(ValueError, match="doesn't match any account"):
        a.load_registry()


def test_credentials_read_from_env(fresh_accounts, monkeypatch):
    a = fresh_accounts
    acc = a.Account(
        key="x", phone=None, display_name="x",
        default_partner=None, cr_state_subdir="x",
        cr_username_env="MY_USER", cr_password_env="MY_PASS",
        tool_scope="full",
    )
    monkeypatch.setenv("MY_USER", "alice@example.com")
    monkeypatch.setenv("MY_PASS", "hunter2")
    assert acc.cr_credentials() == ("alice@example.com", "hunter2")


def test_user_data_dir_is_per_account(fresh_accounts):
    a = fresh_accounts
    acc = a.Account(
        key="x", phone=None, display_name="x",
        default_partner=None, cr_state_subdir="shirley",
        cr_username_env="X", cr_password_env="Y", tool_scope="full",
    )
    p = acc.cr_user_data_dir()
    assert p.name == "shirley"
    assert p.parent.name == ".cr_state"
