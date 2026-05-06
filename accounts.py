"""Per-user accounts for Boris.

Maps a WhatsApp ``sender`` phone (the bare digits in the ``messages``
table — e.g. ``447xxxxxxxxx``) to a profile that says:

  * what to call them in chat (``display_name``)
  * which CourtReserve credentials to use (env-var names — secrets
    themselves stay in ``.env`` and are never committed)
  * which Chromium ``user-data-dir`` to load when invoking CR
    (so each account has its own session/cookies)
  * which subset of bot tools they're allowed to invoke
  * an optional default partner used to short-cut booking commands
    (e.g. Shirley → Maggie Cochrane)

Multi-account mode is opt-in. If ``accounts.json`` is missing, the bot
runs in legacy single-account mode using ``COURTRESERVE_USERNAME`` /
``COURTRESERVE_PASSWORD`` and the original ``.cr_state/`` directory —
existing behaviour, no migration needed.

When ``accounts.json`` exists:

* a sender whose phone matches an entry → that account
* a sender whose phone is unknown → the ``default_account_key``
  account (so admin-group members typing `boris ...` continue to
  work without being individually enrolled)

Tool scope is policy, not magic. ``full`` sees everything; the
narrower scopes are intersected with the channel-based
``PROTECTED_TOOLS`` filter that admin_bot already applies.

``accounts.json`` shape::

    {
      "default_account_key": "geoff",
      "accounts": [
        {
          "key": "geoff",
          "phone": "447xxxxxxxxx",
          "display_name": "Geoff Chapman",
          "default_partner": null,
          "cr_state_subdir": "geoff",
          "cr_username_env": "COURTRESERVE_USERNAME",
          "cr_password_env": "COURTRESERVE_PASSWORD",
          "tool_scope": "full"
        },
        {
          "key": "shirley",
          "phone": null,
          "display_name": "Shirley Chapman",
          "default_partner": "Maggie Cochrane",
          "cr_state_subdir": "shirley",
          "cr_username_env": "COURTRESERVE_USERNAME_SHIRLEY",
          "cr_password_env": "COURTRESERVE_PASSWORD_SHIRLEY",
          "tool_scope": "read_and_book"
        }
      ]
    }
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
ACCOUNTS_PATH = ROOT / "accounts.json"
LEGACY_CR_STATE_DIR = ROOT / ".cr_state"

# ---- tool scopes --------------------------------------------------------

# These names mirror the bot tool names declared in admin_bot.TOOL_IMPLS.
# Listed here (rather than imported) so a typo lands as a quick test
# failure instead of a circular import. Tests assert the sets are valid.

READ_TOOLS: frozenset[str] = frozenset({
    "read_players_roster",
    "read_pairings_history",
    "list_club_sessions",
    "get_session_registrants",
    "get_tonight",
    "validate_member_name",
    "list_validated_members",
})

BOOKING_TOOLS: frozenset[str] = frozenset({
    "book_session",
    "list_my_bookings",
    "cancel_booking",
    "book_court",
    "cancel_court_booking",
    "schedule_court_booking",
    "list_scheduled_bookings",
    "cancel_scheduled_booking",
    "add_validated_member",
})

# Maps the ``tool_scope`` string in accounts.json to either a concrete
# allowlist (``frozenset``) or ``None`` meaning "no scope filter — all
# tools visible". Channel-based filtering (PROTECTED_TOOLS) is applied
# separately and composes with this.
TOOL_SCOPES: dict[str, Optional[frozenset[str]]] = {
    "full": None,
    "read_and_book": READ_TOOLS | BOOKING_TOOLS,
    "booking_only": BOOKING_TOOLS,
    "read_only": READ_TOOLS,
}


# ---- data class ---------------------------------------------------------


@dataclass(frozen=True)
class Account:
    key: str                    # short stable id (e.g. "geoff")
    phone: Optional[str]        # WhatsApp digits-only phone, may be None
    display_name: str
    default_partner: Optional[str]
    cr_state_subdir: str
    cr_username_env: str
    cr_password_env: str
    tool_scope: str             # one of TOOL_SCOPES keys

    def cr_user_data_dir(self) -> Path:
        return LEGACY_CR_STATE_DIR / self.cr_state_subdir

    def cr_credentials(self) -> tuple[Optional[str], Optional[str]]:
        return (
            os.environ.get(self.cr_username_env),
            os.environ.get(self.cr_password_env),
        )

    def is_tool_allowed(self, tool_name: str) -> bool:
        allow = TOOL_SCOPES.get(self.tool_scope)
        if allow is None:                         # "full"
            return True
        return tool_name in allow


# ---- legacy single-account fallback -------------------------------------


_LEGACY_ACCOUNT = Account(
    key="legacy",
    phone=None,
    display_name="(legacy single-account)",
    default_partner=None,
    cr_state_subdir="",                           # uses LEGACY_CR_STATE_DIR
    cr_username_env="COURTRESERVE_USERNAME",
    cr_password_env="COURTRESERVE_PASSWORD",
    tool_scope="full",
)


def _legacy_dir(account: Account) -> bool:
    return account.key == "legacy"


# ---- registry -----------------------------------------------------------


@dataclass(frozen=True)
class AccountRegistry:
    """Loaded view of accounts.json, plus the fallback single-account."""
    accounts: tuple[Account, ...]
    default_key: str
    multi_account: bool                           # True iff accounts.json present

    def by_phone(self, phone: Optional[str]) -> Account:
        """Return the matching account, or the default if no match."""
        if phone:
            for a in self.accounts:
                if a.phone and a.phone == phone:
                    return a
        return self.default()

    def default(self) -> Account:
        for a in self.accounts:
            if a.key == self.default_key:
                return a
        # Misconfigured default_account_key — return first.
        return self.accounts[0]

    def by_key(self, key: str) -> Optional[Account]:
        for a in self.accounts:
            if a.key == key:
                return a
        return None


def _build_legacy_registry() -> AccountRegistry:
    return AccountRegistry(
        accounts=(_LEGACY_ACCOUNT,),
        default_key="legacy",
        multi_account=False,
    )


def _account_from_dict(raw: dict) -> Account:
    return Account(
        key=str(raw["key"]),
        phone=(str(raw["phone"]) if raw.get("phone") else None),
        display_name=str(raw.get("display_name") or raw["key"]),
        default_partner=(
            str(raw["default_partner"]) if raw.get("default_partner") else None
        ),
        cr_state_subdir=str(raw.get("cr_state_subdir") or raw["key"]),
        cr_username_env=str(
            raw.get("cr_username_env") or "COURTRESERVE_USERNAME"
        ),
        cr_password_env=str(
            raw.get("cr_password_env") or "COURTRESERVE_PASSWORD"
        ),
        tool_scope=str(raw.get("tool_scope") or "full"),
    )


def load_registry(path: Path | None = None) -> AccountRegistry:
    """Load ``accounts.json`` and return a registry.

    Returns the legacy single-account registry when the file doesn't
    exist (preserving pre-multi-account behaviour). Raises ValueError
    on a malformed file rather than silently falling back, so a typo
    surfaces loudly instead of routing every command to a default.
    """
    target = path or ACCOUNTS_PATH
    if not target.exists():
        return _build_legacy_registry()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"accounts.json is not valid JSON: {e}") from e
    accounts_list = raw.get("accounts") or []
    if not accounts_list:
        raise ValueError("accounts.json has no 'accounts' entries")
    accounts: list[Account] = []
    seen_keys: set[str] = set()
    for entry in accounts_list:
        a = _account_from_dict(entry)
        if a.key in seen_keys:
            raise ValueError(f"accounts.json has duplicate key {a.key!r}")
        if a.tool_scope not in TOOL_SCOPES:
            raise ValueError(
                f"accounts.json: account {a.key!r} has unknown "
                f"tool_scope {a.tool_scope!r} (valid: {sorted(TOOL_SCOPES)})"
            )
        accounts.append(a)
        seen_keys.add(a.key)
    default_key = str(raw.get("default_account_key") or accounts[0].key)
    if default_key not in seen_keys:
        raise ValueError(
            f"accounts.json: default_account_key {default_key!r} "
            f"doesn't match any account key"
        )
    return AccountRegistry(
        accounts=tuple(accounts),
        default_key=default_key,
        multi_account=True,
    )


# Module-level singleton, loaded lazily so the tests can monkey-patch
# ACCOUNTS_PATH before the first call.
_REGISTRY: AccountRegistry | None = None


def get_registry() -> AccountRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_registry()
    return _REGISTRY


def reset_registry() -> None:
    """Drop the cached registry — useful from tests after editing the
    JSON or env. Not used by the live code path."""
    global _REGISTRY
    _REGISTRY = None


def account_for_phone(phone: Optional[str]) -> Account:
    """Convenience: return the right account for an incoming sender."""
    return get_registry().by_phone(phone)


def account_for_key(key: str) -> Optional[Account]:
    return get_registry().by_key(key)


def default_account() -> Account:
    return get_registry().default()


# ---- CR client factory --------------------------------------------------


def cr_client(account: Account, *, headless: bool = True):
    """Open a CourtReserveClient for ``account``.

    Returns a context manager — ``with cr_client(acc) as cr:``. Raises
    ``RuntimeError`` if the account's env-var credentials are missing.
    """
    from courtreserve import CourtReserveClient

    if _legacy_dir(account):
        # Pre-multi-account behaviour: default user-data-dir + default
        # env vars (already handled inside CourtReserveClient).
        return CourtReserveClient(headless=headless)

    username, password = account.cr_credentials()
    if not username or not password:
        raise RuntimeError(
            f"CourtReserve credentials for account {account.key!r} not set "
            f"({account.cr_username_env} / {account.cr_password_env})"
        )
    return CourtReserveClient(
        user_data_dir=account.cr_user_data_dir(),
        headless=headless,
        username=username,
        password=password,
    )
