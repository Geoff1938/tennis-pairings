"""Validated club-member whitelist.

Court-booking commands need to know that a partner name belongs to a
real club member before they're scheduled. We don't want to rely
solely on ``players.json`` because that's the Thursday-social roster —
the user's wife's regular partner, for example, doesn't play on
Thursdays but is still a valid club member to book a court with.

This module maintains a small JSON whitelist alongside
``players.json``. The lookup order for ``is_known_member(name)`` is:

1. fuzzy match against the Thursday roster (``players.json`` via
   ``Roster.match``-style logic);
2. fuzzy match against this whitelist.

Either hit returns the canonical name. Two or more matches in either
list are reported back so the admin can disambiguate; zero hits
returns the close suggestions (if any).

Storage: ``validated_members.json`` at repo root, gitignored::

    {
      "members": [
        {"name": "Maggie Cochrane",
         "added_by": "geoff",
         "added_at": "2026-05-06"}
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
VALIDATED_MEMBERS_PATH = ROOT / "validated_members.json"


@dataclass(frozen=True)
class MemberLookup:
    """Result of looking a name up against the union of Thursday roster
    and the whitelist."""
    found: bool
    canonical_name: Optional[str]
    source: Optional[str]                # "roster" | "validated_members" | None
    candidates: tuple[str, ...]          # >1 partial match: ambiguous


# ---- store --------------------------------------------------------------


def _load(path: Path | None = None) -> list[dict]:
    target = path or VALIDATED_MEMBERS_PATH
    if not target.exists():
        return []
    raw = json.loads(target.read_text(encoding="utf-8"))
    return list(raw.get("members") or [])


def _save(members: list[dict], path: Path | None = None) -> None:
    target = path or VALIDATED_MEMBERS_PATH
    target.write_text(
        json.dumps({"members": members}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def list_members(path: Path | None = None) -> list[dict]:
    """Return the whitelist as a list of dicts (newest entries last)."""
    return _load(path)


def add_member(
    name: str,
    *,
    added_by: str = "",
    today: Optional[date] = None,
    path: Path | None = None,
) -> dict:
    """Append a name to the whitelist if not already present.

    Returns ``{"ok": True, "added": bool, "name": str, ...}``. Idempotent:
    a duplicate name is a no-op (returns ``added=False``).
    """
    name = (name or "").strip()
    if not name:
        return {"ok": False, "error": "empty_name"}
    members = _load(path)
    existing = {m["name"].lower() for m in members}
    if name.lower() in existing:
        return {"ok": True, "added": False, "name": name, "reason": "already_validated"}
    entry = {
        "name": name,
        "added_by": added_by,
        "added_at": (today or date.today()).isoformat(),
    }
    members.append(entry)
    _save(members, path)
    return {"ok": True, "added": True, "name": name, "entry": entry}


# ---- lookup -------------------------------------------------------------


def _fuzzy(query: str, names: list[str]) -> list[str]:
    """Case-insensitive substring match. Returns all hits in stable order."""
    q = query.strip().lower()
    if not q:
        return []
    return [n for n in names if q in n.lower()]


def is_known_member(
    query: str,
    *,
    roster_names: Optional[list[str]] = None,
    path: Path | None = None,
) -> MemberLookup:
    """Resolve ``query`` against the Thursday roster and the whitelist.

    Pass ``roster_names`` to control which Thursday names to match (the
    bot supplies these from ``Roster.names()``); when ``None`` the
    whitelist is the only source.
    """
    query = (query or "").strip()
    if not query:
        return MemberLookup(False, None, None, ())

    # Exact match in either list short-circuits to canonical.
    roster = list(roster_names or [])
    whitelist = [m["name"] for m in _load(path)]

    for src, names in (("roster", roster), ("validated_members", whitelist)):
        for n in names:
            if n.lower() == query.lower():
                return MemberLookup(True, n, src, ())

    # Otherwise: fuzzy substring match in roster first, then whitelist.
    for src, names in (("roster", roster), ("validated_members", whitelist)):
        hits = _fuzzy(query, names)
        if len(hits) == 1:
            return MemberLookup(True, hits[0], src, ())
        if len(hits) > 1:
            return MemberLookup(False, None, src, tuple(hits))

    return MemberLookup(False, None, None, ())
