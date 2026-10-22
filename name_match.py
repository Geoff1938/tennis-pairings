"""Shared name matching for roster / attendee / waitlist lookups.

Tries progressively looser rules and stops at the first that finds
anything, so a more exact match always wins:

1. the whole name, ignoring case and extra spaces ("tom smith");
2. the query as whole words inside a name ("Smith" finds "Tom Smith"
   but not "Tom Smithson");
3. the query as the start of words ("Al" finds "Alan Jones", not
   "Sally");
4. anywhere inside a name, only for queries of 4+ characters, as a
   last resort for partial spellings.
"""
from __future__ import annotations

import re
from typing import Iterable

MIN_SUBSTRING_LEN = 4


def _norm(s: str) -> str:
    return " ".join(str(s).split()).lower()


def match_names(
    query: str, names: Iterable[str], *, partial: bool = True,
) -> list[str]:
    """Names matching ``query``, best tier only, in input order.

    ``partial=False`` stops after the whole-word tier — for choosing a
    real person on CourtReserve, where "Sam Smith" must never select
    "Sam Smithson".
    """
    q = _norm(query or "")
    if not q:
        return []
    pool = list(names)
    exact = [n for n in pool if _norm(n) == q]
    if exact:
        return exact
    word = re.compile(r"(?<!\w)" + re.escape(q) + r"(?!\w)")
    hits = [n for n in pool if word.search(_norm(n))]
    if hits or not partial:
        return hits
    prefix = re.compile(r"(?<!\w)" + re.escape(q))
    hits = [n for n in pool if prefix.search(_norm(n))]
    if hits:
        return hits
    if len(q) >= MIN_SUBSTRING_LEN:
        return [n for n in pool if q in _norm(n)]
    return []
