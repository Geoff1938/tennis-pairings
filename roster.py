"""The player roster, stored locally in ``players.json``.

The roster used to live in the "Players" tab of a Google Sheet. Nobody
edited the sheet by hand, so it moved back to a local file (Nov 2026):
no Google API calls, no network round trip on every ``Roster()``, and
the file is in the nightly encrypted backup with the other state files.

File layout — one object keyed by full name::

    {
      "Alice Smith": {"gender": "F", "rating": 5, "notes": "",
                      "phone": "+447...", "singles": "", "provisional": false,
                      "mixed": ""},
      ...
    }

Values:
  * key — full name as it appears in CourtReserve. Primary key.
  * ``gender`` — ``M`` / ``F`` / ``?``.
  * ``rating`` — integer 1-10 or literal ``"?"``.
  * ``notes`` — free text.
  * ``phone`` — E.164 (``+447...``); blank if unknown.
  * ``singles`` — singles-court preference: ``avoid`` (don't put in
    singles unless forced), ``prefer`` (pick for singles first), or
    blank (neutral).
  * ``provisional`` — true if the rating was bulk-imported from
    history (registered but never confirmed by the team). Cleared as
    soon as an admin sets the rating explicitly via ``set_rating``.
  * ``mixed`` — ``prefer`` opts in to mixed doubles; blank otherwise.

Every write re-reads the file, applies the change and replaces the file
atomically (``atomic_io``), under a process-wide lock, so two ``Roster``
instances alive at once can't overwrite each other's changes.
"""

from __future__ import annotations

import json
import threading
from functools import wraps
from pathlib import Path
from typing import Any

from atomic_io import atomic_write_text

try:
    from gender_guesser.detector import Detector  # type: ignore

    _GENDER = Detector(case_sensitive=False)
except Exception:  # pragma: no cover
    _GENDER = None

# ---------- config -------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
PLAYERS_PATH = PROJECT_ROOT / "players.json"
VALID_GENDERS = {"M", "F", "?"}
VALID_SINGLES = {"", "avoid", "prefer"}
# Mixed-doubles opt-in (Jul 2026): "prefer" asks the pairing engine to
# give this player at least one mixed rotation per evening when feasible
# (see mixed_match_missed in pairings.py). Blank = no preference.
VALID_MIXED = {"", "prefer"}

# Serialises read-modify-write cycles across threads (the scheduler and
# the agent can both hold a Roster at once).
_LOCK = threading.RLock()


# ---------- helpers ------------------------------------------------------


def guess_gender(full_name: str) -> str:
    """Guess M/F/? from the first name using the gender-guesser library."""
    if not full_name or _GENDER is None:
        return "?"
    first = full_name.strip().split()[0]
    result = _GENDER.get_gender(first)
    if result in ("male", "mostly_male"):
        return "M"
    if result in ("female", "mostly_female"):
        return "F"
    return "?"


def normalise_rating(rating: Any) -> Any:
    """Normalise a rating value to int 1-10 or the string ``"?"``."""
    if rating is None or rating == "":
        return "?"
    if isinstance(rating, str) and rating.strip() == "?":
        return "?"
    try:
        r = int(rating)
    except (TypeError, ValueError):
        return "?"
    if not (1 <= r <= 10):
        raise ValueError(f"rating must be between 1 and 10 (got {rating!r})")
    return r


def clean_entry(raw: dict) -> dict:
    """Coerce a stored (or imported) entry to the canonical shape,
    replacing anything invalid with its blank default."""
    gender = str(raw.get("gender", "?") or "?").strip().upper()
    if gender not in VALID_GENDERS:
        gender = "?"
    try:
        rating = normalise_rating(raw.get("rating", "?"))
    except ValueError:
        rating = "?"
    singles = str(raw.get("singles", "") or "").strip().lower()
    if singles not in VALID_SINGLES:
        singles = ""
    mixed = str(raw.get("mixed", "") or "").strip().lower()
    if mixed not in VALID_MIXED:
        mixed = ""
    return {
        "gender": gender,
        "rating": rating,
        "notes": str(raw.get("notes", "") or ""),
        "phone": str(raw.get("phone", "") or "").strip(),
        "singles": singles,
        "provisional": bool(raw.get("provisional", False)),
        "mixed": mixed,
    }


def _mutation(method):
    """Run a write method as lock → reload from disk → change → save."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with _LOCK:
            self.load()
            before = json.dumps(self._data, sort_keys=True)
            result = method(self, *args, **kwargs)
            if json.dumps(self._data, sort_keys=True) != before:
                self._save()
            return result

    return wrapper


# ---------- Roster -------------------------------------------------------


class Roster:
    """Player roster backed by ``players.json``.

    Construct a fresh instance per logical unit of work (e.g. per bot
    tool call). Reads come from the cache loaded at construction; each
    write reloads the file first, so it never works from stale data.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or PLAYERS_PATH)
        self._data: dict[str, dict] = {}
        self.load()

    # --- persistence ------------------------------------------------------

    def load(self) -> None:
        """Refresh the in-memory cache from the file.

        A missing file is an error rather than an empty roster: an empty
        roster would silently re-add everyone from CourtReserve with
        default ratings.
        """
        if not self.path.exists():
            raise RuntimeError(
                f"Roster file not found at {self.path}. Restore it from "
                "the nightly backup, or create it with "
                "scripts/migrate_sheet_to_local.py."
            )
        raw = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        self._data = {}
        for name, entry in raw.items():
            name = str(name).strip()
            if name:
                self._data[name] = clean_entry(entry or {})

    def _save(self) -> None:
        ordered = {k: self._data[k] for k in sorted(self._data)}
        atomic_write_text(
            self.path,
            json.dumps(ordered, indent=2, ensure_ascii=False) + "\n",
        )

    # --- reads ------------------------------------------------------------

    def all(self) -> dict[str, dict]:
        return {k: dict(v) for k, v in self._data.items()}

    def get(self, name: str) -> dict | None:
        entry = self._data.get(name)
        return dict(entry) if entry is not None else None

    def names(self) -> list[str]:
        return sorted(self._data)

    def find_missing(self, names: list[str]) -> list[str]:
        """Names not present in the roster."""
        return [n for n in names if n not in self._data]

    def find_by_fuzzy(self, query: str) -> list[str]:
        """Name keys matching ``query`` (see name_match: exact, then
        whole word, then word start, then substring)."""
        from name_match import match_names

        return match_names(query, self._data)

    # --- writes -----------------------------------------------------------

    @_mutation
    def add(self, name: str, **fields: Any) -> dict:
        """Add a new player. Returns the stored entry (existing or new).
        Keyword fields as in ``_add``."""
        return self._add(name, **fields)

    def _add(
        self,
        name: str,
        *,
        gender: str | None = None,
        rating: Any = "?",
        notes: str = "",
        phone: str = "",
        singles: str = "",
        provisional: bool = False,
        mixed: str = "",
    ) -> dict:
        """Add a new player (caller holds the lock and saves).

        The name is whitespace-normalised before storage AND before
        the duplicate check — so a CR registration of ``"Jack  Fenner"``
        (double space, accidental) will land in the same slot as an
        existing ``"Jack Fenner"`` instead of creating a near-duplicate.
        """
        name = _normalise_whitespace(name)
        if not name:
            raise ValueError("name must be non-empty")
        if name in self._data:
            return dict(self._data[name])
        if gender is None:
            gender = guess_gender(name)
        if gender not in VALID_GENDERS:
            gender = "?"
        rating = normalise_rating(rating)
        if singles not in VALID_SINGLES:
            raise ValueError(f"singles must be in {sorted(VALID_SINGLES)}")
        mixed = (mixed or "").strip().lower()
        if mixed not in VALID_MIXED:
            raise ValueError(f"mixed must be in {sorted(VALID_MIXED)}")
        entry = {
            "gender": gender, "rating": rating, "notes": notes,
            "phone": phone, "singles": singles,
            "provisional": bool(provisional), "mixed": mixed,
        }
        self._data[name] = entry
        return dict(entry)

    @_mutation
    def add_many_from_cr(self, names: list[str]) -> list[dict]:
        """Auto-add names seen in a CourtReserve registrant list.

        Returns list of newly-added entries (``[{"name", ...}, ...]``).
        Each name is whitespace-normalised before the existence check
        (see ``add``), so a CR upstream that delivers
        ``"Jack  Fenner"`` and ``"Jack Fenner"`` for the same person
        only produces one entry. Saved once at the end.
        """
        added: list[dict] = []
        for raw in names:
            if not raw:
                continue
            name = _normalise_whitespace(raw)
            if not name or name in self._data:
                continue
            entry = self._add(name)
            added.append({"name": name, **entry})
        return added

    @_mutation
    def rename(self, old_name: str, new_name: str) -> dict:
        """Rename an existing entry in place. ``new_name`` is
        whitespace-normalised. Raises ``KeyError`` if ``old_name``
        isn't in the roster, ``ValueError`` if ``new_name`` is empty
        after normalisation OR would collide with another existing
        entry. No-op when normalising ``new_name`` equals ``old_name``
        — common when an admin just wants to "fix the spelling" but
        the value is already canonical."""
        new_name = _normalise_whitespace(new_name)
        if not new_name:
            raise ValueError("new_name must be non-empty")
        if old_name not in self._data:
            raise KeyError(old_name)
        if new_name == old_name:
            return dict(self._data[old_name])
        if new_name in self._data:
            raise ValueError(
                f"can't rename {old_name!r} -> {new_name!r}: target name "
                "is already in the roster (use merge_and_delete to combine)"
            )
        self._data[new_name] = self._data.pop(old_name)
        return dict(self._data[new_name])

    @_mutation
    def set_rating(self, name: str, rating: Any) -> dict:
        """Update a player's rating. Raises ``KeyError`` if not found.

        Side-effect: clears the ``provisional`` flag if it was set. An
        admin running ``boris rate <name> <N>`` is the team's explicit
        confirmation that the rating is right (or the right correction
        to it), so the (P) marker should drop off after this call.
        """
        if name not in self._data:
            raise KeyError(name)
        self._data[name]["rating"] = normalise_rating(rating)
        self._data[name]["provisional"] = False
        return dict(self._data[name])

    @_mutation
    def clear_provisional(self, name: str) -> dict:
        """Drop the ``provisional`` flag for ``name`` without touching
        any other field. No-op (still returns the entry) when the flag
        wasn't set, so callers can run this idempotently across a
        whole lineup. Raises ``KeyError`` if the player isn't in the
        roster.

        Use this from the bulk-confirm path (admin reviewed a session's
        lineup and is happy with all ratings). For a single-player
        explicit rate change, ``set_rating`` already clears the flag
        as a side-effect — no need to call this on top.
        """
        if name not in self._data:
            raise KeyError(name)
        self._data[name]["provisional"] = False
        return dict(self._data[name])

    @_mutation
    def set_singles(self, name: str, singles: str) -> dict:
        """Update a player's singles preference. Raises ``KeyError``
        if not found, ``ValueError`` for an unknown preference value.
        """
        if name not in self._data:
            raise KeyError(name)
        singles = (singles or "").strip().lower()
        if singles not in VALID_SINGLES:
            raise ValueError(f"singles must be in {sorted(VALID_SINGLES)}")
        self._data[name]["singles"] = singles
        return dict(self._data[name])

    @_mutation
    def set_mixed(self, name: str, mixed: str) -> dict:
        """Update a player's mixed-doubles opt-in. Raises ``KeyError``
        if not found, ``ValueError`` for an unknown value. ``"prefer"``
        opts the player in; ``""`` clears it.
        """
        if name not in self._data:
            raise KeyError(name)
        mixed = (mixed or "").strip().lower()
        if mixed not in VALID_MIXED:
            raise ValueError(f"mixed must be in {sorted(VALID_MIXED)}")
        self._data[name]["mixed"] = mixed
        return dict(self._data[name])

    @_mutation
    def set_phone(self, name: str, phone: str) -> dict:
        """Update a player's phone. Raises ``KeyError`` if not found.

        ``phone`` should be E.164 (``+447...``) or empty string to clear.
        """
        if name not in self._data:
            raise KeyError(name)
        self._data[name]["phone"] = (phone or "").strip()
        return dict(self._data[name])

    @_mutation
    def set_gender(self, name: str, gender: str) -> dict:
        if name not in self._data:
            raise KeyError(name)
        if gender not in VALID_GENDERS:
            raise ValueError("gender must be M / F / ?")
        self._data[name]["gender"] = gender
        return dict(self._data[name])

    @_mutation
    def delete(self, name: str) -> dict:
        """Remove a player from the roster.

        Returns the entry that was deleted (so the caller can show or
        log what went away). Raises ``KeyError`` if the player isn't
        in the roster.

        Note that the next CourtReserve scrape will re-add this name
        with default values (gender guessed from first name, rating
        ``?``) if it appears in a future registration list — so this
        is meant for cleaning up duplicate / stale entries, not for
        excluding people from sessions.
        """
        if name not in self._data:
            raise KeyError(name)
        return self._data.pop(name)


# ---------- duplicate detection -----------------------------------------
#
# Common ways the same person ends up in the roster twice:
#
# 1. Apostrophe variants. CourtReserve writes curly U+2019 ("Luke
#    O’Mahoney"); WhatsApp keyboards type straight U+0027 ("Luke
#    O'Mahoney"). Boris adds whichever form it first sees as a new row.
# 2. Nickname variants. Same surname but the first name swings between
#    e.g. "Ben"/"Benjamin", "Mike"/"Michael", "Tom"/"Thomas" — usually
#    because CR registration was edited or a different format was used.
# 3. Whitespace / casing differences. Double spaces, trailing spaces,
#    inconsistent capitalisation.

NICKNAMES_TO_FULL = {
    "ben": "benjamin",
    "benji": "benjamin",
    "bill": "william",
    "billy": "william",
    "bob": "robert",
    "bobby": "robert",
    "chris": "christopher",
    "dan": "daniel",
    "danny": "daniel",
    "dave": "david",
    "dick": "richard",
    "ed": "edward",
    "eddie": "edward",
    "fred": "frederick",
    "freddie": "frederick",
    "greg": "gregory",
    "jack": "john",
    "jim": "james",
    "jimmy": "james",
    "joe": "joseph",
    "kate": "katherine",
    "kathy": "katherine",
    "ken": "kenneth",
    "kenny": "kenneth",
    "matt": "matthew",
    "matty": "matthew",
    "mike": "michael",
    "nate": "nathan",
    "nick": "nicholas",
    "pat": "patrick",
    "patty": "patricia",
    "rich": "richard",
    "rick": "richard",
    "rob": "robert",
    "ron": "ronald",
    "sam": "samuel",
    "steve": "stephen",
    "sue": "susan",
    "suzy": "susan",
    "ted": "edward",
    "tim": "timothy",
    "tom": "thomas",
    "tommy": "thomas",
    "tony": "anthony",
    "will": "william",
}


def _normalise_apostrophes(s: str) -> str:
    """Collapse curly apostrophes to straight ones — the most common
    cause of duplicate roster rows."""
    return s.replace("’", "'").replace("‘", "'")


def _normalise_whitespace(s: str) -> str:
    """Collapse internal whitespace runs and trim."""
    return " ".join(s.split())


def _expand_first_name(first: str) -> str:
    """Map a nickname first-name to its canonical form, lowercase. Names
    not in the table pass through unchanged (lowercased)."""
    lower = first.lower()
    return NICKNAMES_TO_FULL.get(lower, lower)


def _canonical_key(name: str) -> str:
    """Reduce a roster name to a comparison key that catches the
    common duplicate patterns. Two names sharing this key are likely
    the same person:
      * apostrophes normalised (curly → straight)
      * whitespace collapsed, case-folded
      * first-name run through the nickname table
    """
    n = _normalise_apostrophes(_normalise_whitespace(name)).lower()
    parts = n.split(" ", 1)
    if not parts or not parts[0]:
        return n
    first = _expand_first_name(parts[0])
    rest = parts[1] if len(parts) > 1 else ""
    return f"{first} {rest}".strip()


def find_duplicates(roster_data: dict[str, dict]) -> list[dict]:
    """Scan a roster cache for groups of likely-duplicate names.

    Each returned group is::

        {"key": "<canonical key>",
         "names": [name, name, ...],   # ≥ 2 entries
         "hint": "apostrophe variant" | "nickname/whitespace variant"}

    Singletons (names that share no canonical key with any other) are
    omitted. Groups are sorted by their canonical key for deterministic
    output. Read-only; the caller decides what to do.
    """
    by_key: dict[str, list[str]] = {}
    for name in roster_data:
        by_key.setdefault(_canonical_key(name), []).append(name)
    groups: list[dict] = []
    for key, names in sorted(by_key.items()):
        if len(names) < 2:
            continue
        # Hint: if the raw apostrophe-only normalisation already
        # equates them, that's the most common case worth surfacing
        # plainly. Otherwise the difference is in case / nickname.
        apos_norms = {_normalise_apostrophes(n).lower(): None for n in names}
        if len(apos_norms) == 1:
            hint = "apostrophe variant"
        else:
            hint = "nickname/whitespace variant"
        groups.append({"key": key, "names": sorted(names), "hint": hint})
    return groups
