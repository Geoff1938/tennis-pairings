"""The player roster, backed by a Google Sheet.

Source of truth is the "Players" tab of the spreadsheet pointed to by
``GOOGLE_SHEET_ID`` in ``.env``. Admins edit the sheet directly for names,
genders, ratings, notes; Boris reads/writes through this module.

Expected sheet layout (tab named ``Players``):

    A        | B       | C       | D      | E      | F        | G
    Name     | Gender  | Rating  | Notes  | Phone  | Singles  | Provisional

Values:
  * ``Name`` — full name as it appears in CourtReserve. Primary key.
  * ``Gender`` — ``M`` / ``F`` / ``?``.
  * ``Rating`` — integer 1-10 or literal ``?``.
  * ``Notes`` — free text.
  * ``Phone`` — E.164 (``+447...``); blank if unknown.
  * ``Singles`` — singles-court preference: ``avoid`` (don't put in
    singles unless forced), ``prefer`` (pick for singles first), or
    blank (neutral).
  * ``Provisional`` — ``Y`` if the rating was bulk-imported from
    history (registered but never confirmed by the team); blank
    otherwise. The flag is cleared as soon as an admin sets the
    rating explicitly via ``set_rating``. Treated as blank when the
    column is absent from the sheet — older sheets stay compatible.

The public ``Roster`` facade is unchanged from the prior local-JSON
implementation so no callers need updating. Each ``Roster()`` instance
does a single fetch of the full tab on construction (roughly 300–600 ms)
and caches it. Writes flush immediately to the sheet.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from gender_guesser.detector import Detector  # type: ignore

    _GENDER = Detector(case_sensitive=False)
except Exception:  # pragma: no cover
    _GENDER = None

# ---------- config -------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")

PROJECT_ROOT = Path(__file__).parent
DEFAULT_CREDENTIALS_PATH = PROJECT_ROOT / "gcp_service_account.json"
PLAYERS_TAB = "Players"
# Column indices within the Players tab (1-based; matches gspread cell API).
COL_NAME = 1
COL_GENDER = 2
COL_RATING = 3
COL_NOTES = 4
COL_PHONE = 5
COL_SINGLES = 6
COL_PROVISIONAL = 7
VALID_GENDERS = {"M", "F", "?"}
VALID_SINGLES = {"", "avoid", "prefer"}
PROVISIONAL_TRUE = "Y"
PROVISIONAL_FALSE = ""


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


# ---------- Roster -------------------------------------------------------


class Roster:
    """Google-Sheets-backed player roster.

    Not thread-safe. Construct a fresh instance per logical unit of work
    (e.g. per bot tool call) — the initial fetch is cheap enough.
    """

    def __init__(
        self,
        sheet_id: str | None = None,
        credentials_path: Path | str | None = None,
    ) -> None:
        self.sheet_id = sheet_id or os.environ.get("GOOGLE_SHEET_ID")
        if not self.sheet_id:
            raise RuntimeError(
                "GOOGLE_SHEET_ID missing from .env — re-run the Google "
                "Sheets setup (docs/google_sheets_setup.md)."
            )
        self.credentials_path = Path(credentials_path or DEFAULT_CREDENTIALS_PATH)
        if not self.credentials_path.exists():
            raise RuntimeError(
                f"Service-account JSON not found at {self.credentials_path}. "
                "See docs/google_sheets_setup.md step 3."
            )
        # Defer the gspread import so modules that never touch the roster
        # don't pay an import cost.
        import gspread

        self._gc = gspread.service_account(filename=str(self.credentials_path))
        self._sh = self._gc.open_by_key(self.sheet_id)
        self._ws = self._sh.worksheet(PLAYERS_TAB)
        self._data: dict[str, dict] = {}
        self.load()

    # --- persistence ------------------------------------------------------

    def load(self) -> None:
        """Refresh the in-memory cache from the Players tab."""
        # Phone column (E=5) holds E.164 numbers that gspread's default
        # auto-numericise will mangle — pass it through verbatim.
        rows = self._ws.get_all_records(numericise_ignore=[COL_PHONE])
        self._data = {}
        for row in rows:
            name = str(row.get("Name", "")).strip()
            if not name:
                continue
            gender = str(row.get("Gender", "?")).strip().upper() or "?"
            if gender not in VALID_GENDERS:
                gender = "?"
            rating_raw = row.get("Rating", "?")
            # Sheets auto-parses numeric cells to int; strings pass through.
            if isinstance(rating_raw, str) and rating_raw.strip() == "":
                rating: Any = "?"
            else:
                try:
                    rating = normalise_rating(rating_raw)
                except ValueError:
                    rating = "?"
            notes = str(row.get("Notes", "") or "")
            phone = str(row.get("Phone", "") or "").strip()
            singles = str(row.get("Singles", "") or "").strip().lower()
            if singles not in VALID_SINGLES:
                singles = ""
            # The Provisional column was added later — treat as blank
            # when the header is missing so older sheets keep working.
            provisional_raw = str(row.get("Provisional", "") or "").strip().upper()
            provisional = provisional_raw == PROVISIONAL_TRUE
            self._data[name] = {
                "gender": gender,
                "rating": rating,
                "notes": notes,
                "phone": phone,
                "singles": singles,
                "provisional": provisional,
            }

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
        """Name keys containing ``query`` case-insensitively."""
        q = query.strip().lower()
        if not q:
            return []
        return [name for name in self._data if q in name.lower()]

    # --- writes -----------------------------------------------------------

    def _find_row(self, name: str) -> int | None:
        """Return the 1-based row index for ``name``, or ``None`` if absent.

        The name-column search goes via ``ws.find`` which matches on the
        literal cell text. Case-sensitive; pass the canonical stored name.
        """
        try:
            cell = self._ws.find(name, in_column=COL_NAME)
        except Exception:
            return None
        return cell.row if cell else None

    def add(
        self,
        name: str,
        *,
        gender: str | None = None,
        rating: Any = "?",
        notes: str = "",
        phone: str = "",
        singles: str = "",
        provisional: bool = False,
    ) -> dict:
        """Add a new player. Returns the stored entry (existing or new).

        The name is whitespace-normalised before storage AND before
        the duplicate check — so a CR registration of ``"Jack  Fenner"``
        (double space, accidental) will land in the same slot as an
        existing ``"Jack Fenner"`` instead of creating a near-duplicate
        row.
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
        prov_cell = PROVISIONAL_TRUE if provisional else PROVISIONAL_FALSE
        row = [name, gender, str(rating), notes, phone, singles, prov_cell]
        self._ws.append_row(row, value_input_option="USER_ENTERED")
        entry = {
            "gender": gender, "rating": rating, "notes": notes,
            "phone": phone, "singles": singles,
            "provisional": bool(provisional),
        }
        self._data[name] = entry
        return dict(entry)

    def add_many_from_cr(self, names: list[str]) -> list[dict]:
        """Auto-add names seen in a CourtReserve registrant list.

        Returns list of newly-added entries (``[{"name", ...}, ...]``).
        Each name is whitespace-normalised before the existence check
        (see ``add``), so a CR upstream that delivers
        ``"Jack  Fenner"`` and ``"Jack Fenner"`` for the same person
        only produces one row.
        """
        added: list[dict] = []
        for raw in names:
            if not raw:
                continue
            name = _normalise_whitespace(raw)
            if not name or name in self._data:
                continue
            entry = self.add(name)
            added.append({"name": name, **entry})
        return added

    def rename(self, old_name: str, new_name: str) -> dict:
        """Rename an existing entry in place. Updates the Name cell on
        the sheet and the in-memory cache key. ``new_name`` is
        whitespace-normalised. Raises ``KeyError`` if ``old_name``
        isn't in the cache, ``ValueError`` if ``new_name`` is empty
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
        row = self._find_row(old_name)
        if row is None:
            raise KeyError(
                f"Name {old_name!r} was in the local cache but not found "
                "on the sheet — local cache may be stale."
            )
        self._ws.update_cell(row, COL_NAME, new_name)
        self._data[new_name] = self._data.pop(old_name)
        return dict(self._data[new_name])

    def set_rating(self, name: str, rating: Any) -> dict:
        """Update a player's rating cell. Raises ``KeyError`` if not found.

        Side-effect: clears the ``Provisional`` flag if it was set. An
        admin running ``boris rate <name> <N>`` is the team's explicit
        confirmation that the rating is right (or the right correction
        to it), so the (P) marker should drop off after this call.
        """
        if name not in self._data:
            raise KeyError(name)
        new_rating = normalise_rating(rating)
        row = self._find_row(name)
        if row is None:
            raise KeyError(
                f"Name {name!r} was in the local cache but not found on the "
                "sheet — local cache may be stale."
            )
        self._ws.update_cell(row, COL_RATING, str(new_rating))
        self._data[name]["rating"] = new_rating
        # Clear the provisional flag — admin's explicit rating call
        # IS the confirmation. No-op if it wasn't set.
        if self._data[name].get("provisional"):
            self._ws.update_cell(row, COL_PROVISIONAL, PROVISIONAL_FALSE)
            self._data[name]["provisional"] = False
        return dict(self._data[name])

    def clear_provisional(self, name: str) -> dict:
        """Drop the ``Provisional`` flag for ``name`` without touching
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
        if not self._data[name].get("provisional"):
            return dict(self._data[name])
        row = self._find_row(name)
        if row is None:
            raise KeyError(
                f"Name {name!r} was in the local cache but not found on the "
                "sheet — local cache may be stale."
            )
        self._ws.update_cell(row, COL_PROVISIONAL, PROVISIONAL_FALSE)
        self._data[name]["provisional"] = False
        return dict(self._data[name])

    def set_singles(self, name: str, singles: str) -> dict:
        """Update a player's singles-preference cell. Raises ``KeyError``
        if not found, ``ValueError`` for an unknown preference value.
        """
        if name not in self._data:
            raise KeyError(name)
        singles = (singles or "").strip().lower()
        if singles not in VALID_SINGLES:
            raise ValueError(f"singles must be in {sorted(VALID_SINGLES)}")
        row = self._find_row(name)
        if row is None:
            raise KeyError(name)
        self._ws.update_cell(row, COL_SINGLES, singles)
        self._data[name]["singles"] = singles
        return dict(self._data[name])

    def set_phone(self, name: str, phone: str) -> dict:
        """Update a player's phone cell. Raises ``KeyError`` if not found.

        ``phone`` should be E.164 (``+447...``) or empty string to clear.
        Sheets would otherwise mangle the leading ``+`` as a formula, so
        the cell is written with a leading apostrophe to force text mode;
        the apostrophe is invisible on display and on read-back.
        """
        if name not in self._data:
            raise KeyError(name)
        row = self._find_row(name)
        if row is None:
            raise KeyError(name)
        cell_value = "'" + phone if phone else ""
        self._ws.update_cell(row, COL_PHONE, cell_value)
        self._data[name]["phone"] = phone
        return dict(self._data[name])

    def set_gender(self, name: str, gender: str) -> dict:
        if name not in self._data:
            raise KeyError(name)
        if gender not in VALID_GENDERS:
            raise ValueError("gender must be M / F / ?")
        row = self._find_row(name)
        if row is None:
            raise KeyError(name)
        self._ws.update_cell(row, COL_GENDER, gender)
        self._data[name]["gender"] = gender
        return dict(self._data[name])

    def delete(self, name: str) -> dict:
        """Remove a player row from the sheet AND the local cache.

        Returns the entry that was deleted (so the caller can show or
        log what went away). Raises ``KeyError`` if the player isn't
        in the roster.

        Note that the next CourtReserve scrape will re-add this name
        with default values (gender guessed from first name, rating
        ``?``) if it appears in a future registration list — so this
        is meant for cleaning up duplicate / stale rows, not for
        excluding people from sessions.
        """
        if name not in self._data:
            raise KeyError(name)
        row = self._find_row(name)
        if row is None:
            raise KeyError(
                f"Name {name!r} was in the local cache but not found on "
                "the sheet — local cache may be stale."
            )
        deleted = dict(self._data[name])
        self._ws.delete_rows(row)
        del self._data[name]
        return deleted


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
