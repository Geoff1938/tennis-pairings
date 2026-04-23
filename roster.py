"""The player roster, backed by a Google Sheet.

Source of truth is the "Players" tab of the spreadsheet pointed to by
``GOOGLE_SHEET_ID`` in ``.env``. Admins edit the sheet directly for names,
genders, ratings, notes; Boris reads/writes through this module.

Expected sheet layout (tab named ``Players``):

    A        | B       | C       | D
    Name     | Gender  | Rating  | Notes

Values:
  * ``Name`` — full name as it appears in CourtReserve. Primary key.
  * ``Gender`` — ``M`` / ``F`` / ``?``.
  * ``Rating`` — integer 1-5 or literal ``?``.
  * ``Notes`` — free text.

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
VALID_GENDERS = {"M", "F", "?"}


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
    """Normalise a rating value to int 1-5 or the string ``"?"``."""
    if rating is None or rating == "":
        return "?"
    if isinstance(rating, str) and rating.strip() == "?":
        return "?"
    try:
        r = int(rating)
    except (TypeError, ValueError):
        return "?"
    if not (1 <= r <= 5):
        raise ValueError(f"rating must be between 1 and 5 (got {rating!r})")
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
        rows = self._ws.get_all_records()  # list[dict] keyed by header
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
            self._data[name] = {
                "gender": gender,
                "rating": rating,
                "notes": notes,
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
    ) -> dict:
        """Add a new player. Returns the stored entry (existing or new)."""
        if name in self._data:
            return dict(self._data[name])
        if gender is None:
            gender = guess_gender(name)
        if gender not in VALID_GENDERS:
            gender = "?"
        rating = normalise_rating(rating)
        row = [name, gender, str(rating), notes]
        self._ws.append_row(row, value_input_option="USER_ENTERED")
        entry = {"gender": gender, "rating": rating, "notes": notes}
        self._data[name] = entry
        return dict(entry)

    def add_many_from_cr(self, names: list[str]) -> list[dict]:
        """Auto-add names seen in a CourtReserve registrant list.

        Returns list of newly-added entries (``[{"name", ...}, ...]``).
        """
        added: list[dict] = []
        for name in names:
            if not name or name in self._data:
                continue
            entry = self.add(name)
            added.append({"name": name, **entry})
        return added

    def set_rating(self, name: str, rating: Any) -> dict:
        """Update a player's rating cell. Raises ``KeyError`` if not found."""
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
