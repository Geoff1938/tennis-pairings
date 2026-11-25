"""One-off: copy the Google Sheet into local files (Nov 2026).

Boris used to keep the roster and three logs in a Google Sheet. Nobody
edited the sheet by hand, so the data moved to local files in the repo
root (all git-ignored and in the nightly backup):

    Players tab      -> players.json
    Session log tab  -> session_log.csv
    Pair log tab     -> pair_log.csv
    Usage log tab    -> usage_log.csv

Needs the old credentials: ``GOOGLE_SHEET_ID`` in ``.env`` and
``gcp_service_account.json`` in the repo root, plus ``gspread``.
Run on the Pi with the bot stopped::

    .venv/bin/python scripts/migrate_sheet_to_local.py            # dry run
    .venv/bin/python scripts/migrate_sheet_to_local.py --write

Refuses to overwrite a file that already exists unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from roster import clean_entry  # noqa: E402
from session_log import _write_rows  # noqa: E402

LOG_TABS = {
    "Session log": ROOT / "session_log.csv",
    "Pair log": ROOT / "pair_log.csv",
    "Usage log": ROOT / "usage_log.csv",
}
PLAYERS_PATH = ROOT / "players.json"


def _players(rows: list[list[str]]) -> dict[str, dict]:
    """Players tab (header row + values) -> {name: entry}."""
    header = [h.strip().lower() for h in rows[0]]
    out: dict[str, dict] = {}
    for row in rows[1:]:
        cells = dict(zip(header, row))
        name = " ".join(str(cells.get("name", "")).split())
        if not name:
            continue
        if name in out:
            print(f"  warn: duplicate name {name!r} — keeping the first row")
            continue
        out[name] = clean_entry({
            "gender": cells.get("gender", "?"),
            "rating": cells.get("rating", "?"),
            "notes": cells.get("notes", ""),
            "phone": str(cells.get("phone", "")).lstrip("'"),
            "singles": cells.get("singles", ""),
            "provisional": str(cells.get("provisional", "")).strip().upper() == "Y",
            "mixed": cells.get("mixed", ""),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the files (default: dry run)")
    ap.add_argument("--force", action="store_true", help="overwrite files that already exist")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    if not sheet_id:
        raise SystemExit("GOOGLE_SHEET_ID missing from .env")
    import gspread

    gc = gspread.service_account(filename=str(ROOT / "gcp_service_account.json"))
    sh = gc.open_by_key(sheet_id)
    tabs = {ws.title: ws for ws in sh.worksheets()}
    print(f"Sheet tabs: {', '.join(tabs)}")

    targets: dict[Path, object] = {}
    players_rows = tabs["Players"].get_all_values()
    players = _players(players_rows)
    print(f"Players: {len(players_rows) - 1} rows -> {len(players)} players")
    targets[PLAYERS_PATH] = players
    for tab, path in LOG_TABS.items():
        if tab not in tabs:
            print(f"{tab}: tab not found, skipped")
            continue
        rows = [r for r in tabs[tab].get_all_values() if any(c.strip() for c in r)]
        print(f"{tab}: {max(0, len(rows) - 1)} data rows -> {path.name}")
        if rows:
            targets[path] = rows

    clashes = [p for p in targets if p.exists()]
    if clashes and not args.force:
        raise SystemExit(
            "Already exist (use --force to overwrite): "
            + ", ".join(p.name for p in clashes)
        )
    if not args.write:
        print("Dry run — nothing written. Re-run with --write.")
        return
    for path, content in targets.items():
        if path == PLAYERS_PATH:
            ordered = {k: content[k] for k in sorted(content)}
            path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        else:
            _write_rows(path, content)
        os.chmod(path, 0o600)
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
