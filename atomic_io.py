"""Crash-safe file writes for the JSON state files (history.json,
session_state.json, scheduled_bookings.json, validated_members.json).

A plain ``Path.write_text`` truncates the file first, so a power cut or
service restart mid-write leaves an empty / half-written file that then
fails to parse. Writing to a temp file in the same directory and
``os.replace``-ing it over the target is atomic on both POSIX and
Windows: readers see either the old file or the new one, never a mix.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Replace ``path`` with ``text`` atomically (fsynced before the swap)."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            # Keep the original's permissions (e.g. chmod 600) — mkstemp
            # creates the temp file 0600 regardless.
            try:
                os.chmod(tmp, path.stat().st_mode & 0o777)
            except OSError:
                pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
