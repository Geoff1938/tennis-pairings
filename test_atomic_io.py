"""Tests for atomic_io.atomic_write_text."""
from __future__ import annotations

import os

import pytest

from atomic_io import atomic_write_text


def test_writes_new_and_replaces_existing(tmp_path):
    p = tmp_path / "state.json"
    atomic_write_text(p, "one")
    assert p.read_text(encoding="utf-8") == "one"
    atomic_write_text(p, "two — ü")
    assert p.read_text(encoding="utf-8") == "two — ü"
    assert [f.name for f in tmp_path.iterdir()] == ["state.json"]


def test_failed_write_keeps_original_and_cleans_up(tmp_path, monkeypatch):
    p = tmp_path / "history.json"
    p.write_text("original", encoding="utf-8")

    def boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(p, "new")
    assert p.read_text(encoding="utf-8") == "original"
    assert [f.name for f in tmp_path.iterdir()] == ["history.json"]
