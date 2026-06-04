"""Tests for tool_list_roster_grouped — the deterministic roster
listing tool that the LLM emits verbatim to avoid name-dropouts.

The whole reason the tool exists is reliability under long lists,
so the tests focus on: every roster entry appears, ordering is
deterministic, and grouping for the four `group_by` modes is
correct."""

from __future__ import annotations

import pytest


@pytest.fixture
def fake_roster(monkeypatch):
    """Patch admin_bot.Roster with a stub returning a controllable
    `all()` mapping."""
    import admin_bot

    sample = {
        "Anita Reid":      {"rating": 3, "gender": "F", "singles": "",        "provisional": False},
        "Patrick Gibbs":   {"rating": 3, "gender": "M", "singles": "prefer",  "provisional": False},
        "Vuk Mijic":       {"rating": 3, "gender": "M", "singles": "",        "provisional": True},
        "Djordje Mijic":   {"rating": 4, "gender": "M", "singles": "",        "provisional": False},
        "Geoff Chapman":   {"rating": 4, "gender": "M", "singles": "prefer",  "provisional": False},
        "Ana Mijic":       {"rating": 7, "gender": "F", "singles": "avoid",   "provisional": False},
        "Mystery Player":  {"rating": "?", "gender": "?", "singles": "",      "provisional": False},
    }

    class _FakeRoster:
        def all(self):
            return {k: dict(v) for k, v in sample.items()}

    monkeypatch.setattr(admin_bot, "Roster", lambda: _FakeRoster())
    return admin_bot, sample


# ---------- by rating (the use case that triggered this) -----------------


def test_by_rating_includes_every_player(fake_roster):
    admin_bot, sample = fake_roster
    res = admin_bot.tool_list_roster_grouped(group_by="rating")
    assert res["ok"] is True
    text = res["text"]
    # Every name must appear somewhere.
    for name in sample:
        assert name in text, f"missing from text: {name}"


def test_by_rating_groups_in_numeric_order_question_mark_last(fake_roster):
    admin_bot, _ = fake_roster
    text = admin_bot.tool_list_roster_grouped(group_by="rating")["text"]
    # Headings appear in this order.
    headings_in_order = []
    for line in text.splitlines():
        if line.startswith("*Rating"):
            headings_in_order.append(line)
    assert headings_in_order == [
        "*Rating 3*", "*Rating 4*", "*Rating 7*", "*Rating ?*",
    ]


def test_by_rating_names_sorted_first_name_alphabetical(fake_roster):
    """Within each group, alphabetical by first name."""
    admin_bot, _ = fake_roster
    text = admin_bot.tool_list_roster_grouped(group_by="rating")["text"]
    # Rating 3 block contains Anita, Patrick, Vuk in that order.
    rating3_block = text.split("*Rating 3*\n", 1)[1].split("\n\n", 1)[0]
    lines = rating3_block.splitlines()
    assert lines == ["Anita Reid", "Patrick Gibbs", "Vuk Mijic"]


def test_no_per_group_count_in_heading(fake_roster):
    """Headings are just the bold label — no '(N players)' so the
    LLM can't get it wrong by paraphrasing."""
    admin_bot, _ = fake_roster
    text = admin_bot.tool_list_roster_grouped(group_by="rating")["text"]
    for line in text.splitlines():
        if line.startswith("*Rating"):
            assert "players" not in line.lower()
            assert "(" not in line, f"unexpected paren in heading: {line!r}"


# ---------- by gender ---------------------------------------------------


def test_by_gender_groups_into_m_f_unknown(fake_roster):
    admin_bot, _ = fake_roster
    text = admin_bot.tool_list_roster_grouped(group_by="gender")["text"]
    assert "*Male*" in text
    assert "*Female*" in text
    assert "*Unknown gender*" in text
    # Mystery Player only appears under Unknown.
    unknown_block = text.split("*Unknown gender*\n", 1)[1]
    assert "Mystery Player" in unknown_block


# ---------- by singles --------------------------------------------------


def test_by_singles_lists_prefer_avoid_neutral(fake_roster):
    admin_bot, _ = fake_roster
    text = admin_bot.tool_list_roster_grouped(group_by="singles")["text"]
    # Headings present.
    assert "*Singles: prefer*" in text
    assert "*Singles: avoid*" in text
    assert "*Singles: neutral*" in text
    # Geoff prefers; Ana avoids; Anita is neutral — verify membership.
    prefer_block = text.split("*Singles: prefer*\n", 1)[1].split("\n\n", 1)[0]
    avoid_block = text.split("*Singles: avoid*\n", 1)[1].split("\n\n", 1)[0]
    assert "Geoff Chapman" in prefer_block
    assert "Ana Mijic" in avoid_block


# ---------- by provisional ----------------------------------------------


def test_by_provisional_shows_only_provisional_names(fake_roster):
    admin_bot, _ = fake_roster
    text = admin_bot.tool_list_roster_grouped(group_by="provisional")["text"]
    assert "Vuk Mijic" in text
    # Non-provisional names must NOT appear.
    assert "Geoff Chapman" not in text
    assert "Anita Reid" not in text


def test_by_provisional_empty_when_nobody_flagged(monkeypatch):
    import admin_bot

    class _R:
        def all(self):
            return {
                "Alice": {"rating": 5, "gender": "F", "singles": "",
                          "provisional": False},
            }

    monkeypatch.setattr(admin_bot, "Roster", lambda: _R())
    res = admin_bot.tool_list_roster_grouped(group_by="provisional")
    assert res["ok"] is True
    assert "no players currently have a provisional rating" in res["text"]


# ---------- error paths -------------------------------------------------


def test_unknown_group_by_returns_error(fake_roster):
    admin_bot, _ = fake_roster
    res = admin_bot.tool_list_roster_grouped(group_by="zodiac_sign")
    assert res["ok"] is False
    assert res["error"] == "unknown_group_by"
    assert "rating" in res["valid"]


def test_empty_roster_returns_friendly_message(monkeypatch):
    import admin_bot

    class _R:
        def all(self):
            return {}

    monkeypatch.setattr(admin_bot, "Roster", lambda: _R())
    res = admin_bot.tool_list_roster_grouped(group_by="rating")
    assert res["ok"] is True
    assert "empty" in res["text"].lower()
