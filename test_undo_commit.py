"""Tests for the undo-commit feature: session_state.last_commit
tracking, session_log.unlog_plan reversal, and admin_bot.tool_undo_commit.
"""
from __future__ import annotations

import json

import pytest


# ---------- session_state.last_commit -----------------------------------


@pytest.fixture
def ss(tmp_path, monkeypatch):
    import session_state as s
    monkeypatch.setattr(s, "SESSION_STATE_PATH",
                        tmp_path / "session_state.json")
    return s


def test_record_and_get_and_clear_last_commit(ss):
    plan = {"date": "2026-05-21", "attendees": ["A"]}
    ss.record_commit(plan, sheet_session_rows=1, sheet_pair_rows=9)
    lc = ss.get_last_commit()
    assert lc["date"] == "2026-05-21"
    assert lc["plan"] == plan
    assert lc["sheet_session_rows"] == 1
    assert lc["sheet_pair_rows"] == 9
    assert "committed_at" in lc
    ss.clear_last_commit()
    assert ss.get_last_commit() is None


def test_clear_tonight_wipes_last_commit(ss):
    ss.record_commit({"date": "x"}, sheet_session_rows=1, sheet_pair_rows=2)
    assert ss.get_last_commit() is not None
    ss.clear_tonight()
    assert ss.get_last_commit() is None


# ---------- session_log.unlog_plan --------------------------------------


class _FakeWS:
    def __init__(self, rows):
        self.rows = [list(r) for r in rows]  # incl. header at index 0

    def col_values(self, n):
        return [r[n - 1] if len(r) >= n else "" for r in self.rows]

    def delete_rows(self, start, end=None):
        end = end or start
        # 1-indexed inclusive
        del self.rows[start - 1:end]


@pytest.fixture
def fake_tabs(monkeypatch):
    import session_log as sl
    tabs: dict[str, _FakeWS] = {}
    monkeypatch.setattr(sl, "_open_tab", lambda name: tabs[name])
    return sl, tabs


def test_unlog_plan_deletes_matching_trailing_rows(fake_tabs):
    sl, tabs = fake_tabs
    tabs[sl.SESSION_LOG_TAB] = _FakeWS([
        ["Date", "..."],
        ["2026-05-07", "old"],
        ["2026-05-21", "the committed one"],
    ])
    tabs[sl.PAIR_LOG_TAB] = _FakeWS([
        ["Date", "..."],
        ["2026-05-07", "old pair"],
        ["2026-05-21", "p1"],
        ["2026-05-21", "p2"],
        ["2026-05-21", "p3"],
    ])
    res = sl.unlog_plan("2026-05-21", session_rows=1, pair_rows=3)
    assert res["session_rows_deleted"] == 1
    assert res["pair_rows_deleted"] == 3
    assert res["warnings"] == []
    # Only the 2026-05-21 rows went; the old 2026-05-07 rows remain.
    assert tabs[sl.SESSION_LOG_TAB].rows == [
        ["Date", "..."], ["2026-05-07", "old"],
    ]
    assert tabs[sl.PAIR_LOG_TAB].rows == [
        ["Date", "..."], ["2026-05-07", "old pair"],
    ]


def test_unlog_plan_refuses_on_trailing_mismatch(fake_tabs):
    sl, tabs = fake_tabs
    # Something else appended after the commit → trailing row is a
    # different date → must NOT delete it.
    tabs[sl.SESSION_LOG_TAB] = _FakeWS([
        ["Date"], ["2026-05-21"], ["2026-05-28", "newer, unrelated"],
    ])
    tabs[sl.PAIR_LOG_TAB] = _FakeWS([
        ["Date"], ["2026-05-21"], ["2026-05-28", "unrelated"],
    ])
    res = sl.unlog_plan("2026-05-21", session_rows=1, pair_rows=1)
    assert res["session_rows_deleted"] == 0
    assert res["pair_rows_deleted"] == 0
    assert len(res["warnings"]) == 2
    # Nothing deleted.
    assert len(tabs[sl.SESSION_LOG_TAB].rows) == 3
    assert len(tabs[sl.PAIR_LOG_TAB].rows) == 3


# ---------- admin_bot.tool_undo_commit ----------------------------------


@pytest.fixture
def bot(tmp_path, monkeypatch):
    import admin_bot
    import session_state as s

    hist = tmp_path / "history.json"
    monkeypatch.setattr(admin_bot, "HISTORY_PATH", hist)
    monkeypatch.setattr(s, "SESSION_STATE_PATH",
                        tmp_path / "session_state.json")
    return admin_bot, s, hist


def test_undo_commit_end_to_end(bot):
    admin_bot, s, hist = bot
    plan = {"date": "2026-05-21", "attendees": ["A", "B"],
            "rotations": [], "display_names": {}}
    # Simulate the state right after a real commit_plan: history has
    # the appended record, draft cleared, last_commit recorded.
    hist.write_text(json.dumps([
        {"date": "2026-05-07", "attendees": []},   # earlier session
        plan,                                        # the committed one
    ]))
    s.record_commit(plan, sheet_session_rows=0, sheet_pair_rows=0)
    s.set_phase("finalised")

    res = admin_bot.tool_undo_commit()
    assert res["ok"] is True
    assert res["history_entry_removed"] is True

    # history.json: only the earlier session remains.
    remaining = json.loads(hist.read_text())
    assert remaining == [{"date": "2026-05-07", "attendees": []}]
    # Draft restored, phase reopened, last_commit cleared.
    assert s.get_draft_plan() == plan
    assert s.get_phase() == "draft_ready"
    assert s.get_last_commit() is None

    # Second undo → nothing to do.
    res2 = admin_bot.tool_undo_commit()
    assert res2["ok"] is False
    assert res2["error"] == "nothing_to_undo"


def test_undo_commit_invokes_sheet_unlog_when_rows_recorded(bot, monkeypatch):
    admin_bot, s, hist = bot
    plan = {"date": "2026-05-22", "attendees": ["A"]}
    hist.write_text(json.dumps([plan]))
    s.record_commit(plan, sheet_session_rows=1, sheet_pair_rows=4)

    called = {}
    import session_log as sl
    monkeypatch.setattr(
        sl, "unlog_plan",
        lambda date, s_rows, p_rows: called.update(
            date=date, s=s_rows, p=p_rows) or {
            "session_rows_deleted": s_rows,
            "pair_rows_deleted": p_rows, "warnings": []},
    )
    res = admin_bot.tool_undo_commit()
    assert res["ok"] is True
    assert called == {"date": "2026-05-22", "s": 1, "p": 4}
    assert res["sheet_result"]["pair_rows_deleted"] == 4
