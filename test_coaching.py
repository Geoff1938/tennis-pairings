"""Tests for the monthly Thursday "Coaching & Social Tennis" group:
players + courts with a coach for R1 who join the social for R2-R3.

Covers:
  * make_plan: R1 excludes the coaching courts/players (listed as
    ``with_coach``, not sit-outs); later rotations use everyone
  * polish never moves a coaching player onto an R1 court
  * validation errors (unknown player/court, no rotation left, pins)
  * session_state merge / set / clear / remove
  * kickoff: CR coaching event picked up on the same day
  * kickoff message + DOCX "With coach" line
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from pairings import make_plan


REGULAR = [f"Reg{n:02d}" for n in range(20)]
COACHED = [f"Coach{n}" for n in range(7)]
REG_COURTS = ["Court #7 - Floodlit", "Court #8 - Floodlit",
              "Court #9 - Floodlit", "Court #10 - Floodlit",
              "Court #11 - Floodlit"]
COACH_COURTS = ["Court #5 - Floodlit", "Court #6 - Floodlit"]
TEMPLATE = Path(__file__).parent / "tmp" / "Thursday Social Tennis.docx"


@pytest.fixture
def roster(tmp_path):
    players = {
        name: {
            "gender": "MF"[i % 2],
            "rating": 2 + (i % 7),
            "notes": "",
        }
        for i, name in enumerate(REGULAR + COACHED)
    }
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps([]), encoding="utf-8")
    return players, history_path


def _plan(roster, coaching, **kw):
    players, history_path = roster
    return make_plan(
        REGULAR + COACHED, players, history_path,
        court_labels=REG_COURTS + COACH_COURTS,
        num_rotations=3, seed=7, num_seed_attempts=1,
        coaching=coaching, **kw,
    )


def _on_court(rot) -> list[str]:
    return [p for c in rot.courts for p in c.players]


COACHING = {"players": COACHED, "courts": COACH_COURTS, "rotations": 1}


# ---------- pairing engine ----------------------------------------------


def test_r1_excludes_coaching_courts_and_players(roster):
    plan = _plan(roster, COACHING)
    r1 = plan.rotations[0]
    assert sorted(c.court_label for c in r1.courts) == sorted(REG_COURTS)
    assert sorted(_on_court(r1)) == sorted(REGULAR)
    assert r1.with_coach == COACHED
    # 20 regulars on 5 courts: nobody sits out, coaching group isn't
    # counted as sitting out.
    assert r1.sit_outs == []


def test_later_rotations_use_everyone(roster):
    plan = _plan(roster, COACHING)
    for rot in plan.rotations[1:]:
        assert rot.with_coach == []
        labels = {c.court_label for c in rot.courts}
        assert labels == set(REG_COURTS + COACH_COURTS)
        everyone = _on_court(rot) + rot.sit_outs
        assert sorted(everyone) == sorted(REGULAR + COACHED)
        # 27 players: one odd sit-out per rotation.
        assert len(rot.sit_outs) == 1


def test_with_coach_survives_to_dict(roster):
    d = _plan(roster, COACHING).to_dict()
    assert d["rotations"][0]["with_coach"] == COACHED
    assert d["rotations"][1]["with_coach"] == []
    assert "coach" in d["notes"]


def test_polish_keeps_coaching_players_off_r1(roster):
    # Many seeds so a cross-rotation swap is very likely proposed.
    for seed in range(5):
        players, history_path = roster
        plan = make_plan(
            REGULAR + COACHED, players, history_path,
            court_labels=REG_COURTS + COACH_COURTS,
            num_rotations=3, seed=seed, num_seed_attempts=1,
            coaching=COACHING,
        )
        assert not set(_on_court(plan.rotations[0])) & set(COACHED)
        assert plan.rotations[0].with_coach == COACHED


def test_no_coaching_arg_is_unchanged_behaviour(roster):
    plan = _plan(roster, None)
    assert all(r.with_coach == [] for r in plan.rotations)
    assert len(plan.rotations[0].courts) == 7


def test_coaching_player_not_in_attendees_raises(roster):
    with pytest.raises(ValueError, match="not in attendees"):
        _plan(roster, {**COACHING, "players": COACHED + ["Stranger"]})


def test_coaching_court_not_in_labels_raises(roster):
    with pytest.raises(ValueError, match="not in court_labels"):
        _plan(roster, {**COACHING, "courts": ["Court #1 - Floodlit"]})


def test_coaching_for_every_rotation_raises(roster):
    with pytest.raises(ValueError, match="only has 3"):
        _plan(roster, {**COACHING, "rotations": 3})


def test_pinning_coaching_player_in_r1_raises(roster):
    with pytest.raises(ValueError, match="with the coach"):
        _plan(roster, COACHING, pinned_singles=[{
            "rotation_num": 1,
            "players": [COACHED[0], REGULAR[0]],
        }])


# ---------- session state ------------------------------------------------


@pytest.fixture
def ss(tmp_path, monkeypatch):
    import session_state as ss

    monkeypatch.setattr(ss, "SESSION_STATE_PATH", tmp_path / "state.json")
    return ss


def test_start_tonight_merges_coaching_group(ss):
    state = ss.start_tonight(
        ["A", "B", "C"], court_labels=["Court #7 - Floodlit"],
        coaching_players=["C", "D"],
        coaching_courts=["Court #5 - Floodlit"],
        coaching_rotations=1,
    )
    assert state.attendees == ["A", "B", "C", "D"]
    assert state.court_labels == ["Court #7 - Floodlit", "Court #5 - Floodlit"]
    reloaded = ss.get_tonight()
    assert reloaded.coaching_players == ["C", "D"]
    assert reloaded.coaching_courts == ["Court #5 - Floodlit"]
    assert reloaded.coaching_rotations == 1


def test_start_tonight_without_coaching_has_zero_rotations(ss):
    state = ss.start_tonight(["A"], coaching_rotations=1)
    assert state.coaching_rotations == 0


def test_set_coaching_group_matches_existing_court_labels(ss):
    ss.start_tonight(
        ["A", "B"], court_labels=["Court #5 - Floodlit", "Court #7 - Floodlit"],
    )
    state = ss.set_coaching_group(["A"], ["5", "6"])
    assert state.coaching_courts == ["Court #5 - Floodlit", "6"]
    assert state.court_labels == [
        "Court #5 - Floodlit", "Court #7 - Floodlit", "6",
    ]
    with pytest.raises(ValueError, match="not in tonight's attendees"):
        ss.set_coaching_group(["Z"], ["5"])


def test_clear_and_remove_update_coaching_group(ss):
    ss.start_tonight(
        ["A", "B"], coaching_players=["C", "D"],
        coaching_courts=["5"], coaching_rotations=1,
    )
    state, removed = ss.remove_from_tonight("C")
    assert removed == "C"
    assert state.coaching_players == ["D"]
    state = ss.clear_coaching_group()
    assert state.coaching_players == []
    assert state.coaching_rotations == 0
    assert "D" in state.attendees and "5" in state.court_labels


# ---------- kickoff -----------------------------------------------------


def _event(name, day, res):
    return SimpleNamespace(
        name=name,
        date_str=f"Thu, Oct {day}th, 19:30 - 21:30",
        reservation_number=res,
        start_dt=datetime(2099, 10, day, 19, 30),
    )


class _FakeRoster:
    def __init__(self):
        self._names = ["Ann", "Bob", "Cat", "Dan"]

    def names(self):
        return list(self._names)

    def add_many_from_cr(self, names):
        new = [n for n in names if n not in self._names]
        self._names += new
        return [{"name": n} for n in new]

    def all(self):
        return {n: {"rating": 4} for n in self._names}


def _patch_cr(monkeypatch, events, regs):
    import courtreserve
    import roster

    class FakeCR:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def list_events(self, day_of_week):
            return events

        def get_event_registrants(self, res):
            return regs[res]

    monkeypatch.setattr(courtreserve, "CourtReserveClient", FakeCR)
    monkeypatch.setattr(roster, "Roster", _FakeRoster)


def _reg(names, courts):
    return SimpleNamespace(
        registrants=[SimpleNamespace(name=n) for n in names],
        waitlist=[], courts=courts,
    )


def test_collect_picks_up_same_day_coaching_event(monkeypatch):
    import thursday_kickoff
    from session_types import SESSION_TYPES

    events = [
        _event("Coaching & Social Tennis", 22, "COACH"),
        _event("Thursday Evening Club Social for Intermediate+", 22, "SOC"),
        _event("Coaching & Social Tennis", 29, "COACH2"),
    ]
    _patch_cr(monkeypatch, events, {
        "SOC": _reg(["Ann", "Bob", "Cat"], REG_COURTS),
        # Cat registered for both → counted with the coaching group.
        "COACH": _reg(["Cat", "Dan", "Eve"], COACH_COURTS),
    })
    data = thursday_kickoff._collect_session_data(SESSION_TYPES["thursday"])
    assert [r["name"] for r in data["registrants"]] == ["Ann", "Bob"]
    coaching = data["coaching"]
    assert [r["name"] for r in coaching["registrants"]] == ["Cat", "Dan", "Eve"]
    assert coaching["cr_courts"] == COACH_COURTS
    assert coaching["rotations"] == 1
    assert "Eve" in data["new_player_names"]

    text = thursday_kickoff.format_kickoff_message(
        data, SESSION_TYPES["thursday"],
    )
    assert "Coaching & Social Tennis (3)" in text
    assert "join the social from R2" in text
    assert "Court #5 - Floodlit, Court #6 - Floodlit" in text


def test_collect_ignores_coaching_event_on_other_day(monkeypatch):
    import thursday_kickoff
    from session_types import SESSION_TYPES

    events = [
        _event("Thursday Evening Club Social for Intermediate+", 22, "SOC"),
        _event("Coaching & Social Tennis", 29, "COACH2"),
    ]
    _patch_cr(monkeypatch, events, {"SOC": _reg(["Ann"], REG_COURTS)})
    data = thursday_kickoff._collect_session_data(SESSION_TYPES["thursday"])
    assert data["coaching"] is None
    text = thursday_kickoff.format_kickoff_message(
        data, SESSION_TYPES["thursday"],
    )
    assert "Coaching" not in text


def test_only_regular_thursday_has_coaching():
    from session_types import SESSION_TYPES

    assert [k for k, st in SESSION_TYPES.items()
            if st.coaching_cr_name_contains] == ["thursday"]


# ---------- docx --------------------------------------------------------


def test_docx_lists_with_coach_line_under_r1(tmp_path):
    from docx import Document

    from pairings_docx import render_final_docx

    court = {
        "court_label": "7", "mode": "doubles",
        "players": ["A", "B", "C", "D"],
        "pairs": [["A", "B"], ["C", "D"]],
    }
    plan = {
        "date": "2026-10-22",
        "display_names": {},
        "rotations": [
            {"rotation_num": 1, "start_time": "19:30", "end_time": "20:15",
             "sit_outs": [], "with_coach": ["E", "F"], "courts": [court]},
            {"rotation_num": 2, "start_time": "20:15", "end_time": "20:55",
             "sit_outs": [], "with_coach": [], "courts": [court]},
        ],
        "notes": "",
    }
    out = tmp_path / "out.docx"
    render_final_docx(plan, TEMPLATE, out)
    texts = [p.text for p in Document(str(out)).paragraphs]
    i1 = texts.index("Rotation 1 (19:30-20:15)")
    i2 = texts.index("Rotation 2 (20:15-20:55)")
    assert texts.index("With coach: E, F") in range(i1, i2)
    assert sum(t.startswith("With coach") for t in texts) == 1
