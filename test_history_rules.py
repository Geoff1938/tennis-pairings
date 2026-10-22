"""Recent-history rule: past partners and past opponents are penalised
separately (former partners facing each other is fine), and seed
candidates are compared on the full whole-evening score."""
from __future__ import annotations

import json
from datetime import date

from pairings import (
    Court,
    _make_plan_one,
    _plan_total,
    _explain_score_items,
    _score_doubles_court,
    _score_doubles_courts,
    _score_singles_courts,
    make_plan,
    opponent_key,
    recent_pair_weights,
)

TODAY = date(2026, 10, 22)


def _entry(date_iso, *courts):
    return {"date": date_iso, "rotations": [{"courts": list(courts)}]}


def _doubles(pa, pb):
    return {"mode": "doubles", "players": [*pa, *pb],
            "pairs": [list(pa), list(pb)]}


def _singles(a, b):
    return {"mode": "singles", "players": [a, b], "pairs": [[a, b]]}


def test_weights_split_partners_and_opponents():
    w = recent_pair_weights(
        [_entry("2026-10-20", _doubles(("A", "B"), ("C", "D")),
                _singles("E", "F"))],
        today=TODAY,
    )
    assert w[frozenset("AB")] == 18 and w[frozenset("CD")] == 18
    for a, c in (("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")):
        assert w[opponent_key(frozenset((a, c)))] == 18
    # Former opponents are not recorded as partners, and vice versa.
    assert frozenset("AC") not in w
    assert opponent_key(frozenset("AB")) not in w
    # A singles match is a matchup, not a partnership.
    assert frozenset("EF") not in w
    assert w[opponent_key(frozenset("EF"))] == 18


def _court(pa, pb):
    return Court(court_label="5", mode="doubles", players=[*pa, *pb],
                 pairs=[pa, pb])


def _score(court, weights):
    ratings = {p: 5 for p in "ABCD"}
    genders = {p: "?" for p in "ABCD"}
    return _score_doubles_court(court, weights, set(), set(), set(),
                                ratings, genders)


def test_former_opponents_facing_again_is_penalised():
    w = recent_pair_weights(
        [_entry("2026-10-20", _doubles(("A", "B"), ("C", "D")))],
        today=TODAY,
    )
    base = _score(_court(("A", "B"), ("C", "D")), {})
    # Same four, same pairs: 2 partner repeats + 4 opponent repeats.
    assert _score(_court(("A", "B"), ("C", "D")), w) - base == 6 * 18


def test_former_partners_facing_each_other_is_free():
    w = recent_pair_weights(
        [_entry("2026-10-20", _doubles(("A", "B"), ("C", "D")))],
        today=TODAY,
    )
    # A-C v B-D: A/B and C/D were partners, now opponents (free); A v C
    # and B v D were opponents, now partners (free); A v D and B v C
    # were opponents and still are (penalised).
    court = _court(("A", "C"), ("B", "D"))
    assert _score(court, w) - _score(court, {}) == 2 * 18
    items = _explain_score_items(
        [court], w, set(), set(), set(), {p: 5 for p in "ABCD"},
        {p: "?" for p in "ABCD"},
    )
    hist = [it for it in items if it["rule"].startswith("weekly_history")]
    assert {it["rule"] for it in hist} == {"weekly_history_opponents"}
    assert sorted(tuple(it["pair"]) for it in hist) == [("A", "D"),
                                                       ("B", "C")]


def test_singles_rematch_is_penalised():
    w = recent_pair_weights([_entry("2026-10-20", _singles("E", "F"))],
                            today=TODAY)
    c = Court(court_label="6", mode="singles", players=["E", "F"],
              pairs=[("E", "F")])
    assert _score_singles_courts([c], set(), set(), None, None, w) == 18
    assert _score_singles_courts([c], set(), set()) == 0


# ---------- seed candidates include the whole-evening rules ------------


def _files(tmp_path, names):
    pp, hp = tmp_path / "players.json", tmp_path / "history.json"
    pp.write_text(json.dumps({
        n: {"gender": "?", "rating": 5, "notes": "", "singles": ""}
        for n in names
    }))
    hp.write_text("[]")
    return pp, hp


def test_seed_plan_score_includes_whole_evening_rules(tmp_path):
    # One hard court ("1") for two rotations: all four players get two
    # hard-court rotations, a whole-evening rule the seed comparison
    # used to ignore.
    names = ["A", "B", "C", "D"]
    pp, hp = _files(tmp_path, names)
    plan = _make_plan_one(
        attendees=names, players_path=pp, history_path=hp,
        court_labels=["1"], num_rotations=2, seed=1, today=TODAY,
        verbose=False,
    )
    rules = {it["rule"] for r in plan.metrics["rotations"]
             for it in r["breakdown_items"]}
    assert "hard_court_repeat" in rules
    assert _plan_total(plan) >= 4 * 50


def test_multi_seed_total_is_the_full_score(tmp_path):
    names = ["A", "B", "C", "D"]
    pp, hp = _files(tmp_path, names)
    plan = make_plan(names, pp, hp, court_labels=["1"], num_rotations=2,
                     seed=1, num_seed_attempts=3, polish=False,
                     today=TODAY)
    assert plan.metrics["multi_seed"]["chosen_total"] == _plan_total(plan)
    assert _plan_total(plan) >= 4 * 50


# ---------- score and explanation agree --------------------------------


def test_score_matches_explanation_on_random_layouts():
    """The scorer and the explanation are separate code paths (the
    scorer is the hot loop); this keeps them from drifting apart."""
    import random

    rng = random.Random(7)
    names = [f"P{i}" for i in range(16)]
    for _ in range(3000):
        ratings = {n: rng.randint(1, 10) for n in names}
        genders = {n: rng.choice("MFF?") for n in names}
        rng.shuffle(names)
        courts = [
            Court(court_label=str(i + 4), mode="doubles",
                  players=names[4 * i:4 * i + 4],
                  pairs=[tuple(names[4 * i:4 * i + 2]),
                         tuple(names[4 * i + 2:4 * i + 4])],
                  pinned=rng.random() < 0.2)
            for i in range(3)
        ]
        courts.append(Court(court_label="9", mode="singles",
                            players=names[12:14],
                            pairs=[tuple(names[12:14])]))
        w = {}
        for _ in range(10):
            w[frozenset(rng.sample(names, 2))] = 18
            w[opponent_key(frozenset(rng.sample(names, 2)))] = 10
        ip = {frozenset(rng.sample(names, 2)) for _ in range(5)}
        io = {frozenset(rng.sample(names, 2)) for _ in range(5)}
        pc = {frozenset(rng.sample(names, 2)) for _ in range(8)}
        ub = {n: rng.choice([0, 0, 1, 2]) for n in names}
        pg = {n: rng.choice([0, 0, 1]) for n in names}
        ff = rng.choice([0, 1, 2])
        score = (
            _score_doubles_courts(courts, w, ip, io, pc, ratings, genders,
                                  ub, pg, ff)
            + _score_singles_courts(courts, io, pc, ratings, ub, w)
        )
        items = _explain_score_items(courts, w, ip, io, pc, ratings,
                                     genders, ub, pg, ff)
        assert score == sum(it["points"] for it in items)
