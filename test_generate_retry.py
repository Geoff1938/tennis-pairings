"""Tests for the best-of-N retry behaviour in admin_bot._generate_with_retry.

Driven by a stub make_plan that returns canned PairingPlan-shaped
objects with controllable score/metric content — keeps the test fast
and free of Sheet / Roster / file-system dependencies."""

from __future__ import annotations

from types import SimpleNamespace

from admin_bot import (
    GENERATE_RETRY_THRESHOLD,
    _generate_with_retry,
    _merge_loser_work_into_winner,
    _plan_total_score,
)


def _fake_plan(total_score: int, *, permutations: int = 100,
               wall_seconds: float = 10.0):
    """Tiny PairingPlan-look-alike with just the metric fields the
    retry helper inspects."""
    return SimpleNamespace(
        metrics={
            "rotations": [
                {"best_score": total_score},
            ],
            "wall_seconds": wall_seconds,
            "multi_seed": {
                "total_permutations_tried": permutations,
                "wall_seconds": wall_seconds,
            },
        },
    )


# ---------- helpers in isolation ----------------------------------------


def test_plan_total_score_sums_per_rotation_best_scores():
    plan = SimpleNamespace(metrics={"rotations": [
        {"best_score": 10}, {"best_score": 20}, {"best_score": 5},
    ]})
    assert _plan_total_score(plan) == 35


def test_plan_total_score_handles_missing_or_none_fields():
    plan = SimpleNamespace(metrics={"rotations": [
        {"best_score": None}, {"best_score": 7}, {},
    ]})
    assert _plan_total_score(plan) == 7


def test_merge_folds_loser_permutations_and_wall_into_winner():
    winner = _fake_plan(40, permutations=100, wall_seconds=10.0)
    loser = _fake_plan(120, permutations=80, wall_seconds=8.5)
    _merge_loser_work_into_winner(winner, loser)
    assert winner.metrics["wall_seconds"] == 18.5
    assert winner.metrics["multi_seed"]["total_permutations_tried"] == 180
    assert winner.metrics["multi_seed"]["wall_seconds"] == 18.5


# ---------- _generate_with_retry behaviour ------------------------------


def test_no_retry_when_first_run_under_threshold():
    """Single make_plan call; the notice callback is NEVER invoked."""
    plans_returned = []
    notices_fired = []

    def stub_make_plan(**kwargs):
        p = _fake_plan(GENERATE_RETRY_THRESHOLD - 1, permutations=50)
        plans_returned.append(p)
        return p

    result = _generate_with_retry(
        notice_callback=lambda: notices_fired.append("fired"),
        make_plan_fn=stub_make_plan,
        seed=42,
    )
    assert len(plans_returned) == 1
    assert result is plans_returned[0]
    assert notices_fired == []  # short-circuit; no retry message


def test_retry_when_first_run_at_threshold():
    """The condition is >= threshold — exact-threshold scores retry."""
    calls = []

    def stub_make_plan(**kwargs):
        calls.append(kwargs.get("seed"))
        # First call at threshold; second call lower.
        return _fake_plan(
            GENERATE_RETRY_THRESHOLD if len(calls) == 1 else 30,
        )

    notices = []
    _generate_with_retry(
        notice_callback=lambda: notices.append("fired"),
        make_plan_fn=stub_make_plan,
        seed=7,
    )
    assert len(calls) == 2
    assert notices == ["fired"]
    # Retry seed = original + 1.
    assert calls == [7, 8]


def test_retry_when_first_run_above_threshold_picks_lower():
    """The lower-scoring plan wins; loser's permutations get folded in."""
    plan_a = _fake_plan(150, permutations=100, wall_seconds=10.0)
    plan_b = _fake_plan(40, permutations=120, wall_seconds=12.0)
    plans = [plan_a, plan_b]
    calls = []

    def stub_make_plan(**kwargs):
        calls.append(kwargs.get("seed"))
        return plans.pop(0)

    notices = []
    winner = _generate_with_retry(
        notice_callback=lambda: notices.append("fired"),
        make_plan_fn=stub_make_plan,
        seed=10,
    )
    assert winner is plan_b
    assert _plan_total_score(winner) == 40
    # Loser's work folded into winner's metrics.
    assert winner.metrics["multi_seed"]["total_permutations_tried"] == 220
    assert winner.metrics["wall_seconds"] == 22.0
    assert notices == ["fired"]


def test_retry_keeps_first_when_second_is_worse():
    """If the second attempt scores higher, the first is kept and
    the second's wall+permutations are folded into the kept plan."""
    plan_a = _fake_plan(150, permutations=100, wall_seconds=10.0)
    plan_b = _fake_plan(200, permutations=80, wall_seconds=8.0)
    plans = [plan_a, plan_b]

    def stub_make_plan(**kwargs):
        return plans.pop(0)

    winner = _generate_with_retry(
        notice_callback=lambda: None,
        make_plan_fn=stub_make_plan,
        seed=10,
    )
    assert winner is plan_a
    assert winner.metrics["multi_seed"]["total_permutations_tried"] == 180
    assert winner.metrics["wall_seconds"] == 18.0


def test_retry_uses_none_seed_when_first_seed_was_none():
    """When the caller didn't pin a seed, the retry should also pass
    None so make_plan picks a fresh random seed (rather than (None+1)
    which would be a type error)."""
    seeds_seen = []

    def stub_make_plan(**kwargs):
        seeds_seen.append(kwargs.get("seed"))
        return _fake_plan(150)  # both runs at 150 → retry triggered

    _generate_with_retry(
        notice_callback=lambda: None,
        make_plan_fn=stub_make_plan,
        seed=None,
    )
    assert seeds_seen == [None, None]


def test_retry_notice_failure_does_not_block_retry():
    """A bad notice_callback (e.g. bridge unreachable) must not stop
    the retry from running."""
    plans = [_fake_plan(150), _fake_plan(40)]

    def stub_make_plan(**kwargs):
        return plans.pop(0)

    def boom():
        raise RuntimeError("bridge down")

    winner = _generate_with_retry(
        notice_callback=boom,
        make_plan_fn=stub_make_plan,
        seed=1,
    )
    assert _plan_total_score(winner) == 40  # retry still happened
