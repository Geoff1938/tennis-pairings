"""Tests for the parallel best-of-N behaviour in admin_bot._generate_parallel.

Driven by a stub make_plan that returns canned PairingPlan-shaped
objects with controllable score / metric content — keeps the tests
fast and free of Sheet / Roster / process-pool dependencies. The
production path uses ProcessPoolExecutor; the tests inject
``make_plan_fn`` to run sequentially within the test process."""

from __future__ import annotations

from types import SimpleNamespace

from admin_bot import (
    GENERATE_PARALLEL_WORKERS,
    _generate_parallel,
    _merge_parallel_losers_into_winner,
    _plan_total_score,
)


def _fake_plan(
    total_score: int,
    *,
    permutations: int = 100,
    wall_seconds: float = 10.0,
):
    """Tiny PairingPlan-look-alike."""
    return SimpleNamespace(
        metrics={
            "rotations": [{"best_score": total_score}],
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


def test_merge_takes_max_walltime_and_sums_permutations():
    """Parallel workers race — winner's reported wall_seconds should
    be the SLOWEST worker's wall (the admin's actual wait), not the
    sum of all of them. Permutations are summed (total exploration)."""
    winner = _fake_plan(40, permutations=100, wall_seconds=10.0)
    losers = [
        _fake_plan(80, permutations=80, wall_seconds=8.5),
        _fake_plan(120, permutations=90, wall_seconds=12.0),
        _fake_plan(60, permutations=110, wall_seconds=9.2),
    ]
    _merge_parallel_losers_into_winner(winner, losers)
    # Max wall, not sum.
    assert winner.metrics["wall_seconds"] == 12.0
    assert winner.metrics["multi_seed"]["wall_seconds"] == 12.0
    # Sum permutations.
    assert (
        winner.metrics["multi_seed"]["total_permutations_tried"]
        == 100 + 80 + 90 + 110
    )


def test_merge_with_zero_losers_is_a_noop():
    winner = _fake_plan(40, permutations=100, wall_seconds=10.0)
    _merge_parallel_losers_into_winner(winner, [])
    assert winner.metrics["wall_seconds"] == 10.0
    assert winner.metrics["multi_seed"]["total_permutations_tried"] == 100


# ---------- _generate_parallel behaviour --------------------------------


def test_picks_lowest_scoring_plan_across_workers():
    """Among N workers, the winner is the one with the lowest re-scored
    total. Loser work is folded in."""
    scores = [150, 30, 80, 120]
    seeds_seen: list[int] = []

    def stub_make_plan(**kwargs):
        seeds_seen.append(kwargs["seed"])
        idx = kwargs["seed"] - 42  # default base
        return _fake_plan(
            scores[idx], permutations=10 * (idx + 1), wall_seconds=5.0 + idx,
        )

    winner = _generate_parallel(
        num_workers=4,
        make_plan_fn=stub_make_plan,
    )
    assert _plan_total_score(winner) == 30
    # All four seeds were tried.
    assert sorted(seeds_seen) == [42, 43, 44, 45]
    # Permutations totalled across all 4 workers.
    assert winner.metrics["multi_seed"]["total_permutations_tried"] == (
        10 + 20 + 30 + 40
    )
    # Wall time = max across workers.
    assert winner.metrics["wall_seconds"] == max(5.0, 6.0, 7.0, 8.0)


def test_uses_base_seed_42_when_seed_is_none():
    seeds_seen = []

    def stub_make_plan(**kwargs):
        seeds_seen.append(kwargs["seed"])
        return _fake_plan(50)

    _generate_parallel(
        num_workers=3,
        make_plan_fn=stub_make_plan,
        seed=None,
    )
    assert sorted(seeds_seen) == [42, 43, 44]


def test_uses_explicit_seed_base_when_provided():
    seeds_seen = []

    def stub_make_plan(**kwargs):
        seeds_seen.append(kwargs["seed"])
        return _fake_plan(50)

    _generate_parallel(
        num_workers=3,
        make_plan_fn=stub_make_plan,
        seed=100,
    )
    assert sorted(seeds_seen) == [100, 101, 102]


def test_one_worker_failure_doesnt_block_the_run():
    """If one worker raises (e.g. the make_plan call hit an edge case),
    the others still finish and the best of THOSE wins."""
    call_count = [0]

    def stub_make_plan(**kwargs):
        call_count[0] += 1
        if kwargs["seed"] == 43:
            raise RuntimeError("oh no")
        return _fake_plan(70 if kwargs["seed"] == 42 else 40)

    winner = _generate_parallel(
        num_workers=3,
        make_plan_fn=stub_make_plan,
    )
    # 3 attempts; 1 failed; 2 succeeded; lower score wins.
    assert call_count[0] == 3
    assert _plan_total_score(winner) == 40


def test_all_workers_failing_raises():
    """If every worker raises, _generate_parallel re-raises with
    context — better than returning nothing."""
    import pytest

    def stub_make_plan(**kwargs):
        raise RuntimeError(f"boom {kwargs['seed']}")

    with pytest.raises(RuntimeError, match="parallel make_plan workers failed"):
        _generate_parallel(num_workers=3, make_plan_fn=stub_make_plan)


def test_default_num_workers_matches_constant():
    """When num_workers is None, the constant is used (4 by default
    so the Pi's 4 cores get filled)."""
    seeds_seen = []

    def stub_make_plan(**kwargs):
        seeds_seen.append(kwargs["seed"])
        return _fake_plan(50)

    _generate_parallel(make_plan_fn=stub_make_plan, seed=42)
    assert len(seeds_seen) == GENERATE_PARALLEL_WORKERS
    assert sorted(seeds_seen) == [
        42 + i for i in range(GENERATE_PARALLEL_WORKERS)
    ]


def test_num_workers_clamped_to_at_least_one():
    """num_workers=0 or negative is silently bumped to 1 — would
    otherwise leave plans empty and raise."""
    seeds_seen = []

    def stub_make_plan(**kwargs):
        seeds_seen.append(kwargs["seed"])
        return _fake_plan(50)

    _generate_parallel(
        num_workers=0,
        make_plan_fn=stub_make_plan,
        seed=42,
    )
    assert len(seeds_seen) == 1


def test_make_plan_kwargs_are_forwarded_with_seed_override():
    """The seed kwarg gets overridden per worker; everything else
    passes through unchanged."""
    received: list[dict] = []

    def stub_make_plan(**kwargs):
        received.append(dict(kwargs))
        return _fake_plan(50)

    _generate_parallel(
        num_workers=2,
        make_plan_fn=stub_make_plan,
        seed=42,
        attendees=["A", "B", "C", "D"],
        court_labels=["5"],
        num_rotations=3,
        custom_field="hello",
    )
    assert len(received) == 2
    for kw in received:
        assert kw["attendees"] == ["A", "B", "C", "D"]
        assert kw["court_labels"] == ["5"]
        assert kw["num_rotations"] == 3
        assert kw["custom_field"] == "hello"
    # The two seeds are 42 and 43.
    assert sorted(kw["seed"] for kw in received) == [42, 43]
