"""A/B compare the production greedy/polish algorithm with the CP-SAT
spike on past committed sessions.

For each compatible entry in ``history.json`` we:

1. Take the attendees + court_labels from the entry.
2. Build a "history-up-to-just-before-this-entry" view (so the
   recency rule sees only sessions that actually preceded it — neither
   engine cheats with future knowledge).
3. Run both ``pairings.make_plan`` (current) and
   ``pairings_cpsat.make_plan_cpsat`` (spike).
4. Re-score BOTH outputs through the production scoring function
   (``pairings._rescore_layout``) so they're compared under one
   yardstick — even though CP-SAT internally optimises a simpler
   objective.

Prints a side-by-side summary.

Entries the CP-SAT spike can't handle (odd attendance, pins, sit-outs)
are skipped with a reason. Singles aren't supported in Phase 1.
"""

from __future__ import annotations

import json
import time
from datetime import date as _date
from pathlib import Path

from pairings import (
    UNKNOWN_RATING,
    _rescore_layout,
    _build_ratings,
    load_history,
    load_players,
    make_plan,
    recent_pair_weights,
)
from pairings_cpsat import make_plan_cpsat


PROJECT_ROOT = Path(__file__).parent
HISTORY_PATH = PROJECT_ROOT / "history.json"
PLAYERS_PATH = PROJECT_ROOT / "players.json"


def _is_compatible(entry: dict) -> tuple[bool, str]:
    """Can the CP-SAT spike handle this entry? Return (ok, reason)."""
    attendees = entry.get("attendees") or []
    court_labels = entry.get("court_labels") or []
    if not attendees or not court_labels:
        return False, "missing attendees or court_labels"
    if len(attendees) != 4 * len(court_labels):
        return (
            False,
            f"not all-doubles: {len(attendees)} attendees, "
            f"{len(court_labels)} courts (need 4×courts)",
        )
    # Sit-outs (would indicate odd attendance somewhere mid-evening).
    for rot in entry.get("rotations", []):
        if rot.get("sit_outs"):
            return False, "has sit-outs"
        for c in rot.get("courts", []):
            if c.get("mode") == "singles":
                return False, "has singles courts"
            if c.get("pinned"):
                return False, "has pinned courts"
    return True, ""


def _score_plan_through_current_scoring(plan, weekly_pair_penalties, genders):
    """Pass the plan's 4-player-per-court groupings through
    ``_rescore_layout`` to get a production-yardstick total score plus
    per-rotation breakdown. Lets _rescore_layout pick the optimal pair
    split per court — same treatment for both engines."""
    layout = [
        [list(c.players) for c in rot.courts]
        for rot in plan.rotations
    ]
    rotation_modes = [
        [c.mode for c in rot.courts] for rot in plan.rotations
    ]
    rotation_labels = [
        [c.court_label for c in rot.courts] for rot in plan.rotations
    ]
    rotation_sit_outs = [
        list(rot.sit_outs) for rot in plan.rotations
    ]
    total, per_rot, _ = _rescore_layout(
        layout,
        rotation_modes=rotation_modes,
        rotation_labels=rotation_labels,
        rotation_sit_outs=rotation_sit_outs,
        weekly_pair_penalties=weekly_pair_penalties,
        ratings=plan.ratings,
        genders=genders,
    )
    return total, per_rot


def _summarise_breakdown(per_rot) -> str:
    """Sum points by rule across all rotations, return a short
    "key=points, key=points" string sorted by points desc."""
    totals: dict[str, int] = {}
    for pr in per_rot:
        for it in pr.get("breakdown_items", []):
            rule = it.get("rule", "?")
            totals[rule] = totals.get(rule, 0) + int(it.get("points") or 0)
    parts = sorted(totals.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{k}={v}" for k, v in parts)


def compare_entry(
    history: list[dict],
    entry_idx: int,
    *,
    cpsat_time_limit: float = 30.0,
    seed: int = 42,
) -> dict:
    """Run both engines on the entry at ``entry_idx``, using only
    history[:entry_idx] as the recency window (so neither engine sees
    its own session as 'recent past'). Returns a result row."""
    entry = history[entry_idx]
    attendees = list(entry["attendees"])
    court_labels = list(entry["court_labels"])
    entry_date = entry.get("date")
    parsed_date = None
    try:
        parsed_date = _date.fromisoformat(entry_date)
    except (TypeError, ValueError):
        parsed_date = None

    # Write a "history slice" to a temp file so both engines see the same
    # truncated history.
    sliced_path = PROJECT_ROOT / ".tmp_history_slice.json"
    sliced_path.write_text(
        json.dumps(history[:entry_idx], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    players = load_players(PLAYERS_PATH)
    ratings = _build_ratings(players)
    genders = {
        n: (str(info.get("gender", "?")).strip().upper() or "?")
        for n, info in players.items()
    }
    weekly_pair_penalties = recent_pair_weights(
        history[:entry_idx], today=parsed_date,
    )

    # Production engine.
    t0 = time.perf_counter()
    plan_current = make_plan(
        attendees,
        players_path=players,
        history_path=str(sliced_path),
        court_labels=court_labels,
        num_rotations=int(entry.get("num_rotations") or 3),
        seed=seed,
        today=parsed_date,
    )
    current_wall = round(time.perf_counter() - t0, 2)

    # CP-SAT engine.
    t0 = time.perf_counter()
    try:
        plan_cpsat = make_plan_cpsat(
            attendees=attendees,
            players_path=players,
            history_path=str(sliced_path),
            court_labels=court_labels,
            num_rotations=int(entry.get("num_rotations") or 3),
            today=parsed_date,
            time_limit_seconds=cpsat_time_limit,
            seed=seed,
        )
    except (NotImplementedError, RuntimeError) as e:
        sliced_path.unlink(missing_ok=True)
        return {
            "date": entry_date,
            "attendees": len(attendees),
            "courts": len(court_labels),
            "cpsat_error": str(e),
        }
    cpsat_wall = round(time.perf_counter() - t0, 2)

    sliced_path.unlink(missing_ok=True)

    # Re-score both under the production scoring function.
    current_total, current_per_rot = _score_plan_through_current_scoring(
        plan_current, weekly_pair_penalties, genders,
    )
    cpsat_total, cpsat_per_rot = _score_plan_through_current_scoring(
        plan_cpsat, weekly_pair_penalties, genders,
    )

    return {
        "date": entry_date,
        "attendees": len(attendees),
        "courts": len(court_labels),
        "current_score": current_total,
        "current_wall": current_wall,
        "current_breakdown": _summarise_breakdown(current_per_rot),
        "cpsat_score": cpsat_total,
        "cpsat_wall": cpsat_wall,
        "cpsat_breakdown": _summarise_breakdown(cpsat_per_rot),
        "winner": (
            "tie" if current_total == cpsat_total
            else "current" if current_total < cpsat_total
            else "cpsat"
        ),
    }


def main(max_entries: int = 10, cpsat_time_limit: float = 30.0):
    history = load_history(HISTORY_PATH)
    print(f"Loaded {len(history)} history entries from {HISTORY_PATH}.\n")

    candidates: list[int] = []
    skipped: list[tuple[str, str]] = []
    # Walk newest → oldest, pick the most recent N compatible ones.
    for idx in range(len(history) - 1, -1, -1):
        ok, reason = _is_compatible(history[idx])
        if ok:
            candidates.append(idx)
            if len(candidates) >= max_entries:
                break
        else:
            skipped.append((history[idx].get("date", "?"), reason))

    if skipped:
        print("Skipped (CP-SAT spike incompatible):")
        for d, r in skipped:
            print(f"  {d}: {r}")
        print()

    if not candidates:
        print("No CP-SAT-compatible history entries to compare against.")
        return

    candidates.reverse()  # chronological for the table
    print(
        f"Comparing {len(candidates)} session(s) — "
        f"production make_plan vs CP-SAT spike. "
        f"Both re-scored under the current scoring function."
    )
    print()
    print(
        f"{'Date':<14}{'N':>4}{'C':>4}"
        f"{'Current':>9}{'CP-SAT':>8}"
        f"{'Diff':>6}"
        f"{'Cur(s)':>8}{'CP(s)':>8}{'Winner':>10}"
    )
    print("-" * 78)
    rows = []
    for idx in candidates:
        row = compare_entry(history, idx, cpsat_time_limit=cpsat_time_limit)
        rows.append(row)
        if "cpsat_error" in row:
            print(
                f"{row['date']:<14}{row['attendees']:>4}{row['courts']:>4}"
                f"   CP-SAT error: {row['cpsat_error']}"
            )
            continue
        delta = row["cpsat_score"] - row["current_score"]
        print(
            f"{row['date']:<14}{row['attendees']:>4}{row['courts']:>4}"
            f"{row['current_score']:>9}{row['cpsat_score']:>8}"
            f"{delta:>+6}"
            f"{row['current_wall']:>8.1f}{row['cpsat_wall']:>8.1f}"
            f"{row['winner']:>10}"
        )

    # Aggregate
    valid = [r for r in rows if "cpsat_error" not in r]
    if valid:
        wins_current = sum(1 for r in valid if r["winner"] == "current")
        wins_cpsat = sum(1 for r in valid if r["winner"] == "cpsat")
        ties = sum(1 for r in valid if r["winner"] == "tie")
        avg_curr = sum(r["current_score"] for r in valid) / len(valid)
        avg_cpsat = sum(r["cpsat_score"] for r in valid) / len(valid)
        avg_curr_wall = sum(r["current_wall"] for r in valid) / len(valid)
        avg_cpsat_wall = sum(r["cpsat_wall"] for r in valid) / len(valid)
        print("-" * 78)
        print(
            f"Summary over {len(valid)} session(s): "
            f"current wins {wins_current}, cpsat wins {wins_cpsat}, "
            f"ties {ties}"
        )
        print(
            f"  Average score:  current={avg_curr:.1f}  cpsat={avg_cpsat:.1f}"
        )
        print(
            f"  Average wall:   current={avg_curr_wall:.1f}s  "
            f"cpsat={avg_cpsat_wall:.1f}s"
        )

    print()
    print("Per-rule breakdowns (rule_key=points across the whole evening):")
    for r in valid:
        print(f"  {r['date']}:")
        print(f"    current: {r['current_breakdown']}")
        print(f"    cpsat:   {r['cpsat_breakdown']}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--max-entries", type=int, default=10)
    p.add_argument("--cpsat-time-limit", type=float, default=30.0)
    args = p.parse_args()
    main(
        max_entries=args.max_entries,
        cpsat_time_limit=args.cpsat_time_limit,
    )
