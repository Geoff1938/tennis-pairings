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
    """Can the CP-SAT spike handle this entry's attendee+court shape?
    Return (ok, reason). Only the attendees / court_labels matter — we
    regenerate the rotations from scratch in the comparison harness, so
    the old plan's mode/sit-outs/pins are irrelevant."""
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
    return True, ""


def _trim_to_all_doubles(entry: dict) -> dict | None:
    """Return a modified copy of ``entry`` re-shaped to all-doubles.

    The real sessions with singles courts have ``N_attendees`` players
    on ``N_courts`` courts where ``N_attendees < 4 * N_courts`` (some
    courts seat 2 not 4). To compare CP-SAT (which is all-doubles in
    Phase 1) we drop both:
      * ``N_attendees mod 4`` attendees (the last ones, deterministic);
      * Enough trailing court labels to match ``new_N_attendees / 4``.

    Simulates "a couple of late drop-outs AND one less court booked"
    so the comparison runs against a realistic-but-trimmed Westside
    session. Returns None when no trim would help (already
    all-doubles, or attendees too few to fill even one court).
    """
    attendees = list(entry.get("attendees") or [])
    court_labels = list(entry.get("court_labels") or [])
    if not attendees or not court_labels:
        return None
    if len(attendees) == 4 * len(court_labels):
        return None  # already compatible
    if len(attendees) < 4:
        return None  # can't fill even one doubles court
    # Drop attendees down to a multiple of 4.
    dropped_atts = len(attendees) % 4
    new_atts = attendees[: len(attendees) - dropped_atts] if dropped_atts else attendees
    target_courts = len(new_atts) // 4
    if target_courts < 1 or target_courts > len(court_labels):
        return None
    dropped_courts = len(court_labels) - target_courts
    trimmed = dict(entry)
    trimmed["attendees"] = new_atts
    trimmed["court_labels"] = court_labels[:target_courts]
    trimmed["_trimmed_attendees"] = dropped_atts
    trimmed["_trimmed_courts"] = dropped_courts
    return trimmed


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
    prod_seeds: int = 1,
) -> dict:
    """Run both engines on the entry at ``entry_idx``, using only
    history[:entry_idx] as the recency window (so neither engine sees
    its own session as 'recent past'). Returns a result row.

    ``prod_seeds`` controls how many production runs we do (with
    seeds 42, 43, ... 42+prod_seeds-1). We report the BEST score
    across those runs as the "current_score" — gives production the
    benefit of variance. CP-SAT is deterministic; one run."""
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

    # Production engine — run multiple times to surface its variance.
    prod_runs: list[tuple[int, float, "PairingPlan", list]] = []
    for k in range(max(1, prod_seeds)):
        t0 = time.perf_counter()
        plan_k = make_plan(
            attendees,
            players_path=players,
            history_path=str(sliced_path),
            court_labels=court_labels,
            num_rotations=int(entry.get("num_rotations") or 3),
            seed=seed + k,
            today=parsed_date,
        )
        wall_k = round(time.perf_counter() - t0, 2)
        score_k, per_rot_k = _score_plan_through_current_scoring(
            plan_k, weekly_pair_penalties, genders,
        )
        prod_runs.append((score_k, wall_k, plan_k, per_rot_k))
    # Pick the BEST production run (lowest re-scored total).
    prod_runs.sort(key=lambda t: t[0])
    current_total, _, plan_current, current_per_rot = prod_runs[0]
    current_wall = round(sum(t[1] for t in prod_runs), 2)
    prod_scores = [t[0] for t in prod_runs]
    prod_worst = max(prod_scores)

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

    # CP-SAT re-scored under production rules.
    cpsat_total, cpsat_per_rot = _score_plan_through_current_scoring(
        plan_cpsat, weekly_pair_penalties, genders,
    )

    return {
        "date": entry_date,
        "attendees": len(attendees),
        "prod_scores": prod_scores,
        "prod_worst": prod_worst,
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


def main(
    max_entries: int = 10,
    cpsat_time_limit: float = 30.0,
    trim_to_all_doubles: bool = False,
    prod_seeds: int = 1,
):
    history = load_history(HISTORY_PATH)
    print(f"Loaded {len(history)} history entries from {HISTORY_PATH}.\n")

    candidates: list[int] = []
    trimmed: list[tuple[int, dict, int]] = []  # (idx, modified_entry, dropped_n)
    skipped: list[tuple[str, str]] = []

    # Walk newest → oldest, pick the most recent N compatible ones.
    for idx in range(len(history) - 1, -1, -1):
        ok, reason = _is_compatible(history[idx])
        if ok:
            candidates.append(idx)
        elif trim_to_all_doubles:
            modified = _trim_to_all_doubles(history[idx])
            if modified is not None and _is_compatible(modified)[0]:
                trimmed.append((idx, modified, 0))
                candidates.append(idx)
            else:
                skipped.append((history[idx].get("date", "?"), reason))
        else:
            skipped.append((history[idx].get("date", "?"), reason))
        if len(candidates) >= max_entries:
            break

    if skipped:
        print("Skipped (CP-SAT spike incompatible, not trimmable):")
        for d, r in skipped:
            print(f"  {d}: {r}")
        print()
    if trimmed:
        print(
            f"Trimmed {len(trimmed)} entries to all-doubles "
            "(dropping trailing attendees + courts to simulate a "
            "late drop-out and one fewer court):"
        )
        for idx, mod, _ in trimmed:
            d_a = int(mod.get("_trimmed_attendees") or 0)
            d_c = int(mod.get("_trimmed_courts") or 0)
            print(
                f"  {mod.get('date')}: -{d_a} attendees, -{d_c} courts "
                f"-> {len(mod['attendees'])} players, "
                f"{len(mod['court_labels'])} courts"
            )
        print()

    if not candidates:
        print("No CP-SAT-compatible history entries to compare against.")
        return

    # Build the effective entries: trimmed where applicable, original otherwise.
    trimmed_by_idx = {idx: mod for idx, mod, _ in trimmed}
    # Override history[idx] in-place for the comparison run so
    # compare_entry sees the trimmed attendees. We restore originals
    # after each compare so the recency-window slicing for LATER (older)
    # entries still uses the real attendee lists.
    history_local = [dict(h) for h in history]  # shallow-copy entries
    for idx, mod, _ in trimmed:
        history_local[idx] = mod

    candidates.reverse()  # chronological for the table
    print(
        f"Comparing {len(candidates)} session(s) — "
        f"production make_plan vs CP-SAT spike. "
        f"Both re-scored under the current scoring function."
    )
    print()
    cur_label = (
        "Cur(best-worst)" if prod_seeds > 1 else "Current"
    )
    print(
        f"{'Date':<14}{'N':>4}{'C':>4}"
        f"{cur_label:>11}{'CP-SAT':>8}"
        f"{'Diff':>6}"
        f"{'Cur(s)':>8}{'CP(s)':>8}{'Winner':>10}"
    )
    print("-" * 78)
    rows = []
    for idx in candidates:
        row = compare_entry(
            history_local, idx,
            cpsat_time_limit=cpsat_time_limit,
            prod_seeds=prod_seeds,
        )
        rows.append(row)
        if "cpsat_error" in row:
            print(
                f"{row['date']:<14}{row['attendees']:>4}{row['courts']:>4}"
                f"   CP-SAT error: {row['cpsat_error']}"
            )
            continue
        delta = row["cpsat_score"] - row["current_score"]
        prod_str = str(row["current_score"])
        if "prod_scores" in row and len(row["prod_scores"]) > 1:
            prod_str = (
                f"{row['current_score']}-{row['prod_worst']}"
            )
        print(
            f"{row['date']:<14}{row['attendees']:>4}{row['courts']:>4}"
            f"{prod_str:>11}{row['cpsat_score']:>8}"
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
    p.add_argument(
        "--trim-to-all-doubles", action="store_true",
        help=(
            "When a history entry has odd attendance (i.e. used "
            "singles courts), trim its attendee list to the nearest "
            "multiple of 4 so the CP-SAT spike can handle it. "
            "Simulates a couple of late drop-outs."
        ),
    )
    p.add_argument(
        "--prod-seeds", type=int, default=1,
        help=(
            "Run the production engine this many times per test case "
            "(with seeds 42, 43, ...) and use the BEST result. "
            "Surfaces production's variance — useful for checking "
            "whether one-shot losses to CP-SAT would survive a "
            "best-of-N replay."
        ),
    )
    args = p.parse_args()
    main(
        max_entries=args.max_entries,
        cpsat_time_limit=args.cpsat_time_limit,
        trim_to_all_doubles=args.trim_to_all_doubles,
        prod_seeds=args.prod_seeds,
    )
