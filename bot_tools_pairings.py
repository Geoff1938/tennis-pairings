"""Boris tools for pairings: generate (parallel best-of-N),
edit, commit/undo, history, final document and rules PDF.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from roster import Roster
from bot_bridge import send_doc_to_group, send_to_group
from bot_config import (
    BOT_REPLY_PREFIX,
    DOCX_ATTACHMENT_CAPTION,
    FINAL_DOCX_OUTPUT_DIR,
    GENERATE_PARALLEL_WORKERS,
    HISTORY_PATH,
    ROOT,
    RULES_PDF_CAPTION,
    RULES_PDF_PATH,
    _CURRENT_GROUP_JID,
)


def _docx_template_for(session_type: str) -> Path:
    """Resolve the DOCX template path for a session type. Falls back to
    Thursday's template when the type is unknown or empty (legacy /
    manual starts)."""
    from session_types import SESSION_TYPES

    st = SESSION_TYPES.get(session_type) if session_type else None
    relpath = (
        st.docx_template_relpath
        if st is not None
        else SESSION_TYPES["thursday"].docx_template_relpath
    )
    return ROOT / relpath


def _docx_preamble_count_for(session_type: str) -> int:
    """How many template paragraphs to keep verbatim before the date
    heading. All three current session types use the Westside
    template which opens straight on the heading = 0; kept
    parameterised in case a future bespoke template re-introduces a
    preamble block."""
    from session_types import SESSION_TYPES

    st = SESSION_TYPES.get(session_type) if session_type else None
    return (
        st.docx_preamble_paragraph_count
        if st is not None
        else SESSION_TYPES["thursday"].docx_preamble_paragraph_count
    )


def _plan_total_score(plan) -> int:
    """Sum of per-rotation best_score from a PairingPlan's metrics.
    Accepts the live ``PairingPlan`` object — mirrors ``pairings._plan_total``
    but kept here so the parallel helper doesn't reach into pairings'
    private module."""
    return sum(
        int(r.get("best_score") or 0)
        for r in (plan.metrics.get("rotations") or [])
    )


def _merge_parallel_losers_into_winner(winner, losers: list) -> None:
    """Combine wall time + permutations from the LOSER plans into the
    WINNER's metrics so the rendered-to-WhatsApp footer ("tried N
    permutations in Xs") reflects the total effort.

    Wall time uses MAX across all workers — they ran in parallel, so
    the admin's actual elapsed wait is roughly the slowest worker.
    Permutations use SUM — total exploration across the four searches.
    """
    if not losers:
        return
    all_plans = [winner, *losers]
    all_walls = [
        float(p.metrics.get("wall_seconds") or 0) for p in all_plans
    ]
    winner.metrics["wall_seconds"] = round(max(all_walls), 2)

    w_ms = winner.metrics.setdefault("multi_seed", {})
    total_perms = int(w_ms.get("total_permutations_tried") or 0)
    for loser in losers:
        l_ms = loser.metrics.get("multi_seed") or {}
        total_perms += int(l_ms.get("total_permutations_tried") or 0)
    w_ms["total_permutations_tried"] = total_perms

    all_ms_walls = [
        float(p.metrics.get("multi_seed", {}).get("wall_seconds") or 0)
        for p in all_plans
    ]
    w_ms["wall_seconds"] = round(max(all_ms_walls), 2)


def _make_plan_worker(kwargs_dict: dict):
    """Module-level entry-point for ProcessPoolExecutor workers.

    Must be importable by spawned subprocesses (so no closure, no
    nested function). Re-imports ``pairings.make_plan`` on entry so
    each worker has a fresh module state.
    """
    from pairings import make_plan
    return make_plan(**kwargs_dict)


def _generate_parallel(
    *,
    num_workers: int | None = None,
    make_plan_fn=None,
    seed: int | None = None,
    **make_plan_kwargs,
):
    """Run ``make_plan`` in parallel across ``num_workers`` processes
    with different seeds, return the lowest-scoring plan with the
    other workers' wall time + permutations folded into its metrics.

    Each worker is a separate Python process so they actually run on
    different CPU cores (no GIL contention). On the Pi (4 cores)
    setting num_workers=4 fills the box; the wall time is roughly
    the slowest single ``make_plan`` call, but the total exploration
    is num_workers × that of a single call.

    ``make_plan_fn`` is for tests — when supplied, workers run
    sequentially through this function instead of via the process
    pool. ``seed=None`` picks a random base, so a re-roll with nothing
    changed gives a fresh draft; pass an integer for repeatable output.

    Each make_plan tries the consecutive seeds ``seed, seed+1, …`` (up
    to MAX_SEED_ATTEMPTS of them), so worker bases are spaced that far
    apart — adjacent bases would make the workers repeat each other's
    seeds and overstate the search.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if num_workers is None:
        num_workers = GENERATE_PARALLEL_WORKERS
    num_workers = max(1, int(num_workers))

    from pairings import MAX_SEED_ATTEMPTS

    base_seed = (
        random.SystemRandom().randrange(2**31) if seed is None else int(seed)
    )
    worker_kwargs = []
    for k in range(num_workers):
        kw = dict(make_plan_kwargs)
        kw["seed"] = base_seed + k * MAX_SEED_ATTEMPTS
        worker_kwargs.append(kw)

    plans: list = []
    errors: list = []

    if make_plan_fn is not None:
        # Test path — sequential, with the injected function.
        for kw in worker_kwargs:
            try:
                plans.append(make_plan_fn(**kw))
            except Exception as e:
                errors.append(e)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [
                pool.submit(_make_plan_worker, kw) for kw in worker_kwargs
            ]
            for fut in as_completed(futures):
                try:
                    plans.append(fut.result())
                except Exception as e:
                    errors.append(e)

    if not plans:
        # All workers failed — surface the first error. A ValueError
        # is bad input (same for every seed), so re-raise it as-is for
        # tool_generate_pairings to report cleanly.
        value_errors = [e for e in errors if isinstance(e, ValueError)]
        if value_errors:
            raise value_errors[0]
        raise RuntimeError(
            f"All {num_workers} parallel make_plan workers failed; "
            f"first error: {errors[0]!r}"
        )

    plans.sort(key=_plan_total_score)
    winner, *losers = plans
    _merge_parallel_losers_into_winner(winner, losers)
    return winner


def _docx_header_text_for(session_type: str) -> str:
    """Page-header banner per session type. Falls back to Thursday."""
    from session_types import SESSION_TYPES

    st = SESSION_TYPES.get(session_type) if session_type else None
    return (
        st.docx_header_text
        if st is not None
        else SESSION_TYPES["thursday"].docx_header_text
    )


def _docx_basename_for(session_type: str) -> str:
    """Filename prefix for the rendered final doc (date is appended).
    Sessions sharing a template (Tue+Sat) still get distinct output
    filenames so the file says what kind of session it was for."""
    from session_types import SESSION_TYPES

    st = SESSION_TYPES.get(session_type) if session_type else None
    if st is None:
        return "Thursday Social Tennis"
    if st.key == "thursday":
        return "Thursday Social Tennis"
    if st.key == "tuesday":
        return "Tuesday Social Tennis"
    if st.key == "saturday":
        return "Saturday Social Tennis"
    return "Westside Social Tennis"


def tool_read_pairings_history(lookback: int = 4) -> list:
    if not HISTORY_PATH.exists():
        return []
    with HISTORY_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return data[-lookback:] if lookback else data


def tool_generate_pairings(
    attendee_names: Optional[list[str]] = None,
    court_labels: Optional[list] = None,
    num_courts: Optional[int] = None,
    num_rotations: Optional[int] = None,
    seed: Optional[int] = None,
    start_time_hhmm: Optional[str] = None,
    rotation_durations: Optional[list[int]] = None,
    singles_exclude: Optional[list[str]] = None,
    singles_include: Optional[list[str]] = None,
    pinned_singles: Optional[list[dict]] = None,
) -> dict:
    """Build a pairing plan and stash it as the session's draft.

    If ``attendee_names`` is None, uses the current session state (what
    the admin has set via start_tonight / add / remove). Same for
    ``court_labels``. The plan is saved to ``session_state.draft_plan``
    so the admin can iterate (swap_players / swap_rotations) before
    finalising via commit_plan.

    ``start_time_hhmm`` / ``rotation_durations`` default to the
    in-flight session_type's values (Tue/Thu evening 19:30 + 45/40/35;
    Saturday afternoon 14:00 + 45/40/35; Friday Tapas 19:00 +
    25/25/20/20) when not explicitly passed. A re-roll that passes
    none of ``num_rotations`` / ``start_time_hhmm`` /
    ``rotation_durations`` keeps the current draft's rotation count and
    times (so an earlier set_rotation_times survives the re-roll). When
    ``num_rotations`` differs from the session type's count and no
    durations are given, the session's usual span is split evenly.
    """
    from pairings import make_plan
    from session_state import get_tonight, set_draft_plan, set_phase
    from session_types import SESSION_TYPES

    state = get_tonight()
    names = list(attendee_names) if attendee_names is not None else list(state.attendees)
    labels = court_labels if court_labels is not None else (state.court_labels or None)
    # Resolve session-type defaults for start time + rotation durations
    # when the caller didn't pin them explicitly.
    st = SESSION_TYPES.get(state.session_type) if state.session_type else None
    draft_rots = (state.draft_plan or {}).get("rotations") or []
    if (
        num_rotations is None and start_time_hhmm is None
        and rotation_durations is None and draft_rots
    ):
        # Re-roll: keep the draft's rotation structure and times.
        from pairings import _minutes_between
        num_rotations = len(draft_rots)
        start_time_hhmm = draft_rots[0]["start_time"]
        rotation_durations = [
            _minutes_between(r["start_time"], r["end_time"])
            for r in draft_rots
        ]
    if num_rotations is None:
        num_rotations = st.num_rotations if st is not None else 3
    if start_time_hhmm is None:
        start_time_hhmm = st.start_time_hhmm if st is not None else "19:30"
    if rotation_durations is None and st is not None:
        if len(st.rotation_durations) == num_rotations:
            rotation_durations = list(st.rotation_durations)
        else:
            # Different rotation count: split the session's usual span
            # evenly, earlier rotations taking any spare minutes.
            span = sum(st.rotation_durations)
            base, extra = divmod(span, num_rotations)
            rotation_durations = [
                base + (1 if i < extra else 0) for i in range(num_rotations)
            ]
    if not names:
        return {
            "ok": False,
            "error": "no_attendees",
            "message": "No attendees supplied and session state is empty. "
            "Run start_tonight first, or pass attendee_names.",
        }

    # Heads-up: pairings generation can take up to a minute (multi-seed
    # greedy + hill-climb polish). Post a notice into the calling
    # channel before kicking off so the admin doesn't think the bot
    # has stalled. Best-effort — if we can't resolve a channel jid,
    # carry on without the notice.
    notice_jid = _CURRENT_GROUP_JID.get(None)
    if notice_jid:
        try:
            send_to_group(
                notice_jid,
                BOT_REPLY_PREFIX
                + "Generating pairings — this may take up to a minute…",
            )
        except Exception:
            pass
    if not labels and num_courts is None:
        return {
            "ok": False,
            "error": "no_courts",
            "message": "No court_labels in session state and num_courts not given. "
            "Set courts via set_courts_for_tonight or pass court_labels.",
        }

    # Pick up the late-court config (if any) from session state so it
    # auto-applies to the run — the admin sets it via set_late_court
    # and we honour it here without needing it on every call.
    late_court_arg: dict | None = None
    if state.late_court_label and state.late_court_first_rotation >= 1:
        late_court_arg = {
            "label": state.late_court_label,
            "first_rotation": state.late_court_first_rotation,
            "pinned_players": list(state.late_court_pinned_players),
        }
    # Coaching group (monthly Thursday): resolve its courts against
    # tonight's labels by court number so "5" matches "Court #5 -
    # Floodlit" whichever form the admin or CR used.
    coaching_arg: dict | None = None
    if state.coaching_rotations >= 1 and state.coaching_players:
        from pairings import _court_label_key
        by_key = {_court_label_key(str(l)): str(l) for l in (labels or [])}
        coaching_arg = {
            "players": [p for p in state.coaching_players if p in names],
            "courts": [
                by_key.get(_court_label_key(c), c)
                for c in state.coaching_courts
            ],
            "rotations": state.coaching_rotations,
        }
    # Same for any admin-pinned doubles match-ups (pin_doubles tool).
    pinned_doubles_arg: list[dict] | None = (
        list(state.pinned_doubles) if state.pinned_doubles else None
    )
    # Stamp the plan with the session's date (not the day the draft
    # happens to be generated — kickoff is often days ahead). Also
    # anchors the recent-history weighting to the session date.
    session_day = None
    if state.session_date:
        from datetime import date as _date
        session_day = _date.fromisoformat(state.session_date)

    # Parallel best-of-N: GENERATE_PARALLEL_WORKERS independent
    # make_plan calls with non-overlapping seeds, on separate cores. Wall
    # time ≈ slowest single call; total exploration ≈ N × that of one
    # call. Lowest-scoring plan wins.
    try:
        plan = _generate_parallel(
            attendees=names,
            players_path=Roster().all(),
            history_path=str(HISTORY_PATH),
            num_courts=num_courts,
            court_labels=labels,
            num_rotations=num_rotations,
            start_time_hhmm=start_time_hhmm,
            rotation_durations=rotation_durations,
            singles_exclude=singles_exclude,
            singles_include=singles_include,
            pinned_singles=pinned_singles,
            pinned_doubles=pinned_doubles_arg,
            late_court=late_court_arg,
            coaching=coaching_arg,
            today=session_day,
            seed=seed,
        )
    except ValueError as e:
        return {"ok": False, "error": "over_capacity_or_bad_input", "message": str(e)}
    result = plan.to_dict()
    # Tag the plan with the in-flight session type so it persists into
    # history.json on commit (and downstream — DOCX template choice,
    # replay-from-history, sheet log column).
    if state.session_type:
        result["session_type"] = state.session_type
    set_draft_plan(result)
    # Atomic phase transition. The system prompt asks the model to
    # call set_phase("draft_ready") manually after generate_pairings,
    # but that's fragile — if the model forgets, the session is
    # stuck mid-state. Doing it here guarantees the invariant
    # "draft_plan exists ⇒ phase == draft_ready". Idempotent if
    # already in draft_ready.
    try:
        set_phase("draft_ready")
    except Exception:
        pass
    return result


def tool_log_pairings_to_sheet(plan: dict) -> dict:
    """Escape hatch: append a plan dict to the Session/Pair-log tabs.

    Prefer ``commit_plan`` for the normal admin flow — it appends to
    ``history.json`` AND logs to the Sheet, then clears the draft.
    """
    from session_log import log_plan

    return log_plan(plan)


def tool_swap_players(
    name1: str,
    name2: str,
    rotation_num: Optional[int] = None,
) -> dict:
    """Swap the schedule slots of two players in the current draft plan.

    If ``rotation_num`` is omitted, the swap is applied in every rotation
    where both players appear (typical "swap their evening" intent).
    """
    from pairings import swap_players_in_plan
    from session_state import get_draft_plan, set_draft_plan

    plan = get_draft_plan()
    if not plan:
        return {
            "ok": False,
            "error": "no_draft",
            "message": "No draft plan in session state — run generate_pairings first.",
        }
    try:
        swapped = swap_players_in_plan(plan, name1, name2, rotation_num)
    except (KeyError, ValueError) as e:
        return {"ok": False, "error": "swap_failed", "message": str(e)}
    set_draft_plan(plan)
    return {"ok": True, "rotations_changed": swapped, "plan": plan}


def tool_swap_courts(
    label_a: str,
    label_b: str,
    rotation_nums: Optional[list[int]] = None,
) -> dict:
    """Swap the contents of two courts in the current draft plan.

    Use for admin requests like 'put singles on court 5' (then swap the
    current singles court with court 5) or 'move courts 7 and 9'. Court
    labels stay put — only the matchups move — so pinned slots elsewhere
    in the plan are unaffected.

    ``rotation_nums`` scopes the swap to specific rotations (1-based).
    Omit for an all-rotations swap (the default). Use the list form
    for messages like 'swap courts 1 and 5 for rotations 2 and 3'
    when only some rotations should change (e.g. the same group is on
    the less-preferred hard court for those rotations and should move
    to the clay court).
    """
    from pairings import swap_courts_in_plan
    from session_state import get_draft_plan, set_draft_plan

    plan = get_draft_plan()
    if not plan:
        return {
            "ok": False,
            "error": "no_draft",
            "message": "No draft plan in session state — run generate_pairings first.",
        }
    try:
        swapped = swap_courts_in_plan(
            plan, label_a, label_b, rotation_nums=rotation_nums,
        )
    except ValueError as e:
        return {"ok": False, "error": "swap_failed", "message": str(e)}
    set_draft_plan(plan)
    return {
        "ok": True,
        "swapped": [str(label_a), str(label_b)],
        "rotations_swapped": swapped,
        "plan": plan,
    }


def tool_set_rotation_times(
    rotation_durations: Optional[list[int]] = None,
    start_time_hhmm: Optional[str] = None,
) -> dict:
    """Re-time the current draft's rotations without re-pairing.

    ``rotation_durations`` (minutes, one per rotation) and/or a new
    ``start_time_hhmm``. Changing the NUMBER of rotations needs a
    re-roll (generate_pairings with num_rotations).
    """
    from pairings import retime_plan_rotations
    from session_state import get_draft_plan, set_draft_plan

    plan = get_draft_plan()
    if not plan:
        return {
            "ok": False,
            "error": "no_draft",
            "message": "No draft plan in session state — run generate_pairings first.",
        }
    if rotation_durations is None and start_time_hhmm is None:
        return {
            "ok": False,
            "error": "nothing_to_change",
            "message": "Pass rotation_durations and/or start_time_hhmm.",
        }
    try:
        retime_plan_rotations(plan, rotation_durations, start_time_hhmm)
    except ValueError as e:
        return {"ok": False, "error": "retime_failed", "message": str(e)}
    set_draft_plan(plan)
    return {
        "ok": True,
        "times": [
            f"{r['start_time']}-{r['end_time']}" for r in plan["rotations"]
        ],
        "plan": plan,
    }


def tool_swap_rotations(a: int, b: int) -> dict:
    """Swap the contents of two rotations in the current draft plan.

    Times stay tied to position — rotation 1 always runs at the first
    time slot, etc.
    """
    from pairings import swap_rotations_in_plan
    from session_state import get_draft_plan, set_draft_plan

    plan = get_draft_plan()
    if not plan:
        return {
            "ok": False,
            "error": "no_draft",
            "message": "No draft plan in session state — run generate_pairings first.",
        }
    try:
        swap_rotations_in_plan(plan, a, b)
    except ValueError as e:
        return {"ok": False, "error": "swap_failed", "message": str(e)}
    set_draft_plan(plan)
    return {"ok": True, "swapped": [a, b], "plan": plan}


def _plan_is_last_in_history(plan: dict) -> bool:
    """True when ``plan`` is already the most recent history record."""
    from pairings import load_history

    try:
        history = load_history(HISTORY_PATH)
    except Exception:
        return False
    if not history:
        return False
    # Compare via JSON so tuples-vs-lists in the in-memory plan don't
    # matter.
    norm = lambda d: json.dumps(d, sort_keys=True, ensure_ascii=False)
    return norm(history[-1]) == norm(plan)


def tool_commit_plan(pairings_text: str) -> dict:
    """Finalise the current draft plan AND end the session.

    Single-step finish: writes to ``history.json``, mirrors to the
    Google Sheet ``Session log`` / ``Pair log`` tabs, posts the FINAL
    pairings text + Word doc into the calling channel, and then
    CLEARS the session (phase, draft, attendees, late-court /
    pinned-doubles config — all gone). After commit the admin is back
    at no-session, ready for the next kickoff.

    Required: ``pairings_text`` — the fully-rendered FINAL pairings
    block (full preamble, NO ratings, copy-paste hint at the bottom).
    The model produces this; the tool posts it verbatim before the
    Word doc.

    Triggered by "boris commit", "boris save to history", "boris save
    and finalise", "boris that's the one" — the admin's explicit
    decision to lock in the current draft.

    Because the tool posts the FINAL text + doc itself, the model
    should emit an empty assistant reply after calling it.
    """
    from pairings import append_to_history
    from session_state import (
        clear_tonight, get_draft_plan, record_commit,
    )
    from session_log import log_plan

    plan = get_draft_plan()
    if not plan:
        return {
            "ok": False,
            "error": "no_draft",
            "message": "No draft plan to commit — run generate_pairings first.",
        }
    if not isinstance(pairings_text, str) or not pairings_text.strip():
        return {
            "ok": False,
            "error": "no_text",
            "message": "pairings_text is required — render the FINAL "
            "pairings block and pass it as pairings_text.",
        }

    # 1. History write — authoritative. If this fails, abort before
    #    touching anything else. A retried commit (an earlier attempt
    #    wrote history, then failed before clearing the session) finds
    #    the identical plan already last in history and skips the
    #    history + Sheet writes rather than recording the night twice.
    already_recorded = _plan_is_last_in_history(plan)
    if not already_recorded:
        try:
            append_to_history(plan, str(HISTORY_PATH))
        except Exception as e:
            return {
                "ok": False,
                "error": "history_write_failed",
                "message": str(e),
            }

    # 2. Sheet mirror — best-effort; don't abort if it fails (history
    #    is the source of truth).
    sheet_log = None
    sheet_error = None
    if not already_recorded:
        try:
            sheet_log = log_plan(plan)
        except Exception as e:
            sheet_error = str(e)

    # 3. Render + send FINAL text + Word doc into the channel. Never
    #    let a failure here stop the session being cleared below —
    #    history is already written.
    try:
        render_result = _render_and_send_final(plan, pairings_text)
    except Exception as e:
        render_result = {"ok": False, "error": "send_failed",
                         "message": str(e)}

    # 4. Clear the session entirely — the run is done. Whatever the
    #    admin did this evening (attendees, ratings, draft, pins, late
    #    courts) is gone. Roster changes (ratings, gender, singles
    #    preferences) PERSIST on the Sheet — those aren't part of
    #    session_state.
    clear_tonight()

    # 5. Stash the commit so undo_commit can find what to remove from
    #    history + Sheet (but NOT to restore the draft — undo is
    #    intentionally state-less now). Must come AFTER clear_tonight,
    #    which resets last_commit along with everything else.
    record_commit(
        plan,
        sheet_session_rows=(
            (sheet_log or {}).get("session_rows_appended", 0)
        ),
        sheet_pair_rows=(sheet_log or {}).get("pair_rows_appended", 0),
    )

    return {
        "ok": True,
        "history_appended": not already_recorded,
        "already_recorded": already_recorded,
        "sheet_log": sheet_log,
        "sheet_error": sheet_error,
        "final_sent": render_result,
        "session_cleared": True,
        "undo_hint": (
            "Say 'boris undo commit' to remove the entry from history. "
            "Note: undo does NOT recover the session state — you'd need "
            "to 'boris kickoff' to start again."
        ),
    }


def tool_undo_commit() -> dict:
    """Remove the most recent commit's traces from history + the Sheet.

    DOES NOT restore session state — by design. The session was
    cleared by commit_plan and stays cleared. After undo, the admin
    starts over with ``boris kickoff`` if they want a different plan.

    Removes:
      * The just-appended entry from history.json (matched by date
        + last-position).
      * The mirrored rows from the Sheet's Session / Pair log tabs
        (best-effort — sheet may be down; the authoritative undo is
        the history.json rewrite).

    Idempotent-ish: a second undo with nothing to reverse returns
    ``error="nothing_to_undo"``.
    """
    import json
    from session_state import clear_last_commit, get_last_commit

    lc = get_last_commit()
    if not lc:
        return {
            "ok": False,
            "error": "nothing_to_undo",
            "message": (
                "No recent commit to undo (already undone, or none has "
                "happened since the bot started)."
            ),
        }
    plan = lc.get("plan") or {}
    date = lc.get("date", "") or plan.get("date", "")

    # 1) Remove the committed entry from history.json — the LAST entry
    #    matching this date (the one commit_plan appended).
    history_removed = False
    try:
        if HISTORY_PATH.exists():
            with HISTORY_PATH.open(encoding="utf-8") as f:
                hist = json.load(f)
            if isinstance(hist, list):
                for i in range(len(hist) - 1, -1, -1):
                    rec = hist[i]
                    if isinstance(rec, dict) and rec.get("date") == date:
                        hist.pop(i)
                        history_removed = True
                        break
                if history_removed:
                    from atomic_io import atomic_write_text

                    atomic_write_text(
                        HISTORY_PATH,
                        json.dumps(hist, indent=2, ensure_ascii=False),
                    )
    except Exception as e:
        return {
            "ok": False,
            "error": "history_rewrite_failed",
            "message": str(e),
        }

    # 2) Delete the mirrored Sheet rows (best-effort).
    sheet_result = None
    sheet_error = None
    s_rows = int(lc.get("sheet_session_rows", 0) or 0)
    p_rows = int(lc.get("sheet_pair_rows", 0) or 0)
    if s_rows or p_rows:
        try:
            from session_log import unlog_plan
            sheet_result = unlog_plan(date, s_rows, p_rows)
        except Exception as e:
            sheet_error = str(e)

    clear_last_commit()

    return {
        "ok": True,
        "date": date,
        "history_entry_removed": history_removed,
        "sheet_result": sheet_result,
        "sheet_error": sheet_error,
        "message": (
            "Commit undone — removed from history"
            + (" and Sheet log" if (s_rows or p_rows) and not sheet_error
               else "")
            + ". The session state was NOT restored. Say "
            + "'boris kickoff' to start a new run."
            + ("" if history_removed
               else " (Note: no matching history.json entry was found "
                    "to remove.)")
        ),
    }


def tool_send_rules_pdf() -> dict:
    """Generate the pairing-rules PDF and send it to this channel.

    Always re-renders from the live constants in pairings.RULE_DOCS,
    so the attachment is up-to-date the moment any weight changes
    (no separate sync step needed). Sends as a WhatsApp attachment
    with a short covering caption.
    """
    from rules_pdf import render_rules_pdf

    jid = _CURRENT_GROUP_JID.get(None)
    if not jid:
        return {
            "ok": False,
            "error": "no_channel",
            "message": "no calling-channel JID available to send to.",
        }
    try:
        render_rules_pdf(RULES_PDF_PATH)
    except Exception as e:
        return {"ok": False, "error": "render_failed", "message": str(e)}
    ok = send_doc_to_group(
        jid, RULES_PDF_PATH,
        caption=BOT_REPLY_PREFIX + RULES_PDF_CAPTION,
    )
    return {
        "ok": ok,
        "path": str(RULES_PDF_PATH),
        "sent_to": jid,
        "message": (
            "Sent the rules-and-weights PDF to this channel."
            if ok else
            f"PDF generated but the bridge refused to send it; file is at {RULES_PDF_PATH}."
        ),
    }


def _render_and_send_final(plan: dict, pairings_text: str) -> dict:
    """Render the FINAL Word doc for ``plan`` and post it + the text
    block to the calling channel. Pure side effect on WhatsApp — does
    NOT touch session_state, history.json or the Sheet (the caller is
    responsible for those, if any).

    Returns the same shape as the legacy ``tool_send_final_docx``:
    ``{ok, text_sent, doc_sent, path, sent_to, message}``. Used by
    both ``show_final_pairings`` (preview) and ``commit_plan`` (final).
    """
    from pairings_docx import render_final_docx
    from session_state import get_tonight

    if not isinstance(pairings_text, str) or not pairings_text.strip():
        return {
            "ok": False,
            "error": "no_text",
            "message": "pairings_text is required — render the FINAL "
            "pairings block and pass it as pairings_text.",
        }

    jid = _CURRENT_GROUP_JID.get(None)
    if not jid:
        return {
            "ok": False,
            "error": "no_channel",
            "message": "no calling-channel JID available to send to.",
        }

    # Resolve template + filename from the session_type carried on the
    # plan (preferred — survives session clears) or the live session.
    session_type = (
        plan.get("session_type") if isinstance(plan, dict) else ""
    ) or get_tonight().session_type
    template_path = _docx_template_for(session_type)
    if not template_path.exists():
        return {
            "ok": False,
            "error": "template_missing",
            "message": f"Template not found at {template_path}.",
        }

    safe_date = (plan.get("date") or "session").replace("/", "-")
    basename = _docx_basename_for(session_type)
    out_path = FINAL_DOCX_OUTPUT_DIR / f"{basename} - {safe_date}.docx"
    try:
        render_final_docx(
            plan, template_path, out_path,
            preamble_paragraph_count=_docx_preamble_count_for(session_type),
            header_text=_docx_header_text_for(session_type),
        )
    except Exception as e:
        return {"ok": False, "error": "render_failed", "message": str(e)}

    # 1) Text message — full FINAL pairings, with Boris's prefix.
    text_ok = send_to_group(jid, BOT_REPLY_PREFIX + pairings_text)
    # 2) Doc attachment — with explanatory caption (also prefixed).
    doc_ok = send_doc_to_group(
        jid, out_path,
        caption=BOT_REPLY_PREFIX + DOCX_ATTACHMENT_CAPTION,
    )
    return {
        "ok": text_ok and doc_ok,
        "text_sent": text_ok,
        "doc_sent": doc_ok,
        "path": str(out_path),
        "sent_to": jid,
        "message": (
            "Posted FINAL pairings text + Word-doc attachment."
            if (text_ok and doc_ok) else
            f"Partial send: text_ok={text_ok}, doc_ok={doc_ok}. "
            f"File is at {out_path}."
        ),
    }


def tool_show_final_pairings(pairings_text: str) -> dict:
    """Preview the FINAL pairings (text + Word doc) without committing.

    Posts the rendered FINAL pairings block + the Word doc attachment
    into the calling channel. Reads the current draft_plan; does NOT
    touch history.json, the Sheet, or session_state — the draft stays
    editable. Use when the admin says "boris show final pairings",
    "boris preview the final", "boris what will the final look like".

    Because the tool posts the content itself, the model should emit
    an empty assistant reply after calling it.
    """
    from session_state import get_draft_plan

    plan = get_draft_plan()
    if not plan:
        return {
            "ok": False,
            "error": "no_draft",
            "message": "No draft plan to preview — run generate_pairings first.",
        }
    return _render_and_send_final(plan, pairings_text)
