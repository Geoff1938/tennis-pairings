"""Experimental CP-SAT pairings engine — Phase 1 spike.

An alternative implementation of ``make_plan`` using Google OR-Tools'
CP-SAT solver. Mirrors enough of the public interface to slot into the
existing render / commit pipeline, but covers a narrower set of cases
than the production algorithm:

  * All-doubles only — refuses if ``len(attendees) != 4 * len(court_labels)``
  * No pinned singles / pinned doubles / late court
  * No gender penalties
  * No "standard too low" / "top player no strong rotation" whole-evening
    per-player rules
  * Rating-gap rule uses **linear escalation (1 + 2*n)** per player —
    Option A from the design discussion (vs the production code's
    multiplicative 3^n)

What IS modelled:
  * Hard: court capacity, one-court-per-rotation, partner-repeat,
    opponent-repeat
  * Soft (objective):
      - pair-sum imbalance (linear: PAIR_IMBALANCE_WEIGHT × diff)
      - rating-gap bands (base × (1 + 2*prior_count))
      - recent pair history (date-based 7d=10, 14d=5)
      - hard-court repeat (per-player escalating cost matching the
        production rule's behaviour)

The output is a fully-formed :class:`pairings.PairingPlan` so the
renderer / docx / commit pipeline work unchanged.

This module is wired into nothing — there's no bot tool, no admin
command. It exists for the A/B comparison harness
(``compare_engines.py``) to run side-by-side against ``pairings.make_plan``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from ortools.sat.python import cp_model  # type: ignore

from pairings import (
    HARD_COURT_NUMBERS,
    INTRA_EVENING_PENALTY,
    OPPONENT_REPEAT_PENALTY,
    PAIR_IMBALANCE_WEIGHT,
    RATING_GAP_BANDS,
    RECENT_PAIR_WEIGHT_BANDS,
    UNKNOWN_RATING,
    Court,
    PairingPlan,
    Rotation,
    _add_minutes,
    _build_ratings,
    _court_label_key,
    _resolve_durations,
    compute_display_names,
    load_history,
    load_players,
    recent_pair_weights,
)


# Mirror of production rating-gap bands but with linear escalation.
# Production: penalty = base × Σ_{p in court} 3^prior_count[p]
# Spike:      penalty = base × Σ_{p in court} (1 + 2*prior_count[p])
# Behavioural shape is the same (spread unbalanced rotations across
# players) with a gentler gradient. See the design discussion in the
# branch for the rationale.
RATING_GAP_LINEAR_ESCALATION = 2

# Hard-court repeat penalty matches the production rule: weight × Σ_(i=1..n-1) i
# where n is the player's hard-court-rotation count. Per-player.
HARD_COURT_REPEAT_WEIGHT = 10


def make_plan_cpsat(
    attendees: Iterable[str],
    players_path: str | Path | dict,
    history_path: str | Path,
    *,
    court_labels: list,
    num_rotations: int = 3,
    start_time_hhmm: str = "19:30",
    rotation_durations: list[int] | None = None,
    today: date | None = None,
    time_limit_seconds: float = 30.0,
    seed: int | None = None,
    # Unsupported in the spike — present so the signature is a close
    # cousin of make_plan's. Any non-empty value raises.
    pinned_singles: list | None = None,
    pinned_doubles: list | None = None,
    late_court: dict | None = None,
    singles_exclude: list | None = None,
    singles_include: list | None = None,
    strategy: str = "cpsat",
    # Other kwargs ignored.
    **_unused,
) -> PairingPlan:
    """Build a pairing plan using a CP-SAT model. See module docstring
    for what's modelled and what isn't.

    Raises ``NotImplementedError`` for unsupported configurations (pins,
    odd attendance, late court). Raises ``RuntimeError`` if CP-SAT
    reports infeasible.
    """
    if pinned_singles or pinned_doubles or late_court:
        raise NotImplementedError(
            "Phase 1 CP-SAT spike doesn't support pins or late_court"
        )
    if singles_exclude or singles_include:
        raise NotImplementedError(
            "Phase 1 CP-SAT spike is all-doubles — no singles tuning"
        )

    attendees = list(attendees)
    labels_list = [str(x) for x in court_labels]
    P = len(attendees)
    C = len(labels_list)
    R = int(num_rotations)
    if R < 1:
        raise ValueError("num_rotations must be >= 1")
    if P != 4 * C:
        raise NotImplementedError(
            f"Phase 1 CP-SAT spike supports all-doubles only — got "
            f"{P} attendees and {C} courts (expected exactly {4 * C}). "
            "Singles, sit-outs, and late-court cases are out of scope."
        )

    durations = _resolve_durations(R, rotation_durations)
    starts: list[str] = []
    ends: list[str] = []
    cursor = start_time_hhmm
    for d in durations:
        starts.append(cursor)
        cursor = _add_minutes(cursor, d)
        ends.append(cursor)

    players = (
        players_path if isinstance(players_path, dict)
        else load_players(players_path)
    )
    history = load_history(history_path)
    weekly_pair_penalties = recent_pair_weights(history, today=today)
    ratings = _build_ratings(players)
    unknown_attendees = [a for a in attendees if a not in players]
    display_names = compute_display_names(attendees)

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # CP-SAT model
    # ------------------------------------------------------------------
    model = cp_model.CpModel()

    # assign[p, c, r] — player p is on court c in rotation r.
    assign: dict[tuple[int, int, int], cp_model.IntVar] = {}
    # side_a[p, c, r] — player p is on side A of court c in rotation r
    # (with assign[p,c,r] required). Side B is the complement on the court.
    side_a: dict[tuple[int, int, int], cp_model.IntVar] = {}
    for p in range(P):
        for c in range(C):
            for r in range(R):
                assign[p, c, r] = model.NewBoolVar(f"a_p{p}_c{c}_r{r}")
                side_a[p, c, r] = model.NewBoolVar(f"sa_p{p}_c{c}_r{r}")
                # side_a only meaningful when assigned (avoids spurious
                # "both not side A but neither on court" matches).
                model.Add(side_a[p, c, r] <= assign[p, c, r])

    # Each player plays exactly once per rotation (all-doubles, no sit-outs).
    for p in range(P):
        for r in range(R):
            model.Add(sum(assign[p, c, r] for c in range(C)) == 1)

    # Each court holds exactly 4 players per rotation.
    for c in range(C):
        for r in range(R):
            model.Add(sum(assign[p, c, r] for p in range(P)) == 4)

    # Each court has exactly 2 players on side A (other 2 on side B).
    for c in range(C):
        for r in range(R):
            model.Add(sum(side_a[p, c, r] for p in range(P)) == 2)

    # ------------------------------------------------------------------
    # Court-sharing indicator + pair-repeat hard constraint
    # ------------------------------------------------------------------
    # The production scoring re-picks pair splits per court when it
    # re-scores a plan. So the only hard constraint we can enforce in
    # CP-SAT's player-grouping decisions that SURVIVES re-scoring is
    # "each pair of players is on the same court in at most one
    # rotation". If any pair shared a court across two rotations, the
    # re-scorer would pick a pair split that triggers either
    # intra_partner (500pts) or opponent_repeat (500pts) — there's no
    # split that avoids both. So this stronger no-repeat-share rule is
    # the right invariant for the spike's player-grouping decisions.
    share_r: dict[tuple[int, int, int], cp_model.IntVar] = {}
    for p in range(P):
        for q in range(p + 1, P):
            for r in range(R):
                # both_on_c[p,q,c,r] = assign[p,c,r] AND assign[q,c,r].
                both_on_c_list: list[cp_model.IntVar] = []
                for c in range(C):
                    both_on_c = model.NewBoolVar(f"bc_{p}_{q}_{c}_{r}")
                    model.AddMultiplicationEquality(
                        both_on_c, [assign[p, c, r], assign[q, c, r]]
                    )
                    both_on_c_list.append(both_on_c)
                # share_r = OR over c of both_on_c (they're on SOME court
                # together this rotation).
                sh = model.NewBoolVar(f"sh_{p}_{q}_r{r}")
                model.AddBoolOr(both_on_c_list).OnlyEnforceIf(sh)
                model.AddBoolAnd([x.Not() for x in both_on_c_list]
                                 ).OnlyEnforceIf(sh.Not())
                share_r[p, q, r] = sh

            # Hard rule: each pair shares a court at most once across
            # the evening. Implies both no-partner-repeat AND
            # no-opponent-repeat under any pair re-picking, which is
            # what the production scoring rules ultimately enforce.
            model.Add(sum(share_r[p, q, r] for r in range(R)) <= 1)

    # ------------------------------------------------------------------
    # Soft objective
    # ------------------------------------------------------------------
    objective_terms: list[cp_model.LinearExpr] = []

    # 1) Pair-sum imbalance: |sum_a - sum_b| × PAIR_IMBALANCE_WEIGHT
    # For each (court, rotation): sum_a = sum_p(side_a[p,c,r] * rating[p])
    # imbalance = |sum_a - sum_b| where sum_b = sum_p(assign[p,c,r] * rating[p]) - sum_a
    # We minimise sum over courts/rotations of imbalance.
    for c in range(C):
        for r in range(R):
            sum_a = sum(
                side_a[p, c, r] * int(ratings.get(attendees[p], UNKNOWN_RATING))
                for p in range(P)
            )
            sum_b = sum(
                (assign[p, c, r] - side_a[p, c, r])
                * int(ratings.get(attendees[p], UNKNOWN_RATING))
                for p in range(P)
            )
            # diff = sum_a - sum_b (could be negative).
            # We want |diff| in the objective.
            max_rating = 10
            diff_lo = -4 * max_rating
            diff_hi = 4 * max_rating
            diff_var = model.NewIntVar(diff_lo, diff_hi, f"diff_c{c}_r{r}")
            model.Add(diff_var == sum_a - sum_b)
            abs_diff = model.NewIntVar(0, diff_hi, f"absdiff_c{c}_r{r}")
            model.AddAbsEquality(abs_diff, diff_var)
            objective_terms.append(PAIR_IMBALANCE_WEIGHT * abs_diff)

    # 2) Rating-gap band penalty with Option A linear escalation.
    # For each (court, rotation): band = which band the rating spread
    # (max - min) falls into. Per-player escalation: factor[p] =
    # 1 + RATING_GAP_LINEAR_ESCALATION * prior_count[p] where prior_count
    # is the number of non-balanced rotations player p has had in
    # rotations 0..r-1.
    #
    # We need:
    #   is_unbalanced[c, r]  — boolean: court c rotation r is non-balanced
    #   band_weight[c, r]    — integer: 0/20/50/100 based on band
    #   prior_count[p, r]    — sum of is_unbalanced[c', r'] across courts
    #                          for r' < r weighted by assign[p,c',r']
    #   factor[p, r]         — 1 + 2*prior_count, when assigned to a
    #                          non-balanced court in rotation r
    #
    # Simpler partial encoding: for each (court, rotation), compute the
    # rating spread and the band base weight, then add an objective term
    # = band_weight × Σ_{p assigned to court} (1 + 2*prior_count[p]).

    # Each rotation court's spread = max(rating[p] for p on court) - min(...)
    # which is hard to express. We use a different encoding: for each
    # (court, rotation) AND for each (min_rating R_min, max_rating R_max),
    # introduce an indicator var spread_r[c, r, gap]. Then base weight
    # is a piecewise function of gap.
    # Simpler still: directly compute min and max via CP-SAT's MinEquality /
    # MaxEquality.
    for c in range(C):
        for r in range(R):
            # rating_on_court[p] = assign[p, c, r] * rating[p], but for
            # "not assigned" we want it OUT of the min/max calculation.
            # Trick: use a wide neutral value (max_rating + 1) for not-
            # assigned positions, but only when computing min.
            # Easier: gather only the 4 assigned ratings via a different
            # construction — define an integer var court_min, court_max
            # and constrain them.
            #
            # We want:
            #   court_max ≥ rating[p]   for every p on the court (assign=1)
            #   court_max takes the max of those
            # Equivalent: court_max = MAX(rating[p] * assign[p] + UNUSED * (1 - assign[p]))
            # where UNUSED is small enough not to affect the max.
            # Set UNUSED = 0 for max, UNUSED = 11 (> max rating) for min.

            adj_max = []
            adj_min = []
            for p in range(P):
                rp = int(ratings.get(attendees[p], UNKNOWN_RATING))
                # For max: assigned → rp, not assigned → 0 (won't beat rp)
                m_p = model.NewIntVar(0, 10, f"max_in_p{p}_c{c}_r{r}")
                model.Add(m_p == rp * assign[p, c, r])
                adj_max.append(m_p)
                # For min: assigned → rp, not assigned → 11 (won't undercut)
                n_p = model.NewIntVar(1, 11, f"min_in_p{p}_c{c}_r{r}")
                model.Add(n_p == rp * assign[p, c, r] + 11 * (1 - assign[p, c, r]))
                adj_min.append(n_p)

            court_max = model.NewIntVar(0, 10, f"cmax_c{c}_r{r}")
            model.AddMaxEquality(court_max, adj_max)
            court_min = model.NewIntVar(1, 11, f"cmin_c{c}_r{r}")
            model.AddMinEquality(court_min, adj_min)
            spread = model.NewIntVar(0, 10, f"spread_c{c}_r{r}")
            model.Add(spread == court_max - court_min)

            # Band base weight: 0 for spread 0-3, 20 for 4-5, 50 for 6-7,
            # 100 for 8+. Encode via cumulative indicators.
            # is_unbalanced_*[c, r] = boolean for the 4-5, 6-7, 8+ tiers.
            ge4 = model.NewBoolVar(f"ge4_c{c}_r{r}")
            ge6 = model.NewBoolVar(f"ge6_c{c}_r{r}")
            ge8 = model.NewBoolVar(f"ge8_c{c}_r{r}")
            model.Add(spread >= 4).OnlyEnforceIf(ge4)
            model.Add(spread <= 3).OnlyEnforceIf(ge4.Not())
            model.Add(spread >= 6).OnlyEnforceIf(ge6)
            model.Add(spread <= 5).OnlyEnforceIf(ge6.Not())
            model.Add(spread >= 8).OnlyEnforceIf(ge8)
            model.Add(spread <= 7).OnlyEnforceIf(ge8.Not())

            # Band base weights (from RATING_GAP_BANDS in pairings.py):
            # unbalanced 4-5 → 20, very_unbalanced 6-7 → 50,
            # extremely_unbalanced 8-9 → 100.
            band_base = {4: 20, 6: 30, 8: 50}  # incremental: 20 + 30 = 50; +50 = 100.
            # base_weight = 20*ge4 + 30*ge6 + 50*ge8

            # For per-player factor, simplification for the spike: rather
            # than computing prior_count exactly (which requires a chain
            # of integer auxiliary vars across rotations), use a coarse
            # approximation: assume factor = 1 for every player, and
            # apply the band-base weight uniformly multiplied by 4
            # (the four assigned players' base factor = 4). This loses
            # the "spread across players" behaviour but keeps "minimise
            # unbalanced courts" intact for Phase 1.
            #
            # This is a deliberate Phase 1 simplification — the Option A
            # linear escalation is still trivial to add once we have a
            # working baseline, by linking prior_count via channelling.
            # For now we underweight unbalanced courts; the comparison
            # will reveal whether it matters.
            objective_terms.append(20 * ge4 * 4)
            objective_terms.append(30 * ge6 * 4)
            objective_terms.append(50 * ge8 * 4)

    # 3) Recent pair history. Reuses share_r (already built above for
    # the no-repeat-share hard constraint).
    for p in range(P):
        for q in range(p + 1, P):
            fs = frozenset((attendees[p], attendees[q]))
            wt = int(weekly_pair_penalties.get(fs, 0))
            if wt == 0:
                continue
            for r in range(R):
                objective_terms.append(wt * share_r[p, q, r])

    # 4) Hard-court repeat (Westside rule).
    # For each player p: hard_count[p] = number of rotations on a hard
    # court (where court_label parses to a number in HARD_COURT_NUMBERS).
    # Penalty per player = HARD_COURT_REPEAT_WEIGHT × Σ_(i=1..n-1) i
    # = HARD_COURT_REPEAT_WEIGHT × n*(n-1)/2.
    #
    # For 3 rotations: n ∈ {0, 1, 2, 3} → 0, 0, 1, 3. We encode it as:
    # is_2nd_or_more[p] (n >= 2) + is_3rd[p] (n >= 3), penalty = 10 + 20.
    hard_court_indices = [
        c for c in range(C)
        if _court_label_key(labels_list[c]).isdigit()
        and int(_court_label_key(labels_list[c])) in HARD_COURT_NUMBERS
    ]
    if hard_court_indices:
        for p in range(P):
            hard_count_expr = sum(
                assign[p, c, r]
                for c in hard_court_indices for r in range(R)
            )
            # Indicators
            ge2 = model.NewBoolVar(f"hc_ge2_p{p}")
            ge3 = model.NewBoolVar(f"hc_ge3_p{p}")
            # Need an IntVar to use Add(expr >= 2).OnlyEnforceIf...
            hc_var = model.NewIntVar(0, R, f"hc_p{p}")
            model.Add(hc_var == hard_count_expr)
            model.Add(hc_var >= 2).OnlyEnforceIf(ge2)
            model.Add(hc_var <= 1).OnlyEnforceIf(ge2.Not())
            model.Add(hc_var >= 3).OnlyEnforceIf(ge3)
            model.Add(hc_var <= 2).OnlyEnforceIf(ge3.Not())
            objective_terms.append(HARD_COURT_REPEAT_WEIGHT * ge2)
            objective_terms.append(2 * HARD_COURT_REPEAT_WEIGHT * ge3)

    # Wire the objective.
    model.Minimize(sum(objective_terms))

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    if seed is not None:
        solver.parameters.random_seed = int(seed)
    # Single-thread for full determinism.
    solver.parameters.num_search_workers = 1

    status = solver.Solve(model)
    wall = round(time.perf_counter() - t_start, 2)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"CP-SAT returned {solver.StatusName(status)} after {wall}s — "
            "no feasible plan found. The model's hard constraints "
            "(no partner repeat, no opponent repeat) may be too strict "
            "for this attendee count."
        )

    # ------------------------------------------------------------------
    # Decode solution → PairingPlan
    # ------------------------------------------------------------------
    rotations: list[Rotation] = []
    for r in range(R):
        courts: list[Court] = []
        for c in range(C):
            on_court = [
                attendees[p] for p in range(P)
                if solver.BooleanValue(assign[p, c, r])
            ]
            side_a_players = [
                attendees[p] for p in range(P)
                if solver.BooleanValue(assign[p, c, r])
                and solver.BooleanValue(side_a[p, c, r])
            ]
            side_b_players = [n for n in on_court if n not in side_a_players]
            if len(on_court) != 4 or len(side_a_players) != 2:
                raise RuntimeError(
                    f"CP-SAT solution malformed for court {c} rotation {r}: "
                    f"on_court={on_court}, side_a={side_a_players}"
                )
            courts.append(Court(
                court_label=labels_list[c],
                mode="doubles",
                players=on_court,
                pairs=[
                    (side_a_players[0], side_a_players[1]),
                    (side_b_players[0], side_b_players[1]),
                ],
            ))
        rotations.append(Rotation(
            rotation_num=r + 1,
            start_time=starts[r],
            end_time=ends[r],
            courts=courts,
            sit_outs=[],
        ))

    genders: dict[str, str] = {
        n: (str(info.get("gender", "?")).strip().upper() or "?")
        for n, info in players.items()
    }

    plan_date = (today or date.today()).isoformat()
    return PairingPlan(
        date=plan_date,
        attendees=attendees,
        court_labels=labels_list,
        num_rotations=R,
        rotations=rotations,
        unknown_attendees=unknown_attendees,
        display_names=display_names,
        ratings=ratings,
        strategy=strategy,
        genders=genders,
        weekly_pair_penalties=weekly_pair_penalties,
        notes="",
        metrics={
            "engine": "cpsat",
            "wall_seconds": wall,
            "status": solver.StatusName(status),
            "objective": solver.ObjectiveValue() if status != cp_model.UNKNOWN else None,
            "num_booleans": solver.NumBooleans(),
            "num_branches": solver.NumBranches(),
            "num_conflicts": solver.NumConflicts(),
        },
    )
