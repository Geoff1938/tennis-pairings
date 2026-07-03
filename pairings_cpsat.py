"""Experimental CP-SAT pairings engine — Phase 1 spike.

An alternative implementation of ``make_plan`` using Google OR-Tools'
CP-SAT solver. Mirrors enough of the public interface to slot into the
existing render / commit pipeline, but covers a narrower set of cases
than the production algorithm:

  * All-doubles only — refuses if ``len(attendees) != 4 * len(court_labels)``
  * No pinned singles / pinned doubles / late court
  * Rating-gap rule uses a **flat court-level cost (base × 4)** with
    no per-player escalation — Option B from the revival notes. The
    old per-player factor was the model's documented bottleneck
    (~1,600 multiplication constraints).
  * partner_skill_gap modelled WITHOUT its per-player escalation
    (factor fixed at 1), on the model's own pair splits; the
    comparison harness re-picks splits per court, so the modelled
    value is a lower bound for the same grouping.
  * ``standard_too_low`` rounding approximated by floor division
    (±1 pt vs production's round-to-nearest).

What IS modelled (updated Jul 2026 to the post-June rules):
  * Hard: court capacity, one-court-per-rotation, each pair shares a
    court at most TWICE per evening (three shares can't avoid a
    same-role repeat under any splits); a second share is soft-priced
    since it usually re-scores as partners-once-opponents-once
  * Soft (objective):
      - pair-sum imbalance (exact quadratic lookup)
      - rating-gap bands (thresholds/bases derived from
        RATING_GAP_BANDS; per-player linear escalation)
      - recent pair history (date-based, bands from
        RECENT_PAIR_WEIGHT_BANDS via recent_pair_weights)
      - gender: 3F1M, 3M1F, MM-vs-FF, and the per-evening FFFF cap
      - top_player_no_strong_rotation / standard_too_low
      - new_faces_missed / cross_band_missed (Jul 2026 variety rules)
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
    CROSS_BAND_NEIGHBOUR_GAP,
    CROSS_BAND_MIN_OTHERS,
    CROSS_BAND_WEIGHT,
    GENDER_3F1M_PENALTY,
    GENDER_3M1F_PENALTY,
    GENDER_FFFF_FREE_ALLOWANCE,
    GENDER_FFFF_PENALTY_BASE,
    GENDER_FFFF_PENALTY_ESCALATION,
    GENDER_MM_VS_FF_PENALTY,
    HARD_COURT_NUMBERS,
    INTRA_EVENING_PENALTY,
    NEW_FACES_WEIGHT,
    OPPONENT_REPEAT_PENALTY,
    PAIR_IMBALANCE_WEIGHT,
    PARTNER_SKILL_GAP_THRESHOLD,
    PARTNER_SKILL_GAP_WEIGHT,
    RATING_GAP_BANDS,
    RECENT_PAIR_WEIGHT_BANDS,
    STANDARD_RULE_RATING_MAX,
    STANDARD_RULE_RATING_MIN,
    STANDARD_TOO_LOW_MATERIAL,
    STANDARD_TOO_LOW_WEIGHT,
    TOP_PLAYER_MAX_RATING,
    TOP_PLAYER_STRONG_WEIGHT,
    UNKNOWN_RATING,
    Court,
    PairingPlan,
    Rotation,
    _add_minutes,
    _build_ratings,
    _court_label_key,
    _pair_imbalance_penalty,
    _resolve_durations,
    compute_display_names,
    cross_band_due_players,
    load_history,
    load_players,
    never_met_pairs,
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
    # Optional warm start: a PairingPlan (e.g. production's output for
    # the same attendees/courts/rotations) used as a solution hint, so
    # the solver starts from a known-good layout and spends its budget
    # trying to improve it rather than finding feasibility from scratch.
    hint_plan: PairingPlan | None = None,
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
    # both_on[(p, q, r)] — per-court AND indicators, kept for the
    # partner_skill_gap rule below (which needs to know WHICH court a
    # pair shares to test side membership).
    both_on: dict[tuple[int, int, int], list[cp_model.IntVar]] = {}
    # Pairs sharing a court twice this evening (soft-priced below).
    repeat_share_terms: list[cp_model.IntVar] = []
    REPEAT_SHARE_SOFT_COST = 15
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
                both_on[p, q, r] = both_on_c_list

            # Pair-repeat rule. Production allows a pair to share a
            # court TWICE when the re-picked splits give them different
            # roles (partners once, opponents once) — only same-role
            # repeats cost 500. The old spike forbade any second share
            # outright, which (a) solves a strictly harder problem than
            # production and (b) makes real production plans infeasible
            # as warm-start hints. Relaxed Jul 2026: at most 2 shares
            # (three shares can't avoid a same-role repeat under any
            # splits), with the second share priced softly — usually it
            # re-scores as ~1pt same_court_successive, occasionally the
            # split juggling fails and costs 500, so the price sits well
            # above 1 to keep second shares rare.
            share_count = sum(share_r[p, q, r] for r in range(R))
            model.Add(share_count <= 2)
            twice = model.NewBoolVar(f"share2_{p}_{q}")
            model.Add(share_count >= 2).OnlyEnforceIf(twice)
            model.Add(share_count <= 1).OnlyEnforceIf(twice.Not())
            repeat_share_terms.append(twice)

    # ------------------------------------------------------------------
    # Soft objective
    # ------------------------------------------------------------------
    objective_terms: list[cp_model.LinearExpr] = []

    # Second court-shares (see the pair-repeat rule above).
    for tw in repeat_share_terms:
        objective_terms.append(REPEAT_SHARE_SOFT_COST * tw)

    # Pre-compute the rating array (attendees → integer rating).
    rating_arr = [
        int(ratings.get(attendees[p], UNKNOWN_RATING)) for p in range(P)
    ]
    max_rating = 10  # ratings are 1..10 (or UNKNOWN_RATING=6 for "?")

    # ------------------------------------------------------------------
    # Rule 1: Pair-sum imbalance — EXACT quadratic via lookup table.
    # ------------------------------------------------------------------
    # For each (court, rotation): diff = |sum_a - sum_b|, then look up
    # _pair_imbalance_penalty(diff) from pairings.py.
    pair_imbalance_table = [
        int(_pair_imbalance_penalty(d)) for d in range(4 * max_rating + 1)
    ]
    for c in range(C):
        for r in range(R):
            sum_a = sum(
                side_a[p, c, r] * rating_arr[p] for p in range(P)
            )
            sum_b = sum(
                (assign[p, c, r] - side_a[p, c, r]) * rating_arr[p]
                for p in range(P)
            )
            diff_lo = -4 * max_rating
            diff_hi = 4 * max_rating
            diff_var = model.NewIntVar(diff_lo, diff_hi, f"diff_c{c}_r{r}")
            model.Add(diff_var == sum_a - sum_b)
            abs_diff = model.NewIntVar(0, diff_hi, f"absdiff_c{c}_r{r}")
            model.AddAbsEquality(abs_diff, diff_var)
            pen = model.NewIntVar(
                0, max(pair_imbalance_table), f"imbpen_c{c}_r{r}"
            )
            model.AddElement(abs_diff, pair_imbalance_table, pen)
            objective_terms.append(pen)

    # ------------------------------------------------------------------
    # Rule 2: Rating-gap band penalty (flat court-level cost).
    # ------------------------------------------------------------------
    # Band thresholds/bases come from production's RATING_GAP_BANDS
    # (post-Jun 2026: 5→20, 7→50, 9→100). Encoded as cumulative >=
    # indicators with INCREMENTAL weights so the per-band totals match
    # the production base exactly; each non-balanced court is charged
    # the first-time production cost (base × 4) — see 2b for why the
    # per-player escalation is deliberately not modelled.
    band_steps = sorted(
        (thr, base) for thr, _name, base in RATING_GAP_BANDS if thr > 0
    )
    band_increments: list[tuple[int, int]] = []
    _prev_base = 0
    for thr, base in band_steps:
        band_increments.append((thr, base - _prev_base))
        _prev_base = base
    lowest_thr = band_increments[0][0]

    # 2a. Compute spread + band indicators per court per rotation.
    spread_vars: dict[tuple[int, int], cp_model.IntVar] = {}
    band_vars: dict[tuple[int, int], list[cp_model.IntVar]] = {}
    for c in range(C):
        for r in range(R):
            adj_max = []
            adj_min = []
            for p in range(P):
                rp = rating_arr[p]
                m_p = model.NewIntVar(0, max_rating, f"max_in_p{p}_c{c}_r{r}")
                model.Add(m_p == rp * assign[p, c, r])
                adj_max.append(m_p)
                n_p = model.NewIntVar(
                    1, max_rating + 1, f"min_in_p{p}_c{c}_r{r}"
                )
                model.Add(
                    n_p == rp * assign[p, c, r]
                    + (max_rating + 1) * (1 - assign[p, c, r])
                )
                adj_min.append(n_p)

            court_max = model.NewIntVar(0, max_rating, f"cmax_c{c}_r{r}")
            model.AddMaxEquality(court_max, adj_max)
            court_min = model.NewIntVar(
                1, max_rating + 1, f"cmin_c{c}_r{r}"
            )
            model.AddMinEquality(court_min, adj_min)
            spread = model.NewIntVar(0, max_rating, f"spread_c{c}_r{r}")
            model.Add(spread == court_max - court_min)

            indicators: list[cp_model.IntVar] = []
            for thr, _inc in band_increments:
                ge = model.NewBoolVar(f"ge{thr}_c{c}_r{r}")
                model.Add(spread >= thr).OnlyEnforceIf(ge)
                model.Add(spread <= thr - 1).OnlyEnforceIf(ge.Not())
                indicators.append(ge)

            spread_vars[c, r] = spread
            band_vars[c, r] = indicators

    # 2b. FLAT court-level penalty — Option B from the revival notes.
    # Production charges base × Σ_p RATING_GAP_MULT^prior_count[p],
    # which is base × 4 for a court of first-timers and escalates for
    # repeat offenders. The old spike modelled the per-player factor
    # with ~1,600 multiplication constraints — the documented
    # bottleneck that stopped convergence on 32-player sessions. Under
    # the loosened Jun 2026 bands (spread 0-4 free) unbalanced courts
    # are rare in good solutions, so we price every non-balanced court
    # at the first-time cost (base × 4) and drop the escalation
    # machinery entirely. Mismatch only when the SAME player lands on
    # multiple non-balanced courts — which the flat cost already
    # discourages globally.
    for c in range(C):
        for r in range(R):
            for (thr, inc), indicator in zip(
                band_increments, band_vars[c, r]
            ):
                objective_terms.append(4 * inc * indicator)

    # ------------------------------------------------------------------
    # Rule 3: Recent pair history (date-based). Reuses share_r.
    # ------------------------------------------------------------------
    for p in range(P):
        for q in range(p + 1, P):
            fs = frozenset((attendees[p], attendees[q]))
            wt = int(weekly_pair_penalties.get(fs, 0))
            if wt == 0:
                continue
            for r in range(R):
                objective_terms.append(wt * share_r[p, q, r])

    # ------------------------------------------------------------------
    # Rule 4: Gender penalties — 3F1M and MM-vs-FF.
    # ------------------------------------------------------------------
    # Per doubles court per rotation:
    #   3F1M: 50 pts if f_count == 3 and m_count == 1 (no ? players).
    #   MM-vs-FF: 50 pts if 2F+2M with both side-A same gender, both
    #             side-B same gender, sides differ.
    genders_map: dict[str, str] = {
        n: (str(info.get("gender", "?")).strip().upper() or "?")
        for n, info in players.items()
    }
    f_idx = [
        p for p in range(P)
        if genders_map.get(attendees[p], "?") == "F"
    ]
    m_idx = [
        p for p in range(P)
        if genders_map.get(attendees[p], "?") == "M"
    ]
    ffff_indicators: list[cp_model.IntVar] = []
    for c in range(C):
        for r in range(R):
            # Counts of F and M on this court this rotation.
            f_count = model.NewIntVar(0, 4, f"fc_{c}_{r}")
            m_count = model.NewIntVar(0, 4, f"mc_{c}_{r}")
            model.Add(f_count == sum(assign[p, c, r] for p in f_idx))
            model.Add(m_count == sum(assign[p, c, r] for p in m_idx))

            # 3F1M: f_count==3 AND m_count==1.
            f_eq_3 = model.NewBoolVar(f"feq3_{c}_{r}")
            model.Add(f_count == 3).OnlyEnforceIf(f_eq_3)
            model.Add(f_count != 3).OnlyEnforceIf(f_eq_3.Not())
            m_eq_1 = model.NewBoolVar(f"meq1_{c}_{r}")
            model.Add(m_count == 1).OnlyEnforceIf(m_eq_1)
            model.Add(m_count != 1).OnlyEnforceIf(m_eq_1.Not())
            is_3f1m = model.NewBoolVar(f"is3f1m_{c}_{r}")
            model.AddBoolAnd([f_eq_3, m_eq_1]).OnlyEnforceIf(is_3f1m)
            model.AddBoolOr([f_eq_3.Not(), m_eq_1.Not()]
                            ).OnlyEnforceIf(is_3f1m.Not())
            objective_terms.append(GENDER_3F1M_PENALTY * is_3f1m)

            # 3M1F: m_count==3 AND f_count==1 (mirror of the above,
            # lighter weight).
            m_eq_3 = model.NewBoolVar(f"meq3_{c}_{r}")
            model.Add(m_count == 3).OnlyEnforceIf(m_eq_3)
            model.Add(m_count != 3).OnlyEnforceIf(m_eq_3.Not())
            f_eq_1 = model.NewBoolVar(f"feq1_{c}_{r}")
            model.Add(f_count == 1).OnlyEnforceIf(f_eq_1)
            model.Add(f_count != 1).OnlyEnforceIf(f_eq_1.Not())
            is_3m1f = model.NewBoolVar(f"is3m1f_{c}_{r}")
            model.AddBoolAnd([m_eq_3, f_eq_1]).OnlyEnforceIf(is_3m1f)
            model.AddBoolOr([m_eq_3.Not(), f_eq_1.Not()]
                            ).OnlyEnforceIf(is_3m1f.Not())
            objective_terms.append(GENDER_3M1F_PENALTY * is_3m1f)

            # FFFF (all-female court) indicator — the per-evening cap is
            # applied after the loop, over the whole evening's count.
            is_ffff = model.NewBoolVar(f"isffff_{c}_{r}")
            model.Add(f_count == 4).OnlyEnforceIf(is_ffff)
            model.Add(f_count != 4).OnlyEnforceIf(is_ffff.Not())
            ffff_indicators.append(is_ffff)

            # MM-vs-FF: 2F+2M with same-gender pairs.
            # Side-A counts:
            f_side_a = model.NewIntVar(0, 2, f"fsa_{c}_{r}")
            m_side_a = model.NewIntVar(0, 2, f"msa_{c}_{r}")
            model.Add(f_side_a == sum(side_a[p, c, r] for p in f_idx))
            model.Add(m_side_a == sum(side_a[p, c, r] for p in m_idx))

            # MM_vs_FF needs:
            #   f_count == 2, m_count == 2 (so 0 ? on court)
            # AND
            #   side-A is all-F (f_side_a == 2) OR side-A is all-M
            #   (m_side_a == 2)
            #   — both imply side-B is the opposite all-gender.
            f_eq_2 = model.NewBoolVar(f"feq2_{c}_{r}")
            model.Add(f_count == 2).OnlyEnforceIf(f_eq_2)
            model.Add(f_count != 2).OnlyEnforceIf(f_eq_2.Not())
            m_eq_2 = model.NewBoolVar(f"meq2_{c}_{r}")
            model.Add(m_count == 2).OnlyEnforceIf(m_eq_2)
            model.Add(m_count != 2).OnlyEnforceIf(m_eq_2.Not())
            two_each = model.NewBoolVar(f"two_each_{c}_{r}")
            model.AddBoolAnd([f_eq_2, m_eq_2]).OnlyEnforceIf(two_each)
            model.AddBoolOr([f_eq_2.Not(), m_eq_2.Not()]
                            ).OnlyEnforceIf(two_each.Not())

            fsa_eq_2 = model.NewBoolVar(f"fsa_eq2_{c}_{r}")
            model.Add(f_side_a == 2).OnlyEnforceIf(fsa_eq_2)
            model.Add(f_side_a != 2).OnlyEnforceIf(fsa_eq_2.Not())
            msa_eq_2 = model.NewBoolVar(f"msa_eq2_{c}_{r}")
            model.Add(m_side_a == 2).OnlyEnforceIf(msa_eq_2)
            model.Add(m_side_a != 2).OnlyEnforceIf(msa_eq_2.Not())
            segregated = model.NewBoolVar(f"seg_{c}_{r}")
            model.AddBoolOr([fsa_eq_2, msa_eq_2]).OnlyEnforceIf(segregated)
            model.AddBoolAnd([fsa_eq_2.Not(), msa_eq_2.Not()]
                             ).OnlyEnforceIf(segregated.Not())

            is_mmff = model.NewBoolVar(f"is_mmff_{c}_{r}")
            model.AddBoolAnd([two_each, segregated]).OnlyEnforceIf(is_mmff)
            model.AddBoolOr([two_each.Not(), segregated.Not()]
                            ).OnlyEnforceIf(is_mmff.Not())
            objective_terms.append(GENDER_MM_VS_FF_PENALTY * is_mmff)

    # ------------------------------------------------------------------
    # Rule 4b: FFFF cap — per-EVENING escalating cost on all-female
    # courts. Production: first GENDER_FFFF_FREE_ALLOWANCE free, then
    # base × esc^(n - allowance - 1) per extra court (50, 150, 450 …).
    # Encoded as cumulative count >= k indicators with incremental
    # weights.
    # ------------------------------------------------------------------
    max_ffff = min(len(ffff_indicators), (len(f_idx) // 4) * R)
    if max_ffff > GENDER_FFFF_FREE_ALLOWANCE:
        ffff_total = model.NewIntVar(0, max_ffff, "ffff_total")
        model.Add(ffff_total == sum(ffff_indicators))
        for k in range(GENDER_FFFF_FREE_ALLOWANCE + 1, max_ffff + 1):
            ge_k = model.NewBoolVar(f"ffff_ge{k}")
            model.Add(ffff_total >= k).OnlyEnforceIf(ge_k)
            model.Add(ffff_total <= k - 1).OnlyEnforceIf(ge_k.Not())
            inc = GENDER_FFFF_PENALTY_BASE * (
                GENDER_FFFF_PENALTY_ESCALATION
                ** (k - GENDER_FFFF_FREE_ALLOWANCE - 1)
            )
            objective_terms.append(inc * ge_k)

    # ------------------------------------------------------------------
    # Rule 4b2: partner_skill_gap on the model's own pair splits.
    # Production penalises a PAIR whose ratings differ by more than
    # PARTNER_SKILL_GAP_THRESHOLD: weight × excess × (1 + prior
    # high-gap rotations of both players). The re-scorer re-picks
    # splits per court, so the modelled value is a lower bound for the
    # same grouping. Only pairs whose gap exceeds the threshold get
    # variables, keeping the model small.
    # partner_on_court = both_on AND NOT sides-split; sides-split ⇔
    # side_a[p]+side_a[q] == 1 given both on the court.
    # ------------------------------------------------------------------
    gap_pairs: list[tuple[int, int, int]] = []  # (p, q, excess)
    for p in range(P):
        for q in range(p + 1, P):
            excess = (
                abs(rating_arr[p] - rating_arr[q])
                - PARTNER_SKILL_GAP_THRESHOLD
            )
            if excess > 0:
                gap_pairs.append((p, q, excess))

    is_partner: dict[tuple[int, int, int], cp_model.IntVar] = {}
    for p, q, _excess in gap_pairs:
        for r in range(R):
            partner_on_c: list[cp_model.IntVar] = []
            for c in range(C):
                side_sum = side_a[p, c, r] + side_a[q, c, r]
                is_split = model.NewBoolVar(f"psplit_{p}_{q}_{c}_{r}")
                model.Add(side_sum == 1).OnlyEnforceIf(is_split)
                model.Add(side_sum != 1).OnlyEnforceIf(is_split.Not())
                part = model.NewBoolVar(f"ppart_{p}_{q}_{c}_{r}")
                model.AddBoolAnd(
                    [both_on[p, q, r][c], is_split.Not()]
                ).OnlyEnforceIf(part)
                model.AddBoolOr(
                    [both_on[p, q, r][c].Not(), is_split]
                ).OnlyEnforceIf(part.Not())
                partner_on_c.append(part)
            ip = model.NewBoolVar(f"ppair_{p}_{q}_r{r}")
            model.AddBoolOr(partner_on_c).OnlyEnforceIf(ip)
            model.AddBoolAnd([v.Not() for v in partner_on_c]
                             ).OnlyEnforceIf(ip.Not())
            is_partner[p, q, r] = ip

    # Per-player "had a high-gap partner in rotation r" indicator, for
    # the escalation factor (mirrors production's partner_gap_count).
    gap_players = sorted(
        {p for p, _q, _e in gap_pairs} | {q for _p, q, _e in gap_pairs}
    )
    gp_at: dict[tuple[int, int], cp_model.IntVar] = {}
    for p in gap_players:
        for r in range(R):
            involved = [
                is_partner[a, b, r] for a, b, _e in gap_pairs
                if p in (a, b)
            ]
            g = model.NewBoolVar(f"gpat_p{p}_r{r}")
            model.AddBoolOr(involved).OnlyEnforceIf(g)
            model.AddBoolAnd([v.Not() for v in involved]
                             ).OnlyEnforceIf(g.Not())
            gp_at[p, r] = g

    # Penalty per gap-pair per rotation:
    #   weight × excess × is_partner × (1 + prior_p + prior_q)
    pgap_factor_max = 1 + 2 * (R - 1)
    for p, q, excess in gap_pairs:
        for r in range(R):
            prior = sum(gp_at[p, rp] for rp in range(r)) + sum(
                gp_at[q, rp] for rp in range(r)
            )
            fac = model.NewIntVar(1, pgap_factor_max, f"pgf_{p}_{q}_r{r}")
            model.Add(fac == 1 + prior)
            pen = model.NewIntVar(
                0, pgap_factor_max, f"pgp_{p}_{q}_r{r}"
            )
            model.AddMultiplicationEquality(
                pen, [is_partner[p, q, r], fac]
            )
            objective_terms.append(
                PARTNER_SKILL_GAP_WEIGHT * excess * pen
            )

    # ------------------------------------------------------------------
    # Rule 4c: new_faces_missed (Jul 2026). A player with a never-met
    # co-attendee present tonight should share a court with at least
    # one of them; reuses share_r so the encoding is one OR per player.
    # ------------------------------------------------------------------
    idx_of = {name: i for i, name in enumerate(attendees)}
    never_met = never_met_pairs(history, attendees)
    nm_targets: dict[int, list[int]] = {}
    for fs in never_met:
        a, b = tuple(fs)
        if a in idx_of and b in idx_of:
            nm_targets.setdefault(idx_of[a], []).append(idx_of[b])
            nm_targets.setdefault(idx_of[b], []).append(idx_of[a])
    for p, targets in nm_targets.items():
        share_vars = [
            share_r[(min(p, q), max(p, q), r)]
            for q in targets for r in range(R)
        ]
        met = model.NewBoolVar(f"nf_met_p{p}")
        model.AddBoolOr(share_vars).OnlyEnforceIf(met)
        model.AddBoolAnd([v.Not() for v in share_vars]
                         ).OnlyEnforceIf(met.Not())
        objective_terms.append(NEW_FACES_WEIGHT * met.Not())

    # ------------------------------------------------------------------
    # Rule 4d: cross_band_missed (Jul 2026). Each due player should get
    # one rotation with CROSS_BAND_MIN_OTHERS+ court-mates rated
    # CROSS_BAND_NEIGHBOUR_GAP+ points away. Linear in share_r.
    # ------------------------------------------------------------------
    ratings_all = dict(ratings)
    due = cross_band_due_players(history, attendees, ratings_all, today=today)
    for p in range(P):
        if attendees[p] not in due:
            continue
        far_qs = [
            q for q in range(P)
            if q != p
            and abs(rating_arr[q] - rating_arr[p]) >= CROSS_BAND_NEIGHBOUR_GAP
        ]
        if len(far_qs) < CROSS_BAND_MIN_OTHERS:
            continue  # infeasible tonight (mirrors the production filter)
        qual_rs = []
        for r in range(R):
            far_count = sum(
                share_r[(min(p, q), max(p, q), r)] for q in far_qs
            )
            qual = model.NewBoolVar(f"cb_qual_p{p}_r{r}")
            model.Add(far_count >= CROSS_BAND_MIN_OTHERS).OnlyEnforceIf(qual)
            model.Add(far_count <= CROSS_BAND_MIN_OTHERS - 1
                      ).OnlyEnforceIf(qual.Not())
            qual_rs.append(qual)
        crossed = model.NewBoolVar(f"cb_crossed_p{p}")
        model.AddBoolOr(qual_rs).OnlyEnforceIf(crossed)
        model.AddBoolAnd([v.Not() for v in qual_rs]
                         ).OnlyEnforceIf(crossed.Not())
        objective_terms.append(CROSS_BAND_WEIGHT * crossed.Not())

    # ------------------------------------------------------------------
    # Rules 5 & 6 share a helper that exposes per-rotation co-player
    # rating statistics for a focal player p (no extra IntVar per other
    # player — linear expressions inlined to keep the model small).
    # ------------------------------------------------------------------
    def _co_rating_sum_expr(p_idx: int, r_idx: int):
        """LinearExpr: sum of co-players' ratings on p_idx's court in
        rotation r_idx (= 0 if alone). Uses share_r so no quadratic
        terms with assign needed."""
        return sum(
            rating_arr[q] * share_r[(min(p_idx, q), max(p_idx, q), r_idx)]
            for q in range(P) if q != p_idx
        )

    # ------------------------------------------------------------------
    # Rule 5: top_player_no_strong_rotation.
    # ------------------------------------------------------------------
    # For each player p with rating ≤ TOP_PLAYER_MAX_RATING:
    #   For each rotation r: max_other[p, r] = max rating among the
    #     other players on p's court.
    #   best_max_other[p] = min over r of max_other[p, r]
    #   shortfall = max(0, best_max_other - (rating[p] + 1))
    #   penalty = TOP_PLAYER_STRONG_WEIGHT × shortfall
    top_players = [
        p for p in range(P) if rating_arr[p] <= TOP_PLAYER_MAX_RATING
    ]
    for p in top_players:
        ceiling = rating_arr[p] + 1
        rotation_max_others: list[cp_model.IntVar] = []
        for r in range(R):
            # max_other = max over q of (rating[q] if same-court else 0).
            # Express as linear exprs in AddMaxEquality.
            terms: list = []
            for q in range(P):
                if q == p:
                    continue
                lo, hi = (p, q) if p < q else (q, p)
                terms.append(rating_arr[q] * share_r[lo, hi, r])
            mo = model.NewIntVar(0, max_rating, f"maxother_p{p}_r{r}")
            model.AddMaxEquality(mo, terms)
            rotation_max_others.append(mo)
        best_mo = model.NewIntVar(0, max_rating, f"bestmo_p{p}")
        model.AddMinEquality(best_mo, rotation_max_others)
        shortfall = model.NewIntVar(0, max_rating, f"tpshort_p{p}")
        model.AddMaxEquality(shortfall, [best_mo - ceiling, 0])
        objective_terms.append(TOP_PLAYER_STRONG_WEIGHT * shortfall)

    # ------------------------------------------------------------------
    # Rule 6: standard_too_low.
    # ------------------------------------------------------------------
    # For each player p with rating in [STANDARD_RULE_RATING_MIN,
    # STANDARD_RULE_RATING_MAX]:
    #   sum_others[p, r] = sum of co-players' ratings on p's court.
    #   best_sum_others = min over r (lowest = strongest company).
    #   shortfall_x3 = best_sum_others - 3*rating  (= 3 × (mean - rating))
    #   Material check: shortfall_x3 ≥ STANDARD_TOO_LOW_MATERIAL * 3 = 3.
    #   penalty ≈ STANDARD_TOO_LOW_WEIGHT × shortfall_x3 / 3 (int div).
    standard_players = [
        p for p in range(P)
        if STANDARD_RULE_RATING_MIN <= rating_arr[p] <= STANDARD_RULE_RATING_MAX
    ]
    for p in standard_players:
        rating_p = rating_arr[p]
        material_x3 = STANDARD_TOO_LOW_MATERIAL * 3
        rotation_sum_others: list[cp_model.IntVar] = []
        for r in range(R):
            so = model.NewIntVar(
                0, 3 * max_rating, f"sumother_p{p}_r{r}"
            )
            model.Add(so == _co_rating_sum_expr(p, r))
            rotation_sum_others.append(so)
        best_so = model.NewIntVar(0, 3 * max_rating, f"bestso_p{p}")
        model.AddMinEquality(best_so, rotation_sum_others)
        # shortfall_x3 = best_so - 3*rating_p (negative if strong company).
        sf_x3 = model.NewIntVar(
            -3 * max_rating, 3 * max_rating, f"stsf_x3_p{p}"
        )
        model.Add(sf_x3 == best_so - 3 * rating_p)
        # Gate on shortfall_x3 >= material_x3. We can absorb the
        # gating directly into the penalty:
        #   penalty = max(0, WEIGHT * (sf_x3 - material_x3 + material_x3)) / 3
        # but cleaner: clamp shortfall to >= 0 (already), then below
        # material multiply by 0 via the gate.
        gate = model.NewBoolVar(f"sttrig_p{p}")
        model.Add(sf_x3 >= material_x3).OnlyEnforceIf(gate)
        model.Add(sf_x3 < material_x3).OnlyEnforceIf(gate.Not())
        # weighted = WEIGHT * sf_x3 when gate=1 else 0.
        weighted_x3 = model.NewIntVar(
            0, STANDARD_TOO_LOW_WEIGHT * 3 * max_rating, f"stwx3_p{p}"
        )
        # weighted_x3 = WEIGHT * sf_x3 * gate
        # Encode as: when gate=1, weighted_x3 == WEIGHT * sf_x3;
        # when gate=0, weighted_x3 == 0.
        model.Add(weighted_x3 == STANDARD_TOO_LOW_WEIGHT * sf_x3
                  ).OnlyEnforceIf(gate)
        model.Add(weighted_x3 == 0).OnlyEnforceIf(gate.Not())
        # Approximate "round" via integer division by 3.
        penalty = model.NewIntVar(
            0, STANDARD_TOO_LOW_WEIGHT * max_rating, f"stpen_p{p}"
        )
        model.AddDivisionEquality(penalty, weighted_x3, 3)
        objective_terms.append(penalty)

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

    # Optional warm start from a hint plan (same attendees + courts +
    # rotation count required; silently skipped otherwise).
    if hint_plan is not None:
        p_idx = {name: i for i, name in enumerate(attendees)}
        c_idx = {lbl: i for i, lbl in enumerate(labels_list)}
        compatible = (
            len(hint_plan.rotations) == R
            and set(hint_plan.attendees) == set(attendees)
            and all(
                c.court_label in c_idx and c.mode == "doubles"
                for rot in hint_plan.rotations for c in rot.courts
            )
        )
        if compatible:
            for r, rot in enumerate(hint_plan.rotations):
                for court in rot.courts:
                    c = c_idx[court.court_label]
                    pair_a = set(court.pairs[0])
                    for name in court.players:
                        p = p_idx[name]
                        model.AddHint(assign[p, c, r], 1)
                        model.AddHint(side_a[p, c, r], 1 if name in pair_a else 0)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    if seed is not None:
        solver.parameters.random_seed = int(seed)
    # 4 workers — gives us a parallel portfolio (CP-SAT runs different
    # search strategies in different workers and shares solutions).
    # Loses strict determinism but cuts wall time significantly on the
    # bigger models. The Pi has 4 cores, so this maxes them out.
    solver.parameters.num_search_workers = 4

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
        never_met_pairs=never_met,
        cross_band_due=due,
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
