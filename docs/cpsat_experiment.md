# CP-SAT pairings experiment — notes for future revival

A spike on `feature/or-tools-pairings` (final commit `57bf256`) explored
replacing — or running alongside — the production randomised-greedy +
hill-climb pairings algorithm with a Google OR-Tools CP-SAT
constraint-programming model. This document summarises what was
built, what worked, what didn't, and what would need to happen for
CP-SAT to become viable in production.

The branch was left in place rather than deleted. To re-engage:
`git checkout feature/or-tools-pairings`.

## Motivation

`pairings.make_plan` is a randomised algorithm: same inputs, different
runs can produce different scores (we saw a single 32-player session
land at 116 once and 36 on a different run). The variance comes from
the stochastic polish hill-climb. CP-SAT is a deterministic constraint
solver that can prove optimality — if the model captures all the
production rules, it would close that variance gap.

The investigation aimed to answer: **is CP-SAT good enough to replace
or augment the current algorithm?**

## What was built

Three new modules on the branch:

### `pairings_cpsat.py`

A drop-in alternative to `make_plan` using CP-SAT. Final state covers:

**Hard constraints:**
- Each player plays exactly once per rotation (no sit-outs — spike only).
- Each court has exactly 4 players per rotation (all-doubles only).
- Each court has exactly 2 players on side A.
- No pair of players shares a court in more than one rotation
  (stronger than production's "no partner repeat AND no opponent
  repeat" — necessary because the comparison harness re-picks pair
  splits and the weaker rule lets the re-picker introduce 500-pt
  violations).

**Soft constraints (objective minimised):**

| Rule | Production weight | CP-SAT encoding |
|---|---|---|
| Pair-sum imbalance | quadratic above diff 1 | Exact, via `AddElement` lookup table |
| Recent pair history | 7d=10, 14d=5 (date-based) | Exact, reuses `share_r` indicators |
| Rating-gap bands | 20/50/100 × per-player escalation | Linear escalation `1 + 2n` (Option A); per-player ✓ |
| Hard-court repeat | escalating: 0, 0, 10, 30 | Cumulative indicators `≥2`, `≥3` |
| Gender 3F1M | 50pt | Exact |
| Gender MM-vs-FF | 50pt | Exact |
| `top_player_no_strong_rotation` | 5pt × shortfall | Per-rotation max-others + min, AddMaxEquality |
| `standard_too_low` | 5pt × shortfall (rounded) | Integer division by 3 (floor, not round-to-nearest — ±1 mismatch) |

**Out of scope:**
- Singles courts (court mode is fixed as doubles).
- Sit-outs (assumes `len(attendees) == 4 × len(court_labels)`).
- Pinned singles / pinned doubles / late court.
- Per-player rating-gap *3^n exponential* escalation (used linear
  Option A — `1 + 2n` — instead).

### `compare_engines.py`

Replays past committed sessions through both engines and re-scores
both plans under `pairings._rescore_layout` (production scoring),
giving an apples-to-apples comparison. Flags:

- `--max-entries N` — how many history entries to compare.
- `--cpsat-time-limit S` — CP-SAT solve budget in seconds.
- `--trim-to-all-doubles` — for entries with singles courts, drop
  enough attendees + courts to make it all-doubles. Simulates a couple
  of late drop-outs.
- `--prod-seeds N` — run the production engine N times with seeds 42,
  43, … and use the best. Surfaces production's variance.

### `compare_engines_stress.py`

Synthetic 32-player case sampled from the live roster (real ratings
1-6, real M/F mix). Stress-tests the harder problem size where
production's variance becomes visible.

## Headline results

### Easy cases — history-based comparison (24-28 players, all-doubles)

5-of-5 ties between CP-SAT and production-best-of-5. Both find score
0 on every case. **CP-SAT 2.4s avg, production 7.0s avg** (production
runs 5 seeds sequentially; CP-SAT uses 4 workers in parallel).

### Hard case — synthetic 32-player with real rating spread + gender mix

| Engine | Score | Wall time |
|---|---|---|
| Production (best of 5 seeds) | 168 | ~37s |
| CP-SAT (full rules, 180s budget) | 187 | 180s (FEASIBLE not OPTIMAL) |

**Gap: 19 points.** CP-SAT couldn't converge to OPTIMAL within 180s
on the harder problem — the model's per-player rating-gap escalation
adds ~1,600 multiplication constraints, which is the bottleneck.

Internal objective (187) matches the re-scored value (187), so the
rules ARE correctly modelled — it's purely a search-time issue.

## What we learned

### CP-SAT can be competitive but isn't a free win

The Phase 1 spike (before adding the missing rules) lost by 244 points
on the stress test. After encoding all the rules, the gap closed to
19 points — most of which is just incomplete convergence. So the
encoding can faithfully reproduce production's outputs.

### Per-player escalation is the model's most expensive piece

The rating-gap rule's per-player factor (`1 + 2 × prior_unbalanced[p,r]`)
requires:
- 32 players × 8 courts × 3 rotations = 768 `assign × ge4` boolean
  products (for "is player p on an unbalanced court in rotation r").
- 32 × 8 × 3 = 768 `factor × assign` integer products (for the
  per-court factor sum).

So ~1,600 multiplication constraints just for this one rule. CP-SAT
handles them, but each multiplication compiles to many SAT clauses
and slows the search. On the 32-player case, the model can't prove
optimality within 180s; on the 24-player case it's 2-3s.

If we wanted CP-SAT to scale to 40+ player sessions, the per-player
escalation rule would need a simpler encoding — for example, Option B
(soft cap on per-player unbalanced count) or Option C (minimax across
players) from the original design discussion. Either would dramatically
shrink the model.

### Hard rule choice matters

We initially encoded "no partner repeat" and "no opponent repeat" as
the hard constraints, mirroring production. But the comparison harness
re-picks pair splits per court (it uses `_build_best_doubles_court`),
which led to CP-SAT plans triggering 500-pt intra_partner /
opponent_repeat penalties under re-scoring.

The fix: replace the hard rule with "each pair shares a court in at
most one rotation." Stronger than production's two soft rules, but
correct under ANY pair re-picking — there's no pair split that avoids
both intra_partner and opponent_repeat when a pair has shared a court
twice. This rewrite resolved that bug.

If CP-SAT ever becomes the sole engine (no re-scoring re-picking),
it could go back to the weaker pair-of-rules. As long as both engines
co-exist, the stronger rule is required.

### Parallel workers help a lot

The CP-SAT solver supports parallel search workers
(`num_search_workers=4`). Initially the spike used 1 worker for full
determinism. Switching to 4 workers on the bigger model:
- Loses strict reproducibility (workers race to find solutions).
- Substantially cuts wall time on harder cases.
- On the Pi (4 cores) this matches the available parallelism.

The harness handles non-determinism fine — both engines are
re-scored under production rules, so any solution that meets the
constraints can win.

## Why we didn't ship it

After Phase 2 (all rules encoded), the conclusion was: CP-SAT is
**competitive but not strictly better**, and shipping it would mean:

- Maintaining a second optimisation engine.
- Carrying the `ortools` dependency (a substantial C++ library).
- Accepting a wall-time spread (very fast on easy cases, slow on hard
  ones) that's hard to predict for an admin in the middle of a session.
- Still missing singles + sit-outs + pins (the un-done task #26).

The alternative — **running 4 production `make_plan` instances in
parallel with different seeds** — gives roughly the same exploration
in roughly the same wall time as one current call, using existing
well-tested code. That landed on `main` as the practical evolution.

## How to revive if needed

Re-engage by checking out the branch:

```
git checkout feature/or-tools-pairings
py -3 -m pip install ortools  # if not already installed
py -3 compare_engines.py --max-entries 10 --cpsat-time-limit 60 --trim-to-all-doubles --prod-seeds 5
py -3 compare_engines_stress.py
```

### What to do FIRST if reviving:

1. **Replace per-player rating-gap escalation with a simpler form** —
   Option B (soft cap, ~50pt per extra unbalanced rotation) or
   Option C (minimax across players). Should drop the multiplication
   count from ~1,600 to a few dozen, dramatically improving solve
   time on bigger problems.

2. **Add singles + sit-outs** (the un-done task #26). Court mode
   becomes a decision variable. Required to run on real un-trimmed
   Westside sessions (most of which have singles courts).

3. **Decide on the deployment model**:
   - As a parallel engine alongside production: spawn both, race, pick
     best. Requires production's parallel-best-of-N as the baseline
     (already on main) plus CP-SAT as the 5th worker.
   - As the sole engine: requires adding pins (singles + doubles + late
     court) which weren't done.
   - As an optional engine for diagnostics: "boris what's the optimal
     score?" — runs CP-SAT to prove a lower bound on what production
     could possibly achieve, useful for debugging.

### Triggers that might justify revival:

- Production's parallel-best-of-N proves insufficient (variance still
  bites in practice).
- A new constraint kind emerges that's awkward to express in the
  current scoring function (e.g. multi-week fairness across sessions).
- Attendance grows substantially (50+ players) where production's
  wall time becomes uncomfortable.

## Reference: branch / commits

- Branch: `feature/or-tools-pairings`
- Final commit: `57bf256` ("CP-SAT: add the missing scoring rules")
- Earlier commits document the incremental work:
  - `982931d` — initial spike with hard rules only
  - `f058c16` — comparison harness extended with --trim-to-all-doubles
  - `51a42b6` — --prod-seeds N + synthetic stress test (revealed gap)
  - `57bf256` — all scoring rules added (gap closed from 244 → 19 pts)
