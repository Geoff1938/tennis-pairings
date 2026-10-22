# Pairing algorithm

How `pairings.py` turns an attendee list and a set of court labels into
a multi-rotation plan: who plays whom, on which court, at what time.

The public entry point is `make_plan(...)` in [`pairings.py`](../pairings.py).
The bot calls it through `bot_tools_pairings._generate_parallel`, which runs
several `make_plan` calls side by side and keeps the best. There is one
strategy, `skill_balanced` (`random` is kept as an alias for the same
function, `skill_balanced_multi_rotation`).

Weights and thresholds are constants at the top of `pairings.py`.
This document names the constant and gives its value at the time of
writing; if they disagree, the code is right. `RULE_DOCS` in the same
file is the plain-English rule list that feeds the members' rules PDF
(`rules_pdf.py`), so it always shows the live weights.

---

## Overview

1. **Seed search.** For each of several RNG seeds, build a whole
   evening greedily, one rotation at a time, by rejection sampling.
2. **Score each seed's evening** on the full score (per-rotation rules
   plus whole-evening rules) and keep the best few.
3. **Polish.** Hill-climb on the best plan(s) by swapping players
   between courts and rotations, keeping only swaps that lower the
   total.
4. **Parallel workers.** The bot runs steps 1-3 in several processes
   with different seeds and keeps the lowest-scoring result.

Lower scores are better; 0 means no rule is broken.

---

## Inputs

* **`attendees`** — full names of tonight's players.
* **`court_labels`** (or `num_courts`) — court numbers, e.g.
  `["5","6","7","8"]`. Order matters: singles courts are taken from the
  end of the list.
* **`num_rotations`** — default 3.
* **`rotation_durations`** — minutes per rotation. Default
  `DEFAULT_ROTATION_DURATIONS_3` (`[45, 40, 35]`) for 3 rotations,
  otherwise 40 each.
* **`start_time_hhmm`** — default `"19:30"`.
* **`today`** — the session date; anchors the history recency windows.
  The bot passes the session date, not the day the draft is generated.
* **`seed`** — RNG seed (see below).
* **Per-session options (no roster change):**
  * `singles_exclude` / `singles_include` — treat as singles `avoid` /
    `prefer` for this run.
  * `pinned_singles` — `[{rotation_num, players, court_label?}]`, force
    a singles match.
  * `pinned_doubles` — `[{rotation_num?, players, pairs, court_label?}]`,
    force a doubles four with a fixed pair split. A pin without
    `rotation_num` is placed in a rotation chosen by the seed's RNG.
  * `late_court` — `{label, first_rotation, pinned_players}`: a court
    that only opens from `first_rotation`, with its 4 players.
  * `coaching` — `{players, courts, rotations}`: a group and their
    courts away with a coach for the first `rotations` rotations.
* **From the roster (Google Sheet):** `rating` (1-10, 1 is strongest;
  `?` counts as `UNKNOWN_RATING` = 6), `gender` (`M` / `F` / `?`),
  `singles` (`prefer` / `avoid` / blank), `mixed` (`prefer` opts in to
  the mixed-match rule), `provisional`.
* **From `history.json`:** past sessions, used for the recent-partner
  and recent-opponent penalties, never-met pairs, cross-band due
  players and mixed-match droughts.

---

## Court layout per rotation

Each rotation is laid out independently, because the coaching group
and late court can change the number of players and courts. With `n`
players available and `c` courts, capacity is `4c`:

* `n > 4c` — the surplus sits out (plus one more if that leaves an odd
  number); all courts are doubles.
* `n` even and ≤ `4c` — no sit-outs; `(4c − n) / 2` singles courts,
  the rest doubles.
* `n` odd — one player sits out, then as above.
* Fewer than 2 players per court is an error ("drop a court or add
  players").

Sit-outs go to whoever has sat out least so far, random tie-break.
Pinned players are never chosen to sit out. In practice the club keeps
numbers even, so sit-outs are rare.

**Pins, late court and coaching group in the layout:**

* Courts pinned for doubles (and the late court in its first rotation)
  are kept in the doubles range; a court pinned for singles is moved to
  the end so it becomes a singles court.
* Before the late court opens, its 4 players sit out and the court is
  not used. In its first rotation those 4 are placed on it (the best
  2-v-2 split is still chosen). After that it is an ordinary court.
* While the coaching group is with the coach, its players and courts
  are removed from the rotation entirely (they do not count as
  sit-outs).
* Admin-pinned doubles courts are built as given and marked `pinned`:
  they score 0 themselves, but their partners, opponents, gap counts
  and so on still feed the cross-rotation trackers, so (for example)
  a later partner repeat with a pinned pair is still penalised.

---

## Choosing singles players

`_select_singles_players` sorts the eligible players (not pinned to a
doubles court or the late court) by:

1. already at `MAX_SINGLES_PER_EVENING` (1) — these go last;
2. singles preference — `prefer`, neutral, `avoid`;
3. rating — stronger first;
4. singles played so far tonight;
5. random.

A pinned singles pair is placed first and the rest of the slots are
filled from this order.

---

## Greedy rotation build (rejection sampling)

`skill_balanced_multi_rotation` builds rotations in order. For each
rotation it makes up to `MAX_ATTEMPTS` (1000) random layouts and keeps
the lowest-scoring, stopping early at 0. One attempt (`_try_layout`):

1. Shuffle the doubles players and the singles players separately.
2. Place any forced four (late court) or pinned singles pair on its
   court.
3. Cut the doubles shuffle into fours, one per court. For each four,
   `_build_best_doubles_court` tries all three pair splits (AB-CD,
   AC-BD, AD-BC) and keeps the cheapest.
4. Pair up the singles players.
5. Score the rotation with the per-rotation rules below, given
   everything already decided in earlier rotations.

After choosing a rotation it updates the trackers: tonight's partner
pairs and opponent pairs, who shared a court in this rotation (used
only by the next one), each player's count of non-balanced courts and
high-gap partnerships, singles appearances, sit-outs and all-female
courts so far.

Being greedy, a good rotation 1 can force a poor rotation 3. The seed
search and the polish exist to get round this.

---

## Multi-seed search (`make_plan`)

* Seeds: with an integer `seed`, the candidates are `seed, seed+1, …`
  up to `MAX_SEED_ATTEMPTS` (25). With `seed=None` they are random.
* First `DEFAULT_SEED_ATTEMPTS` (5) seeds are run. If the best total is
  still above `HIGH_SCORE_TRIGGER` (50), more seeds are tried, up to 25
  in all, stopping once one reaches `TARGET_SCORE` (25) or below.
* Each seed's plan is compared on its **full score**: the per-rotation
  scores plus the whole-evening rules (folded in by
  `_fold_evening_items`). This is the same score the polish step
  optimises, so seeds are no longer ranked on per-rotation scores
  alone.
* Any hard rule still present in the chosen plan (`HARD_RULE_KEYS`,
  currently `opponent_repeat`) is reported in
  `metrics["multi_seed"]["blocking_rules"]` so the bot can mention it.

## Polish (hill-climb)

If the best plan scores above 0, `polish_plan` runs on the
`POLISH_MULTISTART_K` (5) lowest-scoring seed plans, each with its own
RNG, and the lowest result wins. (With a single seed plan, or a best
of 0, there is at most one polish run.)

Each step swaps two player slots, in the same rotation (different
courts) or across rotations, rescoring the whole evening with
`_rescore_layout`. A swap is kept only if the total goes down. It
stops after `POLISH_MAX_ITERATIONS` (15000) steps or
`POLISH_MAX_NO_IMPROVEMENT` (3000) steps without improvement.

`_rescore_layout` replays the evening from scratch: it re-picks the
best pair split on every doubles court given the trackers so far,
rescores each rotation, then adds the whole-evening rules.

Swaps that could never be legal are not proposed in the first place:

* courts that are locked — the late court in its first rotation,
  admin-pinned doubles courts, and pinned singles matches;
* swaps between a doubles and a singles court (that would undo the
  singles selection, cap and preferences);
* swaps between two rotations with exactly the same people on court.
  A cross-rotation swap is only possible when someone is on court in
  one rotation and not the other (sit-outs, coaching group, late
  court); otherwise it would always put a player on two courts at once.

The remaining checks (no duplicate player in a rotation, never moving
someone who is off court onto a court) stay as a safety net.

## Parallel workers (`bot_tools_pairings._generate_parallel`)

The bot runs `GENERATE_PARALLEL_WORKERS` (4) `make_plan` calls in
separate processes (one per Pi core) and keeps the lowest-scoring
plan. Wall time is roughly that of one call.

* Each worker gets base seed `base + k × MAX_SEED_ATTEMPTS`
  (k = 0, 1, 2, 3). Because each `make_plan` uses consecutive seeds
  from its base, spacing the bases 25 apart means the workers never
  repeat each other's seeds.
* With no seed given, `base` is drawn at random, so a re-roll with
  nothing changed gives a fresh draft. Passing a seed gives repeatable
  output.
* The winner's metrics report the slowest worker's wall time and the
  total number of layouts tried across all workers.
* If every worker fails with a `ValueError` (bad input, e.g. too few
  players for the courts) that error is passed back as-is.

---

## Scoring rules

### Per-rotation rules

Scored on each court (pinned courts excepted) as the rotation is built.

| Rule (breakdown key) | Constant | Current value | What it penalises |
|---|---|---:|---|
| `opponent_repeat` | `OPPONENT_REPEAT_PENALTY` | 500 | Two players facing each other again tonight (doubles across the net, or singles). Hard. |
| `intra_partner` | `INTRA_EVENING_PENALTY` | 500 | Two players partnering again tonight. Hard. |
| Recent history | `RECENT_PAIR_WEIGHT_BANDS` | `[(7, 18), (14, 10), (28, 5)]` | See below. |
| `imbalance` | `PAIR_IMBALANCE_WEIGHT`, `PAIR_IMBALANCE_ESCALATION` | 1, 5 | Difference between the two pairs' rating sums on a doubles court: `diff` for 0-1, then `1 + 5 × (diff − 1)²` (2 → 6, 3 → 21, 4 → 46, 5 → 81). |
| Rating gap (3 bands) | `RATING_GAP_BANDS`, `RATING_GAP_MULT` | 20 / 50 / 100, ×3 | See below. Doubles and singles. |
| `partner_skill_gap` | `PARTNER_SKILL_GAP_WEIGHT`, `PARTNER_SKILL_GAP_THRESHOLD` | 5, 2 | Partners more than 2 rating points apart: `5 × (gap − 2) × (1 + n)`, where n is the two players' earlier high-gap partnerships tonight. |
| `gender_3F1M` | `GENDER_3F1M_PENALTY` | 50 | Three women and one man on a doubles court. |
| `gender_MM_vs_FF` | `GENDER_MM_VS_FF_PENALTY` | 50 | A 2M+2F court split men v women (mixed pairs are fine). |
| `gender_3M1F` | `GENDER_3M1F_PENALTY` | 10 | Three men and one woman. |
| `ffff_court_cap` | `GENDER_FFFF_FREE_ALLOWANCE`, `GENDER_FFFF_PENALTY_BASE`, `GENDER_FFFF_PENALTY_ESCALATION` | 1, 50, ×3 | All-female courts across the evening: the first is free, then 50, 150, 450, … |
| `same_court_successive` | `SAME_COURT_SUCCESSIVE_PENALTY` | 1 | Two players who shared a court in the previous rotation sharing one again. Tie-breaker. |

Gender rules apply to doubles only; `?` genders don't count towards
either side.

**Recent history (partners and opponents).** `recent_pair_weights`
reads `history.json` and puts each past session in the first band whose
`max_days` covers its age in calendar days from `today`: 0-7 days
scores 18, 8-14 days 10, 15-28 days 5, older nothing. Weights add up
across sessions, so partners 3 days ago and again 10 days ago cost
18 + 10 = 28.

* Past **partners** who partner again are penalised. They are keyed by
  the bare pair (`frozenset`).
* Past **opponents** who face each other again are penalised by the
  same bands. They are keyed with `opponent_key(pair)`, so the two
  kinds never mix. This covers doubles cross-pairs and singles
  matches.
* Former partners **facing** each other is not penalised, nor are
  former opponents partnering.
* A past singles match counts as an opponent meeting, not a
  partnership.

**Rating gap.** A court's gap is its strongest-to-weakest rating
difference (`RATING_GAP_BANDS`, with ranges in `RATING_GAP_RANGES`):

| Gap | Band | Base |
|---|---|---:|
| 0-4 | balanced | 0 |
| 5-6 | unbalanced | 20 |
| 7-8 | very unbalanced | 50 |
| 9 | extremely unbalanced | 100 |

The penalty is `base × Σ RATING_GAP_MULT ^ (that player's non-balanced
courts so far tonight)` over the players on the court. Everyone's
first non-balanced rotation is at base cost and each further one is
three times dearer, so mismatches are spread around and kept mild.

### Whole-evening rules

These need the complete evening, so they are added after all rotations
exist (`_fold_evening_items`) and each item is attributed to one
rotation so the breakdown still adds up. They count towards the seed
comparison and the polish, but are not seen by the greedy per-rotation
sampler.

| Rule | Constant(s) | Current value | What it penalises |
|---|---|---:|---|
| `standard_too_low` | `STANDARD_TOO_LOW_WEIGHT`, `STANDARD_TOO_LOW_MATERIAL`, `STANDARD_RULE_RATING_MIN/MAX` | 5 per point | A player rated 3-8 whose best rotation's company (mean rating of the others on court) is still at least 1 point weaker than them. |
| `top_player_no_strong_rotation` | `TOP_PLAYER_STRONG_WEIGHT`, `TOP_PLAYER_MAX_RATING` | 5 per point | A player rated 1-2 who never gets a court where every other player is no worse than their rating + 1. |
| `hard_court_repeat` | `HARD_COURT_REPEAT_WEIGHT`, `HARD_COURT_NUMBERS` | 50 | Players on hard courts (1-4) more than once: `50 × (1 + … + (count − 1))`, so 2 → 50, 3 → 150. |
| `new_faces_missed` | `NEW_FACES_WEIGHT`, `NEW_FACES_MIN_COATTEND` | 3 | A player with a "never-met" co-attendee tonight (2+ shared sessions, never shared a court) who meets none of them. |
| `cross_band_missed` | `CROSS_BAND_WEIGHT`, `CROSS_BAND_NEIGHBOUR_GAP`, `CROSS_BAND_MIN_OTHERS`, `CROSS_BAND_DUE_DAYS` | 5 | A player with no cross-band rotation (2+ others on court rated 2+ points away) in the last 21 days who gets none tonight. |
| `mixed_match_missed` | `MIXED_MATCH_WEIGHT`, `MIXED_MATCH_WEIGHT_CAP` | 5, cap 40 | A player who opted in to mixed play, with an opposite-gender attendee present, who gets no mixed court. Doubles for each session since their last mixed game (5, 10, 20, then 40). |

### Relative sizes

The two hard rules (500) sit well above everything else, so a repeat
is only accepted when nothing else is feasible. The gender, hard-court
and rating-gap rules are in the tens to low hundreds. The recent-history
bands, imbalance at small diffs and the whole-evening nudges are smaller
still, and `same_court_successive` is a tie-breaker. Change weights with
the relative sizes in mind: raising a tie-breaker by an order of
magnitude lets it override balance.

---

## Output and editing

`make_plan` returns a `PairingPlan` with per-rotation `metrics`
(`best_score`, `breakdown`, `breakdown_items` attributing each point to
a court, pair or player), plus `multi_seed` and `polish` blocks. The
bot stores it as the draft in `session_state.json`.

Edits to a draft do not rerun the optimiser; they change the saved
plan directly:

* `swap_players_in_plan` — swap two players for the whole evening or
  one rotation;
* `swap_rotations_in_plan` — swap two rotations' contents (times stay
  with the position);
* `swap_courts_in_plan` — swap two courts' contents across every
  rotation;
* `retime_plan_rotations` — change start time or rotation lengths
  without touching the pairings.

To get a different draft, generate again (a fresh random seed is used
unless one is given).

`boris commit` (`tool_commit_plan`) appends the draft to `history.json`
(`append_to_history`), mirrors it to the Sheet's `Session log` and
`Pair log` tabs, and clears the session. `history.json` is the record
the history-based rules read; the Sheet tabs are for people.
