# Pairing algorithm

How `pairings.py` turns an attendee list and a set of court labels into
a multi-rotation plan: who plays who, on which court, at what time.

The implementation lives in [`pairings.py`](../pairings.py); the entry
point is `make_plan(...)`. Everything below describes the default
strategy `skill_balanced` (currently the only strategy — `random` is
just an alias).

---

## Inputs

* **`attendees`** — full names of the players for the session.
* **`court_labels`** — physical court numbers (e.g. `["3","5","7","8","9","10","11"]`).
  Order matters: singles courts get assigned to the trailing labels
  unless `swap_courts` is used to relocate them.
* **`num_rotations`** — how many rotation blocks (default 3).
* **`rotation_durations`** — per-rotation length in minutes. Defaults to
  `[45, 40, 35]` for 3 rotations (the Thursday club standard) else
  `[40] * num_rotations`.
* **`start_time_hhmm`** — first rotation start (default `"19:30"`).
* **Per-session overrides (no roster change):**
  * `singles_exclude: list[str]` — treat as `avoid` for this run.
  * `singles_include: list[str]` — treat as `prefer` for this run.
  * `pinned_singles: list[{rotation_num, players, court_label?}]` —
    force specific singles matchups before generation.
* **From the roster (Google Sheet "Players" tab):**
  * `rating` — integer 1-5 (lower is stronger) or `?` (treated as 3).
  * `gender` — `M` / `F` / `?`.
  * `singles` — `prefer` / `avoid` / blank (neutral).
* **From `history.json`:** the previous week's plan, used to penalise
  pairs who played together last week.

---

## Stage 1 — court layout

For `n` attendees and `c` courts, capacity is `4c`.

* `n > 4c` → error: drop someone or add a court.
* Even `n` ≤ `4c` → no sit-outs. `(4c − n) / 2` singles courts + the
  rest as doubles. Singles go on the **trailing** court labels.
* Odd `n` → 1 player sits out each rotation (rotated fairly), then the
  same logic on the remaining `n − 1`.

Sit-outs are picked by lowest `sitout_count` so far, random tiebreak.
Pinned-singles players are excluded from the sit-out pool for that
rotation (they have to play singles, so can't sit out).

---

## Stage 2 — pick who plays singles each rotation

Each rotation needs `2 × (singles courts)` singles slots filled.

If a `pinned_singles` entry exists for the rotation, those two players
are forced into the singles slate first; the remainder is picked from
the rest of the active players (excluding pins) by `_select_singles_players`.

`_select_singles_players` sorts candidates by, in order:

1. **Cap reached?** — players already at `MAX_SINGLES_PER_EVENING`
   appearances go last. Only repeated when there aren't enough fresh
   candidates.
2. **Singles preference** — `prefer` first, neutral second, `avoid`
   last.
3. **Rating** — lower is stronger; the strongest play singles first
   (singles is more demanding).
4. **`singles_count` so far** — gentle rotation among ties.
5. **Random** tiebreak.

The first N from the sorted list are the singles slate; everyone else
plays doubles.

---

## Stage 3 — rejection-sample the layout

For each rotation, build up to `MAX_ATTEMPTS = 500` random layouts
and keep the lowest-scoring one. Within one attempt:

1. **Shuffle** doubles players and singles players independently.
2. **Slice** the doubles shuffle into groups of 4 — one per doubles
   court.
3. **For each court, pick the best of 3 pair structures** —
   `_build_best_doubles_court` tries all three ways to partner up the
   four players (`AB-vs-CD`, `AC-vs-BD`, `AD-vs-BC`) and keeps the
   lowest-scoring. Pure local optimisation: same 4 players on the same
   court, no side effects.
4. **Place pinned singles** on the requested court label (or the first
   singles court). Remaining singles players fill the rest.
5. **Score** the whole layout.

Short-circuit if a layout scores zero.

---

## Scoring

Lower is better. The score is summed across all courts in a rotation.

| Penalty | Weight | Applies to | Type |
|---|---:|---|---|
| `INTRA_EVENING_PENALTY` | 100 | A partner pair already used earlier tonight | Hard-ish |
| `OPPONENT_REPEAT_PENALTY` | 500 | A pair who've already faced each other tonight (cross-pair on doubles, or singles match) | **Hard** |
| `WEEKLY_REPEAT_WEIGHTS` | `[10, 5, 2]` | Per pair drawn from `history.json`, indexed by recency: 10 for last week, 5 for 2 weeks ago, 2 for 3 weeks ago. A pair appearing in multiple recent weeks accumulates the sum (so a 3-week-running pair scores 17, not 10). | Soft |
| `PAIR_IMBALANCE_WEIGHT` | 2 × \|sumA − sumB\| | Per doubles court — rating-sum imbalance | Soft (linear) |
| `GENDER_HARD_PENALTY` | 1000 | 3F+1M court, OR a 2M+2F court paired MM-vs-FF | **Hard** |
| `ISOLATED_WOMAN_PENALTY` | 1 | 3M+1F court — gentle nudge toward consolidating women | Tiebreaker |
| `EXCESS_4F_COURT_PENALTY` | 50 | Each all-female (4F) court beyond the **first** across the whole evening — at most 1 4F court per evening unless avoiding it would breach a hard rule | Moderate |
| `SAME_COURT_SUCCESSIVE_PENALTY` | 1 | A pair that shared a court (any role) in the **immediately previous** rotation, sharing again now | Tiebreaker |

Notes on the hard rules:

* `OPPONENT_REPEAT_PENALTY = 500` — Hannah-vs-Louise once means they
  shouldn't face each other again tonight. Algorithm only accepts a
  repeat if no feasible alternative exists.
* `GENDER_HARD_PENALTY = 1000` — 3F+1M and segregated 2v2 courts are
  effectively forbidden. 3M+1F is allowed (different rule set, different
  social dynamic).
* The two hard rules can in principle conflict in tiny groups; the
  500/1000 split means gender wins if it ever comes to it.

Tiebreakers are deliberately tiny (1) so they never override balance
or repeat-avoidance. Their job is to break ties between equivalent
candidate layouts.

---

## State carried across rotations

The strategy maintains four cumulative trackers within a single
`make_plan` call:

* **`sitout_count`** — fairness across rotations.
* **`singles_count`** — feeds the cap rule (`MAX_SINGLES_PER_EVENING = 1`).
* **`intra_partners`** — every doubles partner pair used so far tonight.
* **`intra_opponents`** — every opponent pair used so far tonight
  (4 per doubles court + 1 per singles court).

And one rolling tracker:

* **`prev_court_pairs`** — same-court pairs from the **immediately
  previous** rotation only. Replaced (not unioned) each rotation; that
  is what "successive" means in `SAME_COURT_SUCCESSIVE_PENALTY`.

---

## Per-session overrides

These don't change the roster — they only affect the current generation.

| Parameter | Effect |
|---|---|
| `singles_exclude=[name, …]` | Treat as `avoid` for this run. "Don't put Geoff in singles tonight." |
| `singles_include=[name, …]` | Treat as `prefer` for this run. "Try to put Tim on singles." |
| `pinned_singles=[{rotation_num, players, court_label?}, …]` | Force a singles matchup. Each pinned player counts as their one singles appearance under the cap. The same player can't be pinned to more than one rotation. |

Pinned matchups are placed before the rejection sampler runs, so
doubles balance is optimised **around** the pin instead of being
disrupted by post-hoc swaps.

---

## Post-generation editing

After `make_plan` returns, the bot persists the plan dict as
`session_state.draft_plan` so admins can iterate. Three edit primitives
exist (in `pairings.py`):

* **`swap_players_in_plan(plan, name1, name2, rotation_num=None)`** —
  swap two players' slots. Without `rotation_num`, swap their entire
  evening; with it, only that rotation. Refreshes `bracket_values`
  on affected courts.
* **`swap_rotations_in_plan(plan, a, b)`** — swap two rotations'
  contents. Times stay tied to position (rotation 1 always runs at
  the first time slot).
* **`swap_courts_in_plan(plan, label_a, label_b)`** — swap two courts'
  contents (mode + players + pairs + bracket_values) across every
  rotation. Court labels themselves stay put. Used for "put singles
  on Ct 5".

Edits do **not** re-run the scoring or rebalance anything — they're
literal mutations of the saved draft. Use `generate_pairings(seed=…)`
for a re-roll instead.

---

## Finalisation

`commit_plan` is the only finalisation path. It:

1. Appends the current draft to `history.json` (so next week's
   `weekly_pairs` penalty can see it).
2. Mirrors to the Google Sheet `Session log` and `Pair log` tabs for
   admin browsing.
3. Clears the draft from session state.

`history.json` is the authoritative record of past sessions. The Sheet
mirrors are for humans.

---

## Constants — quick reference

Defined at the top of `pairings.py`:

```python
INTRA_EVENING_PENALTY = 100
WEEKLY_REPEAT_WEIGHTS = [10, 5, 2]   # last week, 2 weeks ago, 3 weeks ago
PAIR_IMBALANCE_WEIGHT = 2
UNKNOWN_RATING = 3
MAX_ATTEMPTS = 500
GENDER_HARD_PENALTY = 1000
ISOLATED_WOMAN_PENALTY = 1
EXCESS_4F_COURT_PENALTY = 50         # at most 1 all-female court per evening
OPPONENT_REPEAT_PENALTY = 500
SAME_COURT_SUCCESSIVE_PENALTY = 1
MAX_SINGLES_PER_EVENING = 1
DEFAULT_ROTATION_DURATIONS_3 = [45, 40, 35]
```

Tweak with care — the relative magnitudes are what make hard rules
hard and tiebreakers tiebreaky. Bumping `SAME_COURT_SUCCESSIVE_PENALTY`
by an order of magnitude, for example, would let it override skill
balance, which is probably not what you want.
