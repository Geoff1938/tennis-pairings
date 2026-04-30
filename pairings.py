"""Thursday Tennis pairings generator (multi-rotation, skill-balanced).

Inputs the caller supplies:
  * the list of attendee full names (as in the roster);
  * the labels of the courts reserved (e.g. ``["4", "5", "6", "7"]``);
  * the number of rotations in the evening (default 3 × 40 min).
  * optional RNG seed + start time.

Capacity policy
---------------

Let ``n = len(attendees)`` and ``c = len(court_labels)``. Each court seats 4
for doubles or 2 for singles, so total capacity is ``4c``:

  * ``n > 4c``          → error, the admin must drop someone.
  * ``n == 4c``         → every court is doubles; no sit-outs.
  * ``n < 4c``, even    → split into ``(4c - n) / 2`` singles courts and the
                          rest as doubles. **No sit-outs.** Singles courts
                          get the highest-labelled courts in the list.
  * ``n < 4c``, odd     → one player sits out each rotation (rotating
                          fairly); the remaining even count is handled as
                          above.

Singles
-------

When singles courts are needed, the players are picked from the stronger end
of the roster (lower ratings = stronger; ``?`` treated as 3 for sorting).
Across the evening, singles participation is rotated among the strongest
players so their match-ups differ from rotation to rotation where possible.

Scoring
-------

Rejection sampling. For each candidate rotation layout we score:

  * ``+INTRA_EVENING_PENALTY`` per pair/matchup that already played earlier
    this evening (dominant signal — mixing partners across blocks is the
    point).
  * ``+WEEKLY_REPEAT_PENALTY`` per pair present in last week's ``history.json``.
  * ``+PAIR_IMBALANCE_WEIGHT × |sumA - sumB|`` per doubles court, where the
    sums are rating totals for each of the two pairs (``?`` → 3).
  * ``+GENDER_HARD_PENALTY`` per doubles court that is 3F+1M (3M+1F is fine)
    or that pairs MM-vs-FF on a 2M+2F court (mixed-doubles MF-vs-MF is
    fine).
  * ``+ISOLATED_WOMAN_PENALTY`` per 3M+1F court — small enough to act as
    a tie-breaker only, so two such courts will, all else equal, collapse
    into 2M+2F + 4M+0F.

The layout with the lowest score wins; we short-circuit at 0.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

# ---------- scoring constants -------------------------------------------

INTRA_EVENING_PENALTY = 100   # partner / singles-matchup already played tonight
WEEKLY_REPEAT_PENALTY = 10    # pair from last week's history
PAIR_IMBALANCE_WEIGHT = 2     # per unit of |pairA_sum - pairB_sum|
UNKNOWN_RATING = 3            # neutral treatment for rating == "?"
MAX_ATTEMPTS = 500            # rejection-sampling cap per rotation
# Gender-composition penalties.
# Hard rule: 3F+1M on a doubles court (3M+1F is allowed) — discouraged
# strongly, alongside 2M-vs-2F segregated matchups (MM pair vs FF pair).
GENDER_HARD_PENALTY = 1000
# Soft preference: each 3M+1F court gets a small penalty so that, all
# else equal, two such courts collapse into one 2M+2F + one 4M+0F.
ISOLATED_WOMAN_PENALTY = 1


# ---------- data classes ------------------------------------------------


@dataclass
class Court:
    """One court's arrangement. ``mode`` is ``"doubles"`` or ``"singles"``.

    For doubles: ``players`` has 4 names, ``pairs`` has two 2-tuples (the
    two partnerships).
    For singles: ``players`` has 2 names, ``pairs`` has one 2-tuple (the
    matchup).
    """

    court_label: str
    mode: str
    players: list[str]
    pairs: list[tuple[str, str]]


@dataclass
class Rotation:
    rotation_num: int
    start_time: str  # "HH:MM"
    end_time: str    # "HH:MM"
    courts: list[Court]
    sit_outs: list[str]


@dataclass
class PairingPlan:
    date: str
    attendees: list[str]
    court_labels: list[str]
    num_rotations: int
    rotations: list[Rotation]
    unknown_attendees: list[str]
    display_names: dict[str, str]
    ratings: dict[str, int]
    strategy: str
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        for rot in d["rotations"]:
            for court in rot["courts"]:
                court["pairs"] = [list(p) for p in court["pairs"]]
        return d


# ---------- display names -----------------------------------------------


def compute_display_names(full_names: Iterable[str]) -> dict[str, str]:
    """Return ``{full_name: short_display_name}`` for a list of full names.

    ``First L`` style where ``L`` is the shortest surname prefix unique
    within each first-name group. Single-token names stay as-is.
    """
    names = list(dict.fromkeys(full_names))
    parsed: dict[str, tuple[str, str]] = {}
    for n in names:
        tokens = n.strip().split()
        if not tokens:
            parsed[n] = ("", "")
            continue
        parsed[n] = (tokens[0], " ".join(tokens[1:]) if len(tokens) > 1 else "")

    from collections import defaultdict

    groups: dict[str, list[str]] = defaultdict(list)
    for n in names:
        groups[parsed[n][0].lower()].append(n)

    display: dict[str, str] = {}
    for bucket in groups.values():
        if len(bucket) == 1:
            n = bucket[0]
            first, _ = parsed[n]
            display[n] = first or n
            continue
        surnames_in_bucket = [parsed[n][1] for n in bucket if parsed[n][1]]
        for n in bucket:
            first, surname = parsed[n]
            if not surname:
                display[n] = first
                continue
            chosen: str | None = None
            for k in range(1, len(surname) + 1):
                prefix = surname[:k].lower()
                collides = any(
                    other_surname != surname
                    and other_surname.lower().startswith(prefix)
                    for other_surname in surnames_in_bucket
                )
                if not collides:
                    chosen = (
                        surname if k >= len(surname) else surname[:k].capitalize()
                    )
                    break
            display[n] = f"{first} {chosen}" if chosen else f"{first} {surname}"
        seen: dict[str, int] = {}
        for n in bucket:
            rendered = display[n]
            if rendered in seen:
                seen[rendered] += 1
                display[n] = f"{rendered} #{seen[rendered]}"
            else:
                seen[rendered] = 1
    return display


# ---------- loaders -----------------------------------------------------


def load_players(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_history(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def recent_pairs(history: list[dict], lookback: int = 1) -> set[frozenset]:
    """Pairs played in the most recent ``lookback`` weeks."""
    pairs: set[frozenset] = set()
    for week in history[-lookback:]:
        for rot in week.get("rotations", []):
            for court in rot.get("courts", []):
                for pair in court.get("pairs", []):
                    if len(pair) == 2:
                        pairs.add(frozenset(pair))
    return pairs


# ---------- time helpers ------------------------------------------------


def _add_minutes(hhmm: str, minutes: int) -> str:
    h, m = map(int, hhmm.split(":"))
    total = h * 60 + m + minutes
    h2, m2 = divmod(total, 60)
    return f"{h2:02d}:{m2:02d}"


# Default rotation durations for the standard Thursday session.
DEFAULT_ROTATION_DURATIONS_3 = [45, 40, 35]


def _resolve_durations(
    num_rotations: int, rotation_durations: list[int] | None
) -> list[int]:
    """Pick the per-rotation length list, falling back to club defaults."""
    if rotation_durations is not None:
        if len(rotation_durations) != num_rotations:
            raise ValueError(
                f"rotation_durations has {len(rotation_durations)} entries; "
                f"expected {num_rotations}"
            )
        return list(rotation_durations)
    if num_rotations == 3:
        return list(DEFAULT_ROTATION_DURATIONS_3)
    return [40] * num_rotations


# ---------- ratings helper ----------------------------------------------


def _build_ratings(players: dict[str, dict]) -> dict[str, int]:
    """Extract name -> numeric rating, treating '?' / missing as UNKNOWN_RATING."""
    ratings: dict[str, int] = {}
    for name, info in players.items():
        r = info.get("rating", "?")
        if isinstance(r, int):
            ratings[name] = r
        elif isinstance(r, str):
            try:
                ratings[name] = int(r)
            except ValueError:
                ratings[name] = UNKNOWN_RATING
        else:
            ratings[name] = UNKNOWN_RATING
    return ratings


# ---------- strategy ----------------------------------------------------


StrategyFn = Callable[
    [
        list[str],
        list[str],
        int,
        dict[str, int],
        dict[str, str],
        dict[str, str],
        set[frozenset],
        random.Random,
    ],
    list[tuple[list[Court], list[str]]],
]


def _pair_rating_sum(
    pair: tuple[str, str], ratings: dict[str, int]
) -> int:
    return (
        ratings.get(pair[0], UNKNOWN_RATING)
        + ratings.get(pair[1], UNKNOWN_RATING)
    )


def _gender_court_penalty(c: Court, genders: dict[str, str]) -> int:
    """Gender-composition penalty for one doubles court.

    Three rules:
      1. 3F+1M is forbidden (3M+1F is fine) → ``GENDER_HARD_PENALTY``.
      2. A 2M+2F court paired as MM-vs-FF is forbidden (mixed pairings
         within the same 2+2 court are fine) → ``GENDER_HARD_PENALTY``.
      3. 3M+1F is mildly disfavoured so two such courts will, all else
         equal, collapse into 2M+2F + 4M+0F → ``ISOLATED_WOMAN_PENALTY``.
    Singles courts have no gender penalty.
    """
    if c.mode != "doubles":
        return 0
    g = [genders.get(p, "?") for p in c.players]
    f_count = g.count("F")
    m_count = g.count("M")
    penalty = 0
    if f_count == 3 and m_count == 1:
        penalty += GENDER_HARD_PENALTY
    if f_count == 2 and m_count == 2:
        pair_a, pair_b = c.pairs
        gen_a = sorted(genders.get(p, "?") for p in pair_a)
        gen_b = sorted(genders.get(p, "?") for p in pair_b)
        if {tuple(gen_a), tuple(gen_b)} == {("F", "F"), ("M", "M")}:
            penalty += GENDER_HARD_PENALTY
    if m_count == 3 and f_count == 1:
        penalty += ISOLATED_WOMAN_PENALTY
    return penalty


def _score_doubles_courts(
    courts: list[Court],
    weekly_pairs: set[frozenset],
    intra_pairs: set[frozenset],
    ratings: dict[str, int],
    genders: dict[str, str],
) -> int:
    score = 0
    for c in courts:
        if c.mode != "doubles":
            continue
        pair_a, pair_b = c.pairs[0], c.pairs[1]
        for pair in (pair_a, pair_b):
            fs = frozenset(pair)
            if fs in intra_pairs:
                score += INTRA_EVENING_PENALTY
            if fs in weekly_pairs:
                score += WEEKLY_REPEAT_PENALTY
        imbalance = abs(
            _pair_rating_sum(pair_a, ratings) - _pair_rating_sum(pair_b, ratings)
        )
        score += PAIR_IMBALANCE_WEIGHT * imbalance
        score += _gender_court_penalty(c, genders)
    return score


def _score_singles_courts(
    courts: list[Court],
    intra_pairs: set[frozenset],
) -> int:
    score = 0
    for c in courts:
        if c.mode != "singles":
            continue
        match = c.pairs[0]
        if frozenset(match) in intra_pairs:
            score += INTRA_EVENING_PENALTY
    return score


def _try_layout(
    doubles_players: list[str],
    singles_players: list[str],
    doubles_labels: list[str],
    singles_labels: list[str],
    weekly_pairs: set[frozenset],
    intra_pairs: set[frozenset],
    ratings: dict[str, int],
    genders: dict[str, str],
    rng: random.Random,
) -> tuple[list[Court], int]:
    """Build one random layout and return (courts, score)."""
    shuffled_d = doubles_players[:]
    rng.shuffle(shuffled_d)
    shuffled_s = singles_players[:]
    rng.shuffle(shuffled_s)

    courts: list[Court] = []
    for i, label in enumerate(doubles_labels):
        four = shuffled_d[i * 4 : (i + 1) * 4]
        pair_a = (four[0], four[1])
        pair_b = (four[2], four[3])
        courts.append(
            Court(
                court_label=label,
                mode="doubles",
                players=four,
                pairs=[pair_a, pair_b],
            )
        )
    for i, label in enumerate(singles_labels):
        two = shuffled_s[i * 2 : (i + 1) * 2]
        courts.append(
            Court(
                court_label=label,
                mode="singles",
                players=two,
                pairs=[(two[0], two[1])],
            )
        )
    score = _score_doubles_courts(
        courts, weekly_pairs, intra_pairs, ratings, genders
    ) + _score_singles_courts(courts, intra_pairs)
    return courts, score


_SINGLES_PREF_RANK = {"prefer": 0, "": 1, "avoid": 2}


def _select_singles_players(
    active: list[str],
    num_singles_slots: int,
    ratings: dict[str, int],
    singles_prefs: dict[str, str],
    singles_count: dict[str, int],
    rng: random.Random,
) -> list[str]:
    """Pick the subset of ``active`` to send to singles courts this rotation.

    Ordering keys (all ascending):
      1. ``singles_prefs`` — ``"prefer"`` players come first, ``"avoid"``
         only get picked if forced (slots > prefer + neutral).
      2. ``rating`` — lower is stronger; unknown ratings → ``UNKNOWN_RATING``.
      3. ``singles_count`` so far — rotate the strongest through singles.
      4. Random tie-break.
    """
    if num_singles_slots <= 0:
        return []
    keyed = sorted(
        active,
        key=lambda p: (
            _SINGLES_PREF_RANK.get(singles_prefs.get(p, ""), 1),
            ratings.get(p, UNKNOWN_RATING),
            singles_count.get(p, 0),
            rng.random(),
        ),
    )
    return keyed[:num_singles_slots]


def skill_balanced_multi_rotation(
    attendees: list[str],
    court_labels: list[str],
    num_rotations: int,
    ratings: dict[str, int],
    genders: dict[str, str],
    singles_prefs: dict[str, str],
    weekly_pairs: set[frozenset],
    rng: random.Random,
) -> list[tuple[list[Court], list[str]]]:
    """Build ``num_rotations`` rotations of mixed doubles+singles courts."""
    n = len(attendees)
    c = len(court_labels)
    capacity = 4 * c
    if n > capacity:
        raise ValueError(
            f"{n} attendees exceeds capacity ({capacity} = 4×{c} courts). "
            "Drop someone or add a court."
        )
    # Odd count: one rotating sit-out. Even count → zero.
    sit_outs_per_rotation = 1 if n % 2 == 1 else 0
    effective_n = n - sit_outs_per_rotation
    # Singles courts absorb the shortfall; each singles court is −2 capacity.
    num_singles_courts = (capacity - effective_n) // 2
    num_doubles_courts = c - num_singles_courts
    # Singles go on the highest-labelled courts (last entries of court_labels).
    doubles_labels = list(court_labels[:num_doubles_courts])
    singles_labels = list(court_labels[num_doubles_courts:])

    sitout_count: dict[str, int] = {p: 0 for p in attendees}
    singles_count: dict[str, int] = {p: 0 for p in attendees}
    intra_pairs: set[frozenset] = set()
    rotations: list[tuple[list[Court], list[str]]] = []

    for _ in range(num_rotations):
        # Fair sit-out selection
        if sit_outs_per_rotation:
            ranked = sorted(
                attendees, key=lambda p: (sitout_count[p], rng.random())
            )
            sit_outs = ranked[:sit_outs_per_rotation]
            for s in sit_outs:
                sitout_count[s] += 1
        else:
            sit_outs = []
        active = [p for p in attendees if p not in sit_outs]

        # Pick singles-destined players this rotation (with matchup-rotation
        # bias via singles_count).
        singles_slots = 2 * num_singles_courts
        singles_players = _select_singles_players(
            active, singles_slots, ratings, singles_prefs, singles_count, rng
        )
        doubles_players = [p for p in active if p not in singles_players]

        # Rejection-sample layouts
        best_courts: list[Court] | None = None
        best_score: int | None = None
        for _attempt in range(MAX_ATTEMPTS):
            courts, score = _try_layout(
                doubles_players,
                singles_players,
                doubles_labels,
                singles_labels,
                weekly_pairs,
                intra_pairs,
                ratings,
                genders,
                rng,
            )
            if best_score is None or score < best_score:
                best_courts = courts
                best_score = score
                if score == 0:
                    break
        assert best_courts is not None

        # Update tracking
        for s in singles_players:
            singles_count[s] += 1
        for c_ in best_courts:
            for pair in c_.pairs:
                intra_pairs.add(frozenset(pair))

        rotations.append((best_courts, sit_outs))

    return rotations


STRATEGIES: dict[str, StrategyFn] = {
    "skill_balanced": skill_balanced_multi_rotation,
    "random": skill_balanced_multi_rotation,  # alias — old name kept
}


# ---------- public API --------------------------------------------------


def make_plan(
    attendees: Iterable[str],
    players_path: str | Path | dict,
    history_path: str | Path,
    num_courts: int | None = None,
    court_labels: list | None = None,
    num_rotations: int = 3,
    start_time_hhmm: str = "19:30",
    rotation_durations: list[int] | None = None,
    strategy: str = "skill_balanced",
    seed: int | None = None,
    today: date | None = None,
    singles_exclude: list[str] | None = None,
    singles_include: list[str] | None = None,
) -> PairingPlan:
    """Build a pairing plan spanning ``num_rotations`` blocks of play.

    Supply either ``num_courts`` (labels default to ``"1", "2", …``) or
    explicit ``court_labels`` (e.g. ``[4, 5, 6, 7]`` — stringified).
    """
    attendees = list(attendees)
    if court_labels is not None:
        labels_list = [str(x) for x in court_labels]
    elif num_courts is not None:
        if num_courts < 1:
            raise ValueError("num_courts must be >= 1")
        labels_list = [str(i + 1) for i in range(num_courts)]
    else:
        raise ValueError("must supply num_courts or court_labels")
    if num_rotations < 1:
        raise ValueError("num_rotations must be >= 1")

    durations = _resolve_durations(num_rotations, rotation_durations)
    starts: list[str] = []
    ends: list[str] = []
    cursor = start_time_hhmm
    for d in durations:
        starts.append(cursor)
        cursor = _add_minutes(cursor, d)
        ends.append(cursor)

    players = (
        players_path
        if isinstance(players_path, dict)
        else load_players(players_path)
    )
    history = load_history(history_path)
    weekly_pairs = recent_pairs(history, lookback=1)
    ratings = _build_ratings(players)
    genders: dict[str, str] = {
        n: (str(info.get("gender", "?")).strip().upper() or "?")
        for n, info in players.items()
    }
    singles_prefs: dict[str, str] = {
        n: str(info.get("singles", "")).strip().lower()
        for n, info in players.items()
    }
    # Per-session overrides that don't touch the roster.
    exclude_set = set(singles_exclude or [])
    include_set = set(singles_include or [])
    overlap = exclude_set & include_set
    if overlap:
        raise ValueError(
            "names appear in both singles_exclude and singles_include: "
            f"{sorted(overlap)}"
        )
    for n in exclude_set:
        singles_prefs[n] = "avoid"
    for n in include_set:
        singles_prefs[n] = "prefer"
    unknown_attendees = [a for a in attendees if a not in players]
    display_names = compute_display_names(attendees)

    notes_parts: list[str] = []
    rotations: list[Rotation] = []

    if len(attendees) < 4:
        notes_parts.append(f"Only {len(attendees)} attendees; need at least 4.")
        for r in range(num_rotations):
            rotations.append(
                Rotation(
                    rotation_num=r + 1,
                    start_time=starts[r],
                    end_time=ends[r],
                    courts=[],
                    sit_outs=attendees[:],
                )
            )
        return PairingPlan(
            date=(today or date.today()).isoformat(),
            attendees=attendees,
            court_labels=labels_list,
            num_rotations=num_rotations,
            rotations=rotations,
            unknown_attendees=unknown_attendees,
            display_names=display_names,
            ratings=ratings,
            strategy=strategy,
            notes=" ".join(notes_parts),
        )

    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}; available: {sorted(STRATEGIES)}"
        )

    rng = random.Random(seed)
    per_rotation = STRATEGIES[strategy](
        attendees, labels_list, num_rotations,
        ratings, genders, singles_prefs, weekly_pairs, rng,
    )

    rotations = [
        Rotation(
            rotation_num=i + 1,
            start_time=starts[i],
            end_time=ends[i],
            courts=courts,
            sit_outs=sit_outs,
        )
        for i, (courts, sit_outs) in enumerate(per_rotation)
    ]

    # Note if we had to run any singles courts / sit-outs.
    any_singles = any(c.mode == "singles" for r in rotations for c in r.courts)
    if any_singles:
        n_s = sum(1 for c in rotations[0].courts if c.mode == "singles")
        notes_parts.append(
            f"{n_s} singles court(s) per rotation (attendees < full doubles capacity)."
        )
    if any(r.sit_outs for r in rotations):
        notes_parts.append(
            "Odd attendee count — 1 player sits out each rotation, rotated fairly."
        )

    return PairingPlan(
        date=(today or date.today()).isoformat(),
        attendees=attendees,
        court_labels=labels_list,
        num_rotations=num_rotations,
        rotations=rotations,
        unknown_attendees=unknown_attendees,
        display_names=display_names,
        ratings=ratings,
        strategy=strategy,
        notes=" ".join(notes_parts),
    )


def append_to_history(plan: PairingPlan | dict, history_path: str | Path) -> None:
    """Append ``plan`` as a new record in ``history.json``.

    Accepts either a ``PairingPlan`` (CLI / fresh-from-make_plan) or its
    ``to_dict`` form (admin_bot path, where the plan has been persisted
    to session_state and possibly edited via swap_players / etc.).
    """
    plan_dict = plan if isinstance(plan, dict) else plan.to_dict()
    history_path = Path(history_path)
    history = load_history(history_path) if history_path.exists() else []
    history.append(plan_dict)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# ---------- plan editing (post-generation, pre-commit) ------------------


def swap_players_in_plan(
    plan: dict, name1: str, name2: str, rotation_num: int | None = None
) -> list[int]:
    """Swap the schedule slots of ``name1`` and ``name2`` in ``plan``.

    If ``rotation_num`` is given, only that rotation is affected. Otherwise
    every rotation in which BOTH names appear (on a court or in the sit-out
    list) is updated. Returns the list of rotation numbers actually swapped.
    Raises ``KeyError`` if no rotation contains both names.
    """
    rots = plan.get("rotations", [])
    if rotation_num is not None:
        if not 1 <= rotation_num <= len(rots):
            raise ValueError(
                f"rotation_num {rotation_num} out of range 1..{len(rots)}"
            )
        target_indices = [rotation_num - 1]
    else:
        target_indices = list(range(len(rots)))

    swapped: list[int] = []
    for idx in target_indices:
        rot = rots[idx]
        names_here = {p for c in rot["courts"] for p in c["players"]} | set(
            rot.get("sit_outs", [])
        )
        if name1 not in names_here or name2 not in names_here:
            continue
        replace = {name1: name2, name2: name1}
        for c in rot["courts"]:
            c["players"] = [replace.get(p, p) for p in c["players"]]
            c["pairs"] = [
                [replace.get(p, p) for p in pair] for pair in c["pairs"]
            ]
        rot["sit_outs"] = [
            replace.get(p, p) for p in rot.get("sit_outs", [])
        ]
        swapped.append(idx + 1)
    if not swapped:
        raise KeyError(
            f"no rotation contained both {name1!r} and {name2!r}"
        )
    return swapped


def swap_rotations_in_plan(plan: dict, a: int, b: int) -> None:
    """Swap the content of rotations ``a`` and ``b`` (1-indexed).

    Times stay attached to position — the rotation that ends up first
    runs at the first time slot, and so on. ``rotation_num`` /
    ``start_time`` / ``end_time`` are NOT swapped, only the courts and
    sit-outs payloads.
    """
    rots = plan.get("rotations", [])
    if not (1 <= a <= len(rots) and 1 <= b <= len(rots)):
        raise ValueError(
            f"rotations {a},{b} out of range 1..{len(rots)}"
        )
    if a == b:
        return
    ra, rb = rots[a - 1], rots[b - 1]
    ra["courts"], rb["courts"] = rb["courts"], ra["courts"]
    ra["sit_outs"], rb["sit_outs"] = rb["sit_outs"], ra["sit_outs"]


# ---------- CLI ----------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Generate Thursday Tennis pairings (skill-balanced)."
    )
    parser.add_argument("--players", default="players.json")
    parser.add_argument("--history", default="history.json")
    parser.add_argument(
        "--courts",
        help="Comma-separated court labels, e.g. '4,5,6,7'. Overrides --num-courts.",
    )
    parser.add_argument("--num-courts", type=int)
    parser.add_argument("--rotations", type=int, default=3)
    parser.add_argument("--start-time", default="19:30", help="HH:MM")
    parser.add_argument(
        "--rotation-durations",
        help="Comma-separated minutes per rotation, e.g. '45,40,35'. "
        "Defaults to 45,40,35 for 3 rotations, else 40 each.",
    )
    parser.add_argument(
        "--strategy", default="skill_balanced", choices=sorted(STRATEGIES)
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--attendees-file",
        help="path to a text file with one name per line (otherwise reads from stdin)",
    )
    args = parser.parse_args()

    court_labels = [c.strip() for c in args.courts.split(",")] if args.courts else None
    rotation_durations = (
        [int(x) for x in args.rotation_durations.split(",")]
        if args.rotation_durations
        else None
    )
    if args.attendees_file:
        with open(args.attendees_file, encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
    else:
        names = [ln.strip() for ln in sys.stdin if ln.strip()]

    plan = make_plan(
        names,
        players_path=args.players,
        history_path=args.history,
        num_courts=args.num_courts,
        court_labels=court_labels,
        num_rotations=args.rotations,
        start_time_hhmm=args.start_time,
        rotation_durations=rotation_durations,
        strategy=args.strategy,
        seed=args.seed,
    )
    json.dump(plan.to_dict(), sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
