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
            first, surname = parsed[n]
            display[n] = f"{first} {surname[0].upper()}" if surname else first
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


def _score_doubles_courts(
    courts: list[Court],
    weekly_pairs: set[frozenset],
    intra_pairs: set[frozenset],
    ratings: dict[str, int],
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
        courts, weekly_pairs, intra_pairs, ratings
    ) + _score_singles_courts(courts, intra_pairs)
    return courts, score


def _select_singles_players(
    active: list[str],
    num_singles_slots: int,
    ratings: dict[str, int],
    singles_count: dict[str, int],
    rng: random.Random,
) -> list[str]:
    """Pick the subset of ``active`` to send to singles courts this rotation.

    Ordering keys (all ascending):
      1. ``rating`` — lower is stronger; unknown ratings become ``UNKNOWN_RATING``.
      2. ``singles_count`` so far — rotate the strongest players through singles.
      3. Random tie-break.
    """
    if num_singles_slots <= 0:
        return []
    keyed = sorted(
        active,
        key=lambda p: (
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
            active, singles_slots, ratings, singles_count, rng
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
    rotation_minutes: int = 40,
    strategy: str = "skill_balanced",
    seed: int | None = None,
    today: date | None = None,
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

    players = (
        players_path
        if isinstance(players_path, dict)
        else load_players(players_path)
    )
    history = load_history(history_path)
    weekly_pairs = recent_pairs(history, lookback=1)
    ratings = _build_ratings(players)
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
                    start_time=_add_minutes(start_time_hhmm, r * rotation_minutes),
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
            strategy=strategy,
            notes=" ".join(notes_parts),
        )

    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}; available: {sorted(STRATEGIES)}"
        )

    rng = random.Random(seed)
    per_rotation = STRATEGIES[strategy](
        attendees, labels_list, num_rotations, ratings, weekly_pairs, rng
    )

    rotations = [
        Rotation(
            rotation_num=i + 1,
            start_time=_add_minutes(start_time_hhmm, i * rotation_minutes),
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
        strategy=strategy,
        notes=" ".join(notes_parts),
    )


def append_to_history(plan: PairingPlan, history_path: str | Path) -> None:
    """Append ``plan`` as a new record in ``history.json``."""
    history_path = Path(history_path)
    history = load_history(history_path) if history_path.exists() else []
    history.append(plan.to_dict())
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


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
    parser.add_argument("--rotation-minutes", type=int, default=40)
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
        rotation_minutes=args.rotation_minutes,
        strategy=args.strategy,
        seed=args.seed,
    )
    json.dump(plan.to_dict(), sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
