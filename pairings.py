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

  * ``+INTRA_EVENING_PENALTY`` per partner pair already played tonight
    (mixing partners across blocks is the point).
  * ``+OPPONENT_REPEAT_PENALTY`` per opponent matchup already played
    tonight — hard rule, e.g. Hannah-vs-Louise once means they shouldn't
    face each other again.
  * ``+SAME_COURT_SUCCESSIVE_PENALTY`` per pair that shared a court in
    the immediately previous rotation. Soft preference; often
    unavoidable, partners-then-opponents is acceptable.
  * Per pair drawn from ``history.json``, the weight at
    ``WEEKLY_REPEAT_WEIGHTS[recency]`` (default ``[10, 5, 2]`` for the
    last 3 weeks). A pair appearing in multiple recent weeks accumulates
    the sum, so a 3-week-running pair is penalised more than one that
    only played together once.
  * ``+PAIR_IMBALANCE_WEIGHT × |sumA - sumB|`` per doubles court, where the
    sums are rating totals for each of the two pairs (``?`` → 3).
  * ``+GENDER_HARD_PENALTY`` per doubles court that is 3F+1M (3M+1F is fine)
    or that pairs MM-vs-FF on a 2M+2F court (mixed-doubles MF-vs-MF is
    fine).
  * ``+ISOLATED_WOMAN_PENALTY`` per 3M+1F court — small enough to act as
    a tie-breaker only, so two such courts will, all else equal, collapse
    into 2M+2F + 4M+0F.
  * ``+EXCESS_4F_COURT_PENALTY`` per all-female (4F) court beyond the
    first across the WHOLE evening. Moderate priority — accepted only
    when the alternative would breach a hard rule.

The layout with the lowest score wins; we short-circuit at 0.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

# ---------- scoring constants -------------------------------------------

INTRA_EVENING_PENALTY = 100   # partner pair already played tonight
# Weights applied per pair drawn from history.json, indexed by recency.
# Index 0 = last week, index 1 = 2 weeks ago, etc. A pair appearing in
# multiple recent weeks accumulates the sum of those weights, so someone
# you've played 3 weeks running is penalised more than someone you only
# saw once.
WEEKLY_REPEAT_WEIGHTS: list[int] = [10, 5, 2]
PAIR_IMBALANCE_WEIGHT = 2     # per unit of |pairA_sum - pairB_sum|
UNKNOWN_RATING = 3            # neutral treatment for rating == "?"
MAX_ATTEMPTS = 2000           # rejection-sampling cap per rotation
# Hard rule: a pair that has already been opponents this evening (cross-
# pair on a doubles court, or the singles matchup) shouldn't face each
# other again in any later rotation. High penalty; algorithm only
# accepts a repeat when there's no feasible alternative.
OPPONENT_REPEAT_PENALTY = 500
# Soft rule: if two players shared a court (as partners or opponents)
# in the IMMEDIATELY previous rotation, prefer them not to share a
# court again in the current rotation. Tiebreaker-grade penalty —
# often unavoidable, so kept low so it doesn't dominate balance.
SAME_COURT_SUCCESSIVE_PENALTY = 1
# Gender-composition penalties.
# Hard rule: 3F+1M on a doubles court (3M+1F is allowed) — discouraged
# strongly, alongside 2M-vs-2F segregated matchups (MM pair vs FF pair).
GENDER_HARD_PENALTY = 1000
# Soft preference: each 3M+1F court gets a small penalty so that, all
# else equal, two such courts collapse into one 2M+2F + one 4M+0F.
ISOLATED_WOMAN_PENALTY = 1
# Moderate-priority rule: at most ONE all-female (4F) court in the
# whole evening. The first 4F court is free; any additional 4F court
# (across all rotations) attracts this penalty per court. Sits between
# soft preferences and hard rules so it can be overridden when the
# alternative would be worse (e.g. forcing a 3F+1M court).
EXCESS_4F_COURT_PENALTY = 50


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
    # Diagnostics: total wall-clock seconds for the make_plan call, and
    # one entry per rotation with {attempts_made, best_score}. Useful
    # for tuning MAX_ATTEMPTS and weights, and for reporting back to the
    # admin how hard the algorithm had to work.
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        ratings = d.get("ratings", {})

        def _r(name: str) -> int:
            v = ratings.get(name, UNKNOWN_RATING)
            return v if isinstance(v, int) else UNKNOWN_RATING

        for rot in d["rotations"]:
            for court in rot["courts"]:
                court["pairs"] = [list(p) for p in court["pairs"]]
                # Pre-compute the bracket numbers Boris renders in DRAFT
                # mode so it doesn't have to do arithmetic in its head.
                # For doubles, two pair sums; for singles, two individual
                # ratings (one per player).
                if court["mode"] == "doubles":
                    court["bracket_values"] = [
                        _r(court["pairs"][0][0]) + _r(court["pairs"][0][1]),
                        _r(court["pairs"][1][0]) + _r(court["pairs"][1][1]),
                    ]
                elif court["mode"] == "singles":
                    court["bracket_values"] = [
                        _r(court["players"][0]),
                        _r(court["players"][1]),
                    ]
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
    """Pairs played in the most recent ``lookback`` weeks (unweighted)."""
    pairs: set[frozenset] = set()
    for week in history[-lookback:]:
        for rot in week.get("rotations", []):
            for court in rot.get("courts", []):
                for pair in court.get("pairs", []):
                    if len(pair) == 2:
                        pairs.add(frozenset(pair))
    return pairs


def recent_pair_weights(
    history: list[dict],
    weights: list[int] | None = None,
) -> dict[frozenset, int]:
    """Map each recent pair to its accumulated penalty weight.

    ``weights[0]`` applies to pairs from the most-recent session,
    ``weights[1]`` to the session before that, and so on. Pairs
    appearing in multiple recent sessions accumulate the sum (so a pair
    that's played together each of the last 3 weeks is penalised more
    than one that played only once). Defaults to
    ``WEEKLY_REPEAT_WEIGHTS``.
    """
    weights = list(weights if weights is not None else WEEKLY_REPEAT_WEIGHTS)
    if not weights:
        return {}
    out: dict[frozenset, int] = {}
    # Walk the tail of history, applying weights[0] to the LAST entry
    # (most recent), weights[1] to the second-to-last, etc.
    recent = history[-len(weights):][::-1]  # most-recent first
    for offset, week in enumerate(recent):
        if offset >= len(weights):
            break
        w = weights[offset]
        for rot in week.get("rotations", []):
            for court in rot.get("courts", []):
                for pair in court.get("pairs", []):
                    if len(pair) == 2:
                        fs = frozenset(pair)
                        out[fs] = out.get(fs, 0) + w
    return out


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
        dict[frozenset, int],
        dict[int, dict],
        random.Random,
    ],
    list[tuple[list[Court], list[str], dict]],
]


def _validate_pinned_singles(
    pinned_singles: list[dict] | None,
    attendees_set: set[str],
    num_rotations: int,
    court_labels_set: set[str],
) -> dict[int, dict]:
    """Validate ``pinned_singles`` and return ``{rotation_num: pin}``.

    Each pin is ``{"players": (n1, n2), "court_label": str | None}``.
    Raises ``ValueError`` for malformed input — at most one pin per
    rotation, both players must be attendees, and no player may be
    pinned to more than one rotation (would breach the per-evening
    cap).
    """
    if not pinned_singles:
        return {}
    out: dict[int, dict] = {}
    seen: set[str] = set()
    for pin in pinned_singles:
        rot = pin.get("rotation_num")
        players = pin.get("players")
        court_label = pin.get("court_label")
        if not isinstance(rot, int) or not (1 <= rot <= num_rotations):
            raise ValueError(
                f"pinned_singles.rotation_num {rot!r} must be int in 1..{num_rotations}"
            )
        if rot in out:
            raise ValueError(f"pinned_singles: rotation {rot} pinned twice")
        if not isinstance(players, (list, tuple)) or len(players) != 2:
            raise ValueError(
                f"pinned_singles.players must list exactly 2 names (got {players!r})"
            )
        for p in players:
            if p not in attendees_set:
                raise ValueError(f"pinned_singles player {p!r} not in attendees")
            if p in seen:
                raise ValueError(
                    f"pinned_singles player {p!r} pinned to multiple rotations "
                    "(would exceed the singles-per-evening cap)"
                )
            seen.add(p)
        if court_label is not None and str(court_label) not in court_labels_set:
            raise ValueError(
                f"pinned_singles.court_label {court_label!r} not in court_labels"
            )
        out[rot] = {
            "players": (players[0], players[1]),
            "court_label": str(court_label) if court_label is not None else None,
        }
    return out


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


def _doubles_opponent_pairs(pa, pb) -> list[frozenset]:
    """The 4 opponent pairs on a doubles court (cross-pair combinations)."""
    return [
        frozenset([pa[0], pb[0]]),
        frozenset([pa[0], pb[1]]),
        frozenset([pa[1], pb[0]]),
        frozenset([pa[1], pb[1]]),
    ]


def _court_pair_combinations(players: list[str]) -> list[frozenset]:
    """All pairs of players on a court (used for the same-court tracking)."""
    out: list[frozenset] = []
    for i, p1 in enumerate(players):
        for p2 in players[i + 1 :]:
            out.append(frozenset([p1, p2]))
    return out


def _score_doubles_court(
    court: Court,
    weekly_pair_penalties: dict[frozenset, int],
    intra_partners: set[frozenset],
    intra_opponents: set[frozenset],
    prev_court_pairs: set[frozenset],
    ratings: dict[str, int],
    genders: dict[str, str],
) -> int:
    """Score a single doubles court — repeats + imbalance + gender penalty."""
    if court.mode != "doubles":
        return 0
    score = 0
    pair_a, pair_b = court.pairs[0], court.pairs[1]
    for pair in (pair_a, pair_b):
        fs = frozenset(pair)
        if fs in intra_partners:
            score += INTRA_EVENING_PENALTY
        score += weekly_pair_penalties.get(fs, 0)
    for op in _doubles_opponent_pairs(pair_a, pair_b):
        if op in intra_opponents:
            score += OPPONENT_REPEAT_PENALTY
    for cp in _court_pair_combinations(court.players):
        if cp in prev_court_pairs:
            score += SAME_COURT_SUCCESSIVE_PENALTY
    imbalance = abs(
        _pair_rating_sum(pair_a, ratings) - _pair_rating_sum(pair_b, ratings)
    )
    score += PAIR_IMBALANCE_WEIGHT * imbalance
    score += _gender_court_penalty(court, genders)
    return score


def _score_doubles_courts(
    courts: list[Court],
    weekly_pair_penalties: dict[frozenset, int],
    intra_partners: set[frozenset],
    intra_opponents: set[frozenset],
    prev_court_pairs: set[frozenset],
    ratings: dict[str, int],
    genders: dict[str, str],
) -> int:
    return sum(
        _score_doubles_court(
            c, weekly_pair_penalties, intra_partners, intra_opponents,
            prev_court_pairs, ratings, genders,
        )
        for c in courts
    )


def _build_best_doubles_court(
    four: list[str],
    label: str,
    weekly_pair_penalties: dict[frozenset, int],
    intra_partners: set[frozenset],
    intra_opponents: set[frozenset],
    prev_court_pairs: set[frozenset],
    ratings: dict[str, int],
    genders: dict[str, str],
) -> Court:
    """Return the lowest-scoring Court for these 4 players (mode=doubles).

    Tries all 3 ways to split four players into two pairs and picks the
    one minimising the per-court score (imbalance + repeat penalties +
    gender rules). This is a local optimisation over the random shuffle:
    once player-to-court assignment is decided, picking the best pair
    structure is free — same 4 players, no side effects elsewhere.
    """
    a, b, c, d = four
    candidates = (
        ((a, b), (c, d)),
        ((a, c), (b, d)),
        ((a, d), (b, c)),
    )
    best: Court | None = None
    best_score = float("inf")
    for pa, pb in candidates:
        court = Court(
            court_label=label,
            mode="doubles",
            players=four,
            pairs=[pa, pb],
        )
        s = _score_doubles_court(
            court, weekly_pair_penalties, intra_partners, intra_opponents,
            prev_court_pairs, ratings, genders,
        )
        if s < best_score:
            best = court
            best_score = s
    assert best is not None
    return best


def _score_singles_courts(
    courts: list[Court],
    intra_opponents: set[frozenset],
    prev_court_pairs: set[frozenset],
) -> int:
    score = 0
    for c in courts:
        if c.mode != "singles":
            continue
        match = frozenset(c.pairs[0])
        if match in intra_opponents:
            score += OPPONENT_REPEAT_PENALTY
        if match in prev_court_pairs:
            score += SAME_COURT_SUCCESSIVE_PENALTY
    return score


def _explain_score(
    courts: list[Court],
    weekly_pair_penalties: dict[frozenset, int],
    intra_partners: set[frozenset],
    intra_opponents: set[frozenset],
    prev_court_pairs: set[frozenset],
    ratings: dict[str, int],
    genders: dict[str, str],
    prior_4f_count: int,
) -> dict[str, int]:
    """Break the score for ``courts`` into per-rule contributions.

    Returns a dict of ``{rule_key: total_contribution}`` for non-zero
    rules only. Mirrors the logic of the scoring functions exactly so
    the values sum back to the layout's total score.
    """
    out: dict[str, int] = {}

    def add(key: str, value: int) -> None:
        if value > 0:
            out[key] = out.get(key, 0) + value

    for c in courts:
        if c.mode == "doubles":
            pa, pb = c.pairs[0], c.pairs[1]
            for pair in (pa, pb):
                fs = frozenset(pair)
                if fs in intra_partners:
                    add("intra_partner", INTRA_EVENING_PENALTY)
                add("weekly_history", weekly_pair_penalties.get(fs, 0))
            for op in _doubles_opponent_pairs(pa, pb):
                if op in intra_opponents:
                    add("opponent_repeat", OPPONENT_REPEAT_PENALTY)
            for cp in _court_pair_combinations(c.players):
                if cp in prev_court_pairs:
                    add("same_court_successive", SAME_COURT_SUCCESSIVE_PENALTY)
            imbalance = abs(
                _pair_rating_sum(pa, ratings) - _pair_rating_sum(pb, ratings)
            )
            add("imbalance", PAIR_IMBALANCE_WEIGHT * imbalance)
            g = [genders.get(p, "?") for p in c.players]
            f_count = g.count("F")
            m_count = g.count("M")
            if f_count == 3 and m_count == 1:
                add("gender_hard_3F1M", GENDER_HARD_PENALTY)
            if f_count == 2 and m_count == 2:
                gen_a = sorted(genders.get(p, "?") for p in pa)
                gen_b = sorted(genders.get(p, "?") for p in pb)
                if {tuple(gen_a), tuple(gen_b)} == {("F", "F"), ("M", "M")}:
                    add("gender_hard_MM_vs_FF", GENDER_HARD_PENALTY)
            if m_count == 3 and f_count == 1:
                add("isolated_woman_3M1F", ISOLATED_WOMAN_PENALTY)
        elif c.mode == "singles":
            match = frozenset(c.pairs[0])
            if match in intra_opponents:
                add("opponent_repeat", OPPONENT_REPEAT_PENALTY)
            if match in prev_court_pairs:
                add("same_court_successive", SAME_COURT_SUCCESSIVE_PENALTY)

    this_4f = sum(
        1 for c in courts
        if c.mode == "doubles"
        and sum(1 for p in c.players if genders.get(p) == "F") == 4
    )
    excess_4f = max(0, prior_4f_count + this_4f - 1)
    if excess_4f > 0:
        add("excess_4f_court", excess_4f * EXCESS_4F_COURT_PENALTY)
    return out


def _try_layout(
    doubles_players: list[str],
    singles_players: list[str],
    doubles_labels: list[str],
    singles_labels: list[str],
    weekly_pair_penalties: dict[frozenset, int],
    intra_partners: set[frozenset],
    intra_opponents: set[frozenset],
    prev_court_pairs: set[frozenset],
    ratings: dict[str, int],
    genders: dict[str, str],
    rng: random.Random,
    forced_singles_pair: tuple[str, str] | None = None,
    forced_singles_label: str | None = None,
    prior_4f_count: int = 0,
) -> tuple[list[Court], int]:
    """Build one random layout and return (courts, score).

    ``forced_singles_pair`` (and optional ``forced_singles_label``) pin a
    specific pair to a singles court. Both names must already appear in
    ``singles_players``. If no label is given, the pair lands on the
    first singles court (``singles_labels[0]``).
    """
    shuffled_d = doubles_players[:]
    rng.shuffle(shuffled_d)
    shuffled_s = singles_players[:]
    rng.shuffle(shuffled_s)
    if forced_singles_pair is not None and singles_labels:
        forced_idx = (
            singles_labels.index(forced_singles_label)
            if forced_singles_label is not None
            else 0
        )
        forced_list = list(forced_singles_pair)
        others = [p for p in shuffled_s if p not in set(forced_list)]
        rebuilt: list[str] = []
        for i in range(len(singles_labels)):
            if i == forced_idx:
                rebuilt.extend(forced_list)
            else:
                rebuilt.extend(others[:2])
                others = others[2:]
        shuffled_s = rebuilt

    courts: list[Court] = []
    for i, label in enumerate(doubles_labels):
        four = shuffled_d[i * 4 : (i + 1) * 4]
        courts.append(
            _build_best_doubles_court(
                four, label, weekly_pair_penalties, intra_partners,
                intra_opponents, prev_court_pairs, ratings, genders,
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
        courts, weekly_pair_penalties, intra_partners, intra_opponents,
        prev_court_pairs, ratings, genders,
    ) + _score_singles_courts(courts, intra_opponents, prev_court_pairs)
    # "At most 1 4F court per evening" — penalise each 4F court beyond
    # the first across the whole evening (this rotation + earlier ones).
    this_4f = sum(
        1 for c in courts
        if c.mode == "doubles"
        and sum(1 for p in c.players if genders.get(p) == "F") == 4
    )
    excess_4f = max(0, prior_4f_count + this_4f - 1)
    score += excess_4f * EXCESS_4F_COURT_PENALTY
    return courts, score


_SINGLES_PREF_RANK = {"prefer": 0, "": 1, "avoid": 2}
# Hard cap on singles-court appearances per evening. Sorting puts anyone
# already at the cap last, so the algorithm only repeats a singles player
# when there aren't enough fresh candidates to fill the slots.
MAX_SINGLES_PER_EVENING = 1


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
      1. ``cap_reached`` — players already at MAX_SINGLES_PER_EVENING
         are deprioritised so each player gets at most one singles slot
         per evening; only repeated when there aren't enough fresh
         candidates.
      2. ``singles_prefs`` — ``"prefer"`` first, ``"avoid"`` last.
      3. ``rating`` — lower is stronger; unknown ratings → ``UNKNOWN_RATING``.
      4. ``singles_count`` so far — gentle rotation among ties.
      5. Random tie-break.
    """
    if num_singles_slots <= 0:
        return []
    keyed = sorted(
        active,
        key=lambda p: (
            singles_count.get(p, 0) >= MAX_SINGLES_PER_EVENING,
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
    weekly_pair_penalties: dict[frozenset, int],
    pinned_per_rotation: dict[int, dict],
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
    intra_partners: set[frozenset] = set()
    intra_opponents: set[frozenset] = set()
    prev_court_pairs: set[frozenset] = set()
    four_f_court_count = 0  # cumulative across rotations
    rotations: list[tuple[list[Court], list[str]]] = []

    for rot_idx in range(num_rotations):
        rotation_num = rot_idx + 1
        pin = pinned_per_rotation.get(rotation_num)
        # Fair sit-out selection — pinned singles players must NOT sit out.
        if sit_outs_per_rotation:
            forced_in = set(pin["players"]) if pin else set()
            sittable = [p for p in attendees if p not in forced_in]
            ranked = sorted(
                sittable, key=lambda p: (sitout_count[p], rng.random())
            )
            sit_outs = ranked[:sit_outs_per_rotation]
            for s in sit_outs:
                sitout_count[s] += 1
        else:
            sit_outs = []
        active = [p for p in attendees if p not in sit_outs]

        # Pick singles-destined players this rotation (with matchup-rotation
        # bias via singles_count). Honour any pin first.
        singles_slots = 2 * num_singles_courts
        forced_pair: tuple[str, str] | None = None
        forced_label: str | None = None
        if pin is not None:
            if singles_slots < 2:
                raise ValueError(
                    f"pinned_singles for rotation {rotation_num}: this "
                    "rotation has no singles court"
                )
            forced_pair = pin["players"]
            forced_label = pin["court_label"]
            forced_set = set(forced_pair)
            remaining_active = [p for p in active if p not in forced_set]
            remaining_singles = _select_singles_players(
                remaining_active, singles_slots - 2,
                ratings, singles_prefs, singles_count, rng,
            )
            singles_players = list(forced_pair) + remaining_singles
        else:
            singles_players = _select_singles_players(
                active, singles_slots, ratings, singles_prefs, singles_count, rng
            )
        doubles_players = [p for p in active if p not in singles_players]

        # Rejection-sample layouts
        best_courts: list[Court] | None = None
        best_score: int | None = None
        attempts_made = 0
        for _attempt in range(MAX_ATTEMPTS):
            attempts_made += 1
            courts, score = _try_layout(
                doubles_players,
                singles_players,
                doubles_labels,
                singles_labels,
                weekly_pair_penalties,
                intra_partners,
                intra_opponents,
                prev_court_pairs,
                ratings,
                genders,
                rng,
                forced_singles_pair=forced_pair,
                forced_singles_label=forced_label,
                prior_4f_count=four_f_court_count,
            )
            if best_score is None or score < best_score:
                best_courts = courts
                best_score = score
                if score == 0:
                    break
        assert best_courts is not None

        # Score breakdown — must happen BEFORE the tracking sets are
        # updated, otherwise the layout's own pairs get flagged as
        # repeats and the breakdown stops summing back to best_score.
        breakdown = (
            _explain_score(
                best_courts, weekly_pair_penalties, intra_partners,
                intra_opponents, prev_court_pairs, ratings, genders,
                four_f_court_count,
            )
            if best_score
            else {}
        )
        rotations.append((
            best_courts,
            sit_outs,
            {
                "attempts_made": attempts_made,
                "best_score": best_score,
                "breakdown": breakdown,
            },
        ))

        # Update tracking. intra_partners holds doubles partner pairs;
        # intra_opponents holds doubles cross-pair AND singles matchup
        # pairs (anyone who's faced each other tonight). prev_court_pairs
        # is replaced (not unioned) with this rotation's same-court pairs
        # so it only ever holds the immediately previous rotation.
        for s in singles_players:
            singles_count[s] += 1
        new_court_pairs: set[frozenset] = set()
        for c_ in best_courts:
            if c_.mode == "doubles":
                pa, pb = c_.pairs
                intra_partners.add(frozenset(pa))
                intra_partners.add(frozenset(pb))
                for op in _doubles_opponent_pairs(pa, pb):
                    intra_opponents.add(op)
            elif c_.mode == "singles":
                intra_opponents.add(frozenset(c_.pairs[0]))
            for cp in _court_pair_combinations(c_.players):
                new_court_pairs.add(cp)
            if (
                c_.mode == "doubles"
                and sum(1 for p in c_.players if genders.get(p) == "F") == 4
            ):
                four_f_court_count += 1
        prev_court_pairs = new_court_pairs

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
    pinned_singles: list[dict] | None = None,
) -> PairingPlan:
    """Build a pairing plan spanning ``num_rotations`` blocks of play.

    Supply either ``num_courts`` (labels default to ``"1", "2", …``) or
    explicit ``court_labels`` (e.g. ``[4, 5, 6, 7]`` — stringified).
    """
    _t_start = time.perf_counter()
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
    weekly_pair_penalties = recent_pair_weights(history)
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
    pinned_per_rotation = _validate_pinned_singles(
        pinned_singles,
        attendees_set=set(attendees),
        num_rotations=num_rotations,
        court_labels_set=set(labels_list),
    )
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
            metrics={
                "total_seconds": round(time.perf_counter() - _t_start, 4),
                "max_attempts_cap": MAX_ATTEMPTS,
                "rotations": [],
            },
        )

    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}; available: {sorted(STRATEGIES)}"
        )

    rng = random.Random(seed)
    per_rotation = STRATEGIES[strategy](
        attendees, labels_list, num_rotations,
        ratings, genders, singles_prefs, weekly_pair_penalties,
        pinned_per_rotation, rng,
    )

    rotation_metrics: list[dict] = []
    rotations = []
    for i, (courts, sit_outs, rot_metrics) in enumerate(per_rotation):
        rotations.append(
            Rotation(
                rotation_num=i + 1,
                start_time=starts[i],
                end_time=ends[i],
                courts=courts,
                sit_outs=sit_outs,
            )
        )
        rotation_metrics.append({
            "rotation_num": i + 1,
            "attempts_made": rot_metrics.get("attempts_made"),
            "best_score": rot_metrics.get("best_score"),
            "breakdown": rot_metrics.get("breakdown") or {},
        })

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

    total_seconds = round(time.perf_counter() - _t_start, 4)
    metrics = {
        "total_seconds": total_seconds,
        "max_attempts_cap": MAX_ATTEMPTS,
        "rotations": rotation_metrics,
    }
    # Multi-line summary so the admin_bot stdout log captures algo cost
    # AND the rule contributions to any non-zero score.
    print(
        f"[pairings] {len(attendees)} attendees, {num_rotations} rotations, "
        f"{total_seconds:.3f}s, cap={MAX_ATTEMPTS}"
    )
    for r in rotation_metrics:
        line = (
            f"  R{r['rotation_num']}={r['best_score']}/{r['attempts_made']}"
        )
        if r["best_score"] and r["breakdown"]:
            parts = ", ".join(
                f"{k}:{v}" for k, v in sorted(
                    r["breakdown"].items(), key=lambda kv: -kv[1]
                )
            )
            line += f"  [{parts}]"
        print(line)

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
        metrics=metrics,
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


def _recompute_bracket_values(court: dict, ratings: dict) -> None:
    """Refresh ``court['bracket_values']`` after the players/pairs change."""
    def _r(name: str) -> int:
        v = ratings.get(name, UNKNOWN_RATING)
        return v if isinstance(v, int) else UNKNOWN_RATING

    if court.get("mode") == "doubles":
        pa, pb = court["pairs"]
        court["bracket_values"] = [_r(pa[0]) + _r(pa[1]), _r(pb[0]) + _r(pb[1])]
    elif court.get("mode") == "singles":
        court["bracket_values"] = [_r(court["players"][0]), _r(court["players"][1])]


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
        ratings = plan.get("ratings", {})
        for c in rot["courts"]:
            c["players"] = [replace.get(p, p) for p in c["players"]]
            c["pairs"] = [
                [replace.get(p, p) for p in pair] for pair in c["pairs"]
            ]
            _recompute_bracket_values(c, ratings)
        rot["sit_outs"] = [
            replace.get(p, p) for p in rot.get("sit_outs", [])
        ]
        swapped.append(idx + 1)
    if not swapped:
        raise KeyError(
            f"no rotation contained both {name1!r} and {name2!r}"
        )
    return swapped


def swap_courts_in_plan(plan: dict, label_a: str, label_b: str) -> None:
    """Swap the contents of two courts across every rotation.

    The court labels stay put; their `mode` / `players` / `pairs` payloads
    are exchanged. So `swap_courts_in_plan(plan, "5", "11")` makes the
    matchups that were scheduled on Ct 11 (e.g. the singles) play on
    Ct 5 instead, and vice versa — without regenerating the plan or
    disturbing pinned matchups elsewhere.
    """
    label_a = str(label_a)
    label_b = str(label_b)
    if label_a == label_b:
        return
    rots = plan.get("rotations", [])
    for rot in rots:
        court_a = next(
            (c for c in rot["courts"] if c["court_label"] == label_a), None
        )
        court_b = next(
            (c for c in rot["courts"] if c["court_label"] == label_b), None
        )
        if court_a is None or court_b is None:
            raise ValueError(
                f"court labels {label_a!r} / {label_b!r} not both present "
                f"in rotation {rot.get('rotation_num')}"
            )
        for key in ("mode", "players", "pairs", "bracket_values"):
            if key in court_a or key in court_b:
                court_a[key], court_b[key] = (
                    court_b.get(key),
                    court_a.get(key),
                )


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
