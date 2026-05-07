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
  * ``+GENDER_HARD_PENALTY`` per doubles court that pairs MM-vs-FF on
    a 2M+2F court (mixed-doubles MF-vs-MF is fine). Hard rule.
  * ``+GENDER_3F1M_PENALTY`` per doubles court that is 3F+1M.
    Discouraged but not forbidden. 3M+1F is allowed and not penalised.
  * ``+RATING_1_5_PENALTY`` per court (doubles or singles) that mixes
    a rating-1 player with a rating-5 player. Effectively a hard rule
    — the algorithm only accepts it when no alternative exists.
  * Per-court rating spread (max rating gap among the 4 players, ``?`` → 3):
      * gap ≤ 1 → balanced, no penalty;
      * gap == 2 → "unbalanced", contributes to per-player count;
      * gap ≥ 3 → "very unbalanced", adds
        ``VERY_UNBALANCED_ROTATION_PENALTY`` per court AND contributes
        to per-player count.
    Per-player count of unbalanced rotations (across the WHOLE evening)
    accrues penalties: the 2nd unbalanced rotation for a player adds
    ``UNBALANCED_PLAYER_PENALTY_AT_2``; the 3rd (or any further) adds
    ``UNBALANCED_PLAYER_PENALTY_AT_3``. The algorithm naturally
    distributes unbalanced courts so each player ends the evening with
    at most one.

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
MAX_ATTEMPTS = 1000           # rejection-sampling cap per rotation
# Per-evening, run the greedy algorithm with N different seeds and keep
# the plan with the lowest total score. Diversifies the path through
# the rotation-cascade tree — a better R1 can paint R3 into a corner
# under greedy per-rotation scoring, so trying different starting RNGs
# usually beats throwing more attempts at the same path.
DEFAULT_SEED_ATTEMPTS = 3
# Extended-search policy: if the best total across the initial seed
# attempts still exceeds HIGH_SCORE_TRIGGER (i.e. a 500-pt opponent
# repeat or worse), keep trying more seeds — up to MAX_SEED_ATTEMPTS in
# total — and stop early if any attempt drops to TARGET_SCORE or below.
HIGH_SCORE_TRIGGER = 500
TARGET_SCORE = 100
MAX_SEED_ATTEMPTS = 10
# Penalty rules treated as "hard" when reporting why an extended search
# couldn't reach a clean score — surfaced to the admin in metrics so
# the bot can flag the unavoidable constraint in its WhatsApp reply.
HARD_RULE_KEYS: set[str] = {
    "opponent_repeat",
    "gender_hard_MM_vs_FF",
    "rating_1_5_same_court",
}
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
# Hard rule: a 2M+2F court paired as MM-vs-FF (segregated). Mixed-
# doubles MF-vs-MF is fine. 3M+1F is allowed and not penalised.
GENDER_HARD_PENALTY = 1000
# Soft preference: a 3F+1M court (one woman with three men). Used to
# be a hard rule; now low/medium so it can be overridden when nothing
# better is available.
GENDER_3F1M_PENALTY = 50
# Hard rule: a rating-1 player and a rating-5 player on the same
# court (doubles or singles). Treated as effectively forbidden —
# accepted only when no alternative layout exists.
RATING_1_5_PENALTY = 500
# Per-court rating-spread penalties.
# A doubles court whose max rating gap is >= 3 ("very unbalanced") gets
# a small per-court penalty. Tuned to be lower than INTRA_EVENING_PENALTY
# so a partner repeat still dominates, but high enough to discourage
# extreme mismatches when alternatives exist.
RATING_DIFF_UNBALANCED = 2
RATING_DIFF_VERY_UNBALANCED = 3
VERY_UNBALANCED_ROTATION_PENALTY = 5
# Per-evening per-player penalties for accumulating unbalanced rotations
# (max gap >= 2). Triggered when a candidate court would push a player
# from 1 → 2 (medium) or from 2 → 3+ (high) unbalanced rotations across
# the evening. Encourages spreading unbalanced courts across players.
UNBALANCED_PLAYER_PENALTY_AT_2 = 30
UNBALANCED_PLAYER_PENALTY_AT_3 = 500


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
    # Captured from the original make_plan call so post-hoc transforms
    # (notably polish_plan) can re-score without re-loading the roster.
    genders: dict[str, str] = field(default_factory=dict)
    weekly_pair_penalties: dict[frozenset, int] = field(default_factory=dict)
    notes: str = ""
    # Diagnostics: total wall-clock seconds for the make_plan call, and
    # one entry per rotation with {attempts_made, best_score}. Useful
    # for tuning MAX_ATTEMPTS and weights, and for reporting back to the
    # admin how hard the algorithm had to work.
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # weekly_pair_penalties is a dict keyed by frozenset(pair) — fine
        # in-memory but JSON keys must be strings. Strip it from the
        # serialised view; it's plumbing for in-process polish/scoring,
        # not something downstream consumers (session_state.json, the
        # WhatsApp tool result, the Sheet log) need to see.
        d.pop("weekly_pair_penalties", None)
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

      1. 3F+1M (one man with three women) → ``GENDER_3F1M_PENALTY``.
         Soft/medium — discouraged but not forbidden. (3M+1F is fine
         and is NOT penalised.)
      2. A 2M+2F court paired as MM-vs-FF is forbidden (mixed pairings
         within the same 2+2 court are fine) → ``GENDER_HARD_PENALTY``.
    Singles courts have no gender penalty.
    """
    if c.mode != "doubles":
        return 0
    g = [genders.get(p, "?") for p in c.players]
    f_count = g.count("F")
    m_count = g.count("M")
    penalty = 0
    if f_count == 3 and m_count == 1:
        penalty += GENDER_3F1M_PENALTY
    if f_count == 2 and m_count == 2:
        pair_a, pair_b = c.pairs
        gen_a = sorted(genders.get(p, "?") for p in pair_a)
        gen_b = sorted(genders.get(p, "?") for p in pair_b)
        if {tuple(gen_a), tuple(gen_b)} == {("F", "F"), ("M", "M")}:
            penalty += GENDER_HARD_PENALTY
    return penalty


def _has_rating_1_5_mix(c: Court, ratings: dict[str, int]) -> bool:
    """True if the court mixes a rating-1 player with a rating-5 player.

    Applies to both doubles (4 players) and singles (2 players). The
    mix is treated as an effective hard rule via ``RATING_1_5_PENALTY``.
    Unknown ratings (``?`` → ``UNKNOWN_RATING`` = 3) never trigger.
    """
    rs = [ratings.get(p, UNKNOWN_RATING) for p in c.players]
    return 1 in rs and 5 in rs


def _court_max_rating_diff(c: Court, ratings: dict[str, int]) -> int:
    """Max rating gap among players on a doubles court (``?`` → 3).

    Returns 0 for non-doubles courts (singles have only 2 players;
    rating-spread doesn't apply for our purposes).
    """
    if c.mode != "doubles":
        return 0
    rs = [ratings.get(p, UNKNOWN_RATING) for p in c.players]
    return max(rs) - min(rs)


def _classify_balance(diff: int) -> str:
    """Bucket a max-rating-diff into balanced / unbalanced / very_unbalanced."""
    if diff < RATING_DIFF_UNBALANCED:
        return "balanced"
    if diff < RATING_DIFF_VERY_UNBALANCED:
        return "unbalanced"
    return "very_unbalanced"


def _player_unbalanced_increment(
    new_count: int,
) -> int:
    """Penalty added when a player would transition INTO ``new_count``
    unbalanced rotations after this rotation. ``new_count`` of 1 is
    free (everyone gets one); 2 is medium; 3+ is high."""
    if new_count <= 1:
        return 0
    if new_count == 2:
        return UNBALANCED_PLAYER_PENALTY_AT_2
    return UNBALANCED_PLAYER_PENALTY_AT_3


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
    unbalanced_count: dict[str, int] | None = None,
) -> int:
    """Score a single doubles court — repeats + imbalance + gender + spread."""
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
    if _has_rating_1_5_mix(court, ratings):
        score += RATING_1_5_PENALTY
    # Per-court rating spread + per-player accumulated unbalanced count.
    diff = _court_max_rating_diff(court, ratings)
    kind = _classify_balance(diff)
    if kind == "very_unbalanced":
        score += VERY_UNBALANCED_ROTATION_PENALTY
    if kind != "balanced" and unbalanced_count is not None:
        for p in court.players:
            new_count = unbalanced_count.get(p, 0) + 1
            score += _player_unbalanced_increment(new_count)
    return score


def _score_doubles_courts(
    courts: list[Court],
    weekly_pair_penalties: dict[frozenset, int],
    intra_partners: set[frozenset],
    intra_opponents: set[frozenset],
    prev_court_pairs: set[frozenset],
    ratings: dict[str, int],
    genders: dict[str, str],
    unbalanced_count: dict[str, int] | None = None,
) -> int:
    return sum(
        _score_doubles_court(
            c, weekly_pair_penalties, intra_partners, intra_opponents,
            prev_court_pairs, ratings, genders, unbalanced_count,
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
    unbalanced_count: dict[str, int] | None = None,
) -> Court:
    """Return the lowest-scoring Court for these 4 players (mode=doubles).

    Tries all 3 ways to split four players into two pairs and picks the
    one minimising the per-court score (imbalance + repeat penalties +
    gender + spread rules). This is a local optimisation over the random
    shuffle: once player-to-court assignment is decided, picking the
    best pair structure is free — same 4 players, no side effects
    elsewhere. ``unbalanced_count`` is a no-op for the per-pair pick
    (the four-player set is fixed), but threading it keeps the score
    consistent with the post-pick layout score.
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
            prev_court_pairs, ratings, genders, unbalanced_count,
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
    ratings: dict[str, int] | None = None,
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
        if ratings is not None and _has_rating_1_5_mix(c, ratings):
            score += RATING_1_5_PENALTY
    return score


def _explain_score_items(
    courts: list[Court],
    weekly_pair_penalties: dict[frozenset, int],
    intra_partners: set[frozenset],
    intra_opponents: set[frozenset],
    prev_court_pairs: set[frozenset],
    ratings: dict[str, int],
    genders: dict[str, str],
    unbalanced_count: dict[str, int] | None = None,
) -> list[dict]:
    """Break the score for ``courts`` into a list of attributed items.

    Each item is a dict with ``rule`` (str), ``points`` (int > 0), and
    optional attribution: ``court`` (court_label), ``pair`` ([a, b])
    and/or ``player`` (name). Items mirror the logic of the scoring
    functions exactly so the points sum back to the layout's total.
    """
    items: list[dict] = []

    def emit(rule: str, points: int, **attrs) -> None:
        if points > 0:
            items.append({"rule": rule, "points": int(points), **attrs})

    for c in courts:
        if c.mode == "doubles":
            pa, pb = c.pairs[0], c.pairs[1]
            for pair in (pa, pb):
                fs = frozenset(pair)
                if fs in intra_partners:
                    emit(
                        "intra_partner", INTRA_EVENING_PENALTY,
                        court=c.court_label, pair=list(pair),
                    )
                emit(
                    "weekly_history", weekly_pair_penalties.get(fs, 0),
                    court=c.court_label, pair=list(pair),
                )
            for op in _doubles_opponent_pairs(pa, pb):
                if op in intra_opponents:
                    emit(
                        "opponent_repeat", OPPONENT_REPEAT_PENALTY,
                        court=c.court_label, pair=sorted(op),
                    )
            for cp in _court_pair_combinations(c.players):
                if cp in prev_court_pairs:
                    emit(
                        "same_court_successive", SAME_COURT_SUCCESSIVE_PENALTY,
                        court=c.court_label, pair=sorted(cp),
                    )
            imbalance = abs(
                _pair_rating_sum(pa, ratings) - _pair_rating_sum(pb, ratings)
            )
            emit(
                "imbalance", PAIR_IMBALANCE_WEIGHT * imbalance,
                court=c.court_label, magnitude=imbalance,
            )
            g = [genders.get(p, "?") for p in c.players]
            f_count = g.count("F")
            m_count = g.count("M")
            if f_count == 3 and m_count == 1:
                emit("gender_3F1M", GENDER_3F1M_PENALTY, court=c.court_label)
            if f_count == 2 and m_count == 2:
                gen_a = sorted(genders.get(p, "?") for p in pa)
                gen_b = sorted(genders.get(p, "?") for p in pb)
                if {tuple(gen_a), tuple(gen_b)} == {("F", "F"), ("M", "M")}:
                    emit(
                        "gender_hard_MM_vs_FF", GENDER_HARD_PENALTY,
                        court=c.court_label,
                    )
            if _has_rating_1_5_mix(c, ratings):
                emit(
                    "rating_1_5_same_court", RATING_1_5_PENALTY,
                    court=c.court_label,
                )
            diff = _court_max_rating_diff(c, ratings)
            kind = _classify_balance(diff)
            if kind == "very_unbalanced":
                emit(
                    "very_unbalanced_court", VERY_UNBALANCED_ROTATION_PENALTY,
                    court=c.court_label,
                )
            if kind != "balanced" and unbalanced_count is not None:
                for p in c.players:
                    new_count = unbalanced_count.get(p, 0) + 1
                    inc = _player_unbalanced_increment(new_count)
                    if inc and new_count == 2:
                        emit(
                            "unbalanced_player_2", inc,
                            court=c.court_label, player=p,
                        )
                    elif inc:
                        emit(
                            "unbalanced_player_3plus", inc,
                            court=c.court_label, player=p,
                        )
        elif c.mode == "singles":
            match = frozenset(c.pairs[0])
            if match in intra_opponents:
                emit(
                    "opponent_repeat", OPPONENT_REPEAT_PENALTY,
                    court=c.court_label, pair=sorted(match),
                )
            if match in prev_court_pairs:
                emit(
                    "same_court_successive", SAME_COURT_SUCCESSIVE_PENALTY,
                    court=c.court_label, pair=sorted(match),
                )
            if _has_rating_1_5_mix(c, ratings):
                emit(
                    "rating_1_5_same_court", RATING_1_5_PENALTY,
                    court=c.court_label,
                )
    return items


def _aggregate_breakdown(items: list[dict]) -> dict[str, int]:
    """Sum item points by rule key — the legacy shape used internally."""
    out: dict[str, int] = {}
    for it in items:
        out[it["rule"]] = out.get(it["rule"], 0) + int(it["points"])
    return out


def _explain_score(
    courts: list[Court],
    weekly_pair_penalties: dict[frozenset, int],
    intra_partners: set[frozenset],
    intra_opponents: set[frozenset],
    prev_court_pairs: set[frozenset],
    ratings: dict[str, int],
    genders: dict[str, str],
    unbalanced_count: dict[str, int] | None = None,
) -> dict[str, int]:
    """Back-compat wrapper: aggregate ``_explain_score_items`` to a dict."""
    return _aggregate_breakdown(
        _explain_score_items(
            courts, weekly_pair_penalties, intra_partners,
            intra_opponents, prev_court_pairs, ratings, genders,
            unbalanced_count,
        )
    )


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
    unbalanced_count: dict[str, int] | None = None,
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
                unbalanced_count,
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
        prev_court_pairs, ratings, genders, unbalanced_count,
    ) + _score_singles_courts(
        courts, intra_opponents, prev_court_pairs, ratings,
    )
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
    unbalanced_count: dict[str, int] = {p: 0 for p in attendees}
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
                unbalanced_count=unbalanced_count,
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
        breakdown_items = (
            _explain_score_items(
                best_courts, weekly_pair_penalties, intra_partners,
                intra_opponents, prev_court_pairs, ratings, genders,
                unbalanced_count,
            )
            if best_score
            else []
        )
        breakdown = _aggregate_breakdown(breakdown_items)
        rotations.append((
            best_courts,
            sit_outs,
            {
                "attempts_made": attempts_made,
                "best_score": best_score,
                "breakdown": breakdown,
                "breakdown_items": breakdown_items,
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
                if _classify_balance(_court_max_rating_diff(c_, ratings)) != "balanced":
                    for p in c_.players:
                        unbalanced_count[p] = unbalanced_count.get(p, 0) + 1
            elif c_.mode == "singles":
                intra_opponents.add(frozenset(c_.pairs[0]))
            for cp in _court_pair_combinations(c_.players):
                new_court_pairs.add(cp)
        prev_court_pairs = new_court_pairs

    return rotations


STRATEGIES: dict[str, StrategyFn] = {
    "skill_balanced": skill_balanced_multi_rotation,
    "random": skill_balanced_multi_rotation,  # alias — old name kept
}


# ---------- polish: hill-climb refinement on a complete plan ------------


# Polish defaults — picked for a "slow but better" tradeoff. The user
# explicitly opted in to longer wall time for tighter scores.
POLISH_MAX_ITERATIONS = 4000
POLISH_MAX_NO_IMPROVEMENT = 800
POLISH_MIN_BASELINE = 1  # don't bother if baseline is already 0.


def _rescore_layout(
    layout: list[list[list[str]]],
    *,
    rotation_modes: list[list[str]],
    rotation_labels: list[list[str]],
    rotation_sit_outs: list[list[str]],
    weekly_pair_penalties: dict[frozenset, int],
    ratings: dict[str, int],
    genders: dict[str, str],
) -> tuple[int, list[dict], list[Rotation]]:
    """Replay a plan from scratch given the player assignments.

    ``layout[i][j]`` is the list of player names on rotation i, court j
    (in any order — the function picks the best pair split for doubles
    courts internally). The function returns
    ``(total_score, per_rotation_metrics, rebuilt_rotations)``.
    """
    intra_partners: set[frozenset] = set()
    intra_opponents: set[frozenset] = set()
    prev_court_pairs: set[frozenset] = set()
    unbalanced_count: dict[str, int] = {}
    per_rotation: list[dict] = []
    rebuilt: list[Rotation] = []
    total = 0

    for rot_idx, courts_players in enumerate(layout):
        modes = rotation_modes[rot_idx]
        labels = rotation_labels[rot_idx]
        sit_outs = rotation_sit_outs[rot_idx]
        new_courts: list[Court] = []
        for ci, players in enumerate(courts_players):
            label = labels[ci]
            mode = modes[ci]
            if mode == "doubles":
                court = _build_best_doubles_court(
                    list(players), label, weekly_pair_penalties,
                    intra_partners, intra_opponents, prev_court_pairs,
                    ratings, genders, unbalanced_count,
                )
            else:
                # Singles: 2 players, single matchup pair.
                two = list(players)
                court = Court(
                    court_label=label,
                    mode="singles",
                    players=two,
                    pairs=[(two[0], two[1])],
                )
            new_courts.append(court)

        score = (
            _score_doubles_courts(
                new_courts, weekly_pair_penalties, intra_partners,
                intra_opponents, prev_court_pairs, ratings, genders,
                unbalanced_count,
            )
            + _score_singles_courts(
                new_courts, intra_opponents, prev_court_pairs, ratings,
            )
        )
        breakdown_items = _explain_score_items(
            new_courts, weekly_pair_penalties, intra_partners,
            intra_opponents, prev_court_pairs, ratings, genders,
            unbalanced_count,
        )
        per_rotation.append({
            "rotation_num": rot_idx + 1,
            "best_score": score,
            "breakdown": _aggregate_breakdown(breakdown_items),
            "breakdown_items": breakdown_items,
        })
        total += score

        # Update trackers for next rotation.
        new_court_pairs: set[frozenset] = set()
        for c in new_courts:
            if c.mode == "doubles":
                pa, pb = c.pairs
                intra_partners.add(frozenset(pa))
                intra_partners.add(frozenset(pb))
                for op in _doubles_opponent_pairs(pa, pb):
                    intra_opponents.add(op)
                if (
                    _classify_balance(_court_max_rating_diff(c, ratings))
                    != "balanced"
                ):
                    for p in c.players:
                        unbalanced_count[p] = unbalanced_count.get(p, 0) + 1
            else:
                intra_opponents.add(frozenset(c.pairs[0]))
            for cp in _court_pair_combinations(c.players):
                new_court_pairs.add(cp)
        prev_court_pairs = new_court_pairs

        rebuilt.append(Rotation(
            rotation_num=rot_idx + 1,
            start_time="",  # filled in by caller from original plan
            end_time="",
            courts=new_courts,
            sit_outs=list(sit_outs),
        ))

    return total, per_rotation, rebuilt


def polish_plan(
    plan: PairingPlan,
    *,
    seed: int | None = None,
    max_iterations: int = POLISH_MAX_ITERATIONS,
    max_no_improvement: int = POLISH_MAX_NO_IMPROVEMENT,
    verbose: bool = True,
) -> PairingPlan:
    """Hill-climb refinement over a complete pairing plan.

    Starting from the multi-seed greedy result, randomly swap two
    player-slots (anywhere across the evening, including across
    different rotations) and accept only score-reducing moves.
    Each move triggers a full plan rescore (best pair-split is
    re-picked per doubles court given the new player set + the
    accumulated cross-rotation trackers up to that point).

    Side-stepping the sequential greediness of make_plan: by mutating
    a complete plan instead of building it left-to-right, R3 quality
    can be improved by trading R1 / R2 quality where it helps the
    total. Returns a new ``PairingPlan`` with refreshed metrics
    (including a ``polish`` block) — does NOT mutate the input.
    """
    rng = random.Random(seed)

    # Snapshot the original rotation metadata that polish doesn't touch
    # (start/end times, modes, labels, sit-outs).
    rotation_modes = [
        [c.mode for c in rot.courts] for rot in plan.rotations
    ]
    rotation_labels = [
        [c.court_label for c in rot.courts] for rot in plan.rotations
    ]
    rotation_sit_outs = [list(rot.sit_outs) for rot in plan.rotations]
    rotation_starts = [rot.start_time for rot in plan.rotations]
    rotation_ends = [rot.end_time for rot in plan.rotations]

    # Mutable layout: rotation → court → list of player names.
    layout: list[list[list[str]]] = [
        [list(c.players) for c in rot.courts] for rot in plan.rotations
    ]

    # Prerequisites for rescoring (drawn from the input plan + roster).
    weekly_pair_penalties = plan.weekly_pair_penalties or {}
    ratings = dict(plan.ratings)
    genders = dict(plan.genders)

    baseline_total, _, _ = _rescore_layout(
        layout,
        rotation_modes=rotation_modes,
        rotation_labels=rotation_labels,
        rotation_sit_outs=rotation_sit_outs,
        weekly_pair_penalties=weekly_pair_penalties,
        ratings=ratings, genders=genders,
    )
    current_total = baseline_total

    if baseline_total < POLISH_MIN_BASELINE:
        if verbose:
            print(
                f"[polish] baseline already {baseline_total} — skipping"
            )
        return _build_polished_plan(
            plan, layout,
            rotation_modes, rotation_labels, rotation_sit_outs,
            rotation_starts, rotation_ends,
            weekly_pair_penalties, ratings, genders,
            polish_meta={
                "iterations": 0,
                "accepted": 0,
                "baseline_total": baseline_total,
                "final_total": baseline_total,
                "skipped": True,
            },
        )

    # Cache for memoisation: layout signatures → score, to short-circuit
    # repeated proposals. Keys are tuples of rotation-tuples-of-frozenset
    # of player names (order-insensitive within a court).
    cache: dict[tuple, int] = {}

    def _signature(lay: list[list[list[str]]]) -> tuple:
        return tuple(
            tuple(frozenset(c) for c in rot) for rot in lay
        )

    cache[_signature(layout)] = current_total
    iterations = 0
    accepted = 0
    no_improvement = 0
    t_start = time.perf_counter()

    while iterations < max_iterations and no_improvement < max_no_improvement:
        iterations += 1
        # Pick two random positions to swap.
        rot_a = rng.randint(0, len(layout) - 1)
        rot_b = rng.randint(0, len(layout) - 1)
        court_a = rng.randint(0, len(layout[rot_a]) - 1)
        court_b = rng.randint(0, len(layout[rot_b]) - 1)
        # Reject same-court (no-op).
        if rot_a == rot_b and court_a == court_b:
            no_improvement += 1
            continue
        # Reject swaps that would move a player between modes
        # (doubles ↔ singles). The original singles-selection logic
        # caps singles appearances per evening and respects user
        # preferences ('prefer'/'avoid'); polish must not undo those.
        if rotation_modes[rot_a][court_a] != rotation_modes[rot_b][court_b]:
            no_improvement += 1
            continue
        slot_a = rng.randint(0, len(layout[rot_a][court_a]) - 1)
        slot_b = rng.randint(0, len(layout[rot_b][court_b]) - 1)
        p_a = layout[rot_a][court_a][slot_a]
        p_b = layout[rot_b][court_b][slot_b]
        if p_a == p_b:
            no_improvement += 1
            continue
        # For cross-rotation swaps, also reject if the swap would
        # duplicate a player WITHIN a rotation, OR move a player out
        # of/into a sit-out slot. Same-rotation swaps are safe (the
        # players are just changing courts).
        if rot_a != rot_b:
            if any(p_b in court for court in layout[rot_a]):
                no_improvement += 1
                continue
            if any(p_a in court for court in layout[rot_b]):
                no_improvement += 1
                continue
            # If p_b is currently sitting out in rotation_a (or p_a in
            # rotation_b), the swap would put them in a court while
            # they're also flagged as sit-out — broken invariant.
            if p_b in rotation_sit_outs[rot_a]:
                no_improvement += 1
                continue
            if p_a in rotation_sit_outs[rot_b]:
                no_improvement += 1
                continue
        else:
            # Same rotation, different courts — only need to check the
            # target courts (each player is in exactly one court within
            # a rotation by construction).
            if p_b in layout[rot_a][court_a]:
                no_improvement += 1
                continue
            if p_a in layout[rot_b][court_b]:
                no_improvement += 1
                continue

        # Apply.
        layout[rot_a][court_a][slot_a] = p_b
        layout[rot_b][court_b][slot_b] = p_a

        sig = _signature(layout)
        if sig in cache:
            new_total = cache[sig]
        else:
            new_total, _, _ = _rescore_layout(
                layout,
                rotation_modes=rotation_modes,
                rotation_labels=rotation_labels,
                rotation_sit_outs=rotation_sit_outs,
                weekly_pair_penalties=weekly_pair_penalties,
                ratings=ratings, genders=genders,
            )
            cache[sig] = new_total

        if new_total < current_total:
            current_total = new_total
            accepted += 1
            no_improvement = 0
        else:
            # Revert.
            layout[rot_a][court_a][slot_a] = p_a
            layout[rot_b][court_b][slot_b] = p_b
            no_improvement += 1

    if verbose:
        elapsed = time.perf_counter() - t_start
        print(
            f"[polish] {iterations} iterations, {accepted} accepted, "
            f"baseline {baseline_total} -> final {current_total} "
            f"({elapsed:.2f}s)"
        )

    return _build_polished_plan(
        plan, layout,
        rotation_modes, rotation_labels, rotation_sit_outs,
        rotation_starts, rotation_ends,
        weekly_pair_penalties, ratings, genders,
        polish_meta={
            "iterations": iterations,
            "accepted": accepted,
            "baseline_total": baseline_total,
            "final_total": current_total,
            "skipped": False,
        },
    )


def _build_polished_plan(
    original: PairingPlan,
    layout: list[list[list[str]]],
    rotation_modes: list[list[str]],
    rotation_labels: list[list[str]],
    rotation_sit_outs: list[list[str]],
    rotation_starts: list[str],
    rotation_ends: list[str],
    weekly_pair_penalties: dict[frozenset, int],
    ratings: dict[str, int],
    genders: dict[str, str],
    polish_meta: dict,
) -> PairingPlan:
    """Build a fresh PairingPlan from the polished layout."""
    total, per_rotation, rebuilt = _rescore_layout(
        layout,
        rotation_modes=rotation_modes,
        rotation_labels=rotation_labels,
        rotation_sit_outs=rotation_sit_outs,
        weekly_pair_penalties=weekly_pair_penalties,
        ratings=ratings, genders=genders,
    )
    # Restore start/end times that polish doesn't touch.
    for r, st, en in zip(rebuilt, rotation_starts, rotation_ends):
        r.start_time = st
        r.end_time = en

    # Preserve attempts_made from the original (greedy) per-rotation
    # metrics — polish doesn't redo rejection-sampling so the count is
    # identical to the multi-seed run that produced the input plan.
    original_per_rot = original.metrics.get("rotations", []) or []
    for i, rot_m in enumerate(per_rotation):
        if i < len(original_per_rot):
            rot_m["attempts_made"] = original_per_rot[i].get("attempts_made")

    new_metrics = dict(original.metrics)
    new_metrics["rotations"] = per_rotation
    new_metrics["polish"] = polish_meta
    if "multi_seed" in new_metrics:
        new_metrics["multi_seed"] = dict(new_metrics["multi_seed"])
        new_metrics["multi_seed"]["chosen_total"] = total
    # Re-run the blocking-rules detector against the polished plan.
    new_metrics["multi_seed"] = new_metrics.get("multi_seed", {})
    blocking: list[dict] = []
    for r in per_rotation:
        for rule, value in r.get("breakdown", {}).items():
            if rule in HARD_RULE_KEYS:
                blocking.append({
                    "rotation_num": r["rotation_num"],
                    "rule": rule,
                    "penalty": value,
                })
    new_metrics["multi_seed"]["blocking_rules"] = blocking

    return PairingPlan(
        date=original.date,
        attendees=list(original.attendees),
        court_labels=list(original.court_labels),
        num_rotations=original.num_rotations,
        rotations=rebuilt,
        unknown_attendees=list(original.unknown_attendees),
        display_names=dict(original.display_names),
        ratings=dict(ratings),
        strategy=original.strategy,
        genders=dict(genders),
        weekly_pair_penalties=dict(weekly_pair_penalties),
        notes=original.notes,
        metrics=new_metrics,
    )


# ---------- public API --------------------------------------------------


def _make_plan_one(
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
    verbose: bool = True,
) -> PairingPlan:
    """Single-seed pairing run — see ``make_plan`` for the public entry.

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
            genders=genders,
            weekly_pair_penalties=weekly_pair_penalties,
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
            "breakdown_items": rot_metrics.get("breakdown_items") or [],
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
    # AND the rule contributions to any non-zero score. Suppress when
    # invoked from the multi-seed wrapper, which prints its own summary.
    if verbose:
        print(
            f"[pairings] {len(attendees)} attendees, {num_rotations} rotations, "
            f"{total_seconds:.3f}s, cap={MAX_ATTEMPTS}, seed={seed}"
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
        genders=genders,
        weekly_pair_penalties=weekly_pair_penalties,
        notes=" ".join(notes_parts),
        metrics=metrics,
    )


def _plan_total(plan: PairingPlan) -> int:
    """Sum of the per-rotation best_scores from the plan's metrics."""
    return sum(
        r.get("best_score", 0) or 0 for r in plan.metrics.get("rotations", [])
    )


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
    num_seed_attempts: int = DEFAULT_SEED_ATTEMPTS,
    polish: bool = True,
) -> PairingPlan:
    """Build a pairing plan, optionally trying multiple seeds.

    The greedy per-rotation algorithm can paint itself into a corner —
    a locally-best rotation 1 may force rotation 3 into accepting a
    hard-rule violation that a different rotation 1 would have avoided.
    Running multiple independent seeds and keeping the lowest-total
    plan diversifies the path through that cascade.

    With ``num_seed_attempts == 1`` this is identical to the old
    single-run behaviour. The default (``DEFAULT_SEED_ATTEMPTS``) tries
    3 seeds. When ``seed`` is provided the candidate seeds are
    ``[seed, seed+1, seed+2, …]`` so single-seed callers stay
    deterministic; when ``seed`` is None the seeds are random.

    When ``polish=True`` (default), a hill-climb refinement runs on top
    of the multi-seed result — see ``polish_plan``. Adds about a second
    of wall time for typical 24-player / 6-court / 3-rotation evenings
    and routinely cuts hard cases by 95%+. Pass ``polish=False`` to
    skip (mostly useful in tests).
    """
    common = dict(
        attendees=list(attendees),
        players_path=players_path,
        history_path=history_path,
        num_courts=num_courts,
        court_labels=court_labels,
        num_rotations=num_rotations,
        start_time_hhmm=start_time_hhmm,
        rotation_durations=rotation_durations,
        strategy=strategy,
        today=today,
        singles_exclude=singles_exclude,
        singles_include=singles_include,
        pinned_singles=pinned_singles,
    )

    if num_seed_attempts <= 1:
        single = _make_plan_one(seed=seed, **common)
        if polish:
            single = polish_plan(single, seed=seed, verbose=True)
        return single

    # Seed plan: deterministic when ``seed`` is given, random otherwise.
    if seed is None:
        master = random.Random()
        all_seeds = [
            master.randint(0, 2**31 - 1)
            for _ in range(MAX_SEED_ATTEMPTS)
        ]
    else:
        all_seeds = [seed + i for i in range(MAX_SEED_ATTEMPTS)]

    _t0 = time.perf_counter()
    initial_n = max(1, num_seed_attempts)
    plans: list[PairingPlan] = []
    seeds_used: list[int] = []
    totals: list[int] = []
    best_total = float("inf")
    best_idx = 0

    def run_seed(s: int) -> None:
        nonlocal best_total, best_idx
        plan = _make_plan_one(seed=s, verbose=False, **common)
        plans.append(plan)
        seeds_used.append(s)
        t = _plan_total(plan)
        totals.append(t)
        if t < best_total:
            best_total = t
            best_idx = len(plans) - 1

    # 1) Initial round of attempts.
    for s in all_seeds[:initial_n]:
        run_seed(s)

    # 2) Extended search if even the best initial total still exceeds
    #    the high-score trigger (typically meaning a 500-pt opponent
    #    repeat). Keep adding seeds until we hit the target or run out.
    extended = False
    if best_total > HIGH_SCORE_TRIGGER:
        extended = True
        for s in all_seeds[initial_n:MAX_SEED_ATTEMPTS]:
            if best_total <= TARGET_SCORE:
                break
            run_seed(s)

    chosen = plans[best_idx]

    # Identify any hard-rule contributions still present in the chosen
    # plan — surfaced for the bot to mention in WhatsApp when the run
    # couldn't be coerced into a clean score.
    blocking: list[dict] = []
    for r in chosen.metrics.get("rotations", []):
        for rule, value in r.get("breakdown", {}).items():
            if rule in HARD_RULE_KEYS:
                blocking.append({
                    "rotation_num": r["rotation_num"],
                    "rule": rule,
                    "penalty": value,
                })

    # Total candidate layouts evaluated across all seeds tried (sum of
    # per-rotation attempts_made for every seed). Surfaced so the bot
    # can tell admins how much search the optimizer did, in
    # admin-friendly wording.
    total_permutations_tried = sum(
        int(r.get("attempts_made") or 0)
        for p in plans
        for r in p.metrics.get("rotations", [])
    )

    chosen.metrics["multi_seed"] = {
        "seeds_tried": list(seeds_used),
        "totals_by_seed": dict(zip(seeds_used, totals)),
        "chosen_seed": seeds_used[best_idx],
        "chosen_total": int(best_total),
        "wall_seconds": round(time.perf_counter() - _t0, 4),
        "extended_search": extended,
        "blocking_rules": blocking,
        "total_permutations_tried": total_permutations_tried,
    }

    # Headline + per-rotation breakdown for the chosen plan.
    print(
        f"[pairings] multi-seed{' (extended)' if extended else ''}: "
        f"{dict(zip(seeds_used, totals))} -> chose seed={seeds_used[best_idx]} "
        f"total={int(best_total)} "
        f"({chosen.metrics['multi_seed']['wall_seconds']:.3f}s)"
    )
    for r in chosen.metrics.get("rotations", []):
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
    if blocking:
        rule_list = ", ".join(
            f"{b['rule']}@R{b['rotation_num']}" for b in blocking
        )
        print(f"  ! blocking rules in chosen plan: {rule_list}")

    if polish:
        chosen = polish_plan(chosen, seed=seed, verbose=True)

    return chosen


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
