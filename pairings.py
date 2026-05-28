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
of the roster (lower ratings = stronger; ``?`` treated as 6 for sorting).
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
  * Per pair drawn from ``history.json``, weighted by how recently
    they last played together. ``RECENT_PAIR_WEIGHT_BANDS`` is a list
    of ``(max_days, weight)`` bands evaluated in order; a pair is
    penalised by the weight of the first band whose ``max_days``
    covers any prior shared session, accumulated across bands so a
    pair seen in BOTH the 0-7d and 8-14d windows scores 10+5=15.
    Date-based rather than entry-indexed: with three sessions a week
    (Tue/Thu/Sat) "last week" means "within the last 7 calendar
    days", not "the last entry in history.json".
  * ``+PAIR_IMBALANCE_WEIGHT × |sumA - sumB|`` per doubles court, where the
    sums are rating totals for each of the two pairs (``?`` → 6).
  * ``+GENDER_MM_VS_FF_PENALTY`` per doubles court that pairs MM-vs-FF
    on a 2M+2F court (mixed-doubles MF-vs-MF is fine). Soft.
  * ``+GENDER_3F1M_PENALTY`` per doubles court that is 3F+1M.
    Discouraged but not forbidden. 3M+1F is allowed and not penalised.
  * Rating-gap band penalty per court (doubles OR singles), on the
    court's min↔max rating gap (``?`` → 6):
      * gap 0-3 → balanced, no penalty;
      * gap 4-5 → unbalanced (base 20);
      * gap 6-7 → very unbalanced (base 50);
      * gap 8-9 → extremely unbalanced (base 100).
    The base is multiplied by ``Σ over the court's players of
    RATING_GAP_MULT ** (that player's non-balanced rotations so
    far)``. Every player's first non-balanced rotation is at base
    cost; each further one for the same player is ~3× dearer, so the
    algorithm gives players all-balanced rotations where possible,
    else at most one non-balanced and that one as mild as it can.
  * Whole-evening per-player ``standard_too_low``: a mid-rated
    player (3-8) whose BEST rotation's company is still materially
    weaker than them (never got an at-or-better-standard game) is
    penalised ``STANDARD_TOO_LOW_WEIGHT × shortfall``. Stronger
    company is free.
  * Whole-evening per-player ``top_player_no_strong_rotation``: a
    top player (rating 1-2) who never gets a rotation where every
    other player is no worse than own+1 is penalised
    ``TOP_PLAYER_STRONG_WEIGHT × (best-attempt worst-other −
    (rating+1))``. Complements the above for the elite end it
    can't cover.

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

INTRA_EVENING_PENALTY = 500   # partner pair already played tonight
# Recency bands for the "played together recently" penalty. Each band
# is ``(max_days, weight)``: a history entry is classified into the
# FIRST band whose ``max_days`` covers its age in calendar days. So
# an entry 3 days back hits the 7d band (weight 10) only; an entry
# 10 days back hits the 14d band (weight 5) only; older entries score
# 0. Pairs accumulate ACROSS entries — a pair playing together 3 days
# AND 10 days ago picks up 10+5=15. With three sessions per week we
# need date-based windows rather than the old entry-indexed weighting,
# otherwise "last week" silently means "the last 1-2 sessions" which
# under-penalises pairs that played a few days apart on the same week.
RECENT_PAIR_WEIGHT_BANDS: list[tuple[int, int]] = [(7, 10), (14, 5)]
PAIR_IMBALANCE_WEIGHT = 1     # per unit of |pairA_sum - pairB_sum| (1-10 scale)
# Quadratic escalation for pair-sum imbalance ≥ 2. Small diffs are
# routinely unavoidable; large diffs (one side much stronger than the
# other) usually mean an opponent-repeat constraint forced a poor
# split — but the rest of the time they're worth pushing the polish
# hill-climb to redistribute players across courts to fix. Penalty is
#   diff ≤ 1: PAIR_IMBALANCE_WEIGHT × diff (0 or 1)
#   diff ≥ 2: PAIR_IMBALANCE_WEIGHT + PAIR_IMBALANCE_ESCALATION × (diff − 1)²
# which gives: 0/1/6/21/46/81/126/… — strong enough at diff 4-5 to
# matter against soft rules, well below the 500 hard rules so it
# never overrides them.
PAIR_IMBALANCE_ESCALATION = 5
UNKNOWN_RATING = 6            # neutral treatment for rating == "?" (1-10 scale)
MAX_ATTEMPTS = 1000           # rejection-sampling cap per rotation
# Per-evening, run the greedy algorithm with N different seeds and keep
# the plan with the lowest total score. Diversifies the path through
# the rotation-cascade tree — a better R1 can paint R3 into a corner
# under greedy per-rotation scoring, so trying different starting RNGs
# usually beats throwing more attempts at the same path.
DEFAULT_SEED_ATTEMPTS = 5
# Extended-search policy: if the best total across the initial seed
# attempts still exceeds HIGH_SCORE_TRIGGER, keep trying more seeds —
# up to MAX_SEED_ATTEMPTS in total — and stop early if any attempt
# drops to TARGET_SCORE or below. The trigger is set deliberately low
# so we keep exploring unless the very first runs are already close
# to the floor.
HIGH_SCORE_TRIGGER = 50
TARGET_SCORE = 25
MAX_SEED_ATTEMPTS = 25
# Penalty rules treated as "hard" when reporting why an extended search
# couldn't reach a clean score — surfaced to the admin in metrics so
# the bot can flag the unavoidable constraint in its WhatsApp reply.
HARD_RULE_KEYS: set[str] = {
    "opponent_repeat",
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
# Soft preference: a 2M+2F court paired as MM-vs-FF (genders
# segregated). Mixed-doubles MF-vs-MF is fine; 3M+1F is allowed and
# not penalised. Low weight — discouraged but readily overridden.
GENDER_MM_VS_FF_PENALTY = 50
# Soft preference: a 3F+1M court (one man with three women). Used to
# be a hard rule; now low/medium so it can be overridden when nothing
# better is available.
GENDER_3F1M_PENALTY = 50

# Rating-gap bands on a court's min↔max rating spread (the difference
# between the highest- and lowest-rated player on the court). Applies
# to BOTH doubles (4 players) and singles (2 players). Bands:
#   gap 0-3 → balanced (free), 4-5 → unbalanced, 6-7 → very
#   unbalanced, 8-9 → extremely unbalanced.
# The per-court penalty is the band's base weight multiplied by the
# sum, over the court's players, of RATING_GAP_MULT ** (that player's
# count of non-balanced rotations SO FAR this evening). So a
# non-balanced court costs progressively more the more non-balanced
# rotations its players have already had — driving the algorithm to
# give every player all-balanced rotations where possible, else at
# most one non-balanced rotation, and that one as mild as possible.
# Balanced courts (base 0) always cost 0 regardless of the multiplier.
RATING_GAP_BANDS: list[tuple[int, str, int]] = [
    # (minimum gap inclusive, band name, base weight)
    (8, "extremely_unbalanced", 100),
    (6, "very_unbalanced", 50),
    (4, "unbalanced", 20),
    (0, "balanced", 0),
]
RATING_GAP_MULT = 3
# band name → base weight, derived from RATING_GAP_BANDS for lookups.
_RATING_GAP_BASE: dict[str, int] = {
    name: base for _thr, name, base in RATING_GAP_BANDS
}

# "Played down all night" rule. On the 1(strong)–10(weak) scale, a
# player rated ``r`` is playing UP in a rotation when the mean rating
# of the OTHER players on their court is below ``r`` (stronger
# company) — that's fine and never penalised. They're playing DOWN
# when that mean is above ``r`` (weaker company). We don't want a
# player playing down in every rotation with no compensating
# up/level one. So for each eligible player we take their BEST
# rotation (the one with the strongest company = lowest other-mean);
# if even that is materially weaker than them, penalise by how far
# short. Only mid-band ratings: the very strongest (≤2) structurally
# can't get an up rotation (almost everyone is weaker), and the very
# weakest (≥9) essentially never trigger. ``?`` → UNKNOWN_RATING (6),
# so unrated players fall in-band and are covered.
STANDARD_TOO_LOW_WEIGHT = 5
STANDARD_TOO_LOW_MATERIAL = 1  # only "materially" worse counts
STANDARD_RULE_RATING_MIN = 3
STANDARD_RULE_RATING_MAX = 8

# Elite-player guarantee (complements standard_too_low, which can't
# cover ratings 1-2 since their company is almost always weaker).
# A top player (rating ≤ TOP_PLAYER_MAX_RATING) should get at least
# one rotation where EVERY other player on their court is rated no
# more than their own rating + 1 (so a 1 wants an all 1-2 court; a 2
# wants an all 1-3 court). Unlike standard_too_low this keys off the
# court's WORST other player (max rating), not the mean — one weak
# link spoils a top game even if the average looks fine. Graded:
# penalty = TOP_PLAYER_STRONG_WEIGHT × (best-attempt court's
# max-other-rating − (own rating + 1)) when positive; 0 once they
# get a qualifying rotation. Playing with stronger company is free.
TOP_PLAYER_STRONG_WEIGHT = 5
TOP_PLAYER_MAX_RATING = 2

# Hard-vs-clay court preference (Westside). Courts 1-4 are hard; 5-12
# are clay. Players prefer clay, so 2+ hard-court rotations in the
# same evening earns a penalty. First hard rotation is free (someone
# has to play there when hard courts are in the pool); the penalty
# escalates so 3-out-of-3 hard is much worse than 2-out-of-3. Per
# player, weight = HARD_COURT_REPEAT_WEIGHT × Σ_{i=1..(hard_count-1)} i.
#   hard_count 0-1 → 0
#   hard_count 2   → 10 (small, "you got unlucky once")
#   hard_count 3   → 30 (no clay all night — significant)
#   hard_count 4+  → 60, 100, …  (rare, escalates).
# Attributed to the player's first hard-court rotation so the
# breakdown reconciles with the rotation totals.
HARD_COURT_REPEAT_WEIGHT = 10
HARD_COURT_NUMBERS: frozenset[int] = frozenset({1, 2, 3, 4})


# ---------- rule documentation (single source of truth) -----------------
#
# Human-readable summary of every rule the pairing algorithm applies.
# Constants are referenced directly so when a weight changes in this
# file, the published-to-members PDF auto-reflects it on next render.
# Keep titles and descriptions plain-English — the PDF goes to admins
# who aren't reading Python.

RULE_DOCS: list[dict] = [
    # Hard rules — algorithm only accepts these when no alternative
    # layout exists.
    {
        "key": "opponent_repeat",
        "category": "hard",
        "weight": OPPONENT_REPEAT_PENALTY,
        "title": "Opponent repeat in the same evening",
        "description": (
            "Any pair of players who have already faced each other "
            "tonight (across-the-net in doubles, or as singles "
            "opponents) shouldn't face each other again in a later "
            "rotation."
        ),
    },
    {
        "key": "intra_partner",
        "category": "hard",
        "weight": INTRA_EVENING_PENALTY,
        "title": "Partner repeat in the same evening",
        "description": (
            "Two players who have already partnered tonight being "
            "paired together again in a later rotation — strongly "
            "discouraged, since mixing partners across the evening is "
            "the whole point of rotating."
        ),
    },

    # Soft preferences — accumulated and balanced against each other.
    {
        "key": "rating_gap_unbalanced",
        "category": "soft",
        "weight": _RATING_GAP_BASE["unbalanced"],
        "weight_label": f"{_RATING_GAP_BASE['unbalanced']} × n",
        "title": "Unbalanced court (rating gap 4-5)",
        "description": (
            "A court (doubles or singles) whose rating gap — the "
            "difference between its strongest and weakest player — is "
            "4 or 5. Base "
            f"{_RATING_GAP_BASE['unbalanced']} points, but multiplied "
            f"by {RATING_GAP_MULT}× for each non-balanced rotation a "
            "player on the court has already had this evening (summed "
            "over the players). So everyone's first non-balanced "
            "rotation is cheap; second and third get rapidly dearer, "
            "spreading the unavoidable mismatches around."
        ),
    },
    {
        "key": "rating_gap_very_unbalanced",
        "category": "soft",
        "weight": _RATING_GAP_BASE["very_unbalanced"],
        "weight_label": f"{_RATING_GAP_BASE['very_unbalanced']} × n",
        "title": "Very unbalanced court (rating gap 6-7)",
        "description": (
            "As 'unbalanced' but for a rating gap of 6 or 7 — base "
            f"{_RATING_GAP_BASE['very_unbalanced']} points, same "
            f"{RATING_GAP_MULT}×-per-prior-non-balanced escalation."
        ),
    },
    {
        "key": "rating_gap_extremely_unbalanced",
        "category": "soft",
        "weight": _RATING_GAP_BASE["extremely_unbalanced"],
        "weight_label": f"{_RATING_GAP_BASE['extremely_unbalanced']} × n",
        "title": "Extremely unbalanced court (rating gap 8-9)",
        "description": (
            "The widest mismatch — a rating gap of 8 or 9 (e.g. a 1 "
            f"with a 9 or 10). Base "
            f"{_RATING_GAP_BASE['extremely_unbalanced']} points with "
            f"the same {RATING_GAP_MULT}×-per-prior-non-balanced "
            "escalation. Replaces the old hard 'extreme rating mix' "
            "rule — strongly discouraged but no longer an absolute "
            "veto."
        ),
    },
    {
        "key": "gender_MM_vs_FF",
        "category": "soft",
        "weight": GENDER_MM_VS_FF_PENALTY,
        "title": "Genders segregated on a doubles court (2M vs 2F)",
        "description": (
            "A doubles court paired as 2 men against 2 women. "
            "Mixed-doubles (MF vs MF) is fine, and 3M+1F is allowed. "
            "Discouraged but readily overridden when it improves the "
            "rest of the evening."
        ),
    },
    {
        "key": "gender_3F1M",
        "category": "soft",
        "weight": GENDER_3F1M_PENALTY,
        "title": "3 women + 1 man on a doubles court",
        "description": (
            "Discouraged but not forbidden — sometimes there's no "
            "alternative when the gender counts are uneven. 3M+1F "
            "is allowed and not penalised."
        ),
    },
    {
        "key": "standard_too_low",
        "category": "soft",
        "weight": STANDARD_TOO_LOW_WEIGHT,
        "weight_label": f"{STANDARD_TOO_LOW_WEIGHT} × n",
        "title": "Player kept among weaker players all evening",
        "description": (
            "For a player rated "
            f"{STANDARD_RULE_RATING_MIN}-{STANDARD_RULE_RATING_MAX} "
            "(1 = strongest), if even their BEST rotation's company "
            "(mean rating of the other players on their court) is "
            f"materially weaker than them — weaker by at least "
            f"{STANDARD_TOO_LOW_MATERIAL} rating point — they never "
            "got an at-or-better-standard game all evening. Penalty "
            f"is {STANDARD_TOO_LOW_WEIGHT} × (how much weaker the "
            "best-rotation company is than them, in rating points), "
            "rounded. Playing with stronger company is never "
            "penalised; one decent rotation clears it. The strongest "
            "(rating 1-2) and weakest (rating 9-10) players are "
            "exempt (their company is structurally always weaker / "
            "stronger, so it can't be balanced)."
        ),
    },
    {
        "key": "top_player_no_strong_rotation",
        "category": "soft",
        "weight": TOP_PLAYER_STRONG_WEIGHT,
        "weight_label": f"{TOP_PLAYER_STRONG_WEIGHT} × n",
        "title": "Top player never got a strong rotation",
        "description": (
            "Complements the rule above for the very best players "
            f"(rating ≤ {TOP_PLAYER_MAX_RATING}, 1 = strongest), who "
            "almost always have weaker company so could never satisfy "
            "the mean-based rule. Each should get at least one "
            "rotation where EVERY other player on their court is rated "
            "no worse than their own rating + 1 (a 1 wants an all 1-2 "
            "court; a 2 an all 1-3 court). Keyed off the WORST other "
            "player, not the average — one weak link spoils a top "
            "game. If even their best such rotation falls short, "
            f"penalty is {TOP_PLAYER_STRONG_WEIGHT} × (that rotation's "
            "worst other rating − (their rating + 1)), rounded. One "
            "qualifying rotation clears it; stronger company is free."
        ),
    },
    {
        "key": "hard_court_repeat",
        "category": "soft",
        "weight": HARD_COURT_REPEAT_WEIGHT,
        "weight_label": (
            f"{HARD_COURT_REPEAT_WEIGHT} × Σ(1..hard_count-1)"
        ),
        "title": "Player stuck on hard courts for multiple rotations",
        "description": (
            "Westside players prefer clay (courts "
            f"{min(HARD_COURT_NUMBERS)}-{max(HARD_COURT_NUMBERS)} are "
            "hard; the rest are clay). One hard-court rotation per "
            "player is free — that's the cost of having hard courts "
            "in the pool. Two or more hard-court rotations earn an "
            "escalating per-player penalty: "
            f"{HARD_COURT_REPEAT_WEIGHT} × Σ_(i=1..hard_count-1) i. So "
            "2 hard rotations = 10 points (a small nudge), 3 hard "
            "rotations = 30 (significant — they never got a clay "
            "game), 4+ rotations escalate further. Attributed to the "
            "player's first hard-court rotation."
        ),
    },
    {
        "key": "recent_history_within_7d",
        "category": "soft",
        "weight": RECENT_PAIR_WEIGHT_BANDS[0][1] if RECENT_PAIR_WEIGHT_BANDS else 0,
        "title": "Pair played together within the last 7 days",
        "description": (
            "Each pair that played together (partners OR opponents) "
            "in a session whose date falls in the last 7 calendar "
            "days. Covers any session type — Tue, Thu or Sat. A pair "
            "playing across multiple recent sessions accumulates one "
            "weight per session — e.g. seen 3 days AND 10 days ago "
            "earns 10 (7d band) plus 5 (8-14d band)."
        ),
    },
    {
        "key": "recent_history_8_to_14d",
        "category": "soft",
        "weight": (
            RECENT_PAIR_WEIGHT_BANDS[1][1]
            if len(RECENT_PAIR_WEIGHT_BANDS) > 1 else 0
        ),
        "title": "Pair played together 8-14 days ago",
        "description": "Same as above for the second week back; weighted lower.",
    },
    {
        "key": "imbalance",
        "category": "soft",
        "weight": PAIR_IMBALANCE_WEIGHT,
        "weight_label": (
            f"{PAIR_IMBALANCE_WEIGHT}×diff if diff ≤ 1; "
            f"else {PAIR_IMBALANCE_WEIGHT} + {PAIR_IMBALANCE_ESCALATION}×(diff−1)²"
        ),
        "title": "Pair-sum imbalance on a doubles court",
        "description": (
            "Absolute difference between the two pairs' rating sums on "
            "a doubles court. Linear at small diffs (0 or 1 — often "
            f"unavoidable, costs {PAIR_IMBALANCE_WEIGHT}× the diff), "
            "then quadratic for diff ≥ 2 so a badly-skewed split is "
            "sharply more expensive than a near-balanced one. Penalty "
            "= "
            f"{PAIR_IMBALANCE_WEIGHT} + {PAIR_IMBALANCE_ESCALATION}×"
            "(diff − 1)² for diff ≥ 2 — so e.g. diff 2 → 6, diff 3 → "
            "21, diff 4 → 46, diff 5 → 81. Strong at large diffs but "
            "well below the 500-point hard rules so a forced unbalanced "
            "split (opponent_repeat constraint) still wins out."
        ),
    },
    {
        "key": "same_court_successive",
        "category": "soft",
        "weight": SAME_COURT_SUCCESSIVE_PENALTY,
        "title": "Same pair of players sharing a court two rotations in a row",
        "description": (
            "Tie-breaker — often unavoidable, so kept very low. "
            "Discourages players who shared a court last rotation "
            "from sharing one again immediately."
        ),
    },
]


# ---------- data classes ------------------------------------------------


@dataclass
class Court:
    """One court's arrangement. ``mode`` is ``"doubles"`` or ``"singles"``.

    For doubles: ``players`` has 4 names, ``pairs`` has two 2-tuples (the
    two partnerships).
    For singles: ``players`` has 2 names, ``pairs`` has one 2-tuple (the
    matchup).

    ``pinned`` marks a court whose players + pair structure were fixed
    by an admin (``pinned_doubles``). Pinned courts contribute 0 to the
    score (the admin chose the line-up, we don't second-guess it) but
    still feed the cross-rotation tallies — partner/opponent repeats,
    unbalanced-count escalation, and the whole-evening per-player
    rules. The hill-climb is forbidden from moving players in or out.
    """

    court_label: str
    mode: str
    players: list[str]
    pairs: list[tuple[str, str]]
    pinned: bool = False


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
    # Names of attendees whose rating is still flagged as provisional in
    # the roster (bulk-imported from history, not yet confirmed by the
    # team). Pairings render as e.g. "Geoff(6P)" for these players;
    # otherwise plain "Geoff(6)". Persisted in the plan so the renderer
    # doesn't need to re-load the roster, and so historic plans keep
    # the marker even after the roster entry is confirmed.
    provisional_players: list[str] = field(default_factory=list)
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
    *,
    today: date | None = None,
    bands: list[tuple[int, int]] | None = None,
) -> dict[frozenset, int]:
    """Map each recent pair to its accumulated penalty weight.

    Date-based windows over ``history``. Each band is
    ``(max_days, weight)``: a history entry is classified into the
    FIRST band whose ``max_days`` covers ``(today - entry.date).days``,
    contributing that band's weight to every pair in the entry. Bands
    later in the list with a wider ``max_days`` are NOT also applied
    — they only catch entries that fell outside earlier bands.

    A pair accumulates weight ACROSS entries: under the default
    ``[(7, 10), (14, 5)]`` a pair that played 3 days AND 10 days ago
    scores 10 (from the 3d entry, in the 7d band) + 5 (from the 10d
    entry, in the 14d band) = 15.

    Entries whose ``date`` field isn't a parseable ISO date are
    silently skipped (no shared date → no penalty).
    """
    from datetime import date as _date_cls

    bands_list = list(bands if bands is not None else RECENT_PAIR_WEIGHT_BANDS)
    if not bands_list:
        return {}
    ref = today or _date_cls.today()
    out: dict[frozenset, int] = {}
    for entry in history:
        raw = entry.get("date")
        if not isinstance(raw, str):
            continue
        try:
            entry_date = _date_cls.fromisoformat(raw)
        except ValueError:
            continue
        age_days = (ref - entry_date).days
        if age_days < 0:
            continue  # future-dated entry — ignore
        # First-match-wins: an entry contributes ONE band weight (the
        # narrowest band it falls inside). Pairs accumulate across
        # multiple entries.
        weight = next(
            (w for max_days, w in bands_list if age_days <= max_days),
            0,
        )
        if weight == 0:
            continue
        for rot in entry.get("rotations", []):
            for court in rot.get("courts", []):
                for pair in court.get("pairs", []):
                    if len(pair) == 2:
                        fs = frozenset(pair)
                        out[fs] = out.get(fs, 0) + weight
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


def _validate_pinned_doubles(
    pinned_doubles: list[dict] | None,
    attendees_set: set[str],
    num_rotations: int,
    court_labels_set: set[str],
    pinned_singles_per_rotation: dict[int, dict],
    late_court: dict | None,
) -> list[dict]:
    """Validate ``pinned_doubles`` and return normalised entries.

    Each entry is normalised to::

        {"rotation_num": int | None,   # None = any rotation
         "players": (a, b, c, d),
         "pairs": ((a, b), (c, d)),    # tuple-of-tuples
         "court_label": str | None}

    Validation rules:
      * exactly 4 distinct attendees per pin;
      * pairs partition the 4 players cleanly;
      * ``rotation_num`` (when given) is in ``1..num_rotations``;
      * ``court_label`` (when given) is one of ``court_labels``;
      * no player appears in more than one pin assigned to the same
        rotation (or in two free-rotation pins, which would force two
        matches simultaneously by elimination);
      * no overlap with a ``pinned_singles`` pin on the same rotation;
      * no overlap with the ``late_court`` pin (the late court already
        forces those 4 to its first rotation; pinning them again
        would conflict).
    """
    if not pinned_doubles:
        return []
    lc_pinned_players: set[str] = set()
    lc_first: int = 0
    if late_court:
        lc_first = int(late_court.get("first_rotation") or 0)
        lc_pinned_players = set(late_court.get("pinned_players") or [])

    out: list[dict] = []
    # Track which players are claimed by FIXED-rotation pins
    # (rotation_num set) so a later free-rotation pin can't collide.
    fixed_claims: dict[int, set[str]] = {}
    free_pin_players: set[str] = set()
    # court_label claims per rotation (only for fixed-rotation pins).
    fixed_label_claims: dict[int, set[str]] = {}

    for raw in pinned_doubles:
        rot = raw.get("rotation_num")
        players = raw.get("players")
        pairs = raw.get("pairs")
        court_label = raw.get("court_label")

        if rot is not None and (
            not isinstance(rot, int) or not (1 <= rot <= num_rotations)
        ):
            raise ValueError(
                f"pinned_doubles.rotation_num {rot!r} must be int in "
                f"1..{num_rotations} or None"
            )
        if (
            not isinstance(players, (list, tuple))
            or len(players) != 4
        ):
            raise ValueError(
                f"pinned_doubles.players must list exactly 4 names "
                f"(got {players!r})"
            )
        players_tuple = tuple(str(p) for p in players)
        if len(set(players_tuple)) != 4:
            raise ValueError(
                f"pinned_doubles.players must be distinct (got {players_tuple!r})"
            )
        for p in players_tuple:
            if p not in attendees_set:
                raise ValueError(
                    f"pinned_doubles player {p!r} not in attendees"
                )
        if (
            not isinstance(pairs, (list, tuple))
            or len(pairs) != 2
            or any(
                not isinstance(pair, (list, tuple)) or len(pair) != 2
                for pair in pairs
            )
        ):
            raise ValueError(
                f"pinned_doubles.pairs must be two 2-tuples (got {pairs!r})"
            )
        norm_pairs = tuple(
            (str(pair[0]), str(pair[1])) for pair in pairs
        )
        flat = [p for pair in norm_pairs for p in pair]
        if sorted(flat) != sorted(players_tuple):
            raise ValueError(
                "pinned_doubles.pairs must partition players exactly "
                f"(players={players_tuple!r}, pairs={norm_pairs!r})"
            )
        if court_label is not None and str(court_label) not in court_labels_set:
            raise ValueError(
                f"pinned_doubles.court_label {court_label!r} not in court_labels"
            )

        # Overlap with pinned_singles on the same rotation.
        if rot is not None:
            ps = pinned_singles_per_rotation.get(rot)
            if ps is not None:
                shared = set(players_tuple) & set(ps["players"])
                if shared:
                    raise ValueError(
                        f"pinned_doubles for rotation {rot}: player(s) "
                        f"{sorted(shared)} are also pinned as singles "
                        "in the same rotation"
                    )
        # Overlap with the late court (whose 4 players are pinned in
        # its first_rotation).
        if lc_pinned_players and (rot is None or rot == lc_first):
            shared = set(players_tuple) & lc_pinned_players
            if shared:
                raise ValueError(
                    f"pinned_doubles: player(s) {sorted(shared)} are "
                    f"already pinned to the late court in rotation {lc_first}"
                )

        # Cross-pin collisions inside pinned_doubles itself.
        if rot is None:
            collide = set(players_tuple) & free_pin_players
            if collide:
                raise ValueError(
                    f"pinned_doubles: player(s) {sorted(collide)} appear "
                    "in more than one free-rotation pin"
                )
            # A free pin can also collide with any fixed pin in the
            # rotation it ends up in — but we don't know yet, so the
            # expansion step rechecks.
            free_pin_players.update(players_tuple)
        else:
            existing = fixed_claims.setdefault(rot, set())
            collide = set(players_tuple) & existing
            if collide:
                raise ValueError(
                    f"pinned_doubles for rotation {rot}: player(s) "
                    f"{sorted(collide)} are pinned to two doubles courts "
                    "in the same rotation"
                )
            existing.update(players_tuple)
            if court_label is not None:
                claimed = fixed_label_claims.setdefault(rot, set())
                if str(court_label) in claimed:
                    raise ValueError(
                        f"pinned_doubles for rotation {rot}: court "
                        f"{court_label!r} is pinned twice"
                    )
                claimed.add(str(court_label))

        out.append({
            "rotation_num": rot,
            "players": players_tuple,
            "pairs": norm_pairs,
            "court_label": (
                str(court_label) if court_label is not None else None
            ),
        })
    return out


def _expand_pinned_doubles(
    pins: list[dict],
    num_rotations: int,
    rng: random.Random,
    pinned_singles_per_rotation: dict[int, dict],
    late_court: dict | None,
) -> list[dict]:
    """Assign every free-rotation pin (``rotation_num is None``) to a
    concrete rotation. Returns a new list with no ``None`` entries.

    Picks a rotation that avoids player collisions with other pins
    already assigned there, and (if the pin specifies ``court_label``)
    avoids label conflicts. Different seeds will explore different
    assignments because the RNG drives the choice. Raises
    ``ValueError`` if no valid rotation is found.
    """
    if not pins:
        return []
    expanded: list[dict] = []
    # Player-claims per rotation across already-assigned pins.
    claims: dict[int, set[str]] = {}
    label_claims: dict[int, set[str]] = {}
    lc_first: int = 0
    lc_pinned_players: set[str] = set()
    if late_court:
        lc_first = int(late_court.get("first_rotation") or 0)
        lc_pinned_players = set(late_court.get("pinned_players") or [])

    def _conflicts(rot: int, pin: dict) -> str | None:
        players = set(pin["players"])
        # Pinned singles overlap.
        ps = pinned_singles_per_rotation.get(rot)
        if ps and (players & set(ps["players"])):
            return "pinned_singles overlap"
        # Late court overlap.
        if lc_pinned_players and rot == lc_first and (players & lc_pinned_players):
            return "late_court overlap"
        if claims.get(rot) and (players & claims[rot]):
            return "player double-booked"
        label = pin.get("court_label")
        if (
            label is not None
            and label_claims.get(rot)
            and label in label_claims[rot]
        ):
            return "court_label clash"
        return None

    # First pass: place every fixed-rotation pin so the free-rotation
    # pass sees them.
    for pin in pins:
        rot = pin["rotation_num"]
        if rot is None:
            continue
        claims.setdefault(rot, set()).update(pin["players"])
        if pin["court_label"] is not None:
            label_claims.setdefault(rot, set()).add(pin["court_label"])
        expanded.append(dict(pin))

    # Second pass: place free-rotation pins, RNG-randomised.
    candidate_rotations = list(range(1, num_rotations + 1))
    for pin in pins:
        if pin["rotation_num"] is not None:
            continue
        rng.shuffle(candidate_rotations)
        chosen: int | None = None
        for rot in candidate_rotations:
            if _conflicts(rot, pin) is None:
                chosen = rot
                break
        if chosen is None:
            raise ValueError(
                f"pinned_doubles: no available rotation for free pin "
                f"{pin['players']!r}"
            )
        claims.setdefault(chosen, set()).update(pin["players"])
        if pin["court_label"] is not None:
            label_claims.setdefault(chosen, set()).add(pin["court_label"])
        assigned = dict(pin)
        assigned["rotation_num"] = chosen
        expanded.append(assigned)
    return expanded


def _pair_rating_sum(
    pair: tuple[str, str], ratings: dict[str, int]
) -> int:
    return (
        ratings.get(pair[0], UNKNOWN_RATING)
        + ratings.get(pair[1], UNKNOWN_RATING)
    )


def _pair_imbalance_penalty(diff: int) -> int:
    """Penalty for a doubles court whose pair-sum imbalance is ``diff``.

    Linear at small diffs (0 or 1), then quadratic for diff ≥ 2 so a
    badly-skewed split (e.g. 2-vs-7) is sharply more expensive than a
    near-balanced one (4-vs-5). See PAIR_IMBALANCE_ESCALATION for the
    quadratic coefficient.

      diff 0 → 0
      diff 1 → 1
      diff 2 → 6
      diff 3 → 21
      diff 4 → 46
      diff 5 → 81
      …
    """
    if diff <= 0:
        return 0
    if diff <= 1:
        return PAIR_IMBALANCE_WEIGHT * diff
    return (
        PAIR_IMBALANCE_WEIGHT
        + PAIR_IMBALANCE_ESCALATION * (diff - 1) * (diff - 1)
    )


def _gender_court_penalty(c: Court, genders: dict[str, str]) -> int:
    """Gender-composition penalty for one doubles court.

      1. 3F+1M (one man with three women) → ``GENDER_3F1M_PENALTY``.
         Soft/medium — discouraged but not forbidden. (3M+1F is fine
         and is NOT penalised.)
      2. A 2M+2F court paired as MM-vs-FF (genders segregated; mixed
         pairings within the same 2+2 court are fine) →
         ``GENDER_MM_VS_FF_PENALTY``.
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
            penalty += GENDER_MM_VS_FF_PENALTY
    return penalty


def _court_max_rating_diff(c: Court, ratings: dict[str, int]) -> int:
    """The court's rating gap — highest minus lowest player rating
    (``?`` → ``UNKNOWN_RATING``). Works for doubles (4 players) AND
    singles (2 players); a lopsided singles match has just as much of
    a gap as a lopsided doubles court. Returns 0 if the court somehow
    has no players.
    """
    rs = [ratings.get(p, UNKNOWN_RATING) for p in c.players]
    return (max(rs) - min(rs)) if rs else 0


def _rating_gap_band(diff: int) -> tuple[str, int]:
    """Return ``(band_name, base_weight)`` for a court's rating gap.
    0-3 balanced (0), 4-5 unbalanced (20), 6-7 very_unbalanced (50),
    8-9 extremely_unbalanced (100)."""
    for threshold, name, base in RATING_GAP_BANDS:
        if diff >= threshold:
            return name, base
    return "balanced", 0


def _classify_balance(diff: int) -> str:
    """Band name for a court's rating gap (``balanced`` /
    ``unbalanced`` / ``very_unbalanced`` / ``extremely_unbalanced``)."""
    return _rating_gap_band(diff)[0]


def _rating_gap_penalty(
    c: Court,
    ratings: dict[str, int],
    unbalanced_count: dict[str, int] | None,
) -> tuple[str, int]:
    """``(band_name, penalty)`` for one court.

    ``penalty = base × Σ_{p in court} RATING_GAP_MULT ** prior[p]``
    where ``prior[p]`` is how many non-balanced rotations player ``p``
    has already had this evening (``unbalanced_count``, prior
    rotations only). Balanced courts (base 0) always score 0. The
    per-player escalation makes a 2nd/3rd non-balanced rotation for
    the same player progressively dearer, so the algorithm spreads
    the unavoidable mismatches across people and keeps each player to
    at most one where it can.
    """
    diff = _court_max_rating_diff(c, ratings)
    band, base = _rating_gap_band(diff)
    if base == 0:
        return band, 0
    uc = unbalanced_count or {}
    factor = sum(RATING_GAP_MULT ** uc.get(p, 0) for p in c.players)
    return band, base * factor


def _standard_too_low_items(
    rotations: "list[Rotation]",
    ratings: dict[str, int],
) -> list[dict]:
    """Whole-evening per-player "played down all night" penalty.

    For each player, look at the mean rating of the OTHER players on
    their court in every rotation they played. Their BEST rotation is
    the one with the strongest company (lowest mean — remember 1 is
    strongest). If even that best rotation's company is materially
    WEAKER than the player (mean − rating ≥ ``STANDARD_TOO_LOW_MATERIAL``),
    they never got an at-or-above game all evening → emit a penalty of
    ``round(STANDARD_TOO_LOW_WEIGHT × (best_mean − rating))`` attributed
    to that best rotation. Playing up (stronger company, mean < rating)
    is never penalised. Only ratings in
    ``[STANDARD_RULE_RATING_MIN, STANDARD_RULE_RATING_MAX]`` are
    eligible; ``?`` → ``UNKNOWN_RATING``.
    """
    # player -> list of (rotation_num, mean-of-others-on-court)
    per_player: dict[str, list[tuple[int, float]]] = {}
    for rot in rotations:
        for c in rot.courts:
            n = len(c.players)
            if n < 2:
                continue
            rs = [ratings.get(p, UNKNOWN_RATING) for p in c.players]
            total = sum(rs)
            for p, pr in zip(c.players, rs):
                others_mean = (total - pr) / (n - 1)
                per_player.setdefault(p, []).append(
                    (rot.rotation_num, others_mean)
                )

    items: list[dict] = []
    for p, seen in per_player.items():
        r = ratings.get(p, UNKNOWN_RATING)
        if not isinstance(r, int):
            r = UNKNOWN_RATING
        if not (STANDARD_RULE_RATING_MIN <= r <= STANDARD_RULE_RATING_MAX):
            continue
        best_rot, best_mean = min(seen, key=lambda x: x[1])
        shortfall = best_mean - r
        if shortfall >= STANDARD_TOO_LOW_MATERIAL:
            pts = int(round(STANDARD_TOO_LOW_WEIGHT * shortfall))
            if pts > 0:
                items.append({
                    "rule": "standard_too_low",
                    "points": pts,
                    "player": p,
                    "rotation_num": best_rot,
                })
    return items


def _top_player_no_strong_items(
    rotations: "list[Rotation]",
    ratings: dict[str, int],
) -> list[dict]:
    """Whole-evening guarantee for the very top players.

    A player rated ``≤ TOP_PLAYER_MAX_RATING`` should get at least one
    rotation whose every OTHER player is rated no worse than their own
    rating + 1. We take the player's best-attempt rotation (the one
    whose WORST other player is the strongest, i.e. min of the
    per-rotation max-other-rating). If even that exceeds the ceiling
    (own rating + 1) they never got a top game → penalty of
    ``round(TOP_PLAYER_STRONG_WEIGHT × (best_max_other − ceiling))``,
    attributed to that rotation. Zero once they get a qualifying
    rotation; stronger company is never penalised. ``?`` → 6 so an
    unrated co-player correctly disqualifies a court.
    """
    # player -> list of (rotation_num, max rating among the OTHERS)
    per_player: dict[str, list[tuple[int, int]]] = {}
    for rot in rotations:
        for c in rot.courts:
            n = len(c.players)
            if n < 2:
                continue
            rs = [ratings.get(p, UNKNOWN_RATING) for p in c.players]
            for idx, p in enumerate(c.players):
                others_max = max(
                    rs[j] for j in range(n) if j != idx
                )
                per_player.setdefault(p, []).append(
                    (rot.rotation_num, others_max)
                )

    items: list[dict] = []
    for p, seen in per_player.items():
        r = ratings.get(p, UNKNOWN_RATING)
        if not isinstance(r, int):
            r = UNKNOWN_RATING
        if r > TOP_PLAYER_MAX_RATING:
            continue
        ceiling = r + 1
        best_rot, best_max = min(seen, key=lambda x: x[1])
        shortfall = best_max - ceiling
        if shortfall > 0:
            pts = int(round(TOP_PLAYER_STRONG_WEIGHT * shortfall))
            if pts > 0:
                items.append({
                    "rule": "top_player_no_strong_rotation",
                    "points": pts,
                    "player": p,
                    "rotation_num": best_rot,
                })
    return items


def _court_label_to_number(label: str) -> int | None:
    """Return the numeric court number from a label (e.g.
    ``"Court #5 - Floodlit"`` → 5), or ``None`` for non-numeric labels
    like ``"AY1"`` or ``"Outdoor"``. Uses the same normalisation rules
    as :func:`_court_label_key`."""
    key = _court_label_key(label)
    return int(key) if key.isdigit() else None


def _is_hard_court(
    label: str, hard_set: frozenset[int] = HARD_COURT_NUMBERS,
) -> bool:
    """True iff ``label`` resolves to a court number in ``hard_set``.
    Non-numeric labels are treated as not-hard (no penalty), since
    we can't be sure of their surface."""
    n = _court_label_to_number(label)
    return n is not None and n in hard_set


def _hard_court_repeat_items(
    rotations: "list[Rotation]",
    hard_set: frozenset[int] = HARD_COURT_NUMBERS,
) -> list[dict]:
    """Whole-evening per-player penalty for ≥2 hard-court rotations.

    Players prefer clay; hard courts (1-4) are tolerated for one
    rotation per player. Two or more earn an escalating penalty:
    ``HARD_COURT_REPEAT_WEIGHT × Σ_{i=1..(hard_count-1)} i``.

      hard_count 0-1 → 0
      hard_count 2   → 10
      hard_count 3   → 30 (10 + 20 — no clay all evening)
      hard_count 4+  → 60, 100, …

    Attributed to the player's FIRST hard-court rotation so the
    per-rotation breakdown reconciles with the running total (and so
    the hill-climb, which rescores via _rescore_layout, optimises
    against this rule). Non-numeric court labels are treated as clay
    (no penalty) — better than penalising for an unknown surface.
    """
    per_player_rots: dict[str, list[int]] = {}
    for rot in rotations:
        for c in rot.courts:
            if _is_hard_court(c.court_label, hard_set):
                for p in c.players:
                    per_player_rots.setdefault(p, []).append(rot.rotation_num)

    items: list[dict] = []
    for p, rot_nums in per_player_rots.items():
        n = len(rot_nums)
        if n <= 1:
            continue
        # Escalating weight: 1× for the 2nd hard rotation, 2× for the
        # 3rd, etc. — so a "3 of 3 on hard" is 3× worse than "2 of 3".
        pts = HARD_COURT_REPEAT_WEIGHT * sum(range(1, n))
        if pts > 0:
            items.append({
                "rule": "hard_court_repeat",
                "points": pts,
                "player": p,
                "hard_rotations": n,
                "rotation_num": min(rot_nums),
            })
    return items


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
    if getattr(court, "pinned", False):
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
    score += _pair_imbalance_penalty(imbalance)
    score += _gender_court_penalty(court, genders)
    # Rating-gap band penalty (escalates per-player by prior
    # non-balanced rotations). Replaces the old extreme-mix +
    # very-unbalanced + per-player-increment rules.
    score += _rating_gap_penalty(court, ratings, unbalanced_count)[1]
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
    unbalanced_count: dict[str, int] | None = None,
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
        if ratings is not None:
            score += _rating_gap_penalty(c, ratings, unbalanced_count)[1]
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
        if getattr(c, "pinned", False):
            # Admin-pinned court — scoring-exempt. No items emitted here;
            # cross-rotation effects propagate via the tracking sets
            # updated by the caller after the rotation completes.
            continue
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
                "imbalance", _pair_imbalance_penalty(imbalance),
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
                        "gender_MM_vs_FF", GENDER_MM_VS_FF_PENALTY,
                        court=c.court_label,
                    )
            band, gap_pts = _rating_gap_penalty(
                c, ratings, unbalanced_count,
            )
            if gap_pts > 0:
                emit(
                    f"rating_gap_{band}", gap_pts,
                    court=c.court_label,
                    players=list(c.players),
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
            if ratings is not None:
                band, gap_pts = _rating_gap_penalty(
                    c, ratings, unbalanced_count,
                )
                if gap_pts > 0:
                    emit(
                        f"rating_gap_{band}", gap_pts,
                        court=c.court_label,
                        players=list(c.players),
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
    forced_doubles_pin: list[str] | None = None,
    forced_doubles_label: str | None = None,
) -> tuple[list[Court], int]:
    """Build one random layout and return (courts, score).

    ``forced_singles_pair`` (and optional ``forced_singles_label``) pin a
    specific pair to a singles court. Both names must already appear in
    ``singles_players``. If no label is given, the pair lands on the
    first singles court (``singles_labels[0]``).

    ``forced_doubles_pin`` pins exactly 4 players to a doubles court
    (``forced_doubles_label``, defaulting to ``doubles_labels[0]``). All
    4 must already appear in ``doubles_players``. The 2-pair split among
    those 4 is still optimised via ``_build_best_doubles_court``.
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

    forced_d_idx: int | None = None
    if forced_doubles_pin is not None and doubles_labels:
        if len(forced_doubles_pin) != 4:
            raise ValueError(
                f"forced_doubles_pin must be exactly 4 players, "
                f"got {len(forced_doubles_pin)}"
            )
        forced_d_idx = (
            doubles_labels.index(forced_doubles_label)
            if forced_doubles_label is not None
            else 0
        )
        forced_d_list = list(forced_doubles_pin)
        others_d = [p for p in shuffled_d if p not in set(forced_d_list)]
        rebuilt_d: list[str] = []
        for i in range(len(doubles_labels)):
            if i == forced_d_idx:
                rebuilt_d.extend(forced_d_list)
            else:
                rebuilt_d.extend(others_d[:4])
                others_d = others_d[4:]
        shuffled_d = rebuilt_d

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
        unbalanced_count,
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
    late_court: dict | None = None,
    pinned_doubles: list[dict] | None = None,
) -> list[tuple[list[Court], list[str]]]:
    """Build ``num_rotations`` rotations of mixed doubles+singles courts.

    ``late_court`` (when supplied) configures a court that's only
    available from rotation ``first_rotation`` onwards. Shape::

        {"label": "5", "first_rotation": 2, "pinned_players": [4 names]}

    Effect:
      * rotations < first_rotation: the 4 pinned players sit out and
        the late court label is excluded from the available set.
      * rotation == first_rotation: the late court is in the pool as a
        doubles court with the 4 pinned players (Boris picks the 2v2
        split optimally).
      * rotations > first_rotation: the late court is fully in the
        pool with no pinning.

    ``pinned_doubles`` (when supplied) is a list of already-expanded
    pinned-doubles entries — every entry has a concrete ``rotation_num``
    (free-rotation pins must be expanded by the caller, see
    ``_expand_pinned_doubles``). Each entry pins 4 players to a doubles
    court in that rotation with a fixed pair structure; the pinned
    court contributes 0 to its own per-court score but still feeds the
    cross-rotation tallies (so a partner repeat with the pinned pair
    elsewhere is penalised, the players' ``unbalanced_count`` ticks up
    if the pinned court is non-balanced, etc.).
    """
    n = len(attendees)
    c = len(court_labels)
    capacity = 4 * c

    lc_label: str | None = None
    lc_first: int = 0
    lc_pinned: list[str] = []
    if late_court:
        lc_label = str(late_court.get("label") or "").strip() or None
        lc_first = int(late_court.get("first_rotation") or 0)
        lc_pinned = list(late_court.get("pinned_players") or [])
        if lc_label is not None and lc_label not in court_labels:
            raise ValueError(
                f"late court label {lc_label!r} not in court_labels {court_labels!r}"
            )
        if lc_label is not None and lc_first < 1:
            raise ValueError(
                f"late_court first_rotation must be >= 1, got {lc_first}"
            )
        if lc_label is not None and len(lc_pinned) != 4:
            raise ValueError(
                f"late court needs exactly 4 pinned players, got {lc_pinned!r}"
            )
        if lc_label is not None:
            missing = [p for p in lc_pinned if p not in attendees]
            if missing:
                raise ValueError(
                    f"late court pinned players not in attendees: {missing!r}"
                )

    if n > capacity:
        raise ValueError(
            f"{n} attendees exceeds capacity ({capacity} = 4×{c} courts). "
            "Drop someone or add a court."
        )

    # Group pinned-doubles entries by rotation. Caller is responsible
    # for expanding any free-rotation pins to concrete rotation numbers.
    pinned_doubles_by_rot: dict[int, list[dict]] = {}
    for pin in pinned_doubles or []:
        rot = pin.get("rotation_num")
        if not isinstance(rot, int):
            raise ValueError(
                "pinned_doubles entries reaching the strategy must have "
                f"concrete rotation_num; got {pin!r}"
            )
        pinned_doubles_by_rot.setdefault(rot, []).append(pin)

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
        rot_pinned_doubles = pinned_doubles_by_rot.get(rotation_num, [])
        # Per-rotation court availability — late court drops out before
        # its first rotation, comes back from first_rotation onwards.
        if lc_label is not None and rotation_num < lc_first:
            rot_court_labels_initial = [
                x for x in court_labels if x != lc_label
            ]
            forced_doubles_pin: list[str] | None = None
            forced_doubles_label: str | None = None
            late_sit_outs: list[str] = list(lc_pinned)
        elif lc_label is not None and rotation_num == lc_first:
            rot_court_labels_initial = list(court_labels)
            forced_doubles_pin = list(lc_pinned)
            forced_doubles_label = lc_label
            late_sit_outs = []
        else:
            rot_court_labels_initial = list(court_labels)
            forced_doubles_pin = None
            forced_doubles_label = None
            late_sit_outs = []

        # Labels that must end up in the doubles range so the pinned /
        # late-court pins can be placed. Late-court first (it appears
        # in the existing single-pin scheme at index 0), then any
        # admin-pinned-doubles labels.
        forced_doubles_labels_here: list[str] = []
        if forced_doubles_label is not None:
            forced_doubles_labels_here.append(forced_doubles_label)
        for pd in rot_pinned_doubles:
            cl = pd.get("court_label")
            if cl and cl not in forced_doubles_labels_here:
                if cl not in court_labels:
                    raise ValueError(
                        f"pinned_doubles rotation {rotation_num}: court "
                        f"{cl!r} not in court_labels {court_labels!r}"
                    )
                forced_doubles_labels_here.append(cl)
        # Rearrange so all forced doubles labels lead, preserving the
        # original order for the remainder.
        rest_labels = [
            x for x in rot_court_labels_initial
            if x not in forced_doubles_labels_here
        ]
        rot_court_labels = forced_doubles_labels_here + rest_labels

        rot_c = len(rot_court_labels)
        rot_capacity = 4 * rot_c
        pinned_doubles_player_set: set[str] = set()
        for pd in rot_pinned_doubles:
            pinned_doubles_player_set.update(pd["players"])
        rot_n = n - len(late_sit_outs)
        if rot_n > rot_capacity:
            raise ValueError(
                f"rotation {rotation_num}: {rot_n} active players exceed "
                f"capacity ({rot_capacity} = 4×{rot_c} courts) "
                "after late-court sit-outs"
            )
        sit_outs_extra = 1 if rot_n % 2 == 1 else 0
        effective_n = rot_n - sit_outs_extra
        num_singles_courts = (rot_capacity - effective_n) // 2
        num_doubles_courts = rot_c - num_singles_courts
        doubles_labels = list(rot_court_labels[:num_doubles_courts])
        singles_labels = list(rot_court_labels[num_doubles_courts:])
        if forced_doubles_label is not None and forced_doubles_label not in doubles_labels:
            raise ValueError(
                f"rotation {rotation_num}: late court {forced_doubles_label!r} "
                f"could not be allocated as doubles "
                f"(doubles_labels={doubles_labels!r})"
            )
        for pd in rot_pinned_doubles:
            cl = pd.get("court_label")
            if cl and cl not in doubles_labels:
                raise ValueError(
                    f"pinned_doubles rotation {rotation_num}: court "
                    f"{cl!r} could not be allocated as doubles "
                    f"(doubles_labels={doubles_labels!r}) — too many "
                    "singles courts this rotation?"
                )
        # Assign labels to pinned doubles entries that didn't specify
        # one. Prefer doubles labels not already claimed by the late
        # court or by another pinned_doubles entry.
        already_claimed = {forced_doubles_label} - {None}
        already_claimed.update(
            pd["court_label"] for pd in rot_pinned_doubles
            if pd.get("court_label")
        )
        free_doubles_labels = [
            l for l in doubles_labels if l not in already_claimed
        ]
        resolved_pins: list[dict] = []
        for pd in rot_pinned_doubles:
            cl = pd.get("court_label")
            if cl is None:
                if not free_doubles_labels:
                    raise ValueError(
                        f"pinned_doubles rotation {rotation_num}: no "
                        "free doubles court to assign this pin to"
                    )
                cl = free_doubles_labels.pop(0)
            resolved_pins.append({
                "players": pd["players"],
                "pairs": pd["pairs"],
                "court_label": cl,
            })

        # Fair sit-out selection — pinned-singles players, late-court
        # pins, and pinned-doubles players must NOT be in the rotating
        # sit-out pool.
        if sit_outs_extra:
            forced_in = set(pin["players"]) if pin else set()
            forced_in.update(forced_doubles_pin or [])
            forced_in.update(pinned_doubles_player_set)
            sittable = [
                p for p in attendees
                if p not in forced_in and p not in late_sit_outs
            ]
            ranked = sorted(
                sittable, key=lambda p: (sitout_count[p], rng.random())
            )
            extra_sit_outs = ranked[:sit_outs_extra]
            for s in extra_sit_outs:
                sitout_count[s] += 1
        else:
            extra_sit_outs = []
        sit_outs = list(late_sit_outs) + list(extra_sit_outs)
        for s in late_sit_outs:
            sitout_count[s] += 1
        active = [p for p in attendees if p not in sit_outs]

        # Pick singles-destined players this rotation (with matchup-rotation
        # bias via singles_count). Honour any pin first. Players pinned
        # to the late court OR to an admin-pinned doubles court are
        # excluded from singles candidates entirely.
        singles_slots = 2 * num_singles_courts
        forced_pair: tuple[str, str] | None = None
        forced_label: str | None = None
        late_pin_set = set(forced_doubles_pin or [])
        singles_candidates = [
            p for p in active
            if p not in late_pin_set
            and p not in pinned_doubles_player_set
        ]
        if pin is not None:
            if singles_slots < 2:
                raise ValueError(
                    f"pinned_singles for rotation {rotation_num}: this "
                    "rotation has no singles court"
                )
            forced_pair = pin["players"]
            forced_label = pin["court_label"]
            forced_set = set(forced_pair)
            remaining_active = [p for p in singles_candidates if p not in forced_set]
            remaining_singles = _select_singles_players(
                remaining_active, singles_slots - 2,
                ratings, singles_prefs, singles_count, rng,
            )
            singles_players = list(forced_pair) + remaining_singles
        else:
            singles_players = _select_singles_players(
                singles_candidates, singles_slots,
                ratings, singles_prefs, singles_count, rng,
            )
        # The non-pinned doubles players that _try_layout will shuffle.
        doubles_players = [
            p for p in active
            if p not in singles_players
            and p not in pinned_doubles_player_set
        ]
        # Likewise, the doubles labels _try_layout sees exclude the
        # admin-pinned ones (we build those directly below).
        pinned_doubles_label_set = {rp["court_label"] for rp in resolved_pins}
        try_doubles_labels = [
            l for l in doubles_labels if l not in pinned_doubles_label_set
        ]

        # Pre-build the admin-pinned doubles courts (fixed pair
        # structure, ``pinned=True`` so per-court scoring skips them).
        pinned_courts_built: list[Court] = []
        for rp in resolved_pins:
            pa, pb = rp["pairs"]
            pinned_courts_built.append(Court(
                court_label=rp["court_label"],
                mode="doubles",
                players=list(rp["players"]),
                pairs=[tuple(pa), tuple(pb)],
                pinned=True,
            ))

        # Rejection-sample layouts for the non-pinned remainder.
        best_courts: list[Court] | None = None
        best_score: int | None = None
        attempts_made = 0
        for _attempt in range(MAX_ATTEMPTS):
            attempts_made += 1
            courts, score = _try_layout(
                doubles_players,
                singles_players,
                try_doubles_labels,
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
                forced_doubles_pin=forced_doubles_pin,
                forced_doubles_label=forced_doubles_label,
            )
            if best_score is None or score < best_score:
                best_courts = courts
                best_score = score
                if score == 0:
                    break
        assert best_courts is not None
        # Combine the pinned courts with the optimiser-built courts and
        # re-sort by the rotation's label order so display is stable.
        combined = list(best_courts) + pinned_courts_built
        label_order = {l: i for i, l in enumerate(rot_court_labels)}
        combined.sort(key=lambda c_: label_order.get(c_.court_label, 1_000_000))
        best_courts = combined

        # Score breakdown — must happen BEFORE the tracking sets are
        # updated, otherwise the layout's own pairs get flagged as
        # repeats and the breakdown stops summing back to best_score.
        # Pinned courts contribute 0 to the per-court score and emit no
        # items (they're cross-rotation tracking only).
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
            elif c_.mode == "singles":
                intra_opponents.add(frozenset(c_.pairs[0]))
            # Tally non-balanced rotations per player (doubles OR
            # singles) so the gap-band penalty escalates next rotation.
            if _classify_balance(
                _court_max_rating_diff(c_, ratings)
            ) != "balanced":
                for p in c_.players:
                    unbalanced_count[p] = unbalanced_count.get(p, 0) + 1
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
POLISH_MAX_ITERATIONS = 15000
POLISH_MAX_NO_IMPROVEMENT = 3000
POLISH_MIN_BASELINE = 1  # don't bother if baseline is already 0.
# Multi-start polish: when the best seed plan still scores > 0, run
# the hill-climb from the K lowest-scoring seed plans (each with its
# own RNG) and keep whichever polishes lowest. The pre-polish best
# isn't always the one that polishes to the best local optimum.
# Skipped entirely when the best seed plan already scored 0 (nothing
# to improve). Cost is ~linear in K, but only on constrained runs.
POLISH_MULTISTART_K = 5


def _rescore_layout(
    layout: list[list[list[str]]],
    *,
    rotation_modes: list[list[str]],
    rotation_labels: list[list[str]],
    rotation_sit_outs: list[list[str]],
    weekly_pair_penalties: dict[frozenset, int],
    ratings: dict[str, int],
    genders: dict[str, str],
    pinned_courts: dict[tuple[int, int], list[tuple[str, str]]] | None = None,
) -> tuple[int, list[dict], list[Rotation]]:
    """Replay a plan from scratch given the player assignments.

    ``layout[i][j]`` is the list of player names on rotation i, court j
    (in any order — the function picks the best pair split for doubles
    courts internally). The function returns
    ``(total_score, per_rotation_metrics, rebuilt_rotations)``.

    ``pinned_courts`` maps ``(rot_idx, court_idx)`` of admin-pinned
    doubles courts to their fixed pair structure. Those courts use the
    given pairs (no best-split search), are marked ``pinned=True``, and
    so contribute 0 to the per-court score. They still update the
    cross-rotation tracking sets the same as any other court.
    """
    intra_partners: set[frozenset] = set()
    intra_opponents: set[frozenset] = set()
    prev_court_pairs: set[frozenset] = set()
    unbalanced_count: dict[str, int] = {}
    per_rotation: list[dict] = []
    rebuilt: list[Rotation] = []
    total = 0
    pinned_courts = pinned_courts or {}

    for rot_idx, courts_players in enumerate(layout):
        modes = rotation_modes[rot_idx]
        labels = rotation_labels[rot_idx]
        sit_outs = rotation_sit_outs[rot_idx]
        new_courts: list[Court] = []
        for ci, players in enumerate(courts_players):
            label = labels[ci]
            mode = modes[ci]
            pinned_pairs = pinned_courts.get((rot_idx, ci))
            if pinned_pairs is not None and mode == "doubles":
                # Admin-pinned: keep the fixed pair structure. The
                # ``pinned`` flag suppresses per-court scoring.
                court = Court(
                    court_label=label,
                    mode="doubles",
                    players=list(players),
                    pairs=[tuple(pp) for pp in pinned_pairs],
                    pinned=True,
                )
            elif mode == "doubles":
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
                unbalanced_count,
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
            else:
                intra_opponents.add(frozenset(c.pairs[0]))
            if (
                _classify_balance(_court_max_rating_diff(c, ratings))
                != "balanced"
            ):
                for p in c.players:
                    unbalanced_count[p] = unbalanced_count.get(p, 0) + 1
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

    # Whole-evening "played down all night" penalty — needs all
    # rotations, so applied here once the plan is complete. Attributed
    # to each player's best rotation so per-rotation best_score +
    # breakdown still reconcile with the total (and so the hill-climb,
    # which scores via this function, optimises against it).
    by_rot = {pr["rotation_num"]: pr for pr in per_rotation}
    evening_items = (
        _standard_too_low_items(rebuilt, ratings)
        + _top_player_no_strong_items(rebuilt, ratings)
        + _hard_court_repeat_items(rebuilt)
    )
    for it in evening_items:
        pr = by_rot.get(it["rotation_num"]) or (
            per_rotation[0] if per_rotation else None
        )
        if pr is None:
            continue
        pr["best_score"] += it["points"]
        pr["breakdown_items"].append(it)
        pr["breakdown"] = _aggregate_breakdown(pr["breakdown_items"])
        total += it["points"]

    return total, per_rotation, rebuilt


def polish_plan(
    plan: PairingPlan,
    *,
    seed: int | None = None,
    max_iterations: int = POLISH_MAX_ITERATIONS,
    max_no_improvement: int = POLISH_MAX_NO_IMPROVEMENT,
    verbose: bool = True,
    late_court: dict | None = None,
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

    # Identify the (rot_idx, court_idx) of the late-court forced
    # doubles slot — polish must NEVER move players into or out of it.
    # In rotations before first_rotation the 4 pinned players are
    # already in rotation_sit_outs, so the existing sit-out checks
    # cover them; here we just lock the forced rotation.
    locked_court: tuple[int, int] | None = None
    if late_court:
        lc_label = str(late_court.get("label") or "").strip() or None
        lc_first = int(late_court.get("first_rotation") or 0)
        if lc_label and lc_first >= 1:
            target_rot_idx = lc_first - 1
            if 0 <= target_rot_idx < len(rotation_labels):
                try:
                    court_idx = rotation_labels[target_rot_idx].index(lc_label)
                    locked_court = (target_rot_idx, court_idx)
                except ValueError:
                    locked_court = None
    # Admin-pinned doubles courts (Court.pinned == True): same lock as
    # the late court — hill-climb is forbidden from swapping players
    # into or out of these positions — and ALSO carry their fixed pair
    # structure into _rescore_layout so the partnerships survive.
    locked_pinned_doubles: set[tuple[int, int]] = set()
    pinned_courts_map: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for rot_idx, rot in enumerate(plan.rotations):
        for court_idx, c in enumerate(rot.courts):
            if getattr(c, "pinned", False):
                locked_pinned_doubles.add((rot_idx, court_idx))
                pinned_courts_map[(rot_idx, court_idx)] = [
                    tuple(pair) for pair in c.pairs
                ]
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
        pinned_courts=pinned_courts_map,
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
            pinned_courts_map=pinned_courts_map,
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
        # Reject any swap that touches the late-court forced doubles
        # slot — its 4 players are pinned by the admin.
        if locked_court is not None and (
            (rot_a, court_a) == locked_court
            or (rot_b, court_b) == locked_court
        ):
            no_improvement += 1
            continue
        # Same for admin-pinned doubles courts — players + pair
        # structure are admin's choice; hill-climb can't touch them.
        if locked_pinned_doubles and (
            (rot_a, court_a) in locked_pinned_doubles
            or (rot_b, court_b) in locked_pinned_doubles
        ):
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
                pinned_courts=pinned_courts_map,
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
            "wall_seconds": round(time.perf_counter() - t_start, 3),
            "skipped": False,
        },
        pinned_courts_map=pinned_courts_map,
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
    pinned_courts_map: dict[tuple[int, int], list[tuple[str, str]]] | None = None,
) -> PairingPlan:
    """Build a fresh PairingPlan from the polished layout."""
    total, per_rotation, rebuilt = _rescore_layout(
        layout,
        rotation_modes=rotation_modes,
        rotation_labels=rotation_labels,
        rotation_sit_outs=rotation_sit_outs,
        weekly_pair_penalties=weekly_pair_penalties,
        ratings=ratings, genders=genders,
        pinned_courts=pinned_courts_map,
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
        provisional_players=list(original.provisional_players),
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
    pinned_doubles: list[dict] | None = None,
    late_court: dict | None = None,
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
    weekly_pair_penalties = recent_pair_weights(history, today=today)
    ratings = _build_ratings(players)
    genders: dict[str, str] = {
        n: (str(info.get("gender", "?")).strip().upper() or "?")
        for n, info in players.items()
    }
    # Provisional flag from the roster, restricted to attendees. The
    # plan only needs the players who are actually going to be
    # rendered, not the whole roster, so the snapshot stays small.
    attendees_set = set(attendees)
    provisional_players: list[str] = sorted(
        n for n, info in players.items()
        if n in attendees_set and info.get("provisional")
    )
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
    validated_pinned_doubles = _validate_pinned_doubles(
        pinned_doubles,
        attendees_set=set(attendees),
        num_rotations=num_rotations,
        court_labels_set=set(labels_list),
        pinned_singles_per_rotation=pinned_per_rotation,
        late_court=late_court,
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
            provisional_players=provisional_players,
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
    # Expand any free-rotation pinned-doubles entries to a concrete
    # rotation (RNG-driven, so different seeds explore different
    # rotation assignments — that's the diversification we want).
    expanded_pinned_doubles = _expand_pinned_doubles(
        validated_pinned_doubles, num_rotations, rng,
        pinned_singles_per_rotation=pinned_per_rotation,
        late_court=late_court,
    )
    per_rotation = STRATEGIES[strategy](
        attendees, labels_list, num_rotations,
        ratings, genders, singles_prefs, weekly_pair_penalties,
        pinned_per_rotation, rng,
        late_court=late_court,
        pinned_doubles=expanded_pinned_doubles,
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
        provisional_players=provisional_players,
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
    pinned_doubles: list[dict] | None = None,
    late_court: dict | None = None,
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
        pinned_doubles=pinned_doubles,
        late_court=late_court,
    )

    if num_seed_attempts <= 1:
        single = _make_plan_one(seed=seed, **common)
        if polish:
            single = polish_plan(
                single, seed=seed, verbose=True, late_court=late_court,
            )
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

    if polish and best_total > 0 and len(plans) > 1:
        # Multi-start polish: the lowest pre-polish plan doesn't always
        # polish to the best local optimum. Polish the K lowest-scoring
        # seed plans (each with its own RNG seed) and keep whichever
        # ends up lowest. Each candidate carries the seed-search
        # metadata so polish_plan can refresh chosen_total/blocking.
        order = sorted(
            range(len(plans)), key=lambda i: totals[i]
        )[:POLISH_MULTISTART_K]
        polished: list[PairingPlan] = []
        for rank, i in enumerate(order):
            cand = plans[i]
            cand.metrics["multi_seed"] = dict(
                chosen.metrics.get("multi_seed", {})
            )
            polished.append(polish_plan(
                cand, seed=seeds_used[i],
                verbose=(rank == 0), late_court=late_court,
            ))
        chosen = min(polished, key=_plan_total)
        # Every candidate's polish cost is real wall time, not just the
        # winner's — sum them so the admin-facing time isn't ~Kx low.
        polish_secs = sum(
            p.metrics.get("polish", {}).get("wall_seconds", 0) or 0
            for p in polished
        )
        # Each polish iteration evaluates a full candidate evening, so
        # it counts toward "candidate layouts tried" exactly like a
        # seed attempt does — across ALL K hill-climbs.
        polish_iters = sum(
            int(p.metrics.get("polish", {}).get("iterations", 0) or 0)
            for p in polished
        )
        print(
            f"[pairings] multi-start polish: {len(order)} plan(s) "
            f"(pre-polish totals "
            f"{[totals[i] for i in order]}) -> best post-polish "
            f"total={int(_plan_total(chosen))} "
            f"(polish {polish_secs:.2f}s total)"
        )
    elif polish:
        chosen = polish_plan(
            chosen, seed=seed, verbose=True, late_court=late_court,
        )
        polish_secs = (
            chosen.metrics.get("polish", {}).get("wall_seconds", 0) or 0
        )
        polish_iters = int(
            chosen.metrics.get("polish", {}).get("iterations", 0) or 0
        )
    else:
        polish_secs = 0.0
        polish_iters = 0

    # Fold the polish hill-climb work into the surfaced
    # "candidate layouts tried" number — it was previously seed-phase
    # attempts only, which understated the real search (especially
    # with K-way multi-start polish).
    ms = chosen.metrics.get("multi_seed")
    if ms is not None:
        ms["total_permutations_tried"] = (
            int(ms.get("total_permutations_tried", 0) or 0) + polish_iters
        )

    # Top-level wall-time summary: multi-seed + polish combined, so the
    # bot can quote one number to the admin in the score footer.
    multi_seed_secs = chosen.metrics.get("multi_seed", {}).get("wall_seconds", 0) or 0
    chosen.metrics["wall_seconds"] = round(multi_seed_secs + polish_secs, 2)

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


import re as _re_for_label_key


def _court_label_key(label: str) -> str:
    """Return a comparison key that equates the same court written in
    different ways. CourtReserve stores labels as ``"Court #5 - Floodlit"``
    while admins type just ``"5"`` (or ``"court 5"``), and the renderer
    shortens to ``"Court 5"``. All of these should match. For non-numeric
    labels (e.g. ``"AY1"``, ``"Outdoor"``) we fall back to the trimmed,
    lowercased string.
    """
    s = str(label or "").strip().rstrip(":").strip()
    if "#" in s:
        after = s.split("#", 1)[1]
        m = _re_for_label_key.match(r"\s*(\d+)", after)
        if m:
            return m.group(1)
    m = _re_for_label_key.match(r"^\s*[Cc]ourt\s+(\d+)\s*$", s)
    if m:
        return m.group(1)
    if s.isdigit():
        return s
    return s.lower()


def swap_courts_in_plan(
    plan: dict,
    label_a: str,
    label_b: str,
    rotation_nums: list[int] | None = None,
) -> list[int]:
    """Swap the contents of two courts across selected rotations.

    The court labels stay put; their ``mode`` / ``players`` / ``pairs``
    payloads are exchanged. So ``swap_courts_in_plan(plan, "5", "11")``
    makes the matchups that were scheduled on Ct 11 (e.g. the singles)
    play on Ct 5 instead, and vice versa — without regenerating the
    plan or disturbing pinned matchups elsewhere.

    Court labels match leniently — ``"5"``, ``"Court 5"`` and
    ``"Court #5 - Floodlit"`` all refer to the same court. Useful
    because CourtReserve stores the long form but admins type the
    short one.

    ``rotation_nums`` is a list of 1-based rotation numbers; ``None``
    (the default) means "every rotation". Used when an admin wants to
    swap two courts for only some of the evening — e.g. "swap courts
    1 and 5 for rotations 2 and 3" when court 1 is a less-preferred
    hard court and the group currently scheduled there for those
    rotations should move to the clay court.

    Returns the list of 1-based rotation numbers actually swapped.
    Raises ``ValueError`` if either label is missing from a targeted
    rotation, or if any value in ``rotation_nums`` is out of range.
    """
    label_a = str(label_a)
    label_b = str(label_b)
    key_a = _court_label_key(label_a)
    key_b = _court_label_key(label_b)
    if key_a == key_b:
        return []
    rots = plan.get("rotations", [])
    if rotation_nums is None:
        target_indices = list(range(len(rots)))
    else:
        target_indices = []
        for rn in rotation_nums:
            if not isinstance(rn, int) or not (1 <= rn <= len(rots)):
                raise ValueError(
                    f"rotation_num {rn!r} out of range 1..{len(rots)}"
                )
            target_indices.append(rn - 1)
    swapped: list[int] = []
    for idx in target_indices:
        rot = rots[idx]
        court_a = next(
            (c for c in rot["courts"]
             if _court_label_key(c["court_label"]) == key_a),
            None,
        )
        court_b = next(
            (c for c in rot["courts"]
             if _court_label_key(c["court_label"]) == key_b),
            None,
        )
        if court_a is None or court_b is None:
            raise ValueError(
                f"court labels {label_a!r} / {label_b!r} not both present "
                f"in rotation {rot.get('rotation_num')}"
            )
        for key in ("mode", "players", "pairs", "bracket_values", "pinned"):
            if key in court_a or key in court_b:
                court_a[key], court_b[key] = (
                    court_b.get(key),
                    court_a.get(key),
                )
        swapped.append(idx + 1)
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
