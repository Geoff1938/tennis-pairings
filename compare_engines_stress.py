"""Stress test: 32-player case using players with REAL ratings from the
live Google-Sheet roster.
"""
import json
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(".env")

from pairings import make_plan, _rescore_layout
from pairings_cpsat import make_plan_cpsat
from roster import Roster

players_dict = Roster().all()

rated = [
    (n, info) for n, info in players_dict.items()
    if isinstance(info.get("rating"), int)
]
print(f"Roster has {len(rated)} rated players")

# Pick 32 players sampled across the rating range.
rated.sort(key=lambda nv: nv[1]["rating"])
step = max(1, len(rated) // 32)
sampled = rated[::step][:32]
if len(sampled) < 32:
    sampled = rated[:32]
attendees = [n for n, _ in sampled]
print(f"Selected ratings: {sorted(info['rating'] for _, info in sampled)}")
print(f"Gender mix: "
      f"M={sum(1 for _, info in sampled if info.get('gender') == 'M')}, "
      f"F={sum(1 for _, info in sampled if info.get('gender') == 'F')}, "
      f"?={sum(1 for _, info in sampled if info.get('gender') not in ('M', 'F'))}")

court_labels = [str(i + 1) for i in range(8)]

history_path = "_empty_history.json"
Path(history_path).write_text("[]")

genders = {
    n: str(info.get("gender", "?")).strip().upper() or "?"
    for n, info in players_dict.items()
}

def score(plan):
    layout = [[list(c.players) for c in rot.courts] for rot in plan.rotations]
    modes = [[c.mode for c in rot.courts] for rot in plan.rotations]
    labels = [[c.court_label for c in rot.courts] for rot in plan.rotations]
    sit_outs = [list(rot.sit_outs) for rot in plan.rotations]
    total, per_rot, _ = _rescore_layout(
        layout, rotation_modes=modes, rotation_labels=labels,
        rotation_sit_outs=sit_outs, weekly_pair_penalties={},
        ratings=plan.ratings, genders=genders,
    )
    rules = {}
    for pr in per_rot:
        for it in pr.get("breakdown_items", []):
            rules[it["rule"]] = rules.get(it["rule"], 0) + int(it.get("points") or 0)
    return total, rules

today = date(2026, 5, 28)

print("\n=== Production (5 seeds) ===")
prod_results = []
for k in range(5):
    t0 = time.perf_counter()
    plan = make_plan(
        attendees, players_path=players_dict, history_path=history_path,
        court_labels=court_labels, num_rotations=3, seed=42 + k, today=today,
    )
    s, rules = score(plan)
    wall = time.perf_counter() - t0
    prod_results.append((s, rules, wall))
    print(f"  seed {42+k}: score={s} ({wall:.1f}s) -- {rules}")
prod_best = min(prod_results, key=lambda t: t[0])
print(f"  Best: {prod_best[0]} with rules {prod_best[1]}")

print("\n=== CP-SAT ===")
t0 = time.perf_counter()
plan_cp = make_plan_cpsat(
    attendees=attendees, players_path=players_dict, history_path=history_path,
    court_labels=court_labels, num_rotations=3, today=today,
    time_limit_seconds=60, seed=42,
)
s_cp, rules_cp = score(plan_cp)
print(f"  score={s_cp} ({time.perf_counter()-t0:.1f}s) -- {rules_cp}")
print(f"  CP-SAT internal objective: {plan_cp.metrics.get('objective')}")

Path(history_path).unlink()

print()
print("=== Result ===")
diff = s_cp - prod_best[0]
if diff < 0:
    print(f"CP-SAT wins by {-diff} points")
elif diff > 0:
    print(f"Production wins by {diff} points")
else:
    print("Tie")
