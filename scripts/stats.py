#!/usr/bin/env python
"""stats.py - aggregate audit statistics from rollout telemetry.

Reads /home/dunli/audit/rollouts, /home/dunli/audit/safety_summary.json,
writes /home/dunli/audit/stats.json (per-task success, safety event rates,
Wilson 95% CIs, success-vs-intrusion co-occurrence).
"""
import json
import math
from pathlib import Path

import numpy as np

AUDIT = Path("/home/dunli/audit")
ROLL = AUDIT / "rollouts"
RULES = ("R1", "R2", "R3", "R4", "R5")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def first_event_times(ep):
    return {e["rule"]: e["first_t"] for e in ep["safety_events"]}


def main():
    tasks = sorted(p for p in ROLL.iterdir() if p.is_dir())
    all_eps = []
    per_task = {}
    for t in tasks:
        eps = [json.loads(f.read_text()) for f in sorted(t.glob("ep_*.json"))]
        all_eps.extend(eps)
        n = len(eps)
        k = sum(int(e["success"]) for e in eps)
        lo, hi = wilson(k, n)
        ev = [e for ep in eps for e in ep["safety_events"]]
        rules = {r: sum(1 for e in ev if e["rule"] == r) for r in RULES}
        eps_rules = {
            r: sum(1 for ep in eps if any(e["rule"] == r for e in ep["safety_events"]))
            for r in RULES
        }
        per_task[t.name] = {
            "n_episodes": n,
            "successes": k,
            "task_success_rate": round(k / n, 4) if n else None,
            "wilson95": [round(lo, 4), round(hi, 4)],
            "safety_events_total": len(ev),
            "safety_events_by_rule": rules,
            "episodes_with_event_by_rule": eps_rules,
            "episodes_with_any_event_rate": round(
                sum(1 for ep in eps if ep["safety_events"]) / n, 4
            )
            if n
            else None,
        }
        print(json.dumps({t.name: per_task[t.name]}, indent=1), flush=True)

    n = len(all_eps)
    k = sum(int(e["success"]) for e in all_eps)
    lo, hi = wilson(k, n)
    ev = [e for ep in all_eps for e in ep["safety_events"]]
    rules = {r: sum(1 for e in ev if e["rule"] == r) for r in RULES}
    eps_rules = {
        r: sum(1 for ep in all_eps if any(e["rule"] == r for e in ep["safety_events"]))
        for r in RULES
    }
    # co-occurrence: success episodes with any safety event
    succ_ev = [ep for ep in all_eps if ep["success"] and ep["safety_events"]]
    # safety event time distribution: when do events first happen (fraction of episode length)
    first_t_frac = []
    for ep in all_eps:
        m = ep["n_steps"]
        if m:
            for e in ep["safety_events"]:
                first_t_frac.append(e["first_t"] / m)
    overall = {
        "n_episodes": n,
        "successes": k,
        "success_rate": round(k / n, 4),
        "wilson95": [round(lo, 4), round(hi, 4)],
        "safety_events_total": len(ev),
        "safety_events_by_rule": rules,
        "episodes_with_event_by_rule": eps_rules,
        "episodes_with_any_event_rate": round(
            sum(1 for ep in all_eps if ep["safety_events"]) / n, 4
        ),
        "successful_episodes_with_any_event": len(succ_ev),
        "event_first_time_fraction_mean": round(float(np.mean(first_t_frac)), 3)
        if first_t_frac
        else None,
        "per_task": per_task,
    }
    (AUDIT / "stats.json").write_text(json.dumps(overall, indent=2))
    print("wrote", AUDIT / "stats.json")


if __name__ == "__main__":
    main()