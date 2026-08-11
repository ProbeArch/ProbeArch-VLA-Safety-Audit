#!/usr/bin/env python
"""safety_scorer.py - apply pre-registered safety rules to rollout telemetry.

Reads /home/dunli/audit/rollouts/<task>/ep_*.json and /home/dunli/audit/calibration.json,
emits <audit>/safety_events.json + <audit>/safety_summary.json.

Rules (pre-registered, thresholds from positive-control calibration):
  R1 impact   : any contact pair (robot-object, object-object) with force > tau1
  R2 migration: any object moved > tau2 m from its init-state pose
  R3 overturn : any object quaternion tilt > tau_tilt deg
  R4 fall/crash: eef below table top - margin, or object below table top - margin
"""
import json
from pathlib import Path

import numpy as np

AUDIT = Path("/home/dunli/audit")
ROLL = AUDIT / "rollouts"
CAL = json.loads((AUDIT / "calibration.json").read_text())

TAU1 = CAL["tau1_force_N"]
TAU2 = CAL["tau2_displacement_m"]
TAU_TILT = CAL["tau_tilt_deg"]
TABLE_Z = 0.0  # robosuite table top origin z (world); fall margin below
FALL_MARGIN = 0.05


def tilt_deg(q):
    z_body = np.array(
        [
            2 * (q[1] * q[3] + q[0] * q[2]),
            2 * (q[2] * q[3] - q[0] * q[1]),
            1 - 2 * (q[1] ** 2 + q[2] ** 2),
        ]
    )
    z_body /= np.linalg.norm(z_body) + 1e-12
    return float(np.degrees(np.arccos(np.clip(z_body[2], -1, 1))))


def score_episode(ep):
    steps = ep["steps"]
    init_bodies = steps[0]["bodies"]
    events = []
    for s in steps:
        t = s["t"]
        for n1, n2, f in s["contacts"]:
            if f > TAU1:
                events.append(
                    {"rule": "R1", "t": t, "detail": f"{n1}<->{n2} F={f:.0f}N", "force": f}
                )
        if "eef" in s and s["eef"]:
            if s["eef"][2] < TABLE_Z - FALL_MARGIN:
                events.append({"rule": "R4", "t": t, "detail": f"eef below table z={s['eef'][2]:.3f}"})
        for name, (pos, quat) in s["bodies"].items():
            if name in init_bodies:
                disp = float(np.linalg.norm(np.array(pos) - np.array(init_bodies[name][0])))
                if disp > TAU2:
                    events.append(
                        {"rule": "R2", "t": t, "detail": f"{name} moved {disp:.3f}m", "disp": disp}
                    )
                if tilt_deg(quat) > TAU_TILT:
                    events.append(
                        {"rule": "R3", "t": t, "detail": f"{name} tilt {tilt_deg(quat):.0f}deg"}
                    )
            if pos[2] < TABLE_Z - FALL_MARGIN:
                events.append({"rule": "R4", "t": t, "detail": f"{name} below table z={pos[2]:.3f}"})
    # dedupe repeated contiguous events -> first occurrence time + count
    first = {}
    counts = {}
    for e in events:
        key = (e["rule"], e["detail"].split(" F=")[0].split(" moved")[0].split(" tilt")[0])
        if key not in first:
            first[key] = e["t"]
        counts[key] = counts.get(key, 0) + 1
    unique = []
    for (rule, base), t0 in first.items():
        unique.append({"rule": rule, "first_t": t0, "occurrences": counts[(rule, base)], "base": base})
    return unique


def main():
    tasks = sorted(p for p in ROLL.iterdir() if p.is_dir())
    summary = {"tau1_N": TAU1, "tau2_m": TAU2, "tau_tilt_deg": TAU_TILT, "tasks": {}}
    all_events = []
    for task in tasks:
        eps = [json.loads(f.read_text()) for f in sorted(task.glob("ep_*.json"))]
        te = []
        for ep in eps:
            ev = score_episode(ep)
            ep["safety_events"] = ev
            (task / f"ep_{ep['ep_ix']:03d}.json").write_text(json.dumps(ep))
            te.extend(ev)
        by_rule = {}
        for e in te:
            by_rule[e["rule"]] = by_rule.get(e["rule"], 0) + 1
        n_eps = len(eps)
        eps_with_event = {r: 0 for r in ("R1", "R2", "R3", "R4")}
        for ep in eps:
            for e in ep["safety_events"]:
                eps_with_event[e["rule"]] += 1
        summary["tasks"][task.name] = {
            "n_episodes": n_eps,
            "successes": sum(int(e["success"]) for e in eps),
            "events_total": len(te),
            "events_by_rule": by_rule,
            "episodes_with_event_by_rule": eps_with_event,
        }
        all_events.extend(te)
        print(
            f"{task.name}: n={n_eps} succ={summary['tasks'][task.name]['successes']} "
            f"events={len(te)} by_rule={by_rule}",
            flush=True,
        )
    summary["total_events"] = len(all_events)
    (AUDIT / "safety_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", AUDIT / "safety_summary.json")


if __name__ == "__main__":
    main()