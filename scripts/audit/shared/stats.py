#!/usr/bin/env python
"""stats.py - aggregate audit statistics from rollout telemetry.

Reads scored episodes from $AUDIT_DIR/rollouts (only episodes whose provenance
matches the task's run_manifest.json) plus $AUDIT_DIR/safety_summary.json for
thresholds, and writes $AUDIT_DIR/stats.json (per-task success, safety event
rates, Wilson 95% CIs, success-vs-intrusion co-occurrence). AUDIT_DIR defaults
to ~/audit.

Episodes not covered by the current run manifest (stale telemetry from an
earlier run) are excluded, mirroring safety_scorer.py.
"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

AUDIT = Path(os.environ.get("AUDIT_DIR", str(Path.home() / "audit")))
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


def task_run_id(task_dir):
    """run_id from a task's run_manifest.json, or None when absent/unreadable."""
    manifest_path = task_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    run_id = manifest.get("run_id")
    return run_id if isinstance(run_id, str) else None


def episode_matches_manifest(ep, run_id):
    """An episode is included only when its provenance matches the run manifest."""
    if run_id is None:
        return False
    provenance = ep.get("provenance")
    if not isinstance(provenance, dict):
        return False
    return provenance.get("run_id") == run_id


def main():
    tasks = sorted(p for p in ROLL.iterdir() if p.is_dir())
    all_eps = []
    per_task = {}
    for t in tasks:
        run_id = task_run_id(t)
        eps = []
        for f in sorted(t.glob("ep_*.json")):
            try:
                ep = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if episode_matches_manifest(ep, run_id):
                eps.append(ep)
        unscored = [ep for ep in eps if "safety_events" not in ep]
        if unscored:
            raise RuntimeError(
                f"{t.name}: {len(unscored)} episode(s) not scored; run safety_scorer.py first"
            )
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
        task_aware_events = [
            e for ep in eps for e in ep.get("task_aware_events", [])
        ]
        task_aware_diagnostics = [
            e
            for ep in eps
            for e in (ep.get("task_aware") or {}).get("diagnostic_events", [])
        ]
        task_aware_rules = {}
        for event in task_aware_events:
            task_aware_rules[event["rule"]] = task_aware_rules.get(event["rule"], 0) + 1
        task_aware_outcomes = {}
        for ep in eps:
            outcome = (ep.get("task_aware") or {}).get("outcome")
            if outcome:
                task_aware_outcomes[outcome] = task_aware_outcomes.get(outcome, 0) + 1
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
            "initial_state_violations": sum(
                len(ep.get("initial_state_violations", [])) for ep in eps
            ),
            "task_aware_events_total": len(task_aware_events),
            "task_aware_events_by_rule": task_aware_rules,
            "task_aware_diagnostic_events_total": len(task_aware_diagnostics),
            "task_aware_outcomes": task_aware_outcomes,
            "episodes_with_expected_target_motion": sum(
                bool((ep.get("task_aware") or {}).get("expected_target_motion")) for ep in eps
            ),
            "episodes_with_destination_motion": sum(
                bool((ep.get("task_aware") or {}).get("destination_motion_measurements"))
                for ep in eps
            ),
            "episodes_with_distractor_motion": sum(
                bool((ep.get("task_aware") or {}).get("distractor_motion_measurements")) for ep in eps
            ),
        }
        print(json.dumps({t.name: per_task[t.name]}, indent=1), flush=True)

    n = len(all_eps)
    if n == 0:
        # Explicit guard: never divide by zero on an empty rollout set.
        raise RuntimeError(f"no episode telemetry found under {ROLL}")
    safety_summary = json.loads((AUDIT / "safety_summary.json").read_text())
    thresholds = safety_summary["thresholds"]
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
        m = ep.get("n_steps") or 0
        if m:
            for e in ep["safety_events"]:
                first_t_frac.append(e["first_t"] / m)
    all_task_aware_events = [
        e for ep in all_eps for e in ep.get("task_aware_events", [])
    ]
    all_task_aware_diagnostics = [
        e
        for ep in all_eps
        for e in (ep.get("task_aware") or {}).get("diagnostic_events", [])
    ]
    task_aware_rules = {}
    for event in all_task_aware_events:
        task_aware_rules[event["rule"]] = task_aware_rules.get(event["rule"], 0) + 1
    task_aware_outcomes = {}
    for ep in all_eps:
        outcome = (ep.get("task_aware") or {}).get("outcome")
        if outcome:
            task_aware_outcomes[outcome] = task_aware_outcomes.get(outcome, 0) + 1
    overall = {
        "thresholds": thresholds,
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
        "task_aware": {
            "events_total": len(all_task_aware_events),
            "events_by_rule": task_aware_rules,
            "diagnostic_events_total": len(all_task_aware_diagnostics),
            "diagnostic_events_by_rule": {
                rule: sum(
                    1
                    for event in all_task_aware_diagnostics
                    if event.get("rule") == rule
                )
                for rule in sorted(
                    {event.get("rule") for event in all_task_aware_diagnostics}
                )
                if rule
            },
            "outcomes": task_aware_outcomes,
            "episodes_with_expected_target_motion": sum(
                bool((ep.get("task_aware") or {}).get("expected_target_motion")) for ep in all_eps
            ),
            "episodes_with_destination_motion": sum(
                bool((ep.get("task_aware") or {}).get("destination_motion_measurements"))
                for ep in all_eps
            ),
            "episodes_with_distractor_motion": sum(
                bool((ep.get("task_aware") or {}).get("distractor_motion_measurements")) for ep in all_eps
            ),
            "hazard_assessment": "not_assessed",
        },
        "per_task": per_task,
    }
    (AUDIT / "stats.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print("wrote", AUDIT / "stats.json")


def _self_test():
    """Synthetic unit tests; plain python, no gymnasium/lerobot/mujoco."""
    with tempfile.TemporaryDirectory(prefix="probearch-stats-selftest-") as tmp:
        root = Path(tmp)
        roll = root / "rollouts"

        # Task with one scored episode covered by its run manifest.
        task = roll / "libero_spatial_0"
        task.mkdir(parents=True)
        ep = {
            "ep_ix": 0,
            "success": True,
            "n_steps": 10,
            "safety_events": [{"rule": "R1", "first_t": 2}],
            "initial_state_violations": [],
            "provenance": {"run_id": "run-1"},
        }
        (task / "ep_000.json").write_text(json.dumps(ep))
        (task / "run_manifest.json").write_text(json.dumps({"run_id": "run-1"}))

        # Empty task directory: per-task rates must be None, not ZeroDivisionError.
        (roll / "libero_spatial_1").mkdir(parents=True)

        # Stale episodes (different run_id): must be excluded from aggregates.
        stale = roll / "libero_spatial_2"
        stale.mkdir(parents=True)
        (stale / "ep_000.json").write_text(
            json.dumps(
                {
                    "ep_ix": 0,
                    "success": True,
                    "n_steps": 5,
                    "safety_events": [{"rule": "R4", "first_t": 1}],
                    "initial_state_violations": [],
                    "provenance": {"run_id": "stale-run"},
                }
            )
        )
        (stale / "run_manifest.json").write_text(json.dumps({"run_id": "run-2"}))

        (root / "safety_summary.json").write_text(
            json.dumps(
                {
                    "thresholds": {
                        "tau1_force_N": 1.0,
                        "tau2_displacement_m": 0.1,
                        "tau_tilt_deg": 45.0,
                        "fall_margin_m": 0.1,
                    }
                }
            )
        )
        old_audit, old_roll = AUDIT, ROLL
        globals()["AUDIT"] = root
        globals()["ROLL"] = roll
        try:
            main()
        finally:
            globals()["AUDIT"], globals()["ROLL"] = old_audit, old_roll
        out = json.loads((root / "stats.json").read_text())
        assert out["n_episodes"] == 1, out  # stale episode excluded
        assert out["success_rate"] == 1.0
        assert out["safety_events_by_rule"] == {"R1": 1, "R2": 0, "R3": 0, "R4": 0, "R5": 0}
        assert out["per_task"]["libero_spatial_0"]["task_success_rate"] == 1.0
        assert out["per_task"]["libero_spatial_1"]["task_success_rate"] is None
        assert out["per_task"]["libero_spatial_1"]["wilson95"] == [0.0, 0.0]
        assert "libero_spatial_2" not in out["per_task"] or (
            out["per_task"]["libero_spatial_2"]["n_episodes"] == 0
        )

        # Empty rollout set raises RuntimeError (guarded), not ZeroDivisionError.
        empty = root / "empty_rollouts"
        empty.mkdir(parents=True)
        globals()["ROLL"] = empty
        try:
            try:
                main()
                raise AssertionError("expected RuntimeError on empty rollouts")
            except RuntimeError:
                pass
        finally:
            globals()["ROLL"] = roll

        # Unscored episodes fail loudly with a clear message.
        unscored_dir = roll / "libero_spatial_3"
        unscored_dir.mkdir(parents=True)
        (unscored_dir / "ep_000.json").write_text(
            json.dumps(
                {
                    "ep_ix": 0,
                    "success": True,
                    "n_steps": 5,
                    "provenance": {"run_id": "run-1"},
                }
            )
        )
        (unscored_dir / "run_manifest.json").write_text(json.dumps({"run_id": "run-1"}))
        try:
            try:
                main()
                raise AssertionError("expected RuntimeError on unscored episodes")
            except RuntimeError as exc:
                assert "not scored" in str(exc)
        finally:
            (unscored_dir / "ep_000.json").unlink()
            (unscored_dir / "run_manifest.json").unlink()

    print("stats self-test passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(_self_test())
    main()
