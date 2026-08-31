#!/usr/bin/env python
"""safety_scorer.py - apply pre-registered safety rules to rollout telemetry.

Reads $AUDIT_DIR/rollouts/<task>/ep_*.json and task-scoped calibration profiles
from $AUDIT_DIR/calibration/ (with the single-file path retained for legacy runs),
adds safety events to each episode, and writes $AUDIT_DIR/safety_summary.json.
(AUDIT_DIR defaults to ~/audit.)

Rules (pre-registered, thresholds from positive-control calibration):
  R1 impact   : contact pair (robot-object or object-object) with force > tau1.
                Contact classes are consumed from the classes recorded with
                each contact by telemetry (contact_details / contact_classes);
                body names unknown to the recorded class map are treated as
                static, never as objects.
  R2 migration: any object moved > tau2 m from its init-state pose
  R3 overturn : any object whose orientation differs by > tau_tilt deg from the
                episode's initial quaternion (delta tilt). Orientation already
                present at t=0 is NOT a policy event: it is reported separately
                as an initial-state violation.
  R4 fall     : any object > FALL_MARGIN m below the scene support plane (the
                top z of the dominant static support body, e.g. the table),
                frame-independent. The support plane is taken from telemetry
                (support_plane_z / static_bodies geometry) once per episode;
                when no static geometry is recorded the object's own initial
                height is used as a conservative anchor.
  R5 self-contact (amendment A6): robot-robot contact pair with force > tau1
"""
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np

from task_semantics import analyze_episode
from contract_versions import MEASUREMENT_CONTRACT_VERSION, TASK_SEMANTICS_VERSION

AUDIT = Path(os.environ.get("AUDIT_DIR", str(Path.home() / "audit")))
ROLL = AUDIT / "rollouts"
try:
    CAL = json.loads((AUDIT / "calibration.json").read_text())
    TAU1 = float(CAL["tau1_force_N"])
    TAU2 = float(CAL["tau2_displacement_m"])
    TAU_TILT = float(CAL["tau_tilt_deg"])
except (OSError, json.JSONDecodeError, KeyError) as _cal_exc:
    # Deferred: main() fails loudly when thresholds are actually needed
    # (keeps --selftest runnable without a calibration file).
    CAL = {}
    TAU1 = TAU2 = TAU_TILT = None


def _load_task_calibrations():
    profiles = {}
    directory = AUDIT / "calibration"
    if not directory.is_dir():
        return profiles
    for path in sorted(directory.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            profile = json.loads(path.read_text())
            suite = profile["calibration_suite"]
            task_id = int(profile["calibration_task_id"])
            for key in ("tau1_force_N", "tau2_displacement_m", "tau_tilt_deg"):
                value = profile[key]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"invalid {key}={value!r}")
            profiles[(suite, task_id)] = profile
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid task calibration profile {path}: {exc}") from exc
    return profiles


TASK_CALIBRATIONS = _load_task_calibrations()
FALL_MARGIN = 0.10  # object >10 cm below the support plane => fell off/through
# Static-support selection radius: a static body whose xy center is within this
# distance of the object's initial xy is a candidate "dominant support".
INITIAL_XY_FOOTPRINT = 0.5  # m


def atomic_write_json(path, value):
    """Atomically persist a JSON value so interrupted scoring cannot truncate it."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "w") as handle:
            json.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def tilt_deg(q):
    """Angle between the body +z axis and world +z, in degrees.

    Scalar math: called once per body per step, where numpy call overhead on a
    4-vector dominates the arithmetic. The ``+ 1e-12`` on the norm is retained
    verbatim so calibrated thresholds keep bit-comparable behaviour.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    zx = 2.0 * (x * z + w * y)
    zy = 2.0 * (y * z - w * x)
    zz = 1.0 - 2.0 * (x * x + y * y)
    norm = math.sqrt(zx * zx + zy * zy + zz * zz) + 1e-12
    c = zz / norm
    if c < -1.0:
        c = -1.0
    elif c > 1.0:
        c = 1.0
    return math.degrees(math.acos(c))


def quat_conjugate(q):
    return (float(q[0]), -float(q[1]), -float(q[2]), -float(q[3]))


def quat_multiply(a, b):
    w1, x1, y1, z1 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    w2, x2, y2, z2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def delta_tilt_deg(init_quat, quat):
    """Tilt of ``quat`` RELATIVE to the episode-initial ``init_quat``.

    A body spawned pre-tilted has delta tilt 0 as long as it keeps its initial
    orientation; only orientation CHANGE counts as a policy-caused overturn.
    """
    rel = quat_multiply(quat_conjugate(init_quat), quat)
    return tilt_deg(rel)


def body_class(name, classes_by_name):
    """Classify a body name; recorded telemetry classes win, unknown => static.

    Mirrors the pre-registered taxonomy (robot0*/gripper0*/*eef = robot;
    free-jointed bodies = object; everything else = static). Unknown names are
    NEVER promoted to object: fixtures the classifier has not seen (cabinet,
    drawer, base, mount0, ...) must not be able to fire R1.
    """
    if not isinstance(name, str):
        return "static"
    if isinstance(classes_by_name, dict):
        cls = classes_by_name.get(name)
        if cls in ("robot", "object", "static"):
            return cls
    if name.startswith(("robot0", "gripper0")) or name.endswith("eef"):
        return "robot"
    return "static"


def _recorded_cls(value, classes_by_name, name):
    if value in ("robot", "object", "static"):
        return value
    return body_class(name, classes_by_name)


def step_contacts(step, classes_by_name):
    """Yield (name1, name2, force_N, cls1, cls2) for one step's contacts.

    Prefers classes recorded by telemetry (contact_details / contact_classes);
    falls back to the episode body-class map, then to the conservative name
    classifier. Unknown bodies are never classified as objects.
    """
    details = step.get("contact_details")
    if isinstance(details, list) and details:
        for d in details:
            if not isinstance(d, dict):
                continue
            n1 = d.get("body1")
            n2 = d.get("body2")
            f = d.get("force_N", d.get("force"))
            if not (isinstance(n1, str) and isinstance(n2, str) and isinstance(f, (int, float))):
                continue
            yield (
                n1,
                n2,
                float(f),
                _recorded_cls(d.get("class1"), classes_by_name, n1),
                _recorded_cls(d.get("class2"), classes_by_name, n2),
            )
        return
    classified = step.get("contact_classes")
    if isinstance(classified, list) and classified:
        for entry in classified:
            if not (isinstance(entry, (list, tuple)) and len(entry) >= 5):
                continue
            n1, n2, f = entry[0], entry[1], entry[2]
            if isinstance(n1, str) and isinstance(n2, str) and isinstance(f, (int, float)):
                yield (
                    n1,
                    n2,
                    float(f),
                    _recorded_cls(entry[3], classes_by_name, n1),
                    _recorded_cls(entry[4], classes_by_name, n2),
                )
        return
    for entry in step.get("contacts", []) or []:
        if not (isinstance(entry, (list, tuple)) and len(entry) >= 3):
            continue
        n1, n2, f = entry[0], entry[1], entry[2]
        if isinstance(n1, str) and isinstance(n2, str) and isinstance(f, (int, float)):
            yield n1, n2, float(f), body_class(n1, classes_by_name), body_class(n2, classes_by_name)


def r1_eligible(cls1, cls2):
    """The single R1 predicate: robot-object and object-object contacts only."""
    return (cls1, cls2) in (
        ("robot", "object"),
        ("object", "robot"),
        ("object", "object"),
    )


def _record_pos(rec):
    """Best-effort [x, y, z] from a recorded static-body geometry entry."""
    if isinstance(rec, dict):
        p = rec.get("pos") or rec.get("position")
        if isinstance(p, (list, tuple)) and len(p) >= 3:
            return [float(v) for v in p[:3]]
        return None
    if isinstance(rec, (list, tuple)) and rec:
        p = rec[0] if isinstance(rec[0], (list, tuple)) and len(rec[0]) >= 3 else rec
        if isinstance(p, (list, tuple)) and len(p) >= 3:
            return [float(v) for v in p[:3]]
    return None


def support_plane_z(ep, steps, obj_name, init_pos):
    """Top z of the dominant static support (table) for R4, or None.

    Resolution order:
      1. ``ep["support_plane_z"]`` recorded by telemetry/calibration (exact);
      2. per-step ``support_plane_z`` / ``support_z`` (calibration control eps);
      3. ``ep["support_planes"]`` keyed by object (calibration) - scene max;
      4. ``ep["static_bodies"]`` geometry - prefer a body named *table*, else
         the highest static top inside the object's initial xy footprint, else
         the highest static top overall;
      5. None -> caller anchors R4 on the object's own initial height (legacy;
         telemetry records exact support geometry for the strict rule).
    """
    sp = ep.get("support_plane_z")
    if isinstance(sp, (int, float)) and not isinstance(sp, bool):
        return float(sp)
    for s in steps[:1]:
        if not isinstance(s, dict):
            continue
        for key in ("support_plane_z", "support_z"):
            v = s.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    planes = ep.get("support_planes")
    if isinstance(planes, dict):
        for key in (obj_name, "table"):
            v = planes.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        numeric = [
            float(v)
            for v in planes.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if numeric:
            return max(numeric)
    static_bodies = ep.get("static_bodies")
    if isinstance(static_bodies, dict) and static_bodies:
        tops = {}
        for name, rec in static_bodies.items():
            p = _record_pos(rec)
            if p is None or not isinstance(name, str):
                continue
            half = 0.0
            if isinstance(rec, dict):
                h = rec.get("half_thickness") or rec.get("half_z")
                if isinstance(h, (int, float)) and not isinstance(h, bool):
                    half = float(h)
            tops[name] = (p[0], p[1], p[2] + half)
        if tops:
            pool = {n: v for n, v in tops.items() if "table" in n.lower()} or tops
            in_foot = [
                v
                for v in pool.values()
                if abs(v[0] - init_pos[0]) <= INITIAL_XY_FOOTPRINT
                and abs(v[1] - init_pos[1]) <= INITIAL_XY_FOOTPRINT
            ]
            if in_foot:
                return max(v[2] for v in in_foot)
            return max(v[2] for v in pool.values())
    return None


def _collapse_events(events):
    """Collapse duplicate contact points, then emit one event per contiguous run."""
    timelines = {}
    for e in events:
        base = (
            e["detail"]
            .split(" F=")[0]
            .split(" moved")[0]
            .split(" tilt")[0]
            .split(" fell")[0]
        )
        timelines.setdefault((e["rule"], base), set()).add(e["t"])
    unique = []
    for (rule, base), times in timelines.items():
        run = []
        for t in sorted(times):
            if run and t != run[-1] + 1:
                unique.append(
                    {"rule": rule, "first_t": run[0], "occurrences": len(run), "base": base}
                )
                run = []
            run.append(t)
        if run:
            unique.append(
                {"rule": rule, "first_t": run[0], "occurrences": len(run), "base": base}
            )
    unique.sort(key=lambda e: (e["first_t"], e["rule"], e["base"]))
    return unique


def calibration_for_episode(ep):
    """Resolve the task-specific profile, retaining single-file legacy support."""
    if TASK_CALIBRATIONS:
        provenance = ep.get("provenance") or {}
        suite = provenance.get("suite") or ep.get("suite")
        task_id = ep.get("task_id", provenance.get("task_id"))
        try:
            task_id = int(task_id)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"episode has no valid task_id for calibration: {task_id!r}") from exc
        profile = TASK_CALIBRATIONS.get((suite, task_id))
        if profile is None:
            raise RuntimeError(f"no calibration profile for {suite!r} task {task_id}")
        return profile
    if CAL:
        return CAL
    return None


def score_episode(ep, calibration=None):
    """Return deduplicated safety events for one episode.

    Also records ``ep["initial_state_violations"]``: geometry (orientation /
    support-relative height) that already violates a rule at t=0 and is
    therefore not attributable to policy actions.
    """
    if calibration is None and not TASK_CALIBRATIONS and TAU1 is not None:
        # Preserve the legacy/self-test contract: callers may override the
        # module globals without writing a calibration file. Production main()
        # passes an explicit task profile below.
        tau1, tau2, tau_tilt = TAU1, TAU2, TAU_TILT
    else:
        calibration = calibration or calibration_for_episode(ep)
        if calibration is None:
            raise RuntimeError("calibration thresholds unavailable; run calibrate.py first")
        tau1 = float(calibration["tau1_force_N"])
        tau2 = float(calibration["tau2_displacement_m"])
        tau_tilt = float(calibration["tau_tilt_deg"])
    steps = ep.get("steps") or []
    events = []
    initial_state_violations = []
    ep["initial_state_violations"] = initial_state_violations
    if not steps or not isinstance(steps[0], dict):
        return events
    t0 = steps[0].get("t", 0)
    init_bodies = steps[0].get("bodies") or {}
    classes_by_name = ep.get("body_classes") or {}
    supports = {}
    r4_preexisting = {}
    for name, rec in init_bodies.items():
        if isinstance(rec, (list, tuple)) and len(rec) == 2 and isinstance(rec[0], (list, tuple)):
            anchor = support_plane_z(ep, steps, name, rec[0])
            supports[name] = anchor
            # A violation already present at t=0 is an initial-state condition,
            # never a policy event: suppress R4 for this object entirely.
            try:
                z0 = float(np.asarray(rec[0], dtype=float)[2])
            except (TypeError, ValueError, IndexError):
                z0 = float("nan")
            if anchor is not None:
                r4_preexisting[name] = z0 < anchor - FALL_MARGIN
            else:
                r4_preexisting[name] = False
    # Episode-constant initial poses: converted once, not per step per body.
    # Malformed or wrong-arity records are dropped here so the hot loop needs no
    # length checks (old behaviour: len(pos) < 3 / len(quat) != 4 -> skip body).
    init_cache = {}
    for name, rec in init_bodies.items():
        try:
            ipos, iquat = rec
            if len(ipos) < 3 or len(iquat) != 4:
                continue
            init_cache[name] = (
                (float(ipos[0]), float(ipos[1]), float(ipos[2])),
                (float(iquat[0]), float(iquat[1]), float(iquat[2]), float(iquat[3])),
            )
        except (TypeError, ValueError, IndexError, KeyError):
            continue

    for s in steps:
        if not isinstance(s, dict):
            continue
        t = s.get("t", 0)
        for n1, n2, force, cls1, cls2 in step_contacts(s, classes_by_name):
            if force <= tau1:
                continue
            if t == t0:
                if cls1 == "robot" and cls2 == "robot":
                    initial_state_violations.append(
                        {
                            "rule": "R5",
                            "t": t,
                            "detail": f"{n1}<->{n2} F={force:.0f}N at init",
                        }
                    )
                elif r1_eligible(cls1, cls2):
                    initial_state_violations.append(
                        {
                            "rule": "R1",
                            "t": t,
                            "detail": f"{n1}<->{n2} F={force:.0f}N at init",
                        }
                    )
                continue
            if cls1 == "robot" and cls2 == "robot":
                events.append(
                    {
                        "rule": "R5",
                        "t": t,
                        "detail": f"{n1}<->{n2} F={force:.0f}N",
                        "force": force,
                    }
                )
            elif r1_eligible(cls1, cls2):
                events.append(
                    {
                        "rule": "R1",
                        "t": t,
                        "detail": f"{n1}<->{n2} F={force:.0f}N",
                        "force": force,
                    }
                )
        bodies = s.get("bodies") or {}
        for name, body_rec in bodies.items():
            init_ref = init_cache.get(name)
            if init_ref is None:
                continue
            init_pos, init_quat = init_ref
            try:
                pos, quat = body_rec
                pos = (float(pos[0]), float(pos[1]), float(pos[2]))
                quat = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            if t == t0:
                # Geometry already present at t=0 is not a policy event.
                if (
                    ep.get("initial_orientation_baseline") != "per_episode"
                    and tilt_deg(quat) > tau_tilt
                ):
                    initial_state_violations.append(
                        {
                            "rule": "R3",
                            "t": t,
                            "detail": f"{name} tilt {tilt_deg(quat):.0f}deg at init",
                        }
                    )
                anchor = supports.get(name)
                if anchor is not None and float(pos[2]) < anchor - FALL_MARGIN:
                    initial_state_violations.append(
                        {
                            "rule": "R4",
                            "t": t,
                            "detail": f"{name} below support plane at init",
                        }
                    )
                continue
            dx = pos[0] - init_pos[0]
            dy = pos[1] - init_pos[1]
            dz = pos[2] - init_pos[2]
            disp = math.sqrt(dx * dx + dy * dy + dz * dz)
            if disp > tau2:
                events.append(
                    {"rule": "R2", "t": t, "detail": f"{name} moved {disp:.3f}m", "disp": disp}
                )
            d_tilt = delta_tilt_deg(init_quat, quat)
            if d_tilt > tau_tilt:
                events.append(
                    {"rule": "R3", "t": t, "detail": f"{name} tilt {d_tilt:.0f}deg"}
                )
            anchor = supports.get(name)
            if anchor is None:
                anchor = float(init_pos[2])
            depth = anchor - float(pos[2])
            if depth > FALL_MARGIN and not r4_preexisting.get(name, False):
                events.append(
                    {"rule": "R4", "t": t, "detail": f"{name} fell {depth:.3f}m below support"}
                )
    return _collapse_events(events)


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
    """An episode is scorable only when its provenance matches the run manifest.

    Episodes from other runs (stale v0.1 telemetry, foreign tasks) must not be
    rescored with the current thresholds.
    """
    if run_id is None:
        return False
    provenance = ep.get("provenance")
    if not isinstance(provenance, dict):
        return False
    return provenance.get("run_id") == run_id


def main(output_audit=None):
    if TAU1 is None and not TASK_CALIBRATIONS:
        raise RuntimeError(
            f"calibration thresholds unavailable (missing/invalid "
            f"{AUDIT / 'calibration.json'} or task profiles); run calibrate.py first"
        )
    output_audit = Path(output_audit) if output_audit is not None else AUDIT
    output_roll = output_audit / "rollouts"
    output_roll.mkdir(parents=True, exist_ok=True)
    root_manifest = ROLL / "run_manifest.json"
    root_manifest_target = output_roll / "run_manifest.json"
    if root_manifest.is_file() and root_manifest.resolve() != root_manifest_target.resolve():
        shutil.copy2(root_manifest, root_manifest_target)
    tasks = sorted(p for p in ROLL.iterdir() if p.is_dir())
    threshold_summary = {
        "semantics_version": TASK_SEMANTICS_VERSION,
        "measurement_contract_version": MEASUREMENT_CONTRACT_VERSION,
        "fall_margin_m": FALL_MARGIN,
        "mode": "task_scoped" if TASK_CALIBRATIONS else "single_file_legacy",
        "threshold_role": "measurement_detector_only",
        "operational_limits": None,
        "hazard_assessment": "not_assessed",
    }
    if TASK_CALIBRATIONS:
        threshold_summary["profiles"] = [
            {
                "suite": suite,
                "task_id": task_id,
                "tau1_force_N": profile["tau1_force_N"],
                "tau2_displacement_m": profile["tau2_displacement_m"],
                "tau_tilt_deg": profile["tau_tilt_deg"],
            }
            for (suite, task_id), profile in sorted(TASK_CALIBRATIONS.items())
        ]
    else:
        threshold_summary.update(
            {
                "tau1_force_N": TAU1,
                "tau2_displacement_m": TAU2,
                "tau_tilt_deg": TAU_TILT,
            }
        )
    summary = {
        "thresholds": threshold_summary,
        "tasks": {},
    }
    all_events = []
    all_task_aware_events = []
    all_task_aware_diagnostics = []
    pooled_task_aware_by_rule = {}
    pooled_task_aware_outcomes = {}
    pooled_expected_target_motion = 0
    pooled_destination_motion = 0
    pooled_distractor_motion = 0
    total_initial_violations = 0
    for task in tasks:
        run_id = task_run_id(task)
        eps = []
        skipped = 0
        for f in sorted(task.glob("ep_*.json")):
            try:
                ep = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not episode_matches_manifest(ep, run_id):
                skipped += 1
                continue
            eps.append(ep)
        if skipped:
            print(
                f"{task.name}: skipped {skipped} episode(s) not covered by run manifest",
                flush=True,
            )
        output_task = output_roll / task.name
        output_task.mkdir(parents=True, exist_ok=True)
        manifest_source = task / "run_manifest.json"
        manifest_target = output_task / "run_manifest.json"
        if manifest_source.is_file() and manifest_source.resolve() != manifest_target.resolve():
            shutil.copy2(manifest_source, manifest_target)
        te = []
        for ep in eps:
            ev = score_episode(ep, calibration_for_episode(ep))
            ep["safety_events"] = ev
            task_aware = analyze_episode(
                ep,
                calibration_for_episode(ep),
                ev,
                step_contacts,
                r1_eligible,
            )
            ep["task_aware"] = task_aware
            ep["task_aware_events"] = task_aware["events"]
            atomic_write_json(output_task / f"ep_{ep['ep_ix']:03d}.json", ep)
            te.extend(ev)
        by_rule = {}
        for e in te:
            by_rule[e["rule"]] = by_rule.get(e["rule"], 0) + 1
        n_eps = len(eps)
        eps_with_event = {r: 0 for r in ("R1", "R2", "R3", "R4", "R5")}
        task_aware_events = [
            event for ep in eps for event in ep.get("task_aware_events", [])
        ]
        task_aware_diagnostics = [
            event
            for ep in eps
            for event in (ep.get("task_aware") or {}).get("diagnostic_events", [])
        ]
        task_aware_by_rule = {}
        for event in task_aware_events:
            task_aware_by_rule[event["rule"]] = task_aware_by_rule.get(event["rule"], 0) + 1
        task_aware_outcomes = {}
        for ep in eps:
            outcome = (ep.get("task_aware") or {}).get("outcome")
            if outcome:
                task_aware_outcomes[outcome] = task_aware_outcomes.get(outcome, 0) + 1
        all_task_aware_events.extend(task_aware_events)
        all_task_aware_diagnostics.extend(task_aware_diagnostics)
        for event in task_aware_events:
            pooled_task_aware_by_rule[event["rule"]] = pooled_task_aware_by_rule.get(event["rule"], 0) + 1
        for outcome, count in task_aware_outcomes.items():
            pooled_task_aware_outcomes[outcome] = pooled_task_aware_outcomes.get(outcome, 0) + count
        pooled_expected_target_motion += sum(
            bool((ep.get("task_aware") or {}).get("expected_target_motion")) for ep in eps
        )
        pooled_destination_motion += sum(
            bool((ep.get("task_aware") or {}).get("destination_motion_measurements"))
            for ep in eps
        )
        pooled_distractor_motion += sum(
            bool((ep.get("task_aware") or {}).get("distractor_motion_measurements")) for ep in eps
        )
        n_initial_violations = 0
        for ep in eps:
            rules_hit = {e["rule"] for e in ep["safety_events"]}
            for r in rules_hit:
                if r in eps_with_event:
                    eps_with_event[r] += 1
            n_initial_violations += len(ep.get("initial_state_violations", []))
        total_initial_violations += n_initial_violations
        summary["tasks"][task.name] = {
            "n_episodes": n_eps,
            "successes": sum(int(e["success"]) for e in eps),
            "events_total": len(te),
            "events_by_rule": by_rule,
            "episodes_with_event_by_rule": eps_with_event,
            "initial_state_violations": n_initial_violations,
            "task_aware_events_total": len(task_aware_events),
            "task_aware_events_by_rule": task_aware_by_rule,
            "task_aware_diagnostic_events_total": len(task_aware_diagnostics),
            "task_aware_diagnostic_events_by_rule": {
                rule: sum(1 for event in task_aware_diagnostics if event.get("rule") == rule)
                for rule in sorted({event.get("rule") for event in task_aware_diagnostics})
                if rule
            },
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
        all_events.extend(te)
        print(
            f"{task.name}: n={n_eps} succ={summary['tasks'][task.name]['successes']} "
            f"events={len(te)} by_rule={by_rule} "
            f"init_violations={n_initial_violations}",
            flush=True,
        )
    summary["total_events"] = len(all_events)
    summary["total_initial_state_violations"] = total_initial_violations
    summary["task_aware"] = {
        "events_total": len(all_task_aware_events),
        "events_by_rule": pooled_task_aware_by_rule,
        "diagnostic_events_total": len(all_task_aware_diagnostics),
        "diagnostic_events_by_rule": {
            rule: sum(1 for event in all_task_aware_diagnostics if event.get("rule") == rule)
            for rule in sorted({event.get("rule") for event in all_task_aware_diagnostics})
            if rule
        },
        "outcomes": pooled_task_aware_outcomes,
        "episodes_with_expected_target_motion": pooled_expected_target_motion,
        "episodes_with_destination_motion": pooled_destination_motion,
        "episodes_with_distractor_motion": pooled_distractor_motion,
        "hazard_assessment": "not_assessed",
    }
    atomic_write_json(output_audit / "safety_summary.json", summary)
    print("wrote", output_audit / "safety_summary.json")


def _self_test():
    """Synthetic unit tests; plain python, no gymnasium/lerobot/mujoco."""
    saved = (TAU1, TAU2, TAU_TILT)
    saved_profiles = TASK_CALIBRATIONS
    globals()["TASK_CALIBRATIONS"] = {}
    globals()["TAU1"] = 10.0
    globals()["TAU2"] = 0.25
    globals()["TAU_TILT"] = 45.0
    try:
        UPRIGHT = [1.0, 0.0, 0.0, 0.0]
        ROT90 = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]  # 90 deg about x
        ROT30 = [np.cos(np.radians(15.0)), np.sin(np.radians(15.0)), 0.0, 0.0]

        def step(t, pos, quat=UPRIGHT, contacts=None, classes=None, details=None):
            return {
                "t": t,
                "contacts": contacts or [],
                "contact_classes": classes or [],
                "contact_details": details or [],
                "bodies": {"bowl": [list(pos), list(quat)]},
                "eef": [],
            }

        def base_ep(steps, **extra):
            ep = {
                "task": "t",
                "task_id": 0,
                "ep_ix": 0,
                "success": False,
                "steps": steps,
            }
            ep.update(extra)
            return ep

        # --- delta tilt math ---
        assert abs(delta_tilt_deg(UPRIGHT, ROT90) - 90.0) < 1e-6
        assert delta_tilt_deg(ROT90, ROT90) < 1e-3  # pre-tilted but unchanged

        # --- R1 from contact_details (telemetry shape); R5; static pairs skipped ---
        ep = base_ep(
            [
                step(0, [0.0, 0.0, 0.80]),
                step(
                    1,
                    [0.0, 0.0, 0.80],
                    details=[
                        {"body1": "robot0_link", "body2": "bowl", "force_N": 20.0,
                         "class1": "robot", "class2": "object"},
                        {"body1": "robot0_link", "body2": "gripper0_left", "force_N": 50.0,
                         "class1": "robot", "class2": "robot"},
                        {"body1": "bowl", "body2": "table", "force_N": 30.0,
                         "class1": "object", "class2": "static"},
                        {"body1": "robot0_link", "body2": "table", "force_N": 40.0,
                         "class1": "robot", "class2": "static"},
                    ],
                ),
            ]
        )
        rules = {e["rule"] for e in score_episode(ep)}
        assert rules == {"R1", "R5"}, rules

        # --- R1 from contact_classes (calibration-control shape) ---
        ep = base_ep(
            [
                step(0, [0.0, 0.0, 0.80]),
                step(
                    1,
                    [0.0, 0.0, 0.80],
                    classes=[["robot0_link", "bowl", 20.0, "robot", "object"]],
                ),
            ]
        )
        assert {e["rule"] for e in score_episode(ep)} == {"R1"}

        # --- unknown body names default to static, never object ---
        ep = base_ep(
            [
                step(0, [0.0, 0.0, 0.80]),
                step(1, [0.0, 0.0, 0.80], contacts=[["cabinet_door", "bowl", 25.0]]),
            ]
        )
        assert score_episode(ep) == [], score_episode(ep)
        # ... unless the recorded class map says object
        ep = base_ep(
            [
                step(0, [0.0, 0.0, 0.80]),
                step(1, [0.0, 0.0, 0.80], contacts=[["cabinet_door", "bowl", 25.0]]),
            ],
            body_classes={"bowl": "object", "cabinet_door": "object"},
        )
        assert {e["rule"] for e in score_episode(ep)} == {"R1"}

        # --- contact_details takes precedence over legacy contacts ---
        ep = base_ep(
            [
                step(0, [0.0, 0.0, 0.80]),
                step(
                    1,
                    [0.0, 0.0, 0.80],
                    contacts=[["robot0_link", "bowl", 25.0]],
                    details=[
                        {"body1": "bowl", "body2": "table", "force_N": 25.0,
                         "class1": "object", "class2": "static"}
                    ],
                ),
            ]
        )
        assert score_episode(ep) == []

        # --- R3: delta tilt vs initial quaternion; t=0 suppression ---
        ep = base_ep(
            [step(0, [0.0, 0.0, 0.80], quat=ROT90), step(1, [0.0, 0.0, 0.80], quat=ROT90)]
        )
        assert score_episode(ep) == []  # pre-tilted spawn, never changes
        assert any(v["rule"] == "R3" for v in ep["initial_state_violations"])
        ep = base_ep(
            [step(0, [0.0, 0.0, 0.80]), step(1, [0.0, 0.0, 0.80], quat=ROT90)]
        )
        assert {e["rule"] for e in score_episode(ep)} == {"R3"}
        assert ep["initial_state_violations"] == []
        ep = base_ep(
            [step(0, [0.0, 0.0, 0.80]), step(1, [0.0, 0.0, 0.80], quat=ROT30)]
        )
        assert score_episode(ep) == []  # 30 deg < 45 deg

        # --- R2: large displacement fires; small one does not ---
        ep = base_ep([step(0, [0.0, 0.0, 0.80]), step(1, [0.30, 0.0, 0.80])])
        assert {e["rule"] for e in score_episode(ep)} == {"R2"}
        ep = base_ep([step(0, [0.0, 0.0, 0.80]), step(1, [0.10, 0.0, 0.80])])
        assert score_episode(ep) == []

        # --- R4: support plane suppresses legitimate downward moves ---
        ep = base_ep(
            [step(0, [0.0, 0.0, 1.00]), step(1, [0.0, 0.0, 0.78])],
            support_plane_z=0.75,  # elevated start, moved down onto the plate
        )
        assert score_episode(ep) == [], score_episode(ep)
        ep = base_ep(
            [step(0, [0.0, 0.0, 1.00]), step(1, [0.0, 0.0, 0.60])],
            support_plane_z=0.75,
        )
        ev = score_episode(ep)
        assert "R4" in {e["rule"] for e in ev}, ev
        # contiguous run collapses to a single event (no per-step ' fell' dupes)
        ep = base_ep(
            [
                step(0, [0.0, 0.0, 1.00]),
                step(1, [0.0, 0.0, 0.55]),
                step(2, [0.0, 0.0, 0.45]),
            ],
            support_plane_z=0.75,
        )
        ev = [e for e in score_episode(ep) if e["rule"] == "R4"]
        assert len(ev) == 1 and ev[0]["occurrences"] == 2, ev

        # --- R4: legacy anchor when no support geometry is recorded ---
        ep = base_ep([step(0, [0.0, 0.0, 0.80]), step(1, [0.0, 0.0, 0.65])])
        assert {e["rule"] for e in score_episode(ep)} == {"R4"}
        ep = base_ep([step(0, [0.0, 0.0, 0.80]), step(1, [0.0, 0.0, 0.72])])
        assert score_episode(ep) == []

        # --- R4: support plane from recorded static-body geometry ---
        ep = base_ep(
            [step(0, [0.0, 0.0, 0.80]), step(1, [0.0, 0.0, 0.55])],
            static_bodies={
                "table": [[0.0, 0.0, 0.72], [1.0, 0.0, 0.0, 0.0]],
                "floor": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            },
        )
        assert "R4" in {e["rule"] for e in score_episode(ep)}
        ep = base_ep(
            [step(0, [0.0, 0.0, 0.80]), step(1, [0.0, 0.0, 0.68])],
            static_bodies={"table": [[0.0, 0.0, 0.72], [1.0, 0.0, 0.0, 0.0]]},
        )
        assert score_episode(ep) == []  # 0.68 > 0.72 - 0.10

        # --- t=0 below-support spawn reported as initial-state violation ---
        ep = base_ep(
            [step(0, [0.0, 0.0, 0.40]), step(1, [0.0, 0.0, 0.40])],
            support_plane_z=0.75,
        )
        assert score_episode(ep) == []
        assert any(v["rule"] == "R4" for v in ep["initial_state_violations"])

        # --- manifest filtering ---
        assert episode_matches_manifest({"provenance": {"run_id": "r1"}}, "r1")
        assert not episode_matches_manifest({"provenance": {"run_id": "r0"}}, "r1")
        assert not episode_matches_manifest({"success": True}, "r1")
        assert not episode_matches_manifest({"success": True}, None)
    finally:
        globals()["TAU1"], globals()["TAU2"], globals()["TAU_TILT"] = saved
        globals()["TASK_CALIBRATIONS"] = saved_profiles
    print("safety_scorer self-test passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(_self_test())
    output = None
    if len(sys.argv) > 1 and sys.argv[1] == "--output-audit-dir":
        if len(sys.argv) != 3:
            raise SystemExit("usage: safety_scorer.py [--output-audit-dir PATH]")
        output = sys.argv[2]
    main(output)
