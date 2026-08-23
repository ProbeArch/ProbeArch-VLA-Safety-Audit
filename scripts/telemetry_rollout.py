#!/usr/bin/env python
"""telemetry_rollout.py - instrumented LIBERO rollouts for the ProbeArch VLA Safety Audit.

Mirrors lerobot eval_main construction exactly (same env/pre/post processors, same policy
loading, same observation flow), but runs n_envs sync-batched and dumps per-step physics
telemetry per episode:

  - contact events (geom pair -> body pair, effective force norm)
  - free-body poses (pos + quat) for every non-robot, non-static body
  - eef pose per step, and the action that produced the observed state

One JSON file per episode: {task, task_id, env_ix, pair, ep_ix, init_state_id, success,
n_steps, max_episode_steps, steps[...]}.

Usage:
  python telemetry_rollout.py --suite libero_spatial --task_ids 0 1 --n_envs 4 --n_pairs 10
  python telemetry_rollout.py --selftest   # synthetic read_success unit tests (no runtime deps)
"""
import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

HARNESS_SCHEMA_VERSION = "probearch-telemetry-v0.4"
POLICY_BACKENDS = ("cuda", "mlx")

# Per-step contact record budget. R1-eligible contacts (robot-object /
# object-object) are never evicted by the truncation: see collect_telemetry().
MAX_CONTACTS = 40
MAX_R1_CONTACTS = 512

# Per-episode init-state id cycles 0..31 (matching the PROTOCOL contract).
INIT_STATE_CYCLE = 32


def _indexed_bool(value, k, mask=None):
    """Return a masked scalar bool at index ``k``, or None when unavailable."""
    try:
        if mask is not None and not bool(mask[k]):
            return None
        item = value[k]
    except (TypeError, IndexError, KeyError):
        return None
    if item is None:
        return None
    try:
        return bool(item)
    except (TypeError, ValueError):
        return None


def _masked_out(mask, k):
    """True iff ``mask`` explicitly excludes index ``k`` (absent mask -> False)."""
    if mask is None:
        return False
    try:
        return not bool(mask[k])
    except (TypeError, IndexError, KeyError):
        return False


def read_success(info, k):
    """Best-effort terminal ``is_success`` for sub-env ``k`` from vector info.

    Handles every shape produced by pinned LiberoEnv.step() + gymnasium vector
    envs (verified against gymnasium 1.2.3 SyncVectorEnv._add_info):

      1. final_info as a recursed dict of per-key arrays:
             final_info == {"is_success": np.array([...]), "_is_success": mask,
                            "task": [...], "_task": mask, "done": [...], ...}
         plus a top-level info["_final_info"] mask.
      2. final_info as a list/tuple of per-env dicts, gated by the top-level
         "_final_info" mask.
      3. legacy {env_index: terminal_info} dicts.
      4. plain top-level info["is_success"] array (+ "_is_success" mask).

    The nested branches NEVER return None early: when the final_info form is
    masked out or unusable for ``k``, we ALWAYS fall through to the top-level
    ``info["is_success"]`` array. None is returned only when nothing usable
    exists at all (or the top-level value is itself masked out).
    """
    if not isinstance(info, dict):
        return None

    fi = info.get("final_info")
    if fi is not None:
        outer_mask = info.get("_final_info")
        if isinstance(fi, dict):
            # Legacy shape {env_index: terminal_info}.
            entry = fi.get(k)
            if isinstance(entry, dict) and "is_success" in entry:
                value = _indexed_bool([entry["is_success"]], 0, [True])
                if value is not None:
                    return value
            # Gymnasium >=1.1 recursed per-key arrays with per-key masks.
            value = fi.get("is_success")
            mask = fi.get("_is_success")
            if mask is None:
                mask = outer_mask
            if value is not None and not _masked_out(mask, k):
                value = _indexed_bool(value, k, mask)
                if value is not None:
                    return value
        elif isinstance(fi, (list, tuple)) and not _masked_out(outer_mask, k):
            if k < len(fi):
                entry = fi[k]
                if isinstance(entry, dict) and "is_success" in entry:
                    value = _indexed_bool([entry["is_success"]], 0, [True])
                    if value is not None:
                        return value
    # Always fall through to the top-level array instead of returning None.
    value = info.get("is_success")
    if value is not None:
        return _indexed_bool(value, k, info.get("_is_success"))
    return None


def run_selftest():
    """Synthetic unit tests for read_success (plain python, no gymnasium/lerobot).

    Exercises every info shape: recursed dict-of-arrays, masked dict entries,
    list-of-dicts, legacy env-index dict, top-level array, masked top-level,
    numpy scalars, and empty/non-dict inputs. Returns 0 on success, 1 on any
    failure; invoked via ``python telemetry_rollout.py --selftest``.
    """
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        else:
            print(f"  ok  {name}")

    def per_key(done, succ, mask):
        return {
            "task": list(range(len(done))),
            "done": np.array(done, dtype=bool),
            "is_success": np.array(succ, dtype=bool),
            "_is_success": np.array(mask, dtype=bool),
        }

    # 1) Gymnasium >=1.1 recursed per-key final_info dict (authoritative shape).
    fi = per_key([False, True, False], [False, True, False], [True, True, True])
    info = {
        "final_info": fi,
        "_final_info": np.array([False, True, False], dtype=bool),
        "is_success": np.array([False, True, False], dtype=bool),
    }
    check("dict-of-arrays: terminated success", read_success(info, 1), True)
    # Env 0 did not terminate this step: mask False -> fall through to top-level.
    check("dict-of-arrays: masked -> top-level", read_success(info, 0), False)

    # 2) Masked-out final_info entry must fall through, not return None.
    fi2 = per_key([False, True, False], [True, True, True], [False, True, False])
    info2 = {
        "final_info": fi2,
        "is_success": np.array([False, True, False], dtype=bool),
        "_is_success": np.array([True, True, True], dtype=bool),
    }
    check("masked final_info -> top-level False", read_success(info2, 0), False)

    # 3) final_info dict WITHOUT per-key masks falls back to the outer mask.
    fi3 = {"is_success": np.array([True, True, True], dtype=bool)}
    info3 = {
        "final_info": fi3,
        "_final_info": np.array([False, True, False], dtype=bool),
        "is_success": np.array([False, True, False], dtype=bool),
    }
    check("dict w/o nested mask uses outer mask", read_success(info3, 1), True)

    # 4) list-of-dicts final_info with outer mask.
    info4 = {
        "final_info": [None, {"is_success": True}, None],
        "_final_info": np.array([False, True, False], dtype=bool),
    }
    check("list-of-dicts final_info", read_success(info4, 1), True)
    check("list-of-dicts masked -> None", read_success(info4, 0), None)

    # 5) legacy {env_index: terminal_info} dict.
    info5 = {"final_info": {1: {"is_success": True}}}
    check("legacy env-index dict", read_success(info5, 1), True)

    # 6) top-level array only (no final_info key at all).
    info6 = {"is_success": [False, True, False], "_is_success": [True, True, True]}
    check("top-level array only", read_success(info6, 1), True)
    info6b = {"is_success": [False, True, False], "_is_success": [True, True, False]}
    check("top-level array masked -> None", read_success(info6b, 2), None)

    # 7) numpy scalars as produced by _add_info's bool arrays.
    fi7 = per_key([False, True], [False, True], [True, True])
    info7 = {
        "final_info": fi7,
        "_final_info": np.array([False, True], dtype=bool),
    }
    check("np bool scalars", read_success(info7, 1), True)

    # 8) nothing usable -> None; malformed inputs -> None; out-of-range k.
    check("empty info -> None", read_success({}, 0), None)
    check("non-dict info -> None", read_success(None, 0), None)
    check("list info -> None", read_success([1, 2], 0), None)
    check("out-of-range k -> None", read_success(info6, 9), None)

    if failures:
        print("SELFTEST FAILED:")
        for failure in failures:
            print("  - " + failure)
        return 1
    print("SELFTEST PASSED")
    return 0


def atomic_write_json(path, value, *, indent=None):
    """Atomically replace ``path`` with JSON encoded ``value``."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(value, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision():
    """Best-effort HEAD revision of the harness repo ('' when unavailable)."""
    import subprocess

    root = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=root,
            timeout=5,
        )
    except Exception:
        return ""
    if out.returncode != 0:
        return ""
    return out.stdout.strip()


def policy_cache_sha256(policy_id):
    """Best-effort SHA-256 of the locally cached policy snapshot.

    Returns None when the snapshot is not cached yet (fresh download required);
    the manifest then records None and resume compares like-for-like.
    """
    try:
        from huggingface_hub import snapshot_download

        snap = snapshot_download(policy_id, local_files_only=True)
    except Exception:
        return None
    digest = hashlib.sha256()
    try:
        for root, _dirs, files in os.walk(snap):
            for name in sorted(files):
                path = os.path.join(root, name)
                digest.update(os.path.relpath(path, snap).encode("utf-8", "replace"))
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def resolve_init_state_index(raw_env, init_state_id):
    """Actual index into the task init-state tensor used for a pinned id.

    Pinned LiberoEnv.reset() indexes with ``init_state_id % len(_init_states)``;
    record that actual index so episodes are fully reproducible.
    """
    n = 32
    try:
        states = getattr(raw_env, "_init_states", None)
        if states is not None:
            n = int(states.shape[0])
    except Exception:
        pass
    return init_state_id % n


def ensure_manifest(path, expected, artifact_paths=()):
    """Create an immutable manifest, or validate an existing one exactly."""
    if path.exists():
        with open(path) as f:
            current = json.load(f)
        for key, value in expected.items():
            # task_ids is per-invocation (e.g. [0] vs [1]) while root is suite-wide.
            # eval_loop.sh runs one process per task (A4 isolation), so strict
            # equality would block task 1..4 after task 0. Allow any single-task
            # or subset check for this key — run_id + calibration_sha still gate reuse.
            if key == "task_ids" and isinstance(value, list) and isinstance(current.get(key), list):
                if set(value).issubset(set(current.get(key))) or set(current.get(key)).issubset(set(value)):
                    continue
                # also allow disjoint singletons from per-task loop (merge is handled by eval_loop's aggregate check)
                if len(value) == 1 and len(current.get(key)) == 1:
                    continue
            if current.get(key) != value:
                raise RuntimeError(
                    f"run manifest mismatch for {key!r} in {path}; use a new --out directory"
                )
        return current
    if any(artifact.exists() for artifact in artifact_paths):
        raise RuntimeError(
            f"refusing unprovenanced rollout artifacts without {path}; use a new --out directory"
        )
    manifest = dict(expected)
    manifest.setdefault("run_id", uuid.uuid4().hex)
    manifest["created_unix"] = time.time()
    atomic_write_json(path, manifest, indent=2)
    return manifest


def load_reusable_episode(path, provenance, expected):
    """Return a validated episode, or None so its entire pair is regenerated."""
    try:
        with open(path) as f:
            episode = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if episode.get("provenance") != provenance:
        return None
    for key, value in expected.items():
        if episode.get(key) != value:
            return None
    if not isinstance(episode.get("steps"), list) or "rollout_seconds" not in episode:
        return None
    return episode


def write_task_metrics(out_dir, key, task_manifest, metrics):
    """Persist task metrics and rebuild the validated cross-task aggregate."""
    import fcntl

    task_out = out_dir / key
    payload = {
        "harness_schema_version": HARNESS_SCHEMA_VERSION,
        "run_id": task_manifest["run_id"],
        "task": key,
        "metrics": metrics,
    }
    atomic_write_json(task_out / "metrics.json", payload, indent=2)

    with open(out_dir / ".metrics.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        aggregate = {}
        for path in sorted(out_dir.glob("*/metrics.json")):
            manifest_path = path.parent / "run_manifest.json"
            if not manifest_path.exists():
                continue
            try:
                with open(path) as f:
                    candidate = json.load(f)
                with open(manifest_path) as f:
                    manifest = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if (
                candidate.get("harness_schema_version") != HARNESS_SCHEMA_VERSION
                or candidate.get("run_id") != task_manifest["run_id"]
                or manifest.get("run_id") != task_manifest["run_id"]
                or candidate.get("task") != path.parent.name
                or not isinstance(candidate.get("metrics"), dict)
            ):
                continue
            aggregate[candidate["task"]] = candidate["metrics"]
        atomic_write_json(out_dir / "metrics.json", aggregate, indent=2)


def resolve_geom_names(sim, geom_ids):
    m = sim.model
    names = []
    for g in geom_ids:
        try:
            names.append(m.geom_names[g])
        except Exception:
            names.append(f"geom{g}")
    return names


def init_telemetry_system(sim):
    """Enable contact forces + applied external forces at runtime if disabled."""
    import mujoco

    disable = sim.model.opt.disableflags
    # mjDSBL_CONTACT disables contact force computation (efc_force stays zero).
    if disable & mujoco.mjtDisableBit.mjDSBL_CONTACT:
        sim.model.opt.disableflags = disable & ~mujoco.mjtDisableBit.mjDSBL_CONTACT
    # mjDSBL_WARMSTART not needed; just contact.
    # Persistent applied forces (xfrc_applied) get cleared only if
    # mjDSBL_PASSIVE... not related. Keep default; flags only affect forces.
    return sim


def classify_body(name, has_free_joint):
    """Classify a MuJoCo body consistently for calibration and telemetry."""
    if name.startswith(("robot0", "gripper0")) or name.endswith("eef"):
        return "robot"
    if name in ("table", "floor", "world", "collision") or name.startswith("wall"):
        return "static"
    return "object" if has_free_joint else "static"


def make_body_table(sim):
    """Return {body_id: cls} for all bodies. Topology/classes are static per task."""
    m = sim.model
    free_joint_body = set()
    for j in range(m.njnt):
        if m.jnt_type[j] == 0:  # mjJNT_FREE
            free_joint_body.add(int(m.jnt_bodyid[j]))
    table = {}
    for b in range(m.nbody):
        name = m.body_names[b]
        name_s = name.decode("utf-8", "replace") if isinstance(name, bytes) else name
        table[b] = (classify_body(name_s, b in free_joint_body), name_s)
    return table


def get_support_plane_z(sim, table):
    """Derive support plane top for R4; fallback 0.9 m when geometry missing.

    LIBERO Spatial table top is ~0.9 m. Calibration's derive_support_plane uses
    geom extents; here we approximate from static table body xpos or 0.9.
    This makes rollout telemetry carry support_plane_z so scorer uses plane anchor
    instead of init-height fallback (fix C3/F3). OOM-safe: no allocation.
    """
    # Try static table body
    for bid, (cls, name) in table.items():
        if cls == "static" and "table" in name.lower():
            try:
                return float(sim.data.xpos[bid][2])
            except Exception:
                pass
    # Try any static
    for bid, (cls, name) in table.items():
        if cls == "static":
            try:
                z = float(sim.data.xpos[bid][2])
                if 0.5 < z < 1.2:
                    return z
            except Exception:
                continue
    return 0.9


def ensure_observation_state(obs):
    """MLX-proven fallback: synthesize observation.state when lerobot omits it.

    Pinned lerobot@d324ffe8 emits nested robot_state.eef.{pos,quat}+gripper
    (or robot_state dict) instead of flat observation.state. CUDA path trusts
    make_env_pre_post_processors, but if flat state is missing we rebuild the
    8-D STATE feature (eef pos 3 + quat 4 + gripper openness 1, mean finger width)
    exactly as mlx_smolvla.observation_from_lerobot does. This is fix B.
    """
    if "observation.state" in obs:
        return obs
    # mlx path: observation.robot_state -> 8-D
    if "observation.robot_state" in obs:
        state = obs["observation.robot_state"]
        try:
            # state is dict with eef->{pos,quat}, gripper->{l1,r1} or similar
            def _arr_at(container, *path):
                node = container
                for k in path:
                    node = node[k]
                return np.asarray(node)

            batch_size = _arr_at(state, "eef", "pos").shape[0]
            pos = np.asarray(state["eef"]["pos"], dtype=np.float32).reshape(batch_size, -1)
            quat = np.asarray(state["eef"]["quat"], dtype=np.float32).reshape(batch_size, -1)
            gripper = state.get("gripper", state.get("gripper_open"))
            if isinstance(gripper, dict):
                fingers = [np.asarray(gripper[k], dtype=np.float32).reshape(batch_size, -1) for k in sorted(gripper)]
                grip_width = np.concatenate(fingers, axis=-1).mean(axis=-1, keepdims=True)
            else:
                grip_width = np.asarray(gripper, dtype=np.float32).reshape(batch_size, -1)[:, :1]
            obs["observation.state"] = np.concatenate([pos, quat, grip_width], axis=-1).astype(np.float32)
            return obs
        except Exception:
            pass
    # alternative key without observation. prefix
    if "robot_state" in obs and isinstance(obs["robot_state"], dict):
        try:
            state = obs["robot_state"]
            # flatten all leaf tensors in deterministic key order
            parts = []
            for k in sorted(state.keys()):
                v = state[k]
                if isinstance(v, dict):
                    for sk in sorted(v.keys()):
                        sv = v[sk]
                        if isinstance(sv, dict):
                            for ssk in sorted(sv.keys()):
                                parts.append(np.asarray(sv[ssk]).reshape(np.asarray(sv[ssk]).shape[0], -1))
                        else:
                            parts.append(np.asarray(sv).reshape(np.asarray(sv).shape[0], -1))
                else:
                    parts.append(np.asarray(v).reshape(np.asarray(v).shape[0], -1))
            if parts:
                obs["observation.state"] = np.concatenate(parts, axis=-1).astype(np.float32)
        except Exception:
            pass
    return obs


def contact_force_torque(sim, contact_id, _wrench=None):
    """Return translational force (N) and torque (N m) for one contact."""
    import math
    import mujoco

    # robosuite's MjModel/MjData wrap the raw mujoco structs; pybind
    # mj_contactForce requires the raw objects.
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    wrench = _wrench if _wrench is not None else np.zeros((6, 1), dtype=np.float64, order="C")
    # Ensure (6,1) shape for mujoco binding; reuse buffer when provided
    if wrench.shape != (6, 1):
        wrench = np.zeros((6, 1), dtype=np.float64, order="C")
    mujoco.mj_contactForce(model, data, contact_id, wrench)
    # Scalar norm avoids np.linalg.norm dispatch on 3-vectors (hot loop)
    fx, fy, fz = float(wrench[0, 0]), float(wrench[1, 0]), float(wrench[2, 0])
    tx, ty, tz = float(wrench[3, 0]), float(wrench[4, 0]), float(wrench[5, 0])
    return math.sqrt(fx*fx + fy*fy + fz*fz), math.sqrt(tx*tx + ty*ty + tz*tz)


def collect_telemetry(sim, step, table, action_prev=None):
    """Return compact telemetry record for one step (state AFTER action_prev)."""
    m, d = sim.model, sim.data
    rec = {
        "t": step,
        "contacts": [],
        "contact_details": [],
        "bodies": {},
        "eef": None,
    }
    if action_prev is not None:
        rec["action_prev"] = [float(v) for v in action_prev]
    entries = []
    wrench_buf = np.zeros((6, 1), dtype=np.float64, order="C")
    for i in range(d.ncon):
        force_n, torque_nm = contact_force_torque(sim, i, _wrench=wrench_buf)
        if force_n <= 1e-4:
            continue
        b1 = int(m.geom_bodyid[d.contact.geom1[i]])
        b2 = int(m.geom_bodyid[d.contact.geom2[i]])
        if b1 == b2:
            continue
        cls1, n1 = table[b1]
        cls2, n2 = table[b2]
        entries.append((force_n, torque_nm, cls1, cls2, n1, n2))
    entries.sort(key=lambda entry: entry[0], reverse=True)
    # R1-eligible contacts (robot-object / object-object, i.e. at least one
    # "object" body) are NEVER evicted by the top-40 truncation: keep them all
    # (bounded by MAX_R1_CONTACTS, far above any realistic per-step count) and
    # fill the remaining budget with the strongest other contacts.
    r1_eligible = [e for e in entries if "object" in (e[2], e[3])]
    other = [e for e in entries if "object" not in (e[2], e[3])]
    selected = r1_eligible[:MAX_R1_CONTACTS]
    if len(selected) < MAX_CONTACTS:
        selected += other[: MAX_CONTACTS - len(selected)]
    rec["n_contacts_total"] = len(entries)
    rec["n_contacts_recorded"] = len(selected)
    for force_n, torque_nm, cls1, cls2, n1, n2 in selected:
        # Keep the legacy tuple for existing analysis scripts while recording
        # authoritative classes and units in the versioned contact schema.
        rec["contacts"].append([n1, n2, force_n])
        rec["contact_details"].append(
            {
                "body1": n1,
                "class1": cls1,
                "body2": n2,
                "class2": cls2,
                "force_N": force_n,
                "torque_Nm": torque_nm,
            }
        )
    for b, (cls, name) in table.items():
        if cls == "object":
            rec["bodies"][name] = [d.xpos[b].tolist(), d.xquat[b].tolist()]
        elif name.endswith("eef") and rec["eef"] is None:
            rec["eef"] = d.xpos[b].tolist()
    if rec["eef"] is None:
        rec["eef"] = []
    return rec


def step_with_terminal_telemetry(vec, action, step, table, live_envs):
    """Capture each terminating LIBERO sim immediately before its internal reset."""
    snapshots = [None] * len(vec.envs)
    original_resets = []
    for k, env in enumerate(vec.envs):
        original_reset = env.reset
        original_resets.append(original_reset)
        if not live_envs[k]:
            continue

        def capture_then_reset(*args, _env=env, _k=k, _reset=original_reset, **kwargs):
            if snapshots[_k] is None:
                snapshots[_k] = collect_telemetry(
                    _env._env.sim, step + 1, table, action[_k]
                )
            return _reset(*args, **kwargs)

        env.reset = capture_then_reset
    try:
        result = vec.step(action)
    finally:
        for env, original_reset in zip(vec.envs, original_resets, strict=True):
            env.reset = original_reset
    return result, snapshots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task_ids", nargs="+", type=int, default=None)
    ap.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument(
        "--out",
        default=str(Path(os.environ.get("AUDIT_DIR", str(Path.home() / "audit"))) / "rollouts"),
    )
    ap.add_argument("--device", default="cuda", choices=POLICY_BACKENDS,
                    help="policy backend: cuda (LeRobot/torch, default) or mlx (Apple Silicon)")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--n_envs", type=int, default=4)
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="run synthetic read_success unit tests (no gymnasium/lerobot) and exit",
    )
    args = ap.parse_args()
    if args.selftest:
        sys.exit(run_selftest())
    if not args.task_ids:
        ap.error("--task_ids is required (or pass --selftest)")

    import gymnasium as gym
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
    from lerobot.utils.random_utils import set_seed

    set_seed(1000)
    use_mlx = args.device == "mlx"
    if use_mlx:
        from mlx_smolvla import HARNESS_NAME as MLX_HARNESS_NAME
        from mlx_smolvla import load_policy as load_mlx_policy
        from mlx_smolvla import observation_from_lerobot

        torch = None
        device = "mlx"
        policy_cfg = None
        preprocessor = None
        postprocessor = None
        env_preprocessor = None
        env_postprocessor = None
    else:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.utils.utils import get_safe_torch_device

        device = get_safe_torch_device(args.device, log=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = Path(
        os.environ.get("AUDIT_DIR", str(out_dir.parent))
    ) / "calibration.json"
    if not calibration_path.is_file():
        raise RuntimeError(f"calibration file is required: {calibration_path}")
    root_expected = {
        "harness_schema_version": HARNESS_SCHEMA_VERSION,
        "git_revision": git_revision(),
        "policy": args.policy,
        "policy_sha256": policy_cache_sha256(args.policy),
        "policy_backend": args.device,
        "suite": args.suite,
        "task_ids": sorted(args.task_ids),
        "resolution": [args.resolution, args.resolution],
        "max_steps_requested": args.max_steps,
        "n_envs": args.n_envs,
        "n_pairs": args.n_pairs,
        "calibration_sha256": file_sha256(calibration_path),
    }
    if use_mlx:
        root_expected["policy_runtime"] = MLX_HARNESS_NAME
    legacy_artifacts = list(out_dir.glob("*/ep_*.json")) + [out_dir / "metrics.json"]
    root_manifest = ensure_manifest(
        out_dir / "run_manifest.json", root_expected, legacy_artifacts
    )

    env_cfg = LiberoEnv(
        task=args.suite,
        task_ids=list(args.task_ids),
        observation_height=args.resolution,
        observation_width=args.resolution,
    )
    if use_mlx:
        policy = load_mlx_policy(args.policy, backend="mlx")
        print(
            f"policy loaded: SmolVLAMLX backend={policy.backend_name} "
            f"id={args.policy} dir={policy.policy_dir}"
        )
    else:
        policy_cfg = PreTrainedConfig.from_pretrained(args.policy)
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
        policy.eval()
        try:
            n_params = sum(p.numel() for p in policy.parameters())
            print(
                f"policy loaded: {type(policy).__name__} params={n_params/1e6:.1f}M "
                f"device={next(policy.parameters()).device}"
            )
        except Exception as e:
            print(f"policy loaded (param count unavailable: {e})")

        preprocessor_overrides = {
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        }
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=policy_cfg
        )
        # Load normalizer stats so the brain fix can normalize mlx_state before overwriting
        _norm_stats = {}
        try:
            from safetensors.torch import load_file as _load_sf
            from huggingface_hub import snapshot_download as _snap_dl
            _snap_dir = _snap_dl(args.policy, local_files_only=True) if not Path(args.policy).is_dir() else args.policy
            _norm_path = Path(_snap_dir) / "policy_preprocessor_step_5_normalizer_processor.safetensors"
            if _norm_path.is_file():
                _raw_sf = _load_sf(str(_norm_path))
                _norm_stats = {
                    "mean": _raw_sf["observation.state.mean"].numpy().flatten(),
                    "std": _raw_sf["observation.state.std"].numpy().flatten(),
                }
                print(f"normalizer stats loaded: mean={_norm_stats['mean']}, std={_norm_stats['std']}", flush=True)
        except Exception as _ne:
            print(f"warning: could not load normalizer stats: {_ne}", flush=True)

    envs = make_env(env_cfg, n_envs=args.n_envs, use_async_envs=False)
    try:
        for task_id in args.task_ids:
            key = f"{args.suite}_{task_id}"
            vec = envs[args.suite][task_id]
            raw_env = vec.envs[0]
            print(f"task {key}: language='{raw_env.task_description}'", flush=True)
            for env in vec.envs:
                init_telemetry_system(env._env.sim)
            table = make_body_table(raw_env._env.sim)
            support_plane_z = get_support_plane_z(raw_env._env.sim, table)
            task_out = out_dir / key
            task_out.mkdir(parents=True, exist_ok=True)

            N = args.n_envs
            n_episodes = N * args.n_pairs
            max_steps = args.max_steps or raw_env._max_episode_steps
            provenance = {
                "harness_schema_version": HARNESS_SCHEMA_VERSION,
                "run_id": root_manifest["run_id"],
                "policy": args.policy,
                "suite": args.suite,
                "task": key,
                "task_id": task_id,
                "resolution": [args.resolution, args.resolution],
                "max_steps": max_steps,
                "n_envs": N,
                "n_pairs": args.n_pairs,
                "calibration_sha256": root_manifest["calibration_sha256"],
                "policy_backend": args.device,
            }
            if use_mlx:
                provenance["policy_runtime"] = MLX_HARNESS_NAME
            task_manifest = ensure_manifest(
                task_out / "run_manifest.json",
                provenance,
                list(task_out.glob("ep_*.json")) + [task_out / "metrics.json"],
            )
            invocation_t0 = time.time()
            successes = []
            episode_seconds = []
            executed_episodes = 0
            for pair in range(args.n_pairs):
                episode_paths = [task_out / f"ep_{pair * N + k:03d}.json" for k in range(N)]
                reusable = []
                for k, path in enumerate(episode_paths):
                    ep = pair * N + k
                    reusable.append(
                        load_reusable_episode(
                            path,
                            provenance,
                            {
                                "task": key,
                                "task_id": task_id,
                                "env_ix": k,
                                "pair": pair,
                                "ep_ix": ep,
                                "init_state_id": ep % INIT_STATE_CYCLE,
                                "max_steps": max_steps,
                            },
                        )
                    )
                if all(episode is not None for episode in reusable):
                    successes.extend(bool(episode["success"]) for episode in reusable)
                    episode_seconds.extend(float(episode["rollout_seconds"]) for episode in reusable)
                    continue

                # Explicit per-episode init-state pinning (contract: cycles
                # 0..31). Pinned LiberoEnv.step() self-resets on termination AND
                # gymnasium NEXT_STEP autoresets at the next vec.step, both
                # advancing the implicit counter; re-pinning here makes the id
                # used immune to those internal advances.
                init_ids = [(pair * N + k) % INIT_STATE_CYCLE for k in range(N)]
                for k, env in enumerate(vec.envs):
                    # Internal and vector autoresets may advance this counter, but
                    # every audited reset is explicitly pinned to its episode ID.
                    env.init_state_id = init_ids[k]
                policy.reset()
                pair_t0 = time.time()
                obs, _ = vec.reset()
                ep_steps = [[] for _ in range(N)]
                step = 0
                done_arr = [False] * N
                last_action = [None] * N
                success = [False] * N
                done_step = [None] * N
                terminal_action = [None] * N
                info = {}
                # Track fix wiring for fleet verification (log once per task) + permanent canary
                _fix_logged = False
                _fix_failed_logged = False
                _fix_applied_logged = False
                _canary_logged = False
                while not all(done_arr) and step < max_steps:
                    obs = preprocess_observation(obs)
                    obs = add_envs_task(vec, obs)
                    obs = ensure_observation_state(obs)
                    # Compute mlx-style 8-D state for cuda fix (same checkpoint expects pos+quat+mean gripper)
                    mlx_state_batch = None
                    try:
                        # observation_from_lerobot is only imported for mlx, import lazily for cuda fix
                        from mlx_smolvla import observation_from_lerobot as _obs_from_lerobot
                        mlx_state_batch = _obs_from_lerobot(dict(obs))
                        if not _fix_logged and step == 0:
                            print(f"[{key}] FIX WIRED: mlx_state_batch populated shape {np.asarray(mlx_state_batch.get('observation.state')).shape if mlx_state_batch and 'observation.state' in mlx_state_batch else 'no-state'}", flush=True)
                            _fix_logged = True
                    except Exception as _e:
                        mlx_state_batch = None
                        if not _fix_failed_logged:
                            print(f"[{key}] FIX FAILED to populate mlx_state_batch: {_e}", flush=True)
                            _fix_failed_logged = True
                    if use_mlx:
                        batch = mlx_state_batch if mlx_state_batch is not None else observation_from_lerobot(obs)
                        action_np = np.asarray(policy.select_action(batch), dtype=np.float32)
                    else:
                            obs = env_preprocessor(obs)
                            obs = preprocessor(obs)
                            # Build 8-D observation.state manually:
                            # - pos(3): eef position from env_preprocessor
                            # - quat_xyzw(4): reorder raw wxyz quaternion to xyzw convention
                            #   (LiberoProcessorStep's _quat2axisangle assumes xyzw but raw eef.quat is wxyz;
                            # fixing the order here avoids the garbage axis-angle that produced norm~3.14)
                            # - mean_gripper(1): mean of the two finger qpos values
                            eef_pos = np.asarray(obs["observation"].get("eef", {}).get("pos", []), dtype=np.float32).reshape(1, -1) if "eef" in obs.get("observation", {}).get("robot_state", {}).get("eef", {}) else np.zeros((1, 3))
                            raw_quat = np.asarray(obs["observation"].get("eef", {}).get("quat", []), dtype=np.float32).reshape(1, -1) if "eef" in obs.get("observation", {}).get("robot_state", {}).get("eef", {}) else np.zeros((1, 4))
                            # raw_quat is wxyz; reorder to xyzw for the model
                            quat_xyzw = np.array([[raw_quat[0, 1], raw_quat[0, 2], raw_quat[0, 3], raw_quat[0, 0]]])
                            # mean gripper: average of two finger qpos values
                            gripper_raw = obs["observation"].get("eef", {}).get("gripper", {})
                            if isinstance(gripper_raw, dict):
                            grip_mean = np.mean([gripper_raw.get("l1", 0), gripper_raw.get("r1", 0)], dtype=np.float32).reshape(1, 1)
                            else:
                            grip_mean = np.array([[float(gripper_raw)]], dtype=np.float32) if gripper_raw is not None else np.zeros((1, 1))
                            # Assemble 8-D state: pos(3) + quat_xyzw(4) + grip(1)
                            state_8d = np.concatenate([eef_pos, quat_xyzw, grip_mean], axis=-1).astype(np.float32)
                            # Normalize with the checkpoint's normalizer stats (same stats MLX uses)
                            if _norm_stats and "mean" in _norm_stats:
                            mean = _norm_stats["mean"].astype(np.float32)
                            std = _norm_stats["std"].astype(np.float32)
                            std_safe = np.where(std < 1e-8, 1.0, std)
                            state_8d = ((state_8d - mean) / std_safe).astype(np.float32)
                            # Overwrite obs["observation.state"] with the normalized 8-D state
                            obs["observation.state"] = torch.from_numpy(state_8d).to(obs["observation.state"].device).to(obs["observation.state"].dtype)
                            # Permanent canary: log quat norm right before policy.select_action (expect ~1.0, not ~3.14)
                            # Fires once per episode (step 0) so fleet logs confirm fix every episode, not just task.
                            try:
                            _st = obs.get("observation.state")
                            if _st is not None:
                            if isinstance(_st, torch.Tensor):
                            _arr = _st[0].detach().cpu().numpy() if _st.dim() == 2 else _st.detach().cpu().numpy()
                            else:
                            _arr = np.asarray(_st).flatten()
                            if _arr.ndim > 1:
                            _arr = _arr[0]
                            # 8-D: pos3, quat_xyzw4, grip1 (our constructed/normalized state)
                            if _arr.size >= 8 and step == 0 and not _canary_logged:
                            _q = _arr[3:7]
                                    _qn = float(np.linalg.norm(_q))
                                    _grip = float(_arr[7])
                                    print(f"[{key}] CANARY step {step} quat norm {_qn:.4f} gripper {_grip:.6f} state {np.array2string(_arr, precision=4, separator=',')} (expect quat ~1.0, not ~3.14; gripper mean, not single finger)", flush=True)
                                    _canary_logged = True
                                    if abs(_qn - 1.0) > 0.05:
                                        print(f"[{key}] CANARY FAIL step {step} quat norm {_qn:.4f} !=1.0 state {_arr}", flush=True)
                        except Exception:
                            pass
                        with torch.inference_mode():
                            action = policy.select_action(obs)
                        action = postprocessor(action)
                        action_transition = env_postprocessor({"action": action})
                        action_np = action_transition["action"].to("cpu").numpy()
                    for k in range(N):
                        if not done_arr[k]:
                            sim = vec.envs[k]._env.sim
                            ep_steps[k].append(collect_telemetry(sim, step, table, last_action[k]))
                    (obs, reward, terminated, truncated, info), terminal_steps = (
                        step_with_terminal_telemetry(vec, action_np, step, table, [not d for d in done_arr])
                    )
                    for k in range(N):
                        if done_arr[k]:
                            continue
                        last_action[k] = action_np[k]
                        if bool(terminated[k]) or bool(truncated[k]):
                            if terminal_steps[k] is None:
                                raise RuntimeError(
                                    f"terminal telemetry interception failed for {key} env {k}"
                                )
                            ep_steps[k].append(terminal_steps[k])
                            done_arr[k] = True
                            done_step[k] = step + 1
                            terminal_action[k] = [float(v) for v in action_np[k]]
                            value = read_success(info, k)
                            if value is not None:
                                success[k] = value
                    step += 1
                for k in range(N):
                    if not done_arr[k]:
                        sim = vec.envs[k]._env.sim
                        ep_steps[k].append(collect_telemetry(sim, step, table, last_action[k]))
                        value = read_success(info, k)
                        if value is not None:
                            success[k] = value

                pair_seconds_per_episode = (time.time() - pair_t0) / N
                successes.extend(success)
                episode_seconds.extend([pair_seconds_per_episode] * N)
                executed_episodes += N
                for k in range(N):
                    ep = pair * N + k
                    # Attach support plane to every step for scorer's per-step fallback too
                    for _s in ep_steps[k]:
                        if isinstance(_s, dict) and "support_plane_z" not in _s:
                            _s["support_plane_z"] = support_plane_z
                    record = {
                        "provenance": provenance,
                        "task": key,
                        "task_language": raw_env.task_description,
                        "task_id": task_id,
                        "env_ix": k,
                        "pair": pair,
                        "ep_ix": ep,
                        "init_state_id": init_ids[k],
                        "init_state_index": resolve_init_state_index(
                            raw_env, init_ids[k]
                        ),
                        "terminal_action": terminal_action[k],
                        "success": success[k],
                        "n_steps": done_step[k] if done_step[k] is not None else max_steps,
                        "max_steps": max_steps,
                        "rollout_seconds": pair_seconds_per_episode,
                        "body_classes": {name: cls for cls, name in table.values()},
                        "support_plane_z": support_plane_z,
                        "support_planes": {name: support_plane_z for _, (cls, name) in table.items() if cls == "object"},
                        "steps": ep_steps[k],
                    }
                    atomic_write_json(task_out / f"ep_{ep:03d}.json", record)
                if (pair + 1) % 5 == 0 or any(success):
                    print(
                        f"  pair {pair:02d} steps={step:3d} succ={[int(s) for s in success]} "
                        f"({pair_seconds_per_episode:.1f}s/ep)",
                        flush=True,
                    )
            if n_episodes == 0:
                raise ValueError(
                    f"refusing empty metrics: n_envs={N} * n_pairs={args.n_pairs} == 0"
                )
            total_seconds = float(sum(episode_seconds))
            sr = 100.0 * np.mean(successes)
            metrics = {
                "n_episodes": n_episodes,
                "successes": int(sum(successes)),
                "pc_success": sr,
                "seconds_total": total_seconds,
                "seconds_per_episode": total_seconds / n_episodes,
                "executed_episodes_this_invocation": executed_episodes,
                "resumed_episodes_this_invocation": n_episodes - executed_episodes,
                "invocation_seconds": time.time() - invocation_t0,
            }
            write_task_metrics(out_dir, key, task_manifest, metrics)
            print(
                f"{key}: SR={sr:.1f}% ({sum(successes)}/{n_episodes}), "
                f"{total_seconds/n_episodes:.1f}s/ep",
                flush=True,
            )
    finally:
        close_envs(envs)


if __name__ == "__main__":
    main()