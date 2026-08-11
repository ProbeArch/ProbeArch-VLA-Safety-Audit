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
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


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
        if name_s.startswith("robot0") or name_s.endswith("eef"):
            cls = "robot"
        elif name_s in ("table", "floor", "world", "collision") or name_s.startswith("wall"):
            cls = "static"
        elif b in free_joint_body:
            cls = "object"
        else:
            cls = "static"
        table[b] = (cls, name_s)
    return table


def collect_telemetry(sim, step, table, action_prev=None):
    """Return compact telemetry record for one step (state AFTER action_prev)."""
    m, d = sim.model, sim.data
    rec = {"t": step, "contacts": [], "bodies": {}, "eef": None}
    if action_prev is not None:
        rec["action_prev"] = [float(v) for v in action_prev]
    ncon = d.ncon
    if ncon:
        addr_arr = d.contact.efc_address
        dim_arr = d.contact.dim
        geom1 = d.contact.geom1
        geom2 = d.contact.geom2
        efc = d.efc_force
        entries = []
        for i in range(ncon):
            addr = int(addr_arr[i])
            dim = int(dim_arr[i])
            if addr < 0 or dim <= 0:
                continue
            f = float(np.sqrt(efc[addr : addr + dim] @ efc[addr : addr + dim]))
            if f <= 1e-4:
                continue
            b1 = int(m.geom_bodyid[geom1[i]])
            b2 = int(m.geom_bodyid[geom2[i]])
            if b1 == b2:
                continue
            cls1, n1 = table[b1]
            cls2, n2 = table[b2]
            entries.append((f, cls1, cls2, n1, n2))
        entries.sort(reverse=True)
        for f, c1, c2, n1, n2 in entries[:40]:
            rec["contacts"].append([n1, n2, f])
    for b, (cls, name) in table.items():
        if cls == "object":
            rec["bodies"][name] = [d.xpos[b].tolist(), d.xquat[b].tolist()]
        elif name.endswith("eef") and rec["eef"] is None:
            rec["eef"] = d.xpos[b].tolist()
    if rec["eef"] is None:
        rec["eef"] = []
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task_ids", nargs="+", type=int, required=True)
    ap.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--out", default="/home/dunli/audit/rollouts")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--n_envs", type=int, default=4)
    ap.add_argument("--n_pairs", type=int, default=10)
    args = ap.parse_args()

    import gymnasium as gym
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed
    from lerobot.utils.utils import get_safe_torch_device

    set_seed(1000)
    device = get_safe_torch_device(args.device, log=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = LiberoEnv(
        task=args.suite,
        task_ids=list(args.task_ids),
        observation_height=args.resolution,
        observation_width=args.resolution,
    )
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
    policy.eval()
    try:
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"policy loaded: {type(policy).__name__} params={n_params/1e6:.1f}M device={next(policy.parameters()).device}")
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

    envs = make_env(env_cfg, n_envs=args.n_envs, use_async_envs=False)
    try:
        all_metrics = {}
        for task_id in args.task_ids:
            key = f"{args.suite}_{task_id}"
            vec = envs[args.suite][task_id]
            raw_env = vec.envs[0]
            print(f"task {key}: language='{raw_env.task_description}'", flush=True)
            init_telemetry_system(raw_env._env.sim)
            table = make_body_table(raw_env._env.sim)
            task_out = out_dir / key
            task_out.mkdir(parents=True, exist_ok=True)

            N = args.n_envs
            n_episodes = N * args.n_pairs
            max_steps = args.max_steps or raw_env._max_episode_steps
            t0 = time.time()
            successes = []
            for pair in range(args.n_pairs):
                if all((task_out / f"ep_{pair * N + k:03d}.json").exists() for k in range(N)):
                    continue
                init_ids = [vec.envs[k].init_state_id for k in range(N)]
                policy.reset()
                obs, _ = vec.reset()  # deterministic init-state cycling per env
                ep_steps = [[] for _ in range(N)]
                step = 0
                done_arr = [False] * N
                last_action = [None] * N
                while not all(done_arr) and step < max_steps:
                    obs = preprocess_observation(obs)
                    obs = add_envs_task(vec, obs)
                    obs = env_preprocessor(obs)
                    obs = preprocessor(obs)
                    with torch.inference_mode():
                        action = policy.select_action(obs)
                    action = postprocessor(action)
                    action_transition = {"action": action}
                    action_transition = env_postprocessor(action_transition)
                    action = action_transition["action"]
                    action_np = action.to("cpu").numpy()
                    for k in range(N):
                        if not done_arr[k]:
                            sim = vec.envs[k]._env.sim
                            ep_steps[k].append(collect_telemetry(sim, step, table, last_action[k]))
                    obs, reward, terminated, truncated, info = vec.step(action_np)
                    for k in range(N):
                        if not done_arr[k]:
                            last_action[k] = action_np[k]
                            done_arr[k] = bool(terminated[k] or truncated[k])
                    step += 1
                for k in range(N):
                    sim = vec.envs[k]._env.sim
                    ep_steps[k].append(collect_telemetry(sim, step, table, last_action[k]))
                success = [False] * N
                if "final_info" in info:
                    fi = info["final_info"]
                    if isinstance(fi, dict):
                        for k in range(N):
                            if k in fi:
                                success[k] = bool(fi[k].get("is_success", False))
                    elif isinstance(fi, (list, tuple)):
                        for k in range(N):
                            success[k] = bool(fi[k].get("is_success", False))
                successes.extend(success)
                for k in range(N):
                    ep = pair * N + k
                    record = {
                        "task": key,
                        "task_language": raw_env.task_description,
                        "task_id": task_id,
                        "env_ix": k,
                        "pair": pair,
                        "ep_ix": ep,
                        "init_state_id": init_ids[k],  # deterministic cycling
                        "success": success[k],
                        "n_steps": step if done_arr[k] else max_steps,
                        "max_steps": max_steps,
                        "steps": ep_steps[k],
                    }
                    with open(task_out / f"ep_{ep:03d}.json", "w") as f:
                        json.dump(record, f)
                if (pair + 1) % 5 == 0 or any(success):
                    print(
                        f"  pair {pair:02d} steps={step:3d} succ={[int(s) for s in success]} "
                        f"({(time.time()-t0)/((pair+1)*N):.1f}s/ep)",
                        flush=True,
                    )
            dt = time.time() - t0
            sr = 100.0 * np.mean(successes)
            all_metrics[key] = {
                "n_episodes": n_episodes,
                "successes": int(sum(successes)),
                "pc_success": sr,
                "seconds_total": dt,
                "seconds_per_episode": dt / n_episodes,
            }
            print(f"{key}: SR={sr:.1f}% ({sum(successes)}/{n_episodes}), {dt/n_episodes:.1f}s/ep", flush=True)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
    finally:
        close_envs(envs)


if __name__ == "__main__":
    main()