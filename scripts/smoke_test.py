#!/usr/bin/env python
"""smoke_test.py - env bring-up + egl render + random action sanity + policy load check."""
import os, sys, time, json
import numpy as np
import torch

def main():
    import gymnasium as gym
    import numpy as np
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env
    from lerobot.envs.utils import close_envs
    from lerobot.utils.random_utils import set_seed

    set_seed(0)
    env_cfg = LiberoEnv(task="libero_spatial", task_ids=[0], observation_height=256, observation_width=256)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec = envs["libero_spatial"][0]
    raw = vec.envs[0]
    print("env type:", type(raw).__name__)
    print("task:", raw.task)
    print("task_language:", raw.task_description)
    print("max_episode_steps:", raw._max_episode_steps)
    print("action_space:", raw.action_space)
    print("obs space keys:", list(raw.observation_space["pixels"].spaces.keys()))

    sim = raw._env.sim
    print("model bodies:", len(sim.model.body_names))
    obs, _ = vec.reset()
    print("obs pixel shapes:", {k: v.shape for k, v in obs["pixels"].items()})
    print("obs robot_state keys:", list(obs["robot_state"].keys()) if "robot_state" in obs else "none")

    # random policy roll 20 steps
    t0 = time.time()
    for i in range(20):
        a = np.random.uniform(-1, 1, size=(1, 7)).astype(np.float32)
        obs, r, term, trunc, info = vec.step(a)
        if term or trunc:
            print("terminated early at", i)
            break
    dt = time.time() - t0
    print(f"random-roll 20 steps in {dt:.2f}s -> sim {20/dt:.1f} Hz (without policy)")

    # check contact telemetry accessibility
    import mujoco
    sim = raw._env.sim
    flags = int(sim.model.opt.disableflags)
    has_contact_flag = bool(flags & int (mujoco.mjtDisableBit.mjDSBL_CONTACT))
    print("disableflags:", flags, "contact-disabled:", has_contact_flag)

    # render sanity
    try:
        frame = raw.render()
        print("render OK, frame", frame.shape, frame.dtype)
        np.save("/home/dunli/audit/smoke_frame.npy", frame)
    except Exception as e:
        print("render FAILED:", type(e).__name__, e)

    close_envs(envs)

    # policy load (GPU, bf16)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy
    t0 = time.time()
    cfg = PreTrainedConfig.from_pretrained("HuggingFaceVLA/smolvla_libero")
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map={})
    policy.eval()
    n = sum(p.numel() for p in policy.parameters())
    dt = time.time() - t0
    dev = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype
    print(f"policy load: {dt:.1f}s params={n/1e6:.1f}M device={dev} dtype={dtype}")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"VRAM: {free/1e9:.2f} free / {total/1e9:.2f} total")

main()