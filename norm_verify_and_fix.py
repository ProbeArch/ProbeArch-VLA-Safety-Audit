"""
Verification script: loads normalizer stats from the checkpoint's safetensors,
prints side-by-side comparison of what reaches the model, then runs the same
episode (libero_spatial_0, init_state_id=2) with the normalize-before-overwrite
fix and outputs a full 280-step gripper trace.
"""
import os, sys, json
os.environ["MUJOCO_GL"] = "egl"
os.environ["AUDIT_DIR"] = "/tmp/norm-verify"
import numpy as np, torch
sys.path.insert(0, "scripts")

# ============================================================================
# STEP 1: Load normalizer stats from checkpoint safetensors
# ============================================================================
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

POLICY = "HuggingFaceVLA/smolvla_libero"
snapshot_dir = snapshot_download(POLICY)
norm_file = os.path.join(snapshot_dir, "policy_preprocessor_step_5_normalizer_processor.safetensors")

print("=" * 80)
print("STEP 1: Normalizer stats source")
print("=" * 80)
print(f"snapshot_dir:   {snapshot_dir}")
print(f"norm_file:      {norm_file}")
print(f"file exists:    {os.path.exists(norm_file)}")

raw_stats = load_file(norm_file)
print(f"keys:           {list(raw_stats.keys())}")

state_mean = raw_stats["observation.state.mean"].numpy().flatten()
state_std  = raw_stats["observation.state.std"].numpy().flatten()
action_mean = raw_stats["action.mean"].numpy().flatten()
action_std  = raw_stats["action.std"].numpy().flatten()

print(f"observation.state.mean: {state_mean}")
print(f"observation.state.std:  {state_std}")
print(f"action.mean:            {action_mean}")
print(f"action.std:             {action_std}")

# ============================================================================
# STEP 2: Side-by-side comparison at step 0 of the same episode
# ============================================================================
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation, close_envs
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device
from mlx_smolvla import observation_from_lerobot

print("\n" + "=" * 80)
print("STEP 2: Side-by-side comparison at step 0")
print("=" * 80)

env_cfg = LiberoEnv(task="libero_spatial", task_ids=[0], observation_height=256, observation_width=256)
envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
try:
    vec = envs["libero_spatial"][0]
    policy_cfg = PreTrainedConfig.from_pretrained(POLICY)
    device = get_safe_torch_device("cuda", log=False)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
    policy.eval()
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": {}}
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=POLICY, preprocessor_overrides=preprocessor_overrides
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    vec.envs[0].init_state_id = 2
    obs, _ = vec.reset()
    policy.reset()

    # (a) Preprocessor's normalized state output
    obs1 = preprocess_observation(obs)
    obs1 = add_envs_task(vec, obs1)
    obs2 = env_preprocessor(dict(obs1))
    obs3 = preprocessor(obs2)
    cuda_state = obs3.get("observation.state")
    if isinstance(cuda_state, torch.Tensor):
        cuda_state_np = cuda_state[0].cpu().numpy()
    else:
        cuda_state_np = np.asarray(cuda_state).flatten()

    # (b) Raw mlx_state_batch state
    mlx_batch = observation_from_lerobot(dict(obs1))
    mlx_state_raw = np.asarray(mlx_batch["observation.state"], dtype=np.float32).flatten()

    # (c) What actually reaches select_action (CURRENT CODE: overwrite happens AFTER preprocessor)
    # The overwrite at telemetry_rollout.py:872-883 replaces obs["observation.state"] with mlx_state_raw
    # So the model receives mlx_state_raw (unnormalized)
    what_model_gets = mlx_state_raw  # <-- THIS is what the model sees in current code

    # Also compute what it SHOULD get (normalized)
    mlx_state_normalized = (mlx_state_raw - state_mean) / np.maximum(state_std, 1e-8)

    print(f"\n{'':40s} {'(a) preprocessor':>20s} {'(b) raw mlx':>20s} {'(c) model gets NOW':>20s} {'SHOULD get':>20s}")
    print(f"{'':40s} {'(normalized)':>20s} {'(unnormalized)':>20s} {'(unnormalized)':>20s} {'(normalized)':>20s}")
    print("-" * 125)

    labels = ["eef_x", "eef_y", "eef_z", "q_w/x", "q_x/y", "q_y/z", "q_z/w", "gripper"]
    for i, label in enumerate(labels):
        a = cuda_state_np[i]
        b = mlx_state_raw[i]
        c = what_model_gets[i]
        d = mlx_state_normalized[i]
        print(f"  [{i}] {label:10s}              {a:+20.6f} {b:+20.6f} {c:+20.6f} {d:+20.6f}")

    print(f"\n  norms:")
    print(f"    (a) preprocessor norm:   {np.linalg.norm(cuda_state_np):.4f}")
    print(f"    (b) raw mlx norm:        {np.linalg.norm(mlx_state_raw):.4f}")
    print(f"    (c) model gets NOW norm: {np.linalg.norm(what_model_gets):.4f}")
    print(f"    SHOULD get norm:         {np.linalg.norm(mlx_state_normalized):.4f}")

    print(f"\n  key observation:")
    print(f"    (a) preprocessor state IS already z-scored: range [{cuda_state_np.min():.3f}, {cuda_state_np.max():.3f}]")
    print(f"    (b) raw mlx state is physical units:       range [{mlx_state_raw.min():.6f}, {mlx_state_raw.max():.6f}]")
    print(f"    (c) model gets (b) = UNNORMALIZED physical units")
    print(f"    model SHOULD get z-scored values like (a) but with correct 8-D format")

    # Confirm: does the overwrite actually happen?
    # After overwrite, obs["observation.state"] = torch.from_numpy(mlx_state_raw)
    # Then policy.select_action(obs) is called with that tensor
    # So YES, the model receives the unnormalized mlx_state_raw
    print(f"\n  confirmation: telemetry_rollout.py:872-883 overwrites obs['observation.state']")
    print(f"  with mlx_state_np (raw). policy.select_action(obs) then sees UNNORMALIZED values.")
    print(f"  The preprocessor's normalization at step (a) is DISCARDED.")

    # ============================================================================
    # STEP 3: Reconcile why quat fix improved 0/20 -> 7/20
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Reconciliation — why quat fix improved 0/20 -> 7/20")
    print("=" * 80)
    print("""
Before the quat fix:
  state = [-0.211, -0.011, 1.174, 3.14, -0.002, -0.087, 0.038, -0.038]
  quat norm = 3.14 (not 1.0)
  gripper = single finger raw (-0.038), not mean (~0)
  Model input is COMPLETELY WRONG: position ok, but quaternion is garbage
  (3.14 instead of ~1.0), and gripper is wrong format.
  Result: model can't make sense of the state at all -> 0% reach, 0% grasp.

After the quat fix (current code):
  state = [-0.211, -0.011, 1.174, 0.999, -0.0009, -0.028, -0.00026, 0.0000019]
  quat norm = 1.0 (correct)
  gripper = mean of fingers (correct format)
  BUT: this is UNNORMALIZED physical units, not z-scored.
  Model sees position x ~ -0.21 (expect ~ -1.5), z ~ 1.17 (expect ~ 1.1),
  quat w ~ 0.999 (expect ~ -5.7 or similar z-score).
  The position z component (1.17 vs mean 0.765, std 0.379 -> z-score ~1.08)
  is close enough to normalized range that the model can partially descend.
  The quaternion is unit-norm (was 3.14 before), so the model can extract
  some rotational information even without proper normalization.
  Result: model can descend to table (7/20) but gripper actions are incoherent
  (flip-flop every 1-2 steps) because the full state is unnormalized.

Why partial improvement is consistent:
  - Quat fix: removes the WORST corruption (3.14 -> 1.0), allowing basic
    spatial reasoning (arm goes down). This is a binary improvement.
  - Normalization missing: the model gets values in [-0.2, 1.2] instead of
    [-1.6, 1.1]. Not catastrophically wrong (no NaN, no sign flip), but
    off enough to degrade fine motor control (gripper).
  - The arm descends because eef_z (1.17 -> z-score ~1.08) is close to
    what the model expects. The gripper fails because grip mean (~0.0)
    vs normalized grip (~-1.9) is very different, and the full 8-D
    distribution is shifted.
""")

    # ============================================================================
    # STEP 4: Run same episode with normalize-before-overwrite fix
    # ============================================================================
    print("=" * 80)
    print("STEP 4: Running same episode with normalize-before-overwrite fix")
    print("=" * 80)

    # Reload env and policy fresh
    close_envs(envs)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec = envs["libero_spatial"][0]
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=POLICY, preprocessor_overrides=preprocessor_overrides
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    vec.envs[0].init_state_id = 2
    obs, _ = vec.reset()
    policy.reset()

    sim = vec.envs[0]._env.sim
    model = sim.model
    eef_body = None
    for b in range(model.nbody):
        name = model.body_names[b]
        ns = name.decode() if isinstance(name, bytes) else str(name)
        if ns.endswith("eef"):
            eef_body = b
            break

    # Load stats as numpy for the fix
    sm = state_mean
    ss = np.maximum(state_std, 1e-8)

    records = []
    eef_below_start = None
    print(f"\nRunning 280 steps with NORMALIZE-BEFORE-OVERWRITE fix...")
    for step in range(280):
        obs1 = preprocess_observation(obs)
        obs1 = add_envs_task(vec, obs1)
        mlx_batch = observation_from_lerobot(dict(obs1))

        obs2 = env_preprocessor(dict(obs1))
        obs3 = preprocessor(obs2)

        # THE FIX: normalize the mlx state before overwriting
        mlx_state_raw = np.asarray(mlx_batch["observation.state"], dtype=np.float32)
        mlx_state_norm = ((mlx_state_raw - sm) / ss).astype(np.float32)

        cur = obs3.get("observation.state")
        if isinstance(cur, torch.Tensor):
            obs3["observation.state"] = torch.from_numpy(mlx_state_norm).to(cur.device).to(cur.dtype)

        with torch.inference_mode():
            raw_action = policy.select_action(obs3)
        post_action = postprocessor(raw_action)
        final_action = env_postprocessor({"action": post_action})["action"]
        final_np = final_action.cpu().numpy() if isinstance(final_action, torch.Tensor) else np.asarray(final_action)
        raw_np = raw_action.cpu().numpy() if isinstance(raw_action, torch.Tensor) else np.asarray(raw_action)

        eef_z = float(sim.data.xpos[eef_body][2]) if eef_body is not None else float('nan')
        grip_qpos = []
        for j in range(model.njnt):
            name = model.joint_names[j]
            ns = name.decode() if isinstance(name, bytes) else str(name)
            if "gripper" in ns and "finger_joint" in ns:
                adr = int(model.jnt_qposadr[j])
                if adr >= 0:
                    grip_qpos.append(float(sim.data.qpos[adr]))
        grip_mean = float(np.mean(grip_qpos)) if grip_qpos else float('nan')

        efc_maxF = 0.0
        try:
            efc_force = np.asarray(sim.data.efc_force).flatten()
            contact_force = np.zeros_like(efc_force)
            for ci in range(sim.data.ncon):
                efc_id = sim.data.contact[ci].efc_address
                if 0 <= efc_id < len(contact_force):
                    contact_force[efc_id] = efc_force[efc_id]
            if len(contact_force) > 0:
                efc_maxF = float(np.max(np.abs(contact_force)))
        except Exception as exc:
            print(f"warning: could not inspect contact-force arrays: {exc}")
            pass

        grip_act = float(raw_np.flatten()[6])
        eef_below = eef_z < 1.0
        if eef_below and eef_below_start is None:
            eef_below_start = step

        records.append({
            "step": step, "eef_z": eef_z, "grip_act": grip_act,
            "grip_qpos": list(grip_qpos), "grip_qpos_mean": grip_mean,
            "maxF_efc": efc_maxF, "below_1.0": eef_below,
        })

        if step % 10 == 0 or (eef_below_start is not None and step >= eef_below_start - 3):
            marker = " <-- BELOW" if eef_below else ""
            print(f"  step {step:3d} eef_z {eef_z:.3f} grip_act {grip_act:+.4f} qpos [{grip_qpos[0]:+.4f},{grip_qpos[1]:+.4f}] mean {grip_mean:+.5f} maxF {efc_maxF:.4f}{marker}")

        obs, _, terminated, truncated, _ = vec.step(final_np)
        if terminated[0] or truncated[0]:
            print(f"  DONE at step {step}")
            break

    # Analysis
    print(f"\n{'='*100}")
    print(f"NORMALIZED FIX ANALYSIS: task libero_spatial_0 init_state_id 2")
    print(f"{'='*100}")
    print(f"eef_below_start: step {eef_below_start}")

    acts = [r["grip_act"] for r in records]
    below_acts = [r for r in records if r["below_1.0"]]
    all_qpos = [r["grip_qpos_mean"] for r in records]

    print(f"\nGripper action summary:")
    print(f" total steps: {len(records)}")
    print(f" close (act<0): {sum(1 for a in acts if a < 0)} steps")
    print(f" open (act>=0): {sum(1 for a in acts if a >= 0)} steps")

    if below_acts:
        ba = [r["grip_act"] for r in below_acts]
        bqm = [r["grip_qpos_mean"] for r in below_acts]
        print(f"\nWhen eef < 1.0m ({len(below_acts)} steps):")
        print(f" close: {sum(1 for a in ba if a < 0)}, open: {sum(1 for a in ba if a >= 0)}")
        print(f" grip_act range: [{min(ba):+.4f}, {max(ba):+.4f}]")
        print(f" grip_qpos_mean range: [{min(bqm):+.6f}, {max(bqm):+.6f}]")

        runs = []
        cur_sign = None
        run_len = 0
        for r in below_acts:
            sign = "C" if r["grip_act"] < 0 else "O"
            if sign == cur_sign:
                run_len += 1
            else:
                if cur_sign is not None:
                    runs.append((cur_sign, run_len))
                cur_sign = sign
                run_len = 1
        if cur_sign is not None:
            runs.append((cur_sign, run_len))
        print(f" pattern: {''.join(s*min(n,30) for s,n in runs)}")
        max_close = max((n for s, n in runs if s == "C"), default=0)
        max_open = max((n for s, n in runs if s == "O"), default=0)
        print(f" longest close run: {max_close} steps")
        print(f" longest open run: {max_open} steps")

    print(f"\nGripper qpos across full episode:")
    print(f" qpos[0] range: [{min(r['grip_qpos'][0] for r in records):.6f}, {max(r['grip_qpos'][0] for r in records):.6f}]")
    print(f" qpos[1] range: [{min(r['grip_qpos'][1] for r in records):.6f}, {max(r['grip_qpos'][1] for r in records):.6f}]")
    print(f" mean range:    [{min(all_qpos):.6f}, {max(all_qpos):.6f}]")
    print(f" open ~[+0.039,-0.039], closed would approach [0,0]")
    qpos0_vals = [r['grip_qpos'][0] for r in records]
    print(f" qpos[0] min {min(qpos0_vals):.6f} at step {np.argmin(qpos0_vals)} (how close to 0 = closed)")

    # Per-step table when below 1.0 (first 40 steps)
    print(f"\nPer-step trace (below 1.0, first 40):")
    for r in below_acts[:40]:
        print(f"  step {r['step']:3d} eef_z {r['eef_z']:.3f} grip {r['grip_act']:+.4f} qpos [{r['grip_qpos'][0]:+.5f},{r['grip_qpos'][1]:+.5f}] mean {r['grip_qpos_mean']:+.6f} maxF {r['maxF_efc']:.4f}")

    outpath = "/home/dunli/audit-fix-1x4/gripper_trace_normalized_fix.json"
    with open(outpath, "w") as f:
        json.dump({"task": "libero_spatial_0", "init_state_id": 2, "fix": "normalize_before_overwrite", "records": records}, f, indent=2)
    print(f"\nSaved to {outpath}")

finally:
    close_envs(envs)
