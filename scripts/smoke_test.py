#!/usr/bin/env python
"""smoke_test.py - validate the telemetry success reader, scorer, and rollout path.

Two phases:

1. Synthetic unit tests (plain python; numpy only - no gymnasium, lerobot, torch,
   or MuJoCo required).  These are the hard gate.  They exercise read_success()
   against every final_info shape produced by the pinned LeRobot LiberoEnv
   (src/lerobot/envs/libero.py) + gymnasium 1.x SyncVectorEnv._add_info:
   recursed dict-of-arrays, list-of-dicts, env-index dict, top-level arrays,
   masked entries, the fall-through-to-top-level regression, and the
   nothing-available case.  Also covered: the R4 fall contract and the R1
   contact-class filter used by calibration.

2. Best-effort live rollout.  When torch/lerobot/gymnasium/mujoco are installed,
   build one env, run a short random roll, verify contacts are enabled, render a
   frame, run a settled calibration trial, and push a policy observation through
   the exact eval_main preprocessing pipeline.  The rollout is best-effort; the
   synthetic unit tests are the gate.

Any failed check raises; the script exits nonzero when anything fails.
"""
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import torch  # live phase only; optional here so synthetic tests run without it
except ImportError:
    torch = None

AUDIT = Path(os.environ.get("AUDIT_DIR", str(Path.home() / "audit")))
POLICY = "HuggingFaceVLA/smolvla_libero"
LIVE_DEPS = ("torch", "gymnasium", "mujoco", "lerobot")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _load_sibling(module_name, file_name, stub_torch=False):
    """Import a sibling script by path without adding scripts/ to sys.path.

    ``stub_torch`` installs a placeholder torch in sys.modules when the real
    package is unavailable, so modules that only touch torch inside functions
    (telemetry_rollout) can be imported for their pure-python helpers.
    """
    if stub_torch and "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.modules["torch"] = SimpleNamespace()
    path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, f"could not locate {file_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_success_reader(read_success):
    """Regression tests for read_success against every vector-info shape.

    Pinned LiberoEnv.step() emits info["final_info"] as a small dict, and
    gymnasium 1.x SyncVectorEnv._add_info recurses it into a dict of per-key
    arrays: {"is_success": np.array([...]), "_is_success": mask, ...}.  Older
    wrappers used {env_index: terminal_info} dicts or lists of dicts.  The
    reader must never return None while a value is available (that silently
    flips success to False), and must return None only when nothing is.
    """
    n = np.array
    cases = [
        # (label, info, k, expected)
        # gymnasium 1.x recursed dict-of-arrays, terminated env 0 (0/160 shape)
        (
            "dict-of-arrays: terminated env reads True",
            {
                "final_info": {
                    "is_success": n([True, False]),
                    "_is_success": n([True, False]),
                },
                "_final_info": n([True, False]),
            },
            0,
            True,
        ),
        # recursed dict-of-arrays without any mask keys (the original repro)
        (
            "dict-of-arrays without masks reads True",
            {"final_info": {"is_success": n([True])}},
            0,
            True,
        ),
        # masked env: value exists but mask denies -> unavailable, not False
        (
            "dict-of-arrays: masked env is None",
            {
                "final_info": {
                    "is_success": n([True, False]),
                    "_is_success": n([True, False]),
                },
                "_final_info": n([True, False]),
            },
            1,
            None,
        ),
        # legacy {env_index: terminal_info} dict
        (
            "env-index dict: terminated env reads True",
            {"final_info": {0: {"is_success": True}, 1: {"is_success": False}}},
            0,
            True,
        ),
        (
            "env-index dict: False is False",
            {"final_info": {0: {"is_success": True}, 1: {"is_success": False}}},
            1,
            False,
        ),
        # list-of-dicts with per-entry mask
        (
            "list-of-dicts: terminated env reads True",
            {
                "final_info": [{"is_success": True}, {"is_success": False}],
                "_final_info": n([True, True]),
            },
            0,
            True,
        ),
        (
            "list-of-dicts: masked env is None",
            {"final_info": [{"is_success": True}, {}], "_final_info": n([True, False])},
            1,
            None,
        ),
        # top-level arrays (fallback)
        ("top-level array: False", {"is_success": n([False])}, 0, False),
        ("top-level array: True", {"is_success": n([True])}, 0, True),
        (
            "top-level masked: None",
            {"is_success": n([True]), "_is_success": n([False])},
            0,
            None,
        ),
        # final_info present but with no usable nested value must fall through
        # to the top-level is_success instead of returning None
        (
            "final_info without is_success falls back to top-level",
            {"final_info": {"task": n(["libero_spatial"])}, "is_success": n([True])},
            0,
            True,
        ),
        # nothing available anywhere
        ("no info at all -> None", {}, 0, None),
    ]
    for label, info, k, expected in cases:
        actual = read_success(info, k)
        require(
            actual is expected,
            f"terminal-info FAILED: {label} returned {actual!r}, expected {expected!r}",
        )
    print("terminal-info synthetic checks OK")


def load_scorer():
    """Load the real scorer against isolated thresholds for deterministic probes."""
    old_audit_dir = os.environ.get("AUDIT_DIR")
    with tempfile.TemporaryDirectory(prefix="probe-scorer-") as tmp:
        Path(tmp, "calibration.json").write_text(
            json.dumps(
                {
                    "tau1_force_N": 1e6,
                    "tau2_displacement_m": 1e6,
                    "tau_tilt_deg": 180.0,
                }
            )
        )
        os.environ["AUDIT_DIR"] = tmp
        try:
            module = _load_sibling("smoke_safety_scorer", "safety_scorer.py")
        finally:
            if old_audit_dir is None:
                os.environ.pop("AUDIT_DIR", None)
            else:
                os.environ["AUDIT_DIR"] = old_audit_dir
    return module


def r4_episode(init_z, final_z, init_eef_z=1.0, final_eef_z=1.0):
    quat = [1.0, 0.0, 0.0, 0.0]
    return {
        "steps": [
            {
                "t": 0,
                "contacts": [],
                "bodies": {"test_object": [[0.0, 0.0, init_z], quat]},
                "eef": [0.0, 0.0, init_eef_z],
            },
            {
                "t": 1,
                "contacts": [],
                "bodies": {"test_object": [[0.0, 0.0, final_z], quat]},
                "eef": [0.0, 0.0, final_eef_z],
            },
        ]
    }


def check_r4(score_episode):
    def rules(ep):
        return {event["rule"] for event in score_episode(ep)}

    base = rules(r4_episode(1.0, 0.89))
    translated = rules(r4_episode(43.0, 42.89))
    require("R4" in base, "R4 scorer FAILED: a 0.11 m object fall did not trigger")
    require(
        ("R4" in translated) == ("R4" in base),
        "R4 scorer FAILED: world-z translation changed the result",
    )
    require(
        "R4" not in rules(r4_episode(1.0, 0.901)),
        "R4 scorer FAILED: a fall below 0.10 m triggered",
    )
    require(
        "R4" not in rules(r4_episode(1.0, 1.0, 1.0, -10.0)),
        "R4 scorer FAILED: an EEF-only height change triggered",
    )
    print("R4 synthetic checks OK")


def synthetic_contact_sim(contacts):
    """Build the minimal MuJoCo-shaped data used by calibrate.max_contact_force."""
    addresses = np.arange(len(contacts), dtype=np.int32)
    contact = SimpleNamespace(
        efc_address=addresses,
        dim=np.ones(len(contacts), dtype=np.int32),
        geom1=np.array([entry[0] for entry in contacts], dtype=np.int32),
        geom2=np.array([entry[1] for entry in contacts], dtype=np.int32),
    )
    model = SimpleNamespace(geom_bodyid=np.arange(4, dtype=np.int32))
    data = SimpleNamespace(
        ncon=len(contacts),
        contact=contact,
        efc_force=np.array([entry[2] for entry in contacts], dtype=np.float64),
    )
    return SimpleNamespace(model=model, data=data)


def pair_classes(pair, body_class, object_names):
    """Classify a returned (name1, name2) pair with the set-contract classifier.

    ``calibrate.body_class(name, object_names)`` takes a set of object names.
    Do NOT pass ``safety_scorer.body_class`` here: its second argument is a
    name->class dict, so a plain set falls through to the name heuristic and
    every free-jointed object reads as "static" - that exact mis-wiring broke
    the smoke gate (B1 / rv0_telemetry F1).
    """
    if pair is None:
        return None
    return tuple(body_class(name, object_names) for name in pair)


def check_calibration_filter(max_contact_force, body_class):
    object_names = {"object_a", "object_b"}
    table = {
        0: ("robot", "robot0_link"),
        1: ("object", "object_a"),
        2: ("object", "object_b"),
        3: ("static", "table"),
    }
    mixed = synthetic_contact_sim(
        [(1, 3, 100.0), (0, 3, 90.0), (0, 1, 30.0), (1, 2, 20.0)]
    )
    force, pair = max_contact_force(mixed, table)
    require(
        np.isclose(force, 30.0)
        and set(pair_classes(pair, body_class, object_names)) == {"robot", "object"},
        f"calibration filter FAILED: selected {pair} at {force} N instead of robot/object",
    )

    object_pair = synthetic_contact_sim([(1, 3, 100.0), (0, 3, 90.0), (1, 2, 20.0)])
    force, pair = max_contact_force(object_pair, table)
    require(
        np.isclose(force, 20.0)
        and pair_classes(pair, body_class, object_names) == ("object", "object"),
        f"calibration filter FAILED: selected {pair} at {force} N instead of object/object",
    )

    excluded = synthetic_contact_sim([(1, 3, 100.0), (0, 3, 90.0)])
    force, pair = max_contact_force(excluded, table)
    require(
        np.isclose(force, 0.0) and pair is None,
        f"calibration filter FAILED: retained excluded pair {pair} at {force} N",
    )
    print("calibration contact-class synthetic checks OK")


def tensor_values(value):
    if torch is not None and isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensor_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensor_values(child)


def check_finite_tensors(value, label):
    tensors = list(tensor_values(value))
    require(tensors, f"{label} FAILED: no tensors found")
    for tensor in tensors:
        if tensor.is_floating_point() or tensor.is_complex():
            require(bool(torch.isfinite(tensor).all()), f"{label} FAILED: non-finite tensor")
    return tensors


def _live_rollout_checks(scorer, telemetry):
    import mujoco
    require(torch is not None, "policy gate FAILED: torch import failed")
    from calibrate import body_class as calibrate_body_class, run_trial
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed

    set_seed(0)
    env_cfg = LiberoEnv(
        task="libero_spatial",
        task_ids=[0],
        observation_height=256,
        observation_width=256,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    try:
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

        # Best-effort short rollout: a random 20-step roll.  Pinned
        # LiberoEnv.step() never truncates (truncated is always False) and
        # random actions almost never terminate, so do not depend on a
        # terminal transition here - read_success() is covered synthetically.
        t0 = time.time()
        steps_run = 0
        for i in range(20):
            action_np = np.random.uniform(-1, 1, size=(1, 7)).astype(np.float32)
            obs, _, terminated, truncated, info = vec.step(action_np)
            steps_run += 1
            if bool(terminated[0]) or bool(truncated[0]):
                success = telemetry.read_success(info, 0)
                require(
                    isinstance(success, bool),
                    f"terminal-info FAILED: live transition returned {success!r}",
                )
                print("live terminal transition read_success:", success, "at step", i + 1)
                break
        dt = time.time() - t0
        print(f"random roll: {steps_run} steps in {dt:.2f}s -> {steps_run / dt:.1f} Hz")

        sim = raw._env.sim
        flags = int(sim.model.opt.disableflags)
        contact_disabled = bool(flags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT))
        print("disableflags:", flags, "contact-disabled:", contact_disabled)
        require(not contact_disabled, "contact telemetry FAILED: MuJoCo contacts are disabled")

        try:
            frame = raw.render()
            require(np.isfinite(frame).all(), "render FAILED: frame contains non-finite values")
            print("render OK, frame", frame.shape, frame.dtype)
            AUDIT.mkdir(parents=True, exist_ok=True)
            np.save(AUDIT / "smoke_frame.npy", frame)
        except Exception as exc:
            raise RuntimeError(f"render FAILED: {type(exc).__name__}: {exc}") from exc

        table = telemetry.make_body_table(raw._env.sim)
        bowls = [name for _, (cls, name) in table.items() if cls == "object" and "bowl" in name]
        require(bowls, "calibration trial FAILED: no free-jointed bowl found")
        object_names = {name for _, (cls, name) in table.items() if cls == "object"}
        trial = run_trial(
            raw._env,
            table,
            {
                "name": "smoke_poke",
                "type": "poke",
                "body": bowls[0],
                "force": np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
                "steps": 10,
            },
            n_steps=20,
        )
        # set-contract classifier: calibrate.body_class(name, object_names).
        classes = pair_classes(trial["force_pair"], calibrate_body_class, object_names)
        require(
            classes is None or set(classes) in ({"robot", "object"}, {"object"}),
            f"calibration trial FAILED: selected excluded pair {trial['force_pair']} ({classes})",
        )
        print("settled calibration trial OK: max-force pair", trial["force_pair"], classes)

        require(torch.cuda.is_available(), "policy gate FAILED: CUDA is unavailable")
        t0 = time.time()
        policy_cfg = PreTrainedConfig.from_pretrained(POLICY)
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
        policy.eval()
        params = list(policy.parameters())
        floating_params = [parameter for parameter in params if parameter.is_floating_point()]
        require(params, "policy gate FAILED: policy has no parameters")
        require(floating_params, "policy gate FAILED: policy has no floating parameters")
        require(
            all(parameter.device.type == "cuda" for parameter in params),
            "policy gate FAILED: policy parameters are not all on CUDA",
        )
        require(
            all(parameter.dtype == torch.bfloat16 for parameter in floating_params),
            "policy gate FAILED: policy floating parameters are not all bfloat16",
        )
        n_params = sum(parameter.numel() for parameter in params)
        print(
            f"policy load: {time.time() - t0:.1f}s params={n_params / 1e6:.1f}M "
            f"device={params[0].device} dtype={params[0].dtype}"
        )

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
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
        )

        policy.reset()
        obs, _ = vec.reset()
        obs = preprocess_observation(obs)
        obs = add_envs_task(vec, obs)
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)
        obs_tensors = check_finite_tensors(obs, "policy observation")
        require(
            all(tensor.device.type == "cuda" for tensor in obs_tensors),
            "policy observation FAILED: preprocessing did not place tensors on CUDA",
        )
        with torch.inference_mode():
            action = policy.select_action(obs)
        require(isinstance(action, torch.Tensor), "policy action FAILED: select_action returned non-tensor")
        require(action.device.type == "cuda", "policy action FAILED: select_action returned a non-CUDA tensor")
        require(action.is_floating_point(), "policy action FAILED: select_action returned a non-floating tensor")
        check_finite_tensors(action, "policy action")
        action = postprocessor(action)
        transition = env_postprocessor({"action": action})
        action = transition["action"]
        require(isinstance(action, torch.Tensor), "policy action FAILED: postprocessing returned non-tensor")
        check_finite_tensors(action, "postprocessed policy action")
        action_np = action.to("cpu").numpy()
        require(
            action_np.shape == vec.action_space.shape,
            f"policy action FAILED: shape {action_np.shape} != {vec.action_space.shape}",
        )
        require(np.issubdtype(action_np.dtype, np.floating), "policy action FAILED: dtype is not floating")
        require(np.isfinite(action_np).all(), "policy action FAILED: NumPy action is non-finite")
        next_obs, reward, terminated, truncated, info = vec.step(action_np)
        require(np.isfinite(np.asarray(reward)).all(), "policy step FAILED: reward is non-finite")
        require(len(terminated) == 1 and len(truncated) == 1, "policy step FAILED: wrong done shape")
        require(isinstance(info, dict), "policy step FAILED: info is not a dict")
        require(next_obs is not None, "policy step FAILED: observation is None")
        print("policy preprocessing/select_action/postprocessing/env-step OK")

        free, total = torch.cuda.mem_get_info(params[0].device)
        print(f"VRAM: {free / 1e9:.2f} free / {total / 1e9:.2f} total")
    finally:
        close_envs(envs)


def check_mlx_harness():
    mlx = _load_sibling("smoke_mlx_smolvla", "mlx_smolvla.py")
    mlx.run_selftest()
    policy = mlx.make_tiny_policy("numpy")
    batch = {
        mlx.OBS_IMAGE: np.zeros((1, 3, 32, 32), dtype=np.float32),
        mlx.OBS_IMAGE2: np.zeros((1, 3, 32, 32), dtype=np.float32),
        mlx.OBS_STATE: np.zeros((1, 8), dtype=np.float32),
        "task": "pick the bowl\n",
    }
    action = policy.select_action(batch, noise=np.zeros((1, mlx.CHUNK_SIZE, mlx.MAX_ACTION_DIM), np.float32))
    require(action.shape == (1, 7), f"mlx smoke FAILED: action shape {action.shape}")
    require(np.isfinite(action).all(), "mlx smoke FAILED: non-finite action")
    print(f"mlx harness OK backend={policy.backend_name}")


def main():
    # Phase 1: synthetic unit tests - plain python, fail loudly.
    scorer = load_scorer()
    telemetry = _load_sibling("smoke_telemetry_rollout", "telemetry_rollout.py", stub_torch=True)
    check_success_reader(telemetry.read_success)
    check_r4(scorer.score_episode)

    # calibrate.body_class is the set-contract classifier pair_classes expects;
    # scorer.body_class takes a name->class dict and would misclassify every
    # object name as static (B1 in docs/REVIEW_telemetry.md).
    from calibrate import body_class, max_contact_force

    check_calibration_filter(max_contact_force, body_class)
    check_mlx_harness()

    # Phase 2: best-effort live rollout when the runtime deps are installed.
    missing = [name for name in LIVE_DEPS if importlib.util.find_spec(name) is None]
    if missing:
        print(f"live rollout skipped (runtime deps unavailable: {', '.join(missing)})")
        return
    _live_rollout_checks(scorer, telemetry)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
    print("SMOKE PASSED")
