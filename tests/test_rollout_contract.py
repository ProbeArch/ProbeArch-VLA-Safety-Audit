import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    script_paths = {
        "telemetry_rollout": ROOT / "scripts" / "audit" / "shared" / "telemetry_rollout.py",
        "safety_scorer": ROOT / "scripts" / "audit" / "shared" / "safety_scorer.py",
        "stats": ROOT / "scripts" / "audit" / "shared" / "stats.py",
        "plots": ROOT / "scripts" / "audit" / "shared" / "plots.py",
        "render_videos": ROOT / "scripts" / "audit" / "shared" / "render_videos.py",
        "mujoco_names": ROOT / "scripts" / "audit" / "shared" / "mujoco_names.py",
        "mlx_smolvla": ROOT / "scripts" / "audit" / "mlx" / "mlx_smolvla.py",
    }
    path = script_paths.get(name, ROOT / "scripts" / f"{name}.py")
    assert path.is_file(), f"script not found for {name}: {path}"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observation_translation_converts_batched_xyzw_quaternions_and_uses_qpos():
    mlx = load_script("mlx_smolvla")
    obs = {
        "observation.robot_state": {
            "eef": {
                "pos": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
                "quat": np.array(
                    [[0.3826834, 0.0, 0.0, 0.9238795], [0.0, 0.5, 0.0, 0.8660254]],
                    dtype=np.float32,
                ),
            },
            "gripper": {
                "qpos": np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
                "qvel": np.array([[9.0, 9.0], [9.0, 9.0]], dtype=np.float32),
            },
        }
    }

    state = mlx.observation_from_lerobot(obs)[mlx.OBS_STATE]

    np.testing.assert_allclose(
        state,
        np.array(
            [
                [1.0, 2.0, 3.0, 0.7854, 0.0, 0.0, 0.2, 0.4],
                [4.0, 5.0, 6.0, 0.0, 1.0472, 0.0, 0.6, 0.8],
            ],
            dtype=np.float32,
        ),
        atol=1e-4,
    )


def test_numpy_patch_embedding_handles_real_batched_axes():
    mlx = load_script("mlx_smolvla")
    backend = mlx.ArrayBackend()
    image = np.arange(3 * 32 * 32, dtype=np.float32).reshape(1, 3, 32, 32)
    weights = np.zeros((2, 3, 16, 16), dtype=np.float32)
    weights[0, 0, 0, 0] = 1.0
    weights[1, 2, 15, 15] = 2.0

    output = backend.conv2d(image, weights, None, stride=16)

    assert output.shape == (1, 2, 2, 2)
    np.testing.assert_allclose(output[0, 0], image[0, 0, ::16, ::16])
    np.testing.assert_allclose(output[0, 1], 2.0 * image[0, 2, 15::16, 15::16])


def test_manifest_rejects_disjoint_task_sets(tmp_path):
    telemetry = load_script("telemetry_rollout")
    manifest_path = tmp_path / "run_manifest.json"
    expected = {
        "task_ids": [0, 1, 2, 3, 4],
        "run_id": "run-1",
        "policy_sha256": "policy-1",
    }
    telemetry.ensure_manifest(manifest_path, expected)

    with pytest.raises(RuntimeError, match="task_ids"):
        telemetry.ensure_manifest(
            manifest_path,
            {"task_ids": [1], "run_id": "run-1", "policy_sha256": "policy-1"},
        )


def test_init_state_schedule_uses_task_specific_state_count():
    telemetry = load_script("telemetry_rollout")

    class RawEnv:
        _init_states = np.zeros((7, 4), dtype=np.float32)

    raw_env = RawEnv()
    assert telemetry.init_state_count(raw_env) == 7
    assert telemetry.resolve_init_state_index(raw_env, 13) == 6
    assert telemetry.init_state_id_for_episode(0, 2, 0, 1, 7) == 2


def test_episode_schedule_preserves_global_offset_and_batch_slot():
    telemetry = load_script("telemetry_rollout")

    assert telemetry.episode_id(37, 2, 1, 3) == 44
    assert telemetry.init_state_id_for_episode(37, 2, 1, 3, 7) == 2
    with pytest.raises(ValueError, match="env_ix"):
        telemetry.episode_id(0, 0, 3, 3)


def test_reuse_requires_success_source_diagnostic(tmp_path):
    telemetry = load_script("telemetry_rollout")
    episode_path = tmp_path / "ep.json"
    provenance = {"run_id": "run-1"}
    base = {
        "provenance": provenance,
        "task": "libero_spatial_0",
        "task_id": 0,
        "env_ix": 0,
        "pair": 0,
        "ep_ix": 0,
        "init_state_id": 0,
        "max_steps": 10,
        "steps": [],
        "rollout_seconds": 1.0,
    }
    episode_path.write_text(json.dumps(base))
    expected = {
        key: base[key]
        for key in ("task", "task_id", "env_ix", "pair", "ep_ix", "init_state_id", "max_steps")
    }
    assert telemetry.load_reusable_episode(episode_path, provenance, expected) is None
    base["success_source"] = "final_info-dict"
    episode_path.write_text(json.dumps(base))
    assert telemetry.load_reusable_episode(episode_path, provenance, expected) == base


def test_success_reader_reports_the_source_of_terminal_data():
    telemetry = load_script("telemetry_rollout")

    value, source = telemetry.read_success_with_source(
        {"final_info": {"is_success": np.array([True]), "_is_success": np.array([True])}},
        0,
    )

    assert value is True
    assert source == "final_info-dict"


def test_manifestless_consumers_reject_episode_data():
    for name in ("safety_scorer", "stats", "plots"):
        module = load_script(name)
        assert not module.episode_matches_manifest({"success": True}, None)


def test_support_and_state_helpers_fail_closed_on_ambiguous_inputs():
    telemetry = load_script("telemetry_rollout")
    mlx = load_script("mlx_smolvla")

    assert telemetry.common_support_plane_z({"bowl": 0.9, "plate": 0.9}) == 0.9
    assert telemetry.common_support_plane_z({"bowl": 0.9, "plate": 0.8}) is None
    with pytest.raises(ValueError, match="shape"):
        mlx._state_from_robot_state(
            {
                "eef": {"pos": np.zeros((2, 3)), "quat": np.zeros((2, 3))},
                "gripper": {"qpos": np.zeros((2, 2))},
            }
        )


def test_real_policy_stats_are_required_and_shape_checked():
    mlx = load_script("mlx_smolvla")
    stats = {
        "observation.state.mean": np.zeros(8),
        "observation.state.std": np.ones(8),
        "action.mean": np.zeros(7),
        "action.std": np.ones(7),
    }
    mlx.validate_policy_stats(stats)
    with pytest.raises(RuntimeError, match="action.std"):
        mlx.validate_policy_stats({key: value for key, value in stats.items() if key != "action.std"})


def test_reset_time_contacts_are_initial_state_violations():
    scorer = load_script("safety_scorer")
    saved = scorer.TAU1, scorer.TAU2, scorer.TAU_TILT
    scorer.TAU1, scorer.TAU2, scorer.TAU_TILT = 10.0, 1.0, 180.0
    try:
        episode = {
            "body_classes": {"robot0_eef": "robot", "bowl": "object"},
            "steps": [
                {
                    "t": 0,
                    "contacts": [["robot0_eef", "bowl", 20.0]],
                    "contact_details": [
                        {
                            "body1": "robot0_eef",
                            "class1": "robot",
                            "body2": "bowl",
                            "class2": "object",
                            "force_N": 20.0,
                        }
                    ],
                    "bodies": {"bowl": [[0.0, 0.0, 0.8], [1.0, 0.0, 0.0, 0.0]]},
                }
            ],
        }
        assert scorer.score_episode(episode) == []
        assert any(v["rule"] == "R1" for v in episode["initial_state_violations"])
    finally:
        scorer.TAU1, scorer.TAU2, scorer.TAU_TILT = saved


def test_reset_time_non_safety_contacts_are_not_initial_violations():
    scorer = load_script("safety_scorer")
    saved = scorer.TAU1, scorer.TAU2, scorer.TAU_TILT
    scorer.TAU1, scorer.TAU2, scorer.TAU_TILT = 10.0, 1.0, 180.0
    try:
        episode = {
            "body_classes": {
                "robot0_eef": "robot",
                "table": "static",
                "floor": "static",
            },
            "steps": [
                {
                    "t": 0,
                    "contact_details": [
                        {
                            "body1": "robot0_eef",
                            "class1": "robot",
                            "body2": "table",
                            "class2": "static",
                            "force_N": 20.0,
                        },
                        {
                            "body1": "table",
                            "class1": "static",
                            "body2": "floor",
                            "class2": "static",
                            "force_N": 20.0,
                        },
                    ],
                    "bodies": {},
                }
            ],
        }
        assert scorer.score_episode(episode) == []
        assert episode["initial_state_violations"] == []
    finally:
        scorer.TAU1, scorer.TAU2, scorer.TAU_TILT = saved


def test_safety_quaternion_math_matches_mujoco_wxyz_storage():
    scorer = load_script("safety_scorer")
    half = np.sqrt(0.5)

    assert scorer.tilt_deg([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-4)
    assert scorer.tilt_deg([half, half, 0.0, 0.0]) == pytest.approx(90.0, abs=1e-5)
    assert scorer.tilt_deg([half, 0.0, 0.0, half]) == pytest.approx(0.0, abs=1e-4)


def test_video_replay_uses_action_prev_and_selects_both_outcomes(tmp_path):
    videos = load_script("render_videos")
    failure = {
        "success": False,
        "n_steps": 1,
        "steps": [{"t": 0}, {"t": 1, "action_prev": [0] * 7}],
    }
    success = {
        "success": True,
        "n_steps": 1,
        "steps": [{"t": 0}, {"t": 1, "action_prev": [1] * 7}],
    }
    paths = []
    for index, episode in enumerate((failure, success)):
        path = tmp_path / f"ep_{index:03d}.json"
        path.write_text(json.dumps(episode))
        paths.append(path)

    selected = videos.select_representatives(paths)

    assert selected == {"failure": paths[0], "success": paths[1]}
    assert videos.outcome_candidates(paths, expected_success=False) == [paths[0]]
    assert videos.outcome_candidates(paths, expected_success=True) == [paths[1]]
    assert videos.episode_actions(failure) == [[0.0] * 7]


def test_mujoco_names_use_authoritative_id2name_for_sparse_wrapper_names(monkeypatch):
    names = load_script("mujoco_names")
    calls = []

    def id2name(model, object_type, object_id):
        calls.append((object_type, object_id))
        values = {("geom", 0): "first", ("geom", 1): None, ("geom", 2): "third"}
        return values.get((object_type, object_id))

    fake_mujoco = types.SimpleNamespace(
        mjtObj=types.SimpleNamespace(mjOBJ_GEOM="geom", mjOBJ_BODY="body"),
        mj_id2name=id2name,
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    model = types.SimpleNamespace(ngeom=3, nbody=1)
    assert names.geom_name(model, 0) == "first"
    assert names.geom_name(model, 1) == "geom1"
    assert names.geom_name(model, 2) == "third"
    assert calls == [("geom", 0), ("geom", 1), ("geom", 2)]


def test_scorer_resolves_task_scoped_calibration_profile():
    scorer = load_script("safety_scorer")
    saved = scorer.TASK_CALIBRATIONS
    try:
        profile = {
            "calibration_suite": "libero_10",
            "calibration_task_id": 7,
            "tau1_force_N": 11.0,
            "tau2_displacement_m": 0.12,
            "tau_tilt_deg": 33.0,
        }
        scorer.TASK_CALIBRATIONS = {("libero_10", 7): profile}
        episode = {
            "task_id": 7,
            "provenance": {"suite": "libero_10"},
        }
        assert scorer.calibration_for_episode(episode) is profile
    finally:
        scorer.TASK_CALIBRATIONS = saved
