import json
from pathlib import Path

from probearch.cli import _validate_config
from scripts.audit.shared.robustness_manifest import build
from scripts.analysis.matrix_ablation import matrix


def load_pilot():
    return json.loads(
        Path("configs/robustness_pilot.yaml").read_text(encoding="utf-8")
    )


def test_robustness_pilot_config_is_rtx3050_safe_and_valid():
    config = load_pilot()
    assert _validate_config(config) == []
    assert config["resources"]["n_envs"] == 1


def test_robustness_manifest_preserves_matched_pairs():
    manifest = build(load_pilot())
    assert manifest["schema_version"] == "probearch-robustness-manifest-v1"
    assert len(manifest["pairs"]) == 8
    assert len(manifest["pairs"]) * len(manifest["conditions"]) == 40
    assert all(pair["conditions"][0] == "clean" for pair in manifest["pairs"])


def test_matrix_ablation_keeps_diagnostics_separate():
    episodes = [
        {
            "success": True,
            "safety_events": [],
            "task_aware_events": [],
            "task_aware": {"events": [], "diagnostic_events": [{"rule": "R5"}]},
        }
    ]
    assert matrix(episodes, "primary")["counts"]["recorded_success"]["safe"] == 1
    assert matrix(episodes, "include_diagnostics")["counts"]["recorded_success"]["unsafe"] == 1
