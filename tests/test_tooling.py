import json
from pathlib import Path

import pytest

from probearch.cli import _validate_config
from scripts.audit.shared.robustness_manifest import build
from scripts.analysis.check_schemas import check_schema
from scripts.analysis.label_agreement import (
    binary_metrics,
    validate_annotation_splits,
    validate_double_annotation,
)
from scripts.analysis.matrix_ablation import matrix
from scripts.analysis.threshold_sensitivity import matrix_for
from scripts.audit.shared.stats import evidence_available, evidence_coverage


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


def test_full_robustness_manifest_covers_all_declared_conditions():
    config = json.loads(Path("configs/robustness_full.json").read_text(encoding="utf-8"))
    manifest = build(config)
    assert len(manifest["conditions"]) == 8
    assert len(manifest["pairs"]) * len(manifest["conditions"]) == 64


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


def test_threshold_sensitivity_matrix_tracks_not_evaluated():
    result = matrix_for(["safe_success", "unsafe_failure", "not_evaluated"])
    assert result["recorded_success"] == {"safe": 1, "unsafe": 0, "not_evaluated": 0}
    assert result["recorded_failure"] == {"safe": 0, "unsafe": 1, "not_evaluated": 1}


def test_label_metrics_report_unsafe_precision_recall_and_coverage():
    predicted = {"a": "UNSAFE_SUCCESS", "b": "SAFE_FAILURE", "c": "UNSAFE_FAILURE"}
    reference = {"a": "UNSAFE_SUCCESS", "b": "UNSAFE_FAILURE", "c": "SAFE_FAILURE"}
    result = binary_metrics(predicted, reference)
    assert result["evaluated_episodes"] == 3
    assert result["confusion"] == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 0,
    }
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)


def test_strict_double_annotation_rejects_duplicate_or_unpaired_rows():
    complete = [
        {"annotator_id": "a", "episode_path": "ep0", "label": "SAFE_SUCCESS"},
        {"annotator_id": "b", "episode_path": "ep0", "label": "SAFE_SUCCESS"},
    ]
    assert validate_double_annotation(complete) == ("a", "b")
    with pytest.raises(ValueError, match="duplicate"):
        validate_double_annotation(complete + [complete[0]])
    with pytest.raises(ValueError, match="same episode set"):
        validate_double_annotation(complete + [{"annotator_id": "a", "episode_path": "ep1", "label": "SAFE_FAILURE"}])


def test_strict_annotation_split_validation_requires_both_splits():
    rows = [
        {"annotator_id": "a", "episode_path": "ep0", "label": "SAFE_SUCCESS", "split": "development"},
        {"annotator_id": "b", "episode_path": "ep0", "label": "SAFE_SUCCESS", "split": "development"},
    ]
    with pytest.raises(ValueError, match="development and heldout"):
        validate_annotation_splits(rows)


def test_versioned_json_schemas_have_consistent_required_fields():
    for path in Path("schemas").glob("*.schema.json"):
        assert check_schema(path) == []


def test_annotation_manifest_is_blinded_and_balanced():
    import csv

    with Path("annotations/sample_manifest.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 100
    assert set(reader.fieldnames or []) == {
        "episode_path", "source_sha256", "suite", "task_id", "episode", "split",
    }
    assert all("candidate" not in row for row in rows)
    assert all("stratum" not in row for row in rows)
    assert all("recorded_success" not in row for row in rows)
    assert all("review_priority" not in row for row in rows)
    assert {row["split"] for row in rows} == {"development", "heldout"}


def test_evidence_coverage_distinguishes_contacts_from_pose_telemetry():
    complete = {
        "body_classes": {"robot": "robot", "object": "object"},
        "steps": [{"bodies": {}, "contact_details": []}],
    }
    assert evidence_available(complete, "R1")
    assert evidence_available(complete, "R2")
    coverage = evidence_coverage([complete, {"steps": []}])
    assert coverage["R1"]["episodes_with_evidence"] == 1
    assert coverage["R2"]["episodes_with_evidence"] == 1
