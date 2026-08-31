#!/usr/bin/env python3
"""Verify rollout hashes, provenance, calibration, aggregates, and videos.

The original verifier was tied to one 200-episode LIBERO-Spatial run. This
version derives the suite, task names, episode counts, and indices from the
frozen manifest so the same checks apply to pilots and additional policies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes root: {relative!r}") from exc
    return candidate


def _count_matrix(matrix: dict) -> int:
    return sum(
        int(value)
        for row in (matrix.get("counts") or {}).values()
        if isinstance(row, dict)
        for value in row.values()
        if isinstance(value, int)
    )


def verify(
    result_dir: Path,
    rollouts_dir: Path,
    video_source_rollouts: Path | None = None,
) -> dict:
    result_dir = result_dir.resolve()
    rollouts_dir = rollouts_dir.resolve()
    errors: list[str] = []

    def load(path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # diagnostic boundary for malformed artifacts
            try:
                label = path.relative_to(result_dir)
            except ValueError:
                label = path
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return None

    freeze = load(result_dir / "dataset_freeze.json")
    stats = load(result_dir / "stats.json")
    summary = load(result_dir / "safety_summary.json")
    matrix = load(result_dir / "confusion_matrix.json")
    result_manifest = load(result_dir / "run_manifest.json")
    if any(value is None for value in (freeze, stats, summary, matrix, result_manifest)):
        return {"passed": False, "errors": errors}

    frozen = freeze.get("episodes")
    if not isinstance(frozen, list) or not frozen:
        errors.append("dataset freeze contains no episodes")
        frozen = []
    expected_n = len(frozen)
    if freeze.get("n_episodes") != expected_n:
        errors.append(
            f"freeze n_episodes {freeze.get('n_episodes')!r} != episode list {expected_n}"
        )
    if freeze.get("errors"):
        errors.extend([f"freeze error: {item}" for item in freeze["errors"]])

    suite = result_manifest.get("suite")
    if not isinstance(suite, str) or not suite:
        errors.append("result run manifest has no suite")
    task_counts = freeze.get("task_counts")
    if not isinstance(task_counts, dict) or not task_counts:
        errors.append("freeze has no task_counts")
        task_counts = {}

    hash_mismatches = 0
    source_successes = 0
    for item in frozen:
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            errors.append(f"malformed frozen episode entry: {item!r}")
            continue
        relative = item["file"]
        source_successes += int(bool(item.get("success")))
        try:
            path = _safe_child(rollouts_dir, relative)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.is_file():
            errors.append(f"missing frozen file: {relative}")
            continue
        if path.stat().st_size != item.get("bytes") or file_sha256(path) != item.get("sha256"):
            hash_mismatches += 1
            errors.append(f"frozen hash/size mismatch: {relative}")

    root_manifest_meta = freeze.get("root_run_manifest") or {}
    source_root_manifest = rollouts_dir / str(
        root_manifest_meta.get("file", "run_manifest.json")
    )
    published_manifest = result_dir / "run_manifest.json"
    if not source_root_manifest.is_file():
        errors.append("source root run manifest is missing")
    else:
        expected_hash = root_manifest_meta.get("sha256")
        if expected_hash and file_sha256(source_root_manifest) != expected_hash:
            errors.append("source root run manifest hash mismatch")
        if published_manifest.is_file() and (
            file_sha256(source_root_manifest) != file_sha256(published_manifest)
        ):
            errors.append("published run manifest does not match source run manifest")

    run_ids: set[str] = set()
    policy_hashes: set[str] = set()
    raw_run_id = result_manifest.get("run_id")
    raw_policy_hash = result_manifest.get("policy_sha256")
    if isinstance(raw_run_id, str):
        run_ids.add(raw_run_id)
    else:
        errors.append("result run manifest has no run_id")
    if isinstance(raw_policy_hash, str):
        policy_hashes.add(raw_policy_hash)
    else:
        errors.append("result run manifest has no policy_sha256")

    task_manifest_hashes: dict[str, str] = {}
    raw_task_manifest_meta = freeze.get("task_run_manifests") or []
    if isinstance(raw_task_manifest_meta, list):
        for value in raw_task_manifest_meta:
            if isinstance(value, dict):
                task_manifest_hashes.update(
                    {str(key): str(digest) for key, digest in value.items()}
                )
    elif isinstance(raw_task_manifest_meta, dict):
        task_manifest_hashes.update(
            {str(key): str(digest) for key, digest in raw_task_manifest_meta.items()}
        )

    checked_episode_payloads = 0
    for task_name, task_meta in sorted(task_counts.items()):
        if not isinstance(task_meta, dict):
            errors.append(f"{task_name}: malformed task_counts entry")
            continue
        task_dir = rollouts_dir / task_name
        files = sorted(task_dir.glob("ep_*.json")) if task_dir.is_dir() else []
        expected_files = int(task_meta.get("files", 0))
        if len(files) != expected_files:
            errors.append(
                f"{task_name}: expected {expected_files} episodes, found {len(files)}"
            )
        expected_indices = task_meta.get("episode_indices") or []
        actual_indices = []
        for path in files:
            payload = load(path)
            if not isinstance(payload, dict):
                continue
            checked_episode_payloads += 1
            actual_indices.append(payload.get("ep_ix"))
            for field in (
                "task",
                "task_id",
                "ep_ix",
                "success",
                "n_steps",
                "provenance",
                "success_source",
            ):
                if field not in payload:
                    errors.append(f"{task_name}/{path.name}: missing {field}")
            provenance = payload.get("provenance") or {}
            if provenance.get("run_id") != raw_run_id:
                errors.append(f"{task_name}/{path.name}: run_id mismatch")
            if provenance.get("policy_sha256") != raw_policy_hash:
                errors.append(f"{task_name}/{path.name}: policy hash mismatch")
            task_aware = payload.get("task_aware")
            if isinstance(task_aware, dict):
                if (task_aware.get("spec") or {}).get("status") != "resolved":
                    errors.append(f"{task_name}/{path.name}: task-aware spec not resolved")
                if task_aware.get("evidence_status", "evaluated") != "evaluated":
                    errors.append(
                        f"{task_name}/{path.name}: task-aware evidence not evaluated"
                    )
        if actual_indices != expected_indices:
            errors.append(
                f"{task_name}: episode indices {actual_indices!r} != frozen {expected_indices!r}"
            )
        task_manifest = task_dir / "run_manifest.json"
        relative_manifest = f"{task_name}/run_manifest.json"
        if not task_manifest.is_file():
            errors.append(f"{task_name}: missing run_manifest.json")
        else:
            expected_manifest_hash = task_manifest_hashes.get(relative_manifest)
            if expected_manifest_hash and file_sha256(task_manifest) != expected_manifest_hash:
                errors.append(f"{task_name}: run manifest hash mismatch")
            manifest = load(task_manifest)
            if isinstance(manifest, dict):
                if isinstance(manifest.get("run_id"), str):
                    run_ids.add(manifest["run_id"])
                if isinstance(manifest.get("policy_sha256"), str):
                    policy_hashes.add(manifest["policy_sha256"])

    if len(run_ids) != 1:
        errors.append(f"provenance run_id mismatch: {sorted(run_ids)}")
    if len(policy_hashes) != 1:
        errors.append(f"provenance policy hash mismatch: {sorted(policy_hashes)}")

    calibration_index = load(result_dir / "calibration" / "index.json")
    calibration_profiles = 0
    if isinstance(calibration_index, dict):
        if calibration_index.get("suite") != suite:
            errors.append("calibration index suite mismatch")
        for profile in calibration_index.get("profiles", []):
            if not isinstance(profile, dict) or not isinstance(profile.get("file"), str):
                errors.append(f"malformed calibration index entry: {profile!r}")
                continue
            path = result_dir / "calibration" / profile["file"]
            if not path.is_file():
                errors.append(f"missing calibration profile: {profile['file']}")
            elif file_sha256(path) != profile.get("sha256"):
                errors.append(f"calibration hash mismatch: {profile['file']}")
            else:
                payload = load(path)
                if isinstance(payload, dict):
                    if payload.get("calibration_suite") != suite:
                        errors.append(f"{profile['file']}: calibration suite mismatch")
                    if payload.get("calibration_task_id") != profile.get("task_id"):
                        errors.append(f"{profile['file']}: calibration task mismatch")
                calibration_profiles += 1
    if calibration_profiles != len(task_counts):
        errors.append(
            f"expected {len(task_counts)} validated calibration profiles, "
            f"found {calibration_profiles}"
        )

    if stats.get("n_episodes") != expected_n:
        errors.append(f"stats n_episodes {stats.get('n_episodes')!r} != {expected_n}")
    if stats.get("successes") != source_successes:
        errors.append(
            f"stats successes {stats.get('successes')!r} != frozen {source_successes}"
        )
    if matrix.get("n_episodes") != expected_n:
        errors.append(f"matrix n_episodes {matrix.get('n_episodes')!r} != {expected_n}")
    if matrix.get("semantics_version") != "probearch-task-semantics-v2":
        errors.append("matrix semantics version mismatch")
    if matrix.get("measurement_contract_version") != "probearch-measurement-v2":
        errors.append("matrix measurement-contract version mismatch")
    if _count_matrix(matrix) != expected_n:
        errors.append(f"matrix cell total {_count_matrix(matrix)} != {expected_n}")
    matrix_counts = matrix.get("counts") or {}
    matrix_successes = sum(
        value
        for value in (matrix_counts.get("recorded_success") or {}).values()
        if isinstance(value, int)
    )
    if matrix_successes != source_successes:
        errors.append(f"matrix success row {matrix_successes} != frozen {source_successes}")
    if len(summary.get("tasks", {})) != len(task_counts):
        errors.append(
            f"safety summary task count {len(summary.get('tasks', {}))} "
            f"!= {len(task_counts)}"
        )
    outcomes = (summary.get("task_aware") or {}).get("outcomes") or {}
    expected_outcome_cells = {
        "safe_success": (matrix_counts.get("recorded_success") or {}).get(
            "task_aware_safe", 0
        ),
        "unsafe_success": (matrix_counts.get("recorded_success") or {}).get(
            "task_aware_unsafe", 0
        ),
        "safe_failure": (matrix_counts.get("recorded_failure") or {}).get(
            "task_aware_safe", 0
        ),
        "unsafe_failure": (matrix_counts.get("recorded_failure") or {}).get(
            "task_aware_unsafe", 0
        ),
        "not_evaluated": sum(
            (matrix_counts.get(row) or {}).get("not_evaluated", 0)
            for row in ("recorded_success", "recorded_failure")
        ),
    }
    for name, count in expected_outcome_cells.items():
        if int(outcomes.get(name, 0)) != int(count):
            errors.append(
                f"task-aware outcome {name}={outcomes.get(name, 0)!r} "
                f"!= matrix {count}"
            )

    video_records_verified = 0
    video_index_path = result_dir / "videos" / "index.json"
    if video_index_path.is_file():
        video_index = load(video_index_path)
        if isinstance(video_index, dict):
            if video_index.get("run_id") != raw_run_id:
                errors.append("video index run_id mismatch")
            if video_index.get("suite") != suite:
                errors.append("video index suite mismatch")
            for record in video_index.get("records", []):
                if not isinstance(record, dict) or not isinstance(record.get("file"), str):
                    errors.append(f"malformed video record: {record!r}")
                    continue
                video_path = result_dir / "videos" / record["file"]
                if not video_path.is_file():
                    errors.append(f"missing video: {record['file']}")
                    continue
                if file_sha256(video_path) != record.get("video_sha256"):
                    errors.append(f"video hash mismatch: {record['file']}")
                    continue
                if video_source_rollouts is not None:
                    episode_path = (
                        video_source_rollouts.resolve()
                        / str(record.get("task"))
                        / f"ep_{int(record.get('episode')):03d}.json"
                    )
                    if not episode_path.is_file():
                        errors.append(f"missing video source episode: {episode_path}")
                    elif file_sha256(episode_path) != record.get("source_episode_sha256"):
                        errors.append(
                            f"video source episode hash mismatch: {record['file']}"
                        )
                video_records_verified += 1

    return {
        "passed": not errors,
        "suite": suite,
        "n_episodes": expected_n,
        "source_successes": source_successes,
        "hash_mismatches": hash_mismatches,
        "task_directories": len(task_counts),
        "episode_payloads_checked": checked_episode_payloads,
        "calibration_profiles_verified": calibration_profiles,
        "video_records_verified": video_records_verified,
        "run_ids": sorted(run_ids),
        "policy_sha256": sorted(policy_hashes),
        "stats_n_episodes": stats.get("n_episodes"),
        "matrix_n_episodes": matrix.get("n_episodes"),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        help="legacy shorthand: use this directory and its rollouts/ child",
    )
    parser.add_argument("--result-dir", type=Path, help="curated result package")
    parser.add_argument("--rollouts-dir", type=Path, help="raw source rollouts directory")
    parser.add_argument(
        "--video-source-rollouts",
        type=Path,
        help="optional rollout root for source hashes in videos/index.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy = args.audit_dir or Path(
        os.environ.get("AUDIT_DIR", Path.home() / "audit")
    )
    result_dir = (args.result_dir or legacy).resolve()
    rollouts_dir = (args.rollouts_dir or (legacy / "rollouts")).resolve()
    result = verify(result_dir, rollouts_dir, args.video_source_rollouts)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
