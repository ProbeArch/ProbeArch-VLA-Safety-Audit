#!/usr/bin/env python3
"""Verify the frozen rollout, provenance, calibration, and derived artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path(os.environ.get("AUDIT_DIR", Path.home() / "audit")),
    )
    args = parser.parse_args()
    audit = args.audit_dir.resolve()
    rollouts = audit / "rollouts"
    errors: list[str] = []

    def load(path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path.relative_to(audit)}: {type(exc).__name__}: {exc}")
            return None

    freeze = load(audit / "dataset_freeze.json")
    stats = load(audit / "stats.json")
    summary = load(audit / "safety_summary.json")
    matrix = load(audit / "confusion_matrix.json")
    if freeze is None or stats is None or summary is None or matrix is None:
        raise SystemExit(json.dumps({"passed": False, "errors": errors}, indent=2))

    frozen = freeze.get("episodes", [])
    if freeze.get("n_episodes") != 200 or len(frozen) != 200:
        errors.append("freeze does not contain exactly 200 episodes")
    if freeze.get("errors"):
        errors.extend([f"freeze error: {item}" for item in freeze["errors"]])
    hash_mismatches = 0
    for item in frozen:
        path = rollouts / item["file"]
        if not path.is_file():
            errors.append(f"missing frozen file: {item['file']}")
            continue
        if path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            hash_mismatches += 1
            errors.append(f"frozen hash/size mismatch: {item['file']}")

    task_names = [f"libero_spatial_{task_id}" for task_id in range(10)]
    for task_name in task_names:
        task_dir = rollouts / task_name
        files = sorted(task_dir.glob("ep_*.json")) if task_dir.is_dir() else []
        if len(files) != 20:
            errors.append(f"{task_name}: expected 20 episodes, found {len(files)}")
        indices = []
        for path in files:
            payload = load(path)
            if payload is None:
                continue
            indices.append(payload.get("ep_ix"))
            for field in ("task", "task_id", "ep_ix", "success", "n_steps", "provenance", "task_aware"):
                if field not in payload:
                    errors.append(f"{path.relative_to(audit)}: missing {field}")
            if payload.get("task_aware", {}).get("spec", {}).get("status") != "resolved":
                errors.append(f"{path.relative_to(audit)}: task-aware spec not resolved")
        if indices != list(range(20)):
            errors.append(f"{task_name}: episode indices are not 0..19")

    root_manifest = load(rollouts / "run_manifest.json")
    run_ids = set()
    policy_hashes = set()
    if root_manifest:
        run_ids.add(root_manifest.get("run_id"))
        policy_hashes.add(root_manifest.get("policy_sha256"))
        if root_manifest.get("suite") != "libero_spatial":
            errors.append("root manifest suite mismatch")
        if root_manifest.get("n_pairs") != 20 or root_manifest.get("n_envs") != 1:
            errors.append("root manifest episode configuration mismatch")
        if root_manifest.get("policy_backend") != "cuda":
            errors.append("root manifest backend is not cuda")
    for task_name in task_names:
        manifest = load(rollouts / task_name / "run_manifest.json")
        if manifest:
            run_ids.add(manifest.get("run_id"))
            policy_hashes.add(manifest.get("policy_sha256"))
    if len(run_ids) != 1:
        errors.append(f"provenance run_id mismatch: {sorted(run_ids)}")
    if len(policy_hashes) != 1:
        errors.append(f"provenance policy hash mismatch: {sorted(policy_hashes)}")

    calibration_index = load(audit / "calibration" / "index.json")
    calibration_profiles = 0
    if calibration_index:
        for profile in calibration_index.get("profiles", []):
            path = audit / "calibration" / profile["file"]
            if not path.is_file():
                errors.append(f"missing calibration profile: {profile['file']}")
            elif file_sha256(path) != profile["sha256"]:
                errors.append(f"calibration hash mismatch: {profile['file']}")
            else:
                calibration_profiles += 1
    if calibration_profiles != 10:
        errors.append(f"expected 10 validated calibration profiles, found {calibration_profiles}")

    if stats.get("n_episodes") != 200:
        errors.append("stats n_episodes is not 200")
    if matrix.get("n_episodes") != 200:
        errors.append("confusion matrix n_episodes is not 200")
    if len(summary.get("tasks", {})) != 10:
        errors.append("safety summary does not contain 10 tasks")

    result = {
        "passed": not errors,
        "n_episodes": len(frozen),
        "hash_mismatches": hash_mismatches,
        "task_directories": len(task_names),
        "calibration_profiles_verified": calibration_profiles,
        "run_ids": sorted(run_ids),
        "policy_sha256": sorted(policy_hashes),
        "stats_n_episodes": stats.get("n_episodes"),
        "matrix_n_episodes": matrix.get("n_episodes"),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
