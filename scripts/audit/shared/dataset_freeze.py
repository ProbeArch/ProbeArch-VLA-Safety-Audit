#!/usr/bin/env python3
"""Create a content-addressed freeze manifest for an audit rollout set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path(os.environ.get("AUDIT_DIR", Path.home() / "audit")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = args.audit_dir.resolve()
    rollouts = audit / "rollouts"
    if not rollouts.is_dir():
        raise SystemExit(f"missing rollouts directory: {rollouts}")

    task_dirs = sorted(
        path for path in rollouts.iterdir() if path.is_dir() and path.name.startswith("libero_")
    )
    episodes: list[dict] = []
    task_counts: dict[str, dict] = {}
    run_ids: set[str] = set()
    policy_hashes: set[str] = set()
    calibration_hashes: set[str] = set()
    provenance_schemas: set[str] = set()
    errors: list[str] = []
    total_bytes = 0

    for task_dir in task_dirs:
        task_files = sorted(task_dir.glob("ep_*.json"))
        successes = 0
        indices: list[int] = []
        for episode_path in task_files:
            relative = episode_path.relative_to(rollouts).as_posix()
            try:
                payload = json.loads(episode_path.read_text(encoding="utf-8"))
                stat = episode_path.stat()
                provenance = payload.get("provenance", {})
                task = payload.get("task", task_dir.name)
                task_id = int(payload["task_id"])
                ep_ix = int(payload["ep_ix"])
                success = bool(payload["success"])
                run_id = str(provenance.get("run_id", payload.get("run_id")))
                policy_sha = str(provenance.get("policy_sha256", payload.get("policy_sha256")))
                calibration_sha = str(
                    provenance.get("calibration_sha256", payload.get("calibration_sha256"))
                )
                schema = provenance.get("harness_schema_version")
                if schema:
                    provenance_schemas.add(str(schema))
                if task != task_dir.name:
                    errors.append(f"{relative}: task field {task!r} != directory {task_dir.name!r}")
                if ep_ix in indices:
                    errors.append(f"{relative}: duplicate ep_ix {ep_ix}")
                indices.append(ep_ix)
                if success:
                    successes += 1
                run_ids.add(run_id)
                policy_hashes.add(policy_sha)
                calibration_hashes.add(calibration_sha)
                total_bytes += stat.st_size
                episodes.append(
                    {
                        "file": relative,
                        "sha256": sha256(episode_path),
                        "bytes": stat.st_size,
                        "task": task,
                        "task_id": task_id,
                        "ep_ix": ep_ix,
                        "success": success,
                        "steps": int(payload.get("n_steps", payload.get("steps", 0))),
                        "run_id": run_id,
                        "policy_sha256": policy_sha,
                        "calibration_sha256": calibration_sha,
                    }
                )
            except Exception as exc:  # pragma: no cover - exercised by corrupt inputs
                errors.append(f"{relative}: {type(exc).__name__}: {exc}")

        task_counts[task_dir.name] = {
            "task_id": int(task_dir.name.rsplit("_", 1)[-1]),
            "episode_indices": sorted(indices),
            "successes": successes,
            "files": len(task_files),
        }
        if sorted(indices) != list(range(len(task_files))):
            errors.append(f"{task_dir.name}: episode indices are not contiguous from zero")

    def manifest_record(path: Path) -> dict:
        return {"file": path.relative_to(rollouts).as_posix(), "sha256": sha256(path)}

    root_manifest = rollouts / "run_manifest.json"
    task_manifests = {
        path.relative_to(rollouts).as_posix(): sha256(path)
        for path in sorted(rollouts.glob("libero_*/run_manifest.json"))
    }
    if not root_manifest.is_file():
        errors.append("missing root run_manifest.json")

    document = {
        "schema_version": "probearch-dataset-freeze-v1",
        "dataset_id": f"{audit.name}-episodes",
        "frozen_unix": time.time(),
        "source_rollouts": str(rollouts),
        "n_episodes": len(episodes),
        "total_bytes": total_bytes,
        "task_counts": task_counts,
        "run_ids": sorted(run_ids),
        "policy_sha256": sorted(policy_hashes),
        "calibration_sha256": sorted(calibration_hashes),
        "provenance_schemas": sorted(provenance_schemas),
        "root_run_manifest": manifest_record(root_manifest) if root_manifest.is_file() else None,
        "task_run_manifests": task_manifests,
        "episodes": sorted(episodes, key=lambda item: (item["task"], item["ep_ix"])),
        "errors": errors,
    }
    output = audit / "dataset_freeze.json"
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "n_episodes": len(episodes), "errors": errors}, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
