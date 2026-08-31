#!/usr/bin/env python3
"""Validate and materialize a matched robustness experiment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError(
                "YAML configuration requires PyYAML; install probearch-audit[yaml]"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("configuration must be an object")
    return value


def build(config: dict) -> dict:
    errors = []
    if config.get("schema_version") != "probearch-robustness-config-v1":
        errors.append("unsupported schema_version")
    conditions = config.get("conditions") or []
    names = [item.get("name") for item in conditions if isinstance(item, dict)]
    if "clean" not in names or len(names) != len(set(names)):
        errors.append("conditions need one unique clean reference")
    resources = config.get("resources") or {}
    if resources.get("n_envs") != 1:
        errors.append("n_envs must remain 1 for the RTX-3050 pilot")
    if errors:
        raise ValueError("; ".join(errors))
    tasks = config.get("tasks") or {}
    selected_tasks = sorted({int(task) for group in tasks.values() for task in group})
    n = int(config["episodes_per_condition"])
    pairs = [
        {
            "pair_id": f"{config['suite']}-{task:02d}-{episode:03d}",
            "suite": config["suite"],
            "task_id": task,
            "episode": episode,
            "conditions": names,
        }
        for task in selected_tasks
        for episode in range(n)
    ]
    return {
        "schema_version": "probearch-robustness-manifest-v1",
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "suite": config["suite"],
        "policy": config.get("policy"),
        "matched_initial_states": bool(config.get("matched_initial_states")),
        "conditions": conditions,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(load(args.config))
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pairs": len(manifest["pairs"])}, indent=2))


if __name__ == "__main__":
    main()
