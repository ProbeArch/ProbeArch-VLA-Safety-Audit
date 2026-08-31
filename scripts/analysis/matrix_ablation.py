#!/usr/bin/env python3
"""Compute transparent outcome ablations from scored episode JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def episodes(root: Path) -> list[dict]:
    values = []
    for path in sorted(root.glob("*/ep_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            values.append(value)
    if not values:
        raise ValueError(f"no episode files found under {root}")
    return values


def matrix(eps: list[dict], mode: str) -> dict:
    counts = {
        "recorded_success": {"safe": 0, "unsafe": 0},
        "recorded_failure": {"safe": 0, "unsafe": 0},
    }
    for ep in eps:
        row = "recorded_success" if ep.get("success") else "recorded_failure"
        if mode == "generic":
            unsafe = bool(ep.get("safety_events"))
        elif mode == "include_diagnostics":
            task_aware = ep.get("task_aware") or {}
            unsafe = bool(task_aware.get("events") or task_aware.get("diagnostic_events"))
        else:
            unsafe = bool(ep.get("task_aware_events"))
        counts[row]["unsafe" if unsafe else "safe"] += 1
    return {
        "schema_version": "probearch-matrix-ablation-v1",
        "mode": mode,
        "n_episodes": len(eps),
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    eps = episodes(args.rollouts)
    result = {
        mode: matrix(eps, mode)
        for mode in ("primary", "include_diagnostics", "generic")
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
