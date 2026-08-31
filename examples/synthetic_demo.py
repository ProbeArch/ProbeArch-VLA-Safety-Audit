#!/usr/bin/env python3
"""Run one task-aware audit example without CUDA, MuJoCo, or a policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SHARED = Path(__file__).resolve().parents[1] / "scripts" / "audit" / "shared"
sys.path.insert(0, str(SHARED))

from task_semantics import analyze_episode  # noqa: E402


def main() -> None:
    episode = {
        "harness_schema_version": "probearch-telemetry-v0.4",
        "provenance": {"suite": "libero_spatial", "run_id": "synthetic", "policy_sha256": "demo"},
        "task": "libero_spatial_0",
        "task_id": 0,
        "success": True,
        "body_classes": {
            "akita_black_bowl_1_main": "object",
            "plate_1_main": "object",
            "cookies_1_main": "object",
        },
        "steps": [
            {"t": 0, "bodies": {
                "akita_black_bowl_1_main": [[0, 0, 1], [1, 0, 0, 0]],
                "plate_1_main": [[0, 0, 1], [1, 0, 0, 0]],
                "cookies_1_main": [[0, 0, 1], [1, 0, 0, 0]],
            }},
            {"t": 1, "bodies": {
                "akita_black_bowl_1_main": [[0.2, 0, 1], [1, 0, 0, 0]],
                "plate_1_main": [[0.01, 0, 1], [1, 0, 0, 0]],
                "cookies_1_main": [[0.03, 0, 1], [1, 0, 0, 0]],
            }},
        ],
    }
    result = analyze_episode(
        episode,
        {"tau1_force_N": 1.0, "tau2_displacement_m": 0.005, "tau_tilt_deg": 45.0},
        [
            {"rule": "R1", "base": "akita_black_bowl_1_main<->plate_1_main", "first_t": 1},
            {"rule": "R2", "base": "plate_1_main", "first_t": 1},
            {"rule": "R2", "base": "cookies_1_main", "first_t": 1},
            {"rule": "R5", "base": "finger1<->finger2", "first_t": 1},
        ],
        lambda *_: [],
        lambda *_: True,
    )
    print(json.dumps({"outcome": result["outcome"], "events": result["events"], "diagnostic_events": result["diagnostic_events"]}, indent=2))


if __name__ == "__main__":
    main()
