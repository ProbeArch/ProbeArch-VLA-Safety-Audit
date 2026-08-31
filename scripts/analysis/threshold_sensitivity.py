#!/usr/bin/env python3
"""Re-score frozen telemetry across declared detector-threshold factors."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

SHARED = Path(__file__).resolve().parents[1] / "audit" / "shared"
sys.path.insert(0, str(SHARED))

import safety_scorer as scorer
from contract_versions import (
    MEASUREMENT_CONTRACT_VERSION,
    TASK_SEMANTICS_VERSION,
)

OUTCOMES = ("safe_success", "unsafe_success", "safe_failure", "unsafe_failure", "not_evaluated")


def load_episodes(rollouts: Path) -> list[dict]:
    episodes = []
    for task_dir in sorted(p for p in rollouts.iterdir() if p.is_dir()):
        manifest_path = task_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = manifest.get("run_id")
        for path in sorted(task_dir.glob("ep_*.json")):
            episode = json.loads(path.read_text(encoding="utf-8"))
            if (episode.get("provenance") or {}).get("run_id") != run_id:
                continue
            episodes.append(episode)
    if not episodes:
        raise ValueError(f"no manifest-matched episodes under {rollouts}")
    return episodes


def matrix_for(outcomes: list[str]) -> dict:
    counts = {
        "recorded_success": {"safe": 0, "unsafe": 0, "not_evaluated": 0},
        "recorded_failure": {"safe": 0, "unsafe": 0, "not_evaluated": 0},
    }
    for outcome in outcomes:
        row = "recorded_success" if outcome.endswith("success") else "recorded_failure"
        if outcome == "not_evaluated":
            column = "not_evaluated"
        else:
            column = "unsafe" if outcome.startswith("unsafe") else "safe"
        counts[row][column] += 1
    return counts


def evaluate(
    episodes: list[dict],
    calibration_dir: Path,
    factor: float,
    fall_margin: float,
    *,
    tau1_factor: float | None = None,
    tau2_factor: float | None = None,
    tilt_factor: float | None = None,
) -> dict:
    """Evaluate one threshold setting without changing the source episodes."""
    tau1_factor = factor if tau1_factor is None else tau1_factor
    tau2_factor = factor if tau2_factor is None else tau2_factor
    tilt_factor = factor if tilt_factor is None else tilt_factor
    old_fall_margin = scorer.FALL_MARGIN
    scorer.FALL_MARGIN = fall_margin
    outcomes = []
    events_by_rule = Counter()
    diagnostics_by_rule = Counter()
    try:
        for original in episodes:
            episode = copy.deepcopy(original)
            task_id = int(episode["task_id"])
            suite = (episode.get("provenance") or {}).get("suite")
            calibration = json.loads(
                (calibration_dir / f"{suite}_{task_id}.json").read_text(encoding="utf-8")
            )
            variant = {
                **calibration,
                "tau1_force_N": float(calibration["tau1_force_N"]) * tau1_factor,
                "tau2_displacement_m": float(calibration["tau2_displacement_m"]) * tau2_factor,
                "tau_tilt_deg": float(calibration["tau_tilt_deg"]) * tilt_factor,
            }
            generic = scorer.score_episode(episode, variant)
            task_aware = scorer.analyze_episode(
                episode,
                variant,
                generic,
                scorer.step_contacts,
                scorer.r1_eligible,
            )
            outcome = task_aware["outcome"]
            outcomes.append(outcome)
            for event in task_aware.get("events", []):
                events_by_rule[event.get("rule", "unknown")] += 1
            for event in task_aware.get("diagnostic_events", []):
                diagnostics_by_rule[event.get("rule", "unknown")] += 1
    finally:
        scorer.FALL_MARGIN = old_fall_margin
    return {
        "factor": factor,
        "tau1_factor": tau1_factor,
        "tau2_factor": tau2_factor,
        "tilt_factor": tilt_factor,
        "fall_margin_m": fall_margin,
        "threshold_role": "measurement_detector_only",
        "n_episodes": len(outcomes),
        "outcome_counts": dict(Counter(outcomes)),
        "matrix": matrix_for(outcomes),
        "candidate_events_by_rule": dict(events_by_rule),
        "diagnostic_events_by_rule": dict(diagnostics_by_rule),
        "hazard_assessment": "not_assessed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factors", nargs="+", type=float, default=[0.75, 1.0, 1.25])
    parser.add_argument("--fall-margin-m", type=float, default=scorer.FALL_MARGIN)
    parser.add_argument(
        "--independent",
        action="store_true",
        help="also sweep tau1, tau2, tilt, and fall margin one parameter at a time",
    )
    args = parser.parse_args()
    if any(value <= 0 for value in args.factors) or args.fall_margin_m <= 0:
        raise SystemExit("threshold factors and fall margin must be positive")
    episodes = load_episodes(args.rollouts)
    result = {
        "schema_version": "probearch-threshold-sensitivity-v1",
        "semantics_version": TASK_SEMANTICS_VERSION,
        "measurement_contract_version": MEASUREMENT_CONTRACT_VERSION,
        "suite": (episodes[0].get("provenance") or {}).get("suite"),
        "factors": args.factors,
        "fall_margin_m": args.fall_margin_m,
        "results": [
            evaluate(episodes, args.calibration, factor, args.fall_margin_m * factor)
            for factor in args.factors
        ],
    }
    if args.independent:
        independent = []
        for parameter in ("tau1", "tau2", "tilt", "fall_margin"):
            for factor in args.factors:
                kwargs = {
                    "tau1_factor": 1.0,
                    "tau2_factor": 1.0,
                    "tilt_factor": 1.0,
                }
                margin = args.fall_margin_m
                if parameter == "tau1":
                    kwargs["tau1_factor"] = factor
                elif parameter == "tau2":
                    kwargs["tau2_factor"] = factor
                elif parameter == "tilt":
                    kwargs["tilt_factor"] = factor
                else:
                    margin *= factor
                item = evaluate(
                    episodes,
                    args.calibration,
                    1.0,
                    margin,
                    **kwargs,
                )
                item["parameter"] = parameter
                item["parameter_factor"] = factor
                independent.append(item)
        result["independent_results"] = independent
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
