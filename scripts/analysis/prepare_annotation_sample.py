#!/usr/bin/env python3
"""Prepare a blinded, stratified annotation manifest from frozen rollouts.

The script derives the current candidate outcome only for sampling. It never
writes that candidate label to the annotator CSV, preventing label leakage.
The candidate outcome and the selected strata are printed as an audit summary
for the study owner; annotators receive only the manifest rows and evidence
references.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "scripts" / "audit" / "shared"
sys.path.insert(0, str(SHARED))

from safety_scorer import r1_eligible, score_episode, step_contacts  # noqa: E402
from task_semantics import analyze_episode  # noqa: E402


STRATA = (
    "success/safe",
    "success/unsafe",
    "failure/safe",
    "failure/unsafe",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def candidate_record(suite: str, rollouts: Path, calibration_dir: Path, path: Path) -> dict:
    episode = load_json(path)
    task_id = int(episode["task_id"])
    calibration = load_json(calibration_dir / f"{suite}_{task_id}.json")
    generic_events = score_episode(episode, calibration)
    task_aware = analyze_episode(
        episode,
        calibration,
        generic_events,
        step_contacts,
        r1_eligible,
    )
    outcome = task_aware["outcome"]
    if outcome not in {"safe_success", "unsafe_success", "safe_failure", "unsafe_failure"}:
        raise ValueError(f"{path}: cannot stratify {outcome}")
    destination_event = any(
        "DESTINATION" in str(event.get("rule", "")).upper()
        or "destination" in str(event.get("classification", "")).lower()
        for event in task_aware.get("events", [])
    )
    destination_event = destination_event or bool(
        task_aware.get("destination_motion_measurements")
    )
    priority = "high" if destination_event else "normal"
    relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    return {
        "episode_path": relative,
        "source_sha256": file_sha256(path),
        "suite": suite,
        "task_id": task_id,
        "episode": int(episode["ep_ix"]),
        "recorded_success": str(bool(episode.get("success"))).lower(),
        "stratum": f"{'success' if episode.get('success') else 'failure'}/"
        f"{'unsafe' if outcome.startswith('unsafe') else 'safe'}",
        "review_priority": priority,
    }


def prepare(config: dict, output: Path) -> dict:
    if config.get("schema_version") != "probearch-annotation-sample-v1":
        raise ValueError("unsupported annotation sample schema")
    n_per_cell = int(config.get("n_per_cell", 25))
    if n_per_cell < 1:
        raise ValueError("n_per_cell must be positive")
    records = []
    for dataset in config.get("datasets", []):
        suite = str(dataset["suite"])
        rollouts = (ROOT / dataset["rollouts"]).resolve()
        calibration_dir = (ROOT / dataset["calibration"]).resolve()
        for task_dir in sorted(p for p in rollouts.iterdir() if p.is_dir()):
            for path in sorted(task_dir.glob("ep_*.json")):
                records.append(candidate_record(suite, rollouts, calibration_dir, path))

    grouped = {stratum: [] for stratum in STRATA}
    for record in records:
        grouped[record["stratum"]].append(record)
    missing = {key: n_per_cell - len(value) for key, value in grouped.items() if len(value) < n_per_cell}
    if missing:
        raise ValueError(f"not enough episodes for requested strata: {missing}")

    rng = random.Random(int(config.get("seed", 0)))
    selected = []
    for stratum in STRATA:
        high = [item for item in grouped[stratum] if item["review_priority"] == "high"]
        normal = [item for item in grouped[stratum] if item["review_priority"] != "high"]
        rng.shuffle(high)
        rng.shuffle(normal)
        selected.extend((high + normal)[:n_per_cell])
    rng.shuffle(selected)
    heldout_n = int(round(len(selected) * float(config.get("heldout_fraction", 0.2))))
    heldout_n = max(1, min(len(selected) - 1, heldout_n))
    for index, record in enumerate(selected):
        record["split"] = "heldout" if index < heldout_n else "development"

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_path", "source_sha256", "suite", "task_id", "episode",
                "recorded_success", "stratum", "review_priority", "split",
            ],
        )
        writer.writeheader()
        writer.writerows(selected)
    return {
        "schema_version": "probearch-annotation-sample-manifest-v1",
        "seed": int(config.get("seed", 0)),
        "n_selected": len(selected),
        "n_development": sum(item["split"] == "development" for item in selected),
        "n_heldout": sum(item["split"] == "heldout" for item in selected),
        "strata": dict(Counter(item["stratum"] for item in selected)),
        "suites": dict(Counter(item["suite"] for item in selected)),
        "borderline_destination_priority": sum(
            item["review_priority"] == "high" for item in selected
        ),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    print(json.dumps(prepare(config, args.output), indent=2))


if __name__ == "__main__":
    main()
