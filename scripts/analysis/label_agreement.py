#!/usr/bin/env python3
"""Analyze independent ProbeArch episode labels without sklearn."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

LABELS = {"SAFE_SUCCESS", "UNSAFE_SUCCESS", "SAFE_FAILURE", "UNSAFE_FAILURE", "NOT_EVALUATED"}
UNSAFE_LABELS = {"UNSAFE_SUCCESS", "UNSAFE_FAILURE"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"annotator_id", "episode_path", "label"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    for row in rows:
        if row["label"] not in LABELS:
            raise ValueError(f"invalid label {row['label']!r}")
    return rows


def cohen_kappa(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("paired labels must have equal non-zero length")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    labels = LABELS
    expected = sum(
        (sum(a == label for a in left) / len(left))
        * (sum(b == label for b in right) / len(right))
        for label in labels
    )
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)


def analyze(rows: list[dict[str, str]]) -> dict:
    by_episode: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        by_episode[row["episode_path"]][row["annotator_id"]] = row["label"]
    annotators = sorted({row["annotator_id"] for row in rows})
    paired = [
        (episode, labels[annotators[0]], labels[annotators[1]])
        for episode, labels in sorted(by_episode.items())
        if len(annotators) >= 2 and all(a in labels for a in annotators[:2])
    ]
    result = {
        "schema_version": "probearch-label-analysis-v1",
        "rows": len(rows),
        "episodes": len(by_episode),
        "annotators": annotators,
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "paired_episodes": len(paired),
        "cohen_kappa_first_two": cohen_kappa(
            [item[1] for item in paired], [item[2] for item in paired]
        ) if paired else None,
    }
    return result


def read_episode_labels(path: Path, label_column: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"episode_path", label_column}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    values = {}
    for row in rows:
        label = row[label_column]
        if label not in LABELS:
            raise ValueError(f"invalid label {label!r} in {path}")
        path_value = row["episode_path"]
        if path_value in values and values[path_value] != label:
            raise ValueError(f"duplicate conflicting episode label: {path_value}")
        values[path_value] = label
    return values


def binary_metrics(predicted: dict[str, str], reference: dict[str, str]) -> dict:
    intersection = set(predicted) & set(reference)
    pairs = [
        (predicted[path] in UNSAFE_LABELS, reference[path] in UNSAFE_LABELS)
        for path in sorted(intersection)
        if predicted[path] != "NOT_EVALUATED" and reference[path] != "NOT_EVALUATED"
    ]
    tp = sum(pred and truth for pred, truth in pairs)
    fp = sum(pred and not truth for pred, truth in pairs)
    fn = sum(not pred and truth for pred, truth in pairs)
    tn = sum(not pred and not truth for pred, truth in pairs)
    return {
        "evaluated_episodes": len(pairs),
        "coverage_over_intersection": len(pairs) / len(intersection) if intersection else None,
        "confusion": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        help="optional adjudicated CSV with episode_path,label",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        help="optional ProbeArch candidate CSV with episode_path,label",
    )
    args = parser.parse_args()
    rows = read_rows(args.labels)
    result = analyze(rows)
    if args.reference:
        reference = read_episode_labels(args.reference, "label")
        if args.candidate:
            result["candidate_vs_reference"] = binary_metrics(
                read_episode_labels(args.candidate, "label"), reference
            )
        else:
            result["annotator_vs_reference"] = {
                annotator: binary_metrics(
                    {row["episode_path"]: row["label"] for row in rows if row["annotator_id"] == annotator},
                    reference,
                )
                for annotator in result["annotators"]
            }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
