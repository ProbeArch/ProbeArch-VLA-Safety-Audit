#!/usr/bin/env python3
"""Analyze independent ProbeArch episode labels without sklearn."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

LABELS = {"SAFE_SUCCESS", "UNSAFE_SUCCESS", "SAFE_FAILURE", "UNSAFE_FAILURE", "NOT_EVALUATED"}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(read_rows(args.labels))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
