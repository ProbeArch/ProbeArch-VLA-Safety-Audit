"""Build the task-aware success/safety 2x2 matrix.

This is a contingency table between recorded task success and the task-aware
measurement view.  It is intentionally labelled as such: without independent
human or expert safety labels, it is not an ML confusion matrix and must not be
reported as validated precision/recall.
"""

import json
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AUDIT = Path(os.environ.get("AUDIT_DIR", str(Path.home() / "audit")))
ROLL = AUDIT / "rollouts"


def task_run_id(task_dir):
    path = task_dir / "run_manifest.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value.get("run_id") if isinstance(value.get("run_id"), str) else None


def load_episodes():
    episodes = []
    for task in sorted(p for p in ROLL.iterdir() if p.is_dir()):
        run_id = task_run_id(task)
        if run_id is None:
            continue
        for path in sorted(task.glob("ep_*.json")):
            try:
                episode = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"unreadable episode {path}: {exc}") from exc
            if (episode.get("provenance") or {}).get("run_id") != run_id:
                continue
            if "task_aware" not in episode or "task_aware_events" not in episode:
                raise RuntimeError(f"episode {path} has no task-aware score; run safety_scorer.py first")
            episodes.append(episode)
    if not episodes:
        raise RuntimeError(f"no task-aware episodes found under {ROLL}")
    return episodes


def build_matrix(episodes):
    rows = ("recorded_success", "recorded_failure")
    columns = ("task_aware_safe", "task_aware_unsafe")
    matrix = {row: {column: 0 for column in columns} for row in rows}
    for episode in episodes:
        row = "recorded_success" if episode.get("success") else "recorded_failure"
        unsafe = bool(episode.get("task_aware_events"))
        matrix[row]["task_aware_unsafe" if unsafe else "task_aware_safe"] += 1
    row_normalized = {}
    for row in rows:
        total = sum(matrix[row].values())
        row_normalized[row] = {
            column: round(matrix[row][column] / total, 6) if total else None
            for column in columns
        }
    return {
        "schema_version": "probearch-success-task-aware-matrix-v1",
        "interpretation": (
            "Success-by-task-aware-safety contingency table; no independent safety labels "
            "are available, so this is not a validated ML confusion matrix."
        ),
        "rows": list(rows),
        "columns": list(columns),
        "counts": matrix,
        "row_normalized": row_normalized,
        "n_episodes": len(episodes),
    }


def render_matrix(matrix, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.array(
        [[matrix["counts"][row][column] for column in matrix["columns"]] for row in matrix["rows"]],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    image = ax.imshow(values, cmap="Blues", vmin=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, int(values[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks(range(len(matrix["columns"])), ["safe", "unsafe"])
    ax.set_yticks(range(len(matrix["rows"])), ["success", "failure"])
    ax.set_xlabel("Task-aware measurement status")
    ax.set_ylabel("Recorded task outcome")
    ax.set_title("Success × task-aware safety")
    fig.colorbar(image, ax=ax, label="episodes")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    episodes = load_episodes()
    matrix = build_matrix(episodes)
    (AUDIT / "confusion_matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    render_matrix(matrix, AUDIT / "figures" / "confusion_matrix.png")
    print(json.dumps(matrix, indent=2))
    print("wrote", AUDIT / "confusion_matrix.json")


def _selftest():
    episodes = [
        {"success": True, "task_aware_events": []},
        {"success": True, "task_aware_events": [{"rule": "TA-R2-DISTRACTOR_MOTION"}]},
        {"success": False, "task_aware_events": []},
        {"success": False, "task_aware_events": [{"rule": "R3"}]},
    ]
    matrix = build_matrix(episodes)
    assert matrix["counts"]["recorded_success"]["task_aware_safe"] == 1
    assert matrix["counts"]["recorded_success"]["task_aware_unsafe"] == 1
    assert matrix["counts"]["recorded_failure"]["task_aware_safe"] == 1
    assert matrix["counts"]["recorded_failure"]["task_aware_unsafe"] == 1
    with tempfile.TemporaryDirectory(prefix="probearch-confusion-selftest-") as temp:
        render_matrix(matrix, Path(temp) / "confusion_matrix.png")
        assert (Path(temp) / "confusion_matrix.png").is_file()
    print("confusion_matrix self-test passed")


if __name__ == "__main__":
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--selftest":
        _selftest()
    else:
        main()
