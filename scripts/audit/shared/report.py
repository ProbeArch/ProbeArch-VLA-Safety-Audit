"""Generate a compact, reproducible report from derived audit artifacts."""

import json
import os
from pathlib import Path


AUDIT = Path(os.environ.get("AUDIT_DIR", str(Path.home() / "audit")))


def _fmt(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main():
    stats = json.loads((AUDIT / "stats.json").read_text())
    summary = json.loads((AUDIT / "safety_summary.json").read_text())
    matrix_path = AUDIT / "confusion_matrix.json"
    matrix = json.loads(matrix_path.read_text()) if matrix_path.is_file() else None
    manifest_path = AUDIT / "rollouts" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    task_aware = stats.get("task_aware", {})
    suite = manifest.get("suite", "unknown")

    lines = [
        f"# ProbeArch `{suite}` Task-Aware Audit",
        "",
        "> This is a derived offline rescore of frozen telemetry. Generic calibrated",
        "> measurements and task-aware candidates are reported separately. A calibration",
        "> threshold is a detector threshold, not a physical damage or hazard limit.",
        "",
        "## Dataset and runtime",
        "",
        f"- Episodes: **{stats['n_episodes']}** ({sum(v['n_episodes'] for v in stats['per_task'].values())} manifest-matched)",
        f"- Policy: `{manifest.get('policy', 'unknown')}`",
        f"- Backend: `{manifest.get('policy_backend', 'unknown')}`",
        f"- Runtime: Python `{(manifest.get('runtime') or {}).get('python', 'unknown')}`, "
        f"PyTorch `{(manifest.get('runtime') or {}).get('torch', 'unknown')}`, "
        f"MuJoCo `{(manifest.get('runtime') or {}).get('mujoco', 'unknown')}`",
        f"- Source run ID: `{manifest.get('run_id', 'unknown')}`",
        "",
        "## Headline results",
        "",
        f"- Recorded task success: **{stats['successes']}/{stats['n_episodes']} ({stats['success_rate']:.1%})**",
        f"- Generic calibrated measurement events: **{stats['safety_events_total']}**",
        f"- Task-aware candidate events: **{task_aware.get('events_total', 0)}**",
        f"- Episodes with expected target motion: **{task_aware.get('episodes_with_expected_target_motion', 0)}**",
        f"- Episodes with measured distractor motion: **{task_aware.get('episodes_with_distractor_motion', 0)}**",
        "",
        "## Success × task-aware measurement status",
        "",
    ]
    if matrix:
        counts = matrix["counts"]
        lines.extend(
            [
                "| Recorded outcome | Task-aware safe | Task-aware unsafe |",
                "|---|---:|---:|",
                f"| Success | {counts['recorded_success']['task_aware_safe']} | {counts['recorded_success']['task_aware_unsafe']} |",
                f"| Failure | {counts['recorded_failure']['task_aware_safe']} | {counts['recorded_failure']['task_aware_unsafe']} |",
                "",
                f"> {matrix['interpretation']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Per-task results",
            "",
            "| Task | Success | Generic events | Task-aware events | Expected target motion | Distractor motion |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task, values in sorted(stats["per_task"].items()):
        lines.append(
            f"| {task.rsplit('_', 1)[-1]} | {values['successes']}/{values['n_episodes']} "
            f"| {values['safety_events_total']} | {values.get('task_aware_events_total', 0)} "
            f"| {values.get('episodes_with_expected_target_motion', 0)} "
            f"| {values.get('episodes_with_distractor_motion', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and limits",
            "",
            "- Target displacement above the calibrated tau2 is retained as an expected-motion measurement when the task explicitly requires moving that object.",
            "- Distractor displacement and contact are task-aware candidate regressions, not independently labeled hazards.",
            "- R3/R4 measurements remain visible because target overturn/fall can still be harmful even when target motion is intended.",
            "- No human or expert safety labels are available in this pilot, so the matrix is a co-occurrence table rather than validated classification precision/recall.",
            "- The next research gate is independent semantic labeling and operational-limit specification.",
            "",
            "## Artifacts",
            "",
            "- `safety_summary.json` — generic and task-aware scoring summary",
            "- `stats.json` — pooled and per-task statistics",
            "- `confusion_matrix.json` — success/task-aware co-occurrence table",
            "- `figures/` — regenerated plots",
            "- `videos/` — outcome-verified representative reconstructions when available",
        ]
    )
    out = AUDIT / "report.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
