"""Dependency-light command line entry point for ProbeArch."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "scripts" / "audit" / "shared"
ROBUSTNESS = SHARED / "robustness_manifest.py"
SCHEMA_CHECK = ROOT / "scripts" / "analysis" / "check_schemas.py"


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SystemExit(
                "YAML config requires PyYAML; install probearch-audit[yaml] "
                "or use the JSON-compatible example"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise SystemExit(f"config must contain an object: {path}")
    return value


def _validate_config(config: dict[str, Any]) -> list[str]:
    errors = []
    if config.get("schema_version") != "probearch-robustness-config-v1":
        errors.append("schema_version must be probearch-robustness-config-v1")
    suite = config.get("suite")
    if suite not in {"libero_10", "libero_spatial"}:
        errors.append("suite must be libero_10 or libero_spatial")
    resources = config.get("resources") or {}
    if resources.get("n_envs") != 1:
        errors.append("resources.n_envs must be 1 for the RTX-3050 pilot")
    episodes = config.get("episodes_per_condition")
    if not isinstance(episodes, int) or episodes < 1:
        errors.append("episodes_per_condition must be a positive integer")
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        errors.append("conditions must be a non-empty list")
    else:
        if any(
            not isinstance(item, dict) or not item.get("name") or not item.get("type")
            for item in conditions
        ):
            errors.append("each condition must have a non-empty name and type")
        names = [item.get("name") for item in conditions if isinstance(item, dict)]
        if len(names) != len(set(names)):
            errors.append("condition names must be unique")
        if "clean" not in names:
            errors.append("conditions must include a clean reference")
    tasks = config.get("tasks")
    if not isinstance(tasks, dict) or not tasks or not any(tasks.values()):
        errors.append("tasks must contain at least one non-empty group")
    elif any(
        not isinstance(values, list) or any(not isinstance(task, int) for task in values)
        for values in tasks.values()
    ):
        errors.append("task groups must be lists of integer task ids")
    return errors


def _run(script: Path, *args: str, env: dict[str, str] | None = None) -> int:
    command = [sys.executable, str(script), *args]
    completed = subprocess.run(command, cwd=ROOT, env=env)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probearch")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config", help="validate a YAML/JSON config")
    validate.add_argument("path", type=Path)

    sub.add_parser("check-schemas", help="check the repository JSON schema contracts")

    calibrate = sub.add_parser("calibrate", help="run task-scoped calibration")
    calibrate.add_argument("--suite", required=True)
    calibrate.add_argument("--task-id", required=True, type=int)
    calibrate.add_argument("--out", required=True, type=Path)
    calibrate.add_argument("--n-trials", type=int, default=5)
    calibrate.add_argument("--max-trials", type=int, default=100)

    score = sub.add_parser("score", help="score an existing trajectory directory")
    score.add_argument("--output-audit-dir", type=Path)
    score.add_argument("--audit-dir", type=Path)

    report = sub.add_parser("report", help="regenerate stats, plots, matrix, and report")
    report.add_argument("--audit-dir", type=Path)

    robustness = sub.add_parser("robustness-manifest", help="materialize a matched perturbation manifest")
    robustness.add_argument("config", type=Path)
    robustness.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run", help="run the guarded CUDA/MLX evaluation loop")
    run.add_argument("suite", choices=["libero_10", "libero_spatial"])
    run.add_argument("--n-pairs", type=int, default=8)
    run.add_argument("--n-envs", type=int, default=1)
    run.add_argument("--audit-dir", type=Path)
    run.add_argument("--backend", choices=["cuda", "mlx"], default="cuda")

    verify = sub.add_parser("verify", help="verify a result package against raw rollouts")
    verify.add_argument("--result-dir", type=Path, required=True)
    verify.add_argument("--rollouts-dir", type=Path, required=True)
    verify.add_argument("--video-source-rollouts", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-config":
        errors = _validate_config(_load_config(args.path))
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 2
        print(f"config OK: {args.path}")
        return 0
    if args.command == "check-schemas":
        return _run(SCHEMA_CHECK)
    if args.command == "calibrate":
        return _run(
            SHARED / "calibrate.py",
            "--suite", args.suite,
            "--task-id", str(args.task_id),
            "--out", str(args.out),
            "--n-trials", str(args.n_trials),
            "--max-trials", str(args.max_trials),
        )
    if args.command == "score":
        env = os.environ.copy()
        if args.audit_dir:
            env["AUDIT_DIR"] = str(args.audit_dir)
        command = ["--output-audit-dir", str(args.output_audit_dir)] if args.output_audit_dir else []
        return _run(SHARED / "safety_scorer.py", *command, env=env)
    if args.command == "robustness-manifest":
        return _run(ROBUSTNESS, str(args.config), "--output", str(args.output))
    if args.command == "run":
        audit_dir = args.audit_dir or Path.home() / "probearch-audits" / args.suite
        env = os.environ.copy()
        env.update({"AUDIT_DIR": str(audit_dir), "POLICY_BACKEND": args.backend})
        return subprocess.run(
            ["bash", str(ROOT / "scripts" / "audit" / "shared" / "eval_loop.sh"),
             args.suite, str(args.n_pairs), str(args.n_envs), "--force"],
            cwd=ROOT, env=env,
        ).returncode
    if args.command == "report":
        audit_dir = args.audit_dir or Path(os.environ.get("AUDIT_DIR", Path.home() / "audit"))
        env = os.environ.copy()
        env["AUDIT_DIR"] = str(audit_dir)
        for script in ("stats.py", "plots.py", "confusion_matrix.py", "report.py"):
            if _run(SHARED / script, env=env):
                return 1
        return 0
    if args.command == "verify":
        command = ["--result-dir", str(args.result_dir), "--rollouts-dir", str(args.rollouts_dir)]
        if args.video_source_rollouts:
            command += ["--video-source-rollouts", str(args.video_source_rollouts)]
        return _run(SHARED / "verify_audit.py", *command)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
