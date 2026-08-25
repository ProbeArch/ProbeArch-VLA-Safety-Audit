#!/usr/bin/env bash
# The ProbeArch eval loop: smoke gate -> calibrate -> rollouts -> score -> stats -> plots.
# Every pin in pins.md must hold before this produces numbers.
#
# Usage:  scripts/_backend_map/shared/eval_loop.sh [SUITE] [N_PAIRS] [N_ENVS] [--resume|--force]
#   (defaults: libero_spatial, 8 pairs, 4 envs; flags may appear in any position)
#   --resume  continue a manifest-matched run. Reuses existing episodes only when
#             the run manifest matches (policy, suite, resolution, n_envs/n_pairs,
#             calibration sha256 - enforced by telemetry_rollout.py). Does NOT
#             re-run calibration: its sha256 anchors the manifest.
#   --force   discard $AUDIT_DIR/rollouts, $AUDIT_DIR/calibration.json and any
#             stale aggregates (safety_summary.json, stats.json, figures/) and
#             start a fully fresh run.
#   With neither flag, the loop fails fast if rollouts already contain episode
#   files (stale v0.1 telemetry must never be rescored with v0.2 thresholds).
#
# Suite note: calibration and task dispatch are suite-aware. Each suite gets its
# own calibration and task-id range; never reuse thresholds across suites.
#
#         N_TRIALS         calibrate.py repetitions per control set (default 5, max MAX_TRIALS)
#         MAX_TRIALS       upper bound passed to calibrate --max-trials (default 100)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export AUDIT_DIR="${AUDIT_DIR:-$HOME/audit}"
if [[ -z "${MUJOCO_GL:-}" ]]; then
  case "$(uname -s 2>/dev/null || echo Linux)" in
    Darwin*) export MUJOCO_GL="glfw" ;;
    *) export MUJOCO_GL="egl" ;;
  esac
fi
POLICY="${POLICY:-HuggingFaceVLA/smolvla_libero}"
POLICY_BACKEND="${POLICY_BACKEND:-cuda}"

SUITE=""
N_PAIRS=""
N_ENVS=""
RESUME=0
N_TRIALS="${N_TRIALS:-5}"
MAX_TRIALS="${MAX_TRIALS:-100}"
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --resume) RESUME=1 ;;
    --force) FORCE=1 ;;
    *)
      if [[ -z "$SUITE" ]]; then
        SUITE="$arg"
      elif [[ -z "$N_PAIRS" ]]; then
        N_PAIRS="$arg"
      elif [[ -z "$N_ENVS" ]]; then
        N_ENVS="$arg"
      else
        echo "unexpected argument: $arg" >&2
        exit 2
      fi
      ;;
  esac
done
SUITE="${SUITE:-libero_spatial}"
N_PAIRS="${N_PAIRS:-8}"
N_ENVS="${N_ENVS:-4}"

case "$N_PAIRS" in
  ''|*[!0-9]*) echo "n_pairs must be a positive integer, got '$N_PAIRS'" >&2; exit 2 ;;
esac
case "$N_ENVS" in
  ''|*[!0-9]*) echo "n_envs must be a positive integer, got '$N_ENVS'" >&2; exit 2 ;;
esac
if ! [[ "$N_TRIALS" =~ ^[0-9]+$ && "$MAX_TRIALS" =~ ^[0-9]+$ ]]; then
  echo "N_TRIALS and MAX_TRIALS must be non-negative integers" >&2
  exit 2
fi
if [[ "$N_TRIALS" -lt 1 || "$MAX_TRIALS" -lt 1 ]]; then
  echo "N_TRIALS and MAX_TRIALS must be >= 1" >&2
  exit 2
fi
if [[ "$N_TRIALS" -gt "$MAX_TRIALS" ]]; then
  echo "N_TRIALS ($N_TRIALS) must be <= MAX_TRIALS ($MAX_TRIALS)" >&2
  exit 2
fi
if [[ "$POLICY_BACKEND" != "cuda" && "$POLICY_BACKEND" != "mlx" ]]; then
  echo "POLICY_BACKEND must be cuda or mlx, got '$POLICY_BACKEND'" >&2
  exit 2
fi

# Resolve the installed suite's task ids explicitly. Keep this shell launcher
# aligned with telemetry_rollout.py, which validates the same suite registry.
case "$SUITE" in
  libero_spatial|libero_object|libero_goal|libero_10)
    TASK_IDS=(0 1 2 3 4 5 6 7 8 9)
    ;;
  libero_90)
    TASK_IDS=($(seq 0 89))
    ;;
  *)
    echo "Refusing: suite '$SUITE' is not supported." >&2
    echo "Supported suites: libero_spatial, libero_object, libero_goal, libero_10, libero_90" >&2
    exit 2
    ;;
esac

mkdir -p "$AUDIT_DIR"
ROLLOUTS_DIR="$AUDIT_DIR/rollouts"
CALIBRATION="$AUDIT_DIR/calibration.json"

validate_calibration() {
  "$PYTHON_BIN" - "$CALIBRATION" <<'PY'
import json
import math
import sys

path = sys.argv[1]
try:
    with open(path) as f:
        cal = json.load(f)
except OSError as exc:
    sys.exit(f"calibration file missing/unreadable: {exc}")
except json.JSONDecodeError as exc:
    sys.exit(f"calibration file is not valid JSON: {exc}")
for key in ("tau1_force_N", "tau2_displacement_m", "tau_tilt_deg"):
    value = cal.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        sys.exit(f"calibration.json missing/invalid {key}: {value!r}")
print(
    f"calibration OK: tau1={cal['tau1_force_N']:.3g} N, "
    f"tau2={cal['tau2_displacement_m']:.3g} m, tau_tilt={cal['tau_tilt_deg']:.3g} deg"
)
PY
}

# Fail fast / resume / force gate on the rollouts directory.
if [[ "$FORCE" == "1" ]]; then
  echo "== --force: discarding $ROLLOUTS_DIR, $CALIBRATION and stale aggregates =="
  rm -rf "$ROLLOUTS_DIR"
  rm -f "$CALIBRATION"
  # Stale aggregates from a previous (possibly retracted v0.1) run must not
  # linger in a reused AUDIT_DIR: a fresh run that dies mid-pipeline would
  # otherwise leave old numbers looking current.
  rm -f "$AUDIT_DIR/safety_summary.json" "$AUDIT_DIR/stats.json" "$AUDIT_DIR/metrics.json"
  rm -rf "$AUDIT_DIR/figures"
elif [[ "$RESUME" == "1" ]]; then
  if [[ ! -d "$ROLLOUTS_DIR" ]]; then
    echo "Refusing: --resume given but $ROLLOUTS_DIR does not exist; start fresh instead." >&2
    exit 1
  fi
  echo "== --resume: continuing manifest-matched run in $ROLLOUTS_DIR =="
elif compgen -G "$ROLLOUTS_DIR/*/ep_*.json" >/dev/null; then
  echo "Refusing: $ROLLOUTS_DIR already contains episode files." >&2
  echo "  Stale telemetry (e.g. retracted v0.1) must not be rescored with v0.2 thresholds." >&2
  echo "  Pass --resume to continue a manifest-matched run, or --force to discard and restart." >&2
  echo "  Tip: use separate AUDIT_DIR per backend, e.g. AUDIT_DIR=~/audit-cuda vs ~/audit-mlx." >&2
  exit 1
fi

MODE=fresh
if [[ "$RESUME" == "1" ]]; then
  MODE=resume
elif [[ "$FORCE" == "1" ]]; then
  MODE=force
fi
echo "== $(date) == eval start: policy=$POLICY backend=$POLICY_BACKEND suite=$SUITE n_pairs=$N_PAIRS n_envs=$N_ENVS n_trials=$N_TRIALS max_trials=$MAX_TRIALS audit_dir=$AUDIT_DIR mode=$MODE"

# 0) synthetic self-tests (plain python, no runtime deps) + smoke gate.
#    Every gate must exit 0 - a nonzero exit aborts the whole run before any
#    calibration or rollout. The smoke gate adds a best-effort live rollout
#    when the runtime deps are installed.
for spec in "shared/telemetry_rollout" "shared/safety_scorer" "shared/stats" "mlx/mlx_smolvla"; do
  echo "== selftest: $spec =="
  "$PYTHON_BIN" "$REPO/scripts/_backend_map/$spec.py" --selftest
done
echo "== selftest: calibrate =="
"$PYTHON_BIN" "$REPO/scripts/_backend_map/shared/calibrate.py" --self-test
echo "== smoke gate =="
"$PYTHON_BIN" "$REPO/scripts/_backend_map/shared/smoke_test.py"

# 1) positive-control calibration -> $AUDIT_DIR/calibration.json (scorer-validated).
#    Skipped on --resume: re-running would change its sha256 and break the run manifest.
if [[ "$RESUME" != "1" ]]; then
  "$PYTHON_BIN" "$REPO/scripts/_backend_map/shared/calibrate.py" \
    --suite "$SUITE" \
    --task-id 0 \
    --n-trials "$N_TRIALS" \
    --max-trials "$MAX_TRIALS"
fi
validate_calibration

# 2) instrumented rollouts (per-step states + contacts saved per episode).
#    Keep each task in its own process: building all task envs together can segfault.
for task_id in "${TASK_IDS[@]}"; do
  "$PYTHON_BIN" "$REPO/scripts/_backend_map/shared/telemetry_rollout.py" \
    --suite "$SUITE" \
    --task_ids "$task_id" \
    --policy "$POLICY" \
    --device "$POLICY_BACKEND" \
    --n_envs "$N_ENVS" \
    --n_pairs "$N_PAIRS"
done

# Verify per-task metrics survived and the aggregate is complete (each invocation
# writes <task>/metrics.json; the merged metrics.json must cover every task and
# hold only finite values - NaN would silently corrupt the JSON).
"$PYTHON_BIN" - "$ROLLOUTS_DIR" "$SUITE" "$N_ENVS" "$N_PAIRS" <<'PY'
import json
import math
import sys
from pathlib import Path

roll, suite, n_envs, n_pairs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

# Run-manifest layer: root + every task dir must carry a manifest and share one
# run_id. A manifest-less task dir would admit unprovenanced episodes (e.g. a
# manual scorer/stats run against a legacy dir must never be part of this run).
with open(f"{roll}/run_manifest.json") as f:
    root_manifest = json.load(f)
root_run_id = root_manifest.get("run_id")
if not isinstance(root_run_id, str):
    sys.exit("root run_manifest.json has no run_id")
missing_manifest = [
    p.name for p in sorted(Path(roll).iterdir()) if p.is_dir()
    and not (p / "run_manifest.json").is_file()
]
if missing_manifest:
    sys.exit(f"task dirs missing run_manifest.json: {sorted(missing_manifest)}")

with open(f"{roll}/metrics.json") as f:
    agg = json.load(f)
task_count = 90 if suite == "libero_90" else 10
expected = {f"{suite}_{t}" for t in range(task_count)}
missing = expected - set(agg)
if missing:
    sys.exit(f"aggregate metrics.json missing tasks: {sorted(missing)}")
for task in sorted(agg):
    with open(f"{roll}/{task}/run_manifest.json") as f:
        task_manifest = json.load(f)
    if task_manifest.get("run_id") != root_run_id:
        sys.exit(
            f"{task}: run_id {task_manifest.get('run_id')!r} != root {root_run_id!r}"
        )
planned_n = n_envs * n_pairs
for task, metrics in sorted(agg.items()):
    if metrics.get("n_episodes") != planned_n:
        sys.exit(f"{task}: n_episodes {metrics.get('n_episodes')!r} != planned {planned_n}")
    for key in ("successes", "pc_success", "seconds_per_episode"):
        value = metrics.get(key)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            sys.exit(f"{task}: invalid metric {key}={value!r}")
print(f"metrics OK: {len(agg)} tasks, {sum(m['n_episodes'] for m in agg.values())} episodes")
PY

# 3) safety scoring (pre-registered rules, positive-control validated)
"$PYTHON_BIN" "$REPO/scripts/_backend_map/shared/safety_scorer.py"

# 4) aggregate stats + figures
"$PYTHON_BIN" "$REPO/scripts/_backend_map/shared/stats.py"
"$PYTHON_BIN" "$REPO/scripts/_backend_map/shared/plots.py"

echo "== $(date) == eval done -> $AUDIT_DIR"
