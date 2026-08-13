#!/usr/bin/env bash
# The ProbeArch eval loop: smoke gate -> calibrate -> rollouts -> score -> stats -> plots.
# Every pin in pins.md must hold before this produces numbers.
#
# Usage:  scripts/eval_loop.sh [SUITE] [N_PAIRS] [N_ENVS] [--resume|--force]
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
# Suite note: calibration (calibrate.py builds the LIBERO Spatial task-0 scene)
# and the per-task loop (Spatial task ids 0-4) are Spatial-specific. Any other
# suite is rejected rather than silently given Spatial thresholds.
#
# Env:    AUDIT_DIR        output root (default ~/audit)
#         POLICY           HF policy id (default HuggingFaceVLA/smolvla_libero)
#         POLICY_BACKEND   cuda (default, LeRobot/torch) or mlx (Apple Silicon)
#         MUJOCO_GL        render backend (default egl)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AUDIT_DIR="${AUDIT_DIR:-$HOME/audit}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
POLICY="${POLICY:-HuggingFaceVLA/smolvla_libero}"
POLICY_BACKEND="${POLICY_BACKEND:-cuda}"

SUITE=""
N_PAIRS=""
N_ENVS=""
RESUME=0
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
if [[ "$N_PAIRS" -lt 1 || "$N_ENVS" -lt 1 ]]; then
  echo "n_pairs and n_envs must be >= 1" >&2
  exit 2
fi
if [[ "$POLICY_BACKEND" != "cuda" && "$POLICY_BACKEND" != "mlx" ]]; then
  echo "POLICY_BACKEND must be cuda or mlx, got '$POLICY_BACKEND'" >&2
  exit 2
fi

# Calibration and the task loop are LIBERO Spatial-specific (see header).
if [[ "$SUITE" != "libero_spatial" ]]; then
  echo "Refusing: suite '$SUITE' is not supported." >&2
  echo "This harness calibrates and rolls out LIBERO Spatial task ids 0-4 only;" >&2
  echo "other suites would receive Spatial thresholds." >&2
  exit 2
fi

mkdir -p "$AUDIT_DIR"
ROLLOUTS_DIR="$AUDIT_DIR/rollouts"
CALIBRATION="$AUDIT_DIR/calibration.json"

validate_calibration() {
  python3 - "$CALIBRATION" <<'PY'
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
  exit 1
fi

echo "== $(date) == eval start: policy=$POLICY backend=$POLICY_BACKEND suite=$SUITE n_pairs=$N_PAIRS n_envs=$N_ENVS audit_dir=$AUDIT_DIR mode=$([ "$RESUME" = 1 ] && echo resume || ([ "$FORCE" = 1 ] && echo force || echo fresh))"

# 0) synthetic self-tests (plain python, no runtime deps) + smoke gate.
#    Every gate must exit 0 - a nonzero exit aborts the whole run before any
#    calibration or rollout. The smoke gate adds a best-effort live rollout
#    when the runtime deps are installed.
for t in telemetry_rollout safety_scorer stats mlx_smolvla; do
  echo "== selftest: $t =="
  python3 "$REPO/scripts/$t.py" --selftest
done
echo "== selftest: calibrate =="
python3 "$REPO/scripts/calibrate.py" --self-test
echo "== smoke gate =="
python3 "$REPO/scripts/smoke_test.py"

# 1) positive-control calibration -> $AUDIT_DIR/calibration.json (scorer-validated).
#    Skipped on --resume: re-running would change its sha256 and break the run manifest.
if [[ "$RESUME" != "1" ]]; then
  python3 "$REPO/scripts/calibrate.py" --suite "$SUITE" --task-id 0
fi
validate_calibration

# 2) instrumented rollouts (per-step states + contacts saved per episode).
#    Keep each task in its own process: building all task envs together can segfault.
for task_id in 0 1 2 3 4; do
  python3 "$REPO/scripts/telemetry_rollout.py" \
    --suite "$SUITE" \
    --task_ids "$task_id" \
    --policy "$POLICY" \
    --device "$POLICY_BACKEND" \
    --n_envs "$N_ENVS" \
    --n_pairs "$N_PAIRS"
done

# Verify per-task metrics survived and the aggregate is complete (each invocation
# writes <task>/metrics.json; the merged metrics.json must cover all 5 tasks and
# hold only finite values - NaN would silently corrupt the JSON).
python3 - "$ROLLOUTS_DIR" "$SUITE" "$N_ENVS" "$N_PAIRS" <<'PY'
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
expected = {f"{suite}_{t}" for t in range(5)}
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
python3 "$REPO/scripts/safety_scorer.py"

# 4) aggregate stats + figures
python3 "$REPO/scripts/stats.py"
python3 "$REPO/scripts/plots.py"

echo "== $(date) == eval done -> $AUDIT_DIR"
