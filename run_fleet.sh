#!/usr/bin/env bash
set -euo pipefail
export AUDIT_DIR=/home/dunli/audit-v0.2-fleet
export MUJOCO_GL=egl
export POLICY_BACKEND=cuda
export PATH="/home/dunli/miniconda3/envs/vla-audit/bin:$PATH"
cd /mnt/d/ProbeArch-VLA-Safety-Audit

try_fleet() {
  local n_envs=$1
  local n_pairs=$2
  local label="fleet ${n_pairs}x${n_envs}"
  echo "== TRY $label N_TRIALS=5 =="
  rm -rf "$AUDIT_DIR"
  mkdir -p "$AUDIT_DIR"
  # Disk guard before each try
  df -h /home/dunli | tail -n 1
  if N_TRIALS=5 MAX_TRIALS=5 bash scripts/audit/shared/eval_loop.sh libero_spatial "$n_pairs" "$n_envs" --force 2>&1 | tee "$AUDIT_DIR/fleet.log"; then
    echo "== $label SUCCEEDED =="
    return 0
  else
    echo "== $label FAILED with exit $? =="
    # Check for OOM signatures
    if grep -qi "out of memory\|OOM\|CUDA.*memory\|RuntimeError.*memory" "$AUDIT_DIR/fleet.log"; then
      echo "OOM detected for $label"
      return 2
    fi
    return 1
  fi
}

# Try 8x4 first
if try_fleet 4 8; then
  echo "fleet 8x4 done"
else
  rc=$?
  if [ $rc -eq 2 ]; then
    echo "Falling back to 8x1 due to OOM"
    if try_fleet 1 8; then
      echo "fleet 8x1 fallback succeeded"
    else
      echo "fleet 8x1 also failed"
      exit 1
    fi
  else
    echo "fleet 8x4 failed non-OOM, trying 8x1 anyway"
    try_fleet 1 8 || exit 1
  fi
fi

echo "== fleet final == "
ls -R "$AUDIT_DIR" | head -n 300
echo "--- calibration taus ---"
grep -E "tau|fall_margin" "$AUDIT_DIR/calibration.json" | head -n 20
echo "--- safety_summary ---"
cat "$AUDIT_DIR/safety_summary.json" 2>&1 | head -n 120
echo "--- stats ---"
cat "$AUDIT_DIR/stats.json" 2>&1 | head -n 120
echo "--- run_manifest ---"
cat "$AUDIT_DIR/rollouts/run_manifest.json" 2>&1 | head -n 60
echo "--- nvidia-smi ==="
nvidia-smi --query-gpu=memory.free,memory.total --format=csv || true
echo "--- df ==="
df -h /home/dunli | tail -n 1
du -sh "$AUDIT_DIR" | head -n 5
