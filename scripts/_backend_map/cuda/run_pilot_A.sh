#!/usr/bin/env bash
set -euo pipefail
export AUDIT_DIR=/home/dunli/audit-v0.2-pilot-A
rm -rf "$AUDIT_DIR"
mkdir -p "$AUDIT_DIR"
cd /mnt/d/ProbeArch-VLA-Safety-Audit
echo "== A synthetic =="
/home/dunli/miniconda3/envs/vla-audit/bin/python scripts/_backend_map/cuda/cuda_sanity.py
/home/dunli/miniconda3/envs/vla-audit/bin/python scripts/_backend_map/shared/telemetry_rollout.py --selftest
/home/dunli/miniconda3/envs/vla-audit/bin/python scripts/_backend_map/shared/safety_scorer.py --selftest
/home/dunli/miniconda3/envs/vla-audit/bin/python scripts/_backend_map/shared/stats.py --selftest
/home/dunli/miniconda3/envs/vla-audit/bin/python scripts/_backend_map/shared/calibrate.py --self-test
/home/dunli/miniconda3/envs/vla-audit/bin/python scripts/_backend_map/mlx/mlx_smolvla.py --selftest
/home/dunli/miniconda3/envs/vla-audit/bin/python scripts/_backend_map/shared/smoke_test.py
echo "== B/C pilot A 1x1 N_TRIALS=1 FIXED A+B+D =="
export MUJOCO_GL=egl
export POLICY_BACKEND=cuda
export PATH="/home/dunli/miniconda3/envs/vla-audit/bin:$PATH"
echo "== disk before pilot =="; df -h /home/dunli | tail -n 1
echo "== gpu before pilot =="; nvidia-smi --query-gpu=memory.free,memory.total --format=csv
N_TRIALS=1 MAX_TRIALS=1 bash scripts/_backend_map/shared/eval_loop.sh libero_spatial 1 1 --force 2>&1 | tee "$AUDIT_DIR/pilot.log"
echo "== pilot A done =="
echo "== disk after =="; df -h /home/dunli | tail -n 1
echo "== gpu after =="; nvidia-smi --query-gpu=memory.free,memory.total --format=csv
ls -R "$AUDIT_DIR" | head -n 200
echo "--- calibration taus ---"
grep -E "tau|fall_margin" "$AUDIT_DIR/calibration.json" | head -n 20
echo "--- safety_summary ---"
cat "$AUDIT_DIR/safety_summary.json" 2>&1 | head -n 80
echo "--- stats ---"
cat "$AUDIT_DIR/stats.json" 2>&1 | head -n 80
echo "--- run_manifest ---"
cat "$AUDIT_DIR/rollouts/run_manifest.json" 2>&1 | head -n 40
echo "--- pilot.log tail ---"
tail -n 50 "$AUDIT_DIR/pilot.log"
