#!/usr/bin/env bash
set -euo pipefail
export AUDIT_DIR=/home/dunli/audit-v0.2-pilot-B
rm -rf "$AUDIT_DIR"
mkdir -p "$AUDIT_DIR"
cd /mnt/d/ProbeArch-VLA-Safety-Audit
echo "== A synthetic + cuda sanity =="
/home/dunli/miniconda3/envs/vla-audit/bin/python scripts/audit/cuda/cuda_sanity.py
/home/dunli/miniconda3/envs/vla-audit/bin/python scripts/audit/cuda/cuda_scorer_batch.py /home/dunli/audit-v0.2-pilot-A 2>&1 | head -n 20 || echo "no prior A to parity, will parity after B"
echo "== B pilot B 1x1 N_TRIALS=1 with B1 kernel ==="
export MUJOCO_GL=egl
export POLICY_BACKEND=cuda
export PATH="/home/dunli/miniconda3/envs/vla-audit/bin:$PATH"
echo "== disk before B =="; df -h /home/dunli | tail -n 1
echo "== gpu before B =="; nvidia-smi --query-gpu=memory.free,memory.total --format=csv
N_TRIALS=1 MAX_TRIALS=1 bash scripts/audit/shared/eval_loop.sh libero_spatial 1 1 --force 2>&1 | tee "$AUDIT_DIR/pilot.log"
echo "== pilot B done =="
echo "== disk after B =="; df -h /home/dunli | tail -n 1
echo "== gpu after B =="; nvidia-smi --query-gpu=memory.free,memory.total --format=csv
ls -R "$AUDIT_DIR" | head -n 200
echo "--- calibration ---"
grep -E "tau|fall_margin" "$AUDIT_DIR/calibration.json" | head -n 20
echo "--- safety_summary ---"
cat "$AUDIT_DIR/safety_summary.json" 2>&1 | head -n 80
echo "--- stats ---"
cat "$AUDIT_DIR/stats.json" 2>&1 | head -n 80
echo "--- verify B cuda batch parity on B data ---"
/home/dunli/miniconda3/envs/vla-audit/bin/python scripts/audit/cuda/cuda_scorer_batch.py "$AUDIT_DIR" 2>&1
echo "--- byte-identical check A vs B safety_summary ---"
diff -q /home/dunli/audit-v0.2-pilot-A/safety_summary.json "$AUDIT_DIR/safety_summary.json" && echo "byte-identical safety_summary A vs B" || (echo "diff A vs B safety_summary:"; diff /home/dunli/audit-v0.2-pilot-A/safety_summary.json "$AUDIT_DIR/safety_summary.json" | head -n 40)
echo "--- pilot.log tail B ---"
tail -n 50 "$AUDIT_DIR/pilot.log"
