#!/usr/bin/env bash
set -e
export AUDIT_DIR=/tmp/test-brain2
rm -rf "$AUDIT_DIR"
mkdir -p "$AUDIT_DIR"
export PATH="/home/dunli/miniconda3/envs/vla-audit/bin:$PATH"
export MUJOCO_GL=egl
export POLICY_BACKEND=cuda
cd /mnt/d/ProbeArch-VLA-Safety-Audit
N_TRIALS=1 MAX_TRIALS=1 bash scripts/audit/shared/eval_loop.sh libero_spatial 1 1 --force 2>&1 | tee /tmp/test-brain2.log
echo EXIT:$?
cat /tmp/test-brain2/stats.json 2>&1 | head -n 40
echo ---ep---
/home/dunli/miniconda3/envs/vla-audit/bin/python /mnt/d/ProbeArch-VLA-Safety-Audit/check_ep.py /tmp/test-brain2
