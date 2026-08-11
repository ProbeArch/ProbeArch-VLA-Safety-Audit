#!/bin/bash
# The ProbeArch eval loop. Every pin in pins.md must hold before this produces numbers.
set -euo pipefail

cd /home/dunli
export PATH="/home/dunli/miniconda3/bin:$PATH"
source /home/dunli/miniconda3/bin/activate vla-audit
export MUJOCO_GL=egl

POLICY="HuggingFaceVLA/smolvla_libero"
SUITE="${1:-libero_spatial}"
EPISODES="${2:-10}"
SEED="${3:-0}"
OUT="$(git -C /mnt/d/ProbeArch-VLA-Safety-Audit rev-parse --show-toplevel)/analysis/results"

echo "== $(date) == eval start: policy=$POLICY suite=$SUITE episodes=$EPISODES seed=$SEED"

# 1) model rollouts via the custom telemetry harness (states+contacts per step saved)
python /mnt/d/ProbeArch-VLA-Safety-Audit/scripts/telemetry_rollout.py \
  --policy.path="$POLICY" \
  --env.task="$SUITE" \
  --eval.n_episodes="$EPISODES" \
  --seed="$SEED" \
  --outdir="$OUT"

# 2) safety scoring (pre-registered rules, Positive-control validated)
python /mnt/d/ProbeArch-VLA-Safety-Audit/scripts/safety_scorer.py \
  --rollouts="$OUT/rollouts_${SUITE}_seed${SEED}" \
  --spec /mnt/d/ProbeArch-VLA-Safety-Audit/pre-registration/spec_${SUITE}.json

echo "== $(date) == eval done"