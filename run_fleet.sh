#!/usr/bin/env bash
# Portable CUDA fleet launcher. Configuration comes from arguments/environment;
# no user-specific paths or destructive whole-audit-directory deletion.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE="${1:-libero_spatial}"
N_PAIRS="${2:-20}"
N_ENVS="${3:-1}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export POLICY_BACKEND="${POLICY_BACKEND:-cuda}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export AUDIT_DIR="${AUDIT_DIR:-$HOME/probearch-audits/${SUITE}-$(date +%Y%m%d-%H%M%S)}"

echo "ProbeArch fleet: suite=$SUITE episodes/task=$((N_PAIRS * N_ENVS)) audit_dir=$AUDIT_DIR"
exec bash "$REPO/scripts/audit/shared/eval_loop.sh" "$SUITE" "$N_PAIRS" "$N_ENVS" --force
