#!/usr/bin/env bash
set -e
export AUDIT_DIR=/tmp/fixed_single
rm -rf "$AUDIT_DIR"
mkdir -p "$AUDIT_DIR"
export PATH="/home/dunli/miniconda3/envs/vla-audit/bin:$PATH"
export MUJOCO_GL=egl
export POLICY_BACKEND=cuda
cd /mnt/d/ProbeArch-VLA-Safety-Audit
echo "calibrate"
N_TRIALS=1 MAX_TRIALS=1 python scripts/_backend_map/shared/calibrate.py --suite libero_spatial --task-id 0 --n-trials 1 --max-trials 1 --out "$AUDIT_DIR/calibration.json" 2>&1 | tail -n 20
echo "rollout 1 task"
python scripts/_backend_map/shared/telemetry_rollout.py --suite libero_spatial --task_ids 0 --n_envs 1 --n_pairs 1 --out "$AUDIT_DIR/rollouts" 2>&1 | tail -n 50
echo "score"
python scripts/_backend_map/shared/safety_scorer.py 2>&1 | tail -n 20
echo "stats"
python scripts/_backend_map/shared/stats.py 2>&1 | tail -n 20
echo "ep check"
if [[ -f /mnt/d/ProbeArch-VLA-Safety-Audit/check_ep.py ]]; then
  python /mnt/d/ProbeArch-VLA-Safety-Audit/check_ep.py /tmp/fixed_single
else
  echo "check_ep.py not present; skipping legacy diagnostic"
fi
echo "eef check"
/home/dunli/miniconda3/envs/vla-audit/bin/python -c "
import json, pathlib
p = pathlib.Path('/tmp/fixed_single/rollouts/libero_spatial_0/ep_000.json')
ep = json.loads(p.read_text())
zs = [s['eef'][2] for s in ep['steps'] if s.get('eef')]
print('eef min', min(zs), 'max', max(zs), 'mean', sum(zs)/len(zs))
print('reaches table 0.9?', min(zs) < 0.95)
maxf=0
for s in ep['steps']:
 for d in s.get('contact_details',[]):
  maxf=max(maxf, d.get('force_N',0))
print('max force', maxf)
print('safety', ep.get('safety_events'))
"
