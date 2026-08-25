#!/usr/bin/env bash
set -e
export AUDIT_DIR=/tmp/test-brain
rm -rf "$AUDIT_DIR"
mkdir -p "$AUDIT_DIR"
export PATH="/home/dunli/miniconda3/envs/vla-audit/bin:$PATH"
export MUJOCO_GL=egl
export POLICY_BACKEND=cuda
cd /mnt/d/ProbeArch-VLA-Safety-Audit
timeout 600 bash scripts/_backend_map/shared/eval_loop.sh libero_spatial 1 1 --force 2>&1 | tee /tmp/test-brain.log
echo ---done
cat /tmp/test-brain/stats.json 2>&1 | head -n 80
cat /tmp/test-brain/safety_summary.json 2>&1 | head -n 80
echo ---ep check
/home/dunli/miniconda3/envs/vla-audit/bin/python -c "
import json, pathlib
p = pathlib.Path('/tmp/test-brain/rollouts/libero_spatial_0/ep_000.json')
ep = json.loads(p.read_text())
print('success', ep['success'], 'n_steps', ep['n_steps'])
zs = [s['eef'][2] for s in ep['steps'] if s.get('eef')]
print('eef z', min(zs), max(zs))
maxf=0
for s in ep['steps']:
 for d in s.get('contact_details',[]):
  maxf=max(maxf, d.get('force_N',0))
print('max force', maxf)
print('safety', ep.get('safety_events'))
"
