# TurboVLA evaluation runbook

TurboVLA is the only additional policy being advanced in the current phase.
X-VLA is intentionally deferred until TurboVLA has completed its compatibility
gate and matched pilot.

## Current state

- Official TurboVLA source revision:
  `c7c2ba9f94ef3a734c5033be96ef3f3f5a5f3c18`
- Official LIBERO-Spatial checkpoint: retained under the ignored
  `.verification/turbovla-assets/checkpoints/libero/spatial.pth` directory.
- Checkpoint SHA-256:
  `a7c3faa825a6c68d365df0647c39845c3a7bb553e1e24be3729b76de22f703fa`
- Target: RTX 3050 Laptop GPU, approximately 4 GB VRAM.
- Required visual backbone:
  `facebook/dinov3-vitb16-pretrain-lvd1689m`
- Gate result: `NOT_EVALUATED`. The Hugging Face account is valid, but the
  account is not authorized for the gated DINOv3 repository. No substitute
  backbone or checkpoint is permitted.

## After DINOv3 access is granted

Run these steps in the isolated TurboVLA environment. Do not use the existing
SmolVLA environment and do not start a fleet until the dry load succeeds.

```powershell
hf auth login
& .verification/turbovla-venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='facebook/dinov3-vitb16-pretrain-lvd1689m', local_dir='.verification/turbovla-assets/dinov3-vitb16-pretrain-lvd1689m', token=True))"
```

Then run the official TurboVLA load-only path from the TurboVLA checkout. The
two `PYTHONPATH` entries are required because the upstream source keeps
`turbovla` and the VLA-Adapter package in separate source roots:

```powershell
$env:PYTHONPATH = "$PWD\.verification\TurboVLA-src;$PWD\.verification\TurboVLA-src\third_party\vla_adapter"
Push-Location .verification/TurboVLA-src
try {
& ..\turbovla-venv\Scripts\python.exe -m vla_adapter.rollout `
  --ckpt_path ..\turbovla-assets\checkpoints\libero\spatial.pth `
  --dinov3_path ..\turbovla-assets\dinov3-vitb16-pretrain-lvd1689m `
  --bert_path bert-base-uncased `
  --stats_path experiments/libero/configs/libero_all4_stats.json `
  --stats_key libero_all4_no_noops `
  --task_suite_name libero_spatial `
  --num_trials_per_task 1 `
  --chunk_size 12 `
  --num_open_loop_steps 12 `
  --seed 7 `
  --precision bf16 `
  --max_tasks 1 `
  --dry_run_model_load true `
  --video_out_path ..\turbovla-load-test
} finally {
  Pop-Location
}
```

Record the exit code, model-load time, dtype, peak VRAM, and any OOM. A
successful load is still not a valid comparison: it only unlocks the next
pilot gate.

## Pilot and fleet gates

1. Verify action dimension 7, action chunk 12, two DINOv3 image views, BERT
   text preprocessing, image resolution, normalization, and the Franka
   LIBERO embodiment.
2. Run one fixed LIBERO-Spatial task with the official TurboVLA evaluator and
   compare the task ID, initial state, horizon, and success extraction against
   the ProbeArch baseline.
3. Add a thin adapter that converts the official 7-D action output into the
   ProbeArch rollout contract. The scorer, task semantics, calibration rules,
   and safety thresholds must remain unchanged.
4. Run the corrected live smoke gate and save telemetry plus one success/failure
   video where available.
5. Run the matched low-cost pilot before allocating a fleet. Use the same task
   IDs, seeds, horizon, and episode budget as the baseline pilot.
6. Only if the pilot passes, run TurboVLA LIBERO-10 and LIBERO-Spatial. Keep
   each suite in a new audit directory and never silently retry failed episodes.
7. Generate TurboVLA-specific matrices, reports, videos, provenance hashes,
   and model-delta tables. Keep TurboVLA outside pooled claims until every
   protocol field is directly comparable.

## Stop conditions

Stop and record `NOT_EVALUATED` if DINOv3 access is unavailable, the official
checkpoint cannot load, the GPU runs out of memory, preprocessing/action shape
does not match, telemetry is incomplete, or the stock-parity check fails. Do
not lower safety thresholds, substitute a different backbone, or count a
partial rollout as a TurboVLA result.
