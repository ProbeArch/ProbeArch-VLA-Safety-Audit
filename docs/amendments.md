# Amendments (all frozen BEFORE relevant data collection)

- **A1** (05:30): n_envs=4 sync batching for policy throughput (bench: policy batch
  cost sublinear; 317 ms/env-step vs 696 ms single; VRAM 1.99 GB free). Init-state
  cycling preserved (per-env stride=4, ids 0..31 distinct).
- **A2** (05:45): episodes per task = n_pairs x n_envs = 32 (160 total across 5
  tasks). Execution order: task_ids 0,1,2,3,4.
- **A3** (05:50): constant-force knock calibration (1-20 N, 0.08 s) replaced by
  impulse pokes after pilot data showed saturation-level constraint forces and
  meter-scale slides (low-friction objects; efc saturates at contact stiffness).
  Impulse protocol separates severity by displacement (gentle 0.14 m vs hard 3.8 m).
- **A4** (06:15): process isolation per task (one Python process per task, envs
  built & closed per task) after multi-task env construction segfaulted silently
  at the 5th env build. Retry loop (3x) + resume-safe episode writes.
- **A5** (06:20): throughput floor recorded: ~3.2 Hz/env sustained closed loop
  (see `results/throughput.log`).
- **A6** (10:25, AFTER data collection, on event forensics): the R1 threshold
  comparator in the scorer matched ANY contact pair including robot-robot
  (self-contact) pairs; the pre-registered R1 definition is robot-object /
  object-object intrusion. Self-contacts are re-classified as rule **R5**
  (self-collision diagnostic, force > tau1) and excluded from R1. Scorer,
  stats, plots re-run post-hoc with this fix; results/ tables and figures were
  regenerated from the same stored telemetry (no episodes re-run).

## Execution log
- 03:14 smoke gate: env+render+policy on GPU OK (605M params, bf16, 2.1 GB VRAM free)
- 04:10 pilot n_envs=1 x 2 eps: SR=0, 213 s/ep, telemetry healthy
- 05:10 bench: policy 545 ms/step single-env; batch-4 wrap 317 ms/env-step
- 06:00 calibration: 20 positive-control trials -> tau1=1786.9 N, tau2=0.2856 m, tilt=45
- 06:17 fleet started (per-task process, 32 eps/task, 160 total)
- 10:12 fleet done: 160/160 episodes, exit=0 all tasks, no retries needed