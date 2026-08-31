# ProbeArch result packages

## Current

- [`libero_10-taskaware-20260830/`](libero_10-taskaware-20260830/README.md) —
  SmolVLA, 200 episodes, 64 task successes; corrected task-semantics matrix.
- [`libero_spatial-200-20260830/`](libero_spatial-200-20260830/README.md) —
  SmolVLA, 200 episodes, 151 task successes; corrected task-semantics matrix.

The current packages contain curated reports, aggregate JSON, figures,
calibration profiles, manifests, and outcome-verified videos. Large raw rollout
telemetry remains under the local ignored `audits/` tree; `dataset_freeze.json`
records its hashes and provenance.

## Historical

- `libero_10-200-20260825/` predates the current task-aware package and is kept
  for history, not as the headline result.
- `v0.1-retracted/` contains the invalid 2026-08-12 run. Its 0/160 success and
  safety claims were instrumentation artifacts and must not be cited.
- Each current package contains `superseded-task-semantics-v1/`, preserving an
  initial offline interpretation that incorrectly treated destination objects
  as distractors. Those matrices must not be cited as current.
