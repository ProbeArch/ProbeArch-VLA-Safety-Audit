# Robustness benchmark protocol v1

The robustness benchmark measures how a fixed policy and fixed audit contract
respond to controlled changes. It must not silently change the policy,
normalization, task success definition, or safety thresholds between clean and
perturbed conditions.

## Matched design

Each condition reuses the same suite, task, initial-state ID, language intent,
episode budget, horizon, and runtime as its `clean` partner. Every episode gets
a unique condition-aware run ID. Failed episodes are retained; retries require
an explicit new episode index and are never substituted silently.

The starter configuration in `configs/robustness_pilot.yaml` is a 40-episode
Spatial pilot: four representative tasks × five conditions × two matched
episodes. It includes two high-success tasks, one medium-success task, and one
low-success task while keeping the budget explicit in its manifest.

`configs/robustness_full.json` expands the same matched design to all seven
planned perturbation types: four representative tasks × eight conditions
(clean plus seven perturbations) × two episodes = 64 manifest entries. It is a
planning/full-coverage manifest, not evidence that those rollouts have already
been executed.

## Conditions

1. `object_displacement`: offset one object at reset by a declared magnitude.
2. `action_noise`: add declared, seeded noise after policy output and before
   environment step; record the exact seed and noise statistics.
3. `instruction_paraphrase`: use a prewritten meaning-preserving variant.
4. `camera_shift`: modify only the declared camera transform.
5. `initial_state`: choose an alternate valid initial state from the same task.
6. `long_horizon`: change horizon only as an explicit ablation.

## Analysis

Report paired success delta, task-aware matrix delta, per-rule episode rates,
95% confidence intervals, evidence coverage, policy latency, and peak VRAM.
Use paired bootstrap intervals or an exact paired test for clean-versus-
perturbed comparisons. Do not call a perturbation result a safety failure
without independent hazard labels.

The pilot remains intentionally one-policy/one-environment at a time for the
RTX-3050 budget. The manifest generator and scorer are local and reproducible;
environment-specific perturbation injection and GPU rollouts are still a
target-machine execution gate.
