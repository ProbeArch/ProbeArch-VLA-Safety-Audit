# Stewardship Backlog — current validated baseline and next steps

## Current baseline (2026-08-31)

- [x] Complete fresh corrected CUDA calibration and 200-episode LIBERO-10 run.
- [x] Complete fresh corrected CUDA calibration and 200-episode LIBERO-Spatial run.
- [x] Verify all 400 raw episode hashes, manifests, task success labels,
  calibration profiles, aggregate totals, and representative video hashes.
- [x] Correct task semantics so commanded destinations are not distractors, R5
  is diagnostic-only, and missing semantic evidence becomes `NOT_EVALUATED`.
- [x] Regenerate both published matrices/reports and preserve the superseded
  interpretations for audit history.
- [x] Add a dependency-light `probearch` CLI, adapter interfaces, versioned
  schemas, annotation guide, and matched robustness-pilot manifest.
- [x] Add the paper outline, model compatibility gates, matrix-ablation utility,
  and execution-status handoff.
- [x] Add a blinded 100-episode annotation sampler with balanced outcome strata,
  destination-case oversampling, and development/held-out split assignment.
- [x] Add threshold sensitivity re-scoring for both published suites and attach
  the machine-readable outputs to their result packages, including independent
  τ1/τ2/tilt/fall-margin sweeps.
- [x] Add detector/schema versioning, a RuleDetector extension interface, full
  perturbation-manifest coverage, dependency-free schema checks, and CI.
- [x] Push the completed baseline branch. PR creation is pending GitHub token
  permission (`createPullRequest`).
- [ ] Obtain independent human/expert labels and define operational limits;
  current safe/unsafe cells remain candidate measurement statuses, not hazards.

## Short-term (recommended next)
- [x] Fix Gymnasium recursed `final_info` success extraction and align tau1
  calibration contact classes with R1 (done in the audit-fix round:
  `telemetry_rollout.read_success` + `calibrate.r1_eligible`, scorer-validated)
- [x] Record the compatible Gymnasium pin (`>=1.1.1,<2.0.0`) in `pins.md` — done
  (the last pre-run static item; nothing left on the static checklist)
- [x] Fix the F1 smoke-gate blocker (re-review F1, `docs/REVIEW_telemetry.md`):
  the two `body_class` call sites in `smoke_test.py` now use `calibrate.body_class`
  (set contract); `python3 scripts/audit/shared/smoke_test.py` → `SMOKE PASSED` locally
  (numpy-only phase). Done in the final-fix round; live-phase checks still need
  the target machine.
- [x] Run the corrected validation pipeline with fresh task-scoped calibration
  and frozen telemetry — completed for LIBERO-10 and LIBERO-Spatial.
- [ ] Re-run with `n_pairs=8` + cross-seed re-seeding (beyond deterministic init cycling)
- [ ] Reproduce on the Visual/NEW suites (needs full LIBERO datasets; method identical)
- [ ] Share the corrected report with design partners while stating that the
  task-aware labels have not yet been independently validated as hazards.
- [ ] Publish raw telemetry archive (tar.gz of rollouts JSON) + offline viewer

## Cross-model audit expansion

The original audit remains the primary experiment. Add models one at a time
under the same corrected harness, LIBERO task protocol, calibration artifacts,
safety rules, and report schema. A model must not be compared using a different
success definition, horizon, seed policy, image resolution, or action format.

### Model 1 — TurboVLA (first)

- [ ] Pin the official TurboVLA repository commit and the exact LIBERO
  checkpoint; record the checkpoint SHA-256, license, and model configuration.
- [ ] Create a separate Python 3.10 evaluation environment so TurboVLA
  dependencies cannot silently change the validated SmolVLA environment.
- [ ] Confirm the official LIBERO entry points for `libero_10` and
  `libero_spatial`; do not assume a RoboTwin checkpoint is interchangeable.
- [ ] Run a model-load smoke test on the RTX 3050 and record peak VRAM,
  inference latency, dtype, image resolution, and sustained closed-loop rate.
- [ ] Run stock-policy parity on one fixed LIBERO-Spatial task before any
  safety fleet run.
- [ ] Run the corrected live smoke gate, including success extraction,
  telemetry fields, calibration compatibility, and video rendering.
- [ ] Run a small matched pilot: the same task IDs, seeds, horizon, and number
  of episodes used for the existing policy pilot.
- [ ] Compare TurboVLA against the existing policy only after both manifests
  pass validation; report task success, safety coverage, and all four outcome
  classes separately.
- [ ] Run the final TurboVLA LIBERO-10 and LIBERO-Spatial evaluation only after
  the pilot passes. Preserve failed episodes and do not silently retry them.
- [ ] Generate TurboVLA-specific reports, confusion matrices, calibration
  outputs, representative success/failure videos, and model-delta tables.
- [ ] Mark the run `BLOCKED` or `NOT_EVALUATED` if the checkpoint cannot run on
  the available GPU or does not emit the required telemetry; do not replace it
  with a different checkpoint without a new manifest.

**TurboVLA exit gate:** a clean, reproducible pilot completes on the target
machine and its report is directly comparable with the corrected baseline.

### Model 2 — X-VLA (second)

- [ ] Pin the official X-VLA implementation and the exact LIBERO checkpoint;
  record the checkpoint SHA-256, action mode, domain ID, and processor version.
- [ ] Confirm whether the chosen X-VLA checkpoint uses the same LIBERO robot,
  camera layout, action convention, normalization, and control mode as the
  audit harness.
- [ ] Create a separate evaluation environment or lockfile for X-VLA.
- [ ] Run model-load, preprocessing, action-shape, and latency smoke tests
  before allocating episodes.
- [ ] Add a thin policy adapter that converts X-VLA output into the existing
  rollout action contract without changing the scorer or safety rules.
- [ ] Run stock-policy parity and the corrected live smoke gate.
- [ ] Run the same small matched pilot used for the baseline and TurboVLA.
- [ ] Run final LIBERO-10 and LIBERO-Spatial evaluations only if the pilot,
  manifest, and telemetry checks pass.
- [ ] Add X-VLA to the cross-model matrix only when all models share the same
  protocol; otherwise publish a descriptive, non-ranking comparison.

**X-VLA exit gate:** the model produces valid, protocol-compatible rollouts and
the audit conclusions remain traceable to the same task and safety contract.

### Cross-model reporting rules

- [ ] Keep model identity, checkpoint, environment, and policy configuration
  visible in every report and video index.
- [ ] Report per-suite and per-task results before any pooled number.
- [ ] Include task success, `SAFE_SUCCESS`, `UNSAFE_SUCCESS`, `SAFE_FAILURE`,
  `UNSAFE_FAILURE`, `NOT_EVALUATED`, safety coverage, and confidence intervals.
- [ ] Do not rank models using safety percentages when evidence coverage differs.
- [ ] Include compute cost, peak VRAM, latency, throughput, and episode budget
  as reproducibility fields—not as safety outcomes.
- [ ] Treat TurboVLA and X-VLA as research evaluation targets, not evidence of
  physical-robot safety or certification.

## Design partners (people/orgs to review before wider release)
- Community: LeRobot HF team (policy load + eval parity findings, GR00T dataclass bug report + patch)
- Community: HuggingFaceVLA maintainers (smolvla_libero checkpoint; eval harness trace)
- Research: university robotics-safety labs (safety-case methodology review)
- Industry: embodied-AI evaluation groups (threat-model feedback for R1-R4 rule set)

## "Failure First" outreach (interest-check draft)
Subject: VLA safety audit — pre-registered, open code, public prereg

Body: We ran a pre-registered safety audit of a published open-source 0.5B VLA
(smolvla_libero) on vanilla LIBERO Spatial, measuring the success-safety gap:
task success rate vs. pre-registered intrusion rules (impact forces, object
migration, overturns, fall-through) from positive-control calibration. Code and
protocol are public. **The v0.1 results are RETRACTED** pending a corrected
re-run (harness defects invalidated the headline numbers; see docs/REPORT.md) —
do not cite them as current findings. If your team works on robot VLA safety
cases, we'd like to swap notes on rule design and calibration protocols — reply
and we'll share the report draft early.

## Telemetry re-review closure (F3–F7, docs/REVIEW_telemetry.md)
F3–F7 are closed in the current producer/consumer path. The remaining work is
target-machine validation and re-derived calibration, not static hardening.
- [x] **F3 — record support geometry in rollout telemetry.** Per-object support
  planes use the same `calibrate.derive_support_plane` geometry-top calculation;
  a common support height is emitted in the compact field when applicable.
- [x] **F4 — dirty-tree digest in the run manifest.** `git_revision` includes a
  digest of the tracked `git diff HEAD` when source is dirty, and policy digest
  absence is a hard error for manifest validation.
- [x] **F5 — `success_source` diagnostic.** `read_success_with_source` records
  the terminal info shape used for every episode, including explicit masked/none
  outcomes; synthetic shape tests cover the source labels.
- [x] **F6 — hard-fail on missing run manifest in standalone consumers.**
  `safety_scorer.py`, `stats.py`, and `plots.py` reject episodes when a task
  manifest is absent or unreadable, in addition to filtering mismatched run IDs.
- [x] **F7 — preserve R1-eligible contacts in calibration truncation (low).**
  `calibrate.prioritize_r1` retains eligible contacts before filling the remaining
  contact budget, matching `collect_telemetry`.

## Long-term
- Extend rule set: task semantics (action preconditions), temporal ordering, reward hacking
- LLM-assist "event narrative" reconstruction from telemetry (open question)
- Expand the cross-model matrix to additional policies such as GR00T-N1.5 or
  pi0 only after TurboVLA and X-VLA pass the same protocol and the hardware
  budget is understood.
