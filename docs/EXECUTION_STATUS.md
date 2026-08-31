# Execution status

Updated 2026-08-31.

## Completed locally

- Corrected SmolVLA LIBERO-10 and LIBERO-Spatial baseline: 400 episodes.
- End-to-end provenance/video/result verification: passed.
- `probearch` package/CLI and adapter interfaces.
- Telemetry, task-spec, and measurement JSON schemas.
- Annotation guide and blank label template.
- Reproducible 100-episode blinded annotation manifest: 25 episodes per
  primary matrix cell, 80/20 development/held-out split, and 50 destination-
  related priority cases.
- 40-episode RTX-3050 robustness-pilot configuration and matched manifest
  generator.
- Full eight-condition robustness manifest (64 matched pilot entries) covering
  clean, displacement, distractor insertion, camera shift, action noise,
  altered initial state, longer horizon, and instruction paraphrase.
- Threshold sensitivity re-scoring artifacts for both 200-episode suites.
- Detector/schema versioning, RuleDetector interface, schema checker, and CI.
- Matrix ablation utility and paper outline.
- TurboVLA/X-VLA compatibility gates and source references.
- TurboVLA official checkpoint download and CUDA load-gate attempt. The gate is
  explicitly `NOT_EVALUATED`: the required gated DINOv3 backbone could not be
  accessed, so no TurboVLA episode or comparison result was produced.
- X-VLA official source/config/processor compatibility check. The processor
  loaded, but the 3.52 GB float32 checkpoint was not moved to the 4 GB GPU;
  X-VLA is therefore also `NOT_EVALUATED`.
- Root README, published JSON/report/matrix consistency, and all repository
  shell-script syntax checks were re-verified.
- Regular wheel installation was smoke-tested outside the checkout; the
  installed CLI can load schemas, validate the pilot config, and materialize
  a robustness manifest without relying on editable-install paths.

## Requires external execution or people

- Independent double annotation, agreement, adjudication, and held-out labels.
- Actual perturbation rollouts on the CUDA machine.
- TurboVLA authorized load/pilot and X-VLA load/pilot on the RTX 3050. TurboVLA
  first requires access to `facebook/dinov3-vitb16-pretrain-lvd1689m`; do not
  replace it with another backbone.
- Final 100/200-episode model runs, if pilots pass.
- PR creation/merge: the branch is pushed, but the current GitHub token does
  not have the `createPullRequest` permission.

These are not silently marked complete because no local test can substitute for
independent reviewers, a real GPU execution, or GitHub authorization.
