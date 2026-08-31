# Execution status

Updated 2026-08-31.

## Completed locally

- Corrected SmolVLA LIBERO-10 and LIBERO-Spatial baseline: 400 episodes.
- End-to-end provenance/video/result verification: passed.
- `probearch` package/CLI and adapter interfaces.
- Telemetry, task-spec, and measurement JSON schemas.
- Annotation guide and blank label template.
- 40-episode RTX-3050 robustness-pilot configuration and matched manifest
  generator.
- Matrix ablation utility and paper outline.
- TurboVLA/X-VLA compatibility gates and source references.

## Requires external execution or people

- Independent double annotation, agreement, adjudication, and held-out labels.
- Actual perturbation rollouts on the CUDA machine.
- TurboVLA load/pilot and X-VLA load/pilot on the RTX 3050.
- Final 100/200-episode model runs, if pilots pass.
- PR creation/merge: the branch is pushed, but the current GitHub token does
  not have the `createPullRequest` permission.

These are not silently marked complete because no local test can substitute for
independent reviewers, a real GPU execution, or GitHub authorization.
