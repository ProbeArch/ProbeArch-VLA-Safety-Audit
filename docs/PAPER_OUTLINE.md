# ProbeArch preprint outline

## Working title

Task Success Is Not Enough: Evidence-First, Task-Aware Measurement of Robot
Policy Behavior

## Claims we can responsibly make

1. A benchmark-success bit and a task-aware measurement status answer different
   questions and should be reported separately.
2. A reproducible audit must preserve telemetry, calibration, semantics,
   provenance, and replay evidence together.
3. The current LIBERO results demonstrate a measurement gap, not a validated
   physical hazard rate.

## Required experiments before submission

- Corrected SmolVLA evaluation on LIBERO-10 and LIBERO-Spatial.
- Independent human labels on a stratified held-out sample.
- Threshold and semantic ablations, including destination handling and R5.
- At least one controlled robustness perturbation.
- TurboVLA and X-VLA only if their protocol gates pass; otherwise document the
  incompatibility as a result.

## Paper structure

1. Motivation and failure of success-only evaluation.
2. Measurement contract and task-aware semantics.
3. Telemetry, calibration, provenance, and replay architecture.
4. Experimental protocol and hardware limits.
5. LIBERO results and success × measurement matrices.
6. Independent-label agreement and error analysis.
7. Robustness/ablation results.
8. Limitations: simulation, detector thresholds, missing operational limits,
   no physical-arm validation, and single-device compute constraints.
9. Reproducibility appendix with manifests, schemas, configs, and commands.
