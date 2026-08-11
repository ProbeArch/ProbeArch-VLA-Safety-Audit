# ProbeArch VLA Safety Audit

**The success-safety gap in edge-class vision-language-action models: SmolVLA (0.5B) on LIBERO-Spatial, audited with pre-registered specifications and positive controls.**

## Status

- [ ] Phase 0 — Environment bring-up (WSL2, pinned LIBERO + LeRobot, checkpoint verification)
- [ ] Phase 1 — Pre-registration (tasks, unsafe definitions, audit report template)
- [ ] Phase 2 — Positive controls (scripted dangerous rollouts trip the detector)
- [ ] Phase 3 — Baseline n=10/task with CP-95% CIs
- [ ] Phase 4 — Verification (re-seeded re-runs, named mechanisms)
- [ ] Phase 5 — Repower n=50/task; obstacle-perturbed escalation if signal is flat
- [ ] Phase 6 — Write-up, tests, tagged release

## Why this exists

SafeVLA-Bench (arXiv:2606.00773), LIBERO-Safety (ECCV 2026) and SafeLIBERO measure the
success-safety gap (Succ-But-Unsafe rate, violation severity) for 1.7B–7B VLA policies.
None measures the sub-1B edge-class population — the models actually deployed on
Jetson-class hardware. This audit closes that gap, on hardware representative of the
deployment class.

Method discipline (inherited from the exfil audit line of work):
- pre-registered specs, written before any model rollout
- three-way reporting: success / unsafe / no-op — aggregate scores hide cause
- positive controls prove the detector fires
- every anomaly hand-verified from saved rollouts; harness artifacts versioned, not silently fixed

## Layout

```
pre-registration/  per-task specs + audit report template (pre-registered)
scripts/           eval loop, positive controls, scenario tooling
analysis/          result tables, CIs (rollouts gitignored)
pins.md            exact version pins + rationale (the reproducibility contract)
experiment-notes-robotics.md  pre-reg, numbers, findings
```

## License

Apache-2.0