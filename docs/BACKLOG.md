# Stewardship Backlog — next steps after v0.1

## Short-term (recommended next)
- [x] Fix Gymnasium recursed `final_info` success extraction and align tau1
  calibration contact classes with R1 (done in the audit-fix round:
  `telemetry_rollout.read_success` + `calibrate.r1_eligible`, scorer-validated)
- [x] Record the compatible Gymnasium pin (`>=1.1.1,<2.0.0`) in `pins.md` — done
  (the last pre-run static item; nothing left on the static checklist)
- [x] Fix the F1 smoke-gate blocker (re-review F1, `docs/REVIEW_telemetry.md`):
  the two `body_class` call sites in `smoke_test.py` now use `calibrate.body_class`
  (set contract); `python3 scripts/smoke_test.py` → `SMOKE PASSED` locally
  (numpy-only phase). Done in the final-fix round; live-phase checks still need
  the target machine.
- [ ] Run the corrected validation pipeline (fresh `AUDIT_DIR` + re-derived
  tau1/tau2; see docs/HANDOFF.md "REQUIRED next step") — blocks everything below
- [ ] Re-run with `n_pairs=8` + cross-seed re-seeding (beyond deterministic init cycling)
- [ ] Reproduce on the Visual/NEW suites (needs full LIBERO datasets; method identical)
- [ ] Share pilot report with design partners (below) — **BLOCKED until the
  validation run completes**: the v0.1 results are retracted and must not be
  presented as current findings
- [ ] Publish raw telemetry archive (tar.gz of rollouts JSON) + offline viewer

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
- Cross-model matrix: same protocol on 2B GR00T-N1.5, Pi-0, etc. (needs >4GB VRAM)
