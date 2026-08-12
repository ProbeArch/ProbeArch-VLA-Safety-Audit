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

## Open hardening items from the telemetry re-review (F3–F7, docs/REVIEW_telemetry.md)
Referenced as pending items from PROTOCOL §2/§3 and HANDOFF risk notes. None of
these block the validation run; all are post-validation hardening.
- [ ] **F3 — record support geometry in rollout telemetry.** `telemetry_rollout.py`
  records no `support_plane_z` / `support_planes` / `static_bodies`, so
  production R4 episodes are scored against the own-init-height fallback anchor;
  only `calibrate.run_trial` control episodes (incl. `off_table_fall`) carry the
  support plane. Emit it per task from static contact geometry as
  `calibrate.derive_support_plane` does; then the scorer's support-plane R4 is
  the production path, not just the control path.
- [ ] **F4 — dirty-tree digest in the run manifest.** `git_revision` is plain
  `git rev-parse HEAD` (currently the v0.1-era commit `647b191`); all fixes are
  uncommitted, so `--resume` cannot detect working-tree code changes. Add a
  digest of the working-tree `scripts/` (e.g. sha256 of the sorted file hashes)
  to `run_manifest.json` and match on it.
- [ ] **F5 — `success_source` diagnostic.** `read_success` `None` reads are
  silently recorded as `False` with no record of which info source produced the
  value; a drift of the pinned gymnasium `final_info` shape would silently
  re-create C1. Record `success_source` (e.g. `final_info-dict` /
  `final_info-list` / `legacy` / `top-level` / `none`) per episode; the synthetic
  shape tests in `smoke_test.py` and `telemetry_rollout.py --selftest` are the
  tripwire until then.
- [ ] **F6 — hard-fail on missing run manifest in standalone scorer/stats.**
  `eval_loop.sh` verifies every task dir carries a `run_manifest.json` with the
  root `run_id` (post-rollout check), but a standalone `safety_scorer.py` /
  `stats.py` run still admits episodes when a task dir has no manifest at all.
  Make `episode_matches_manifest` reject a missing run id outside the loop too.
- [ ] **F7 — preserve R1-eligible contacts in calibration truncation (low).**
  `calibrate.prioritize_r1` truncates by force rank only (`[:CONTACT_LIMIT]`),
  unlike rollout telemetry which never evicts robot-object / object-object
  contacts. Low practical risk (calibration scenes have few contacts), but make
  it symmetric with `collect_telemetry`.

## Long-term
- Extend rule set: task semantics (action preconditions), temporal ordering, reward hacking
- LLM-assist "event narrative" reconstruction from telemetry (open question)
- Cross-model matrix: same protocol on 2B GR00T-N1.5, Pi-0, etc. (needs >4GB VRAM)