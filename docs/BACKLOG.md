# Stewardship Backlog — next steps after v0.1

## Short-term (recommended next)
- [ ] Re-run with `n_pairs=8` + cross-seed re-seeding (beyond deterministic init cycling)
- [ ] Reproduce on the Visual/NEW suites (needs full LIBERO datasets; method identical)
- [ ] Share pilot report with design partners (below); collect pre-audit expectations
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
migration, overturns, fall-through) from positive-control calibration. Code,
protocol, and results are public. If your team works on robot VLA safety cases,
we'd like to swap notes on rule design and calibration protocols — reply and
we'll share the report draft early.

## Long-term
- Extend rule set: task semantics (action preconditions), temporal ordering, reward hacking
- LLM-assist "event narrative" reconstruction from telemetry (open question)
- Cross-model matrix: same protocol on 2B GR00T-N1.5, Pi-0, etc. (needs >4GB VRAM)