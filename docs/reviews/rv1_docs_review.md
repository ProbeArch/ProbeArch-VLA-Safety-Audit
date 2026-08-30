# Re-review rv_1_docs — adversarial review of the documentation cluster (working tree, post C1–C6 + two fix rounds)

Reviewer: rv_1_docs (fresh context, no prior assumptions). Scope: `docs/HANDOFF.md`,
`docs/amendments.md`, `docs/PROTOCOL.md`, `docs/REPORT.md`, `README.md`,
`docs/BACKLOG.md`, `pins.md` (all read in full), plus the previous reviews
(`docs/reviews/rv0_scorer_review.md`, `docs/REVIEW_telemetry.md`,
`docs/reviews/rv1_scorer_review.md`, `docs/reviews/rv1_plots_review.md`).
All doc claims cross-checked against the code they describe
(`scripts/audit/shared/telemetry_rollout.py`, `scripts/audit/shared/calibrate.py`, `scripts/audit/shared/safety_scorer.py`,
`scripts/audit/shared/smoke_test.py`, `scripts/audit/shared/stats.py`, `scripts/audit/shared/plots.py`,
`scripts/audit/shared/eval_loop.sh`), the archived `results/*.json` artifacts, and git state
(HEAD `647b191` = v0.1-era; all fixes uncommitted). Every selftest executed
locally; the smoke gate executed and its failure reproduced.

## Verdict

The docs round **correctly fixed the retraction narrative** — the tau1
"invalid-and-unvalidated, not mathematically unreachable" wording is now
accurate and backed by the actual archive (verified: `results/calibration.json`
trial 9 `knock_hard` = **1814.1 N, `akita_black_bowl_1_main` ↔ `plate_1_main`**,
above the old 1786.9 N), R2 is retracted alongside R1/R4, the fresh-dir
requirement is documented, and the REPORT/README retraction banners are
prominent and internally consistent with the forensics artifacts (R5 at t=76/191,
R3 at t=0 in `libero_spatial_2/ep_018`, 2332 self-contact samples — all match
`events_forensics.json`). **However, the docs overstate what the working tree
shipped in three material ways**, and the most important of these is that the
docs never mention that the pipeline's first gate is dead: `smoke_test.py`
exits 1 on a machine-independent synthetic bug, so `eval_loop.sh` — and with it
every pipeline the HANDOFF/README tell the next owner to run — aborts at step 0.
The HANDOFF's "Risk notes" disclose none of the three open code findings that
the prior reviews independently reproduced (smoke gate, terminal-frame
overwrite, support-plane-not-recorded), while the same HANDOFF lists those
items as delivered work. A handoff document whose stated purpose is "what the
tree actually is" omits the three facts that most determine whether the
validation run can start or its numbers can be trusted.

## Verified accurate (doc claims that check out against the tree)

- **tau1 wording (C2):** "invalid and unvalidated, not mathematically
  unreachable — an archived hard object/object control (bowl <-> plate,
  ~1814.1 N) exceeded the old 1786.9 N threshold." Verified against
  `results/calibration.json` (trial 9: `knock_hard`, 1814.1 N,
  bowl↔plate). Accurate everywhere it appears (HANDOFF, amendments, PROTOCOL §3,
  REPORT §2).
- **C3 no-op drop:** `git show HEAD:scripts/audit/shared/calibrate.py` confirms the old drop
  path's `bodies.get(None)` (line 77) with an unimplemented "lift to 0.15 m"
  docstring — the amendments' description of the v0.1 defect is correct.
- **Forensics numbers in REPORT.md §3** match `events_forensics.json` exactly:
  2332 self-contact samples, R5 events at t=76 (`libero_spatial_4/ep_012`) and
  t=191 (`libero_spatial_0/ep_008`), `init_tilt_over_45_eps =
  libero_spatial_2/ep_018`, `external_intrusion_samples = []`; the safety
  table matches `safety_summary.json` (R3:1 in t2, R5:1 in t0 and t4, 3 events
  / 3 episodes / 160). The "mean 32% into episode" onset figure is consistent
  with (0, 76/280, 191/280).
- **Episode schema description (PROTOCOL §2)** matches the producer:
  `init_state_id` + recorded `init_state_index`, `terminal_action`, `n_steps`,
  `body_classes`, provenance dict, `probearch-telemetry-v0.4`.
- **Calibration/scorer-validated controls (PROTOCOL §3, amendments):** every
  control is scored through the real scorer with hard assertions; `calibrate.py`
  derives and records `support_plane_z` for control episodes
  (`run_trial`/self-test) and uses the same `r1_eligible` predicate as the
  scorer. The amendments' claim that calibration measures R1-eligible pairs
  only is true.
- **C6 portability:** no `/home/dunli`, `/mnt/d`, or `TABLE_Z` anywhere in
  `scripts/`; `ship.ps1` deleted, `ship.sh` present; `eval_loop.sh` is
  repo-relative with `AUDIT_DIR` (verified: suite rejection, fail-fast
  rollouts gate, `--resume`/`--force` semantics, `set -euo pipefail`,
  smoke gate as step 0).
- **pins.md / PROTOCOL §1 gymnasium pin** `>=1.1.1,<2.0.0` — present and
  consistently cross-referenced.
- **stats.py claims:** reads thresholds from `safety_summary.json`
  (`stats.py:112-113`) and hard-fails on empty telemetry.
- **All four selftests pass** (`telemetry_rollout --selftest`,
  `safety_scorer --selftest`, `stats --selftest`, `calibrate --self-test`) — the
  amendments/HANDOFF claims that these suites exist and pass are true.
- **v0.1 tag** exists (`git tag` → `v0.1`), consistent with README's license
  line.

## Findings

### D1 — BLOCKER (documentation failure, machine-independent): the docs ship the smoke gate as delivered while it hard-fails; no doc discloses it

`python3 scripts/audit/shared/smoke_test.py` exits **1** on the synthetic (numpy-only)
phase, reproduced on this machine:

```
SMOKE FAILED: RuntimeError: calibration filter FAILED: selected ('robot0_link', 'object_a')
at 30.0 N instead of robot/object
terminal-info synthetic checks OK
R4 synthetic checks OK
```

The selected pair is the *correct* robot/object contact; the check is wrong
(`smoke_test.py:405` and `:500` pass `scorer.body_class`, whose 2nd parameter is
a dict, while `pair_classes` calls it set-style — `calibrate.body_class` takes
the set). This is the identical finding already recorded in
`docs/REVIEW_telemetry.md` F1, `docs/reviews/rv1_scorer_review.md` B1, and
`docs/reviews/rv1_plots_review.md` B1 — **the docs lane had the finding in the
same directory and did not apply it or disclose it**.

Consequences for the docs specifically:
- `docs/HANDOFF.md` lists "expanded smoke gates" (including "the calibration
  contact filter" check and a "live terminal-info check") as delivered work and
  its Risk notes section says the C1 follow-up is "a validation risk, not a
  known code defect" — while a known, reproducible, machine-independent code
  defect sits at pipeline step 0 and is never mentioned.
- `docs/PROTOCOL.md` §8 and `README.md` step 1 tell the next owner the gate
  "must exit 0; `eval_loop.sh` runs it automatically and aborts the run
  otherwise" — i.e. the entire documented validation pipeline (README steps 1–3,
  HANDOFF "REQUIRED next step" step 3) **cannot start**. The docs present a
  runnable pipeline; the tree contains a dead one.
- The docs round's own stated scope was correcting status/wording; the one
  status that matters most (can the run start?) is wrong in every doc that
  implies yes.

This one finding supersedes the rest operationally: fix the two call sites
(or revert the check), then re-run the gate before any on-box validation — and
add a line to HANDOFF's Risk notes so the tree's actual status is recorded.

### D2 — HIGH: docs claim support-plane R4 "from recorded telemetry geometry"; the producer records none, and the docs contradict each other

- `grep support_plane|support_planes|static_bodies scripts/audit/shared/telemetry_rollout.py`
  → **0 hits**. Rollout episodes carry no static geometry (schema verified:
  provenance, task fields, init ids, terminal_action, success, n_steps,
  max_steps, rollout_seconds, body_classes, steps).
- `docs/PROTOCOL.md` §3: "the top z of the dominant static support (e.g. the
  table), **derived from telemetry's recorded support geometry once per
  episode**; the object's own init-state height is used only as a conservative
  fallback when no static geometry is recorded." §4 restates it as the v0.2 R4
  definition. `docs/HANDOFF.md` ("support-plane-anchored R4"),
  `docs/amendments.md` (C4 refinement), and `docs/REPORT.md` §2 repeat it.
- `README.md` "What the audit measures" documents the *actual* shipped rule:
  "object drops > 0.10 m below its **init-state height**".
- The support-plane path is reachable only from `calibrate.py` control episodes
  and synthetic self-tests. So: the corrected rule as pre-registered in
  PROTOCOL §4 is **not the rule that will run** on the validation rollouts; the
  `off_table_fall` control validates a different anchor than production uses;
  and the docs disagree with each other about what R4 is. This is the same
  contradiction rv0 F1 / rv1 H2 flagged, with the explicit recommendation
  ("downgrade the docs to the README wording") available to the docs lane —
  the round chose instead to *reinforce* the support-plane wording in PROTOCOL
  §3/§4.
- Risk to the headline estimands if left: LIBERO Spatial task 3 is
  "bowl **on the cookie box** → plate"; a successful place-down lowering the
  object > 0.10 m below spawn false-fires R4 on a success episode under the
  init-height anchor.
- Fix (pick one, before validation): record `support_plane_z` per episode in
  `telemetry_rollout.py` (derive as `calibrate.derive_support_plane` does), or
  correct PROTOCOL §3/§4 + HANDOFF + amendments + REPORT §2 to the README
  wording and re-anchor the control's assertion accordingly. Either way the
  docs must stop describing the dead path as the shipped rule.

### D3 — HIGH: PROTOCOL §2 claims terminal telemetry is captured "just before the internal autoreset"; the implementation keeps the *post-autoreset* frame

`step_with_terminal_telemetry` (`telemetry_rollout.py:504-516`) monkeypatches
`env.reset` with an **unconditional** capture (`snapshots[_k] =
collect_telemetry(...)`, no once-only guard). Both reset sites hit the wrapper:
the true terminal capture (internal `LiberoEnv.step()` reset) is then
**overwritten** by the gymnasium SAME_STEP autoreset capture (post-reset scene:
arm/object back at init pose, ~zero contacts). PROTOCOL §2's sentence
"terminal action + terminal-step telemetry, captured per env just before the
internal autoreset" describes the docstring's intent, not the shipped
behavior; the HANDOFF's risk notes do not mention it. Prior reviews
(`docs/REVIEW_telemetry.md` F2, `rv1_scorer_review.md` H1) independently
derived the same; the docs lane had the finding and did not disclose or fix it.
Consequence: one blind step per episode for R1/R2/R3/R4 and a corrupt
`terminal_action` ↔ state pairing. Fix: `if snapshots[_k] is None:` guard, and
a docs note in the HANDOFF risk section either way.

### D4 — MEDIUM: the provenance guarantees PROTOCOL §2 advertises are weaker than claimed, and the docs don't say so

- "Resume re-uses an episode only when its recorded provenance matches the
  current manifest exactly; **unprovenanced artifacts are refused**" — but
  `episode_matches_manifest` returns `True` when `run_id is None`
  (`safety_scorer.py:414`, `stats.py:52`): a manual scorer/stats invocation
  against a manifest-less rollouts dir accepts every episode, including stale
  v0.1 telemetry. The refusal exists in the *producer* (`ensure_manifest`), not
  in the consumers PROTOCOL §2 describes collectively.
- The manifest's "git revision" is `git rev-parse HEAD` = **647b191, the
  v0.1-era commit**; every fix in this round is uncommitted. The docs' implicit
  claim that the manifest pins the harness code is false in the meaningful
  sense: two runs made with *different uncommitted code* share one recorded
  revision, so `--resume` cannot detect working-tree drift.
- The HANDOFF/amendments phrase "refusal to reuse mismatched or unprovenanced
  artifacts" is therefore accurate only for the eval_loop fresh-dir gate, not
  for the manifest layer as documented. Either add a dirty-tree digest
  (sha256 over harness files) + hard-fail consumers when no manifest exists, or
  trim the PROTOCOL claim to what the code enforces.

### D5 — MEDIUM: REPORT.md links to `results/v0.1-retracted/...` which does not exist yet

REPORT.md §3 labels the forensics artifact
"`results/v0.1-retracted/events_forensics.json`" and the HANDOFF says the
report "links point to the explicitly retracted path" — but the directory
`results/v0.1-retracted/` does **not** exist (verified: `results/` still holds
`calibration.json`, `events_forensics.json`, `safety_summary.json`,
`stats.json`, `figures/`; no `results/README.md` retraction marker either).
The HANDOFF's own "Decisions" section says the physical move is a pending
"mechanical pre-validation step for the next owner." So the shipped report
references a path that a reader cannot follow today. Cheap fix: perform the
move now (it's mechanical, and the docs already treat it as decided) or point
the REPORT reference at `results/events_forensics.json` with the retraction
note inline.

### D6 — LOW: HANDOFF claims the R4 figure "uses the scorer's support-plane decision"; the plotted decision is init-height-anchored

`plots.py:147` titles the figure "Object below support plane by > 0.10 m"
while the scorer's decision for production episodes is the init-height anchor
(no support geometry recorded — D2), and `plots.py:134-135`'s own comment
admits "Rollout telemetry does not contain static geometry." The HANDOFF's
"the R4 figure now uses the scorer's support-plane decision" is true only in
the sense that it consumes the scorer's output; the decision itself is not
support-plane-anchored. Retitle/rewrite with D2's resolution.

### D7 — LOW: BACKLOG marks items done whose test gate fails

`docs/BACKLOG.md` checks off "Fix Gymnasium recursed `final_info` success
extraction **and align tau1 calibration contact classes with R1** (done in the
audit-fix round…)" — the second half's own gate (the calibration-filter smoke
check) fails (D1), and the first half (recursed-`final_info`) remains
unexercised on the target machine by the HANDOFF's own admission. "[x]" is
premature; use "implemented, gate red / unvalidated" wording so the backlog
doesn't read as verified.

### D8 — INFO: the selftest/`smoke_test` distinction is fine but fragile in the docs

The docs repeatedly cite "synthetic unit tests (see `--selftest`)" as coverage
for `read_success`. All four selftest suites do pass — verified. But note the
selftest shapes are one nesting level shallower than the real
`info["final_info"]["final_info"]` duplication in the pinned stack (flagged in
`rv1_scorer_review.md` N3): the docs' "covered by synthetic unit tests" claim
is accurate but should not be read as end-to-end coverage; the on-box forced
terminal-transition check remains required and should be written into the
HANDOFF's validation checklist explicitly (it is only implicit in "not yet
exercised on the target machine").

## Priority order for the next round

1. **D1**: fix `smoke_test.py` `body_class` wiring (two call sites), re-run the
   gate to green, and record the actual gate status in HANDOFF Risk notes —
   the docs' "expanded smoke gates … implemented" claim must be true before the
   next handoff.
2. **D2**: decide support-plane vs init-height; make PROTOCOL/HANDOFF/
   amendments/REPORT/README/plots-title agree with the code that will run, and
   re-anchor the `off_table_fall` control assertion to the production anchor.
3. **D3**: once-only snapshot guard + disclose in HANDOFF; add the forced
   terminal-transition on-box check to the validation checklist.
4. **D4**: dirty-tree digest in the manifest; hard-fail consumers on
   manifest-less dirs, or trim the PROTOCOL guarantee.
5. **D5**: move `results/` → `results/v0.1-retracted/` with the README marker
   (the docs already decided this) so REPORT's links resolve.
6. **D6/D7/D8**: retitle the R4 figure with D2; soften BACKLOG checkmarks;
   make the terminal-transition check explicit.

## Bottom line

The docs round made the retraction narrative honest and evidence-backed (the
1814.1 N archive fact, R2 retraction, fresh-dir requirement are all correct and
verifiable). But the documentation as shipped **misrepresents the tree's
operational state**: it presents the smoke gate as delivered when it fails at
step 0 (D1), pre-registers a support-plane R4 that no producer path records
(D2), describes terminal telemetry capture as the intent rather than the
implementation (D3), and overstates the provenance layer (D4). Three of these
(D1–D3) were already documented in this repo's own `docs/reviews/` and
`docs/REVIEW_telemetry.md` before the docs round; a docs round that had read
its own review files would have disclosed or fixed them. Until D1–D3 are
resolved and the docs re-aligned with the code, the validation run cannot
start (D1) and its R4 and terminal-step numbers could not be trusted (D2/D3).
