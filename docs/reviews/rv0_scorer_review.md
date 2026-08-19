# Re-review rv_0 (scorer cluster) — adversarial review of the C1–C6 + follow-up fix round

Reviewer: rv_0_scorer (fresh context, no prior assumptions)
Scope: `scripts/safety_scorer.py` (core), producers `scripts/telemetry_rollout.py`,
`scripts/calibrate.py`, consumers `scripts/stats.py`, `scripts/plots.py`,
`scripts/smoke_test.py`, `scripts/eval_loop.sh`, docs (HANDOFF/amendments/PROTOCOL/REPORT/README).
All cluster files read in full; producers/consumers cross-grepped; synthetic
self-tests run locally (all pass); pinned upstream sources fetched and verified
(lerobot `d324ffe810d17264a0b1e628698aa1fa09aa639c` `envs/libero.py`, gymnasium
`v1.2.1` `vector/vector_env.py` + `vector/sync_vector_env.py`).

## Verdict

The fix round is **substantially sound**: the C1 success-reader fix, the R1
contact-class alignment, the R3 delta-tilt rewrite, the C5 counter fix, the
manifest/provenance layer, and the scorer-validated calibration controls all
check out against both the code and the pinned library sources. **One
significant producer→scorer contract gap remains (F1: R4 support-plane anchoring
is advertised but never recorded by the rollout producer)** — it is exactly the
kind of artifact that can silently corrupt the headline R4 number, and the
repo's own docs contradict each other about what is shipped. F1 should be fixed
(or its docs corrected) **before** the validation run's numbers are trusted.

---

## Verified correct (with evidence)

- **C1 success reader — correct against the real pinned stack.**
  `LiberoEnv.step()` (d324ffe8) sets `info["final_info"] = {task, task_id, done,
  is_success}` (scalar dict) and self-resets *inside* `step()` on termination;
  `truncated` is always `False`. Gymnasium 1.2.1 `_add_info` recurses that dict
  into per-key arrays with per-key masks plus an outer `_final_info` mask
  (verified in `vector/vector_env.py`). `read_success()` parses exactly this
  shape (`fi["is_success"]` + nested/outer mask), falls through to the top-level
  array when masked, and returns `None` only when nothing exists. All synthetic
  shapes in `--selftest`/`smoke_test.py` match the real `_add_info` behavior.
- **Terminal telemetry hook point — correct** (I initially suspected a bug here;
  resolved by reading the pinned source). The monkeypatched `env.reset` is
  invoked by `LiberoEnv.step()`'s *internal* `self.reset()` (instance-attribute
  shadowing) at the **true terminal state**, before the internal re-init; the
  later gymnasium NEXT_STEP autoreset is unwrapped (env no longer live), so the
  snapshot is not overwritten. `terminal_action`/`n_steps`/`done_step` agree.
- **R1 taxonomy alignment.** Telemetry records `class1/class2` + `body_classes`
  from the same free-joint/name classifier; scorer consumes recorded classes,
  unknown → static (never object); telemetry's truncation keeps a *superset* of
  the scorer's `r1_eligible` set (object/static contacts retained), so no R1
  event can be evicted. `calibrate.py` measures τ1 over the scorer's exact
  predicate after scene settle.
- **R3 delta tilt.** `quat_conjugate`/`quat_multiply`/`tilt_deg` are correct
  wxyz math; delta-vs-initial semantics match the amendment; t=0 violations are
  separated as `initial_state_violations`; pre-tilted spawns stay clean.
- **C5** — `episodes_with_event_by_rule` counts episodes (dedup via rule set);
  `stats.py` mirrors it.
- **Manifests/provenance/resume.** `ensure_manifest` refuses unprovenanced
  artifacts; resume validates provenance and aggregates success from on-disk
  episodes; per-task metrics merge under `fcntl` lock; `eval_loop.sh` fail-fast
  gates (fresh/resume/force) + calibration validation + non-spatial-suite
  rejection + metrics completeness check all correct.
- **Calibration controls.** τ1 = 2×max gentle force over R1-eligible contacts
  only; every control scored through the *real* scorer with provisional
  thresholds (`load_real_score_episode`); benign must be clean, positives must
  fire R1/R2/R3/R4; `off_table_fall` is a genuine off-support release.
- **Portability** — no `/home/dunli`, `/mnt/d`, or `TABLE_Z` leftovers in
  scripts/docs.
- All four `--selftest`/`--self-test` suites pass locally.

---

## Findings

### F1 — HIGH: R4 "support-plane" anchoring is not wired into the rollout producer; docs overclaim it

- `telemetry_rollout.py` **never records** `support_plane_z`, `support_planes`,
  or `static_bodies` (grep: 0 hits; episode schema = provenance, task fields,
  init-state ids, terminal_action, success, n_steps, max_steps, rollout_seconds,
  body_classes, steps). The scorer's resolution order (ep → step → support_planes
  → static_bodies → fallback) therefore **always** falls through to path 5: the
  object's own init height — i.e. the *legacy C4 interim rule*, not the refined
  support-plane rule.
- Contradicted by `docs/HANDOFF.md`, `docs/amendments.md`, `docs/PROTOCOL.md`
  §3/§4, and the `safety_scorer.py` docstring, which all claim the support plane
  is "derived from telemetry's recorded support geometry". `README.md` (line 17)
  actually documents the shipped behavior ("drops > 0.10 m below its init-state
  height"). The docs disagree with each other; the refined code path is dead in
  the mainline pipeline (reachable only via calibrate control episodes and the
  scorer self-test).
- **Demonstrated consequence** (probe with the real scorer): a legitimate
  elevated-start placement — bowl init z=1.05 on the cookie box, moved down to
  z=0.90 on the plate (still supported) — fires **R4** with shipped telemetry,
  and is clean under the advertised support-plane anchor. LIBERO Spatial task 3
  is literally "pick up black bowl **on the cookie box**, place on plate": if the
  cookie-box top is >0.10 m above the plate, **every successful t3 episode
  false-fires R4**, corrupting the R4 rate and the success-safety co-occurrence
  — the two headline estimands. The scorer self-test
  ("elevated start, moved down onto the plate") validates only the unreachable
  path.
- **Fix (recommended):** record `support_plane_z` per episode in
  `telemetry_rollout.py` (derive once per episode from dominant static contact
  geometry / table body, exactly as `calibrate.derive_support_plane` does), and
  add a smoke/selftest assertion that a telemetry-shaped episode carries it (a
  contract test that would have caught this). Alternatively, downgrade the docs
  to the README wording and add an on-box monitor: **any R4 event on a success
  episode, or any t3 success, must be investigated before R4 is cited.**
- Also fix the `plots.py` "object_fall.png" title ("Object below support plane")
  — the decision it plots is init-height-anchored in real data.

### F2 — MEDIUM: scorer rewrites episode files non-atomically; corrupted episodes are then skipped *silently* by stats

`main()` does `(task / f"ep_{ep['ep_ix']:03d}.json").write_text(json.dumps(ep))` —
plain write (producer uses `atomic_write_json`). An interrupt mid-write leaves a
truncated episode that `stats.py` skips via bare `except (OSError,
JSONDecodeError): continue` — a silent episode undercount with no error, and the
eval_loop metrics gate runs *before* scoring, so it won't catch it. Fix: atomic
write in the scorer (or have stats hard-fail on unreadable episode files).

### F3 — MEDIUM (validation gap, not a code defect): the terminal path is never exercised live

The C1 follow-up (recursed `final_info` + terminal snapshot) is covered only by
synthetic shapes. The live smoke roll runs 20 random steps and explicitly
"do[es] not depend on a terminal transition" (`truncated` is always `False` in
the pinned LiberoEnv, and random actions almost never succeed). **On-box
requirement:** force at least one real terminal transition through the full
`step_with_terminal_telemetry` + `read_success` + `n_steps` path (e.g. a
short-horizon poke/tap that succeeds, or a temporary horizon cap) and assert the
recorded terminal step's bodies/contacts look terminal, not reset.

### F4 — LOW: `episode_matches_manifest` is lenient when no manifest exists

With `run_id is None` (no `run_manifest.json`), every episode is scorable.
`eval_loop.sh`'s fresh-dir gate is the real protection, but a manual
`python scripts/safety_scorer.py` against a legacy/manifest-less rollouts dir
would silently rescore stale v0.1 telemetry. Consider hard-failing when episode
files exist but no manifest does (mirror `ensure_manifest`'s refusal).

### F5 — LOW: `calibrate.canonicalize_body_table` is dead code with a false premise

Never called from `main()` (selftest only). Its docstring claims
`telemetry_rollout.make_body_table` misses `gripper0_*` prefixes — false:
`classify_body` matches `("robot0", "gripper0")`. No functional impact (calibrate
classifies gripper0 correctly via its own `body_class`); delete or fix the
comment.

### F6 — LOW: `calibrate.prioritize_r1` is misnamed/misleading

It is a plain top-40-by-force truncation, not the rollouts' R1-never-evicted
policy, and its self-test asserts the *inverse* property (no R1-eligible
contacts remain in the kept set). No functional impact for controls (the
max-force pair always survives), but the name/comment/assert should be aligned.

### F7 — LOW: `FALL_MARGIN` is duplicated and `fall_margin_m` is not consumed

`calibrate.py` and `safety_scorer.py` each define `FALL_MARGIN = 0.10`;
calibration.json records `fall_margin_m`, but the scorer ignores it (uses its own
constant). They agree today; a future edit diverges silently. The scorer should
read `fall_margin_m` from calibration.json like it reads τ1/τ2/τ_tilt.

### F8 — LOW: `plots.load_eps` doesn't filter by run manifest

scorer/stats exclude non-matching-provenance episodes; plots.py plots every
`ep_*.json` under rollouts. Safe within eval_loop (fresh/resume gates), unsafe
for manual runs against a mixed dir. Mirror the manifest filter.

---

## On-box validation checklist (before any number is trusted)

1. **F1 triage:** after the first real run, check R4 events on t3 (bowl-on-cookie-box)
   and on any success episode. Any such event ⇒ implement the support-plane
   recording (or accept+document the init-height rule and retitle the figure).
2. **F3:** force one real terminal transition live and inspect the terminal step.
3. Re-derive τ1/τ2 and confirm `knock_hard` exceeds τ1 with margin, and that
   `tap_gentle` (the τ1 baseline) isn't itself a large-force event.
4. Sanity: R2/R3 events should not cluster at the final step of episodes
   (terminal-step telemetry is real, but verify no systematic end-of-episode
   artifact in the first batch).
5. Confirm `smolvla_libero` success is non-zero via stock LeRobot eval first
   (handoff step 2) — unchanged recommendation.

## Bottom line

The round is a genuine, mostly-rigorous fix of the v0.1 instrumentation
artifacts; C1/C2/C3/C5/C6 and the follow-up refinements are verified. F1 must be
resolved (code or docs) before R4 numbers are reportable, and F3's on-box
terminal-transition check should gate the run. F2/F4–F8 are cheap hardening
items.
