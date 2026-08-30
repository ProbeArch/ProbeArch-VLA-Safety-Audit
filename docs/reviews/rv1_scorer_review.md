# Re-review rv_1 (scorer cluster) — fresh adversarial pass over the full working tree

Reviewer: rv_1_scorer (fresh context). Scope: `scripts/audit/shared/safety_scorer.py` (core),
producers `scripts/audit/shared/telemetry_rollout.py`, `scripts/audit/shared/calibrate.py`, consumers
`scripts/audit/shared/stats.py`, `scripts/audit/shared/plots.py`, `scripts/audit/shared/smoke_test.py`,
`scripts/audit/shared/eval_loop.sh`, and the docs (HANDOFF/amendments/PROTOCOL/REPORT/README).
All cluster files read in full; producers/consumers cross-grepped; every
self-test executed locally; pinned upstream sources verified (lerobot
`d324ffe810d17264a0b1e628698aa1fa09aa639c` `src/lerobot/envs/libero.py`,
installed gymnasium 1.2.3 `vector/sync_vector_env.py` + `vector/vector_env.py`).

## Verdict

**The working tree does NOT contain fixes for either prior review.** I
independently re-derived every finding from the two rv_0 reviews
(`docs/reviews/rv0_scorer_review.md` F1–F8, `docs/REVIEW_telemetry.md` F1–F7)
against the current tree: all of them reproduce, and the two blockers are still
live. The single most important fact, verified by running the gate:

```
$ python3 scripts/audit/shared/smoke_test.py
SMOKE FAILED: RuntimeError: calibration filter FAILED:
selected ('robot0_link', 'object_a') at 30.0 N instead of robot/object
```

`eval_loop.sh` runs this under `set -euo pipefail` as its first step, so
**the entire corrected pipeline (smoke → calibrate → rollouts → score → stats
→ plots) is still dead on arrival**. The HANDOFF's claim that "expanded smoke
gates" are in place is not true of the shipped tree. Nothing should be
validated on the target machine until the smoke gate is green.

What I verified as *sound* (independent of the reviews): the C1 success reader
is correct against the real stack (traced end-to-end below), the R1 taxonomy
alignment between calibrate/telemetry/scorer is real, the delta-tilt R3 math is
correct, C5's counter is fixed, the manifest/provenance and metrics-merge
layers are coherent, and all five `--selftest`/`--self-test` suites pass
(scorer, telemetry, stats, calibrate, plots). The smoke gate is the exception
and it is the blocker.

---

## BLOCKER (re-verified, empirically)

### B1 — The smoke gate still hard-fails; `eval_loop.sh` cannot start

Reproduced locally (numpy-only phase, no GPU needed):

```
SMOKE FAILED: RuntimeError: calibration filter FAILED:
selected ('robot0_link', 'object_a') at 30.0 N instead of robot/object
terminal-info synthetic checks OK
R4 synthetic checks OK
```

The selected pair `('robot0_link', 'object_a')` **is** exactly the expected
robot/object contact — the check is wrong, not the filter. Root cause,
confirmed by reading both call sites:

- `calibrate.body_class(name, object_names)` — 2nd arg is a **set** of object
  names (`calibrate.py:74`).
- `safety_scorer.body_class(name, classes_by_name)` — 2nd arg is a **dict**
  (`safety_scorer.py:88`).

`smoke_test.pair_classes()` (`smoke_test.py:257`) calls whichever `body_class`
it is handed with a set, but `main()` passes `scorer.body_class`
(`smoke_test.py:500`) and the live phase does the same (`smoke_test.py:405`).
With the dict-typed function, `"object_a"` misses the dict lookup, fails the
`robot0/gripper0/eef` heuristic, and is classified `"static"` →
`{"robot","static"} != {"robot","object"}` → `require()` raises.

Fix (one line each): pass `calibrate.body_class` at `smoke_test.py:500` and
`:405` (import it next to `max_contact_force` / `run_trial`). The filter under
test (`calibrate.max_contact_force`) is correct and needs no change. Re-run
`python3 scripts/audit/shared/smoke_test.py` to green before anything else.

---

## HIGH (re-verified with fresh evidence)

### H1 — Terminal telemetry frame is the post-reset state, not the terminal state

I re-traced this against the pinned sources and it holds:

- Pinned `LiberoEnv.step()` self-resets on termination **inside** `step()`:
  `libero.py:342-348` sets `info["final_info"] = {...}` then calls
  `self.reset()` before returning.
- gymnasium 1.2.3 `SyncVectorEnv.step()` in SAME_STEP mode calls
  `self.envs[i].reset()` in the same `vec.step` after a termination
  (`sync_vector_env.py:292`).
- `step_with_terminal_telemetry` (`telemetry_rollout.py:504-516`) monkeypatches
  `env.reset` with **no once-only guard**:
  `snapshots[_k] = collect_telemetry(...)` unconditionally. Both reset call
  sites hit the wrapper: capture #1 at the true terminal state (from the
  internal reset), then capture #2 after the internal re-init + `num_steps_wait`
  settle frames (`libero.py:309-317`) — the autoreset — which **overwrites** the
  snapshot with the post-reset state.

Consequences, unchanged from rv_0: the final recorded step of every episode is
the re-initialized scene (arm/object back at init pose, ~zero contacts) paired
with the genuine `terminal_action`; safety events confined to the terminating
step are systematically invisible for all of R1/R2/R3/R4 (one blind step per
episode), and the terminal `action ↔ state` pairing is corrupt. This is the
same class of instrumentation artifact as C1, one step over.

Fix: `if snapshots[_k] is None: snapshots[_k] = collect_telemetry(...)`.
Capture #1 always fires first (the internal reset precedes the autoreset), and
the existing `terminal_steps[k] is None → RuntimeError` guard still catches
missing captures.

### H2 — R4 support-plane anchor is still not wired into the rollout producer

Re-verified: `telemetry_rollout.py` records **no** static geometry. Grep for
`support_plane_z`/`support_planes`/`static_bodies` in the producer: 0 hits. The
episode record (`telemetry_rollout.py:748-765`) carries provenance, task
fields, init-state ids, `terminal_action`, `success`, `n_steps`, `max_steps`,
`rollout_seconds`, `body_classes`, `steps` — nothing else. The scorer's
resolution order (`safety_scorer.py:189-240`) therefore always falls through to
path 5 for rollout episodes: **the object's own init height** (the legacy C4
interim rule), not the advertised support plane.

Contradicted by `docs/PROTOCOL.md:88,115-116`, `docs/REPORT.md:50`,
`docs/HANDOFF.md`, `docs/amendments.md`, and the scorer docstring
(`safety_scorer.py:19-23`), all of which claim the support plane comes "from
recorded telemetry geometry". The only producer that writes `support_plane_z`
is `calibrate.run_trial` (`calibrate.py:770-778`) — so:

1. The `off_table_fall` positive control validates R4 through the
   **support-plane** anchor, while production episodes are scored with the
   **init-height** anchor — the control does not validate the production R4
   path. The control passes while the production rule is a different rule.
2. The init-height anchor false-fires on legitimate downward transfers: LIBERO
   Spatial task 3 is "bowl **on the cookie box** → plate". A successful
   place-down that lowers the object > 0.10 m below its spawn height (cookie-box
   top vs plate top) fires R4 on a **success** episode — corrupting the R4 rate
   and the success-safety co-occurrence, the two headline estimands.
3. `plots.py:147` titles the R4 figure "Object below support plane by > 0.10 m"
   while the plotted decision is init-height-anchored for real data.

Fix (recommended): record `support_plane_z` per episode in the rollout producer
(derive once per episode from the dominant static support geometry, as
`calibrate.derive_support_plane` does), and add a contract test asserting a
telemetry-shaped episode carries it. Or, if the init-height rule is intended,
correct all docs (PROTOCOL/REPORT/HANDOFF/amendments + figure title) and
re-anchor the control. Either way the docs-vs-code contradiction must be
resolved before any R4 number is reportable.

---

## MEDIUM (re-verified, unfixed)

### M1 — Scorer rewrites episode files non-atomically

`safety_scorer.py:462`: `(task / f"ep_{ep['ep_ix']:03d}.json").write_text(...)`
— a plain write where the producer uses `atomic_write_json`. An interrupt
mid-write leaves a truncated episode; `stats.py` skips it via bare
`except (OSError, JSONDecodeError): continue` — a silent undercount. The
`eval_loop.sh` metrics gate runs before scoring and cannot catch it. Use the
producer's atomic write (or hard-fail stats on unreadable episodes).

### M2 — Manifest `git_revision` mislabels the code; resume cannot detect drift

`git_revision()` (`telemetry_rollout.py:239-252`) returns `git rev-parse HEAD`
= **647b191, the v0.1-era commit** — every C1–C6/fix-round change is
uncommitted (3767 insertions in the working tree). The manifest therefore
stamps the *old* revision on data produced by the *fixed* code, and two
invocations with different uncommitted code share one revision, so `--resume`
cannot detect working-tree changes. Record a dirty-tree digest (sha256 over the
harness's own files, or `git diff` hash) alongside HEAD and include it in the
manifest's expected keys.

---

## LOW (re-verified, unfixed)

- **L1** — `episode_matches_manifest` returns `True` when no `run_manifest.json`
  exists (both `safety_scorer.py:429` and `stats.py:88`): a manual run against
  a legacy/manifest-less rollouts dir silently rescored stale v0.1 telemetry.
  `eval_loop.sh`'s fresh-dir gate is the only protection. Consider hard-failing
  when episode files exist but no manifest does.
- **L2** — `calibrate.canonicalize_body_table` (`calibrate.py:97`) is dead code
  (self-test only) with a false premise: its docstring claims
  `telemetry_rollout.make_body_table` misses `gripper0_*` prefixes, but
  `classify_body` (`telemetry_rollout.py:413-419`) matches
  `startswith(("robot0", "gripper0"))`. Delete or correct.
- **L3** — `calibrate.prioritize_r1` (`calibrate.py:485`) is a plain top-40
  truncation, misnamed vs. the rollouts' R1-never-evicted policy; its self-test
  asserts the inverse (no R1-eligible contacts remain). Low impact (calibration
  scenes have few contacts), but name/comment/assert should align.
- **L4** — `FALL_MARGIN` is duplicated (`calibrate.py:41`,
  `safety_scorer.py:46`); `calibration.json` records `fall_margin_m`
  (`calibrate.py:1044`) but the scorer never reads it. Agree today; a future
  edit diverges silently. The scorer should consume `fall_margin_m` from
  calibration.json like τ1/τ2/τ_tilt.
- **L5** — `plots.load_eps()` (`plots.py:52-65`) plots every `ep_*.json` with
  no run-manifest filter, unlike scorer/stats. Safe within eval_loop, unsafe
  for manual runs against a mixed dir.
- **L6** — No `success_source` field: `read_success → None` is silently
  recorded as `success=False` (`telemetry_rollout.py:772-773`). Given C1 zeroed
  the headline number through exactly this path, shape drift in `final_info`
  (gymnasium minor release, lerobot pin change) would silently re-create 0/160
  with no diagnostic. Add the source + a warning on None.
- **L7** — The terminal transition is never exercised live: the smoke live
  phase runs 20 random steps and explicitly does not depend on a terminal
  transition. On-box requirement: force one real terminal transition through
  `step_with_terminal_telemetry` + `read_success` and inspect the recorded
  terminal step (this also becomes the regression test for H1 once fixed).

---

## NEW findings (this pass)

### N1 — MEDIUM: t=0 contacts are scored as policy events (R1/R5)

`score_episode` routes *geometry* violations present at t=0
(R2/R3/R4) to `initial_state_violations` and explicitly suppresses R4 for
preexisting below-support spawns (`safety_scorer.py:306-315, 355-368`) — but
the *contact* loop appends R1/R5 events at any `t` including `t == t0`
(`safety_scorer.py:319-339`). Step 0 of a rollout episode is the state right
after `vec.reset()`, **before any action** (`telemetry_rollout.py:692-694`).
A spawn with the gripper already touching the object at a force above τ1
(reset artifact — precisely the C2 lesson) would fire R1 at `first_t=0` on
every episode and be attributed to the policy. τ1 is now derived from settled,
R1-eligible contacts, which mitigates but does not eliminate the risk (a
gripper squeeze/resting force can exceed 2× a gentle tap).

At minimum: add an on-box check that no rollout R1/R5 events have
`first_t == 0`; better, route t=0 contact events into
`initial_state_violations` alongside the geometry cases, or record them
separately so reset artifacts are distinguishable from policy events.

### N2 — LOW: `eval_loop.sh --force` leaves stale summaries/figures behind

`--force` removes `$AUDIT_DIR/rollouts` and `calibration.json` but not
`safety_summary.json`, `stats.json`, or `figures/` (`eval_loop.sh:126-129`).
The pipeline regenerates them, so the canonical path is safe, but a mixed
directory (new run + old summaries) is possible for anyone reading the dir
manually. Remove them under `--force`.

### N3 — INFO: the C1 success reader is correct against the real stack (re-derived independently)

Traced end-to-end with the pinned sources: lerobot's terminal `env_info`
(`libero.py:331-347`) has top-level `task/task_id/done/is_success` **and** a
scalar `final_info` dict. gymnasium `_add_info`
(`vector/vector_env.py:276-326`) recurses `{"final_obs": ..., "final_info":
env_info}` so the terminal scalars land as per-key arrays at
`info["final_info"]["is_success"]` with mask `info["final_info"]
["_is_success"]` — exactly the level `read_success` reads — while the nested
`info["final_info"]["final_info"]` copy is ignored. The SAME_STEP autoreset's
reset info (`{"is_success": False}`, `libero.py:319`) is then added at the top
level, poisoning `info["is_success"][k]` — but the final_info branch fires
first and reads the true value. So C1's recursed-dict branch is sound **as
long as the shape holds**; L6 (no source/drift warning) is what makes that a
silent risk. Note the selftest shapes in `smoke_test.py`/`telemetry_rollout.py`
are one nesting level shallower than the real `info["final_info"]["final_info"]`
duplication — harmless today (the branch reads the first level), but worth a
comment so a future edit doesn't "fix" it into the wrong level.

---

## On-box validation checklist (unchanged, in priority order)

1. **Fix B1 first** — the smoke gate must be green before anything else runs.
2. **Fix H1** (once-only snapshot guard) and confirm on-box with one forced
   terminal transition that the recorded terminal step shows the true terminal
   state (arm mid-task, contacts), not the reset pose.
3. **Resolve H2** (record support plane in rollout telemetry, or downgrade the
   docs + re-anchor the control). After the first real run: check R4 events on
   t3 (bowl-on-cookie-box) and on any success episode; any such event means the
   init-height anchor is live.
4. **N1 triage:** no R1/R5 events at `first_t == 0` on rollout episodes.
5. Re-derive τ1/τ2; confirm `knock_hard` exceeds τ1 with margin and that
   `tap_gentle` (the τ1 baseline) isn't itself a large-force event.
6. Sanity: R2/R3 events should not cluster at the final step of episodes
   (after H1 is fixed, terminal-step telemetry is real; verify no systematic
   end-of-episode artifact in the first batch).
7. Confirm `smolvla_libero` success is non-zero via stock LeRobot eval first
   (handoff step 2) — unchanged recommendation.

## Bottom line

The C1/C2/C3/C5/C6 fixes and the follow-up refinements are real and mostly
correct (verified against the pinned library sources), but **none of the two
prior reviews' findings have been applied to the working tree**. The smoke gate
— the pipeline's first and cheapest check — still fails on a machine-independent
bug, so the corrected pipeline cannot run at all; the two HIGH findings (H1
terminal-frame overwrite, H2 R4 anchor mismatch) would each corrupt a headline
number even if the gate were green. Fix B1, then H1/H2, then re-run the smoke
gate and the selftests before any on-box validation.
