# Static verification gate — vrfy_static (final gate, pre-validation)

Date: 2026-08-12. Machine: macOS (no runtime deps: no lerobot/mujoco/torch —
numpy-only). Scope: whole working tree on `fix/harness-v0.2` (uncommitted), all
scripts, docs, pins. Verdict: **PASS for the local gates; ONE unresolved blocker
class (F2 autoreset dispute) must be closed on the target machine before any
number is trusted.**

## Green — verified by execution or direct code reading

1. **F1 (smoke-gate blocker from `REVIEW_telemetry.md`) — FIXED, re-verified.**
   Both `body_class` call sites in `scripts/audit/shared/smoke_test.py` use the set-contract
   `calibrate.body_class` (imports at :329, live phase at :414, synthetic phase
   at :510–512; `pair_classes` docstring documents the contract). Execution:
   `python3 scripts/audit/shared/smoke_test.py` → `SMOKE PASSED`, exit 0 (numpy-only phase;
   live phase correctly skips with a message when mujoco/lerobot are absent).
2. **All five synthetic gates exit 0** (executed, exit codes captured):
   `smoke_test.py` 0, `telemetry_rollout.py --selftest` 0, `safety_scorer.py
   --selftest` 0, `stats.py --selftest` 0, `calibrate.py --self-test` 0.
   Note the flag asymmetry (`--self-test` vs `--selftest`); `eval_loop.sh`
   invokes each correctly (:133–141).
3. **`eval_loop.sh` gate structure** (`set -euo pipefail`): step 0 = the four
   synthetic selftests + smoke gate (any nonzero aborts before calibration);
   fresh-dir gate refuses `$AUDIT_DIR/rollouts` containing `ep_*.json` unless
   `--resume`/`--force` (:118–125); non-`libero_spatial` suites rejected (:70);
   `--force` also clears stale aggregates + figures (:112–115); post-loop gate
   verifies root↔per-task `run_id` match, per-task episode counts vs planned
   n_envs×n_pairs, and finite metrics (:195–209).
4. **C1 success extraction** — `read_success` handles recursed per-key
   `final_info` (+ `_is_success`/`_final_info` masks), list-of-dicts, legacy
   env-index dict, and top-level `is_success` fallback; never returns `None`
   early while a usable shape exists; synthetic coverage in
   `telemetry_rollout.py --selftest` (passes).
5. **C2/C3 calibration** — `r1_eligible` (:127) is the shared predicate with the
   scorer; scene settles post-reset; controls validated through the real scorer
   with hard asserts: `knock_hard`→{R1,R2}, `displacement`→{R2},
   `overturn`→{R3}, `off_table_fall`→{R4} + >0.10 m descent assert (:882–885,
   :762–766).
6. **C4 R4 anchor** — `support_plane_z()` 3-tier resolution (:189–212) with the
   own-init-height fallback; the F3 gap (rollout telemetry records no support
   geometry → production episodes use the fallback) is disclosed accurately in
   HANDOFF/amendments/REPORT, not overstated. R3 is delta-tilt vs the episode's
   initial quaternion with t=0 violations reported as
   `initial_state_violations`.
7. **C5** — `episodes_with_event_by_rule` counts episodes per rule (set-based
   per-episode dedup, :468–474); `events_by_rule` reported separately.
8. **C6/portability** — `AUDIT_DIR` default everywhere; `ship.ps1` deleted
   (staged); `ship.sh` (bash -n OK) guards: `--i-am-sure`, dirty tree,
   RETRACTED markers, branch name, foreign origin.
9. **Provenance** — per-run + per-task `run_manifest.json` (git_revision,
   policy sha256, calibration sha256, suite, resolution, max_steps, n_envs,
   n_pairs); `load_reusable_episode` validates provenance; atomic writes;
   fcntl-merged per-task metrics (no last-task-wins).
10. **stats/plots guards** — stats reads thresholds from `safety_summary.json`
    and hard-fails on empty rollouts (selftest asserts RuntimeError); plots
    uses staged atomic figure install with rollback; R4 figure is
    `object_fall.png` (support-plane decision).
11. **pins.md** — `gymnasium>=1.1.1,<2.0.0` recorded in the resolved matrix.
12. **Syntax** — all 6 `scripts/*.py` pass `py_compile`; `eval_loop.sh` and
    `ship.sh` pass `bash -n`.
13. **Docs** — REPORT.md retraction banner is accurate (R1/R2/R4 invalid, F1
    fixed, F3 open); amendments.md corrected wording (tau1 invalid-but-not-
    unreachable, R2 retracted, fresh AUDIT_DIR, off-support R4 control);
    `.gitignore` covers `.pi/ .pi-glla/ .codegraph/ .claude/` so the ship
    guard sees real changes only.

## BLOCKER — unresolved: F2 autoreset-mode dispute (cannot close statically)

- Four rv1 reviews (`rv1_scorer`, `rv1_telemetry`, `rv1_shell`,
  `rv1_calibrate`) cite the pinned lerobot `d324ffe8` `envs/factory.py:215`
  verbatim: `env_cls([...], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)`,
  with a mechanical stub trace showing the terminal snapshot is overwritten by
  the post-reset capture (arm/object back at init, ~0 contacts, paired with the
  terminal action) → one blind terminal step per episode for R1–R4.
- HANDOFF/PROTOCOL/amendments assert the LIBERO path uses the `SyncVectorEnv`
  default `NEXT_STEP` ("SAME_STEP is only in the non-LIBERO `gym.make` branch")
  with **no source citation**, and the pinned lerobot is **not installed on this
  machine** (only gymnasium 1.2.3, whose default is NEXT_STEP but which the
  pinned lerobot may override).
- The code at `telemetry_rollout.py:508` has **no once-only guard**
  (`snapshots[_k] = collect_telemetry(...)`, unconditional). Under SAME_STEP the
  F2 defect is live; under NEXT_STEP it is not. The docstring promises capture
  "immediately before its internal reset" — i.e., capture #1 — which only holds
  under NEXT_STEP.
- HANDOFF itself concedes the mechanism ("if the autoreset mode ever becomes
  SAME_STEP, the vector autoreset would overwrite the captured terminal frame").

Required before validation: verify the autoreset mode of the installed pinned
lerobot `d324ffe8` on the target machine at install time (one grep of
`envs/factory.py`), and/or apply the reviewer's one-line fix, which is safe
under **both** modes because capture #1 always fires first (the internal
`self.reset()` runs inside `LiberoEnv.step()`, before any vector autoreset):

```python
def capture_then_reset(*args, _env=env, _k=k, _reset=original_reset, **kwargs):
    if snapshots[_k] is None:
        snapshots[_k] = collect_telemetry(_env._env.sim, step + 1, table, action[_k])
    return _reset(*args, **kwargs)
```

## Open (documented, non-blocking for this gate)

- F3 (production R4 uses own-init-height fallback anchor) — disclosed, backlogged.
- F4 (manifest `git_revision` is v0.1-era HEAD; uncommitted fixes invisible) — backlogged.
- F5 (`read_success` None → silent False; no `success_source`) — synthetic shape
  tests are the tripwire.
- F6 (standalone scorer/stats admit unprovenanced episodes) — protected by
  eval_loop gates only.
- F7 (calibrate `prioritize_r1` truncation lacks R1-eligible preservation;
  rollouts side has it) — low risk.
- `results/` still holds v0.1 artifacts; the decided move to
  `results/v0.1-retracted/` + retraction marker is pending (REPORT.md already
  links to the future path — dangling until the move executes).
- Commit/push deliberately deferred until the validation run (ship.sh enforces
  this via the RETRACTED/dirty-tree guards).

## Bottom line

Every locally runnable gate is green and every code claim in the handoff that
could be checked statically checked out — including the previously-blocking F1.
The pipeline cannot be certified end-to-end from this machine: the F2
autoreset-mode question is the one load-bearing unknown (it decides whether
terminal-step safety events are visible at all), and the on-target validation
run (fresh AUDIT_DIR, stock-LeRobot ground-truth sanity check, pinned stack)
remains mandatory before any number is trusted or cited.
