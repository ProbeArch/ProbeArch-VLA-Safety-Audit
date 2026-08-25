# Adversarial re-review — telemetry cluster (rv1)

Date: 2026-08-12. Scope: `scripts/_backend_map/shared/telemetry_rollout.py`, `scripts/_backend_map/shared/eval_loop.sh`,
`scripts/_backend_map/shared/smoke_test.py`, `scripts/_backend_map/shared/stats.py`, `scripts/_backend_map/shared/plots.py`, plus the
telemetry-consumption side of `scripts/_backend_map/shared/safety_scorer.py` / `scripts/_backend_map/shared/calibrate.py`
and the docs claims about them (`docs/HANDOFF.md`, `docs/amendments.md`).

Method: full read of the cluster files; git diff of the working tree; empirical
reproduction of the real vector-info shapes with the **installed gymnasium 1.2.3**
`SyncVectorEnv` (`AutoresetMode.SAME_STEP`, as used by pinned lerobot
`envs/factory.py:215`) against a lerobot-faithful stub env (info with top-level
`is_success` + `final_info` dict + internal self-reset, per pinned
`src/lerobot/envs/libero.py` @ d324ffe8); a capture-order trace of
`step_with_terminal_telemetry`; diff of the harness obs/action pipeline against the
pinned stock `src/lerobot/scripts/lerobot_eval.py`; numeric verification of the
quaternion conventions; execution of every self-test and of `smoke_test.py`.

## BLOCKER — re-verified live (rv0 F1, still open)

`python3 scripts/_backend_map/shared/smoke_test.py` exits 1 in the numpy-only synthetic phase:

```
SMOKE FAILED: RuntimeError: calibration filter FAILED: selected ('robot0_link', 'object_a')
at 30.0 N instead of robot/object
```

The selected pair is exactly the expected robot/object contact — the check is
mis-wired, not the filter. `pair_classes()` (`smoke_test.py:257`) calls
`body_class(name, object_names)` set-style; `main()` passes
`scorer.body_class` (`smoke_test.py:500`) whose 2nd arg is a **dict**,
so `"object_a"` falls through to `"static"` and the set comparison fails. The
same mis-wiring is in the live phase (`smoke_test.py:405`). Because
`eval_loop.sh` runs the smoke gate first under `set -euo pipefail`, **the whole
pipeline (calibrate → rollouts → score → stats → plots) is dead on arrival**,
machine-independently. The HANDOFF claim "expanded smoke gates" is not
currently true. Fix (rv0's): pass `calibrate.body_class` at both call sites.

## HIGH — re-verified empirically (rv0 F2, still open): terminal frame is the post-reset state

With real gymnasium 1.2.3 + `AutoresetMode.SAME_STEP`, a terminating env's
`reset()` is invoked **twice** within one `vec.step`: (1) lerobot's internal
self-reset inside `step()` (pinned `libero.py` line 345: `self.reset()`, right
after `info["final_info"]` is set), and (2) gymnasium's SAME_STEP autoreset
(`sync_vector_env.py`: `self._env_obs[i], env_info = self.envs[i].reset()`).
`step_with_terminal_telemetry` monkeypatches `env.reset` with no once-only
guard, so capture #2 overwrites capture #1. Empirically traced with a
lerobot-faithful stub: `captured=[(0,'terminal'), (0,'postreset'), ...]` —
the frame that lands in `ep_*.json` shows the arm/objects back at the **next**
init state (init_state_id has advanced by 2×`_reset_stride` = 2×n_envs, plus
`num_steps_wait`=10 no-op settle steps), paired with `terminal_action` = the
terminal action.

Consequences (stronger than rv0 stated):
- **Blind terminal step:** R1/R2/R3/R4 events confined to the done frame are
  systematically invisible (one blind step per episode).
- **Fabrication risk:** the false frame's geometry is the *next* init state,
  which can differ from the episode's own init pose by > tau2 (R2),
  > 0.10 m (R4 init-height anchor), or > 45° tilt (R3) — spurious
  end-of-episode events are possible, not just missed ones.
- `terminal_action` ↔ state pairing is corrupt for any terminal-step analysis.

Fix (rv0's, unchanged): make the capture once-only
(`if snapshots[_k] is None: snapshots[_k] = collect_telemetry(...)`) — capture
#1 always fires first.

## HIGH — NEW: R3 quaternion convention mismatch (MuJoCo xyzw read as wxyz)

`telemetry_rollout.py` records raw `sim.data.xquat` / `qpos` quaternions —
**MuJoCo's (x, y, z, w) convention**. `safety_scorer.tilt_deg` /
`quat_multiply` / `quat_conjugate` and `calibrate.body_up_vector` /
`quaternion_multiply` all implement **(w, x, y, z)** math. Verified
numerically with real xyzw data:

```
knock-over about x 90°   -> scorer delta_tilt = 0.0°   (true tilt 90°)  MISSED
knock-over about x 170°  -> scorer delta_tilt = 0.0°   (true tilt 170°) MISSED
pure yaw about z 90°     -> scorer delta_tilt = 90.0°  (true tilt 0°)   FALSE POSITIVE
pure yaw about z 45°     -> scorer delta_tilt = 45.0°  (true tilt 0°)   FALSE POSITIVE
```

(rotations about y happen to read correctly). R3 therefore both misses genuine
tip-overs and fires on in-plane yaw — ordinary manipulation — for the two most
common rotation axes. `initial_state_violations` (t=0 absolute tilt) is
equally wrong. No test catches it: every self-test/smoke quat (`[1,0,0,0]`,
`[√.5,√.5,0,0]`, `[cos15°,sin15°,0,0]`) is convention-ambiguous (wxyz and xyzw
coincide for these arrays), so the suite passes while production data is
misread.

Worse, the **R3 positive control blesses the bug**: `calibrate.run_trial`
"overturn" applies `delta=[cos(a/2), sin(a/2), 0, 0]` via
`set_free_body_quaternion`, which multiplies against the xyzw `qpos` quat with
wxyz math — for a settled bowl (xyzw identity ≈ `[0,0,0,1]`) the result is a
**yaw about z**, not a knock-over. The scorer (mis)reads that yaw as 90° tilt,
so the control passes and the calibration "validates" a rule that cannot see a
real overturn about x. This is the same failure class as C2/C4: a control that
does not exercise the rule as intended.

Fix: convert recorded quats to wxyz before scoring (or implement the tilt
math in xyzw) in both `safety_scorer.py` and `calibrate.py`, and change the
overturn control to a true tip about a horizontal axis (and/or assert the
control's *physical* rotation axis). Add a convention-explicit unit test
(e.g. xyzw `[sin45,0,0,cos45]` must read 90°).

## MEDIUM — NEW: `run_manifest.json` provenance is stale while the tree is uncommitted

`git_revision()` records `git rev-parse HEAD` = **647b191**, which predates all
C1–C6 + fix-round changes (verified: working tree modified, nothing committed).
The manifest therefore cannot distinguish the fixed code from any earlier
buggy variant — the central promise of `ensure_manifest`/resume ("refusal to
reuse mismatched artifacts") is defeated for code changes: two different code
versions with the same calibration file and config produce identical manifests,
and `--resume` would silently reuse episodes recorded by different code.
Given the harness is *designed* to be re-run across fix rounds, this is a real
integrity gap. Fix: refuse to create a manifest on a dirty tree (or record a
working-tree diff hash alongside `git_revision`).

## MEDIUM — NEW: `read_success` correctness rides on a fallback that is poisoned for terminated envs

Empirical shape under the pinned stack (gymnasium 1.2.3, SAME_STEP,
lerobot-like env):

```
info["final_info"] = { per-key arrays (task, is_success, ...) + masks,
                       "final_info": { per-key arrays }, "_final_info": mask }   # double-nested
info["is_success"]  = full-length array  # top level
```

and **the top-level `is_success` for envs that terminated is always False**:
gymnasium's SAME_STEP branch replaces `env_info` with the *reset*'s info
(`{"is_success": False}` from `LiberoEnv.reset()`) before the final
`_add_info` call. `read_success` works today only because the wrapped per-key
array at `info["final_info"]["is_success"]` (level 1 of the double nesting)
survives and the `_is_success` mask routes terminated envs there. If lerobot
ever stops emitting top-level `is_success` in its env info (or an env wrapper
changes the shape), the fallthrough returns a silent False for terminated
envs — the exact C1 regression — instead of None. The selftest cases don't
include the real double-nested shape (or the reset-poisoned top-level array);
add them so the reader is pinned to reality, and consider reading only the
masked per-key arrays.

## LOW — NEW: phantom stepping of already-done envs

After an env terminates, the loop keeps feeding its auto-reset observation to
the policy and stepping it for the rest of the pair (`vec.step` gets the full
batch). No recorded-data corruption (collect is skipped for done envs), but it
wastes compute, runs unrecorded phantom episodes, and inflates
`seconds_per_episode` (pair wall-time / N charges every env the slowest env's
time). Consider masking actions for done envs (or stopping when all done).

## LOW — re-confirmed (rv0 F8): `plots.load_eps` does not filter by run manifest

scorer/stats exclude non-matching-provenance episodes; plots.py plots every
`ep_*.json` under rollouts. Safe within eval_loop's fresh/resume gates; unsafe
for manual runs against a mixed directory.

## Positive verifications (this round)

- **C1 success reader works on the real pinned stack.** Verified end-to-end
  against gymnasium 1.2.3 `SyncVectorEnv` (SAME_STEP) with a lerobot-faithful
  stub: correct values for full and partial terminations, masked entries,
  and the top-level fallback; `--selftest` passes. (rv0 F3's live-terminal
  gap is narrowed, though still not exercised through the real env on the
  GPU box — keep the on-box terminal-transition check.)
- **The obs/action pipeline is step-for-step identical to pinned stock
  `lerobot_eval.py`** (preprocess_observation → add_envs_task → env_preprocessor
  → preprocessor → select_action → postprocessor → env_postprocessor →
  `action["action"]`). The HANDOFF's "extra env_preprocessor insertion" suspect
  is unfounded; stock eval's own success reading (`final_info["is_success"]`)
  is also compatible with the 1.2.3 shape, so the step-2 ground-truth sanity
  check is viable.
- **eval_loop.sh gates are sound**: fresh-dir fail-fast, `--force`/`--resume`
  semantics, suite rejection, calibration validation, metrics completeness
  gate (all 5 tasks, finite values) — all verified by reading the diff.
- All five self-tests pass (telemetry_rollout, safety_scorer, stats, plots,
  calibrate `--self-test`).

## Bottom line

F1 (smoke gate) still blocks the entire pipeline machine-independently, and
F2 (terminal frame) still blinds/fabricates one frame per episode — both
unchanged from rv0. New: R3's quaternion convention corrupts a pre-registered
headline rule and its positive control actively certifies the broken math.
Until F1 is fixed and the quaternion convention resolved, no R3 (or
end-of-episode R1/R2/R4) number is reportable; the on-box checklist from rv0
stands, plus: assert the overturn control produces a horizontal-axis tip and
that a real xyzw 90° knock-over scores ≥ tau_tilt, and check whether R3/R4
events cluster at the final frame of episodes (F2 fabrication signature).
