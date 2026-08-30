# Adversarial re-review — telemetry cluster (rv_0_telemetry)

Date: 2026-08-12. Scope: `scripts/audit/shared/telemetry_rollout.py`, `scripts/audit/shared/safety_scorer.py`,
`scripts/audit/shared/calibrate.py`, `scripts/audit/shared/smoke_test.py`, `scripts/audit/shared/stats.py`, `scripts/audit/shared/plots.py`,
`scripts/audit/shared/eval_loop.sh`, plus `docs/HANDOFF.md` / `docs/amendments.md` claims about them.

Verification performed: full read of all cluster files; git diff of the working tree;
comparison against the **installed gymnasium 1.2.3** `SyncVectorEnv`/`_add_info` source
and the **pinned lerobot d324ffe8** `envs/libero.py` + `scripts/lerobot_eval.py`;
empirical reproduction of the real info shapes through `read_success`; execution of all
six selftests; a stub-level execution trace of the terminal-telemetry capture order.

## BLOCKER

### F1 — The smoke gate hard-fails: `eval_loop.sh` cannot run at all

`python3 scripts/audit/shared/smoke_test.py` exits 1 on the **synthetic (numpy-only) phase**:

```
SMOKE FAILED: RuntimeError: calibration filter FAILED: selected ('robot0_link', 'object_a')
at 30.0 N instead of robot/object
terminal-info synthetic checks OK
R4 synthetic checks OK
```

The selected pair is in fact exactly the expected `robot/object` contact — the check is
wrong, not the filter. Root cause: a **signature mismatch between two `body_class`
functions**:

- `calibrate.body_class(name, object_names)` — 2nd arg is a **set** of object names
  (`scripts/audit/shared/calibrate.py:74`).
- `safety_scorer.body_class(name, classes_by_name)` — 2nd arg is a **dict** name→class
  (`scripts/audit/shared/safety_scorer.py:88`).

`smoke_test.pair_classes()` (`smoke_test.py:257`) calls whichever `body_class` it is given
set-style, but `main()` passes `scorer.body_class` (`smoke_test.py:500`), so `"object_a"`
falls through the dict check, fails the name heuristic, and is classified `"static"` →
`{"robot","static"} != {"robot","object"}` → `require()` raises. The same mis-wiring is
in the live phase (`smoke_test.py:405`), so it would also fail on the GPU box even after
the synthetic phase is fixed.

Consequence: `eval_loop.sh` runs `smoke_test.py` first under `set -euo pipefail` and
aborts — **the entire corrected pipeline (calibrate → rollouts → score → stats → plots)
is dead on arrival**. This is machine-independent (numpy only), so it should have been
caught before handoff; the HANDOFF's "expanded smoke gates" claim is not currently true.

Fix: pass `calibrate.body_class` at `smoke_test.py:500` and `:405` (import it next to
`max_contact_force`), or reimplement `pair_classes` from the authoritative table classes.
The filter under test (`calibrate.max_contact_force`) is correct and needs no change.

## HIGH

### F2 — Terminal telemetry frame is the post-reset state, not the terminal state

The pinned lerobot `LiberoEnv.step()` **self-resets internally on termination**
(`self.reset()` in `src/lerobot/envs/libero.py`, right after setting `info["final_info"]`),
and lerobot's `make_env` uses `AutoresetMode.SAME_STEP` (confirmed in the pinned
`envs/factory.py:215`), so gymnasium also calls `env.reset()` in the same `vec.step`.

`step_with_terminal_telemetry` (`telemetry_rollout.py`) monkeypatches `env.reset`, which
both call sites hit:

1. internal self-reset → **capture #1 at the true terminal state** (correct),
2. gymnasium SAME_STEP autoreset → **capture #2 after the internal reset + its 10
   no-op settle steps** → **overwrites** `snapshots[k]` (no once-only guard).

Empirically confirmed with a stub trace: the frame that lands in `ep_*.json` shows the
arm back at init pose, the object back at its init pose, ~zero contacts — paired with
`action_prev`/`terminal_action` = the terminal action. Effects:

- **Safety events confined to the terminating step are systematically invisible** — one
  blind step per episode for all of R1/R2/R3/R4 (e.g., a knock whose only
  above-τ1 contact frame is the done frame; a fall initiated by the terminal action).
  This is precisely the C1-class instrumentation artifact the fix round was meant to
  eliminate, just one step over.
- `terminal_action` ↔ state pairing is corrupt, so any "what was the policy doing at the
  event" analysis using the last frame is misleading.
- The docstring "Capture each terminating LIBERO sim immediately before its internal
  reset" describes capture #1; the implementation keeps capture #2.

Fix: make the capture once-only (`if snapshots[_k] is None: ...`) — capture #1 always
fires first because the internal reset happens inside `env.step()`, before gymnasium's
autoreset.

## MEDIUM-HIGH

### F3 — R4's "support-plane anchor" is not what runs on rollout episodes

`collect_telemetry()` records only object bodies + eef; it never records static geometry
(`support_plane_z`, `support_planes`, `static_bodies` are absent from every rollout
episode — only `body_classes` is written). `support_plane_z()` in the scorer therefore
resolves to `None` for rollout episodes and R4 falls back to the **object's own init
height** — the legacy C4 anchor. The support-plane anchor runs only on **calibration
control episodes** (which `run_trial` annotates with `support_plane_z`).

So the HANDOFF/amendments/PROTOCOL claim ("R4 anchored to the scene support plane from
recorded telemetry geometry; init height only a conservative fallback") overstates the
production path; worse, the `off_table_fall` positive control validates R4 with the
support-plane anchor — a **different anchor than the one production episodes use** — so
the control does not validate the production R4 path. `plots.py`'s own comment admits
this ("Rollout telemetry does not contain static geometry").

Impact for LIBERO Spatial is bounded (full off-table falls drop ~0.8 m ≫ 0.10 m, so both
anchors fire), but the margin is now anchored to the object top, which can false-positive
on a tall object toppling while remaining supported (center drops > 0.10 m below its own
init height). Fix: record the support plane (per-task, from static contact geometry, as
calibrate does) in rollout telemetry, or correct the docs and re-anchor the control.

## MEDIUM

### F4 — Manifest `git_revision` is the pre-fix HEAD; uncommitted code is invisible to provenance

`git_revision()` returns `git rev-parse HEAD` = **647b191 — the v0.1-era commit**, while
every C1–C6 + audit-fix change is uncommitted. Two consequences:

1. The manifest's "harness git revision" mislabels the code that actually produced the
   data (a run using all the fixes is recorded as the old revision).
2. `--resume` cannot detect working-tree changes: two invocations made with **different
   uncommitted code** share the same recorded revision, so the manifest check passes and
   new pairs are appended under the same `run_id` — a mixed-code dataset under one
   provenance stamp. This undermines the "refusal to reuse mismatched or unprovenanced
   artifacts" guarantee that the round advertises.

Fix: record a dirty-tree digest alongside HEAD (e.g., sha256 over the sorted content of
the harness's own files, or `git diff` hash), and include it in the manifest expected
keys.

## LOW–MEDIUM

### F5 — The top-level success fallback is poisoned in this stack; shape drift would silently re-create C1

Verified against the real gymnasium 1.2.3 `_add_info` + pinned lerobot info shapes:
in SAME_STEP mode the autoreset's reset-info (`LiberoEnv.reset` returns
`{"is_success": False}`) **overwrites the top-level `info["is_success"][k]` slot of a
terminated env with False**. The recursed `final_info` branch reads the correct value
(empirically confirmed: `read_success → True` for the terminated env, `False` for live
envs, both single- and multi-termination steps — the C1 fix itself is sound), but if the
`final_info` shape ever drifts (gymnasium 1.x minor release, lerobot pin change), the
fallback silently returns False, and the harness records `success=False` with no
diagnostic. There is no `success_source` field (final_info vs fallback vs default) and
`read_success → None` is silently treated as False. For an audit whose headline number
was zeroed by exactly this path, add the source field + a warning on None.

## LOW

### F6 — Scorer/stats admit unprovenanced episodes when a task dir lacks a run manifest

`episode_matches_manifest` returns `True` when `run_id is None` (no
`run_manifest.json`), so stale v0.1-style telemetry (no provenance, no manifest) would
be rescored with v0.2 thresholds if `safety_scorer.py`/`stats.py` are pointed at it
standalone. `eval_loop.sh`'s fresh-dir gate protects the canonical path; consider
hard-failing when the manifest is absent instead.

### F7 — `calibrate.prioritize_r1` does not preserve R1-eligible contacts (asymmetric with rollouts)

Rollout telemetry guarantees R1-eligible contacts survive the top-40 truncation
(`MAX_R1_CONTACTS=512`); calibration control steps truncate to `CONTACT_LIMIT=40` by
force rank with no class preservation. Low practical risk (calibration scenes have few
contacts), but the calibrate self-test's R1-eviction assertion is vacuous (its synthetic
contacts are all classified `static`), and the documented invariant is not mirrored on
the calibration side.

## Verified sound (no change needed)

- `read_success` handles the **real** gymnasium 1.2.3 recursed `final_info` shape
  correctly (reproduced against installed source + pinned lerobot info shapes, including
  multi-env termination in one step); legacy/list/top-level shapes behave as documented.
- τ1/τ2 calibration: settle-before-measure, R1-eligible-only force baseline, and
  scorer-validated controls (incl. `off_table_fall` → R4 through the real scorer) match
  the amendments; `max_contact_force` uses table classes directly and is correct.
- No force-measurement mismatch between calibration (efc_force norm) and rollout
  telemetry (`mj_contactForce` translational norm) — the constraint basis is orthogonal,
  so the norms agree.
- Observation/action pipeline matches pinned `lerobot_eval.py` exactly
  (preprocess_observation → add_envs_task → env_preprocessor → preprocessor →
  select_action → postprocessor → env_postprocessor), including the env processors.
- Fresh-dir gate, per-task process isolation + manifest validation, atomic writes,
  fcntl-merged per-task metrics, and the aggregate-metrics verification in `eval_loop.sh`
  are coherent; C5 (episodes-with-event per rule) is fixed; stats/plots guards (empty
  telemetry, unscored episodes, atomic figure install) are in place.
- All five other selftests (`telemetry_rollout --selftest`, `safety_scorer --selftest`,
  `stats --selftest`, `calibrate --self-test`) pass.

## Priority order for the next fix round

1. F1 (blocker): fix the two `body_class` call sites in `smoke_test.py`; re-run
   `smoke_test.py` to green before anything else.
2. F2: once-only guard in `step_with_terminal_telemetry`.
3. F3: record the support plane in rollout telemetry (or correct docs + re-anchor
   control).
4. F4: dirty-tree digest in the run manifest.
5. F5/F6/F7: diagnostics and hardening.
