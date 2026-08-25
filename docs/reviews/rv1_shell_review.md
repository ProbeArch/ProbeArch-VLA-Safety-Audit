# Re-review rv_1_shell — adversarial review of the shell lane (eval_loop.sh / ship.sh) + pipeline contract

Reviewer: rv_1_shell (fresh context, no prior assumptions). Scope: `scripts/_backend_map/shared/eval_loop.sh`,
`scripts/ship.sh` (core), producers/consumers `scripts/_backend_map/shared/telemetry_rollout.py`,
`scripts/_backend_map/shared/calibrate.py`, `scripts/_backend_map/shared/safety_scorer.py`, `scripts/_backend_map/shared/smoke_test.py`,
`scripts/_backend_map/shared/stats.py`, `scripts/_backend_map/shared/plots.py`, `.gitignore`, `pins.md`, and docs
(HANDOFF / amendments / PROTOCOL / REPORT / README). All files read in full;
producers/consumers cross-grepped; every selftest executed on this machine;
`ensure_manifest` root-manifest behavior reproduced standalone; the F2 terminal-
telemetry dispute settled against the **installed gymnasium 1.2.3** `SyncVectorEnv.step`
source and the **pinned lerobot d324ffe8** `envs/factory.py`. Prior reviews
(`docs/REVIEW_telemetry.md`, `docs/reviews/rv0_scorer_review.md`,
`docs/reviews/rv1_plots_review.md`) read and their findings re-checked.

## Verdict

The shell lane is coherent in design but **cannot run end-to-end on the shipped tree**:
two independent blockers — B1 (smoke gate, previously reported twice, still unfixed)
and **B2 (NEW: root-manifest `task_ids` exact-match kills the per-task process loop at
task 1)** — plus one data-integrity defect that a prior review wrongly certified as
correct (H1, terminal frame is post-reset, settled with installed-source evidence).
H2 (support-plane R4 dead in production) remains open and the docs still overclaim it.
Nothing here contradicts the retraction of v0.1; the round is a genuine fix effort, but
the validation run must not start until B1/B2 are fixed and H1 is decided.

## BLOCKERS

### B1 — Smoke gate still hard-fails; `eval_loop.sh` step 0 aborts (3rd report, unfixed)

Reproduced on this machine (numpy-only phase):

```
$ python3 scripts/_backend_map/shared/smoke_test.py ; echo $?
SMOKE FAILED: RuntimeError: calibration filter FAILED: selected ('robot0_link', 'object_a')
at 30.0 N instead of robot/object
terminal-info synthetic checks OK
R4 synthetic checks OK
1
```

The selected pair **is** the expected robot/object contact — the check is wrong, not
the filter. Root cause (unchanged since `REVIEW_telemetry.md` F1 and `rv1_plots` B1):
`pair_classes` (`smoke_test.py:257`) passes `object_names` as a **set** to
`safety_scorer.body_class(name, classes_by_name)` (`safety_scorer.py:87–98`), whose
second parameter is a **dict**; a set fails the `isinstance(..., dict)` branch, so
`"object_a"` falls to the name heuristics → `"static"` → `{"robot","static"} !=
{"robot","object"}`. Two call sites are mis-wired: `smoke_test.py:500` (synthetic) and
`:405` (live phase — would also fail on the GPU box). `calibrate.body_class` takes the
set and is correct. Because `eval_loop.sh` runs `smoke_test.py` first under
`set -euo pipefail`, the entire pipeline is dead on arrival. The HANDOFF's "expanded
smoke gates … must exit 0" claim is not true of the shipped tree. **One-line fix:
pass `calibrate.body_class` at both call sites (or adapt `pair_classes` to the dict
signature); re-run to green.** This was reported twice before; it should gate the next
handoff.

### B2 — NEW: root `run_manifest.json` `task_ids` exact-match breaks the per-task process loop; the fleet can never run past task 0

`eval_loop.sh` runs `telemetry_rollout.py` **once per task** (A4 process isolation:
`for task_id in 0 1 2 3 4; do python3 scripts/_backend_map/shared/telemetry_rollout.py --task_ids "$task_id" ...`).
But `telemetry_rollout.py` builds `root_expected["task_ids"] = sorted(args.task_ids)`
(→ `[0]`, then `[1]`, …) and `ensure_manifest()` validates the shared root
`$AUDIT_DIR/rollouts/run_manifest.json` with **exact equality on every key**
("run manifest mismatch for 'task_ids'").

Reproduced standalone with the exact manifest logic:

```
task 0: OK (manifest task_ids=[0])
task 1: FAILED -> run manifest mismatch for 'task_ids' in .../run_manifest.json
```

Consequences:
- **Fresh fleet:** task 0 creates the root manifest with `task_ids=[0]`; task 1 hard-
  fails. Only 32 of 160 episodes could ever be rolled out.
- **`--resume`:** identical mismatch for tasks 1–4; resume is equally dead.
- **README's manual per-task steps** (documented as "equivalent to what the loop
  does") carry the same defect.
- The handoff/PROTOCOL claim that the manifest gate "works" is false for the shipped
  orchestration. No prior review caught this; it is in this cluster's direct line of
  sight.

Fix options: (a) drop `task_ids` from the **root** manifest (each task's own
`<task>/run_manifest.json` and per-episode `provenance` already pin the task id), or
(b) root-manifest compare as subset/superset instead of equality, or (c) pass all five
`--task_ids` to every invocation — **not** acceptable: that reintroduces the A4
multi-task env-build segfault. (a) is minimal and preserves provenance.

## HIGH

### H1 — Terminal-step telemetry frame is the post-reset state, not the terminal state (settled; a prior review certified this wrongly)

`REVIEW_telemetry.md` F2 (HIGH) claimed the recorded terminal frame is the post-reset
state because gymnasium's autoreset re-invokes the monkeypatched `env.reset` and
overwrites the snapshot. `rv0_scorer_review.md` claimed the opposite ("hook point —
correct"; "the later gymnasium NEXT_STEP autoreset is unwrapped"). I settled it with
primary sources:

1. Installed gymnasium **1.2.3** `SyncVectorEnv.step()` (matches pin `>=1.1.1,<2.0.0`),
   SAME_STEP branch:
   ```python
   ... = self.envs[i].step(action)
   if self._terminations[i] or self._truncations[i]:
       infos = self._add_info(infos, {"final_obs": ..., "final_info": env_info}, i)
       self._env_obs[i], env_info = self.envs[i].reset()   # autoreset, SAME call
   ```
2. Pinned lerobot `d324ffe8` `envs/factory.py:215`:
   `vec = env_cls([...], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)`.

So on a terminating step the patched `capture_then_reset` fires **twice**: (1) lerobot
`LiberoEnv.step()`'s internal `self.reset()` at the true terminal state, then (2)
gymnasium's SAME_STEP autoreset right after `step` returns. `snapshots[k]` ends up =
**capture #2 = the post-internal-reset state** (object/arm back at init, contacts ~0),
recorded as the terminal step with `t=step+1` and paired with the terminal action.

Effects: one **blind step per episode** for R1/R2/R3/R4 (a knock/fall whose only
above-threshold or below-support frame is the done frame is invisible), and
`terminal_action`↔state pairing is corrupt. This is the same C1-class artifact the
round was meant to eliminate, one step over. **Fix (one line): once-only guard**
(`if snapshots[_k] is None: snapshots[_k] = collect_telemetry(...)`) — capture #1
always fires first because it happens inside `env.step()`. The on-box terminal-
transition check (handoff step 4/F3) must assert the recorded terminal frame shows
terminal contacts, not reset pose.

### H2 — R4 "support-plane anchor" is dead code in the production path; the positive control validates a different anchor than production (3rd confirmation, still open)

`telemetry_rollout.py` records **no** `support_plane_z` / `support_planes` /
`static_bodies` anywhere (grep: 0 hits; episode schema = provenance + task fields +
init-state ids + terminal_action + success + n_steps + steps + body_classes).
`safety_scorer.support_plane_z()` therefore always resolves to path 5 (object's own
init height) for rollout episodes; the support-plane branches run only on calibration
control episodes (which `run_trial` annotates) and synthetic tests. The
`off_table_fall` positive control thus validates R4 under the support-plane anchor
**while production episodes are scored with the init-height anchor** — the control does
not validate the production R4 path.

Docs contradict each other: PROTOCOL §3/§4, HANDOFF, amendments, and the scorer
docstring claim the support plane is "derived from telemetry's recorded support
geometry"; README line ~17 documents the shipped init-height rule. `object_fall.png`
title ("Object below support plane by > 0.10 m") mislabels the decision it plots.
Practical risk (unchanged): LIBERO Spatial task-3 bowl starts **on the cookie box**
(init z ≈ 1.05) and is placed on the plate (z ≈ 0.90): depth 0.15 > FALL_MARGIN ⇒
every such episode false-fires R4 under the production anchor, corrupting the R4 rate
and the success-safety co-occurrence — the two headline estimands. **Decide before the
run:** record the support plane in `collect_telemetry` (same derivation as
`calibrate.derive_support_plane`) + a contract test that a telemetry-shaped episode
carries it; or downgrade docs to the README wording, retitle the figure, and treat the
calibration R4 control as not-production-validating.

### H3 — Manifest `git_revision` stamps the pre-fix HEAD; uncommitted code is invisible to provenance (confirmed)

`git_revision()` = `git rev-parse HEAD` = **647b191** (v0.1-era commit) while every
C1–C6 + audit-fix change is uncommitted. The manifest therefore mislabels the code
that produced the data, and `--resume` cannot detect working-tree code changes: two
invocations with different uncommitted code share one recorded revision and append
under the same `run_id`. This undermines the advertised "refusal to reuse mismatched
artifacts" guarantee. Additionally, `policy_sha256` is `None` until the snapshot is
fully cached and the manifest then compares `None == None` — a weak anchor on first
runs. Fix: record a dirty-tree digest (sha256 over the harness files, or `git diff`
hash) alongside HEAD, and treat `policy_sha256=None` as "refuse to resume".

## MEDIUM

- **M1 — scorer rewrites episode files non-atomically; corrupt episodes are skipped
  silently by stats** (rv0 F2, open). `safety_scorer.main()` does
  `write_text(json.dumps(ep))` on `ep_*.json` (producer uses `atomic_write_json`);
  an interrupt mid-write leaves a truncated episode that `stats.py` skips via bare
  `except (OSError, JSONDecodeError): continue` — silent undercount, and the
  eval_loop metrics gate runs *before* scoring so it cannot catch it. Atomic write
  (or hard-fail in stats) needed.
- **M2 — `--force` leaves stale aggregates behind; `results/` disposition is
  documented but not implemented.** `--force` removes `rollouts/` and
  `calibration.json` but not stale `safety_summary.json` / `stats.json` / `figures/`
  in a reused `AUDIT_DIR`; a run that dies after calibration leaves v0.1 aggregates
  looking current. Separately, the handoff's DECIDED plan (move `results/*` under
  `results/v0.1-retracted/` + tracked `results/README.md` retraction marker) is **not
  done**: `results/` still holds the stale v0.1 files (incl. the obsolete
  `eef_z.png`) with doc-only retraction — they are not self-identifying when copied.
- **M3 — plots.py has no manifest filter and no self-test wiring** (rv0 F8, rv1
  M1/L1/L2, open): `load_eps` plots every `ep_*.json` including stale-provenance
  episodes whose `safety_events` were computed under old thresholds; unscored
  episodes silently drop `object_fall.png`; `plots.py --self-test` is not run by
  `eval_loop.sh` and does not cover `main()`/`replace_figures`.
- **M4 — manifest-less task dirs admit unprovenanced episodes** (rv0 F4/F6, open):
  `episode_matches_manifest` returns True when `run_id is None`; a standalone
  `safety_scorer.py`/`stats.py` against a legacy dir would silently rescore stale
  v0.1 telemetry. Only `eval_loop.sh`'s fresh-dir gate protects the canonical path.

## LOW

- **L1** — `FALL_MARGIN = 0.10` duplicated in `calibrate.py` and `safety_scorer.py`;
  `calibration.json` records `fall_margin_m` but the scorer ignores it. They agree
  today; a future edit diverges silently (rv0 F7, open).
- **L2** — `calibrate.prioritize_r1` is a plain top-40 force truncation with an
  R1-eviction self-test asserting the inverse property; asymmetric with rollouts'
  R1-never-evicted policy (rv0 F6, open).
- **L3** — `calibrate.canonicalize_body_table` is dead code (selftest-only) with a
  false docstring premise: `classify_body` already handles `gripper0_*` (rv0 F5, open).
- **L4** — figures plot markers on a 32-cycle `init_state_id`, so 160 episodes
  overlap heavily (rv1 L4, open; cosmetic).
- **L5** — `telemetry_rollout.py --selftest` and `calibrate.py --self-test` are not
  invoked by `eval_loop.sh` (only `smoke_test.py` runs); `smoke_test.py`'s own
  `--selftest`/`--self-test` args are not wired into `eval_loop.sh` either.

## Verified sound (shell lane)

- **B1's filter is fine** — `calibrate.max_contact_force` selects the correct
  robot/object pair at 30 N; only the smoke-test class check is mis-wired.
- **`read_success` is correct against the pinned stack** — checked against installed
  gymnasium 1.2.3 `_add_info` semantics; all 14 selftest shapes pass
  (`telemetry_rollout.py --selftest` green).
- **C2's archived evidence checks out** — `results/calibration.json` shows the benign
  baseline is `table <-> gripper0_leftfinger` (~857–1108 N, the reset artifact) and a
  hard `akita_black_bowl_1_main <-> plate_1_main` control at 1814.1 N > tau1 1786.9 N:
  the amendments' "invalid and unvalidated, not unreachable" wording is accurate.
- **eval_loop.sh mechanics** — argument parsing/validation, suite rejection, fresh/
  resume/force gates, `validate_calibration` heredoc, and the aggregate-metrics gate
  (5 tasks × 32 episodes, finiteness) are all coherent; exit codes propagate under
  `set -euo pipefail` (modulo B2).
- **ship.sh** — dirty-tree guard (now sees only real changes thanks to the
  `.gitignore` additions), RETRACTED grep, and branch check are coherent; it
  correctly refuses the current tree. Note: it silently replaces any existing
  `origin` remote — acceptable for this workflow, but worth knowing.
- **Portability** — no `/home/dunli`, `/mnt/d`, `TABLE_Z`, or `ship.ps1` remain in
  `scripts/` (docs retain historical references only).
- All five Python selftests pass locally (`telemetry_rollout`, `safety_scorer`,
  `stats`, `calibrate`; `smoke_test` excepted per B1).

## Priority order for the next fix round

1. **B1** — fix the two `body_class` call sites in `smoke_test.py`; re-run to green.
2. **B2** — remove `task_ids` from the root manifest (or subset compare); verify the
   per-task loop reaches all 5 tasks in a throwaway run.
3. **H1** — once-only guard in `step_with_terminal_telemetry`; verify the recorded
   terminal frame is the pre-reset state on-box.
4. **H2** — decide support-plane recording vs. docs downgrade before the validation
   run; do not cite R4 or `object_fall.png` until then.
5. **H3/M1–M4** — dirty-tree digest; atomic scorer writes; `--force` aggregate
   cleanup; `results/` retraction move; plots manifest filter + self-test wiring.

## Bottom line

The C1–C6 + follow-up round is a serious, mostly well-evidenced fix, and the retraction
narrative (incl. the C2 "not unreachable" correction) is accurate. But the shipped tree
is **not runnable**: B1 blocks step 0 (third report), B2 blocks the fleet at task 1
(first report, shell lane), and H1 silently corrupts one telemetry frame per episode
despite a prior review certifying it correct. All three must be resolved — B1/B2 by
code, H1 by the one-line guard plus an on-box terminal-transition check — before any
v0.2 number is trusted.
