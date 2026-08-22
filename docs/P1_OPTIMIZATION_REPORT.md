# P1 Optimization Report — Scorer Hot-Loop De-numpyfication

**Amendment P1 (2026-08-19)** — code-only throughput optimization, no threshold/rule/output change.

## Summary

`scripts/safety_scorer.py` scoring path rewritten from small-array numpy to scalar `math`. Episode-initial poses hoisted into a pre-computed cache. **Result: 2.00x median scorer wall time**, byte-identical output on protocol-sized synthetic telemetry.

## Motivation

Profile showed `tilt_deg`, `quat_multiply`, and the per-step body loop calling numpy on 3- and 4-element vectors, where dispatch overhead dominates the arithmetic.

## Changes

1. **Scalar quaternion math.** `tilt_deg`, `quat_conjugate`, `quat_multiply`, and `delta_tilt_deg` rewritten to scalar `math` instead of `np.array` / `np.linalg.norm`. The `+ 1e-12` norm guard is retained verbatim for bit-comparable behaviour with calibrated thresholds.

2. **Hoisted init pose conversions.** Episode-initial poses converted once per episode into an `init_cache` instead of re-parsed per step per body. Malformed/wrong-arity records dropped at cache-build time, preserving the old skip-body behaviour.

3. **Scalar displacement.** Replaced `np.linalg.norm(pos - init_pos)` with explicit `math.sqrt(dx*dx + dy*dy + dz*dz)`.

## Verification — Byte-Identical Output

### 40-episode equivalence check (amendment P1, first logged measurement)

| artifact | HEAD | patched |
|---|---|---|
| `safety_summary.json` sha256 | `e2cdadc79080fe46…` | identical |
| `stats.json` sha256 | `9619d1cb3ace41d6…` | identical |
| episode files | — | 0 / 40 differing |
| scorer wall time | 1.418s | 0.673s (**2.11x**) |

40 episodes, 4 tasks, 27306 events. HEAD vs patched run over identical pristine synthetic telemetry via the real CLI.

### 128-episode repeated benchmark (protocol-sized fixture)

**Fixture:** 128 episodes / 66560 steps / 101 MB / 87309 events, 4 tasks (libero_spatial_0, libero_spatial_1, libero_object_0, libero_goal_0), structured as `rollouts/<task>/ep_NNN.json` with per-task `run_manifest.json` matching the producer contract.

**Method:** 5 alternating trials per side (HEAD scorer, then patched, then HEAD, ...) to spread any thermal drift evenly. Each trial copies the pristine fixture into a fresh `AUDIT_DIR`, runs the scorer CLI, then runs `stats.py`.

| trial | HEAD scorer | patched scorer |
|---|---|---|
| 1 | 5.442s | 2.646s |
| 2 | 4.843s | 2.486s |
| 3 | 5.947s | 3.111s |
| 4 | 6.371s | 2.785s |
| 5 | 5.260s | 2.722s |

**Median:** 5.442s → 2.722s = **2.00x**  
**Min:** 4.843s → 2.486s = 1.95x  
**Per-episode:** 34.0 ms → 17.0 ms

The spread on HEAD (4.84–6.37s) is wider than the gap, which is why the median is the honest figure.

| artifact | identical |
|---|---|
| `safety_summary.json` sha256 | yes — `301ed43bbf283a40…` |
| `stats.json` sha256 | yes — `787fb93d1cdcc8ab…` |
| 128 episode files | 0 differing |

87309 events, `R1: 86413, R2: 512, R3: 384, R4: 0, R5: 0`, success_rate 0.25.

## Gates

- `safety_scorer.py --selftest` PASS
- `stats.py --selftest` PASS
- `telemetry_rollout.py --selftest` PASS
- `smoke_test.py` → `SMOKE PASSED`

## Limitations

Throughput measurement on synthetic telemetry. R1 counts are synthetic-fixture artifacts; R4/R5 never firing means those paths are essentially untimed. Real LIBERO episodes have different step counts and contact density, so expect wall-clock to move even though ~2.0x should carry.

Not a validation run: no mujoco, lerobot, CUDA, or LIBERO on this machine. v0.1 numbers stay retracted; the calibration-sha gate still prevents rescoring old telemetry. The blocking item in `docs/BACKLOG.md` (real validation run on the GPU box) is unchanged — this just halves what each scoring pass costs once you're there.

## Honest Cross-Scale Figure

The 40-episode check measured 2.11x; the 128-episode repeated benchmark measured 2.00x median. Honest figure is **~2.0x**. The 2.11x is the 40-episode sample and is left as-measured in amendment P1 rather than retro-edited.
