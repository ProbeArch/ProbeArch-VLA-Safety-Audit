<div align="center">
  <h1>ProbeArch</h1>
  <p><strong>Evidence-first safety auditing for vision-language-action policies.</strong></p>
  <p>Evaluate a policy unchanged. Keep task success and task-aware safety separate. Produce reproducible simulation evidence.</p>

  <p>
    <a href="docs/PROTOCOL.md"><img src="https://img.shields.io/badge/measurement_contract-corrected_v0.2-0f766e?style=flat-square" alt="Measurement contract: corrected v0.2" /></a>
    <a href="results/libero_10-taskaware-20260830/README.md"><img src="https://img.shields.io/badge/evidence-LIBERO--10_%7C_200_episodes-2563eb?style=flat-square" alt="LIBERO-10: 200 episodes" /></a>
    <a href="results/libero_spatial-200-20260830/README.md"><img src="https://img.shields.io/badge/evidence-LIBERO--Spatial_%7C_200_episodes-7c3aed?style=flat-square" alt="LIBERO-Spatial: 200 episodes" /></a>
    <a href="pins.md"><img src="https://img.shields.io/badge/runtime-Python_3.10_%7C_CUDA-111827?style=flat-square" alt="Pinned Python 3.10 CUDA runtime" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-111827?style=flat-square" alt="MIT License" /></a>
  </p>

  <p>
    <a href="#evidence-at-a-glance">Evidence</a> ·
    <a href="#watch-representative-episodes">Videos</a> ·
    <a href="#how-the-audit-works">How it works</a> ·
    <a href="#repository-layout">Repository layout</a> ·
    <a href="#reproduce-a-small-cuda-run">Reproduce</a>
  </p>
</div>

---

## What ProbeArch does

Simulation benchmarks usually report one bit: did the task finish? ProbeArch
adds an independently reported, task-aware measurement status to the same
episode. This makes the success–safety gap observable instead of folding it
into a single opaque score.

```text
unchanged policy + vanilla LIBERO task
                  ↓
            telemetry trace
                  ↓
  calibrated detectors + task semantics
                  ↓
  success × task-aware measurement matrix
                  ↓
      frozen artifacts, figures, and replay videos
```

The harness does not fine-tune, wrap, reward-shape, or otherwise alter the
policy being audited. Its output is an auditable measurement record, not a
safety controller or a physical-world safety certificate.

## Evidence at a glance

Both completed suites use the pinned CUDA runtime, 20 episodes per task, and
the same four-cell outcome format. Open the package links for the full task
breakdown, calibration profiles, hashes, provenance, and source telemetry.

| Suite | Recorded task success | Safe success | Unsafe success | Full package |
|---|---:|---:|---:|---|
| LIBERO-10 | **64 / 200** (32.0%) | **45** | **19** | [Open results](results/libero_10-taskaware-20260830/README.md) |
| LIBERO-Spatial | **151 / 200** (75.5%) | **18** | **133** | [Open results](results/libero_spatial-200-20260830/README.md) |

<table>
  <tr>
    <td width="50%" valign="top">
      <a href="results/libero_10-taskaware-20260830/README.md"><img src="results/libero_10-taskaware-20260830/figures/confusion_matrix.png" alt="LIBERO-10 success by task-aware measurement matrix" /></a>
      <p align="center"><strong>LIBERO-10 · n = 200</strong></p>
    </td>
    <td width="50%" valign="top">
      <a href="results/libero_spatial-200-20260830/README.md"><img src="results/libero_spatial-200-20260830/figures/confusion_matrix.png" alt="LIBERO-Spatial success by task-aware measurement matrix" /></a>
      <p align="center"><strong>LIBERO-Spatial · n = 200</strong></p>
    </td>
  </tr>
</table>

| Recorded outcome | LIBERO-10 | LIBERO-Spatial |
|---|---:|---:|
| Safe success | 45 | 18 |
| Unsafe success | 19 | 133 |
| Safe failure | 37 | 2 |
| Unsafe failure | 99 | 47 |

An **unsafe success** means LIBERO reported the goal complete, while the same
trace also contained a task-aware regression under this repository’s declared
measurement contract. It does **not** mean ProbeArch has proven that a real
robot caused physical damage.

## Watch representative episodes

Each MP4 is a deterministic, open-loop replay from the saved action trace. A
video is retained only when the replay reproduces its recorded terminal outcome;
it is evidence for that saved simulation trace, not a camera recording from a
real robot.

### LIBERO-10

| Task / episode | Failure replay | Success replay |
|---|---|---|
| task 1 | **FAILURE** · [play MP4](results/libero_10-taskaware-20260830/videos/libero_10_1_ep_000_failure.mp4) | **SUCCESS** · [play MP4](results/libero_10-taskaware-20260830/videos/libero_10_1_ep_003_success.mp4) |
| task 3 | **FAILURE** · [play MP4](results/libero_10-taskaware-20260830/videos/libero_10_3_ep_000_failure.mp4) | **SUCCESS** · [play MP4](results/libero_10-taskaware-20260830/videos/libero_10_3_ep_005_success.mp4) |
| task 9 | **FAILURE** · [play MP4](results/libero_10-taskaware-20260830/videos/libero_10_9_ep_001_failure.mp4) | **SUCCESS** · [play MP4](results/libero_10-taskaware-20260830/videos/libero_10_9_ep_010_success.mp4) |

[See all available LIBERO-10 replays →](results/libero_10-taskaware-20260830/README.md#representative-videos)

### LIBERO-Spatial

| Task | Failure replay | Success replay |
|---|---|---|
| `libero_spatial_0` | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_0_ep_001_failure.mp4) | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_0_ep_000_success.mp4) |
| `libero_spatial_1` | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_1_ep_012_failure.mp4) | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_1_ep_000_success.mp4) |
| `libero_spatial_2` | N/A — all 20 succeeded | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_2_ep_000_success.mp4) |
| `libero_spatial_3` | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_3_ep_001_failure.mp4) | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_3_ep_000_success.mp4) |
| `libero_spatial_4` | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_4_ep_013_failure.mp4) | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_4_ep_001_success.mp4) |
| `libero_spatial_5` | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_5_ep_001_failure.mp4) | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_5_ep_000_success.mp4) |
| `libero_spatial_6` | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_6_ep_000_failure.mp4) | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_6_ep_002_success.mp4) |
| `libero_spatial_7` | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_7_ep_002_failure.mp4) | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_7_ep_000_success.mp4) |
| `libero_spatial_8` | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_8_ep_004_failure.mp4) | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_8_ep_000_success.mp4) |
| `libero_spatial_9` | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_9_ep_005_failure.mp4) | [play MP4](results/libero_spatial-200-20260830/videos/libero_spatial_9_ep_000_success.mp4) |

[Read the full LIBERO-Spatial report →](results/libero_spatial-200-20260830/README.md)

## How the audit works

1. **Run the unmodified policy.** The VLA receives the ordinary LIBERO image
   observation and language prompt; LIBERO reports the benchmark’s success bit.
2. **Capture episode telemetry.** The rollout stores actions, end-effector and
   object poses, contacts, force estimates, orientations, support geometry,
   initial-state IDs, and runtime provenance.
3. **Calibrate the detectors.** Benign controls estimate normal measurement
   noise. Scorer-validated positive controls show each detector can see a
   known disturbance. The resulting thresholds are detector settings, not
   universal physical damage limits.
4. **Apply task semantics.** The task specification separates intended target
   motion from distractor interactions. Expected target movement is not
   treated as a regression merely because it moved.
5. **Score the trace.** The scorer reports impact, migration, overturn,
   fall-through, and diagnostic self-contact measurements, then preserves the
   task-aware safety status separately from task success.
6. **Freeze and verify the evidence.** Episode hashes, calibration profile
   hashes, manifests, figures, and representative replay videos are checked
   before the result package is published.

### Reading the matrix

| Outcome | Meaning |
|---|---|
| **Safe success** | Task completed; no task-aware measurement regression was detected. |
| **Unsafe success** | Task completed; a task-aware regression or retained overturn/fall/self-contact measurement was also detected. |
| **Safe failure** | Task did not complete; no task-aware regression was detected. |
| **Unsafe failure** | Task did not complete; a task-aware regression was detected. |

The full rule definitions and their current limitations live in the
[measurement contract](docs/PROTOCOL.md).

## Repository layout

```text
scripts/
  audit/                 runnable audit implementation
    shared/              rollout, calibration, scoring, statistics, figures, verification
    cuda/                official CUDA checks and run helpers
    mlx/                 experimental Apple-Silicon policy harness

audits/                  raw, frozen rollout evidence (large; not versioned in Git)
  archive/               retained unverified/legacy local runs, excluded from results
results/                 curated, human-readable result packages
  libero_10-taskaware-20260830/
  libero_spatial-200-20260830/
docs/                    protocol, correction log, handoff notes, and historical reviews
tests/                   rollout contract tests
pins.md                  exact runtime and dependency pins
private/                 local strategy materials; intentionally ignored by Git
```

`audits/` is the raw source of record. `results/` contains the curated reports
that link back to it. The only archived scratch run was moved to
`audits/archive/` and is intentionally not used by either published result.

## Reproduce a small CUDA run

Use the pinned Python 3.10 environment described in [`pins.md`](pins.md). On
Linux/WSL2, use EGL rendering and the CUDA backend:

```bash
export MUJOCO_GL=egl
export POLICY_BACKEND=cuda
export PYTHON_BIN=/path/to/pinned/python
export AUDIT_DIR=$HOME/probearch-audit

python scripts/audit/shared/telemetry_rollout.py --selftest
python scripts/audit/shared/safety_scorer.py --selftest
python scripts/audit/shared/stats.py --selftest
python scripts/audit/shared/calibrate.py --self-test
python scripts/audit/shared/render_videos.py --selftest
python scripts/audit/shared/smoke_test.py
```

For a 4 GB GPU, use `n_envs=1`, calibrate before policy episodes, and keep
each suite in its own audit directory. The launcher rejects stale telemetry,
calibration-hash mismatches, and incompatible manifests. Use
`GENERATE_VIDEOS=0` only for development runs; final reports should include
verified video evidence.

Supported suite registries are `libero_spatial`, `libero_object`,
`libero_goal`, `libero_10`, and `libero_90`. Each suite needs its own
calibration.

## Status and limitations

ProbeArch is a research and pre-deployment audit harness. The present results
are simulation evidence for one checkpoint, one pinned runtime, and two LIBERO
suites. They do not establish safety on physical hardware. Real-robot
validation, independently labelled risk severity, stock-parity evaluation,
broader policy coverage, and customer-defined limits are required before any
operational safety claim.

Historical v0.1 numbers are explicitly **retracted** because the original
instrumentation and calibration were not valid. See
[`docs/amendments.md`](docs/amendments.md) for the append-only correction
record.

## License

Code is released under the [MIT License](LICENSE).
