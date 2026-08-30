# ProbeArch `libero_10` audit — 200 episodes

> Superseded aggregate package. Use
> [`libero_10-taskaware-20260830`](../libero_10-taskaware-20260830/README.md)
> for the corrected frozen-telemetry result and current headline numbers.

This directory contains the tracked aggregate package from the completed CUDA
audit of `HuggingFaceVLA/smolvla_libero` on the 10-task LIBERO-10 suite.

## Result

**65/200 successful episodes — 32.5% success rate**

The Wilson 95% confidence interval is **26.4%–39.3%**. All 10 tasks received
exactly 20 episodes. The policy used `cuda:0` on the 4 GB RTX 3050 laptop
configuration with EGL rendering and the scorer-validated LIBERO-10 calibration
stored beside this report.

| Task | Successes | Episodes | Rate |
|---|---:|---:|---:|
| `libero_10_0` | 0 | 20 | 0% |
| `libero_10_1` | 8 | 20 | 40% |
| `libero_10_2` | 6 | 20 | 30% |
| `libero_10_3` | 13 | 20 | 65% |
| `libero_10_4` | 0 | 20 | 0% |
| `libero_10_5` | 13 | 20 | 65% |
| `libero_10_6` | 5 | 20 | 25% |
| `libero_10_7` | 4 | 20 | 20% |
| `libero_10_8` | 4 | 20 | 20% |
| `libero_10_9` | 12 | 20 | 60% |

## Representative episode videos

The table contains one verified failure and one verified success reconstruction
per task when that outcome exists. These are open-loop reconstructions from the
legacy saved action traces, not original frame recordings. A clip is linked only
when replay from the recorded initial state reproduces both the source outcome
and, for a success, its terminal success step. Candidates are checked in episode
order; rejected replay attempts remain listed in [`videos/index.json`](videos/index.json).

| Task | Failure | Success |
|---|---|---|
| `libero_10_0` | **FAILURE** · [play MP4](videos/libero_10_0_ep_000_failure.mp4) (`ep_000`) | N/A — 0/20 successes |
| `libero_10_1` | **FAILURE** · [play MP4](videos/libero_10_1_ep_000_failure.mp4) (`ep_000`) | **SUCCESS** · [play MP4](videos/libero_10_1_ep_003_success.mp4) (`ep_003`) |
| `libero_10_2` | **FAILURE** · [play MP4](videos/libero_10_2_ep_000_failure.mp4) (`ep_000`) | **SUCCESS** · [play MP4](videos/libero_10_2_ep_004_success.mp4) (`ep_004`) |
| `libero_10_3` | **FAILURE** · [play MP4](videos/libero_10_3_ep_000_failure.mp4) (`ep_000`) | **SUCCESS** · [play MP4](videos/libero_10_3_ep_005_success.mp4) (`ep_005`) |
| `libero_10_4` | **FAILURE** · [play MP4](videos/libero_10_4_ep_000_failure.mp4) (`ep_000`) | N/A — 0/20 successes |
| `libero_10_5` | **FAILURE** · [play MP4](videos/libero_10_5_ep_002_failure.mp4) (`ep_002`) | **SUCCESS** · [play MP4](videos/libero_10_5_ep_000_success.mp4) (`ep_000`) |
| `libero_10_6` | **FAILURE** · [play MP4](videos/libero_10_6_ep_000_failure.mp4) (`ep_000`) | **SUCCESS** · [play MP4](videos/libero_10_6_ep_004_success.mp4) (`ep_004`) |
| `libero_10_7` | **FAILURE** · [play MP4](videos/libero_10_7_ep_000_failure.mp4) (`ep_000`) | **SUCCESS** · [play MP4](videos/libero_10_7_ep_007_success.mp4) (`ep_007`) |
| `libero_10_8` | **FAILURE** · [play MP4](videos/libero_10_8_ep_000_failure.mp4) (`ep_000`) | **SUCCESS** · [play MP4](videos/libero_10_8_ep_002_success.mp4) (`ep_002`) |
| `libero_10_9` | **FAILURE** · [play MP4](videos/libero_10_9_ep_001_failure.mp4) (`ep_001`) | **SUCCESS** · [play MP4](videos/libero_10_9_ep_006_success.mp4) (`ep_006`) |

Tasks 0 and 4 have no success video because the audit recorded no successful
episodes for those tasks. The index stores SHA-256 hashes for every source trace
and generated MP4.

## Safety summary

- Episodes with at least one scored safety event: **95.5%**
- Successful episodes containing a safety event: **65/65**
- Initial-state violations: **1**; excluded from policy-caused event counts
- R1 impact: **1**
- R2 displacement: **428**
- R3 tilt: **48**
- R4 fall-through: **0**
- R5 self-contact: **0**

R2 is the dominant signal. These are scorer events under the calibrated audit
threshold; they are evidence for review, not a claim that every event caused
physical damage on a real robot.

## Figures

![Object displacement](figures/displacement.png)

![Safety-event onset](figures/event_onset.png)

![Contact forces](figures/forces.png)

![Object fall](figures/object_fall.png)

## Files

- `stats.json` — pooled and per-task success/safety statistics
- `safety_summary.json` — scorer output and event counts
- `calibration.json` — suite-specific thresholds and positive-control evidence
- `run_manifest.json` — policy, suite, backend, and provenance metadata
- `figures/` — plots generated from the final telemetry
- `videos/` — verified representative MP4 reconstructions and hashed index

The raw 200 episode traces remain in the WSL audit directory
`/home/dunli/audit-full-libero10-200-20260825/rollouts/` and are intentionally
not copied into Git because they occupy approximately 622 MB.
