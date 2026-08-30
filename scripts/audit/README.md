# ProbeArch audit implementation

This directory contains the runnable measurement pipeline. It is organized by
what is portable across runtimes and what is backend-specific.

| Directory | Purpose |
|---|---|
| [`shared/`](shared/) | Core audit pipeline: telemetry capture, calibration, scoring, statistics, plots, evidence freezing, replay rendering, and verification. |
| [`cuda/`](cuda/) | CUDA sanity checks and run helpers for the official LIBERO audit path. |
| [`mlx/`](mlx/) | Experimental Apple-Silicon policy adapter. Keep its results separate until a paired parity evaluation is complete. |

## Core flow

```text
calibrate.py → telemetry_rollout.py → safety_scorer.py → stats.py / plots.py
                                      ↓
                        dataset_freeze.py → render_videos.py → verify_audit.py
```

Run [`shared/eval_loop.sh`](shared/eval_loop.sh) for the manifest-gated
end-to-end flow. It executes self-tests, creates one calibration profile per
task, collects telemetry, scores it, produces figures, renders representative
videos, and refuses incompatible artifacts.

The public protocol is in [`../../docs/PROTOCOL.md`](../../docs/PROTOCOL.md);
the exact environment is documented in [`../../pins.md`](../../pins.md).
