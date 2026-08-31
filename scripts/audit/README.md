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
calibrate.py → telemetry_rollout.py → safety_scorer.py → stats / plots / matrix / report
                                                              ↓
                                          render_videos → dataset_freeze → verify_audit
```

Run [`shared/eval_loop.sh`](shared/eval_loop.sh) for the manifest-gated
end-to-end flow. It executes self-tests, creates one calibration profile per
task, collects telemetry, scores it, produces figures, renders representative
videos, and refuses incompatible artifacts.

The public protocol is in [`../../docs/PROTOCOL.md`](../../docs/PROTOCOL.md);
the exact environment is documented in [`../../pins.md`](../../pins.md).

## CLI

After `pip install -e .`, the dependency-light command is available as
`probearch`. Use `probearch validate-config configs/robustness_pilot.yaml` and
`probearch robustness-manifest ...` for the matched pilot scaffold. Existing
audit directories can be processed with `probearch score`, `probearch report`,
and `probearch verify`; `probearch run` delegates to the guarded shell loop on
Linux/WSL2.
