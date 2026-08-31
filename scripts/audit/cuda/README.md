# CUDA helpers

The supported CUDA execution path is
`scripts/audit/shared/eval_loop.sh` with `POLICY_BACKEND=cuda`. It uses the
official policy/environment preprocessing and is the path used by the current
400-episode evidence set.

- `cuda_sanity.py` checks the installed CUDA runtime.
- `cuda_scorer_batch.py` is an optional parity/benchmark experiment. It is not
  used to produce audit results and reports explicitly when it falls back to CPU.
- `run_pilot_*.sh`, `run_fixed_single.sh`, and `run_test_brain*.sh` are retained
  historical target-machine diagnostics with machine-specific paths. They are
  not portable launchers and must not be used for a current audit.

Use the root `run_fleet.sh` for a portable fleet launch.
