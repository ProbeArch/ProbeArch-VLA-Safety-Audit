# Backend file map - inside scripts/

Originals stay in scripts/ - no duplicates. This map tells you what each file is for.

## SHARED - used by BOTH backends (90% of repo)
- telemetry_rollout.py  -> main harness, branches on --device cuda/mlx (line 652)
- calibrate.py          -> physics calibration
- safety_scorer.py      -> R1-R5 scoring
- stats.py, plots.py    -> stats/figures
- eval_loop.sh          -> pipeline, POLICY_BACKEND=cuda|mlx
- smoke_test.py         -> synthetic + live gate
- ship.sh, calibrate.py etc. -> shared
- pins.md Common section

## CUDA ONLY (official audit)
- cuda_sanity.py
- cuda_scorer_batch.py
- run_pilot_A/B.sh, run_fixed_single.sh, run_test_brain*.sh
- telemetry_rollout.py else-branch (torch/lerobot, line 665)
- pins.md CUDA section

## MLX ONLY (experimental Apple Silicon)
- mlx_smolvla.py
- docs/MLX_HARNESS.md (ref)
- docs/reports/mlx_safety_benchmark_report.md (ref)
- telemetry_rollout.py if use_mlx branch (line 653)
