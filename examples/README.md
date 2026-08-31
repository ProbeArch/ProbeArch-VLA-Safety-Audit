# Examples

The audit core has dependency-light synthetic checks that do not load CUDA,
MuJoCo, or a policy:

```bash
python scripts/audit/shared/telemetry_rollout.py --selftest
python scripts/audit/shared/safety_scorer.py --selftest
python scripts/audit/shared/confusion_matrix.py --selftest
python -m probearch check-schemas
python examples/synthetic_demo.py
```

The curated LIBERO packages under `results/` are the small public-data example:
they show the expected report, matrix, calibration index, replay-video index,
hashes, and threshold-sensitivity artifact without requiring the raw rollout
archive to be checked into Git.
