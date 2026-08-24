# MLX Harness — remaining components and test procedure

Optional Apple Silicon policy backend for ProbeArch. It swaps only `select_action`. Physics, success reading, scoring, and stats stay on the existing CUDA/LeRobot contract.

**Official audit numbers still come from `POLICY_BACKEND=cuda` on the pinned WSL/CUDA box.** Do not cite MLX rates until T4 (parity) passes.

Entry points:

- `POLICY_BACKEND=mlx scripts/eval_loop.sh …`
- `python3 scripts/telemetry_rollout.py --device mlx …`
- `python3 scripts/mlx_smolvla.py --selftest|--probe`

Runtime: [`scripts/mlx_smolvla.py`](../scripts/mlx_smolvla.py). Manifests record `policy_backend` and, for MLX, `policy_runtime: probearch-mlx-smolvla`.

## Status on this Mac (2026-08-13)

**Green**

- `mlx` 0.32.0 + `mlx-metal` (Metal `Device(gpu, 0)`)
- `safetensors` 0.7.0, `transformers` 5.5.4, `tokenizers` 0.22.2
- `scripts/mlx_smolvla.py --selftest`
- telemetry / scorer / stats / calibrate self-tests
- `scripts/smoke_test.py` synthetic phase (`SMOKE PASSED`)

**Not run**

- live env construction
- real checkpoint load (`--probe`)
- MLX closed-loop episode
- CUDA vs MLX action parity

## Remaining components

| Component | Pin / note | Needed for |
|---|---|---|
| `lerobot` editable | `d324ffe810d17264a0b1e628698aa1fa09aa639c` + [`pins/lerobot-d324ffe8-groot-dataclass.patch`](../pins/lerobot-d324ffe8-groot-dataclass.patch) | env factory, LIBERO, official CUDA policy |
| `hf-libero` | `0.1.4` (`>=0.1.4,<0.2.0`) | Spatial suite |
| `mujoco` | `3.8.1` | physics + contacts |
| `robosuite` | `1.4.0` | LIBERO backend |
| LIBERO Spatial datasets | Spatial for this audit; Visual/NEW later | env construction |
| Policy snapshot | `HuggingFaceVLA/smolvla_libero` @ `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` (~1.22 GB) | MLX `--probe` and any real rollout |
| SmolVLM tokenizer | `HuggingFaceTB/SmolVLM2-500M-Instruct` | MLX language tokens |
| Empty `$AUDIT_DIR` | default `~/audit` | calibration + rollouts |
| Render backend | macOS: `MUJOCO_GL=glfw` or `cgl`. Linux/WSL pin is `egl` | live smoke render |
| Official CUDA stack | torch `2.9.1+cu128`, CUDA GPU, Python 3.10.20 | official numbers + T4 parity |
| Smoke MLX live branch | **not written** | `smoke_test.py` live policy gate still requires CUDA |

Present on this Mac but **not** the official pin: `torch` 2.11.0, `gymnasium` 1.2.3.

Full official matrix: [`pins.md`](../pins.md).

## Test procedure

Run in order. Stop on the first fail.

### T0 — install (once)

```bash
# Clean venv. Install the pins in pins.md, including lerobot @ d324ffe8 + patch.
python3 -c "import lerobot, mujoco, libero; print('lerobot', lerobot.__version__)"

hf download HuggingFaceVLA/smolvla_libero \
  --revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102

export AUDIT_DIR=~/audit          # must be empty for a fresh run
export MUJOCO_GL=glfw             # macOS; use egl on the WSL box
```

### T1 — synthetic gates

Re-run after any harness edit. Already green on this Mac.

```bash
python3 scripts/mlx_smolvla.py --selftest
python3 scripts/telemetry_rollout.py --selftest
python3 scripts/safety_scorer.py --selftest
python3 scripts/stats.py --selftest
python3 scripts/calibrate.py --self-test
python3 scripts/smoke_test.py
```

Pass: each exits 0.

Smoke today prints `live rollout skipped` until `mujoco` and `lerobot` exist. After they exist, the live policy gate still hard-requires CUDA (`torch.cuda.is_available()`, bf16 params on CUDA). On Apple Silicon that gate will fail even if MLX works. That is a leftover, not an MLX live test.

### T2 — MLX weight load (no env)

```bash
python3 scripts/mlx_smolvla.py --probe --backend mlx
```

Pass: JSON with `"backend": "mlx"`, `"action_shape": [1, 7]`, `"finite": true`.

First run downloads the ~1.22 GB snapshot plus tokenizer files if they are not cached.

### T3 — MLX closed loop

This is the actual harness test.

```bash
export AUDIT_DIR=~/audit-mlx
rm -rf "$AUDIT_DIR"
POLICY_BACKEND=mlx scripts/eval_loop.sh libero_spatial 1 1
```

If `eval_loop.sh` dies in smoke's CUDA policy gate, run stages by hand:

```bash
export AUDIT_DIR=~/audit-mlx
python3 scripts/calibrate.py --suite libero_spatial --task-id 0
python3 scripts/telemetry_rollout.py \
  --device mlx --suite libero_spatial --task_ids 0 --n_envs 1 --n_pairs 1
python3 scripts/safety_scorer.py && python3 scripts/stats.py
```

Pass:

- `$AUDIT_DIR/rollouts/run_manifest.json` has `policy_backend: mlx` and `policy_runtime: probearch-mlx-smolvla`
- one `ep_000.json` with finite actions, contacts, and a boolean `success`
- `calibration.json` positive controls fire R1–R4 through the real scorer
- scorer/stats write without refusing provenance

### T4 — CUDA vs MLX parity (required before any comparison)

Same task, same init-state id, identical observations. One CUDA episode and one MLX episode. Compare action L2 / cosine on the first N steps.

If they diverge, keep MLX labeled experimental. Do not mix rates.

### T5 — fleet (only after T3 and T4)

```bash
export AUDIT_DIR=~/audit-mlx
POLICY_BACKEND=mlx scripts/eval_loop.sh libero_spatial 8 4
```

## Non-goals

- Replacing the official CUDA audit pin
- Treating MLX success/safety rates as interchangeable with CUDA rates
- Running Visual/NEW suites (still Spatial-only)
- Target-machine validation and paired CUDA/MLX parity; F3–F7 are closed statically in [`docs/BACKLOG.md`](BACKLOG.md)

## Related

- [`README.md`](../README.md) — reproduce steps, including `POLICY_BACKEND=mlx`
- [`pins.md`](../pins.md) — official matrix; MLX is optional and unlabeled for citation
- [`docs/PROTOCOL.md`](PROTOCOL.md) — telemetry / rule contract
- [`docs/HANDOFF.md`](HANDOFF.md) — v0.2 harness defects and validation blocker
