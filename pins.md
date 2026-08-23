# Repro pins (contract: reproduce exact env on a fresh machine)

## Changes log
- 2026-08-12: initial pins registered before any model rollout.
- 2026-08-12 02:46: install resolved. LeRobot pinned to `d324ffe8…` (parent of `59ab2862…`): the 2026-08-10 HEAD requires Python>=3.12 (torch/torchvision cp310 wheels pinned earlier kept the env at 3.10; eval harness code identical, change is dependency-metadata only).
- 2026-08-12 (fix round 2): `gymnasium>=1.1.1,<2.0.0` added to the resolved matrix. The telemetry success reader depends on the gymnasium 1.x `SyncVectorEnv._add_info` behavior of recursively vectorizing `final_info` into per-key arrays; the `<2.0.0` cap guards against future vector-info shape changes. Matches the pinned lerobot `pyproject.toml` constraint (resolved `gymnasium==1.2.1` in its requirements files). Post-hoc entry — recorded during harness re-verification after the v0.1 retraction.
- 2026-08-13: optional Apple Silicon policy backend (`POLICY_BACKEND=mlx` / `telemetry_rollout.py --device mlx`) added. It loads `HuggingFaceVLA/smolvla_libero` through `scripts/mlx_smolvla.py` (MLX when installed, NumPy self-tests otherwise). The official audit pin remains CUDA/LeRobot; MLX results must be labeled separately until a paired parity run exists.

## Environment matrix (final resolved versions — see notes for deviations)

### Common (both platforms)
| component | pin | reason |
|---|---|---|
| lerobot (editable) | **d324ffe810d17264a0b1e628698aa1fa09aa639c** | 59ab2862 requires Python>=3.12; d324ffe8 is its parent → last `>=3.10` commit |
| lerobot.version() | 0.4.5 | resolved at install time |
| gymnasium | **>=1.1.1,<2.0.0** | success reader depends on 1.x recursed `final_info` shape (per-key arrays); `<2.0.0` caps future shape changes; matches lerobot pin (resolved 1.2.1) |
| hf-libero | 0.1.4 (>=0.1.4,<0.2.0) | LeRobot-maintained LIBERO fork |
| mujoco | 3.8.1 | |
| robosuite | 1.4.0 | |
| transformers | 5.15.0 | |
| torchcodec | 0.10.0 | |
| av | 15.1.0 | |
| datasets | 4.8.5 | |
| egl_probe | 1.0.2 **patched** | sdist declares cmake_minimum_required 2.8.12 → patched to 3.5 in CMakeLists before building |
| policy | HuggingFaceVLA/smolvla_libero sha 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 (bf16) | |

### CUDA (official audit, WSL2/Linux, `POLICY_BACKEND=cuda`)
| component | pin | reason |
|---|---|---|
| Python | 3.10.20 | lerobot pin requires `>=3.10`; kept per `dunli`/`moses` agreement |
| torch | 2.9.1+cu128 (cp310, manylinux_2_28, local wheel) | |
| torchvision | 0.24.1+cu128 (cp310, local wheel) | |
| MUJOCO_GL | egl | headless; auto-detected on Linux via `eval_loop.sh` |

### MLX (experimental Apple Silicon, `POLICY_BACKEND=mlx`)
| component | pin | reason |
|---|---|---|
| Python | 3.10.20 (uv venv) or 3.11 | validated with 3.10.20 on M5 via `uv` per `docs/reports/mlx_safety_benchmark_report.md` |
| mlx | 0.32.1 + mlx-metal (Metal `Device(gpu,0)`) | Apple Silicon backend; `scripts/mlx_smolvla.py` |
| torch | 2.11.0 (optional, for tokenizer fallback) | not required for MLX inference; present on M5 per report |
| MUJOCO_GL | glfw (or cgl) | macOS window system; auto-detected on Darwin via `eval_loop.sh` |
| transformers | 5.5.4 / tokenizers 0.22.2 | validated on M5 |

## Install procedure quirks (learned; do not "fix" silently)
1. PyTorch CDN (download.pytorch.org) direct wheel URLs intermittently return AccessDenied; pip installs can hang with established-but-idle sockets. → Download wheels with `curl -C -` (resumable; retry on AccessDenied) and `pip install ./wheel.whl` (no `--index-url cu128`).
2. conda 25.x `conda create` does NOT install pip into new envs → always `conda create -y -n <env> python=<ver> pip`.
3. Never `pip install` as root into a user env: leaves root-owned site-packages; partial deletions poison later `conda create` reuses (stale `.dist-info` with missing package dirs). If poisoned: `rm -rf` the env dir as root before recreating.
4. pip cache is content-addressed: copy `/root/.cache/pip` → user cache to reuse bulk downloads (nvidia-* wheels etc).
5. WSL arg mangling: never pass inline quoted commands through `wsl -- sh -c "..."`; always run script files from /mnt/d (or /home/<user>).
6. Background jobs need `setsid bash -c '...' < /dev/null &` with output file on ext4 (NOT /mnt/d) to survive the wsl client exit.
7. VHDX: after `--import --vhd`, verify growth (write 14GB probe) — a stale fixed-size VHDX silently hits ENOSPC with `df` still showing free space.

## Artifacts on disk (WSL rootfs, user dunli)
- /home/dunli/torch_wheels/ — torch + torchvision wheels (keep as reinstall source; ~1.8GB)
- /home/dunli/lerobot — clone at pinned SHA (editable install)
- /home/dunli/.cache/pip — 3.0GB seeded cache
- /home/dunli/.cache/huggingface — checkpoint cache (populated by hf download step)