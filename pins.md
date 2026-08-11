# Repro pins (contract: reproduce exact env on a fresh machine)

## Changes log
- 2026-08-12: initial pins registered before any model rollout.
- 2026-08-12 02:46: install resolved. LeRobot pinned to `d324ffe8…` (parent of `59ab2862…`): the 2026-08-10 HEAD requires Python>=3.12 (torch/torchvision cp310 wheels pinned earlier kept the env at 3.10; eval harness code identical, change is dependency-metadata only).

## Environment matrix (final resolved versions — see notes for deviations)
| component | pin | reason |
|---|---|---|
| Python | 3.10.20 | lerobot pin requires `>=3.10` (see deviation below) |
| torch | 2.9.1+cu128 (cp310, manylinux_2_28, local wheel) | |
| torchvision | 0.24.1+cu128 (cp310, local wheel) | |
| lerobot (editable) | **d324ffe810d17264a0b1e628698aa1fa09aa639c** | 59ab2862 requires Python>=3.12; d324ffe8 is its parent → last `>=3.10` commit |
| lerobot.version() | 0.4.5 | resolved at install time |
| hf-libero | 0.1.4 (>=0.1.4,<0.2.0) | LeRobot-maintained LIBERO fork |
| mujoco | 3.8.1 | |
| robosuite | 1.4.0 | |
| transformers | 5.15.0 | |
| torchcodec | 0.10.0 | |
| av | 15.1.0 | |
| datasets | 4.8.5 | |
| egl_probe | 1.0.2 **patched** | sdist declares cmake_minimum_required 2.8.12 → patched to 3.5 in CMakeLists before building |
| policy | HuggingFaceVLA/smolvla_libero sha 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 (bf16) | |
| MUJOCO_GL | egl | headless |

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