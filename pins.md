# Version pins — the reproducibility contract

Every number in this audit is a function of this exact stack. This file is the contract.
The SmolVLA LIBERO eval is documented-fragile: MuJoCo version, render backend, LeRobot
revision, and camera mapping all move the result by multiple points (community replication
logs: 81.5% vs official-default ~75.5% on LIBERO-Spatial from harness drift alone).

## Method

Each pin below must round-trip: `pip show <pkg>` / `git rev-parse HEAD` match this file at
the moment the final numbers were produced. Any change = re-register the eval.

## Stack

| Component          | Pin | Verification |
|--------------------|-----|--------------|
| OS                 | Ubuntu 26.04 LTS (WSL2) | `cat /etc/os-release` |
| Python             | 3.10.20 (conda env `vla-audit`) | `conda run -n vla-audit python --version` |
| GPU (host)         | RTX 3050 Laptop 4 GiB, driver 591.86 | `nvidia-smi` |
| LeRobot            | `59ab28620f3f2385f808bd4bcac7fc50cf14217a` (2026-08-10) | `git -C ~/lerobot rev-parse HEAD` |
| hf-libero          | >=0.1.4,<0.2.0 (resolved version TBD after install) | `pip show hf-libero` |
| mujoco             | pinned by hf-libero (resolved TBD) | `pip show mujoco` |
| robosuite          | pinned by hf-libero (resolved TBD) | `pip show robosuite` |
| torch / torchvision| 2.9.1+cu128 / 0.24.x+cu128 | `pip show torch` |
| Render backend     | EGL (`MUJOCO_GL=egl`), OSMesa available | env var |

## Policy under test

| Component | Pin |
|-----------|-----|
| Checkpoint | `HuggingFaceVLA/smolvla_libero` sha `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, bf16 |
| Base model   | `lerobot/smolvla_base` (SmolVLM-256M + SmolLM2-1.7B, half layers) |
| Eval protocol | 10 episodes/task, 3 seeds (LeRobot benchmark protocol), hard resets, relative control |

## Changes log

- 2026-08-12: initial pins registered before any model rollout. Tolerance: floor-matching
  versions accepted only with explicit justification recorded here.