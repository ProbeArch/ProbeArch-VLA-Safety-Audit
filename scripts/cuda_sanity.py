#!/usr/bin/env python
"""cuda_sanity.py - 20-line torch.cuda matmul proving 3050 4GB OOM-safe.

Runs a tiny fp16 matmul on cuda:0 and reports peak VRAM. Must stay <4GB
(3050 Laptop 4GB / WSL2 pins.md:27-33). Called by pilot A before any rollout.
"""
import sys

def main():
    try:
        import torch
    except ImportError as e:
        print(f"cuda_sanity FAILED: torch not installed: {e}", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("cuda_sanity FAILED: torch.cuda.is_available() is False", file=sys.stderr)
        return 1
    # 1024x1024 fp16 = 2 MB per matrix, matmul = 2 MB, sum = trivial vs 4GB
    # Use bf16 when available to match policy dtype (smolvla_libero bf16)
    dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
    device = torch.device("cuda:0")
    # Reset peak stats so pilot measures clean
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass
    try:
        a = torch.randn(1024, 1024, device=device, dtype=dtype)
        b = torch.randn(1024, 1024, device=device, dtype=dtype)
    except RuntimeError as e:
        print(f"cuda_sanity FAILED: OOM at alloc: {e}", file=sys.stderr)
        return 1
    try:
        # Tiny kernel: fused matmul + sum
        c = a @ b
        s = c.sum().item()
    except RuntimeError as e:
        print(f"cuda_sanity FAILED: OOM at matmul: {e}", file=sys.stderr)
        return 1
    try:
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        cur = torch.cuda.memory_allocated(device) / 1e9
        free, total = torch.cuda.mem_get_info(device)
        free_g, total_g = free / 1e9, total / 1e9
        print(f"cuda_sanity OK: sum={s:.3g} cur={cur:.2f}GB peak={peak:.2f}GB free={free_g:.2f}/{total_g:.2f}GB dtype={dtype}")
        if peak > 3.5:
            print(f"cuda_sanity WARNING: peak {peak:.2f}GB >3.5GB near 4GB limit", file=sys.stderr)
        # Cleanup for next stage (policy load needs ~1.9GB)
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"cuda_sanity OK but mem query failed: {e}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
