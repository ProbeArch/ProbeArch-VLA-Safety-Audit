#!/usr/bin/env python
"""plots.py - generate report figures from rollout telemetry + stats.json."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AUDIT = Path("/home/dunli/audit")
ROLL = AUDIT / "rollouts"
OUT = AUDIT / "figures"
OUT.mkdir(exist_ok=True)


def load_eps():
    eps = []
    for t in sorted(p for p in ROLL.iterdir() if p.is_dir()):
        for f in sorted(t.glob("ep_*.json")):
            eps.append(json.loads(f.read_text()))
    return eps


def main():
    eps = load_eps()
    if not eps:
        print("no episodes yet")
        return
    # 1. contact force distribution (all contacts across all steps)
    all_f = []
    for e in eps:
        for s in e["steps"]:
            for _, _, f in s["contacts"]:
                all_f.append(f)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(np.log10(np.clip(all_f, 1e-6, None)), bins=60)
    ax.set_xlabel("log10 contact force (N)")
    ax.set_ylabel("count")
    ax.set_title(f"Contact force distribution ({len(all_f):,} samples)")
    fig.tight_layout()
    fig.savefig(OUT / "forces.png", dpi=150)
    plt.close(fig)

    # 2. per-episode max displacement of each object (R2 view)
    fig, ax = plt.subplots(figsize=(6, 3))
    for e in eps:
        init = e["steps"][0]["bodies"]
        mx = 0.0
        for s in e["steps"]:
            for name, (pos, _) in s["bodies"].items():
                if name in init:
                    mx = max(mx, float(np.linalg.norm(np.array(pos) - np.array(init[name][0]))))
        ax.plot([e["init_state_id"]], [mx], "o", ms=3)
    ax.set_xlabel("init_state_id")
    ax.set_ylabel("max object displacement (m)")
    ax.set_title("Max displacement per episode")
    fig.tight_layout()
    fig.savefig(OUT / "displacement.png", dpi=150)
    plt.close(fig)

    # 3. first safety event timing
    if "safety_events" in eps[0]:
        fracs = []
        for e in eps:
            m = e["n_steps"]
            for ev in e["safety_events"]:
                fracs.append(ev["first_t"] / m)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(fracs, bins=40)
        ax.set_xlabel("first event time / episode length")
        ax.set_ylabel("count")
        ax.set_title("Safety event onset")
        fig.tight_layout()
        fig.savefig(OUT / "event_onset.png", dpi=150)
        plt.close(fig)

    # 4. eef height min per episode (R4 view)
    fig, ax = plt.subplots(figsize=(6, 3))
    for e in eps:
        zs = [s["eef"][2] for s in e["steps"] if s.get("eef")]
        ax.plot([e["init_state_id"]], [min(zs) if zs else 0], "o", ms=3)
    ax.axhline(-0.05, color="r", ls="--", lw=0.8)
    ax.set_ylabel("min eef z (m)")
    ax.set_title("Lowest eef height per episode")
    fig.tight_layout()
    fig.savefig(OUT / "eef_z.png", dpi=150)
    plt.close(fig)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()