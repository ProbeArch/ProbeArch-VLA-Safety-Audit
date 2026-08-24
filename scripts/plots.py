#!/usr/bin/env python
"""plots.py - generate report figures from rollout telemetry (rollouts/<task>/ep_*.json).

Figures are derived from the rollout telemetry files only (not stats.json):
contact force distribution, per-episode max object displacement, safety event
onset timing, and the R4 verdict figure (taken from the scorer's own decisions;
its title states the anchor the scorer actually used - recorded support plane
when present, otherwise the object's initial height).  Run with --self-test to
generate all figures from synthetic producer-shaped episodes without any
rollout data.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AUDIT = Path(os.environ.get("AUDIT_DIR", str(Path.home() / "audit")))
ROLL = AUDIT / "rollouts"
OUT = AUDIT / "figures"

# Keys every episode produced by telemetry_rollout.py must carry (the scorer
# adds safety_events later; episodes are valid in either state).
_EPISODE_KEYS = ("task", "ep_ix", "init_state_id", "n_steps", "steps")


def _validate_episode(episode, path):
    """Fail loudly and specifically when a producer drops a key plots.py reads."""
    for key in _EPISODE_KEYS:
        if key not in episode:
            raise RuntimeError(f"episode {path} missing producer key {key!r}")
    steps = episode.get("steps")
    if not isinstance(steps, list) or not steps:
        raise RuntimeError(f"episode {path} has no steps")
    for s in steps:
        if not isinstance(s.get("contacts"), list):
            raise RuntimeError(f"episode {path} step {s.get('t')} missing contacts list")
        if not isinstance(s.get("bodies"), dict):
            raise RuntimeError(f"episode {path} step {s.get('t')} missing bodies dict")
    events = episode.get("safety_events")
    if events is not None:
        for ev in events:
            if "rule" not in ev or "first_t" not in ev:
                raise RuntimeError(f"episode {path} has a safety event missing rule/first_t")


def task_run_id(task):
    """run_id from a task's run_manifest.json, or None when absent/unreadable."""
    manifest_path = task / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    run_id = manifest.get("run_id")
    return run_id if isinstance(run_id, str) else None


def episode_matches_manifest(episode, run_id):
    """An episode is plotted only when its provenance matches the run manifest.

    Mirrors stats.py / safety_scorer.py: episodes from other runs (stale v0.1
    telemetry, foreign tasks) must not be mixed into the figures, and a stale
    episode with an old schema must not abort the figure run (M1 in
    rv1_plots_review).
    """
    if run_id is None:
        return False
    provenance = episode.get("provenance")
    if not isinstance(provenance, dict):
        return False
    return provenance.get("run_id") == run_id


def load_eps():
    eps = []
    if not ROLL.is_dir():
        return eps
    for t in sorted(p for p in ROLL.iterdir() if p.is_dir()):
        run_id = task_run_id(t)
        skipped = 0
        for f in sorted(t.glob("ep_*.json")):
            try:
                episode = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                # Deliberately fail loud: a truncated file silently dropping
                # telemetry from an audit figure is worse than a clear error.
                raise RuntimeError(f"unreadable episode {f}: {exc}") from exc
            if not episode_matches_manifest(episode, run_id):
                skipped += 1
                continue
            _validate_episode(episode, f)
            eps.append(episode)
        if skipped:
            print(
                f"{t.name}: skipped {skipped} episode(s) not covered by run manifest",
                flush=True,
            )
    return eps


def main():
    eps = load_eps()
    if not eps:
        raise SystemExit(f"no episodes found under {ROLL}")

    AUDIT.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".figures-", dir=AUDIT))
    try:
        generate_figures(eps, staged)
        replace_figures(staged)
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        raise
    print("figures written to", OUT)


def _has_support_geometry(eps):
    """True when any episode carries recorded support-plane geometry.

    The scorer resolves R4's anchor as: episode ``support_plane_z``, per-step
    ``support_plane_z``/``support_z``, ``support_planes``, ``static_bodies``,
    else the object's own initial height. Production rollout episodes currently
    record none of the geometry keys, so the figure title must state the
    anchor that was actually used (H1 in rv1_plots_review).
    """
    for e in eps:
        v = e.get("support_plane_z")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return True
        planes = e.get("support_planes")
        if isinstance(planes, dict) and planes:
            return True
        statics = e.get("static_bodies")
        if isinstance(statics, dict) and statics:
            return True
        for s in e.get("steps", []):
            if isinstance(s, dict):
                for key in ("support_plane_z", "support_z"):
                    v = s.get(key)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        return True
    return False


def generate_figures(eps, out):
    from safety_scorer import FALL_MARGIN

    # 1. contact force distribution (all contacts across all steps)
    all_f = []
    for e in eps:
        for s in e["steps"]:
            for contact in s["contacts"]:
                # The stable first three fields are body1, body2, and force;
                # newer telemetry may append preclassified body classes.
                all_f.append(contact[2])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(np.log10(np.clip(all_f, 1e-6, None)), bins=60)
    ax.set_xlabel("log10 contact force (N)")
    ax.set_ylabel("count")
    ax.set_title(f"Contact force distribution ({len(all_f):,} samples)")
    fig.tight_layout()
    fig.savefig(out / "forces.png", dpi=150)
    plt.close(fig)

    # 2. per-episode max displacement of each object (R2 view)
    fig, ax = plt.subplots(figsize=(6, 3))
    rng = np.random.default_rng(0)
    for e in eps:
        init = e["steps"][0]["bodies"]
        mx = 0.0
        for s in e["steps"]:
            for name, (pos, _) in s["bodies"].items():
                if name in init:
                    mx = max(mx, float(np.linalg.norm(np.array(pos) - np.array(init[name][0]))))
        # jitter x so 32-cycle init_state_id markers do not stack (L4)
        ax.plot([e["init_state_id"] + float(rng.uniform(-0.12, 0.12))], [mx], "o", ms=3)
    ax.set_xlabel("init_state_id")
    ax.set_ylabel("max object displacement (m)")
    ax.set_title("Max displacement per episode")
    fig.tight_layout()
    fig.savefig(out / "displacement.png", dpi=150)
    plt.close(fig)

    # 3. first safety event timing
    fracs = []
    for e in eps:
        m = e["n_steps"]
        for ev in e.get("safety_events", []):
            if m > 0:
                fracs.append(ev["first_t"] / m)
    if fracs:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(fracs, bins=40)
        ax.set_xlabel("first event time / episode length")
        ax.set_ylabel("count")
        ax.set_title("Safety event onset")
        fig.tight_layout()
        fig.savefig(out / "event_onset.png", dpi=150)
        plt.close(fig)

    # 4. R4 verdicts, taken from the scorer's authoritative decisions (never
    # reconstructed here). The anchor is the recorded support plane when the
    # producer wrote one; otherwise the scorer falls back to each object's own
    # initial height, and the title below states the anchor actually used (H1
    # in rv1_plots_review).
    scored_eps = [e for e in eps if "safety_events" in e]
    unscored = len(eps) - len(scored_eps)
    if unscored:
        if not scored_eps:
            raise RuntimeError(
                f"no scored episodes found under {ROLL}: {unscored} of {len(eps)} "
                f"episode(s) unscored; run safety_scorer.py before plots.py"
            )
        print(
            f"warning: {unscored} of {len(eps)} episode(s) unscored; "
            f"R4/onset figures exclude them",
            file=sys.stderr,
            flush=True,
        )
    if scored_eps:
        fig, ax = plt.subplots(figsize=(6, 3))
        rng = np.random.default_rng(0)
        for e in scored_eps:
            hit = int(any(ev["rule"] == "R4" for ev in e["safety_events"]))
            ax.plot([e["init_state_id"] + float(rng.uniform(-0.12, 0.12))], [hit], "o", ms=3)
        ax.set_yticks([0, 1], labels=["clear", "violation"])
        ax.set_ylim(-0.25, 1.25)
        ax.set_xlabel("init_state_id")
        ax.set_ylabel("R4 verdict")
        if _has_support_geometry(eps):
            ax.set_title(f"R4: object below support plane by > {FALL_MARGIN:.2f} m")
        else:
            ax.set_title(
                f"R4: object below its init height by > {FALL_MARGIN:.2f} m "
                "(support plane not recorded)"
            )
        fig.tight_layout()
        fig.savefig(out / "object_fall.png", dpi=150)
        plt.close(fig)

def replace_figures(staged):
    """Install a complete figure set, rolling back if the final rename fails."""
    backup = Path(tempfile.mkdtemp(prefix=".figures-old-", dir=AUDIT))
    backup.rmdir()
    had_output = OUT.exists()
    if had_output:
        os.replace(OUT, backup)
    try:
        os.replace(staged, OUT)
    except Exception:
        if had_output:
            os.replace(backup, OUT)
        raise
    if had_output:
        # Replacing the directory removes stale managed files, including eef_z.png.
        shutil.rmtree(backup)


def _synthetic_episode(ep_ix, scored):
    """Episode shaped exactly as telemetry_rollout.py + safety_scorer.py write it."""
    quat = [1.0, 0.0, 0.0, 0.0]
    episode = {
        "task": "libero_spatial",
        "task_id": 0,
        "env_ix": ep_ix,
        "pair": ep_ix,
        "ep_ix": ep_ix,
        "init_state_id": ep_ix % 32,
        "success": False,
        "n_steps": 2,
        "max_episode_steps": 280,
        "steps": [
            {
                "t": 0,
                "contacts": [["robot0_link0", "bowl", 12.3]],
                "contact_details": [
                    {
                        "body1": "robot0_link0",
                        "class1": "robot",
                        "body2": "bowl",
                        "class2": "object",
                        "force_N": 12.3,
                        "torque_Nm": 0.0,
                    }
                ],
                "bodies": {"bowl": [[0.5, 0.2, 1.0], quat]},
                "eef": [0.4, 0.2, 1.1],
            },
            {
                "t": 1,
                "contacts": [],
                "bodies": {"bowl": [[0.6, 0.2, 0.85], quat]},
                "eef": [0.4, 0.2, 1.1],
            },
        ],
    }
    if scored:
        episode["safety_events"] = [
            {"rule": "R2", "first_t": 0, "occurrences": 1, "base": "bowl moved"},
            {"rule": "R4", "first_t": 1, "occurrences": 1, "base": "bowl fell"},
        ]
    return episode


def _episode_with_provenance(episode, run_id="selftest-run"):
    ep = dict(episode)
    ep["provenance"] = {"run_id": run_id}
    return ep


_FIGURE_NAMES = ("forces.png", "displacement.png", "event_onset.png", "object_fall.png")


def _selftest_pipeline(root, episodes):
    """Cover load_eps, main()'s staged swap, and the replace_figures rollback."""
    task_dir = ROLL / "libero_spatial"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "run_manifest.json").write_text(json.dumps({"run_id": "selftest-run"}))
    for ep_ix, ep in enumerate(episodes):
        (task_dir / f"ep_{ep_ix:03d}.json").write_text(
            json.dumps(_episode_with_provenance(ep))
        )
    # A stale episode from another run (old schema, missing producer keys) must
    # be skipped by the manifest filter, not abort the figure run (M1).
    (task_dir / "ep_002.json").write_text(
        json.dumps({"provenance": {"run_id": "other-run"}})
    )
    loaded = load_eps()
    if len(loaded) != len(episodes):
        raise SystemExit(
            f"self-test FAILED: load_eps returned {len(loaded)} episodes, "
            f"expected {len(episodes)} (manifest filter broken)"
        )

    # main() end to end: staged generation + atomic swap into OUT.
    main()
    for name in _FIGURE_NAMES:
        path = OUT / name
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"self-test FAILED: main() did not install {name}")

    # replace_figures rollback: when the staged->OUT rename fails, the previous
    # complete figure set must be restored, not lost (L2).
    real_replace = os.replace
    failed = {"flag": False}

    def flaky_replace(src, dst):
        if Path(dst) == OUT and not failed["flag"]:
            failed["flag"] = True
            raise OSError("simulated replace failure")
        return real_replace(src, dst)

    os.replace = flaky_replace
    try:
        staged = Path(tempfile.mkdtemp(prefix=".figures-", dir=AUDIT))
        (staged / "partial.txt").write_text("partial")
        try:
            replace_figures(staged)
            raise SystemExit(
                "self-test FAILED: replace_figures did not raise on staged->OUT failure"
            )
        except OSError:
            pass
    finally:
        os.replace = real_replace
    for name in _FIGURE_NAMES:
        if not (OUT / name).is_file():
            raise SystemExit(f"self-test FAILED: rollback did not restore {name}")


def self_test():
    """Generate every figure from producer-shaped synthetic episodes."""
    global AUDIT, ROLL, OUT
    old_audit_dir = os.environ.get("AUDIT_DIR")
    with tempfile.TemporaryDirectory(prefix="plots-self-test-") as tmp:
        root = Path(tmp)
        # safety_scorer reads calibration.json at import time for FALL_MARGIN.
        (root / "calibration.json").write_text(
            json.dumps(
                {
                    "tau1_force_N": 1e6,
                    "tau2_displacement_m": 1e6,
                    "tau_tilt_deg": 180.0,
                }
            )
        )
        os.environ["AUDIT_DIR"] = str(root)
        saved_dirs = (AUDIT, ROLL, OUT)
        AUDIT, ROLL, OUT = root, root / "rollouts", root / "figures"
        try:
            episodes = [_synthetic_episode(0, scored=True), _synthetic_episode(1, scored=False)]
            out = root / "figures"
            out.mkdir(parents=True, exist_ok=True)
            generate_figures(episodes, out)
            for name in _FIGURE_NAMES:
                path = out / name
                if not path.is_file() or path.stat().st_size == 0:
                    raise SystemExit(f"self-test FAILED: {name} was not produced")
            _selftest_pipeline(root, episodes)
        finally:
            AUDIT, ROLL, OUT = saved_dirs
            if old_audit_dir is None:
                os.environ.pop("AUDIT_DIR", None)
            else:
                os.environ["AUDIT_DIR"] = old_audit_dir
    print(
        "plots self-test OK: figures generated from producer-shaped episodes; "
        "load_eps/main/replace_figures(rollback) covered"
    )


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
