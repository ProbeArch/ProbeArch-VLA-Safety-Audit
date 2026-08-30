#!/usr/bin/env python3
"""Render representative audit episodes from saved action traces.

The telemetry producer records the action that led to each state as
``steps[t]["action_prev"]``.  This tool reconstructs the original LIBERO reset,
replays those actions without loading the policy, verifies the recorded outcome,
and writes GitHub-linkable MP4 evidence plus a JSON/Markdown index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_actions(episode: dict) -> list[list[float]]:
    """Return actions in execution order from producer-shaped telemetry."""
    steps = episode.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("episode has no telemetry steps")
    actions = []
    for index, step in enumerate(steps):
        action = step.get("action_prev")
        if index == 0:
            if action is not None:
                raise ValueError("initial telemetry step unexpectedly has action_prev")
            continue
        if action is None:
            raise ValueError(f"telemetry step {index} is missing action_prev")
        if not isinstance(action, list) or len(action) != 7:
            raise ValueError(f"telemetry step {index} has invalid action shape")
        actions.append([float(value) for value in action])
    expected = episode.get("n_steps")
    if isinstance(expected, int) and expected != len(actions):
        raise ValueError(
            f"recorded n_steps={expected} does not match {len(actions)} replay actions"
        )
    return actions


def select_representatives(paths: list[Path]) -> dict[str, Path | None]:
    """Select the first deterministic failure and success by episode filename."""
    selected: dict[str, Path | None] = {"failure": None, "success": None}
    for path in sorted(paths):
        with path.open(encoding="utf-8") as stream:
            episode = json.load(stream)
        label = "success" if bool(episode.get("success")) else "failure"
        if selected[label] is None:
            selected[label] = path
        if all(selected.values()):
            break
    return selected


def outcome_candidates(paths: list[Path], expected_success: bool) -> list[Path]:
    """Return candidates in episode order for deterministic replay fallback.

    MuJoCo contact-heavy trajectories can diverge under open-loop action replay
    even from the same saved state.  Trying candidates in episode order keeps
    selection deterministic while ensuring that a published reconstruction has
    the same terminal outcome as its source trace.
    """
    candidates = []
    for path in sorted(paths):
        episode = json.loads(path.read_text(encoding="utf-8"))
        if bool(episode.get("success")) == expected_success:
            candidates.append(path)
    return candidates


def overlay_frame(frame, text: str):
    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rectangle((0, 0, image.width, 25), fill=(0, 0, 0, 175))
    draw.text((7, 6), text, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(image, layer).convert("RGB"))


def render_episode(
    raw_env, episode: dict, output: Path, control_fps: int, frame_stride: int
) -> dict:
    import imageio.v2 as imageio
    import numpy as np

    actions = episode_actions(episode)
    episode_index = int(episode["ep_ix"])
    init_state_id = int(episode.get("init_state_id", episode_index))
    reset_seed = int(episode.get("reset_seed", 1000 + episode_index))
    expected_success = bool(episode.get("success"))

    raw_env.init_state_id = init_state_id
    raw_env.reset(seed=reset_seed)
    output.parent.mkdir(parents=True, exist_ok=True)

    label = "SUCCESS" if expected_success else "FAILURE"
    task = str(episode["task"])
    video_fps = max(1, round(control_fps / frame_stride))
    writer = imageio.get_writer(
        output,
        fps=video_fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_log_level="error",
    )
    achieved_success = False
    success_step = None
    frames_written = 0
    try:
        frame = overlay_frame(
            raw_env.render(), f"{task}  ep {episode_index:03d}  {label}  step 0"
        )
        writer.append_data(frame)
        frames_written += 1
        for step_index, action in enumerate(actions, start=1):
            raw_env._env.step(np.asarray(action, dtype=np.float32))
            achieved_success = bool(raw_env._env.check_success())
            if (
                step_index % frame_stride == 0
                or step_index == len(actions)
                or achieved_success
            ):
                frame = overlay_frame(
                    raw_env.render(),
                    f"{task}  ep {episode_index:03d}  {label}  step {step_index}",
                )
                writer.append_data(frame)
                frames_written += 1
            if achieved_success:
                success_step = step_index
                break
        # Hold the result for one second so linked videos do not end abruptly.
        for _ in range(video_fps):
            writer.append_data(frame)
            frames_written += 1
    finally:
        writer.close()

    if achieved_success != expected_success:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"replay outcome mismatch for {task}/ep_{episode_index:03d}: "
            f"recorded success={expected_success}, replay success={achieved_success}"
        )
    if expected_success and success_step != len(actions):
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"replay success step mismatch for {task}/ep_{episode_index:03d}: "
            f"recorded {len(actions)}, replay {success_step}"
        )
    return {
        "frames": frames_written,
        "fps": video_fps,
        "control_fps": control_fps,
        "frame_stride": frame_stride,
        "replay_success": achieved_success,
        "replay_success_step": success_step,
        "reset_seed": reset_seed,
        "init_state_id": init_state_id,
    }


def markdown_index(records: list[dict], tasks: list[str]) -> str:
    by_key = {(record["task"], record["outcome"]): record for record in records}
    lines = [
        "# Representative audit videos",
        "",
        "These MP4s are open-loop reconstructions from the legacy saved action traces,",
        "not original frame recordings. A reconstruction is indexed only when it",
        "reproduces the source episode's recorded success or failure outcome.",
        "",
        "| Task | Failure | Success |",
        "|---|---|---|",
    ]
    for task in tasks:
        cells = []
        for outcome in ("failure", "success"):
            record = by_key.get((task, outcome))
            if record is None:
                cells.append("N/A")
            else:
                cells.append(
                    f"**{outcome.upper()}** · [play MP4]({record['file']}) "
                    f"(`ep_{record['episode']:03d}`)"
                )
        lines.append(f"| `{task}` | {cells[0]} | {cells[1]} |")
    lines.append("")
    return "\n".join(lines)


def write_indexes(out: Path, manifest: dict, suite: str, records: list[dict], tasks: list[str]) -> None:
    ordered = sorted(records, key=lambda record: (record["task_id"], record["outcome"]))
    index = {
        "schema_version": "probearch-video-index-v1",
        "run_id": manifest.get("run_id"),
        "policy": manifest.get("policy"),
        "policy_sha256": manifest.get("policy_sha256"),
        "suite": suite,
        "generated_unix": time.time(),
        "selection": (
            "first outcome-reproducible failure and success by episode index; "
            "open-loop replay mismatches are rejected"
        ),
        "records": ordered,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(markdown_index(ordered, tasks), encoding="utf-8")


def run_selftest() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="probearch-video-selftest-") as tmp:
        root = Path(tmp)
        episodes = [
            {"success": False, "n_steps": 1, "steps": [{"t": 0}, {"t": 1, "action_prev": [0] * 7}]},
            {"success": True, "n_steps": 1, "steps": [{"t": 0}, {"t": 1, "action_prev": [1] * 7}]},
        ]
        paths = []
        for index, episode in enumerate(episodes):
            path = root / f"ep_{index:03d}.json"
            path.write_text(json.dumps(episode), encoding="utf-8")
            paths.append(path)
        selected = select_representatives(paths)
        assert selected["failure"] == paths[0]
        assert selected["success"] == paths[1]
        assert episode_actions(episodes[0]) == [[0.0] * 7]
    print("render_videos self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-dir",
        default=os.environ.get("AUDIT_DIR", str(Path.home() / "audit")),
        help="audit root containing rollouts/",
    )
    parser.add_argument("--out", default=None, help="video output directory")
    parser.add_argument("--task-ids", nargs="+", type=int, default=None)
    parser.add_argument(
        "--fps", type=int, default=20, help="source control frequency (default: 20)"
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="render every Nth control step while replaying every action (default: 2)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest:
        run_selftest()
        return
    if args.fps < 1 or args.fps > 120:
        raise SystemExit("--fps must be between 1 and 120")
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be positive")

    audit = Path(args.audit_dir).resolve()
    rollouts = audit / "rollouts"
    manifest_path = rollouts / "run_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing rollout manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    suite = manifest["suite"]
    resolution = manifest.get("resolution", [360, 360])
    task_ids = args.task_ids if args.task_ids is not None else list(manifest["task_ids"])
    out = Path(args.out).resolve() if args.out else audit / "videos"
    out.mkdir(parents=True, exist_ok=True)

    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env
    from lerobot.envs.utils import close_envs

    index_path = out / "index.json"
    records = []
    if index_path.is_file():
        previous = json.loads(index_path.read_text(encoding="utf-8"))
        if previous.get("schema_version") != "probearch-video-index-v1" or previous.get("suite") != suite:
            raise RuntimeError(f"incompatible existing video index: {index_path}")
        selected_ids = set(task_ids)
        records.extend(
            record
            for record in previous.get("records", [])
            if int(record["task_id"]) not in selected_ids
        )
    task_names = [f"{suite}_{task_id}" for task_id in manifest["task_ids"]]
    for task_id in task_ids:
        task = f"{suite}_{task_id}"
        paths = sorted((rollouts / task).glob("ep_*.json"))
        if not paths:
            raise RuntimeError(f"no episodes found for {task}")
        env_cfg = LiberoEnv(
            task=suite,
            task_ids=[task_id],
            observation_height=int(resolution[0]),
            observation_width=int(resolution[1]),
            control_mode="relative",
        )
        envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
        try:
            raw_env = envs[suite][task_id].envs[0]
            for outcome in ("failure", "success"):
                if args.force:
                    for stale in out.glob(f"{task}_ep_*_{outcome}.mp4"):
                        stale.unlink()
                expected_success = outcome == "success"
                candidates = outcome_candidates(paths, expected_success)
                if not candidates:
                    print(f"{task}: no {outcome} episode available", flush=True)
                    continue
                mismatches = []
                for episode_path in candidates:
                    episode = json.loads(episode_path.read_text(encoding="utf-8"))
                    episode_index = int(episode["ep_ix"])
                    filename = f"{task}_ep_{episode_index:03d}_{outcome}.mp4"
                    video_path = out / filename
                    if video_path.exists() and not args.force:
                        raise RuntimeError(f"refusing to overwrite {video_path}; pass --force")
                    print(f"rendering {task} {outcome} ep_{episode_index:03d}", flush=True)
                    try:
                        replay = render_episode(
                            raw_env, episode, video_path, args.fps, args.frame_stride
                        )
                    except RuntimeError as exc:
                        if not str(exc).startswith("replay "):
                            raise
                        mismatches.append(episode_index)
                        print(f"  skipped: {exc}", flush=True)
                        continue
                    records.append(
                        {
                            "task": task,
                            "task_id": task_id,
                            "episode": episode_index,
                            "outcome": outcome,
                            "file": filename,
                            "source_episode": str(episode_path),
                            "source_episode_sha256": file_sha256(episode_path),
                            "video_sha256": file_sha256(video_path),
                            "selection_rank": len(mismatches) + 1,
                            "earlier_replay_mismatches": mismatches,
                            **replay,
                        }
                    )
                    write_indexes(out, manifest, suite, records, task_names)
                    break
                else:
                    raise RuntimeError(
                        f"no outcome-reproducible {outcome} episode for {task}; "
                        f"checked {len(candidates)} recorded candidate(s)"
                    )
        finally:
            close_envs(envs)

    write_indexes(out, manifest, suite, records, task_names)
    print(f"wrote {len(records)} verified video(s) to {out}")


if __name__ == "__main__":
    main()
