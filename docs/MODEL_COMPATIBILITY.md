# VLA compatibility gates

This document records what can be established before allocating GPU episodes.
A model is included in a comparison only after its checkpoint, action space,
preprocessing, embodiment, and rollout protocol are verified in its own
environment.

## TurboVLA

The official [TurboVLA repository](https://github.com/H-EmbodVis/TurboVLA)
publishes a LIBERO evaluation recipe and states that the released LIBERO setup
uses two DINOv3 views, BERT text, 7-D actions, and action chunks of 12. Its
reference LIBERO environment lists PyTorch 2.3.1, torchvision 0.18.1,
transformers 4.56, and TensorFlow/TensorFlow Datasets dependencies. The
[official checkpoint page](https://huggingface.co/H-EmbodVis/TurboVLA) lists
LIBERO and RoboTwin weights.

Therefore TurboVLA is a separate adapter/environment candidate, not a drop-in
replacement for the current SmolVLA LeRobot path. Before episodes:

- pin the exact official checkpoint and repository revision;
- verify the Franka/LIBERO embodiment and action ordering;
- verify DINOv3/BERT preprocessing, image views, normalization, chunking, and
  control horizon;
- run the official `dry_run_model_load` or equivalent load check;
- record VRAM, dtype, latency, and license metadata;
- convert only the final action into ProbeArch’s rollout contract.

The RTX 3050 may not fit the reference stack and should be treated as an
unknown until measured. A failed load is `NOT_EVALUATED`, not a substitute
checkpoint.

## X-VLA

The [official X-VLA repository](https://github.com/2toinf/X-VLA) lists
`2toINF/X-VLA-Libero` as a Franka LIBERO checkpoint. X-VLA is designed around
soft prompts for cross-embodiment transfer, but that does not by itself prove
compatibility with this repository’s camera, action, state, and control
conventions. The [official LeRobot model card](https://huggingface.co/lerobot/xvla-libero)
is an additional provenance reference.

Before episodes:

- pin the exact X-VLA implementation, processor, domain/embodiment ID, and
  checkpoint hash;
- verify camera layout, image size, language format, state inputs, action
  dimension, normalization, and chunk execution;
- run a one-task stock-parity rollout;
- keep X-VLA in a separate environment if dependencies conflict;
- include it in pooled comparisons only if the protocol matches exactly.

## Comparison rule

A model comparison table must show model identity, checkpoint hash, runtime,
action mode, horizon, seed policy, image configuration, latency, peak VRAM,
episode count, and evidence coverage. If any of these differ materially, the
result is descriptive rather than a ranking.
