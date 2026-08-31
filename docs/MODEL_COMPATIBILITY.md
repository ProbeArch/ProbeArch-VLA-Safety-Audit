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

### Current gate result (2026-08-31)

The official source was cloned at revision
`c7c2ba9f94ef3a734c5033be96ef3f3f5a5f3c18`. The official LIBERO-Spatial
checkpoint was downloaded from the [TurboVLA model repository](https://huggingface.co/H-EmbodVis/TurboVLA)
and verified locally with SHA-256
`a7c3faa825a6c68d365df0647c39845c3a7bb553e1e24be3729b76de22f703fa`.
CUDA was available on an NVIDIA GeForce RTX 3050 Laptop GPU with 4.0 GB
reported VRAM, and the official evaluator reached the policy/DINO encoder
construction path.

The load gate then stopped before full policy construction because the required
`facebook/dinov3-vitb16-pretrain-lvd1689m` backbone is a gated Hugging Face
repository and this environment has no authorization. No TurboVLA episode was
run and no TurboVLA result is included in the baseline comparison. Status:
`NOT_EVALUATED` pending an authorized Hugging Face account/token and a rerun of
the same pinned command. The downloaded TurboVLA checkpoint is retained only
as a provenance/load-test asset under the ignored `.verification/` directory.

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

### Current gate result (2026-08-31)

The official repository currently resolves to revision
`6bc2513f5f1cbec715cc668b414392a6cae5c671`. The official
`2toINF/X-VLA-Libero` model revision currently resolves to
`129e71460678b7236cee6fc9707f09d9fa0c3590`; its small processor/config files
loaded successfully in the isolated Python 3.11 environment. The model
checkpoint is a 3,519,068,172-byte `model.safetensors` file (HEAD ETag/SHA
`ccda9b23b8b274ef1f3fa0d4f111d178313e8b6af1a0d305a059b287c0765933`). Its
config declares `action_mode=ee6d`, `num_actions=30`, `max_len_seq=512`,
`use_proprio=true`, `torch_dtype=float32`, and Transformers `4.51.3`.

The full checkpoint was not downloaded or moved to the GPU: the official
deployment script loads it as float32 and the laptop exposes only 4.0 GB VRAM,
so a stock CUDA load is not a responsible assumption. No X-VLA episode was
run and X-VLA remains outside all reported baseline results. Status:
`NOT_EVALUATED` pending an approved low-memory loading strategy or a machine
with sufficient VRAM, followed by action/camera parity and a one-task pilot.

## Comparison rule

A model comparison table must show model identity, checkpoint hash, runtime,
action mode, horizon, seed policy, image configuration, latency, peak VRAM,
episode count, and evidence coverage. If any of these differ materially, the
result is descriptive rather than a ranking.
