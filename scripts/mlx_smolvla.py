#!/usr/bin/env python
"""Apple-Silicon MLX runtime for HuggingFaceVLA/smolvla_libero.

Mirrors the pinned LeRobot SmolVLA inference path (prepare_images / prepare_state /
embed_prefix / KV-cached flow-matching denoise / action queue) without requiring
CUDA or the full LeRobot policy factory. The existing telemetry loop stays the
source of truth for physics; this module only replaces ``select_action``.

Self-tests run on NumPy. Live inference prefers ``mlx`` when installed.
"""
from __future__ import annotations

import json
import math
import os
from collections import deque
from pathlib import Path

import numpy as np

HARNESS_NAME = "probearch-mlx-smolvla"
DEFAULT_POLICY = "HuggingFaceVLA/smolvla_libero"
DEFAULT_VLM = "HuggingFaceTB/SmolVLM2-500M-Instruct"

OBS_STATE = "observation.state"
OBS_IMAGE = "observation.images.image"
OBS_IMAGE2 = "observation.images.image2"
OBS_LANG_TOKENS = "observation.language.tokens"
OBS_LANG_MASK = "observation.language.attention_mask"
ACTION = "action"

# Architecture recovered from HuggingFaceVLA/smolvla_libero + SmolVLM2-500M.
VLM_HIDDEN = 960
VLM_INTERMEDIATE = 2560
VLM_HEADS = 15
VLM_KV_HEADS = 5
VLM_HEAD_DIM = 64
VLM_LAYERS = 32
VOCAB_SIZE = 49280
VISION_HIDDEN = 768
VISION_HEADS = 12
VISION_LAYERS = 12
VISION_PATCH = 16
VISION_IMAGE = 512
VISION_MLP = 3072
SCALE_FACTOR = 4
EXPERT_HIDDEN = 480
EXPERT_INTERMEDIATE = 1280
EXPERT_HEADS = 15
EXPERT_KV_HEADS = 5
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32
CHUNK_SIZE = 50
NUM_STEPS = 10
TOKENIZER_MAX_LENGTH = 48
RMS_EPS = 1e-5
ROPE_THETA = 100_000.0
MIN_PERIOD = 0.004
MAX_PERIOD = 4.0
SELF_ATTN_EVERY_N = 2
NORM_EPS = 1e-8


class ArrayBackend:
    """Tiny ndarray facade so math is identical on NumPy and MLX."""

    name = "numpy"

    def array(self, value, dtype=None):
        return np.asarray(value, dtype=dtype)

    def asarray(self, value, dtype=None):
        arr = np.asarray(value)
        if dtype is not None and arr.dtype != np.dtype(dtype):
            arr = arr.astype(dtype, copy=False)
        return arr

    def zeros(self, shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=np.float32):
        return np.ones(shape, dtype=dtype)

    def arange(self, n, dtype=np.float32):
        return np.arange(n, dtype=dtype)

    def reshape(self, x, shape):
        return np.reshape(x, shape)

    def transpose(self, x, axes):
        return np.transpose(x, axes)

    def concat(self, xs, axis=0):
        return np.concatenate(xs, axis=axis)

    def expand_dims(self, x, axis):
        return np.expand_dims(x, axis)

    def broadcast_to(self, x, shape):
        return np.broadcast_to(x, shape)

    def matmul(self, a, b):
        return np.matmul(a, b)

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def where(self, cond, a, b):
        return np.where(cond, a, b)

    def softmax(self, x, axis=-1):
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def silu(self, x):
        return x / (1.0 + np.exp(-x))

    def gelu(self, x):
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))

    def rsqrt(self, x):
        return 1.0 / np.sqrt(x)

    def sqrt(self, x):
        return np.sqrt(x)

    def sin(self, x):
        return np.sin(x)

    def cos(self, x):
        return np.cos(x)

    def astype(self, x, dtype):
        return x.astype(dtype, copy=False)

    def to_numpy(self, x):
        return np.asarray(x)

    def eval(self, x):
        return x

    def sum(self, x, axis=None, keepdims=False):
        return np.sum(x, axis=axis, keepdims=keepdims)

    def cumsum(self, x, axis=0):
        return np.cumsum(x, axis=axis)

    def mean(self, x, axis=None, keepdims=False):
        return np.mean(x, axis=axis, keepdims=keepdims)

    def pad(self, x, pad_width, value=0):
        return np.pad(x, pad_width, constant_values=value)

    def take(self, table, indices):
        return table[np.asarray(indices, dtype=np.int64)]

    def conv2d(self, x_nchw, weight, bias, stride=1):
        # x: B,C,H,W  weight: out,in,kh,kw
        x = np.asarray(x_nchw)
        w = np.asarray(weight)
        bsz, _c, h, w_in = x.shape
        _out_c, _, kh, kw = w.shape
        oh = (h - kh) // stride + 1
        ow = (w_in - kw) // stride + 1
        patches = np.lib.stride_tricks.as_strided(
            x,
            shape=(bsz, _c, oh, ow, kh, kw),
            strides=x.strides[:2]
            + (x.strides[2] * stride, x.strides[3] * stride, x.strides[2], x.strides[3]),
            writeable=False,
        )
        y = np.einsum("bchwij,oijc->bohw", patches.transpose(0, 2, 3, 4, 5, 1), w.transpose(0, 2, 3, 1))
        if bias is not None:
            y = y + np.asarray(bias)[None, :, None, None]
        return y


class MlxBackend(ArrayBackend):
    name = "mlx"

    def __init__(self):
        import mlx.core as mx

        self.mx = mx

    def array(self, value, dtype=None):
        if dtype is None:
            return self.mx.array(value)
        return self.mx.array(value, dtype=_mlx_dtype(self.mx, dtype))

    def asarray(self, value, dtype=None):
        if hasattr(value, "dtype") and type(value).__module__.startswith("mlx"):
            if dtype is None:
                return value
            return value.astype(_mlx_dtype(self.mx, dtype))
        return self.array(value, dtype=dtype)

    def zeros(self, shape, dtype=np.float32):
        return self.mx.zeros(shape, dtype=_mlx_dtype(self.mx, dtype))

    def ones(self, shape, dtype=np.float32):
        return self.mx.ones(shape, dtype=_mlx_dtype(self.mx, dtype))

    def arange(self, n, dtype=np.float32):
        return self.mx.arange(n, dtype=_mlx_dtype(self.mx, dtype))

    def reshape(self, x, shape):
        return x.reshape(shape)

    def transpose(self, x, axes):
        return self.mx.transpose(x, axes)

    def concat(self, xs, axis=0):
        return self.mx.concatenate(xs, axis=axis)

    def expand_dims(self, x, axis):
        return self.mx.expand_dims(x, axis)

    def broadcast_to(self, x, shape):
        return self.mx.broadcast_to(x, shape)

    def matmul(self, a, b):
        return a @ b

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def where(self, cond, a, b):
        return self.mx.where(cond, a, b)

    def softmax(self, x, axis=-1):
        return self.mx.softmax(x, axis=axis)

    def silu(self, x):
        return x * self.mx.sigmoid(x)

    def gelu(self, x):
        return 0.5 * x * (1.0 + self.mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))

    def rsqrt(self, x):
        return self.mx.rsqrt(x)

    def sqrt(self, x):
        return self.mx.sqrt(x)

    def sin(self, x):
        return self.mx.sin(x)

    def cos(self, x):
        return self.mx.cos(x)

    def astype(self, x, dtype):
        return x.astype(_mlx_dtype(self.mx, dtype))

    def to_numpy(self, x):
        self.mx.eval(x)
        return np.array(x)

    def eval(self, x):
        self.mx.eval(x)
        return x

    def sum(self, x, axis=None, keepdims=False):
        return self.mx.sum(x, axis=axis, keepdims=keepdims)

    def cumsum(self, x, axis=0):
        return self.mx.cumsum(x, axis=axis)

    def mean(self, x, axis=None, keepdims=False):
        return self.mx.mean(x, axis=axis, keepdims=keepdims)

    def pad(self, x, pad_width, value=0):
        return self.mx.pad(x, pad_width, constant_values=value)

    def take(self, table, indices):
        return table[self.mx.array(indices, dtype=self.mx.int32)]

    def conv2d(self, x_nchw, weight, bias, stride=1):
        x = self.transpose(x_nchw, (0, 2, 3, 1))
        w = self.transpose(weight, (0, 2, 3, 1))
        y = self.mx.conv2d(x, w, stride=stride)
        if bias is not None:
            y = y + bias
        return self.transpose(y, (0, 3, 1, 2))


def _mlx_dtype(mx, dtype):
    if dtype is None:
        return mx.float32
    if isinstance(dtype, str):
        return getattr(mx, dtype)
    mapping = {
        np.float32: mx.float32,
        np.float16: mx.float16,
        np.int32: mx.int32,
        np.int64: mx.int32,
        np.bool_: mx.bool_,
        bool: mx.bool_,
    }
    if dtype in mapping:
        return mapping[dtype]
    name = getattr(dtype, "name", str(dtype))
    if name in ("bfloat16", "bf16"):
        return getattr(mx, "bfloat16", mx.float16)
    return mx.float32


def resolve_backend(name="auto"):
    requested = (name or "auto").lower()
    if requested in ("mlx", "auto"):
        try:
            import mlx.core as mx  # noqa: F401

            return MlxBackend()
        except ImportError:
            if requested == "mlx":
                raise
    if requested in ("numpy", "cpu", "auto"):
        return ArrayBackend()
    raise ValueError(f"unknown array backend: {name}")


def interpolate_bilinear(img, height, width):
    """img: B,C,H,W float32 -> resized B,C,height,width."""
    img = np.asarray(img, dtype=np.float32)
    _, _, src_h, src_w = img.shape
    if src_h == height and src_w == width:
        return img
    ys = (np.arange(height, dtype=np.float32) + 0.5) * (src_h / height) - 0.5
    xs = (np.arange(width, dtype=np.float32) + 0.5) * (src_w / width) - 0.5
    ys = np.clip(ys, 0.0, src_h - 1.000001)
    xs = np.clip(xs, 0.0, src_w - 1.000001)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    wy = (ys - y0).reshape(1, 1, height, 1)
    wx = (xs - x0).reshape(1, 1, 1, width)
    Ia = img[:, :, y0][:, :, :, x0]
    Ib = img[:, :, y0][:, :, :, x1]
    Ic = img[:, :, y1][:, :, :, x0]
    Id = img[:, :, y1][:, :, :, x1]
    return (Ia * (1 - wy) * (1 - wx) + Ib * (1 - wy) * wx + Ic * wy * (1 - wx) + Id * wy * wx)


def resize_with_pad(img, width, height, pad_value=0.0):
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, got {img.shape}")
    cur_h, cur_w = img.shape[2:]
    ratio = max(cur_w / width, cur_h / height)
    resized_h = max(1, int(cur_h / ratio))
    resized_w = max(1, int(cur_w / ratio))
    resized = interpolate_bilinear(img, resized_h, resized_w)
    pad_h = max(0, height - resized_h)
    pad_w = max(0, width - resized_w)
    return np.pad(resized, ((0, 0), (0, 0), (pad_h, 0), (pad_w, 0)), constant_values=pad_value)


def pad_vector(vector, new_dim):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape[-1] == new_dim:
        return vector
    out = np.zeros(vector.shape[:-1] + (new_dim,), dtype=np.float32)
    out[..., : vector.shape[-1]] = vector
    return out


def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, xp):
    time = xp.asarray(time, dtype=np.float32)
    if time.ndim != 1:
        raise ValueError("time must be (batch,)")
    if dimension % 2:
        raise ValueError("dimension must be even")
    fraction = xp.arange(dimension // 2, dtype=np.float32) / float(dimension // 2 - 1 or 1)
    # Match torch.linspace(0, 1, dimension//2)
    if dimension // 2 == 1:
        fraction = xp.zeros((1,), dtype=np.float32)
    else:
        fraction = xp.arange(dimension // 2, dtype=np.float32) / float(dimension // 2 - 1)
    period = min_period * (max_period / min_period) ** fraction
    scaling = (1.0 / period) * (2.0 * math.pi)
    sin_input = xp.expand_dims(time, 1) * xp.expand_dims(scaling, 0)
    return xp.concat([xp.sin(sin_input), xp.cos(sin_input)], axis=1)


def make_att_2d_masks(pad_masks, att_masks, xp):
    pad = xp.asarray(pad_masks)
    att = xp.asarray(att_masks)
    if hasattr(pad, "astype"):
        pad_f = xp.astype(pad, np.float32)
        att_f = xp.astype(att, np.float32)
    else:
        pad_f = pad.astype(np.float32)
        att_f = att.astype(np.float32)
    cumsum = xp.cumsum(att_f, axis=1)
    att_2d = xp.expand_dims(cumsum, 1) <= xp.expand_dims(cumsum, 2)
    pad_2d = xp.expand_dims(pad_f, 1) * xp.expand_dims(pad_f, 2)
    return att_2d & (pad_2d > 0.5)


def apply_rope(x, positions, xp, max_wavelength=10_000.0):
    # x: B,L,H,D
    d_half = x.shape[-1] // 2
    x32 = xp.astype(x, np.float32)
    freq = (2.0 / x.shape[-1]) * xp.arange(d_half, dtype=np.float32)
    timescale = max_wavelength ** freq
    pos = xp.astype(xp.expand_dims(positions, 2), np.float32)
    radians = xp.expand_dims(pos / timescale, 2)  # B,L,1,Dh
    sin = xp.sin(radians)
    cos = xp.cos(radians)
    x1 = x32[..., :d_half]
    x2 = x32[..., d_half:]
    rot = xp.concat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return xp.astype(rot, x.dtype) if hasattr(x, "dtype") else rot

def linear(x, weight, bias, xp):
    x_np = xp.to_numpy(x) if hasattr(xp, "to_numpy") else np.asarray(x)
    if x_np.size and not np.any(x_np):
        # All-zero activations times a well-scaled weight are exactly zero.
        # NumPy/Accelerate still emits overflow warnings on some macOS BLAS paths.
        y = xp.zeros(x_np.shape[:-1] + (np.asarray(weight).shape[0],), dtype=np.float32)
    else:
        y = xp.matmul(x, xp.transpose(weight, (1, 0)))
    if bias is not None:
        y = y + bias
    return y


def rms_norm(x, weight, xp, eps=RMS_EPS):
    x32 = xp.astype(x, np.float32)
    var = xp.mean(x32 * x32, axis=-1, keepdims=True)
    y = x32 * xp.rsqrt(var + eps)
    return xp.astype(y, x.dtype) * weight if hasattr(x, "dtype") else y * weight


def layer_norm(x, weight, bias, xp, eps=1e-6):
    x32 = xp.astype(x, np.float32)
    mean = xp.mean(x32, axis=-1, keepdims=True)
    var = xp.mean((x32 - mean) * (x32 - mean), axis=-1, keepdims=True)
    y = (x32 - mean) * xp.rsqrt(var + eps)
    y = y * weight
    if bias is not None:
        y = y + bias
    return xp.astype(y, x.dtype) if hasattr(x, "dtype") else y


def pixel_shuffle(x, scale_factor, xp):
    # Idefics3 / SmolVLM connector: B,S,C with S = H*W
    bsz, seq, embed_dim = x.shape
    height = width = int(seq ** 0.5)
    if height * width != seq:
        raise ValueError(f"sequence {seq} is not a square grid")
    x = xp.reshape(x, (bsz, height, width, embed_dim))
    x = xp.reshape(x, (bsz, height, width // scale_factor, embed_dim * scale_factor))
    x = xp.transpose(x, (0, 2, 1, 3))
    x = xp.reshape(
        x,
        (
            bsz,
            width // scale_factor,
            height // scale_factor,
            embed_dim * (scale_factor ** 2),
        ),
    )
    x = xp.transpose(x, (0, 2, 1, 3))
    return xp.reshape(
        x,
        (bsz, (height // scale_factor) * (width // scale_factor), embed_dim * (scale_factor ** 2)),
    )


def eager_attention(mask, query, key, value, xp, num_heads, num_kv_heads, head_dim):
    # q/k/v: B,L,H,D
    batch, seq_k, _, _ = key.shape
    groups = num_heads // num_kv_heads
    key = xp.reshape(
        xp.broadcast_to(xp.expand_dims(key, 3), (batch, seq_k, num_kv_heads, groups, head_dim)),
        (batch, seq_k, num_heads, head_dim),
    )
    value = xp.reshape(
        xp.broadcast_to(xp.expand_dims(value, 3), (batch, seq_k, num_kv_heads, groups, head_dim)),
        (batch, seq_k, num_heads, head_dim),
    )
    q = xp.astype(xp.transpose(query, (0, 2, 1, 3)), np.float32)
    k = xp.astype(xp.transpose(key, (0, 2, 1, 3)), np.float32)
    scores = xp.matmul(q, xp.transpose(k, (0, 1, 3, 2))) * (head_dim ** -0.5)
    big_neg = np.finfo(np.float32).min
    mask_b = xp.expand_dims(xp.asarray(mask), 1)
    scores = xp.where(mask_b, scores, xp.asarray(big_neg, dtype=np.float32))
    probs = xp.softmax(scores, axis=-1)
    v = xp.transpose(value, (0, 2, 1, 3))
    out = xp.matmul(xp.astype(probs, value.dtype) if hasattr(value, "dtype") else probs, v)
    out = xp.transpose(out, (0, 2, 1, 3))
    return xp.reshape(out, (batch, -1, num_heads * head_dim))


def mlp_swiglu(x, gate_w, up_w, down_w, xp):
    return linear(xp.silu(linear(x, gate_w, None, xp)) * linear(x, up_w, None, xp), down_w, None, xp)


def load_safetensors(path):
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to load SmolVLA weights") from exc
    return load_file(str(path))


def snapshot_or_path(policy_id, cache_dir=None):
    local = Path(policy_id)
    if local.exists():
        return local
    from huggingface_hub import snapshot_download

    kwargs = {"repo_id": policy_id}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return Path(snapshot_download(**kwargs))


class HashTokenizer:
    """Deterministic stand-in used only by self-tests."""

    def __call__(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        ids = []
        masks = []
        for text in texts:
            tokens = [min(VOCAB_SIZE - 1, (ord(ch) + 17) % (VOCAB_SIZE - 8) + 4) for ch in text[:TOKENIZER_MAX_LENGTH]]
            if not tokens:
                tokens = [4]
            pad = TOKENIZER_MAX_LENGTH - len(tokens)
            ids.append(tokens + [0] * pad)
            masks.append([1] * len(tokens) + [0] * pad)
        return {
            "input_ids": np.asarray(ids, dtype=np.int64),
            "attention_mask": np.asarray(masks, dtype=np.int64),
        }


def load_tokenizer(name=DEFAULT_VLM):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    except Exception:
        try:
            from tokenizers import Tokenizer
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(name, "tokenizer.json")
            tok = Tokenizer.from_file(path)

            class _Wrap:
                def __call__(self, texts, padding="max_length", truncation=True, max_length=TOKENIZER_MAX_LENGTH, **_):
                    if isinstance(texts, str):
                        texts = [texts]
                    ids, masks = [], []
                    for text in texts:
                        encoded = tok.encode(text)
                        pieces = encoded.ids[:max_length]
                        pad = max_length - len(pieces) if padding else 0
                        ids.append(pieces + [0] * pad)
                        masks.append([1] * len(pieces) + [0] * pad)
                    return {
                        "input_ids": np.asarray(ids, dtype=np.int64),
                        "attention_mask": np.asarray(masks, dtype=np.int64),
                    }

            return _Wrap()
        except Exception as exc:
            raise RuntimeError(
                "could not load SmolVLM tokenizer; install transformers or tokenizers"
            ) from exc


class SmolVLAMLX:
    """Weight-driven SmolVLA inference graph."""

    def __init__(self, weights, xp, tokenizer=None, n_action_steps=1):
        self.w = {k: xp.asarray(v) for k, v in weights.items()}
        self.xp = xp
        self.tokenizer = tokenizer or HashTokenizer()
        self.n_action_steps = n_action_steps
        self.action_queue = deque()
        self.stats = {}

    def reset(self):
        self.action_queue.clear()

    def _w(self, key):
        return self.w[key]

    def embed_language(self, tokens):
        return self.xp.take(self._w("model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight"), tokens)

    def embed_image(self, images):
        xp = self.xp
        # images: B,C,H,W in [-1, 1]
        x = images
        w = self._w("model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight")
        b = self._w("model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.bias")
        x = xp.conv2d(x, w, b, stride=VISION_PATCH)
        bsz, hidden, gh, gw = x.shape
        x = xp.transpose(x, (0, 2, 3, 1))
        x = xp.reshape(x, (bsz, gh * gw, hidden))
        pos = self._w("model.vlm_with_expert.vlm.model.vision_model.embeddings.position_embedding.weight")
        x = x + pos[None, : x.shape[1], :]
        prefix = "model.vlm_with_expert.vlm.model.vision_model.encoder.layers."
        for i in range(VISION_LAYERS):
            residual = x
            x = layer_norm(
                x,
                self._w(f"{prefix}{i}.layer_norm1.weight"),
                self._w(f"{prefix}{i}.layer_norm1.bias"),
                xp,
            )
            q = linear(x, self._w(f"{prefix}{i}.self_attn.q_proj.weight"), self._w(f"{prefix}{i}.self_attn.q_proj.bias"), xp)
            k = linear(x, self._w(f"{prefix}{i}.self_attn.k_proj.weight"), self._w(f"{prefix}{i}.self_attn.k_proj.bias"), xp)
            v = linear(x, self._w(f"{prefix}{i}.self_attn.v_proj.weight"), self._w(f"{prefix}{i}.self_attn.v_proj.bias"), xp)
            seq = x.shape[1]
            q = xp.reshape(q, (bsz, seq, VISION_HEADS, VISION_HIDDEN // VISION_HEADS))
            k = xp.reshape(k, (bsz, seq, VISION_HEADS, VISION_HIDDEN // VISION_HEADS))
            v = xp.reshape(v, (bsz, seq, VISION_HEADS, VISION_HIDDEN // VISION_HEADS))
            mask = xp.ones((bsz, seq, seq), dtype=np.bool_)
            attn = eager_attention(mask, q, k, v, xp, VISION_HEADS, VISION_HEADS, VISION_HIDDEN // VISION_HEADS)
            x = residual + linear(
                attn,
                self._w(f"{prefix}{i}.self_attn.out_proj.weight"),
                self._w(f"{prefix}{i}.self_attn.out_proj.bias"),
                xp,
            )
            residual = x
            x = layer_norm(
                x,
                self._w(f"{prefix}{i}.layer_norm2.weight"),
                self._w(f"{prefix}{i}.layer_norm2.bias"),
                xp,
            )
            x = residual + linear(
                xp.gelu(linear(x, self._w(f"{prefix}{i}.mlp.fc1.weight"), self._w(f"{prefix}{i}.mlp.fc1.bias"), xp)),
                self._w(f"{prefix}{i}.mlp.fc2.weight"),
                self._w(f"{prefix}{i}.mlp.fc2.bias"),
                xp,
            )
        x = layer_norm(
            x,
            self._w("model.vlm_with_expert.vlm.model.vision_model.post_layernorm.weight"),
            self._w("model.vlm_with_expert.vlm.model.vision_model.post_layernorm.bias"),
            xp,
        )
        x = pixel_shuffle(x, SCALE_FACTOR, xp)
        x = linear(x, self._w("model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight"), None, xp)
        return x

    def _layer_pair(self, layer_idx):
        vlm_p = f"model.vlm_with_expert.vlm.model.text_model.layers.{layer_idx}."
        exp_p = f"model.vlm_with_expert.lm_expert.layers.{layer_idx}."
        return vlm_p, exp_p

    def _split_heads(self, tensor, n_heads, head_dim):
        b, s, _ = tensor.shape
        return self.xp.reshape(tensor, (b, s, n_heads, head_dim))

    def forward_attn_layer(self, inputs_embeds, layer_idx, position_ids, attention_mask, past=None, fill_cache=False):
        xp = self.xp
        vlm_p, exp_p = self._layer_pair(layer_idx)
        prefixes = []
        queries, keys, values = [], [], []
        for hidden, prefix, n_heads, n_kv, hidden_size in (
            (inputs_embeds[0], vlm_p, VLM_HEADS, VLM_KV_HEADS, VLM_HIDDEN),
            (inputs_embeds[1], exp_p, EXPERT_HEADS, EXPERT_KV_HEADS, EXPERT_HIDDEN),
        ):
            if hidden is None:
                prefixes.append(None)
                continue
            prefixes.append(prefix)
            h = rms_norm(hidden, self._w(prefix + "input_layernorm.weight"), xp)
            q = linear(h, self._w(prefix + "self_attn.q_proj.weight"), None, xp)
            k = linear(h, self._w(prefix + "self_attn.k_proj.weight"), None, xp)
            v = linear(h, self._w(prefix + "self_attn.v_proj.weight"), None, xp)
            queries.append(self._split_heads(q, n_heads, VLM_HEAD_DIM))
            keys.append(self._split_heads(k, n_kv, VLM_HEAD_DIM))
            values.append(self._split_heads(v, n_kv, VLM_HEAD_DIM))
        query = xp.concat(queries, axis=1)
        key = xp.concat(keys, axis=1)
        value = xp.concat(values, axis=1)
        seq = query.shape[1]
        pos = position_ids[:, :seq] if position_ids.shape[1] > seq else position_ids
        mask = attention_mask[:, :seq, :seq] if attention_mask.shape[1] > seq else attention_mask
        query = apply_rope(query, pos, xp)
        key = apply_rope(key, pos, xp)
        if past is None:
            past = {}
        if fill_cache:
            past[layer_idx] = {"key_states": key, "value_states": value}
        elif layer_idx in past:
            key = xp.concat([past[layer_idx]["key_states"], key], axis=1)
            value = xp.concat([past[layer_idx]["value_states"], value], axis=1)
        attn = eager_attention(mask, query, key, value, xp, VLM_HEADS, VLM_KV_HEADS, VLM_HEAD_DIM)
        return [attn], past

    def forward_cross_attn_layer(self, inputs_embeds, layer_idx, position_ids, attention_mask, past):
        xp = self.xp
        vlm_p, exp_p = self._layer_pair(layer_idx)
        outputs = []
        if inputs_embeds[0] is not None and not past:
            hidden = rms_norm(inputs_embeds[0], self._w(vlm_p + "input_layernorm.weight"), xp)
            q = self._split_heads(linear(hidden, self._w(vlm_p + "self_attn.q_proj.weight"), None, xp), VLM_HEADS, VLM_HEAD_DIM)
            k = self._split_heads(linear(hidden, self._w(vlm_p + "self_attn.k_proj.weight"), None, xp), VLM_KV_HEADS, VLM_HEAD_DIM)
            v = self._split_heads(linear(hidden, self._w(vlm_p + "self_attn.v_proj.weight"), None, xp), VLM_KV_HEADS, VLM_HEAD_DIM)
            seq = hidden.shape[1]
            pos, expert_pos = position_ids[:, :seq], position_ids[:, seq:]
            mask = attention_mask[:, :seq, :seq]
            q = apply_rope(q, pos, xp)
            k = apply_rope(k, pos, xp)
            outputs.append(eager_attention(mask, q, k, v, xp, VLM_HEADS, VLM_KV_HEADS, VLM_HEAD_DIM))
            key_states, value_states = k, v
        else:
            expert_pos = position_ids
            key_states = past[layer_idx]["key_states"]
            value_states = past[layer_idx]["value_states"]

        if past is not None and layer_idx not in past:
            past[layer_idx] = {"key_states": key_states, "value_states": value_states}

        expert = inputs_embeds[1]
        if expert is None:
            outputs.append(None)
            return outputs, past
        hidden = rms_norm(expert, self._w(exp_p + "input_layernorm.weight"), xp)
        q = self._split_heads(linear(hidden, self._w(exp_p + "self_attn.q_proj.weight"), None, xp), EXPERT_HEADS, VLM_HEAD_DIM)
        flat_k = xp.reshape(xp.astype(key_states, self._w(exp_p + "self_attn.k_proj.weight").dtype), (*key_states.shape[:2], -1))
        flat_v = xp.reshape(xp.astype(value_states, self._w(exp_p + "self_attn.v_proj.weight").dtype), (*value_states.shape[:2], -1))
        k = self._split_heads(linear(flat_k, self._w(exp_p + "self_attn.k_proj.weight"), None, xp), EXPERT_KV_HEADS, VLM_HEAD_DIM)
        v = self._split_heads(linear(flat_v, self._w(exp_p + "self_attn.v_proj.weight"), None, xp), EXPERT_KV_HEADS, VLM_HEAD_DIM)
        min_pos = xp.reshape(xp.asarray(np.min(self.xp.to_numpy(expert_pos), axis=1)), (-1, 1))
        expert_pos = expert_pos - min_pos
        mask = attention_mask[:, -expert.shape[1] :, : k.shape[1]]
        q = apply_rope(q, expert_pos, xp)
        outputs.append(eager_attention(mask, q, k, v, xp, EXPERT_HEADS, EXPERT_KV_HEADS, VLM_HEAD_DIM))
        return outputs, past

    def transformer(self, inputs_embeds, attention_mask, position_ids, past=None, use_cache=False, fill_cache=False):
        xp = self.xp
        if past is None:
            past = {}
        for layer_idx in range(VLM_LAYERS):
            use_self = fill_cache or (SELF_ATTN_EVERY_N > 0 and layer_idx % SELF_ATTN_EVERY_N == 0)
            if use_self:
                att_outputs, past = self.forward_attn_layer(
                    inputs_embeds, layer_idx, position_ids, attention_mask, past=past, fill_cache=fill_cache
                )
            else:
                att_outputs, past = self.forward_cross_attn_layer(
                    inputs_embeds, layer_idx, position_ids, attention_mask, past
                )
            next_embeds = []
            start = 0
            for i, hidden in enumerate(inputs_embeds):
                if hidden is None:
                    next_embeds.append(None)
                    continue
                prefix = self._layer_pair(layer_idx)[i]
                att = att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                end = start + hidden.shape[1]
                piece = att[:, start:end] if len(att_outputs) == 1 else att
                out = linear(piece, self._w(prefix + "self_attn.o_proj.weight"), None, xp) + hidden
                residual = out
                out = rms_norm(out, self._w(prefix + "post_attention_layernorm.weight"), xp)
                if i == 0:
                    out = mlp_swiglu(
                        out,
                        self._w(prefix + "mlp.gate_proj.weight"),
                        self._w(prefix + "mlp.up_proj.weight"),
                        self._w(prefix + "mlp.down_proj.weight"),
                        xp,
                    )
                else:
                    out = mlp_swiglu(
                        out,
                        self._w(prefix + "mlp.gate_proj.weight"),
                        self._w(prefix + "mlp.up_proj.weight"),
                        self._w(prefix + "mlp.down_proj.weight"),
                        xp,
                    )
                next_embeds.append(out + residual)
                start = end if len(att_outputs) == 1 else 0
            inputs_embeds = next_embeds
        outs = []
        norms = (
            "model.vlm_with_expert.vlm.model.text_model.norm.weight",
            "model.vlm_with_expert.lm_expert.norm.weight",
        )
        for hidden, key in zip(inputs_embeds, norms):
            outs.append(None if hidden is None else rms_norm(hidden, self._w(key), xp))
        return outs, past

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
        xp = self.xp
        embs, pads, att = [], [], []
        for img, mask in zip(images, img_masks):
            img_emb = self.embed_image(img)
            img_emb = img_emb * math.sqrt(img_emb.shape[-1])
            bsz, n = img_emb.shape[:2]
            mask = np.asarray(mask)
            if mask.ndim == 1:
                mask = np.broadcast_to(mask[:, None], (bsz, n))
            else:
                mask = np.broadcast_to(mask[:, None], (bsz, n))
            embs.append(img_emb)
            pads.append(xp.asarray(mask.astype(bool)))
            att.extend([0] * n)
        lang_emb = self.embed_language(lang_tokens) * math.sqrt(VLM_HIDDEN)
        embs.append(lang_emb)
        pads.append(xp.asarray(np.asarray(lang_masks).astype(bool)))
        att.extend([0] * lang_emb.shape[1])
        state_emb = linear(xp.asarray(state, dtype=np.float32), self._w("model.state_proj.weight"), self._w("model.state_proj.bias"), xp)
        if state_emb.ndim == 2:
            state_emb = xp.expand_dims(state_emb, 1)
        embs.append(state_emb)
        bsz = state_emb.shape[0]
        pads.append(xp.ones((bsz, state_emb.shape[1]), dtype=np.bool_))
        att.extend([1] * state_emb.shape[1])
        embs = xp.concat(embs, axis=1)
        pads = xp.concat(pads, axis=1)
        att = xp.asarray(np.broadcast_to(np.asarray(att, dtype=bool)[None, :], (bsz, len(att))))
        return embs, pads, att

    def embed_suffix(self, noisy_actions, timestep):
        xp = self.xp
        action_emb = linear(noisy_actions, self._w("model.action_in_proj.weight"), self._w("model.action_in_proj.bias"), xp)
        time_emb = create_sinusoidal_pos_embedding(timestep, EXPERT_HIDDEN, MIN_PERIOD, MAX_PERIOD, xp)
        time_emb = xp.astype(time_emb, action_emb.dtype) if hasattr(action_emb, "dtype") else time_emb
        time_emb = xp.broadcast_to(xp.expand_dims(time_emb, 1), action_emb.shape)
        fused = xp.concat([action_emb, time_emb], axis=2)
        fused = linear(fused, self._w("model.action_time_mlp_in.weight"), self._w("model.action_time_mlp_in.bias"), xp)
        fused = xp.silu(fused)
        fused = linear(fused, self._w("model.action_time_mlp_out.weight"), self._w("model.action_time_mlp_out.bias"), xp)
        bsz, steps = fused.shape[:2]
        pads = xp.ones((bsz, steps), dtype=np.bool_)
        att = xp.asarray(np.broadcast_to(np.ones((1, steps), dtype=bool), (bsz, steps)))
        return fused, pads, att

    def denoise_step(self, x_t, prefix_pad_masks, past, timestep):
        suffix, suffix_pad, suffix_att = self.embed_suffix(x_t, timestep)
        bsz = prefix_pad_masks.shape[0]
        suffix_len = suffix_pad.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d = xp_broadcast_prefix(self.xp, prefix_pad_masks, bsz, suffix_len, prefix_len)
        suffix_2d = make_att_2d_masks(suffix_pad, suffix_att, self.xp)
        full = self.xp.concat([prefix_pad_2d, suffix_2d], axis=2)
        prefix_offsets = self.xp.expand_dims(self.xp.sum(self.xp.astype(prefix_pad_masks, np.float32), axis=-1), 1)
        position_ids = prefix_offsets + self.xp.cumsum(self.xp.astype(suffix_pad, np.float32), axis=1) - 1
        outs, _ = self.transformer(
            [None, suffix],
            full,
            position_ids,
            past=past,
            use_cache=True,
            fill_cache=False,
        )
        suffix_out = self.xp.astype(outs[1][:, -CHUNK_SIZE :], np.float32)
        return linear(suffix_out, self._w("model.action_out_proj.weight"), self._w("model.action_out_proj.bias"), self.xp)

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
        xp = self.xp
        state = xp.asarray(state, dtype=np.float32)
        bsz = state.shape[0]
        if noise is None:
            noise = np.random.normal(0.0, 1.0, size=(bsz, CHUNK_SIZE, MAX_ACTION_DIM)).astype(np.float32)
        noise = xp.asarray(noise, dtype=np.float32)
        prefix, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
        prefix_2d = make_att_2d_masks(prefix_pad, prefix_att, xp)
        prefix_pos = xp.cumsum(xp.astype(prefix_pad, np.float32), axis=1) - 1
        _, past = self.transformer([prefix, None], prefix_2d, prefix_pos, past={}, use_cache=True, fill_cache=True)
        dt = -1.0 / NUM_STEPS
        x_t = noise
        for step in range(NUM_STEPS):
            time = 1.0 + step * dt
            time_tensor = xp.asarray(np.full((bsz,), time, dtype=np.float32))
            v_t = self.denoise_step(x_t, prefix_pad, past, time_tensor)
            x_t = x_t + dt * v_t
        return x_t

    def prepare_images(self, batch):
        images, masks = [], []
        for key in (OBS_IMAGE, OBS_IMAGE2):
            if key not in batch:
                continue
            img = np.asarray(batch[key], dtype=np.float32)
            if img.ndim == 5:
                img = img[:, -1]
            if img.shape[1] not in (1, 3) and img.shape[-1] in (1, 3):
                img = np.transpose(img, (0, 3, 1, 2))
            img = resize_with_pad(img, VISION_IMAGE, VISION_IMAGE, pad_value=0.0)
            img = img * 2.0 - 1.0
            images.append(self.xp.asarray(img, dtype=np.float32))
            masks.append(np.ones((img.shape[0],), dtype=bool))
        if not images:
            raise ValueError("batch is missing camera observations")
        return images, masks

    def prepare_state(self, batch):
        state = np.asarray(batch[OBS_STATE], dtype=np.float32)
        if state.ndim > 2:
            state = state[:, -1]
        return pad_vector(state, MAX_STATE_DIM)

    def tokenize_tasks(self, tasks):
        if isinstance(tasks, str):
            tasks = [tasks]
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]
        encoded = self.tokenizer(
            tasks,
            padding="max_length",
            truncation=True,
            max_length=TOKENIZER_MAX_LENGTH,
            return_tensors=None,
        )
        tokens = np.asarray(encoded["input_ids"], dtype=np.int64)
        mask = np.asarray(encoded["attention_mask"], dtype=bool)
        if tokens.shape[-1] > TOKENIZER_MAX_LENGTH:
            tokens = tokens[:, :TOKENIZER_MAX_LENGTH]
            mask = mask[:, :TOKENIZER_MAX_LENGTH]
        elif tokens.shape[-1] < TOKENIZER_MAX_LENGTH:
            pad = TOKENIZER_MAX_LENGTH - tokens.shape[-1]
            tokens = np.pad(tokens, ((0, 0), (0, pad)))
            mask = np.pad(mask, ((0, 0), (0, pad)))
        return tokens, mask

    def normalize_obs(self, batch):
        out = dict(batch)
        if self.stats:
            mean = np.asarray(self.stats["observation.state.mean"], dtype=np.float32)
            std = np.asarray(self.stats["observation.state.std"], dtype=np.float32)
            state = np.asarray(out[OBS_STATE], dtype=np.float32)
            out[OBS_STATE] = (state - mean) / np.maximum(std, NORM_EPS)
        return out

    def unnormalize_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if not self.stats:
            return action
        mean = np.asarray(self.stats["action.mean"], dtype=np.float32)
        std = np.asarray(self.stats["action.std"], dtype=np.float32)
        return action * np.maximum(std, NORM_EPS) + mean

    def select_action(self, batch, noise=None):
        batch = self.normalize_obs(batch)
        if not self.action_queue:
            images, masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            if OBS_LANG_TOKENS in batch:
                tokens = np.asarray(batch[OBS_LANG_TOKENS])
                lang_masks = np.asarray(batch[OBS_LANG_MASK]).astype(bool)
            else:
                tasks = batch.get("task") or batch.get("observation.language") or ""
                tokens, lang_masks = self.tokenize_tasks(tasks)
            actions = self.sample_actions(
                images,
                masks,
                self.xp.asarray(tokens),
                lang_masks,
                state,
                noise=noise,
            )
            actions = self.xp.to_numpy(actions)[:, :, :7]
            for step in range(min(self.n_action_steps, actions.shape[1])):
                self.action_queue.append(actions[:, step, :])
        action = self.unnormalize_action(self.action_queue.popleft())
        return action


def xp_broadcast_prefix(xp, prefix_pad, batch, suffix_len, prefix_len):
    pad = xp.astype(prefix_pad, np.bool_) if hasattr(xp, "astype") else prefix_pad.astype(bool)
    pad = xp.expand_dims(pad, 1)
    return xp.broadcast_to(pad, (batch, suffix_len, prefix_len))


def load_stats(policy_dir):
    path = Path(policy_dir) / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    if not path.is_file():
        return {}
    return load_safetensors(path)


def load_config(policy_dir):
    path = Path(policy_dir) / "config.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def load_policy(policy_id=DEFAULT_POLICY, backend="auto", tokenizer=None, cache_dir=None):
    root = snapshot_or_path(policy_id, cache_dir=cache_dir)
    weights = load_safetensors(root / "model.safetensors")
    xp = resolve_backend(backend)
    cfg = load_config(root)
    policy = SmolVLAMLX(
        weights,
        xp,
        tokenizer=tokenizer or load_tokenizer(cfg.get("vlm_model_name", DEFAULT_VLM)),
        n_action_steps=int(cfg.get("n_action_steps", 1)),
    )
    policy.stats = load_stats(root)
    policy.policy_id = policy_id
    policy.policy_dir = root
    policy.backend_name = xp.name
    policy.config = cfg
    return policy


def observation_from_lerobot(obs):
    """Translate LeRobot preprocess_observation output into SmolVLA keys."""
    batch = {}
    if "observation.images.image" in obs:
        batch[OBS_IMAGE] = _as_numpy(obs["observation.images.image"])
        if "observation.images.image2" in obs:
            batch[OBS_IMAGE2] = _as_numpy(obs["observation.images.image2"])
    elif isinstance(obs, dict) and "pixels" in obs:
        pixels = obs["pixels"]
        keys = list(pixels)
        batch[OBS_IMAGE] = _maybe_hwc_to_chw(_as_numpy(pixels[keys[0]]))
        if len(keys) > 1:
            batch[OBS_IMAGE2] = _maybe_hwc_to_chw(_as_numpy(pixels[keys[1]]))
    if "observation.state" in obs:
        batch[OBS_STATE] = _as_numpy(obs["observation.state"])
    elif "robot_state" in obs:
        state = obs["robot_state"]
        if isinstance(state, dict):
            batch[OBS_STATE] = np.concatenate(
                [_as_numpy(state[k]).reshape(len(_as_numpy(state[k])), -1) for k in state],
                axis=-1,
            )
        else:
            batch[OBS_STATE] = _as_numpy(state)
    if "task" in obs:
        batch["task"] = obs["task"]
    if OBS_LANG_TOKENS in obs:
        batch[OBS_LANG_TOKENS] = _as_numpy(obs[OBS_LANG_TOKENS])
        batch[OBS_LANG_MASK] = _as_numpy(obs[OBS_LANG_MASK])
    return batch


def _maybe_hwc_to_chw(img):
    img = np.asarray(img)
    if img.ndim == 4 and img.shape[-1] in (1, 3) and img.shape[1] not in (1, 3):
        img = np.transpose(img, (0, 3, 1, 2))
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def _as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def synthetic_weights(seed=0, layers=2, vision_layers=1):
    """Tiny weight pack that exercises every numeric path without the 1.2 GB dump."""
    rng = np.random.default_rng(seed)
    w = {}

    def rand(*shape, dtype=np.float32, scale=0.001):
        return (rng.standard_normal(shape) * scale).astype(dtype)

    w["model.state_proj.weight"] = rand(VLM_HIDDEN, MAX_STATE_DIM)
    w["model.state_proj.bias"] = np.zeros((VLM_HIDDEN,), np.float32)
    w["model.action_in_proj.weight"] = rand(EXPERT_HIDDEN, MAX_ACTION_DIM)
    w["model.action_in_proj.bias"] = np.zeros((EXPERT_HIDDEN,), np.float32)
    w["model.action_out_proj.weight"] = rand(MAX_ACTION_DIM, EXPERT_HIDDEN)
    w["model.action_out_proj.bias"] = np.zeros((MAX_ACTION_DIM,), np.float32)
    w["model.action_time_mlp_in.weight"] = rand(EXPERT_HIDDEN, EXPERT_HIDDEN * 2)
    w["model.action_time_mlp_in.bias"] = np.zeros((EXPERT_HIDDEN,), np.float32)
    w["model.action_time_mlp_out.weight"] = rand(EXPERT_HIDDEN, EXPERT_HIDDEN)
    w["model.action_time_mlp_out.bias"] = np.zeros((EXPERT_HIDDEN,), np.float32)
    w["model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight"] = rand(VOCAB_SIZE, VLM_HIDDEN, scale=0.001)
    w["model.vlm_with_expert.vlm.model.text_model.norm.weight"] = np.ones((VLM_HIDDEN,), np.float32)
    w["model.vlm_with_expert.lm_expert.norm.weight"] = np.ones((EXPERT_HIDDEN,), np.float32)
    w["model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight"] = rand(
        VISION_HIDDEN, 3, VISION_PATCH, VISION_PATCH, scale=0.01
    )
    w["model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.bias"] = np.zeros((VISION_HIDDEN,), np.float32)
    w["model.vlm_with_expert.vlm.model.vision_model.embeddings.position_embedding.weight"] = rand(1024, VISION_HIDDEN, scale=0.01)
    w["model.vlm_with_expert.vlm.model.vision_model.post_layernorm.weight"] = np.ones((VISION_HIDDEN,), np.float32)
    w["model.vlm_with_expert.vlm.model.vision_model.post_layernorm.bias"] = np.zeros((VISION_HIDDEN,), np.float32)
    w["model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight"] = rand(VLM_HIDDEN, VISION_HIDDEN * SCALE_FACTOR ** 2)
    for i in range(vision_layers):
        p = f"model.vlm_with_expert.vlm.model.vision_model.encoder.layers.{i}."
        scale = 0.02
        for name, shape in {
            "layer_norm1.weight": (VISION_HIDDEN,),
            "layer_norm1.bias": (VISION_HIDDEN,),
            "layer_norm2.weight": (VISION_HIDDEN,),
            "layer_norm2.bias": (VISION_HIDDEN,),
            "self_attn.q_proj.weight": (VISION_HIDDEN, VISION_HIDDEN),
            "self_attn.q_proj.bias": (VISION_HIDDEN,),
            "self_attn.k_proj.weight": (VISION_HIDDEN, VISION_HIDDEN),
            "self_attn.k_proj.bias": (VISION_HIDDEN,),
            "self_attn.v_proj.weight": (VISION_HIDDEN, VISION_HIDDEN),
            "self_attn.v_proj.bias": (VISION_HIDDEN,),
            "self_attn.out_proj.weight": (VISION_HIDDEN, VISION_HIDDEN),
            "self_attn.out_proj.bias": (VISION_HIDDEN,),
            "mlp.fc1.weight": (VISION_MLP, VISION_HIDDEN),
            "mlp.fc1.bias": (VISION_MLP,),
            "mlp.fc2.weight": (VISION_HIDDEN, VISION_MLP),
            "mlp.fc2.bias": (VISION_HIDDEN,),
        }.items():
            if name.endswith("weight") and "norm" in name:
                w[p + name] = np.ones(shape, np.float32)
            elif name.endswith("bias"):
                w[p + name] = np.zeros(shape, np.float32)
            else:
                w[p + name] = rand(*shape, scale=scale)
    for i in range(layers):
        scale = 0.002
        vp = f"model.vlm_with_expert.vlm.model.text_model.layers.{i}."
        ep = f"model.vlm_with_expert.lm_expert.layers.{i}."
        w[vp + "input_layernorm.weight"] = np.ones((VLM_HIDDEN,), np.float32)
        w[vp + "post_attention_layernorm.weight"] = np.ones((VLM_HIDDEN,), np.float32)
        w[vp + "self_attn.q_proj.weight"] = rand(VLM_HIDDEN, VLM_HIDDEN, scale=scale)
        w[vp + "self_attn.k_proj.weight"] = rand(VLM_KV_HEADS * VLM_HEAD_DIM, VLM_HIDDEN, scale=scale)
        w[vp + "self_attn.v_proj.weight"] = rand(VLM_KV_HEADS * VLM_HEAD_DIM, VLM_HIDDEN, scale=scale)
        w[vp + "self_attn.o_proj.weight"] = rand(VLM_HIDDEN, VLM_HIDDEN, scale=scale)
        w[vp + "mlp.gate_proj.weight"] = rand(VLM_INTERMEDIATE, VLM_HIDDEN, scale=scale)
        w[vp + "mlp.up_proj.weight"] = rand(VLM_INTERMEDIATE, VLM_HIDDEN, scale=scale)
        w[vp + "mlp.down_proj.weight"] = rand(VLM_HIDDEN, VLM_INTERMEDIATE, scale=scale)
        w[ep + "input_layernorm.weight"] = np.ones((EXPERT_HIDDEN,), np.float32)
        w[ep + "post_attention_layernorm.weight"] = np.ones((EXPERT_HIDDEN,), np.float32)
        w[ep + "self_attn.q_proj.weight"] = rand(EXPERT_HEADS * VLM_HEAD_DIM, EXPERT_HIDDEN, scale=scale)
        w[ep + "self_attn.o_proj.weight"] = rand(EXPERT_HIDDEN, EXPERT_HEADS * VLM_HEAD_DIM, scale=scale)
        if i % SELF_ATTN_EVERY_N == 0:
            w[ep + "self_attn.k_proj.weight"] = rand(EXPERT_KV_HEADS * VLM_HEAD_DIM, EXPERT_HIDDEN, scale=scale)
            w[ep + "self_attn.v_proj.weight"] = rand(EXPERT_KV_HEADS * VLM_HEAD_DIM, EXPERT_HIDDEN, scale=scale)
        else:
            w[ep + "self_attn.k_proj.weight"] = rand(EXPERT_KV_HEADS * VLM_HEAD_DIM, VLM_KV_HEADS * VLM_HEAD_DIM, scale=scale)
            w[ep + "self_attn.v_proj.weight"] = rand(EXPERT_KV_HEADS * VLM_HEAD_DIM, VLM_KV_HEADS * VLM_HEAD_DIM, scale=scale)
        w[ep + "mlp.gate_proj.weight"] = rand(EXPERT_INTERMEDIATE, EXPERT_HIDDEN, scale=scale)
        w[ep + "mlp.up_proj.weight"] = rand(EXPERT_INTERMEDIATE, EXPERT_HIDDEN, scale=scale)
        w[ep + "mlp.down_proj.weight"] = rand(EXPERT_HIDDEN, EXPERT_INTERMEDIATE, scale=scale)
    return w


class TinySmolVLAMLX(SmolVLAMLX):
    """Self-test model: same graph, fewer layers, 64px vision."""

    def embed_image(self, images):
        # Downscale so the synthetic graph stays cheap. Do not use the 12288-wide
        # connector here: random init overflows float32 before the transformer.
        img = self.xp.to_numpy(images)
        if img.ndim != 4:
            raise ValueError(img.shape)
        pooled = img.mean(axis=(2, 3))
        tokens = np.zeros((pooled.shape[0], 1, VLM_HIDDEN), dtype=np.float32)
        tokens[:, 0, : pooled.shape[-1]] = pooled
        return self.xp.asarray(tokens, dtype=np.float32)

    def transformer(self, inputs_embeds, attention_mask, position_ids, past=None, use_cache=False, fill_cache=False):
        # Only the first two layers carry random weights in the synthetic pack.
        saved = VLM_LAYERS
        globals_override = globals()
        try:
            # Local loop copy with 2 layers
            xp = self.xp
            if past is None:
                past = {}
            for layer_idx in range(2):
                use_self = fill_cache or (SELF_ATTN_EVERY_N > 0 and layer_idx % SELF_ATTN_EVERY_N == 0)
                if use_self:
                    att_outputs, past = self.forward_attn_layer(
                        inputs_embeds, layer_idx, position_ids, attention_mask, past=past, fill_cache=fill_cache
                    )
                else:
                    att_outputs, past = self.forward_cross_attn_layer(
                        inputs_embeds, layer_idx, position_ids, attention_mask, past
                    )
                next_embeds = []
                start = 0
                for i, hidden in enumerate(inputs_embeds):
                    if hidden is None:
                        next_embeds.append(None)
                        continue
                    prefix = self._layer_pair(layer_idx)[i]
                    att = att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                    piece = att[:, start : start + hidden.shape[1]] if len(att_outputs) == 1 else att
                    out = linear(piece, self._w(prefix + "self_attn.o_proj.weight"), None, xp) + hidden
                    residual = out
                    out = rms_norm(out, self._w(prefix + "post_attention_layernorm.weight"), xp)
                    out = mlp_swiglu(
                        out,
                        self._w(prefix + "mlp.gate_proj.weight"),
                        self._w(prefix + "mlp.up_proj.weight"),
                        self._w(prefix + "mlp.down_proj.weight"),
                        xp,
                    )
                    next_embeds.append(out + residual)
                    start = start + hidden.shape[1] if len(att_outputs) == 1 else 0
                inputs_embeds = next_embeds
            outs = []
            norms = (
                "model.vlm_with_expert.vlm.model.text_model.norm.weight",
                "model.vlm_with_expert.lm_expert.norm.weight",
            )
            for hidden, key in zip(inputs_embeds, norms):
                outs.append(None if hidden is None else rms_norm(hidden, self._w(key), xp))
            return outs, past
        finally:
            globals_override["VLM_LAYERS"] = saved


def make_tiny_policy(backend="numpy"):
    xp = resolve_backend(backend)
    policy = TinySmolVLAMLX(synthetic_weights(), xp, tokenizer=HashTokenizer(), n_action_steps=1)
    policy.stats = {
        "observation.state.mean": np.zeros((8,), np.float32),
        "observation.state.std": np.ones((8,), np.float32),
        "action.mean": np.zeros((7,), np.float32),
        "action.std": np.ones((7,), np.float32),
    }
    policy.backend_name = xp.name
    policy.policy_id = "synthetic-smolvla"
    return policy


def run_selftest():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    img = np.zeros((1, 3, 40, 80), dtype=np.float32)
    img[:, :, 10:30, 20:60] = 1.0
    padded = resize_with_pad(img, 64, 64, pad_value=0.0)
    check(padded.shape == (1, 3, 64, 64), f"resize_with_pad shape {padded.shape}")
    check(padded[0, 0, 0, 0] == 0.0, "padding should land on the top/left")
    check(padded.max() == 1.0, "foreground should survive resize")

    xp = ArrayBackend()
    time = xp.array([0.25, 0.75], dtype=np.float32)
    pe = create_sinusoidal_pos_embedding(time, 8, MIN_PERIOD, MAX_PERIOD, xp)
    check(pe.shape == (2, 8), f"time embed shape {pe.shape}")
    check(np.isfinite(pe).all(), "time embed must be finite")

    pad = np.array([[1, 1, 1, 0]], dtype=bool)
    att = np.array([[0, 0, 1, 1]], dtype=bool)
    mask = make_att_2d_masks(pad, att, xp)
    check(mask.shape == (1, 4, 4), f"attn mask {mask.shape}")
    check(bool(mask[0, 2, 0]) and not bool(mask[0, 0, 2]), "prefix tokens must not attend into suffix")

    tokens = np.zeros((1, 4, 8), dtype=np.float32)
    shuffled = pixel_shuffle(xp.array(tokens), 2, xp)
    check(shuffled.shape == (1, 1, 32), f"pixel_shuffle {shuffled.shape}")

    policy = make_tiny_policy("numpy")
    batch = {
        OBS_IMAGE: np.random.default_rng(0).random((2, 3, 32, 32)).astype(np.float32),
        OBS_IMAGE2: np.random.default_rng(1).random((2, 3, 32, 32)).astype(np.float32),
        OBS_STATE: np.zeros((2, 8), dtype=np.float32),
        "task": ["pick the bowl\n", "pick the bowl\n"],
    }
    noise = np.zeros((2, CHUNK_SIZE, MAX_ACTION_DIM), dtype=np.float32)
    action = policy.select_action(batch, noise=noise)
    check(action.shape == (2, 7), f"select_action shape {action.shape}")
    check(np.isfinite(action).all(), "select_action produced non-finite values")
    policy.reset()
    again = policy.select_action(batch, noise=noise)
    check(np.allclose(action, again, atol=1e-5), "deterministic noise must replay")

    # MEAN_STD invert
    policy.stats["action.mean"] = np.arange(7, dtype=np.float32)
    policy.stats["action.std"] = np.full((7,), 2.0, np.float32)
    raw = np.ones((1, 7), dtype=np.float32)
    check(np.allclose(policy.unnormalize_action(raw), np.arange(7) + 2.0), "unnormalize failed")

    # Backend resolution: numpy always works; mlx optional.
    check(resolve_backend("numpy").name == "numpy", "numpy backend missing")
    try:
        mlx_backend = resolve_backend("mlx")
        check(mlx_backend.name == "mlx", "mlx backend did not identify itself")
    except ImportError:
        pass

    if failures:
        raise AssertionError("mlx harness selftest failed:\n- " + "\n- ".join(failures))
    print("MLX HARNESS SELFTEST PASSED")
    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ProbeArch MLX SmolVLA harness")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--backend", default="auto", choices=("auto", "mlx", "numpy"))
    parser.add_argument("--probe", action="store_true", help="load weights and run one dummy select_action")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(run_selftest())
    if args.probe:
        policy = load_policy(args.policy, backend=args.backend)
        batch = {
            OBS_IMAGE: np.zeros((1, 3, 256, 256), dtype=np.float32),
            OBS_IMAGE2: np.zeros((1, 3, 256, 256), dtype=np.float32),
            OBS_STATE: np.zeros((1, 8), dtype=np.float32),
            "task": "pick up the black bowl on the ramekin and place it on the plate\n",
        }
        action = policy.select_action(batch)
        print(
            json.dumps(
                {
                    "harness": HARNESS_NAME,
                    "backend": policy.backend_name,
                    "policy": args.policy,
                    "action_shape": list(action.shape),
                    "finite": bool(np.isfinite(action).all()),
                }
            )
        )
        return
    parser.print_help()


if __name__ == "__main__":
    main()
