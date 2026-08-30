"""Authoritative MuJoCo model-ID to name resolution.

MuJoCo's Python ``mj_id2name`` API is the source of truth.  Wrapper-provided
name tuples can omit unnamed objects and therefore shift every later index.
"""

from __future__ import annotations


def _decode(value, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _resolve(model, object_type_name: str, object_id: int, fallback_prefix: str) -> str:
    object_id = int(object_id)
    # robosuite exposes a binding wrapper at ``sim.model``; the MuJoCo API
    # requires its native ``mujoco.MjModel`` stored in ``._model``.
    native_model = getattr(model, "_model", model)
    count_attr = {"mjOBJ_GEOM": "ngeom", "mjOBJ_BODY": "nbody"}[object_type_name]
    count = int(getattr(native_model, count_attr))
    if object_id < 0 or object_id >= count:
        raise IndexError(f"{object_type_name} id {object_id} outside 0..{count - 1}")

    import mujoco

    object_type = getattr(mujoco.mjtObj, object_type_name)
    name = mujoco.mj_id2name(native_model, object_type, object_id)
    return _decode(name, f"{fallback_prefix}{object_id}")


def geom_name(model, geom_id: int) -> str:
    return _resolve(model, "mjOBJ_GEOM", geom_id, "geom")


def body_name(model, body_id: int) -> str:
    return _resolve(model, "mjOBJ_BODY", body_id, "body")
