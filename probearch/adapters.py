"""Small public interfaces for connecting policies and trajectory sources.

Adapters deliberately contain no safety logic. They normalize an external
policy/environment into the existing telemetry contract so the scorer remains
the single implementation of measurement rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence


class PolicyAdapter(Protocol):
    """Policy interface expected by a rollout adapter."""

    policy_id: str

    def reset(self, *, task: str, seed: int) -> None:
        ...

    def act(self, observation: dict[str, Any], instruction: str) -> Sequence[float]:
        ...


class EnvironmentAdapter(Protocol):
    """Environment interface for instrumented collection."""

    def reset(self, *, task_id: int, seed: int) -> dict[str, Any]:
        ...

    def step(self, action: Sequence[float]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        ...

    def telemetry(self) -> dict[str, Any]:
        ...


class TrajectorySource(Protocol):
    """Offline source consumed by a scorer without policy inference."""

    def episodes(self) -> Iterable[Path]:
        ...


@dataclass(frozen=True)
class AdapterContract:
    """Metadata that must be written into a run manifest."""

    schema_version: str
    policy_id: str
    action_dim: int
    image_size: tuple[int, int]
    control_mode: str
    camera_names: tuple[str, ...]
