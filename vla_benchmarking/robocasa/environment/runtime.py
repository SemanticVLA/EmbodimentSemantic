"""Pure runtime contracts for the official RoboCasa PandaOmron interface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from ..shared.config import (
    CAMERA,
    DEFAULT_CONTROL_FREQUENCY,
    DEFAULT_ROBOT,
    target_env_kwargs,
)


@dataclass(frozen=True)
class CameraObservationContract:
    """Keys and conventions expected from one RoboCasa RGB-D observation."""

    camera_name: str = CAMERA.name
    # RoboCasa's Gym wrapper exposes camera observations under the ``video.``
    # namespace, not the robosuite ``*_image`` spelling.
    rgb_key: str = f"video.{CAMERA.name}"
    depth_key: str = f"video.{CAMERA.name}_depth"
    width: int = CAMERA.width
    height: int = CAMERA.height
    depth_encoding: str = CAMERA.depth_encoding
    image_frame: str = CAMERA.image_frame


@dataclass(frozen=True)
class PandaOmronActionLayout:
    """Raw 12D PandaOmron action layout used by RoboCasa's Gym wrapper.

    The canonical arrow controller emits six arm values followed by one
    gripper value.  Only those seven values are copied; mobile base and torso
    commands remain zero for this fixed-arm transfer test.
    """

    dimension: int = 12
    arm_start: int = 0
    arm_stop: int = 6
    gripper_index: int = 6
    base_start: int = 7
    base_stop: int = 10
    torso_index: int = 10
    control_mode_index: int = 11
    control_mode_value: float = 0.0
    robot: str = DEFAULT_ROBOT
    control_frequency: int = DEFAULT_CONTROL_FREQUENCY

    @property
    def arm_dimension(self) -> int:
        return self.arm_stop - self.arm_start

    @property
    def controller_dimension(self) -> int:
        return self.arm_dimension + 1

    def validate(self, action: Sequence[float]) -> None:
        if len(action) != self.dimension:
            raise ValueError(f"expected {self.dimension}D PandaOmron action, got {len(action)}D")
        if any(not isinstance(value, (int, float)) for value in action):
            raise TypeError("PandaOmron action values must be numeric")


PANDA_OMRON_ACTION_LAYOUT = PandaOmronActionLayout()


def create_robocasa_env(task_name: str, *, seed: int | None = None, **overrides: Any) -> Any:
    """Create one official RoboCasa target-split environment lazily.

    RoboCasa and robosuite are intentionally imported only when a caller
    requests a live environment.  This keeps the LIBERO interpreter safe and
    makes missing-runtime failures explicit at the execution boundary.
    """

    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError("task_name must be a non-empty string")
    try:
        from robocasa.utils.env_utils import create_env
    except ImportError as exc:  # pragma: no cover - depends on isolated env
        raise RuntimeError(
            "RoboCasa is not installed in the active interpreter; use its isolated environment"
        ) from exc
    kwargs = target_env_kwargs(seed=seed)
    kwargs.update(overrides)
    # env_utils.create_env accepts the task class name as its first argument.
    return create_env(task_name, **kwargs)


def compose_panda_omron_action(
    controller_action: Sequence[float],
    *,
    layout: PandaOmronActionLayout = PANDA_OMRON_ACTION_LAYOUT,
) -> tuple[float, ...]:
    """Embed the canonical 7D arm/gripper command in the raw composite action."""

    if len(controller_action) != layout.controller_dimension:
        raise ValueError(
            f"expected {layout.controller_dimension}D controller action, "
            f"got {len(controller_action)}D"
        )
    if any(not isinstance(value, (int, float)) for value in controller_action):
        raise TypeError("controller action values must be numeric")
    if any(not math.isfinite(float(value)) for value in controller_action):
        raise ValueError("controller action values must be finite")
    action = [0.0] * layout.dimension
    action[layout.arm_start : layout.arm_stop] = [
        float(value) for value in controller_action[: layout.arm_dimension]
    ]
    action[layout.gripper_index] = float(controller_action[layout.arm_dimension])
    action[layout.control_mode_index] = float(layout.control_mode_value)
    layout.validate(action)
    return tuple(action)


def validate_action_layout(
    action: Sequence[float], *, layout: PandaOmronActionLayout = PANDA_OMRON_ACTION_LAYOUT
) -> None:
    """Validate a composed action and enforce parked base/torso fields."""

    layout.validate(action)
    if any(not math.isfinite(float(value)) for value in action):
        raise ValueError("PandaOmron action values must be finite")
    if any(action[index] != 0.0 for index in range(layout.base_start, layout.base_stop)):
        raise ValueError("PandaOmron base commands must remain zero in this experiment")
    if action[layout.torso_index] != 0.0:
        raise ValueError("PandaOmron torso command must remain zero in this experiment")


__all__ = [
    "CameraObservationContract",
    "PANDA_OMRON_ACTION_LAYOUT",
    "PandaOmronActionLayout",
    "compose_panda_omron_action",
    "create_robocasa_env",
    "validate_action_layout",
]
