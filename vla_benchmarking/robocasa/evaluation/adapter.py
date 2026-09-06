"""RoboCasa-to-controller observation and PandaOmron action contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .arrow import render_bbox_center_arrow
from ..environment.runtime import PANDA_OMRON_ACTION_LAYOUT, PandaOmronActionLayout
from .task_manifest import TaskSpec


ROBOCASA_CAMERA = "robot0_agentview_left"
ROBOCASA_RESOLUTION = 256
CONTROLLER_ACTION_DIM = 7
PANDA_OMRON_ACTION_DIM = 12


@dataclass(frozen=True)
class RoboCasaFrame:
    """One aligned RoboCasa observation in the controller's image contract."""

    capture: Any
    camera_name: str
    resolution: int
    bboxes: Mapping[str, Sequence[float]]
    task_name: str
    source_key: str
    destination_key: str

    @property
    def rgb(self) -> np.ndarray:
        return np.asarray(self.capture.rgb)

    @property
    def metric_depth_m(self) -> np.ndarray:
        return np.asarray(self.capture.metric_depth)


class RoboCasaObservationAdapter:
    """Capture RoboCasa RGB-D while keeping simulator state input-side only."""

    def __init__(
        self,
        env: Any,
        *,
        camera_name: str = ROBOCASA_CAMERA,
        resolution: int = ROBOCASA_RESOLUTION,
    ) -> None:
        if not camera_name:
            raise ValueError("camera_name must be non-empty")
        if int(resolution) <= 0:
            raise ValueError("resolution must be positive")
        self.env = env
        self.camera_name = str(camera_name)
        self.resolution = int(resolution)

    def capture(
        self,
        *,
        bboxes: Mapping[str, Sequence[float]],
        task: TaskSpec,
        source_key: str | None = None,
    ) -> RoboCasaFrame:
        """Capture a synchronized frame through the RoboCasa-local seam."""

        if task.source_key is None and source_key is None:
            raise ValueError(f"task {task.name} requires a selected source key")
        selected_source = str(source_key or task.source_key)
        if selected_source not in bboxes or task.destination_key not in bboxes:
            raise KeyError(
                f"missing projected bbox for {selected_source!r} or "
                f"{task.destination_key!r}"
            )
        try:
            # Import lazily to keep the package importable without MuJoCo.  The
            # live helper owns RoboCasa's one-time vertical flip and depth
            # conversion; this adapter must not reimplement that contract.
            from .live import _capture
            capture = _capture(self.env, resolution=self.resolution)
        except ImportError as exc:  # pragma: no cover - dependency-free CLI path
            raise RuntimeError(
                "the RoboCasa RGB-D capture seam is unavailable; install RoboCasa "
                "and its robosuite runtime before executing"
            ) from exc
        return RoboCasaFrame(
            capture=capture,
            camera_name=self.camera_name,
            resolution=self.resolution,
            bboxes=dict(bboxes),
            task_name=task.name,
            source_key=selected_source,
            destination_key=task.destination_key,
        )

    def render_arrow(self, frame: RoboCasaFrame) -> tuple[np.ndarray, dict[str, Any]]:
        arrow, audit = render_bbox_center_arrow(
            frame.rgb,
            frame.bboxes,
            source=frame.source_key,
            destination=frame.destination_key,
        )
        return arrow, {**audit, "camera_name": frame.camera_name}


class PandaOmronActionAdapter:
    """Embed the canonical 7D arm/gripper command without changing it."""

    def __init__(self, *, layout: PandaOmronActionLayout | None = None) -> None:
        self.layout = layout or PANDA_OMRON_ACTION_LAYOUT
        if self.layout.dimension != PANDA_OMRON_ACTION_DIM:
            raise ValueError("unexpected PandaOmron composite action dimension")

    @classmethod
    def from_env(cls, env: Any) -> "PandaOmronActionAdapter":
        """Accept only the currently verified 12D PandaOmron layout.

        RoboCasa's Gym wrapper exposes a dict space; the legacy robosuite
        object exposes a flat 12D space.  Both are checked when available so a
        changed controller split cannot silently receive this adapter.
        """

        action_dim = getattr(env, "action_dim", None)
        if action_dim is None:
            action_space = getattr(env, "action_space", None)
            action_dim = getattr(action_space, "shape", (None,))[0]
        if action_dim is not None and int(action_dim) != PANDA_OMRON_ACTION_DIM:
            raise ValueError(f"expected 12D PandaOmron action, got {action_dim!r}")
        action_space = getattr(env, "action_space", None)
        if action_space is not None:
            expected = {
                "action.end_effector_position": 3,
                "action.end_effector_rotation": 3,
                "action.gripper_close": 1,
                "action.base_motion": 4,
                "action.control_mode": 1,
            }
            for key, size in expected.items():
                try:
                    shape = tuple(int(item) for item in action_space[key].shape)
                except (KeyError, TypeError, AttributeError):
                    continue
                if shape != (size,):
                    raise ValueError(f"unexpected PandaOmron action field {key}: {shape}")
        return cls(layout=PANDA_OMRON_ACTION_LAYOUT)

    def pack(self, controller_action: Sequence[float], *, control_mode: float = 0.0) -> np.ndarray:
        values = np.asarray(controller_action, dtype=np.float64).reshape(-1)
        if values.shape != (CONTROLLER_ACTION_DIM,) or not np.isfinite(values).all():
            raise ValueError("controller_action must contain seven finite values")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("controller_action must be normalized to [-1, 1]")
        if not np.isfinite(control_mode):
            raise ValueError("control_mode must be finite")
        result = np.zeros(self.layout.dimension, dtype=np.float64)
        result[self.layout.arm_start : self.layout.arm_stop] = values[: self.layout.arm_dimension]
        result[self.layout.gripper_index] = values[6]
        result[self.layout.control_mode_index] = float(control_mode)
        return result

    def audit(self) -> dict[str, Any]:
        return {
            "dimension": self.layout.dimension,
            "arm_indices": list(range(self.layout.arm_start, self.layout.arm_stop)),
            "gripper_index": self.layout.gripper_index,
            "base_indices": list(range(self.layout.base_start, self.layout.base_stop)),
            "torso_indices": [self.layout.torso_index],
            "control_mode_index": self.layout.control_mode_index,
            "base_policy": "zero",
            "torso_policy": "zero",
        }


__all__ = [
    "CONTROLLER_ACTION_DIM",
    "PANDA_OMRON_ACTION_DIM",
    "PandaOmronActionAdapter",
    "PandaOmronActionLayout",
    "ROBOCASA_CAMERA",
    "ROBOCASA_RESOLUTION",
    "RoboCasaFrame",
    "RoboCasaObservationAdapter",
]
