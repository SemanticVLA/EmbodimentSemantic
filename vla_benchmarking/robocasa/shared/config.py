"""Frozen RoboCasa runtime configuration used by the PickPlace evaluation.

This module intentionally contains configuration only.  Environment creation is
performed by the RoboCasa evaluation adapter so importing the benchmark package
does not require MuJoCo or RoboCasa to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BENCHMARK_NAME = "robocasa_pick_place_21"

# Source revisions are recorded with every result.  RoboCasa's setup metadata
# requires robosuite >= 1.5.2 and its documentation currently recommends the
# robosuite master branch; this is the exact revision used by the initial port.
ROBOCASA_COMMIT = "4f8a2980def75a55dff96b990745b83540425f09"
ROBOSUITE_COMMIT = "5ce6643f3092639d08f7b0f90ed1c6a84f50552c"

DEFAULT_ROBOT = "PandaOmron"
DEFAULT_CAMERA_NAME = "robot0_agentview_left"
DEFAULT_CAMERA_NAMES = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
DEFAULT_CAMERA_WIDTH = 256
DEFAULT_CAMERA_HEIGHT = 256
DEFAULT_CONTROL_FREQUENCY = 20
DEFAULT_SPLIT = "target"


@dataclass(frozen=True)
class CameraConfig:
    """The single RGB-D camera used to produce controller arrows."""

    name: str = DEFAULT_CAMERA_NAME
    width: int = DEFAULT_CAMERA_WIDTH
    height: int = DEFAULT_CAMERA_HEIGHT
    depth: bool = True
    # RoboCasa/robosuite returns RGB and depth in its native image convention.
    # The adapter records the explicit post-flip convention before projection.
    image_frame: str = "post_flip_xy"
    # RoboCasa's camera depth is the robosuite normalized z-buffer encoding;
    # the adapter converts it to metric metres with get_real_depth_map.
    depth_encoding: str = "normalized"


CAMERA = CameraConfig()


@dataclass(frozen=True)
class TargetSplitConfig:
    """Official RoboCasa target split parameters."""

    split: str = DEFAULT_SPLIT
    obj_instance_split: str = "target"
    layout_and_style_ids: tuple[tuple[int, int], ...] = tuple(
        (value, value) for value in range(1, 11)
    )

    def as_env_kwargs(self) -> dict[str, Any]:
        """Return kwargs accepted by ``robocasa.utils.env_utils.create_env``."""

        return {
            "split": self.split,
            "obj_instance_split": self.obj_instance_split,
            "layout_and_style_ids": list(self.layout_and_style_ids),
            "layout_ids": None,
            "style_ids": None,
        }


TARGET_SPLIT = TargetSplitConfig()


def target_env_kwargs(
    *,
    camera: CameraConfig = CAMERA,
    robot: str = DEFAULT_ROBOT,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build deterministic, explicit kwargs for one target-split environment.

    No RoboCasa objects are constructed here.  Keeping this pure makes it safe
    for manifest/preflight tests and keeps dependency loading in the adapter.
    """

    kwargs = {
        "robots": robot,
        "camera_names": [camera.name],
        "camera_widths": camera.width,
        "camera_heights": camera.height,
        "camera_depths": camera.depth,
        "seed": seed,
        "render_onscreen": False,
    }
    kwargs.update(TARGET_SPLIT.as_env_kwargs())
    return kwargs


__all__ = [
    "BENCHMARK_NAME",
    "CAMERA",
    "CameraConfig",
    "DEFAULT_CAMERA_NAME",
    "DEFAULT_CAMERA_NAMES",
    "DEFAULT_CAMERA_HEIGHT",
    "DEFAULT_CAMERA_WIDTH",
    "DEFAULT_CONTROL_FREQUENCY",
    "DEFAULT_ROBOT",
    "DEFAULT_SPLIT",
    "ROBOSUITE_COMMIT",
    "ROBOCASA_COMMIT",
    "TARGET_SPLIT",
    "TargetSplitConfig",
    "target_env_kwargs",
]
