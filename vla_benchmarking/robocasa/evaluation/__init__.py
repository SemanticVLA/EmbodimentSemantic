"""Dependency-light RoboCasa evaluation seams.

Simulator and model imports are intentionally deferred until an episode is
requested.  This keeps manifest, prompt, geometry, and CLI preflight checks
usable before the RoboCasa simulator is installed.
"""

from .arrow import bbox_center, render_bbox_center_arrow
from .capture import CameraCalibration, CapturedRGBD, capture_robocasa_rgbd
from .live import RoboCasaControllerEnv, official_success, project_task_bboxes, run_live_cell
from .prompt import adapt_source_noun, canonical_prompt
from .task_manifest import PICK_PLACE_TASKS, TaskSpec, get_task

__all__ = [
    "PICK_PLACE_TASKS",
    "TaskSpec",
    "CameraCalibration",
    "CapturedRGBD",
    "adapt_source_noun",
    "bbox_center",
    "canonical_prompt",
    "capture_robocasa_rgbd",
    "get_task",
    "render_bbox_center_arrow",
    "RoboCasaControllerEnv",
    "official_success",
    "project_task_bboxes",
    "run_live_cell",
]
