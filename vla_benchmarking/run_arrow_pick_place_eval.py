#!/usr/bin/env python3
"""Arrow-only RGB-D pick/place MVP for LIBERO.

The runner deliberately keeps perception and motion separate.  The only data
passed to :mod:`arrow_controller` are the clean RGB frame, the one-arrow RGB
frame, aligned metric depth, and camera calibration.  LIBERO object poses,
meshes, bboxes, and evaluator state never cross that boundary.

Motion is opt-in.  ``python run_arrow_pick_place_eval.py`` captures and audits
one episode without calling ``step``; pass ``--execute-motion`` only after
reviewing the generated artifacts.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:  # Optional outside the LIBERO environment (all pure seams stay importable).
    from robosuite.utils import camera_utils
except ImportError:  # pragma: no cover - exercised by dependency-free tests
    camera_utils = None

try:
    from radomize_scenes import settle_physics as _settle_physics
except ImportError:  # pragma: no cover - dependency-free import guard
    _settle_physics = None

try:
    from arrow_controller import (
        build_bowl_waypoints,
        decode_arrow,
        deproject_endpoint,
        normalized_osc_action,
    )
except ImportError:  # pragma: no cover - a clear error is raised on use
    build_bowl_waypoints = decode_arrow = deproject_endpoint = normalized_osc_action = None


CAMERA_NAME = "agentview"
# The only resolution covered by the live-validated calibration/profile.  Keep
# this conservative default explicit; other resolutions are dry-run friendly
# but require an opt-in before sending motion to LIBERO.
DEFAULT_RESOLUTION = 256
DEFAULT_GOAL_OBJECT = "plate_1"
DEFAULT_SUBJECT = "akita_black_bowl_1"
# Fixed, camera/profile-specific offsets from visual endpoints to a robust
# bowl-rim grasp/release point. These are constant transforms, never runtime
# simulator object coordinates.
DEFAULT_PROFILE_NAME = "libero_spatial_akita_bowl_agentview_v1"
DEFAULT_SOURCE_GRASP_OFFSET_M = (0.0146, 0.0432, 0.0244)
DEFAULT_DESTINATION_RELEASE_OFFSET_M = (-0.0057, 0.0484, 0.0310)
DEFAULT_GRIPPER_DWELL_STEPS = 20
# The OSC controller can settle within a few millimetres of a waypoint while
# contact dynamics keep the final error from crossing exactly 1 cm.  Keep this
# explicit and conservative so task-specific diagnostics can distinguish a
# reachable near-stop from a true runaway.
WAYPOINT_POSITION_TOLERANCE_M = 0.015
VERIFIED_PROFILE_TASK_ID = 0
VERIFIED_PROFILE_SEED = 1000
VERIFIED_PROFILE_RESOLUTION = 256
VERIFIED_PROFILE_CONDITIONS = {
    "task_id": VERIFIED_PROFILE_TASK_ID,
    "seed": VERIFIED_PROFILE_SEED,
    "resolution": VERIFIED_PROFILE_RESOLUTION,
}
# Coarse finite safety volume for the adjusted controller endpoints.  This is
# a workspace sanity check only, not object detection or collision validation.
# The limits cover the validated LIBERO tabletop scene and intentionally leave
# a generous margin around its nominal bowl/plate region.
WORKSPACE_BOUNDS_M = {
    "x": (-0.8, 0.8),
    "y": (-0.8, 0.8),
    "z": (0.0, 1.6),
}
OSC_ACTION_DIM = 7
OSC_LOW = -1.0
OSC_HIGH = 1.0
PHASES = (
    "pregrasp",
    "descend",
    "close",
    "lift",
    "preplace",
    "descend_place",
    "open",
    "retreat",
)
PHASE_WAYPOINT_INDEX = {
    "pregrasp": 0,
    "descend": 1,
    "close": 1,
    "lift": 2,
    "preplace": 3,
    "descend_place": 4,
    "open": 4,
    "retreat": 5,
}
DEFAULT_OSC_SCALES = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)


@dataclass(frozen=True)
class CameraCalibration:
    """Calibration and orientation provenance for one RGB-D capture."""

    camera_name: str
    width: int
    height: int
    intrinsic: list[list[float]]
    world_from_camera: list[list[float]]
    raw_projection_intrinsic: list[list[float]] | None = None
    pixel_origin: str = "top_left"
    camera_frame: str = "opencv_optical_x_right_y_down_z_forward"
    world_frame: str = "libero_mujoco_world"
    extrinsic_direction: str = "world_from_camera"
    rgb_depth_alignment: str = "same_sim_render_call"
    source: str = "robosuite.camera_utils"
    projection_vertical_axis: str = "robosuite_projection_v_up_converted_to_image_v_down"


@dataclass
class CapturedRGBD:
    rgb: np.ndarray
    normalized_depth: np.ndarray
    metric_depth: np.ndarray
    calibration: CameraCalibration
    observation: Mapping[str, Any] | None = None


def _require(function: Callable[..., Any] | None, name: str) -> Callable[..., Any]:
    if function is None:
        raise RuntimeError(
            f"arrow controller function {name} is unavailable; install/provide arrow_controller"
        )
    return function


def settle_libero_env(env: Any, *, max_steps: int) -> dict[str, Any]:
    """Settle the inner LIBERO env with the robot frozen and retain safe diagnostics."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if _settle_physics is None:
        raise RuntimeError("radomize_scenes.settle_physics is required before motion")
    inner = getattr(env, "env", env)
    result = _settle_physics(inner, max_steps=int(max_steps))
    if not isinstance(result, Mapping):
        raise RuntimeError("settle_physics returned a non-mapping diagnostic")
    try:
        diagnostics = {
            "steps": int(result["steps_taken"]),
            "final_max_velocity_m_s": float(result["final_max_vel"]),
            "settled": bool(result["settled"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("settle_physics returned incomplete diagnostics") from exc
    setattr(env, "_arrow_settle_diagnostics", diagnostics)
    return diagnostics


def _as_rgb(array: Any) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB frame must have shape HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _as_depth(array: Any, shape: tuple[int, int]) -> np.ndarray:
    depth = np.asarray(array)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.shape != shape:
        raise ValueError(f"RGB/depth are not aligned: RGB={shape}, depth={depth.shape}")
    if not np.isfinite(depth).any():
        raise ValueError("normalized depth contains no finite pixels")
    return np.ascontiguousarray(depth.astype(np.float32, copy=False))


def _raw_observation(env: Any) -> Mapping[str, Any] | None:
    """Read one simulator observation without reading object state."""
    for owner in (getattr(env, "env", None), env):
        getter = getattr(owner, "_get_observations", None)
        if getter is None:
            continue
        try:
            return getter(force_update=True)
        except TypeError:
            return getter()
    return None


def _render_pair(
    env: Any, camera_name: str, width: int, height: int
) -> tuple[Any, Any, Mapping[str, Any] | None]:
    """Return RGB and normalized depth from one aligned render operation."""
    render = getattr(env, "render", None)
    if render is not None:
        try:
            result = render(camera_name=camera_name, width=width, height=height, depth=True)
            if isinstance(result, tuple) and len(result) >= 2:
                return result[0], result[1], None
        except (TypeError, NotImplementedError):
            pass

    observation = _raw_observation(env)
    if observation is not None:
        rgb = observation.get(f"{camera_name}_image")
        depth = observation.get(f"{camera_name}_depth")
        if rgb is not None and depth is not None:
            return rgb, depth, observation
    raise RuntimeError(
        f"could not capture aligned {camera_name} RGB+depth; "
        "construct LIBERO with camera_depths=True"
    )


def build_camera_calibration(sim: Any, camera_name: str, width: int, height: int) -> CameraCalibration:
    """Build image-aligned calibration from robosuite's canonical hooks.

    robosuite returns ``K`` for projection coordinates whose vertical axis is
    positive upward.  The rendered agentview arrays and arrow pixels use image
    coordinates with ``v`` positive downward, matching the existing
    ``world_to_pixel`` implementation, so fy/cy are converted explicitly.
    """
    if camera_utils is None:
        raise RuntimeError("robosuite camera_utils is required for metric deprojection")
    intrinsic = camera_utils.get_camera_intrinsic_matrix(
        sim, camera_name, camera_height=height, camera_width=width
    )
    extrinsic = camera_utils.get_camera_extrinsic_matrix(sim, camera_name)
    raw_intrinsic = np.asarray(intrinsic, dtype=np.float64)
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if raw_intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
        raise ValueError(f"unexpected calibration shapes K={raw_intrinsic.shape}, T={extrinsic.shape}")
    image_intrinsic = raw_intrinsic.copy()
    image_intrinsic[1, 1] = -raw_intrinsic[1, 1]
    image_intrinsic[1, 2] = float(height) - raw_intrinsic[1, 2]
    return CameraCalibration(
        camera_name=camera_name,
        width=int(width),
        height=int(height),
        intrinsic=image_intrinsic.tolist(),
        world_from_camera=extrinsic.tolist(),
        raw_projection_intrinsic=raw_intrinsic.tolist(),
    )


def normalized_depth_to_metric(sim: Any, normalized_depth: np.ndarray) -> np.ndarray:
    """Convert MuJoCo's normalized depth using robosuite, preserving alignment."""
    if camera_utils is None:
        raise RuntimeError("robosuite camera_utils.get_real_depth_map is required")
    converted = camera_utils.get_real_depth_map(sim, normalized_depth)
    metric = np.asarray(converted, dtype=np.float32)
    if metric.ndim == 3 and metric.shape[-1] == 1:
        metric = metric[..., 0]
    if metric.shape != normalized_depth.shape:
        raise ValueError(
            f"metric depth changed shape from {normalized_depth.shape} to {metric.shape}"
        )
    if not np.isfinite(metric).any() or np.nanmax(metric) <= 0:
        raise ValueError("metric depth contains no positive finite pixels")
    return np.ascontiguousarray(metric)


def capture_agentview(
    env: Any,
    *,
    resolution: int = DEFAULT_RESOLUTION,
    camera_name: str = CAMERA_NAME,
) -> CapturedRGBD:
    """Capture exactly one aligned clean RGB + normalized/metric depth pair."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    rgb_raw, normalized_raw, observation = _render_pair(env, camera_name, resolution, resolution)
    rgb = _as_rgb(rgb_raw)
    normalized = _as_depth(normalized_raw, rgb.shape[:2])
    # ``render(depth=True)`` implementations may return only pixels.  A
    # second non-visual read is allowed for EEF/gripper proprioception; it is
    # never used to reconstruct object geometry and does not affect RGB-D
    # alignment.
    if observation is None:
        observation = _raw_observation(env)
    sim = getattr(env, "sim", None)
    if sim is None:
        raise RuntimeError("capture environment does not expose sim for calibration/depth conversion")
    calibration = build_camera_calibration(sim, camera_name, rgb.shape[1], rgb.shape[0])
    metric = normalized_depth_to_metric(sim, normalized)
    return CapturedRGBD(
        rgb=rgb,
        normalized_depth=normalized,
        metric_depth=metric,
        calibration=calibration,
        observation=observation,
    )


# Descriptive aliases make the seam convenient for external smoke tests and
# keep the capture contract discoverable without duplicating implementation.
capture_synchronized_frame = capture_agentview
convert_depth_to_metric = normalized_depth_to_metric


def render_exactly_one_arrow(
    clean_rgb: np.ndarray,
    bboxes: Mapping[str, Sequence[float]],
    *,
    subject: str = DEFAULT_SUBJECT,
    goal_object: str = DEFAULT_GOAL_OBJECT,
    line_width: int = 2,
    head_length: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render only the subject→goal arrow on a copy of the clean frame.

    Ground-truth bbox generation is deliberately kept on the input-generation
    side.  The returned audit is not passed to the controller.
    """
    if subject not in bboxes or goal_object not in bboxes:
        raise ValueError("exactly-one-arrow rendering requires subject and goal bboxes")
    try:
        from visual_scene_graph import draw_scene_graph_arrows
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("visual_scene_graph is required to render the input arrow") from exc
    relation = (subject, "goal", goal_object)
    arrow = draw_scene_graph_arrows(
        _as_rgb(clean_rgb), bboxes, [relation], line_width=line_width,
        head_length=head_length, copy_image=True,
    )
    if np.array_equal(arrow, clean_rgb):
        raise ValueError("one-arrow renderer produced no visible pixels")
    return arrow, {
        "relation_count": 1,
        "relation": list(relation),
        "line_width": int(line_width),
        "head_length": int(head_length),
        "input_generation_only": True,
    }


def _call_controller(function: Callable[..., Any], **kwargs: Any) -> Any:
    """Call a controller seam while allowing a compact positional test fake."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs)
    accepted = {
        key: value for key, value in kwargs.items()
        if any(p.kind == p.VAR_KEYWORD or name == key for name, p in signature.parameters.items())
    }
    return function(**accepted)


def _endpoint(value: Any, names: Sequence[str]) -> np.ndarray:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return np.asarray(value[name], dtype=np.float64)
    for name in names:
        if hasattr(value, name):
            return np.asarray(getattr(value, name), dtype=np.float64)
    if isinstance(value, Sequence) and len(value) >= 2:
        return np.asarray(value[:2], dtype=np.float64)
    raise ValueError(f"arrow controller did not return an endpoint with fields {names}")


def decode_arrow_pixels(clean_rgb: np.ndarray, arrow_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode source-tail and destination-head pixels from the arrow controller."""
    decoded = _call_controller(
        _require(decode_arrow, "decode_arrow"), clean_rgb=clean_rgb, arrow_rgb=arrow_rgb
    )
    source = _endpoint(decoded, ("source_uv", "tail_uv", "source", "tail", "source_xy"))
    target = _endpoint(decoded, ("target_uv", "head_uv", "destination_uv", "target", "head", "target_xy"))
    if source.shape != (2,) or target.shape != (2,):
        raise ValueError(f"arrow endpoints must be (u,v), got {source.shape} and {target.shape}")
    return source, target


def _depth_at(depth: np.ndarray, uv: np.ndarray, radius: int = 2) -> float:
    u, v = np.rint(uv).astype(int)
    if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
        raise ValueError(f"arrow endpoint {(u, v)} lies outside depth frame {depth.shape[::-1]}")
    patch = depth[max(0, v - radius):v + radius + 1, max(0, u - radius):u + radius + 1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        raise ValueError(f"no valid metric depth near arrow endpoint {(u, v)}")
    return float(np.median(valid))


def validate_capture_contract(
    capture: CapturedRGBD, *, resolution: int, camera_name: str = CAMERA_NAME
) -> dict[str, Any]:
    """Validate RGB-D/calibration shape, camera, and requested-size provenance."""
    rgb = np.asarray(capture.rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"capture contract violation: RGB must be HxWx3, got {rgb.shape}")
    height, width = (int(rgb.shape[0]), int(rgb.shape[1]))

    def depth_shape(value: Any, name: str) -> tuple[int, int]:
        shape = np.asarray(value).shape
        if len(shape) == 3 and shape[-1] == 1:
            shape = shape[:2]
        if len(shape) != 2:
            raise ValueError(f"capture contract violation: {name} must be HxW, got {shape}")
        return int(shape[0]), int(shape[1])

    normalized_shape = depth_shape(capture.normalized_depth, "normalized depth")
    metric_shape = depth_shape(capture.metric_depth, "metric depth")
    if normalized_shape != (height, width) or metric_shape != (height, width):
        raise ValueError(
            "capture contract violation: RGB/depth shapes disagree "
            f"(RGB={(height, width)}, normalized={normalized_shape}, metric={metric_shape})"
        )
    calibration = capture.calibration
    if int(calibration.width) != width or int(calibration.height) != height:
        raise ValueError(
            "capture contract violation: calibration dimensions "
            f"={(calibration.width, calibration.height)} do not match RGB={(width, height)}"
        )
    if (width, height) != (int(resolution), int(resolution)):
        raise ValueError(
            "capture contract violation: requested resolution "
            f"={resolution} does not match captured RGB={(width, height)}"
        )
    if calibration.camera_name != camera_name:
        raise ValueError(
            "capture contract violation: calibration camera "
            f"{calibration.camera_name!r} is not {camera_name!r}"
        )
    return {
        "valid": True,
        "camera_name": str(calibration.camera_name),
        "requested_resolution": int(resolution),
        "rgb_shape": [height, width, 3],
        "normalized_depth_shape": list(normalized_shape),
        "metric_depth_shape": list(metric_shape),
        "calibration_shape": [int(calibration.height), int(calibration.width)],
    }


def _profile_conditions(task_id: int, seed: int, resolution: int) -> dict[str, Any]:
    """Return a stable audit record for the live-validated profile gate."""
    actual = {
        "task_id": int(task_id),
        "seed": int(seed),
        "resolution": int(resolution),
    }
    return {
        "verified": dict(VERIFIED_PROFILE_CONDITIONS),
        "actual": actual,
        "conditions_match": actual == VERIFIED_PROFILE_CONDITIONS,
    }


def validate_workspace_points(
    points: Mapping[str, Sequence[float]] | Sequence[tuple[str, Sequence[float]]],
    *,
    bounds: Mapping[str, Sequence[float]] = WORKSPACE_BOUNDS_M,
) -> None:
    """Reject non-finite/out-of-volume controller targets before motion.

    ``bounds`` is deliberately a named, coarse safety volume.  It does not
    establish object identity, collision freedom, or task success.
    """
    items = points.items() if isinstance(points, Mapping) else points
    for name, point in items:
        value = np.asarray(point, dtype=np.float64).reshape(-1)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"workspace validation failed for {name}: expected finite 3D point")
        for axis, coordinate in zip(("x", "y", "z"), value):
            limits = np.asarray(bounds[axis], dtype=np.float64).reshape(-1)
            if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] > limits[1]:
                raise ValueError(f"workspace bounds for {axis} are invalid")
            if coordinate < limits[0] or coordinate > limits[1]:
                raise ValueError(
                    f"workspace validation failed for {name}: {axis}={coordinate:.6f} "
                    f"outside [{limits[0]:.6f}, {limits[1]:.6f}] m"
                )


def _proprioception(observation: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
    """Extract only EEF/gripper proprioception needed by the motion loop."""
    if observation is None:
        return {}
    aliases = {
        "eef_pos": ("robot0_eef_pos", "eef_pos"),
        "eef_quat": ("robot0_eef_quat", "eef_quat"),
        "gripper_qpos": ("robot0_gripper_qpos", "gripper_qpos"),
    }
    result = {}
    for output, candidates in aliases.items():
        for key in candidates:
            if key in observation:
                result[output] = np.asarray(observation[key], dtype=np.float64)
                break
    return result


def _position(waypoint: Any) -> np.ndarray:
    if isinstance(waypoint, Mapping):
        for key in ("position", "pos", "eef_pos"):
            if key in waypoint:
                return np.asarray(waypoint[key], dtype=np.float64)
    for key in ("position", "pos", "eef_pos"):
        if hasattr(waypoint, key):
            return np.asarray(getattr(waypoint, key), dtype=np.float64)
    return np.asarray(waypoint, dtype=np.float64)[:3]


def _phase_waypoint(waypoints: Any, phase: str) -> Any:
    if isinstance(waypoints, Mapping):
        if phase in waypoints:
            return waypoints[phase]
        aliases = {"descend_place": "place", "preplace": "pre_place"}
        if aliases.get(phase) in waypoints:
            return waypoints[aliases[phase]]
    if isinstance(waypoints, (np.ndarray, Sequence)) and not isinstance(waypoints, (str, bytes)):
        index = PHASE_WAYPOINT_INDEX[phase]
        if index < len(waypoints):
            return waypoints[index]
    raise ValueError(f"controller did not provide waypoint for phase {phase!r}")


def normalized_action_for_waypoint(
    current_proprio: Mapping[str, np.ndarray], waypoint: Any, *, gripper: float,
    held_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Produce one finite, seven-dimensional normalized OSC_POSE action."""
    current = current_proprio.get("eef_pos")
    if current is None:
        raise ValueError("EEF position proprioception is required for OSC action generation")
    target = _position(waypoint)
    current_rot = current_proprio.get("eef_quat")
    if current_rot is None:
        raise ValueError("EEF orientation proprioception is required for OSC action generation")
    if held_rotation is None:
        held_rotation = current_rot
    # Waypoints are positions only by design.  Holding the initial EEF
    # orientation makes the bowl transfer a top-down, deterministic primitive.
    action = _require(normalized_osc_action, "normalized_osc_action")(
        current_pos=current[:3],
        current_rot=current_rot,
        target_pos=target[:3],
        target_rot=held_rotation,
        gripper=float(gripper),
        scales=DEFAULT_OSC_SCALES,
    )
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape != (OSC_ACTION_DIM,):
        raise ValueError(f"OSC action must have shape (7,), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("OSC action contains non-finite values")
    if np.any(action < OSC_LOW) or np.any(action > OSC_HIGH):
        raise ValueError("OSC action exceeded normalized [-1, 1] bounds")
    return action.astype(np.float32)


def _step_once(env: Any, action: np.ndarray) -> tuple[Mapping[str, Any] | None, bool, dict[str, Any]]:
    result = env.step(action)
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("LIBERO step must return observation and status")
    observation = result[0] if isinstance(result[0], Mapping) else None
    if len(result) == 4:  # robosuite/LIBERO legacy: obs, reward, done, info
        done = bool(result[2])
        info = result[3] if isinstance(result[3], Mapping) else {}
    else:  # Gymnasium: obs, reward, terminated, truncated, info
        done = bool(result[2]) or bool(result[3])
        info = result[4] if isinstance(result[4], Mapping) else {}
    return observation, done, dict(info)


def _run_motion(
    env: Any,
    waypoints: Any,
    initial_observation: Mapping[str, Any] | None,
    *,
    phase_timeout_steps: int,
    gripper_dwell_steps: int,
    stop_after_phase: str,
    dry_run: bool,
    phase_frame_callback: Callable[[str, int], str | Path | None] | None = None,
    motion_started_callback: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    if phase_timeout_steps <= 0:
        raise ValueError("phase_timeout_steps must be positive")
    if gripper_dwell_steps <= 0:
        raise ValueError("gripper_dwell_steps must be positive")
    if stop_after_phase not in PHASES:
        raise ValueError(f"stop_after_phase must be one of {PHASES}, got {stop_after_phase!r}")
    # Preserve a failure-safe motion marker for batch diagnostics.  A phase can
    # fail during its first simulator step, before an episode audit is written;
    # the matrix runner must still distinguish "motion never began" from a
    # controller/environment failure after an action was sent.
    setattr(env, "_arrow_motion_began", False)
    motion_notified = False
    proprio = _proprioception(initial_observation)
    held_rotation = proprio.get("eef_quat")
    if held_rotation is None:
        raise ValueError("initial observation lacks EEF orientation proprioception")
    phase_audit: list[dict[str, Any]] = []
    # Keep a live reference so a timeout or other controller exception can be
    # serialized by the matrix runner even though no final audit is produced.
    setattr(env, "_arrow_phase_audit", phase_audit)
    for phase in PHASES:
        waypoint = _phase_waypoint(waypoints, phase)
        gripper = 1.0 if phase == "close" else -1.0 if phase == "open" else 0.0
        is_gripper_phase = phase in {"close", "open"}
        required_steps = gripper_dwell_steps if is_gripper_phase else phase_timeout_steps
        record = {
            "phase": phase,
            "steps": 0,
            "status": "dry_run" if dry_run else "pending",
            "gripper_command": float(gripper),
            "dwell_steps": int(gripper_dwell_steps) if is_gripper_phase else 0,
        }
        for step in range(required_steps):
            action = normalized_action_for_waypoint(
                proprio, waypoint, gripper=gripper, held_rotation=held_rotation
            )
            record["last_action"] = action.tolist()
            record["steps"] = step + 1
            if dry_run:
                break
            if not motion_notified:
                # Let a batch coordinator durably record the transition to
                # physical motion before the first simulator action is sent.
                if motion_started_callback is not None:
                    motion_started_callback()
                motion_notified = True
            setattr(env, "_arrow_motion_began", True)
            observation, done, _info = _step_once(env, action)
            proprio = _proprioception(observation)
            if observation is None or "eef_pos" not in proprio:
                raise RuntimeError(f"phase {phase} lost EEF proprioception; failing closed")
            # LIBERO's ``done`` includes evaluator success.  It is intentionally
            # ignored while executing the bounded state machine: reading it or
            # changing phase based on it would make evaluator state influence
            # motion.  Success is queried only after retreat completes.
            if is_gripper_phase:
                if step + 1 >= gripper_dwell_steps:
                    record["status"] = "dwell"
                    break
            elif np.linalg.norm(_position(waypoint) - proprio["eef_pos"][:3]) < WAYPOINT_POSITION_TOLERANCE_M:
                record["status"] = "reached"
                break
        else:
            record["status"] = "timeout"
            if "eef_pos" in proprio:
                eef_pos = np.asarray(proprio["eef_pos"][:3], dtype=np.float64)
                record["eef_pos_m"] = eef_pos.tolist()
                record["position_error_norm_m"] = float(
                    np.linalg.norm(_position(waypoint)[:3] - eef_pos)
                )
            phase_audit.append(record)
            if is_gripper_phase:
                raise TimeoutError(f"phase {phase} failed to complete {gripper_dwell_steps}-step dwell")
            raise TimeoutError(f"phase {phase} exceeded {phase_timeout_steps} steps")
        # Record only robot proprioception observed at phase end.  These
        # diagnostics are not fed back into the controller or evaluator.
        if "eef_pos" in proprio:
            eef_pos = np.asarray(proprio["eef_pos"][:3], dtype=np.float64)
            record["eef_pos_m"] = eef_pos.tolist()
            record["position_error_norm_m"] = float(
                np.linalg.norm(_position(waypoint)[:3] - eef_pos)
            )
        if "gripper_qpos" in proprio:
            record["gripper_qpos"] = np.asarray(proprio["gripper_qpos"], dtype=np.float64).tolist()
        if phase == stop_after_phase:
            record["stop_after_phase"] = True
            if dry_run:
                record["status"] = "dry_run_stop"
            else:
                record["status"] = "stop"
        phase_audit.append(record)
        setattr(env, "_arrow_phase_audit", phase_audit)
        if not dry_run and phase_frame_callback is not None:
            try:
                frame_path = phase_frame_callback(phase, len(phase_audit) - 1)
            except Exception as exc:  # diagnostics are best-effort, never control-critical
                record["diagnostic_frame"] = None
                record["diagnostic_frame_error"] = str(exc)
            else:
                if frame_path is not None:
                    record["diagnostic_frame"] = (
                        Path(frame_path).expanduser().resolve().as_posix()
                    )
        if phase == stop_after_phase:
            break
    return phase_audit


def _save_phase_snapshot(
    env: Any,
    output_dir: Path,
    phase_index: int,
    phase: str,
    *,
    width: int,
    height: int,
) -> Path:
    """Save a clean post-phase RGB render without feeding it to control."""
    render = getattr(env, "render", None)
    if render is None:
        # OffScreenRenderEnv exposes camera pixels through its observation
        # helper, not a public render() method.  This is a post-step diagnostic
        # read and does not feed back into the controller.
        rendered, _depth, _observation = _render_pair(env, CAMERA_NAME, width, height)
    else:
        try:
            rendered = render(camera_name=CAMERA_NAME, width=width, height=height, depth=False)
        except TypeError:
            try:
                rendered = render(camera_name=CAMERA_NAME, width=width, height=height)
            except Exception:
                rendered = None
        except Exception:
            rendered = None
        if rendered is None:
            observation = _raw_observation(env)
            rendered = observation.get(f"{CAMERA_NAME}_image") if observation is not None else None
    if isinstance(rendered, tuple):
        rendered = rendered[0]
    if isinstance(rendered, Mapping):
        rendered = rendered.get(f"{CAMERA_NAME}_image")
    if rendered is None:
        raise RuntimeError(f"phase {phase} render returned no RGB image")
    image = _as_rgb(rendered)
    path = output_dir / "phase_frames" / f"{phase_index:02d}_{phase}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(image).save(path)
    return path


def _save_capture(capture: CapturedRGBD, arrow_rgb: np.ndarray, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    paths = {
        "clean_rgb": output_dir / "clean_agentview.png",
        "arrow_rgb": output_dir / "one_arrow_agentview.png",
        "normalized_depth": output_dir / "agentview_depth_normalized.npy",
        "metric_depth": output_dir / "agentview_depth_metric_m.npy",
    }
    Image.fromarray(capture.rgb).save(paths["clean_rgb"])
    Image.fromarray(_as_rgb(arrow_rgb)).save(paths["arrow_rgb"])
    np.save(paths["normalized_depth"], capture.normalized_depth)
    np.save(paths["metric_depth"], capture.metric_depth)
    return {name: path.as_posix() for name, path in paths.items()}


def run_episode(
    *,
    env: Any,
    task_id: int,
    seed: int,
    output_dir: str | Path,
    arrow_rgb: np.ndarray | None = None,
    bboxes: Mapping[str, Sequence[float]] | None = None,
    dry_run: bool = True,
    resolution: int = DEFAULT_RESOLUTION,
    goal_object: str = DEFAULT_GOAL_OBJECT,
    subject: str = DEFAULT_SUBJECT,
    phase_timeout_steps: int = 80,
    gripper_dwell_steps: int = DEFAULT_GRIPPER_DWELL_STEPS,
    stop_after_phase: str = "retreat",
    source_grasp_offset: Sequence[float] = DEFAULT_SOURCE_GRASP_OFFSET_M,
    destination_release_offset: Sequence[float] = DEFAULT_DESTINATION_RELEASE_OFFSET_M,
    clearance_m: float = 0.08,
    evaluator: Callable[[Any], bool] | None = None,
    capture: CapturedRGBD | None = None,
    allow_unvalidated_profile: bool = False,
    motion_started_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Capture, decode, and optionally execute one bounded arrow episode."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if gripper_dwell_steps <= 0:
        raise ValueError("gripper_dwell_steps must be positive")
    if stop_after_phase not in PHASES:
        raise ValueError(f"stop_after_phase must be one of {PHASES}, got {stop_after_phase!r}")
    settle_diagnostics = getattr(env, "_arrow_settle_diagnostics", None)
    if not dry_run and (
        not isinstance(settle_diagnostics, Mapping)
        or not bool(settle_diagnostics.get("settled", False))
    ):
        raise RuntimeError("refusing motion: LIBERO physics was not confirmed settled")
    np.random.seed(seed)
    capture = capture if capture is not None else capture_agentview(env, resolution=resolution)
    # A caller-supplied capture is treated exactly like a live capture: its
    # actual pixels and calibration must agree with the requested agentview
    # contract before profile validation or any possible motion.
    capture_contract = validate_capture_contract(capture, resolution=resolution)
    profile = _profile_conditions(task_id, seed, capture_contract["rgb_shape"][1])
    # Task/seed come from the seeded LIBERO episode setup; camera and
    # resolution are taken from the validated capture, never inferred solely
    # from CLI arguments.
    profile["verified"]["camera_name"] = CAMERA_NAME
    profile["actual"]["camera_name"] = capture_contract["camera_name"]
    if not dry_run and not allow_unvalidated_profile and not profile["conditions_match"]:
        raise RuntimeError(
            "refusing motion outside the verified LIBERO arrow profile "
            f"(expected task={VERIFIED_PROFILE_TASK_ID}, seed={VERIFIED_PROFILE_SEED}, "
            f"resolution={VERIFIED_PROFILE_RESOLUTION}; got task={task_id}, seed={seed}, "
            f"resolution={capture_contract['rgb_shape'][1]}); pass "
            "allow_unvalidated_profile=True only after review"
        )
    if arrow_rgb is None:
        if bboxes is None:
            raise ValueError("provide arrow_rgb or input-generation bboxes")
        arrow_rgb, arrow_audit = render_exactly_one_arrow(
            capture.rgb, bboxes, subject=subject, goal_object=goal_object,
        )
    else:
        arrow_rgb = _as_rgb(arrow_rgb)
        if arrow_rgb.shape != capture.rgb.shape:
            raise ValueError("clean RGB and arrow RGB must have identical shape")
        arrow_audit = {"controller_input": "caller_supplied_one_arrow"}

    # Controller receives no bboxes, names, evaluator, or env handle.
    source_uv, target_uv = decode_arrow_pixels(capture.rgb, arrow_rgb)
    source_depth = _depth_at(capture.metric_depth, source_uv)
    target_depth = _depth_at(capture.metric_depth, target_uv)
    calibration = asdict(capture.calibration)
    K = capture.calibration.intrinsic
    T_world_camera = capture.calibration.world_from_camera
    deproject = _require(deproject_endpoint, "deproject_endpoint")
    # Deproject with the exact robust scalar depth audited below.  Passing the
    # whole depth image would allow a controller/helper to silently choose a
    # different neighborhood than the one used for provenance.
    source_visual_point = np.asarray(
        deproject(source_uv, source_depth, K, T_world_camera), dtype=np.float64
    ).reshape(-1)
    destination_visual_point = np.asarray(
        deproject(target_uv, target_depth, K, T_world_camera), dtype=np.float64
    ).reshape(-1)
    if source_visual_point.shape != (3,) or destination_visual_point.shape != (3,):
        raise ValueError("deproject_endpoint must return finite 3D points")
    source_offset = np.asarray(source_grasp_offset, dtype=np.float64).reshape(-1)
    destination_offset = np.asarray(destination_release_offset, dtype=np.float64).reshape(-1)
    if source_offset.shape != (3,) or destination_offset.shape != (3,):
        raise ValueError("source_grasp_offset and destination_release_offset must each have 3 values")
    if not np.isfinite(source_offset).all() or not np.isfinite(destination_offset).all():
        raise ValueError("source/destination offsets must be finite")
    source_offset_overridden = not np.allclose(source_offset, DEFAULT_SOURCE_GRASP_OFFSET_M, atol=1e-12)
    destination_offset_overridden = not np.allclose(
        destination_offset, DEFAULT_DESTINATION_RELEASE_OFFSET_M, atol=1e-12
    )
    bowl_point = source_visual_point + source_offset
    destination_point = destination_visual_point + destination_offset
    if not np.isfinite(source_visual_point).all() or not np.isfinite(destination_visual_point).all():
        raise ValueError("deproject_endpoint returned non-finite visual endpoint")
    workspace_validation = {
        "status": "not_run_dry_run" if dry_run else "passed",
        "kind": "coarse_finite_volume_only",
        "bounds_m": {axis: list(limits) for axis, limits in WORKSPACE_BOUNDS_M.items()},
        "points": {
            "source_grasp_target": bowl_point.tolist(),
            "destination_release_target": destination_point.tolist(),
        },
    }
    initial_proprio = _proprioception(capture.observation)
    if not np.isfinite(clearance_m) or clearance_m <= 0:
        raise ValueError("clearance_m must be finite and positive")
    waypoints = _require(build_bowl_waypoints, "build_bowl_waypoints")(
        bowl_point, destination_point,
        initial_proprio.get("eef_quat"),
        {"lift_height_m": float(clearance_m)},
    )
    if isinstance(waypoints, Mapping):
        waypoint_points = {
            f"waypoint_{name}": _position(value)[:3] for name, value in waypoints.items()
        }
    else:
        try:
            waypoint_values = list(waypoints)
        except TypeError as exc:
            raise ValueError("controller waypoints must be a sequence or mapping") from exc
        waypoint_points = {
            f"waypoint_{index}": _position(value)[:3]
            for index, value in enumerate(waypoint_values)
        }
    workspace_validation["points"].update(
        {name: np.asarray(point, dtype=np.float64).tolist() for name, point in waypoint_points.items()}
    )
    if not dry_run:
        validate_workspace_points(
            {
                "source_grasp_target": bowl_point,
                "destination_release_target": destination_point,
                **waypoint_points,
            }
        )
        workspace_validation["status"] = "passed"

    output_root = Path(output_dir).expanduser().resolve()
    frame_paths = _save_capture(capture, arrow_rgb, output_root)
    phase_frame_paths: list[str] = []

    def phase_frame_callback(phase: str, phase_index: int) -> Path:
        path = _save_phase_snapshot(
            env,
            output_root,
            phase_index,
            phase,
            width=capture.rgb.shape[1],
            height=capture.rgb.shape[0],
        )
        phase_frame_paths.append(path.as_posix())
        return path

    phase_audit = _run_motion(
        env,
        waypoints,
        capture.observation,
        phase_timeout_steps=phase_timeout_steps,
        gripper_dwell_steps=gripper_dwell_steps,
        stop_after_phase=stop_after_phase,
        dry_run=dry_run,
        phase_frame_callback=phase_frame_callback if not dry_run else None,
        motion_started_callback=motion_started_callback if not dry_run else None,
    )
    # Evaluator state is intentionally queried only after all motion phases.
    full_execution = stop_after_phase == "retreat"
    success = None if dry_run or not full_execution else (bool(evaluator(env)) if evaluator is not None else None)
    audit = {
        "schema_version": "arrow_pick_place_mvp.v1",
        "task_id": int(task_id),
        "seed": int(seed),
        "camera": CAMERA_NAME,
        "resolution": [capture.rgb.shape[1], capture.rgb.shape[0]],
        "dry_run": bool(dry_run),
        "motion_executed": not bool(dry_run),
        "stop_after_phase": stop_after_phase,
        "full_state_machine_completed": bool(full_execution and not dry_run),
        "gripper_dwell_steps": int(gripper_dwell_steps),
        "settle_diagnostics": settle_diagnostics,
        "capture_contract": capture_contract,
        "profile": {
            "name": DEFAULT_PROFILE_NAME,
            "verified_conditions": profile["verified"],
            "actual_conditions": profile["actual"],
            "conditions_match": bool(profile["conditions_match"]),
            "allow_unvalidated_profile": bool(allow_unvalidated_profile),
        },
        "profile_validated": bool(profile["conditions_match"]),
        "allow_unvalidated_profile": bool(allow_unvalidated_profile),
        "controller_input": "pixels_depth_calibration",
        "arrow": arrow_audit,
        "arrow_endpoints_uv": {"source_tail": source_uv.tolist(), "destination_head": target_uv.tolist()},
        "endpoint_depths_m": {
            "source_tail": float(source_depth),
            "destination_head": float(target_depth),
        },
        "deprojected_visual_endpoint_world_points_m": {
            "source_tail": source_visual_point.tolist(),
            "destination_head": destination_visual_point.tolist(),
        },
        "control_targets_world_m": {
            "source_grasp": bowl_point.tolist(),
            "destination_release": destination_point.tolist(),
        },
        "waypoints_world_m": np.asarray(waypoints, dtype=np.float64).tolist(),
        "source_grasp_offset_m": source_offset.tolist(),
        "destination_release_offset_m": destination_offset.tolist(),
        "offset_profile": DEFAULT_PROFILE_NAME,
        "offsets_overridden": {
            "source_grasp": bool(source_offset_overridden),
            "destination_release": bool(destination_offset_overridden),
        },
        "clearance_m": float(clearance_m),
        "workspace_validation": workspace_validation,
        "calibration": calibration,
        "frame_orientation": {
            "pixel_origin": "top_left",
            "rgb_depth": "aligned_same_capture",
            "intrinsic_vertical_conversion": "K_image[1,1]=-K_raw[1,1]; K_image[1,2]=H-K_raw[1,2]",
            "world_from_camera": "robosuite_camera_utils_with_axis_correction",
        },
        "frames": frame_paths,
        "phase_frames": phase_frame_paths,
        "phase_frame_errors": {
            record["phase"]: record["diagnostic_frame_error"]
            for record in phase_audit
            if "diagnostic_frame_error" in record
        },
        "phases": phase_audit,
        "evaluator_success": success,
        "evaluator_read_after_action": bool(not dry_run and full_execution),
        "timestamp_unix": time.time(),
    }
    audit_path = output_root / "arrow_pick_place_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    audit["audit_path"] = audit_path.as_posix()
    return audit


def build_libero_env(task_id: int, seed: int, resolution: int) -> Any:
    """Construct direct LIBERO OffScreenRenderEnv with aligned camera depth."""
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("LIBERO and robosuite are required for live execution") from exc
    from config import BENCHMARK_NAME

    suite = benchmark.get_benchmark_dict()[BENCHMARK_NAME]()
    bddl_file = suite.get_task_bddl_file_path(int(task_id))
    np.random.seed(seed)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_names=[CAMERA_NAME],
        camera_heights=int(resolution),
        camera_widths=int(resolution),
        camera_depths=True,
        controller="OSC_POSE",
    )
    # Match render_visual_arrow_pair.py's seed/init-state setup while avoiding
    # LeRobot's terminal autoreset.  The direct wrapper owns the same init-state
    # files and lets us preserve the terminal observation through retreat.
    seed_fn = getattr(env, "seed", None)
    if seed_fn is not None:
        seed_fn(int(seed))
    env.reset()
    init_state_diagnostics = {
        "source": "seeded_reset_fallback",
        "available_count": None,
        "selected_index": None,
        "fallback": True,
    }
    try:
        from lerobot.envs.libero import get_task_init_states

        init_states = get_task_init_states(suite, int(task_id))
        available_count = len(init_states)
        init_state_diagnostics.update({
            "source": "lerobot.get_task_init_states",
            "available_count": int(available_count),
        })
        if available_count:
            selected_index = int(seed) % int(available_count)
            env.set_init_state(init_states[selected_index])
            init_state_diagnostics.update({
                "selected_index": int(selected_index),
                "fallback": False,
            })
    except (ImportError, FileNotFoundError, AttributeError, TypeError):
        # Some lightweight LIBERO installs omit LeRobot's torch loader; the
        # seeded reset remains a valid fallback and is recorded by the caller.
        pass
    setattr(env, "_arrow_init_state_diagnostics", init_state_diagnostics)
    try:
        from preview_visual_arrows import _apply_task_swaps

        _apply_task_swaps(env.env, int(task_id))
    except (ImportError, KeyError):
        pass
    from config import SETTLE_STEPS_INIT

    settle_libero_env(env, max_steps=SETTLE_STEPS_INIT)
    return env


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--output-dir", type=Path, default=Path("arrow_pick_place_outputs"))
    parser.add_argument("--phase-timeout-steps", type=int, default=80)
    parser.add_argument("--gripper-dwell-steps", type=int, default=DEFAULT_GRIPPER_DWELL_STEPS)
    parser.add_argument("--stop-after-phase", choices=PHASES, default="retreat")
    parser.add_argument("--source-grasp-offset", type=float, nargs=3, default=DEFAULT_SOURCE_GRASP_OFFSET_M, metavar=("DX", "DY", "DZ"))
    parser.add_argument("--destination-release-offset", type=float, nargs=3, default=DEFAULT_DESTINATION_RELEASE_OFFSET_M, metavar=("DX", "DY", "DZ"))
    parser.add_argument("--clearance-m", type=float, default=0.08)
    parser.add_argument(
        "--allow-unvalidated-profile",
        action="store_true",
        help="allow motion outside task=0/seed=1000/resolution=256 after manual review",
    )
    parser.add_argument("--dry-run", "--no-motion", dest="dry_run", action="store_true", default=True)
    parser.add_argument("--execute-motion", dest="dry_run", action="store_false")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    env = build_libero_env(args.task, args.seed, args.resolution)
    try:
        # Input-generation integration is intentionally explicit: a caller can
        # replace this with a human/annotated arrow PNG while the controller
        # contract remains unchanged.  The built-in path uses GT bboxes only
        # to create the one-arrow input, never for control.
        from libero_live_semantic_context import LiveSemanticContextGenerator
        from config import SCENE_GRAPH_SUBJECT_FILTER, TASK_GOAL_OBJECT_CONFIG
        from types import SimpleNamespace

        generator = LiveSemanticContextGenerator()
        generator.scene_graph_subject_filter = SCENE_GRAPH_SUBJECT_FILTER
        capture = capture_agentview(env, resolution=args.resolution)
        # The semantic helper is input-generation-only and expects the small
        # LeRobot wrapper surface.  It reads bboxes for rendering the arrow;
        # none of this namespace is passed to the controller.
        task_text = getattr(env, "language_instruction", "")
        context_env = SimpleNamespace(
            _env=env.env,
            observation_height=args.resolution,
            observation_width=args.resolution,
            task=task_text,
            task_id=args.task,
        )
        context = generator.observe_visual_graph(context_env, camera=CAMERA_NAME)
        goal = TASK_GOAL_OBJECT_CONFIG.get(args.task, DEFAULT_GOAL_OBJECT)
        result = run_episode(
            env=env,
            task_id=args.task,
            seed=args.seed,
            output_dir=args.output_dir,
            bboxes=context["bboxes"],
            dry_run=args.dry_run,
            resolution=args.resolution,
            goal_object=goal,
            subject=SCENE_GRAPH_SUBJECT_FILTER,
            phase_timeout_steps=args.phase_timeout_steps,
            gripper_dwell_steps=args.gripper_dwell_steps,
            stop_after_phase=args.stop_after_phase,
            source_grasp_offset=args.source_grasp_offset,
            destination_release_offset=args.destination_release_offset,
            clearance_m=args.clearance_m,
            allow_unvalidated_profile=args.allow_unvalidated_profile,
            capture=capture,
            evaluator=lambda candidate: bool(candidate.check_success()),
        )
        print(json.dumps({"audit_path": result["audit_path"], "success": result["evaluator_success"]}))
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
