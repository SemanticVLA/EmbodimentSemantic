"""Live RoboCasa bridge for the existing arrow controller.

All RoboCasa and MuJoCo imports are lazy.  The bridge keeps simulator-only
role resolution and projection on the input side, then passes only an RGB-D
capture plus a single rendered arrow to the unchanged motion engine.
The wrapped environment converts the engine's unchanged 7D OSC command into
RoboCasa's official 12D PandaOmron dictionary action.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..environment.runtime import create_robocasa_env, validate_action_layout
from ..shared.task_manifest import PickPlaceTask, RoleSpec, get_task
from .adapter import (
    PandaOmronActionAdapter,
    ROBOCASA_CAMERA,
    adapt_capture_to_base,
    world_base_transform,
)
from .arrow import render_bbox_center_arrow
from .prompt import adapt_source_noun


# RoboCasa's PandaOmron exposes two fixed third-person agent views.  The
# controller keeps the LIBERO-facing camera identity ``agentview`` while the
# input seam may choose the physical view that can actually see both task
# roles.  The right view is a deterministic fallback for layouts where the
# left view clips the source object.
ROBOCASA_CAMERA_FALLBACKS = (ROBOCASA_CAMERA, "robot0_agentview_right")


class RoboCasaLiveError(RuntimeError):
    """Raised when a live RoboCasa cell cannot satisfy the frozen contract."""


class RoboCasaHorizonError(RoboCasaLiveError):
    """Raised when the official task horizon is exhausted before success."""


def _lookup_fixture(env: Any, name: str) -> Any:
    fixtures = getattr(env, "fixtures", None)
    if isinstance(fixtures, Mapping):
        if name in fixtures:
            return fixtures[name]
        lowered = name.lower()
        for key, fixture in fixtures.items():
            if lowered in str(key).lower():
                return fixture
    getter = getattr(env, "get_fixture", None)
    if callable(getter):
        for candidate in (name,):
            try:
                return getter(candidate)
            except Exception:
                pass
    for getter_name in ("get_fixture_by_name", "fixture_by_name"):
        getter = getattr(env, getter_name, None)
        if callable(getter):
            try:
                return getter(name)
            except Exception:
                pass
    return None


def _lookup_object(env: Any, name: str) -> Any:
    objects = getattr(env, "objects", None)
    if isinstance(objects, Mapping) and name in objects:
        return objects[name]
    for getter_name in ("get_object", "get_obj", "get_object_by_name"):
        getter = getattr(env, getter_name, None)
        if callable(getter):
            try:
                value = getter(name)
            except Exception:
                value = None
            if value is not None:
                return value
    value = getattr(env, name, None)
    if value is not None:
        return value
    # RoboCasa exposes auxiliary task objects such as ``blender_lid`` as
    # children of their owning fixture rather than in ``env.objects``.  Keep
    # this lookup on the official object graph; do not consult simulator body
    # ids or task-specific poses here.
    fixtures = getattr(env, "fixtures", None)
    if isinstance(fixtures, Mapping):
        for fixture in fixtures.values():
            child = getattr(fixture, name, None)
            if child is not None:
                return child
    return None


def _reset_region_world_points(entity: Any, region: Mapping[str, Any]) -> np.ndarray:
    """Convert an official fixture reset-region record to world corners."""

    offset = np.asarray(region.get("offset"), dtype=np.float64).reshape(-1)
    size = np.asarray(region.get("size"), dtype=np.float64).reshape(-1)
    height = float(region.get("height", 0.01))
    if offset.size != 3 or size.size != 2 or not np.isfinite(offset).all() or not np.isfinite(size).all():
        raise RoboCasaLiveError("fixture reset region has invalid offset/size")
    if np.any(size <= 0.0) or not np.isfinite(height) or height <= 0.0:
        raise RoboCasaLiveError("fixture reset region has invalid extent")
    center_offset = offset.copy()
    # Official reset regions store the lower z face while x/y offsets are
    # centered.  Move the z center upward by half the region height.
    center_offset[2] += height / 2.0
    local = np.asarray([
        (sx * size[0] / 2.0, sy * size[1] / 2.0, sz * height / 2.0)
        for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
    ], dtype=np.float64) + center_offset[None, :]
    pos = np.asarray(getattr(entity, "pos", np.zeros(3)), dtype=np.float64).reshape(-1)
    if pos.size != 3 or not np.isfinite(pos).all():
        raise RoboCasaLiveError("fixture reset region owner has invalid position")
    angle = float(getattr(entity, "rot", 0.0))
    if not np.isfinite(angle):
        raise RoboCasaLiveError("fixture reset region owner has invalid rotation")
    c, s = float(np.cos(angle)), float(np.sin(angle))
    rotation = np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    return local @ rotation.T + pos[None, :]


def _role_entity(env: Any, role: RoleSpec) -> Any:
    key = str(role.key)
    if key.startswith("self."):
        value = getattr(env, key[5:], None)
        if value is not None:
            return value
        return _lookup_fixture(env, key[5:])
    if role.kind == "object":
        return _lookup_object(env, key)
    fixture = _lookup_fixture(env, key)
    if fixture is not None:
        return fixture
    # Auxiliary RoboCasa task entities may be registered in ``objects`` even
    # though their manifest role is a fixture (for example BlenderLid).
    return _lookup_object(env, key)


def _object_pose(env: Any, entity: Any, key: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Read the current MuJoCo object pose for local MJCF bbox geometry."""

    sim = getattr(env, "sim", None)
    if sim is None:
        return None
    body_id = None
    body_ids = getattr(env, "obj_body_id", None)
    if isinstance(body_ids, Mapping):
        body_id = body_ids.get(key)
        if body_id is None:
            body_id = body_ids.get(str(key))
    if body_id is None:
        model = getattr(sim, "model", None)
        body_name = getattr(entity, "root_body", None) or getattr(entity, "name", None)
        body_name2id = getattr(model, "body_name2id", None)
        if callable(body_name2id) and body_name:
            try:
                body_id = body_name2id(str(body_name))
            except Exception:
                body_id = None
    data = getattr(sim, "data", None)
    if body_id is None or data is None:
        return None
    try:
        translation = np.asarray(data.body_xpos[int(body_id)], dtype=np.float64)
        quat_wxyz = np.asarray(data.body_xquat[int(body_id)], dtype=np.float64)
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if translation.shape != (3,) or quat_wxyz.shape != (4,):
        raise RoboCasaLiveError(f"object {key!r} returned invalid simulator pose")
    # MuJoCo stores ``wxyz``; RoboSuite's MJCF bbox helper consumes ``xyzw``.
    quat_xyzw = np.asarray(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
    )
    if not np.isfinite(translation).all() or not np.isfinite(quat_xyzw).all():
        raise RoboCasaLiveError(f"object {key!r} returned a non-finite simulator pose")
    return translation, quat_xyzw


def _runtime_region_names(env: Any, entity: Any, role: RoleSpec) -> set[str] | None:
    """Resolve official fixture-region selectors to the fixture's runtime names."""

    selector = role.region_selector
    if not selector:
        return None
    owner = getattr(env, "_env", env)
    if selector == "chosen_toaster_receptacle":
        value = getattr(owner, selector, None)
        if value is None:
            value = getattr(env, selector, None)
        if value is None:
            raise RoboCasaLiveError(
                f"RoboCasa did not expose runtime region selector {selector!r}"
            )
        return {str(value)}
    if selector not in {"fridge_drawer", "fridge_shelf"}:
        raise RoboCasaLiveError(f"unknown RoboCasa region selector {selector!r}")
    getter = getattr(entity, "get_reset_regions", None)
    if not callable(getter):
        raise RoboCasaLiveError(
            f"fixture {role.key!r} does not expose official get_reset_regions"
        )
    reg_type = "drawer" if selector == "fridge_drawer" else "shelf"
    kwargs = {
        "compartment": "fridge",
        "reg_type": reg_type,
        # The official shelf->drawer task targets the highest drawer.  The
        # drawer->shelf task accepts every valid fridge shelf.
        "rack_index": -1 if reg_type == "drawer" else None,
    }
    try:
        regions = getter(owner, **kwargs)
    except TypeError:
        regions = getter(env, **kwargs)
    if not isinstance(regions, Mapping) or not regions:
        raise RoboCasaLiveError(
            f"RoboCasa fixture returned no {reg_type} regions for {role.key!r}"
        )
    return {str(name) for name in regions}


def _world_points(env: Any, entity: Any, role: RoleSpec) -> np.ndarray:
    if entity is None:
        raise RoboCasaLiveError(f"could not resolve role entity {role.key!r}")
    if role.region == "lid_closed" and callable(getattr(entity, "get_lid_closed_pos", None)):
        closed = np.asarray(entity.get_lid_closed_pos(env), dtype=np.float64).reshape(-1)
        if closed.size != 3 or not np.isfinite(closed).all():
            raise RoboCasaLiveError(f"role {role.key!r} returned invalid closed-lid position")
        # A small measured target volume keeps projection useful while the
        # official success predicate still decides whether the lid closed.
        return closed[None, :] + np.asarray([
            (-0.01, -0.01, -0.005), (0.01, -0.01, -0.005),
            (-0.01, 0.01, 0.005), (0.01, 0.01, 0.005),
        ], dtype=np.float64)
    if role.kind in {"region", "fixture"} and callable(getattr(entity, "get_int_sites", None)):
        try:
            regions = entity.get_int_sites(all_points=True, relative=False)
        except TypeError:
            regions = entity.get_int_sites()
        if isinstance(regions, Mapping) and regions:
            requested = str(role.region or "")
            runtime_names = _runtime_region_names(env, entity, role)
            if runtime_names is not None:
                selected = [
                    points
                    for name, points in regions.items()
                    if str(name) in runtime_names
                ]
                if not selected:
                    raise RoboCasaLiveError(
                        f"role {role.key!r} runtime selector {role.region_selector!r} "
                        f"resolved to {sorted(runtime_names)}, "
                        f"available={sorted(str(name) for name in regions)}"
                    )
                points = np.vstack(
                    [np.asarray(item, dtype=np.float64) for item in selected]
                )
            else:
                selected = None
                for name, region_points in regions.items():
                    if requested and requested.lower() == str(name).lower():
                        selected = region_points
                        break
                if requested:
                    if selected is None:
                        reset_getter = getattr(entity, "get_reset_regions", None)
                        reset_regions = None
                        if callable(reset_getter):
                            try:
                                reset_regions = reset_getter(env)
                            except TypeError:
                                reset_regions = reset_getter()
                        if isinstance(reset_regions, Mapping) and requested in reset_regions:
                            points = _reset_region_world_points(entity, reset_regions[requested])
                        else:
                            raise RoboCasaLiveError(
                                f"role {role.key!r} requested region {requested!r}, "
                                f"available={sorted(str(name) for name in regions)}"
                            )
                    else:
                        points = np.asarray(selected, dtype=np.float64)
                else:
                    points = np.vstack(
                        [np.asarray(item, dtype=np.float64) for item in regions.values()]
                    )
        else:
            # RoboCasa fixtures return either a named mapping (for multiple
            # racks/compartments) or the four world-space interior corners
            # directly.  Never substitute the fixture's exterior bbox for an
            # interior placement target.
            reset_getter = getattr(entity, "get_reset_regions", None)
            reset_regions = None
            if callable(reset_getter):
                try:
                    reset_regions = reset_getter(env)
                except TypeError:
                    reset_regions = reset_getter()
            requested = str(role.region or "")
            if requested and isinstance(reset_regions, Mapping) and requested in reset_regions:
                points = _reset_region_world_points(entity, reset_regions[requested])
            else:
                points = np.asarray(regions, dtype=np.float64)
    elif callable(getattr(entity, "get_bbox_points", None)):
        if role.kind in {"fixture", "region"}:
            raise RoboCasaLiveError(
                f"fixture role {role.key!r} exposes no interior placement region; "
                "refusing to target its exterior bbox"
            )
        pose = _object_pose(env, entity, str(role.key)) if role.kind == "object" else None
        if pose is None:
            if role.kind == "object" and getattr(env, "sim", None) is not None:
                raise RoboCasaLiveError(
                    f"could not resolve current MuJoCo pose for object {role.key!r}"
                )
            points = np.asarray(entity.get_bbox_points(), dtype=np.float64)
        else:
            translation, quat_xyzw = pose
            points = np.asarray(
                entity.get_bbox_points(trans=translation, rot=quat_xyzw), dtype=np.float64
            )
    else:
        raise RoboCasaLiveError(f"role {role.key!r} has no bbox/region geometry API")
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise RoboCasaLiveError(f"role {role.key!r} returned invalid world bbox points")
    return points


def _project_bbox(points_world: np.ndarray, capture: Any, *, width: int, height: int) -> tuple[float, float, float, float]:
    K = np.asarray(capture.calibration.intrinsic, dtype=np.float64)
    T_world_camera = np.asarray(capture.calibration.world_from_camera, dtype=np.float64)
    if K.shape != (3, 3) or T_world_camera.shape != (4, 4):
        raise RoboCasaLiveError("capture calibration has invalid K/T")
    T_camera_world = np.linalg.inv(T_world_camera)
    homogeneous = np.concatenate((points_world, np.ones((len(points_world), 1))), axis=1)
    camera = (T_camera_world @ homogeneous.T).T[:, :3]
    visible = camera[:, 2] > 1e-6
    if not np.any(visible):
        raise RoboCasaLiveError("role bbox is entirely behind the camera")
    camera = camera[visible]
    u = K[0, 0] * camera[:, 0] / camera[:, 2] + K[0, 2]
    v = K[1, 1] * camera[:, 1] / camera[:, 2] + K[1, 2]
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise RoboCasaLiveError("projected bbox is non-finite")
    # A valid object can occupy less than one pixel in one image dimension
    # (or collapse to a line after perspective projection).  Keep the
    # projection useful at the controller's low-resolution Molmo input by
    # giving an in-frame subpixel footprint one pixel of support.  This is an
    # image-space quantization rule only: it does not enlarge world geometry,
    # consult simulator masks, or recover an entirely out-of-view role.
    # A projected rectangle can overlap the image while all of its vertices
    # lie outside it (for example, a near camera-facing box).  Visibility is
    # therefore an interval-overlap test, rather than a vertex-in-frame test.
    if float(np.max(u)) < 0.0 or float(np.min(u)) > width - 1 or float(np.max(v)) < 0.0 or float(np.min(v)) > height - 1:
        raise RoboCasaLiveError("role bbox has no visible area after projection")
    clipped_u = np.clip(u, 0.0, width - 1)
    clipped_v = np.clip(v, 0.0, height - 1)
    x1 = float(np.clip(np.min(u), 0, width - 1))
    x2 = float(np.clip(np.max(u), 0, width - 1))
    y1 = float(np.clip(np.min(v), 0, height - 1))
    y2 = float(np.clip(np.max(v), 0, height - 1))
    if x2 <= x1:
        center_x = float(np.median(clipped_u))
        x1 = max(0.0, center_x - 0.5)
        x2 = min(float(width - 1), center_x + 0.5)
    if y2 <= y1:
        center_y = float(np.median(clipped_v))
        y1 = max(0.0, center_y - 0.5)
        y2 = min(float(height - 1), center_y + 0.5)
    if x2 <= x1 or y2 <= y1:
        raise RoboCasaLiveError("role bbox has no visible area after projection")
    return x1, y1, x2, y2


def _capture(
    env: Any, *, resolution: int, camera_name: str = ROBOCASA_CAMERA
) -> Any:
    """Capture one aligned RoboCasa RGB-D pair with exactly one vertical flip."""
    try:
        from .capture import capture_robocasa_rgbd
        return capture_robocasa_rgbd(
            env, camera_name=str(camera_name), resolution=int(resolution)
        )
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        raise RoboCasaLiveError(str(exc)) from exc


def project_task_bboxes(env: Any, capture: Any, task: PickPlaceTask) -> tuple[dict[str, tuple[float, float, float, float]], str]:
    """Resolve and project the frozen source/destination roles for one state."""

    source_keys = (task.source.key,)
    if task.source_selector:
        source_keys = ("ice_cube1", "ice_cube2")
    bboxes: dict[str, tuple[float, float, float, float]] = {}
    for key in (*source_keys, task.destination.key):
        role = replace(task.source, key=key) if key in source_keys else task.destination
        entity = _role_entity(env, role)
        points = _world_points(env, entity, role)
        try:
            bboxes[key] = _project_bbox(
                points, capture, width=int(capture.rgb.shape[1]), height=int(capture.rgb.shape[0])
            )
        except RoboCasaLiveError as exc:
            # Keep the original projection behavior while making the failed
            # source/destination role explicit in the persisted cell result.
            raise RoboCasaLiveError(f"role {role.key!r}: {exc}") from exc
    if task.source_selector:
        selected = max(source_keys, key=lambda key: (bboxes[key][2] - bboxes[key][0]) * (bboxes[key][3] - bboxes[key][1]))
    else:
        selected = task.source.key
    return bboxes, selected


def _source_label(env: Any, task: PickPlaceTask, source_key: str) -> str:
    """Resolve RoboCasa's official object language without changing the prompt."""

    getter = getattr(env, "get_obj_lang", None)
    if callable(getter):
        for call in (
            lambda: getter(obj_name=source_key),
            lambda: getter(source_key),
        ):
            try:
                label = str(call()).strip()
            except Exception:
                continue
            if label:
                return label
    entity = _role_entity(env, replace(task.source, key=source_key))
    for attribute in ("name", "object_name", "category_name"):
        label = str(getattr(entity, attribute, "")).strip()
        if label:
            return label.replace("_", " ")
    if task.source.static_label:
        return task.source.static_label
    if source_key != "obj":
        return source_key.replace("_", " ")
    raise RoboCasaLiveError("RoboCasa did not expose a language label for source object 'obj'")


class RoboCasaControllerEnv:
    """Expose RoboCasa through the 7D step contract expected by LIBERO motion."""

    def __init__(self, env: Any):
        self._env = env
        self._last_info: dict[str, Any] = {}
        self._last_observation: Mapping[str, Any] | None = None
        self._action_adapter = PandaOmronActionAdapter.from_env(env)
        horizon = getattr(env, "horizon", None)
        if horizon is None:
            horizon = getattr(env, "_horizon", None)
        self._horizon = None if horizon is None else int(horizon)
        if self._horizon is not None and self._horizon <= 0:
            raise RoboCasaLiveError(f"invalid RoboCasa horizon {self._horizon}")
        self._steps = 0
        # Keep the canonical command boundary auditable.  This history is
        # deliberately local to the wrapper; the standalone runtime probe
        # creates a separate environment and never touches scored cells.
        self._action_history: list[dict[str, Any]] = []
        self._arrow_depth_encoding = "normalized"
        self._arrow_physical_camera_name = ROBOCASA_CAMERA
        self._world_from_base_B0: np.ndarray | None = None
        self._base_from_world_B0: np.ndarray | None = None
        self._last_raw_base: dict[str, Any] = {}
        self._last_raw_proprio: dict[str, Any] = {}
        self._initial_raw_base: dict[str, Any] = {}
        self._initial_raw_proprio: dict[str, Any] = {}
        self._arrow_settle_diagnostics = {"settled": True, "source": "robocasa_reset"}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def last_info(self) -> Mapping[str, Any]:
        return self._last_info

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        # A reset starts a new frozen frame.  Never let the previous episode's
        # B0 transform or raw sensor snapshot contaminate this episode.
        self._world_from_base_B0 = None
        self._base_from_world_B0 = None
        self._last_raw_base = {}
        self._last_raw_proprio = {}
        self._initial_raw_base = {}
        self._initial_raw_proprio = {}
        result = self._env.reset(*args, **kwargs)
        self._steps = 0
        self._action_history = []
        observation = result[0] if isinstance(result, tuple) and result else result
        if isinstance(observation, Mapping):
            self._freeze_base_frame(observation)
            adapted = self._last_observation
            if isinstance(result, tuple):
                return (adapted, *result[1:])
            return adapted
        return result

    def _freeze_base_frame(self, observation: Mapping[str, Any]) -> None:
        """Freeze B0 at reset and expose both EEF aliases in B0 consistently."""
        raw_base, raw_proprio = _raw_robot_diagnostics(observation)
        self._last_raw_base = raw_base
        self._last_raw_proprio = raw_proprio
        if self._steps == 0 and not self._initial_raw_base:
            self._initial_raw_base = dict(raw_base)
            self._initial_raw_proprio = dict(raw_proprio)
        if self._base_from_world_B0 is None:
            if "robot0_base_pos" not in observation or "robot0_base_quat" not in observation:
                # Dependency-light contract fixtures may intentionally omit
                # robot observations; keep their identity frame without making
                # this path executable for a real RoboCasa cell.
                if observation:
                    raise RoboCasaLiveError(
                        "RoboCasa reset observation must expose robot0_base_pos and robot0_base_quat"
                    )
                self._world_from_base_B0 = np.eye(4, dtype=np.float64)
                self._base_from_world_B0 = np.eye(4, dtype=np.float64)
            else:
                self._world_from_base_B0, self._base_from_world_B0 = world_base_transform(observation)
        adapted = dict(observation)
        base_pos = raw_proprio.get("robot0_base_to_eef_pos")
        base_quat = raw_proprio.get("robot0_base_to_eef_quat")
        if base_pos is None or base_quat is None:
            if not observation:
                self._last_observation = adapted
                return
            raise RoboCasaLiveError(
                "RoboCasa reset observation must expose robot0_base_to_eef_pos and "
                "robot0_base_to_eef_quat"
            )
        # Compose T_B0G = T_B0W * T_WBt * T_BtG.  The old implementation
        # aliased the current-base sensor directly, which was only correct at
        # reset and made episode_contract consume mixed frames after base drift.
        current_world_from_base, _ = world_base_transform(self._last_raw_base)
        frozen_base_from_world = np.asarray(self._base_from_world_B0, dtype=np.float64)
        current_base_from_frozen = frozen_base_from_world @ current_world_from_base
        raw_pos = np.asarray(base_pos, dtype=np.float64).reshape(-1)
        raw_quat = np.asarray(base_quat, dtype=np.float64).reshape(-1)
        if raw_pos.shape != (3,) or not np.isfinite(raw_pos).all() or raw_quat.shape != (4,):
            raise RoboCasaLiveError("raw robot0_base_to_eef pose has invalid shape")
        eef_transform = np.eye(4, dtype=np.float64)
        eef_transform[:3, :3] = _rotation_from_quaternion_xyzw(raw_quat)
        eef_transform[:3, 3] = raw_pos
        b0_eef = current_base_from_frozen @ eef_transform
        b0_pos = b0_eef[:3, 3]
        b0_quat = _quaternion_xyzw_from_matrix(b0_eef[:3, :3])
        if b0_quat is None:
            raise RoboCasaLiveError("composed B0 EEF rotation is invalid")
        # Both aliases are consumed by the controller's alias priority list;
        # keep them identical and retain raw values only in diagnostic fields.
        adapted["robot0_base_to_eef_pos"] = b0_pos.copy()
        adapted["robot0_base_to_eef_quat"] = b0_quat.copy()
        adapted["robot0_eef_pos"] = b0_pos.copy()
        adapted["robot0_eef_quat"] = b0_quat.copy()
        adapted.pop("eef_pos", None)
        adapted.pop("eef_quat", None)
        self._last_observation = adapted

    def _action_in_current_base(self, canonical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rotate B0 translational/axis-angle deltas into the current OSC base."""

        transformed = np.asarray(canonical, dtype=np.float64).reshape(-1).copy()
        if transformed.shape != (7,):
            raise RoboCasaLiveError("canonical action must have dimension 7")
        rotation = np.eye(3, dtype=np.float64)
        raw_base = self._last_raw_base
        if self._world_from_base_B0 is not None and raw_base:
            current_world_from_base, _ = world_base_transform(raw_base)
            # R_BtB0 = R_WBt^T R_WB0.
            rotation = current_world_from_base[:3, :3].T @ np.asarray(
                self._world_from_base_B0, dtype=np.float64
            )[:3, :3]
            transformed[:3] = rotation @ transformed[:3]
            transformed[3:6] = rotation @ transformed[3:6]
            # Preserve the adapter's normalized contract across floating
            # point roundoff at unit action boundaries.
            transformed[:6] = np.clip(transformed[:6], -1.0, 1.0)
        return transformed, rotation

    def step(self, action: Sequence[float]) -> Any:
        if self._horizon is not None and self._steps >= self._horizon:
            raise RoboCasaHorizonError(
                f"RoboCasa task horizon exhausted at {self._steps}/{self._horizon} actions"
            )
        canonical = np.asarray(action, dtype=np.float64).reshape(-1)
        transformed, action_rotation = self._action_in_current_base(canonical)
        packed = self._action_adapter.pack(transformed)
        validate_action_layout(tuple(float(value) for value in packed))
        # The official create_env factory returns raw robosuite (flat 12D),
        # while a Gym wrapper exposes the same split as a Dict space.
        action_dict = {
            "action.end_effector_position": np.asarray(packed[0:3], dtype=np.float32),
            "action.end_effector_rotation": np.asarray(packed[3:6], dtype=np.float32),
            "action.gripper_close": np.asarray(packed[6:7], dtype=np.float32),
            "action.base_motion": np.asarray(packed[7:11], dtype=np.float32),
            "action.control_mode": np.asarray(packed[11:12], dtype=np.float32),
        }
        command = {
            "index": int(self._steps),
            "controller_action": canonical.tolist(),
            "controller_action_b0": canonical.tolist(),
            "controller_action_current_base": transformed.tolist(),
            "action_frame_provenance": {
                "input_frame": "robocasa_pandaomron_base_B0",
                "osc_frame": "robocasa_pandaomron_base_current",
                "rotation_current_base_from_B0": action_rotation.tolist(),
                "normalized_scales_isotropic": {
                    "translation": 0.05,
                    "axis_angle": 0.5,
                },
            },
            "packed_action": np.asarray(packed, dtype=np.float64).tolist(),
            "arm_command": bool(np.any(np.abs(canonical[:6]) > 0.0)),
            "gripper_only_command": bool(
                not np.any(np.abs(canonical[:6]) > 0.0)
                and abs(float(canonical[6])) > 0.0
            ),
            "status": "attempted",
        }
        self._action_history.append(command)
        action_space = getattr(self._env, "action_space", None)
        use_dict = False
        if action_space is not None:
            try:
                use_dict = "action.end_effector_position" in action_space
            except Exception:
                try:
                    action_space["action.end_effector_position"]
                    use_dict = True
                except (KeyError, TypeError, AttributeError):
                    use_dict = False
        try:
            result = self._env.step(action_dict if use_dict else np.asarray(packed, dtype=np.float32))
        except Exception as exc:
            command.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
            raise
        command["status"] = "sent"
        self._steps += 1
        observation = result[0] if isinstance(result, tuple) and result else result
        if isinstance(observation, Mapping):
            self._freeze_base_frame(observation)
            if isinstance(result, tuple):
                result = (self._last_observation, *result[1:])
            else:
                result = self._last_observation
        if isinstance(result, tuple) and result and isinstance(result[-1], Mapping):
            self._last_info = dict(result[-1])
        position = _current_eef_b0(self)
        if position is not None:
            command["eef_after_b0_m"] = position.tolist()
        command["base_drift_after"] = _base_drift(self)
        return result

    def render(self, *args: Any, **kwargs: Any) -> Any:
        camera = kwargs.get("camera_name")
        if camera == "agentview":
            kwargs["camera_name"] = str(
                getattr(self, "_arrow_physical_camera_name", ROBOCASA_CAMERA)
            )
        sim = getattr(self._env, "sim", None)
        render_owner = sim if sim is not None else self._env
        if sim is not None:
            width = int(kwargs.pop("width", 256))
            height = int(kwargs.pop("height", 256))
            depth = bool(kwargs.pop("depth", False))
            result = render_owner.render(height=height, width=width, depth=depth, **kwargs)
        else:
            result = render_owner.render(*args, **kwargs)
        if sim is not None:
            # ``sim.render`` is the sole native bottom-left producer.  Flip
            # that raw result once; wrapper observation fallbacks pass through.
            if isinstance(result, tuple) and len(result) >= 2:
                return (np.asarray(result[0])[::-1].copy(), np.asarray(result[1])[::-1].copy(), *result[2:])
            return np.asarray(result)[::-1].copy()
        if isinstance(result, tuple) and len(result) >= 2:
            return (np.asarray(result[0]).copy(), np.asarray(result[1]).copy(), *result[2:])
        return np.asarray(result).copy()

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def horizon(self) -> int | None:
        return self._horizon


def official_success(env: RoboCasaControllerEnv) -> bool:
    """Read RoboCasa's official post-step success signal.

    The Gym wrapper places this predicate in ``info['success']``.  The
    official ``create_env`` helper returns raw robosuite, whose step tuple has
    no info mapping, so the same task-owned predicate is read from
    ``_check_success`` only after the controller has completed its step.
    """

    if "success" in env.last_info:
        return bool(env.last_info["success"])
    checker = getattr(getattr(env, "_env", env), "_check_success", None)
    if callable(checker):
        return bool(checker())
    return False


def _official_outcome_from_controller_audit(audit: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return ``(evaluator_called, official_success)`` from the canary manifest."""

    final_result = audit.get("final_result")
    if not isinstance(final_result, Mapping):
        return False, False
    evaluator_called = bool(final_result.get("evaluator_called"))
    return evaluator_called, evaluator_called and final_result.get("evaluator_success") is True


def _diagnostic_safe(value: Any) -> Any:
    """Convert runtime values to JSON without changing the live contract."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _diagnostic_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _runtime_class(value: Any) -> str | None:
    if value is None:
        return None
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _model_count(model: Any, kind: str) -> int | None:
    value = getattr(model, f"n{kind}", None)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    names = getattr(model, "names", None)
    if isinstance(names, Mapping):
        try:
            return len(names.get(kind, ()))
        except TypeError:
            pass
    values = getattr(model, f"{kind}_names", None)
    try:
        return len(values) if values is not None else None
    except TypeError:
        return None


def _model_name(model: Any, kind: str, index: int) -> str:
    """Resolve a compiled MuJoCo name, preferring the authoritative API."""
    accessor = getattr(model, kind, None)
    if callable(accessor):
        try:
            name = getattr(accessor(int(index)), "name", None)
            if name:
                return str(name)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            pass
    resolver = getattr(model, f"{kind}_id2name", None)
    if callable(resolver):
        try:
            name = resolver(int(index))
            if name:
                return str(name)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            pass
    names = getattr(model, "names", None)
    if isinstance(names, Mapping):
        try:
            values = names.get(kind, ())
            if int(index) < len(values) and values[int(index)]:
                return str(values[int(index)])
        except (IndexError, TypeError, ValueError):
            pass
    values = getattr(model, f"{kind}_names", None)
    try:
        if values is not None and int(index) < len(values) and values[int(index)]:
            return str(values[int(index)])
    except (IndexError, TypeError, ValueError):
        pass
    # MuJoCo 3's compiled model may expose no names on the Python view.  Use
    # its official id2name API as the final source of truth.
    try:
        mujoco = importlib.import_module("mujoco")
        object_type = getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}")
        raw_model = getattr(model, "_model", model)
        name = mujoco.mj_id2name(raw_model, object_type, int(index))
        return str(name or "")
    except (ImportError, AttributeError, IndexError, KeyError, TypeError, ValueError):
        return ""


def _model_names(model: Any, kind: str) -> list[str]:
    count = _model_count(model, kind) or 0
    return [_model_name(model, kind, index) for index in range(count)]


def _array_shape(owner: Any, name: str) -> list[int] | None:
    value = getattr(owner, name, None)
    if value is None:
        return None
    try:
        return [int(item) for item in np.asarray(value).shape]
    except (TypeError, ValueError):
        return None


def _eef_site_mapping(env: Any, model: Any, names: Sequence[str]) -> dict[str, Any]:
    raw = getattr(env, "_env", env)
    robots = getattr(raw, "robots", None)
    selected: tuple[int, str] | None = None
    authoritative: dict[str, Any] = {}
    if isinstance(robots, (list, tuple)) and robots:
        robot = robots[0]
        eef_ids = getattr(robot, "eef_site_id", None)
        if isinstance(eef_ids, Mapping):
            raw_id = eef_ids.get("right")
            if raw_id is None and len(eef_ids) == 1:
                raw_id = next(iter(eef_ids.values()))
            try:
                site_id = int(raw_id)
                if site_id >= 0:
                    selected = (site_id, _model_name(model, "site", site_id))
            except (TypeError, ValueError):
                pass
        gripper = getattr(robot, "gripper", None)
        gripper_right = gripper.get("right") if isinstance(gripper, Mapping) else gripper
        authoritative = {
            "eef_site_id": _diagnostic_safe(eef_ids),
            "robot_model_eef_name": _diagnostic_safe(
                getattr(getattr(robot, "robot_model", None), "eef_name", None)
            ),
            "gripper_important_sites": _diagnostic_safe(
                getattr(gripper_right, "important_sites", None)
            ),
        }
    return {
        "authoritative": authoritative,
        "resolved": selected is not None,
        "site_id": None if selected is None else int(selected[0]),
        "site_name": None if selected is None else str(selected[1]),
        "source": "robots[0].eef_site_id['right']",
    }


def _current_eef_b0(env: Any, eef: Mapping[str, Any] | None = None) -> np.ndarray | None:
    observation = getattr(env, "_last_observation", None)
    if isinstance(observation, Mapping):
        for key in ("robot0_eef_pos", "robot0_base_to_eef_pos", "eef_pos"):
            try:
                value = np.asarray(observation[key], dtype=np.float64).reshape(-1)
            except (KeyError, TypeError, ValueError):
                continue
            if value.shape == (3,) and np.isfinite(value).all():
                return value
    raw = getattr(env, "_env", env)
    sim = getattr(raw, "sim", None) or getattr(env, "sim", None)
    data = getattr(sim, "data", None)
    site_id = None if eef is None else eef.get("site_id")
    if data is None or site_id is None:
        return None
    try:
        world = np.asarray(data.site_xpos[int(site_id)], dtype=np.float64).reshape(-1)
        transform = np.asarray(getattr(env, "_base_from_world_B0"), dtype=np.float64)
        value = (transform @ np.r_[world, 1.0])[:3]
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    return value if value.shape == (3,) and np.isfinite(value).all() else None


def _base_drift(env: Any) -> dict[str, Any]:
    observation = getattr(env, "_last_observation", None)
    result: dict[str, Any] = {"available": False, "translation_m": None, "rotation_angle_rad": None}
    if not isinstance(observation, Mapping):
        return result
    try:
        current_pos = np.asarray(observation["robot0_base_pos"], dtype=np.float64).reshape(-1)
        current_quat = np.asarray(observation["robot0_base_quat"], dtype=np.float64).reshape(-1)
        frozen = np.asarray(getattr(env, "_base_from_world_B0"), dtype=np.float64)
        if current_pos.shape != (3,) or current_quat.shape != (4,):
            return result
        current_world_from_base, _ = world_base_transform(
            {"robot0_base_pos": current_pos, "robot0_base_quat": current_quat}
        )
        translation = (frozen @ current_world_from_base)[:3, 3]
        relative_rotation = frozen[:3, :3] @ current_world_from_base[:3, :3]
        cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
        result.update({
            "available": bool(np.isfinite(translation).all()),
            "translation_m": translation.tolist(),
            "rotation_angle_rad": float(np.arccos(cosine)),
        })
    except (AttributeError, KeyError, TypeError, ValueError):
        return result
    return result


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = {
        "live.py": Path(__file__),
        "adapter.py": Path(__file__).with_name("adapter.py"),
        "runner.py": root / "arrow_grasp_controller" / "controller" / "runner.py",
        "episode_contract.py": root / "arrow_grasp_controller" / "controller" / "episode_contract.py",
        "policy.py": root / "arrow_grasp_controller" / "controller" / "policy.py",
        "canonical_config.json": root / "arrow_grasp_controller" / "configs" / "canonical_molmo_rgbd_grasp.json",
        "active_policy.lock.json": root / "arrow_grasp_controller" / "configs" / "active_policy.lock.json",
    }
    result: dict[str, str] = {}
    for name, path in paths.items():
        try:
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            result[name] = "missing"
    return result


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("robocasa", "robosuite", "mujoco", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _controller_runtime_metadata(raw: Any) -> dict[str, Any]:
    """Read the instantiated right-arm controller's actual runtime contract."""
    expected = {
        "name": "OSC_POSE",
        "input_ref_frame": "base",
        "input_type": "delta",
        "control_dim": 6,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    }
    actual: dict[str, Any] = {}
    robots = getattr(raw, "robots", None)
    robot = robots[0] if isinstance(robots, (list, tuple)) and robots else None
    controllers = getattr(robot, "part_controllers", None)
    controller = controllers.get("right") if isinstance(controllers, Mapping) else None
    if controller is not None:
        for field in ("name", "input_ref_frame", "input_type", "control_dim", "output_max", "output_min"):
            actual[field] = _diagnostic_safe(getattr(controller, field, None))
    matches: dict[str, bool] = {}
    for field, expected_value in expected.items():
        value = actual.get(field)
        if field in {"output_max", "output_min"}:
            try:
                matches[field] = bool(np.asarray(value, dtype=np.float64).shape == (6,)) and bool(
                    np.allclose(np.asarray(value, dtype=np.float64), expected_value)
                )
            except (TypeError, ValueError):
                matches[field] = False
        elif field == "name":
            matches[field] = str(value).upper() == str(expected_value).upper()
        else:
            matches[field] = value == expected_value
    return {
        "actual": actual,
        "expected": expected,
        "matches": matches,
        "contract_matches": bool(actual) and all(matches.values()),
    }


def _quaternion_xyzw_from_matrix(matrix: Any) -> np.ndarray | None:
    try:
        r = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(r).all():
            return None
        q = np.empty(4, dtype=np.float64)
        trace = float(np.trace(r))
        if trace > 0.0:
            s = 2.0 * np.sqrt(trace + 1.0)
            q[3], q[0], q[1], q[2] = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
        else:
            index = int(np.argmax(np.diag(r)))
            if index == 0:
                s = 2.0 * np.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 1e-12))
                q[3], q[0], q[1], q[2] = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
            elif index == 1:
                s = 2.0 * np.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 1e-12))
                q[3], q[0], q[1], q[2] = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 1e-12))
                q[3], q[0], q[1], q[2] = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
        q /= np.linalg.norm(q)
        return q if np.isfinite(q).all() else None
    except (TypeError, ValueError, FloatingPointError):
        return None


def _rotation_from_quaternion_xyzw(quaternion: Any) -> np.ndarray:
    """Return an SO(3) matrix for a finite RoboSuite/MuJoCo ``xyzw`` quat."""

    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise RoboCasaLiveError("EEF quaternion must be a finite 4-vector")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise RoboCasaLiveError("EEF quaternion must be non-zero")
    x, y, z, w = q / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _raw_robot_diagnostics(observation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy raw base pose and proprioception before B0 adaptation."""

    base: dict[str, Any] = {}
    proprio: dict[str, Any] = {}
    for key, value in observation.items():
        if key in {"robot0_base_pos", "robot0_base_quat"}:
            try:
                base[key] = np.asarray(value, dtype=np.float64).copy()
            except (TypeError, ValueError):
                base[key] = value
        elif str(key).startswith("robot0_") and any(
            marker in str(key).lower() for marker in ("eef", "gripper", "hand")
        ):
            try:
                proprio[str(key)] = np.asarray(value, dtype=np.float64).copy()
            except (TypeError, ValueError):
                proprio[str(key)] = value
    return base, proprio


def _eef_site_runtime_evidence(env: Any, model: Any, data: Any, eef: Mapping[str, Any]) -> dict[str, Any]:
    """Compare compiled grip-site/hand poses with raw world-frame sensors."""

    result: dict[str, Any] = {"available": False, "matches": False}
    try:
        site_id = int(eef["site_id"])
        grip_site_world = np.asarray(data.site_xpos[site_id], dtype=np.float64).reshape(-1)
        raw_base = getattr(env, "_last_raw_base", {})
        raw_proprio = getattr(env, "_last_raw_proprio", {})
        base_pos = np.asarray(raw_base["robot0_base_pos"], dtype=np.float64).reshape(-1)
        base_quat = np.asarray(raw_base["robot0_base_quat"], dtype=np.float64).reshape(-1)
        eef_pos = np.asarray(raw_proprio["robot0_base_to_eef_pos"], dtype=np.float64).reshape(-1)
        eef_quat = np.asarray(raw_proprio["robot0_base_to_eef_quat"], dtype=np.float64).reshape(-1)
        current_world_from_base, _ = world_base_transform({
            "robot0_base_pos": base_pos,
            "robot0_base_quat": base_quat,
        })
        expected_world = (current_world_from_base @ np.r_[eef_pos, 1.0])[:3]
        expected_rot = current_world_from_base[:3, :3] @ _rotation_from_quaternion_xyzw(eef_quat)
        # RoboSuite's base-to-EEF quaternion is the hand/body orientation,
        # while the position is the authoritative grip site.  Compare those
        # against their corresponding compiled runtime sources instead of
        # assuming the grip site's orientation has no fixed tool offset.
        hand_xmat = None
        hand_rotation_source = "robot_model.eef_name['right']"
        raw = getattr(env, "_env", env)
        robots = getattr(raw, "robots", None)
        robot = robots[0] if isinstance(robots, (list, tuple)) and robots else None
        eef_names = getattr(getattr(robot, "robot_model", None), "eef_name", None)
        if isinstance(eef_names, Mapping):
            hand_name = eef_names.get("right")
        else:
            hand_name = eef_names
        body_id = None
        resolver = getattr(model, "body_name2id", None)
        if callable(resolver) and hand_name:
            body_id = int(resolver(str(hand_name)))
        if body_id is not None:
            hand_xmat = np.asarray(data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)
        if hand_xmat is None:
            hand_xmat = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
            hand_rotation_source = "grip_site_xmat_fallback"
        compiled_quat = _quaternion_xyzw_from_matrix(hand_xmat)
        expected_quat = _quaternion_xyzw_from_matrix(expected_rot)
        if (
            grip_site_world.shape != (3,)
            or not np.isfinite(grip_site_world).all()
            or compiled_quat is None
            or expected_quat is None
        ):
            raise ValueError("invalid compiled EEF site/hand pose")
        position_error = float(np.linalg.norm(grip_site_world - expected_world))
        rotation_abs_dot = float(abs(np.dot(compiled_quat, expected_quat)))
        result.update({
            "available": True,
            "matches": bool(position_error <= 1e-5 and rotation_abs_dot >= 1.0 - 1e-5),
            "grip_site_id": site_id,
            "grip_site_world_m": grip_site_world.tolist(),
            "expected_world_from_base_sensor_m": expected_world.tolist(),
            "position_error_m": position_error,
            "grip_site_xmat_shape": list(np.asarray(data.site_xmat).shape),
            "hand_xmat_shape": list(np.asarray(data.body_xmat).shape) if hasattr(data, "body_xmat") else None,
            "hand_rotation_source": hand_rotation_source,
            "hand_rotation_sensor": "robot0_base_to_eef_quat",
            "rotation_abs_dot": rotation_abs_dot,
        })
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ZeroDivisionError, RoboCasaLiveError):
        return result
    return result


def _base_center_evidence(env: Any, model: Any, data: Any) -> dict[str, Any]:
    """Compare reset sensors against the compiled mobilebase center site."""
    result: dict[str, Any] = {"available": False, "matches": False, "source": "robot_model.base.correct_naming('center')"}
    raw = getattr(env, "_env", env)
    robots = getattr(raw, "robots", None)
    observation = getattr(env, "_last_observation", None)
    try:
        robot = robots[0]
        base = getattr(getattr(robot, "robot_model", None), "base", None)
        correct = getattr(base, "correct_naming", None)
        name = str(correct("center")) if callable(correct) else ""
        resolver = getattr(model, "site_name2id", None)
        site_id = int(resolver(name)) if callable(resolver) else None
        if site_id is None:
            names = _model_names(model, "site")
            site_id = names.index(name)
        compiled_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).reshape(-1)
        compiled_rot = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        observed_pos = np.asarray(observation["robot0_base_pos"], dtype=np.float64).reshape(-1)
        observed_quat = np.asarray(observation["robot0_base_quat"], dtype=np.float64).reshape(-1)
        compiled_quat = _quaternion_xyzw_from_matrix(compiled_rot)
        if compiled_pos.shape != (3,) or observed_pos.shape != (3,) or observed_quat.shape != (4,) or compiled_quat is None:
            raise ValueError("invalid base center/site sensor shapes")
        position_error = float(np.linalg.norm(compiled_pos - observed_pos))
        quaternion_abs_dot = float(abs(np.dot(compiled_quat, observed_quat / np.linalg.norm(observed_quat))))
        result.update({
            "available": True,
            "site_name": name,
            "site_id": site_id,
            "compiled_site_xpos_m": compiled_pos.tolist(),
            "observed_base_pos_m": observed_pos.tolist(),
            "compiled_site_quat_xyzw": compiled_quat.tolist(),
            "observed_base_quat_xyzw": observed_quat.tolist(),
            "position_error_m": position_error,
            "quaternion_abs_dot": quaternion_abs_dot,
            "matches": bool(position_error <= 1e-5 and quaternion_abs_dot >= 1.0 - 1e-5),
            "xmat_shape": list(np.asarray(data.site_xmat).shape),
        })
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return result
    return result


def collect_pre_motion_diagnostics(
    env: RoboCasaControllerEnv,
    *,
    output_dir: Path | None = None,
    task_name: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Collect read-only runtime and calibration evidence before Molmo loads."""
    raw = getattr(env, "_env", env)
    sim = getattr(raw, "sim", None) or getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    names = _model_names(model, "site") if model is not None else []
    eef = _eef_site_mapping(env, model, names) if model is not None else {
        "authoritative": {}, "resolved": False,
        "site_id": None, "site_name": None, "source": "robots[0].eef_site_id['right']",
    }
    important = [
        {"id": index, "name": name}
        for index, name in enumerate(names)
        if any(marker in name.lower() for marker in ("grip", "finger", "hand", "eef", "pad"))
    ]
    robot_mappings: list[dict[str, Any]] = []
    robots = getattr(raw, "robots", None)
    if isinstance(robots, (list, tuple)):
        for robot_index, robot in enumerate(robots):
            eef_ids = getattr(robot, "eef_site_id", None)
            gripper = getattr(robot, "gripper", None)
            important_sites = None
            if isinstance(gripper, Mapping):
                important_sites = {
                    str(key): _diagnostic_safe(getattr(value, "important_sites", value))
                    for key, value in gripper.items()
                }
            elif gripper is not None:
                important_sites = _diagnostic_safe(getattr(gripper, "important_sites", None))
            robot_mappings.append({
                "robot_index": robot_index,
                "eef_site_id": _diagnostic_safe(eef_ids),
                "robot_model_eef_name": _diagnostic_safe(
                    getattr(getattr(robot, "robot_model", None), "eef_name", None)
                ),
                "gripper_important_sites": important_sites,
            })
    result: dict[str, Any] = {
        "schema": "robocasa_runtime_diagnostics.v1",
        "task": task_name,
        "seed": None if seed is None else int(seed),
        "runtime": {
            "model_class": _runtime_class(model),
            "data_class": _runtime_class(data),
            "versions": _runtime_versions(),
            "model_counts": {
                kind: _model_count(model, kind) if model is not None else None
                for kind in ("body", "site", "geom", "joint", "actuator", "sensor")
            },
            "important_gripper_sites": important,
            "robot_eef_mappings": robot_mappings,
            "eef_mapping": eef,
            "data_shapes": {
                name: _array_shape(data, name) if data is not None else None
                for name in ("site_xpos", "site_xmat", "body_xmat", "geom_xmat", "body_xpos", "geom_xpos")
            },
        },
        "b0": {
            "base_from_world": _diagnostic_safe(getattr(env, "_base_from_world_B0", None)),
            "world_from_base": _diagnostic_safe(getattr(env, "_world_from_base_B0", None)),
            "base_drift": _base_drift(env),
            "raw_base_pose": _diagnostic_safe(getattr(env, "_last_raw_base", {})),
            "raw_proprioception": _diagnostic_safe(getattr(env, "_last_raw_proprio", {})),
            "base_center_comparison": _base_center_evidence(env, model, data) if model is not None and data is not None else {
                "available": False, "matches": False, "source": "robot_model.base.correct_naming('center')"
            },
            "compiled_eef_comparison": _eef_site_runtime_evidence(env, model, data, eef)
            if model is not None and data is not None and eef.get("resolved") else {
                "available": False, "matches": False
            },
        },
        "controller": {
            "input_dimension": 7,
            "canonical_action": "[dx,dy,dz,rx,ry,rz,gripper]",
            "osc_scales": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "osc_position_scale_m_override": None,
            "action_layout": _diagnostic_safe(env._action_adapter.audit()),
            "runtime_controller": _controller_runtime_metadata(raw),
            "source_hashes": _source_hashes(),
        },
        "calibration_passed": False,
        "calibration": None,
        "errors": [],
    }
    try:
        from ..arrow_grasp_controller.controller.runner import probe_robot_calibration
        _calibration, transform, record = probe_robot_calibration(env)
        result["calibration_passed"] = bool(record.get("passed", True)) if isinstance(record, Mapping) else True
        result["calibration"] = _diagnostic_safe(record)
        result["calibration_transform"] = _diagnostic_safe(transform)
    except Exception as exc:
        result["errors"].append({"stage": "calibration", "type": type(exc).__name__, "error": str(exc)})
        probe_record = getattr(env, "_grasp_controller_robot_calibration_probe", None)
        if probe_record is not None:
            result["calibration"] = _diagnostic_safe(probe_record)
    result["eef_initial_b0_m"] = _diagnostic_safe(_current_eef_b0(env, eef))
    if output_dir is not None:
        _write_diagnostic_json(Path(output_dir) / "preflight_diagnostics.json", result)
    return result


def begin_motion_diagnostics(env: RoboCasaControllerEnv) -> dict[str, Any]:
    """Snapshot reset-time EEF/base state; this function performs no steps."""
    return {
        "initial_eef_b0_m": _diagnostic_safe(_current_eef_b0(env)),
        "initial_base_drift": _base_drift(env),
        "initial_raw_base_pose": _diagnostic_safe(getattr(env, "_last_raw_base", {})),
        "initial_raw_proprioception": _diagnostic_safe(getattr(env, "_last_raw_proprio", {})),
        "probe_action_count": 0,
    }


# Stable descriptive alias for callers that do not need the pre-motion name.
collect_runtime_diagnostics = collect_pre_motion_diagnostics


def finalize_motion_diagnostics(
    env: RoboCasaControllerEnv,
    initial: Mapping[str, Any] | None,
    *,
    output_dir: Path | None = None,
    outcome: str | None = None,
) -> dict[str, Any]:
    """Summarize all wrapper commands and EEF motion, including failures."""
    start = None
    if isinstance(initial, Mapping):
        try:
            start = np.asarray(initial.get("initial_eef_b0_m"), dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            start = None
    trajectory: list[list[float]] = []
    if start is not None and np.isfinite(start).all():
        trajectory.append(start.tolist())
    for command in getattr(env, "_action_history", ()):
        position = command.get("eef_after_b0_m")
        if position is not None:
            try:
                value = np.asarray(position, dtype=np.float64).reshape(3)
                if np.isfinite(value).all():
                    trajectory.append(value.tolist())
            except (TypeError, ValueError):
                pass
    final = _current_eef_b0(env)
    if final is not None and (not trajectory or not np.allclose(final, trajectory[-1])):
        trajectory.append(final.tolist())
    points = np.asarray(trajectory, dtype=np.float64) if trajectory else np.empty((0, 3))
    displacement = None
    max_displacement = None
    if start is not None and points.size:
        distances = np.linalg.norm(points - start.reshape(1, 3), axis=1)
        displacement = float(np.linalg.norm(points[-1] - start))
        max_displacement = float(np.max(distances))
    commands = list(getattr(env, "_action_history", ()))
    arm_count = sum(bool(item.get("arm_command")) and item.get("status") == "sent" for item in commands)
    gripper_only_count = sum(bool(item.get("gripper_only_command")) and item.get("status") == "sent" for item in commands)
    result = {
        "schema": "robocasa_motion_diagnostics.v2",
        "outcome": outcome,
        "initial_eef_b0_m": _diagnostic_safe(start),
        "final_eef_b0_m": _diagnostic_safe(final),
        "eef_trajectory_b0_m": trajectory,
        "eef_displacement_m": displacement,
        "max_eef_displacement_m": max_displacement,
        "action_count": int(sum(item.get("status") == "sent" for item in commands)),
        "attempted_action_count": len(commands),
        "arm_command_count": int(arm_count),
        "gripper_only_command_count": int(gripper_only_count),
        "base_drift_initial": _diagnostic_safe(initial.get("initial_base_drift") if isinstance(initial, Mapping) else None),
        "base_drift_final": _base_drift(env),
        "raw_base_pose_initial": _diagnostic_safe(initial.get("initial_raw_base_pose") if isinstance(initial, Mapping) else None),
        "raw_proprioception_initial": _diagnostic_safe(initial.get("initial_raw_proprioception") if isinstance(initial, Mapping) else None),
        "raw_base_pose_final": _diagnostic_safe(getattr(env, "_last_raw_base", {})),
        "raw_proprioception_final": _diagnostic_safe(getattr(env, "_last_raw_proprio", {})),
        "actions": _diagnostic_safe(commands),
        "probe_action_count": 0,
    }
    if output_dir is not None:
        _write_diagnostic_json(Path(output_dir) / "motion_diagnostics.json", result)
    return result


def _write_diagnostic_json(path: Path, payload: Mapping[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(_diagnostic_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        return True
    except Exception:
        # Diagnostics must never replace or mask the controller outcome.
        return False


def _save_projection_failure_rgb(capture: Any, output_dir: Path) -> dict[str, Any]:
    """Persist the already captured RGB frame when role projection fails."""

    path = Path(output_dir) / "projection_failure.png"
    try:
        from PIL import Image

        image = np.asarray(capture.rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"captured RGB must be HxWx3, got {image.shape}")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGB").save(path)
        return {"path": path.as_posix(), "saved": True}
    except Exception as exc:
        return {
            "path": path.as_posix(),
            "saved": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_live_cell(
    *, task_name: str, seed: int, output_dir: Path, resolution: int = 256,
    execute_motion: bool = True, molmo_runtime: Any | None = None,
    grasp_profile: str = "canonical_rim",
) -> dict[str, Any]:
    """Run one cell through environment reset, arrow generation, and motion.

    The final motion call goes through the RoboCasa-local controller boundary.
    No RoboCasa state is passed to the motion policy; it receives only the
    aligned capture and arrow.
    """

    if grasp_profile not in {"canonical_rim", "object_contact_v1", "object_contact_v2", "object_contact_v3", "object_contact_v4", "object_contact_v5"}:
        raise ValueError(f"unknown RoboCasa grasp profile: {grasp_profile!r}")
    task = get_task(task_name)
    raw_env = create_robocasa_env(task_name, seed=seed)
    env = RoboCasaControllerEnv(raw_env)
    motion_initial: Mapping[str, Any] | None = None
    preflight: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    try:
        env.reset()
        motion_initial = begin_motion_diagnostics(env)
        # This is intentionally before capture, Molmo construction, or any
        # controller action.  The calibration helper is no-motion and its
        # complete record is persisted even when the cell cannot continue.
        preflight = collect_pre_motion_diagnostics(
            env, output_dir=output_dir, task_name=task_name, seed=seed
        )
        if not bool(preflight.get("calibration_passed")):
            result = {
                "task": task_name,
                "compatibility_task_id": 0,
                "seed": int(seed),
                "status": "controller_failure",
                "terminal_reason": "calibration_preflight_failed",
                "error": "read-only Panda calibration probe failed",
                "pre_motion_diagnostics": preflight,
                "actions_executed": env.steps,
                "official_success": False,
            }
            return result
        capture = None
        bboxes = None
        source_key = None
        selected_physical_camera = None
        valid_camera_candidates: list[
            tuple[float, str, Any, dict[str, tuple[float, float, float, float]], str]
        ] = []
        projection_attempts: list[dict[str, Any]] = []
        for physical_camera in ROBOCASA_CAMERA_FALLBACKS:
            try:
                candidate_capture = _capture(
                    env, resolution=resolution, camera_name=physical_camera
                )
            except Exception as exc:
                projection_attempts.append({
                    "camera_name": physical_camera,
                    "stage": "capture",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            try:
                candidate_bboxes, candidate_source = project_task_bboxes(
                    env, candidate_capture, task
                )
            except Exception as exc:
                projection_attempts.append({
                    "camera_name": physical_camera,
                    "stage": "projection",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                # Preserve the most informative frame for diagnostics if all
                # declared agent views fail the visibility contract.
                if capture is None:
                    capture = candidate_capture
                continue
            visible_area = sum(
                max(0.0, float(box[2]) - float(box[0]))
                * max(0.0, float(box[3]) - float(box[1]))
                for box in candidate_bboxes.values()
            )
            valid_camera_candidates.append(
                (
                    visible_area,
                    physical_camera,
                    candidate_capture,
                    candidate_bboxes,
                    candidate_source,
                )
            )
        if valid_camera_candidates:
            # Preserve LIBERO's primary ``agentview`` whenever its role
            # projections are valid.  The alternate physical view is a
            # visibility fallback, not an optimization target: choosing by
            # projected area can select a camera whose calibration is valid
            # yet whose source object is visually occluded by the robot.
            selected = next(
                (item for item in valid_camera_candidates if item[1] == ROBOCASA_CAMERA),
                valid_camera_candidates[0],
            )
            _, selected_physical_camera, capture, bboxes, source_key = selected
        if capture is None or bboxes is None or source_key is None:
            projection_error = "; ".join(
                f"{item['camera_name']}: {item['error']}"
                for item in projection_attempts
            ) or "no agent-view capture was available"
            projection_failure = (
                _save_projection_failure_rgb(capture, Path(output_dir))
                if capture is not None else None
            )
            result = {
                "task": task_name,
                "compatibility_task_id": 0,
                "seed": int(seed),
                "status": "controller_failure",
                "terminal_reason": "all_agent_views_failed_projection",
                "error": projection_error,
                "pre_motion_diagnostics": preflight,
                "actions_executed": env.steps,
                "official_success": False,
                "projection_failure": projection_failure,
                "projection_attempts": projection_attempts,
            }
            return result
        # The motion engine's contract remains ``agentview``.  Preserve the
        # physical camera in provenance while presenting the selected view
        # through the same controller-facing calibration identity used by
        # LIBERO.
        env._arrow_physical_camera_name = str(selected_physical_camera)
        calibration = getattr(capture, "calibration", None)
        if calibration is None:
            raise RoboCasaLiveError("selected RoboCasa capture has no calibration")
        if str(getattr(calibration, "camera_name", "")) != "agentview":
            capture = replace(
                capture,
                calibration=replace(calibration, camera_name="agentview"),
            )
        arrow_rgb, arrow_audit = render_bbox_center_arrow(
            np.asarray(capture.rgb, dtype=np.uint8), bboxes,
            source=source_key, destination=task.destination.key,
            allow_fallback=False,
        )
        source_label = _source_label(env, task, source_key)
        if grasp_profile in {"object_contact_v1", "object_contact_v2", "object_contact_v3", "object_contact_v4", "object_contact_v5"}:
            from .prompt import object_contact_prompt
            effective_source_prompt = object_contact_prompt(source_label)
        else:
            effective_source_prompt = adapt_source_noun(source_label)
        result = {
            "task": task_name,
            # The local high-level episode contract retains LIBERO's internal
            # task id for compatibility; the outer task identity is RoboCasa.
            "compatibility_task_id": 0,
            "seed": int(seed),
            "source_key": source_key,
            "destination_key": task.destination.key,
            "official_success_rule": task.success_rule,
            "prompt": effective_source_prompt,
            "grasp_profile": grasp_profile,
            "arrow_audit": arrow_audit,
            "bboxes": {str(key): [float(value) for value in bbox] for key, bbox in bboxes.items()},
            "bbox_sha256": hashlib.sha256(repr(sorted(bboxes.items())).encode()).hexdigest(),
            "capture_provenance": {
                "camera_name": str(capture.calibration.camera_name),
                "physical_camera_name": str(selected_physical_camera),
                "camera_selection_policy": "left_then_right_if_role_projection_fails",
                "projection_attempts": projection_attempts,
                "resolution": [int(capture.calibration.width), int(capture.calibration.height)],
                "world_frame": str(capture.calibration.world_frame),
                "depth_conversion_mode": str(getattr(capture, "depth_conversion_mode", "unknown")),
                "image_frame": "post_flip_xy",
            },
            "action_layout": env._action_adapter.audit(),
            "pre_motion_diagnostics": preflight,
            "horizon": env.horizon,
            "actions_executed_before_motion": env.steps,
            "official_success": False,
        }
        if not execute_motion:
            result.update({"status": "preflight_complete", "terminal_reason": "motion_not_requested"})
            return result
        def arrow_refresh_builder(controller_env: RoboCasaControllerEnv, controller_capture: Any):
            """Reproject current roles against the exact fresh RGB-D frame."""
            if controller_env._world_from_base_B0 is None:
                raise RoboCasaLiveError("cannot refresh arrow before freezing B0")
            world_capture = adapt_capture_to_base(
                controller_capture, controller_env._world_from_base_B0
            )
            # ``adapt_capture_to_base`` is also used for the controller's B0
            # view and therefore labels its result as base_from_camera.  This
            # refresh path intentionally converts back to world coordinates;
            # preserve the matrices while correcting only their provenance.
            calibration = getattr(world_capture, "calibration", None)
            if calibration is not None and hasattr(calibration, "world_frame"):
                calibration = replace(
                    calibration,
                    world_frame="robocasa_mujoco_world",
                    extrinsic_direction="world_from_camera",
                )
                world_capture = replace(world_capture, calibration=calibration)
            refreshed_bboxes, refreshed_source = project_task_bboxes(
                controller_env, world_capture, task
            )
            refreshed_arrow, _ = render_bbox_center_arrow(
                np.asarray(controller_capture.rgb, dtype=np.uint8),
                refreshed_bboxes,
                source=refreshed_source,
                destination=task.destination.key,
                allow_fallback=False,
            )
            return refreshed_arrow, refreshed_bboxes, refreshed_source, task.destination.key
        try:
            from ..arrow_grasp_controller.controller.runner import run_episode
            if env._base_from_world_B0 is None:
                raise RoboCasaLiveError("reset did not establish frozen PandaOmron base frame")
            controller_capture = adapt_capture_to_base(capture, env._base_from_world_B0)
            audit = run_episode(
                env=env,
                seed=seed,
                output_dir=output_dir,
                arrow_rgb=arrow_rgb,
                bboxes=bboxes,
                source=source_key,
                destination=task.destination.key,
                resolution=resolution,
                capture=controller_capture,
                evaluator=official_success,
                arrow_refresh_builder=arrow_refresh_builder,
                source_prompt=effective_source_prompt,
                molmo_runtime=molmo_runtime,
                grasp_profile=grasp_profile,
            )
            evaluator_called, succeeded = _official_outcome_from_controller_audit(audit)
            result.update({
                "status": "success" if succeeded else "task_failure",
                "terminal_reason": (
                    "official_success_evaluated"
                    if evaluator_called
                    else "controller_completed_without_official_evaluation"
                ),
                "audit": audit,
                "official_success": succeeded,
                "actions_executed": env.steps,
            })
        except Exception as exc:
            status = "horizon_exhaustion" if isinstance(exc, RoboCasaHorizonError) else "controller_failure"
            result.update({"status": status, "terminal_reason": type(exc).__name__, "error": str(exc), "actions_executed": env.steps})
        return result
    except Exception as exc:
        if result is None:
            result = {
                "task": task_name,
                "compatibility_task_id": 0,
                "seed": int(seed),
                "status": "controller_failure",
                "terminal_reason": type(exc).__name__,
                "error": str(exc),
                "pre_motion_diagnostics": preflight,
                "actions_executed": env.steps,
                "official_success": False,
            }
        return result
    finally:
        if motion_initial is not None:
            try:
                motion = finalize_motion_diagnostics(
                    env, motion_initial, output_dir=output_dir,
                    outcome=None if result is None else result.get("status"),
                )
                if result is not None:
                    result["motion_diagnostics"] = motion
            except Exception as exc:
                if result is not None:
                    result.setdefault("diagnostic_errors", []).append({
                        "stage": "motion_diagnostics",
                        "type": type(exc).__name__,
                        "error": str(exc),
                    })
        close = getattr(raw_env, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                close_error = {
                    "stage": "close",
                    "type": type(exc).__name__,
                    "error": str(exc),
                }
                if result is not None:
                    result.setdefault("diagnostic_errors", []).append(close_error)
                    motion = result.get("motion_diagnostics")
                    if isinstance(motion, Mapping):
                        motion_with_close_error = dict(motion)
                        motion_with_close_error["close_error"] = close_error
                        result["motion_diagnostics"] = motion_with_close_error
                        _write_diagnostic_json(
                            Path(output_dir) / "motion_diagnostics.json",
                            motion_with_close_error,
                        )


__all__ = [
    "RoboCasaControllerEnv",
    "RoboCasaLiveError",
    "RoboCasaHorizonError",
    "official_success",
    "project_task_bboxes",
    "collect_pre_motion_diagnostics",
    "collect_runtime_diagnostics",
    "begin_motion_diagnostics",
    "finalize_motion_diagnostics",
    "run_live_cell",
]
