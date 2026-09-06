"""Live RoboCasa bridge for the existing arrow controller.

All RoboCasa and MuJoCo imports are lazy.  The bridge keeps simulator-only
role resolution and projection on the input side, then passes only an RGB-D
capture plus a single rendered arrow to the unchanged motion engine.
The wrapped environment converts the engine's unchanged 7D OSC command into
RoboCasa's official 12D PandaOmron dictionary action.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..environment.runtime import create_robocasa_env, validate_action_layout
from ..shared.task_manifest import PickPlaceTask, RoleSpec, get_task
from .adapter import PandaOmronActionAdapter, ROBOCASA_CAMERA
from .arrow import render_bbox_center_arrow
from .prompt import adapt_source_noun


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
    return value


def _role_entity(env: Any, role: RoleSpec) -> Any:
    key = str(role.key)
    if key.startswith("self."):
        value = getattr(env, key[5:], None)
        if value is not None:
            return value
        return _lookup_fixture(env, key[5:])
    if role.kind == "object":
        return _lookup_object(env, key)
    return _lookup_fixture(env, key)


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
                        raise RoboCasaLiveError(
                            f"role {role.key!r} requested region {requested!r}, "
                            f"available={sorted(str(name) for name in regions)}"
                        )
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
            points = np.asarray(regions, dtype=np.float64)
    elif callable(getattr(entity, "get_bbox_points", None)):
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
    x1, x2 = float(np.clip(np.min(u), 0, width - 1)), float(np.clip(np.max(u), 0, width - 1))
    y1, y2 = float(np.clip(np.min(v), 0, height - 1)), float(np.clip(np.max(v), 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        raise RoboCasaLiveError("role bbox has no visible area after projection")
    return x1, y1, x2, y2


def _capture(env: Any, *, resolution: int) -> Any:
    """Capture one aligned RoboCasa RGB-D pair with exactly one vertical flip."""
    try:
        from .capture import capture_robocasa_rgbd
        return capture_robocasa_rgbd(
            env, camera_name=ROBOCASA_CAMERA, resolution=int(resolution)
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
        bboxes[key] = _project_bbox(points, capture, width=int(capture.rgb.shape[1]), height=int(capture.rgb.shape[0]))
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
        self._arrow_settle_diagnostics = {"settled": True, "source": "robocasa_reset"}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def last_info(self) -> Mapping[str, Any]:
        return self._last_info

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        result = self._env.reset(*args, **kwargs)
        self._steps = 0
        observation = result[0] if isinstance(result, tuple) and result else result
        if isinstance(observation, Mapping):
            self._last_observation = observation
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result

    def step(self, action: Sequence[float]) -> Any:
        if self._horizon is not None and self._steps >= self._horizon:
            raise RoboCasaHorizonError(
                f"RoboCasa task horizon exhausted at {self._steps}/{self._horizon} actions"
            )
        packed = self._action_adapter.pack(action)
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
        result = self._env.step(action_dict if use_dict else np.asarray(packed, dtype=np.float32))
        self._steps += 1
        observation = result[0] if isinstance(result, tuple) and result else result
        if isinstance(observation, Mapping):
            self._last_observation = observation
        if isinstance(result, tuple) and result and isinstance(result[-1], Mapping):
            self._last_info = dict(result[-1])
        return result

    def render(self, *args: Any, **kwargs: Any) -> Any:
        camera = kwargs.get("camera_name")
        if camera == "agentview":
            kwargs["camera_name"] = ROBOCASA_CAMERA
        sim = getattr(self._env, "sim", None)
        render_owner = sim if sim is not None else self._env
        if sim is not None:
            width = int(kwargs.pop("width", 256))
            height = int(kwargs.pop("height", 256))
            kwargs.pop("depth", None)
            result = render_owner.render(height=height, width=width, **kwargs)
        else:
            result = render_owner.render(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 2:
            return (np.asarray(result[0])[::-1].copy(), np.asarray(result[1])[::-1].copy(), *result[2:])
        return np.asarray(result)[::-1].copy()

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


def run_live_cell(*, task_name: str, seed: int, output_dir: Path, resolution: int = 256, execute_motion: bool = True) -> dict[str, Any]:
    """Run one cell through environment reset, arrow generation, and motion.

    The final motion call goes through the RoboCasa-local controller boundary.
    No RoboCasa state is passed to the motion policy; it receives only the
    aligned capture and arrow.
    """

    task = get_task(task_name)
    raw_env = create_robocasa_env(task_name, seed=seed)
    env = RoboCasaControllerEnv(raw_env)
    try:
        env.reset()
        capture = _capture(env, resolution=resolution)
        bboxes, source_key = project_task_bboxes(env, capture, task)
        arrow_rgb, arrow_audit = render_bbox_center_arrow(
            np.asarray(capture.rgb, dtype=np.uint8), bboxes,
            source=source_key, destination=task.destination.key,
            allow_fallback=False,
        )
        source_label = _source_label(env, task, source_key)
        result: dict[str, Any] = {
            "task": task_name,
            "seed": int(seed),
            "source_key": source_key,
            "destination_key": task.destination.key,
            "official_success_rule": task.success_rule,
            "prompt": adapt_source_noun(source_label),
            "arrow_audit": arrow_audit,
            "bboxes": {str(key): [float(value) for value in bbox] for key, bbox in bboxes.items()},
            "bbox_sha256": hashlib.sha256(repr(sorted(bboxes.items())).encode()).hexdigest(),
            "capture_provenance": {
                "camera_name": str(capture.calibration.camera_name),
                "resolution": [int(capture.calibration.width), int(capture.calibration.height)],
                "world_frame": str(capture.calibration.world_frame),
                "depth_conversion_mode": str(getattr(capture, "depth_conversion_mode", "unknown")),
                "image_frame": "post_flip_xy",
            },
            "action_layout": env._action_adapter.audit(),
            "horizon": env.horizon,
            "actions_executed_before_motion": env.steps,
            "official_success": False,
        }
        if not execute_motion:
            result.update({"status": "preflight_complete", "terminal_reason": "motion_not_requested"})
            return result
        try:
            from ..arrow_grasp_controller.controller.runner import run_episode
            audit = run_episode(
                env=env,
                seed=seed,
                output_dir=output_dir,
                arrow_rgb=arrow_rgb,
                bboxes=bboxes,
                source=source_key,
                destination=task.destination.key,
                resolution=resolution,
                capture=capture,
                evaluator=official_success,
            )
            result.update({"status": "success" if bool(audit.get("evaluator_success")) else "task_failure", "terminal_reason": "official_success_evaluated", "audit": audit, "official_success": bool(audit.get("evaluator_success")), "actions_executed": env.steps})
        except Exception as exc:
            status = "horizon_exhaustion" if isinstance(exc, RoboCasaHorizonError) else "controller_failure"
            result.update({"status": status, "terminal_reason": type(exc).__name__, "error": str(exc), "actions_executed": env.steps})
        return result
    finally:
        close = getattr(raw_env, "close", None)
        if callable(close):
            close()


__all__ = [
    "RoboCasaControllerEnv",
    "RoboCasaLiveError",
    "RoboCasaHorizonError",
    "official_success",
    "project_task_bboxes",
    "run_live_cell",
]
