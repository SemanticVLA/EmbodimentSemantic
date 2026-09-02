"""Dependency-free contracts for the isolated ZeroGrasp experiment.

The LIBERO controller is intentionally only an RGB-D client of the external
ZeroGrasp process.  Keeping these dataclasses small and strict makes it hard
to accidentally pass simulator state, task metadata, or evaluator results to
the learned component.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# These tags are part of the process-boundary schema, not descriptive
# diagnostics.  ZeroGrasp emits GraspNet poses whose rotation and translation
# are both expressed in the captured camera frame.  Keeping one exact value
# for each component prevents a consumer from silently interpreting a worker
# pose as world-, palm-, or EEF-frame geometry.
GRASPNET_CAMERA_FRAME = "camera_graspnet"
GRASPNET_TRANSLATION_REFERENCE = "grasp_center"


def _finite_array(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite numeric values")
    return arr


def _pixel(value: Sequence[float], name: str) -> tuple[float, float]:
    arr = _finite_array(value, name).reshape(-1)
    if arr.size != 2:
        raise ValueError(f"{name} must contain exactly two values (u, v)")
    return float(arr[0]), float(arr[1])


def validate_se3(value: Any, name: str = "transform") -> np.ndarray:
    arr = _finite_array(value, name, (4, 4)).astype(np.float64)
    if not np.allclose(arr[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    rotation = arr[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation must be a proper orthonormal matrix")
    return arr


@dataclass(frozen=True)
class ZeroGraspObservation:
    """The complete, permitted controller observation.

    Deliberately no ``task_id``, bbox, object pose, environment, simulator,
    scene graph, or evaluator fields exist on this contract.
    """

    clean_rgb: np.ndarray
    arrow_rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    T_world_camera: np.ndarray
    source_px: tuple[float, float]
    destination_px: tuple[float, float]
    gripper_qpos: np.ndarray | None = None
    eef_position_world_m: np.ndarray | None = None
    eef_quaternion_right_hand_xyzw: np.ndarray | None = None

    def __post_init__(self) -> None:
        rgb = np.asarray(self.clean_rgb)
        arrow = np.asarray(self.arrow_rgb)
        if rgb.dtype != np.uint8 or arrow.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise TypeError("clean_rgb and arrow_rgb must be HxWx3 uint8 arrays")
        if arrow.shape != rgb.shape:
            raise ValueError("clean_rgb and arrow_rgb must be aligned and identically shaped")
        depth = np.asarray(self.depth_m)
        if depth.ndim != 2 or depth.shape != rgb.shape[:2] or not np.issubdtype(depth.dtype, np.floating):
            raise TypeError("depth_m must be an aligned HxW floating-point metric-depth array")
        if np.any(np.isinf(depth)) or np.any(depth < 0):
            raise ValueError("depth_m may contain NaN invalid samples but no infinities or negative values")
        K = _finite_array(self.K, "K", (3, 3)).astype(np.float64)
        if np.isclose(K[0, 0], 0) or np.isclose(K[1, 1], 0):
            raise ValueError("K focal lengths must be non-zero")
        validate_se3(self.T_world_camera, "T_world_camera")
        h, w = rgb.shape[:2]
        for name, px in (("source_px", self.source_px), ("destination_px", self.destination_px)):
            u, v = _pixel(px, name)
            if not (0 <= u < w and 0 <= v < h):
                raise ValueError(f"{name} must lie inside the image")
        if self.gripper_qpos is not None:
            q = _finite_array(self.gripper_qpos, "gripper_qpos").reshape(-1)
            if q.size == 0:
                raise ValueError("gripper_qpos cannot be empty")
        if self.eef_position_world_m is not None:
            position = _finite_array(self.eef_position_world_m, "eef_position_world_m").reshape(-1)
            if position.size != 3:
                raise ValueError("eef_position_world_m must be a finite world-frame 3-vector")
        if self.eef_quaternion_right_hand_xyzw is not None:
            quaternion = _finite_array(self.eef_quaternion_right_hand_xyzw, "eef_quaternion_right_hand_xyzw").reshape(-1)
            if quaternion.size != 4 or not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-5):
                raise ValueError("eef_quaternion_right_hand_xyzw must be a unit xyzw quaternion")


ControllerObservation = ZeroGraspObservation


@dataclass(frozen=True)
class ZeroGraspConfig:
    """Explicit experiment/runtime configuration; no hidden per-task knobs."""

    model_height: int = 1024
    model_width: int = 1280
    fixed_seed: int = 0
    max_candidates: int = 16
    mask_seed_radius_px: int = 10
    mask_min_area_px: int = 12
    mask_max_area_fraction: float = 0.35
    mask_depth_tolerance_m: float = 0.025
    mask_color_tolerance: float = 48.0
    grasp_workspace_min_m: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    grasp_workspace_max_m: tuple[float, float, float] = (1.0, 1.0, 2.0)
    pregrasp_distance_m: float = 0.08
    swept_path_clearance_m: float = 0.012
    placement_clearance_m: float = 0.004
    grasp_width_range_m: tuple[float, float] = (0.005, 0.08)
    grasp_depth_range_m: tuple[float, float] = (1e-6, 0.04)
    grasp_height_range_m: tuple[float, float] = (0.0, 0.25)
    approach_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0)
    approach_axis_cos_min: float = 0.70710678
    max_approach_tilt_deg: float = 45.0
    R_grasp_eef: np.ndarray | None = None
    tip_to_eef_residual_m: tuple[float, float, float] | None = None
    eef_calibration_verified: bool = False
    calibration_source: str | None = None
    calibration_sha256: str | None = None
    probe_sha256: str | None = None
    translation_rule: str = "center_plus_depth_x_then_R_G_E_delta_E_v1"
    R_H_E: np.ndarray | None = None
    request_timeout_s: float = 20.0
    external_repo: str | None = None
    checkpoint: str | None = None
    runtime_config: str | None = None
    entrypoint: str | None = None
    max_json_line_bytes: int = 64 * 1024 * 1024

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None = None) -> "ZeroGraspConfig":
        if value is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown ZeroGrasp config fields: {sorted(unknown)}")
        kwargs = dict(value)
        for field_name in ("grasp_workspace_min_m", "grasp_workspace_max_m", "grasp_width_range_m", "grasp_depth_range_m", "grasp_height_range_m", "approach_axis_local", "tip_to_eef_residual_m"):
            if field_name in kwargs:
                kwargs[field_name] = tuple(float(x) for x in kwargs[field_name])
        return cls(**kwargs)

    def validate(self) -> "ZeroGraspConfig":
        ints = ("model_height", "model_width", "max_candidates", "mask_seed_radius_px", "mask_min_area_px", "max_json_line_bytes")
        for name in ints:
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_height != 1024 or self.model_width != 1280:
            raise ValueError("ZeroGrasp model input must be the fixed 1024x1280 letterbox")
        if not 0 < self.mask_max_area_fraction <= 1:
            raise ValueError("mask_max_area_fraction must be in (0, 1]")
        for name in ("mask_depth_tolerance_m", "mask_color_tolerance", "pregrasp_distance_m", "swept_path_clearance_m", "placement_clearance_m", "request_timeout_s"):
            if not np.isfinite(getattr(self, name)) or float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("grasp_workspace_min_m", "grasp_workspace_max_m"):
            value = tuple(float(x) for x in getattr(self, name))
            if len(value) != 3 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain three finite values")
        if np.any(np.asarray(self.grasp_workspace_min_m) >= np.asarray(self.grasp_workspace_max_m)):
            raise ValueError("grasp workspace min must be strictly below max")
        for name in ("grasp_width_range_m", "grasp_depth_range_m", "grasp_height_range_m"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (2,) or np.any(~np.isfinite(value)) or np.any(value < 0) or value[0] > value[1]:
                raise ValueError(f"{name} must be a finite non-negative (min, max) range")
        approach = np.asarray(self.approach_axis_local, dtype=float)
        if approach.shape != (3,) or not np.all(np.isfinite(approach)) or np.linalg.norm(approach) <= 1e-9:
            raise ValueError("approach_axis_local must be a finite nonzero 3-vector")
        if not 0 <= self.approach_axis_cos_min <= 1:
            raise ValueError("approach_axis_cos_min must be in [0, 1]")
        if not 0 <= self.max_approach_tilt_deg <= 90:
            raise ValueError("max_approach_tilt_deg must be in [0, 90]")
        if self.R_grasp_eef is not None:
            rotation = _finite_array(self.R_grasp_eef, "R_grasp_eef", (3, 3)).astype(float)
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
                raise ValueError("R_grasp_eef must be proper orthonormal")
        if self.tip_to_eef_residual_m is not None:
            residual = np.asarray(self.tip_to_eef_residual_m, dtype=float)
            if residual.shape != (3,) or not np.all(np.isfinite(residual)):
                raise ValueError("tip_to_eef_residual_m must be a finite 3-vector")
        if not isinstance(self.eef_calibration_verified, (bool, np.bool_)):
            raise TypeError("eef_calibration_verified must be boolean")
        if self.eef_calibration_verified:
            if self.R_grasp_eef is None or self.R_H_E is None:
                raise ValueError("verified EEF calibration requires R_grasp_eef and R_H_E")
            if self.translation_rule != "center_plus_depth_x_then_R_G_E_delta_E_v1":
                raise ValueError("verified calibration requires the exact dynamic translation rule")
            if not self.calibration_source or not str(self.calibration_source).strip():
                raise ValueError("verified calibration requires a nonblank calibration_source")
            for name in ("calibration_sha256", "probe_sha256"):
                value = getattr(self, name)
                if value is None or not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
                    raise ValueError(f"verified calibration requires a valid 64-hex {name}")
            expected = _calibration_sha256(self)
            if self.calibration_sha256.lower() != expected:
                raise ValueError("calibration_sha256 does not match calibration formula/matrices/residual/source")
        if self.R_H_E is not None:
            rotation = _finite_array(self.R_H_E, "R_H_E", (3, 3)).astype(float)
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
                raise ValueError("R_H_E must be proper orthonormal")
        return self


def validate_config(value: ZeroGraspConfig | Mapping[str, Any] | None = None) -> ZeroGraspConfig:
    return (value if isinstance(value, ZeroGraspConfig) else ZeroGraspConfig.from_mapping(value)).validate()


@dataclass(frozen=True)
class MaskDiagnostics:
    ok: bool
    reason: str | None
    source_area: int
    destination_area: int
    source_confidence: float
    destination_confidence: float
    source_depth_m: float | None
    destination_depth_m: float | None
    disjoint: bool


@dataclass(frozen=True)
class ArrowMasks:
    source_mask: np.ndarray
    destination_mask: np.ndarray
    diagnostics: MaskDiagnostics


@dataclass(frozen=True)
class PreparedZeroGraspInput:
    clean_rgb: np.ndarray
    depth_m: np.ndarray
    source_mask: np.ndarray
    destination_mask: np.ndarray
    K_model: np.ndarray
    source_px_model: tuple[float, float]
    destination_px_model: tuple[float, float]
    pixel_affine: np.ndarray
    vertically_flipped: bool
    request_hash: str
    source_mask_hash: str = ""
    destination_mask_hash: str = ""
    source_mask_area: int = 0
    destination_mask_area: int = 0
    mask_rejections: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraspCandidate:
    T_camera_gripper: np.ndarray
    score: float = 0.0
    clearance_m: float = 0.0
    source_index: int = 0
    width_m: float | None = None
    height_m: float | None = None
    depth_m: float | None = None
    collision_free: bool = False
    source_role: str = "source"
    eef_frame: str | None = None
    translation_reference: str | None = None
    translation_frame: str | None = None
    rotation_frame: str | None = None

    def __post_init__(self) -> None:
        validate_se3(self.T_camera_gripper, "T_camera_gripper")
        if not np.isfinite(self.score) or not np.isfinite(self.clearance_m):
            raise ValueError("grasp candidate score and clearance must be finite")
        for name in ("width_m", "height_m", "depth_m"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or float(value) < 0):
                raise ValueError(f"{name} must be finite and non-negative when present")
        if not isinstance(self.collision_free, (bool, np.bool_)):
            raise TypeError("collision_free must be boolean")
        if self.source_role != "source":
            raise ValueError("grasp candidates may only have source role")
        if self.eef_frame is not None and not str(self.eef_frame).strip():
            raise ValueError("eef_frame cannot be blank")
        if self.translation_reference != GRASPNET_TRANSLATION_REFERENCE:
            raise ValueError("translation_reference must be exactly grasp_center")
        if self.translation_frame != GRASPNET_CAMERA_FRAME:
            raise ValueError("translation_frame must be exactly camera_graspnet")
        if self.rotation_frame != GRASPNET_CAMERA_FRAME:
            raise ValueError("rotation_frame must be exactly camera_graspnet")


@dataclass(frozen=True)
class ReconstructionSummary:
    dimensions_m: tuple[float, float, float]
    confidence: float = 0.0
    source_frame: str = "camera"
    centroid_camera_m: tuple[float, float, float] | None = None
    bounds_camera_m: tuple[float, float, float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if len(self.dimensions_m) != 3 or not np.all(np.isfinite(self.dimensions_m)) or any(float(x) <= 0 for x in self.dimensions_m):
            raise ValueError("reconstruction dimensions_m must contain three positive finite values")
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("reconstruction confidence must be in [0, 1]")
        if self.centroid_camera_m is not None and (len(self.centroid_camera_m) != 3 or not np.all(np.isfinite(self.centroid_camera_m))):
            raise ValueError("centroid_camera_m must be a finite 3-vector")
        if self.bounds_camera_m is not None and (len(self.bounds_camera_m) != 6 or not np.all(np.isfinite(self.bounds_camera_m))):
            raise ValueError("bounds_camera_m must be six finite values")


@dataclass(frozen=True)
class ZeroGraspInferenceResult:
    candidates: tuple[GraspCandidate, ...]
    reconstruction: ReconstructionSummary | None
    request_hash: str
    output_hash: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlacementPlan:
    T_world_eef: np.ndarray
    support_point_world: tuple[float, float, float]
    support_normal_world: tuple[float, float, float]
    footprint_m: tuple[float, float]
    confidence: float

    def __post_init__(self) -> None:
        validate_se3(self.T_world_eef, "T_world_eef")
        for name, value in (("support_point_world", self.support_point_world), ("support_normal_world", self.support_normal_world), ("footprint_m", self.footprint_m)):
            if len(value) not in (2, 3) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite values")
        if len(self.support_normal_world) != 3 or not np.isclose(np.linalg.norm(self.support_normal_world), 1.0, atol=1e-5):
            raise ValueError("support_normal_world must be unit length")
        if not 0 <= self.confidence <= 1:
            raise ValueError("placement confidence must be in [0, 1]")

def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_array(value: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    header = json.dumps({"dtype": arr.dtype.str, "shape": arr.shape}, separators=(",", ":")).encode()
    return hash_bytes(header + b"\0" + arr.tobytes(order="C"))


def encode_array(value: Any) -> dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(value))
    return {"dtype": arr.dtype.str, "shape": list(arr.shape), "data_b64": base64.b64encode(arr.tobytes()).decode("ascii")}


def decode_array(value: Mapping[str, Any]) -> np.ndarray:
    try:
        dtype = np.dtype(value["dtype"])
        shape = tuple(int(x) for x in value["shape"])
        raw = base64.b64decode(value["data_b64"], validate=True)
    except Exception as exc:
        raise ValueError("malformed encoded array") from exc
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if expected != len(raw):
        raise ValueError("encoded array byte length does not match dtype/shape")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def stable_json_hash(value: Any) -> str:
    return hash_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode())


def _calibration_sha256(config: ZeroGraspConfig) -> str:
    residual = config.tip_to_eef_residual_m
    payload = {"translation_rule": config.translation_rule, "R_grasp_eef": None if config.R_grasp_eef is None else np.asarray(config.R_grasp_eef, dtype=float).round(12).tolist(), "R_H_E": None if config.R_H_E is None else np.asarray(config.R_H_E, dtype=float).round(12).tolist(), "tip_to_eef_residual_m": None if residual is None else tuple(float(x) for x in residual), "calibration_source": config.calibration_source}
    return stable_json_hash(payload)


def calibration_sha256(config: ZeroGraspConfig) -> str:
    """Return the recomputable calibration identity used by validation."""
    return _calibration_sha256(validate_config(config))


def serialize_audit(result: Any) -> dict[str, Any]:
    """Return JSON-safe provenance without arrays or simulator-owned values."""
    if hasattr(result, "__dataclass_fields__"):
        result = asdict(result)
    if isinstance(result, np.ndarray):
        return {"array_hash": hash_array(result), "shape": list(result.shape), "dtype": result.dtype.str}
    if isinstance(result, Mapping):
        return {str(k): serialize_audit(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return [serialize_audit(v) for v in result]
    if isinstance(result, (np.integer, np.floating)):
        return result.item()
    if isinstance(result, Path):
        return result.as_posix()
    return result


__all__ = [name for name in globals() if not name.startswith("_")]
