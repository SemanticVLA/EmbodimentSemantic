"""No-motion Panda grip-site frame probe.

The probe audits the fixed orientation relationship used by the LIBERO Panda
model::

    R_grip_site_expected = R_right_hand @ Rz(-90 degrees)

It reads only MuJoCo model/data frame values.  It does not step a simulator,
send an action, reset an environment, inspect objects, or query evaluation
state.  Position values are recorded as metadata only; this probe does not
claim that a grasp-center-to-tip residual has been measured or verified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


GRIP_SITE_NAME = "grip_site"
RIGHT_HAND_NAME = "right_hand"
RZ_MINUS_90 = np.asarray(
    (
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)


def _rotation(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (9,):
        matrix = matrix.reshape(3, 3)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 3x3 or length-9 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6, rtol=1e-6):
        raise ValueError(f"{label} must be an orthonormal SO(3) rotation")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError(f"{label} must have determinant +1")
    return matrix


def _vector(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a finite 3-vector")
    return vector


def _named_id(model: Any, kind: str, names_to_try: tuple[str, ...]) -> tuple[int, str]:
    resolver = getattr(model, f"{kind}_name2id", None)
    if callable(resolver):
        for name in names_to_try:
            try:
                index = int(resolver(name))
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            if index >= 0:
                return index, name
    available = getattr(model, f"{kind}_names", None)
    for name in names_to_try:
        if isinstance(available, Mapping) and name in available:
            return int(available[name]), name
        if isinstance(available, (list, tuple)) and name in available:
            return int(available.index(name)), name
    joined = ", ".join(repr(name) for name in names_to_try)
    raise AttributeError(f"model does not expose any {kind} name in ({joined})")


def _axes(matrix: np.ndarray) -> dict[str, list[float]]:
    # MuJoCo stores orientation matrices row-major; the columns are the
    # frame's x/y/z axes in the parent frame.
    return {axis: matrix[:, index].tolist() for index, axis in enumerate(("x", "y", "z"))}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported calibration input type: {type(value).__name__}")


def _calibration_record(calibration_input: Any) -> dict[str, Any]:
    if calibration_input is None:
        return {"input": None, "sha256": None, "source": "unspecified"}
    if isinstance(calibration_input, (str, Path)):
        path = Path(calibration_input).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"calibration input is not a regular file: {path}")
        raw = path.read_bytes()
        return {
            "input": path.resolve().as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "source": "file",
        }
    normalized = _jsonable(calibration_input)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"input": normalized, "sha256": hashlib.sha256(encoded).hexdigest(), "source": "value"}


def probe_grip_site_frame(
    sim: Any,
    *,
    site_name: str | None = None,
    body_name: str | None = None,
    calibration_input: Any = None,
    angular_tolerance_deg: float = 1.0,
) -> dict[str, Any]:
    """Read and audit Panda frame orientation without moving ``sim``.

    ``sim`` may be a MuJoCo sim or a robosuite environment exposing ``.sim``.
    The returned ``position`` record is observational metadata only;
    no position tolerance is applied because body-to-grip-site translation is
    not the center-to-tip residual being calibrated elsewhere.
    """
    tolerance = float(angular_tolerance_deg)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("angular_tolerance_deg must be finite and positive")
    source = sim
    nested_sim = getattr(source, "sim", None)
    if nested_sim is not None and nested_sim is not source:
        source = nested_sim
    model = getattr(source, "model", None)
    data = getattr(source, "data", None)
    if model is None or data is None:
        raise AttributeError("sim or env.sim must expose model and data")

    requested_site_names = (
        (site_name,)
        if site_name
        else (GRIP_SITE_NAME, "gripper0_grip_site", "robot0_grip_site")
    )
    requested_body_names = (body_name,) if body_name else (RIGHT_HAND_NAME, "robot0_right_hand")
    site_id, resolved_site_name = _named_id(model, "site", requested_site_names)
    body_id, resolved_body_name = _named_id(model, "body", requested_body_names)
    site_xmat = _rotation(np.asarray(data.site_xmat)[site_id], f"data.site_xmat[{resolved_site_name}]")
    body_xmat = _rotation(np.asarray(data.body_xmat)[body_id], f"data.body_xmat[{resolved_body_name}]")
    expected = body_xmat @ RZ_MINUS_90
    observed_body_to_site = body_xmat.T @ site_xmat
    cosine = (float(np.trace(expected.T @ site_xmat)) - 1.0) / 2.0
    angular_error_rad = float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    site_xpos = _vector(np.asarray(data.site_xpos)[site_id], f"data.site_xpos[{resolved_site_name}]")
    body_xpos = _vector(np.asarray(data.body_xpos)[body_id], f"data.body_xpos[{resolved_body_name}]")
    offset = site_xpos - body_xpos
    calibration = _calibration_record(calibration_input)
    passed = angular_error_rad <= np.deg2rad(tolerance)
    return {
        "schema_version": "panda_grip_site_frame_probe.v1",
        "passed": bool(passed),
        "pass": bool(passed),
        "frame_contract": {
            "requested_site_name": site_name,
            "requested_body_name": body_name,
            "resolved_site_name": resolved_site_name,
            "resolved_body_name": resolved_body_name,
            "site_frame": resolved_site_name,
            "body_frame": resolved_body_name,
            "expected_rotation": f"R_{resolved_body_name} @ Rz(-90deg)",
            "rotation_matrix_storage": "MuJoCo row-major; columns are frame axes",
            "position_units": "meters",
            "angular_units": "radians_and_degrees",
        },
        "site_id": site_id,
        "body_id": body_id,
        "resolved_site_name": resolved_site_name,
        "resolved_body_name": resolved_body_name,
        "expected_rotation_matrix": expected.tolist(),
        "observed_rotation_matrix": site_xmat.tolist(),
        "expected_body_to_site_rotation_matrix": RZ_MINUS_90.tolist(),
        "observed_body_to_site_rotation_matrix": observed_body_to_site.tolist(),
        "expected_axes": _axes(expected),
        "observed_axes": _axes(site_xmat),
        "angular": {
            "error_rad": angular_error_rad,
            "error_deg": float(np.rad2deg(angular_error_rad)),
            "tolerance_deg": tolerance,
            "passed": bool(passed),
        },
        "position": {
            "site_name": resolved_site_name,
            "body_name": resolved_body_name,
            "site_xpos_m": site_xpos.tolist(),
            "right_hand_xpos_m": body_xpos.tolist(),
            "site_minus_right_hand_m": offset.tolist(),
            "site_offset_norm_m": float(np.linalg.norm(offset)),
            "status": "observed_metadata_only",
            "center_to_tip_residual_verified": False,
        },
        "calibration": calibration,
        "calibration_input": calibration["input"],
        "calibration_sha256": calibration["sha256"],
    }


def _load_factory(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("factory must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError(f"factory is not callable: {spec}")
    return factory()


class _ArtifactModel:
    def __init__(self, site_names: list[str], body_names: list[str]):
        self.site_names = site_names
        self.body_names = body_names

    def site_name2id(self, name: str) -> int:
        return self.site_names.index(name)

    def body_name2id(self, name: str) -> int:
        return self.body_names.index(name)


def _load_artifact(path: Path) -> tuple[Any, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_payload = payload.get("model", {})
    data_payload = payload.get("data", payload)
    site_names = list(model_payload.get("site_names", [GRIP_SITE_NAME]))
    body_names = list(model_payload.get("body_names", [RIGHT_HAND_NAME]))
    model = _ArtifactModel(site_names, body_names)
    data = SimpleNamespace(
        site_xmat=np.asarray(data_payload["site_xmat"], dtype=float),
        body_xmat=np.asarray(data_payload["body_xmat"], dtype=float),
        site_xpos=np.asarray(data_payload["site_xpos"], dtype=float),
        body_xpos=np.asarray(data_payload["body_xpos"], dtype=float),
    )
    return SimpleNamespace(model=model, data=data), payload.get("calibration")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--factory", help="module:function returning an already-created sim")
    source.add_argument("--artifact", type=Path, help="JSON frame artifact; read-only, no simulator motion")
    parser.add_argument("--site-name", help="explicit MuJoCo site name; default tries grip_site then robot0_grip_site")
    parser.add_argument("--body-name", help="explicit MuJoCo body name; default tries right_hand then robot0_right_hand")
    parser.add_argument("--calibration-input", type=Path, help="optional calibration file to hash")
    parser.add_argument("--output", type=Path, help="optional output JSON path")
    args = parser.parse_args(argv)
    artifact_calibration = None
    if args.factory:
        sim = _load_factory(args.factory)
    else:
        sim, artifact_calibration = _load_artifact(args.artifact)
    calibration_input = args.calibration_input if args.calibration_input is not None else artifact_calibration
    result = probe_grip_site_frame(
        sim,
        site_name=args.site_name,
        body_name=args.body_name,
        calibration_input=calibration_input,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
