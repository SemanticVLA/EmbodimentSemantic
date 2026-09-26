"""Pure, versioned geometric rules for LIBERO visual/reference graphs.

The production graph is built from public RGB masks.  A benchmark-side
diagnostic may build a second graph from projected simulator geometry, but it
must use this same relation contract and keep the privileged graph outside the
production resolver.  Keeping the rules here (rather than in a detector,
tracker, or graph-brain prompt) makes initial SAM3.1 detections and later live
tracking frames comparable.

The coordinate adjustment is intentional and explicit: LIBERO's raw
``agentview`` pixels use a top-left origin, while the accepted relation
calibration reverses the image-y direction and weights it by 1.32.  This is a
semantic image-space adjustment, not access to simulator pose or calibration.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull, QhullError

from .geometry_profiles import (
    SAMGRAPH_SPATIAL_MASK_GEOMETRY,
    SAMGRAPH_SPATIAL_MASK_GEOMETRY_PROFILE,
)


GEOMETRIC_GRAPH_SCHEMA = "samgraph.geometric_graph.v1"
GEOMETRIC_RELATION_RULES_REVISION = "libero_geometric_relation_rules_v2"
NOTEBOOK_MASK_PROXY_RULES_REVISION = SAMGRAPH_SPATIAL_MASK_GEOMETRY_PROFILE
PIXEL_FRAME = "agentview_raw_rgb_top_left_xyxy"

INVERSE_RELATIONS: dict[str, str] = {
    "is_left_of": "is_right_of",
    "is_right_of": "is_left_of",
    "is_in_front_of": "is_behind",
    "is_behind": "is_in_front_of",
    "is_on_top_of": "is_below_of",
    "is_below_of": "is_on_top_of",
    "is_inside": "contains",
    "contains": "is_inside",
}

@dataclass(frozen=True, slots=True)
class GeometricRelationConfig:
    """Resolved knobs for the image-space graph relation contract.

    The defaults intentionally reproduce the original SamGraph behavior.  A
    benchmark may supply another configuration, but that configuration is
    always serialized into the graph manifest and therefore changes both its
    rule revision and digest.
    """

    vertical_scale: float = 1.32
    hull_coverage_min: float = 0.80
    hull_coverage_near_min: float = 0.70
    near_distance_max_normalized: float = 0.08
    cookies_reverse_coverage_min: float = 0.40
    cabinet_top_layer_quantile: float = 0.15
    mask_overlap_iou_max: float = 0.90
    black_bowl_rgb_std_min: float = 8.0

    def __post_init__(self) -> None:
        values = {
            "vertical_scale": self.vertical_scale,
            "hull_coverage_min": self.hull_coverage_min,
            "hull_coverage_near_min": self.hull_coverage_near_min,
            "near_distance_max_normalized": self.near_distance_max_normalized,
            "cookies_reverse_coverage_min": self.cookies_reverse_coverage_min,
            "cabinet_top_layer_quantile": self.cabinet_top_layer_quantile,
            "mask_overlap_iou_max": self.mask_overlap_iou_max,
            "black_bowl_rgb_std_min": self.black_bowl_rgb_std_min,
        }
        if values["vertical_scale"] <= 0:
            raise ValueError("vertical_scale must be positive")
        for name in (
            "hull_coverage_min",
            "hull_coverage_near_min",
            "cookies_reverse_coverage_min",
            "cabinet_top_layer_quantile",
            "mask_overlap_iou_max",
        ):
            value = float(values[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if values["hull_coverage_near_min"] > values["hull_coverage_min"]:
            raise ValueError("hull_coverage_near_min cannot exceed hull_coverage_min")
        if values["near_distance_max_normalized"] < 0:
            raise ValueError("near_distance_max_normalized must be non-negative")
        if values["black_bowl_rgb_std_min"] < 0:
            raise ValueError("black_bowl_rgb_std_min must be non-negative")


DEFAULT_GEOMETRIC_RELATION_CONFIG = GeometricRelationConfig()


@dataclass(frozen=True, slots=True)
class NotebookMaskProxyConfig:
    """Mask-only replay of the notebook's bbox graph calibration.

    This profile deliberately contains no world coordinates or simulator
    state.  Bboxes are half-open pixel extents, and all relation decisions are
    derived from the cached/current masks supplied by the caller.
    """

    # SAM masks are evaluated at their observed extent.  Padding is available
    # only as an explicit downstream geometry calibration, never by default.
    bbox_padding_x: float = float(SAMGRAPH_SPATIAL_MASK_GEOMETRY["bbox_padding_x"])
    bbox_padding_y: float = float(SAMGRAPH_SPATIAL_MASK_GEOMETRY["bbox_padding_y"])
    containment_iomin_threshold: float = float(
        SAMGRAPH_SPATIAL_MASK_GEOMETRY["containment_iomin_threshold"]
    )
    direction_horizontal_axis: str = str(
        SAMGRAPH_SPATIAL_MASK_GEOMETRY["direction_horizontal_axis"]
    )
    direction_horizontal_sign: int = int(
        SAMGRAPH_SPATIAL_MASK_GEOMETRY["direction_horizontal_sign"]
    )
    direction_vertical_axis: str = str(
        SAMGRAPH_SPATIAL_MASK_GEOMETRY["direction_vertical_axis"]
    )
    direction_vertical_sign: int = int(
        SAMGRAPH_SPATIAL_MASK_GEOMETRY["direction_vertical_sign"]
    )
    direction_vertical_scale: float = float(
        SAMGRAPH_SPATIAL_MASK_GEOMETRY["direction_vertical_scale"]
    )
    # Optional 2-D affine map from image-space centroid displacement
    # ``(du, dv)`` to the notebook's two dominant-direction scores.  When
    # omitted, the legacy axis/sign/vertical-scale rule below is used exactly
    # as before.  Both rows must be supplied together so a candidate cannot
    # accidentally compare an affine front score with a legacy left score.
    direction_front_coefficients: tuple[float, float] | None = None
    direction_left_coefficients: tuple[float, float] | None = None
    direction_tie_axis: str = str(SAMGRAPH_SPATIAL_MASK_GEOMETRY["direction_tie_axis"])
    stack_order: str = str(SAMGRAPH_SPATIAL_MASK_GEOMETRY["stack_order"])
    drawer_stack_order: str = str(SAMGRAPH_SPATIAL_MASK_GEOMETRY["drawer_stack_order"])
    drawer_mode: str = str(SAMGRAPH_SPATIAL_MASK_GEOMETRY["drawer_mode"])
    # Opt-in branch semantics from the root ground-truth notebook. Geometry
    # remains mask-derived; this does not provide world-height observations.
    notebook_pair_semantics: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.notebook_pair_semantics, bool):
            raise ValueError("notebook_pair_semantics must be boolean")
        for name in ("bbox_padding_x", "bbox_padding_y"):
            value = _typed_number(getattr(self, name), name)
            object.__setattr__(self, name, value)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        threshold = _typed_number(self.containment_iomin_threshold, "containment_iomin_threshold")
        object.__setattr__(self, "containment_iomin_threshold", threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("containment_iomin_threshold must be in [0, 1]")
        scale = _typed_number(self.direction_vertical_scale, "direction_vertical_scale")
        object.__setattr__(self, "direction_vertical_scale", scale)
        if scale <= 0:
            raise ValueError("direction_vertical_scale must be positive")
        front = _typed_coefficients(
            self.direction_front_coefficients, "direction_front_coefficients"
        )
        left = _typed_coefficients(
            self.direction_left_coefficients, "direction_left_coefficients"
        )
        if (front is None) != (left is None):
            raise ValueError(
                "direction_front_coefficients and direction_left_coefficients "
                "must be supplied together"
            )
        if front is not None and left is not None:
            determinant = front[0] * left[1] - front[1] * left[0]
            if np.isclose(determinant, 0.0, atol=1e-12, rtol=0.0):
                raise ValueError(
                    "affine direction coefficient rows must be non-degenerate"
                )
        object.__setattr__(self, "direction_front_coefficients", front)
        object.__setattr__(self, "direction_left_coefficients", left)
        for name in ("direction_horizontal_sign", "direction_vertical_sign"):
            value = _typed_int(getattr(self, name), name)
            object.__setattr__(self, name, value)
            if value not in {-1, 1}:
                raise ValueError(f"{name} must be -1 or 1")
        if self.direction_horizontal_axis not in {"x", "y"}:
            raise ValueError("direction_horizontal_axis must be x or y")
        if self.direction_vertical_axis not in {"x", "y"}:
            raise ValueError("direction_vertical_axis must be x or y")
        if self.direction_horizontal_axis == self.direction_vertical_axis:
            raise ValueError("direction axes must be distinct")
        if self.direction_tie_axis not in {"horizontal", "vertical"}:
            raise ValueError("direction_tie_axis must be horizontal or vertical")
        valid_stack_orders = {
            "higher_vertical_is_top", "lower_vertical_is_top",
            "higher_vertical_is_inside", "lower_vertical_is_inside",
            "subject_first", "destination_first", "first_is_top", "first_is_bottom",
        }
        if self.stack_order not in valid_stack_orders:
            raise ValueError(f"unsupported notebook stack-order name: {self.stack_order!r}")
        if self.drawer_stack_order not in valid_stack_orders:
            raise ValueError(
                f"unsupported notebook drawer stack-order name: {self.drawer_stack_order!r}"
            )
        if self.drawer_mode not in {
            "disabled", "instruction_overlap", "instruction_overlap_cabinet_band"
        }:
            raise ValueError(f"unsupported notebook drawer mode: {self.drawer_mode!r}")
        if not self.stack_order or not self.drawer_stack_order:
            raise ValueError("stack-order names must be non-empty")


def _typed_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        result = float(value) if isinstance(value, (int, float, str)) else None
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result is None or not np.isfinite(result):
        raise ValueError(f"{name} must be finite numeric")
    return result


def _typed_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not boolean")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and np.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if parsed.is_integer():
            return int(parsed)
    raise ValueError(f"{name} must be an integer")


def _typed_coefficients(
    value: Any,
    name: str,
) -> tuple[float, float] | None:
    """Validate one row of the optional image-to-direction affine map."""

    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a two-element numeric sequence")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a two-element numeric sequence") from exc
    if len(items) != 2:
        raise ValueError(f"{name} must contain exactly two coefficients")
    coefficients = tuple(_typed_number(item, f"{name}[{index}]") for index, item in enumerate(items))
    if coefficients == (0.0, 0.0):
        raise ValueError(f"{name} must not be the all-zero row")
    return coefficients


def _is_notebook_proxy(value: Any) -> bool:
    if isinstance(value, NotebookMaskProxyConfig):
        return True
    if not isinstance(value, Mapping):
        return False
    profile = str(value.get("profile", value.get("rule_profile", value.get("revision", ""))))
    return profile == NOTEBOOK_MASK_PROXY_RULES_REVISION


def _notebook_proxy_from_rules(value: NotebookMaskProxyConfig | Mapping[str, Any] | None) -> NotebookMaskProxyConfig:
    if value is None:
        return NotebookMaskProxyConfig()
    if isinstance(value, NotebookMaskProxyConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("notebook mask proxy rules must be a config or mapping")
    defaults = NotebookMaskProxyConfig()
    bbox = value.get("mask_bbox", value.get("bbox", {}))
    containment = value.get("containment", value.get("stack", {}))
    direction = value.get("direction", value.get("coordinate_adjustment", {}))
    support = value.get("support", {})
    stack_value = value.get("stack_order", value.get("stack", defaults.stack_order))
    drawer = value.get("drawer_stack_order", value.get("drawer", defaults.drawer_stack_order))
    drawer_mode = value.get("drawer_mode", defaults.drawer_mode)
    stack = stack_value
    if isinstance(stack_value, Mapping):
        stack = stack_value.get("name", stack_value.get("order", defaults.stack_order))
        drawer = stack_value.get("drawer_name", stack_value.get("drawer_order", drawer))
        drawer_mode = stack_value.get("drawer_mode", drawer_mode)
    if isinstance(drawer, Mapping):
        drawer = drawer.get("name", drawer.get("order", defaults.drawer_stack_order))
    def _get(*names: str, default: Any) -> Any:
        for source in (value, bbox if isinstance(bbox, Mapping) else {}, containment if isinstance(containment, Mapping) else {}, direction if isinstance(direction, Mapping) else {}, support if isinstance(support, Mapping) else {}):
            for name in names:
                if name in source:
                    return source[name]
        return default
    config = NotebookMaskProxyConfig(
        bbox_padding_x=_get(
            "bbox_padding_x_fraction", "padding_x_fraction", "bbox_padding_x",
            "padding_x", "padding_x_px",
            default=SAMGRAPH_SPATIAL_MASK_GEOMETRY["bbox_padding_x"],
        ),
        bbox_padding_y=_get(
            "bbox_padding_y_fraction", "padding_y_fraction", "bbox_padding_y",
            "padding_y", "padding_y_px",
            default=SAMGRAPH_SPATIAL_MASK_GEOMETRY["bbox_padding_y"],
        ),
        containment_iomin_threshold=_get(
            "containment_iomin_threshold", "iomin_threshold", "threshold",
            default=SAMGRAPH_SPATIAL_MASK_GEOMETRY["containment_iomin_threshold"],
        ),
        direction_horizontal_axis=_get(
            "direction_horizontal_axis", "horizontal_axis",
            default=SAMGRAPH_SPATIAL_MASK_GEOMETRY["direction_horizontal_axis"],
        ),
        direction_horizontal_sign=_get(
            "direction_horizontal_sign", "horizontal_sign",
            default=SAMGRAPH_SPATIAL_MASK_GEOMETRY["direction_horizontal_sign"],
        ),
        direction_vertical_axis=_get(
            "direction_vertical_axis", "vertical_axis",
            default=SAMGRAPH_SPATIAL_MASK_GEOMETRY["direction_vertical_axis"],
        ),
        direction_vertical_sign=_get(
            "direction_vertical_sign", "vertical_sign",
            default=SAMGRAPH_SPATIAL_MASK_GEOMETRY["direction_vertical_sign"],
        ),
        direction_vertical_scale=_get("direction_vertical_scale", "vertical_scale", "scale", default=defaults.direction_vertical_scale),
        direction_front_coefficients=_get(
            "direction_front_coefficients",
            "front_score_coefficients",
            "front_score",
            "front_coefficients",
            default=defaults.direction_front_coefficients,
        ),
        direction_left_coefficients=_get(
            "direction_left_coefficients",
            "left_score_coefficients",
            "left_score",
            "left_coefficients",
            default=defaults.direction_left_coefficients,
        ),
        direction_tie_axis=_get(
            "direction_tie_axis", "tie_axis",
            default=SAMGRAPH_SPATIAL_MASK_GEOMETRY["direction_tie_axis"],
        ),
        stack_order=stack,
        drawer_stack_order=drawer,
        drawer_mode=_get("drawer_mode", "drawer_proxy_mode", default=drawer_mode),
        notebook_pair_semantics=_get("notebook_pair_semantics", default=False),
    )
    return config


def _config_from_rules(
    value: GeometricRelationConfig | NotebookMaskProxyConfig | Mapping[str, Any] | None,
) -> GeometricRelationConfig | NotebookMaskProxyConfig:
    if value is None:
        return DEFAULT_GEOMETRIC_RELATION_CONFIG
    if isinstance(value, GeometricRelationConfig):
        return value
    if isinstance(value, NotebookMaskProxyConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("geometry rules must be a GeometricRelationConfig or mapping")
    profile = value.get("profile", value.get("rule_profile", value.get("revision")))
    if profile is not None and str(profile).startswith("notebook_mask_proxy"):
        raise ValueError(
            "unsupported geometry profile; use "
            f"{NOTEBOOK_MASK_PROXY_RULES_REVISION}"
        )
    if _is_notebook_proxy(value):
        return _notebook_proxy_from_rules(value)
    # Accept either the small flat configuration or a previously serialized
    # graph manifest.  This makes saved benchmark manifests reusable as input.
    coordinate = value.get("coordinate_adjustment", {})
    support = value.get("support", {})
    quality = value.get("quality", {})
    flat = {
        "vertical_scale": value.get("vertical_scale", coordinate.get("vertical_scale", 1.32)),
        "hull_coverage_min": value.get("hull_coverage_min", support.get("hull_coverage_min", 0.80)),
        "hull_coverage_near_min": value.get("hull_coverage_near_min", support.get("hull_coverage_near_min", 0.70)),
        "near_distance_max_normalized": value.get(
            "near_distance_max_normalized", support.get("near_distance_max_normalized", 0.08)
        ),
        "cookies_reverse_coverage_min": value.get(
            "cookies_reverse_coverage_min", support.get("cookies_reverse_coverage_min", 0.40)
        ),
        "cabinet_top_layer_quantile": value.get(
            "cabinet_top_layer_quantile", support.get("cabinet_top_layer_quantile", 0.15)
        ),
        "mask_overlap_iou_max": value.get(
            "mask_overlap_iou_max", quality.get("mask_overlap_iou_max", 0.90)
        ),
        "black_bowl_rgb_std_min": value.get(
            "black_bowl_rgb_std_min", quality.get("black_bowl_rgb_std_min", 8.0)
        ),
    }
    return GeometricRelationConfig(**{key: _typed_number(item, key) for key, item in flat.items()})


def _notebook_proxy_rules(config: NotebookMaskProxyConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved = _notebook_proxy_from_rules(config)
    rules = {
        "revision": NOTEBOOK_MASK_PROXY_RULES_REVISION,
        "profile": NOTEBOOK_MASK_PROXY_RULES_REVISION,
        "schema": GEOMETRIC_GRAPH_SCHEMA,
        "pixel_frame": PIXEL_FRAME,
        "mask_bbox": {
            "convention": "half_open_xyxy",
            "padding_x_fraction": resolved.bbox_padding_x,
            "padding_y_fraction": resolved.bbox_padding_y,
            "padding_reference": "image_extent_per_axis",
            "clamp": "image_extent",
        },
        "containment": {
            "metric": "intersection_over_min_area",
            "threshold": resolved.containment_iomin_threshold,
            "comparison": "strict_gt",
        },
        "coordinate_adjustment": {
            "direction_coordinates": "notebook_mask_proxy_pixels",
            "horizontal_axis": resolved.direction_horizontal_axis,
            "horizontal_sign": resolved.direction_horizontal_sign,
            "vertical_axis": resolved.direction_vertical_axis,
            "vertical_sign": resolved.direction_vertical_sign,
            "vertical_scale": resolved.direction_vertical_scale,
            "dominant_axis_tie": resolved.direction_tie_axis,
        },
        "stack_order": {
            "name": resolved.stack_order,
            "drawer_name": resolved.drawer_stack_order,
            "drawer_mode": resolved.drawer_mode,
        },
        "drawer_mode": resolved.drawer_mode,
        "support": {
            "subject_role": "bowl",
            "destination_roles": ["support", "cabinet"],
        },
        "quality": {
            "mask_overlap_iou_max": 0.90,
            "black_bowl_rgb_std_min": 8.0,
        },
        "object_goal": {
            "on_tokens": ["on", "on top of"],
            "inside_tokens": ["in", "inside"],
            "relation_source": "instruction_goal_grammar",
            "cabinet_storage_tokens": ["top layer", "top drawer"],
        },
        "live": {
            "target_geometry_source": "current_nonempty_tracker_mask",
            "held_geometry_is_observed": False,
            "selection_calls": 1,
        },
        "simulator_geometry_consumed": False,
    }
    # Keep the legacy serialized profile byte-for-byte stable when no affine
    # map is selected.  Explicit affine candidates are self-describing and
    # round-trip through the same mapping parser.
    if resolved.notebook_pair_semantics:
        rules["notebook_pair_semantics"] = True
        rules["support"] = {"pair_eligibility": "either_role_is_bowl",
                            "ordered_pairs": True,
                            "height_source": "mask_midpoint_proxy_not_world_z"}
    if resolved.direction_front_coefficients is not None:
        rules["coordinate_adjustment"]["front_score_coefficients"] = list(
            resolved.direction_front_coefficients
        )
        rules["coordinate_adjustment"]["left_score_coefficients"] = list(
            resolved.direction_left_coefficients or ()
        )
    return rules


def _config_digest(config: GeometricRelationConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def geometric_relation_revision(
    config: GeometricRelationConfig | Mapping[str, Any] | None = None,
) -> str:
    if _is_notebook_proxy(config):
        resolved = _notebook_proxy_from_rules(config)
        default = _notebook_proxy_rules()
        current = _notebook_proxy_rules(resolved)
        if current == default:
            return NOTEBOOK_MASK_PROXY_RULES_REVISION
        digest = hashlib.sha256(json.dumps(current, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return f"{NOTEBOOK_MASK_PROXY_RULES_REVISION}+{digest[:12]}"
    resolved = _config_from_rules(config)
    digest = _config_digest(resolved)
    default_digest = _config_digest(DEFAULT_GEOMETRIC_RELATION_CONFIG)
    if digest == default_digest:
        return GEOMETRIC_RELATION_RULES_REVISION
    return f"{GEOMETRIC_RELATION_RULES_REVISION}+{digest[:12]}"


def geometric_relation_rules(
    config: GeometricRelationConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the resolved, JSON-compatible relation contract."""

    if _is_notebook_proxy(config):
        rules = _notebook_proxy_rules(config)
        rules["revision"] = geometric_relation_revision(config)
        return rules
    resolved = _config_from_rules(config)
    return {
        "revision": geometric_relation_revision(resolved),
        "schema": GEOMETRIC_GRAPH_SCHEMA,
        "pixel_frame": PIXEL_FRAME,
        "coordinate_adjustment": {
            "direction_coordinates": "calibrated_agentview",
            "vertical_sign": -1,
            "vertical_scale": resolved.vertical_scale,
            "dominant_axis_tie": "vertical",
            "normalization": "native_pixel_displacement",
        },
        "support": {
            "subject_role": "bowl",
            "destination_roles": ["support", "cabinet"],
            "hull_coverage_min": resolved.hull_coverage_min,
            "hull_coverage_near_min": resolved.hull_coverage_near_min,
            "near_distance_max_normalized": resolved.near_distance_max_normalized,
            "cookies_reverse_coverage_min": resolved.cookies_reverse_coverage_min,
            "cabinet_top_layer_quantile": resolved.cabinet_top_layer_quantile,
        },
        "quality": {
            "mask_overlap_iou_max": resolved.mask_overlap_iou_max,
            "black_bowl_rgb_std_min": resolved.black_bowl_rgb_std_min,
        },
        "object_goal": {
            "on_tokens": ["on", "on top of"],
            "inside_tokens": ["in", "inside"],
            "relation_source": "instruction_goal_grammar",
            "cabinet_storage_tokens": ["top layer", "top drawer"],
        },
        "live": {
            "target_geometry_source": "current_nonempty_tracker_mask",
            "held_geometry_is_observed": False,
            "selection_calls": 1,
        },
    }


GEOMETRIC_RELATION_RULES: dict[str, Any] = geometric_relation_rules()


def geometric_relation_rules_sha256(
    config: GeometricRelationConfig | Mapping[str, Any] | None = None,
) -> str:
    payload = json.dumps(
        geometric_relation_rules(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_mask(value: np.ndarray) -> np.ndarray:
    mask = np.ascontiguousarray(np.asarray(value, dtype=bool))
    if mask.ndim != 2 or not mask.any():
        raise ValueError("geometric graph masks must be nonempty 2-D arrays")
    return mask


def mask_centroid(mask: np.ndarray) -> np.ndarray:
    value = _as_mask(mask)
    ys, xs = np.nonzero(value)
    return np.asarray((float(xs.mean()), float(ys.mean())), dtype=np.float64)


def mask_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
    value = _as_mask(mask)
    ys, xs = np.nonzero(value)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_bbox_xyxy_half_open(
    mask: np.ndarray,
    *,
    padding_x: float = 0.0,
    padding_y: float = 0.0,
) -> tuple[int, int, int, int]:
    """Return a clamped half-open ``(x0, y0, x1, y1)`` mask bbox.

    Padding is independently applied in each image axis.  The image extent is
    used for clamping, so the returned width/height are always non-negative
    and compatible with the notebook's rectangle-area convention.
    """

    value = _as_mask(mask)
    px, py = _typed_number(padding_x, "padding_x"), _typed_number(padding_y, "padding_y")
    if px < 0 or py < 0:
        raise ValueError("bbox padding must be non-negative")
    ys, xs = np.nonzero(value)
    height, width = value.shape
    x0 = max(0, int(np.floor(float(xs.min()) - px)))
    y0 = max(0, int(np.floor(float(ys.min()) - py)))
    x1 = min(width, int(np.ceil(float(xs.max()) + 1.0 + px)))
    y1 = min(height, int(np.ceil(float(ys.max()) + 1.0 + py)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("padded mask bbox is empty")
    return x0, y0, x1, y1


def notebook_mask_bbox_xyxy(
    mask: np.ndarray,
    *,
    padding_x: float = 0.0,
    padding_y: float = 0.0,
) -> tuple[int, int, int, int]:
    """Return a tight SAM-mask bbox; optional padding is explicit post-processing."""

    value = _as_mask(mask)
    return mask_bbox_xyxy_half_open(
        value,
        padding_x=_typed_number(padding_x, "padding_x_fraction") * value.shape[1],
        padding_y=_typed_number(padding_y, "padding_y_fraction") * value.shape[0],
    )


# Short aliases make the proxy primitive easy to use from offline notebooks
# while retaining the explicit long-form name in the serialized contract.
mask_bbox_half_open = mask_bbox_xyxy_half_open
notebook_mask_bbox = notebook_mask_bbox_xyxy


def bbox_iomin(
    left: tuple[int, int, int, int] | list[int | float],
    right: tuple[int, int, int, int] | list[int | float],
) -> float:
    """Exact intersection-over-minimum-area for half-open rectangles."""

    lx0, ly0, lx1, ly1 = (float(item) for item in left)
    rx0, ry0, rx1, ry1 = (float(item) for item in right)
    intersection = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(
        0.0, min(ly1, ry1) - max(ly0, ry0)
    )
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    denominator = min(left_area, right_area)
    return intersection / denominator if denominator > 0.0 else 0.0


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    a, b = _as_mask(left), _as_mask(right)
    if a.shape != b.shape:
        raise ValueError("geometric graph IoU requires equal mask shapes")
    union = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum()) / union if union else 0.0


def convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    """Rasterize the convex hull using the same native pixel convention."""

    value = _as_mask(mask)
    ys, xs = np.nonzero(value)
    points = np.column_stack((xs, ys))
    if len(points) < 3:
        return np.array(value, copy=True)
    try:
        vertices = points[ConvexHull(points).vertices]
    except QhullError:
        return np.array(value, copy=True)
    image = Image.new("L", (value.shape[1], value.shape[0]), 0)
    ImageDraw.Draw(image).polygon(
        [tuple(map(int, point)) for point in vertices], fill=1
    )
    return np.asarray(image, dtype=np.uint8) > 0


def _notebook_axis_value(center: np.ndarray | tuple[float, float], axis: str) -> float:
    values = np.asarray(center, dtype=np.float64)
    if values.shape != (2,):
        raise ValueError("pixel centers must be two-element x/y coordinates")
    return float(values[0] if axis == "x" else values[1])


def notebook_direction_relation(
    left: np.ndarray | tuple[float, float],
    right: np.ndarray | tuple[float, float],
    *,
    geometry_rules: NotebookMaskProxyConfig | Mapping[str, Any] | None = None,
) -> str:
    """Apply the explicit pixel-axis/sign contract of the notebook proxy."""

    config = _notebook_proxy_from_rules(geometry_rules)
    left_center = np.asarray(left, dtype=np.float64)
    right_center = np.asarray(right, dtype=np.float64)
    if left_center.shape != (2,) or right_center.shape != (2,):
        raise ValueError("pixel centers must be two-element x/y coordinates")
    displacement = left_center - right_center
    if config.direction_front_coefficients is not None:
        # The affine rows are deliberately applied to raw image-space (x, y)
        # displacement.  This permits the camera's cross-axis projection to
        # be calibrated while retaining the notebook's dominant-score rule.
        vertical = float(np.dot(config.direction_front_coefficients, displacement))
        horizontal = float(np.dot(config.direction_left_coefficients, displacement))
    else:
        horizontal = config.direction_horizontal_sign * (
            _notebook_axis_value(left, config.direction_horizontal_axis)
            - _notebook_axis_value(right, config.direction_horizontal_axis)
        )
        vertical = config.direction_vertical_sign * config.direction_vertical_scale * (
            _notebook_axis_value(left, config.direction_vertical_axis)
            - _notebook_axis_value(right, config.direction_vertical_axis)
        )
    if horizontal == 0.0 and vertical == 0.0:
        if config.notebook_pair_semantics:
            return "is_behind"
        raise ValueError("coincident graph centers do not define a direction")
    if config.direction_tie_axis == "vertical":
        use_vertical = abs(vertical) >= abs(horizontal)
    else:
        use_vertical = abs(vertical) > abs(horizontal)
    if use_vertical:
        return "is_in_front_of" if vertical > 0 else "is_behind"
    return "is_left_of" if horizontal > 0 else "is_right_of"


def _notebook_stack_is_top(
    subject: Mapping[str, Any],
    destination: Mapping[str, Any],
    subject_center: np.ndarray,
    destination_center: np.ndarray,
    config: NotebookMaskProxyConfig,
    *,
    drawer: bool,
) -> bool:
    name = config.drawer_stack_order if drawer else config.stack_order
    subject_value = config.direction_vertical_sign * _notebook_axis_value(
        subject_center, config.direction_vertical_axis
    )
    destination_value = config.direction_vertical_sign * _notebook_axis_value(
        destination_center, config.direction_vertical_axis
    )
    if subject_value == destination_value:
        # A named, deterministic tie-break keeps the proxy replayable when
        # mask centroids land on the same pixel row/column.
        return str(subject.get("instance_id")) < str(destination.get("instance_id"))
    if name.startswith("lower_"):
        return subject_value < destination_value
    if name.startswith("higher_"):
        return subject_value > destination_value
    if name in {"subject_first", "first_is_top"}:
        return str(subject.get("instance_id")) < str(destination.get("instance_id"))
    if name in {"destination_first", "first_is_bottom"}:
        return str(subject.get("instance_id")) > str(destination.get("instance_id"))
    raise ValueError(f"unsupported notebook stack-order name: {name!r}")


def direction_relation(
    left: np.ndarray | tuple[float, float],
    right: np.ndarray | tuple[float, float],
    *,
    shape: tuple[int, int] | None = None,
    geometry_rules: GeometricRelationConfig | Mapping[str, Any] | None = None,
) -> str:
    """Apply the calibrated image-space direction rule.

    ``shape`` is accepted for API symmetry and validation, but the locked rule
    intentionally uses native pixel displacement.  This prevents a later
    resize from silently changing the horizontal/vertical decision boundary.
    """

    if shape is not None and (int(shape[0]) <= 0 or int(shape[1]) <= 0):
        raise ValueError("invalid geometric graph image shape")
    if _is_notebook_proxy(geometry_rules):
        return notebook_direction_relation(left, right, geometry_rules=geometry_rules)
    config = _config_from_rules(geometry_rules)
    du, dv = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    dv = -float(config.vertical_scale) * dv
    if du == 0 and dv == 0:
        raise ValueError("coincident graph centers do not define a direction")
    if abs(dv) >= abs(du):
        return "is_in_front_of" if dv > 0 else "is_behind"
    return "is_left_of" if du > 0 else "is_right_of"


def _instruction_relation(instruction: str) -> str | None:
    normalized = " ".join(str(instruction).lower().split())
    # The longer phrase must be tested first so that ``on`` does not consume
    # the prefix of ``on top of``.
    if re.search(r"\bon top of\b", normalized):
        return "is_on_top_of"
    if re.search(r"\b(?:in|inside)\b", normalized):
        return "is_inside"
    if re.search(r"\bon\b", normalized):
        return "is_on_top_of"
    return None


def _is_cabinet_top_storage_instruction(instruction: str) -> bool:
    """Recognize the LIBERO top-layer/top-drawer containment wording.

    The ordinary ``on the wooden cabinet`` task remains a support relation;
    only an explicit top layer or top drawer phrase activates containment.
    """

    normalized = " ".join(str(instruction).lower().split())
    return bool(
        re.search(r"\btop\s+(?:layer|drawer)\b.*\bwooden cabinet\b", normalized)
    )


def _is_notebook_top_drawer_instruction(instruction: str) -> bool:
    """Recognize only the notebook proxy's exact top-drawer wording.

    The generic SamGraph rules intentionally support both ``top layer`` and
    ``top drawer``.  The SamGraph exception is narrower: it must not turn a
    top-layer instruction into the calibrated drawer override.
    """

    normalized = " ".join(str(instruction).lower().split())
    return bool(re.search(r"\btop drawer\b.*\bwooden cabinet\b", normalized))


def _normalized_instance_id(value: Any) -> str:
    """Normalize detector/notebook instance IDs for the drawer exception.

    LIBERO notebook labels may carry the ``akita_`` asset prefix while the
    cached SAM labels use the shorter semantic name.  Only that known prefix
    is removed; arbitrary prefixes remain distinct so this exception cannot
    silently apply to another bowl or cabinet instance.
    """

    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if normalized.startswith("akita_"):
        normalized = normalized[len("akita_"):]
    return normalized


def _is_notebook_drawer_identity_pair(
    subject: Mapping[str, Any], destination: Mapping[str, Any]
) -> bool:
    return (
        _normalized_instance_id(subject.get("instance_id")) == "black_bowl_1"
        and _normalized_instance_id(destination.get("instance_id")) == "wooden_cabinet_1"
    )


def _normalized_center_distance(
    left: np.ndarray, right: np.ndarray, shape: tuple[int, int]
) -> float:
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("invalid geometric graph image shape")
    return float(
        np.linalg.norm((np.asarray(left) - np.asarray(right)) / np.asarray((width, height)))
    )


def support_relation(
    subject: Mapping[str, Any],
    destination: Mapping[str, Any],
    subject_mask: np.ndarray,
    destination_mask: np.ndarray,
    *,
    instruction: str,
    shape: tuple[int, int],
    geometry_rules: GeometricRelationConfig | Mapping[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Classify the visual support/containment relation and record evidence."""

    if _is_notebook_proxy(geometry_rules):
        config = _notebook_proxy_from_rules(geometry_rules)
        subject_center = mask_centroid(subject_mask)
        destination_center = mask_centroid(destination_mask)
        subject_bbox = notebook_mask_bbox_xyxy(
            subject_mask,
            padding_x=config.bbox_padding_x,
            padding_y=config.bbox_padding_y,
        )
        destination_bbox = notebook_mask_bbox_xyxy(
            destination_mask,
            padding_x=config.bbox_padding_x,
            padding_y=config.bbox_padding_y,
        )
        iomin = bbox_iomin(subject_bbox, destination_bbox)
        evidence = {
            "subject": str(subject["instance_id"]),
            "object": str(destination["instance_id"]),
            "bbox_iomin": iomin,
            "containment_threshold": config.containment_iomin_threshold,
            "drawer_mode": config.drawer_mode,
            "qualifies": False,
            "cabinet_exception_applied": False,
            "rules_revision": geometric_relation_revision(config),
        }
        if config.notebook_pair_semantics:
            if "bowl" not in {subject.get("role"), destination.get("role")}:
                return None, evidence
        else:
            if str(subject.get("role", "")) != "bowl":
                return None, evidence
            if str(destination.get("role", "")) not in {"support", "cabinet"}:
                return None, evidence
        qualifies = iomin > config.containment_iomin_threshold
        evidence["qualifies"] = bool(qualifies)
        if not qualifies:
            return None, evidence
        drawer_instruction = _is_notebook_top_drawer_instruction(instruction)
        drawer = False
        if config.notebook_pair_semantics:
            drawer = drawer_instruction and (
                _is_notebook_drawer_identity_pair(subject, destination)
                or _is_notebook_drawer_identity_pair(destination, subject)
            )
        elif config.drawer_mode in {"instruction_overlap", "instruction_overlap_cabinet_band"}:
            destination_bbox = mask_bbox_xyxy_half_open(destination_mask)
            x, y = subject_center
            drawer = (
                drawer_instruction
                and str(destination.get("role", "")) == "cabinet"
                and _is_notebook_drawer_identity_pair(subject, destination)
                # Strict interior overlap avoids treating a centroid on the
                # cabinet boundary as evidence for the special drawer rule.
                and destination_bbox[0] < x < destination_bbox[2]
                and destination_bbox[1] < y < destination_bbox[3]
            )
        top = _notebook_stack_is_top(
            subject, destination, subject_center, destination_center, config, drawer=drawer
        )
        if drawer and top:
            evidence["cabinet_exception_applied"] = True
            return "is_inside", evidence
        if drawer:
            evidence["cabinet_exception_applied"] = True
            return "contains", evidence
        return ("is_on_top_of" if top else "is_below_of"), evidence

    config = _config_from_rules(geometry_rules)
    subject_center = mask_centroid(subject_mask)
    destination_center = mask_centroid(destination_mask)
    destination_hull = convex_hull_mask(destination_mask)
    subject_hull = convex_hull_mask(subject_mask)
    subject_area = max(1, int(np.asarray(subject_mask, dtype=bool).sum()))
    destination_area = max(1, int(np.asarray(destination_mask, dtype=bool).sum()))
    coverage = float(np.logical_and(subject_mask, destination_hull).sum()) / subject_area
    reverse_coverage = float(np.logical_and(destination_mask, subject_hull).sum()) / destination_area
    distance = _normalized_center_distance(subject_center, destination_center, shape)
    evidence: dict[str, Any] = {
        "subject": str(subject["instance_id"]),
        "object": str(destination["instance_id"]),
        "hull_coverage": coverage,
        "reverse_hull_coverage": reverse_coverage,
        "normalized_distance": distance,
        "qualifies": False,
        "cabinet_exception_applied": False,
        "rules_revision": geometric_relation_revision(config),
    }

    if str(subject.get("role", "")) != "bowl":
        return None, evidence
    if str(destination.get("role", "")) not in {"support", "cabinet"}:
        return None, evidence

    qualifies = (
        coverage >= config.hull_coverage_min
        or (
            coverage >= config.hull_coverage_near_min
            and distance <= config.near_distance_max_normalized
        )
        or (
            str(destination.get("class_id", "")) == "cookies"
            and reverse_coverage >= config.cookies_reverse_coverage_min
            and distance <= config.near_distance_max_normalized
        )
    )
    evidence["qualifies"] = bool(qualifies)
    if not qualifies:
        return None, evidence

    top_layer = _is_cabinet_top_storage_instruction(instruction)
    inside = False
    if (
        str(destination.get("role", "")) == "cabinet"
        and top_layer
        and coverage >= config.hull_coverage_min
    ):
        cabinet_ys = np.nonzero(destination_mask)[0]
        inside = bool(
            subject_center[1]
            >= float(
                np.quantile(
                    cabinet_ys,
                    config.cabinet_top_layer_quantile,
                )
            )
        )
        evidence["cabinet_exception_applied"] = inside
    return ("is_inside" if inside else "is_on_top_of"), evidence


def build_directed_relations(
    instances: list[Mapping[str, Any]],
    masks: Mapping[str, np.ndarray],
    *,
    suite: str,
    instruction: str,
    shape: tuple[int, int],
    geometry_rules: GeometricRelationConfig | Mapping[str, Any] | None = None,
    allow_partial_visibility: bool = False,
) -> tuple[dict[tuple[str, str], str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a complete directed graph from current masks.

    The function is deliberately independent of SAM, MuJoCo, HTTP, and the
    graph brain.  Both the initial detector scene and every live tracking frame
    call it with their current mask set.
    """

    config = _config_from_rules(geometry_rules)
    rule_spec: Any = geometry_rules if _is_notebook_proxy(geometry_rules) else config
    if suite not in {"spatial", "object"}:
        raise ValueError(f"unsupported geometric graph suite: {suite!r}")
    by_id = {str(item["instance_id"]): item for item in instances}
    if len(by_id) != len(instances) or len(by_id) < 2:
        raise ValueError("geometric graph requires unique instance IDs")
    normalized_masks = {key: _as_mask(value) for key, value in masks.items()}
    if set(normalized_masks) != set(by_id):
        raise ValueError("geometric graph masks must cover every instance")
    shapes = {mask.shape for mask in normalized_masks.values()}
    if len(shapes) != 1 or next(iter(shapes)) != tuple(shape):
        raise ValueError("geometric graph masks have inconsistent image shapes")
    centers = {key: mask_centroid(value) for key, value in normalized_masks.items()}
    relations: dict[tuple[str, str], str] = {}
    evidence: list[dict[str, Any]] = []

    if suite == "object":
        object_items = [item for item in instances if item.get("role") == "object"]
        destination_items = [item for item in instances if item.get("role") == "destination"]
        relation = _instruction_relation(instruction)
        if relation and len(object_items) == 1 and len(destination_items) == 1:
            source, destination = object_items[0], destination_items[0]
            a, b = str(source["instance_id"]), str(destination["instance_id"])
            relations[(a, b)] = relation
            relations[(b, a)] = INVERSE_RELATIONS[relation]
            evidence.append({
                "subject": a,
                "object": b,
                "relation": relation,
                "source": "instruction_goal_grammar",
                "rules_revision": geometric_relation_revision(config),
            })

    if suite == "spatial":
        for subject in instances:
            notebook_pairs = isinstance(config, NotebookMaskProxyConfig) and config.notebook_pair_semantics
            if subject.get("role") == "cabinet" and not notebook_pairs:
                continue
            qualifying: list[tuple[Mapping[str, Any], str]] = []
            for destination in instances:
                if not notebook_pairs and destination.get("role") not in {"support", "cabinet"}:
                    continue
                if subject["instance_id"] == destination["instance_id"]:
                    continue
                relation, relation_evidence = support_relation(
                    subject,
                    destination,
                    normalized_masks[str(subject["instance_id"])],
                    normalized_masks[str(destination["instance_id"])],
                    instruction=instruction,
                    shape=shape,
                    geometry_rules=rule_spec,
                )
                evidence.append(relation_evidence)
                if relation is not None:
                    qualifying.append((destination, relation))
            if len(qualifying) > 1 and not isinstance(config, NotebookMaskProxyConfig):
                raise ValueError("multiple visual supports are ambiguous")
            for destination, relation in qualifying:
                a, b = str(subject["instance_id"]), str(destination["instance_id"])
                relations[(a, b)] = relation
                if not notebook_pairs:
                    relations[(b, a)] = INVERSE_RELATIONS[relation]
        if _is_cabinet_top_storage_instruction(instruction):
            inside_count = sum(value == "is_inside" for value in relations.values())
            if (
                inside_count > 1 and not isinstance(config, NotebookMaskProxyConfig)
            ) or (inside_count == 0 and not allow_partial_visibility):
                raise ValueError("cabinet exception has no unique visually supported bowl")

    triplets: list[dict[str, Any]] = []
    for subject in instances:
        for destination in instances:
            a, b = str(subject["instance_id"]), str(destination["instance_id"])
            if a == b:
                continue
            relation = relations.get((a, b)) or direction_relation(
                centers[a], centers[b], shape=shape, geometry_rules=rule_spec
            )
            triplets.append({
                "relation_id": f"r{len(triplets):03d}",
                "subject": a,
                "relation": relation,
                "object": b,
            })
    return relations, triplets, evidence


def graph_instance(
    template: Mapping[str, Any],
    mask: np.ndarray,
    *,
    track_state: str | None = None,
    tracking_evidence: str | None = None,
) -> dict[str, Any]:
    """Update one graph instance with current mask geometry."""

    value = _as_mask(mask)
    center = mask_centroid(value)
    x0, y0, x1, y1 = mask_bbox_xyxy(value)
    result = dict(template)
    result.update({
        "area_px": int(value.sum()),
        "center_xy": center.tolist(),
        "bbox_xyxy": [x0, y0, x1, y1],
    })
    if track_state is not None:
        result["track_state"] = str(track_state)
    if tracking_evidence is not None:
        result["tracking_evidence"] = str(tracking_evidence)
    return result


def render_graph_overlay(
    rgb: np.ndarray,
    scene: Mapping[str, Any],
    *,
    selected_pair: tuple[str, str] | None = None,
) -> np.ndarray:
    """Render a diagnostic graph view without changing authoritative RGB.

    One arrow is drawn per unordered entity pair to avoid drawing both inverse
    edges on top of each other.  The complete textual triplets remain in the
    returned graph; this image is only a human-facing diagnostic overlay.
    """

    value = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("graph overlay requires an RGB image")
    image = Image.fromarray(value, mode="RGB")
    draw = ImageDraw.Draw(image)
    by_id = {str(item["instance_id"]): item for item in scene.get("instances", [])}
    seen: set[frozenset[str]] = set()
    selected = frozenset(selected_pair) if selected_pair is not None else frozenset()
    triplets = list(scene.get("triplets", []))
    if selected_pair is not None:
        selected_triplets = [
            item for item in triplets
            if str(item.get("subject")) == str(selected_pair[0])
            and str(item.get("object")) == str(selected_pair[1])
        ]
        triplets = selected_triplets + [
            item for item in triplets if item not in selected_triplets
        ]
    for triplet in triplets:
        subject_id, object_id = str(triplet["subject"]), str(triplet["object"])
        pair = frozenset((subject_id, object_id))
        if len(pair) != 2 or pair in seen:
            continue
        seen.add(pair)
        subject, object_item = by_id.get(subject_id), by_id.get(object_id)
        if (subject is None or object_item is None or subject.get("center_xy") is None
                or object_item.get("center_xy") is None):
            continue
        start = tuple(round(float(item)) for item in subject["center_xy"])
        end = tuple(round(float(item)) for item in object_item["center_xy"])
        if start == end:
            continue
        color = (0, 220, 120)
        draw.line((start, end), fill=color, width=2)
        dx, dy = end[0] - start[0], end[1] - start[1]
        norm = max(1.0, float(np.hypot(dx, dy)))
        ux, uy = dx / norm, dy / norm
        px, py = -uy, ux
        head = min(12.0, max(5.0, norm * 0.18))
        wings = [
            (round(end[0] - head * ux + head * 0.55 * px), round(end[1] - head * uy + head * 0.55 * py)),
            (round(end[0] - head * ux - head * 0.55 * px), round(end[1] - head * uy - head * 0.55 * py)),
        ]
        draw.polygon([end, *wings], fill=color)
    for instance_id, item in by_id.items():
        if item.get("center_xy") is None:
            continue
        center = tuple(round(float(item)) for item in item["center_xy"])
        radius = 3 if instance_id not in selected else 5
        color = (0, 255, 120) if instance_id in selected else (255, 220, 40)
        draw.ellipse(
            (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
            outline=color,
            width=2,
        )
    missing = [key for key, item in by_id.items() if item.get("center_xy") is None]
    if missing:
        draw.rectangle((0, 0, value.shape[1], 12 * len(missing)), fill=(20, 20, 20))
        for index, key in enumerate(missing):
            draw.text((2, 12 * index), f"unobserved: {key}", fill=(255, 190, 70))
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


__all__ = [
    "DEFAULT_GEOMETRIC_RELATION_CONFIG",
    "GEOMETRIC_GRAPH_SCHEMA",
    "GEOMETRIC_RELATION_RULES",
    "GEOMETRIC_RELATION_RULES_REVISION",
    "NOTEBOOK_MASK_PROXY_RULES_REVISION",
    "GeometricRelationConfig",
    "NotebookMaskProxyConfig",
    "INVERSE_RELATIONS",
    "PIXEL_FRAME",
    "build_directed_relations",
    "convex_hull_mask",
    "direction_relation",
    "geometric_relation_rules",
    "geometric_relation_revision",
    "geometric_relation_rules_sha256",
    "graph_instance",
    "bbox_iomin",
    "mask_bbox_xyxy",
    "mask_bbox_xyxy_half_open",
    "mask_bbox_half_open",
    "mask_centroid",
    "mask_iou",
    "render_graph_overlay",
    "notebook_direction_relation",
    "notebook_mask_bbox_xyxy",
    "notebook_mask_bbox",
    "support_relation",
]
