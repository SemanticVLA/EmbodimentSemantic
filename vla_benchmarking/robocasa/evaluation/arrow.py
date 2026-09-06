"""Input-side RoboCasa arrow generation.

The simulator-side caller supplies projected bboxes. The controller sees only
the resulting RGB arrow image; object names and simulator state never cross
this boundary. Rendering intentionally matches the frozen LIBERO parked-arrow
input contract while depending only on RoboCasa-local policy constants.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..arrow_grasp_controller.controller.policy import (
    ARROW_LONG_HEAD_LENGTH_PX,
    ARROW_LONG_LINE_WIDTH,
    ARROW_SHORT_HEAD_FRACTION,
    ARROW_SHORT_HEAD_MAX_PX,
    ARROW_SHORT_HEAD_MIN_PX,
    ARROW_SHORT_LINE_WIDTH,
    ARROW_SHORT_SPAN_THRESHOLD_PX,
)


_ARROW_COLOR_RGB = (0, 166, 107)


def _validated_bbox(value: Sequence[float], *, name: str) -> tuple[float, float, float, float]:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError(f"{name} bbox must be four finite values")
    x1, y1, x2, y2 = (float(item) for item in values)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} bbox must have positive area")
    return x1, y1, x2, y2


def _bbox_center_float(bbox: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_center(bbox: Sequence[float]) -> tuple[int, int]:
    """Return the Python-rounded center of an ``(x1, y1, x2, y2)`` bbox."""

    validated = _validated_bbox(bbox, name="object")
    center_x, center_y = _bbox_center_float(validated)
    return round(center_x), round(center_y)


def _resolved_render_params(
    source_center: tuple[float, float],
    destination_center: tuple[float, float],
) -> dict[str, Any]:
    span = float(
        np.linalg.norm(
            np.asarray(destination_center, dtype=np.float64)
            - np.asarray(source_center, dtype=np.float64)
        )
    )
    if not math.isfinite(span):
        raise ValueError("arrow endpoint span is non-finite")
    if span < ARROW_SHORT_SPAN_THRESHOLD_PX:
        return {
            "endpoint_span_px": span,
            "line_width": ARROW_SHORT_LINE_WIDTH,
            "head_length": max(
                ARROW_SHORT_HEAD_MIN_PX,
                min(
                    ARROW_SHORT_HEAD_MAX_PX,
                    round(ARROW_SHORT_HEAD_FRACTION * span),
                ),
            ),
            "render_policy": "adaptive_short_v1",
        }
    return {
        "endpoint_span_px": span,
        "line_width": ARROW_LONG_LINE_WIDTH,
        "head_length": ARROW_LONG_HEAD_LENGTH_PX,
        "render_policy": "adaptive_short_v1_long_default",
    }


def _render_arrow(
    clean_rgb: np.ndarray,
    source: tuple[int, int],
    destination: tuple[int, int],
    *,
    line_width: int,
    head_length: int,
    allow_fallback: bool,
) -> tuple[np.ndarray, str, str | None, bool]:
    """Render with the sole benchmark backend, OpenCV.

    The benchmark path leaves ``allow_fallback`` false and therefore fails
    closed. The opt-in Pillow path exists only for dependency-light contract
    tests and compatibility callers; it is recorded explicitly in the audit.
    """

    try:
        import cv2
    except ImportError as exc:
        if not allow_fallback:
            raise RuntimeError(
                "OpenCV is required for the frozen RoboCasa arrow renderer; "
                "install the pinned RoboCasa requirements"
            ) from exc
        try:
            from PIL import Image, ImageDraw
        except ImportError as pillow_exc:  # pragma: no cover - minimal env only
            raise RuntimeError("Pillow is required for compatibility rendering") from pillow_exc
        image = Image.fromarray(np.asarray(clean_rgb, dtype=np.uint8).copy())
        draw = ImageDraw.Draw(image)
        draw.line([source, destination], fill=_ARROW_COLOR_RGB, width=int(line_width))
        dx = destination[0] - source[0]
        dy = destination[1] - source[1]
        length = max(float(math.hypot(dx, dy)), 1.0)
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        wing_half_width = head_length / 2.0
        left = (
            destination[0] - head_length * ux + wing_half_width * px,
            destination[1] - head_length * uy + wing_half_width * py,
        )
        right = (
            destination[0] - head_length * ux - wing_half_width * px,
            destination[1] - head_length * uy - wing_half_width * py,
        )
        draw.polygon([destination, left, right], fill=_ARROW_COLOR_RGB)
        return np.asarray(image, dtype=np.uint8), "pillow_compat", None, True

    canvas = np.ascontiguousarray(np.asarray(clean_rgb, dtype=np.uint8).copy())
    cv2.line(
        canvas,
        source,
        destination,
        _ARROW_COLOR_RGB,
        thickness=int(line_width),
        lineType=cv2.LINE_AA,
    )
    angle = math.atan2(destination[1] - source[1], destination[0] - source[0])
    wing_a = (
        round(destination[0] - head_length * math.cos(angle - math.pi / 6)),
        round(destination[1] - head_length * math.sin(angle - math.pi / 6)),
    )
    wing_b = (
        round(destination[0] - head_length * math.cos(angle + math.pi / 6)),
        round(destination[1] - head_length * math.sin(angle + math.pi / 6)),
    )
    cv2.fillConvexPoly(
        canvas,
        np.asarray([destination, wing_a, wing_b], dtype=np.int32),
        _ARROW_COLOR_RGB,
        lineType=cv2.LINE_AA,
    )
    return canvas, "opencv", str(cv2.__version__), False


def render_bbox_center_arrow(
    clean_rgb: np.ndarray,
    bboxes: Mapping[str, Sequence[float]],
    *,
    source: str,
    destination: str,
    allow_fallback: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render one source-center to destination-center adaptive arrow locally."""

    image = np.asarray(clean_rgb)
    if image.dtype != np.uint8:
        raise TypeError(f"clean_rgb must have dtype uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("clean_rgb must have shape HxWx3")
    if source not in bboxes or destination not in bboxes:
        raise KeyError(f"missing source/destination bbox: {source!r}, {destination!r}")
    source_bbox = _validated_bbox(bboxes[source], name="source")
    destination_bbox = _validated_bbox(bboxes[destination], name="destination")
    source_center = _bbox_center_float(source_bbox)
    destination_center = _bbox_center_float(destination_bbox)
    source_uv = (round(source_center[0]), round(source_center[1]))
    destination_uv = (round(destination_center[0]), round(destination_center[1]))
    render_params = _resolved_render_params(source_center, destination_center)

    arrow, renderer_backend, renderer_version, fallback_used = _render_arrow(
        image,
        source_uv,
        destination_uv,
        line_width=int(render_params["line_width"]),
        head_length=int(render_params["head_length"]),
        allow_fallback=bool(allow_fallback),
    )
    audit: dict[str, Any] = {
        "source": str(source),
        "destination": str(destination),
        "source_bbox": list(source_bbox),
        "destination_bbox": list(destination_bbox),
        "source_center_uv": list(source_uv),
        "destination_center_uv": list(destination_uv),
        "anchor_policy": "bbox_center",
        "input_generation_only": True,
        "renderer": "robocasa_local_frozen_style",
        "renderer_backend": renderer_backend,
        "renderer_version": renderer_version,
        "arrow_color_rgb": list(_ARROW_COLOR_RGB),
        "endpoint_span_px": float(render_params["endpoint_span_px"]),
        "line_width": int(render_params["line_width"]),
        "head_length": int(render_params["head_length"]),
        "render_policy": str(render_params["render_policy"]),
        "short_span_threshold_px": int(ARROW_SHORT_SPAN_THRESHOLD_PX),
        "allow_fallback_compat": bool(allow_fallback),
        "fallback_used": fallback_used,
    }
    if arrow.shape != image.shape or np.array_equal(arrow, image):
        raise ValueError("arrow renderer did not produce a changed RGB frame")
    return np.asarray(arrow, dtype=np.uint8), audit


__all__ = ["bbox_center", "render_bbox_center_arrow"]
