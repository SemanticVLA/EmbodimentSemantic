"""Input-side RoboCasa arrow generation.

The simulator-side caller supplies projected bboxes.  The controller sees
only the resulting RGB arrow image; object names and simulator state never
cross this boundary.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _validated_bbox(value: Sequence[float], *, name: str) -> tuple[float, float, float, float]:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError(f"{name} bbox must be four finite values")
    x1, y1, x2, y2 = (float(item) for item in values)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} bbox must have positive area")
    return x1, y1, x2, y2


def bbox_center(bbox: Sequence[float]) -> tuple[int, int]:
    """Return the rounded center of an ``(x1, y1, x2, y2)`` bbox."""

    x1, y1, x2, y2 = _validated_bbox(bbox, name="object")
    return round((x1 + x2) / 2.0), round((y1 + y2) / 2.0)


def _fallback_arrow(
    clean_rgb: np.ndarray,
    source: tuple[int, int],
    destination: tuple[int, int],
    *,
    allow_fallback: bool,
) -> np.ndarray:
    """Render the frozen one-arrow style without importing another suite."""

    try:
        import cv2
        import math

        canvas = np.ascontiguousarray(np.asarray(clean_rgb, dtype=np.uint8).copy())
        colour = (0, 166, 107)
        cv2.line(canvas, source, destination, colour, thickness=1, lineType=cv2.LINE_AA)
        angle = math.atan2(destination[1] - source[1], destination[0] - source[0])
        wings = np.asarray([
            destination,
            (round(destination[0] - 16 * math.cos(angle - math.pi / 6)),
             round(destination[1] - 16 * math.sin(angle - math.pi / 6))),
            (round(destination[0] - 16 * math.cos(angle + math.pi / 6)),
             round(destination[1] - 16 * math.sin(angle + math.pi / 6))),
        ], dtype=np.int32)
        cv2.fillConvexPoly(canvas, wings, colour, lineType=cv2.LINE_AA)
        return canvas
    except ImportError as exc:
        if not allow_fallback:
            raise RuntimeError(
                "OpenCV is required for the frozen RoboCasa arrow renderer; "
                "install the pinned RoboCasa requirements or explicitly enable "
                "the compatibility Pillow fallback"
            ) from exc

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - only minimal envs
        raise RuntimeError("Pillow is required to render a RoboCasa arrow") from exc
    image = Image.fromarray(np.asarray(clean_rgb, dtype=np.uint8).copy())
    draw = ImageDraw.Draw(image)
    colour = (0, 166, 107)
    draw.line([source, destination], fill=colour, width=1)
    dx = destination[0] - source[0]
    dy = destination[1] - source[1]
    length = max(float((dx * dx + dy * dy) ** 0.5), 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = destination
    left = (tip[0] - 16 * ux + 8 * px, tip[1] - 16 * uy + 8 * py)
    right = (tip[0] - 16 * ux - 8 * px, tip[1] - 16 * uy - 8 * py)
    draw.polygon([tip, left, right], fill=colour)
    return np.asarray(image, dtype=np.uint8)


def render_bbox_center_arrow(
    clean_rgb: np.ndarray,
    bboxes: Mapping[str, Sequence[float]],
    *,
    source: str,
    destination: str,
    allow_fallback: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render one source-center to destination-center arrow locally."""

    image = np.asarray(clean_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("clean_rgb must have shape HxWx3")
    if source not in bboxes or destination not in bboxes:
        raise KeyError(f"missing source/destination bbox: {source!r}, {destination!r}")
    source_bbox = _validated_bbox(bboxes[source], name="source")
    destination_bbox = _validated_bbox(bboxes[destination], name="destination")
    source_uv = bbox_center(source_bbox)
    destination_uv = bbox_center(destination_bbox)

    audit: dict[str, Any] = {
        "source": str(source),
        "destination": str(destination),
        "source_bbox": list(source_bbox),
        "destination_bbox": list(destination_bbox),
        "source_center_uv": list(source_uv),
        "destination_center_uv": list(destination_uv),
        "anchor_policy": "bbox_center",
        "input_generation_only": True,
    }
    arrow = _fallback_arrow(
        image,
        source_uv,
        destination_uv,
        allow_fallback=bool(allow_fallback),
    )
    try:
        import cv2  # type: ignore
        renderer = "opencv"
        renderer_version = str(cv2.__version__)
    except ImportError:
        renderer = "pillow_compat"
        renderer_version = None
    audit.update({
        "renderer": "robocasa_local_frozen_style",
        "renderer_backend": renderer,
        "renderer_version": renderer_version,
        "line_width": 1,
        "head_length": 16,
        "allow_fallback_compat": bool(allow_fallback),
    })
    if arrow.shape != image.shape or np.array_equal(arrow, image):
        raise ValueError("arrow renderer did not produce a changed RGB frame")
    return np.asarray(arrow, dtype=np.uint8), audit


__all__ = ["bbox_center", "render_bbox_center_arrow"]
