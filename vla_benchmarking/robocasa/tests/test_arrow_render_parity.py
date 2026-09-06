"""Pixel-exact tests for the RoboCasa copy of the LIBERO arrow contract."""

from __future__ import annotations

import builtins
import math

import numpy as np
import pytest

from vla_benchmarking.robocasa.arrow_grasp_controller.controller.policy import (
    ARROW_LONG_HEAD_LENGTH_PX,
    ARROW_LONG_LINE_WIDTH,
    ARROW_SHORT_HEAD_FRACTION,
    ARROW_SHORT_HEAD_MAX_PX,
    ARROW_SHORT_HEAD_MIN_PX,
    ARROW_SHORT_LINE_WIDTH,
    ARROW_SHORT_SPAN_THRESHOLD_PX,
)
from vla_benchmarking.robocasa.evaluation.arrow import render_bbox_center_arrow


def _reference_render(
    clean_rgb: np.ndarray,
    source_bbox: tuple[float, float, float, float],
    destination_bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, dict[str, float | int]]:
    cv2 = pytest.importorskip("cv2")
    source_center = (
        (source_bbox[0] + source_bbox[2]) / 2.0,
        (source_bbox[1] + source_bbox[3]) / 2.0,
    )
    destination_center = (
        (destination_bbox[0] + destination_bbox[2]) / 2.0,
        (destination_bbox[1] + destination_bbox[3]) / 2.0,
    )
    span = float(
        np.linalg.norm(
            np.asarray(destination_center, dtype=np.float64)
            - np.asarray(source_center, dtype=np.float64)
        )
    )
    if span < ARROW_SHORT_SPAN_THRESHOLD_PX:
        line_width = ARROW_SHORT_LINE_WIDTH
        head_length = max(
            ARROW_SHORT_HEAD_MIN_PX,
            min(
                ARROW_SHORT_HEAD_MAX_PX,
                round(ARROW_SHORT_HEAD_FRACTION * span),
            ),
        )
    else:
        line_width = ARROW_LONG_LINE_WIDTH
        head_length = ARROW_LONG_HEAD_LENGTH_PX

    source = (round(source_center[0]), round(source_center[1]))
    destination = (round(destination_center[0]), round(destination_center[1]))
    output = np.ascontiguousarray(clean_rgb.copy())
    color = (0, 166, 107)
    cv2.line(
        output,
        source,
        destination,
        color,
        thickness=line_width,
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
        output,
        np.asarray([destination, wing_a, wing_b], dtype=np.int32),
        color,
        lineType=cv2.LINE_AA,
    )
    return output, {
        "span": span,
        "line_width": line_width,
        "head_length": head_length,
    }


@pytest.mark.parametrize(
    ("source_bbox", "destination_bbox"),
    [
        ((4.0, 6.0, 12.0, 14.0), (23.0, 17.0, 31.0, 25.0)),
        ((5.0, 9.0, 18.0, 22.0), (72.0, 63.0, 87.0, 76.0)),
    ],
    ids=("adaptive-short", "long-default"),
)
def test_pixel_exact_parity_for_adaptive_arrow(
    source_bbox: tuple[float, float, float, float],
    destination_bbox: tuple[float, float, float, float],
) -> None:
    pytest.importorskip("cv2")
    rgb = np.full((96, 112, 3), 29, dtype=np.uint8)
    bboxes = {"object": source_bbox, "destination": destination_bbox}

    actual, audit = render_bbox_center_arrow(
        rgb,
        bboxes,
        source="object",
        destination="destination",
    )
    expected, params = _reference_render(rgb, source_bbox, destination_bbox)

    assert np.array_equal(actual, expected)
    assert audit["endpoint_span_px"] == pytest.approx(params["span"])
    assert audit["line_width"] == params["line_width"]
    assert audit["head_length"] == params["head_length"]
    assert audit["arrow_color_rgb"] == [0, 166, 107]
    assert audit["renderer_backend"] == "opencv"
    assert audit["fallback_used"] is False


def test_short_threshold_is_strictly_less_than_32_pixels() -> None:
    pytest.importorskip("cv2")
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    _, audit = render_bbox_center_arrow(
        rgb,
        {
            "object": (4.0, 4.0, 8.0, 8.0),
            "destination": (36.0, 4.0, 40.0, 8.0),
        },
        source="object",
        destination="destination",
    )

    assert audit["endpoint_span_px"] == pytest.approx(32.0)
    assert audit["line_width"] == ARROW_LONG_LINE_WIDTH
    assert audit["head_length"] == ARROW_LONG_HEAD_LENGTH_PX
    assert audit["render_policy"] == "adaptive_short_v1_long_default"


def test_missing_opencv_fails_closed_in_benchmark_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def rejecting_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "cv2":
            raise ImportError("simulated missing OpenCV")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", rejecting_import)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    bboxes = {"object": (2, 4, 10, 12), "destination": (20, 16, 28, 24)}

    with pytest.raises(RuntimeError, match="OpenCV is required"):
        render_bbox_center_arrow(
            rgb,
            bboxes,
            source="object",
            destination="destination",
            allow_fallback=False,
        )
