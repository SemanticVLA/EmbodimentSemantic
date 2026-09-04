"""Focused contracts for the parked short-arrow rendering seam."""

from __future__ import annotations

import math

import pytest

from vla_benchmarking import run_molmo_sam3_canary as runner


class _Episode:
    @staticmethod
    def _arrow_anchor_bboxes(bboxes, *, subject, image_shape, policy):
        assert subject in bboxes
        assert image_shape == (256, 256)
        assert policy == "bbox_center"
        return dict(bboxes)


def _bbox(center_x: float, center_y: float) -> tuple[float, float, float, float]:
    return (center_x - 1.0, center_y - 1.0, center_x + 1.0, center_y + 1.0)


def test_short_arrow_exact_repro_uses_rounded_renderer_centers():
    params = runner._parked_arrow_render_params(
        _Episode(), {"bowl": _bbox(204, 68), "plate": _bbox(194, 66)},
        subject="bowl", goal_object="plate", image_shape=(256, 256),
    )
    assert params["rounded_source_center"] == [204, 68]
    assert params["rounded_target_center"] == [194, 66]
    assert params["endpoint_span_px"] == pytest.approx(math.sqrt(104.0))
    assert params["line_width"] == 1
    assert params["head_length"] == 4
    assert params["render_policy"] == "adaptive_short_v1"
    t9 = runner._parked_arrow_render_params(
        _Episode(), {"bowl": _bbox(193, 85), "plate": _bbox(200, 72)},
        subject="bowl", goal_object="plate", image_shape=(256, 256),
    )
    assert t9["rounded_source_center"] == [193, 85]
    assert t9["rounded_target_center"] == [200, 72]
    assert t9["endpoint_span_px"] == pytest.approx(math.sqrt(218.0))
    assert t9["line_width"] == 1
    assert t9["head_length"] == 5


@pytest.mark.parametrize("span", [12, 15, 16, 20, 24, 28, 31])
def test_short_arrow_yaw_endpoint_spans_are_bounded(span):
    params = runner._parked_arrow_render_params(
        _Episode(), {"bowl": _bbox(100, 100), "plate": _bbox(100 + span, 100)},
        subject="bowl", goal_object="plate", image_shape=(256, 256),
    )
    assert params["line_width"] == 1
    assert params["head_length"] == max(3, min(16, round(0.35 * span)))


def test_tiny_arrow_stays_decoder_gated_and_long_arrow_keeps_defaults():
    tiny = runner._parked_arrow_render_params(
        _Episode(), {"bowl": _bbox(40, 40), "plate": _bbox(46, 40)},
        subject="bowl", goal_object="plate", image_shape=(256, 256),
    )
    assert tiny["line_width"] == 1
    assert tiny["head_length"] == 3
    # No endpoint/controller bypass is supplied by this policy; the existing
    # decoder remains the rejection gate for ambiguous tiny rendered arrows.
    long = runner._parked_arrow_render_params(
        _Episode(), {"bowl": _bbox(40, 40), "plate": _bbox(72, 40)},
        subject="bowl", goal_object="plate", image_shape=(256, 256),
    )
    assert long["line_width"] == 2
    assert long["head_length"] == 16
    assert long["render_policy"] == "adaptive_short_v1_long_default"
    assert runner.resolve_observation_profile("baseline")["name"] == "baseline"


def test_parked_profile_identity_records_adaptive_render_rule():
    parked = runner.resolve_observation_profile("parked")
    assert parked["arrow_render_policy"] == "adaptive_short_v1"
    assert parked["arrow_short_span_threshold_px"] == 32
    assert parked["arrow_short_head_length_rule"] == "max(3,min(16,round(0.35*span_px)))"
