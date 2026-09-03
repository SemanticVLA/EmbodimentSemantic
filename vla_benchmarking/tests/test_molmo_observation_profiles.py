"""Focused contracts for the bounded observation-hover profiles."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from vla_benchmarking import run_molmo_sam3_canary as runner


class _Calibration:
    camera_name = runner.AGENTVIEW
    pixel_origin = "top_left"
    camera_frame = "camera"
    world_frame = "world"
    extrinsic_direction = "world_from_camera"
    rgb_depth_alignment = "aligned"
    intrinsic = np.asarray(((20.0, 0.0, 8.0), (0.0, 20.0, 8.0), (0.0, 0.0, 1.0)))
    world_from_camera = np.asarray(((1.0, 0.0, 0.0, 0.1), (0.0, 1.0, 0.0, 0.2), (0.0, 0.0, 1.0, 0.2), (0.0, 0.0, 0.0, 1.0)))


def _capture():
    return SimpleNamespace(
        rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        metric_depth=np.ones((16, 16), dtype=np.float64),
        calibration=_Calibration(),
    )


def _fake_episode(monkeypatch, *, position_error=0.015699, height_error=0.0, final_orientation=None):
    fake_episode = ModuleType("run_arrow_pick_place_eval")
    state = {"final": np.asarray((0.1, 0.2, 1.3 - height_error), dtype=np.float64)}
    fake_episode._raw_observation = lambda _env: {
        "eef_pos": state["final"],
        "eef_quat": np.asarray((0.0, 0.0, 0.0, 1.0)) if final_orientation is None else np.asarray(final_orientation),
    }
    fake_episode._proprioception = lambda observation: observation
    fake_episode._pose_waypoint = lambda position, orientation=None: {
        "position": np.asarray(position, dtype=np.float64),
        "orientation": np.asarray(orientation, dtype=np.float64),
    }
    fake_episode.validate_workspace_points = lambda _points: None

    def run_motion(_env, waypoints, _observation, **kwargs):
        state["final"] = np.asarray(waypoints["hover_lower"]["position"], dtype=np.float64)
        state["final"][2] -= height_error
        phases = []
        for phase in kwargs["phase_order"]:
            commanded = np.asarray(waypoints[phase]["position"], dtype=np.float64)
            actual = commanded.copy()
            actual[0] += position_error
            actual[2] -= height_error
            phases.append({
                "phase": phase,
                "status": "reached",
                "eef_pos_m": actual.tolist(),
                "position_error_norm_m": float(position_error),
                "orientation_error_rad": 0.0,
            })
        return phases

    fake_episode._run_motion = run_motion
    monkeypatch.setitem(sys.modules, "vla_benchmarking.run_arrow_pick_place_eval", fake_episode)
    monkeypatch.setitem(sys.modules, "run_arrow_pick_place_eval", fake_episode)


@pytest.fixture
def region_mask(monkeypatch):
    rgbd_region = __import__("vla_benchmarking.rgbd_region", fromlist=["derive_observed_region_mask"])
    monkeypatch.setattr(
        rgbd_region,
        "derive_observed_region_mask",
        lambda capture, source_uv, profile_target_world: (
            np.ones(np.asarray(capture.metric_depth).shape, dtype=bool), {"source": "test"}
        ),
    )


def _run_hover(tmp_path, monkeypatch, region_mask, **kwargs):
    del region_mask
    _fake_episode(monkeypatch, **kwargs)
    env = SimpleNamespace(_molmo_sam3_action_count=0)
    return runner._perform_observation_hover(
        env, _capture(), (8.0, 8.0), output_dir=tmp_path,
        motion_settings={"observation_profile": runner.resolve_observation_profile("hover20mm")},
    )


def test_hover20mm_allows_15_699mm_error_and_records_actual_transform(tmp_path, monkeypatch, region_mask):
    audit = _run_hover(tmp_path, monkeypatch, region_mask)
    assert audit["phase_tolerance_m"] == pytest.approx(0.020)
    assert audit["orientation_tolerance_rad"] == pytest.approx(0.12)
    assert audit["final_fresh_proprio"]["actual_height_margin_m"] >= 0.08
    assert audit["capture_provenance"]["world_from_camera"][2][3] == pytest.approx(0.2)
    assert len(audit["phase_motion_audit"]) == 3


def test_hover20mm_rejects_error_above_tolerance_and_low_actual_height(tmp_path, monkeypatch, region_mask):
    with pytest.raises(RuntimeError, match="position error exceeds"):
        _run_hover(tmp_path, monkeypatch, region_mask, position_error=0.020001)
    with pytest.raises(RuntimeError, match="actual height margin"):
        _run_hover(tmp_path, monkeypatch, region_mask, position_error=0.010, height_error=0.04)


def test_hover20mm_rejects_invalid_final_orientation(tmp_path, monkeypatch, region_mask):
    with pytest.raises(RuntimeError, match="invalid orientation"):
        _run_hover(tmp_path, monkeypatch, region_mask, position_error=0.010, final_orientation=(0.0, 0.0, 0.0, 0.0))


def test_baseline_profile_keeps_existing_fifteen_mm_policy():
    baseline = runner.resolve_observation_profile("baseline")
    treatment = runner.resolve_observation_profile("hover20mm")
    assert baseline["phase_tolerance_m"] == pytest.approx(0.015)
    assert treatment["phase_tolerance_m"] == pytest.approx(0.020)
    assert baseline["target_offset_m"] == treatment["target_offset_m"] == pytest.approx(0.10)
    assert baseline["orientation_tolerance_rad"] == treatment["orientation_tolerance_rad"] == pytest.approx(0.12)
