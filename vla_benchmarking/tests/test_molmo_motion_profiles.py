"""Focused contracts for the bounded Molmo canary motion treatments."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from vla_benchmarking import run_molmo_sam3_canary as runner


@dataclass
class _Calibration:
    camera_name: str = runner.AGENTVIEW
    pixel_origin: str = "top_left"
    camera_frame: str = "camera"
    world_frame: str = "world"
    extrinsic_direction: str = "world_from_camera"
    rgb_depth_alignment: str = "aligned"


@dataclass
class _Capture:
    calibration: _Calibration
    rgb: np.ndarray
    metric_depth: np.ndarray


def test_motion_profile_resolution_is_explicit_and_rgbd_gated():
    baseline = runner.resolve_motion_profile("baseline", region_backend="sam3")
    assert baseline["micro_correction"] is None
    treatment = runner.resolve_motion_profile("placement_micro5mm", region_backend="rgbd")
    assert treatment["micro_correction"]["phases"] == ("preplace", "descend_place")
    assert treatment["micro_correction"]["residual_max_m"] == pytest.approx(0.005)
    with pytest.raises(ValueError, match="requires --region-backend rgbd"):
        runner.resolve_motion_profile("placement_micro5mm", region_backend="sam3")


def test_profile_manifest_records_provenance_and_configuration_hash(tmp_path):
    capture = _Capture(_Calibration(), np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4), dtype=float))
    candidate = runner.GraspCandidate("candidate_0", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), 0.04)

    class Worker:
        def propose(self, request):
            return {"candidates": [candidate], "diagnostics": {"worker": "fake"}}

    def capture_fn(_env, *, resolution, camera_name):
        del resolution
        return _Capture(_Calibration(camera_name=camera_name), capture.rgb, capture.metric_depth)

    seen = []

    def episode_runner(*, context, evaluator):
        del evaluator
        seen.append(context.candidate.candidate_id)
        return {"status": "selected", "retreat_complete": True, "total_actions": 0}

    treatment = runner.resolve_motion_profile("placement_micro5mm", region_backend="rgbd")
    manifest = runner.run_canary_episode(
        env=object(), task_id=4, seed=1000, output_dir=tmp_path,
        variant=runner.VARIANTS["rgbd_geometry_agentview"], worker=Worker(),
        episode_runner=episode_runner, source_uv=(1.0, 1.0), capture_fn=capture_fn,
        dry_run=True, region_backend="rgbd", motion_profile=treatment["name"],
        motion_profile_params=treatment, motion_diagnostics=True,
    )
    assert seen == ["candidate_0"]
    assert manifest["motion_profile"] == "placement_micro5mm"
    assert manifest["motion_profile_params"]["micro_correction"]["max_actions"] == 16
    assert manifest["motion_diagnostics"] is True
    assert manifest["experiment_config_hash"]


def test_diagnostics_forward_to_helper_using_backend_keyword(tmp_path, monkeypatch):
    """The helper path must use the backend's explicit diagnostics parameter."""
    fake_episode = ModuleType("run_arrow_pick_place_eval")
    fake_episode._raw_observation = lambda _env: {"eef_pos": np.asarray((0.1, 0.2, 0.3))}
    fake_episode._proprioception = lambda observation: observation
    seen = []

    def run_motion(
        _env, _waypoints, _observation, *, phase_timeout_steps,
        gripper_dwell_steps, stop_after_phase, start_phase, dry_run,
        action_budget, motion_started_callback, motion_trace_path,
        motion_trace_segment, stall_window_steps, stall_delta_m,
        osc_position_scale_m, phase_policies, experimental_motion_diagnostics,
    ):
        del (
            phase_timeout_steps, gripper_dwell_steps, stop_after_phase,
            start_phase, dry_run, action_budget, motion_started_callback,
            motion_trace_path, motion_trace_segment, stall_window_steps,
            stall_delta_m, osc_position_scale_m, phase_policies,
        )
        seen.append(experimental_motion_diagnostics)
        return [{"phase": "open", "steps": 1, "status": "stop"}]

    fake_episode._run_motion = run_motion
    monkeypatch.setitem(sys.modules, "vla_benchmarking.run_arrow_pick_place_eval", fake_episode)
    monkeypatch.setitem(sys.modules, "run_arrow_pick_place_eval", fake_episode)
    env = SimpleNamespace(_molmo_sam3_action_count=0)
    runner._perform_gripper_open(
        env, output_dir=tmp_path,
        motion_settings={"motion_diagnostics": True},
    )
    assert seen == [True]
