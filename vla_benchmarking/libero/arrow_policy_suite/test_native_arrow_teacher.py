from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arrow_policy_suite.contracts import ContractError, ObservationFrame
from arrow_policy_suite.native_arrow_teacher import PerFrameArrowTeacher


def _frame(step: int = 0, x: float = 0.0, rotvec=(0.0, 0.0, 0.0)) -> ObservationFrame:
    # Canonical LIBERO state convention: EEF xyz, axis-angle rotvec, gripper.
    return ObservationFrame({"state": [x, 0.0, 0.5, *rotvec, 0.0, 0.0]}, timestep=step)


def test_phase_teacher_proposes_one_action_and_advances_only_after_commit():
    calls = []

    def perceive(frame):
        calls.append(frame.timestep)
        waypoints = [[0.01, 0.0, 0.5], [0.02, 0.0, 0.5], [0.03, 0.0, 0.5],
                     [0.04, 0.0, 0.5], [0.05, 0.0, 0.5], [0.06, 0.0, 0.5]]
        return {"candidate_id": "c0", "waypoints": waypoints, "provenance": {"arrow": "synthetic"}}

    teacher = PerFrameArrowTeacher(perceive, gripper_dwell_steps=1)
    first = teacher.propose(_frame())
    assert first is not None
    assert first.metadata["phase"] == "pregrasp"
    assert calls == [0]
    # A second proposal in the same frame is idempotent and does not re-run
    # perception or advance the phase.
    assert teacher.propose(_frame()) == first
    assert first.metadata["phase"] == "pregrasp"
    teacher.commit(SimpleNamespace(base=first, teacher=first, next_frame=_frame(1, 0.01)))
    second = teacher.propose(_frame(1, 0.01))
    assert second is not None and second.metadata["phase"] == "descend"
    assert calls == [0]


def test_teacher_filters_full_geometry_audit_before_policy_metadata_boundary():
    geometry_audit = {
        "seed_audit": [{"seed_index": 0, "source": "observed_upper_rim"}],
        "candidate_grid_size": 12,
        "returned_count": 1,
        "current_pose_available": True,
        "hand_collision_sphere_count": 0,
        "hand_collision_box_count": 0,
        "terminal_contact_allowance_m": 0.012,
        "contact_mode": "observed_upper_rim",
        "rim_height_quantile": 0.8,
        "rim_height_m": 0.21,
        "upper_rim_threshold_m": 0.2,
        "rim_height_band_m": 0.01,
        "rim_local_radius_m": 0.02,
        "obstruction_clearance_m": 0.005,
        "robot_exclusion_clearance_m": 0.01,
    }

    def perceive(_frame):
        return {
            "candidate_id": "c0",
            "waypoints": [[0.01, 0.0, 0.5]] * 6,
            "provenance": {
                "diagnostics": {
                    "backend": "rgbd_region",
                    "geometry_audit": geometry_audit,
                    "region_audit": {"mask_pixels": 12},
                }
            },
        }

    proposal = PerFrameArrowTeacher(perceive, gripper_dwell_steps=1).propose(_frame())
    assert proposal is not None
    assert "plan_provenance" not in proposal.metadata
    assert proposal.metadata["phase"] == "pregrasp"


def test_phase_teacher_snapshot_restores_plan_and_pending_action():
    teacher = PerFrameArrowTeacher(lambda _frame: {"waypoints": [[0.0, 0.0, 0.5]] * 6})
    frame = _frame()
    proposal = teacher.propose(frame)
    state = teacher.snapshot_state()
    teacher.reset()
    teacher.restore_state(state)
    assert teacher.propose(frame) == proposal


def test_canonical_rotvec_is_converted_to_quaternion_for_osc():
    teacher = PerFrameArrowTeacher(lambda _frame: {"waypoints": [[0.01, 0.0, 0.5]] * 6})
    proposal = teacher.propose(_frame(rotvec=(0.0, 0.0, 1.5707963267948966)))
    assert proposal is not None
    assert all(abs(float(value)) <= 1.0 for value in proposal.action)


def test_teacher_close_destroy_are_idempotent_and_block_new_proposals():
    cleanup_calls = []
    teacher = PerFrameArrowTeacher(
        lambda _frame: {"waypoints": [[0.01, 0.0, 0.5]] * 6},
        cleanup_attempt=lambda: cleanup_calls.append("cleanup"),
    )
    teacher.close()
    teacher.destroy()
    teacher.close()
    assert cleanup_calls == ["cleanup"]
    with pytest.raises(ContractError, match="closed"):
        teacher.propose(_frame())
