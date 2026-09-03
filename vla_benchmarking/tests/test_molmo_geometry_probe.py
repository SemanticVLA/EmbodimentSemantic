from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from molmo_sam3.grasp_candidates import (
    CameraCalibration,
    CandidatePolicy,
    CandidateRejection,
    GraspCandidateResult,
    RobotGraspCalibration,
    generate_grasp_candidates,
)
from run_molmo_geometry_probe import run_geometry_passes


@dataclass
class _Candidate:
    candidate_id: str
    clearance_m: float


def _seam(policy: CandidatePolicy) -> dict:
    return {
        "rgb": np.zeros((3, 4, 3), dtype=np.uint8),
        "metric_depth_m": np.ones((3, 4), dtype=np.float32),
        "sam_mask": np.ones((3, 4), dtype=bool),
        "calibration": CameraCalibration(
            width=4, height=3,
            intrinsic=((2.0, 0.0, 1.5), (0.0, 2.0, 1.0), (0.0, 0.0, 1.0)),
            world_from_camera=np.eye(4).tolist(), camera_name="agentview",
        ),
        "robot_calibration": object(),
        "molmo_points": [(1.0, 1.0, 0.9)],
        "policy": policy,
    }


def test_probe_runs_two_geometry_passes_on_identical_inputs_and_records_details(tmp_path):
    policy = CandidatePolicy(name="molmo_dense", obstruction_clearance_m=None)
    seam = _seam(policy)
    seen = []

    def fake_generate(**kwargs):
        seen.append(kwargs)
        margin = kwargs["policy"].obstruction_clearance_m
        return GraspCandidateResult(
            candidates=(_Candidate(f"c{len(seen)}", float(margin)),),
            rejected=(CandidateRejection(0, 0.0, 0.0, "approach_obstruction", {"gap_m": margin}),),
            seeds_uv=((1, 1),), policy=kwargs["policy"].name,
            audit={"full_sample_minimum": True},
        )

    record = run_geometry_passes(seam, output_dir=tmp_path, label="t4", generate_fn=fake_generate)
    assert len(seen) == 2
    assert seen[0]["rgb"] is seen[1]["rgb"]
    assert seen[0]["metric_depth_m"] is seen[1]["metric_depth_m"]
    assert seen[0]["sam_mask"] is seen[1]["sam_mask"]
    assert seen[0]["policy"].obstruction_clearance_m == 0.006
    assert seen[0]["policy"].robot_exclusion_clearance_m == 0.006
    assert seen[1]["policy"].obstruction_clearance_m == 0.005
    assert seen[1]["policy"].robot_exclusion_clearance_m == 0.006
    assert record["passes"]["baseline_6mm_robot_6mm"]["admitted_candidate_ids"] == ["c1"]
    assert record["passes"]["admission_5mm_robot_6mm"]["actual_clearances_m"] == [0.005]
    assert record["passes"]["baseline_6mm_robot_6mm"]["rejections"][0]["details"]["gap_m"] == 0.006
    assert (tmp_path / "t4__inputs.npz").is_file()
    assert (tmp_path / "t4__geometry.json").is_file()


def test_robot_exclusion_margin_is_independent_but_none_retains_legacy_fallback():
    assert CandidatePolicy(obstruction_clearance_m=None).robot_exclusion_clearance_m is None
    assert CandidatePolicy(obstruction_clearance_m=0.006, robot_exclusion_clearance_m=0.006).robot_exclusion_clearance_m == 0.006


def test_legacy_none_robot_margin_matches_explicit_six_mm():
    height = width = 8
    depth = np.ones((height, width), dtype=np.float64)
    mask = np.ones((height, width), dtype=bool)
    calibration = CameraCalibration(
        width=width, height=height,
        intrinsic=((4.0, 0.0, 4.0), (0.0, 4.0, 4.0), (0.0, 0.0, 1.0)),
        world_from_camera=np.eye(4).tolist(),
    )
    robot = RobotGraspCalibration(
        current_grip_site_world_m=(0.0, 0.0, 1.0),
        current_rotation_world_grip_site=np.eye(3),
        hand_collision_spheres_grasp=(((0.0, 0.0, 0.0), 0.01),),
    )
    common = dict(rgb=np.zeros((height, width, 3), dtype=np.uint8), metric_depth_m=depth,
                  sam_mask=mask, molmo_points=[(4.0, 4.0)], calibration=calibration,
                  robot_calibration=robot)
    legacy = generate_grasp_candidates(**common, policy=CandidatePolicy(
        name="molmo_local", obstruction_clearance_m=0.006))
    explicit = generate_grasp_candidates(**common, policy=CandidatePolicy(
        name="molmo_local", obstruction_clearance_m=0.006,
        robot_exclusion_clearance_m=0.006))
    assert [item.candidate_id for item in legacy.candidates] == [item.candidate_id for item in explicit.candidates]
    assert [(item.reason, item.seed_index, item.yaw_deg, item.insertion_depth_m) for item in legacy.rejected] == [
        (item.reason, item.seed_index, item.yaw_deg, item.insertion_depth_m) for item in explicit.rejected
    ]
