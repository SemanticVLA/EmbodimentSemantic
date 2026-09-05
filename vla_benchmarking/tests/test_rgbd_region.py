from dataclasses import dataclass

import numpy as np
import pytest

from vla_benchmarking.rgbd_region import derive_observed_region_mask, project_observed_region_to_wrist


@dataclass
class Calibration:
    camera_name: str
    intrinsic: tuple = ((400.0, 0.0, 32.0), (0.0, -400.0, 32.0), (0.0, 0.0, 1.0))
    world_from_camera: tuple = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


@dataclass
class Capture:
    camera_name: str

    def __post_init__(self):
        yy, xx = np.indices((64, 64))
        self.rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        self.metric_depth = np.full((64, 64), np.nan, dtype=np.float64)
        self.metric_depth[(xx - 32) ** 2 + (yy - 32) ** 2 <= 10 ** 2] = 1.0
        self.calibration = Calibration(self.camera_name)


def test_rgbd_region_is_arrow_seeded_and_audited_without_sam():
    capture = Capture("agentview")
    mask, audit = derive_observed_region_mask(capture, (32.0, 32.0))
    assert mask.dtype == np.bool_
    assert 200 < int(mask.sum()) < 400
    assert audit["method"] == "arrow_seeded_metric_depth_component_v1"
    assert audit["region_mask_sha256"]


def test_wrist_projection_requires_current_depth_identity():
    agentview, wrist = Capture("agentview"), Capture("robot0_eye_in_hand")
    region, _ = derive_observed_region_mask(agentview, (32.0, 32.0))
    projected, audit = project_observed_region_to_wrist(agentview, wrist, region)
    assert projected.dtype == np.bool_
    assert int(projected.sum()) >= 4
    assert audit["depth_agreement_px"] >= 4
    wrist.metric_depth[:] = np.nan
    with pytest.raises(ValueError, match="strictly match"):
        project_observed_region_to_wrist(agentview, wrist, region)


def test_wrist_projection_rejects_pixels_that_round_outside_frame():
    agentview, wrist = Capture("agentview"), Capture("robot0_eye_in_hand")
    region, _ = derive_observed_region_mask(agentview, (32.0, 32.0))
    wrist.calibration = Calibration(
        "robot0_eye_in_hand",
        intrinsic=((1.0, 0.0, 63.6), (0.0, -1.0, 32.0), (0.0, 0.0, 1.0)),
    )
    with pytest.raises(ValueError, match="rounds outside"):
        project_observed_region_to_wrist(agentview, wrist, region)
