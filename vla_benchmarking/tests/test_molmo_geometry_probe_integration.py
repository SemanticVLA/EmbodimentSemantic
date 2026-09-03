"""Focused CLI-seam checks for the perception-only geometry probe."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from molmo_sam3.grasp_candidates import CameraCalibration, CandidatePolicy
import run_molmo_geometry_probe as probe


def _seam() -> dict:
    return {
        "rgb": np.zeros((3, 4, 3), dtype=np.uint8),
        "metric_depth_m": np.ones((3, 4), dtype=np.float32),
        "sam_mask": np.ones((3, 4), dtype=bool),
        "calibration": CameraCalibration(
            width=4, height=3,
            intrinsic=((2.0, 0.0, 1.5), (0.0, 2.0, 1.0), (0.0, 0.0, 1.0)),
            world_from_camera=np.eye(4).tolist(), camera_name="agentview",
        ),
        "robot_calibration": SimpleNamespace(),
        "molmo_points": [(1.0, 1.0, 0.9)],
        "policy": CandidatePolicy(name="molmo_dense", obstruction_clearance_m=0.006),
    }


def test_main_uses_fixed_single_cell_rim_clearance_and_blocks_motion(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_passes(seam_kwargs, **kwargs):
        seen["seam"] = seam_kwargs
        seen["output"] = kwargs
        return {"passes": {}}

    monkeypatch.setattr(probe, "run_geometry_passes", fake_passes)
    import run_arrow_pick_place_matrix as matrix

    def fake_matrix(**kwargs):
        seen["matrix_kwargs"] = kwargs
        assert callable(kwargs["evaluator"])
        return {"status": "synthetic"}

    monkeypatch.setattr(matrix, "run_matrix", fake_matrix)

    def fake_canary_main(argv, *, cell_completed_callback):
        seen["argv"] = list(argv)
        # main() must replace the live episode runner with a hard motion guard.
        import run_arrow_pick_place_eval as episode
        assert episode.run_episode.__name__ == "guard"
        matrix.run_matrix(output_root=tmp_path / "matrix")
        result = probe.geometry.generate_grasp_candidates(**_seam())
        assert tuple(result.candidates) == ()
        with pytest.raises(probe._StopAfterProbe):
            cell_completed_callback({"status": "completed", "task_id": 4, "seed": 1000, "suite_mode": "vanilla"})
        return 0

    monkeypatch.setattr(probe.canary, "main", fake_canary_main)
    assert probe.main(["--output-dir", str(tmp_path), "--label", "t4_probe"]) == 0
    accepted_args = dict(zip(seen["argv"][::2], seen["argv"][1::2]))
    assert accepted_args["--task-ids"] == "4,6,9"
    assert accepted_args["--suite-modes"] == "vanilla,sealed_randomized"
    output = __import__("json").loads(capsys.readouterr().out)
    args = dict(zip(output["probe"]["requested_cli"][::2], output["probe"]["requested_cli"][1::2]))
    assert args == {
        "--variant": "molmo_dense_agentview",
        "--output-dir": str(tmp_path),
        "--label": "t4_probe",
        "--region-backend": "rgbd",
        "--observation-profile": "hover20mm",
        "--episodes-per-task": "1",
        "--task-ids": "4",
        "--suite-modes": "vanilla",
        "--seed-base": "1000",
        "--molmopoint-prompt-id": "rim_clearance",
    }
    assert set(seen["seam"]) == {
        "rgb", "metric_depth_m", "sam_mask", "calibration",
        "robot_calibration", "molmo_points", "policy",
    }
    assert len(seen["output"]["execution_sha"]) == 40
    assert seen["matrix_kwargs"]["experiment_metadata"]["diagnostic_kind"] == "perception_geometry_probe"



def test_nonzero_canary_status_is_not_promoted_to_success(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "run_geometry_passes", lambda *_args, **_kwargs: {"passes": {}})

    def failed_canary(argv, *, cell_completed_callback):
        del argv, cell_completed_callback
        probe.geometry.generate_grasp_candidates(**_seam())
        return 2

    monkeypatch.setattr(probe.canary, "main", failed_canary)
    # A return-code 2 without the controlled durable-cell callback must remain
    # an operation error (CLI exit is therefore nonzero), never success.
    with pytest.raises(RuntimeError, match="without a geometry seam capture"):
        probe.main(["--output-dir", str(tmp_path)])


def test_failed_first_cell_is_not_reported_as_a_successful_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "run_geometry_passes", lambda *_args, **_kwargs: {"passes": {}})

    def failed_cell_canary(argv, *, cell_completed_callback):
        del argv
        probe.geometry.generate_grasp_candidates(**_seam())
        with pytest.raises(probe._StopAfterProbe):
            cell_completed_callback({"status": "failed", "task_id": 4, "seed": 1000, "suite_mode": "vanilla"})
        return 0

    monkeypatch.setattr(probe.canary, "main", failed_cell_canary)
    with pytest.raises(RuntimeError, match="callback"):
        probe.main(["--output-dir", str(tmp_path)])


def test_second_perception_seam_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "run_geometry_passes", lambda *_args, **_kwargs: {"passes": {}})

    def duplicate_canary(argv, *, cell_completed_callback):
        del argv, cell_completed_callback
        probe.geometry.generate_grasp_candidates(**_seam())
        probe.geometry.generate_grasp_candidates(**_seam())
        return 0

    monkeypatch.setattr(probe.canary, "main", duplicate_canary)
    with pytest.raises(RuntimeError, match="more than one perception seam"):
        probe.main(["--output-dir", str(tmp_path)])
