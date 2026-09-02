from __future__ import annotations

import json
from pathlib import Path

from sanity_checks import run_live_panda_frame_probe as live_probe


def test_run_probe_builds_settled_env_reads_frame_and_closes(monkeypatch, tmp_path: Path):
    controller = tmp_path / "controller.json"
    controller.write_text('{"name":"probe"}\n', encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"schema_version":"test"}\n', encoding="utf-8")

    class Env:
        _arrow_settle_diagnostics = {"settled": True, "steps": 500}
        closed = False

        def close(self):
            self.closed = True

    env = Env()
    variant = type("Variant", (), {"name": "v10-probe"})()
    monkeypatch.setattr(
        live_probe,
        "load_controller_config",
        lambda _path: {"name": "probe", "config_hash": "a" * 64},
    )
    monkeypatch.setattr(
        live_probe,
        "controller_variant_from_config",
        lambda _config, suite_mode: variant,
    )
    monkeypatch.setattr(live_probe, "build_libero_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(
        live_probe,
        "probe_grip_site_frame",
        lambda candidate, **_kwargs: {"passed": candidate is env},
    )

    result = live_probe.run_probe(
        task_id=0,
        seed=1000,
        resolution=256,
        suite_mode="vanilla",
        controller_config=controller,
        calibration_input=calibration,
    )

    assert result["passed"] is True
    assert result["execution"]["commanded_robot_motion"] is False
    assert result["execution"]["controller_inference"] is False
    assert result["execution"]["evaluator_queried"] is False
    assert result["settle_diagnostics"]["settled"] is True
    assert env.closed is True


def test_live_probe_job_is_exact_commit_bounded_and_archived():
    root = Path(__file__).resolve().parents[1]
    script = (root / "legion" / "run_zerograsp_live_frame_probe.sbatch").read_text(
        encoding="utf-8"
    )
    assert "ARROW_MATRIX_EXPECTED_COMMIT" in script
    assert "status --porcelain --untracked-files=all" in script
    assert "vanilla sealed_randomized" in script
    assert "commanded_robot_motion" in script
    assert "evaluator_queried" in script
    assert "probe_manifest.sha256" in script
    assert "trap archive_on_exit EXIT" in script
    assert "--gres=gpu:1" in script


def test_calibration_contract_has_explicit_frames_and_no_evaluator():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "calibrations" / "panda_graspnet_eef_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["candidate_translation_reference"] == "grasp_center"
    assert payload["candidate_translation_frame"] == "camera_graspnet"
    assert payload["candidate_rotation_frame"] == "camera_graspnet"
    assert payload["controller_target_frame"] == "grip_site"
    assert payload["validation"]["simulator_object_state_used_by_controller"] is False
    assert payload["validation"]["evaluator_used"] is False
