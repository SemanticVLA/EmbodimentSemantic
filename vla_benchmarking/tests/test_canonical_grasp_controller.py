"""Focused contracts for the promoted canonical grasp policy."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_benchmarking.arrow_grasp_controller.configs import (
    ACTIVE_CONTROLLER_CONFIG_FILENAME,
    ACTIVE_CONTROLLER_NAME,
    ControllerConfigError,
    load_controller_config,
)
from vla_benchmarking.arrow_grasp_controller.controller.entrypoint import build_engine_argv
from vla_benchmarking.arrow_grasp_controller.controller.policy import canonical_candidate_policy, canonical_settings
from vla_benchmarking.arrow_grasp_controller.controller.molmopoint import MolmoPointRuntime, MolmoPointRuntimeConfig
from vla_benchmarking.arrow_grasp_controller.controller.runner import (
    _candidate_motion_kwargs,
    preflight_local_molmo_runtime,
    resolve_motion_profile,
    resolve_opening_profile,
)


def test_canonical_policy_and_lock_are_frozen():
    config = load_controller_config()
    assert ACTIVE_CONTROLLER_CONFIG_FILENAME == "canonical_molmo_rgbd_grasp.json"
    assert config["name"] == ACTIVE_CONTROLLER_NAME
    metadata = config["policy_metadata"]
    assert metadata["result"] == "87/100"
    assert metadata["sam3_used"] is False
    assert metadata["molmopoint_prompt_id"] == "rim_clearance"
    assert metadata["max_grasp_attempts"] == 4
    assert metadata["shared_action_budget"] == 1200
    assert metadata == canonical_settings()
    lock = json.loads((Path(__file__).parents[1] / "arrow_grasp_controller" / "configs" / "active_policy.lock.json").read_text())
    assert lock["immutable"] is True
    assert lock["per_task_successes"] == {str(i): value for i, value in enumerate((10, 10, 10, 10, 7, 10, 8, 10, 10, 2))}
    assert lock["canonical_config_sha256"] == config["config_hash"]


def test_canonical_lock_matches_authoritative_result_tracker():
    root = Path(__file__).parents[1]
    lock = json.loads(
        (root / "arrow_grasp_controller" / "configs" / "active_policy.lock.json").read_text()
    )
    tracker = json.loads((root / "evaluation_results_tracker.json").read_text())
    result = tracker["canonical_grasp_controller"]

    assert result["status"] == "FINAL"
    assert result["job_id"] == lock["source_job_id"]
    assert result["source_release_commit"] == lock["source_release_commit"]
    assert result["canonical_config_sha256"] == lock["canonical_config_sha256"]
    assert result["scientific_identity_hash"] == lock["scientific_identity_hash"]
    assert result["archive_root"] == lock["archive"]
    assert result["per_task_successes"] == lock["per_task_successes"]
    assert result["successes"] == sum(result["per_task_successes"].values()) == 87
    assert result["planned"] == result["terminal_cells"] == 100


def test_historical_controller_config_is_not_executable():
    with pytest.raises(ControllerConfigError):
        load_controller_config("obsolete_controller.json")


def test_candidate_defaults_match_promoted_treatment():
    policy = canonical_candidate_policy()
    assert policy.name == "molmo_dense"
    assert policy.max_seeds == 16 and policy.max_candidates == 128
    assert policy.molmo_snap_radius_px == 12
    assert policy.local_seed_radius_px == 28
    assert policy.rim_support_radius_px == 24
    assert policy.yaw_offsets_deg == (-15.0, 0.0, 15.0)
    assert policy.insertion_depths_m == (0.0, 0.004, 0.008)


def test_public_entrypoint_exposes_scope_only():
    args = build_engine_argv(output_dir=Path("/tmp/canonical"))
    assert args[0:2] == ["--variant", "canonical"]
    assert "--sealed-100" in args
    assert "--region-backend" in args and args[args.index("--region-backend") + 1] == "rgbd"
    assert "--molmopoint-prompt-id" in args and args[args.index("--molmopoint-prompt-id") + 1] == "rim_clearance"
    assert not any(value.startswith("--sam3") for value in args)


def test_noncanonical_model_runtime_is_rejected():
    with pytest.raises(TypeError, match="project MolmoPointRuntime"):
        preflight_local_molmo_runtime(SimpleNamespace(config=MolmoPointRuntimeConfig()))
    altered = MolmoPointRuntime(MolmoPointRuntimeConfig(device="cpu", device_map=None))
    with pytest.raises(ValueError, match="not canonical"):
        preflight_local_molmo_runtime(altered)


def test_promoted_mechanical_treatment_is_exact():
    opening = resolve_opening_profile(
        "preshape40mm", region_backend="rgbd", camera_name="agentview"
    )
    assert opening["target_opening_m"] == 0.040
    assert opening["accepted_opening_band_m"] == (0.035, 0.045)
    assert opening["max_actions"] == 160
    motion = resolve_motion_profile("release20_retreat80mm", region_backend="rgbd")
    assert motion["release_height_offset_m"] == 0.020
    assert motion["retreat_height_offset_m"] == 0.080
    assert motion["retreat_tolerance_m"] == 0.005
    kwargs = _candidate_motion_kwargs(motion)
    assert kwargs["experimental_release_height_offset_m"] == 0.020
    assert kwargs["experimental_retreat_height_offset_m"] == 0.080
    assert "experimental_transfer_xy_policy" not in kwargs


def test_only_one_operator_launcher_remains():
    legion = Path(__file__).parents[1] / "arrow_grasp_controller" / "legion"
    assert {path.name for path in legion.iterdir()} == {
        "grasp_controller_requirements.txt",
        "run_grasp_controller.sbatch",
    }


def test_launcher_verifies_results_and_serializes_runtime_setup():
    launcher = (
        Path(__file__).parents[1] / "arrow_grasp_controller" / "legion" / "run_grasp_controller.sbatch"
    ).read_text(encoding="utf-8")
    assert 'workload_rc" -eq 0' in launcher
    assert "canonical_grasp_summary.json" in launcher
    assert "canonical_grasp_failed.json" in launcher
    assert "status=PRESERVED_FAILURE" in launcher
    assert "flock 9" in launcher and "flock -u 9" in launcher
    assert 'venv-py312-${REQUIREMENTS_SHA256:0:16}' in launcher
    assert "sys.path.insert(0, str(root))" in launcher
