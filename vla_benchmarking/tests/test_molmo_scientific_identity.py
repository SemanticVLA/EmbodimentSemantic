"""Focused tests for versioned scientific identity and provenance filtering."""

from __future__ import annotations

from copy import deepcopy

from vla_benchmarking.arrow_grasp_controller.controller import runner


def _identity_kwargs():
    return {
        "execution_sha": "a" * 40,
        "controller_config_digest": "b" * 64,
        "model_provenance": {
            "molmopoint_model": "model",
            "molmopoint_revision": "e" * 40,
            "molmopoint_prompt_id": "rim_clearance",
            "molmopoint_prompt": "prompt text",
            "transformers_version": "4.0",
            "dtype": "bfloat16",
            "max_points": 64,
            "cache_path": "transient/path",
            "load_seconds": 17.0,
            "models_loaded": True,
        },
        "variant": "canonical",
        "candidate_policy": "molmo_dense",
        "camera_name": "agentview",
        "region_backend": "rgbd",
        "backend": "rgbd_region",
        "motion_profile": "release20_retreat80mm",
        "motion_profile_params": {"release_height_offset_m": 0.02},
        "motion_diagnostics": False,
        "observation_profile": "parked",
        "observation_profile_params": {"skip_hover": True},
        "task_ids": tuple(range(10)),
        "seed_base": 1000,
        "suite_modes": ("sealed_randomized",),
    }


def test_operational_run_fields_do_not_enter_scientific_identity():
    first_payload, first_hash = runner.build_scientific_identity(**_identity_kwargs())
    second_payload, second_hash = runner.build_scientific_identity(**_identity_kwargs())
    assert first_payload == second_payload
    assert first_hash == second_hash
    assert "cache_path" not in first_payload["model"]
    assert "load_seconds" not in first_payload["model"]
    assert "models_loaded" not in first_payload["model"]


def test_stable_model_configuration_and_scientific_factors_change_identity():
    baseline = _identity_kwargs()
    _, baseline_hash = runner.build_scientific_identity(**baseline)

    transient_only = deepcopy(baseline)
    transient_only["model_provenance"]["cache_path"] = "another/path"
    transient_only["model_provenance"]["load_seconds"] = 99.0
    _, transient_hash = runner.build_scientific_identity(**transient_only)
    assert transient_hash == baseline_hash

    for key, value in (
        ("execution_sha", "f" * 40),
        ("controller_config_digest", "1" * 64),
        ("camera_name", "other_camera"),
        ("region_backend", "other_backend"),
        ("motion_profile", "other_motion"),
        ("observation_profile", "other_observation"),
        ("seed_base", 2000),
        ("suite_modes", ("other_suite",)),
    ):
        changed = deepcopy(baseline)
        changed[key] = value
        _, changed_hash = runner.build_scientific_identity(**changed)
        assert changed_hash != baseline_hash, key

    changed_model = deepcopy(baseline)
    changed_model["model_provenance"]["max_points"] = 128
    _, changed_model_hash = runner.build_scientific_identity(**changed_model)
    assert changed_model_hash != baseline_hash
