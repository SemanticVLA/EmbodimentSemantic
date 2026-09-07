"""Focused tests for exact-artifact and split preflight gates."""

import hashlib
import json

from .config import (
    ExperimentConfig,
    ProvenanceConfig,
    SplitConfig,
    validate_artifact_manifest,
    validate_vla_checkpoints,
)


def test_experiment_config_rejects_duplicate_task_ids():
    config = ExperimentConfig(task_ids=[0, 0], fidelity_mode="algorithmic_port")
    errors = config.validate()
    assert "task_ids contains duplicates" in errors


def test_empty_reference_manifest_is_blocked(tmp_path):
    manifest = tmp_path / "empty.json"
    manifest.write_text("{}\n", encoding="utf-8")
    errors = validate_artifact_manifest(
        {},
        ProvenanceConfig(reference_artifact_manifest=str(manifest)),
        "exact",
    )
    assert errors
    assert any("Exact RoboTTT training is blocked" in error for error in errors)


def test_paper_setting_cannot_override_official_manifest(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"official checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    fields = {
        "fast_mlp_hidden_dim": 128,
        "qkv_dimensions": 64,
        "qkv_normalization": "layernorm",
        "inner_learning_rate_parameterization": "softplus",
        "tbptt_segment_length": 8,
        "action_horizon": 16,
        "denoising_steps": 4,
        "token_packing": "per_timestep",
        "optimizer_betas_eps_clip": {"betas": [0.9, 0.95], "eps": 1e-8, "clip": 1.0},
        "image_crop_stride_preprocessing": {"crop": 224, "stride": 16},
    }
    manifest = tmp_path / "reference.json"
    manifest.write_text(
        json.dumps(
            {
                "checkpoint_uri": str(checkpoint),
                "checkpoint_sha256": digest,
                "code_commit": "official-commit",
                "fields": fields,
            }
        ),
        encoding="utf-8",
    )
    paper_settings = {
        "action_horizon": 99,
        "fast_mlp_width": None,
        "ttt_projection_dim": None,
        "qkv_normalization": None,
        "learned_step_size_parameterization": None,
        "tbptt_segment_length": None,
        "denoising_steps": None,
        "token_packing": None,
        "optimizer_betas_eps_clip": None,
        "image_crop_stride_preprocessing": None,
        "checkpoint": None,
        "checkpoint_sha256": None,
        "official_code_commit": None,
    }
    errors = validate_artifact_manifest(
        paper_settings,
        ProvenanceConfig(reference_artifact_manifest=str(manifest)),
        "exact",
    )
    assert any("paper_settings.action_horizon" in error and "conflicts" in error for error in errors)


def test_split_rejects_unknown_task_and_duplicate_seed():
    split = SplitConfig(eval_episode_ids=["task07_seed4_eval00", "task07_seed4_eval01"])
    errors = split.validate(task_ids=[0], episodes_per_task=2)
    assert any("parses task 7" in error for error in errors)
    assert any("repeats task/seed pair" in error for error in errors)


def test_eval_requires_exact_count_for_each_configured_task():
    split = SplitConfig(
        eval_episode_ids=["task00_seed4_eval00", "task00_seed5_eval01"],
    )
    errors = split.validate(task_ids=[0, 1], episodes_per_task=2)
    assert not any("eval task 0 has" in error for error in errors)
    assert any("eval task 1 has 0 episode IDs; expected exactly 2" in error for error in errors)


def test_local_vla_checkpoint_must_exist():
    errors = validate_vla_checkpoints(
        ProvenanceConfig(vla_checkpoints={"openvla": "missing-checkpoint.bin"}),
        "exact",
        ["openvla"],
    )
    assert any("does not exist" in error for error in errors)
