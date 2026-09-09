from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arrow_policy_suite.cli import _load_config, main
from arrow_policy_suite.config import StudyConfig
from arrow_policy_suite.splits import ResetIdentity, build_split_manifest, write_split_manifest


def _config(path: Path) -> Path:
    payload = {
        "task_ids": [0],
        "test_reset_ids": {"0": [f"test-{index}" for index in range(10)]},
        "validation_reset_ids": {"0": [f"validation-{index}" for index in range(10)]},
        "trace_geometry_provider_revision": "arrow-rgbd-test-v1",
        "model_revision": "model-test-v1",
        "controller_revision": "controller-test-v1",
        "calibration_revision": "calibration-test-v1",
        "environment_revision": "environment-test-v1",
        "processor_revision": "processor-test-v1",
        "action_encoding": "normalized_osc7_v1",
        "n_action_steps": 1,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "image_resolution": [256, 256],
        "depth_units": "meters",
        "renderer_revision": "renderer-test-v1",
        "metric_revision": "success_at_1200_v1",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_handoff_commands_are_create_only_and_do_not_import_factories(capsys):
    commands = (
        "canary", "derive", "train-apprentice", "train-residuals",
        "train-fast-slow", "prepare-fast-support", "evaluate", "audit",
    )
    for command in commands:
        assert main([command, "--factory", "missing.module:executor", "--dry-run"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "DRY_RUN"
        assert payload["operation"] == command
        assert payload["runs_launched"] is False
        assert payload["execution"] == "not_launched"
        assert payload["factory"] == "missing.module:executor"


def test_handoff_with_valid_config_is_create_only_and_invalid_config_blocks(tmp_path, capsys):
    config = _config(tmp_path / "study.json")
    assert main(["evaluate", "--config", str(config)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "CREATE_ONLY"
    assert payload["config"]["status"] == "READY"
    assert payload["runs_launched"] is False

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"task_ids": [0]}), encoding="utf-8")
    assert main(["audit", "--config", str(bad)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "BLOCKED"
    assert payload["runs_launched"] is False


def _split(path: Path, split: str, start: int, count: int) -> Path:
    identities = tuple(
        ResetIdentity(
            task_id=0,
            episode_id=f"{split}-{index}",
            seed=start + index,
            reset_index=1,
            query_index=start + index,
            observation_sha256="a" * 64,
            simulator_state_sha256=f"{start + index:064x}",
            environment_fingerprint="environment-test-v1",
        )
        for index in range(count)
    )
    write_split_manifest(path, build_split_manifest(split, identities))
    return path


def test_preflight_split_seal_dry_run_does_not_write_and_create_only_refuses_overwrite(tmp_path, capsys):
    config = _config(tmp_path / "study.json")
    collection = _split(tmp_path / "collection.json", "collection", 0, 50)
    validation = _split(tmp_path / "validation.json", "validation", 100, 10)
    test = _split(tmp_path / "test.json", "test", 200, 10)
    training = tmp_path / "training.json"
    training.write_text("training-manifest", encoding="utf-8")
    output = tmp_path / "frozen-study.json"
    command = [
        "preflight", str(config), "--output", str(output), "--dry-run",
        "--collection-split", str(collection), "--validation-split", str(validation),
        "--test-split", str(test), "--retained-training-manifest", str(training),
    ]
    assert main(command) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "READY"
    assert receipt["mode"] == "DRY_RUN"
    assert receipt["written"] is False and receipt["runs_launched"] is False
    assert len(receipt["manifest_digest"]) == 64
    assert not output.exists()

    command[command.index("--dry-run")] = "--unused"  # avoid mutating the tested argument shape below
    command.remove("--unused")
    assert main(command) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["mode"] == "CREATE_ONLY" and receipt["written"] is True
    assert output.exists()
    assert main(command) == 2
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "BLOCKED" and receipt["written"] is False


def test_loader_round_trips_signed_config_and_frozen_study_for_preflight_and_canary(tmp_path, capsys):
    config_path = _config(tmp_path / "study.json")
    loaded = _load_config(config_path)
    config_manifest_path = tmp_path / "study-manifest.json"
    config_manifest_path.write_text(json.dumps(loaded.manifest()), encoding="utf-8")
    restored = _load_config(config_manifest_path)
    assert restored.config_sha256() == loaded.config_sha256()

    collection = _split(tmp_path / "collection.json", "collection", 0, 50)
    validation = _split(tmp_path / "validation.json", "validation", 100, 10)
    test = _split(tmp_path / "test.json", "test", 200, 10)
    training = tmp_path / "training.json"
    training.write_text("training-manifest", encoding="utf-8")
    frozen_path = tmp_path / "frozen.json"
    assert main([
        "preflight", str(config_path), "--output", str(frozen_path),
        "--collection-split", str(collection), "--validation-split", str(validation),
        "--test-split", str(test), "--retained-training-manifest", str(training),
    ]) == 0
    capsys.readouterr()

    assert main(["preflight", str(frozen_path)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["manifest_digest"] == receipt["frozen_study"]["composite_sha256"]
    assert receipt["frozen_study"]["collection"]["split"] == "collection"

    assert main(["canary", "--config", str(frozen_path)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["config"]["manifest_digest"] == receipt["config"]["frozen_study"]["composite_sha256"]
