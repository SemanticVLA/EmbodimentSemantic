from __future__ import annotations

import json
from pathlib import Path
import sys
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arrow_policy_suite.cli import _load_config, main
import arrow_policy_suite.cli as cli
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


def test_learned_handoff_requires_immutable_sidecar_and_preserves_hashes(tmp_path, capsys):
    checkpoint = tmp_path / "editor.pt"
    sidecar = tmp_path / "editor.pt.json"
    checkpoint.write_bytes(b"checkpoint")
    sidecar.write_text('{"schema":"arrow_policy_suite.native_residual.v1"}\n', encoding="utf-8")

    assert main([
        "canary", "--policy", "arrow_editor", "--input", str(checkpoint),
    ]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["runtime_privileges"] == {"teacher_required": False, "teacher_free": True}
    assert receipt["inputs"] == [str(checkpoint.resolve())]
    learned = receipt["learned_artifacts"][0]
    assert learned["path"] == str(checkpoint.resolve())
    assert learned["sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert learned["sidecar"]["path"] == str(sidecar.resolve())
    assert learned["sidecar"]["sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()

    checkpoint.unlink()
    assert main([
        "canary", "--policy", "arrow_editor", "--input", str(checkpoint),
    ]) == 2
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "BLOCKED"


def test_apprentice_handoff_accepts_one_immutable_bundle_and_records_manifest(tmp_path, capsys):
    bundle = tmp_path / "apprentice_bundle"
    bundle.mkdir()
    payload = bundle / "adapter_model.safetensors"
    payload.write_bytes(b"weights")
    inventory = {payload.name: hashlib.sha256(payload.read_bytes()).hexdigest()}
    inventory_hash = hashlib.sha256(
        (json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    manifest = bundle / "apprentice_manifest.json"
    manifest.write_text(json.dumps({
        "checkpoint_inventory": inventory,
        "checkpoint_sha256": inventory_hash,
        "adapter_only": True,
    }, sort_keys=True) + "\n", encoding="utf-8")

    assert main(["canary", "--policy", "arrow_apprentice", "--input", str(bundle)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    learned = receipt["learned_artifacts"][0]
    assert learned["path"] == str(bundle.resolve())
    assert learned["bundle_manifest"]["path"] == str(manifest.resolve())
    assert learned["bundle_manifest"]["inventory_sha256"] == inventory_hash
    assert learned["bundle_manifest"]["payload_files"] == 1

    second = tmp_path / "second_bundle"
    second.mkdir()
    (second / "apprentice_manifest.json").write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    (second / "payload").write_bytes(b"other")
    assert main([
        "canary", "--policy", "arrow_apprentice", "--input", str(bundle),
        "--input", str(second),
    ]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"


def test_learned_and_runtime_minimal_privileges_are_distinct(tmp_path, capsys):
    checkpoint = tmp_path / "minimal.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "minimal.pt.json").write_text("{}\n", encoding="utf-8")

    assert main(["canary", "--policy", "arrow_minimal_runtime"]) == 0
    runtime = json.loads(capsys.readouterr().out)
    assert runtime["runtime_privileges"]["teacher_required"] is True
    assert runtime["runtime_privileges"]["teacher_free"] is False

    assert main(["canary", "--policy", "arrow_minimal_learned", "--input", str(checkpoint)]) == 0
    learned = json.loads(capsys.readouterr().out)
    assert learned["runtime_privileges"]["teacher_required"] is False
    assert learned["runtime_privileges"]["teacher_free"] is True


def test_cli_receipt_preserves_explicit_launcher_reset_identity(monkeypatch, capsys):
    monkeypatch.setenv("ARROW_SUITE_TASK_ID", "3")
    monkeypatch.setenv("ARROW_SUITE_SEED", "17")
    monkeypatch.setenv("ARROW_SUITE_INIT_STATE_INDEX", "2")
    assert main(["canary"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["launch_identity"] == {"task_id": 3, "seed": 17, "init_state_index": 2}

    monkeypatch.delenv("ARROW_SUITE_TASK_ID")
    monkeypatch.delenv("ARROW_SUITE_SEED")
    monkeypatch.delenv("ARROW_SUITE_INIT_STATE_INDEX")
    assert main(["canary"]) == 0
    assert json.loads(capsys.readouterr().out)["launch_identity"] == {}


def test_collect_accepts_launcher_contract_and_dispatches_only_when_execute(tmp_path, monkeypatch, capsys):
    config = _config(tmp_path / "study.json")
    assert main(["collect", "--config", str(config)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "READY_TO_COLLECT"

    called = {}
    def fake_execute(args):
        called["args"] = args
        return 17
    monkeypatch.setattr(cli, "_cmd_execute", fake_execute)
    assert main([
        "collect", "--config", str(config), "--factory", "example.factory:build",
        "--execute", "--policy", "arrow_on_call", "--run-dir", str(tmp_path / "run"),
        "--output", str(tmp_path / "receipt.json"), "--steps", "3",
    ]) == 17
    assert called["args"].config == str(config)
    assert called["args"].policy == "arrow_on_call"


def test_execute_collect_and_evaluate_require_explicit_predeclared_horizons(tmp_path, capsys):
    config = _config(tmp_path / "study.json")
    common = [
        "--config", str(config), "--factory", "example.factory:build",
        "--execute", "--policy", "arrow_on_call", "--run-dir", str(tmp_path / "run"),
    ]
    assert main(["collect", *common]) == 2
    assert "explicit --steps" in json.loads(capsys.readouterr().out)["errors"][0]
    assert main(["evaluate", *common, "--steps", "3"]) == 2
    assert "280 or 1200" in json.loads(capsys.readouterr().out)["errors"][0]
