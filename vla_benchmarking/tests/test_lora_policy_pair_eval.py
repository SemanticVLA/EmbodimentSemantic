from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import run_lora_policy_pair_eval as pair_eval
from lora_finetuning_policy import ACTION_ONLY_LORA_V1, ACTION_VISUAL_LORA_V1, get_policy


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    base = tmp_path / f"smolvla_libero-{pair_eval.SEALED_REVISION}"
    base.mkdir()
    sealed_file = base / "sealed_config.json"
    sealed_file.write_text("sealed base", encoding="utf-8")
    (base / "base_snapshot_manifest.json").write_text(
        json.dumps({
            "revision": pair_eval.SEALED_REVISION,
            "files": {"sealed_config.json": _sha(sealed_file)},
        }),
        encoding="utf-8",
    )
    cache = base / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "benign_cache_index.json").write_text("cache", encoding="utf-8")
    base_identity = pair_eval.validate_base_snapshot_identity(base)
    pair = tmp_path / "sealed_lora_pair_manifest.json"
    sentinel = tmp_path / "sealed_lora_pair_verified.json"
    pair.write_text("pair", encoding="utf-8")
    sentinel.write_text("sentinel", encoding="utf-8")
    data_root = tmp_path / "training_data"
    data_root.mkdir()
    (data_root / "sealed_lora_pair_manifest.json").write_bytes(pair.read_bytes())
    bundle = tmp_path / "legacy_action_only_evidence_bundle"
    bundle.mkdir()
    (bundle / "bundle.json").write_text(
        json.dumps({"pair": {"pair_identity": "p" * 64}}), encoding="utf-8"
    )

    def make(policy_id: str, name: str) -> tuple[Path, Path, dict]:
        adapter = tmp_path / name / "029190" / "pretrained_model"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes((name + " adapter").encode())
        inventory = tmp_path / f"{name}_expected_adapter_inventory.json"
        inventory.write_text("{}", encoding="utf-8")
        audit = adapter / "adapter_audit.json"
        audit.write_text(json.dumps({"checkpoint_tree_sha256": name}), encoding="utf-8")
        policy = get_policy(policy_id)
        manifest = {
            "experiment": pair_eval.TRAINING_EXPERIMENT,
            "base_policy": str(base),
            "base_policy_revision": pair_eval.SEALED_REVISION,
            "pair_manifest": str(pair),
            "pair_manifest_sha256": _sha(pair),
            "pair_sentinel": str(sentinel),
            "pair_sentinel_sha256": _sha(sentinel),
            "training_variant": "no_arrow_treatment",
            "dataset_variant": "control",
            "dataset_repo_id": "local/libero_spatial_control",
            "trained_on_visual_condition": "no_arrows",
            "finetuning_policy_id": policy_id,
            "finetuning_policy_target_regex": policy.target_regex,
            "flags": {"steps": 29190, "save_freq": 1946, "batch_size": 32, "seed": 1000, "peft_r": 16},
            "no_arrow_treatment_adapter": {"path": str(adapter), "sha256": _sha(adapter / "adapter_model.safetensors")},
            "expected_adapter_inventory": {"path": str(inventory), "sha256": _sha(inventory)},
            "adapter_audit": {"path": str(audit), "sha256": _sha(audit)},
        }
        return adapter, audit, manifest

    only_adapter, _only_audit, only = make(ACTION_ONLY_LORA_V1, "action_only")
    visual_adapter, _visual_audit, visual = make(ACTION_VISUAL_LORA_V1, "action_visual")
    expected = {
        ACTION_ONLY_LORA_V1: {"finetuning_policy_id": ACTION_ONLY_LORA_V1, "base_policy": str(base)},
        ACTION_VISUAL_LORA_V1: {"finetuning_policy_id": ACTION_VISUAL_LORA_V1, "base_policy": str(base)},
    }
    monkeypatch.setattr(pair_eval, "load_expected_inventory", lambda path: expected[ACTION_ONLY_LORA_V1] if Path(path).name.startswith("action_only_") else expected[ACTION_VISUAL_LORA_V1])
    monkeypatch.setattr(pair_eval, "audit_adapter_checkpoint", lambda *args, **kwargs: {"checkpoint_tree_sha256": "action_only" if Path(args[0]).parts[-3] == "action_only" else "action_visual"})
    monkeypatch.setattr(
        pair_eval,
        "validate_legacy_action_only_evidence",
        lambda *args, **kwargs: {
            "valid": True,
            "pair_identity": "p" * 64,
            "checkpoint_tree_sha256": "historical-checkpoint",
            "base_snapshot_identity_sha256": base_identity,
            "source_snapshots": [
                {"role": "base_policy", "sha256": base_identity},
                {"role": "data_root", "sha256": "data-root"},
            ],
        },
    )
    # The adapter_audit evidence in each fixture must match the monkeypatched
    # runtime audit exactly.
    only["adapter_audit"]["sha256"] = _sha(only_adapter / "adapter_audit.json")
    visual["adapter_audit"]["sha256"] = _sha(visual_adapter / "adapter_audit.json")
    only["_legacy_bundle"] = str(bundle)
    only["_data_root"] = str(data_root)
    return only, visual


def test_build_manifest_has_canonical_two_cell_order_and_clean_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    only, visual = _fixture(tmp_path, monkeypatch)
    only_path = tmp_path / "action_only_training_manifest.json"
    visual_path = tmp_path / "action_visual_training_manifest.json"
    only_path.write_text(json.dumps(only), encoding="utf-8")
    visual_path.write_text(json.dumps(visual), encoding="utf-8")
    manifest = pair_eval.build_manifest(
        action_only_checkpoint=only["no_arrow_treatment_adapter"]["path"],
        action_visual_checkpoint=visual["no_arrow_treatment_adapter"]["path"],
        action_only_training_manifest=str(only_path),
        action_visual_training_manifest=str(visual_path),
        action_only_legacy_evidence_bundle=only["_legacy_bundle"],
        training_data_root=only["_data_root"],
        output_root=tmp_path / "eval",
        device="plan",
    )
    assert [cell["cell_id"] for cell in manifest["cells"]] == list(pair_eval.CELL_IDS)
    assert [cell["policy_id"] for cell in manifest["cells"]] == list(pair_eval.POLICY_IDS)
    assert all(cell["live_arrows"] is False and cell["visual_condition"] == "none" for cell in manifest["cells"])
    assert manifest["tasks"] == list(range(10))
    assert manifest["episodes"] == 10
    assert manifest["batch_size"] == 1
    assert manifest["paired_reset_states"] is True
    contract = manifest["comparability_contract"]
    assert contract["intended_training_difference"] == "finetuning_policy_target_modules"
    assert contract["verified_shared_training_fields"] == [
        "base_policy_revision",
        "dataset_repo_id",
        "dataset_variant",
        "pair_manifest_sha256",
        "training_schedule",
        "training_seed",
    ]
    assert contract["verified_current_compatibility_fields"] == [
        "reconstruction_and_candidate_base_snapshot_identity",
    ]
    assert "historical_training_base_content_identity" in contract["unverified_or_runtime_confound_fields"]
    assert contract["unverified_or_runtime_confound_fields"]
    assert "only_intended_difference" not in contract
    assert manifest["evidence_classes"]["action_visual_candidate"] == "native_policy_evidence_v1"
    assert manifest["conclusion_limits"] == pair_eval.CONCLUSION_LIMITS
    base_identity_evidence = manifest["base_snapshot_identity_evidence"]
    assert base_identity_evidence["historical_training_base_content_identity"] == {
        "status": "unavailable_noncontemporaneous",
        "sha256": None,
    }
    assert base_identity_evidence["reconstruction_and_candidate_base_snapshot_identity"]["status"] == "current_verified_compatibility_evidence"
    assert len(base_identity_evidence["reconstruction_and_candidate_base_snapshot_identity"]["sha256"]) == 64
    assert "base_snapshot_identity_sha256" not in manifest["shared_training_provenance"]
    assert any("historical training base-content identity is unavailable" in item for item in manifest["known_noncontemporaneous_evidence"])


def test_manifest_rejects_legacy_only_intended_difference_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    only, visual = _fixture(tmp_path, monkeypatch)
    only_path = tmp_path / "action_only_training_manifest.json"
    visual_path = tmp_path / "action_visual_training_manifest.json"
    only_path.write_text(json.dumps(only), encoding="utf-8")
    visual_path.write_text(json.dumps(visual), encoding="utf-8")
    manifest = pair_eval.build_manifest(
        action_only_checkpoint=only["no_arrow_treatment_adapter"]["path"],
        action_visual_checkpoint=visual["no_arrow_treatment_adapter"]["path"],
        action_only_training_manifest=str(only_path),
        action_visual_training_manifest=str(visual_path),
        action_only_legacy_evidence_bundle=only["_legacy_bundle"],
        training_data_root=only["_data_root"],
        output_root=tmp_path / "eval",
        device="plan",
    )
    manifest["comparability_contract"]["only_intended_difference"] = ["policy"]
    with pytest.raises(ValueError, match="only_intended_difference"):
        pair_eval._validate_manifest(manifest)


def test_legacy_action_only_manifest_without_policy_fields_resolves_to_v1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    only, visual = _fixture(tmp_path, monkeypatch)
    only.pop("finetuning_policy_id")
    only.pop("finetuning_policy_target_regex")
    only_path = tmp_path / "legacy_action_only_training_manifest.json"
    visual_path = tmp_path / "action_visual_training_manifest.json"
    only_path.write_text(json.dumps(only), encoding="utf-8")
    visual_path.write_text(json.dumps(visual), encoding="utf-8")
    manifest = pair_eval.build_manifest(
        action_only_checkpoint=only["no_arrow_treatment_adapter"]["path"],
        action_visual_checkpoint=visual["no_arrow_treatment_adapter"]["path"],
        action_only_training_manifest=str(only_path),
        action_visual_training_manifest=str(visual_path),
        action_only_legacy_evidence_bundle=only["_legacy_bundle"],
        training_data_root=only["_data_root"],
        output_root=tmp_path / "eval",
        device="plan",
    )
    assert manifest["cells"][0]["policy_id"] == ACTION_ONLY_LORA_V1
    assert manifest["cells"][1]["policy_id"] == ACTION_VISUAL_LORA_V1


def test_build_manifest_rejects_training_data_or_schedule_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    only, visual = _fixture(tmp_path, monkeypatch)
    different_pair = tmp_path / "different_pair.json"
    different_pair.write_text("different pair", encoding="utf-8")
    visual["pair_manifest"] = str(different_pair)
    visual["pair_manifest_sha256"] = _sha(different_pair)
    only_path = tmp_path / "only.json"
    visual_path = tmp_path / "visual.json"
    only_path.write_text(json.dumps(only), encoding="utf-8")
    visual_path.write_text(json.dumps(visual), encoding="utf-8")
    with pytest.raises(ValueError, match="pair-manifest digests differ"):
        pair_eval.build_manifest(
            action_only_checkpoint=only["no_arrow_treatment_adapter"]["path"],
            action_visual_checkpoint=visual["no_arrow_treatment_adapter"]["path"],
            action_only_training_manifest=str(only_path),
            action_visual_training_manifest=str(visual_path),
            action_only_legacy_evidence_bundle=only["_legacy_bundle"],
            training_data_root=only["_data_root"],
            output_root=tmp_path / "eval",
            device="plan",
        )


def _build_manifest_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, Path, Path]:
    only, visual = _fixture(tmp_path, monkeypatch)
    only_path = tmp_path / "only.json"
    visual_path = tmp_path / "visual.json"
    only_path.write_text(json.dumps(only), encoding="utf-8")
    visual_path.write_text(json.dumps(visual), encoding="utf-8")
    return only, visual, only_path, visual_path


def test_missing_or_invalid_legacy_bundle_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    only, visual, only_path, visual_path = _build_manifest_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="legacy action-only evidence bundle is missing"):
        pair_eval.build_manifest(
            action_only_checkpoint=only["no_arrow_treatment_adapter"]["path"],
            action_visual_checkpoint=visual["no_arrow_treatment_adapter"]["path"],
            action_only_training_manifest=str(only_path), action_visual_training_manifest=str(visual_path),
            action_only_legacy_evidence_bundle=str(tmp_path / "missing_bundle"), training_data_root=only["_data_root"],
            output_root=tmp_path / "eval", device="plan",
        )
    (Path(only["_legacy_bundle"]) / "bundle.json").write_text(json.dumps({"pair": {"pair_identity": "q" * 64}}), encoding="utf-8")
    with pytest.raises(ValueError, match="stable pair identity"):
        pair_eval.build_manifest(
            action_only_checkpoint=only["no_arrow_treatment_adapter"]["path"],
            action_visual_checkpoint=visual["no_arrow_treatment_adapter"]["path"],
            action_only_training_manifest=str(only_path), action_visual_training_manifest=str(visual_path),
            action_only_legacy_evidence_bundle=only["_legacy_bundle"], training_data_root=only["_data_root"],
            output_root=tmp_path / "eval", device="plan",
        )


def test_candidate_cannot_use_legacy_policy_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    only, visual, only_path, visual_path = _build_manifest_fixture(tmp_path, monkeypatch)
    visual["finetuning_policy_id"] = ACTION_ONLY_LORA_V1
    visual_path.write_text(json.dumps(visual), encoding="utf-8")
    with pytest.raises(ValueError, match="training manifest policy"):
        pair_eval.build_manifest(
            action_only_checkpoint=only["no_arrow_treatment_adapter"]["path"],
            action_visual_checkpoint=visual["no_arrow_treatment_adapter"]["path"],
            action_only_training_manifest=str(only_path), action_visual_training_manifest=str(visual_path),
            action_only_legacy_evidence_bundle=only["_legacy_bundle"], training_data_root=only["_data_root"],
            output_root=tmp_path / "eval", device="plan",
        )


def test_stable_pair_identity_drift_fails_but_raw_sentinel_drift_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    only, visual, only_path, visual_path = _build_manifest_fixture(tmp_path, monkeypatch)
    # Keep the validated stable identity but use a distinct, valid candidate
    # sentinel byte stream. Raw sentinel equality is intentionally not a gate.
    candidate_sentinel = tmp_path / "candidate_pair_sentinel.json"
    candidate_sentinel.write_text("candidate sentinel", encoding="utf-8")
    visual["pair_sentinel"] = str(candidate_sentinel)
    visual["pair_sentinel_sha256"] = _sha(candidate_sentinel)
    visual_path.write_text(json.dumps(visual), encoding="utf-8")
    manifest = pair_eval.build_manifest(
        action_only_checkpoint=only["no_arrow_treatment_adapter"]["path"],
        action_visual_checkpoint=visual["no_arrow_treatment_adapter"]["path"],
        action_only_training_manifest=str(only_path), action_visual_training_manifest=str(visual_path),
        action_only_legacy_evidence_bundle=only["_legacy_bundle"], training_data_root=only["_data_root"],
        output_root=tmp_path / "eval", device="plan",
    )
    assert manifest["comparison_type"] == "retrospective_matched_checkpoint_evaluation"
    assert manifest["causal_ablation_status"] == "retrospective_not_strict"
    assert manifest["raw_pair_sentinel_digests"]["historical_action_only"] != manifest["raw_pair_sentinel_digests"]["action_visual_candidate"]
    assert manifest["stable_verified_pair_identity"] == "p" * 64


def test_build_manifest_rejects_sealed_base_snapshot_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    only, visual, only_path, visual_path = _build_manifest_fixture(tmp_path, monkeypatch)
    (Path(only["base_policy"]) / "sealed_config.json").write_text("tampered base", encoding="utf-8")
    with pytest.raises(ValueError, match="base snapshot hash mismatch"):
        pair_eval.build_manifest(
            action_only_checkpoint=only["no_arrow_treatment_adapter"]["path"],
            action_visual_checkpoint=visual["no_arrow_treatment_adapter"]["path"],
            action_only_training_manifest=str(only_path),
            action_visual_training_manifest=str(visual_path),
            action_only_legacy_evidence_bundle=only["_legacy_bundle"],
            training_data_root=only["_data_root"],
            output_root=tmp_path / "eval",
            device="plan",
        )


def test_build_manifest_rejects_same_checkpoint_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    only, visual = _fixture(tmp_path, monkeypatch)
    only_path = tmp_path / "only.json"
    visual_path = tmp_path / "visual.json"
    only_path.write_text(json.dumps(only), encoding="utf-8")
    visual_path.write_text(json.dumps(visual), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct checkpoint paths"):
        pair_eval.build_manifest(
            action_only_checkpoint=only["no_arrow_treatment_adapter"]["path"],
            action_visual_checkpoint=only["no_arrow_treatment_adapter"]["path"],
            action_only_training_manifest=str(only_path),
            action_visual_training_manifest=str(visual_path),
            action_only_legacy_evidence_bundle=only["_legacy_bundle"],
            training_data_root=only["_data_root"],
            output_root=tmp_path / "eval",
            device="plan",
        )


def test_build_manifest_rejects_distinct_paths_with_identical_adapter_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    only, visual = _fixture(tmp_path, monkeypatch)
    only_path = tmp_path / "only.json"
    visual_path = tmp_path / "visual.json"
    only_path.write_text(json.dumps(only), encoding="utf-8")
    visual_adapter = Path(visual["no_arrow_treatment_adapter"]["path"])
    only_adapter = Path(only["no_arrow_treatment_adapter"]["path"])
    (visual_adapter / "adapter_model.safetensors").write_bytes((only_adapter / "adapter_model.safetensors").read_bytes())
    visual["no_arrow_treatment_adapter"]["sha256"] = _sha(visual_adapter / "adapter_model.safetensors")
    visual_path.write_text(json.dumps(visual), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct adapter artifact identities"):
        pair_eval.build_manifest(
            action_only_checkpoint=str(only_adapter),
            action_visual_checkpoint=str(visual_adapter),
            action_only_training_manifest=str(only_path),
            action_visual_training_manifest=str(visual_path),
            action_only_legacy_evidence_bundle=only["_legacy_bundle"],
            training_data_root=only["_data_root"],
            output_root=tmp_path / "eval",
            device="plan",
        )


def _reset_record(task_id: int = 0, selected_index: int = 3, state: str = "a" * 64) -> dict:
    return {
        "task_id": task_id,
        "env_index": 0,
        "reset_sequence": 1,
        "details": {
            "removed": ["cookies_1"],
            "projection": {"success": True},
            "protected": {"akita_black_bowl_1": True, "plate_1": True},
            "layout": None,
            "swaps": None,
            "sim_state_sha256": state,
            "init_state": {"selected_index": selected_index, "selected_row_sha256": "b" * 64},
        },
    }


def test_paired_reset_audit_rejects_different_reset_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "randomization_audit.jsonl").write_text(json.dumps(_reset_record()) + "\n", encoding="utf-8")
    (right / "randomization_audit.jsonl").write_text(json.dumps(_reset_record(selected_index=4)) + "\n", encoding="utf-8")
    monkeypatch.setattr(pair_eval, "validate_randomization_audit", lambda *args, **kwargs: None)
    cells = [{"output_dir": str(left)}, {"output_dir": str(right)}]
    with pytest.raises(ValueError, match="identical reset/randomization identities"):
        pair_eval._paired_reset_audit(cells, {})


def test_episode_and_task_results_are_per_task_and_delta_is_visual_minus_action_only():
    manifest = {"paired_reset_states": True}
    cells = [
        {"cell_id": pair_eval.CELL_IDS[0], "policy_id": ACTION_ONLY_LORA_V1, "training_manifest_sha256": "a", "adapter_sha256": "b", "output_dir": "only"},
        {"cell_id": pair_eval.CELL_IDS[1], "policy_id": ACTION_VISUAL_LORA_V1, "training_manifest_sha256": "c", "adapter_sha256": "d", "output_dir": "visual"},
    ]
    info = lambda successes: {"per_task": [{"task_id": task, "metrics": {"successes": successes, "sum_rewards": [1.0] * len(successes), "max_rewards": [1.0] * len(successes)}} for task in range(10)]}
    only_rows = pair_eval._task_results(cells[0], info([True]))
    visual_rows = pair_eval._task_results(cells[1], info([False]))
    assert len(only_rows) == len(visual_rows) == 10
    assert {row["task_id"] for row in only_rows} == set(range(10))
    paired = pair_eval._paired_results({ACTION_ONLY_LORA_V1: only_rows, ACTION_VISUAL_LORA_V1: visual_rows})
    assert len(paired) == 11
    assert paired[0]["delta_pp"] == -100.0
    assert paired[-1]["task_id"] == "all"
    records = pair_eval._emit_episode_records(cells[0], info([True]), manifest)
    assert len(records) == 10
    assert records[0]["task_id"] == 0 and records[0]["episode_index"] == 0
    assert records[0]["live_arrows"] is False


def test_clean_cell_rejects_visual_relation_audit(tmp_path: Path):
    cell = {"output_dir": str(tmp_path)}
    (tmp_path / "visual_relation_audit.jsonl").write_text('{"condition":"visual_arrows"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="visual arrow audit"):
        pair_eval._validate_no_live_arrows(cell)


def test_eval_task_order_is_sealed():
    info = {"per_task": [{"task_id": task} for task in range(10)]}
    pair_eval._validate_task_order(info)
    info["per_task"] = list(reversed(info["per_task"]))
    with pytest.raises(ValueError, match="task order"):
        pair_eval._validate_task_order(info)


def test_historical_schema1_action_audit_is_accepted_but_candidate_audit_is_explicit():
    recorded = {
        "schema_version": 1,
        "target_regex": get_policy(ACTION_ONLY_LORA_V1).target_regex,
        "peft_type": "LORA",
        "rank": 16,
        "modules_to_save": [],
        "tensor_count": 14,
        "tensor_keys": [],
        "tensor_shapes": {},
        "tensor_inventory_sha256": "tensor-hash",
        "adapter_sha256": "adapter-hash",
        "checkpoint_inventory": {"adapter_model.safetensors": "adapter-hash"},
        "checkpoint_tree_sha256": "old-tree",
        "nonzero_lora_b_count": 7,
        "nonzero_lora_b_required": True,
        "expected_inventory_sha256": "inventory-hash",
        "expected_matched_module_names": [],
    }
    live = dict(recorded)
    live.update({"finetuning_policy_id": ACTION_ONLY_LORA_V1, "checkpoint_tree_sha256": "new-tree"})
    # Current and historical reports must agree on every legacy field; the
    # fixture's tree value is intentionally unchanged because it is part of
    # the authenticated legacy report.
    live["checkpoint_tree_sha256"] = "old-tree"
    assert pair_eval._audit_evidence_matches(recorded, live, ACTION_ONLY_LORA_V1)
    assert not pair_eval._audit_evidence_matches(recorded, live, ACTION_VISUAL_LORA_V1)
    live["tensor_count"] = 16
    assert not pair_eval._audit_evidence_matches(recorded, live, ACTION_ONLY_LORA_V1)
    truncated = {"schema_version": 1}
    assert not pair_eval._audit_evidence_matches(truncated, live, ACTION_ONLY_LORA_V1)
