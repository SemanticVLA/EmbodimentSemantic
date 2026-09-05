from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vla_benchmarking.shared.config import TASK_NAMES, TASK_PROMPT_OVERRIDE, TASK_REMOVE_CONFIG, TASK_SWAP_CONFIG

from vla_benchmarking.arrow_finetuned_vla.workflows.run_lora_2x2_eval import (
    CELL_IDS,
    MANIFEST_FILENAME,
    RAW_TRAINING_CAMERAS,
    TRAINING_CAMERAS,
    build_manifest,
    _build_contrast_rows,
    _extract_task_rows,
    compare_manifests,
    parse_int_list,
    validate_eval_info,
    validate_randomization_audit,
    validate_existing_outputs,
    validate_training_manifest,
    _resume_record_digest,
    write_immutable_manifest,
)


def _training_manifest_fixture(tmp_path: Path, steps: list[int]):
    """Build a local, immutable training lineage without invoking evaluation."""
    base = tmp_path / "base-policy"
    base.mkdir(parents=True)
    adapter = tmp_path / "adapter" / "adapter_model.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"adapter-bytes")
    plan = tmp_path / "training_plan.json"
    plan.write_text("{\"sealed\":true}\n", encoding="utf-8")
    pair_manifest = tmp_path / "pair_manifest.json"
    pair_manifest.write_bytes(b"pair")
    pair_sentinel = tmp_path / "pair_sentinel.json"
    pair_sentinel.write_bytes(b"sentinel")

    audits = []
    previous = None
    for step in steps:
        config = tmp_path / "checkpoints" / str(step) / "pretrained_model" / "train_config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({"steps": 29190, "checkpoint_step": step}) + "\n", encoding="utf-8")
        record = {
            "schema_version": 1,
            "chain_index": len(audits) + 1,
            "train_config_path": str(config.resolve()),
            "train_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "checkpoint_step": step,
            "profile": "treatment",
            "dataset_variant": "treatment",
            "previous_record_sha256": previous,
        }
        record["record_sha256"] = _resume_record_digest(record)
        audits.append(record)
        previous = record["record_sha256"]
    chain_digest = hashlib.sha256(
        json.dumps([record["record_sha256"] for record in audits], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    training_manifest = tmp_path / "training_manifest.json"
    data = {
        "experiment": "smolvla_lora_treatment_training",
        "base_policy_revision": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
        "training_variant": "treatment",
        "dataset_variant": "treatment",
        "training_plan": str(plan.resolve()),
        "training_plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "pair_manifest": str(pair_manifest.resolve()),
        "pair_manifest_sha256": hashlib.sha256(pair_manifest.read_bytes()).hexdigest(),
        "pair_sentinel": str(pair_sentinel.resolve()),
        "pair_sentinel_sha256": hashlib.sha256(pair_sentinel.read_bytes()).hexdigest(),
        "base_policy": str(base.resolve()),
        "treatment_adapter": {
            "path": str(adapter.resolve()),
            "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        },
        "resume_audits": audits,
        "resume_chain_digest": chain_digest,
    }
    training_manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    eval_manifest = build_manifest(
        base_checkpoint=str(base),
        treatment_checkpoint=str(adapter),
        seeds=[1000],
        output_root=tmp_path / "eval-output",
    )
    return training_manifest, eval_manifest, data


def test_manifest_contains_all_tasks_fixed_cameras_and_four_cells_per_seed(tmp_path):
    assert CELL_IDS == (
        "treatment_live_arrows",
        "treatment_no_arrows",
        "base_live_arrows",
        "base_no_arrows",
    )
    assert TRAINING_CAMERAS == "agentview_image,robot0_eye_in_hand_image"
    assert RAW_TRAINING_CAMERAS == "agentview,robot0_eye_in_hand"
    manifest = build_manifest(
        base_checkpoint="smolvla-libero",
        treatment_checkpoint="treatment-adapter",
        seeds=[1000, 1001],
        output_root=tmp_path,
    )

    assert manifest["tasks"] == list(range(10))
    assert manifest["tasks"][2] == 2 and manifest["tasks"][6] == 6
    assert manifest["randomize_scenes"] is True
    assert manifest["camera_name"] == "agentview_image,robot0_eye_in_hand_image"
    assert manifest["raw_camera_names"] == "agentview,robot0_eye_in_hand"
    assert manifest["observation_height"] == manifest["observation_width"] == 256
    assert manifest["visual_arrow_width"] == 1
    assert manifest["visual_arrow_head_length"] == 16
    assert {(cell["seed"], cell["cell_id"]) for cell in manifest["cells"]} == {
        (seed, cell_id) for seed in (1000, 1001) for cell_id in CELL_IDS
    }
    assert [cell["cell_id"] for cell in manifest["cells"]] == [
        cell_id for _seed in (1000, 1001) for cell_id in CELL_IDS
    ]
    assert {cell["checkpoint"] for cell in manifest["cells"] if cell["cell_id"].startswith("base")} == {
        "smolvla-libero"
    }
    assert {cell["checkpoint"] for cell in manifest["cells"] if cell["cell_id"].startswith("treatment")} == {
        "treatment-adapter"
    }
    assert manifest["randomization_dimensions"]["2"] == {
        "scene_layout": True,
        "object_removal": True,
        "prompt_variant": True,
    }
    assert manifest["randomization_config"]["prompt"] == {
        str(task_id): prompt for task_id, prompt in sorted(TASK_PROMPT_OVERRIDE.items())
    }
    assert set(manifest["randomization_config"]["prompt"]) == {str(task_id) for task_id in range(10)}
    assert all(
        values["prompt_variant"]
        for values in manifest["randomization_dimensions"].values()
    )
    assert all(
        values["object_removal"] and (values["scene_layout"] or task_id in {"0", "1", "3", "5", "6"})
        for task_id, values in manifest["randomization_dimensions"].items()
    )
    with pytest.raises(ValueError, match="batch_size=1"):
        build_manifest(
            base_checkpoint="base",
            treatment_checkpoint="treatment",
            seeds=[1000],
            batch_size=2,
            output_root=tmp_path / "batch2",
        )


def test_hybrid_layout_has_one_safe_operation_per_task():
    assert set(TASK_REMOVE_CONFIG) == set(range(10))
    assert set(TASK_SWAP_CONFIG) == {2, 4, 7, 8, 9}
    assert TASK_REMOVE_CONFIG[3] == ["glazed_rim_porcelain_ramekin_1"]  # cookies support target
    assert TASK_REMOVE_CONFIG[5] == ["cookies_1"]  # ramekin supports target
    assert all(task_id not in TASK_SWAP_CONFIG for task_id in {0, 1, 3, 5, 6})
    for task_id in range(10):
        removed = TASK_REMOVE_CONFIG[task_id]
        assert len(removed) == 1
        assert removed[0] not in {"akita_black_bowl_1", "akita_black_bowl_2", "plate_1"}
        if task_id in {0, 1, 3, 5, 6}:
            assert task_id not in TASK_SWAP_CONFIG
            continue
        operation = TASK_SWAP_CONFIG[task_id]
        assert len(operation) == 1
        left, right = operation[0]
        assert left not in {"akita_black_bowl_1", "plate_1", removed[0]}
        assert right not in {"akita_black_bowl_1", "plate_1", removed[0], left}


def test_manifest_is_immutable_and_mismatch_is_rejected(tmp_path):
    manifest = build_manifest(
        base_checkpoint="base",
        treatment_checkpoint="treatment",
        seeds=[3],
        output_root=tmp_path,
    )
    path = tmp_path / MANIFEST_FILENAME
    write_immutable_manifest(path, manifest)
    write_immutable_manifest(path, manifest)

    changed = dict(manifest)
    changed["device"] = "cpu"
    with pytest.raises(ValueError, match="does not match"):
        write_immutable_manifest(path, changed)

    prompt_changed = dict(manifest)
    prompt_changed["randomization_config"] = {
        **manifest["randomization_config"],
        "prompt": {"0": "tampered prompt"},
    }
    with pytest.raises(ValueError, match="randomization config"):
        write_immutable_manifest(tmp_path / "prompt_changed.json", prompt_changed)


def test_stale_cell_marker_is_rejected(tmp_path):
    manifest = build_manifest(
        base_checkpoint="base",
        treatment_checkpoint="treatment",
        seeds=[3],
        output_root=tmp_path,
    )
    cell = manifest["cells"][0]
    cell_dir = tmp_path / "seed_3" / cell["cell_id"]
    cell_dir.mkdir(parents=True)
    (cell_dir / "cell_manifest.json").write_text(json.dumps({"cell_id": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        validate_existing_outputs(tmp_path, manifest)


def test_stale_unexpected_seed_directory_is_rejected(tmp_path):
    manifest = build_manifest(
        base_checkpoint="base",
        treatment_checkpoint="treatment",
        seeds=[3],
        output_root=tmp_path,
    )
    (tmp_path / "seed_999" / "base_no_arrows").mkdir(parents=True)
    with pytest.raises(ValueError, match="unexpected stale"):
        validate_existing_outputs(tmp_path, manifest)


def test_seed_parser_requires_unique_explicit_values():
    assert parse_int_list("[2, 6, 10]") == [2, 6, 10]
    with pytest.raises(ValueError, match="duplicates"):
        parse_int_list("1,1")


def test_hybrid_variation_enables_prompt_dimension_for_every_task(tmp_path):
    manifest = build_manifest(
        base_checkpoint="base",
        treatment_checkpoint="treatment",
        seeds=[3],
        output_root=tmp_path,
    )
    assert manifest["randomization_dimensions"]["2"]["scene_layout"] is True
    assert manifest["randomization_dimensions"]["2"]["object_removal"] is True
    assert all(
        manifest["randomization_dimensions"][str(task_id)]["prompt_variant"]
        for task_id in range(10)
    )


def test_prompt_overrides_are_nonempty_and_change_canonical_task_text():
    assert set(TASK_PROMPT_OVERRIDE) == set(range(10))
    assert all(
        isinstance(prompt, str)
        and prompt.strip()
        and prompt != TASK_NAMES[task_id]
        for task_id, prompt in TASK_PROMPT_OVERRIDE.items()
    )


def test_randomization_audit_requires_realized_dimension(tmp_path):
    manifest = build_manifest(
        base_checkpoint="base",
        treatment_checkpoint="treatment",
        seeds=[3],
        output_root=tmp_path,
    )
    output = tmp_path / "cell"
    output.mkdir()
    rows = []
    for task_id in range(10):
        rows.append(json.dumps({
            "task_id": task_id,
            "env_index": 0,
            "reset_sequence": 1,
            "dimensions_enabled": manifest["randomization_dimensions"][str(task_id)],
            "dimensions_realized": manifest["randomization_dimensions"][str(task_id)],
            "details": {
                "removed": manifest["randomization_config"]["remove"][str(task_id)],
                "projection": {"success": True},
                "protected": {"akita_black_bowl_1": True, "plate_1": True},
                **({"layout": {
                    "configured": manifest["randomization_config"]["layout"][str(task_id)],
                    "applied": [label for operation in manifest["randomization_config"]["layout"][str(task_id)] for label in operation],
                    "skipped": [],
                }} if manifest["randomization_dimensions"][str(task_id)]["scene_layout"] else {}),
            },
            "status": "ok",
        }))
    (output / "randomization_audit.jsonl").write_text("\n".join(rows), encoding="utf-8")
    validate_randomization_audit(output, manifest)
    bad = output / "bad"
    bad.mkdir()
    rows[2] = json.dumps({
        "task_id": 2,
        "env_index": 0,
        "reset_sequence": 1,
        "dimensions_enabled": {"scene_layout": True, "object_removal": True, "prompt_variant": False},
        "dimensions_realized": {"scene_layout": True, "object_removal": False, "prompt_variant": False},
        "details": {"removed": ["glazed_rim_porcelain_ramekin_1"], "projection": {"success": True}, "protected": {"akita_black_bowl_1": True, "plate_1": True}, "layout": {"configured": [["akita_black_bowl_2", "cookies_1"]], "applied": ["akita_black_bowl_2", "cookies_1"], "skipped": []}},
        "status": "ok",
    })
    (bad / "randomization_audit.jsonl").write_text("\n".join(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="task 2"):
        validate_randomization_audit(bad, manifest)


def test_randomization_audit_requires_unique_complete_reset_records(tmp_path):
    manifest = build_manifest(
        base_checkpoint="base",
        treatment_checkpoint="treatment",
        seeds=[3],
        episodes=2,
        output_root=tmp_path,
    )

    def record(task_id, reset_sequence):
        dimensions = manifest["randomization_dimensions"][str(task_id)]
        return {
            "task_id": task_id,
            "env_index": 0,
            "reset_sequence": reset_sequence,
            "dimensions_enabled": dimensions,
            "dimensions_realized": dimensions,
            "details": {
                "removed": manifest["randomization_config"]["remove"][str(task_id)],
                "projection": {"success": True},
                "protected": {"akita_black_bowl_1": True, "plate_1": True},
                **({
                    "layout": {
                        "configured": manifest["randomization_config"]["layout"][str(task_id)],
                        "applied": [label for operation in manifest["randomization_config"]["layout"][str(task_id)] for label in operation],
                        "skipped": [],
                    }
                } if dimensions["scene_layout"] else {}),
            },
            "status": "ok",
        }

    output = tmp_path / "complete"
    output.mkdir()
    records = [record(task_id, reset) for task_id in range(10) for reset in (1, 2)]
    (output / "randomization_audit.jsonl").write_text(
        "\n".join(json.dumps(item) for item in records), encoding="utf-8"
    )
    validate_randomization_audit(output, manifest)

    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    duplicate_records = records + [record(0, 1)]
    (duplicate / "randomization_audit.jsonl").write_text(
        "\n".join(json.dumps(item) for item in duplicate_records), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_randomization_audit(duplicate, manifest)

    partial = tmp_path / "partial"
    partial.mkdir()
    partial_records = [item for item in records if not (item["task_id"] == 5 and item["reset_sequence"] == 2)]
    (partial / "randomization_audit.jsonl").write_text(
        "\n".join(json.dumps(item) for item in partial_records), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly 2"):
        validate_randomization_audit(partial, manifest)


def test_compare_manifests_accepts_fixed_camera_contract(tmp_path):
    randomized = build_manifest(
        base_checkpoint="base",
        treatment_checkpoint="treatment",
        seeds=[3],
        output_root=tmp_path / "randomized",
    )
    assert compare_manifests(randomized, randomized) == {}


def _complete_eval_info(episodes: int = 2):
    return {
        "per_task": [
            {
                "task_id": task_id,
                "metrics": {
                    "successes": [True, False][:episodes],
                    "sum_rewards": [1.0, 0.0][:episodes],
                    "max_rewards": [1.0, 0.0][:episodes],
                },
            }
            for task_id in range(10)
        ],
        "overall": {"n_episodes": 10 * episodes, "pc_success": 50.0},
    }


def test_eval_info_requires_all_tasks_and_expected_episode_arrays(tmp_path):
    manifest = build_manifest(
        base_checkpoint="base",
        treatment_checkpoint="treatment",
        seeds=[3],
        episodes=2,
        output_root=tmp_path,
    )
    path = tmp_path / "eval_info.json"
    path.write_text(json.dumps(_complete_eval_info()), encoding="utf-8")
    validate_eval_info(path, manifest)

    partial = _complete_eval_info()
    partial["per_task"][2]["metrics"]["successes"] = [True]
    path.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(ValueError, match="does not have 2 episodes"):
        validate_eval_info(path, manifest)

    missing = _complete_eval_info()
    missing["per_task"] = missing["per_task"][:-1]
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 10 task records"):
        validate_eval_info(path, manifest)


def test_summary_contains_task_rows_and_2x2_interaction(tmp_path):
    manifest = build_manifest(
        base_checkpoint="base",
        treatment_checkpoint="treatment",
        seeds=[3, 4],
        output_root=tmp_path,
    )
    infos = {}
    for seed in (3, 4):
        for cell_id in (
            "base_no_arrows",
            "base_live_arrows",
            "treatment_no_arrows",
            "treatment_live_arrows",
        ):
            cell = next(item for item in manifest["cells"] if item["seed"] == seed and item["cell_id"] == cell_id)
            info = _complete_eval_info(1)
            # Give the treatment live cell a distinct task-0 result.
            if cell_id == "treatment_live_arrows":
                info["per_task"][0]["metrics"]["successes"] = [True]
            infos[(seed, cell_id)] = (cell, info)
    task_rows = _extract_task_rows(infos[(3, "base_no_arrows")][0], infos[(3, "base_no_arrows")][1])
    assert len(task_rows) == 10
    contrast_rows = _build_contrast_rows(infos, manifest)
    assert any(row["contrast"] == "arrow_treatment_interaction" and row["seed"] == "aggregate" for row in contrast_rows)
    assert any(row["task_id"] == "all" for row in contrast_rows)


def test_training_manifest_accepts_empty_resume_audit_chain(tmp_path):
    path, manifest, _ = _training_manifest_fixture(tmp_path, [])
    validated = validate_training_manifest(path, manifest)
    assert validated["resume_audits"] == []


def test_training_manifest_accepts_valid_nonempty_resume_audit_chain(tmp_path):
    path, manifest, data = _training_manifest_fixture(tmp_path, [2, 4])
    validated = validate_training_manifest(path, manifest)
    assert [record["checkpoint_step"] for record in validated["resume_audits"]] == [2, 4]
    assert validated["resume_chain_digest"] == data["resume_chain_digest"]


def test_training_manifest_rejects_retired_target_arrow_profile(tmp_path):
    path, manifest, data = _training_manifest_fixture(tmp_path, [])
    data["training_variant"] = "target_arrow_treatment"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="training manifest is not for the arrow treatment dataset"):
        validate_training_manifest(path, manifest)


def test_training_manifest_rejects_mutated_record_or_broken_chain(tmp_path):
    path, manifest, data = _training_manifest_fixture(tmp_path, [2, 4])
    data["resume_audits"][0]["checkpoint_step"] = 3
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="resume audit record hash mismatch"):
        validate_training_manifest(path, manifest)

    path, manifest, data = _training_manifest_fixture(tmp_path / "linkage", [2, 4])
    data["resume_audits"][1]["previous_record_sha256"] = "broken-link"
    data["resume_audits"][1]["record_sha256"] = _resume_record_digest(data["resume_audits"][1])
    data["resume_chain_digest"] = hashlib.sha256(
        json.dumps([record["record_sha256"] for record in data["resume_audits"]], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="previous-record linkage"):
        validate_training_manifest(path, manifest)

    path, manifest, data = _training_manifest_fixture(tmp_path / "digest", [2])
    data["resume_chain_digest"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="chain digest mismatch"):
        validate_training_manifest(path, manifest)


def test_training_manifest_rejects_missing_or_mutated_referenced_train_config(tmp_path):
    path, manifest, data = _training_manifest_fixture(tmp_path, [2])
    config_path = Path(data["resume_audits"][0]["train_config_path"])
    config_path.unlink()
    with pytest.raises(ValueError, match="train_config.json is missing"):
        validate_training_manifest(path, manifest)

    path, manifest, data = _training_manifest_fixture(tmp_path / "mutated", [2])
    config_path = Path(data["resume_audits"][0]["train_config_path"])
    config_path.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train_config.json hash changed"):
        validate_training_manifest(path, manifest)
