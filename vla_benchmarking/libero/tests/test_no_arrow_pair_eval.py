from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vla_benchmarking.libero.finetuned_vlas.smolvla.workflows.run_lora_no_arrow_pair_eval import (
    CELL_IDS,
    SMOKE_CHECKPOINT_ID,
    SMOKE_EPISODES,
    SMOKE_SAVE_FREQ,
    SMOKE_STEPS,
    SMOKE_TASKS,
    SEALED_CHECKPOINT_ID,
    SEALED_SAVE_FREQ,
    SEALED_STEPS,
    _evaluation_contract,
    _build_contrast_rows,
    build_manifest,
    get_profile,
    validate_existing_outputs,
    validate_training_manifest,
)


REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    base = tmp_path / f"smolvla_libero-{REVISION}"
    base.mkdir()
    (base / "config.json").write_text("{}\n", encoding="utf-8")
    base_files = {"config.json": _sha(base / "config.json")}
    (base / "base_snapshot_manifest.json").write_text(
        json.dumps({"revision": REVISION, "files": base_files}) + "\n", encoding="utf-8"
    )
    adapter = tmp_path / "run" / "checkpoints" / "029190" / "pretrained_model" / "adapter_model.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"no-arrow adapter")
    plan = tmp_path / "run" / "training_plan.json"
    plan.write_text(
        json.dumps(
            {
                "experiment": "smolvla_lora_no_arrow_treatment_training",
                "training_variant": "no_arrow_treatment",
                "dataset_variant": "control",
                "dataset_repo_id": "local/libero_spatial_control",
                "trained_on_visual_condition": "no_arrows",
                "base_policy": str(base.resolve()),
                "base_policy_revision": REVISION,
                "flags": {"steps": 29190, "save_freq": 1946, "batch_size": 32, "seed": 1000, "peft_r": 16},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pair_manifest = tmp_path / "sealed_lora_pair_manifest.json"
    pair_manifest.write_text(
        json.dumps({"pair_kind": "sealed_lora_control_treatment", "full_experiment_ready": True, "launch_eligibility": "full_experiment_ready"}) + "\n",
        encoding="utf-8",
    )
    pair_sentinel = tmp_path / "sealed_lora_pair_verified.json"
    pair_sentinel.write_text(
        json.dumps({"pair_kind": "sealed_lora_control_treatment", "full_experiment_ready": True, "launch_eligibility": "full_experiment_ready"}) + "\n",
        encoding="utf-8",
    )
    training = tmp_path / "run" / "training_manifest.json"
    training.write_text(
        json.dumps(
            {
                "experiment": "smolvla_lora_no_arrow_treatment_training",
                "training_variant": "no_arrow_treatment",
                "dataset_variant": "control",
                "trained_on_visual_condition": "no_arrows",
                "dataset_repo_id": "local/libero_spatial_control",
                "base_policy": str(base.resolve()),
                "base_policy_revision": REVISION,
                "training_plan": str(plan.resolve()),
                "training_plan_sha256": _sha(plan),
                "pair_manifest": str(pair_manifest.resolve()),
                "pair_manifest_sha256": _sha(pair_manifest),
                "pair_sentinel": str(pair_sentinel.resolve()),
                "pair_sentinel_sha256": _sha(pair_sentinel),
                "pair_kind": "sealed_lora_control_treatment",
                "resume_audits": [],
                "resume_chain_digest": hashlib.sha256(b"[]").hexdigest(),
                "final_checkpoint_id": "029190",
                "flags": {"steps": 29190, "save_freq": 1946, "batch_size": 32, "seed": 1000, "peft_r": 16},
                "no_arrow_treatment_adapter": {"path": str(adapter.resolve()), "sha256": _sha(adapter)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "eval"
    manifest = build_manifest(
        adapter_checkpoint=str(adapter.parent),
        seeds=[1000],
        episodes=10,
        training_manifest=str(training),
        output_root=output,
    )
    return training, manifest


def test_no_arrow_manifest_is_all_task_two_cell_ordered_and_same_adapter(tmp_path: Path):
    training, manifest = _fixture(tmp_path)
    assert manifest["tasks"] == list(range(10))
    assert [cell["cell_id"] for cell in manifest["cells"][:2]] == list(CELL_IDS)
    assert all(cell["checkpoint"] == manifest["adapter_checkpoint"] for cell in manifest["cells"])
    assert manifest["contrast"] == "live_arrow_effect_pp"
    assert manifest["randomization_dimensions"]["2"]["object_removal"] is True
    assert manifest["training_manifest_sha256"] == _sha(training)
    assert len({cell["adapter_sha256"] for cell in manifest["cells"]}) == 1


def test_evaluation_scope_contract_seals_smoke_to_target_arrow_two_step_checkpoint():
    full = _evaluation_contract("full", get_profile("no_arrow_treatment"))
    assert full["checkpoint_id"] == SEALED_CHECKPOINT_ID
    assert full["steps"] == SEALED_STEPS
    assert full["save_freq"] == SEALED_SAVE_FREQ
    assert full["tasks"] == tuple(range(10))
    assert full["episodes"] == 10

    smoke = _evaluation_contract("smoke", get_profile("target_arrow_treatment"))
    assert smoke["checkpoint_id"] == SMOKE_CHECKPOINT_ID
    assert smoke["steps"] == SMOKE_STEPS
    assert smoke["save_freq"] == SMOKE_SAVE_FREQ
    assert smoke["tasks"] == SMOKE_TASKS
    assert smoke["episodes"] == SMOKE_EPISODES

    with pytest.raises(ValueError, match="reserved for the target_arrow_treatment"):
        _evaluation_contract("smoke", get_profile("no_arrow_treatment"))


def test_no_arrow_profile_cannot_opt_into_target_arrow_smoke_scope(tmp_path: Path):
    _training, manifest = _fixture(tmp_path)
    with pytest.raises(ValueError, match="reserved for the target_arrow_treatment"):
        build_manifest(
            adapter_checkpoint=manifest["adapter_checkpoint"],
            seeds=[1000],
            tasks=list(SMOKE_TASKS),
            episodes=SMOKE_EPISODES,
            training_manifest=manifest["training_manifest"],
            output_root=tmp_path / "smoke-eval",
            profile_name="no_arrow_treatment",
            evaluation_scope="smoke",
        )


def test_no_arrow_training_lineage_accepts_only_sealed_no_arrow_manifest(tmp_path: Path):
    training, manifest = _fixture(tmp_path)
    validated = validate_training_manifest(training, manifest)
    assert validated["no_arrow_treatment_adapter"]["path"].endswith("adapter_model.safetensors")

    data = json.loads(training.read_text(encoding="utf-8"))
    data["treatment_adapter"] = data.pop("no_arrow_treatment_adapter")
    training.write_text(json.dumps(data), encoding="utf-8")
    manifest["training_manifest_sha256"] = _sha(training)
    with pytest.raises(ValueError, match="must not contain treatment_adapter"):
        validate_training_manifest(training, manifest)


def test_no_arrow_training_lineage_rejects_wrong_condition_or_experiment(tmp_path: Path):
    training, manifest = _fixture(tmp_path)
    data = json.loads(training.read_text(encoding="utf-8"))
    data["trained_on_visual_condition"] = "arrows"
    training.write_text(json.dumps(data), encoding="utf-8")
    manifest["training_manifest_sha256"] = _sha(training)
    with pytest.raises(ValueError, match="does not record no_arrows"):
        validate_training_manifest(training, manifest)

    data["trained_on_visual_condition"] = "no_arrows"
    data["experiment"] = "smolvla_lora_treatment_training"
    training.write_text(json.dumps(data), encoding="utf-8")
    manifest["training_manifest_sha256"] = _sha(training)
    with pytest.raises(ValueError, match="unexpected no-arrow training manifest"):
        validate_training_manifest(training, manifest)


def test_no_arrow_pair_contrast_contains_only_live_arrow_effect(tmp_path: Path):
    _training, manifest = _fixture(tmp_path)
    infos = {}
    for cell in manifest["cells"][:2]:
        success = [True, False] if cell["live_arrows"] else [False, False]
        infos[(cell["seed"], cell["cell_id"])] = (
            cell,
            {
                "per_task": [
                    {
                        "task_id": task_id,
                        "metrics": {"successes": success, "sum_rewards": [1.0, 0.0], "max_rewards": [1.0, 0.0]},
                    }
                    for task_id in range(10)
                ],
            },
        )
    rows = _build_contrast_rows(infos, manifest)
    assert len(rows) == 21
    assert {row["contrast"] for row in rows} == {"live_arrow_effect_pp"}
    assert all(row["pc_success"] == 50.0 for row in rows)


def test_smoke_pair_contrast_uses_only_scope_task_subset(tmp_path: Path):
    _training, full_manifest = _fixture(tmp_path)
    manifest = dict(full_manifest)
    manifest["tasks"] = list(SMOKE_TASKS)
    infos = {}
    for cell in manifest["cells"][:2]:
        success = [True] if cell["live_arrows"] else [False]
        infos[(cell["seed"], cell["cell_id"])] = (
            cell,
            {
                "per_task": [
                    {
                        "task_id": task_id,
                        "metrics": {"successes": success},
                    }
                    for task_id in SMOKE_TASKS
                ],
            },
        )
    rows = _build_contrast_rows(infos, manifest)
    assert [row["task_id"] for row in rows if row["seed"] == "aggregate"] == [0, 4, "all"]


def test_no_arrow_manifest_rejects_non_unit_batch_size(tmp_path: Path):
    adapter = tmp_path / "029190" / "pretrained_model"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    with pytest.raises(ValueError, match="batch_size=1"):
        build_manifest(
            adapter_checkpoint=str(adapter),
            seeds=[1000],
            episodes=10,
            batch_size=2,
            training_manifest=str(tmp_path / "training_manifest.json"),
            output_root=tmp_path / "eval",
        )


@pytest.mark.parametrize("seeds", ([1001], [1000, 1001]))
def test_no_arrow_pair_seals_single_approved_seed(tmp_path: Path, seeds):
    _training, manifest = _fixture(tmp_path)
    with pytest.raises(ValueError, match=r"seeds exactly \[1000\]"):
        build_manifest(
            adapter_checkpoint=manifest["adapter_checkpoint"],
            seeds=seeds,
            episodes=10,
            training_manifest=manifest["training_manifest"],
            output_root=tmp_path / "other-eval",
        )


@pytest.mark.parametrize("episodes", (1, 9, 11))
def test_no_arrow_pair_seals_ten_episodes(tmp_path: Path, episodes: int):
    _training, manifest = _fixture(tmp_path)
    with pytest.raises(ValueError, match="exactly 10 episodes"):
        build_manifest(
            adapter_checkpoint=manifest["adapter_checkpoint"],
            seeds=[1000],
            episodes=episodes,
            training_manifest=manifest["training_manifest"],
            output_root=tmp_path / "other-eval",
        )


def test_no_arrow_eval_rejects_unexpected_cell_directories(tmp_path: Path):
    _training, manifest = _fixture(tmp_path)
    seed_dir = Path(manifest["cells"][0]["output_dir"]).parent
    (seed_dir / "base_no_arrows").mkdir(parents=True)
    with pytest.raises(ValueError, match="unexpected stale cell"):
        validate_existing_outputs(Path(manifest["output_root"]), manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("base_policy_revision", "wrong-revision", "base policy revision"),
        ("dataset_repo_id", "local/libero_spatial_treatment", "dataset_repo_id"),
        ("final_checkpoint_id", "000002", "final checkpoint"),
    ],
)
def test_no_arrow_lineage_rejects_sealed_provenance_mutations(tmp_path: Path, field: str, value: str, message: str):
    training, manifest = _fixture(tmp_path)
    data = json.loads(training.read_text(encoding="utf-8"))
    data[field] = value
    training.write_text(json.dumps(data), encoding="utf-8")
    manifest["training_manifest_sha256"] = _sha(training)
    with pytest.raises(ValueError, match=message):
        validate_training_manifest(training, manifest)
