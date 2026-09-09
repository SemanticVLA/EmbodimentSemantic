from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from arrow_policy_suite.apprentice_training import (
    ApprenticeExportRequest,
    ApprenticeExportReceipt,
    build_apprentice_training_job,
    export_apprentice_dataset_native,
    load_apprentice_runtime_adapter,
    load_apprentice_transitions,
    run_apprentice_training_job,
    save_adapter_checkpoint,
    write_adapter_manifest,
    verify_adapter_reload,
    _canonical,
    _base_tree_sha256,
    DEFAULT_SMOLVLA_PEFT_TARGET_REGEX,
    SmolVLALoRAConfig,
)
from arrow_policy_suite.contracts import ContractError, ObservationFrame, digest
from arrow_policy_suite.learning import DatasetManifest, InterventionRow


def _obs() -> dict[str, object]:
    image = np.zeros((4, 5, 3), dtype=np.uint8).tolist()
    return {"agentview": image, "wrist": image, "state": [0.0] * 8, "instruction": "pick bowl"}


def _row() -> InterventionRow:
    return InterventionRow("ep", 3, 0, _obs(), (0.0,) * 7, (0.2,) + (0.0,) * 6, True)


def _write_peft_bundle(path: Path, *, base: str, revision: str = "revision",
                       processor_revision: str | None = None, rank: int = 16,
                       include_processor_revision: bool = True) -> None:
    import torch
    from safetensors.torch import save_file
    processor_revision = processor_revision or revision

    config = {
        "peft_type": "LORA", "r": rank, "lora_alpha": 8, "lora_dropout": 0.0,
        "bias": "none", "target_modules": DEFAULT_SMOLVLA_PEFT_TARGET_REGEX,
        "modules_to_save": None, "base_model_name_or_path": base, "revision": revision,
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file({"base_model.model.state_proj.lora_A.default.weight": torch.zeros((1, 1)),
               "base_model.model.state_proj.lora_B.default.weight": torch.zeros((1, 1))},
              str(path / "adapter_model.safetensors"))
    processor_payload = {"processor_revision": processor_revision} if include_processor_revision else {"normalization": "uint8"}
    (path / "policy_preprocessor.json").write_text(json.dumps(processor_payload), encoding="utf-8")
    (path / "policy_postprocessor.json").write_text(json.dumps(processor_payload), encoding="utf-8")


def test_loader_requires_raw_images_and_teacher_action(tmp_path: Path):
    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps({"teacher_action": [0.2] + [0.0] * 6, "episode_id": "ep", "task_id": 3,
                                "timestep": 0, "observation": _obs(), "success_episode": True}) + "\n")
    rows = load_apprentice_transitions(good)
    assert len(rows) == 1 and rows[0].teacher_action[0] == pytest.approx(0.2)

    digest_only = tmp_path / "digest.jsonl"
    digest_only.write_text(json.dumps({"teacher_action_digest": "a" * 64, "episode_id": "ep", "task_id": 3,
                                       "timestep": 0, "observation_digest": "b" * 64}) + "\n")
    with pytest.raises(ContractError, match="observation|teacher"):
        load_apprentice_transitions(digest_only)


def test_native_export_contains_images_state_task_and_lineage(tmp_path: Path):
    row = _row()
    parent = DatasetManifest("arrow_policy_suite.interventions.v1", 1, ("ep",), "source", "a" * 64, "filter")
    request = ApprenticeExportRequest((row,), parent, str(tmp_path / "dataset"))

    class FakeDataset:
        def __init__(self):
            self.frames = []
            self.saved = 0

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.saved += 1

        def finalize(self):
            pass

    holder = {}

    class Factory:
        @staticmethod
        def create(**kwargs):
            holder["kwargs"] = kwargs
            holder["dataset"] = FakeDataset()
            return holder["dataset"]

    receipt = export_apprentice_dataset_native(request, dataset_factory=Factory, fps=10)
    frame = holder["dataset"].frames[0]
    assert "observation.images.image" in frame and "observation.images.image2" in frame
    assert frame["observation.state"] == [0.0] * 8 and frame["task"] == "3"
    lineage = Path(receipt.dataset_path) / "meta" / "apprentice_lineage.json"
    assert json.loads(lineage.read_text())["parent_manifest_sha256"] == "a" * 64


def test_job_command_is_seeded_peft_and_frozen_base(tmp_path: Path):
    row = _row()
    parent = DatasetManifest("arrow_policy_suite.interventions.v1", 1, ("ep",), "source", "a" * 64, "filter")
    request = ApprenticeExportRequest((row,), parent, str(tmp_path / "dataset"))
    receipt = ApprenticeExportReceipt(request, str(tmp_path / "dataset"), "b" * 64, 1, None)
    job = build_apprentice_training_job(receipt, base_checkpoint="HuggingFaceVLA/smolvla_libero",
                                        output_dir=tmp_path / "run",
                                        lora=SmolVLALoRAConfig(base_vla_sha256="c" * 64,
                                                                model_revision="revision",
                                                                processor_revision="revision"))
    assert "--peft.r=16" in job.command and "--seed=1000" in job.command
    assert job.manifest["base_frozen"] is True and job.environment["PYTHONHASHSEED"] == "1000"


def test_runtime_loader_is_teacher_free_and_base_bound(tmp_path: Path):
    path = tmp_path / "adapter.bin"
    save_adapter_checkpoint(path, {"adapter.weight": [1.0]}, base_vla_sha256="c" * 64)
    seen = {}

    def loader(_path):
        return lambda frame, base: (seen.update(frame=frame, base=base) or (0.3,) + (0.0,) * 6)

    action = load_apprentice_runtime_adapter(path, base_vla_sha256="c" * 64, adapter_loader=loader)
    frame = ObservationFrame(_obs(), timestep=0, episode_id="eval")
    assert action(frame, (0.0,) * 7)[0] == pytest.approx(0.3)
    assert seen["frame"] is frame and seen["base"] == (0.0,) * 7


def test_target_regex_and_native_loader_composition(monkeypatch, tmp_path: Path):
    assert "lm_expert" in DEFAULT_SMOLVLA_PEFT_TARGET_REGEX
    assert "state_proj" in DEFAULT_SMOLVLA_PEFT_TARGET_REGEX
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / ".cache").mkdir()
    (base / ".cache" / "transient").write_text("ignored")
    base_hash = _base_tree_sha256(base)
    adapter = tmp_path / "pretrained_model"
    adapter.mkdir()
    _write_peft_bundle(adapter, base=str(base))
    write_adapter_manifest(
        adapter, base_vla_sha256=base_hash, base_checkpoint=base,
        model=SmolVLALoRAConfig(model_revision="revision", processor_revision="revision").manifest(),
        processor_provenance={"processor_revision": "revision", "selection_order": ["adapter", "base"]},
    )

    calls = {}
    class FakeConfig:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["peft_config"] = (path, kwargs)
            return cls()

    class FakePolicy:
        config = object()

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["base"] = (path, kwargs)
            return cls()

        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True

        def select_action(self, value):
            return value

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, policy, path, **kwargs):
            calls["adapter"] = (policy, path, kwargs)
            return policy

    peft_module = types.ModuleType("peft")
    peft_module.PeftConfig = FakeConfig
    peft_module.PeftModel = FakePeftModel
    lerobot_module = types.ModuleType("lerobot")
    policies_module = types.ModuleType("lerobot.policies")
    smolvla_module = types.ModuleType("lerobot.policies.smolvla.modeling_smolvla")
    smolvla_module.SmolVLAPolicy = FakePolicy
    policies_module.make_pre_post_processors = lambda **kwargs: (lambda value: value, lambda value: value)
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setitem(sys.modules, "lerobot", lerobot_module)
    monkeypatch.setitem(sys.modules, "lerobot.policies", policies_module)
    monkeypatch.setitem(sys.modules, "lerobot.policies.smolvla", types.ModuleType("lerobot.policies.smolvla"))
    monkeypatch.setitem(sys.modules, "lerobot.policies.smolvla.modeling_smolvla", smolvla_module)

    from arrow_policy_suite.apprentice_training import load_native_smolvla_adapter
    loaded = load_native_smolvla_adapter(adapter, base_checkpoint=base, base_vla_sha256=base_hash)
    assert loaded["policy"] is not None and calls["adapter"][2]["is_trainable"] is False
    assert calls["base"][1]["local_files_only"] is True


def test_training_source_v1_line_is_loaded_as_executed_teacher_transition(tmp_path: Path):
    row = {
        "schema": "arrow_policy_suite.training_source.v1",
        "task_id": 7,
        "reset_id": "r0",
        "episode_id": "ep-source",
        "timestep": 4,
        "identity": {"task_id": 7, "reset_id": "r0", "episode_id": "ep-source"},
        "observation": _obs(),
        "base_proposal": {"policy_id": "smolvla", "action": [0.1] + [0.0] * 6},
        "teacher_proposal": {"policy_id": "arrow_on_call", "action": [0.4] + [0.0] * 6},
        "decision": {"policy_id": "arrow_on_call", "teacher_used": True},
        "executed_action": [0.4] + [0.0] * 6,
        "executed_by": "arrow",
        "outcome": {"success": True, "terminal": False},
        "eligible": True,
    }
    row["observation_digest"] = digest(row["observation"])
    unsigned = dict(row)
    row["transition_sha256"] = __import__("hashlib").sha256(_canonical(unsigned)).hexdigest()
    source = tmp_path / "training_source.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = load_apprentice_transitions(source)
    assert len(loaded) == 1
    assert loaded[0].base_action[0] == pytest.approx(0.1)
    assert loaded[0].teacher_action[0] == pytest.approx(0.4)
    assert loaded[0].executed_action == loaded[0].teacher_action
    assert loaded[0].task_id == 7 and loaded[0].episode_id == "ep-source"
    assert loaded[0].outcome["success"] is True


def test_training_source_v1_rejects_ineligible_and_digest_only_rows(tmp_path: Path):
    skipped = {
        "schema": "arrow_policy_suite.training_source.v1", "task_id": 7,
        "reset_id": "r0", "episode_id": "ep-source", "timestep": 0,
        "observation": {"agentview": {"sha256": "a" * 64}, "wrist": [], "state": [0.0] * 8,
                        "instruction": "pick"},
        "base_proposal": {"action": [0.0] * 7},
        "teacher_proposal": {"policy_id": "arrow_on_call", "action": [0.1] + [0.0] * 6},
        "decision": {"policy_id": "arrow_on_call", "teacher_used": True},
        "executed_action": [0.1] + [0.0] * 6, "executed_by": "arrow",
        "outcome": {}, "eligible": False,
    }
    skipped["observation_digest"] = "a" * 64
    skipped["transition_sha256"] = __import__("hashlib").sha256(_canonical(skipped)).hexdigest()
    non_on_call = dict(skipped)
    non_on_call.update({"task_id": 8, "episode_id": "ep-vla", "timestep": 0, "eligible": True,
                        "decision": {"policy_id": "vla", "teacher_used": False},
                        "executed_by": "vla"})
    non_on_call.pop("transition_sha256")
    non_on_call["transition_sha256"] = __import__("hashlib").sha256(_canonical(non_on_call)).hexdigest()
    row = {
        "schema": "arrow_policy_suite.training_source.v1", "task_id": 7,
        "reset_id": "r0", "episode_id": "ep-source-good", "timestep": 1,
        "identity": {"task_id": 7, "reset_id": "r0", "episode_id": "ep-source-good"},
        "observation": _obs(),
        "base_proposal": {"action": [0.0] * 7},
        "teacher_proposal": {"policy_id": "arrow_on_call", "action": [0.1] + [0.0] * 6},
        "decision": {"policy_id": "arrow_on_call", "teacher_used": True},
        "executed_action": [0.1] + [0.0] * 6, "executed_by": "arrow",
        "outcome": {"success": True}, "eligible": True,
    }
    row["observation_digest"] = digest(row["observation"])
    row["transition_sha256"] = __import__("hashlib").sha256(_canonical(row)).hexdigest()
    source = tmp_path / "bad_source.jsonl"
    source.write_text(json.dumps(skipped) + "\n" + json.dumps(non_on_call) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    loaded = load_apprentice_transitions(source)
    assert [item.episode_id for item in loaded] == ["ep-source-good"]

    malformed = dict(row)
    malformed["observation"] = {"agentview": {"sha256": "a" * 64}, "wrist": [], "state": [0.0] * 8,
                                "instruction": "pick"}
    malformed["observation_digest"] = "a" * 64
    malformed.pop("transition_sha256")
    malformed["transition_sha256"] = __import__("hashlib").sha256(_canonical(malformed)).hexdigest()
    source.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="raw|image|observation"):
        load_apprentice_transitions(source)


def test_training_source_filters_apply_after_teacher_selection(tmp_path: Path):
    def source_row(task: int, reset: str, episode: str, timestep: int) -> dict[str, object]:
        row: dict[str, object] = {
            "schema": "arrow_policy_suite.training_source.v1", "task_id": task,
            "reset_id": reset, "episode_id": episode, "timestep": timestep,
            "identity": {"task_id": task, "reset_id": reset, "episode_id": episode},
            "observation": _obs(), "base_proposal": {"action": [0.0] * 7},
            "teacher_proposal": {"policy_id": "arrow_on_call", "action": [0.1] + [0.0] * 6},
            "decision": {"policy_id": "arrow_on_call", "teacher_used": True},
            "executed_action": [0.1] + [0.0] * 6, "executed_by": "arrow",
            "outcome": {"success": True}, "eligible": True,
        }
        row["observation_digest"] = digest(row["observation"])
        row["transition_sha256"] = __import__("hashlib").sha256(_canonical(row)).hexdigest()
        return row

    rows = [source_row(1, "r1", "ep-1", 0), source_row(2, "r2", "ep-2", 0)]
    source = tmp_path / "filter-source.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    selected = load_apprentice_transitions(source, task_ids=["1"], reset_ids=["r1"], episode_ids=["ep-1"])
    assert len(selected) == 1 and selected[0].task_id == 1
    with pytest.raises(ContractError, match="filters selected no"):
        load_apprentice_transitions(source, task_ids=["missing"])

    from arrow_policy_suite.apprentice_job import build_parser
    parsed = build_parser().parse_args([
        "--transitions", str(source), "--parent-artifact", "parent",
        "--dataset-root", "dataset", "--base-checkpoint", "base", "--base-sha256", "c" * 64,
        "--model-revision", "model-r1", "--processor-revision", "processor-r1", "--output-dir", "run",
        "--task-id", "1", "--reset-id", "r1", "--episode-id", "ep-1",
    ])
    assert parsed.task_ids == ["1"] and parsed.reset_ids == ["r1"] and parsed.episode_ids == ["ep-1"]


def test_simulated_training_emits_verifiable_in_bundle_manifest(tmp_path: Path):
    parent = DatasetManifest("arrow_policy_suite.interventions.v1", 1, ("ep",), "source", "a" * 64, "filter")
    request = ApprenticeExportRequest((_row(),), parent, str(tmp_path / "dataset"))
    receipt = ApprenticeExportReceipt(request, str(tmp_path / "dataset"), "b" * 64, 1, None)
    job = build_apprentice_training_job(
        receipt, base_checkpoint="HuggingFaceVLA/smolvla_libero", output_dir=tmp_path / "run",
        lora=SmolVLALoRAConfig(base_vla_sha256="c" * 64, model_revision="revision", processor_revision="processor-r1"),
    )

    def fake_runner(_command, **_kwargs):
        output = Path(job.output_dir)
        output.mkdir(parents=True)
        _write_peft_bundle(output, base="HuggingFaceVLA/smolvla_libero", revision="revision", processor_revision="processor-r1", include_processor_revision=False)
        return {"returncode": 0, "adapter_checkpoint": str(output / "adapter_model.safetensors")}

    result = run_apprentice_training_job(job, runner=fake_runner)
    bundle = Path(result["adapter_checkpoint"]).parent
    manifest_path = bundle / "apprentice_manifest.json"
    assert manifest_path.is_file()
    assert not Path(str(bundle) + ".json").exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["checkpoint_inventory"]
    assert "apprentice_manifest.json" not in manifest["checkpoint_inventory"]
    assert manifest["inventory_sha256"] == manifest["checkpoint_sha256"]
    assert manifest["base_vla_sha256"] == "c" * 64
    assert manifest["processor_provenance"]["processor_revision"] == "processor-r1"
    assert manifest["adapter_audit"]["processor"]["revision_binding"] == "sealed_training_job_input"
    assert manifest["adapter_audit"]["processor"]["revision_evidence"] == []
    verified = verify_adapter_reload(bundle, base_vla_sha256="c" * 64, loader=lambda _path: {})
    assert verified["reloaded"] is True


def test_simulated_training_rejects_peft_config_mismatch(tmp_path: Path):
    parent = DatasetManifest("arrow_policy_suite.interventions.v1", 1, ("ep",), "source", "a" * 64, "filter")
    request = ApprenticeExportRequest((_row(),), parent, str(tmp_path / "dataset"))
    receipt = ApprenticeExportReceipt(request, str(tmp_path / "dataset"), "b" * 64, 1, None)
    job = build_apprentice_training_job(
        receipt, base_checkpoint="HuggingFaceVLA/smolvla_libero", output_dir=tmp_path / "run",
        lora=SmolVLALoRAConfig(base_vla_sha256="c" * 64, model_revision="revision", processor_revision="processor-r1"),
    )

    def fake_runner(_command, **_kwargs):
        output = Path(job.output_dir)
        output.mkdir(parents=True)
        _write_peft_bundle(output, base="HuggingFaceVLA/smolvla_libero", revision="revision", processor_revision="processor-r1", rank=8)
        return {"returncode": 0, "adapter_checkpoint": str(output / "adapter_model.safetensors")}

    with pytest.raises(ContractError, match="adapter config r"):
        run_apprentice_training_job(job, runner=fake_runner)


def test_simulated_training_rejects_explicit_processor_revision_mismatch(tmp_path: Path):
    parent = DatasetManifest("arrow_policy_suite.interventions.v1", 1, ("ep",), "source", "a" * 64, "filter")
    request = ApprenticeExportRequest((_row(),), parent, str(tmp_path / "dataset"))
    receipt = ApprenticeExportReceipt(request, str(tmp_path / "dataset"), "b" * 64, 1, None)
    job = build_apprentice_training_job(
        receipt, base_checkpoint="HuggingFaceVLA/smolvla_libero", output_dir=tmp_path / "run",
        lora=SmolVLALoRAConfig(base_vla_sha256="c" * 64, model_revision="revision", processor_revision="processor-r1"),
    )

    def fake_runner(_command, **_kwargs):
        output = Path(job.output_dir)
        output.mkdir(parents=True)
        _write_peft_bundle(output, base="HuggingFaceVLA/smolvla_libero", revision="revision", processor_revision="wrong-r1")
        return {"returncode": 0, "adapter_checkpoint": str(output / "adapter_model.safetensors")}

    with pytest.raises(ContractError, match="processor revision field contradicts"):
        run_apprentice_training_job(job, runner=fake_runner)


def test_simulated_training_rejects_full_base_weight_key(tmp_path: Path):
    parent = DatasetManifest("arrow_policy_suite.interventions.v1", 1, ("ep",), "source", "a" * 64, "filter")
    request = ApprenticeExportRequest((_row(),), parent, str(tmp_path / "dataset"))
    receipt = ApprenticeExportReceipt(request, str(tmp_path / "dataset"), "b" * 64, 1, None)
    job = build_apprentice_training_job(
        receipt, base_checkpoint="HuggingFaceVLA/smolvla_libero", output_dir=tmp_path / "run",
        lora=SmolVLALoRAConfig(base_vla_sha256="c" * 64, model_revision="revision", processor_revision="processor-r1"),
    )

    def fake_runner(_command, **_kwargs):
        import torch
        from safetensors.torch import save_file
        output = Path(job.output_dir)
        output.mkdir(parents=True)
        _write_peft_bundle(output, base="HuggingFaceVLA/smolvla_libero", revision="revision", processor_revision="processor-r1")
        save_file({"base_model.model.state_proj.weight": torch.zeros((1, 1))}, str(output / "adapter_model.safetensors"))
        return {"returncode": 0, "adapter_checkpoint": str(output / "adapter_model.safetensors")}

    with pytest.raises(ContractError, match="full-base|non-LoRA"):
        run_apprentice_training_job(job, runner=fake_runner)
