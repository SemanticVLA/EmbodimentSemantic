from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import adapter_audit


class _Tensor:
    def __init__(self, shape, nonzero=True):
        self.shape = shape
        self._nonzero = nonzero

    def any(self):
        return self._nonzero

    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result


class _Parameter(_Tensor):
    def __init__(self, shape, requires_grad):
        super().__init__(shape)
        self.requires_grad = requires_grad


class _WrappedPolicy:
    def __init__(self, modules):
        self._modules = modules
        self._parameters = [
            (f"base_model.model.{module}.lora_A.default.weight", _Parameter((16, 32), True))
            for module in modules
        ] + [
            (f"base_model.model.{module}.lora_B.default.weight", _Parameter((32, 16), True))
            for module in modules
        ] + [("base_model.model.model.vlm_with_expert.lm_expert.embed_tokens.weight", _Parameter((8, 8), False))]

    def named_modules(self):
        return [("", self)] + [(f"base_model.model.{module}", object()) for module in self._modules]

    def named_parameters(self):
        return list(self._parameters)


class _Handle:
    def __init__(self, tensors):
        self.tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def keys(self):
        return list(self.tensors)

    def get_tensor(self, key):
        return self.tensors[key]


def _fixture(tmp_path: Path, *, missing_b=False, target=None, nonzero=True):
    adapter = tmp_path / "pretrained_model"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text(
        json.dumps({
            "peft_type": "LORA", "r": 16, "modules_to_save": [],
            "target_modules": target if target is not None else adapter_audit.ACTION_SIDE_TARGET_REGEX,
        }), encoding="utf-8"
    )
    modules = sorted(adapter_audit.REQUIRED_ACTION_MODULES | {
        "model.vlm_with_expert.lm_expert.layers.0.q_proj",
        "model.vlm_with_expert.lm_expert.layers.0.v_proj",
    })
    tensors = {}
    for module in modules:
        key_a = f"base_model.model.{module}.lora_A.default.weight"
        key_b = f"base_model.model.{module}.lora_B.default.weight"
        tensors[key_a] = _Tensor((16, 32))
        if not missing_b or module != "model.state_proj":
            tensors[key_b] = _Tensor((32, 16), nonzero=nonzero)
    return adapter, tensors


def _patch_safetensors(monkeypatch, tensors):
    module = types.ModuleType("safetensors")
    module.safe_open = lambda *args, **kwargs: _Handle(tensors)
    monkeypatch.setitem(sys.modules, "safetensors", module)


def test_action_side_audit_accepts_exact_paired_rank16_checkpoint(tmp_path, monkeypatch):
    adapter, tensors = _fixture(tmp_path)
    _patch_safetensors(monkeypatch, tensors)
    report = adapter_audit.audit_adapter_checkpoint(adapter)
    assert report["rank"] == 16
    assert report["nonzero_lora_b_count"] == 7
    assert report["tensor_count"] == 14


def test_action_side_audit_rejects_target_subset_and_missing_side(tmp_path, monkeypatch):
    adapter, tensors = _fixture(tmp_path, target=["state_proj"])
    _patch_safetensors(monkeypatch, tensors)
    with pytest.raises(ValueError, match="target_modules"):
        adapter_audit.audit_adapter_checkpoint(adapter)

    adapter, tensors = _fixture(tmp_path / "missing", missing_b=True)
    _patch_safetensors(monkeypatch, tensors)
    with pytest.raises(ValueError, match="exactly one LoRA A"):
        adapter_audit.audit_adapter_checkpoint(adapter)


def test_action_side_audit_rejects_zero_lora_b(tmp_path, monkeypatch):
    adapter, tensors = _fixture(tmp_path, nonzero=False)
    _patch_safetensors(monkeypatch, tensors)
    with pytest.raises(ValueError, match="nonzero lora_B"):
        adapter_audit.audit_adapter_checkpoint(adapter)


def test_action_side_audit_rejects_rank_shape_mismatch(tmp_path, monkeypatch):
    adapter, tensors = _fixture(tmp_path)
    tensors["base_model.model.model.state_proj.lora_A.default.weight"] = _Tensor((15, 32))
    _patch_safetensors(monkeypatch, tensors)
    with pytest.raises(ValueError, match="rank/shape"):
        adapter_audit.audit_adapter_checkpoint(adapter)


def test_action_side_audit_accepts_non_square_lora_projection(tmp_path, monkeypatch):
    adapter, tensors = _fixture(tmp_path)
    tensors["base_model.model.model.state_proj.lora_A.default.weight"] = _Tensor((16, 31))
    tensors["base_model.model.model.state_proj.lora_B.default.weight"] = _Tensor((64, 16))
    _patch_safetensors(monkeypatch, tensors)
    report = adapter_audit.audit_adapter_checkpoint(adapter)
    assert report["tensor_shapes"]["base_model.model.model.state_proj.lora_A.default.weight"] == [16, 31]
    assert report["tensor_shapes"]["base_model.model.model.state_proj.lora_B.default.weight"] == [64, 16]


def test_action_side_audit_rejects_lora_b_rank_mismatch(tmp_path, monkeypatch):
    adapter, tensors = _fixture(tmp_path)
    tensors["base_model.model.model.state_proj.lora_B.default.weight"] = _Tensor((64, 15))
    _patch_safetensors(monkeypatch, tensors)
    with pytest.raises(ValueError, match="rank/shape"):
        adapter_audit.audit_adapter_checkpoint(adapter)


def test_action_side_audit_rejects_prohibited_vision_lora_key(tmp_path, monkeypatch):
    adapter, tensors = _fixture(tmp_path)
    tensors["base_model.model.vision_tower.lora_A.default.weight"] = _Tensor((16, 32))
    _patch_safetensors(monkeypatch, tensors)
    with pytest.raises(ValueError, match="non-action-side"):
        adapter_audit.audit_adapter_checkpoint(adapter)


def test_action_side_audit_detects_assets_added_after_initial_audit(tmp_path, monkeypatch):
    adapter, tensors = _fixture(tmp_path)
    _patch_safetensors(monkeypatch, tensors)
    initial = adapter_audit.audit_adapter_checkpoint(adapter)
    (adapter / "adapter_audit.json").write_text(json.dumps(initial), encoding="utf-8")
    (adapter / "tokenizer").mkdir()
    (adapter / "tokenizer" / "tokenizer.json").write_text("sealed tokenizer", encoding="utf-8")
    final = adapter_audit.audit_adapter_checkpoint(adapter)
    assert final["checkpoint_tree_sha256"] != initial["checkpoint_tree_sha256"]
    assert final["checkpoint_inventory"] != initial["checkpoint_inventory"]


def test_live_expected_inventory_records_complete_module_and_parameter_set():
    modules = sorted(adapter_audit.REQUIRED_ACTION_MODULES | {
        "model.vlm_with_expert.lm_expert.layers.0.q_proj",
        "model.vlm_with_expert.lm_expert.layers.0.v_proj",
    })
    expected = adapter_audit._build_expected_inventory_from_model(_WrappedPolicy(modules), base_policy="/sealed/base")
    assert expected["matched_module_names"] == modules
    assert expected["trainable_parameter_count"] == 2 * len(modules)
    assert expected["total_parameter_count"] == 2 * len(modules) + 1
    assert expected["inventory_sha256"] == adapter_audit.inventory_sha256(expected)


def test_live_expected_inventory_rejects_non_action_trainables():
    modules = sorted(adapter_audit.REQUIRED_ACTION_MODULES | {
        "model.vlm_with_expert.lm_expert.layers.0.q_proj",
        "model.vlm_with_expert.lm_expert.layers.0.v_proj",
    })
    policy = _WrappedPolicy(modules)
    policy._parameters.append(("base_model.model.vision_tower.projection.weight", _Parameter((4, 4), True)))
    with pytest.raises(ValueError, match="unexpected trainable non-LoRA"):
        adapter_audit._build_expected_inventory_from_model(policy)


def test_load_and_wrap_binds_exact_local_base_to_policy_config(tmp_path, monkeypatch):
    base = tmp_path / "base-policy"
    base.mkdir()
    loaded = []

    class _Config:
        pretrained_path = None

    class _PeftConfig:
        peft_type = "LORA"
        r = 16
        target_modules = adapter_audit.ACTION_SIDE_TARGET_REGEX
        modules_to_save = []

    class _Policy:
        @classmethod
        def from_pretrained(cls, pretrained_name_or_path, local_files_only=True):
            assert local_files_only is True
            assert pretrained_name_or_path == str(base.resolve())
            policy = cls()
            policy.config = _Config()
            loaded.append(policy)
            return policy

        def wrap_with_peft(self, *, peft_cli_overrides):
            assert self.config.pretrained_path == str(base.resolve())
            self.peft_config = {"default": _PeftConfig()}
            return self

    lerobot = types.ModuleType("lerobot")
    policies = types.ModuleType("lerobot.policies")
    smolvla = types.ModuleType("lerobot.policies.smolvla")
    smolvla.SmolVLAPolicy = _Policy
    monkeypatch.setitem(sys.modules, "peft", types.ModuleType("peft"))
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.policies", policies)
    monkeypatch.setitem(sys.modules, "lerobot.policies.smolvla", smolvla)

    wrapped = adapter_audit._load_and_wrap_pinned_smolvla(base)

    assert wrapped is loaded[0]
    assert loaded[0].config.pretrained_path == str(base.resolve())


def test_live_expected_inventory_round_trip_is_sealed(tmp_path):
    modules = sorted(adapter_audit.REQUIRED_ACTION_MODULES | {
        "model.vlm_with_expert.lm_expert.layers.0.q_proj",
        "model.vlm_with_expert.lm_expert.layers.0.v_proj",
    })
    value = adapter_audit._build_expected_inventory_from_model(_WrappedPolicy(modules), base_policy="/sealed/base")
    value["base_policy_revision"] = adapter_audit.PINNED_BASE_POLICY_REVISION
    value["effective_peft_config"] = {
        "peft_type": "LORA", "r": 16,
        "target_modules": adapter_audit.ACTION_SIDE_TARGET_REGEX, "modules_to_save": [],
    }
    value["inventory_sha256"] = adapter_audit.inventory_sha256(value)
    path = tmp_path / "expected_adapter_inventory.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert adapter_audit.load_expected_inventory(path) == value
    value["matched_module_names"] = value["matched_module_names"][:-1]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        adapter_audit.load_expected_inventory(path)


def test_checkpoint_audit_rejects_strict_lm_expert_subset(tmp_path, monkeypatch):
    adapter, tensors = _fixture(tmp_path)
    _patch_safetensors(monkeypatch, tensors)
    modules = sorted(adapter_audit.REQUIRED_ACTION_MODULES | {
        "model.vlm_with_expert.lm_expert.layers.0.q_proj",
        "model.vlm_with_expert.lm_expert.layers.0.v_proj",
        "model.vlm_with_expert.lm_expert.layers.1.q_proj",
        "model.vlm_with_expert.lm_expert.layers.1.v_proj",
    })
    expected = {
        "schema_version": 1,
        "base_policy_revision": adapter_audit.PINNED_BASE_POLICY_REVISION,
        "target_regex": adapter_audit.ACTION_SIDE_TARGET_REGEX,
        "peft_type": "LORA", "rank": 16, "modules_to_save": [],
        "effective_peft_config": {
            "peft_type": "LORA", "r": 16,
            "target_modules": adapter_audit.ACTION_SIDE_TARGET_REGEX, "modules_to_save": [],
        },
        "matched_module_names": modules,
        "trainable_parameter_names": sorted(
            f"{module}.lora_{side}.default.weight" for module in modules for side in ("A", "B")
        ),
        "trainable_parameter_count": 2 * len(modules),
    }
    expected["inventory_sha256"] = adapter_audit.inventory_sha256(expected)
    with pytest.raises(ValueError, match="module set"):
        adapter_audit.audit_adapter_checkpoint(adapter, expected_inventory=expected)
