from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from .adapters import AdapterMetadata
from .fidelity import (
    ExactFidelityError,
    REQUIRED_ARTIFACT_FIELDS,
    ReferenceArtifactManifest,
    capture_runtime_attestation,
    validate_runtime_receipt,
)
from .training import train_from_config


def _artifact(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    fields = {name: "resolved" for name in REQUIRED_ARTIFACT_FIELDS if name not in {"checkpoint_uri", "checkpoint_sha256"}}
    return ReferenceArtifactManifest(
        checkpoint_uri=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        code_commit="a" * 40,
        fields=fields,
    )


def test_manifest_cannot_attest_exact_without_runtime_receipt(tmp_path):
    artifact = _artifact(tmp_path)
    with pytest.raises(ExactFidelityError, match="runtime receipt"):
        artifact.label(True)


def test_train_backend_does_not_claim_exact_without_adapter(tmp_path):
    artifact = _artifact(tmp_path)
    config = SimpleNamespace(
        fidelity_mode="exact",
        provenance=SimpleNamespace(reference_artifact_manifest=str(tmp_path / "artifact.json")),
        digest=lambda: "config-digest",
    )
    artifact.to_json(config.provenance.reference_artifact_manifest)
    result = train_from_config(config=config)
    assert result["requested_fidelity"] == "exact"
    assert result["verified_fidelity"] == "blocked_unverified"
    assert result["run_label"] is None


def test_runtime_receipt_rejects_single_layer_kernel(tmp_path):
    artifact = _artifact(tmp_path)
    receipt = {
        "scope": "single_layer_kernel",
        "ttt_layer_count": 1,
        "fields": {name: "resolved" for name in REQUIRED_ARTIFACT_FIELDS},
        "optimizer_metadata": {},
    }
    with pytest.raises(ExactFidelityError, match="opaque"):
        validate_runtime_receipt(artifact, receipt)


def test_runtime_attestation_requires_verifier_hook_trace(tmp_path):
    torch = pytest.importorskip("torch")

    class Layer(torch.nn.Module):
        def forward(self, value):
            return value + 1

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ttt_layers = torch.nn.ModuleList([Layer() for _ in range(16)])

        def forward(self, value):
            for layer in self.ttt_layers:
                value = layer(value)
            return value

        def runtime_receipt(self):
            return {
                "scope": "full_16_layer_contextual",
                "ttt_layer_count": 16,
                "fields": {name: "resolved" for name in REQUIRED_ARTIFACT_FIELDS},
                "adapter_metadata": {
                    "image_preprocessing": "resolved",
                    "state_layout": "resolved",
                    "orientation_representation": "resolved",
                    "instruction_format": "resolved",
                    "prompt_template": "resolved",
                    "action_chunk_horizon": 1,
                    "denoising_steps": 1,
                    "processor_revision": "resolved",
                    "processor_config_digest": "resolved",
                    "native_action_objective": "flow_matching",
                    "compatibility_key": "resolved",
                    "target_task_finetuned": False,
                },
                "optimizer_metadata": {},
            }

    artifact = _artifact(tmp_path)
    model = Model()
    with pytest.raises(ExactFidelityError, match="forward pass"):
        capture_runtime_attestation(model)
    attestation = capture_runtime_attestation(model, probe=lambda: model(torch.zeros(1, 1)))
    validate_runtime_receipt(artifact, attestation)
    other_model = Model()
    with pytest.raises(ExactFidelityError, match="different module"):
        validate_runtime_receipt(artifact, attestation, module=other_model)


def test_exact_optimizer_receipt_is_bound_to_artifact_settings(tmp_path):
    torch = pytest.importorskip("torch")

    class Layer(torch.nn.Module):
        def forward(self, value):
            return value + 1

    optimizer_settings = {
        "pretrain": {"betas": [0.9, 0.95], "eps": 1e-8, "clip_grad_norm": 1.0},
    }

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ttt_layers = torch.nn.ModuleList([Layer() for _ in range(16)])

        def forward(self, value):
            for layer in self.ttt_layers:
                value = layer(value)
            return value

        def runtime_receipt(self):
            return {
                "scope": "full_16_layer_contextual", "ttt_layer_count": 16,
                "fields": {name: "resolved" for name in REQUIRED_ARTIFACT_FIELDS},
                "adapter_metadata": {name: "resolved" for name in (
                    "image_preprocessing", "state_layout", "orientation_representation", "instruction_format",
                    "prompt_template", "action_chunk_horizon", "denoising_steps", "processor_revision",
                    "processor_config_digest", "native_action_objective", "compatibility_key",
                )} | {"native_action_objective": "flow_matching", "target_task_finetuned": False},
                "optimizer_metadata": {
                    "pretrain": {
                        "optimizer": "AdamW", "weight_decay": 1e-5, "peak_learning_rate": 2e-5,
                        "scheduler": "WSD", "trainable_parameter_rule": "sequence_layers_only", "steps": 30000,
                        "betas_eps_clip": optimizer_settings["pretrain"],
                    }
                },
            }

    artifact = _artifact(tmp_path)
    artifact = ReferenceArtifactManifest(
        checkpoint_uri=artifact.checkpoint_uri, checkpoint_sha256=artifact.checkpoint_sha256,
        code_commit=artifact.code_commit,
        fields={**artifact.fields, "optimizer_betas_eps_clip": optimizer_settings},
    )
    model = Model()
    attestation = capture_runtime_attestation(model, probe=lambda: model(torch.zeros(1, 1)))
    validate_runtime_receipt(artifact, attestation, module=model, mode="pretrain")
    bad_fields = {**artifact.fields, "optimizer_betas_eps_clip": {"pretrain": {**optimizer_settings["pretrain"], "clip_grad_norm": 2.0}}}
    bad_artifact = ReferenceArtifactManifest(
        checkpoint_uri=artifact.checkpoint_uri, checkpoint_sha256=artifact.checkpoint_sha256,
        code_commit=artifact.code_commit, fields=bad_fields,
    )
    with pytest.raises(ExactFidelityError, match="optimizer"):
        validate_runtime_receipt(bad_artifact, attestation, module=model, mode="pretrain")


def test_adapter_compatibility_key_changes_with_processor_semantics():
    base = dict(
        name="openvla", checkpoint_id="ckpt", image_preprocessing={"resize": 224}, state_layout="eef8",
        orientation_representation="axis_angle", instruction_format="text", prompt_template="{instruction}",
        action_chunk_horizon=8, denoising_steps=1, processor_revision="r1", processor_config_digest="d1",
        native_action_objective="continuous_l1",
    )
    first = AdapterMetadata(**base)
    second = AdapterMetadata(**{**base, "prompt_template": "Task: {instruction}"})
    assert first.compatibility_key() != second.compatibility_key()
    with pytest.raises(Exception, match="metadata mismatch"):
        first.assert_compatible(second)
