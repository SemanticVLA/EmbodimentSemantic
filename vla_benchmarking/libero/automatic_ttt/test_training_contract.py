from __future__ import annotations

import hashlib

import pytest

from .dataset import TransitionRecord as TrainingTransition
from .fidelity import ExactFidelityError, REQUIRED_ARTIFACT_FIELDS, ReferenceArtifactManifest
from .model import TTTKVBConfig, build_ttt_kvb_module
from .training import RoboTTTTrainer, sample_flow_matching_tau, run_tbptt_segments


torch = pytest.importorskip("torch")


def _artifact(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    fields = {name: "resolved" for name in REQUIRED_ARTIFACT_FIELDS if name not in {"checkpoint_uri", "checkpoint_sha256"}}
    return checkpoint, ReferenceArtifactManifest(
        checkpoint_uri=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        code_commit="a" * 40,
        fields=fields,
    )


def test_exact_gate_verifies_local_checkpoint(tmp_path):
    checkpoint, artifact = _artifact(tmp_path)
    artifact.require_exact()
    bad = ReferenceArtifactManifest(
        checkpoint_uri=str(checkpoint), checkpoint_sha256="0" * 64,
        code_commit="a" * 40, fields=artifact.fields,
    )
    with pytest.raises(ExactFidelityError):
        bad.require_exact()


def test_exact_gate_verifies_sharded_checkpoint_directory(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "part-0.bin").write_bytes(b"part one")
    (checkpoint / "part-1.bin").write_bytes(b"part two")
    digest = hashlib.sha256()
    for item in sorted(checkpoint.rglob("*")):
        if item.is_file():
            relative = item.relative_to(checkpoint).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(item.read_bytes())
    _checkpoint, base = _artifact(tmp_path)
    artifact = ReferenceArtifactManifest(
        checkpoint_uri=str(checkpoint),
        checkpoint_sha256=digest.hexdigest(),
        code_commit=base.code_commit,
        fields=base.fields,
    )
    artifact.require_exact()
    (checkpoint / "part-1.bin").write_bytes(b"tampered")
    with pytest.raises(ExactFidelityError, match="hash mismatch"):
        artifact.require_exact()


def test_remote_exact_checkpoint_requires_host_digest_receipt():
    fields = {name: "resolved" for name in REQUIRED_ARTIFACT_FIELDS if name not in {"checkpoint_uri", "checkpoint_sha256"}}
    artifact = ReferenceArtifactManifest(
        checkpoint_uri="hf://org/private-checkpoint",
        checkpoint_sha256="a" * 64,
        code_commit="a" * 40,
        fields=fields,
    )
    with pytest.raises(ExactFidelityError, match="remote exact checkpoints"):
        artifact.require_exact()
    verified = ReferenceArtifactManifest(
        checkpoint_uri=artifact.checkpoint_uri,
        checkpoint_sha256=artifact.checkpoint_sha256,
        code_commit=artifact.code_commit,
        fields={**fields, "remote_checkpoint_verification": True},
    )
    verified.require_exact()


def test_pretrain_mask_keeps_w0_and_registers_trainable(tmp_path):
    _checkpoint, artifact = _artifact(tmp_path)
    module = build_ttt_kvb_module(TTTKVBConfig(4, 8))
    # The standalone module is intentionally an algorithmic port; only a
    # full 16-layer adapter may request exact mode.
    trainer = RoboTTTTrainer(module, artifact, exact=False)
    trainer.configure("pretrain")
    assert module.w0_1.requires_grad and module.w0_2.requires_grad
    assert module.register_tokens.requires_grad


def test_exact_trainer_rejects_single_layer_kernel(tmp_path):
    _checkpoint, artifact = _artifact(tmp_path)
    module = build_ttt_kvb_module(TTTKVBConfig(4, 8))
    with pytest.raises(ExactFidelityError, match="forward pass"):
        RoboTTTTrainer(module, artifact, exact=True)


def test_tbptt_detached_state_can_update_again():
    module = build_ttt_kvb_module(TTTKVBConfig(4, 8))
    x = torch.randn(2, 4)
    outputs, state, _ = run_tbptt_segments(module, [(x, x, x), (x, x, x), (x, x, x)], segment_length=2)
    assert len(outputs) == 3 and state.segment_index == 1


def test_register_tokens_affect_forward_output():
    module = build_ttt_kvb_module(TTTKVBConfig(4, 8))
    x = torch.randn(2, 4)
    # TTT inference intentionally needs autograd for the inner K/V update;
    # callers must not wrap the module in torch.no_grad().
    with torch.enable_grad():
        first, _state, _loss = module(x)
        with torch.no_grad():
            module.register_tokens.add_(1.0)
        second, _state, _loss = module(x)
    assert not torch.equal(first.detach(), second.detach())


def test_batched_fast_state_isolated_per_episode():
    module = build_ttt_kvb_module(TTTKVBConfig(4, 8))
    torch.manual_seed(4)
    keys = torch.randn(2, 3, 4)
    queries = torch.randn(2, 3, 4)
    values = torch.randn(2, 3, 4)
    first, _state, _loss = module.ttt_step(keys, values, queries)
    changed = values.clone()
    changed[1] += 100.0
    second, _state, _loss = module.ttt_step(keys, changed, queries)
    assert torch.equal(first[0], second[0])
    assert not torch.equal(first[1], second[1])


def test_fast_update_is_invariant_to_batch_duplication():
    torch.manual_seed(5)
    single = build_ttt_kvb_module(TTTKVBConfig(4, 8))
    duplicate = build_ttt_kvb_module(TTTKVBConfig(4, 8))
    duplicate.load_state_dict(single.state_dict())
    keys = torch.randn(1, 3, 4)
    values = torch.randn(1, 3, 4)
    queries = torch.randn(1, 3, 4)
    one, _state, _loss = single.ttt_step(keys, values, queries)
    two, _state, _loss = duplicate.ttt_step(keys.expand(2, -1, -1), values.expand(2, -1, -1), queries.expand(2, -1, -1))
    assert torch.allclose(one[0], two[0], atol=1e-6, rtol=1e-6)


def test_flow_head_receives_noisy_action(tmp_path):
    _checkpoint, artifact = _artifact(tmp_path)
    module = build_ttt_kvb_module(TTTKVBConfig(4, 8))
    trainer = RoboTTTTrainer(module, artifact, exact=False)
    trainer.configure("posttrain")
    x = torch.ones(2, 4)
    seen = []
    chunks = [{"keys": x, "values": x, "queries": x, "action": x, "noise": torch.zeros_like(x), "tau": torch.tensor(.25), "action_loss_mask": 1.0}]

    def head(_output, interpolated, _tau, _chunk):
        seen.append(interpolated.detach().clone())
        return interpolated

    trainer.sequence_loss(chunks, action_forward=head, segment_length=1)
    assert seen and torch.allclose(seen[0], torch.full_like(x, .25))


def test_exact_train_step_requires_sealed_dataset_and_chunk_binding():
    trainer = RoboTTTTrainer.__new__(RoboTTTTrainer)
    trainer.exact = True
    chunks = [{"action_loss_mask": 1.0}]
    with pytest.raises(ValueError, match="sealed dataset"):
        trainer.train_step(None, chunks, action_forward=lambda *args: None, segment_length=1)


def test_malformed_teacher_target_rejected():
    with pytest.raises(ValueError):
        TrainingTransition("e", 0, {}, None, None, 1.0, 1.0, "teacher_correction", 0, 1)
