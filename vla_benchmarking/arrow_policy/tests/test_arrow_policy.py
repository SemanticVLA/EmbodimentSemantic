from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from vla_benchmarking.arrow_policy.bridge import ArrowConditioningBridge, build_prefix_attention_mask
from vla_benchmarking.arrow_policy.config import (
    STAGE_SCHEDULE,
    StageName,
    apply_stage_trainability,
    build_stage_optimizer,
)
from vla_benchmarking.arrow_policy.flow import flow_matching_target, masked_mse_loss, sample_flow_pair
from vla_benchmarking.arrow_policy.routing import route_expert_cache
from vla_benchmarking.arrow_policy.student import ArrowVisualStudent
from vla_benchmarking.arrow_policy.teacher import (
    apply_rope,
    conditioning_feature_loss,
    inverse_rope,
    make_teacher_conditioning_targets,
    pool_teacher_text_kv,
)
from vla_benchmarking.arrow_policy.arrow_lerobot_impl import _PairedDataset, _resolve_normalizer_path, build_parser
from vla_benchmarking.arrow_policy.lerobot_integration import AtomicCheckpointStore


def tiny_bridge(layers: int = 2) -> ArrowConditioningBridge:
    return ArrowConditioningBridge(bridge_layers=1, expert_layers=layers)


def test_bridge_emits_fixed_prefix_and_mask_contract() -> None:
    bridge = tiny_bridge()
    assert isinstance(bridge.final_norm, nn.LayerNorm)
    output = bridge(torch.randn(2, 64, 960), torch.randn(2, 32))
    assert output.prefix_tokens.shape == (2, 73, 480)
    assert output.conditioning.position_ids.tolist() == list(range(73))
    assert all(item.shape == (2, 73, 5, 64) for item in output.conditioning.keys)
    mask = build_prefix_attention_mask()
    assert not mask[0, 64]  # scene reads goals
    assert not mask[64, 0]  # goals read scene
    assert mask[0, 72]  # scene cannot read state
    assert not mask[72].any()  # state reads all prefix rows


def test_teacher_pooling_excludes_padding_and_skips_layer_zero_goal() -> None:
    base = torch.randn(2, 4, 73, 5, 64)
    values = torch.randn_like(base)
    valid = torch.tensor([[True, True, True, False, False, False, False, False], [True, True, False, False, False, False, False, False]])
    altered = base.clone()
    altered[:, :, 67:72] += 10000.0
    first = make_teacher_conditioning_targets(base, values, valid).conditioning.keys
    second = make_teacher_conditioning_targets(altered, values, valid).conditioning.keys
    for left, right in zip(first, second):
        assert torch.allclose(left[:, :64], right[:, :64])
        assert torch.allclose(left[:, 64:72], right[:, 64:72])
    student = tiny_bridge(layers=4)(torch.randn(2, 64, 960), torch.randn(2, 32)).conditioning
    teacher = make_teacher_conditioning_targets(base, values, valid)
    # Layer 0's goal contribution is omitted, while other groups remain valid.
    loss = conditioning_feature_loss(student, teacher)
    assert torch.isfinite(loss)


def test_teacher_scene_and_state_keys_are_inverse_rotated() -> None:
    raw = torch.randn(1, 2, 73, 5, 64)
    values = torch.randn_like(raw)
    valid = torch.ones(1, 8, dtype=torch.bool)
    angles = torch.arange(73, dtype=torch.float32)[:, None] / 10
    cos = torch.cos(angles).expand(73, 32)
    sin = torch.sin(angles).expand(73, 32)
    rotated = raw.clone()
    for layer in range(2):
        rotated[:, layer] = apply_rope(raw[:, layer], cos, sin)
    targets = make_teacher_conditioning_targets(rotated, values, valid, rope_cos=cos, rope_sin=sin)
    assert torch.allclose(targets.conditioning.keys[0][:, :64], raw[:, 0, :64], atol=1e-5)
    assert torch.allclose(targets.conditioning.keys[0][:, 72:73], raw[:, 0, 72:73], atol=1e-5)


def test_batched_rope_factors_and_fp32_inverse_support() -> None:
    raw = torch.randn(2, 73, 5, 64, dtype=torch.float16)
    angles = torch.arange(73, dtype=torch.float32)[None, :, None] / 10
    cos = torch.cos(angles).expand(2, 73, 32)
    sin = torch.sin(angles).expand(2, 73, 32)
    rotated = apply_rope(raw, cos, sin)
    recovered = inverse_rope(rotated, cos, sin)
    assert recovered.dtype == torch.float32
    assert torch.allclose(recovered.float(), raw.float(), atol=2e-3, rtol=2e-3)


def test_bfloat_teacher_targets_keep_inverse_rotations_in_fp32() -> None:
    raw = torch.randn(1, 1, 73, 5, 64, dtype=torch.bfloat16)
    values = torch.randn_like(raw)
    valid = torch.ones(1, 8, dtype=torch.bool)
    angles = torch.arange(73, dtype=torch.float32)[:, None] / 10
    cos = torch.cos(angles).expand(73, 32)
    sin = torch.sin(angles).expand(73, 32)
    rotated = raw.clone()
    rotated[:, 0] = apply_rope(raw[:, 0], cos, sin)
    targets = make_teacher_conditioning_targets(rotated, values, valid, rope_cos=cos, rope_sin=sin)
    assert targets.conditioning.keys[0].dtype == torch.float32


def test_rope_is_applied_once_by_routing() -> None:
    cache = tiny_bridge()(torch.randn(1, 64, 960), torch.randn(1, 32)).conditioning
    positions = torch.arange(73)
    angles = positions[:, None].float() / 10
    cos = torch.cos(angles).expand(73, 32)
    sin = torch.sin(angles).expand(73, 32)
    routed = route_expert_cache(cache, rope_cos=cos, rope_sin=sin)
    assert routed[0].keys_are_rotated
    assert routed[0].attention_mode == "shared_self_attention"
    assert routed[1].attention_mode == "cross_attention"
    rotated_cache = cache.__class__(
        keys=tuple(item.key for item in routed), values=cache.values,
        attention_mask=cache.attention_mask, position_ids=cache.position_ids,
        keys_are_rotated=True,
    )
    with pytest.raises(ValueError, match="twice"):
        route_expert_cache(rotated_cache, rope_cos=cos, rope_sin=sin)


def test_student_inference_contract_has_no_text_argument() -> None:
    signature = inspect.signature(ArrowConditioningBridge.forward)
    assert "text" not in signature.parameters
    bridge = tiny_bridge()
    with pytest.raises(TypeError):
        bridge(torch.randn(1, 64, 960), torch.randn(1, 32), "text")  # type: ignore[call-arg]


class LiveCacheExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, noisy_actions, timestep, conditioning):
        return conditioning[0].key.mean(dim=(1, 2, 3), keepdim=True) * self.weight + noisy_actions[..., :1] * 0


def test_bridge_gets_gradient_through_frozen_expert() -> None:
    student = ArrowVisualStudent(tiny_bridge(layers=2), LiveCacheExpert())
    for parameter in student.action_expert.parameters():
        parameter.requires_grad = False
    actions = torch.randn(2, 3, 7)
    loss = student(torch.randn(2, 64, 960), torch.randn(2, 32), actions, torch.rand(2)).velocity.mean()
    loss.backward()
    assert student.bridge.goal_queries.grad is not None
    assert student.bridge.goal_queries.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in student.action_expert.parameters())


def test_flow_loss_respects_action_mask() -> None:
    noise = torch.tensor([[[1.0], [1.0]], [[1.0], [1.0]]])
    actions = torch.zeros_like(noise)
    assert torch.equal(flow_matching_target(noise, actions), noise)
    prediction = torch.zeros_like(noise)
    mask = torch.tensor([[True, False], [False, False]])
    assert masked_mse_loss(prediction, noise, mask) == pytest.approx(1.0)
    assert masked_mse_loss(torch.zeros(1, 2, 7), torch.ones(1, 2, 7), torch.tensor([[True, False]])) == pytest.approx(1.0)


def test_flow_timestep_uses_clipped_beta_schedule() -> None:
    actions = torch.zeros(4096, 2, 7)
    _, timesteps, _ = sample_flow_pair(actions)
    assert float(timesteps.min()) >= 0.001
    assert float(timesteps.max()) <= 1.0
    assert float(timesteps.mean()) > 0.57  # Beta(1.5, 1) mean is 0.6 before clipping


class Inventory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in ("vision_encoder", "visual_connector", "bridge", "action_expert", "state_projection", "action_in_projection", "action_out_projection", "time_projection"):
            setattr(self, name, nn.Linear(2, 2))


def test_stage_inventories_match_schedule() -> None:
    model = Inventory()
    apply_stage_trainability(model, StageName.A_DISTILL_BRIDGE)
    assert {name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad} == {"bridge"}
    apply_stage_trainability(model, StageName.C_JOINT_ACTION)
    assert {name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad} == {
        "bridge", "action_expert", "state_projection", "action_in_projection", "action_out_projection", "time_projection"
    }
    assert [stage.updates for stage in STAGE_SCHEDULE] == [5000, 5000, 10000, 10000]


class NestedInventory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Linear(2, 2)
        self.visual_connector = nn.Linear(2, 2)
        self.bridge = nn.ModuleDict({
            "state_projection": nn.Linear(2, 2),
            "state_to_width": nn.Linear(2, 2),
            "goal": nn.Linear(2, 2),
        })
        self.action_expert = nn.Linear(2, 2)
        self.state_projection = nn.Linear(2, 2)
        self.action_in_projection = nn.Linear(2, 2)
        self.action_out_projection = nn.Linear(2, 2)
        self.time_projection = nn.Linear(2, 2)


def test_stage_freezes_retained_nested_state_projection_and_groups_lrs() -> None:
    model = NestedInventory()
    apply_stage_trainability(model, StageName.A_DISTILL_BRIDGE)
    assert not model.bridge["state_projection"].weight.requires_grad
    assert model.bridge["state_to_width"].weight.requires_grad
    apply_stage_trainability(model, StageName.C_JOINT_ACTION)
    assert model.bridge["state_projection"].weight.requires_grad

    optimizer = build_stage_optimizer(model, StageName.D_FULL_STUDENT)
    parameter_lrs = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_lrs[id(parameter)] = group["lr"]
    names = dict((id(parameter), name) for name, parameter in model.named_parameters())
    assert parameter_lrs[id(model.vision_encoder.weight)] == pytest.approx(1e-6)
    assert parameter_lrs[id(model.action_expert.weight)] == pytest.approx(1e-5)
    assert parameter_lrs[id(model.bridge["state_projection"].weight)] == pytest.approx(1e-5)
    assert parameter_lrs[id(model.bridge["state_to_width"].weight)] == pytest.approx(5e-5)


def test_strict_joint_stage_rejects_partial_student_inventory() -> None:
    with pytest.raises(ValueError, match="requires retained student components"):
        build_stage_optimizer(ArrowVisualStudent(tiny_bridge(), LiveCacheExpert()), StageName.D_FULL_STUDENT)


def test_legion_entrypoint_is_importable_without_lerobot() -> None:
    parser = build_parser()
    assert parser.parse_args([
        "--stage", "A_distill_bridge", "--run-root", "/run", "--output-root", "/run",
        "--checkpoint-root", "/run/checkpoints", "--dataset-root", "/data",
        "--teacher-checkpoint", "/teacher", "--initial-checkpoint", "/initial",
        "--libero-config-path", "/config", "--resume",
    ]).stage == "A_distill_bridge"


def test_atomic_checkpoint_store_writes_resume_pointer(tmp_path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    store = AtomicCheckpointStore(tmp_path)
    checkpoint = store.save(model, optimizer, scheduler, stage=StageName.A_DISTILL_BRIDGE, update=2, seed=1000)
    assert (checkpoint / "manifest.json").is_file()
    assert (tmp_path / "latest.json").is_file()
    restored = store.load_latest(nn.Linear(2, 2), torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=1e-3), scheduler)
    assert restored["resumed"] is True


class _MappingDataset(torch.utils.data.Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


def _paired_frame(frame_index=0, *, action=1.0):
    return {
        "task": "pick",
        "episode_index": torch.tensor([2, 3]),
        "frame_index": torch.tensor(frame_index),
        "timestamp": torch.tensor(0.05),
        "observation.state": torch.ones(8),
        "action": torch.full((50, 7), action),
        "observation.images.image": torch.zeros(3, 8, 8),
    }


def test_paired_dataset_checks_lineage_and_supervision_but_allows_image_difference():
    arrow = _paired_frame()
    clean = _paired_frame()
    clean["observation.images.image"] = torch.ones(3, 8, 8)
    result = _PairedDataset(_MappingDataset([arrow]), _MappingDataset([clean]))[0]
    assert torch.equal(result["teacher_image"], clean["observation.images.image"])
    bad = _paired_frame()
    bad["action"] = torch.zeros(50, 7)
    with pytest.raises(ValueError, match="raw action"):
        _PairedDataset(_MappingDataset([arrow]), _MappingDataset([bad]))[0]


def test_normalizer_resolution_requires_matching_checkpoint_stats(tmp_path):
    teacher, initial, base = (tmp_path / name for name in ("teacher", "initial", "base"))
    for path in (teacher, initial, base):
        path.mkdir()
    stat_name = "policy_preprocessor_step_5_normalizer_processor.safetensors"
    (teacher / stat_name).write_bytes(b"same")
    (initial / stat_name).write_bytes(b"same")
    selected, provenance = _resolve_normalizer_path(teacher, initial, base)
    assert selected == teacher / stat_name
    assert provenance["source"] == "teacher_and_initial_checkpoint"
    (initial / stat_name).write_bytes(b"different")
    with pytest.raises(ValueError, match="different normalizer"):
        _resolve_normalizer_path(teacher, initial, base)


def test_stage_trainer_keeps_frozen_vision_in_eval_until_stage_d(tmp_path):
    from vla_benchmarking.arrow_policy.lerobot_integration import ArrowStageTrainer

    class ModeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_encoder = nn.Dropout()
            self.visual_connector = nn.Dropout()

    model = ModeModel()
    trainer = ArrowStageTrainer(model, tmp_path, device="cpu")
    trainer._set_stage_modes(StageName.C_JOINT_ACTION)
    assert model.training is True
    assert model.vision_encoder.training is False
    assert model.visual_connector.training is False
    trainer._set_stage_modes(StageName.D_FULL_STUDENT)
    assert model.vision_encoder.training is True
    assert model.visual_connector.training is True
