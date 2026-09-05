from __future__ import annotations

import pytest
import torch
from torch import nn

from vla_benchmarking.arrow_finetuned_vla.language_free_arrow_student.bridge import ArrowConditioningBridge
from vla_benchmarking.arrow_finetuned_vla.language_free_arrow_student.contracts import StudentBatch, StudentOutput
from vla_benchmarking.arrow_finetuned_vla.language_free_arrow_student.flow import flow_matching_target
from vla_benchmarking.arrow_finetuned_vla.language_free_arrow_student.student import ArrowVisualStudent
from vla_benchmarking.arrow_finetuned_vla.language_free_arrow_student.teacher import make_teacher_conditioning_targets
from vla_benchmarking.arrow_finetuned_vla.language_free_arrow_student.training import (
    build_stage_scheduler,
    clip_stage_gradients,
    compute_batch_loss,
    compute_stage_loss,
)
from vla_benchmarking.arrow_finetuned_vla.language_free_arrow_student.config import StageName


class ZeroExpert(nn.Module):
    def forward(self, noisy_actions, timestep, conditioning):
        return torch.zeros_like(noisy_actions)


def _student_output() -> StudentOutput:
    model = ArrowVisualStudent(ArrowConditioningBridge(bridge_layers=1, expert_layers=1), ZeroExpert())
    return model(
        torch.randn(2, 64, 960),
        torch.randn(2, 32),
        torch.randn(2, 3, 7),
        torch.rand(2),
    )


def test_stage_b_matches_masked_flow_target() -> None:
    output = _student_output()
    actions = torch.randn(2, 3, 7)
    noise = torch.randn_like(actions)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    expected = ((flow_matching_target(noise, actions) ** 2) * mask[..., None]).sum() / mask.sum() / 7
    actual = compute_stage_loss(StageName.B_ACTION_BRIDGE, output, actions=actions, noise=noise, action_mask=mask)
    assert actual == pytest.approx(float(expected))


def test_stage_a_requires_teacher_outputs() -> None:
    with pytest.raises(ValueError, match="requires teacher"):
        compute_stage_loss(StageName.A_DISTILL_BRIDGE, _student_output())


def test_batch_wrapper_uses_student_batch_actions() -> None:
    output = _student_output()
    actions = torch.randn(2, 3, 7)
    noise = torch.randn_like(actions)
    batch = StudentBatch(torch.randn(2, 64, 960), torch.randn(2, 32), actions=actions)
    loss = compute_batch_loss(StageName.B_ACTION_BRIDGE, output, batch, noise=noise)
    assert torch.isfinite(loss)


def test_stage_scheduler_warms_and_decays() -> None:
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 1e-4}])
    scheduler = build_stage_scheduler(optimizer, StageName.A_DISTILL_BRIDGE)
    initial = scheduler.get_last_lr()[0]
    optimizer.step()
    scheduler.step()
    warm = scheduler.get_last_lr()[0]
    assert initial == pytest.approx(1e-4 / 250)
    assert warm > initial


def test_clip_stage_gradients_requires_live_gradients() -> None:
    parameter = nn.Parameter(torch.ones(()))
    model = nn.Linear(1, 1)
    model.weight.grad = torch.full_like(model.weight, 10.0)
    before = clip_stage_gradients(model, max_norm=1.0)
    assert float(before) > 1.0
    assert float(model.weight.grad.norm()) <= 1.0 + 1e-6
