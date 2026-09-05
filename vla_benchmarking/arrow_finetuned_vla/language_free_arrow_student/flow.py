"""Flow-matching and action-loss helpers for the SmolVLA action expert."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def flow_matching_target(noise: Tensor, actions: Tensor) -> Tensor:
    """Return the linear flow target used by SmolVLA: noise minus actions."""

    if noise.shape != actions.shape:
        raise ValueError("noise and actions must have the same shape")
    return noise - actions


def masked_mse_loss(prediction: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Mean squared error with optional batch/time mask and safe empty handling.

    Masked denominators count every selected action element, including the
    action dimension, so a selected seven-dimensional zero-error/one-error
    vector contributes its ordinary mean rather than seven times its mean.
    """

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    loss = (prediction - target).square()
    if mask is None:
        return loss.mean()
    expanded = mask
    while expanded.ndim < loss.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.to(dtype=loss.dtype, device=loss.device).expand_as(loss)
    denom = expanded.sum().clamp_min(1.0)
    return (loss * expanded).sum() / denom


def teacher_velocity_loss(
    student_velocity: Tensor, teacher_velocity: Tensor, action_mask: Optional[Tensor] = None
) -> Tensor:
    """Distill the full teacher's flow velocity under identical noise/timestep."""

    return masked_mse_loss(student_velocity, teacher_velocity, action_mask)


def sample_flow_pair(actions: Tensor, *, timestep: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """Sample noise, timestep and interpolated noisy action for flow training."""

    if actions.ndim < 2:
        raise ValueError("actions must include a batch dimension")
    noise = torch.randn_like(actions)
    if timestep is None:
        # Match the pinned SmolVLA schedule: Beta(1.5, 1), clipped away from
        # both endpoints to keep the flow interpolation well-conditioned.
        timestep = torch.distributions.Beta(torch.tensor(1.5, device=actions.device), torch.tensor(1.0, device=actions.device)).sample((actions.shape[0],))
        timestep = (timestep * 0.999 + 0.001).to(dtype=actions.dtype)
    if timestep.ndim != 1 or timestep.shape[0] != actions.shape[0]:
        raise ValueError("timestep must have shape [batch]")
    view_shape = (actions.shape[0],) + (1,) * (actions.ndim - 1)
    t = timestep.reshape(view_shape)
    noisy = t * noise + (1.0 - t) * actions
    return noisy, timestep, noise
