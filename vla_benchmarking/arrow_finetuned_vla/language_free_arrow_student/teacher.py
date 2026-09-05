"""Teacher-target extraction and feature distillation losses."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import ConditioningCache, TeacherConditioningTargets


def rms_normalize(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Normalize each feature vector by its root mean square."""

    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True).clamp_min(eps**2))


def _broadcast_rope_factors(keys: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    half = keys.shape[-1] // 2
    if cos.shape[-1] == keys.shape[-1]:
        cos = cos[..., :half]
        sin = sin[..., :half]
    if cos.shape[-1] != half or sin.shape[-1] != half:
        raise ValueError("cos/sin must have half or full head dimension")
    # Common Transformers layout is [sequence, half_dim]. Convert it to
    # [1, sequence, 1, half_dim] for [batch, sequence, heads, dim] keys.
    if keys.ndim == 4 and cos.ndim == 2 and cos.shape[0] == keys.shape[1]:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    elif keys.ndim == 4 and cos.ndim == 3 and cos.shape[:2] == keys.shape[:2]:
        # Batch-aware factors: [batch, sequence, half_dim].
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
    elif keys.ndim == 4 and cos.ndim == 3 and cos.shape[0] == keys.shape[1]:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    elif keys.ndim == 4 and cos.ndim == 3 and cos.shape[0] == 1 and cos.shape[1] == keys.shape[1]:
        # A position-id vector is commonly expanded to [1, sequence, half].
        # Keep sequence on axis 1 and add the head broadcast axis; treating it
        # as a generic leading batch would incorrectly produce [1,1,T,half].
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
    else:
        while cos.ndim < keys.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
    return cos, sin


def apply_rope(keys: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply standard even/odd rotary embedding exactly once."""

    if keys.shape[-1] % 2:
        raise ValueError("RoPE head dimension must be even")
    cos, sin = _broadcast_rope_factors(keys, cos, sin)
    half = keys.shape[-1] // 2
    output_dtype = keys.dtype
    first, second = keys[..., :half].float(), keys[..., half:].float()
    return torch.cat((first * cos.float() - second * sin.float(), first * sin.float() + second * cos.float()), dim=-1).to(output_dtype)


def inverse_rope(keys: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Undo the standard even/odd rotary embedding on key vectors.

    ``keys`` may be ``[B, T, H, D]`` or any shape whose final dimensions are
    broadcast-compatible with ``cos`` and ``sin``.  The returned keys are in
    the unrotated coordinate system used for bridge feature distillation.
    """

    if keys.shape[-1] % 2:
        raise ValueError("RoPE head dimension must be even")
    cos, sin = _broadcast_rope_factors(keys, cos, sin)
    half = keys.shape[-1] // 2
    # Keep the inverse result in FP32. Teacher K/V often arrive as BF16 and
    # inverse rotations are a numerically sensitive subtraction; targets are
    # converted back only at the student loss boundary if required.
    first, second = keys[..., :half].float(), keys[..., half:].float()
    return torch.cat((first * cos.float() + second * sin.float(), -first * sin.float() + second * cos.float()), dim=-1)


def _pool_valid(values: Tensor, valid: Tensor, slots: int) -> Tensor:
    if values.ndim != 4 or valid.ndim != 2:
        raise ValueError("values must be [B, T, H, D] and valid must be [B, T]")
    if values.shape[:2] != valid.shape:
        raise ValueError("valid mask does not match teacher text dimensions")
    pooled: list[Tensor] = []
    for sample, sample_valid in zip(values, valid):
        sample_values = sample[sample_valid]
        if sample_values.shape[0] == 0:
            raise ValueError("each teacher sample must contain at least one valid text token")
        # Pool along sequence while retaining [heads, head_dim].
        sequence_first = sample_values.permute(1, 2, 0).reshape(1, -1, sample_values.shape[0])
        reduced = F.adaptive_avg_pool1d(sequence_first, slots)
        pooled.append(reduced.reshape(sample_values.shape[1], sample_values.shape[2], slots).permute(2, 0, 1))
    return torch.stack(pooled, dim=0)


def pool_teacher_text_kv(
    keys: Tensor,
    values: Tensor,
    valid_text_mask: Tensor,
    *,
    slots: int = 8,
    rope_cos: Optional[Tensor] = None,
    rope_sin: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Pool valid teacher text K/V rows into exactly eight visual-goal rows.

    Keys are inverse-rotated before pooling. Values are not rotated. If RoPE
    factors are omitted, keys are assumed to already be unrotated. The helper
    does not pool padding and supports a different valid text length per batch
    sample.
    """

    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("keys and values must have matching shape [B, T, H, D]")
    if slots != 8:
        raise ValueError("the arrow bridge requires exactly eight goal slots")
    if rope_cos is not None or rope_sin is not None:
        if rope_cos is None or rope_sin is None:
            raise ValueError("rope_cos and rope_sin must be supplied together")
        keys = inverse_rope(keys, rope_cos, rope_sin)
    return _pool_valid(keys, valid_text_mask, slots), _pool_valid(values, valid_text_mask, slots)


def _rope_rows(factor: Tensor, start: int, length: int, total: int) -> Tensor:
    """Select row factors from a full-sequence or already sliced RoPE table."""

    if factor.ndim == 2:
        if factor.shape[0] == total:
            return factor[start : start + length]
        if factor.shape[0] == length:
            return factor
    elif factor.ndim >= 3:
        # Transformers commonly use [1, sequence, 1, half_dim].
        sequence_axis = 1 if factor.shape[1] >= total else 0
        if factor.shape[sequence_axis] == total:
            index = [slice(None)] * factor.ndim
            index[sequence_axis] = slice(start, start + length)
            return factor[tuple(index)]
        if factor.shape[sequence_axis] == length:
            return factor
    raise ValueError("RoPE factors must provide rows for the teacher sequence")


def make_teacher_conditioning_targets(
    teacher_keys: Tensor,
    teacher_values: Tensor,
    valid_text_mask: Tensor,
    *,
    image_tokens: int = 64,
    state_index: Optional[int] = None,
    rope_cos: Optional[Tensor] = None,
    rope_sin: Optional[Tensor] = None,
) -> TeacherConditioningTargets:
    """Align teacher layer caches to the student's 64+8+1 prefix.

    Teacher tensors have shape ``[B, layers, sequence, heads, head_dim]`` and
    contain image rows, padded text rows, then a state row. Valid text rows
    are inverse-RoPE'd and pooled; when RoPE factors are supplied, image and
    state key rows are inverse-RoPE'd as well. The returned targets are
    detached because the old teacher is frozen.
    """

    if teacher_keys.shape != teacher_values.shape or teacher_keys.ndim != 5:
        raise ValueError("teacher K/V must have shape [B, layers, sequence, heads, head_dim]")
    if valid_text_mask.ndim != 2 or valid_text_mask.shape[0] != teacher_keys.shape[0]:
        raise ValueError("valid_text_mask must have shape [B, text_sequence]")
    text_tokens = valid_text_mask.shape[1]
    expected_state = image_tokens + text_tokens
    state_index = expected_state if state_index is None else state_index
    if image_tokens != 64 or state_index < expected_state or state_index >= teacher_keys.shape[2]:
        raise ValueError("teacher sequence does not contain the required image/state rows")
    if image_tokens + text_tokens > teacher_keys.shape[2]:
        raise ValueError("teacher text span exceeds teacher sequence")

    scene = teacher_keys[:, :, :image_tokens]
    scene_values = teacher_values[:, :, :image_tokens]
    text_keys = teacher_keys[:, :, image_tokens : image_tokens + text_tokens]
    text_values = teacher_values[:, :, image_tokens : image_tokens + text_tokens]
    state = teacher_keys[:, :, state_index : state_index + 1]
    state_values = teacher_values[:, :, state_index : state_index + 1]

    if rope_cos is not None or rope_sin is not None:
        if rope_cos is None or rope_sin is None:
            raise ValueError("rope_cos and rope_sin must be supplied together")
        # All teacher key rows are brought back to one unrotated coordinate
        # system. This is required because the student bridge emits raw K/V;
        # the expert adapter owns the single forward RoPE application later.
        scene_rope_cos = _rope_rows(rope_cos, 0, image_tokens, teacher_keys.shape[2])
        scene_rope_sin = _rope_rows(rope_sin, 0, image_tokens, teacher_keys.shape[2])
        state_rope_cos = _rope_rows(rope_cos, state_index, 1, teacher_keys.shape[2])
        state_rope_sin = _rope_rows(rope_sin, state_index, 1, teacher_keys.shape[2])
        scene = torch.stack(
            [
                inverse_rope(scene[:, layer], scene_rope_cos, scene_rope_sin)
                for layer in range(teacher_keys.shape[1])
            ],
            dim=1,
        )
        state = torch.stack(
            [
                inverse_rope(state[:, layer], state_rope_cos, state_rope_sin)
                for layer in range(teacher_keys.shape[1])
            ],
            dim=1,
        )
        text_rope_cos = _rope_rows(rope_cos, image_tokens, text_tokens, teacher_keys.shape[2])
        text_rope_sin = _rope_rows(rope_sin, image_tokens, text_tokens, teacher_keys.shape[2])
    else:
        text_rope_cos = text_rope_sin = None

    target_keys: list[Tensor] = []
    target_values: list[Tensor] = []
    for layer in range(teacher_keys.shape[1]):
        pooled_keys, pooled_values = pool_teacher_text_kv(
            text_keys[:, layer],
            text_values[:, layer],
            valid_text_mask,
            rope_cos=text_rope_cos,
            rope_sin=text_rope_sin,
        )
        target_keys.append(torch.cat((scene[:, layer], pooled_keys, state[:, layer]), dim=1).detach())
        target_values.append(torch.cat((scene_values[:, layer], pooled_values, state_values[:, layer]), dim=1).detach())

    batch = teacher_keys.shape[0]
    device = teacher_keys.device
    all_valid = torch.ones((batch, 73), dtype=torch.bool, device=device)
    positions = torch.arange(73, dtype=torch.long, device=device)
    cache = ConditioningCache(
        keys=tuple(target_keys),
        values=tuple(target_values),
        attention_mask=all_valid,
        position_ids=positions,
        keys_are_rotated=False,
    )
    return TeacherConditioningTargets(
        conditioning=cache,
        scene_mask=all_valid[:, :64],
        goal_mask=all_valid[:, 64:72],
        state_mask=all_valid[:, 72:73],
    )


def _masked_feature_loss(student: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if student.shape != target.shape:
        raise ValueError("student and teacher features must have identical shape")
    expanded = mask
    while expanded.ndim < student.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.to(dtype=student.dtype)
    expanded = expanded.expand_as(student)
    denom = expanded.sum().clamp_min(1.0)
    student_norm = rms_normalize(student.float())
    target_norm = rms_normalize(target.float())
    smooth = F.smooth_l1_loss(student_norm, target_norm, reduction="none")
    cosine = 1.0 - F.cosine_similarity(student_norm, target_norm, dim=-1).unsqueeze(-1)
    return ((smooth + 0.1 * cosine) * expanded).sum() / denom


def conditioning_feature_loss(
    student: ConditioningCache,
    teacher: TeacherConditioningTargets | ConditioningCache,
    *,
    skip_layer0_goal: bool = True,
) -> Tensor:
    """Compute RMS-normalized SmoothL1 + 0.1 cosine K/V distillation loss."""

    targets = teacher.conditioning if isinstance(teacher, TeacherConditioningTargets) else teacher
    if student.num_layers != targets.num_layers:
        raise ValueError("student and teacher must have the same expert layer count")
    if student.num_tokens != targets.num_tokens:
        raise ValueError("student and teacher must have the same prefix length")
    grouped: dict[str, list[Tensor]] = {"scene": [], "goal": [], "state": []}
    masks = (
        targets.attention_mask[:, :64],
        targets.attention_mask[:, 64:72],
        targets.attention_mask[:, 72:73],
    )
    for layer, (student_key, target_key, student_value, target_value) in enumerate(
        zip(student.keys, targets.keys, student.values, targets.values)
    ):
        for student_part, target_part in (
            (student_key, target_key),
            (student_value, target_value),
        ):
            grouped["scene"].append(_masked_feature_loss(student_part[:, :64], target_part[:, :64], masks[0]))
            if not (skip_layer0_goal and layer == 0):
                grouped["goal"].append(_masked_feature_loss(student_part[:, 64:72], target_part[:, 64:72], masks[1]))
            grouped["state"].append(_masked_feature_loss(student_part[:, 72:73], target_part[:, 72:73], masks[2]))
    group_means = [torch.stack(values).mean() for values in grouped.values() if values]
    return torch.stack(group_means).mean()
