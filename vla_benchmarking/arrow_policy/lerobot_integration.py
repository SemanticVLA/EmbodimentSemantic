"""Optional integration with the pinned LeRobot SmolVLA implementation.

The tensor bridge in this package is intentionally independent of LeRobot.  This
module is the small, explicit adapter used on a training node: imports of
LeRobot happen only when a checkpoint is loaded, and the rest of the module can
still be imported and tested on a CPU-only workstation.

The adapter mirrors the pinned source at commit
``d656da8ccca5989ff0a2207e81fbfa2c2d5bafb1``.  In particular, the old VLM
cache is rotated once before it is consumed; cross-attention expert layers
apply their existing K/V projections to those routed keys/values, while
shared-self layers consume the cache directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .bridge import ArrowConditioningBridge
from .config import STAGE_SCHEDULE, OptimizerConfig, StageName, apply_stage_trainability, build_stage_optimizer
from .contracts import StudentBatch, StudentOutput, TeacherConditioningTargets
from .flow import sample_flow_pair
from .routing import ExpertLayerConditioning, route_expert_cache
from .teacher import make_teacher_conditioning_targets
from .training import build_stage_scheduler, clip_stage_gradients, compute_stage_loss


PINNED_LEROBOT_COMMIT = "d656da8ccca5989ff0a2207e81fbfa2c2d5bafb1"


def _require_lerobot() -> Any:
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as exc:  # pragma: no cover - exercised on Legion
        raise RuntimeError(
            "LeRobot with the pinned SmolVLA extra is required for checkpoint integration; "
            "install requirements-lora.txt in the Legion environment"
        ) from exc
    return SmolVLAPolicy


def _load_policy_class(path: Path) -> Any:
    policy_cls = _require_lerobot()
    kwargs = {"local_files_only": True}
    try:
        return policy_cls.from_pretrained(str(path), **kwargs)
    except TypeError:
        # Older pinned wrappers accept only the positional path.  The path is
        # still local and resolved, so this cannot silently select a Hub model.
        return policy_cls.from_pretrained(str(path))


def load_pinned_smolvla(
    checkpoint: str | os.PathLike[str],
    *,
    base_policy: str | os.PathLike[str] | None = None,
    device: str = "auto",
) -> Any:
    """Load the local merged SmolVLA checkpoint through LeRobot.

    ``from_pretrained`` is the public constructor in the pinned source.  A
    local checkpoint is required so a run cannot silently switch model
    revisions or download an unrelated model.
    """

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"SmolVLA checkpoint directory does not exist: {path}")
    # A LeRobot checkpoint produced by the existing 029190 runs is a PEFT
    # adapter directory.  Materialize it into a plain policy before copying
    # retained modules into the language-free student.  Full snapshots remain
    # accepted directly.
    adapter_marker = (path / "adapter_config.json").is_file() or (path / "adapter_model.safetensors").is_file()
    if adapter_marker:
        if base_policy is None:
            raise ValueError("an adapter checkpoint requires --base-policy pointing to the pinned local snapshot")
        base_path = Path(base_policy).expanduser().resolve()
        if not base_path.is_dir() or not (base_path / "config.json").is_file():
            raise FileNotFoundError(f"pinned base policy directory is missing config.json: {base_path}")
        policy = _load_policy_class(base_path)
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - compute-node only
            raise RuntimeError("peft is required to merge the supplied adapter checkpoint") from exc
        # Adapter keys are rooted at ``base_model.model.model...``: the first
        # ``model`` is PEFT's wrapper and the second is LeRobot's policy.model.
        # Wrap the policy object itself so the serialized target names resolve;
        # wrapping policy.model would silently miss the sealed modules.
        wrapped = PeftModel.from_pretrained(policy, str(path), is_trainable=False)
        policy = wrapped.merge_and_unload()
    else:
        policy = _load_policy_class(path)
    if device != "auto":
        policy.to(device)
    policy.eval()
    return policy


def _source_make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """Exact pinned LeRobot prefix/suffix attention mask construction."""

    cumsum = torch.cumsum(att_masks, dim=1)
    causal = cumsum[:, None, :] <= cumsum[:, :, None]
    valid = pad_masks[:, None, :] & pad_masks[:, :, None]
    return causal & valid


def _pad_last(x: Tensor, width: int) -> Tensor:
    if x.shape[-1] == width:
        return x
    if x.shape[-1] > width:
        raise ValueError(f"tensor width {x.shape[-1]} exceeds configured width {width}")
    result = torch.zeros(*x.shape[:-1], width, device=x.device, dtype=x.dtype)
    result[..., : x.shape[-1]] = x
    return result


def _rope_factors(position_ids: Tensor, head_dim: int) -> tuple[Tensor, Tensor]:
    if position_ids.ndim == 1:
        position_ids = position_ids[None]
    half = head_dim // 2
    exponents = (2.0 / head_dim) * torch.arange(half, dtype=torch.float32, device=position_ids.device)
    timescale = 10_000.0**exponents
    radians = position_ids.to(torch.float32)[..., None] / timescale
    return torch.cos(radians), torch.sin(radians)


def _apply_source_rope(x: Tensor, positions: Tensor) -> Tensor:
    """Equivalent to SmolVLMWithExpertModel.apply_rope."""

    if x.shape[-1] % 2:
        raise ValueError("expert head dimension must be even")
    cos, sin = _rope_factors(positions, x.shape[-1])
    half = x.shape[-1] // 2
    first, second = x[..., :half].float(), x[..., half:].float()
    return torch.cat((first * cos[..., None, :] - second * sin[..., None, :],
                      second * cos[..., None, :] + first * sin[..., None, :]), dim=-1).to(x.dtype)


def _attention(query: Tensor, key: Tensor, value: Tensor, allow: Tensor) -> Tensor:
    """Eager GQA attention matching the pinned LeRobot attention interface."""

    q_heads, kv_heads = query.shape[2], key.shape[2]
    if q_heads % kv_heads:
        raise ValueError(f"expert query heads {q_heads} are not divisible by KV heads {kv_heads}")
    groups = q_heads // kv_heads
    key = key[:, :, :, None, :].expand(-1, -1, -1, groups, -1).reshape(
        key.shape[0], key.shape[1], q_heads, key.shape[-1]
    )
    value = value[:, :, :, None, :].expand(-1, -1, -1, groups, -1).reshape(
        value.shape[0], value.shape[1], q_heads, value.shape[-1]
    )
    scores = torch.einsum("bshd,bthd->bhst", query.float(), key.float()) * (query.shape[-1] ** -0.5)
    scores = scores.masked_fill(~allow[:, None], torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhst,bthd->bshd", weights, value.float()).to(query.dtype)


class ArrowSmolVLAActionExpert(nn.Module):
    """Run the pinned action expert using a language-free routed prefix."""

    def __init__(self, source_model: nn.Module, *, self_attn_every_n_layers: int = 2) -> None:
        super().__init__()
        self.lm_expert = source_model.vlm_with_expert.lm_expert
        self.action_in_proj = source_model.action_in_proj
        self.action_out_proj = source_model.action_out_proj
        self.action_time_mlp_in = source_model.action_time_mlp_in
        self.action_time_mlp_out = source_model.action_time_mlp_out
        self.self_attn_every_n_layers = self_attn_every_n_layers
        config = source_model.config
        self.min_period = float(getattr(config, "min_period", 4.0))
        self.max_period = float(getattr(config, "max_period", 4.0e4))

    @property
    def action_width(self) -> int:
        return int(self.action_in_proj.out_features)

    @property
    def input_action_dim(self) -> int:
        return int(self.action_in_proj.in_features)

    def _time_embedding(self, timestep: Tensor, width: int) -> Tensor:
        if timestep.ndim != 1:
            raise ValueError("timestep must have shape [batch]")
        # Match pinned ``create_sinusoidal_pos_embedding``: safe float64
        # intermediates on CUDA, followed by the caller's expert-dtype cast.
        fraction = torch.linspace(0.0, 1.0, width // 2, device=timestep.device, dtype=torch.float64)
        periods = self.min_period * (self.max_period / self.min_period) ** fraction
        radians = timestep.to(dtype=torch.float64)[:, None] * (2.0 * torch.pi / periods[None, :])
        return torch.cat((torch.sin(radians), torch.cos(radians)), dim=-1)

    def forward(self, noisy_actions: Tensor, timestep: Tensor, conditioning: tuple[ExpertLayerConditioning, ...]) -> Tensor:
        if noisy_actions.ndim != 3 or noisy_actions.shape[-1] != self.input_action_dim:
            raise ValueError(f"noisy_actions must be [batch, horizon, {self.input_action_dim}]")
        if len(conditioning) != len(self.lm_expert.layers):
            raise ValueError("conditioning layer count does not match the retained action expert")
        action = self.action_in_proj(noisy_actions)
        time = self._time_embedding(timestep, action.shape[-1]).to(dtype=action.dtype)
        time = time[:, None, :].expand_as(action)
        action = self.action_time_mlp_in(torch.cat((action, time), dim=-1))
        action = self.action_time_mlp_out(F.silu(action))
        bsz, horizon, width = action.shape
        prefix_len = conditioning[0].key.shape[1]
        action_positions = torch.arange(prefix_len, prefix_len + horizon, device=action.device).expand(bsz, -1)
        for index, (expert_layer, routed) in enumerate(zip(self.lm_expert.layers, conditioning)):
            normalized = expert_layer.input_layernorm(action)
            q = expert_layer.self_attn.q_proj(normalized).view(bsz, horizon, -1, expert_layer.self_attn.head_dim)
            q_positions = action_positions
            if routed.attention_mode == "cross_attention":
                # Pinned source subtracts the minimum suffix position for its
                # cross-attention query RoPE path.
                q_positions = q_positions - q_positions.min(dim=1, keepdim=True).values
            q = _apply_source_rope(q, q_positions)
            if routed.attention_mode == "shared_self_attention":
                suffix_k = expert_layer.self_attn.k_proj(normalized).view(bsz, horizon, -1, expert_layer.self_attn.head_dim)
                suffix_v = expert_layer.self_attn.v_proj(normalized).view(bsz, horizon, -1, expert_layer.self_attn.head_dim)
                key = torch.cat((routed.key, _apply_source_rope(suffix_k, action_positions)), dim=1)
                value = torch.cat((routed.value, suffix_v), dim=1)
                allow = torch.cat(
                    (routed.attention_mask[:, None].expand(-1, horizon, -1),
                     torch.tril(torch.ones(bsz, horizon, horizon, dtype=torch.bool, device=action.device))), dim=2
                )
            elif routed.attention_mode == "cross_attention":
                raw_key = routed.key.reshape(bsz, prefix_len, -1)
                raw_value = routed.value.reshape(bsz, prefix_len, -1)
                key = expert_layer.self_attn.k_proj(raw_key).view(bsz, prefix_len, -1, expert_layer.self_attn.head_dim)
                value = expert_layer.self_attn.v_proj(raw_value).view(bsz, prefix_len, -1, expert_layer.self_attn.head_dim)
                allow = routed.attention_mask[:, None].expand(-1, horizon, -1)
            else:
                raise ValueError(f"unknown expert attention mode: {routed.attention_mode}")
            # SmolVLM uses 15 query heads (960 values) with a 480-wide expert;
            # the expert output projection performs the 960 -> 480 contraction.
            attended = _attention(q, key, value, allow).reshape(bsz, horizon, -1)
            action = action + expert_layer.self_attn.o_proj(attended)
            residual = action
            action = residual + expert_layer.mlp(expert_layer.post_attention_layernorm(residual))
        return self.action_out_proj(self.lm_expert.norm(action))


class ArrowSmolVLAPolicy(nn.Module):
    """Agentview-only student assembled from a pinned SmolVLA checkpoint."""

    def __init__(self, source_policy: Any, bridge: Optional[ArrowConditioningBridge] = None) -> None:
        super().__init__()
        source_model = source_policy.model
        vlm = source_model.vlm_with_expert.get_vlm_model()
        self.vision_encoder = vlm.vision_model
        self.visual_connector = vlm.connector
        self.bridge = bridge or ArrowConditioningBridge()
        if source_model.state_proj.in_features != self.bridge.state_dim or source_model.state_proj.out_features != self.bridge.image_width:
            raise ValueError("source state projection is incompatible with the pinned arrow bridge")
        self.bridge.state_projection.load_state_dict(source_model.state_proj.state_dict())
        self.action_in_projection = source_model.action_in_proj
        self.action_out_projection = source_model.action_out_proj
        self.time_projection = nn.ModuleList([source_model.action_time_mlp_in, source_model.action_time_mlp_out])
        self.action_expert = ArrowSmolVLAActionExpert(
            source_model,
            self_attn_every_n_layers=int(getattr(source_model.config, "self_attn_every_n_layers", 2)),
        )
        self.self_attn_every_n_layers = self.action_expert.self_attn_every_n_layers
        self.action_dim = int(source_model.config.action_feature.shape[0])
        self.max_action_dim = int(source_model.config.max_action_dim)
        self.image_size = tuple(getattr(source_model.config, "resize_imgs_with_padding", None) or (224, 224))

    def encode_agentview(self, image: Tensor) -> Tensor:
        if image.ndim == 5:
            image = image[:, -1]
        if image.ndim != 4:
            raise ValueError("agentview image must have shape [B,C,H,W] or [B,T,C,H,W]")
        if image.shape[1] not in (1, 3):
            raise ValueError("agentview image must be channel-first")
        image = image.to(device=next(self.vision_encoder.parameters()).device)
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif image.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            image = image.float()
        if float(image.detach().max()) > 1.5:
            image = image / 255.0
        height, width = self.image_size
        ratio = max(image.shape[-1] / width, image.shape[-2] / height)
        resized = F.interpolate(image, size=(max(1, int(image.shape[-2] / ratio)), max(1, int(image.shape[-1] / ratio))), mode="bilinear", align_corners=False)
        resized = F.pad(resized, (max(0, width - resized.shape[-1]), 0, max(0, height - resized.shape[-2]), 0), value=0.0)
        pixel_values = resized * 2.0 - 1.0
        hidden = self.vision_encoder(pixel_values=pixel_values.to(dtype=next(self.vision_encoder.parameters()).dtype), patch_attention_mask=None).last_hidden_state
        hidden = self.visual_connector(hidden)
        return hidden * (hidden.shape[-1] ** 0.5)

    def forward(self, image: Tensor, state: Tensor, noisy_actions: Tensor, timestep: Tensor, *, rope_cos: Optional[Tensor] = None, rope_sin: Optional[Tensor] = None):
        image_tokens = self.encode_agentview(image)
        if image_tokens.shape[1:] != (64, 960):
            raise ValueError(f"pinned agentview connector must emit [B,64,960], got {tuple(image_tokens.shape)}")
        # The pinned SigLIP/VLM weights are bfloat16 while the newly-created
        # bridge is initialized in the default training dtype.  Explicitly
        # align the connector output at this boundary instead of relying on an
        # ambient autocast context (the CPU smoke and Legion job both run with
        # different autocast defaults).
        bridge_dtype = next(self.bridge.parameters()).dtype
        image_tokens = image_tokens.to(dtype=bridge_dtype)
        if state.ndim == 3:
            state = state[:, -1]
        if state.shape[-1] < self.bridge.state_dim:
            state = _pad_last(state, self.bridge.state_dim)
        state = state.to(device=image_tokens.device, dtype=bridge_dtype)
        padded = _pad_last(noisy_actions, self.max_action_dim)
        bridge_output = self.bridge(image_tokens, state)
        if rope_cos is None and rope_sin is None:
            rope_cos, rope_sin = _rope_factors(bridge_output.conditioning.position_ids, self.bridge.head_dim)
        routed = route_expert_cache(
            bridge_output.conditioning,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            self_attn_every_n_layers=self.self_attn_every_n_layers,
        )
        velocity = self.action_expert(padded, timestep, routed)
        return StudentOutput(velocity=velocity[..., : self.action_dim], conditioning=bridge_output.conditioning)


@dataclass(frozen=True)
class TeacherBatch:
    images: list[Tensor]
    image_masks: list[Tensor]
    language_tokens: Tensor
    language_mask: Tensor
    state: Tensor


class SmolVLATeacherAdapter:
    """Extract teacher K/V targets and teacher velocities from old SmolVLA."""

    def __init__(self, source_policy: Any) -> None:
        self.policy = source_policy
        self.model = source_policy.model
        self.expert_layers = len(self.model.vlm_with_expert.lm_expert.layers)

    @torch.no_grad()
    def conditioning_targets(self, batch: TeacherBatch) -> TeacherConditioningTargets:
        model = self.model
        prefix, pad, att = model.embed_prefix(batch.images, batch.image_masks, batch.language_tokens, batch.language_mask, state=batch.state)
        if prefix.shape[1] < 73 or prefix.shape[1] < 64 + batch.language_tokens.shape[1] + 1:
            raise ValueError("teacher prefix is shorter than image + language + state")
        state_index = 64 + batch.language_tokens.shape[1]
        if not bool(pad[:, state_index].all()):
            raise ValueError("teacher state row is not aligned after the 64 image and language rows")
        if prefix.shape[1] > 64 + batch.language_tokens.shape[1] + 1 and bool(pad[:, 64 + batch.language_tokens.shape[1] + 1 :].any()):
            raise ValueError("teacher prefix contains valid rows after the state; alignment would be ambiguous")
        attention = _source_make_att_2d_masks(pad, att)
        positions = torch.cumsum(pad, dim=1) - 1
        _, past = model.vlm_with_expert.forward(
            attention_mask=attention, position_ids=positions, past_key_values=None,
            inputs_embeds=[prefix, None], use_cache=True, fill_kv_cache=True,
        )
        if not isinstance(past, dict) or len(past) != self.expert_layers:
            raise RuntimeError("pinned SmolVLA did not return a complete per-layer teacher cache")
        keys, values = [], []
        for layer in range(self.expert_layers):
            entry = past[layer]
            keys.append(entry["key_states"])
            values.append(entry["value_states"])
        teacher_keys = torch.stack(keys, dim=1)
        teacher_values = torch.stack(values, dim=1)
        cos, sin = _rope_factors(positions, teacher_keys.shape[-1])
        return make_teacher_conditioning_targets(
            teacher_keys, teacher_values,
            batch.language_mask,
            image_tokens=64,
            state_index=state_index,
            rope_cos=cos,
            rope_sin=sin,
        )

    @torch.no_grad()
    def velocity(
        self,
        batch: TeacherBatch,
        noisy_actions: Tensor,
        timestep: Tensor,
        *,
        action_dim: int | None = None,
    ) -> Tensor:
        model = self.model
        max_dim = int(model.config.max_action_dim)
        actions = _pad_last(noisy_actions, max_dim)
        action_dtype = getattr(model.action_in_proj, "weight", actions).dtype
        actions = actions.to(dtype=action_dtype)
        prefix, prefix_pad, prefix_att = model.embed_prefix(batch.images, batch.image_masks, batch.language_tokens, batch.language_mask, state=batch.state)
        suffix, suffix_pad, suffix_att = model.embed_suffix(actions, timestep)
        attention = _source_make_att_2d_masks(torch.cat((prefix_pad, suffix_pad), dim=1), torch.cat((prefix_att, suffix_att), dim=1))
        positions = torch.cumsum(torch.cat((prefix_pad, suffix_pad), dim=1), dim=1) - 1
        outputs_embeds, _ = model.vlm_with_expert.forward(
            attention_mask=attention, position_ids=positions, past_key_values=None,
            inputs_embeds=[prefix, suffix], use_cache=False, fill_kv_cache=False,
        )
        if not isinstance(outputs_embeds, (list, tuple)) or len(outputs_embeds) != 2:
            raise RuntimeError("pinned SmolVLA forward did not return [prefix, suffix] embeddings")
        suffix_out = outputs_embeds[1]
        # Match pinned ``VLAFlowMatching.denoise_step``: upcast the expert
        # output before the action head for numerically stable flow targets.
        suffix_out = suffix_out.to(dtype=torch.float32)
        output = model.action_out_proj(suffix_out[:, -actions.shape[1] :])
        width = noisy_actions.shape[-1] if action_dim is None else action_dim
        return output[:, :, :width]


def student_batch_from_mapping(batch: Mapping[str, Tensor], *, image_key: str = "observation.images.image", state_key: str = "observation.state", action_key: str = "action", action_mask_key: str = "action_is_pad") -> StudentBatch:
    """Convert one LeRobot batch to the language-free student contract."""

    image = batch[image_key]
    state = batch[state_key]
    actions = batch.get(action_key)
    mask = batch.get(action_mask_key)
    if mask is not None:
        mask = ~mask.bool()
    return StudentBatch(image_tokens=image, state=state, actions=actions, action_mask=mask)


class AtomicCheckpointStore:
    """Crash-safe, content-addressed stage checkpoints and resume pointer."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.root / "latest.json"

    @staticmethod
    def _write_atomic(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, *, stage: StageName, update: int, seed: int) -> Path:
        directory = self.root / stage.value / f"update_{update:07d}"
        directory.mkdir(parents=True, exist_ok=True)
        files = {
            "model.pt": model.state_dict(),
            "optimizer.pt": optimizer.state_dict(),
            "scheduler.pt": scheduler.state_dict(),
            "rng.pt": {"torch": torch.get_rng_state(), "python": random.getstate(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        for name, value in files.items():
            temporary = directory / f".{name}.tmp"
            torch.save(value, temporary)
            os.replace(temporary, directory / name)
        manifest = {"stage": stage.value, "update": update, "seed": seed, "files": {}, "format": 1}
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.name != "manifest.json":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest["files"][path.name] = digest
        self._write_atomic(directory / "manifest.json", json.dumps(manifest, sort_keys=True, indent=2).encode())
        self._write_atomic(self.latest_path, json.dumps({"path": str(directory), **{k: manifest[k] for k in ("stage", "update", "seed")}}, sort_keys=True).encode())
        return directory

    def _latest_directory(self) -> tuple[Path, dict[str, Any]] | None:
        if not self.latest_path.is_file():
            return None
        pointer = json.loads(self.latest_path.read_text(encoding="utf-8"))
        directory = Path(pointer["path"]).expanduser().resolve()
        if directory.parent.parent != self.root:
            raise RuntimeError("checkpoint pointer escapes the experiment output folder")
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"checkpoint manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest.get("files", {}).items():
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"checkpoint checksum mismatch: {directory / name}")
        return directory, manifest

    def load_model_latest(self, model: nn.Module, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        """Restore only model weights, for a transition into the next stage."""

        latest = self._latest_directory()
        if latest is None:
            return {"resumed": False, "update": 0, "stage": None}
        directory, manifest = latest
        model.load_state_dict(torch.load(directory / "model.pt", map_location=map_location, weights_only=True))
        return {"resumed": True, "stage": StageName(manifest["stage"]), "update": int(manifest["update"]), "seed": int(manifest["seed"]), "path": str(directory)}

    def load_latest(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        latest = self._latest_directory()
        if latest is None:
            return {"resumed": False, "update": 0, "stage": None}
        directory, manifest = latest
        model.load_state_dict(torch.load(directory / "model.pt", map_location=map_location, weights_only=True))
        optimizer.load_state_dict(torch.load(directory / "optimizer.pt", map_location=map_location, weights_only=True))
        scheduler.load_state_dict(torch.load(directory / "scheduler.pt", map_location=map_location, weights_only=True))
        # RNG state tensors are CPU-owned even when the model is restored onto
        # CUDA.  Loading them directly on the GPU makes torch.set_rng_state
        # reject the byte tensor during resume.
        rng = torch.load(directory / "rng.pt", map_location="cpu", weights_only=False)
        torch.set_rng_state(rng["torch"])
        random.setstate(rng["python"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        return {"resumed": True, "stage": StageName(manifest["stage"]), "update": int(manifest["update"]), "seed": int(manifest["seed"]), "path": str(directory)}


class ArrowStageTrainer:
    """Small resumable trainer around the four explicit experiment stages."""

    def __init__(self, model: nn.Module, output_dir: str | os.PathLike[str], *, seed: int = 1000, save_every: int = 250, optimizer_config: OptimizerConfig = OptimizerConfig(), device: str | torch.device = "cuda") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.seed = seed
        self.save_every = save_every
        self.optimizer_config = optimizer_config
        self.checkpoints = AtomicCheckpointStore(output_dir)

    def _set_stage_modes(self, stage: StageName) -> None:
        """Set train/eval modes independently from ``requires_grad``.

        ``Module.train()`` recursively flips every child, including frozen
        vision/connector modules.  Their frozen A-C path must stay in eval
        mode so dropout and running-stat updates cannot change the teacher
        feature distribution. Stage D deliberately enables train mode again.
        """

        self.model.train()
        vision_training = stage is StageName.D_FULL_STUDENT
        for name in ("vision_encoder", "visual_connector"):
            component = getattr(self.model, name, None)
            if component is not None:
                component.train(vision_training)

    def run_stage(self, stage: StageName, batches: Iterable[Mapping[str, Tensor]], *, teacher: Optional[SmolVLATeacherAdapter] = None, updates: Optional[int] = None, resume: bool = True) -> dict[str, Any]:
        optimizer = build_stage_optimizer(self.model, stage, optimizer_config=self.optimizer_config)
        scheduler = build_stage_scheduler(optimizer, stage, warmup_fraction=self.optimizer_config.warmup_fraction, final_lr_fraction=self.optimizer_config.final_lr_fraction)
        start_update = 0
        if resume:
            latest = self.checkpoints._latest_directory()
            if latest is not None:
                directory, manifest = latest
                checkpoint_stage = StageName(manifest["stage"])
                stage_order = {StageName.A_DISTILL_BRIDGE: 0, StageName.B_ACTION_BRIDGE: 1, StageName.C_JOINT_ACTION: 2, StageName.D_FULL_STUDENT: 3}
                if stage_order[checkpoint_stage] > stage_order[stage]:
                    return {"resumed": True, "stage": checkpoint_stage, "update": int(manifest["update"]), "path": str(directory)}
                if checkpoint_stage is stage:
                    state = self.checkpoints.load_latest(self.model, optimizer, scheduler, map_location=self.device)
                    start_update = state["update"]
                else:
                    self.checkpoints.load_model_latest(self.model, map_location=self.device)
        self._set_stage_modes(stage)
        iterator = iter(batches)
        target_updates = updates if updates is not None else next(item.updates for item in STAGE_SCHEDULE if item.name == stage)
        completed = start_update
        while completed < target_updates:
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                raw = next(iterator)
            image = raw["observation.images.image"].to(self.device)
            state = raw["observation.state"].to(self.device)
            actions = raw["action"].to(self.device)
            valid = raw.get("action_is_pad")
            valid = None if valid is None else (~valid.bool()).to(self.device)
            # Sample in the full 32-wide latent action space, just as the
            # pinned SmolVLA does after padding the seven real controls. The
            # objective is reduced to the seven real dimensions below.
            padded_actions = _pad_last(actions, self.model.max_action_dim)
            noisy_full, timestep, noise_full = sample_flow_pair(padded_actions)
            noisy = noisy_full[..., : actions.shape[-1]]
            noise = noise_full[..., : actions.shape[-1]]
            output = self.model(image, state, noisy_full, timestep)
            teacher_targets = teacher_velocity = None
            if stage is StageName.A_DISTILL_BRIDGE:
                if teacher is None or "teacher" not in raw:
                    raise ValueError("stage A requires a TeacherBatch under batch['teacher']")
                teacher_batch = raw["teacher"]
                teacher_targets = teacher.conditioning_targets(teacher_batch)
                teacher_velocity = teacher.velocity(
                    teacher_batch,
                    noisy_full,
                    timestep,
                    action_dim=actions.shape[-1],
                )
            loss = compute_stage_loss(
                stage,
                output,
                actions=actions,
                noise=noise,
                action_mask=valid,
                teacher_targets=teacher_targets,
                teacher_velocity=teacher_velocity,
            )
            if not torch.isfinite(loss).all():
                diagnostic = {
                    "stage": stage.value,
                    "update": completed + 1,
                    "loss": float(loss.detach().cpu()) if loss.numel() == 1 else None,
                    "error": "non-finite loss; optimizer step was skipped",
                }
                self.checkpoints._write_atomic(
                    self.checkpoints.root / "nonfinite_loss.json",
                    (json.dumps(diagnostic, sort_keys=True, indent=2) + "\n").encode("utf-8"),
                )
                raise FloatingPointError(f"non-finite loss at {stage.value} update {completed + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_stage_gradients(self.model, self.optimizer_config.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            completed += 1
            if completed % self.save_every == 0 or completed == target_updates:
                self.checkpoints.save(self.model, optimizer, scheduler, stage=stage, update=completed, seed=self.seed)
        return {"stage": stage.value, "updates": completed, "output_dir": str(self.checkpoints.root)}

    def run_schedule(
        self,
        batches_by_stage: Mapping[StageName, Iterable[Mapping[str, Tensor]]],
        *,
        teacher: Optional[SmolVLATeacherAdapter] = None,
        resume: bool = True,
    ) -> list[dict[str, Any]]:
        """Run A/B/C/D in order, resuming the newest valid stage checkpoint."""

        results: list[dict[str, Any]] = []
        for stage_config in STAGE_SCHEDULE:
            if stage_config.name not in batches_by_stage:
                raise KeyError(f"missing batch provider for {stage_config.name.value}")
            result = self.run_stage(stage_config.name, batches_by_stage[stage_config.name], teacher=teacher, resume=resume)
            results.append(result)
            target = next(item.updates for item in STAGE_SCHEDULE if item.name == stage_config.name)
            if (
                stage_config.name is StageName.D_FULL_STUDENT
                and result.get("stage") in (StageName.D_FULL_STUDENT, StageName.D_FULL_STUDENT.value)
                and int(result.get("updates", 0)) >= target
            ):
                break
        return results
