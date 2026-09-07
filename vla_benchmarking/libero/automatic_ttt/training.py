"""Exact-equation training utilities for automatic RoboTTT experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .fidelity import (
    ReferenceArtifactManifest,
    RuntimeAttestation,
    capture_runtime_attestation,
    validate_runtime_receipt,
)


@dataclass(frozen=True)
class OptimizerMetadata:
    mode: str
    optimizer: str = "AdamW"
    weight_decay: float = 1e-5
    peak_learning_rate: float = 2e-5
    scheduler: str = "WSD"
    trainable_parameter_rule: str = "sequence_layers_only"
    steps: int = 30000
    betas_eps_clip: Mapping[str, Any] = field(default_factory=dict)
    gradient_clip_norm: float | None = None


PRETRAIN_METADATA = OptimizerMetadata(mode="pretrain")
POSTTRAIN_METADATA = OptimizerMetadata(
    mode="posttrain",
    peak_learning_rate=5e-5,
    scheduler="cosine",
    trainable_parameter_rule="all_parameters",
    steps=20000,
)


class TrainingContractError(ValueError):
    pass


SUPPORTED_VLA_NAMES = ("openvla", "pi05", "smolvla", "ours")


class VLASequenceAdapter(Protocol):
    """Model-facing boundary shared by every supported VLA.

    Adapters own processor/checkpoint loading and the VLA's action head.  This
    package owns the TTT state lifecycle and objective; no adapter is silently
    substituted for another one.
    """

    policy_id: str

    def begin_episode(self, task_description: str, episode_seed: int) -> None: ...

    def act(self, observation: Mapping[str, Any]) -> Sequence[float]: ...

    def reset_fast_state(self) -> Any: ...

    def ingest_teacher_context(self, transitions: Sequence[Any]) -> None: ...

    def ttt_keys_values_queries(self, batch: Any) -> tuple[Any, Any, Any]: ...

    def predict_flow_velocity(self, interpolated_action: Any, tau: Any, batch: Any) -> Any: ...

    def load_checkpoint(self, checkpoint: str) -> None: ...


def sample_flow_matching_tau(batch_size: int, *, device: Any = None, generator: Any = None) -> Any:
    """Sample independent per-chunk tau=0.999*(1-u), u~Beta(1.5,1)."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("flow matching requires optional dependency 'torch'") from exc
    if batch_size <= 0:
        raise TrainingContractError("batch_size must be positive")
    # For Beta(a=1.5,b=1), inverse-CDF sampling is u=r**(1/a).  Using
    # torch.rand keeps an optional Generator effective and gives exactly one
    # independent noise level per action chunk, as required by sequence
    # action forcing.
    r = torch.rand((batch_size,), device=device, generator=generator)
    u = r.pow(2.0 / 3.0)
    return 0.999 * (1.0 - u)


def flow_matching_target(clean_action: Any, noise: Any, tau: Any) -> tuple[Any, Any]:
    """Return A^tau and the flow target A-epsilon from the paper."""
    while getattr(tau, "ndim", 0) < getattr(clean_action, "ndim", 0):
        tau = tau.unsqueeze(-1)
    interpolated = tau * clean_action + (1.0 - tau) * noise
    target_velocity = clean_action - noise
    return interpolated, target_velocity


def masked_flow_matching_loss(predicted: Any, target: Any, action_mask: Any) -> Any:
    """MSE with action supervision only where action_mask is one."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("flow matching requires optional dependency 'torch'") from exc
    if hasattr(action_mask, "to"):
        mask = action_mask.to(device=predicted.device, dtype=predicted.dtype)
    else:
        mask = torch.as_tensor(action_mask, device=predicted.device, dtype=predicted.dtype)
    while mask.ndim < predicted.ndim:
        mask = mask.unsqueeze(-1)
    error = (predicted - target).pow(2) * mask
    denominator = mask.expand_as(error).sum().clamp_min(1.0)
    return error.sum() / denominator


def build_optimizer(module: Any, metadata: OptimizerMetadata) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("training requires optional dependency 'torch'") from exc
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise TrainingContractError(f"no trainable parameters for {metadata.mode}")
    options = dict(metadata.betas_eps_clip)
    clip_norm = options.pop("clip_grad_norm", options.pop("gradient_clip_norm", metadata.gradient_clip_norm))
    allowed = {"betas", "eps", "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"}
    unknown = set(options) - allowed
    if unknown:
        raise TrainingContractError(f"unknown AdamW options (clip must be separate): {sorted(unknown)}")
    optimizer = torch.optim.AdamW(parameters, lr=metadata.peak_learning_rate, weight_decay=metadata.weight_decay, **options)
    optimizer._automatic_ttt_gradient_clip_norm = clip_norm
    optimizer._automatic_ttt_optimizer_metadata = metadata
    return optimizer


def build_scheduler(optimizer: Any, metadata: OptimizerMetadata, *, boundaries: Mapping[str, int] | None = None) -> Any:
    """Build the published scheduler without guessing missing WSD boundaries.

    Post-training's cosine schedule is fully determined by its step count.  The
    paper names a WSD pretraining schedule but does not publish warmup/decay
    boundaries; exact mode therefore requires those values in the artifact
    manifest rather than silently inventing them.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("training requires optional dependency 'torch'") from exc
    if metadata.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=metadata.steps)
    if metadata.scheduler != "WSD":
        raise TrainingContractError(f"unsupported scheduler {metadata.scheduler!r}")
    values = dict(boundaries or {})
    required = ("warmup_steps", "decay_start_step", "decay_end_step")
    if any(key not in values for key in required):
        raise TrainingContractError("WSD boundaries are unresolved; obtain warmup/decay steps from the official artifact")
    warmup, decay_start, decay_end = (int(values[key]) for key in required)
    if not (0 <= warmup <= decay_start < decay_end <= metadata.steps):
        raise TrainingContractError("invalid WSD boundaries")

    def scale(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        if step < decay_start:
            return 1.0
        if step >= decay_end:
            return 0.0
        return 1.0 - (step - decay_start) / (decay_end - decay_start)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def set_parameter_mask(module: Any, mode: str, sequence_prefixes: Sequence[str] = ("ttt",)) -> dict[str, bool]:
    """Apply paper's pretrain/posttrain slow-parameter policy."""
    if mode not in {"pretrain", "posttrain"}:
        raise TrainingContractError("mode must be pretrain or posttrain")
    result: dict[str, bool] = {}
    for name, parameter in module.named_parameters():
        trainable = mode == "posttrain" or any(
            name == prefix
            or name.startswith(prefix + ".")
            or name.startswith(prefix + "_")
            for prefix in sequence_prefixes
        )
        parameter.requires_grad_(trainable)
        result[name] = trainable
    if mode == "pretrain" and not any(result.values()):
        raise TrainingContractError("pretrain mask selected no sequence parameters")
    return result


def masked_action_targets(records: Iterable[Any]) -> tuple[list[tuple[float, ...]], list[float]]:
    """Extract correction targets from either dataset or runtime contracts.

    The runtime collector stores one action per record with ``actor`` and
    ``training_eligible`` fields.  The standalone dataset helper stores the
    paired ``robot_action``/``teacher_action`` representation.  Supporting both
    keeps collection and training decoupled without silently supervising VLA
    actions.
    """
    targets: list[tuple[float, ...]] = []
    masks: list[float] = []
    for record in records:
        if hasattr(record, "teacher_action"):
            target = record.teacher_action
            mask = float(record.action_loss_mask)
        else:
            actor = getattr(record, "actor", None)
            actor_name = getattr(actor, "value", actor)
            target = record.action if actor_name == "arrow_grasp_controller" else None
            mask = float(bool(getattr(record, "training_eligible", False)))
        if target is None or not mask:
            continue
        targets.append(tuple(float(value) for value in target))
        masks.append(mask)
    return targets, masks


@dataclass
class TBPTTState:
    fast_state: Any = None
    segment_index: int = 0


def run_tbptt_raw_segments(
    module: Any,
    tokens: Sequence[Any],
    *,
    segment_length: int,
    state: TBPTTState | None = None,
) -> tuple[list[Any], TBPTTState, list[Any]]:
    """Run the contextual module on raw tokens, retaining all slow paths."""
    if not callable(getattr(module, "forward", None)):
        raise TrainingContractError("exact TTT requires a module forward/sequence hook over raw contextual tokens")
    if segment_length <= 0:
        raise TrainingContractError("segment_length must be positive")
    running = state or TBPTTState()
    outputs: list[Any] = []
    inner_losses: list[Any] = []
    for index, token_batch in enumerate(tokens):
        output, running.fast_state, inner_loss = module(token_batch, running.fast_state)
        outputs.append(output)
        inner_losses.append(inner_loss)
        if (index + 1) % segment_length == 0 and index + 1 < len(tokens):
            running.fast_state = module.detach_fast_state(running.fast_state)
            running.segment_index += 1
    return outputs, running, inner_losses


def run_tbptt_segments(
    module: Any,
    chunks: Sequence[tuple[Any, Any, Any]],
    *,
    segment_length: int,
    state: TBPTTState | None = None,
    create_graph: bool = True,
) -> tuple[list[Any], TBPTTState, list[Any]]:
    """Carry fast values across segments and detach only at boundaries."""
    if segment_length <= 0:
        raise TrainingContractError("segment_length must be positive")
    running = state or TBPTTState()
    outputs: list[Any] = []
    inner_losses: list[Any] = []
    for index, (keys, values, queries) in enumerate(chunks):
        output, running.fast_state, inner_loss = module.ttt_step(
            keys, values, queries, running.fast_state, create_graph=create_graph
        )
        outputs.append(output)
        inner_losses.append(inner_loss)
        is_boundary = (index + 1) % segment_length == 0 and index + 1 < len(chunks)
        if is_boundary:
            running.fast_state = module.detach_fast_state(running.fast_state)
            running.segment_index += 1
    return outputs, running, inner_losses


class RoboTTTTrainer:
    """Contract wrapper preventing unlabelled approximate experiments."""

    def __init__(
        self,
        module: Any,
        artifact: ReferenceArtifactManifest,
        *,
        exact: bool = True,
        runtime_receipt: RuntimeAttestation | None = None,
        runtime_probe: Any | None = None,
    ) -> None:
        self.module = module
        self.artifact = artifact
        self.exact = exact
        self.runtime_receipt = runtime_receipt
        self._optimizer_step_count = 0
        self.optimizer_metadata: OptimizerMetadata | None = None
        if exact:
            # The artifact gate alone proves checkpoint provenance, not that
            # this instantiated module has the published 16-layer wiring.
            artifact.require_exact()
            receipt = runtime_receipt
            if receipt is None:
                probe = runtime_probe
                if probe is None:
                    probe = getattr(module, "runtime_probe", None)
                receipt = capture_runtime_attestation(module, probe=probe)
            self.runtime_receipt = receipt
            validate_runtime_receipt(artifact, receipt, module=module)
        self.run_label = artifact.label(exact, runtime_attested=self.runtime_receipt if exact else None)

    def configure(self, mode: str) -> OptimizerMetadata:
        metadata = PRETRAIN_METADATA if mode == "pretrain" else POSTTRAIN_METADATA if mode == "posttrain" else None
        if metadata is None:
            raise TrainingContractError("mode must be pretrain or posttrain")
        if self.exact:
            validate_runtime_receipt(
                self.artifact,
                self.runtime_receipt,
                module=self.module,
                mode=mode,
                optimizer_metadata={
                    "optimizer": metadata.optimizer,
                    "weight_decay": metadata.weight_decay,
                    "peak_learning_rate": metadata.peak_learning_rate,
                    "scheduler": metadata.scheduler,
                    "trainable_parameter_rule": metadata.trainable_parameter_rule,
                    "steps": metadata.steps,
                },
            )
            artifact_optimizer = self.artifact.fields.get("optimizer_betas_eps_clip")
            if isinstance(artifact_optimizer, Mapping) and isinstance(artifact_optimizer.get(mode), Mapping):
                artifact_optimizer = artifact_optimizer[mode]
            if not isinstance(artifact_optimizer, Mapping):
                raise TrainingContractError("exact mode requires artifact optimizer betas/epsilon/clipping metadata")
            metadata = replace(metadata, betas_eps_clip=dict(artifact_optimizer))
        set_parameter_mask(
            self.module,
            mode,
            (
                "ttt", "sequence", "q_proj", "k_proj", "v_proj", "w0", "b0",
                "register_tokens", "alpha", "log_inner_learning_rate",
            ),
        )
        self.optimizer_metadata = metadata
        return metadata

    @staticmethod
    def validate_dataset(dataset: Any, split_manifest: Mapping[str, Any]) -> None:
        """Run split/leakage checks immediately before constructing an optimizer."""
        validator = getattr(dataset, "validate_against_split", None)
        if not callable(validator):
            raise TrainingContractError("training dataset must expose validate_against_split")
        validator(split_manifest)

    def sequence_loss(
        self,
        chunks: Sequence[Mapping[str, Any]],
        *,
        action_forward: Any,
        segment_length: int,
        create_graph: bool = True,
    ) -> tuple[Any, TBPTTState, dict[str, Any]]:
        """Compute one masked outer flow-matching loss over ordered chunks.

        Each mapping contains ``keys``, ``values``, ``queries``, ``action``
        (clean target), ``noise``, ``tau`` and ``action_loss_mask``.  The
        injected ``action_forward(ttt_output, interpolated_action, tau,
        chunk)`` is the selected VLA's official action head; all sequence
        mechanics remain shared.  Robot
        failures may update the fast state but carry mask zero, while Arrow
        corrections carry mask one.  No optimizer step occurs when a batch has
        no supervised correction tokens.
        """

        if not chunks:
            raise TrainingContractError("sequence batch is empty")
        if not callable(action_forward):
            raise TypeError("action_forward must be callable")
        if any("action_loss_mask" not in chunk for chunk in chunks):
            raise TrainingContractError("each chunk requires action_loss_mask")
        if self.exact:
            if not callable(getattr(self.module, "forward", None)):
                raise TrainingContractError("exact TTT requires a raw-token sequence hook")
            if any("tokens" not in chunk for chunk in chunks):
                raise TrainingContractError(
                    "exact TTT chunks must contain raw contextual tokens; direct K/V/Q bypasses projections/registers"
                )
            outputs, state, inner_losses = run_tbptt_raw_segments(
                self.module, [chunk["tokens"] for chunk in chunks], segment_length=segment_length
            )
        else:
            fast_chunks = [(chunk["keys"], chunk["values"], chunk["queries"]) for chunk in chunks]
            outputs, state, inner_losses = run_tbptt_segments(
                self.module, fast_chunks, segment_length=segment_length, create_graph=create_graph
            )
        numerator = None
        denominator = None
        supervised = 0
        supervised_elements = 0.0
        for chunk, output in zip(chunks, outputs):
            action = chunk.get("action")
            noise = chunk.get("noise")
            tau = chunk.get("tau")
            if action is None or noise is None or tau is None:
                raise TrainingContractError("chunk requires action, noise and tau")
            interpolated, target = flow_matching_target(action, noise, tau)
            # The noisy action A^tau and its per-chunk tau are inputs to the
            # official flow head.  Passing the clean action here would remove
            # sequence action forcing and is a different objective.
            predicted = action_forward(output, interpolated, tau, chunk)
            mask = chunk["action_loss_mask"]
            try:
                supervised += int(float(mask) > 0)
            except (TypeError, ValueError):
                raise TrainingContractError("action_loss_mask must be numeric")
            try:
                import torch
                mask_tensor = mask.to(device=predicted.device, dtype=predicted.dtype) if hasattr(mask, "to") else torch.as_tensor(mask, device=predicted.device, dtype=predicted.dtype)
            except ImportError as exc:  # pragma: no cover
                raise ImportError("training requires optional dependency 'torch'") from exc
            while mask_tensor.ndim < predicted.ndim:
                mask_tensor = mask_tensor.unsqueeze(-1)
            squared_error = (predicted - target).pow(2)
            chunk_numerator = (squared_error * mask_tensor).sum()
            chunk_denominator = mask_tensor.expand_as(squared_error).sum()
            numerator = chunk_numerator if numerator is None else numerator + chunk_numerator
            denominator = chunk_denominator if denominator is None else denominator + chunk_denominator
            supervised_elements += float(chunk_denominator.detach())
        assert numerator is not None and denominator is not None
        # Normalize once over all supervised action elements.  Averaging a
        # separately normalized loss per chunk would dilute a correction by
        # the length of the failed robot prefix.
        total = numerator / denominator.clamp_min(1.0)
        if self.exact:
            try:
                import torch
                intended = [(name, parameter) for name, parameter in self.module.named_parameters() if parameter.requires_grad]
                gradients = torch.autograd.grad(
                    total, [parameter for _name, parameter in intended], retain_graph=True, allow_unused=True
                )
            except ImportError as exc:  # pragma: no cover
                raise ImportError("training requires optional dependency 'torch'") from exc
            missing_gradients = [name for (name, _parameter), gradient in zip(intended, gradients) if gradient is None]
            if missing_gradients:
                raise TrainingContractError(
                    "exact TTT outer loss does not reach intended slow parameters: " + ", ".join(missing_gradients)
                )
        receipt = {
            "chunks": len(chunks),
            "supervised_chunks": supervised,
            "supervised_elements": supervised_elements,
            "inner_loss_mean": sum(float(loss.detach()) for loss in inner_losses) / len(inner_losses),
            "tbptt_segment_length": segment_length,
            "update_order": "K,V inner update then Q application",
        }
        return total, state, receipt

    def train_step(
        self,
        optimizer: Any,
        chunks: Sequence[Mapping[str, Any]],
        *,
        action_forward: Any,
        segment_length: int,
        create_graph: bool = True,
        scheduler: Any | None = None,
        optimizer_metadata: OptimizerMetadata | None = None,
        dataset: Any | None = None,
        split_manifest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one outer AdamW step, refusing empty-supervision batches."""

        if not any(float(chunk.get("action_loss_mask", 0.0)) > 0 for chunk in chunks):
            raise TrainingContractError("refusing optimizer step with no teacher-correction supervision")
        if self.exact and (dataset is None or split_manifest is None):
            raise TrainingContractError(
                "exact optimization requires a sealed dataset and explicit split manifest"
            )
        if dataset is not None:
            if split_manifest is None:
                raise TrainingContractError("split_manifest is required when dataset is supplied")
            self.validate_dataset(dataset, split_manifest)
        if self.exact:
            episodes = getattr(dataset, "episodes", None)
            if not isinstance(episodes, (list, tuple)):
                raise TrainingContractError("exact dataset must expose immutable episodes")
            dataset_keys: set[tuple[str, int, int | None]] = set()
            for episode in episodes:
                for record in getattr(episode, "transitions", ()):
                    dataset_keys.add((str(record.episode_id), int(record.timestep), getattr(record, "action_chunk_index", None)))
            for index, chunk in enumerate(chunks):
                episode_id = chunk.get("episode_id")
                timestep = chunk.get("timestep")
                chunk_index = chunk.get("action_chunk_index")
                if episode_id is None or timestep is None:
                    raise TrainingContractError(
                        f"exact chunk {index} must bind episode_id and timestep before optimizer step"
                    )
                key = (str(episode_id), int(timestep), chunk_index)
                # Standalone dataset records have no action_chunk_index; the
                # timestep remains the immutable chunk identity in that case.
                if key not in dataset_keys and not any(
                    known[:2] == (str(episode_id), int(timestep)) for known in dataset_keys
                ):
                    raise TrainingContractError(
                        f"exact chunk {index} is not present in the validated dataset: {key}"
                    )
        optimizer.zero_grad(set_to_none=True)
        loss, _state, receipt = self.sequence_loss(
            chunks, action_forward=action_forward, segment_length=segment_length, create_graph=create_graph
        )
        loss.backward()
        metadata = optimizer_metadata or self.optimizer_metadata or getattr(optimizer, "_automatic_ttt_optimizer_metadata", None)
        clip_norm = getattr(optimizer, "_automatic_ttt_gradient_clip_norm", None)
        if clip_norm is None and metadata is not None:
            clip_norm = metadata.gradient_clip_norm
        gradient_norm = None
        if clip_norm is not None:
            import torch
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(self.module.parameters(), float(clip_norm)))
        optimizer.step()
        self._optimizer_step_count += 1
        scheduler_step = None
        learning_rate = None
        if scheduler is not None:
            scheduler.step()
            scheduler_step = self._optimizer_step_count
            learning_rate = [float(group["lr"]) for group in optimizer.param_groups]
        return {
            **receipt,
            "outer_loss": float(loss.detach()),
            "run_label": self.run_label,
            "optimizer_step": self._optimizer_step_count,
            "gradient_norm_before_clip": gradient_norm,
            "scheduler": type(scheduler).__name__ if scheduler is not None else None,
            "scheduler_step": scheduler_step,
            "learning_rate": learning_rate,
        }


def train_from_config(*, config: Any, args: Any | None = None) -> dict[str, Any]:
    """CLI backend entry point.

    Model construction is intentionally injected by the host VLA adapter.  A
    command-line invocation without that adapter is therefore a hard failure,
    not a fake run that reports improvement from a randomly initialized TTT
    head.  The function still emits a deterministic contract receipt for
    launchers and CI preflight.
    """

    artifact_path = getattr(getattr(config, "provenance", None), "reference_artifact_manifest", "")
    if not artifact_path:
        raise TrainingContractError("training requires provenance.reference_artifact_manifest")
    artifact = ReferenceArtifactManifest.from_json(artifact_path)
    exact = getattr(config, "fidelity_mode", "exact") == "exact"
    if exact:
        artifact.require_exact()
    return {
        "status": "BLOCKED_NEEDS_MODEL_ADAPTER",
        "requested_fidelity": "exact" if exact else "algorithmic_port",
        "verified_fidelity": "blocked_unverified",
        "run_label": None,
        "config_digest": config.digest(),
        "message": "Provide a model adapter and invoke RoboTTTTrainer with the frozen artifact; no training was launched.",
    }
