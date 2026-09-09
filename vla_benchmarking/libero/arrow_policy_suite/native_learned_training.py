"""Optional native residual training for Editor and Minimal-Learned.

This is the shared learned corrector.  It trains only an external head from
offline On-Call transitions; the base VLA is never updated.  Imports of torch
are lazy so policy-suite contracts and collection remain usable on a CPU-only
machine without the training stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import ContractError, _safe, state8
from .learning import DatasetManifest, InterventionRow
from .residual_model import ResidualBatch, ResidualModel, ResidualModelConfig, residual_batch_from_rows


NATIVE_RESIDUAL_SCHEMA = "arrow_policy_suite.native_residual.v1"
FEATURE_DIM = 32


def default_residual_features(row: InterventionRow, *, width: int = FEATURE_DIM) -> tuple[float, ...]:
    """Student-only feature vector: state, base action, progress, and padding.

    Teacher action and episode outcome never enter the feature vector.  This
    makes the same checkpoint valid for Editor and Minimal-Learned at runtime.
    """
    state = state8(row.observation)
    values = tuple(state) + tuple(row.base_action) + (min(1.0, float(row.timestep) / 1200.0),)
    if width < len(values):
        raise ContractError(f"residual feature width {width} is smaller than canonical features {len(values)}")
    return values + (0.0,) * (width - len(values))


@dataclass(frozen=True)
class NativeResidualTrainingConfig:
    seed: int = 1000
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    hidden_dim: int = 64
    output_mode: str = "correction_mask"

    def __post_init__(self) -> None:
        if self.seed < 0 or self.epochs <= 0 or self.batch_size <= 0 or self.hidden_dim <= 0:
            raise ContractError("invalid native residual training configuration")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ContractError("native residual learning rate/weight decay is invalid")
        if self.output_mode not in {"residual", "correction_mask"}:
            raise ContractError("native residual output_mode is invalid")

    @property
    def output_dim(self) -> int:
        return 14 if self.output_mode == "correction_mask" else 7


@dataclass(frozen=True)
class NativeResidualReceipt:
    variant: str
    checkpoint_path: str
    checkpoint_sha256: str
    base_vla_sha256: str
    source_manifest_sha256: str | None
    rows: int
    optimizer_steps: int
    cost: Mapping[str, Any]
    architecture: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return _safe(self.__dict__)


def _require_base_hash(value: str) -> str:
    value = str(value)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise ContractError("native learned training requires a lowercase frozen base VLA SHA-256")
    return value.lower()


def build_residual_batch(rows: Sequence[InterventionRow], *, feature_fn: Callable[[InterventionRow], Sequence[float]] | None = None,
                        source_manifest_sha256: str | None = None) -> ResidualBatch:
    if not rows:
        raise ContractError("native learned training requires at least one transition")
    return residual_batch_from_rows(rows, feature_fn or default_residual_features,
                                    source_manifest_sha256=source_manifest_sha256)


def _module_for(torch: Any, config: NativeResidualTrainingConfig, input_dim: int) -> Any:
    activation = torch.nn.GELU()
    return torch.nn.Sequential(torch.nn.Linear(input_dim, config.hidden_dim), activation,
                               torch.nn.Linear(config.hidden_dim, config.output_dim))


def _save_checkpoint(path: Path, state: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    if path.exists() or Path(str(path) + ".json").exists():
        raise ContractError(f"refusing to overwrite immutable residual checkpoint: {path}")
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise ContractError("torch is required for native learned training") from exc
    buffer = io.BytesIO()
    torch.save(dict(state), buffer)
    data = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    payload = {**_safe(manifest), "checkpoint_sha256": digest}
    Path(str(path) + ".json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return digest


def train_native_residual(
    rows: Sequence[InterventionRow], *, output_path: str | Path, base_vla_sha256: str,
    variant: str, manifest: DatasetManifest | None = None,
    config: NativeResidualTrainingConfig | None = None,
    feature_fn: Callable[[InterventionRow], Sequence[float]] | None = None,
) -> NativeResidualReceipt:
    """Train the shared residual/mask head and write an immutable checkpoint."""
    if variant not in {"arrow_editor", "arrow_minimal_learned"}:
        raise ContractError("native residual variant must be arrow_editor or arrow_minimal_learned")
    frozen_hash = _require_base_hash(base_vla_sha256)
    if manifest is not None and manifest.rows != len(rows):
        raise ContractError("residual rows do not match source dataset manifest")
    batch = build_residual_batch(rows, feature_fn=feature_fn,
                                 source_manifest_sha256=(manifest.content_sha256 if manifest else None))
    resolved = config or NativeResidualTrainingConfig()
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise ContractError("torch is required for native learned training") from exc
    torch.manual_seed(resolved.seed)
    module = _module_for(torch, resolved, batch.input_dim)
    optimizer = torch.optim.AdamW(module.parameters(), lr=resolved.learning_rate, weight_decay=resolved.weight_decay)
    x = torch.tensor(batch.features, dtype=torch.float32)
    target = torch.tensor(batch.target_residuals, dtype=torch.float32)
    # The mask half is trained toward a permissive gate (the residual target
    # is still the only teacher signal).  At runtime the mask decides whether
    # a correction should be applied.
    if resolved.output_mode == "correction_mask":
        target = torch.cat((target, torch.ones((batch.rows, 7), dtype=torch.float32)), dim=1)
    module.train()
    optimizer_steps = 0
    for _epoch in range(resolved.epochs):
        for start in range(0, batch.rows, resolved.batch_size):
            prediction = module(x[start:start + resolved.batch_size])
            loss = torch.nn.functional.mse_loss(prediction, target[start:start + resolved.batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
    path = Path(output_path)
    manifest_payload = {
        "schema": NATIVE_RESIDUAL_SCHEMA, "variant": variant, "base_vla_sha256": frozen_hash,
        "source_manifest_sha256": manifest.content_sha256 if manifest else None,
        "rows": batch.rows, "optimizer_steps": optimizer_steps,
        "architecture": {"input_dim": batch.input_dim, "hidden_dim": resolved.hidden_dim,
                         "output_dim": resolved.output_dim, "output_mode": resolved.output_mode},
        "cost": {"offline_rows": batch.rows, "offline_episodes": len(set(batch.episode_ids)),
                 "optimizer_steps": optimizer_steps, "success_rows": sum(bool(row.success_episode) for row in rows),
                 "failure_rows": sum(not bool(row.success_episode) for row in rows)},
    }
    checkpoint_hash = _save_checkpoint(path, module.state_dict(), manifest_payload)
    return NativeResidualReceipt(variant, str(path), checkpoint_hash, frozen_hash,
                                 manifest_payload["source_manifest_sha256"], batch.rows,
                                 optimizer_steps, manifest_payload["cost"], manifest_payload["architecture"])


def load_native_residual(path: str | Path, *, base_vla_sha256: str) -> ResidualModel:
    """Load a native checkpoint and expose it through the policy-neutral model."""
    frozen_hash = _require_base_hash(base_vla_sha256)
    target = Path(path)
    try:
        metadata = json.loads(Path(str(target) + ".json").read_text(encoding="utf-8"))
        import torch  # type: ignore
        state = torch.load(target, map_location="cpu", weights_only=True)
    except (OSError, json.JSONDecodeError, ImportError, RuntimeError, TypeError) as exc:
        raise ContractError("native residual checkpoint cannot be loaded") from exc
    if metadata.get("base_vla_sha256") != frozen_hash:
        raise ContractError("native residual checkpoint belongs to a different base VLA")
    if hashlib.sha256(target.read_bytes()).hexdigest() != metadata.get("checkpoint_sha256"):
        raise ContractError("native residual checkpoint hash mismatch")
    architecture = metadata.get("architecture", {})
    config = NativeResidualTrainingConfig(hidden_dim=int(architecture["hidden_dim"]),
                                          output_mode=str(architecture["output_mode"]))
    module = _module_for(torch, config, int(architecture["input_dim"]))
    module.load_state_dict(state)
    module.eval()

    def predictor(features: Sequence[float]) -> Sequence[float]:
        with torch.no_grad():
            values = module(torch.tensor([list(features)], dtype=torch.float32))[0]
        return tuple(float(value) for value in values.tolist())

    return ResidualModel(ResidualModelConfig(input_dim=int(architecture["input_dim"]),
                                             hidden_dim=int(architecture["hidden_dim"]),
                                             output_dim=7, output_mode=str(architecture["output_mode"]),
                                             base_vla_sha256=frozen_hash), predictor=predictor)


def _frame_features(frame: Any, base_action: Sequence[float]) -> tuple[float, ...]:
    """Build the same student-only feature schema used offline."""
    # Avoid importing the runtime policy stack here.  The small structural
    # view is sufficient for the shared feature function and contains no
    # teacher action or outcome.
    row = InterventionRow(
        episode_id=str(getattr(frame, "episode_id", None) or "native-eval"),
        task_id=str(getattr(frame, "metadata", {}).get("task_id", "unknown")),
        timestep=int(getattr(frame, "timestep", getattr(frame, "step", 0))),
        observation=getattr(frame, "observation", {}),
        base_action=tuple(float(value) for value in base_action),
        teacher_action=tuple(float(value) for value in base_action),
        success_episode=False,
    )
    return default_residual_features(row)


def build_native_learned_policy(
    policy_id: str,
    *,
    checkpoint_path: str | Path,
    base_vla_sha256: str,
    adapter_action_fn_factory: Callable[[Path, str], Callable[[Any, Sequence[float]], Sequence[float]]] | None = None,
) -> Any:
    """Construct a learned policy only from a verified immutable artifact.

    This is the integration seam for native Legion factories.  It fails closed
    when the checkpoint, sidecar manifest, frozen-base identity, or adapter
    loader is absent; importantly, it never constructs a zero-residual or
    frozen fallback under a learned policy name.

    ``adapter_action_fn_factory`` is intentionally explicit because loading a
    SmolVLA/PEFT adapter is version-specific.  It receives the verified
    checkpoint path and frozen base hash and must return ``(frame, base) ->
    action``.
    """
    normalized = str(policy_id)
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ContractError(f"learned policy checkpoint is missing: {path}")
    frozen_hash = _require_base_hash(base_vla_sha256)
    if normalized == "arrow_apprentice":
        from .apprentice_training import verify_adapter_reload

        verify_adapter_reload(path, base_vla_sha256=frozen_hash)
        if adapter_action_fn_factory is None:
            raise ContractError("Apprentice requires an explicit native adapter action loader")
        action_fn = adapter_action_fn_factory(path, frozen_hash)
        if not callable(action_fn):
            raise ContractError("Apprentice adapter action loader returned a non-callable")
        from .policies import ApprenticePolicy

        return ApprenticePolicy(action_fn=action_fn)
    if normalized not in {"arrow_editor", "arrow_minimal_learned"}:
        raise ContractError(f"unsupported learned policy id: {normalized}")
    model = load_native_residual(path, base_vla_sha256=frozen_hash)

    def corrected(frame: Any, base: Sequence[float]) -> tuple[float, ...]:
        features = _frame_features(frame, base)
        _residual, action = model.predict_correction(features, base)
        return action

    if normalized == "arrow_editor":
        from .policies import EditorPolicy

        def residual(frame: Any, base: Sequence[float]) -> tuple[float, ...]:
            features = _frame_features(frame, base)
            values, _action = model.predict_correction(features, base)
            return values

        return EditorPolicy(residual_fn=residual)
    from .policies import MinimalPolicy

    return MinimalPolicy(variant="learned", learned_fn=corrected)


def train_editor(*args: Any, **kwargs: Any) -> NativeResidualReceipt:
    kwargs["variant"] = "arrow_editor"
    return train_native_residual(*args, **kwargs)


def train_minimal_learned(*args: Any, **kwargs: Any) -> NativeResidualReceipt:
    kwargs["variant"] = "arrow_minimal_learned"
    return train_native_residual(*args, **kwargs)


__all__ = [
    "NATIVE_RESIDUAL_SCHEMA", "FEATURE_DIM", "NativeResidualTrainingConfig", "NativeResidualReceipt",
    "default_residual_features", "build_residual_batch", "train_native_residual", "train_editor",
    "train_minimal_learned", "load_native_residual", "build_native_learned_policy",
]
