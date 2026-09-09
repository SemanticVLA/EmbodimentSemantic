"""Dependency-light residual-policy architecture and training hooks.

This module describes the Editor/Minimal-learned residual contract without
importing torch, LeRobot, LIBERO, or a particular feature extractor.  A native
implementation is injected through ``predictor`` and ``trainer`` callbacks.
The default predictor is deliberately a zero residual; it is a safe shape
checked scaffold, not evidence that a model has been trained.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import ACTION_DIM, ContractError, _safe
from .learning import InterventionRow


RESIDUAL_SCHEMA = "arrow_policy_suite.residual.v1"


def _as_finite_vector(value: Sequence[float], *, name: str, width: int | None = None) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ContractError(f"{name} must be a numeric sequence")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be a numeric sequence") from exc
    if width is not None and len(result) != width:
        raise ContractError(f"{name} must have width {width}, got {len(result)}")
    if any(not math.isfinite(item) for item in result):
        raise ContractError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class ResidualModelConfig:
    """Serializable architecture contract for a residual action head."""

    input_dim: int = 32
    hidden_dim: int = 64
    output_dim: int = ACTION_DIM
    activation: str = "gelu"
    max_abs_residual: float = 1.0
    architecture: str = "mlp"
    output_mode: str = "residual"
    base_vla_sha256: str = ""

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0 or self.output_dim != ACTION_DIM:
            raise ContractError("residual dimensions must be positive and output_dim must be seven")
        if self.activation not in {"gelu", "relu", "identity"}:
            raise ContractError("residual activation must be gelu, relu, or identity")
        if self.architecture != "mlp":
            raise ContractError("only the injectable MLP residual architecture is supported")
        if self.output_mode not in {"residual", "correction_mask"}:
            raise ContractError("output_mode must be residual or correction_mask")
        if not 0.0 < float(self.max_abs_residual) <= 2.0:
            raise ContractError("max_abs_residual must lie in (0, 2]")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_SCHEMA,
            "architecture": self.architecture,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "max_abs_residual": float(self.max_abs_residual),
            "output_mode": self.output_mode,
            "base_vla_sha256": self.base_vla_sha256,
        }


@dataclass(frozen=True)
class ResidualTrainingConfig:
    """Training values passed unchanged to an injected native trainer."""

    seed: int = 1000
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.seed < 0:
            raise ContractError("residual training seed, epochs, and batch_size must be valid")
        if not 0.0 < float(self.learning_rate) or float(self.weight_decay) < 0.0:
            raise ContractError("residual learning_rate must be positive and weight_decay non-negative")


@dataclass(frozen=True)
class ResidualBatch:
    """Feature/target rows with immutable source identity for lineage checks."""

    features: tuple[tuple[float, ...], ...]
    base_actions: tuple[tuple[float, ...], ...]
    teacher_actions: tuple[tuple[float, ...], ...]
    episode_ids: tuple[str, ...]
    observation_digests: tuple[str, ...] = ()
    source_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        count = len(self.features)
        if count == 0:
            raise ContractError("residual batch must contain at least one row")
        if not (len(self.base_actions) == len(self.teacher_actions) == len(self.episode_ids) == count):
            raise ContractError("residual batch fields must have equal row counts")
        width = len(self.features[0])
        if width <= 0:
            raise ContractError("residual features must have a positive width")
        for index, row in enumerate(self.features):
            _as_finite_vector(row, name=f"features[{index}]", width=width)
            _as_finite_vector(self.base_actions[index], name=f"base_actions[{index}]", width=ACTION_DIM)
            _as_finite_vector(self.teacher_actions[index], name=f"teacher_actions[{index}]", width=ACTION_DIM)
            if not self.episode_ids[index]:
                raise ContractError("each residual row requires an episode id")
        if self.observation_digests and len(self.observation_digests) != count:
            raise ContractError("observation_digests must be empty or match row count")

    @property
    def input_dim(self) -> int:
        return len(self.features[0])

    @property
    def rows(self) -> int:
        return len(self.features)

    @property
    def target_residuals(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(self.teacher_actions[row][col] - self.base_actions[row][col] for col in range(ACTION_DIM))
            for row in range(self.rows)
        )

    def lineage(self) -> dict[str, Any]:
        payload = {
            "schema": RESIDUAL_SCHEMA,
            "rows": self.rows,
            "input_dim": self.input_dim,
            "episode_ids": sorted(set(self.episode_ids)),
            "source_manifest_sha256": self.source_manifest_sha256,
            "observation_digests": list(self.observation_digests),
        }
        encoded = json.dumps(_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {**payload, "content_sha256": hashlib.sha256(encoded).hexdigest()}


ResidualPredictor = Callable[[tuple[float, ...]], Sequence[float]]
ResidualFeatureFn = Callable[[InterventionRow], Sequence[float]]


class ResidualTrainer(Protocol):
    def __call__(self, model: "ResidualModel", batch: ResidualBatch, config: ResidualTrainingConfig) -> Any: ...


class ResidualModel:
    """Shape-safe residual model facade around an injected predictor.

    ``predictor`` may be a torch module, JAX function, or a test double.  It is
    called once per row so the contract remains independent of tensor libraries
    and native batching conventions.
    """

    def __init__(self, config: ResidualModelConfig | None = None, *, predictor: ResidualPredictor | None = None) -> None:
        self.config = config or ResidualModelConfig()
        self.predictor = predictor
        self.training_receipt: Any = None

    def _predict_one(self, features: Sequence[float]) -> tuple[float, ...]:
        vector = _as_finite_vector(features, name="features", width=self.config.input_dim)
        width = ACTION_DIM * 2 if self.config.output_mode == "correction_mask" else ACTION_DIM
        raw = self.predictor(vector) if self.predictor is not None else (0.0,) * width
        values = _as_finite_vector(raw, name="residual", width=width)
        residual = values[:ACTION_DIM]
        limit = float(self.config.max_abs_residual)
        return tuple(max(-limit, min(limit, value)) for value in residual)

    def predict_correction(self, features: Sequence[float], base_action: Sequence[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return a teacher-free residual and a per-action gate/mask.

        The mask variant is deliberately usable without an online Arrow
        object: the residual head sees only student features, and the mask is
        applied to the base VLA action at inference time.
        """
        vector = _as_finite_vector(features, name="features", width=self.config.input_dim)
        base = _as_finite_vector(base_action, name="base_action", width=ACTION_DIM)
        width = ACTION_DIM * 2 if self.config.output_mode == "correction_mask" else ACTION_DIM
        raw = self.predictor(vector) if self.predictor is not None else (0.0,) * width
        values = _as_finite_vector(raw, name="residual", width=width)
        residual = values[:ACTION_DIM]
        if self.config.output_mode == "correction_mask":
            # Stable sigmoid without a torch dependency.
            mask = tuple(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value)))) for value in values[ACTION_DIM:])
        else:
            mask = (1.0,) * ACTION_DIM
        limit = float(self.config.max_abs_residual)
        residual = tuple(max(-limit, min(limit, value)) * mask[i] for i, value in enumerate(residual))
        corrected = tuple(max(-1.0, min(1.0, base[i] + residual[i])) for i in range(ACTION_DIM))
        return residual, corrected

    def predict(self, features: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
        if isinstance(features, (str, bytes)):
            raise ContractError("features must be a batch of feature vectors")
        return tuple(self._predict_one(row) for row in features)

    def predict_one(self, features: Sequence[float]) -> tuple[float, ...]:
        return self._predict_one(features)

    def fit(self, batch: ResidualBatch, trainer: ResidualTrainer, *, config: ResidualTrainingConfig | None = None) -> Any:
        if batch.input_dim != self.config.input_dim:
            raise ContractError(f"batch input_dim {batch.input_dim} does not match model input_dim {self.config.input_dim}")
        result = trainer(self, batch, config or ResidualTrainingConfig())
        self.training_receipt = result
        return result

    def architecture_manifest(self) -> dict[str, Any]:
        return self.config.manifest()


def residual_batch_from_rows(
    rows: Sequence[InterventionRow],
    feature_fn: ResidualFeatureFn,
    *,
    source_manifest_sha256: str | None = None,
) -> ResidualBatch:
    """Create a residual batch while retaining source episode/action lineage."""

    if not rows:
        raise ContractError("cannot build a residual batch from zero rows")
    features = tuple(tuple(float(value) for value in feature_fn(row)) for row in rows)
    return ResidualBatch(
        features=features,
        base_actions=tuple(row.base_action for row in rows),
        teacher_actions=tuple(row.teacher_action for row in rows),
        episode_ids=tuple(row.episode_id for row in rows),
        observation_digests=tuple(str(row.metadata.get("observation_digest", "")) for row in rows),
        source_manifest_sha256=source_manifest_sha256,
    )


@dataclass(frozen=True)
class ResidualTrainingReceipt:
    """Auditable wrapper for an injected training result."""

    architecture: Mapping[str, Any]
    data_lineage: Mapping[str, Any]
    config: Mapping[str, Any]
    native_result: Any


def fit_residual_model(
    model: ResidualModel,
    batch: ResidualBatch,
    trainer: ResidualTrainer,
    *,
    config: ResidualTrainingConfig | None = None,
) -> ResidualTrainingReceipt:
    """Run a caller-supplied trainer and retain model/data/config provenance."""

    resolved = config or ResidualTrainingConfig()
    native_result = model.fit(batch, trainer, config=resolved)
    return ResidualTrainingReceipt(
        architecture=model.architecture_manifest(),
        data_lineage=batch.lineage(),
        config=_safe(resolved.__dict__),
        native_result=native_result,
    )


__all__ = [
    "RESIDUAL_SCHEMA", "ResidualBatch", "ResidualModel", "ResidualModelConfig",
    "ResidualConfig", "ResidualFeatureFn", "ResidualPredictor", "ResidualTrainer", "ResidualTrainerHook", "ResidualTrainingConfig",
    "ResidualTrainingReceipt", "fit_residual_model", "residual_batch_from_rows",
]

# Friendly aliases used by launchers that refer to the head/config rather than
# the longer manifest-oriented names.
ResidualConfig = ResidualModelConfig
ResidualTrainerHook = ResidualTrainer
