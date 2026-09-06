"""Native Octo adapter skeleton with explicit JAX PRNG and action contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal
import hashlib
import json

import numpy as np

from .config import (
    COMMUNITY_CHECKPOINT,
    COMMUNITY_EVAL_CONFIG,
    MATCHED_TRAIN_CONFIG,
    OFFICIAL_BASE_CHECKPOINT,
    CheckpointRef,
    checkpoint_download_patterns,
)
from .contracts import (
    OCTO_ACTION_DIM,
    OCTO_ACTION_HORIZON,
    convert_unnormalized_octo_to_libero_action,
    validate_observation,
)

from vla_benchmarking.libero.evaluation.policy_adapter import (
    PolicyMetadata,
    validate_policy_action,
)


def validate_action_chunk(value: Any) -> Any:
    """Require a batched native Octo chunk shaped exactly ``[1, 4, 7]``."""

    shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
    if shape != (1, OCTO_ACTION_HORIZON, OCTO_ACTION_DIM):
        raise ValueError(f"Octo sample_actions must return [1, 4, 7], got {shape}")
    return value


def _find_model_action_statistics(model: Any) -> tuple[Any, Any, Any] | None:
    """Find Octo's native action statistics without assuming one API version."""

    candidates: list[Any] = []
    for attribute in ("dataset_statistics", "action_statistics", "normalization_statistics"):
        value = getattr(model, attribute, None)
        if value is not None:
            candidates.append(value)
    if isinstance(getattr(model, "config", None), Mapping):
        candidates.append(model.config)

    def visit(value: Any) -> tuple[Any, Any, Any] | None:
        if isinstance(value, Mapping):
            if "mean" in value and "std" in value:
                return value["mean"], value["std"], value.get("mask")
            # Check the known Octo LIBERO key first, then recurse through
            # nested suite/statistics mappings used by different checkpoint
            # serialization versions.
            keys = ("libero_spatial", "action", "actions", "action_stats", "normalization")
            visited: set[int] = set()
            for key in keys:
                if key in value and id(value[key]) not in visited:
                    visited.add(id(value[key]))
                    found = visit(value[key])
                    if found is not None:
                        return found
            for child in value.values():
                if id(child) in visited:
                    continue
                found = visit(child)
                if found is not None:
                    return found
        return None

    for candidate in candidates:
        found = visit(candidate)
        if found is not None:
            return found
    return None


def _canonical_statistics(mean: Any, std: Any, mask: Any | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    if mean_arr.shape == (6,):
        mean_arr = np.concatenate([mean_arr, np.zeros(1, dtype=np.float32)])
    if std_arr.shape == (6,):
        std_arr = np.concatenate([std_arr, np.ones(1, dtype=np.float32)])
    if mean_arr.shape != (7,) or std_arr.shape != (7,):
        raise ValueError("Octo action statistics must have seven values")
    if not np.isfinite(mean_arr).all() or not np.isfinite(std_arr).all() or (std_arr[:6] <= 0).any():
        raise ValueError("Octo action statistics are non-finite or have invalid Gaussian scales")
    expected_mask = np.asarray([True] * 6 + [False], dtype=bool)
    if mask is not None and not np.array_equal(np.asarray(mask, dtype=bool), expected_mask):
        raise ValueError("Octo action statistics mask must preserve Gaussian dims 0-5 and unscaled gripper dim 6")
    return mean_arr, std_arr, expected_mask


def resolve_action_statistics(
    model: Any,
    *,
    action_mean: Any | None,
    action_std: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Derive native stats and reject caller values that disagree with them."""

    native = _find_model_action_statistics(model)
    native_canonical = None
    if native is not None:
        native_canonical = _canonical_statistics(*native)
    caller_canonical = None
    if action_mean is not None or action_std is not None:
        if action_mean is None or action_std is None:
            raise ValueError("action_mean and action_std must be supplied together")
        caller_canonical = _canonical_statistics(action_mean, action_std, None)
    if native_canonical is not None and caller_canonical is not None:
        if not np.allclose(native_canonical[0], caller_canonical[0], rtol=1e-6, atol=1e-6) or not np.allclose(native_canonical[1], caller_canonical[1], rtol=1e-6, atol=1e-6):
            raise ValueError("caller action statistics disagree with loaded Octo model statistics")
        selected = native_canonical
        source = "loaded_model_verified_against_caller"
    elif native_canonical is not None:
        selected = native_canonical
        source = "loaded_model"
    elif caller_canonical is not None:
        selected = caller_canonical
        source = "explicit_caller_no_model_stats"
    else:
        raise ValueError("Octo model exposes no action statistics and caller supplied none")
    return (*selected, source)


@dataclass(frozen=True)
class RolloutExecution:
    """Auditable result for one native four-action query."""

    actions_executed: int
    native_action_horizon: int
    query_index: int
    policy_provenance: dict[str, Any]


class OctoPolicyAdapter:
    """Thin, lazy-loading adapter around ``OctoModel``.

    The adapter accepts an injected model and JAX module for unit tests.  The
    production ``from_pretrained`` constructor imports Octo/JAX only on a
    compute node and never silently falls back to another model.
    """

    def __init__(
        self,
        model: Any,
        jax_module: Any,
        *,
        action_mean: Any | None,
        action_std: Any | None,
        mode: Literal["community_eval", "matched_train"],
        policy_kind: str | None = None,
        artifact_id: str | None = None,
        checkpoint_revision: str | None = None,
        dataset_manifest_sha256: str | None = None,
        runtime_sha256: str | None = None,
        io_sha256: str | None = None,
        checkpoint_sha256: str | None = None,
        checkpoint_tree_sha256: str | None = None,
    ):
        self._model = model
        self._jax = jax_module
        if mode not in ("community_eval", "matched_train"):
            raise ValueError(f"unsupported Octo mode: {mode}")
        config = COMMUNITY_EVAL_CONFIG if mode == "community_eval" else MATCHED_TRAIN_CONFIG
        if policy_kind is not None and policy_kind != config.policy_kind:
            raise ValueError(f"policy_kind {policy_kind!r} does not match explicit mode {mode!r}")
        mean, std, mask, stats_source = resolve_action_statistics(
            model, action_mean=action_mean, action_std=action_std
        )
        self._action_mean = mean
        self._action_std = std
        self._action_mask = mask
        self._mode = mode
        self._policy_kind = str(policy_kind or config.policy_kind)
        self._artifact_id = str(artifact_id or config.checkpoint.repository)
        self._checkpoint_revision = str(checkpoint_revision or config.checkpoint.revision)
        self._stats_source = stats_source
        self._dataset_manifest_sha256 = str(dataset_manifest_sha256) if dataset_manifest_sha256 else None
        self._runtime_sha256 = str(runtime_sha256) if runtime_sha256 else None
        self._io_sha256 = str(io_sha256) if io_sha256 else None
        if checkpoint_sha256 and checkpoint_tree_sha256 and str(checkpoint_sha256) != str(checkpoint_tree_sha256):
            raise ValueError("checkpoint_sha256 and deprecated checkpoint_tree_sha256 disagree")
        self._checkpoint_sha256 = str(checkpoint_sha256 or checkpoint_tree_sha256) if (checkpoint_sha256 or checkpoint_tree_sha256) else None
        self._stats_sha256 = hashlib.sha256(
            json.dumps({"mean": mean.tolist(), "std": std.tolist(), "mask": mask.tolist()}, sort_keys=True).encode()
        ).hexdigest()
        self._rng = None
        self._tasks = None
        self._observation_window = config.observation_window
        self._frame_buffer: list[np.ndarray] = []
        self._timestep_buffer: list[int] = []

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointRef | str,
        *,
        seed: int,
        mode: Literal["community_eval", "matched_train"],
        action_mean: Any | None = None,
        action_std: Any | None = None,
        checkpoint_revision: str | None = None,
        artifact_id: str | None = None,
        dataset_manifest_sha256: str | None = None,
        runtime_sha256: str | None = None,
        io_sha256: str | None = None,
        checkpoint_sha256: str | None = None,
        checkpoint_tree_sha256: str | None = None,
        checkpoint_root: str | Path | None = None,
        checkpoint_step: int | None = None,
    ) -> "OctoPolicyAdapter":
        try:
            import jax
        except ImportError as exc:  # pragma: no cover - GPU runtime boundary
            raise RuntimeError("Octo adapter requires JAX on the selected runtime") from exc
        try:
            from octo.model.octo_model import OctoModel
        except ImportError as exc:  # pragma: no cover - GPU runtime boundary
            raise RuntimeError("Octo adapter requires the pinned octo package") from exc
        if mode not in ("community_eval", "matched_train"):
            raise ValueError(f"unsupported Octo mode: {mode}")
        if isinstance(checkpoint, CheckpointRef):
            expected = COMMUNITY_CHECKPOINT if mode == "community_eval" else OFFICIAL_BASE_CHECKPOINT
            if checkpoint != expected:
                raise ValueError(f"{mode} requires its sealed checkpoint provenance")
            identifier = checkpoint.identifier()
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:  # pragma: no cover - runtime boundary
                raise RuntimeError("Octo Hub loading requires huggingface_hub") from exc
            if checkpoint.step is not None:
                run_root = checkpoint.root_identifier()
                step = int(checkpoint.step)
                local_root = snapshot_download(
                    repo_id=checkpoint.repository,
                    revision=checkpoint.revision,
                    allow_patterns=list(checkpoint_download_patterns(checkpoint)),
                )
                model = OctoModel.load_pretrained(
                    str(Path(local_root) / run_root) if run_root else local_root,
                    step=step,
                )
            else:
                local_root = snapshot_download(
                    repo_id=checkpoint.repository,
                    revision=checkpoint.revision,
                )
                model = OctoModel.load_pretrained(local_root)
            artifact_id = artifact_id or identifier
            revision = checkpoint_revision or checkpoint.revision
        else:
            identifier = str(checkpoint)
            # A local published snapshot has the same root/step layout as the
            # Hub artifact.  Accept a leaf for compatibility, but always pass
            # the normalized experiment root and step when supplied.
            if checkpoint_root is not None or checkpoint_step is not None:
                root = Path(checkpoint_root or identifier).expanduser().resolve()
                step = int(checkpoint_step) if checkpoint_step is not None else None
                if step is None:
                    raise ValueError("checkpoint_root and checkpoint_step must be supplied together")
                model = OctoModel.load_pretrained(str(root), step=step)
            else:
                model = OctoModel.load_pretrained(identifier)
            artifact_id = artifact_id or identifier
            revision = checkpoint_revision or "runtime-resolved"
        return cls(
            model,
            jax,
            action_mean=action_mean,
            action_std=action_std,
            mode=mode,
            artifact_id=artifact_id,
            checkpoint_revision=revision,
            dataset_manifest_sha256=dataset_manifest_sha256,
            runtime_sha256=runtime_sha256,
            io_sha256=io_sha256,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_tree_sha256=checkpoint_tree_sha256,
        ).reset(episode_seed=seed)

    @property
    def metadata(self) -> PolicyMetadata:
        extra = {
            "observation_window": self._observation_window,
            "uses_language": True,
            "mode": self._mode,
            "stats_source": self._stats_source,
            "action_mask": self._action_mask.tolist(),
            "action_stats_sha256": self._stats_sha256,
            "frame_rotation_owner": "shared_evaluator",
        }
        if self._dataset_manifest_sha256 is not None:
            extra["dataset_manifest_sha256"] = self._dataset_manifest_sha256
        if self._runtime_sha256 is not None:
            extra["runtime_sha256"] = self._runtime_sha256
        if self._io_sha256 is not None:
            extra["io_sha256"] = self._io_sha256
        if self._checkpoint_sha256 is not None:
            extra["checkpoint_sha256"] = self._checkpoint_sha256
        return PolicyMetadata(
            policy_kind=self._policy_kind,
            artifact_id=self._artifact_id,
            checkpoint_revision=self._checkpoint_revision,
            backend="octo_native",
            native_action_horizon=OCTO_ACTION_HORIZON,
            input_resolution=256,
            camera_keys=("image_primary",),
            state_dim=None,
            extra=extra,
        )

    def reset(
        self,
        task_description: str | None = None,
        episode_seed: int | None = None,
        *,
        seed: int | None = None,
        instruction: str | None = None,
    ) -> "OctoPolicyAdapter":
        if episode_seed is None:
            episode_seed = seed
        if task_description is None:
            task_description = instruction
        if not isinstance(episode_seed, int) or isinstance(episode_seed, bool):
            raise TypeError("Octo reset seed must be an integer")
        self._rng = self._jax.random.PRNGKey(episode_seed)
        self._frame_buffer.clear()
        self._timestep_buffer.clear()
        if task_description is not None:
            self.set_instruction(task_description)
        return self

    def set_instruction(self, instruction: str) -> None:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        self._tasks = self._model.create_tasks(texts=[instruction])

    def act(self, observation: Mapping[str, Any]) -> Any:
        if self._rng is None:
            raise RuntimeError("reset must be called before act")
        if self._tasks is None:
            raise RuntimeError("set_instruction must be called before act")
        image = validate_observation(observation)
        self._frame_buffer.append(np.asarray(image).copy())
        timestep = int(np.asarray(observation.get("timestep", 0)).reshape(-1)[0])
        self._timestep_buffer.append(timestep)
        if len(self._frame_buffer) > self._observation_window:
            self._frame_buffer = self._frame_buffer[-self._observation_window :]
            self._timestep_buffer = self._timestep_buffer[-self._observation_window :]
        pad_count = self._observation_window - len(self._frame_buffer)
        # At the start of a two-frame community rollout, repeat the first
        # canonical frame and mark only the observed slot as valid.  Once the
        # buffer is full, all slots are valid and the oldest frame is dropped.
        frames = ([self._frame_buffer[0]] * pad_count) + list(self._frame_buffer)
        timesteps = ([self._timestep_buffer[0]] * pad_count) + list(self._timestep_buffer)
        pad_mask = ([False] * pad_count) + ([True] * len(self._frame_buffer))
        self._rng, sample_key = self._jax.random.split(self._rng)
        batch = {
            "image_primary": np.asarray(frames)[None, ...],
            "timestep": np.asarray(timesteps, dtype=np.int32).reshape(1, self._observation_window),
            "timestep_pad_mask": np.asarray(pad_mask, dtype=bool).reshape(1, self._observation_window),
        }
        # Keep unnormalization explicit at this boundary.  Octo's native
        # sampler returns the action-space convention from the checkpoint;
        # conversion below enforces the LIBERO gripper convention.
        sampled = self._model.sample_actions(
            batch,
            tasks=self._tasks,
            rng=sample_key,
            unnormalization_statistics={
                "mean": self._action_mean,
                "std": self._action_std,
                "mask": self._action_mask,
            },
        )
        validate_action_chunk(sampled)
        converted = self.action_chunk_to_libero(sampled)
        return validate_policy_action(converted, self.metadata, observation=observation)

    def action_chunk_to_libero(self, sampled_chunk: Any) -> Any:
        """Convert an Octo [1,4,7] chunk to a NumPy float32 LIBERO chunk."""

        validate_action_chunk(sampled_chunk)
        return np.stack(
            [convert_unnormalized_octo_to_libero_action(step) for step in np.asarray(sampled_chunk[0])],
            axis=0,
        )

    def execute_action_chunk(
        self,
        observation: Mapping[str, Any],
        execute_action: Any,
        *,
        query_index: int,
    ) -> RolloutExecution:
        """Execute all four actions or fail closed with provenance.

        ``execute_action`` receives one float32 LIBERO action at a time.  A
        callback returning ``False`` is treated as an execution failure; no
        silent truncation or implicit replanning is allowed.
        """

        if not callable(execute_action):
            raise TypeError("execute_action callback is required for native rollout")
        chunk = self.act(observation)
        if np.asarray(chunk).shape != (OCTO_ACTION_HORIZON, OCTO_ACTION_DIM):
            raise RuntimeError("Octo native rollout refused a malformed action chunk")
        executed = 0
        try:
            for action in np.asarray(chunk, dtype=np.float32):
                result = execute_action(action)
                if result is False:
                    raise RuntimeError("executor rejected Octo action")
                executed += 1
        except Exception as exc:
            provenance = self.metadata.__dict__
            raise RuntimeError(
                f"Octo rollout failed after {executed}/{OCTO_ACTION_HORIZON} actions; "
                f"provenance={provenance}"
            ) from exc
        return RolloutExecution(
            actions_executed=executed,
            native_action_horizon=OCTO_ACTION_HORIZON,
            query_index=int(query_index),
            policy_provenance=self.metadata.__dict__,
        )
