"""Model-neutral policy adapter contracts for LIBERO.

The model packages own loading and inference.  This module only defines the
small boundary used by the shared evaluator: reset a policy for an episode,
request one native action chunk, and expose immutable provenance metadata.
Keeping validation here prevents a backend from silently changing action
horizons, dimensions, or arrow conditions during a comparison run.
"""

from __future__ import annotations

import re
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


_IMMUTABLE_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$", re.IGNORECASE)


def _receipt_mapping(value: Mapping[str, Any] | Any, *, name: str) -> Mapping[str, Any]:
    """Return a mapping for a receipt, accepting the project's ``as_dict`` objects."""

    if isinstance(value, Mapping):
        return value
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        result = as_dict()
        if isinstance(result, Mapping):
            return result
    raise TypeError(f"{name} receipt must be a mapping or expose as_dict()")


def _receipt_value(receipt: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    for alias in aliases:
        value = receipt.get(alias)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _validate_receipt_digest(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if not _IMMUTABLE_SHA256.fullmatch(value):
        raise ValueError(f"{name} receipt digest must be a 64-character SHA-256")
    return value.lower()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _path_sha256(path: Path) -> str:
    """Hash a checkpoint file or directory, including relative file names."""

    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _runtime_closure_digest(paths: Iterable[Path], *, root: Path) -> tuple[str, tuple[str, ...]]:
    digest = hashlib.sha256()
    normalized: list[tuple[str, Path]] = []
    for path in paths:
        if path.is_dir():
            candidates = (
                item for item in path.rglob("*")
                if item.is_file() and not any(part in {".git", "__pycache__", ".pytest_cache"} for part in item.parts)
            )
        elif path.is_file():
            candidates = (path,)
        else:
            raise FileNotFoundError(f"runtime closure path does not exist: {path}")
        for item in candidates:
            label = item.relative_to(root).as_posix() if item.is_relative_to(root) else str(item)
            normalized.append((label, item))
    if not normalized:
        raise ValueError("runtime closure is empty")
    for label, path in sorted(set(normalized)):
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest(), tuple(label for label, _ in sorted(set(normalized)))


def derive_checkpoint_receipt(
    checkpoint_path: str | Path,
    *,
    artifact_id: str,
    checkpoint_revision: str,
) -> dict[str, Any]:
    """Derive artifact identity from the exact local checkpoint bytes."""

    if not _IMMUTABLE_REVISION.fullmatch(str(checkpoint_revision)):
        raise ValueError("loaded checkpoint revision must be an immutable 40- or 64-character SHA")
    target = Path(checkpoint_path).expanduser().resolve()
    checkpoint_sha256 = _path_sha256(target)
    return {
        "id": str(artifact_id),
        "artifact_id": str(artifact_id),
        "revision": str(checkpoint_revision).lower(),
        "checkpoint_revision": str(checkpoint_revision).lower(),
        "checkpoint_sha256": checkpoint_sha256,
        "sha256": checkpoint_sha256,
        "path": str(target),
    }


def derive_runtime_receipt(
    *,
    source_root: str | Path | None = None,
    python_executable: str | Path = sys.executable,
    closure_paths: Sequence[str | Path] | None = None,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Derive runtime identity from source revision and a complete code closure.

    Pinned source paths are checked with Git before hashing.  Any modified,
    deleted, or untracked file causes a fail-closed receipt when ``require_clean``
    is true; callers may disable this only for diagnostic/preflight inspection.
    """

    root = Path(source_root or Path(__file__).resolve().parents[3]).expanduser().resolve()
    if closure_paths is None:
        closure_paths = (root / "vla_benchmarking" / "libero" / "evaluation",)
    resolved_closure = tuple(
        Path(item).expanduser().resolve() if Path(item).is_absolute()
        else (root / Path(item)).resolve()
        for item in closure_paths
    )
    repo_paths = []
    for item in resolved_closure:
        try:
            repo_paths.append(item.relative_to(root).as_posix())
        except ValueError:
            continue
    if require_clean and repo_paths:
        try:
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--", *repo_paths],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"unable to verify pinned runtime source state under {root}") from exc
        if dirty:
            raise RuntimeError("pinned runtime source is dirty or contains untracked files")
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"unable to resolve installed source commit under {root}") from exc
    source_commit = completed.stdout.strip().lower()
    if not _IMMUTABLE_REVISION.fullmatch(source_commit):
        raise ValueError("installed source commit must be an immutable 40-character SHA")
    executable = Path(python_executable).expanduser()
    resolved = str(executable.resolve()) if executable.exists() else str(python_executable)
    payload = {
        "source_commit": source_commit,
        "python": resolved,
        "python_version": ".".join(str(item) for item in sys.version_info[:3]),
    }
    closure_sha256, closure = _runtime_closure_digest(resolved_closure, root=root)
    payload["closure_sha256"] = closure_sha256
    payload["closure"] = list(closure)
    return {"id": f"runtime:{source_commit}", "sha256": _canonical_sha256(payload), **payload}


def derive_io_receipt(metadata: "PolicyMetadata" | Mapping[str, Any]) -> dict[str, Any]:
    """Derive I/O identity from the concrete loaded adapter metadata."""

    def value(name: str, default: Any = None) -> Any:
        if isinstance(metadata, Mapping):
            return metadata.get(name, default)
        return getattr(metadata, name, default)

    payload = {
        "action_horizon": int(value("native_action_horizon", value("action_horizon"))),
        "action_dim": int(value("action_dim", 7)),
        "input_resolution": int(value("input_resolution", 256)),
        "camera_keys": list(value("camera_keys", ())),
        "state_dim": value("state_dim"),
        "visual_input": value("visual_input", "none"),
    }
    digest = _canonical_sha256(payload)
    return {"id": f"io:{digest}", "sha256": digest, **payload}


def derive_dataset_manifest_receipt(path: str | Path) -> dict[str, Any]:
    """Derive dataset identity from the manifest file bytes, not plan labels."""

    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"dataset manifest is not valid JSON: {target}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("dataset manifest must contain a JSON object")
    # Octo binds the normalized completion-manifest identity used by its
    # preflight/evaluator, rather than the incidental file-byte digest.  Keep
    # the generic VLA receipt byte-based for all other manifest schemas.
    if value.get("schema") == "octo_dataset_completion.v1":
        from vla_benchmarking.libero.finetuned_vlas.octo.manifest import completion_manifest_sha256

        digest = completion_manifest_sha256(dict(value))
    else:
        digest = _path_sha256(target)
    return {
        "id": f"dataset:{digest}",
        "manifest_sha256": digest,
        "dataset_manifest_sha256": digest,
        "path": str(target),
        "schema": value.get("schema"),
    }


@dataclass(frozen=True)
class PolicyProvenance:
    """Immutable receipt identities attached to a loaded adapter.

    The artifact identity is represented by the adapter's existing
    ``artifact_id`` and ``checkpoint_revision`` fields.  Runtime, I/O, and
    dataset identities are retained in both ID and digest form when supplied;
    this lets a v2 plan bind either a stable receipt ID or its content digest
    without allowing a caller to silently substitute a different receipt.
    """

    artifact_id: str
    checkpoint_revision: str
    artifact_sha256: str | None = None
    runtime_id: str | None = None
    runtime_sha256: str | None = None
    io_id: str | None = None
    io_sha256: str | None = None
    dataset_manifest_id: str | None = None
    dataset_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not str(self.artifact_id).strip():
            raise ValueError("artifact receipt requires a non-empty artifact id")
        revision = str(self.checkpoint_revision)
        if not _IMMUTABLE_REVISION.fullmatch(revision):
            raise ValueError(
                "artifact checkpoint revision must be an immutable 40- or 64-character SHA"
            )
        _validate_receipt_digest(self.artifact_sha256, name="artifact")
        for name in ("runtime_id", "io_id", "dataset_manifest_id"):
            value = getattr(self, name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{name} must be non-empty when supplied")
        for name in ("runtime_sha256", "io_sha256", "dataset_manifest_sha256"):
            _validate_receipt_digest(getattr(self, name), name=name)
        for name in ("runtime", "io", "dataset_manifest"):
            if getattr(self, f"{name}_id") is None and getattr(self, f"{name}_sha256") is None:
                raise ValueError(f"{name} receipt requires an immutable identity")

    @classmethod
    def from_receipts(
        cls,
        *,
        artifact: Mapping[str, Any] | Any,
        runtime: Mapping[str, Any] | Any,
        io: Mapping[str, Any] | Any,
        dataset_manifest: Mapping[str, Any] | Any,
    ) -> "PolicyProvenance":
        """Build a provenance bundle from provider-neutral receipt mappings."""

        artifact_receipt = _receipt_mapping(artifact, name="artifact")
        artifact_id = _receipt_value(artifact_receipt, ("artifact_id", "id", "model_id"))
        revision = _receipt_value(artifact_receipt, ("checkpoint_revision", "revision", "model_revision"))
        if artifact_id is None or revision is None:
            raise ValueError("artifact receipt requires artifact id and checkpoint revision")

        artifact_sha256 = _validate_receipt_digest(
            _receipt_value(artifact_receipt, ("checkpoint_sha256", "artifact_sha256", "sha256")),
            name="artifact",
        )

        def receipt_fields(value: Mapping[str, Any] | Any, *, name: str) -> tuple[str | None, str | None]:
            receipt = _receipt_mapping(value, name=name)
            identity = _receipt_value(receipt, (f"{name}_id", "receipt_id", "id"))
            digest = _receipt_value(
                receipt,
                (f"{name}_sha256", "manifest_sha256" if name == "dataset_manifest" else "sha256", "receipt_sha256"),
            )
            return identity, _validate_receipt_digest(digest, name=name)

        runtime_id, runtime_sha256 = receipt_fields(runtime, name="runtime")
        io_id, io_sha256 = receipt_fields(io, name="io")
        dataset_id, dataset_sha256 = receipt_fields(dataset_manifest, name="dataset_manifest")
        return cls(
            artifact_id=artifact_id,
            checkpoint_revision=revision,
            artifact_sha256=artifact_sha256,
            runtime_id=runtime_id,
            runtime_sha256=runtime_sha256,
            io_id=io_id,
            io_sha256=io_sha256,
            dataset_manifest_id=dataset_id,
            dataset_manifest_sha256=dataset_sha256,
        )

    def as_extra(self) -> dict[str, str]:
        """Serialize receipt identities into ``PolicyMetadata.extra``."""

        values = {
            "artifact_sha256": self.artifact_sha256,
            "checkpoint_sha256": self.artifact_sha256,
            "runtime_id": self.runtime_id,
            "runtime_sha256": self.runtime_sha256,
            "io_id": self.io_id,
            "io_sha256": self.io_sha256,
            "dataset_manifest_id": self.dataset_manifest_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
        }
        return {key: value for key, value in values.items() if value is not None}


def _float_vector(value: Sequence[float], *, name: str, size: int) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != size:
        raise ValueError(f"{name} must contain {size} values; got {len(result)}")
    if not np.all(np.isfinite(np.asarray(result, dtype=np.float64))):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class PolicyMetadata:
    """Identity and I/O contract for one loaded policy artifact."""

    policy_kind: str
    artifact_id: str
    checkpoint_revision: str
    backend: str
    native_action_horizon: int
    adapter_kind: str | None = None
    action_dim: int = 7
    input_resolution: int = 256
    camera_keys: tuple[str, ...] = ("agentview",)
    state_dim: int | None = None
    visual_input: str = "none"
    action_low: tuple[float, ...] = field(default_factory=lambda: (-1.0,) * 7)
    action_high: tuple[float, ...] = field(default_factory=lambda: (1.0,) * 7)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.policy_kind).strip() or not str(self.artifact_id).strip():
            raise ValueError("policy_kind and artifact_id must be non-empty")
        if not str(self.checkpoint_revision).strip():
            raise ValueError("checkpoint_revision must be non-empty")
        if self.adapter_kind is None:
            object.__setattr__(self, "adapter_kind", self.policy_kind)
        elif not str(self.adapter_kind).strip():
            raise ValueError("adapter_kind must be non-empty")
        if int(self.native_action_horizon) <= 0:
            raise ValueError("native_action_horizon must be positive")
        if int(self.action_dim) != 7:
            raise ValueError("LIBERO policy action_dim must be exactly 7")
        if int(self.input_resolution) <= 0:
            raise ValueError("input_resolution must be positive")
        if not self.camera_keys or any(not str(key).strip() for key in self.camera_keys):
            raise ValueError("camera_keys must contain non-empty names")
        if self.state_dim is not None and int(self.state_dim) < 0:
            raise ValueError("state_dim must be non-negative or None")
        if self.visual_input != "none":
            raise ValueError("new VLA adapters must use visual_input='none'")
        object.__setattr__(self, "action_low", _float_vector(self.action_low, name="action_low", size=7))
        object.__setattr__(self, "action_high", _float_vector(self.action_high, name="action_high", size=7))
        if any(low >= high for low, high in zip(self.action_low, self.action_high)):
            raise ValueError("every action_low value must be less than action_high")

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_kind": self.policy_kind,
            "artifact_id": self.artifact_id,
            "checkpoint_revision": self.checkpoint_revision,
            "backend": self.backend,
            "adapter_kind": self.adapter_kind,
            "native_action_horizon": int(self.native_action_horizon),
            "action_dim": int(self.action_dim),
            "input_resolution": int(self.input_resolution),
            "camera_keys": list(self.camera_keys),
            "state_dim": self.state_dim,
            "visual_input": self.visual_input,
            "action_low": list(self.action_low),
            "action_high": list(self.action_high),
            "extra": dict(self.extra),
        }

    def with_provenance(self, provenance: PolicyProvenance) -> "PolicyMetadata":
        """Return metadata carrying receipts, rejecting conflicting identities."""

        if self.artifact_id != provenance.artifact_id:
            raise ValueError("adapter artifact_id does not match provenance receipt")
        if self.checkpoint_revision != provenance.checkpoint_revision:
            raise ValueError("adapter checkpoint revision does not match provenance receipt")
        merged = dict(self.extra)
        for key, value in provenance.as_extra().items():
            existing = merged.get(key)
            if existing is not None and str(existing) != str(value):
                raise ValueError(f"adapter metadata contains conflicting {key}")
            merged[key] = value
        return replace(self, extra=merged)


@runtime_checkable
class PolicyAdapter(Protocol):
    """Runtime protocol implemented by every native model adapter."""

    @property
    def metadata(self) -> PolicyMetadata: ...

    def reset(self, task_description: str, episode_seed: int) -> None: ...

    def act(self, observation: Mapping[str, Any]) -> np.ndarray: ...


class ProvenanceBoundPolicyAdapter:
    """Transparent adapter view that attaches immutable receipts to metadata.

    Native model packages remain responsible for loading and inference.  This
    wrapper is the shared-evaluator seam for attaching the receipts produced by
    those packages, so their source files do not need to depend on the v2 plan
    implementation.
    """

    def __init__(self, adapter: PolicyAdapter, provenance: PolicyProvenance) -> None:
        self._adapter = adapter
        self._provenance = provenance
        # Validate the artifact identity at construction time rather than
        # allowing a mismatched plan to fail only after an episode starts.
        self.metadata  # force the base metadata contract now

    @property
    def metadata(self) -> PolicyMetadata:
        return self._adapter.metadata.with_provenance(self._provenance)

    def reset(self, task_description: str, episode_seed: int) -> None:
        self._adapter.reset(task_description, int(episode_seed))

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        return self._adapter.act(observation)


def bind_adapter_provenance(
    adapter: PolicyAdapter,
    *,
    artifact: Mapping[str, Any] | Any,
    runtime: Mapping[str, Any] | Any,
    io: Mapping[str, Any] | Any,
    dataset_manifest: Mapping[str, Any] | Any,
) -> ProvenanceBoundPolicyAdapter:
    """Attach immutable artifact/runtime/I/O/dataset receipt identities.

    This helper is intended for the production evaluator after a native model
    adapter has loaded its artifact and runtime receipts.  It does not mutate
    the native adapter and preserves its reset/act behavior.
    """

    provenance = PolicyProvenance.from_receipts(
        artifact=artifact,
        runtime=runtime,
        io=io,
        dataset_manifest=dataset_manifest,
    )
    return ProvenanceBoundPolicyAdapter(adapter, provenance)


# Descriptive alias for callers that prefer the policy-level wording.
bind_policy_provenance = bind_adapter_provenance


def validate_arrow_free_observation(observation: Mapping[str, Any]) -> None:
    """Reject explicit arrow overlays/metadata before a no-arrow rollout.

    Pixel-level arrow detection is intentionally not guessed here.  Renderers
    must provide an explicit audit marker or mask when they create overlays;
    this contract fails closed on those markers while leaving raw RGB values
    untouched.
    """

    for key in ("arrow_overlay", "visual_arrow", "has_arrows", "arrow_mask", "arrows"):
        if key not in observation:
            continue
        marker = observation[key]
        if isinstance(marker, np.ndarray):
            present = bool(marker.any())
        elif isinstance(marker, (list, tuple, set, dict)):
            present = bool(marker)
        else:
            present = bool(marker)
        if present:
            raise ValueError(f"arrow-free policy received an observation with {key}")


def validate_policy_action(
    action: Any,
    metadata: PolicyMetadata,
    *,
    observation: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Validate and normalize one policy-native action chunk to float32."""

    if observation is not None:
        validate_arrow_free_observation(observation)
    if hasattr(action, "detach") and hasattr(action, "cpu"):
        action = action.detach().cpu().numpy()
    values = np.asarray(action)
    expected = (int(metadata.native_action_horizon), int(metadata.action_dim))
    if values.shape != expected:
        raise ValueError(f"policy {metadata.policy_kind} must return shape {expected}; got {values.shape}")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("policy action must be numeric")
    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("policy action contains NaN or infinity")
    low = np.asarray(metadata.action_low, dtype=np.float32)
    high = np.asarray(metadata.action_high, dtype=np.float32)
    if np.any(values < low[None, :]) or np.any(values > high[None, :]):
        raise ValueError("policy action is outside the declared action range")
    return np.ascontiguousarray(values, dtype=np.float32)


def validate_adapter_metadata(adapter: PolicyAdapter, plan: Mapping[str, Any]) -> None:
    """Fail closed when a loaded adapter does not match an evaluation plan."""

    metadata = adapter.metadata
    policy_kind = str(plan.get("policy_kind", ""))
    if metadata.policy_kind != policy_kind:
        raise ValueError(
            f"adapter policy_kind {metadata.policy_kind!r} does not match plan {policy_kind!r}"
        )
    if plan.get("schema") != "shared_evaluation_plan.v2":
        return
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("v2 plan lacks adapter bindings")
    expected_kind = str(bindings.get("adapter_kind", ""))
    if metadata.adapter_kind != expected_kind:
        raise ValueError(
            f"adapter kind {metadata.adapter_kind!r} does not match plan {expected_kind!r}"
        )
    artifact = bindings.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("v2 artifact binding is invalid")
    expected_artifact = artifact.get("artifact_id", artifact.get("id"))
    if expected_artifact is not None and str(expected_artifact) != metadata.artifact_id:
        raise ValueError("adapter artifact_id does not match plan binding")
    expected_revision = artifact.get("checkpoint_revision", artifact.get("revision"))
    if expected_revision is not None and str(expected_revision) != metadata.checkpoint_revision:
        raise ValueError("adapter checkpoint revision does not match plan binding")
    artifact_hash = next(
        (artifact.get(key) for key in ("checkpoint_sha256", "artifact_sha256", "sha256") if artifact.get(key)),
        None,
    )
    if artifact_hash is not None:
        observed_hash = next(
            (
                metadata.extra.get(key)
                for key in ("checkpoint_sha256", "artifact_sha256")
                if metadata.extra.get(key) is not None
            ),
            None,
        )
        if observed_hash is None:
            raise ValueError("adapter metadata lacks artifact checkpoint identity")
        if str(observed_hash) != str(artifact_hash):
            raise ValueError("adapter artifact checkpoint identity does not match plan binding")

    def _check_extra(binding_name: str, *names: str) -> None:
        binding = bindings.get(binding_name)
        if not isinstance(binding, Mapping):
            raise ValueError(f"v2 {binding_name} binding is invalid")
        extras = dict(metadata.extra)
        expected_pairs = []
        for binding_key in ("sha256", "manifest_sha256", "id"):
            expected = binding.get(binding_key)
            if expected is not None and str(expected).strip():
                expected_pairs.append((binding_key, str(expected)))
        if not expected_pairs:
            raise ValueError(f"v2 {binding_name} binding has no identity")
        for binding_key, expected in expected_pairs:
            aliases = names if binding_key != "id" else tuple(
                name.replace("_sha256", "_id") for name in names
            )
            observed = next((extras[name] for name in aliases if extras.get(name) is not None), None)
            if observed is None:
                raise ValueError(f"adapter metadata lacks {binding_name} {binding_key} identity")
            if str(observed) != expected:
                raise ValueError(f"adapter {binding_name} {binding_key} identity does not match plan binding")

    _check_extra("runtime", "runtime_sha256", "runtime_contract_sha256")
    _check_extra("io", "io_sha256", "io_contract_sha256")
    _check_extra("dataset_manifest", "dataset_manifest_sha256", "manifest_sha256")
    io_binding = bindings.get("io")
    if isinstance(io_binding, Mapping):
        for field_name, observed in (
            ("action_horizon", metadata.native_action_horizon),
            ("action_dim", metadata.action_dim),
            ("input_resolution", metadata.input_resolution),
            ("state_dim", metadata.state_dim),
        ):
            if field_name in io_binding and io_binding[field_name] is not None:
                if int(io_binding[field_name]) != int(observed):
                    raise ValueError(f"adapter {field_name} does not match plan I/O binding")
        if "camera_keys" in io_binding and tuple(io_binding["camera_keys"]) != tuple(metadata.camera_keys):
            raise ValueError("adapter camera_keys do not match plan I/O binding")


class CallablePolicyAdapter:
    """Small adapter useful for native backends and unit/integration tests."""

    def __init__(self, metadata: PolicyMetadata, *, reset_fn=None, act_fn=None) -> None:
        if act_fn is None:
            raise ValueError("act_fn is required")
        self._metadata = metadata
        self._reset_fn = reset_fn
        self._act_fn = act_fn

    @property
    def metadata(self) -> PolicyMetadata:
        return self._metadata

    def reset(self, task_description: str, episode_seed: int) -> None:
        if self._reset_fn is not None:
            self._reset_fn(task_description, int(episode_seed))

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        return validate_policy_action(self._act_fn(observation), self._metadata, observation=observation)


__all__ = [
    "CallablePolicyAdapter",
    "PolicyProvenance",
    "PolicyAdapter",
    "PolicyMetadata",
    "ProvenanceBoundPolicyAdapter",
    "bind_adapter_provenance",
    "bind_policy_provenance",
    "derive_checkpoint_receipt",
    "derive_dataset_manifest_receipt",
    "derive_io_receipt",
    "derive_runtime_receipt",
    "validate_arrow_free_observation",
    "validate_adapter_metadata",
    "validate_policy_action",
]
