"""Native SmolVLA/LoRA Apprentice orchestration contracts.

The production exporter and trainer are intentionally injected.  Importing this
module never imports LeRobot/PEFT and never starts a dataset conversion or
training job.  The contract makes the successful On-Call teacher rows,
SmolVLA identity, LoRA settings, and dataset lineage explicit at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import ContractError, _safe
from .learning import DatasetManifest, InterventionRow


APPRENTICE_SCHEMA = "arrow_policy_suite.apprentice.v1"
DEFAULT_SMOLVLA_MODEL = "HuggingFaceVLA/smolvla_libero"


def _canonical(value: Any) -> bytes:
    return (json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


@dataclass(frozen=True)
class SmolVLALoRAConfig:
    """Pinned action-side PEFT values for the native SmolVLA adapter."""

    model_id: str = DEFAULT_SMOLVLA_MODEL
    model_revision: str = "unresolved"
    processor_revision: str = "unresolved"
    rank: int = 16
    alpha: int = 8
    dropout: float = 0.0
    bias: str = "none"
    target_modules: tuple[str, ...] = (
        "action_in_proj",
        "action_out_proj",
        "q_proj",
        "v_proj",
    )
    modules_to_save: tuple[str, ...] = ()
    base_vla_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.model_id or self.rank <= 0 or self.alpha <= 0:
            raise ContractError("SmolVLA model identity and positive LoRA rank/alpha are required")
        if self.bias not in {"none", "all", "lora_only"}:
            raise ContractError("LoRA bias must be none, all, or lora_only")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ContractError("LoRA dropout must lie in [0, 1)")
        if not self.target_modules:
            raise ContractError("native SmolVLA LoRA requires explicit target modules")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": APPRENTICE_SCHEMA,
            "provider": "huggingface/lerobot/peft",
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "processor_revision": self.processor_revision,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": float(self.dropout),
            "bias": self.bias,
            "target_modules": list(self.target_modules),
            "modules_to_save": list(self.modules_to_save),
            "base_vla_sha256": self.base_vla_sha256,
        }


@dataclass(frozen=True)
class ApprenticeTrainingConfig:
    """Native training values; execution is owned by the injected trainer."""

    seed: int = 1000
    effective_batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    warmup_fraction_denominator: int = 30
    max_grad_norm: float = 10.0
    mixed_precision: str = "no"

    def __post_init__(self) -> None:
        if self.seed < 0 or self.effective_batch_size <= 0 or self.epochs <= 0:
            raise ContractError("invalid SmolVLA training seed, batch size, or epoch count")
        if float(self.learning_rate) <= 0.0 or float(self.weight_decay) < 0.0:
            raise ContractError("SmolVLA learning_rate must be positive and weight_decay non-negative")
        if self.warmup_fraction_denominator <= 0 or float(self.max_grad_norm) <= 0.0:
            raise ContractError("invalid scheduler or gradient clipping configuration")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ContractError("mixed_precision must be no, fp16, or bf16")

    def derived_steps(self, frames: int) -> int:
        if frames <= 0:
            raise ContractError("exported dataset must contain at least one frame")
        return max(1, (self.epochs * frames + self.effective_batch_size - 1) // self.effective_batch_size)

    def manifest(self, *, frames: int) -> dict[str, Any]:
        steps = self.derived_steps(frames)
        return {
            "seed": self.seed,
            "effective_batch_size": self.effective_batch_size,
            "epochs": self.epochs,
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "warmup_steps": steps // self.warmup_fraction_denominator,
            "optimizer_steps": steps,
            "max_grad_norm": float(self.max_grad_norm),
            "mixed_precision": self.mixed_precision,
        }


@dataclass(frozen=True)
class ApprenticeExportRequest:
    """Rows and provenance passed to a native LeRobot exporter."""

    rows: tuple[InterventionRow, ...]
    parent_manifest: DatasetManifest
    destination: str
    variant: str = "apprentice_teacher_actions"
    state_key: str = "observation.state"
    action_key: str = "action"

    def __post_init__(self) -> None:
        if not self.rows:
            raise ContractError("Apprentice export requires at least one executed teacher row")
        if not self.destination or self.variant != "apprentice_teacher_actions":
            raise ContractError("Apprentice export destination/variant is invalid")
        if self.parent_manifest.rows != len(self.rows):
            raise ContractError("parent manifest row count does not match export rows")
        if not self.parent_manifest.content_sha256:
            raise ContractError("parent manifest content hash is required")
        seen: set[tuple[str, int]] = set()
        for row in self.rows:
            if row.source != "on_call_teacher" or row.metadata.get("source_policy_family", "arrow_on_call") != "arrow_on_call":
                raise ContractError("Apprentice rows must be executed On-Call teacher rows")
            identity = (row.episode_id, row.timestep)
            if identity in seen:
                raise ContractError(f"duplicate source row {identity!r}")
            seen.add(identity)

    @property
    def episodes(self) -> tuple[str, ...]:
        return tuple(sorted({row.episode_id for row in self.rows}))

    def native_rows(self) -> tuple[dict[str, Any], ...]:
        """Return only observation + teacher action to the native BC exporter."""
        return tuple(
            {
                "episode_id": row.episode_id,
                "task_id": row.task_id,
                "timestep": row.timestep,
                "observation": row.observation,
                self.action_key: list(row.teacher_action),
            }
            for row in self.rows
        )

    def lineage(self) -> dict[str, Any]:
        payload = {
            "schema": APPRENTICE_SCHEMA,
            "parent_manifest_sha256": self.parent_manifest.content_sha256,
            "parent_artifact": self.parent_manifest.parent_artifact,
            "rows": len(self.rows),
            "episodes": list(self.episodes),
            "destination": self.destination,
            "variant": self.variant,
            "state_key": self.state_key,
            "action_key": self.action_key,
            "teacher_source": "executed_on_call_teacher",
            "outcomes": {"success_rows": sum(bool(row.success_episode) for row in self.rows),
                          "failure_rows": sum(not bool(row.success_episode) for row in self.rows)},
            "base_actions_exported": False,
        }
        encoded = json.dumps(_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {**payload, "request_sha256": hashlib.sha256(encoded).hexdigest()}


@dataclass(frozen=True)
class ApprenticeExportReceipt:
    request: ApprenticeExportRequest
    dataset_path: str
    dataset_manifest_sha256: str
    frames: int
    native_result: Any

    def __post_init__(self) -> None:
        if not self.dataset_path or self.frames <= 0 or not self.dataset_manifest_sha256:
            raise ContractError("native export must return path, positive frame count, and dataset hash")

    def lineage(self) -> dict[str, Any]:
        return {
            **self.request.lineage(),
            "dataset_path": str(Path(self.dataset_path)),
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "frames": self.frames,
        }


class NativeSmolVLAExporter(Protocol):
    def __call__(self, request: ApprenticeExportRequest) -> Mapping[str, Any]: ...


class NativeSmolVLATrainer(Protocol):
    def __call__(self, receipt: ApprenticeExportReceipt, *, lora: SmolVLALoRAConfig,
                 training: ApprenticeTrainingConfig) -> Any: ...


def make_export_request(
    rows: Sequence[InterventionRow],
    parent_manifest: DatasetManifest,
    *,
    destination: str,
) -> ApprenticeExportRequest:
    """Build a fail-closed request; does not write data or import native code."""

    return ApprenticeExportRequest(tuple(rows), parent_manifest, destination)


def export_apprentice_dataset(
    request: ApprenticeExportRequest,
    exporter: NativeSmolVLAExporter,
) -> ApprenticeExportReceipt:
    """Invoke the caller's native exporter and check returned lineage fields."""

    result = exporter(request)
    if not isinstance(result, Mapping):
        raise ContractError("native exporter must return a mapping")
    parent_hash = result.get("parent_manifest_sha256")
    if parent_hash is not None and str(parent_hash) != request.parent_manifest.content_sha256:
        raise ContractError("native exporter returned the wrong parent manifest hash")
    try:
        return ApprenticeExportReceipt(
            request=request,
            dataset_path=str(result["dataset_path"]),
            dataset_manifest_sha256=str(result["dataset_manifest_sha256"]),
            frames=int(result["frames"]),
            native_result=result.get("native_result"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("native exporter result lacks required dataset identity fields") from exc


def export_apprentice_dataset_native(
    request: ApprenticeExportRequest,
    *,
    dataset_factory: Callable[..., Any] | None = None,
    fps: int = 10,
    robot_type: str = "libero",
) -> ApprenticeExportReceipt:
    """Export rows through the installed LeRobot dataset API.

    Imports are intentionally lazy.  A caller can inject ``dataset_factory``
    for a pinned LeRobot version or tests; otherwise the function resolves
    ``LeRobotDataset`` at call time and refuses to guess a remote repository.
    """
    if fps <= 0 or not robot_type:
        raise ContractError("native dataset export requires positive fps and robot_type")
    factory = dataset_factory
    if factory is None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
        except ImportError as exc:
            raise ContractError("LeRobot is required for native Apprentice export") from exc
        factory = LeRobotDataset
    destination = Path(request.destination)
    # LeRobot versions differ in their feature declaration.  Prefer the
    # modern create() API, while keeping the factory injectable for a pinned
    # installation.
    features = {request.action_key: {"dtype": "float32", "shape": (7,)}}
    for key in ("observation.state", "state"):
        features[key] = {"dtype": "float32", "shape": (8,)}
    try:
        dataset = factory.create(repo_id=f"local/{destination.name}", root=destination,
                                 fps=fps, robot_type=robot_type, features=features)
    except TypeError:
        dataset = factory.create(repo_id=f"local/{destination.name}", root=destination, fps=fps, features=features)
    for row in request.rows:
        observation = dict(row.observation)
        frame = {request.action_key: list(row.teacher_action), "task": str(row.task_id),
                 "episode_id": row.episode_id, "timestep": row.timestep}
        state = observation.get("observation.state", observation.get("state"))
        if state is not None:
            frame["observation.state"] = list(state)
        for key in request.state_key,:
            if key in observation:
                frame[key] = observation[key]
        add_frame = getattr(dataset, "add_frame", None)
        if not callable(add_frame):
            raise ContractError("LeRobot dataset object lacks add_frame")
        add_frame(frame)
    save_episode = getattr(dataset, "save_episode", None)
    if callable(save_episode):
        save_episode()
    manifest_path = destination / "meta" / "info.json"
    dataset_hash = hashlib.sha256()
    if manifest_path.exists():
        dataset_hash.update(manifest_path.read_bytes())
    else:
        dataset_hash.update(_canonical(request.native_rows()))
    result = {"dataset_path": str(destination), "dataset_manifest_sha256": dataset_hash.hexdigest(),
              "frames": len(request.rows), "parent_manifest_sha256": request.parent_manifest.content_sha256,
              "native_result": {"provider": "lerobot", "fps": fps, "robot_type": robot_type}}
    return export_apprentice_dataset(request, lambda _request: result)


def train_apprentice(
    receipt: ApprenticeExportReceipt,
    trainer: NativeSmolVLATrainer,
    *,
    lora: SmolVLALoRAConfig | None = None,
    training: ApprenticeTrainingConfig | None = None,
) -> dict[str, Any]:
    """Invoke native SmolVLA training only through an explicit injected hook."""

    resolved_lora = lora or SmolVLALoRAConfig()
    resolved_training = training or ApprenticeTrainingConfig()
    native_result = trainer(receipt, lora=resolved_lora, training=resolved_training)
    return {
        "schema": APPRENTICE_SCHEMA,
        "model": resolved_lora.manifest(),
        "training": resolved_training.manifest(frames=receipt.frames),
        "dataset": receipt.lineage(),
        "native_result": native_result,
    }


def save_adapter_checkpoint(
    path: str | Path,
    adapter_state: Mapping[str, Any],
    *,
    base_vla_sha256: str,
    training: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save adapter-only state once and emit a hashable verification record."""
    if not base_vla_sha256 or len(base_vla_sha256) != 64:
        raise ContractError("Apprentice adapter checkpoint requires frozen base_vla_sha256")
    if not isinstance(adapter_state, Mapping) or not adapter_state:
        raise ContractError("adapter_state must be a non-empty mapping")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # torch.save is used only when tensors are present; JSON keeps the module
    # useful in lightweight environments and in contract tests.
    try:
        import torch  # type: ignore
        buffer = io.BytesIO()
        torch.save(dict(adapter_state), buffer)
        payload = buffer.getvalue()
        serialization = "torch"
    except (ImportError, RuntimeError, TypeError):
        payload = (_canonical(adapter_state))
        serialization = "json"
    if target.exists():
        raise ContractError(f"refusing to overwrite immutable adapter checkpoint: {target}")
    target.write_bytes(payload)
    adapter_hash = hashlib.sha256(payload).hexdigest()
    manifest = {"schema": APPRENTICE_SCHEMA + ".adapter", "checkpoint_path": str(target),
                "adapter_sha256": adapter_hash, "base_vla_sha256": base_vla_sha256,
                "adapter_only": True, "serialization": serialization,
                "keys": sorted(str(key) for key in adapter_state), "training": dict(training or {})}
    manifest_path = Path(str(target) + ".json")
    if manifest_path.exists():
        raise ContractError(f"refusing to overwrite immutable adapter manifest: {manifest_path}")
    manifest_path.write_bytes(_canonical(manifest))
    return manifest


def verify_adapter_reload(path: str | Path, *, base_vla_sha256: str,
                          loader: Callable[[Path], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Reload an adapter and verify bytes, frozen-base identity, and keys."""
    target = Path(path)
    manifest_path = Path(str(target) + ".json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = target.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("adapter checkpoint or manifest is unreadable") from exc
    if manifest.get("base_vla_sha256") != base_vla_sha256 or not manifest.get("adapter_only", False):
        raise ContractError("adapter reload failed frozen-base/adapter-only verification")
    if hashlib.sha256(payload).hexdigest() != manifest.get("adapter_sha256"):
        raise ContractError("adapter checkpoint hash changed")
    if loader is None:
        if manifest.get("serialization") == "torch":
            try:
                import torch  # type: ignore
                state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
            except (ImportError, RuntimeError, TypeError) as exc:
                raise ContractError("torch adapter reload unavailable") from exc
        else:
            state = json.loads(payload.decode("utf-8"))
    else:
        state = loader(target)
    if sorted(str(key) for key in state) != list(manifest.get("keys", ())):
        raise ContractError("reloaded adapter keys differ from saved adapter")
    return {**manifest, "reloaded": True}


def invoke_native_trainer(command: Sequence[str], *, cwd: str | Path | None = None,
                          timeout: float | None = None) -> dict[str, Any]:
    """Invoke a pinned LeRobot trainer command without importing it eagerly."""
    if not command:
        raise ContractError("native trainer command cannot be empty")
    completed = subprocess.run(tuple(str(item) for item in command), cwd=cwd,
                               capture_output=True, text=True, timeout=timeout, check=False)
    if completed.returncode != 0:
        raise ContractError(f"native trainer failed with exit code {completed.returncode}: {completed.stderr[-1000:]}")
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


# Short aliases keep integration launchers readable without changing the
# explicit manifest-oriented class names above.
NativeSmolVLAConfig = SmolVLALoRAConfig
ApprenticeDatasetRequest = ApprenticeExportRequest
export_teacher_rows = export_apprentice_dataset


__all__ = [
    "APPRENTICE_SCHEMA", "DEFAULT_SMOLVLA_MODEL", "ApprenticeDatasetRequest", "ApprenticeExportReceipt",
    "ApprenticeExportRequest", "ApprenticeTrainingConfig", "NativeSmolVLAConfig", "NativeSmolVLAExporter",
    "NativeSmolVLATrainer", "SmolVLALoRAConfig", "export_apprentice_dataset", "export_teacher_rows",
    "make_export_request", "train_apprentice", "export_apprentice_dataset_native",
    "save_adapter_checkpoint", "verify_adapter_reload", "invoke_native_trainer",
]
