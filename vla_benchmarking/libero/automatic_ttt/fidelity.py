"""Provenance and fidelity gates for the RoboTTT implementation.

The public paper specifies the algorithm but does not publish the exact model
checkpoint or several architecture/training details.  This module makes that
distinction executable: an exact-paper run must carry an artifact manifest
which resolves every required field.  A user may still run an explicitly
labelled algorithmic port, but it cannot be reported as RoboTTT-exact.
"""

from __future__ import annotations

import hashlib
import json
import re
import weakref
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PAPER_ID = "RoboTTT/arXiv:2607.15275v1"
PAPER_URL = "https://arxiv.org/html/2607.15275v1"

# These are intentionally explicit.  None means that the paper does not give
# enough information to reproduce the value without the authors' artifact.
PAPER_KNOWN = {
    "ttt_variant": "TTT-KVB",
    "fast_network": "two-layer GeLU MLP",
    "register_tokens_per_timestep": 16,
    "diT_layers_with_ttt": 16,
    "gate": "tanh(alpha)",
    "gate_initial_alpha": 0.001,
    "flow_matching_tau": "0.999 * (1-u), u~Beta(1.5, 1)",
    "pretrain_steps": 30000,
    "posttrain_steps": 20000,
    "pretrain_weight_decay": 1e-5,
    "pretrain_peak_lr": 2e-5,
    "posttrain_weight_decay": 1e-5,
    "posttrain_peak_lr": 5e-5,
    "posttrain_optimizer": "AdamW",
    "action_loss_on": "human correction chunks only",
    "fast_state_context": "robot and human chunks",
}

# Values that must be supplied by the official release (or explicitly marked
# unresolved) before claiming exact reproduction.
REQUIRED_ARTIFACT_FIELDS = (
    "checkpoint_sha256",
    "checkpoint_uri",
    "fast_mlp_hidden_dim",
    "qkv_dimensions",
    "qkv_normalization",
    "inner_learning_rate_parameterization",
    "tbptt_segment_length",
    "action_horizon",
    "denoising_steps",
    "token_packing",
    "optimizer_betas_eps_clip",
    "image_crop_stride_preprocessing",
)

# Runtime metadata is supplied by the concrete VLA adapter.  It is separate
# from the paper manifest so a correct paper manifest cannot accidentally bless
# a kernel wired into only one DiT layer.
RUNTIME_REQUIRED_FIELDS = (
    "fast_mlp_hidden_dim",
    "qkv_dimensions",
    "qkv_normalization",
    "inner_learning_rate_parameterization",
    "tbptt_segment_length",
    "action_horizon",
    "denoising_steps",
    "token_packing",
)


class ExactFidelityError(RuntimeError):
    """Raised when a caller requests an exact run without resolved artifacts."""


class RuntimeAttestation:
    """Opaque receipt produced from an inspected, executed model instance.

    Callers cannot construct this from a dictionary.  ``capture_runtime_attestation``
    reads the module's own receipt after its forward hook has observed a real
    execution, preventing a hand-written mapping from claiming exact wiring.
    """

    __slots__ = ("_payload", "_token", "_module_id", "_validated_artifact_id", "__weakref__")

    def __init__(self, payload: Mapping[str, Any], token: object, module: Any) -> None:
        if token is not _ATTESTATION_TOKEN:
            raise TypeError("RuntimeAttestation must be created by capture_runtime_attestation")
        self._payload = dict(payload)
        self._token = token
        self._module_id = id(module)
        self._validated_artifact_id: int | None = None

    def _as_mapping(self) -> Mapping[str, Any]:
        return self._payload


# Descriptive alias used by host adapters and paper-facing reports.  Both names
# refer to the same opaque capability object; neither accepts caller mappings.
ExactRuntimeAttestation = RuntimeAttestation


_ATTESTATION_TOKEN = object()
_ISSUED_ATTESTATIONS: "weakref.WeakSet[RuntimeAttestation]" = weakref.WeakSet()


def _discover_ttt_layers(module: Any) -> tuple[Any, ...]:
    """Find the concrete TTT modules that the verifier will hook.

    This intentionally accepts only an explicit module list supplied by the
    instantiated adapter.  A receipt claiming ``16`` layers is not evidence of
    a sixteen-layer model; the objects themselves must exist and expose a
    framework forward-hook API.
    """
    for name in ("ttt_layers", "ttt_layer_modules", "dit_ttt_layers"):
        candidate = getattr(module, name, None)
        if candidate is None or isinstance(candidate, (str, bytes)):
            continue
        try:
            layers = tuple(candidate)
        except TypeError:
            continue
        if layers:
            return layers
    raise ExactFidelityError(
        "exact runtime attestation requires the instantiated adapter to expose its concrete ttt_layers"
    )


def capture_runtime_attestation(
    module: Any,
    *,
    probe: Callable[[], Any] | None = None,
) -> RuntimeAttestation:
    """Create an attestation only from verifier-installed forward hooks.

    ``probe`` is a zero-argument callable owned by the concrete adapter.  The
    verifier installs hooks on the adapter's 16 actual TTT modules, executes
    the probe once, and records which module identities fired.  A model-set
    boolean or a hand-written ``runtime_receipt`` therefore cannot unlock the
    exact label.
    """
    if not callable(probe):
        raise ExactFidelityError(
            "exact runtime attestation requires a verifier-driven forward pass via a probe callable; "
            "a model-set forward flag is insufficient"
        )
    layers = _discover_ttt_layers(module)
    if len(layers) != 16:
        raise ExactFidelityError(f"runtime topology exposes {len(layers)} TTT layers; expected 16")
    if len({id(layer) for layer in layers}) != len(layers):
        raise ExactFidelityError("runtime topology reuses a TTT module identity across layer slots")
    fired: list[int] = []
    hooks: list[Any] = []
    for index, layer in enumerate(layers):
        register_hook = getattr(layer, "register_forward_hook", None)
        if not callable(register_hook):
            raise ExactFidelityError(f"ttt layer {index} does not support framework forward hooks")

        def _record(_module: Any, _inputs: Any, _output: Any, *, _index: int = index) -> None:
            fired.append(_index)

        hooks.append(register_hook(_record))
    try:
        probe()
    except Exception as exc:
        raise ExactFidelityError(f"verifier-driven runtime probe failed: {exc}") from exc
    finally:
        for hook in hooks:
            remove = getattr(hook, "remove", None)
            if callable(remove):
                remove()
    if fired != list(range(16)):
        raise ExactFidelityError(
            "runtime probe did not execute each contextual TTT layer exactly once; "
            f"observed layer indices={fired!r}"
        )
    getter = getattr(module, "runtime_receipt", None)
    if not callable(getter):
        raise ExactFidelityError("instantiated adapter does not expose runtime_receipt()")
    receipt = getter()
    if not isinstance(receipt, Mapping):
        raise ExactFidelityError("module runtime_receipt() must return a mapping")
    payload = dict(receipt)
    payload["runtime_trace"] = {
        "hooked_layer_count": len(layers),
        "fired_layer_indices": fired,
        "layer_object_ids": [id(layer) for layer in layers],
    }
    attestation = RuntimeAttestation(payload, _ATTESTATION_TOKEN, module)
    _ISSUED_ATTESTATIONS.add(attestation)
    return attestation


def validate_runtime_receipt(
    artifact: "ReferenceArtifactManifest",
    receipt: RuntimeAttestation | Mapping[str, Any] | None,
    *,
    module: Any | None = None,
    mode: str | None = None,
    optimizer_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Require the instantiated adapter to match the official artifact.

    ``scope=single_layer_kernel`` is intentionally rejected: the paper inserts
    TTT in all 16 DiT layers, and a standalone kernel is only an algorithmic
    component.  Nested values are compared exactly after JSON normalization;
    no inferred defaults are accepted in exact mode.
    """
    if not isinstance(receipt, RuntimeAttestation) or receipt not in _ISSUED_ATTESTATIONS:
        raise ExactFidelityError(
            "exact RoboTTT runtime receipt must be an opaque attestation captured from the instantiated adapter"
        )
    receipt_obj = receipt
    if module is not None and receipt_obj._module_id != id(module):
        raise ExactFidelityError("runtime attestation belongs to a different module instance")
    receipt = receipt_obj._as_mapping()
    trace = receipt.get("runtime_trace")
    if not isinstance(trace, Mapping) or trace.get("hooked_layer_count") != 16:
        raise ExactFidelityError("runtime receipt lacks verifier-generated 16-layer forward trace")
    fired = trace.get("fired_layer_indices")
    if fired != list(range(16)):
        raise ExactFidelityError("runtime receipt forward trace did not observe all 16 TTT layers")
    scope = receipt.get("scope")
    layer_count = receipt.get("ttt_layer_count", receipt.get("dit_layers_with_ttt"))
    if scope != "full_16_layer_contextual" or layer_count != 16:
        raise ExactFidelityError(
            "runtime receipt is not full RoboTTT: expected scope='full_16_layer_contextual' and 16 TTT layers; "
            f"got scope={scope!r}, layer_count={layer_count!r}"
        )
    fields = receipt.get("fields")
    if not isinstance(fields, Mapping):
        raise ExactFidelityError("runtime receipt must contain a fields mapping")
    adapter_metadata = receipt.get("adapter_metadata")
    if not isinstance(adapter_metadata, Mapping):
        raise ExactFidelityError("exact mode requires complete adapter_metadata in the runtime receipt")
    adapter_required = (
        "image_preprocessing", "state_layout", "orientation_representation", "instruction_format",
        "prompt_template", "action_chunk_horizon", "denoising_steps", "processor_revision",
        "processor_config_digest", "native_action_objective", "compatibility_key",
        "target_task_finetuned",
    )
    missing_adapter = [name for name in adapter_required if adapter_metadata.get(name) in (None, "")]
    if missing_adapter:
        raise ExactFidelityError("runtime adapter metadata is incomplete: " + ", ".join(missing_adapter))
    if adapter_metadata.get("target_task_finetuned") is not False:
        raise ExactFidelityError(
            "exact zero-shot evaluation requires target_task_finetuned=false in adapter provenance"
        )
    if adapter_metadata.get("native_action_objective") != "flow_matching":
        raise ExactFidelityError(
            "exact RoboTTT requires a DiT flow-matching action head; native objective is "
            f"{adapter_metadata.get('native_action_objective')!r}"
        )
    missing = [name for name in RUNTIME_REQUIRED_FIELDS if fields.get(name) in (None, "")]
    if missing:
        raise ExactFidelityError("runtime receipt has unresolved fields: " + ", ".join(missing))
    for name in RUNTIME_REQUIRED_FIELDS:
        expected = artifact.fields.get(name)
        if expected in (None, ""):
            raise ExactFidelityError(f"official artifact is missing runtime field {name}")
        if _canonical_compare(expected) != _canonical_compare(fields[name]):
            raise ExactFidelityError(
                f"runtime/artifact mismatch for {name}: expected {expected!r}, got {fields[name]!r}"
            )

    runtime_optimizer = receipt.get("optimizer_metadata")
    if not isinstance(runtime_optimizer, Mapping):
        raise ExactFidelityError("exact mode requires optimizer_metadata in the runtime receipt")
    if mode is not None:
        expected = {
            "pretrain": {
                "optimizer": "AdamW", "weight_decay": 1e-5, "peak_learning_rate": 2e-5,
                "scheduler": "WSD", "trainable_parameter_rule": "sequence_layers_only", "steps": 30000,
            },
            "posttrain": {
                "optimizer": "AdamW", "weight_decay": 1e-5, "peak_learning_rate": 5e-5,
                "scheduler": "cosine", "trainable_parameter_rule": "all_parameters", "steps": 20000,
            },
        }.get(mode)
        if expected is None:
            raise ExactFidelityError(f"unknown optimizer mode {mode!r}")
        actual = runtime_optimizer.get(mode, runtime_optimizer) if isinstance(runtime_optimizer, Mapping) else {}
        if not isinstance(actual, Mapping):
            raise ExactFidelityError(f"runtime optimizer metadata for {mode} is not a mapping")
        if optimizer_metadata:
            expected = {**expected, **dict(optimizer_metadata)}
        mismatches = [key for key, value in expected.items() if _canonical_compare(actual.get(key)) != _canonical_compare(value)]
        if mismatches:
            raise ExactFidelityError("runtime optimizer metadata mismatch: " + ", ".join(mismatches))
        artifact_optimizer = artifact.fields.get("optimizer_betas_eps_clip")
        if not isinstance(artifact_optimizer, Mapping):
            raise ExactFidelityError(
                "official artifact must provide optimizer_betas_eps_clip as a mapping including gradient clipping"
            )
        if isinstance(artifact_optimizer.get(mode), Mapping):
            artifact_optimizer = artifact_optimizer[mode]
        actual_optimizer = actual.get("betas_eps_clip", actual.get("optimizer_betas_eps_clip"))
        if not isinstance(actual_optimizer, Mapping):
            raise ExactFidelityError(f"runtime optimizer metadata for {mode} lacks betas_eps_clip")
        required_optimizer_fields = ("betas", "eps", "clip_grad_norm")
        missing_optimizer = [name for name in required_optimizer_fields if name not in artifact_optimizer]
        if missing_optimizer:
            raise ExactFidelityError(
                f"official artifact optimizer metadata for {mode} lacks: {', '.join(missing_optimizer)}"
            )
        if _canonical_compare(actual_optimizer) != _canonical_compare(artifact_optimizer):
            raise ExactFidelityError(f"runtime/artifact optimizer betas/epsilon/clipping mismatch for {mode}")
    # Bind the opaque capability to the exact immutable manifest object that
    # was checked.  ``ReferenceArtifactManifest.label`` will not accept a
    # receipt validated against a different checkpoint/configuration object.
    receipt_obj._validated_artifact_id = id(artifact)


def _canonical_compare(value: Any) -> Any:
    """Normalize nested JSON-like values without stringifying arrays/models."""
    if isinstance(value, Mapping):
        return {str(key): _canonical_compare(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical_compare(item) for item in value]
    return value


@dataclass(frozen=True)
class ReferenceArtifactManifest:
    """Immutable identity/configuration of the code and model used in a run."""

    paper_id: str = PAPER_ID
    paper_url: str = PAPER_URL
    checkpoint_uri: str | None = None
    checkpoint_sha256: str | None = None
    code_uri: str | None = None
    code_commit: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)
    # Omitted means "derive unresolved values from missing fields".  A caller
    # may still list fields explicitly when an official artifact says they are
    # unavailable; those remain blocking for an exact run.
    unresolved_fields: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, path: str | Path) -> "ReferenceArtifactManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        unresolved = tuple(payload.get("unresolved_fields", ()))
        return cls(
            paper_id=payload.get("paper_id", PAPER_ID),
            paper_url=payload.get("paper_url", PAPER_URL),
            checkpoint_uri=payload.get("checkpoint_uri"),
            checkpoint_sha256=payload.get("checkpoint_sha256"),
            code_uri=payload.get("code_uri"),
            code_commit=payload.get("code_commit"),
            fields=payload.get("fields", {}),
            unresolved_fields=unresolved,
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def unresolved(self) -> tuple[str, ...]:
        missing: list[str] = []
        # Checkpoint URI/hash are first-class manifest fields; all other
        # values live in the artifact's ``fields`` mapping.  Treating the
        # first two as absent from ``fields`` would make a valid manifest
        # impossible to satisfy even when its top-level values are present.
        for name in REQUIRED_ARTIFACT_FIELDS:
            if name in self.unresolved_fields:
                missing.append(name)
                continue
            value = self.checkpoint_uri if name == "checkpoint_uri" else self.checkpoint_sha256 if name == "checkpoint_sha256" else self.fields.get(name)
            if not value:
                missing.append(name)
        return tuple(dict.fromkeys(missing))

    def exact_ready(self) -> bool:
        return bool(self.checkpoint_uri and self.checkpoint_sha256 and self.code_commit and not self.unresolved())

    def require_exact(self) -> None:
        if not self.exact_ready():
            missing = ", ".join(self.unresolved() or ("checkpoint_uri/checkpoint_sha256/code_commit",))
            raise ExactFidelityError(
                "Exact RoboTTT training is blocked. Resolve the official artifact fields before running: " + missing
            )
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(self.checkpoint_sha256)):
            raise ExactFidelityError("checkpoint_sha256 must be a 64-character SHA-256 digest")
        # Verify local artifacts before any model is instantiated.  Directories
        # (for example sharded HF checkpoints) are hashed deterministically by
        # verify_checkpoint_hash as well; checking only files would allow a
        # changed directory checkpoint through the exact gate.
        checkpoint = str(self.checkpoint_uri)
        if "://" not in checkpoint:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.exists():
                raise ExactFidelityError(f"official checkpoint does not exist: {checkpoint}")
            if not (checkpoint_path.is_file() or checkpoint_path.is_dir()):
                raise ExactFidelityError(f"official checkpoint is neither a file nor directory: {checkpoint}")
            verify_checkpoint_hash(checkpoint_path, str(self.checkpoint_sha256))
        elif self.fields.get("remote_checkpoint_verification") is not True:
            raise ExactFidelityError(
                "remote exact checkpoints require fields.remote_checkpoint_verification=true "
                "from the host launcher after digest verification"
            )

    def label(self, requested_exact: bool, *, runtime_attested: RuntimeAttestation | None = None) -> str:
        """Return a label only after complete runtime attestation.

        A checkpoint manifest alone does not prove that the instantiated model
        has the 16 contextual TTT layers, matching preprocessing, and the
        published optimizer.  Callers must pass ``runtime_attested=True`` only
        after :func:`validate_runtime_receipt` succeeds.
        """
        if requested_exact:
            if (
                not isinstance(runtime_attested, RuntimeAttestation)
                or runtime_attested not in _ISSUED_ATTESTATIONS
                or runtime_attested._validated_artifact_id != id(self)
            ):
                raise ExactFidelityError(
                    "cannot attest robottt_exact from an artifact manifest alone; pass a verifier-generated runtime receipt (RuntimeAttestation)"
                )
            self.require_exact()
            return "robottt_exact"
        return "robottt_algorithmic_port"

    def as_provenance(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "paper_url": self.paper_url,
            "checkpoint_uri": self.checkpoint_uri,
            "checkpoint_sha256": self.checkpoint_sha256,
            "code_uri": self.code_uri,
            "code_commit": self.code_commit,
            "resolved_fields": dict(self.fields),
            "unresolved_fields": list(self.unresolved()),
        }


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a checkpoint file or directory deterministically."""
    target = Path(path)
    if target.is_dir():
        digest = hashlib.sha256()
        files = sorted(item for item in target.rglob("*") if item.is_file())
        if not files:
            raise FileNotFoundError(f"checkpoint directory contains no files: {target}")
        for item in files:
            relative = item.relative_to(target).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(chunk_size), b""):
                    digest.update(block)
        return digest.hexdigest()
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_hash(path: str | Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ExactFidelityError(f"Checkpoint hash mismatch: expected {expected_sha256}, got {actual}")
