"""Native SmolVLA proposal adapter with an explicit action-chunk queue.

No LeRobot, Torch, or SmolVLA module is imported until an injected policy is
used.  Production launchers can pass the policy/pre/post-processors loaded by
``automatic_ttt.smolvla_arrow_factory``; tests can inject a tiny callable.
"""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import math
import random
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import ActionProposal, ObservationFrame, StepRecord, ContractError, clip_action


_NATIVE_LOAD_ERROR_MAX = 256
_SENSITIVE_ERROR_VALUE = re.compile(
    r"(?i)\b(api[_-]?key|token|password|passwd|secret|authorization|cookie)\b"
    r"(\s*[:=]\s*)[^\s,;]+"
)
_BEARER_ERROR_VALUE = re.compile(r"(?i)\b(bearer\s+)[^\s,;]+")


def _native_load_exception_detail(exc: BaseException) -> str:
    """Return bounded, single-line diagnostics for native loader failures.

    Native model loading crosses several optional dependency boundaries.  The
    public ``ContractError`` must retain the original exception as its cause,
    while exposing enough detail for a Legion log to identify the failing
    dependency/version.  Keep this detail intentionally bounded and redact
    common credential-shaped values; callers should never need a traceback or
    an unbounded model/config dump to diagnose a load failure.
    """
    message = str(exc).strip()
    message = "".join(char if char.isprintable() and char not in "\r\n\t" else " " for char in message)
    message = re.sub(r"\s+", " ", message)
    message = _BEARER_ERROR_VALUE.sub(r"\1<redacted>", message)
    message = _SENSITIVE_ERROR_VALUE.sub(r"\1\2<redacted>", message)
    if not message:
        message = "<no message>"
    if len(message) > _NATIVE_LOAD_ERROR_MAX:
        message = message[: _NATIVE_LOAD_ERROR_MAX - 1].rstrip() + "…"
    return f"{type(exc).__name__}: {message}"


def _proposal(action: Sequence[float], frame: ObservationFrame, producer: str, **metadata: Any) -> ActionProposal:
    return ActionProposal(action, policy_id=producer, timestep=frame.timestep,
                          metadata=metadata, observation_digest=frame.digest)


def _state_pair(component: Any, *, name: str) -> tuple[Callable[[], Any], Callable[[Any], None]] | None:
    snapshot_state = getattr(component, "snapshot_state", None)
    restore_state = getattr(component, "restore_state", None)
    snapshot = getattr(component, "snapshot", None)
    restore = getattr(component, "restore", None)
    if callable(snapshot_state) != callable(restore_state):
        raise ContractError(f"{name} exposes only one of snapshot_state()/restore_state()")
    if callable(snapshot_state):
        return snapshot_state, restore_state
    if callable(snapshot) != callable(restore):
        raise ContractError(f"{name} exposes only one of snapshot()/restore()")
    if callable(snapshot):
        return snapshot, restore
    return None


def _capture_state(component: Any, *, name: str) -> Any:
    hooks = _state_pair(component, name=name)
    return None if hooks is None else hooks[0]()


def _capture_processor_state(component: Any, *, name: str) -> Any:
    """Capture checkpoint-owned processor state without claiming purity.

    LeRobot's processor pipelines do not consistently expose snapshot hooks.
    Their mutable configuration/queue state is nevertheless held in the
    callable object's ``__dict__``.  Capture that dictionary exactly; an
    opaque callable with neither hooks nor state is still rejected unless it
    explicitly declares itself stateless.
    """
    hooks = _state_pair(component, name=name)
    if hooks is not None:
        return ("hooks", hooks[0]())
    if _is_stateless_callable(component):
        return ("stateless", None)
    values = getattr(component, "__dict__", None)
    if isinstance(values, dict) and values:
        try:
            return ("dict", copy.deepcopy(values))
        except Exception as exc:
            raise ContractError(f"{name} state dictionary is not deepcopyable") from exc
    return None


def _restore_state(component: Any, state: Any, *, name: str) -> None:
    hooks = _state_pair(component, name=name)
    if state is not None:
        if hooks is None:
            raise ContractError(f"{name} snapshot exists but no restore hook is available")
        hooks[1](state)


def _restore_processor_state(component: Any, state: Any, *, name: str) -> None:
    if state is None:
        return
    if not isinstance(state, tuple) or len(state) != 2:
        raise ContractError(f"invalid {name} snapshot")
    mode, value = state
    if mode == "hooks":
        hooks = _state_pair(component, name=name)
        if hooks is None:
            raise ContractError(f"{name} snapshot exists but no restore hook is available")
        hooks[1](value)
    elif mode == "dict":
        values = getattr(component, "__dict__", None)
        if not isinstance(values, dict):
            raise ContractError(f"{name} no longer exposes a restorable state dictionary")
        values.clear()
        values.update(copy.deepcopy(value))
    elif mode != "stateless":
        raise ContractError(f"unknown {name} snapshot mode {mode!r}")


def _is_stateless_callable(component: Any) -> bool:
    """Allow a callable without state hooks only with an explicit declaration.

    A Python function is not inherently stateless: closures, globals, and
    mutable callable instances can all affect inference.  Native canary
    rollback therefore treats undeclared callables as mutable and refuses to
    claim complete rollback coverage.
    """
    return callable(component) and bool(getattr(component, "__arrow_stateless__", False))


def _complete_component_state(component: Any, *, name: str) -> bool:
    if component is None:
        return True
    try:
        return _state_pair(component, name=name) is not None or _is_stateless_callable(component)
    except ContractError:
        return False


def _complete_processor_state(component: Any, *, name: str) -> bool:
    if component is None:
        return True
    try:
        if _state_pair(component, name=name) is not None or _is_stateless_callable(component):
            return True
        values = getattr(component, "__dict__", None)
        if isinstance(values, dict) and values:
            copy.deepcopy(values)
            return True
    except (ContractError, Exception):
        return False
    return False


def _native_queue_attrs(policy: Any) -> tuple[str, ...]:
    """Return only LeRobot's mutable inference-cache attributes.

    We deliberately do not deepcopy a whole policy: model parameters and
    compiled modules are expensive, and their weights are frozen during a
    native canary.  The state that changes between proposals is the action
    queue (plus the one-step chunk setting).
    """
    if policy is None:
        return ()
    names: list[str] = []
    for name in ("_queues", "queues", "_action_queue", "_action_queues", "_queue"):
        if hasattr(policy, name):
            names.append(name)
    return tuple(names)


def _native_n_action_steps_owner(policy: Any) -> tuple[Any, str] | None:
    if policy is None:
        return None
    for name in ("config", "cfg"):
        config = getattr(policy, name, None)
        if config is not None and hasattr(config, "n_action_steps"):
            return config, "n_action_steps"
        if isinstance(config, Mapping) and "n_action_steps" in config:
            return config, "n_action_steps"
    if hasattr(policy, "n_action_steps"):
        return policy, "n_action_steps"
    return None


def _force_single_action_step(policy: Any) -> bool:
    """Force a native SmolVLA policy to emit one action per host step."""
    owner = _native_n_action_steps_owner(policy)
    if owner is None:
        return False
    target, name = owner
    try:
        if isinstance(target, Mapping):
            target[name] = 1
        else:
            setattr(target, name, 1)
    except Exception as exc:
        raise ContractError("native SmolVLA n_action_steps is not writable") from exc
    return True


def _capture_native_policy_state(policy: Any) -> tuple[Any, bool]:
    attrs = _native_queue_attrs(policy)
    owner = _native_n_action_steps_owner(policy)
    if not attrs and owner is None:
        return None, False
    queue_values: dict[str, Any] = {}
    try:
        for name in attrs:
            queue_values[name] = copy.deepcopy(getattr(policy, name))
        steps = None if owner is None else copy.deepcopy(owner[0][owner[1]] if isinstance(owner[0], Mapping) else getattr(owner[0], owner[1]))
    except Exception as exc:
        raise ContractError("cannot snapshot native SmolVLA queue state") from exc
    return {"queues": queue_values, "n_action_steps": steps}, True


def _restore_native_policy_state(policy: Any, state: Any) -> None:
    if state is None or policy is None:
        return
    try:
        for name, value in state.get("queues", {}).items():
            setattr(policy, name, copy.deepcopy(value))
        owner = _native_n_action_steps_owner(policy)
        if owner is not None and state.get("n_action_steps") is not None:
            target, name = owner
            if isinstance(target, Mapping):
                target[name] = state["n_action_steps"]
            else:
                setattr(target, name, state["n_action_steps"])
    except Exception as exc:
        raise ContractError("cannot restore native SmolVLA queue state") from exc


def _capture_rng_state() -> Any:
    """Capture host RNGs without making NumPy/Torch hard dependencies."""
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    try:
        import torch
        state["torch"] = torch.random.get_rng_state().clone()
        state["torch_cuda"] = tuple(value.clone() for value in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None
    except (ImportError, RuntimeError):
        state["torch"] = None
        state["torch_cuda"] = None
    return state


def _restore_rng_state(state: Any) -> None:
    if not isinstance(state, Mapping):
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    try:
        import numpy as np
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
    except ImportError:
        pass
    try:
        import torch
        if state.get("torch") is not None:
            torch.random.set_rng_state(state["torch"])
        cuda = state.get("torch_cuda")
        if cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda)
    except (ImportError, RuntimeError):
        pass


class SmolVLAInference(Protocol):
    def __call__(self, observation: Mapping[str, Any], step: int) -> Any: ...


@dataclass(frozen=True)
class SmolVLASnapshot:
    pending_actions: tuple[tuple[float, ...], ...]
    next_chunk_id: int
    chunk_horizon: int = 0
    model_state: Any = None
    # Appended for compatibility with snapshots emitted before in-flight
    # proposal rollback was required.
    inflight: ActionProposal | None = None
    inference_state: Any = None
    preprocessor_state: Any = None
    postprocessor_state: Any = None
    rng_state: Any = None
    native_policy_state: Any = None
    clipped_rows: tuple[bool, ...] = ()


def _to_rows(value: Any) -> tuple[tuple[float, ...], ...]:
    """Convert [7] or [chunk,7] model output to canonical normalized actions."""

    rows, _ = _to_rows_with_clipping(value)
    return rows


def _to_rows_with_clipping(value: Any) -> tuple[tuple[tuple[float, ...], ...], tuple[bool, ...]]:
    """Normalize finite SmolVLA rows and retain per-row clipping diagnostics.

    SmolVLA's action head is trained against normalized actions, but a native
    checkpoint can still emit values outside the environment boundary.
    Clipping finite, correctly-shaped rows here is the canonical adapter
    behavior; malformed shapes and non-finite values remain hard errors.
    """

    if isinstance(value, ActionProposal):
        value = value.action
    if isinstance(value, Mapping):
        for key in ("action", "actions", "action_chunk"):
            if key in value:
                return _to_rows_with_clipping(value[key])
        raise ContractError("SmolVLA output mapping lacks action/action_chunk")
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    as_numpy = getattr(value, "numpy", None)
    if callable(as_numpy):
        value = as_numpy()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (str, bytes)):
        raise ContractError("SmolVLA action output must be [7] or [chunk,7]")
    try:
        rows = list(value)
    except TypeError as exc:
        raise ContractError("SmolVLA action output must be [7] or [chunk,7]") from exc
    if not rows:
        raise ContractError("SmolVLA action chunk cannot be empty")
    first = rows[0]
    if isinstance(first, (int, float)):
        rows = [rows]
    normalized: list[tuple[float, ...]] = []
    clipped: list[bool] = []
    for row in rows:
        if isinstance(row, (str, bytes)):
            raise ContractError("SmolVLA action output must contain numeric rows")
        try:
            numeric = tuple(float(item) for item in row)
        except (TypeError, ValueError) as exc:
            raise ContractError("SmolVLA action output must contain numeric rows") from exc
        if len(numeric) != 7:
            raise ContractError("SmolVLA action output rows must have dimension seven")
        if any(not math.isfinite(item) for item in numeric):
            raise ContractError("SmolVLA action output must contain finite numeric rows")
        bounded = clip_action(numeric)
        normalized.append(bounded)
        clipped.append(bounded != numeric)
    return tuple(normalized), tuple(clipped)


def _native_image_tensor(value: Any, *, name: str) -> Any:
    """Convert one canonical RGB image to the tensor contract of SmolVLA.

    The checkpoint-owned LeRobot pipeline starts with ``AddBatchDimension``;
    unlike the vanilla observation processor it does not convert NumPy HWC
    images to tensors or move channels.  ``ObservationFrame`` also freezes
    NumPy arrays, so passing its image through directly both emits a
    non-writable-tensor warning and leaves the model with ``B,H,W,C``.  Keep
    this conversion at the native SmolVLA boundary, where the expected
    ``B,C,H,W``/``[0, 1]`` contract is explicit.

    A floating-point tensor is treated as already normalized, which avoids
    applying the uint8 scaling twice when a caller has already prepared the
    LeRobot representation.
    """

    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - native runtime boundary
        raise ContractError("native SmolVLA image preparation requires NumPy and PyTorch") from exc

    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone()
    else:
        # ``np.array(copy=True)`` is intentional: ObservationFrame protects
        # its arrays with write=False, while torch.as_tensor would otherwise
        # retain the read-only backing buffer.
        try:
            array = np.array(value, copy=True, order="C")
            tensor = torch.from_numpy(array)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{name} must be an RGB image array or tensor") from exc

    channels = (1, 3, 4)
    if tensor.ndim == 3:
        if tensor.shape[-1] in channels and tensor.shape[0] not in channels:
            tensor = tensor.permute(2, 0, 1)
        elif tensor.shape[0] not in channels:
            raise ContractError(f"{name} must be HWC or CHW with an RGB channel dimension, got {tuple(tensor.shape)}")
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        if tensor.shape[-1] in channels and tensor.shape[1] not in channels:
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.shape[1] not in channels:
            raise ContractError(f"{name} must be BHWC or BCHW with an RGB channel dimension, got {tuple(tensor.shape)}")
    else:
        raise ContractError(f"{name} must be a 3-D or 4-D image, got {tuple(tensor.shape)}")

    if tensor.shape[1] != 3:
        raise ContractError(f"{name} must have three RGB channels, got {tuple(tensor.shape)}")
    if tensor.dtype == torch.uint8 or not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32) / 255.0
    else:
        tensor = tensor.to(dtype=torch.float32)
    return tensor.contiguous()


def _native_state_tensor(value: Any) -> Any:
    """Convert canonical 8-D state to the batched tensor expected by SmolVLA."""

    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - native runtime boundary
        raise ContractError("native SmolVLA state preparation requires NumPy and PyTorch") from exc

    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone()
    else:
        try:
            tensor = torch.from_numpy(np.array(value, dtype=np.float32, copy=True, order="C"))
        except (TypeError, ValueError) as exc:
            raise ContractError("SmolVLA observation.state must be numeric") from exc
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[-1] != 8:
        raise ContractError(f"SmolVLA observation.state must have shape [batch, 8], got {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.float32).contiguous()


class SmolVLAAdapter:
    """Adapt native SmolVLA inference to ``VLA.propose/commit/reset``."""

    def __init__(
        self,
        policy: Any = None,
        *,
        inference: SmolVLAInference | None = None,
        preprocessor: Callable[[Mapping[str, Any]], Any] | None = None,
        postprocessor: Callable[[Any], Any] | None = None,
        task_description: str | None = None,
        producer: str = "smolvla",
        force_single_action_step: bool = False,
        base_vla_sha256: str | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        if policy is None and inference is None:
            raise TypeError("SmolVLAAdapter requires an injected policy or inference callable")
        if not producer:
            raise ValueError("producer is required")
        self.policy = policy
        self.inference = inference
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.task_description = task_description
        self.producer = producer
        # Provenance is intentionally explicit and inert: the adapter does
        # not derive or mutate these identities, while native factories can
        # bind learned artifacts to the exact frozen base they loaded.
        self.base_vla_sha256 = base_vla_sha256
        self.checkpoint_path = checkpoint_path
        self._pending: deque[tuple[float, ...]] = deque()
        self._chunk_id = 0
        self._chunk_horizon = 0
        self._clipped_rows: tuple[bool, ...] = ()
        self._inflight: ActionProposal | None = None
        self.n_action_steps = 1 if force_single_action_step else getattr(policy, "n_action_steps", None)
        if force_single_action_step and policy is not None and not _force_single_action_step(policy):
            raise ContractError(
                "native SmolVLA policy exposes no writable n_action_steps; "
                "cannot guarantee one-action rollback semantics"
            )

    @property
    def rollback_completeness(self) -> Mapping[str, bool]:
        """Report whether every mutable injected component is restorable."""
        return {
            "policy": _complete_component_state(self.policy, name="SmolVLA policy") or _capture_native_policy_state(self.policy)[1],
            "inference": _complete_component_state(self.inference, name="SmolVLA inference"),
            "preprocessor": _complete_processor_state(self.preprocessor, name="SmolVLA preprocessor"),
            "postprocessor": _complete_processor_state(self.postprocessor, name="SmolVLA postprocessor"),
        }

    @property
    def rollback_complete(self) -> bool:
        return all(self.rollback_completeness.values())

    @classmethod
    def from_local_checkpoint(
        cls,
        checkpoint: str,
        *,
        device: str = "cuda",
        task_description: str | None = None,
        producer: str = "smolvla",
        base_vla_sha256: str | None = None,
    ) -> "SmolVLAAdapter":
        """Load the repository's pinned local SmolVLA stack on demand.

        The import is intentionally deferred so contract tests and CPU
        preflight do not require LeRobot.  The existing loader remains the
        source of truth for checkpoint-owned preprocessing and postprocessing.
        """
        try:
            from vla_benchmarking.libero.automatic_ttt.smolvla_arrow_factory import (
                _load_local_smolvla_policy,
            )
            policy, preprocessor, postprocessor = _load_local_smolvla_policy(
                checkpoint, device=device,
            )
        except Exception as exc:  # pragma: no cover - native runtime boundary
            raise ContractError(
                "pinned local SmolVLA could not be loaded: "
                f"{_native_load_exception_detail(exc)}; "
                "inject inference for a dependency-light run"
            ) from exc
        return cls(
            policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task_description=task_description,
            producer=producer,
            force_single_action_step=True,
            base_vla_sha256=base_vla_sha256,
            checkpoint_path=str(checkpoint),
        )

    def reset(self) -> None:
        self._pending.clear()
        self._chunk_id = 0
        self._chunk_horizon = 0
        self._clipped_rows = ()
        self._inflight = None
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
        reset = getattr(self.inference, "reset", None)
        if callable(reset):
            reset()

    def _payload(self, frame: ObservationFrame) -> dict[str, Any]:
        observation = frame.observation
        payload: dict[str, Any] = {}
        for key in ("agentview", "wrist", "state"):
            if key in observation:
                payload[key] = observation[key]
        instruction = observation.get("instruction", self.task_description)
        if instruction is not None:
            if not isinstance(instruction, str) or not instruction.strip():
                raise ContractError("SmolVLA observation instruction must be non-empty text")
            payload["instruction"] = instruction
        return payload

    def _infer(self, frame: ObservationFrame) -> Any:
        payload = self._payload(frame)
        if self.inference is not None:
            return self.inference(payload, frame.step)
        policy = self.policy
        select_action = getattr(policy, "select_action", None)
        if callable(select_action):
            if self.preprocessor is None or self.postprocessor is None:
                raise ContractError(
                    "native SmolVLA select_action requires checkpoint-owned preprocessor and postprocessor"
                )
            native_payload = {
                "observation.images.image": _native_image_tensor(
                    payload.get("agentview"), name="observation.images.image"
                ),
                "observation.images.image2": _native_image_tensor(
                    payload.get("wrist"), name="observation.images.image2"
                ),
                "observation.state": _native_state_tensor(payload.get("state")),
                "task": payload.get("instruction", self.task_description),
            }
            return self.postprocessor(select_action(self.preprocessor(native_payload)))
        for name in ("propose", "act", "predict"):
            method = getattr(policy, name, None)
            if callable(method):
                try:
                    return method(payload, frame.step)
                except TypeError:
                    return method(payload)
        if callable(policy):
            try:
                return policy(payload, frame.step)
            except TypeError:
                return policy(payload)
        raise ContractError("injected SmolVLA policy exposes no select_action/propose/act/predict hook")

    def propose(self, frame: ObservationFrame) -> ActionProposal:
        if self._inflight is not None and self._inflight.timestep == frame.timestep:
            return self._inflight
        if self._inflight is not None:
            raise ContractError("previous SmolVLA proposal was not committed")
        if not self._pending:
            rows, clipped_rows = _to_rows_with_clipping(self._infer(frame))
            self._pending.extend(rows)
            self._clipped_rows = clipped_rows
            self._chunk_id += 1
            self._chunk_horizon = len(rows)
        action_index = self._chunk_horizon - len(self._pending)
        action = self._pending.popleft()
        action_clipped = bool(self._clipped_rows[action_index]) if action_index < len(self._clipped_rows) else False
        proposal = _proposal(
            action,
            frame,
            self.producer,
            action_chunk_id=f"{self.producer}-chunk-{self._chunk_id}",
            action_chunk_index=action_index,
            action_chunk_horizon=self._chunk_horizon,
            queued_actions=len(self._pending),
            smolvla_output_clipped=action_clipped,
            smolvla_chunk_clipped=any(self._clipped_rows),
        )
        self._inflight = proposal
        return proposal

    def commit(self, record: StepRecord) -> None:
        if self._inflight is None:
            raise ContractError("SmolVLA commit has no pending proposal")
        if record.base.timestep != self._inflight.timestep or record.base.action != self._inflight.action:
            raise ContractError("SmolVLA commit does not match the pending proposal")
        commit = getattr(self.policy, "commit", None)
        if callable(commit):
            commit(record)
        self._inflight = None

    def invalidate_pending(self, *, reason: str = "external_action") -> None:
        """Discard an unexecuted action chunk after a takeover/hybrid action.

        SmolVLA action chunks are predictions for consecutive future states.
        Once another controller executes an action, every queued row is stale;
        retaining it would silently apply an action predicted for the wrong
        state.  The hook is deliberately explicit so a native host can enforce
        this rule without calling the model's commit hook for an unexecuted
        action.
        """
        self._pending.clear()
        self._clipped_rows = ()
        self._inflight = None
        invalidate = getattr(self.policy, "invalidate_pending", None)
        if callable(invalidate):
            invalidate(reason=reason)
        invalidate = getattr(self.inference, "invalidate_pending", None)
        if callable(invalidate):
            invalidate(reason=reason)

    # Alias used by host adapters that call the operation "queue invalidation".
    invalidate_queue = invalidate_pending

    def snapshot(self) -> SmolVLASnapshot:
        model_snapshot = _capture_state(self.policy, name="SmolVLA policy") if self.policy is not None else None
        native_policy_snapshot, _ = _capture_native_policy_state(self.policy)
        inference_snapshot = _capture_state(self.inference, name="SmolVLA inference") if self.inference is not None else None
        preprocessor_snapshot = _capture_processor_state(self.preprocessor, name="SmolVLA preprocessor") if self.preprocessor is not None else None
        postprocessor_snapshot = _capture_processor_state(self.postprocessor, name="SmolVLA postprocessor") if self.postprocessor is not None else None
        return SmolVLASnapshot(
            tuple(self._pending), self._chunk_id, self._chunk_horizon,
            model_snapshot, self._inflight, inference_snapshot,
            preprocessor_snapshot, postprocessor_snapshot, _capture_rng_state(), native_policy_snapshot,
            tuple(self._clipped_rows),
        )

    def restore(self, snapshot: SmolVLASnapshot) -> None:
        if not isinstance(snapshot, SmolVLASnapshot):
            raise ContractError("SmolVLA restore requires SmolVLASnapshot")
        self._pending = deque(snapshot.pending_actions)
        self._chunk_id = int(snapshot.next_chunk_id)
        self._chunk_horizon = int(snapshot.chunk_horizon or len(self._pending))
        self._clipped_rows = tuple(snapshot.clipped_rows)
        self._inflight = snapshot.inflight
        _restore_state(self.policy, snapshot.model_state, name="SmolVLA policy") if self.policy is not None else None
        _restore_native_policy_state(self.policy, snapshot.native_policy_state)
        _restore_state(self.inference, snapshot.inference_state, name="SmolVLA inference") if self.inference is not None else None
        _restore_processor_state(self.preprocessor, snapshot.preprocessor_state, name="SmolVLA preprocessor") if self.preprocessor is not None else None
        _restore_processor_state(self.postprocessor, snapshot.postprocessor_state, name="SmolVLA postprocessor") if self.postprocessor is not None else None
        _restore_rng_state(snapshot.rng_state)

    def snapshot_state(self) -> SmolVLASnapshot:
        """Alias used by :class:`TransactionalCoordinator` component rollback."""
        return self.snapshot()

    def restore_state(self, snapshot: SmolVLASnapshot) -> None:
        """Alias used by :class:`TransactionalCoordinator` component rollback."""
        self.restore(snapshot)

    def close(self) -> None:
        """Release an injected native policy/inference runtime, if supported."""
        seen: set[int] = set()
        for component in (self.policy, self.inference):
            if component is None or id(component) in seen:
                continue
            seen.add(id(component))
            close = getattr(component, "close", None)
            if callable(close):
                close()


__all__ = ["SmolVLAAdapter", "SmolVLASnapshot", "SmolVLAInference"]
