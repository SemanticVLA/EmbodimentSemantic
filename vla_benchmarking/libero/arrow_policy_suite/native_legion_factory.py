"""Concrete Legion factory for the bounded native Arrow canary.

This is intentionally a thin runtime composition layer.  It reuses the
repository's production LIBERO builder and checkpoint loader; it does not
silently invent a second environment or processor implementation.  The
optional Arrow/graph callables are explicit environment variables so a
canary cannot accidentally claim a teacher that was not loaded.

Environment variables:
``ARROW_SUITE_CHECKPOINT`` (required for VLA policies),
``ARROW_SUITE_DEVICE`` (default ``cuda``), ``ARROW_SUITE_TASK_ID``,
``ARROW_SUITE_SEED``, ``ARROW_SUITE_INSTRUCTION``,
``ARROW_SUITE_CONTROLLER_HASH`` (or ``ARROW_SUITE_CONTROLLER`` pointing at
the canonical config; required by the default teacher),
optional ``ARROW_SUITE_TEACHER_FACTORY``, and
``ARROW_SUITE_GRAPH_FACTORY`` (required for Fast/Trace).
Vision-only Trace additionally requires
``ARROW_SUITE_TRACE_VISION_ENDPOINT_FACTORY``; simulator-assisted RGB-D is
the only Trace path allowed to use simulator-derived endpoint callbacks.
The factory callable receives ``raw_environment=...`` and may return an
already rollback-capable ``InterruptibleArrow`` or compatible teacher.
"""

from __future__ import annotations

import importlib
import inspect
import hashlib
import json
import os
import random
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import ContractError, ObservationFrame
from .interruptible_arrow import InterruptibleArrow
from .libero_adapter import LiberoEnvironmentAdapter
from .native_factory import NativeHostSpec, build_native_host
from .smolvla_adapter import SmolVLAAdapter
from .native_arrow_teacher import PerFrameArrowTeacher, build_rgbd_perception


_LEARNED_POLICY_IDS = {"arrow_apprentice", "arrow_editor", "arrow_minimal_learned"}
_TEACHER_FREE_POLICY_IDS = _LEARNED_POLICY_IDS | {"frozen_base", "arrow_trace"}


def _hash_base_checkpoint(path: str | Path) -> str:
    """Return the hash convention used by persisted learned artifacts.

    Apprentice manifests intentionally exclude caches and the base snapshot
    manifest from a directory hash.  Reusing that implementation here keeps
    the runtime identity check byte-for-byte compatible with training.
    """
    target = Path(path).expanduser()
    if target.is_file():
        return hashlib.sha256(target.read_bytes()).hexdigest()
    if target.is_dir():
        from .apprentice_training import _base_tree_sha256
        return str(_base_tree_sha256(target))
    raise ContractError(f"base VLA checkpoint is missing: {target}")


def _artifact_path(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink() or not path.exists():
        raise ContractError(f"{label} is missing or is a symlink: {path}")
    if not path.is_file() and not path.is_dir():
        raise ContractError(f"{label} must be an immutable file or directory bundle: {path}")
    return path.resolve()


def _artifact_sha256(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    entries = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        entries.append((child.relative_to(path).as_posix(), hashlib.sha256(child.read_bytes()).hexdigest()))
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode("utf-8")).hexdigest()


def _learned_artifact(policy_id: str, values: Sequence[str | Path] | None) -> Path | None:
    supplied = tuple(values or ())
    if policy_id in _LEARNED_POLICY_IDS:
        if len(supplied) != 1:
            raise ContractError(
                f"{policy_id} requires exactly one immutable learned artifact; refusing a fallback"
            )
        return _artifact_path(supplied[0], label="learned artifact")
    if supplied:
        raise ContractError("learned artifacts are only valid for Apprentice, Editor, and Minimal-Learned")
    return None


def _rng_snapshot() -> dict[str, Any]:
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


def _rng_restore(state: Mapping[str, Any]) -> None:
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
        if state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except (ImportError, RuntimeError):
        pass


class _LiveMinimalPolicy:
    """Frame-aware runtime Minimal wrapper without changing public policies."""

    def __init__(self, branch_runner: Any = None, *, max_decisions: int = 60) -> None:
        from .policies import MinimalPolicy
        self._policy = MinimalPolicy(variant="runtime_oracle", branch_runner=branch_runner,
                                     max_decisions=max_decisions)
        self.policy_id = self._policy.policy_id

    @property
    def variant(self) -> str:
        return self._policy.variant

    @property
    def branch_runner(self) -> Any:
        return self._policy.branch_runner

    @branch_runner.setter
    def branch_runner(self, value: Any) -> None:
        self._policy.branch_runner = value

    def decide(self, frame: ObservationFrame, base: Any, teacher: Any) -> Any:
        runner = self.branch_runner
        if runner is not None:
            setter = getattr(runner, "set_frame", None)
            if callable(setter):
                setter(frame)
        return self._policy.decide(frame, base, teacher)

    def commit(self, record: Any) -> None:
        self._policy.commit(record)

    def reset(self) -> None:
        self._policy.reset()

    def snapshot_state(self) -> Any:
        return self._policy.snapshot_state()

    def restore_state(self, state: Any) -> None:
        self._policy.restore_state(state)


def _build_minimal_branch_runner(environment: Any, vla: Any, teacher: Any, policy: Any) -> Any:
    """Build the real, composite-snapshot Minimal-Runtime branch runner."""
    from .branching import BranchRunner
    from .policies import MinimalPolicy

    if teacher is None:
        raise ContractError("Minimal-Runtime requires an Arrow teacher")
    context: dict[str, Any] = {"frame": None, "base": None, "teacher": None,
                               "branch_signal": {}}

    class _BranchEnvironment:
        def step(self, action: Sequence[float]) -> Any:
            raw = environment.step(action)
            frame = context.get("frame")
            base = context.get("base")
            arrow = context.get("teacher")
            if frame is None or base is None or arrow is None:
                raise ContractError("Minimal branch stepped without fresh proposals")
            next_frame = ObservationFrame(
                environment.observe(), timestep=int(frame.timestep) + 1,
                episode_id=frame.episode_id, metadata=frame.metadata,
            )
            record = SimpleNamespace(frame=frame, base=base, teacher=arrow,
                                     next_frame=next_frame, result=raw,
                                     decision=SimpleNamespace(action=tuple(action)))
            if tuple(action) == tuple(base.action):
                commit = getattr(vla, "commit", None)
                if callable(commit):
                    commit(record)
            else:
                invalidate = getattr(vla, "invalidate_pending", None) or getattr(vla, "invalidate_queue", None)
                if callable(invalidate):
                    invalidate(reason="minimal_branch_action")
            if tuple(action) == tuple(arrow.action):
                commit = getattr(teacher, "commit", None)
                if callable(commit):
                    commit(record)
            else:
                interrupt = getattr(teacher, "interrupt", None)
                if callable(interrupt):
                    interrupt()
            return raw

    branch_environment = _BranchEnvironment()

    def proposal(mask: int, index: int, _baseline: Any) -> Sequence[float]:
        source = context.get("frame")
        if not isinstance(source, ObservationFrame):
            raise ContractError("Minimal branch frame was not set by the runtime policy")
        observation = environment.observe()
        frame = ObservationFrame(
            observation, timestep=int(source.timestep) + int(index), episode_id=source.episode_id,
            metadata=source.metadata,
        )
        base = vla.propose(frame)
        arrow = teacher.propose(frame)
        if arrow is None:
            raise ContractError("Minimal-Runtime branch lost the Arrow proposal")
        metadata = dict(getattr(arrow, "metadata", {}) or {})
        signal = context.setdefault("branch_signal", {})
        if int(index) == 0:
            signal.clear()
            signal.update({
                "phase_start": metadata.get("phase_index"),
                "error_start": metadata.get("phase_error"),
            })
        signal.update({
            "phase_end": metadata.get("phase_index", signal.get("phase_end")),
            "error_end": metadata.get("phase_error", signal.get("error_end")),
            "milestone": bool(metadata.get("milestone", metadata.get("milestone_reached", False))),
        })
        context["frame"], context["base"], context["teacher"] = frame, base, arrow
        return MinimalPolicy._apply_mask(policy._policy, base.action, arrow.action,
                                         MinimalPolicy.mask_from_bits(int(mask)))

    def outcome(mask: int, actions: Sequence[Sequence[float]], raws: Sequence[Any]) -> Mapping[str, Any]:
        score = 0.0
        sufficient = False
        for raw in raws:
            if isinstance(raw, Mapping):
                score += float(raw.get("reward", 0.0) or 0.0)
                sufficient = sufficient or bool(raw.get("success", raw.get("task_success", raw.get("is_success", False))))
            elif isinstance(raw, tuple) and len(raw) >= 3:
                score += float(raw[1] or 0.0)
                sufficient = sufficient or bool(raw[3] if len(raw) >= 4 else raw[2])
        checker = getattr(environment, "check_success", None)
        if callable(checker):
            try:
                value = checker()
                sufficient = sufficient or bool(value.get("success", value.get("task_success", False))) if isinstance(value, Mapping) else sufficient or bool(value)
            except Exception:
                pass
        signal = context.get("branch_signal", {})
        phase_start, phase_end = signal.get("phase_start"), signal.get("phase_end")
        error_start, error_end = signal.get("error_start"), signal.get("error_end")
        phase_advanced = (
            isinstance(phase_start, (int, float)) and isinstance(phase_end, (int, float))
            and float(phase_end) > float(phase_start)
        ) or bool(signal.get("milestone", False))
        error_dropped = False
        if isinstance(error_start, (int, float)) and isinstance(error_end, (int, float)):
            error_dropped = float(error_start) > 0.0 and float(error_end) <= float(error_start) * 0.9
        progress = float(score)
        if isinstance(error_start, (int, float)) and isinstance(error_end, (int, float)):
            progress = max(progress, float(error_start) - float(error_end))
        return {"sufficient": bool(sufficient or phase_advanced or error_dropped),
                "progress": progress, "phase_advanced": bool(phase_advanced),
                "phase_error_dropped": bool(error_dropped), "mask": int(mask),
                "fresh_actions": len(actions)}

    def composite_snapshot() -> Mapping[str, Any]:
        return {
            "environment": environment.snapshot(), "vla": vla.snapshot_state(),
            "teacher": teacher.snapshot_state(), "policy": policy.snapshot_state(),
            "rng": _rng_snapshot(),
        }

    def composite_restore(state: Mapping[str, Any]) -> None:
        environment.restore(state["environment"])
        vla.restore_state(state["vla"])
        teacher.restore_state(state["teacher"])
        policy.restore_state(state["policy"])
        _rng_restore(state["rng"])

    runner = BranchRunner(
        branch_environment, masks=range(8), horizon=20, proposal_fn=proposal,
        composite_snapshot=composite_snapshot, composite_restore=composite_restore,
        policies=(policy,), rng_snapshot=_rng_snapshot, rng_restore=_rng_restore,
        require_state_isolation=True, outcome_fn=outcome,
    )
    runner.set_frame = lambda frame: context.__setitem__("frame", frame)  # type: ignore[attr-defined]
    return runner


def _runtime_trace_callbacks(
    raw_environment: Any, *, task_id: int, resolution: int,
    frame_name: str, calibration_revision: str,
    endpoint_detector: Callable[..., Any] | None = None,
    simulator_assisted: bool = False,
) -> tuple[Callable[..., Any], Callable[..., Any], None]:
    """Build synchronized RGB-D callbacks from the live production camera.

    The canonical LIBERO bbox/arrow renderer is simulator-derived.  It is
    therefore available only when ``simulator_assisted`` is explicit; a
    vision-only Trace must inject a detector operating on the captured RGB-D
    packet.
    """
    if endpoint_detector is None and not simulator_assisted:
        raise ContractError(
            "vision-only Trace requires an explicit RGB-D endpoint detector"
        )
    from vla_benchmarking.libero.arrow_policy_suite.rgbd_geometry import ArrowRGBDObservation
    from vla_benchmarking.libero.evaluation import run_arrow_pick_place_eval as episode
    from vla_benchmarking.libero.evaluation import run_arrow_pick_place_matrix as matrix

    def capture(frame: ObservationFrame) -> Any:
        packet = episode.capture_agentview(raw_environment, resolution=int(resolution), camera_name="agentview")
        calibration = packet.calibration
        return ArrowRGBDObservation(
            rgb=packet.rgb, depth_m=packet.metric_depth,
            intrinsics=calibration.intrinsic,
            world_from_camera=calibration.world_from_camera,
            frame_name=frame_name, calibration_revision=calibration_revision,
            timestamp_s=time.time(), camera_id=str(calibration.camera_name),
            metadata={"frame_timestep": frame.timestep},
        )

    def endpoint(frame: ObservationFrame, capture_packet: Any) -> Mapping[str, Any]:
        if endpoint_detector is not None:
            try:
                value = endpoint_detector(frame, capture_packet)
            except TypeError:
                value = endpoint_detector(frame=frame, capture_packet=capture_packet)
            if not isinstance(value, Mapping):
                raise ContractError("vision-only Trace endpoint detector must return a mapping")
            provenance = dict(value.get("provenance", {}))
            if provenance.get("vision_only") is not True:
                raise ContractError(
                    "vision-only Trace endpoint detector must attest provenance.vision_only=true"
                )
            provenance["source"] = "injected_vision_endpoint_detector"
            return {**value, "provenance": provenance}
        inputs = matrix._default_arrow_inputs(
            raw_environment, int(task_id), int(resolution), record_on_env=False
        )
        rendered, _ = episode.render_exactly_one_arrow(
            capture_packet.rgb, inputs["bboxes"], subject=inputs["subject"],
            goal_object=inputs["goal_object"], anchor_policy="bbox_center",
        )
        source, destination = episode.decode_arrow_pixels(capture_packet.rgb, rendered)
        return {"source_xy": tuple(float(v) for v in source),
                "destination_xy": tuple(float(v) for v in destination),
                "provenance": {"source": "canonical_simulator_endpoint", "vision_only": False}}

    return capture, endpoint, None


def _vision_endpoint_detector(
    spec: str | Callable[..., Any], *, frame_name: str,
    calibration_revision: str, run_dir: str | Path,
    camera_name: str = "agentview",
) -> Callable[..., Any]:
    """Resolve a vision-only endpoint detector without simulator access.

    A factory may receive only camera/calibration/run metadata.  Any factory
    advertising an environment-shaped parameter is rejected before it can be
    called.  A detector callable itself (``frame, capture_packet``) is
    returned unchanged and is still required to attest vision-only output at
    invocation time.
    """
    factory = _callable(spec, label="Trace vision endpoint factory") if isinstance(spec, str) else spec
    if not callable(factory):
        raise ContractError("Trace vision endpoint factory must be callable")
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        parameters = {}
    forbidden = {"raw_environment", "environment", "env", "simulator", "sim_state", "mujoco"}
    if forbidden.intersection(parameters):
        raise ContractError("Trace vision endpoint factory may not receive a simulator/environment object")
    detector_parameters = {"frame", "capture_packet", "capture", "observation"}
    is_detector = bool(detector_parameters.intersection(parameters))
    if is_detector:
        detector = factory
    else:
        allowed = {
            "frame_name": frame_name,
            "calibration_revision": calibration_revision,
            "run_dir": Path(run_dir),
            "camera_name": camera_name,
        }
        required = [
            parameter for parameter in parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        unknown = [parameter.name for parameter in required if parameter.name not in allowed]
        if unknown:
            raise ContractError(
                "Trace vision endpoint factory requires unsupported non-simulator parameters: "
                + ", ".join(unknown)
            )
        detector = factory(**{name: value for name, value in allowed.items() if name in parameters})
    if not callable(detector):
        raise ContractError("Trace vision endpoint factory must return a callable detector")
    return detector


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return None if value is None or not str(value).strip() else str(value).strip()


def _identity_int(name: str, *, operation: str, default: int) -> int:
    """Resolve launcher identity without hidden collect/evaluate defaults."""
    raw = _env(name)
    if raw is None:
        if operation in {"collect", "evaluate"}:
            raise ContractError(f"{name} is required explicitly for {operation}")
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ContractError(f"{name} must be an integer") from exc
    if value < 0:
        raise ContractError(f"{name} must be non-negative")
    return value


def _callable(spec: str, *, label: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ContractError(f"{label} must be module:callable")
    module_name, attr = spec.split(":", 1)
    try:
        value: Any = importlib.import_module(module_name)
        for part in attr.split("."):
            value = getattr(value, part)
    except (ImportError, AttributeError) as exc:
        raise ContractError(f"cannot load {label} {spec!r}") from exc
    if not callable(value):
        raise ContractError(f"{label} {spec!r} is not callable")
    return value


def _graph_callback(spec: str | None) -> Callable[[ObservationFrame], Mapping[str, Any] | None] | None:
    if spec is None:
        return None
    callback = _callable(spec, label="graph factory")

    def provide(frame: ObservationFrame) -> Mapping[str, Any] | None:
        try:
            accepts_named_frame = "frame" in inspect.signature(callback).parameters
        except (TypeError, ValueError):
            accepts_named_frame = False
        value = callback(frame=frame) if accepts_named_frame else callback(frame)
        if value is not None and not isinstance(value, Mapping):
            raise ContractError("graph factory must return a mapping or None")
        return value

    return provide


def _controller_hash() -> str:
    explicit = _env("ARROW_SUITE_CONTROLLER_HASH")
    if explicit is not None:
        return explicit
    path = _env("ARROW_SUITE_CONTROLLER")
    if path is None:
        raise ContractError("ARROW_SUITE_CONTROLLER_HASH or ARROW_SUITE_CONTROLLER is required for the default Arrow teacher")
    try:
        from vla_benchmarking.libero.arrow_grasp_controller.configs import load_controller_config
        payload = load_controller_config(Path(path).expanduser().resolve())
        value = str(payload.get("config_hash", ""))
    except Exception as exc:
        raise ContractError("ARROW_SUITE_CONTROLLER could not be resolved to a canonical config hash") from exc
    if len(value) != 64:
        raise ContractError("resolved Arrow controller config hash is invalid")
    return value


def _trace_calibration_contract(path: str | Path) -> tuple[str, str]:
    target = Path(path).expanduser().resolve()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        frame_name = str(payload["frame_name"])
        revision = str(payload["revision"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ContractError("Trace calibration artifact must expose frame_name and revision") from exc
    if not frame_name.strip() or not revision.strip():
        raise ContractError("Trace calibration frame_name/revision must be non-empty")
    return frame_name, revision


def build_host(*, config: Any, operation: str, policy_id: str, run_dir: str | Path,
               max_steps: int, learned_artifacts: Sequence[str | Path] = ()) -> Any:
    """Build one real, rollback-capable NativeHost for the Legion canary.

    Learned policy artifacts are an explicit one-file input.  Their sidecar
    manifests are verified by the shared loaders against the hash of the
    frozen base checkpoint loaded below; no learned policy is ever replaced by
    the frozen VLA or a full-teacher path.
    """
    learned_artifact = _learned_artifact(policy_id, learned_artifacts)
    checkpoint = _env("ARROW_SUITE_CHECKPOINT")
    if checkpoint is None:
        raise ContractError("ARROW_SUITE_CHECKPOINT is required for native SmolVLA canary")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    base_vla_sha256 = _hash_base_checkpoint(checkpoint_path)
    declared_base_hash = _env("ARROW_SUITE_BASE_VLA_SHA256")
    if declared_base_hash is not None and declared_base_hash != base_vla_sha256:
        raise ContractError("ARROW_SUITE_BASE_VLA_SHA256 does not match the loaded base checkpoint")
    task_id = _identity_int(
        "ARROW_SUITE_TASK_ID", operation=operation, default=int(config.task_ids[0])
    )
    seed = _identity_int(
        "ARROW_SUITE_SEED", operation=operation, default=int(config.learned_seeds[0])
    )
    resolution = int(_env("ARROW_SUITE_RESOLUTION", str(config.image_resolution[0])))
    init_state_raw = _env("ARROW_SUITE_INIT_STATE_INDEX")
    if init_state_raw is None:
        if operation in {"evaluate", "collect"}:
            raise ContractError("ARROW_SUITE_INIT_STATE_INDEX is required for evaluate/collect")
        init_state_index = 1
    else:
        try:
            init_state_index = int(init_state_raw)
        except ValueError as exc:
            raise ContractError("ARROW_SUITE_INIT_STATE_INDEX must be an integer") from exc
    if init_state_index < 0:
        raise ContractError("ARROW_SUITE_INIT_STATE_INDEX must be non-negative")
    suite_mode = _env("ARROW_SUITE_SUITE_MODE", "vanilla")
    instruction = _env("ARROW_SUITE_INSTRUCTION")
    device = _env("ARROW_SUITE_DEVICE", "cuda")
    if instruction is None:
        try:
            from vla_benchmarking.libero.evaluation.native_vla_eval import _task_description
            instruction = _task_description(task_id, {"suite_mode": suite_mode})
        except Exception as exc:
            raise ContractError(
                "ARROW_SUITE_INSTRUCTION is unset and the canonical LIBERO task prompt could not be resolved"
            ) from exc
        if not isinstance(instruction, str) or not instruction.strip():
            raise ContractError("resolved LIBERO task instruction is empty")

    # Imports are deliberately deferred until --execute on a configured node.
    from vla_benchmarking.libero.evaluation.run_arrow_pick_place_eval import build_libero_env

    raw_environment = build_libero_env(
        task_id, seed, resolution, suite_mode=suite_mode,
        extra_camera_names=("robot0_eye_in_hand",), init_state_index=init_state_index,
    )
    try:
        def annotate(host: Any) -> Any:
            # These fields are consumed only as immutable run provenance; the
            # NativeHost action/rollback contract remains unchanged.
            host.base_vla_sha256 = base_vla_sha256
            host.task_id = task_id
            host.seed = seed
            host.init_state_index = init_state_index
            if learned_artifact is not None:
                host.learned_artifact_path = str(learned_artifact)
                host.learned_artifact_sha256 = _artifact_sha256(learned_artifact)
            return host

        environment = LiberoEnvironmentAdapter.from_live_libero(
            raw_environment, instruction=instruction, require_images=True, strict_snapshot=True,
        )
        vla = SmolVLAAdapter.from_local_checkpoint(
            checkpoint, device=device, task_description=instruction,
            base_vla_sha256=base_vla_sha256,
        )
        # Make base identity available to executor-side manifests and any
        # injected adapter loader without changing the VLA action contract.
        teacher = None
        policy: Any | None = None
        if policy_id in _LEARNED_POLICY_IDS:
            if learned_artifact is None:  # defensive: _learned_artifact already checks this
                raise ContractError(f"{policy_id} requires an immutable learned artifact")
            from .native_learned_training import build_native_learned_policy
            adapter_factory = None
            if policy_id == "arrow_apprentice":
                loader_spec = _env("ARROW_SUITE_ADAPTER_LOADER") or _env("ARROW_SUITE_APPRENTICE_LOADER")
                from .apprentice_training import load_apprentice_runtime_adapter
                if loader_spec is None:
                    adapter_factory = lambda path, base_hash: load_apprentice_runtime_adapter(
                        path, base_vla_sha256=base_hash, base_checkpoint=checkpoint_path, device=device
                    )
                else:
                    adapter_loader = _callable(loader_spec, label="Apprentice adapter loader")
                    adapter_factory = lambda path, base_hash: load_apprentice_runtime_adapter(
                        path, base_vla_sha256=base_hash, adapter_loader=adapter_loader
                    )
            if policy_id == "arrow_apprentice" and learned_artifact.is_dir():
                from .policies import ApprenticePolicy
                action_fn = adapter_factory(learned_artifact, base_vla_sha256)
                policy = ApprenticePolicy(action_fn=action_fn)
            else:
                policy = build_native_learned_policy(
                    policy_id, checkpoint_path=learned_artifact,
                    base_vla_sha256=base_vla_sha256,
                    adapter_action_fn_factory=adapter_factory,
                )
            policy.learned_artifact_path = str(learned_artifact)
            policy.learned_artifact_sha256 = _artifact_sha256(learned_artifact)
            policy.base_vla_sha256 = base_vla_sha256
        elif policy_id not in {"frozen_base", "arrow_trace"}:
            teacher_spec = _env("ARROW_SUITE_TEACHER_FACTORY")
            if teacher_spec is not None:
                candidate = _callable(teacher_spec, label="Arrow teacher factory")(
                    raw_environment=raw_environment, run_dir=Path(run_dir), task_id=task_id, seed=seed,
                )
                teacher = candidate if isinstance(candidate, InterruptibleArrow) else InterruptibleArrow(candidate)
            else:
                # Default production path: reuse the pinned Molmo/RGB-D
                # runtime's calibration and worker, but replace its
                # whole-episode runner with a proposal/commit phase machine.
                config_hash = _controller_hash()
                from vla_benchmarking.libero.automatic_ttt.smolvla_arrow_factory import _build_arrow_teacher_factory
                # The legacy loader names its config variable differently;
                # bridge it only for construction and restore the process
                # environment immediately afterwards.
                previous_controller_config = os.environ.get("ARROW_CONTROLLER_CONFIG")
                suite_controller = _env("ARROW_SUITE_CONTROLLER")
                if previous_controller_config is None and suite_controller is not None:
                    os.environ["ARROW_CONTROLLER_CONFIG"] = suite_controller
                try:
                    _legacy_factory, session = _build_arrow_teacher_factory(
                        output_root=Path(run_dir), controller_config_hash=config_hash,
                        resolution=resolution, device=device,
                    )
                finally:
                    if previous_controller_config is None:
                        os.environ.pop("ARROW_CONTROLLER_CONFIG", None)
                prepare = getattr(_legacy_factory, "prepare_for_environment", None)
                if not callable(prepare):
                    raise ContractError("pinned Arrow runtime did not expose calibration preparation")
                prepare(raw_environment)
                if session.worker is None:
                    raise ContractError("pinned Arrow runtime did not create a perception worker")
                perception = build_rgbd_perception(
                    raw_environment=raw_environment, worker=session.worker,
                    task_id=task_id, resolution=resolution, output_dir=Path(run_dir),
                )
                teacher = PerFrameArrowTeacher(
                    perception, eef_orientation_transform=session.transform,
                    cleanup_attempt=session.cleanup_attempt,
                )
        if policy_id in {"arrow_minimal", "arrow_minimal_runtime"}:
            runtime_policy = _LiveMinimalPolicy(max_decisions=int(_env("ARROW_SUITE_MINIMAL_MAX_DECISIONS", "60")))
            runtime_policy.branch_runner = _build_minimal_branch_runner(environment, vla, teacher, runtime_policy)
            policy = runtime_policy
        graph_context_fn = _graph_callback(_env("ARROW_SUITE_GRAPH_FACTORY"))
        if policy_id == "arrow_fast":
            if graph_context_fn is None:
                raise ContractError("native Fast requires ARROW_SUITE_GRAPH_FACTORY")
            from .native_fast_factory import build_fast_native_components
            bundle = build_fast_native_components(
                environment, vla, teacher, graph_context_fn=graph_context_fn,
            )
            host = build_native_host(NativeHostSpec(
                environment=environment, vla=vla, teacher=None,
                policy_id="arrow_fast", policy=bundle.policy,
                graph_context_fn=graph_context_fn,
            ))
            # Preserve the one-shot support evidence for native_executor's
            # preflight/manifest after the teacher is detached.
            host.fast_lifecycle_receipt = bundle.receipt
            return annotate(host)
        if policy_id == "arrow_trace":
            route_artifact = _env("ARROW_SUITE_TRACE_ROUTE_ARTIFACT")
            calibration_artifact = _env("ARROW_SUITE_TRACE_CALIBRATION_ARTIFACT")
            geometry_variant = _env("ARROW_SUITE_TRACE_GEOMETRY_VARIANT")
            graph_revision = _env("ARROW_SUITE_GRAPH_CONTEXT_REVISION")
            if not route_artifact or not calibration_artifact or not geometry_variant or not graph_revision:
                raise ContractError(
                    "native Trace requires explicit route, calibration, geometry-variant, and graph-context artifacts"
                )
            from .trace_native_factory import build_trace_native_fields
            trace_kwargs: dict[str, Any] = {
                "route_artifact": route_artifact,
                "calibration_artifact": calibration_artifact,
                "geometry_variant": geometry_variant,
                "graph_context_fn": graph_context_fn,
                "graph_context_revision": graph_revision,
                "lookahead": int(config.trace_lookahead),
            }
            if geometry_variant in {"rgbd", "simulator_assisted_rgbd"}:
                frame_name, calibration_revision = _trace_calibration_contract(calibration_artifact)
                if geometry_variant == "rgbd":
                    endpoint_spec = _env("ARROW_SUITE_TRACE_VISION_ENDPOINT_FACTORY")
                    if endpoint_spec is None:
                        raise ContractError(
                            "vision-only Trace requires ARROW_SUITE_TRACE_VISION_ENDPOINT_FACTORY"
                        )
                    capture_fn, endpoint_fn, _ = _runtime_trace_callbacks(
                        raw_environment, task_id=task_id, resolution=resolution,
                        frame_name=frame_name, calibration_revision=calibration_revision,
                        endpoint_detector=_vision_endpoint_detector(
                            endpoint_spec, frame_name=frame_name,
                            calibration_revision=calibration_revision,
                            run_dir=run_dir,
                        ),
                    )
                else:
                    capture_fn, endpoint_fn, _ = _runtime_trace_callbacks(
                        raw_environment, task_id=task_id, resolution=resolution,
                        frame_name=frame_name, calibration_revision=calibration_revision,
                        simulator_assisted=True,
                    )
                trace_kwargs.update(capture_fn=capture_fn, endpoint_fn=endpoint_fn)
            else:  # simulator_assisted_arrow
                anchors_spec = _env("ARROW_SUITE_TRACE_SIMULATOR_ANCHORS_FACTORY")
                if anchors_spec is None:
                    raise ContractError(
                        "simulator-assisted Trace requires ARROW_SUITE_TRACE_SIMULATOR_ANCHORS_FACTORY"
                    )
                trace_kwargs["simulator_anchors_fn"] = _callable(
                    anchors_spec, label="Trace simulator-anchor factory"
                )
            fields = build_trace_native_fields(**trace_kwargs)
            return annotate(build_native_host(NativeHostSpec(
                environment=environment, vla=vla, teacher=None,
                policy_id="arrow_trace", policy=fields["policy"],
                graph_context_fn=fields["graph_context_fn"],
            )))
        return annotate(build_native_host(NativeHostSpec(
            environment=environment, vla=vla,
            teacher=None if policy_id in _TEACHER_FREE_POLICY_IDS else teacher,
            policy_id=policy_id, policy=policy, graph_context_fn=graph_context_fn,
        )))
    except BaseException:
        close = getattr(raw_environment, "close", None)
        if callable(close):
            close()
        raise


build_minimal_branch_runner = _build_minimal_branch_runner

__all__ = ["build_host", "build_minimal_branch_runner"]
