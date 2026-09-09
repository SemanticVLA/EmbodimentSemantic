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
The factory callable receives ``raw_environment=...`` and may return an
already rollback-capable ``InterruptibleArrow`` or compatible teacher.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .contracts import ContractError, ObservationFrame
from .interruptible_arrow import InterruptibleArrow
from .libero_adapter import LiberoEnvironmentAdapter
from .native_factory import NativeHostSpec, build_native_host
from .smolvla_adapter import SmolVLAAdapter
from .native_arrow_teacher import PerFrameArrowTeacher, build_rgbd_perception


def _runtime_trace_callbacks(
    raw_environment: Any, *, task_id: int, resolution: int,
    frame_name: str, calibration_revision: str,
) -> tuple[Callable[..., Any], Callable[..., Any], None]:
    """Build synchronized RGB-D callbacks from the live production camera."""
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
                "provenance": {"source": "canonical_visual_arrow"}}

    return capture, endpoint, None


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return None if value is None or not str(value).strip() else str(value).strip()


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


def build_host(*, config: Any, operation: str, policy_id: str, run_dir: str | Path, max_steps: int) -> Any:
    """Build one real, rollback-capable NativeHost for the Legion canary."""
    checkpoint = _env("ARROW_SUITE_CHECKPOINT")
    if checkpoint is None:
        raise ContractError("ARROW_SUITE_CHECKPOINT is required for native SmolVLA canary")
    task_id = int(_env("ARROW_SUITE_TASK_ID", str(config.task_ids[0])))
    seed = int(_env("ARROW_SUITE_SEED", str(config.learned_seeds[0])))
    resolution = int(_env("ARROW_SUITE_RESOLUTION", str(config.image_resolution[0])))
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
        extra_camera_names=("robot0_eye_in_hand",), init_state_index=1,
    )
    try:
        environment = LiberoEnvironmentAdapter.from_live_libero(
            raw_environment, instruction=instruction, require_images=True, strict_snapshot=True,
        )
        vla = SmolVLAAdapter.from_local_checkpoint(
            checkpoint, device=device, task_description=instruction,
        )
        teacher = None
        if policy_id not in {"frozen_base", "arrow_trace"}:
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
            return host
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
                capture_fn, endpoint_fn, _ = _runtime_trace_callbacks(
                    raw_environment, task_id=task_id, resolution=resolution,
                    frame_name=frame_name, calibration_revision=calibration_revision,
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
            return build_native_host(NativeHostSpec(
                environment=environment, vla=vla, teacher=None,
                policy_id="arrow_trace", policy=fields["policy"],
                graph_context_fn=fields["graph_context_fn"],
            ))
        return build_native_host(NativeHostSpec(
            environment=environment, vla=vla, teacher=teacher,
            policy_id=policy_id, graph_context_fn=graph_context_fn,
        ))
    except BaseException:
        close = getattr(raw_environment, "close", None)
        if callable(close):
            close()
        raise


__all__ = ["build_host"]
