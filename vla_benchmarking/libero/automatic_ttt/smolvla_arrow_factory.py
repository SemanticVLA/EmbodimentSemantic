"""Production SmolVLA -> Arrow live-correction runtime.

This module is the concrete runtime factory for
``collect_smolvla_arrow_corrections``.  It deliberately owns no data conversion
or training logic: it builds the direct sealed-randomized LIBERO environment,
loads the pinned local SmolVLA policy with its checkpoint-owned LeRobot
processor, and hands the *same* environment to ``ArrowCanaryBridge`` after a
failed VLA attempt.  The module never reads HDF5 and has no mock/fallback
controller path.

The heavy dependencies are imported lazily.  This keeps the contract tests
usable on a machine without MuJoCo, LeRobot, or the Arrow model runtime while
making every production dependency fail closed with an actionable error.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from .arrow_bridge import ArrowCanaryBridge
from .contracts import ContractError, EpisodeSpec, SourceState
from .live_collection import (
    CollectionResult,
    collect_fresh_arrow_demonstrations,
    collect_task_corrections,
)


PINNED_BASE_POLICY_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
DEFAULT_RESOLUTION = 256
DEFAULT_DEVICE = "cuda"
DEFAULT_VLA_STEP_BUDGET = 280
DEFAULT_ARROW_STEP_BUDGET = 1200
DEFAULT_MAX_ATTEMPTS = 500
DEFAULT_CONTROLLER_CONFIG = "canonical_molmo_rgbd_grasp.json"


def _resolved_arrow_action_budget(resolved: Mapping[str, Any]) -> int:
    """Resolve the controller budget from the hashed canonical config."""
    grasp = resolved.get("grasp_search")
    policy = resolved.get("policy_metadata")
    if not isinstance(grasp, Mapping) or not isinstance(policy, Mapping):
        raise ContractError("canonical Arrow config lacks grasp_search/policy_metadata")
    try:
        search_budget = int(grasp["max_actions"])
        shared_budget = int(policy["shared_action_budget"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("canonical Arrow config lacks integer action budgets") from exc
    if search_budget <= 0 or shared_budget <= 0 or search_budget != shared_budget:
        raise ContractError("canonical Arrow config has inconsistent action budgets")
    return search_budget


def _require_local_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"SmolVLA Arrow preflight: {label} directory is missing: {path}")
    return path


def _load_local_smolvla_policy(
    base_policy: str | Path, *, device: str = DEFAULT_DEVICE
) -> tuple[Any, Any, Any]:
    """Load the pinned SmolVLA policy and its serialized LeRobot processor.

    The loader is intentionally local-only.  The evaluator and training
    launcher use this same checkpoint-owned ``policy_preprocessor.json``;
    constructing a hand-written image or state transform here would silently
    change the VLA's action semantics.
    """

    base = _require_local_directory(base_policy, label="base policy")
    manifest = base / "base_snapshot_manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"SmolVLA Arrow preflight: base snapshot manifest is missing: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SmolVLA Arrow preflight: base snapshot manifest is unreadable: {manifest}") from exc
    if str(payload.get("revision", "")) != PINNED_BASE_POLICY_REVISION:
        raise RuntimeError(
            "SmolVLA Arrow preflight: base policy revision is not the pinned "
            f"{PINNED_BASE_POLICY_REVISION}"
        )
    files = payload.get("files")
    if not isinstance(files, Mapping) or not files:
        raise RuntimeError("SmolVLA Arrow preflight: base snapshot manifest has no file inventory")
    for relative, expected in files.items():
        candidate = base / str(relative)
        if (
            not candidate.is_file()
            or not isinstance(expected, str)
            or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected.lower()
        ):
            raise RuntimeError(
                f"SmolVLA Arrow preflight: base snapshot digest mismatch: {relative}"
            )
    preprocessor_config = base / "policy_preprocessor.json"
    if not preprocessor_config.is_file():
        raise RuntimeError(
            "SmolVLA Arrow preflight: checkpoint-owned policy_preprocessor.json is missing; "
            "refusing to invent processor semantics"
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime boundary
        raise RuntimeError("SmolVLA Arrow preflight: PyTorch is required") from exc
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("SmolVLA Arrow preflight: CUDA is unavailable for the requested SmolVLA device")

    policy = None
    errors: list[str] = []
    candidates = (
        ("lerobot.policies.smolvla", "SmolVLAPolicy"),
        ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
        ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLA"),
        ("lerobot.policies.smolvla.model", "SmolVLAPolicy"),
    )
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            loader = getattr(cls, "from_pretrained", None)
            if not callable(loader):
                raise TypeError(f"{class_name}.from_pretrained is unavailable")
            try:
                policy = loader(pretrained_name_or_path=str(base), local_files_only=True)
            except TypeError:
                policy = loader(str(base), local_files_only=True)
            break
        except Exception as exc:  # pragma: no cover - version/runtime dependent
            errors.append(f"{module_name}.{class_name}: {exc}")
    if policy is None:
        raise RuntimeError(
            "SmolVLA Arrow preflight: pinned LeRobot SmolVLA could not be loaded locally; "
            + " | ".join(errors)
        )
    try:
        to = getattr(policy, "to", None)
        if callable(to):
            to(device)
        eval_method = getattr(policy, "eval", None)
        if callable(eval_method):
            eval_method()
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError("SmolVLA Arrow preflight: failed to place policy on the requested device") from exc

    try:
        from lerobot.policies import make_pre_post_processors
    except ImportError as exc:  # pragma: no cover - runtime boundary
        raise RuntimeError("SmolVLA Arrow preflight: LeRobot processor factory is unavailable") from exc
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(base),
        )
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError(
            "SmolVLA Arrow preflight: checkpoint-owned LeRobot pre/postprocessors could not be loaded locally"
        ) from exc
    if not callable(preprocessor) or not callable(postprocessor):
        raise RuntimeError("SmolVLA Arrow preflight: loaded LeRobot pre/postprocessors are not callable")
    if not callable(getattr(policy, "select_action", None)):
        raise RuntimeError("SmolVLA Arrow preflight: SmolVLA policy exposes no select_action method")
    return policy, preprocessor, postprocessor


def _as_action_chunk(value: Any) -> Any:
    """Preserve a native action chunk while normalizing only batch dimension."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SmolVLA Arrow runtime requires NumPy") from exc
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 1 and array.shape == (7,):
        return tuple(float(item) for item in array)
    if array.ndim == 2 and array.shape[1] == 7 and array.shape[0] > 0:
        return tuple(tuple(float(item) for item in row) for row in array)
    raise ContractError(
        "SmolVLA action output must be a native [7] or [chunk,7] action; "
        f"got shape {tuple(array.shape)}"
    )


def _build_smolvla_action(base_policy: str | Path, task_description: str, *, device: str = DEFAULT_DEVICE) -> Callable[..., Any]:
    policy, preprocessor, postprocessor = _load_local_smolvla_policy(base_policy, device=device)
    reset = getattr(policy, "reset", None)

    def action(observation: Mapping[str, Any], step: int) -> Any:
        if not isinstance(observation, Mapping):
            raise ContractError("SmolVLA observation must be a mapping")
        if int(step) == 0 and callable(reset):
            reset()
        instruction = observation.get("instruction", task_description)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ContractError("SmolVLA observation instruction is missing")
        # These keys match native_vla_eval._canonical_observation and the
        # checkpoint's LeRobot processor pipeline.  No hand-written image
        # resize, normalization, or action scaling is performed here.
        payload = {
            "observation.images.image": observation["agentview"],
            "observation.images.image2": observation["wrist"],
            "observation.state": observation["state"],
            "task": instruction,
        }
        try:
            processed = preprocessor(payload)
            raw = policy.select_action(processed)
            raw = postprocessor(raw)
        except Exception as exc:  # pragma: no cover - runtime dependent
            raise RuntimeError("SmolVLA native processor/action inference failed") from exc
        return _as_action_chunk(raw)

    return action


class _DirectLiberoEnvironment:
    """One direct LIBERO environment with evaluator-compatible reset semantics."""

    def __init__(self, raw: Any, *, episode: EpisodeSpec) -> None:
        self._raw = raw
        self._episode = episode

    @property
    def raw_environment(self) -> Any:
        return self._raw

    def reset(self, *, seed: int, task_id: int, episode_index: int) -> Mapping[str, Any]:
        expected = self._episode
        if (int(seed), int(task_id), int(episode_index)) != (int(expected.seed), int(expected.task_id), 0):
            raise ContractError("LIBERO reset arguments disagree with the canonical adaptation episode")
        try:
            from vla_benchmarking.libero.evaluation.observation import read_raw_observation
            return read_raw_observation(self._raw, required=True)
        except Exception as exc:  # pragma: no cover - runtime dependent
            raise RuntimeError("SmolVLA Arrow environment could not read its post-setup observation") from exc

    def step(self, action: Any) -> Any:
        return self._raw.step(action)

    def check_success(self) -> bool:
        method = getattr(self._raw, "check_success", None)
        if not callable(method):
            raise RuntimeError("SmolVLA Arrow environment has no evaluator check_success method")
        value = method()
        if not isinstance(value, (bool, Mapping)):
            raise RuntimeError("LIBERO check_success returned an unsupported value")
        if isinstance(value, Mapping):
            return any(value.get(key) is True for key in ("success", "task_success", "is_success", "task"))
        return bool(value)

    def close(self) -> None:
        close = getattr(self._raw, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


def _build_live_environment_factory(
    *, resolution: int = DEFAULT_RESOLUTION, adaptation_seed_start: int = 3000
) -> tuple[Callable[[EpisodeSpec], Any], Callable[[Any, EpisodeSpec], Any], Callable[[Any], None]]:
    try:
        from vla_benchmarking.libero.evaluation.run_arrow_pick_place_eval import build_libero_env
    except ImportError as exc:  # pragma: no cover - runtime boundary
        raise RuntimeError("SmolVLA Arrow preflight: direct LIBERO environment builder is unavailable") from exc
    current: dict[str, Any] = {}
    available_count_by_task: dict[int, int] = {}

    def available_init_state_count(task_id: int) -> int:
        count = available_count_by_task.get(int(task_id))
        if count is not None:
            return count
        try:
            from libero.libero import benchmark
            from lerobot.envs.libero import get_task_init_states
            from vla_benchmarking.libero.shared.config import BENCHMARK_NAME
            suite = benchmark.get_benchmark_dict()[BENCHMARK_NAME]()
            count = int(len(get_task_init_states(suite, int(task_id))))
        except Exception as exc:  # pragma: no cover - runtime-only dependency
            raise RuntimeError("SmolVLA Arrow preflight: cannot determine LIBERO init-state count") from exc
        if count <= 10:
            raise RuntimeError(
                f"fresh Arrow collection requires >10 init states for task {task_id}; found {count}"
            )
        available_count_by_task[int(task_id)] = count
        return count

    def make(episode: EpisodeSpec) -> _DirectLiberoEnvironment:
        available_count = available_init_state_count(int(episode.task_id))
        offset = int(episode.seed) - int(adaptation_seed_start)
        if offset < 0:
            raise ContractError("adaptation seed precedes the collection seed namespace")
        allowed_count = available_count - 10
        selected_init_state_index = 10 + (offset % allowed_count)
        try:
            raw = build_libero_env(
                int(episode.task_id), int(episode.seed), int(resolution),
                suite_mode="sealed_randomized", extra_camera_names=("robot0_eye_in_hand",),
                init_state_index=selected_init_state_index,
            )
        except Exception as exc:  # pragma: no cover - runtime dependent
            raise RuntimeError(
                "SmolVLA Arrow preflight: failed to build sealed-randomized LIBERO environment"
            ) from exc
        for attr in ("_arrow_init_state_diagnostics", "_arrow_environment_audit"):
            if not isinstance(getattr(raw, attr, None), Mapping):
                close = getattr(raw, "close", None)
                if callable(close):
                    close()
                raise RuntimeError(f"SmolVLA Arrow environment lacks required {attr} audit evidence")
        diagnostics = getattr(raw, "_arrow_init_state_diagnostics")
        if diagnostics.get("available_count") != available_count or diagnostics.get("selected_index") != selected_init_state_index:
            close = getattr(raw, "close", None)
            if callable(close):
                close()
            raise RuntimeError("LIBERO runtime selected init state differs from deterministic collection mapping")
        wrapped = _DirectLiberoEnvironment(raw, episode=episode)
        current["environment"] = wrapped
        return wrapped

    def reset(environment: _DirectLiberoEnvironment, episode: EpisodeSpec) -> Mapping[str, Any]:
        if current.get("environment") is not environment:
            raise ContractError("LIBERO reset received an environment other than the current live episode")
        return environment.reset(seed=episode.seed, task_id=episode.task_id, episode_index=0)

    def close(environment: _DirectLiberoEnvironment) -> None:
        environment.close()
        if current.get("environment") is environment:
            current.clear()

    return make, reset, close


@dataclass
class _ArrowSession:
    molmo: Any
    resolution: int
    output_root: Path
    action_budget: Any | None = None
    transform: Any | None = None
    opening_m: float | None = None
    worker: Any | None = None
    pending_capture: Any | None = None
    pending_arrow: tuple[Any, Sequence[float], Sequence[float] | None] | None = None
    current_arrow_rgb: Any | None = None
    current_view: Any | None = None
    task_id: int | None = None
    seed: int | None = None
    provenance: Mapping[str, Any] | None = None

    def cleanup_attempt(self) -> None:
        """Release every per-episode Arrow reference before the next reset.

        ``_ArrowSession`` intentionally persists the heavyweight Molmo
        runtime, but none of the current environment, RGB capture, rendered
        arrow, calibration, action-budget, or candidate-worker state may span
        collection attempts.  This hook is called after the environment is
        closed by the collector and is idempotent so exception paths are safe.
        """
        worker = self.worker
        if worker is not None and hasattr(worker, "robot_calibration"):
            worker.robot_calibration = None
        self.worker = None
        self.pending_capture = None
        self.pending_arrow = None
        self.current_arrow_rgb = None
        self.current_view = None
        self.transform = None
        self.opening_m = None
        self.action_budget = None
        self.task_id = None
        self.seed = None

    def capture(self, environment: Any, *, resolution: int, camera_name: str) -> Any:
        """Return the cached post-VLA frame once, then fresh controller frames."""
        if camera_name == "agentview" and self.pending_capture is not None:
            capture = self.pending_capture
            self.pending_capture = None
            return capture
        from vla_benchmarking.libero.evaluation.run_arrow_pick_place_eval import capture_agentview
        return capture_agentview(environment, resolution=int(resolution), camera_name=str(camera_name))

    def refresh_geometry(self, view: Any, request: Any) -> Mapping[str, Any]:
        """Capture and decode the canonical one-arrow input after VLA motion."""
        from vla_benchmarking.libero.evaluation import run_arrow_pick_place_eval as episode

        # Refresh Panda calibration at the exact takeover pose.  The worker
        # receives this observed calibration, never simulator object state.
        self.current_view = view
        self.refresh_calibration(view)
        capture = episode.capture_agentview(view, resolution=self.resolution, camera_name="agentview")
        self.pending_capture = capture
        rendered, source_uv, destination_uv = self._render_arrow(
            view, capture, int(request.episode.task_id)
        )
        self.current_arrow_rgb = None
        self.pending_arrow = (rendered, source_uv, destination_uv)
        calibration = getattr(capture, "calibration", None)
        camera_id = getattr(calibration, "camera_name", "agentview")
        return {
            "source_uv": source_uv,
            "destination_uv": destination_uv,
            "capture_provenance": {
                "timestamp": float(time.time()),
                "camera_id": str(camera_id),
                "resolution": [int(self.resolution), int(self.resolution)],
                "calibration_revision": hashlib.sha256(
                    json.dumps({"camera": str(camera_id), "calibration": str(calibration)}, sort_keys=True).encode()
                ).hexdigest(),
                "captured_after_timestep": len(request.vla_history),
            },
        }

    def refresh_calibration(self, environment: Any) -> None:
        """Refresh the robot-frame calibration at the current takeover pose."""

        from vla_benchmarking.libero.arrow_grasp_controller.controller import runner

        if self.worker is None:
            raise ContractError("Arrow perception worker is unavailable")
        calibration, transform, probe = runner.probe_robot_calibration(environment)
        self.worker.robot_calibration = calibration
        self.transform = transform
        geometry = probe.get("gripper_geometry") if isinstance(probe, Mapping) else None
        if not isinstance(geometry, Mapping) or "measured_opening_m" not in geometry:
            raise ContractError("Arrow calibration probe lacks measured gripper opening")
        self.opening_m = float(geometry["measured_opening_m"])

    def _render_arrow(
        self, environment: Any, capture: Any, task_id: int
    ) -> tuple[Any, Sequence[float], Sequence[float] | None]:
        from vla_benchmarking.libero.evaluation import run_arrow_pick_place_eval as episode
        from vla_benchmarking.libero.evaluation import run_arrow_pick_place_matrix as matrix

        inputs = matrix._default_arrow_inputs(environment, int(task_id), self.resolution)
        rendered, _audit = episode.render_exactly_one_arrow(
            capture.rgb,
            inputs["bboxes"],
            subject=inputs["subject"],
            goal_object=inputs["goal_object"],
            anchor_policy="bbox_center",
        )
        source_uv, destination_uv = episode.decode_arrow_pixels(capture.rgb, rendered)
        return rendered, source_uv, destination_uv

    def refresh_arrow(self, capture: Any) -> tuple[Any, Sequence[float], Sequence[float] | None]:
        """Bind every candidate retry to the RGB-D frame captured for it."""

        if self.pending_arrow is not None:
            value = self.pending_arrow
            self.pending_arrow = None
        else:
            if self.current_view is None or self.task_id is None:
                raise ContractError("Arrow retry requested before a live takeover was bound")
            value = self._render_arrow(self.current_view, capture, int(self.task_id))
        self.current_arrow_rgb = value[0]
        return value

    def evaluate(self, environment: Any) -> bool:
        method = getattr(environment, "check_success", None)
        if not callable(method):
            raise ContractError("Arrow evaluator requires environment.check_success()")
        value = method()
        if isinstance(value, Mapping):
            return any(value.get(key) is True for key in ("success", "task_success", "is_success", "task"))
        if not isinstance(value, bool):
            raise ContractError("Arrow evaluator check_success() must return bool or mapping")
        return value

    def episode_runner(self, *, env: Any, context: Any, evaluator: Callable[[Any], bool] | None,
                       retreat_completed_callback: Callable[[], None] | None = None) -> Mapping[str, Any]:
        if self.transform is None or self.opening_m is None:
            raise ContractError("Arrow controller calibration was not refreshed before takeover")
        if self.current_arrow_rgb is None:
            raise ContractError("Arrow controller motion requires the current rendered arrow frame")
        from vla_benchmarking.libero.evaluation import run_arrow_pick_place_eval as episode
        if self.action_budget is None:
            self.action_budget = episode._ActionBudget(DEFAULT_ARROW_STEP_BUDGET)
        result = episode.run_episode(
            env=env, task_id=int(self.task_id if self.task_id is not None else 0),
            seed=int(self.seed if self.seed is not None else 0), output_dir=context.output_dir,
            arrow_rgb=self.current_arrow_rgb,
            dry_run=False, resolution=self.resolution,
            evaluator=evaluator, capture=context.agentview_capture,
            # ``run_episode`` accepts the Arrow canary variant name
            # ``canonical_molmo_rgbd_grasp`` (or ``None``), not the separate
            # ``runner.run_canary_episode`` spelling ``canonical``.
            allow_unvalidated_profile=True, controller_variant=episode.DEFAULT_PROFILE_NAME,
            suite_mode="sealed_randomized", experimental_candidate=context.candidate,
            experimental_eef_orientation_transform=self.transform,
            experimental_gripper_opening_m=float(self.opening_m),
            retreat_completed_callback=retreat_completed_callback,
            experimental_action_budget=self.action_budget,
        )
        return result


def _build_arrow_teacher_factory(
    *,
    output_root: Path,
    controller_config_hash: str,
    resolution: int = DEFAULT_RESOLUTION,
    device: str = DEFAULT_DEVICE,
    arrow_step_budget: int = DEFAULT_ARROW_STEP_BUDGET,
) -> tuple[Callable[[EpisodeSpec, Path], ArrowCanaryBridge], _ArrowSession]:
    try:
        from vla_benchmarking.libero.arrow_grasp_controller.controller import runner
        from vla_benchmarking.libero.arrow_grasp_controller.configs import load_controller_config
    except ImportError as exc:  # pragma: no cover - runtime boundary
        raise RuntimeError("ArrowGraspControllerTeacher preflight: Arrow controller runtime is unavailable") from exc
    configured_config = os.environ.get("ARROW_CONTROLLER_CONFIG", "").strip()
    config_path = Path(configured_config).expanduser() if configured_config else Path(__file__).resolve().parents[1] / "arrow_grasp_controller" / "configs" / DEFAULT_CONTROLLER_CONFIG
    if not configured_config:
        config_path = Path(__file__).resolve().parents[1] / "arrow_grasp_controller" / "configs" / DEFAULT_CONTROLLER_CONFIG
    if not config_path.is_file():
        raise RuntimeError(f"ArrowGraspControllerTeacher preflight: canonical controller config is missing: {config_path}")
    try:
        resolved = load_controller_config(config_path)
    except Exception as exc:
        raise RuntimeError("ArrowGraspControllerTeacher preflight: canonical controller config could not be resolved") from exc
    resolved_hash = str(resolved.get("config_hash", "")).lower()
    if len(resolved_hash) != 64 or resolved_hash != str(controller_config_hash).lower():
        raise ContractError("ArrowGraspControllerTeacher preflight: controller config hash does not match the supplied contract")
    resolved_action_budget = _resolved_arrow_action_budget(resolved)
    if int(arrow_step_budget) != resolved_action_budget:
        raise ContractError(
            "ArrowGraspControllerTeacher preflight: requested action budget "
            f"{arrow_step_budget} differs from canonical config budget {resolved_action_budget}"
        )
    try:
        molmo = runner.build_local_molmo_runtime(
            molmopoint_model=os.environ.get("ARROW_MOLMOPOINT_MODEL", runner.MOLMOPOINT_MODEL_ID),
            molmopoint_revision=os.environ.get("ARROW_MOLMOPOINT_REVISION", runner.MOLMOPOINT_MODEL_REVISION),
            device=device,
        )
        provenance = runner.preflight_local_molmo_runtime(molmo, load_models=True)
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError(
            "ArrowGraspControllerTeacher preflight: pinned MolmoPoint worker could not be initialized; "
            "the Legion job requires Arrow model credentials/cache and CUDA"
        ) from exc
    session = _ArrowSession(molmo=molmo, resolution=int(resolution), output_root=Path(output_root))
    session.provenance = provenance

    def teacher_factory(episode: EpisodeSpec, episode_output: Path) -> ArrowCanaryBridge:
        if session.worker is None:
            raise RuntimeError("ArrowGraspControllerTeacher preflight: worker was not initialized for the live environment")
        session.task_id = int(episode.task_id)
        session.seed = int(episode.seed)
        return ArrowCanaryBridge(
            worker=session.worker,
            episode_runner=session.episode_runner,
            source_uv=(0.0, 0.0), destination_uv=None,
            output_root=episode_output,
            variant="canonical",
            refresh_fn=session.refresh_geometry,
            capture_fn=session.capture,
            arrow_refresh_fn=session.refresh_arrow,
            before_propose_callback=session.refresh_calibration,
            evaluator=session.evaluate,
            dry_run=False,
            resolution=int(resolution),
        )

    def prepare_for_environment(environment: _DirectLiberoEnvironment) -> None:
        try:
            calibration, transform, probe = runner.probe_robot_calibration(environment)
            session.transform = transform
            geometry = probe.get("gripper_geometry") if isinstance(probe, Mapping) else None
            session.opening_m = float(geometry["measured_opening_m"]) if isinstance(geometry, Mapping) else None
            session.worker = runner.ModelPerceptionWorker(molmo, calibration)
            from vla_benchmarking.libero.evaluation.run_arrow_pick_place_eval import _ActionBudget
            session.action_budget = _ActionBudget(int(arrow_step_budget))
        except Exception as exc:  # pragma: no cover - runtime dependent
            raise RuntimeError("ArrowGraspControllerTeacher preflight: live Panda calibration/worker setup failed") from exc

    # Expose this hook to the outer environment factory without changing the
    # public collector signature.
    teacher_factory.prepare_for_environment = prepare_for_environment  # type: ignore[attr-defined]

    return teacher_factory, session


def build_collection_factory(
    *,
    base_policy: str | Path | None = None,
    device: str = DEFAULT_DEVICE,
    resolution: int = DEFAULT_RESOLUTION,
    vla_step_budget: int = DEFAULT_VLA_STEP_BUDGET,
    arrow_step_budget: int = DEFAULT_ARROW_STEP_BUDGET,
    collection_mode: str = "same_episode_takeover",
) -> Callable[..., Mapping[str, Any]]:
    """Build the CLI-compatible collection callable after runtime preflight."""

    selected_base = base_policy or os.environ.get("SMOLVLA_BASE_POLICY")
    if collection_mode not in {"same_episode_takeover", "fresh_arrow"}:
        raise ValueError("collection_mode must be same_episode_takeover or fresh_arrow")
    if collection_mode == "same_episode_takeover" and not selected_base:
        raise RuntimeError(
            "SmolVLA Arrow preflight: SMOLVLA_BASE_POLICY must name the pinned local base checkpoint"
        )
    if int(vla_step_budget) <= 0 or int(arrow_step_budget) <= 0:
        raise ValueError("VLA and Arrow step budgets must be positive")

    def collect(**kwargs: Any) -> Mapping[str, Any]:
        task_id = int(kwargs["task_id"])
        task_description = str(kwargs["task_description"])
        output_root = Path(kwargs["output_root"]).expanduser().resolve()
        supplied_hash = str(kwargs["controller_config_hash"])
        adaptation_seed_start = int(kwargs.get("adaptation_seed_start", 3000))
        environment_factory, reset_environment, close_environment = _build_live_environment_factory(
            resolution=resolution, adaptation_seed_start=adaptation_seed_start
        )
        teacher_factory, session = _build_arrow_teacher_factory(
            output_root=output_root,
            controller_config_hash=supplied_hash,
            resolution=resolution,
            device=device,
            arrow_step_budget=int(arrow_step_budget),
        )
        original_make = environment_factory

        def make_and_prepare(episode: EpisodeSpec) -> Any:
            environment = original_make(episode)
            prepare = getattr(teacher_factory, "prepare_for_environment", None)
            if not callable(prepare):
                raise RuntimeError("ArrowGraspControllerTeacher preflight: missing per-environment calibration hook")
            prepare(environment)
            return environment

        provenance = {
                "runtime": "smolvla_arrow_factory",
                "base_policy_revision": PINNED_BASE_POLICY_REVISION,
                "arrow_model_provenance": getattr(session, "provenance", {}),
                "processor": "checkpoint-owned LeRobot policy_preprocessor.json",
                "resolution": int(resolution),
                "vla_step_budget": int(vla_step_budget),
                "arrow_step_budget": int(arrow_step_budget),
                "max_attempts": int(kwargs.get("max_attempts", DEFAULT_MAX_ATTEMPTS)),
                "fps": 20,
                "provenance_categories": {
                    "resolution": "SMOLVLA_PINNED", "vla_step_budget": "SMOLVLA_PINNED",
                    "arrow_step_budget": "CONTROLLER_PINNED", "max_attempts": "OPERATIONAL",
                    "fps": "CONTROLLER_PINNED",
                },
            }
        if collection_mode == "fresh_arrow":
            result = collect_fresh_arrow_demonstrations(
                task_id=task_id, task_description=task_description, policy_id="smolvla",
                environment_factory=make_and_prepare, reset_environment=reset_environment,
                close_environment=close_environment, teacher_factory=teacher_factory,
                output_root=output_root, accepted_target=int(kwargs.get("accepted_target", 50)),
                adaptation_seed_start=int(kwargs.get("adaptation_seed_start", 3000)),
                max_attempts=int(kwargs.get("max_attempts", DEFAULT_MAX_ATTEMPTS)),
                teacher_step_budget=int(arrow_step_budget), source_state_fn=classify_source_state,
                controller_config_hash=supplied_hash,
                reserved_eval_init_state_indices=kwargs.get("reserved_eval_init_state_indices"),
                reserved_eval_init_state_hashes=kwargs.get("reserved_eval_init_state_hashes"),
                attempt_cleanup_fn=getattr(session, "cleanup_attempt", None),
                provenance={**provenance, "collection_mode": "fresh_arrow", "vla_loaded": False,
                            "vla_called": False, "method": "successful_arrow_behavior_cloning"},
            )
        else:
            action = _build_smolvla_action(selected_base, task_description, device=device)
            result = collect_task_corrections(
                task_id=task_id, task_description=task_description, policy_id="smolvla",
                environment_factory=make_and_prepare, reset_environment=reset_environment,
                close_environment=close_environment, vla_action=action, teacher_factory=teacher_factory,
                output_root=output_root, accepted_target=int(kwargs.get("accepted_target", 50)),
                adaptation_seed_start=int(kwargs.get("adaptation_seed_start", 3000)),
                max_attempts=int(kwargs.get("max_attempts", DEFAULT_MAX_ATTEMPTS)),
                vla_step_budget=int(vla_step_budget), teacher_step_budget=int(arrow_step_budget),
                source_state_fn=classify_source_state, controller_config_hash=supplied_hash,
                attempt_cleanup_fn=getattr(session, "cleanup_attempt", None),
                provenance={**provenance, "collection_mode": "same_episode_takeover",
                            "base_policy": str(Path(selected_base).expanduser().resolve())},
            )
        return {
            "status": "COLLECTION_COMPLETE", "task_id": result.task_id,
            "accepted_count": result.accepted_count, "attempted_count": result.attempted_count,
            # Failed episode payloads are intentionally discarded by the live
            # collector; preserve that absence instead of serializing None as
            # the literal string ``"None"``.
            "accepted_path": str(result.accepted_path),
            "failed_path": None if result.failed_path is None else str(result.failed_path),
            "manifest_path": str(result.manifest_path), "manifest_sha256": result.manifest_sha256,
        }

    return collect


def classify_source_state(_environment: Any, observation: Mapping[str, Any]) -> SourceState:
    """Allow takeover only when both jaws are clearly open.

    A closed gripper does *not* prove that the failed VLA is holding the
    intended object.  This factory has no placement-only/held-object recovery
    path, so closed and borderline observations are marked ``UNSAFE`` and the
    collector discards that attempt rather than asking Arrow to move an
    unknown payload.
    """

    try:
        import numpy as np
        state = np.asarray(observation.get("state"), dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise ContractError("source-state classifier requires canonical finite 8-D state") from exc
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ContractError("source-state classifier requires canonical finite 8-D state")
    gripper = state[-2:]
    closed_threshold = float(os.environ.get("ARROW_GRIPPER_CLOSED_THRESHOLD", "0.01"))
    if not np.isfinite(closed_threshold) or closed_threshold <= 0:
        raise ContractError("ARROW_GRIPPER_CLOSED_THRESHOLD must be positive")
    open_threshold = float(
        os.environ.get("ARROW_GRIPPER_OPEN_THRESHOLD", str(2.0 * closed_threshold))
    )
    if not np.isfinite(open_threshold) or open_threshold <= closed_threshold:
        raise ContractError(
            "ARROW_GRIPPER_OPEN_THRESHOLD must be finite and greater than "
            "ARROW_GRIPPER_CLOSED_THRESHOLD"
        )
    # Require both finger joints to be beyond the open margin.  ``qpos`` is a
    # bounded safety signal only; it cannot establish object identity.
    if bool(np.all(np.abs(gripper) >= open_threshold)):
        return SourceState.SOURCE_UNHELD
    return SourceState.UNSAFE


def collect(**kwargs: Any) -> Mapping[str, Any]:
    """CLI ``module:callable`` entrypoint for live collection."""
    # The CLI passes these values explicitly so no collection-budget or image
    # resolution default is hidden at the module boundary.
    return build_collection_factory(
        resolution=int(kwargs.pop("resolution", DEFAULT_RESOLUTION)),
        vla_step_budget=int(kwargs.pop("vla_step_budget", DEFAULT_VLA_STEP_BUDGET)),
        arrow_step_budget=int(kwargs.pop("arrow_step_budget", DEFAULT_ARROW_STEP_BUDGET)),
        collection_mode=str(kwargs.pop("collection_mode", "same_episode_takeover")),
    )(**kwargs)


__all__ = [
    "PINNED_BASE_POLICY_REVISION", "build_collection_factory", "classify_source_state", "collect",
]
