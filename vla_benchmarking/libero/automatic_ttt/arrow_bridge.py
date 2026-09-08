"""Opt-in bridge to the repository's existing Arrow canary controller.

The controller's production CLI owns its own episode lifecycle, so it cannot
be composed directly with a VLA rollout.  ``ArrowCanaryBridge`` instead calls
the already-imported ``run_canary_episode`` on the takeover view supplied by
``EpisodeCoordinator``.  The caller provides the controller's perception
worker, episode runner and source/destination geometry exactly as it does for
the canonical canary.  No reset or close is performed by this bridge.
"""

from __future__ import annotations

from pathlib import Path
import math
import inspect
from typing import Any, Callable, Mapping, Sequence

from .contracts import ContractError, TeacherRecoveryRequest, TeacherRecoveryResult
from .teacher import ArrowGraspControllerTeacher, TakeoverEnvironmentView


class ArrowCanaryBridge(ArrowGraspControllerTeacher):
    """Adapt ``arrow_grasp_controller.controller.runner.run_canary_episode``.

    ``episode_runner`` must execute actions through its supplied environment
    and return the controller audit mapping.  Transition data comes only from
    the guarded live-view recorder; controller manifests, overlays, candidate
    boxes and simulator contact state remain metadata.
    """

    def __init__(
        self,
        *,
        worker: Any,
        episode_runner: Callable[..., Any],
        source_uv: Sequence[float],
        destination_uv: Sequence[float] | None = None,
        output_root: str | Path,
        variant: Any = "canonical",
        transition_getter: Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]] | None = None,
        refresh_fn: Callable[[TakeoverEnvironmentView, TeacherRecoveryRequest], Mapping[str, Any]] | None = None,
        allow_stale_geometry_for_tests: bool = False,
        held_recover_fn: Callable[[TakeoverEnvironmentView, TeacherRecoveryRequest], Mapping[str, Any]] | None = None,
        **run_kwargs: Any,
    ) -> None:
        self.worker = worker
        self.episode_runner = episode_runner
        self.source_uv = tuple(float(value) for value in source_uv)
        self.destination_uv = None if destination_uv is None else tuple(float(value) for value in destination_uv)
        self.output_root = Path(output_root)
        self.variant = variant
        # Kept as a compatibility argument for older callers, but never used
        # as the source of training rows.  The guarded view is authoritative:
        # only its pre/action/post records prove that Arrow actually stepped
        # the live post-VLA environment.  A manifest or callback cannot
        # fabricate a transition.
        self.transition_getter = transition_getter
        self.held_recover_fn = held_recover_fn
        self.refresh_fn = refresh_fn
        self.allow_stale_geometry_for_tests = bool(allow_stale_geometry_for_tests)
        if self.refresh_fn is None and not self.allow_stale_geometry_for_tests:
            raise ContractError(
                "Arrow takeover requires refresh_fn to capture source/destination geometry after VLA motion; "
                "allow_stale_geometry_for_tests=True is only for unit tests"
            )
        self.run_kwargs = dict(run_kwargs)
        super().__init__(
            self._recover,
            teacher_id="arrow_grasp_controller",
            teacher_privilege="simulator_bbox_and_contact_state",
            requires_privileged_environment=True,
        )

    def _recover(
        self,
        view: TakeoverEnvironmentView,
        request: TeacherRecoveryRequest,
        *,
        collection_mode: str = "same_episode_takeover",
    ) -> Mapping[str, Any]:
        # ``ArrowGraspControllerTeacher.recover_from_reset`` forwards the
        # collection mode to the injected recovery callable.  The bridge's
        # controller mechanics are identical for a fresh reset and takeover;
        # the superclass owns the mode-specific receipt/trace validation.
        if collection_mode not in {"same_episode_takeover", "fresh_arrow"}:
            raise ContractError(f"unknown Arrow collection mode: {collection_mode!r}")
        if request.source_state.value == "source_held":
            if self.held_recover_fn is None:
                raise ContractError(
                    "canonical Arrow canary assumes an unheld parked start; provide held_recover_fn for "
                    "placement-only takeover instead of opening the gripper on a held object"
                )
            return self.held_recover_fn(view, request)
        # Import lazily so importing automatic_ttt never loads MuJoCo/OpenCV.
        try:
            from vla_benchmarking.libero.arrow_grasp_controller.controller.runner import run_canary_episode
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ContractError("LIBERO Arrow controller dependencies are unavailable") from exc
        if self.refresh_fn is None:
            geometry = {
                "source_uv": self.source_uv,
                "destination_uv": self.destination_uv,
                "capture_provenance": {"status": "unsafe_test_opt_in"},
            }
        else:
            geometry = self._fresh_geometry(self.refresh_fn(view, request), request)
        source_uv = geometry["source_uv"]
        destination_uv = geometry["destination_uv"]
        run_dir = self.output_root / request.episode.episode_id
        runner = self._bind_runner_to_view(view)
        raw = run_canary_episode(
            env=view,
            task_id=request.episode.task_id,
            seed=request.episode.seed,
            output_dir=run_dir,
            variant=self.variant,
            worker=self.worker,
            episode_runner=runner,
            source_uv=source_uv,
            destination_uv=destination_uv,
            **self.run_kwargs,
        )
        if not isinstance(raw, Mapping):
            raise ContractError("run_canary_episode must return a mapping")
        transitions = tuple(view.executed_transitions)
        if self.transition_getter is not None:
            # An optional legacy getter is an audit assertion only.  Do not
            # ingest its rows; callers must migrate to the authoritative view
            # recorder.  Empty output is accepted for legacy manifests.
            legacy_rows = self.transition_getter(raw)
            if legacy_rows and len(tuple(legacy_rows)) != len(transitions):
                raise ContractError(
                    "legacy transition_getter disagrees with the authoritative live-view recorder"
                )
        # ``run_canary_episode`` returns a manifest whose controller result is
        # nested under ``final_result``.  The top-level manifest is not an
        # evaluator verdict: it may only contain attempt bookkeeping.  Require
        # the explicit evaluator boolean so a completed motion or a non-empty
        # manifest cannot be mislabeled as a successful training target.
        evaluator_success, controller_status = self._controller_verdict(raw)
        final_transition_success = bool(transitions and transitions[-1].success)
        # Arrow's evaluator runs after retreat.  Its explicit episode verdict
        # is the acceptance criterion; an individual final env.step may still
        # carry ``success=False`` because it is not the evaluator.  Preserve
        # that raw flag and expose the mismatch instead of rewriting it.
        accepted_success = bool(evaluator_success)
        return {
            "transitions": transitions,
            "success": accepted_success,
            "status": "teacher_success" if accepted_success else "teacher_failed",
            "metadata": {
                "controller_status": controller_status,
                "evaluator_success": evaluator_success,
                "final_transition_success": final_transition_success,
                "acceptance_reason": (
                    "evaluator_success_post_retreat"
                    if accepted_success
                    else "evaluator_rejected"
                ),
                "evaluator_phase": "post_retreat",
                "controller_attempt_count": len(raw.get("attempts", ())) if isinstance(raw.get("attempts", ()), (list, tuple)) else None,
                "teacher_privilege": "simulator_bbox_and_contact_state",
                "source_state": request.source_state.value,
                "capture_provenance": geometry["capture_provenance"],
            },
        }

    def _bind_runner_to_view(self, view: TakeoverEnvironmentView) -> Callable[..., Any]:
        """Inject the guarded live view into runners that expose ``env``.

        Existing canonical runners often close over their environment and are
        left untouched.  New/composable runners should declare ``env`` (or
        ``**kwargs``); this wrapper then makes it impossible to accidentally
        execute actions on the pre-takeover raw environment.
        """

        try:
            parameters = inspect.signature(self.episode_runner).parameters
            accepts_env = "env" in parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
            )
        except (TypeError, ValueError):
            accepts_env = False
        if not accepts_env:
            raise ContractError(
                "Arrow takeover runner must accept env= so actions execute through the guarded live view; "
                "a runner that closes over the raw environment cannot produce a valid training trace"
            )

        def bound_runner(**kwargs: Any) -> Any:
            kwargs["env"] = view
            return self.episode_runner(**kwargs)

        return bound_runner

    @staticmethod
    def _fresh_geometry(raw: Mapping[str, Any], request: TeacherRecoveryRequest) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ContractError("refresh_fn must return a geometry mapping")
        source = raw.get("source_uv")
        destination = raw.get("destination_uv")
        provenance = raw.get("capture_provenance")
        if not isinstance(source, Sequence) or isinstance(source, (str, bytes)) or len(source) != 2:
            raise ContractError("refresh_fn must return a finite two-dimensional source_uv")
        source_uv = tuple(float(value) for value in source)
        if not all(math.isfinite(value) for value in source_uv):
            raise ContractError("refresh_fn source_uv contains a non-finite value")
        if destination is not None:
            if not isinstance(destination, Sequence) or isinstance(destination, (str, bytes)) or len(destination) != 2:
                raise ContractError("refresh_fn destination_uv must be two-dimensional or null")
            destination_uv = tuple(float(value) for value in destination)
            if not all(math.isfinite(value) for value in destination_uv):
                raise ContractError("refresh_fn destination_uv contains a non-finite value")
        else:
            destination_uv = None
        if not isinstance(provenance, Mapping):
            raise ContractError("refresh_fn must return capture_provenance")
        required = ("timestamp", "camera_id", "resolution", "calibration_revision", "captured_after_timestep")
        missing = [key for key in required if key not in provenance]
        if missing:
            raise ContractError(f"refresh_fn capture_provenance missing {missing}")
        timestamp = provenance["timestamp"]
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(float(timestamp)):
            raise ContractError("capture_provenance.timestamp must be finite numeric")
        if not str(provenance["camera_id"]):
            raise ContractError("capture_provenance.camera_id is required")
        resolution = provenance["resolution"]
        if isinstance(resolution, int):
            valid_resolution = resolution > 0
        elif isinstance(resolution, Sequence) and not isinstance(resolution, (str, bytes)):
            valid_resolution = len(resolution) == 2 and all(int(value) > 0 for value in resolution)
        else:
            valid_resolution = False
        if not valid_resolution:
            raise ContractError("capture_provenance.resolution must be a positive int or [height, width]")
        if not str(provenance["calibration_revision"]):
            raise ContractError("capture_provenance.calibration_revision is required")
        captured_after = provenance["captured_after_timestep"]
        if isinstance(captured_after, bool) or int(captured_after) != captured_after or int(captured_after) != len(request.vla_history):
            raise ContractError(
                "capture_provenance.captured_after_timestep must equal the final VLA timestep count"
            )
        return {
            "source_uv": source_uv,
            "destination_uv": destination_uv,
            "capture_provenance": dict(provenance),
        }

    @staticmethod
    def _controller_verdict(raw: Mapping[str, Any]) -> tuple[bool, str]:
        """Return the nested evaluator verdict and controller status.

        ``run_canary_episode``'s top-level ``status`` describes candidate
        bookkeeping (for example ``selected``); only ``final_result`` is the
        controller/evaluator result for the executed attempt.
        """

        final_result = raw.get("final_result")
        if not isinstance(final_result, Mapping):
            return False, "missing_final_result"
        return final_result.get("evaluator_success") is True, str(final_result.get("status", "unknown"))


__all__ = ["ArrowCanaryBridge"]
