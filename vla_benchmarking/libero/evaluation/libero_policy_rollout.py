"""Dependency-light LIBERO rollout seam for native VLA policies.

The module deliberately does not import LIBERO, Gym, JAX, or LeRobot.  A
runtime-specific factory supplies an environment implementing the tiny reset /
step / close surface below.  This keeps simulator setup in the model/runtime
package while making chunk execution and success accounting identical across
Pi0.5, OpenVLA-OFT, and Octo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .run_policy_eval import (
    DEFAULT_EPISODE_STEP_BUDGET,
    ActionCountingAdapter,
    EpisodeOutcome,
    EpisodeSpec,
    resolve_episode_step_budget,
    run_policy_eval,
)
from .observation import read_raw_observation

NATIVE_POLICY_KINDS = frozenset({
    "pi05",
    "openvla",
    "openvla_oft",
    "octo_community_multisuite_190k",
    "octo_base15_spatial_no_arrow_matched",
})


@runtime_checkable
class LiberoEnvironment(Protocol):
    """Minimal environment seam required by the native rollout."""

    def reset(self, *, seed: int, task_id: int, episode_index: int) -> Any: ...

    def step(self, action: np.ndarray) -> Any: ...


@runtime_checkable
class LiberoEnvironmentFactory(Protocol):
    def __call__(self, episode: EpisodeSpec) -> LiberoEnvironment: ...


@dataclass(frozen=True)
class CallableLiberoEnvironmentFactory:
    """Concrete factory seam around a runtime-specific environment builder."""

    builder: Callable[[EpisodeSpec], LiberoEnvironment]

    def __call__(self, episode: EpisodeSpec) -> LiberoEnvironment:
        environment = self.builder(episode)
        if environment is None:
            raise RuntimeError("LIBERO environment factory returned None")
        return environment


@dataclass(frozen=True)
class ProductionLiberoEnvironmentFactory:
    """Factory backed by the repository's direct LIBERO environment builder.

    Imports remain lazy so contract tests and offline evaluators do not require
    MuJoCo/LIBERO.  The model packages are not involved: this factory only
    constructs the direct ``OffScreenRenderEnv`` used by the shared evaluator.
    """

    resolution: int = 256
    suite_mode: str = "vanilla"
    controller_variant: Any | None = None
    extra_camera_names: tuple[str, ...] = ()

    def __call__(self, episode: EpisodeSpec) -> LiberoEnvironment:
        try:
            from .run_arrow_pick_place_eval import build_libero_env
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "production LIBERO factory requires the repository LIBERO/MuJoCo runtime"
            ) from exc
        try:
            environment = build_libero_env(
                int(episode.cell.task_id),
                int(episode.cell.seed),
                int(self.resolution),
                suite_mode=str(self.suite_mode),
                controller_variant=self.controller_variant,
                extra_camera_names=tuple(self.extra_camera_names),
                init_state_index=int(episode.cell.init_state_index),
            )
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "unable to construct the production LIBERO environment; "
                "verify the installed LIBERO/MuJoCo assets and configuration"
            ) from exc
        init_state_audit = getattr(environment, "_arrow_init_state_diagnostics", None)
        environment_audit = getattr(environment, "_arrow_environment_audit", None)
        if not isinstance(init_state_audit, Mapping):
            _close_production_environment(environment)
            raise RuntimeError("production LIBERO environment lacks init-state audit evidence")
        if not isinstance(environment_audit, Mapping):
            _close_production_environment(environment)
            raise RuntimeError("production LIBERO environment lacks mutation audit evidence")
        selected_index = init_state_audit.get("selected_index")
        if selected_index is not None and int(selected_index) != int(episode.cell.init_state_index):
            _close_production_environment(environment)
            raise RuntimeError(
                "production LIBERO environment selected an init state different from the plan"
            )
        expected_scene_mode = "sealed_randomized" if self.suite_mode == "sealed_randomized" else "vanilla"
        if environment_audit.get("scene_randomization") != expected_scene_mode:
            _close_production_environment(environment)
            raise RuntimeError("production LIBERO environment mutation audit disagrees with suite mode")
        observation = read_raw_observation(environment, required=True)
        return _ProductionEnvironmentAdapter(
            environment,
            episode=episode,
            observation=observation,
            setup_audit={
                "init_state": dict(init_state_audit),
                "environment": dict(environment_audit),
            },
        )


def _close_production_environment(environment: Any) -> None:
    close = getattr(environment, "close", None)
    if callable(close):
        close()


class _ProductionEnvironmentAdapter:
    """Expose setup's current observation without resetting the simulator."""

    def __init__(
        self,
        environment: Any,
        *,
        episode: EpisodeSpec,
        observation: Mapping[str, Any],
        setup_audit: Mapping[str, Any],
    ) -> None:
        self._environment = environment
        self._episode = episode
        self._observation = dict(observation)
        self.setup_audit = dict(setup_audit)

    def reset(self, *, seed: int, task_id: int, episode_index: int) -> Any:
        expected = self._episode.cell
        if (int(seed), int(task_id), int(episode_index)) != (
            int(expected.seed), int(expected.task_id), int(expected.episode_index)
        ):
            raise ValueError("production LIBERO reset arguments disagree with the planned episode")
        # The direct builder already performed reset, init-state selection,
        # scene mutation, and settling.  Calling env.reset here would discard
        # that state, so return the captured post-setup observation instead.
        return dict(self._observation)

    def step(self, action: np.ndarray) -> Any:
        return self._environment.step(action)

    def check_success(self) -> Any:
        method = getattr(self._environment, "check_success", None)
        return method() if callable(method) else False

    def close(self) -> None:
        _close_production_environment(self._environment)


def _reset_observation(value: Any) -> Mapping[str, Any]:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], Mapping):
        value = value[0]
    if not isinstance(value, Mapping):
        raise TypeError("LIBERO reset must return an observation mapping or (observation, info)")
    return value


def _step_result(value: Any) -> tuple[Mapping[str, Any], bool, bool, Mapping[str, Any]]:
    if not isinstance(value, tuple) or len(value) not in (4, 5):
        raise TypeError("LIBERO step must return a 4- or 5-tuple")
    if len(value) == 5:
        observation, _reward, terminated, truncated, info = value
    else:
        observation, _reward, done, info = value
        terminated, truncated = bool(done), False
    if not isinstance(observation, Mapping):
        raise TypeError("LIBERO step must return an observation mapping")
    if not isinstance(info, Mapping):
        info = {}
    return observation, bool(terminated), bool(truncated), info


def _success_from_env(environment: Any, info: Mapping[str, Any]) -> bool:
    for key in ("success", "task_success", "is_success"):
        if key in info and bool(info[key]):
            return True
    for method_name in ("check_success", "is_success"):
        method = getattr(environment, method_name, None)
        if callable(method):
            value = method()
            if isinstance(value, Mapping):
                if any(bool(value.get(key)) for key in ("success", "task", "task_success", "is_success")):
                    return True
            elif bool(value):
                return True
    return bool(getattr(environment, "success", False))


def rollout_libero_episode(
    adapter: ActionCountingAdapter,
    episode: EpisodeSpec,
    *,
    env_factory: LiberoEnvironmentFactory,
    max_chunks: int | None = None,
    episode_step_budget: int = DEFAULT_EPISODE_STEP_BUDGET,
) -> EpisodeOutcome:
    """Reset one LIBERO environment and execute native chunks fairly.

    ``episode_step_budget`` counts accepted calls to ``env.step`` and is
    independent of the policy's native action horizon.  A budget stop may
    therefore occur in the middle of a chunk; the pending chunk is discarded
    and the outcome records the truncation without fabricating terminal success.
    """

    if max_chunks is not None and (isinstance(max_chunks, bool) or int(max_chunks) <= 0):
        raise ValueError("max_chunks must be positive")
    if max_chunks is not None:
        max_chunks = int(max_chunks)
    if isinstance(episode_step_budget, bool) or int(episode_step_budget) <= 0:
        raise ValueError("episode_step_budget must be positive")
    episode_step_budget = int(episode_step_budget)
    environment = env_factory(episode)
    environment_steps = 0
    setup_audit = getattr(environment, "setup_audit", None)
    if setup_audit is not None and not isinstance(setup_audit, Mapping):
        close = getattr(environment, "close", None)
        if callable(close):
            close()
        raise TypeError("environment setup_audit must be a mapping when provided")

    def _metadata(*, reason: str, truncated: bool = False, budget_exhausted: bool = False) -> dict[str, Any]:
        metadata = {
            "rollout": "libero_native_chunk",
            "environment_steps": int(environment_steps),
            "environment_step_budget": int(episode_step_budget),
            "budget_exhausted": bool(budget_exhausted),
            "truncated": bool(truncated),
            "termination_reason": str(reason),
        }
        if setup_audit is not None:
            metadata["environment_setup_audit"] = dict(setup_audit)
        return metadata

    try:
        observation = _reset_observation(environment.reset(
            seed=int(episode.cell.seed),
            task_id=int(episode.cell.task_id),
            episode_index=int(episode.cell.episode_index),
        ))
        chunk_index = 0
        while max_chunks is None or chunk_index < max_chunks:
            chunk_index += 1
            chunk = adapter.act(observation)
            for action in np.asarray(chunk, dtype=np.float32):
                if environment_steps >= episode_step_budget:
                    adapter.discard_pending_chunk()
                    return EpisodeOutcome(
                        success=False,
                        terminal=True,
                        failure_category="episode_step_budget",
                        metadata=_metadata(
                            reason="episode_step_budget",
                            truncated=True,
                            budget_exhausted=True,
                        ),
                    )
                # Stop immediately on terminal/success, including in the
                # middle of a chunk.  The counting adapter records only
                # actions accepted by env.step and closes the partial chunk.
                observation, terminated, truncated, info = _step_result(environment.step(action))
                adapter.record_environment_step()
                environment_steps += 1
                success = _success_from_env(environment, info)
                terminal = terminated or truncated or bool(info.get("terminal", False))
                if success:
                    adapter.discard_pending_chunk()
                    return EpisodeOutcome(
                        success=True,
                        terminal=True,
                        metadata=_metadata(
                            reason="success",
                            truncated=bool(truncated),
                            budget_exhausted=False,
                        ),
                    )
                if terminal:
                    adapter.discard_pending_chunk()
                    return EpisodeOutcome(
                        success=False,
                        terminal=True,
                        failure_category="environment_truncated" if truncated else "environment_terminal",
                        metadata=_metadata(
                            reason="environment_truncated" if truncated else "environment_terminal",
                            truncated=bool(truncated),
                            budget_exhausted=False,
                        ),
                    )
            # A chunk can exactly consume the budget without producing a
            # terminal signal.  Return now rather than querying another chunk.
            if environment_steps >= episode_step_budget:
                adapter.discard_pending_chunk()
                return EpisodeOutcome(
                    success=False,
                    terminal=True,
                    failure_category="episode_step_budget",
                    metadata=_metadata(
                        reason="episode_step_budget",
                        truncated=True,
                        budget_exhausted=True,
                    ),
                )
        # max_chunks is an optional separate protocol stop and should remain
        # observable.  With no explicit cap, the sealed step budget is the
        # only non-terminal stopping condition.
        return EpisodeOutcome(
            success=False,
            terminal=True,
            failure_category="max_chunks",
            metadata={
                **_metadata(reason="max_chunks"),
                "max_chunks": int(max_chunks) if max_chunks is not None else None,
            },
        )
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()


def run_libero_policy_eval(
    adapter: Any,
    plan: Mapping[str, Any],
    episodes: Sequence[EpisodeSpec],
    *,
    env_factory: LiberoEnvironmentFactory,
    max_chunks: int | None = None,
    episode_step_budget: int | None = None,
    output_jsonl: str | None = None,
):
    """Run native LIBERO episodes through the shared plan-aware evaluator."""

    resolved_budget = resolve_episode_step_budget(plan, episode_step_budget)
    return run_policy_eval(
        adapter,
        plan,
        episodes,
        lambda counted, episode: rollout_libero_episode(
            counted,
            episode,
            env_factory=env_factory,
            max_chunks=max_chunks,
            episode_step_budget=resolved_budget,
        ),
        output_jsonl=output_jsonl,
    )


def run_native_policy_eval(
    adapter: Any,
    plan: Mapping[str, Any],
    episodes: Sequence[EpisodeSpec],
    *,
    env_factory: LiberoEnvironmentFactory,
    max_chunks: int | None = None,
    episode_step_budget: int | None = None,
    output_jsonl: str | None = None,
):
    """Plan-consuming production seam for the Pi0.5/OpenVLA-OFT/Octo panel."""

    policy_kind = str(adapter.metadata.policy_kind)
    if policy_kind not in NATIVE_POLICY_KINDS:
        raise ValueError(f"native VLA runner does not support policy kind {policy_kind!r}")
    return run_libero_policy_eval(
        adapter,
        plan,
        episodes,
        env_factory=env_factory,
        max_chunks=max_chunks,
        episode_step_budget=episode_step_budget,
        output_jsonl=output_jsonl,
    )


__all__ = [
    "CallableLiberoEnvironmentFactory",
    "ProductionLiberoEnvironmentFactory",
    "DEFAULT_EPISODE_STEP_BUDGET",
    "LiberoEnvironment",
    "LiberoEnvironmentFactory",
    "rollout_libero_episode",
    "run_libero_policy_eval",
    "run_native_policy_eval",
    "NATIVE_POLICY_KINDS",
]
