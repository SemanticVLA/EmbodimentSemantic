"""Native composition seam for the one-observation Arrow Fast policy.

The existing Legion factory owns model/environment construction.  This module
owns only Fast's disposable adaptation lifecycle so the host can call it after
loading those components:

1. snapshot the exact current reset state of environment/VLA/teacher;
2. collect exactly one same-frame base/teacher proposal pair;
3. fit the revisioned 448-slot corrector once;
4. restore the snapshot and detach/destroy Arrow;
5. return a teacher-free Fast policy and selector for scored ``NativeHost``.

There is no unadapted fallback.  A missing artifact identity, graph callback,
rollback hook, or complete support pair fails before a scored host is built.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import os
import re
from typing import Any, Callable, Mapping

from .contracts import ContractError, ObservationFrame
from .fast import FastLifecycleReceipt, FastOneObservationSession, FastPolicy
from .fast_training import (
    FrozenEncoderArtifact,
    FrozenRouterArtifact,
    GraphFastModel,
    GraphRouter,
    GraphSlowEncoder,
    make_deterministic_encoder_artifact,
    make_deterministic_router_artifact,
)
from .native_factory import NativeHostSpec, action_selector_for, build_native_host


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _required_sha(name: str, value: str | None = None) -> str:
    resolved = value or os.environ.get(name)
    if resolved is None or not _SHA256.fullmatch(str(resolved).strip()):
        raise ContractError(f"{name} must be an explicit lowercase SHA-256")
    return str(resolved).strip()


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ContractError(f"{name} is required for native Fast")
    return value.strip()


def _pair(component: Any, name: str) -> tuple[Callable[[], Any], Callable[[Any], None]]:
    if component is None:
        raise ContractError(f"{name} is required")
    for first, second in (("snapshot_state", "restore_state"), ("snapshot", "restore")):
        snapshot = getattr(component, first, None)
        restore = getattr(component, second, None)
        if callable(snapshot) != callable(restore):
            raise ContractError(f"{name} exposes only one of {first}/{second}")
        if callable(snapshot):
            return snapshot, restore
    raise ContractError(f"{name} requires paired snapshot/restore hooks")


@dataclass(frozen=True)
class _NativeSupportSnapshot:
    environment: Any
    vla: Any
    teacher: Any


@dataclass(frozen=True)
class FastNativeBundle:
    """Everything the native spine needs after one-shot Fast adaptation."""

    policy: FastPolicy
    action_selector: Callable[..., Any]
    corrector: GraphFastModel
    receipt: FastLifecycleReceipt
    graph_context_fn: Callable[[ObservationFrame], Mapping[str, Any] | Any]
    source_manifest_sha256: str
    teacher_detached: bool = True

    @property
    def teacher(self) -> None:
        """The scored host must receive no teacher after support adaptation."""

        return None

    def policy_kwargs(self) -> Mapping[str, Any]:
        # NativeHost places one graph mapping in frame.metadata.  Leaving
        # graph_fn unset avoids calling the perception provider twice per step.
        return {"corrector": self.corrector}


def load_frozen_fast_artifacts(
    *,
    encoder_artifact: FrozenEncoderArtifact | None = None,
    router_artifact: FrozenRouterArtifact | None = None,
) -> tuple[FrozenEncoderArtifact, FrozenRouterArtifact]:
    """Load explicitly selected frozen artifacts; never silently substitute."""

    if encoder_artifact is not None or router_artifact is not None:
        if encoder_artifact is None or router_artifact is None:
            raise ContractError("native Fast requires both encoder and router artifacts")
        return encoder_artifact, router_artifact
    kind = _required_env("ARROW_SUITE_FAST_ARTIFACT_KIND")
    if kind != "deterministic":
        raise ContractError(
            "ARROW_SUITE_FAST_ARTIFACT_KIND must be deterministic unless native learned artifacts are injected"
        )
    encoder = make_deterministic_encoder_artifact(
        revision=_required_env("ARROW_SUITE_FAST_ENCODER_REVISION"),
        artifact_sha256=_required_sha("ARROW_SUITE_FAST_ENCODER_SHA256"),
    )
    router = make_deterministic_router_artifact(
        revision=_required_env("ARROW_SUITE_FAST_ROUTER_REVISION"),
        artifact_sha256=_required_sha("ARROW_SUITE_FAST_ROUTER_SHA256"),
    )
    return encoder, router


def build_fast_native_components(
    environment: Any,
    vla: Any,
    teacher: Any,
    *,
    graph_context_fn: Callable[[ObservationFrame], Mapping[str, Any] | Any],
    source_manifest_sha256: str | None = None,
    vla_manifest_sha256: str | None = None,
    episode_id: str | None = None,
    encoder_artifact: FrozenEncoderArtifact | None = None,
    router_artifact: FrozenRouterArtifact | None = None,
) -> FastNativeBundle:
    """Adapt Fast once at current reset t0 and return scored-host components."""

    if not callable(graph_context_fn):
        raise ContractError("native Fast requires an explicit graph context callback")
    source_hash = _required_sha("ARROW_SUITE_FAST_SOURCE_MANIFEST_SHA256", source_manifest_sha256)
    vla_hash = _required_sha("ARROW_SUITE_FAST_VLA_MANIFEST_SHA256", vla_manifest_sha256)
    encoder, router = load_frozen_fast_artifacts(
        encoder_artifact=encoder_artifact, router_artifact=router_artifact,
    )
    _env_hooks = _pair(environment, "environment")
    _vla_hooks = _pair(vla, "VLA")
    _teacher_hooks = _pair(teacher, "teacher")
    def snapshot() -> _NativeSupportSnapshot:
        return _NativeSupportSnapshot(
            copy.deepcopy(_env_hooks[0]()),
            copy.deepcopy(_vla_hooks[0]()),
            copy.deepcopy(_teacher_hooks[0]()),
        )

    def restore(value: _NativeSupportSnapshot) -> None:
        if not isinstance(value, _NativeSupportSnapshot):
            raise ContractError("native Fast snapshot belongs to another runtime")
        _env_hooks[1](copy.deepcopy(value.environment))
        _vla_hooks[1](copy.deepcopy(value.vla))
        _teacher_hooks[1](copy.deepcopy(value.teacher))

    raw = environment.observe()
    frame = ObservationFrame(raw, timestep=0, episode_id=episode_id)
    model = GraphFastModel(
        encoder=GraphSlowEncoder(artifact=encoder, native_mode=True),
        router=GraphRouter(artifact=router, native_mode=True),
        native_mode=True,
        vla_manifest_sha256=vla_hash,
    )
    session = FastOneObservationSession(
        model,
        snapshot_fn=snapshot,
        restore_fn=restore,
        graph_context_fn=graph_context_fn,
        vla=vla,
        teacher=teacher,
        vla_identity_fn=lambda _component: vla_hash,
    )
    result = session.run(
        [frame], [], support_success=True,
        source_manifest_sha256=source_hash,
    )
    if not result.receipt.support_complete:
        raise ContractError(f"native Fast support attempt failed: {result.receipt.support_error}")
    if model.last_receipt is None or not model.last_receipt.adapted:
        raise ContractError("native Fast did not produce an adapted 448-slot artifact")
    if not result.receipt.restored_t0:
        raise ContractError("native Fast did not restore the exact support t0")
    if result.receipt.scored_teacher_calls != 0:
        raise ContractError("native Fast scored path made a teacher call")
    policy = FastPolicy(model)
    selector = action_selector_for("arrow_fast", policy)
    return FastNativeBundle(
        policy=policy, action_selector=selector, corrector=model,
        receipt=result.receipt, graph_context_fn=graph_context_fn,
        source_manifest_sha256=source_hash,
    )


def build_fast_native_host(
    environment: Any,
    vla: Any,
    teacher: Any,
    *,
    graph_context_fn: Callable[[ObservationFrame], Mapping[str, Any] | Any],
    **kwargs: Any,
) -> tuple[Any, FastNativeBundle]:
    """Convenience integration for the native spine.

    The returned host receives ``teacher=None``.  The caller should preserve
    the bundle receipt alongside the run manifest.
    """

    bundle = build_fast_native_components(
        environment, vla, teacher, graph_context_fn=graph_context_fn, **kwargs,
    )
    host = build_native_host(NativeHostSpec(
        environment=environment, vla=vla, teacher=None,
        policy_id="arrow_fast", policy=bundle.policy,
        graph_context_fn=graph_context_fn,
    ))
    return host, bundle


__all__ = [
    "FastNativeBundle", "build_fast_native_components", "build_fast_native_host",
    "load_frozen_fast_artifacts",
]
