"""Genuine Arrow collection for takeover and fresh demonstrations.

This module is the runtime seam for adaptation data.  A caller supplies the
already configured native VLA environment and policy action function.  The
environment is reset exactly once, the VLA acts until failure, and the
``ArrowCanaryBridge`` receives the *same* live object through
``EpisodeCoordinator``.  No HDF5 expert data, controller manifest, or
simulator-only state can become a training transition.

The collector is intentionally dependency-light.  LIBERO, LeRobot, the VLA,
and Arrow's RGB-D worker are injected by the production launcher; this module
can therefore be preflighted and tested with a tiny synthetic environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .contracts import (
    ContractError,
    EpisodeSpec,
    EpisodeStatus,
    SourceState,
    SourceStateFn,
    TerminalFn,
    SuccessFn,
    VLAActionFn,
    _json_safe,
)
from .dataset import CANONICAL_OBSERVATION_SCHEMA, validate_student_observation_schema
from .episode import EpisodeCoordinator
from .teacher import ArrowGraspControllerTeacher, PrivilegedTakeoverEnvironmentView, TakeoverEnvironmentView
from .collection_cache import DurableEpisodeCache


COLLECTION_SCHEMA = "automatic_ttt.arrow_live_collection.v1"
SOURCE_KIND = "arrow_grasp_controller_trajectory"
PEFT_METHOD_LABEL = "peft_lora"
FRESH_COLLECTION_SCHEMA = "automatic_ttt.arrow_fresh_demonstration_collection.v1"
FRESH_SOURCE_KIND = "arrow_grasp_controller_fresh_demonstration"
FRESH_PEFT_METHOD_LABEL = "fresh_arrow_behavior_cloning_peft"
COLLECTION_MANIFEST_VERSION = 1
SEALED_EVAL_INIT_STATE_INDICES = tuple(range(10))


def canonical_student_observation(value: Any, *, instruction: str) -> dict[str, Any]:
    """Project a native VLA observation to the four-field student contract.

    The native evaluator owns image orientation and state construction.  We
    call its ``_canonical_observation`` immediately, then deliberately drop
    every sidecar so the VLA and Arrow both see exactly ``agentview``,
    ``wrist``, ``state``, and ``instruction``.
    """

    if not isinstance(instruction, str) or not instruction.strip():
        raise ContractError("instruction must be non-empty text")
    try:
        from vla_benchmarking.libero.evaluation.native_vla_eval import _canonical_observation
    except ImportError as exc:  # pragma: no cover - runtime-only dependency
        raise ContractError("native VLA observation canonicalizer is unavailable") from exc
    normalized = _canonical_observation(value, original_openvla=False)
    required = {
        "observation.images.image": "agentview",
        "observation.images.image2": "wrist",
        "observation.state": "state",
    }
    missing = [key for key in required if key not in normalized]
    if missing:
        raise ContractError(f"native canonical observation lacks fields: {missing}")
    projected = {
        "agentview": normalized["observation.images.image"],
        "wrist": normalized["observation.images.image2"],
        "state": normalized["observation.state"],
        "instruction": instruction,
    }
    validate_student_observation_schema(
        projected, require_complete=True, schema=CANONICAL_OBSERVATION_SCHEMA
    )
    return projected


class CanonicalLiveEnvironment:
    """Canonical observation facade over one already-reset native environment.

    ``step`` stores the post-step observation and returns a tuple with that
    canonical observation.  ``reset`` and ``close`` are intentionally absent;
    the collector owns those lifecycle calls outside the takeover interval.
    ``__getattr__`` is needed by Arrow's privileged capture/motion helpers and
    delegates only to this exact raw environment object.
    """

    def __init__(
        self,
        raw_environment: Any,
        *,
        initial_observation: Any,
        instruction: str,
        observation_transform: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self._raw_environment = raw_environment
        self._instruction = instruction
        self._observation_transform = observation_transform
        self._observation = self._project_observation(initial_observation)

    def _project_observation(self, raw_observation: Any) -> Mapping[str, Any]:
        observation = canonical_student_observation(
            raw_observation, instruction=self._instruction
        )
        if self._observation_transform is not None:
            observation = self._observation_transform(observation)
            if not isinstance(observation, Mapping):
                raise ContractError("observation_transform must return a mapping")
            validate_student_observation_schema(
                observation,
                require_complete=True,
                schema=CANONICAL_OBSERVATION_SCHEMA,
            )
        return observation

    @property
    def raw_environment(self) -> Any:
        """The exact raw object used for capture/controller hooks."""

        return self._raw_environment

    def observe(self) -> Mapping[str, Any]:
        return dict(self._observation)

    def step(self, action: Sequence[float]) -> Any:
        result, _raw_observation = self._step_and_update(action)
        return _replace_step_observation(result, self._observation, result[1:])

    def step_with_raw_observation(self, action: Sequence[float]) -> Any:
        """Step once while returning raw proprioception to the Arrow controller.

        The facade still updates its canonical observation, so transition
        capture and every student-facing read remain restricted to the four
        canonical fields.  Only the privileged controller view may call this
        method; it needs the raw post-step EEF pose for closed-loop motion.
        """

        result, raw_observation = self._step_and_update(action)
        return _replace_step_observation(result, raw_observation, result[1:])

    def _step_and_update(self, action: Sequence[float]) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
        result = self._raw_environment.step(action)
        observation, rest = _extract_step_observation(result)
        self._observation = self._project_observation(observation)
        return _replace_step_observation(result, observation, rest), observation

    def __getattr__(self, name: str) -> Any:
        # Deliberately delegates only non-lifecycle controller/runtime hooks.
        # TakeoverEnvironmentView performs the stronger reset/close guard.
        if name in {"reset", "close", "reset_model", "hard_reset", "set_state", "set_qpos", "set_qvel"}:
            raise ContractError(f"live collection facade does not expose {name}()")
        return getattr(self._raw_environment, name)


def _extract_step_observation(result: Any) -> tuple[Mapping[str, Any], tuple[Any, ...]]:
    if not isinstance(result, tuple) or len(result) not in (4, 5):
        raise ContractError("native environment step must return a 4- or 5-tuple")
    observation = result[0]
    if not isinstance(observation, Mapping):
        raise ContractError("native environment step observation must be a mapping")
    return observation, result[1:]


def _replace_step_observation(result: Any, observation: Mapping[str, Any], rest: tuple[Any, ...]) -> tuple[Any, ...]:
    return (observation, *rest)


def _default_terminal(_environment: Any, result: Any) -> bool:
    if isinstance(result, Mapping):
        return bool(result.get("done", result.get("terminated", False))) or bool(result.get("truncated", False))
    if isinstance(result, tuple) and len(result) == 5:
        return bool(result[2]) or bool(result[3])
    if isinstance(result, tuple) and len(result) == 4:
        return bool(result[2])
    return False


def _default_success(environment: Any, result: Any) -> bool:
    info: Mapping[str, Any] = {}
    if isinstance(result, Mapping):
        info = result
    elif isinstance(result, tuple) and len(result) in (4, 5) and isinstance(result[-1], Mapping):
        info = result[-1]
    for key in ("success", "task_success", "is_success"):
        if key in info and type(info[key]) is bool:
            if info[key]:
                return True
    check = getattr(environment, "check_success", None)
    if callable(check):
        value = check()
        if type(value) is bool:
            return value
    return False


@dataclass(frozen=True)
class CollectionResult:
    task_id: int
    accepted_target: int
    accepted_count: int
    attempted_count: int
    accepted_path: Path
    failed_path: Path | None
    manifest_path: Path
    manifest_sha256: str


def reset_identity_from_environment(environment: Any, *, task_id: int) -> dict[str, Any]:
    """Return the exact LIBERO reset identity selected by the runtime.

    A numeric episode seed is not sufficient evidence: two seed namespaces can
    still select the same LIBERO init-state row.  The direct LIBERO builder
    therefore publishes the selected row index and its byte-level digest in
    ``_arrow_init_state_diagnostics``.  Collection refuses to continue when
    that authoritative evidence is absent or incomplete.
    """
    diagnostics = getattr(environment, "_arrow_init_state_diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        raise ContractError("fresh Arrow collection requires init-state diagnostics")
    available = diagnostics.get("available_count")
    selected = diagnostics.get("selected_index")
    digest = diagnostics.get("selected_row_sha256", diagnostics.get("init_state_sha256"))
    # Older direct-LIBERO runtimes expose the selected projected table but not
    # its digest. Derive the same dtype/shape/bytes hash used by the evaluator
    # audit rather than falling back to a seed-derived identity.
    if digest is None:
        states = getattr(environment, "_init_states", None)
        if states is None:
            states = getattr(getattr(environment, "raw_environment", None), "_init_states", None)
        if states is not None and selected is not None:
            try:
                import numpy as np
                array = np.asarray(states)
                if array.ndim > 0 and len(array) > int(selected):
                    row = np.ascontiguousarray(np.asarray(array[int(selected)]))
                    if row.dtype != object:
                        digest_builder = hashlib.sha256()
                        digest_builder.update(str(row.dtype).encode("ascii") + b"\0")
                        digest_builder.update(repr(tuple(row.shape)).encode("ascii") + b"\0")
                        digest_builder.update(row.tobytes(order="C"))
                        digest = digest_builder.hexdigest()
                        if available is None:
                            available = len(array)
            except Exception:
                digest = None
    if isinstance(available, bool) or not isinstance(available, int) or available <= 10:
        raise ContractError("fresh Arrow collection requires more than 10 available init states")
    if isinstance(selected, bool) or not isinstance(selected, int) or not 0 <= selected < available:
        raise ContractError("fresh Arrow collection selected init-state index is malformed")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ContractError("fresh Arrow collection requires selected init-state SHA-256 evidence")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ContractError("fresh Arrow selected init-state digest is not SHA-256") from exc
    if int(selected) in SEALED_EVAL_INIT_STATE_INDICES:
        raise ContractError("fresh Arrow collection selected a reserved evaluation init-state index")
    return {
        "task_id": int(task_id),
        "selected_init_state_index": int(selected),
        "init_state_sha256": digest.lower(),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_immutable(path: Path, payload: Any) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable collection artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def _write_jsonl_immutable(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable collection artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (_canonical_json(row) + "\n").encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _collection_contract(
    *, mode: str, task_id: int, task_description: str, policy_id: str,
    accepted_target: int, adaptation_seed_start: int, max_attempts: int,
    teacher_step_budget: int, controller_config_hash: str,
    provenance: Mapping[str, Any] | None, vla_step_budget: int | None = None,
    reserved_eval_init_state_indices: Sequence[int] | None = None,
    reserved_eval_init_state_hashes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the exact restart-binding contract from explicit inputs."""
    contract: dict[str, Any] = {
        "collection_mode": str(mode), "task_id": int(task_id),
        "task_description": str(task_description), "policy_id": str(policy_id),
        "accepted_target": int(accepted_target), "adaptation_seed_start": int(adaptation_seed_start),
        "max_attempts": int(max_attempts), "teacher_step_budget": int(teacher_step_budget),
        "controller_config_hash": str(controller_config_hash).lower(),
        "provenance": _json_safe(dict(provenance or {})),
    }
    if vla_step_budget is not None:
        contract["vla_step_budget"] = int(vla_step_budget)
    if reserved_eval_init_state_indices is not None:
        contract["reserved_eval_init_state_indices"] = [int(value) for value in reserved_eval_init_state_indices]
    if reserved_eval_init_state_hashes is not None:
        contract["reserved_eval_init_state_hashes"] = [str(value).lower() for value in reserved_eval_init_state_hashes]
    return contract


def export_correction_only_lerobot_dataset(
    accepted_episodes_path: str | Path,
    dataset_root: str | Path,
    *,
    repo_id: str = "local/libero_arrow_corrections",
    fps: int = 20,
) -> Mapping[str, Any]:
    """Materialize accepted Arrow rows through the native LeRobot writer.

    Only Arrow suffix rows are emitted.  The complete VLA-prefix/Arrow-suffix
    traces remain in ``accepted_episodes.jsonl`` for audit and replay.  This
    function imports LeRobot lazily so preflight and unit tests remain CPU-only;
    a production collection without LeRobot fails closed.
    """

    try:
        import numpy as np
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ContractError("native LeRobot is required for correction-only PEFT export") from exc
    source = Path(accepted_episodes_path)
    if not source.is_file():
        raise ContractError(f"accepted episode trace does not exist: {source}")
    root = Path(dataset_root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite immutable dataset root: {root}")
    features = {
        "observation.images.image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
        "observation.images.image2": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
    }
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id, fps=int(fps), features=features, root=root,
            robot_type="panda", use_videos=False,
        )
        episode_count = 0
        frame_count = 0
        with source.open("r", encoding="utf-8") as source_handle:
            for line_number, line in enumerate(source_handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ContractError(f"accepted episode trace line {line_number} is invalid JSON") from exc
                if not isinstance(item, Mapping):
                    raise ContractError("accepted episode trace records must be JSON objects")
                current_episode = None
                for transition in item.get("transitions", ()):
                    if transition.get("actor") != "arrow_grasp_controller":
                        continue
                    observation = transition["observation"]
                    frame = {
                        "observation.images.image": np.asarray(observation["agentview"], dtype=np.uint8),
                        "observation.images.image2": np.asarray(observation["wrist"], dtype=np.uint8),
                        "observation.state": np.asarray(observation["state"], dtype=np.float32),
                        "action": np.asarray(transition["action"], dtype=np.float32),
                        "task": str(observation["instruction"]),
                    }
                    if current_episode is None:
                        current_episode = str(item["episode_id"])
                    dataset.add_frame(frame)
                    frame_count += 1
                if current_episode is not None:
                    dataset.save_episode()
                    episode_count += 1
                # Drop the full transition payload before reading the next
                # line.  The cache and collector retain only scalar refs.
                del item
        if frame_count == 0:
            raise ContractError("accepted episodes contain no Arrow correction rows")
        dataset.finalize()
    except Exception as exc:
        raise RuntimeError("native LeRobot correction dataset export failed") from exc
    info = root / "meta" / "info.json"
    if not info.is_file():
        raise ContractError("LeRobot export completed without meta/info.json")
    dataset_manifest = root.parent / f"{root.name}.dataset_manifest.json"
    files = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = path.relative_to(root).as_posix()
        files.append({"path": relative, "sha256": _sha256_file(path), "bytes": path.stat().st_size})
    payload = {
        "schema_version": 1,
        "dataset_root": str(root.resolve()),
        "source_accepted_episodes": str(source.resolve()),
        "source_accepted_episodes_sha256": _sha256_file(source),
        "repo_id": repo_id,
        "fps": int(fps),
        "frames": frame_count,
        "episodes": episode_count,
        "fields": sorted(features),
        "files": files,
        "dataset_tree_sha256": hashlib.sha256(_canonical_json(files).encode("utf-8")).hexdigest(),
    }
    _write_immutable(dataset_manifest, payload)
    return {
        "dataset_root": str(root.resolve()),
        "dataset_manifest_path": str(dataset_manifest.resolve()),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest),
        "frames": frame_count,
        "episodes": episode_count,
    }


def collect_task_corrections(
    *,
    task_id: int,
    task_description: str,
    policy_id: str,
    environment_factory: Callable[[EpisodeSpec], Any],
    reset_environment: Callable[[Any, EpisodeSpec], Any],
    close_environment: Callable[[Any], None],
    vla_action: VLAActionFn,
    teacher_factory: Callable[[EpisodeSpec, Path], ArrowGraspControllerTeacher],
    output_root: str | Path,
    accepted_target: int = 50,
    adaptation_seed_start: int = 3000,
    max_attempts: int | None = 500,
    vla_step_budget: int = 280,
    teacher_step_budget: int = 1200,
    source_state_fn: SourceStateFn,
    success_fn: SuccessFn = _default_success,
    terminal_fn: TerminalFn = _default_terminal,
    controller_config_hash: str,
    dataset_exporter: Callable[[Path, Path], Mapping[str, Any]] | None = None,
    attempt_cleanup_fn: Callable[[], None] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> CollectionResult:
    """Collect exactly ``accepted_target`` evaluator-confirmed corrections.

    Adaptation seeds are monotonically allocated from namespace >=3000 and are
    never reused. Failed Arrow attempts are discarded from training; only
    aggregate failure counts remain in the manifest. Accepted JSONL rows contain the complete
    raw VLA prefix and Arrow suffix from the same environment identity.
    """

    if task_id < 0:
        raise ContractError("task_id must be non-negative")
    if accepted_target <= 0:
        raise ContractError("accepted_target must be positive")
    if adaptation_seed_start < 3000:
        raise ContractError("adaptation seeds must use namespace >=3000")
    if not controller_config_hash or len(controller_config_hash) != 64:
        raise ContractError("controller_config_hash must be a SHA-256 digest")
    if max_attempts is None:
        max_attempts = 500
    if max_attempts < accepted_target:
        raise ContractError("max_attempts must be at least accepted_target")

    root = Path(output_root)
    cache = DurableEpisodeCache(
        root,
        contract=_collection_contract(
            mode="same_episode_takeover", task_id=task_id, task_description=task_description,
            policy_id=policy_id, accepted_target=accepted_target,
            adaptation_seed_start=adaptation_seed_start, max_attempts=max_attempts,
            teacher_step_budget=teacher_step_budget, vla_step_budget=vla_step_budget,
            controller_config_hash=controller_config_hash, provenance=provenance,
        ),
        target=accepted_target,
    )

    for attempt_index in range(cache.next_attempt_index, int(max_attempts)):
        if cache.accepted_count >= accepted_target:
            break
        seed = int(adaptation_seed_start + attempt_index)
        cache.begin_attempt(attempt_index)
        episode = EpisodeSpec(
            episode_id=f"arrow-live-task{task_id}-seed{seed}",
            task_id=int(task_id),
            seed=seed,
            task_description=task_description,
            policy_id=policy_id,
            split="train",
        )
        raw_environment = None
        environment = reset_observation = teacher = view = request = result = None
        transitions = metadata = receipt = accepted_row = None
        try:
            raw_environment = environment_factory(episode)
            reset_observation = reset_environment(raw_environment, episode)
            environment = CanonicalLiveEnvironment(
                raw_environment, initial_observation=reset_observation, instruction=task_description
            )
            teacher = teacher_factory(episode, root / "arrow" / episode.episode_id)
            coordinator = EpisodeCoordinator(
                environment,
                episode,
                observation_fn=lambda env: env.observe(),
                step_fn=lambda env, action: env.step(action),
                observation_schema=CANONICAL_OBSERVATION_SCHEMA,
            )
            result = coordinator.run_vla_then_teacher(
                vla_action,
                teacher,
                success_fn=success_fn,
                terminal_fn=terminal_fn,
                source_state_fn=source_state_fn,
                vla_step_budget=vla_step_budget,
                teacher_step_budget=teacher_step_budget,
                metadata={
                    "source_kind": SOURCE_KIND,
                    "task_id": int(task_id),
                    "seed": seed,
                    "adaptation_seed_namespace": adaptation_seed_start,
                },
            )
            receipt = result.metadata.get("demonstration_receipt")
            teacher_metadata = result.metadata.get("teacher", {})
            evaluator_success = (
                isinstance(teacher_metadata, Mapping)
                and teacher_metadata.get("evaluator_success") is True
            )
            if result.success and result.status is EpisodeStatus.TEACHER_SUCCESS and evaluator_success and receipt:
                cache.add_success({
                    "episode_id": episode.episode_id,
                    "task_id": int(task_id),
                    "seed": seed,
                    "source_kind": SOURCE_KIND,
                    "method_label": PEFT_METHOD_LABEL,
                    "transitions": [row.to_json() for row in result.transitions],
                    "evaluator_receipt": receipt,
                    "metadata": _json_safe(result.metadata),
                })
                # The durable cache owns the only cross-attempt copy.  Do not
                # retain this episode payload in the collector state.
                del result
            else:
                category = str(result.status.value)
                cache.record_failure(category)
        except Exception:
            # Structured VLA/teacher failures are returned above and may be
            # discarded while collection continues. Exceptions mean the
            # runtime contract itself is broken; fail immediately instead of
            # burning hundreds of GPU episodes or selecting around a bug.
            raise
        finally:
            try:
                if raw_environment is not None:
                    close_environment(raw_environment)
            finally:
                if attempt_cleanup_fn is not None:
                    attempt_cleanup_fn()
                print(
                    f"accepted={cache.accepted_count}/{accepted_target} attempted={cache.attempted_count}",
                    flush=True,
                )

    if cache.accepted_count != accepted_target:
        raise ContractError(
            f"accepted target not reached: accepted={cache.accepted_count} target={accepted_target}; "
            f"attempted={cache.attempted_count}"
        )

    accepted_path = root / "accepted_episodes.jsonl"
    accepted_digest = _sha256_file(accepted_path) if accepted_path.exists() else _write_jsonl_immutable(accepted_path, cache.iter_rows())
    dataset_root = root / "lerobot_dataset"
    exporter = dataset_exporter or export_correction_only_lerobot_dataset
    dataset_info = dict(exporter(accepted_path, dataset_root))
    required_dataset_info = {"dataset_root", "dataset_manifest_path", "dataset_manifest_sha256"}
    if not required_dataset_info.issubset(dataset_info):
        raise ContractError(
            "dataset_exporter must return dataset_root, dataset_manifest_path, and dataset_manifest_sha256"
        )
    successful_trajectories = [
        {
            "trajectory_id": ref.episode_id,
            "seed": ref.seed,
            "evaluator_success": True,
        }
        for ref in cache.refs
    ]
    manifest = {
        "schema": COLLECTION_SCHEMA,
        "schema_version": COLLECTION_MANIFEST_VERSION,
        "source_kind": SOURCE_KIND,
        "method_label": PEFT_METHOD_LABEL,
        "task_ids": [int(task_id)],
        "task_id": int(task_id),
        "policy_id": policy_id,
        "accepted_target": int(accepted_target),
        "accepted_count": cache.accepted_count,
        "evaluator_confirmed_successes": cache.accepted_count,
        "successful_trajectories": successful_trajectories,
        "attempted_count": cache.attempted_count,
        "discarded_failure_count": cache.attempted_count - cache.accepted_count,
        "discarded_failure_categories": dict(sorted(cache.failure_categories.items())),
        "accepted_seeds": [ref.seed for ref in cache.refs],
        "adaptation_seeds": [ref.seed for ref in cache.refs],
        "adaptation_seed_namespace": int(adaptation_seed_start),
        "evaluator_receipts": [ref.evaluator_receipt for ref in cache.refs],
        "controller_config_hash": controller_config_hash.lower(),
        "accepted_episodes_jsonl": str(accepted_path),
        "accepted_episodes_sha256": accepted_digest,
        "dataset_root": str(dataset_info["dataset_root"]),
        "dataset_manifest_path": str(dataset_info["dataset_manifest_path"]),
        "dataset_manifest_sha256": str(dataset_info["dataset_manifest_sha256"]),
        "provenance": dict(provenance or {}),
    }
    manifest_path = root / "collection_manifest.json"
    manifest_digest = _write_immutable(manifest_path, manifest)
    return CollectionResult(
        task_id=int(task_id), accepted_target=int(accepted_target), accepted_count=cache.accepted_count,
        attempted_count=cache.attempted_count, accepted_path=accepted_path, failed_path=None,
        manifest_path=manifest_path, manifest_sha256=manifest_digest,
    )


def collect_fresh_arrow_demonstrations(
    *,
    task_id: int,
    task_description: str,
    policy_id: str,
    environment_factory: Callable[[EpisodeSpec], Any],
    reset_environment: Callable[[Any, EpisodeSpec], Any],
    close_environment: Callable[[Any], None],
    teacher_factory: Callable[[EpisodeSpec, Path], ArrowGraspControllerTeacher],
    output_root: str | Path,
    accepted_target: int = 50,
    adaptation_seed_start: int = 3000,
    max_attempts: int | None = 500,
    teacher_step_budget: int = 1200,
    source_state_fn: SourceStateFn,
    controller_config_hash: str,
    dataset_exporter: Callable[[Path, Path], Mapping[str, Any]] | None = None,
    attempt_cleanup_fn: Callable[[], None] | None = None,
    provenance: Mapping[str, Any] | None = None,
    reserved_eval_init_state_indices: Sequence[int] | None = None,
    reserved_eval_init_state_hashes: Sequence[str] | None = None,
    reset_identity_fn: Callable[[Any, int], Mapping[str, Any]] | None = None,
    observation_transform_factory: Callable[
        [Any, EpisodeSpec], Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ] | None = None,
) -> CollectionResult:
    """Collect successful Arrow rollouts, each from a fresh reset.

    Every attempt creates and resets its own sealed-randomized environment;
    Arrow starts at timestep zero and SmolVLA is never constructed or called.
    Failed attempts are discarded together with their temporary controller
    output.  Only evaluator-confirmed Arrow-only traces are written to the
    accepted dataset and immutable manifest.
    """
    if task_id < 0 or accepted_target <= 0:
        raise ContractError("task_id must be non-negative and accepted_target must be positive")
    if adaptation_seed_start < 3000:
        raise ContractError("adaptation seeds must use namespace >=3000")
    if not controller_config_hash or len(controller_config_hash) != 64:
        raise ContractError("controller_config_hash must be a SHA-256 digest")
    max_attempts = 500 if max_attempts is None else int(max_attempts)
    if max_attempts < accepted_target:
        raise ContractError("max_attempts must be at least accepted_target")
    if teacher_step_budget <= 0:
        raise ContractError("teacher_step_budget must be positive")
    reserved_indices = list(SEALED_EVAL_INIT_STATE_INDICES if reserved_eval_init_state_indices is None else reserved_eval_init_state_indices)
    if reserved_indices != list(SEALED_EVAL_INIT_STATE_INDICES):
        raise ContractError("reserved evaluation init-state indices must be exactly 0..9")
    reserved_hashes = list(reserved_eval_init_state_hashes or ())
    if len(reserved_hashes) not in (0, len(reserved_indices)):
        raise ContractError("reserved evaluation init-state hashes must contain exactly ten SHA-256 values")
    if reserved_hashes:
        if any(not isinstance(value, str) or len(value) != 64 for value in reserved_hashes):
            raise ContractError("reserved evaluation init-state hashes must be SHA-256 values")
        if len({value.lower() for value in reserved_hashes}) != len(reserved_hashes):
            raise ContractError("reserved evaluation init-state hashes must be unique")

    root = Path(output_root)
    cache = DurableEpisodeCache(
        root,
        contract=_collection_contract(
            mode="fresh_arrow", task_id=task_id, task_description=task_description,
            policy_id=policy_id, accepted_target=accepted_target,
            adaptation_seed_start=adaptation_seed_start, max_attempts=max_attempts,
            teacher_step_budget=teacher_step_budget,
            controller_config_hash=controller_config_hash, provenance=provenance,
            reserved_eval_init_state_indices=reserved_indices,
            reserved_eval_init_state_hashes=reserved_hashes,
        ),
        target=accepted_target,
    )

    for attempt_index in range(cache.next_attempt_index, max_attempts):
        if cache.accepted_count >= accepted_target:
            break
        seed = int(adaptation_seed_start + attempt_index)
        cache.begin_attempt(attempt_index)
        episode = EpisodeSpec(
            episode_id=f"arrow-fresh-task{task_id}-seed{seed}", task_id=int(task_id), seed=seed,
            task_description=task_description, policy_id=policy_id, split="train",
        )
        raw_environment = None
        observation_transform = None
        # Controller diagnostics are attempt-local.  Removing this directory
        # on both success and failure prevents failed trajectories from
        # becoming an implicit training/raw-trace dataset.
        attempt_output = Path(tempfile.mkdtemp(prefix=f"arrow-attempt-{seed}-"))
        try:
            raw_environment = environment_factory(episode)
            reset_observation = reset_environment(raw_environment, episode)
            if reset_identity_fn is None:
                reset_identity = dict(
                    reset_identity_from_environment(raw_environment, task_id=int(task_id))
                )
            else:
                reset_identity = dict(reset_identity_fn(raw_environment, int(task_id)))
            if reset_identity.get("task_id") != int(task_id):
                raise ContractError("fresh Arrow reset identity task_id differs from collection task")
            selected_index = reset_identity.get("selected_init_state_index")
            digest = str(reset_identity.get("init_state_sha256", "")).lower()
            if not isinstance(selected_index, int) or isinstance(selected_index, bool):
                raise ContractError("fresh Arrow reset identity selected index is malformed")
            if selected_index in reserved_indices or selected_index < 10:
                raise ContractError("fresh Arrow reset identity overlaps reserved evaluation index")
            if reserved_hashes and digest in {value.lower() for value in reserved_hashes}:
                raise ContractError("fresh Arrow reset identity hash overlaps reserved evaluation hash")
            if observation_transform_factory is not None:
                observation_transform = observation_transform_factory(raw_environment, episode)
                if not callable(observation_transform):
                    raise ContractError("observation_transform_factory must return a callable")
            environment = CanonicalLiveEnvironment(
                raw_environment,
                initial_observation=reset_observation,
                instruction=task_description,
                observation_transform=observation_transform,
            )
            source_state = source_state_fn(environment, environment.observe())
            if source_state not in {SourceState.SOURCE_UNHELD, SourceState.SOURCE_HELD}:
                category = str(source_state.value)
                cache.record_failure(category)
                continue
            teacher = teacher_factory(episode, attempt_output)
            if not callable(getattr(teacher, "recover_from_reset", None)):
                raise ContractError("fresh Arrow collection requires teacher.recover_from_reset()")
            view_type = PrivilegedTakeoverEnvironmentView if getattr(
                teacher, "requires_privileged_environment", False
            ) else TakeoverEnvironmentView
            view = view_type(
                environment, expected_identity=id(environment), episode_id=episode.episode_id,
                start_timestep=0, source_state=source_state,
            )
            from .contracts import TeacherRecoveryRequest
            request = TeacherRecoveryRequest(
                episode=episode, source_state=source_state, observation=environment.observe(),
                vla_history=(), remaining_budget=teacher_step_budget,
            )
            try:
                result = teacher.recover_from_reset(view, request)
            except TimeoutError:
                # Motion-phase timeouts are ordinary controller failures for a
                # particular randomized reset, not collection-job failures.
                # Discard the partial trace and advance to the next seeded
                # fresh environment; contract/integrity exceptions still
                # propagate and fail closed.
                cache.record_failure("controller_motion_timeout")
                continue
            if not result.success or result.status is not EpisodeStatus.TEACHER_SUCCESS:
                category = str(result.status.value)
                cache.record_failure(category)
                continue
            metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
            if metadata.get("evaluator_success") is not True:
                cache.record_failure("evaluator_not_confirmed")
                continue
            receipt = metadata.get("demonstration_receipt")
            if not receipt:
                raise ContractError("fresh Arrow success is missing its demonstration receipt")
            transitions = list(result.transitions)
            if not transitions or any(row.actor.value != "arrow_grasp_controller" for row in transitions):
                raise ContractError("fresh Arrow accepted trace contains a non-Arrow transition")
            transform_summary: Mapping[str, Any] | None = None
            summarize_transform = getattr(observation_transform, "audit_summary", None)
            if callable(summarize_transform):
                transform_summary = summarize_transform()
                if not isinstance(transform_summary, Mapping):
                    raise ContractError("observation transform audit_summary must return a mapping")
            accepted_metadata = {
                **dict(metadata), "collection_mode": "fresh_arrow", "vla_called": False
            }
            if transform_summary is not None:
                accepted_metadata["student_observation_transform"] = dict(transform_summary)
            accepted_row = {
                "episode_id": episode.episode_id, "task_id": int(task_id), "seed": seed,
                "source_kind": FRESH_SOURCE_KIND, "method_label": FRESH_PEFT_METHOD_LABEL,
                "collection_mode": "fresh_arrow", "transitions": [row.to_json() for row in transitions],
                "evaluator_receipt": receipt,
                "reset_identity": reset_identity,
                "metadata": _json_safe(accepted_metadata),
            }
            cache.add_success(accepted_row)
        finally:
            try:
                if raw_environment is not None:
                    close_environment(raw_environment)
            finally:
                if attempt_cleanup_fn is not None:
                    attempt_cleanup_fn()
                # Do not carry image-heavy episode objects into the next
                # rollout. Cache files are plain CPU JSON; no CUDA tensor is
                # retained here.
                accepted_row = transitions = result = request = view = teacher = None
                metadata = receipt = environment = reset_observation = raw_environment = None
                observation_transform = None
                shutil.rmtree(attempt_output, ignore_errors=True)
                print(
                    f"accepted={cache.accepted_count}/{accepted_target} attempted={cache.attempted_count}",
                    flush=True,
                )

    if cache.accepted_count != accepted_target:
        raise ContractError(
            f"accepted target not reached: accepted={cache.accepted_count} target={accepted_target}; attempted={cache.attempted_count}"
        )
    root.mkdir(parents=True, exist_ok=True)
    accepted_path = root / "accepted_episodes.jsonl"
    accepted_digest = _sha256_file(accepted_path) if accepted_path.exists() else _write_jsonl_immutable(accepted_path, cache.iter_rows())
    dataset_root = root / "lerobot_dataset"
    exporter = dataset_exporter or export_correction_only_lerobot_dataset
    dataset_info = dict(exporter(accepted_path, dataset_root))
    required_dataset_info = {"dataset_root", "dataset_manifest_path", "dataset_manifest_sha256"}
    if not required_dataset_info.issubset(dataset_info):
        raise ContractError("dataset_exporter must return dataset_root, dataset_manifest_path, and dataset_manifest_sha256")
    successful_trajectories = [
        {"trajectory_id": ref.episode_id, "seed": ref.seed, "evaluator_success": True,
         "reset_identity": dict(ref.reset_identity or {})}
        for ref in cache.refs
    ]
    manifest = {
        "schema": FRESH_COLLECTION_SCHEMA, "schema_version": COLLECTION_MANIFEST_VERSION,
        "source_kind": FRESH_SOURCE_KIND, "method_label": FRESH_PEFT_METHOD_LABEL,
        "collection_mode": "fresh_arrow", "starts_from_reset": True, "vla_called": False,
        "task_ids": [int(task_id)], "task_id": int(task_id), "policy_id": policy_id,
        "accepted_target": int(accepted_target), "accepted_count": cache.accepted_count,
        "evaluator_confirmed_successes": cache.accepted_count, "successful_trajectories": successful_trajectories,
        "attempted_count": cache.attempted_count, "discarded_failure_count": cache.attempted_count - cache.accepted_count,
        "discarded_failure_categories": dict(sorted(cache.failure_categories.items())),
        "accepted_seeds": [ref.seed for ref in cache.refs], "adaptation_seeds": [ref.seed for ref in cache.refs],
        "accepted_reset_identities": [dict(ref.reset_identity or {}) for ref in cache.refs],
        "reserved_eval_init_state_indices": reserved_indices,
        "reserved_eval_init_state_hashes": [value.lower() for value in reserved_hashes],
        "adaptation_seed_namespace": int(adaptation_seed_start),
        "evaluator_receipts": [ref.evaluator_receipt for ref in cache.refs],
        "controller_config_hash": controller_config_hash.lower(),
        "accepted_episodes_jsonl": str(accepted_path), "accepted_episodes_sha256": accepted_digest,
        "dataset_root": str(dataset_info["dataset_root"]),
        "dataset_manifest_path": str(dataset_info["dataset_manifest_path"]),
        "dataset_manifest_sha256": str(dataset_info["dataset_manifest_sha256"]),
        "provenance": dict(provenance or {}),
    }
    manifest_path = root / "collection_manifest.json"
    manifest_digest = _write_immutable(manifest_path, manifest)
    return CollectionResult(
        task_id=int(task_id), accepted_target=int(accepted_target), accepted_count=cache.accepted_count,
        attempted_count=cache.attempted_count, accepted_path=accepted_path, failed_path=None,
        manifest_path=manifest_path, manifest_sha256=manifest_digest,
    )

__all__ = [
    "COLLECTION_SCHEMA", "SOURCE_KIND", "PEFT_METHOD_LABEL", "CanonicalLiveEnvironment",
    "FRESH_COLLECTION_SCHEMA", "FRESH_SOURCE_KIND", "FRESH_PEFT_METHOD_LABEL",
    "CollectionResult", "canonical_student_observation", "collect_task_corrections",
    "collect_fresh_arrow_demonstrations",
    "export_correction_only_lerobot_dataset",
]
