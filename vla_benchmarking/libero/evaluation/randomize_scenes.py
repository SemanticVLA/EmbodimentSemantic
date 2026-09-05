"""Object pose override utilities for LIBERO envs.

Lets you place objects at custom positions before a rollout, with optional
physics-settle while keeping the robot frozen. Designed to be called from
both interactive notebooks and the lerobot eval wrapper.
"""

from __future__ import annotations

import hashlib
import numpy as np
from typing import Any

EXCLUDE_PREFIXES = ("robot0", "gripper")
SETTLE_VEL_THRESHOLD = 0.01  # m/s — below this, an object is considered settled


def sim_state_sha256(sim: Any) -> str:
    """Hash the actual post-randomization simulator state deterministically."""
    getter = getattr(sim, "get_state", None)
    if not callable(getter):
        raise RuntimeError("simulator does not expose get_state(); reset state cannot be audited")
    state = getter()
    chunks: list[tuple[str, np.ndarray]] = []
    if isinstance(state, np.ndarray):
        chunks.append(("state", state))
    else:
        for name in ("time", "qpos", "qvel", "act", "udd_state"):
            value = getattr(state, name, None)
            if value is None or isinstance(value, (dict, list)) and name == "udd_state":
                continue
            try:
                array = np.asarray(value)
            except Exception:
                continue
            if array.dtype == object:
                continue
            chunks.append((name, array))
    if not chunks:
        raise RuntimeError("simulator get_state returned no hashable numeric state")
    digest = hashlib.sha256()
    for name, array in chunks:
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
        digest.update(repr(tuple(contiguous.shape)).encode("ascii") + b"\0")
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def init_state_evidence(env: Any) -> dict[str, Any]:
    """Capture the exact projected init-state row selected by the next reset."""
    index = getattr(env, "_init_state_id", getattr(env, "init_state_id", None))
    states = getattr(env, "_init_states", None)
    if index is None or states is None:
        raise RuntimeError("environment lacks init_state_id/_init_states for reset audit")
    array = np.asarray(states)
    if array.ndim == 0 or len(array) == 0:
        raise RuntimeError("environment init-state table is empty")
    try:
        selected_index = int(index) % len(array)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("environment init_state_id is invalid") from exc
    selected = np.asarray(array[selected_index])
    if selected.dtype == object:
        raise RuntimeError("selected init-state row is not numeric")
    contiguous = np.ascontiguousarray(selected)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
    digest.update(repr(tuple(contiguous.shape)).encode("ascii") + b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return {"selected_index": selected_index, "selected_row_sha256": digest.hexdigest()}


# ═══════════════════════════════════════════════════════════════════════════════
# OBJECT DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def discover_objects(env: Any) -> dict[str, str]:
    """Return {label: body_name} for all manipulable objects in the scene."""
    return {
        env.sim.model.body_id2name(i).replace("_main", ""): env.sim.model.body_id2name(i)
        for i in range(env.sim.model.nbody)
        if env.sim.model.body_id2name(i)
        and env.sim.model.body_id2name(i).endswith("_main")
        and not any(env.sim.model.body_id2name(i).startswith(p) for p in EXCLUDE_PREFIXES)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# POSE GET / SET (low-level, qpos-based)
# ═══════════════════════════════════════════════════════════════════════════════

def _qpos_addr(env: Any, body_name: str) -> int | None:
    """Return the qpos slot index for an object's free joint, or None if fixed."""
    body_id = env.sim.model.body_name2id(body_name)
    joint_id = env.sim.model.body_jntadr[body_id]
    return env.sim.model.jnt_qposadr[joint_id] if joint_id >= 0 else None


def _qvel_addr(env: Any, body_name: str) -> int | None:
    body_id = env.sim.model.body_name2id(body_name)
    joint_id = env.sim.model.body_jntadr[body_id]
    return env.sim.model.jnt_dofadr[joint_id] if joint_id >= 0 else None


def get_object_pose(env: Any, label: str, objects: dict[str, str] | None = None) -> np.ndarray | None:
    """Return [x, y, z, qw, qx, qy, qz] for the given object label."""
    objects = objects or discover_objects(env)
    if label not in objects:
        return None
    addr = _qpos_addr(env, objects[label])
    return env.sim.data.qpos[addr : addr + 7].copy() if addr is not None else None


def set_object_pose(env: Any, label: str, pose: np.ndarray, objects: dict[str, str] | None = None) -> bool:
    """Set object pose via qpos. Returns True on success, False if fixed/missing."""
    objects = objects or discover_objects(env)
    if label not in objects:
        return False
    body_name = objects[label]
    addr = _qpos_addr(env, body_name)
    if addr is None:
        return False
    env.sim.data.qpos[addr : addr + 7] = pose
    # Zero velocity for stability
    vaddr = _qvel_addr(env, body_name)
    if vaddr is not None:
        env.sim.data.qvel[vaddr : vaddr + 6] = 0.0
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# ROBOT STATE FREEZE (for clean physics settle)
# ═══════════════════════════════════════════════════════════════════════════════

class RobotFreezer:
    """Snapshots robot qpos/qvel so we can hold it still during physics settle."""

    def __init__(self, env: Any):
        self.qpos_idx, self.qvel_idx = self._collect_robot_indices(env)
        self.qpos_snapshot = env.sim.data.qpos[self.qpos_idx].copy()
        self.qvel_snapshot = env.sim.data.qvel[self.qvel_idx].copy()

    @staticmethod
    def _collect_robot_indices(env: Any) -> tuple[np.ndarray, np.ndarray]:
        qpos_idx, qvel_idx = [], []
        for body_id in range(env.sim.model.nbody):
            name = env.sim.model.body_id2name(body_id)
            if not name or not name.startswith(("robot0", "gripper")):
                continue
            base_jnt = env.sim.model.body_jntadr[body_id]
            njnt = env.sim.model.body_jntnum[body_id]
            for k in range(njnt):
                jid = base_jnt + k
                if jid < 0 or jid >= env.sim.model.njnt:
                    continue
                jtype = env.sim.model.jnt_type[jid]
                qpos_n, qvel_n = {0: (7, 6), 1: (4, 3)}.get(jtype, (1, 1))
                qpa = env.sim.model.jnt_qposadr[jid]
                qva = env.sim.model.jnt_dofadr[jid]
                qpos_idx.extend(range(qpa, qpa + qpos_n))
                qvel_idx.extend(range(qva, qva + qvel_n))
        return np.array(sorted(set(qpos_idx))), np.array(sorted(set(qvel_idx)))

    def restore(self, env: Any):
        env.sim.data.qpos[self.qpos_idx] = self.qpos_snapshot
        env.sim.data.qvel[self.qvel_idx] = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS SETTLE (with robot frozen)
# ═══════════════════════════════════════════════════════════════════════════════

def settle_physics(
    env: Any,
    max_steps: int = 200,
    vel_threshold: float = SETTLE_VEL_THRESHOLD,
    freezer: RobotFreezer | None = None,
    track_objects: list[str] | None = None,
) -> dict:
    """Step physics with the robot frozen until objects settle or max_steps reached.

    Returns a dict with diagnostics: steps_taken, final_max_vel, traces (if requested).
    """
    if freezer is None:
        freezer = RobotFreezer(env)

    objects = discover_objects(env)
    track_objects = track_objects or []
    traces = {label: [] for label in track_objects}

    steps_taken = 0
    final_max_vel = float("inf")

    for step in range(max_steps):
        freezer.restore(env)
        env.sim.forward()
        env.sim.step()
        steps_taken = step + 1

        # Track requested objects
        for label in track_objects:
            pose = get_object_pose(env, label, objects)
            if pose is not None:
                traces[label].append(pose[:3].copy())

        # Check settling: max velocity across all objects
        max_vel = 0.0
        for label, body_name in objects.items():
            vaddr = _qvel_addr(env, body_name)
            if vaddr is None:
                continue
            v = env.sim.data.qvel[vaddr : vaddr + 3]
            max_vel = max(max_vel, float(np.linalg.norm(v)))
        final_max_vel = max_vel

        if max_vel < vel_threshold:
            break

    # Final restore of robot pose
    freezer.restore(env)
    env.sim.forward()

    return {
        "steps_taken": steps_taken,
        "final_max_vel": final_max_vel,
        "settled": final_max_vel < vel_threshold,
        "traces": traces,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL API
# ═══════════════════════════════════════════════════════════════════════════════

def apply_layout(
    env: Any,
    layout: dict[str, np.ndarray | list],
    settle_steps: int = 0,
    verbose: bool = True,
) -> dict:
    """Apply a custom layout to the env.

    Args:
        env: LIBERO env (the inner MuJoCo env).
        layout: {object_label: [x, y, z, qw, qx, qy, qz]} (list or array, 7 floats).
        settle_steps: If >0, run physics with robot frozen for up to N steps so
            objects rest on their supports. If 0, only kinematics propagated.
        verbose: Print summary.

    Returns:
        Dict with applied/skipped lists and (if settled) physics diagnostics.
    """
    objects = discover_objects(env)
    applied, skipped = [], []

    # Apply each pose
    for label, pose in layout.items():
        pose = np.array(pose, dtype=float)
        if pose.shape != (7,):
            skipped.append((label, f"bad pose shape {pose.shape}"))
            continue
        if not set_object_pose(env, label, pose, objects):
            skipped.append((label, "object not found or fixed"))
            continue
        applied.append(label)

    env.sim.forward()

    # Optional settle
    settle_info = None
    if settle_steps > 0 and applied:
        freezer = RobotFreezer(env)
        # NOTE: snapshot AFTER applying poses so we don't capture pre-layout robot state
        # (actually robot state is unchanged by set_object_pose, so this is moot —
        # but freezer captures whatever's there now)
        settle_info = settle_physics(env, max_steps=settle_steps, freezer=freezer)

    if verbose:
        print(f"[apply_layout] applied: {applied}")
        if skipped:
            print(f"[apply_layout] skipped: {skipped}")
        if settle_info:
            status = "✅ settled" if settle_info["settled"] else "⚠️  not fully settled"
            print(f"[apply_layout] {status} after {settle_info['steps_taken']} steps "
                  f"(final max vel: {settle_info['final_max_vel']:.4f} m/s)")

    return {"applied": applied, "skipped": skipped, "settle_info": settle_info}


def swap_objects(
    env: Any,
    label_a: str,
    label_b: str,
    settle_steps: int = 0,
    verbose: bool = True,
) -> dict:
    """Swap two objects' full poses. Convenience wrapper around apply_layout."""
    pose_a = get_object_pose(env, label_a)
    pose_b = get_object_pose(env, label_b)
    if pose_a is None or pose_b is None:
        if verbose:
            print(f"[swap_objects] cannot swap: missing pose for {label_a} or {label_b}")
        return {"applied": [], "skipped": [(label_a, "no pose"), (label_b, "no pose")]}

    layout = {label_a: pose_b, label_b: pose_a}
    return apply_layout(env, layout, settle_steps=settle_steps, verbose=verbose)


# ═══════════════════════════════════════════════════════════════════════════════
# VEC-ENV HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_envs(vec_env: Any) -> list:
    """Return the sub-env list from any wrapper depth.

    Works for SyncVectorEnv (has .envs directly) and wrappers that forward via
    __getattr__ (TaskContextVecEnv, SceneRandomizerVecEnvWrapper).

    For _LazyAsyncVectorEnv (n_envs>1 async): sub-envs live in worker subprocesses
    and are not accessible as Python objects. Raises RuntimeError with a clear
    message telling the user to switch to batch_size=1.
    """
    visited: set[int] = set()
    obj = vec_env
    while obj is not None and id(obj) not in visited:
        visited.add(id(obj))
        # Direct .envs attribute (SyncVectorEnv, or any wrapper that exposes it)
        if "envs" in getattr(obj, "__dict__", {}):
            return obj.envs
        # _LazyAsyncVectorEnv: ._env is None before first reset, real AsyncVectorEnv after
        inner = obj.__dict__.get("_env")
        if inner is not None:
            if hasattr(inner, "envs"):
                return inner.envs
            raise RuntimeError(
                "_resolve_envs: sub-envs are running in worker subprocesses "
                "(AsyncVectorEnv / batch_size>1) and cannot be accessed as Python "
                "objects. Use --eval.batch_size=1 for per-env operations "
                "(prompt override, scene randomization, semantic context)."
            )
        obj = obj.__dict__.get("env")
    raise RuntimeError(f"_resolve_envs: could not find .envs on {type(vec_env).__name__}")

def _replace_batched_observation_slot(batched_obs, single_obs, index):
    if isinstance(batched_obs, dict):
        for key, value in single_obs.items():
            _replace_batched_observation_slot(batched_obs[key], value, index)
        return
    batched_obs[index] = single_obs

# ═══════════════════════════════════════════════════════════════════════════════
# LEROBOT VEC-ENV WRAPPER (hooks randomization into every episode reset)
# ═══════════════════════════════════════════════════════════════════════════════

class SceneRandomizerVecEnvWrapper:
    """Wraps a lerobot VecEnv and applies TASK_SWAP_CONFIG after every reset.

    Access pattern assumed:
        vec_env.envs[i]        — lerobot gym wrapper
        vec_env.envs[i]._env   — LIBERO ControlEnv (has .sim)
    This matches how libero_live_semantic_context.py accesses sub-envs.
    """

    def __init__(
        self,
        env: Any,
        task_id: int,
        swap_config: dict,
        settle_steps: int = 0,
        verbose: bool = False,
        audit_logger: Any | None = None,
        removal_config: dict | None = None,
    ):
        self.env = env
        self.task_id = int(task_id)
        self.swaps = swap_config.get(self.task_id, [])
        self.settle_steps = settle_steps
        self.verbose = verbose
        self.audit_logger = audit_logger
        self.removal_config = removal_config or {}
        self._reset_sequence = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


    def reset(self, id=None, **kwargs):
        self._reset_sequence += 1
        reset_sequence = self._reset_sequence
        # Capture the row that LeRobot is about to select before reset
        # increments its init_state_id. This is required for paired-cell
        # evaluation and is deliberately fail-closed.
        envs_before = _resolve_envs(self.env)
        # LeRobot's lazy wrapper materializes the underlying environment (and
        # its projected _init_states table) before reset.  This does not select
        # or increment a row; reset still owns that operation below.
        for env_before in envs_before:
            ensure_env = getattr(env_before, "_ensure_env", None)
            if callable(ensure_env):
                ensure_env()
        ids_before = list(range(len(envs_before))) if id is None else ([id] if isinstance(id, int) else list(id))
        init_evidence = {
            env_index: init_state_evidence(envs_before[env_index])
            for env_index in ids_before
        }
        result = self.env.reset(**kwargs) if id is None else self.env.reset(id=id, **kwargs)
        returns_info = isinstance(result, tuple) and len(result) == 2
        obs = result[0] if returns_info else result
        if not self.swaps:
            envs = _resolve_envs(self.env)
            ids = list(range(len(envs))) if id is None else ([id] if isinstance(id, int) else list(id))
            reset_evidence = []
            for env_index in ids:
                lerobot_env = envs[env_index]
                lerobot_env._randomization_env_index = env_index
                lerobot_env._randomization_reset_sequence = reset_sequence
                if self.audit_logger is not None:
                    self.audit_logger.log(
                        task_id=self.task_id,
                        env_index=env_index,
                        reset_sequence=reset_sequence,
                        dimensions_enabled={"scene_layout": False, "object_removal": bool(self.removal_config.get(self.task_id))},
                        dimensions_realized={},
                        details={"status": "pending"},
                    )
                inner_env = lerobot_env._env
                protected = self._snapshot_protected(inner_env)
                removed = self._verify_removed(inner_env)
                projection = self._projection_evidence(lerobot_env, inner_env)
                if not projection.get("success", False) or (projection.get("required") and not projection.get("projected")):
                    raise RuntimeError(f"task {self.task_id} init-state projection failed: {projection}")
                inner_env.sim.forward()
                self._verify_protected(inner_env, protected)
                state_hash = sim_state_sha256(inner_env.sim)
                reset_evidence.append((env_index, removed, projection, protected, state_hash, init_evidence[env_index]))
            if self.audit_logger is not None:
                for env_index, removed, projection, protected, state_hash, init_state in reset_evidence:
                    self.audit_logger.log(
                        task_id=self.task_id,
                        env_index=env_index,
                        reset_sequence=reset_sequence,
                        dimensions_enabled={
                            "scene_layout": False,
                            "object_removal": bool(self.removal_config.get(self.task_id)),
                        },
                        dimensions_realized={
                            "scene_layout": False,
                            "object_removal": bool(self.removal_config.get(self.task_id)),
                        },
                        details={"swaps": [], "removed": removed, "projection": projection, "protected": self._protected_evidence(protected), "sim_state_sha256": state_hash, "init_state": init_state, "status": "environment_ok"},
                    )
            return result
        envs = _resolve_envs(self.env)
        n = len(envs)
        ids = list(range(n)) if id is None else ([id] if isinstance(id, int) else list(id))
        realized_layout: list[dict] = []
        for obs_index, env_index in enumerate(ids):
            lerobot_env = envs[env_index]
            lerobot_env._randomization_env_index = env_index
            lerobot_env._randomization_reset_sequence = reset_sequence
            if self.audit_logger is not None:
                self.audit_logger.log(
                    task_id=self.task_id,
                    env_index=env_index,
                    reset_sequence=reset_sequence,
                    dimensions_enabled={"scene_layout": True, "object_removal": bool(self.removal_config.get(self.task_id))},
                    dimensions_realized={},
                    details={"status": "pending"},
                )
            inner_env = lerobot_env._env
            protected = self._snapshot_protected(inner_env)
            removed = self._verify_removed(inner_env)
            projection = self._projection_evidence(lerobot_env, inner_env)
            if not projection.get("success", False) or (projection.get("required") and not projection.get("projected")):
                raise RuntimeError(f"task {self.task_id} init-state projection failed: {projection}")
            layout_results = self._apply_swaps(inner_env)
            expected_labels = [label for swap in self.swaps for label in swap]
            applied_labels = [label for result_item in layout_results for label in result_item.get("applied", [])]
            skipped = [item for result_item in layout_results for item in result_item.get("skipped", [])]
            if skipped or sorted(applied_labels) != sorted(expected_labels):
                raise RuntimeError(
                    f"task {self.task_id} layout was not fully applied: "
                    f"expected={expected_labels}, applied={applied_labels}, skipped={skipped}"
                )
            self._restore_protected(inner_env, protected)
            inner_env.sim.forward()
            self._verify_protected(inner_env, protected)
            state_hash = sim_state_sha256(inner_env.sim)
            realized_layout.append({
                "configured": self.swaps,
                "applied": [label for result in layout_results for label in result.get("applied", [])],
                "skipped": [item for result in layout_results for item in result.get("skipped", [])],
                "removed": removed,
                "projection": projection,
                "protected": self._protected_evidence(protected),
                "sim_state_sha256": state_hash,
                "init_state": init_evidence[env_index],
            })
            inner_env.sim.forward()
            raw_obs = inner_env.env._get_observations(force_update=True)
            fresh_obs = lerobot_env._format_raw_obs(raw_obs)
            _replace_batched_observation_slot(obs, fresh_obs, obs_index)
        if self.audit_logger is not None:
            for env_index, evidence in zip(ids, realized_layout):
                self.audit_logger.log(
                    task_id=self.task_id,
                    env_index=env_index,
                    reset_sequence=reset_sequence,
                    dimensions_enabled={"scene_layout": True, "object_removal": bool(self.removal_config.get(self.task_id))},
                    dimensions_realized={"scene_layout": True, "object_removal": bool(self.removal_config.get(self.task_id))},
                    details={
                        "swaps": self.swaps,
                        "layout": evidence,
                        "removed": evidence["removed"],
                        "projection": evidence["projection"],
                        "protected": evidence["protected"],
                        "sim_state_sha256": evidence["sim_state_sha256"],
                        "init_state": evidence["init_state"],
                        "status": "environment_ok",
                    },
                )
        return (obs, result[1]) if returns_info else obs

    @staticmethod
    def _projection_evidence(lerobot_env: Any, inner_env: Any) -> dict:
        return getattr(
            lerobot_env,
            "_init_state_projection_evidence",
            getattr(inner_env, "_init_state_projection_evidence", {"required": False, "success": True}),
        )

    def _verify_removed(self, inner_env: Any) -> list[str]:
        configured = list(self.removal_config.get(self.task_id, []))
        present = discover_objects(inner_env)
        remaining = [label for label in configured if label in present]
        if remaining:
            raise RuntimeError(f"configured removed objects still present after reset: {remaining}")
        return configured

    @staticmethod
    def _snapshot_protected(inner_env: Any) -> dict[str, np.ndarray]:
        snapshot = {}
        for label in ("akita_black_bowl_1", "plate_1"):
            pose = get_object_pose(inner_env, label)
            if pose is None:
                raise RuntimeError(f"protected object missing before layout: {label}")
            snapshot[label] = pose
        return snapshot

    @staticmethod
    def _restore_protected(inner_env: Any, snapshot: dict[str, np.ndarray]) -> None:
        for label, pose in snapshot.items():
            if not set_object_pose(inner_env, label, pose):
                raise RuntimeError(f"protected object missing while restoring layout: {label}")

    @staticmethod
    def _verify_protected(inner_env: Any, snapshot: dict[str, np.ndarray]) -> None:
        for label, expected in snapshot.items():
            actual = get_object_pose(inner_env, label)
            if actual is None or not np.allclose(actual, expected, atol=1e-7, rtol=0):
                raise RuntimeError(f"protected object pose changed during layout: {label}")

    @staticmethod
    def _protected_evidence(snapshot: dict[str, np.ndarray]) -> dict[str, bool]:
        return {label: True for label in snapshot}

    def _apply_swaps(self, inner_env: Any) -> list[dict]:
        results = []
        for obj_a, obj_b in self.swaps:
            if not isinstance(obj_a, str) or not isinstance(obj_b, str):
                raise TypeError(
                    "TASK_SWAP_CONFIG entries must be string object-name pairs; "
                    f"got ({obj_a!r}, {obj_b!r})"
                )
            results.append(
                swap_objects(
                    inner_env,
                    obj_a,
                    obj_b,
                    settle_steps=self.settle_steps,
                    verbose=self.verbose,
                )
            )
        return results
