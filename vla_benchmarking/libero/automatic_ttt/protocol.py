"""Frozen, preregistered RoboTTT/LIBERO experiment protocol.

This file contains study design, not claims that the NVIDIA paper used these
LIBERO counts.  The 100-trajectory/round Arrow collection is an explicit
proposed setting for this repository and is labelled accordingly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import re
from typing import Any, Mapping, Sequence


ADAPTATION_TASKS = (0, 2, 4, 6, 8)
TRANSFER_TASKS = (1, 3, 5, 7, 9)
SEEDS = (17, 29, 43)
CHECKPOINTS = (0, 1, 2, 3)


class ProtocolError(ValueError):
    pass


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_lower_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and value == value.lower() and all(
        character in "0123456789abcdef" for character in value
    )


_QUERY_ID_PATTERN = re.compile(r"(?:^|[_-])query[_-]?(?P<index>\d+)(?:$|[_-])", re.IGNORECASE)


def _query_index(entry: "ResetStateEntry") -> int:
    if entry.query_index is not None:
        return entry.query_index
    match = _QUERY_ID_PATTERN.search(entry.episode_id)
    if match is None:
        raise ProtocolError(
            f"scored reset entry {entry.episode_id!r} requires explicit query_index "
            "or canonical queryNN in episode_id"
        )
    query_index = int(match.group("index"))
    if query_index < 0:
        raise ProtocolError("query index must be non-negative")
    return query_index


@dataclass(frozen=True)
class ResetStateEntry:
    """Authoritative reset identity used for paired episode layouts.

    ``observation_sha256`` is retained for auditability, but it is not a
    substitute for the simulator-state digest/replay key when a host can
    export the underlying state.
    """

    task_id: int
    episode_id: str
    seed: int
    reset_index: int
    simulator_state_sha256: str | None
    simulator_replay_key: str | None
    observation_sha256: str
    environment_fingerprint: str
    # ``episode_id`` may carry ``queryNN`` for backward-compatible pilot
    # manifests, but scored locks must expose this index explicitly or use
    # that canonical equivalent.  It is not safe to infer coverage from the
    # number of rows or from opaque episode IDs.
    query_index: int | None = None

    def validate(self) -> None:
        if self.task_id < 0 or self.seed < 0 or self.reset_index != 1:
            raise ProtocolError("reset entries require non-negative task/seed and exactly one reset")
        if not self.episode_id or not self.environment_fingerprint:
            raise ProtocolError("reset entry identity is incomplete")
        if not _is_lower_sha256(self.observation_sha256):
            raise ProtocolError("reset observation digest must be SHA-256")
        if self.simulator_state_sha256 is None and not self.simulator_replay_key:
            raise ProtocolError("reset entry requires simulator state digest or replay key")
        if self.simulator_state_sha256 is not None and not _is_lower_sha256(self.simulator_state_sha256):
            raise ProtocolError("simulator state digest must be SHA-256")
        if self.query_index is not None and (isinstance(self.query_index, bool) or self.query_index < 0):
            raise ProtocolError("query_index must be a non-negative integer")


@dataclass(frozen=True)
class ProtocolLockReceipt:
    """Immutable receipt proving the protocol was frozen before policy actions."""

    protocol_id: str
    protocol_digest: str
    reset_manifest_digest: str
    locked_at_utc: str
    scope: str = "scored"
    first_policy_action_recorded: bool = False

    def validate(self) -> None:
        if not self.protocol_id or not _is_lower_sha256(self.protocol_digest) or not _is_lower_sha256(self.reset_manifest_digest):
            raise ProtocolError("protocol lock receipt has invalid digests")
        if self.scope not in {"scored", "pilot"}:
            raise ProtocolError("protocol lock scope must be scored or pilot")
        try:
            datetime.fromisoformat(self.locked_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ProtocolError("locked_at_utc must be ISO-8601") from exc
        if self.first_policy_action_recorded:
            raise ProtocolError("protocol must be locked before first policy action")

    def to_json(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class CostReceipt:
    run_id: str
    gpu_type: str
    gpu_count: int
    estimated_gpu_hours: float
    actual_gpu_hours: float | None
    artifact_uri: str
    collection_env_steps: int
    training_outer_steps: int

    def validate(self) -> None:
        if not self.run_id or not self.gpu_type or not self.artifact_uri:
            raise ProtocolError("cost receipt requires run_id, gpu_type, and artifact_uri")
        if self.gpu_count <= 0 or self.estimated_gpu_hours <= 0:
            raise ProtocolError("cost receipt GPU count/hours must be positive")
        if self.actual_gpu_hours is not None and self.actual_gpu_hours < 0:
            raise ProtocolError("actual_gpu_hours cannot be negative")
        if self.collection_env_steps <= 0 or self.training_outer_steps <= 0:
            raise ProtocolError("cost receipt step counts must be positive")


@dataclass(frozen=True)
class RoboTTTProtocol:
    """All counts affecting the proposed automatic-TTT experiment."""

    protocol_id: str
    base_policies: tuple[str, ...]
    total_base_trajectories: int
    trajectories_per_base_policy: int
    accepted_corrections_per_round: int
    adaptation_tasks: tuple[int, ...]
    transfer_tasks: tuple[int, ...]
    seeds: tuple[int, ...]
    rounds: int
    attempts_per_task_per_round: int
    query_episodes_per_task: int
    checkpoints: tuple[int, ...]
    outer_training_steps: int
    context_window: int
    effective_batch_size: int
    expected_rollouts_per_arm: int
    cost_receipt: CostReceipt
    collection_setting_label: str = "proposed_LIBERO_setting_not_paper_count"
    bootstrap_resamples: int = 2000
    improvement_target: float = 0.10
    reset_manifest: tuple[ResetStateEntry, ...] = ()

    def validate(self) -> None:
        if self.total_base_trajectories != 100 or self.trajectories_per_base_policy != 50:
            raise ProtocolError("frozen base pool is exactly 100 trajectories: 50 per base policy")
        if len(self.base_policies) != 2:
            raise ProtocolError("the frozen base pool requires exactly two base policies")
        if self.total_base_trajectories != len(self.base_policies) * self.trajectories_per_base_policy:
            raise ProtocolError("base trajectory count does not equal policies times trajectories per policy")
        if self.accepted_corrections_per_round != 100:
            raise ProtocolError("accepted_corrections_per_round must remain the explicit proposed value 100")
        if set(self.adaptation_tasks) & set(self.transfer_tasks):
            raise ProtocolError("adaptation and transfer tasks overlap")
        if set(self.adaptation_tasks) | set(self.transfer_tasks) != set(range(10)):
            raise ProtocolError("adaptation/transfer tasks must partition LIBERO task IDs 0..9")
        if self.adaptation_tasks != ADAPTATION_TASKS or self.transfer_tasks != TRANSFER_TASKS:
            raise ProtocolError("task assignment differs from the frozen protocol")
        if self.seeds != SEEDS:
            raise ProtocolError("seed list differs from the frozen protocol")
        if self.rounds != 3 or self.attempts_per_task_per_round != 20:
            raise ProtocolError("frozen collection count is 3 rounds x 20 attempts/task/round")
        if self.query_episodes_per_task != 50 or self.checkpoints != CHECKPOINTS:
            raise ProtocolError("frozen query/checkpoint counts are 50 episodes/task and checkpoints 0..3")
        if self.outer_training_steps != 20000 or self.context_window != 1000 or self.effective_batch_size != 8:
            raise ProtocolError("training counts must be 20,000 steps, context 1,000, effective batch 8")
        if self.expected_rollouts_per_arm != 6000:
            raise ProtocolError("expected_rollouts_per_arm must be 6000")
        if self.bootstrap_resamples < 1000 or not 0 < self.improvement_target < 1:
            raise ProtocolError("bootstrap count/target are invalid")
        self.cost_receipt.validate()
        for entry in self.reset_manifest:
            entry.validate()
        ids = [entry.episode_id for entry in self.reset_manifest]
        if len(ids) != len(set(ids)):
            raise ProtocolError("reset manifest contains duplicate episode IDs")

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        self.validate()
        encoded = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def lock(
        self,
        reset_manifest: Sequence[ResetStateEntry],
        *,
        locked_at_utc: str | None = None,
        pilot: bool = False,
    ) -> ProtocolLockReceipt:
        """Freeze reset identities and return a receipt required by scored hosts."""
        entries = tuple(reset_manifest)
        if not entries:
            raise ProtocolError("cannot lock a scored protocol without a reset manifest")
        expected_entries = len(self.adaptation_tasks + self.transfer_tasks) * len(self.seeds) * self.query_episodes_per_task
        if not pilot and len(entries) != expected_entries:
            raise ProtocolError(
                f"scored reset manifest requires {expected_entries} task/seed/query entries; observed {len(entries)}; "
                "use pilot=True for an explicitly unscored canary"
            )
        if not pilot:
            self._validate_scored_coverage(entries)
        candidate = replace(self, reset_manifest=entries)
        candidate.validate()
        manifest_blob = json.dumps(
            [asdict(entry) for entry in entries], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        timestamp = locked_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        receipt = ProtocolLockReceipt(
            protocol_id=candidate.protocol_id,
            protocol_digest=candidate.digest(),
            reset_manifest_digest=hashlib.sha256(manifest_blob).hexdigest(),
            locked_at_utc=timestamp,
            scope="pilot" if pilot else "scored",
        )
        receipt.validate()
        return receipt

    def _validate_scored_coverage(self, entries: Sequence[ResetStateEntry]) -> None:
        """Require the exact task×seed×query grid for a scored lock.

        A length check is insufficient: 1,500 rows can all be task 0.  The
        key includes the authoritative task and seed fields plus an explicit
        query index (or the canonical ``queryNN`` episode-id suffix retained
        for older pilot manifests).
        """

        tasks = tuple(self.adaptation_tasks + self.transfer_tasks)
        expected = {(task_id, seed, query_index) for task_id in tasks for seed in self.seeds for query_index in range(self.query_episodes_per_task)}
        observed: set[tuple[int, int, int]] = set()
        for entry in entries:
            query_index = _query_index(entry)
            key = (entry.task_id, entry.seed, query_index)
            if key in observed:
                raise ProtocolError(f"duplicate scored reset coverage key {key}")
            observed.add(key)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ProtocolError(
                "scored reset manifest does not cover the exact task×seed×query grid; "
                f"missing={len(missing)} extra={len(extra)} "
                f"missing_examples={missing[:3]} extra_examples={extra[:3]}"
            )

    def count_report(self) -> dict[str, Any]:
        self.validate()
        return {
            "base_pool_total": self.total_base_trajectories,
            "base_pool_per_policy": self.trajectories_per_base_policy,
            "accepted_arrow_corrections_per_round": self.accepted_corrections_per_round,
            "adaptation_tasks": list(self.adaptation_tasks),
            "transfer_tasks": list(self.transfer_tasks),
            "seeds": list(self.seeds),
            "rounds": self.rounds,
            "attempts_per_task_per_round": self.attempts_per_task_per_round,
            "query_episodes_per_task": self.query_episodes_per_task,
            "checkpoints": list(self.checkpoints),
            "outer_training_steps": self.outer_training_steps,
            "context_window": self.context_window,
            "effective_batch_size": self.effective_batch_size,
            "expected_rollouts_per_arm": self.expected_rollouts_per_arm,
            "study_setting": self.collection_setting_label,
        }


def make_default_protocol(cost_receipt: CostReceipt) -> RoboTTTProtocol:
    """Construct the frozen protocol; no runtime/cost values are hidden."""
    return RoboTTTProtocol(
        protocol_id="automatic-robottt-libero-v1",
        base_policies=("openvla", "pi05"),
        total_base_trajectories=100,
        trajectories_per_base_policy=50,
        accepted_corrections_per_round=100,
        adaptation_tasks=ADAPTATION_TASKS,
        transfer_tasks=TRANSFER_TASKS,
        seeds=SEEDS,
        rounds=3,
        attempts_per_task_per_round=20,
        query_episodes_per_task=50,
        checkpoints=CHECKPOINTS,
        outer_training_steps=20000,
        context_window=1000,
        effective_batch_size=8,
        expected_rollouts_per_arm=6000,
        cost_receipt=cost_receipt,
    )


def count_accepted_arrow_trajectories(receipts: Sequence[Mapping[str, Any]]) -> int:
    """Count only evaluator-confirmed Arrow demonstrations in a round.

    A receipt must identify a unique episode and carry both teacher and
    evaluator success.  Failed attempts stay in the raw archive and are not
    counted as demonstrations.
    """
    seen: set[str] = set()
    accepted = 0
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, Mapping):
            raise ProtocolError(f"Arrow receipt {index} is not a mapping")
        episode_id = receipt.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ProtocolError(f"Arrow receipt {index} lacks episode_id")
        if episode_id in seen:
            raise ProtocolError(f"duplicate Arrow receipt for episode {episode_id!r}")
        seen.add(episode_id)
        if receipt.get("teacher_success") is True and receipt.get("evaluator_success") is True:
            accepted += 1
    return accepted


def require_arrow_target(receipts: Sequence[Mapping[str, Any]], *, target: int = 100) -> int:
    """Require one complete, registered adaptation round of Arrow attempts.

    The input is one round, not an aggregate over rounds.  Every one of the
    five adaptation tasks must contribute exactly 20 unique attempts, each
    carrying a registered seed and round index.  Only attempts with explicit
    teacher and evaluator success become training demonstrations.
    """
    if target <= 0:
        raise ProtocolError("Arrow target must be positive")
    if target != 100:
        # The frozen study target and allocation are inseparable.  Allowing a
        # smaller target would silently turn a partial pilot into scored data.
        raise ProtocolError("scored Arrow target must remain exactly 100 (5 adaptation tasks × 20 attempts)")
    if len(receipts) != 100:
        raise ProtocolError(
            "Arrow round requires exactly 100 registered attempts "
            "(5 adaptation tasks × 20 attempts)"
        )
    allocation: set[tuple[int, int]] = set()
    observed_rounds: set[int] = set()
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, Mapping):
            raise ProtocolError(f"Arrow receipt {index} is not a mapping")
        task_id = receipt.get("task_id")
        seed = receipt.get("seed")
        round_index = receipt.get("round")
        attempt_index = receipt.get("attempt_index")
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id not in ADAPTATION_TASKS:
            raise ProtocolError(f"Arrow receipt {index} task_id must be a registered adaptation task")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
            raise ProtocolError(f"Arrow receipt {index} seed must be one of registered adaptation seeds {SEEDS}")
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index not in range(3):
            raise ProtocolError(f"Arrow receipt {index} round must be one of 0, 1, 2")
        if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index not in range(20):
            raise ProtocolError(f"Arrow receipt {index} attempt_index must be in 0..19")
        key = (task_id, attempt_index)
        if key in allocation:
            raise ProtocolError(f"duplicate Arrow allocation key {key}")
        allocation.add(key)
        observed_rounds.add(round_index)
        if not isinstance(receipt.get("teacher_success"), bool) or not isinstance(receipt.get("evaluator_success"), bool):
            raise ProtocolError(f"Arrow receipt {index} requires boolean teacher_success and evaluator_success")
    expected = {(task_id, attempt_index) for task_id in ADAPTATION_TASKS for attempt_index in range(20)}
    if allocation != expected:
        raise ProtocolError("Arrow receipts do not cover the exact adaptation allocation")
    if len(observed_rounds) != 1:
        raise ProtocolError("Arrow receipts must belong to exactly one registered round")
    accepted = count_accepted_arrow_trajectories(receipts)
    if accepted != target:
        raise ProtocolError(f"accepted Arrow trajectory target is {target}, observed {accepted}")
    return accepted


__all__ = [
    "ADAPTATION_TASKS", "CHECKPOINTS", "CostReceipt", "ProtocolError", "ProtocolLockReceipt",
    "ResetStateEntry", "RoboTTTProtocol", "SEEDS", "TRANSFER_TASKS", "count_accepted_arrow_trajectories",
    "make_default_protocol", "require_arrow_target",
]
