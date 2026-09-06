from __future__ import annotations

import copy

import numpy as np
import pytest

from vla_benchmarking.libero.evaluation.contracts import build_task_seed_matrix
from vla_benchmarking.libero.evaluation.plan import (
    LEGACY_PLAN_SCHEMA,
    build_evaluation_plan,
    validate_plan,
)
from vla_benchmarking.libero.evaluation.policy_adapter import (
    CallablePolicyAdapter,
    bind_adapter_provenance,
    derive_checkpoint_receipt,
    derive_dataset_manifest_receipt,
    derive_io_receipt,
    derive_runtime_receipt,
    PolicyMetadata,
    validate_adapter_metadata,
)
from vla_benchmarking.libero.evaluation.run_policy_eval import (
    EpisodeOutcome,
    EpisodeSpec,
    run_policy_eval,
)
from vla_benchmarking.libero.evaluation.libero_policy_rollout import (
    CallableLiberoEnvironmentFactory,
    run_libero_policy_eval,
)
from vla_benchmarking.libero.finetuned_vlas.common.manifest import (
    build_plan_bindings,
    build_policy_manifest,
    validate_manifest,
)
from vla_benchmarking.libero.finetuned_vlas.common.source_contract import DatasetSourceContract


def _source() -> DatasetSourceContract:
    digest = "a" * 64
    return DatasetSourceContract(
        source_id="libero-no-arrow-500",
        source_revision="source-rev",
        episode_count=500,
        timestep_count=62250,
        schema_sha256=digest,
        source_sha256=digest,
        split_sha256=digest,
    )


def test_action_contract_honors_native_horizon_and_arrow_marker():
    metadata = PolicyMetadata(
        policy_kind="octo_base15_spatial_no_arrow_matched",
        artifact_id="test",
        checkpoint_revision="rev",
        backend="native_policy",
        native_action_horizon=4,
    )
    adapter = CallablePolicyAdapter(metadata, act_fn=lambda _: np.zeros((4, 7), dtype=np.float32))
    observed = {"agentview": np.zeros((256, 256, 3), dtype=np.uint8)}
    assert adapter.act(observed).shape == (4, 7)
    with pytest.raises(ValueError, match="arrow-free"):
        adapter.act({**observed, "arrow_overlay": True})
    bad = CallablePolicyAdapter(metadata, act_fn=lambda _: np.zeros((10, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        bad.act(observed)


def test_real_adapter_metadata_can_be_bound_to_immutable_receipts():
    """The shared seam attaches v2 identities without changing native adapters."""

    revision = "a" * 40
    metadata = PolicyMetadata(
        policy_kind="pi05",
        artifact_id="artifact",
        checkpoint_revision=revision,
        backend="lerobot",
        adapter_kind="pi05",
        native_action_horizon=10,
    )
    native = CallablePolicyAdapter(metadata, act_fn=lambda _: np.zeros((10, 7), dtype=np.float32))
    bound = bind_adapter_provenance(
        native,
        artifact={"id": "artifact", "revision": revision},
        runtime={"id": "runtime-receipt", "sha256": "b" * 64},
        io={"id": "io-receipt", "sha256": "c" * 64},
        dataset_manifest={"id": "dataset-receipt", "manifest_sha256": "d" * 64},
    )
    assert bound.metadata.extra["runtime_id"] == "runtime-receipt"
    assert bound.metadata.extra["runtime_sha256"] == "b" * 64
    assert bound.metadata.extra["io_id"] == "io-receipt"
    assert bound.metadata.extra["dataset_manifest_sha256"] == "d" * 64
    assert bound.metadata.artifact_id == "artifact"
    assert bound.act({"agentview": np.zeros((256, 256, 3), dtype=np.uint8)}).shape == (10, 7)


def test_receipt_binding_rejects_mutable_or_conflicting_identity():
    metadata = PolicyMetadata(
        policy_kind="pi05",
        artifact_id="artifact",
        checkpoint_revision="a" * 40,
        backend="lerobot",
        native_action_horizon=10,
    )
    native = CallablePolicyAdapter(metadata, act_fn=lambda _: np.zeros((10, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="immutable"):
        bind_adapter_provenance(
            native,
            artifact={"id": "artifact", "revision": "main"},
            runtime={"sha256": "b" * 64},
            io={"sha256": "c" * 64},
            dataset_manifest={"manifest_sha256": "d" * 64},
        )
    with pytest.raises(ValueError, match="conflicting runtime_sha256"):
        bind_adapter_provenance(
            CallablePolicyAdapter(
                PolicyMetadata(
                    policy_kind="pi05",
                    artifact_id="artifact",
                    checkpoint_revision="a" * 40,
                    backend="lerobot",
                    native_action_horizon=10,
                    extra={"runtime_sha256": "e" * 64},
                ),
                act_fn=lambda _: np.zeros((10, 7), dtype=np.float32),
            ),
            artifact={"id": "artifact", "revision": "a" * 40},
            runtime={"sha256": "b" * 64},
            io={"sha256": "c" * 64},
            dataset_manifest={"manifest_sha256": "d" * 64},
        )


def test_manifest_binds_dataset_and_all_experiment_hashes():
    digest = "b" * 64
    manifest = build_policy_manifest(
        policy_kind="pi05",
        artifact_id="pi05-test",
        backend="lerobot",
        model_revision="rev",
        checkpoint_sha256=digest,
        dataset=_source(),
        preprocessing_sha256=digest,
        training_sha256=digest,
        evaluation_sha256=digest,
        action_horizon=10,
        camera_keys=("agentview", "robot0_eye_in_hand"),
        state_dim=8,
    )
    assert validate_manifest(manifest)["manifest_sha256"] == manifest["manifest_sha256"]
    tampered = dict(manifest)
    tampered["training_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="manifest hash"):
        validate_manifest(tampered)


def test_new_policy_kinds_build_plans_and_v1_remains_readable():
    bindings = {
        "adapter_kind": "test-native-adapter",
        "artifact": {"id": "artifact", "revision": "a" * 40, "checkpoint_sha256": "d" * 64},
        "runtime": {"id": "runtime", "sha256": "a" * 64},
        "io": {"id": "io", "sha256": "b" * 64},
        "dataset_manifest": {"id": "dataset", "manifest_sha256": "c" * 64},
    }
    for policy_kind in (
        "pi05",
        "openvla_oft",
        "octo_community_multisuite_190k",
        "octo_base15_spatial_no_arrow_matched",
    ):
        plan = build_evaluation_plan(
            policy_kind=policy_kind,
            suite_mode="sealed_randomized",
            task_ids=[0],
            episodes_per_task=1,
            bindings=bindings,
        )
        assert plan["schema"] == "shared_evaluation_plan.v2"
        assert validate_plan(plan)["policy_kind"] == policy_kind
    legacy = build_evaluation_plan(
        policy_kind="smolvla_no_arrow", suite_mode="vanilla", task_ids=[0], episodes_per_task=1,
    )
    legacy["schema"] = LEGACY_PLAN_SCHEMA
    # Recompute the historical digest after changing only the schema field.
    from vla_benchmarking.libero.evaluation.plan import canonical_sha256
    payload = dict(legacy)
    payload.pop("sha256")
    legacy["sha256"] = canonical_sha256(payload)
    assert validate_plan(legacy)["schema"] == LEGACY_PLAN_SCHEMA


def test_v2_plan_requires_complete_bindings():
    with pytest.raises(ValueError, match="require bindings"):
        build_evaluation_plan(
            policy_kind="pi05", suite_mode="vanilla", task_ids=[0], episodes_per_task=1,
        )


def test_v2_plan_requires_canonical_checkpoint_content_hash():
    bindings = {
        "adapter_kind": "pi05",
        "artifact": {"id": "artifact", "revision": "a" * 40},
        "runtime": {"sha256": "a" * 64},
        "io": {"sha256": "b" * 64},
        "dataset_manifest": {"manifest_sha256": "c" * 64},
    }
    with pytest.raises(ValueError, match="canonical checkpoint_sha256"):
        build_evaluation_plan(
            policy_kind="pi05", suite_mode="vanilla", task_ids=[0], episodes_per_task=1,
            bindings=bindings,
        )


def test_receipt_binding_builder_requires_immutable_checkpoint_identity():
    receipt = {"sha256": "a" * 64}
    bindings = build_plan_bindings(
        adapter_kind="pi05",
        artifact={"id": "artifact", "revision": "a" * 40, "checkpoint_sha256": "d" * 64},
        runtime_receipt=receipt,
        io_receipt={"io_sha256": "b" * 64},
        dataset_manifest={"manifest_sha256": "c" * 64},
    )
    assert bindings["artifact"]["revision"] == "a" * 40
    assert bindings["dataset_manifest"]["manifest_sha256"] == "c" * 64
    with pytest.raises(ValueError, match="immutable"):
        build_plan_bindings(
            adapter_kind="pi05",
            artifact={"id": "artifact", "revision": "main"},
            runtime_receipt=receipt,
            io_receipt=receipt,
            dataset_manifest=receipt,
        )


def test_runner_validates_plan_metadata_schedule_and_derives_action_counts():
    bindings = {
        "adapter_kind": "pi05",
        "artifact": {"id": "artifact", "revision": "a" * 40, "checkpoint_sha256": "d" * 64},
        "runtime": {"id": "runtime", "sha256": "a" * 64},
        "io": {"id": "io", "sha256": "b" * 64, "action_horizon": 10, "action_dim": 7},
        "dataset_manifest": {"id": "dataset", "manifest_sha256": "c" * 64},
    }
    plan = build_evaluation_plan(
        policy_kind="pi05", suite_mode="vanilla", task_ids=[0], episodes_per_task=1,
        bindings=bindings,
    )
    metadata = PolicyMetadata(
        policy_kind="pi05", artifact_id="artifact", checkpoint_revision="a" * 40,
        backend="lerobot", adapter_kind="pi05", native_action_horizon=10,
        extra={
            "runtime_id": "runtime", "runtime_sha256": "a" * 64,
            "io_id": "io", "io_sha256": "b" * 64,
            "dataset_manifest_id": "dataset", "dataset_manifest_sha256": "c" * 64,
            "checkpoint_sha256": "d" * 64,
        },
    )
    adapter = CallablePolicyAdapter(metadata, act_fn=lambda _: np.zeros((10, 7), dtype=np.float32))
    cell = build_task_seed_matrix(task_ids=[0], episodes_per_task=1)[0]
    def counted_rollout(policy, _episode):
        action = policy.act({"agentview": np.zeros((256, 256, 3), dtype=np.uint8)})
        for _ in action:
            policy.record_environment_step()
        return EpisodeOutcome(True)

    records = run_policy_eval(
        adapter, plan, [EpisodeSpec(cell, "pick up the bowl")], counted_rollout,
    )
    assert records[0].action_chunks == 1
    assert records[0].executed_actions == 10
    assert records[0].plan_sha256 == plan["sha256"]
    with pytest.raises(RuntimeError, match="without calling"):
        run_policy_eval(
            adapter, plan, [EpisodeSpec(cell, "pick up the bowl")],
            lambda _policy, _episode: EpisodeOutcome(True),
        )


def test_v2_metadata_validation_checks_receipt_ids_as_well_as_digests():
    bindings = {
        "adapter_kind": "pi05",
        "artifact": {"id": "artifact", "revision": "a" * 40, "checkpoint_sha256": "d" * 64},
        "runtime": {"id": "runtime", "sha256": "a" * 64},
        "io": {"id": "io", "sha256": "b" * 64, "action_horizon": 10, "action_dim": 7},
        "dataset_manifest": {"id": "dataset", "manifest_sha256": "c" * 64},
    }
    plan = build_evaluation_plan(
        policy_kind="pi05", suite_mode="vanilla", task_ids=[0], episodes_per_task=1,
        bindings=bindings,
    )
    metadata = PolicyMetadata(
        policy_kind="pi05", artifact_id="artifact", checkpoint_revision="a" * 40,
        backend="lerobot", adapter_kind="pi05", native_action_horizon=10,
        extra={
            "runtime_id": "wrong-runtime", "runtime_sha256": "a" * 64,
            "io_id": "io", "io_sha256": "b" * 64,
            "dataset_manifest_id": "dataset", "dataset_manifest_sha256": "c" * 64,
            "checkpoint_sha256": "d" * 64,
        },
    )
    adapter = CallablePolicyAdapter(metadata, act_fn=lambda _: np.zeros((10, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="runtime id identity"):
        run_policy_eval(
            adapter, plan, [EpisodeSpec(build_task_seed_matrix(task_ids=[0], episodes_per_task=1)[0], "pick up the bowl")],
            lambda _policy, _episode: EpisodeOutcome(False),
        )


def test_derived_receipts_reject_mismatched_checkpoint_and_runtime(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint-a")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema":"test"}\n', encoding="utf-8")
    artifact = derive_checkpoint_receipt(
        checkpoint, artifact_id="artifact", checkpoint_revision="a" * 40,
    )
    runtime = derive_runtime_receipt(require_clean=False)
    io = derive_io_receipt({
        "action_horizon": 10, "action_dim": 7, "input_resolution": 256,
        "camera_keys": ["agentview"], "state_dim": None, "visual_input": "none",
    })
    dataset = derive_dataset_manifest_receipt(manifest)
    metadata = PolicyMetadata(
        policy_kind="pi05", artifact_id="artifact", checkpoint_revision="a" * 40,
        backend="test", adapter_kind="pi05", native_action_horizon=10,
        extra={"runtime_id": runtime["id"], "runtime_sha256": runtime["sha256"],
               "io_id": io["id"], "io_sha256": io["sha256"],
               "dataset_manifest_id": dataset["id"],
               "dataset_manifest_sha256": dataset["manifest_sha256"],
               "checkpoint_sha256": artifact["checkpoint_sha256"]},
    )
    adapter = CallablePolicyAdapter(metadata, act_fn=lambda _: np.zeros((10, 7), dtype=np.float32))
    bindings = build_plan_bindings(
        adapter_kind="pi05", artifact=artifact, runtime_receipt=runtime,
        io_receipt=io, dataset_manifest=dataset,
    )
    plan = build_evaluation_plan(
        policy_kind="pi05", suite_mode="vanilla", task_ids=[0], episodes_per_task=1,
        bindings=bindings,
    )
    # Altering either local evidence after the plan was sealed cannot be
    # hidden by reusing the plan's labels.
    checkpoint.write_bytes(b"checkpoint-b")
    changed_artifact = derive_checkpoint_receipt(
        checkpoint, artifact_id="artifact", checkpoint_revision="a" * 40,
    )
    assert changed_artifact["checkpoint_sha256"] != artifact["checkpoint_sha256"]
    with pytest.raises(ValueError, match="checkpoint identity"):
        validate_plan(plan)
        changed_metadata = PolicyMetadata(
            policy_kind=metadata.policy_kind, artifact_id=metadata.artifact_id,
            checkpoint_revision=metadata.checkpoint_revision, backend=metadata.backend,
            adapter_kind=metadata.adapter_kind, native_action_horizon=metadata.native_action_horizon,
            extra={**metadata.extra, "checkpoint_sha256": changed_artifact["checkpoint_sha256"]},
        )
        validate_adapter_metadata(
            CallablePolicyAdapter(changed_metadata, act_fn=lambda _: np.zeros((10, 7), dtype=np.float32)),
            plan,
        )
    runtime_tampered = dict(metadata.extra)
    runtime_tampered["runtime_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="runtime sha256 identity"):
        validate_adapter_metadata(
            CallablePolicyAdapter(
                PolicyMetadata(
                    policy_kind=metadata.policy_kind, artifact_id=metadata.artifact_id,
                    checkpoint_revision=metadata.checkpoint_revision, backend=metadata.backend,
                    adapter_kind=metadata.adapter_kind, native_action_horizon=metadata.native_action_horizon,
                    extra=runtime_tampered,
                ),
                act_fn=lambda _: np.zeros((10, 7), dtype=np.float32),
            ),
            plan,
        )


def test_runtime_receipt_fails_closed_on_dirty_pinned_sources(monkeypatch):
    from vla_benchmarking.libero.evaluation import policy_adapter as contracts

    original_run = contracts.subprocess.run

    def dirty_status(command, *args, **kwargs):
        if "status" in command:
            return contracts.subprocess.CompletedProcess(command, 0, stdout=" M pinned.py\n", stderr="")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(contracts.subprocess, "run", dirty_status)
    with pytest.raises(RuntimeError, match="dirty or contains untracked"):
        derive_runtime_receipt()


def test_libero_rollout_executes_every_native_action_and_detects_success():
    metadata = PolicyMetadata(
        policy_kind="smolvla_no_arrow", artifact_id="artifact", checkpoint_revision="rev",
        backend="lerobot", native_action_horizon=4,
    )
    adapter = CallablePolicyAdapter(metadata, act_fn=lambda _: np.zeros((4, 7), dtype=np.float32))
    plan = build_evaluation_plan(
        policy_kind="smolvla_no_arrow", suite_mode="vanilla", task_ids=[0], episodes_per_task=1,
    )
    cell = build_task_seed_matrix(task_ids=[0], episodes_per_task=1)[0]

    class FakeEnv:
        def __init__(self):
            self.actions = []
            self.closed = False

        def reset(self, *, seed, task_id, episode_index):
            assert (seed, task_id, episode_index) == (1000, 0, 0)
            return {"agentview": np.zeros((256, 256, 3), dtype=np.uint8)}

        def step(self, action):
            self.actions.append(np.asarray(action))
            # Success is reported part-way through a chunk; rollout must stop
            # and account only the actions accepted by env.step.
            return {"agentview": np.zeros((256, 256, 3), dtype=np.uint8)}, 0.0, False, False, {
                "success": len(self.actions) == 2,
            }

        def close(self):
            self.closed = True

    fake = FakeEnv()
    records = run_libero_policy_eval(
        adapter, plan, [EpisodeSpec(cell, "pick up the bowl")],
        env_factory=CallableLiberoEnvironmentFactory(lambda _: fake),
    )
    assert records[0].success is True
    assert records[0].action_chunks == 1
    assert records[0].executed_actions == 2
    assert len(fake.actions) == 2
    assert fake.closed is True


def test_libero_rollout_fails_closed_on_schedule_drift():
    metadata = PolicyMetadata(
        policy_kind="smolvla_no_arrow", artifact_id="artifact", checkpoint_revision="rev",
        backend="lerobot", native_action_horizon=4,
    )
    adapter = CallablePolicyAdapter(metadata, act_fn=lambda _: np.zeros((4, 7), dtype=np.float32))
    plan = build_evaluation_plan(
        policy_kind="smolvla_no_arrow", suite_mode="vanilla", task_ids=[0], episodes_per_task=1,
    )
    drifted = build_task_seed_matrix(task_ids=[1], episodes_per_task=1)[0]
    with pytest.raises(ValueError, match="native planned cell schedule"):
        run_libero_policy_eval(
            adapter, plan, [EpisodeSpec(drifted, "pick up the bowl")],
            env_factory=CallableLiberoEnvironmentFactory(lambda _: None),
        )
