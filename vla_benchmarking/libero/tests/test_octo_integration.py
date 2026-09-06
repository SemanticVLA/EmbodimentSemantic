from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vla_benchmarking.libero.finetuned_vlas.octo.adapter import OctoPolicyAdapter
from vla_benchmarking.libero.finetuned_vlas.octo.config import (
    A40_BATCH_LADDER,
    A40_MEMORY_LIMIT_GB,
    COMMUNITY_CHECKPOINT,
    COMMUNITY_EVAL_CONFIG,
    MATCHED_TRAIN_CONFIG,
    checkpoint_download_patterns,
    choose_a40_batch_candidate,
)
from vla_benchmarking.libero.finetuned_vlas.octo.contracts import (
    FRAME_ORIENTATION_LIBERO_CANONICAL,
    OCTO_ACTION_HORIZON,
    convert_libero_to_octo_action,
    convert_octo_to_libero_action,
    rotate_stored_frame_180,
)
from vla_benchmarking.libero.finetuned_vlas.octo.dataset import (
    fingerprint_steps,
    serialize_episode,
    validate_episode_steps,
)
from vla_benchmarking.libero.finetuned_vlas.octo.eval import build_command as build_eval_command
from vla_benchmarking.libero.finetuned_vlas.octo.manifest import (
    build_completion_manifest,
    compute_optimizer_updates,
    write_completion_manifest,
)
from vla_benchmarking.libero.finetuned_vlas.octo.preflight import run_preflight
from vla_benchmarking.libero.finetuned_vlas.octo.train import build_command as build_train_command


def test_checkpoint_and_mode_are_pinned_separately() -> None:
    assert COMMUNITY_CHECKPOINT.repository == "cyrusneary/octo-finetuned-libero"
    assert COMMUNITY_CHECKPOINT.revision == "f8a0888cfa7ef3be072417eb012339464a9bb6dc"
    assert COMMUNITY_CHECKPOINT.checkpoint_subpath.endswith("190000/default/checkpoint")
    assert COMMUNITY_EVAL_CONFIG.mode == "community_eval"
    assert MATCHED_TRAIN_CONFIG.mode == "matched_train"
    assert COMMUNITY_EVAL_CONFIG.episodes == 432
    assert COMMUNITY_EVAL_CONFIG.timesteps == 52970
    assert COMMUNITY_EVAL_CONFIG.policy_kind == "octo_community_multisuite_190k"
    assert COMMUNITY_EVAL_CONFIG.optimizer_updates is None
    assert MATCHED_TRAIN_CONFIG.optimizer_updates is None
    assert MATCHED_TRAIN_CONFIG.episodes == 500
    assert MATCHED_TRAIN_CONFIG.timesteps == 62250
    assert MATCHED_TRAIN_CONFIG.policy_kind == "octo_base15_spatial_no_arrow_matched"


def test_checkpoint_download_allowlist_excludes_historical_steps() -> None:
    patterns = checkpoint_download_patterns(COMMUNITY_CHECKPOINT)
    assert any(pattern.endswith("190000/default/checkpoint/**") for pattern in patterns)
    assert not any(pattern.endswith("experiment_20250621_094538/**") for pattern in patterns)
    assert "dataset_statistics.json" in patterns


def test_a40_ladder_preserves_effective_batch_and_selects_first_fit() -> None:
    assert [item.effective_batch for item in A40_BATCH_LADDER] == [32, 32, 32]
    selected = choose_a40_batch_candidate({32: 44.0, 16: 42.0, 8: 40.0})
    assert selected.microbatch == 16
    with pytest.raises(RuntimeError):
        choose_a40_batch_candidate({32: 44.0, 16: 44.0, 8: 44.0})
    assert A40_MEMORY_LIMIT_GB == pytest.approx(43.2)


def test_action_conversion_round_trips_and_maps_gripper() -> None:
    mean = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    std = np.ones(6, dtype=np.float32) * 0.5
    libero = np.array([0.2, -0.1, 0.3, 0.4, -0.5, 0.6, -1.0], dtype=np.float32)
    octo = convert_libero_to_octo_action(libero, mean, std)
    assert octo[6] == pytest.approx(1.0)
    assert np.allclose(convert_octo_to_libero_action(octo, mean, std), libero)


def test_stored_frame_rotation_is_exactly_one_flip() -> None:
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert np.array_equal(rotate_stored_frame_180(frame), frame[::-1, ::-1])


def test_canonical_frame_cannot_be_rotated_again() -> None:
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="second time"):
        list(
            serialize_episode(
                [frame],
                [np.zeros(7, dtype=np.float32)],
                instruction="open the drawer",
                episode_id="canonical-0",
                source_frame_orientation=FRAME_ORIENTATION_LIBERO_CANONICAL,
            )
        )


def test_rlds_serializer_preserves_lineage_and_final_flags() -> None:
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    steps = list(
        serialize_episode(
            [frame, frame + 1],
            [np.zeros(7, dtype=np.float32), np.ones(7, dtype=np.float32)],
            instruction="put the bowl on the plate",
            episode_id="task-0-demo-3",
        )
    )
    assert validate_episode_steps(steps) == 2
    assert steps[0]["episode_id"] == "task-0-demo-3"
    assert steps[0]["frame_id"] == 0
    assert steps[0]["is_first"] is True
    assert steps[0]["is_last"] is False
    assert steps[-1]["is_last"] is True
    assert steps[-1]["is_terminal"] is True
    assert steps[0]["action_space"] == "octo_libero_dataset_v1"
    assert float(steps[0]["action"][6]) == pytest.approx(0.5)
    assert np.array_equal(steps[0]["observation"]["image_primary"], frame[::-1, ::-1])
    fingerprint = fingerprint_steps(steps)
    assert fingerprint["episode_count"] == 1
    assert fingerprint["step_count"] == 2
    assert fingerprint["image_bytes"] == 2 * frame.nbytes

    changed = list(
        serialize_episode(
            [frame + 2, frame + 1],
            [np.zeros(7, dtype=np.float32), np.ones(7, dtype=np.float32)],
            instruction="put the bowl on the plate",
            episode_id="task-0-demo-3",
        )
    )
    assert fingerprint_steps(changed)["sha256"] != fingerprint["sha256"]


class _FakeRandom:
    @staticmethod
    def PRNGKey(seed: int) -> tuple[str, int]:
        return ("key", seed)

    @staticmethod
    def split(key: tuple[str, int]) -> tuple[tuple[str, int], tuple[str, int]]:
        return ("next", key[1] + 1), ("sample", key[1])


class _FakeJax:
    random = _FakeRandom()


class _FakeModel:
    def __init__(self) -> None:
        self.instructions: list[str] = []
        self.last_rng: tuple[str, int] | None = None
        self.last_stats = None
        self.image_shapes: list[tuple[int, ...]] = []
        self.masks: list[np.ndarray] = []

    def create_tasks(self, *, texts: list[str]) -> dict[str, list[str]]:
        self.instructions.extend(texts)
        return {"language_instruction": texts}

    def sample_actions(self, batch, *, tasks, rng, unnormalization_statistics):
        assert batch["image_primary"].shape[0] == 1
        assert batch["image_primary"].shape[2:] == (256, 256, 3)
        assert batch["timestep_pad_mask"].shape == batch["timestep"].shape
        self.image_shapes.append(tuple(batch["image_primary"].shape))
        self.masks.append(batch["timestep_pad_mask"].copy())
        assert np.array_equal(
            unnormalization_statistics["mask"], np.array([True, True, True, True, True, True, False])
        )
        assert tasks["language_instruction"] == ["open the drawer"]
        self.last_rng = rng
        self.last_stats = unnormalization_statistics
        return np.zeros((1, OCTO_ACTION_HORIZON, 7), dtype=np.float32)


def test_native_adapter_splits_prng_and_requires_native_chunk_shape() -> None:
    model = _FakeModel()
    adapter = OctoPolicyAdapter(
        model,
        _FakeJax,
        action_mean=np.zeros(6, dtype=np.float32),
        action_std=np.ones(6, dtype=np.float32),
        mode="matched_train",
    )
    adapter.reset("open the drawer", 1000)
    sampled = adapter.act({"image_primary": np.zeros((256, 256, 3), dtype=np.uint8), "timestep": 0})
    assert sampled.shape == (4, 7)
    assert model.last_rng == ("sample", 1000)


def test_community_adapter_buffers_and_pads_two_frames() -> None:
    model = _FakeModel()
    adapter = OctoPolicyAdapter(
        model,
        _FakeJax,
        action_mean=np.zeros(6, dtype=np.float32),
        action_std=np.ones(6, dtype=np.float32),
        mode="community_eval",
    ).reset("open the drawer", 1000)
    observation = {"image_primary": np.zeros((256, 256, 3), dtype=np.uint8), "timestep": 0}
    adapter.act(observation)
    adapter.act({**observation, "timestep": 1})
    assert model.image_shapes == [(1, 2, 256, 256, 3), (1, 2, 256, 256, 3)]
    assert np.array_equal(model.masks[0], np.array([[False, True]]))
    assert np.array_equal(model.masks[1], np.array([[True, True]]))


def test_loaded_model_statistics_are_authoritative_and_mismatch_is_rejected() -> None:
    model = _FakeModel()
    model.dataset_statistics = {
        "action": {"mean": np.zeros(7, dtype=np.float32), "std": np.ones(7, dtype=np.float32), "mask": [True] * 6 + [False]}
    }
    adapter = OctoPolicyAdapter(
        model,
        _FakeJax,
        action_mean=np.zeros(6, dtype=np.float32),
        action_std=np.ones(6, dtype=np.float32),
        mode="matched_train",
    )
    assert adapter.metadata.extra["stats_source"] == "loaded_model_verified_against_caller"
    with pytest.raises(ValueError, match="disagree"):
        OctoPolicyAdapter(
            model,
            _FakeJax,
            action_mean=np.ones(6, dtype=np.float32),
            action_std=np.ones(6, dtype=np.float32),
            mode="matched_train",
        )


def test_native_rollout_executes_all_four_actions_and_preserves_provenance() -> None:
    model = _FakeModel()
    adapter = OctoPolicyAdapter(
        model,
        _FakeJax,
        action_mean=np.zeros(6, dtype=np.float32),
        action_std=np.ones(6, dtype=np.float32),
        mode="matched_train",
    ).reset("open the drawer", 1000)
    executed = []
    result = adapter.execute_action_chunk(
        {"image_primary": np.zeros((256, 256, 3), dtype=np.uint8), "timestep": 0},
        lambda action: executed.append(action.copy()),
        query_index=7,
    )
    assert result.actions_executed == 4
    assert result.policy_provenance["policy_kind"] == "octo_base15_spatial_no_arrow_matched"
    assert len(executed) == 4
    calls = [0]

    def fail_after_one(action):
        calls[0] += 1
        return calls[0] < 2

    with pytest.raises(RuntimeError, match="failed after 1/4"):
        adapter.execute_action_chunk(
            {"image_primary": np.zeros((256, 256, 3), dtype=np.uint8), "timestep": 0},
            fail_after_one,
            query_index=8,
        )


def test_preflight_derives_updates_from_verified_manifest(tmp_path) -> None:
    fingerprint = {
        "sha256": "a" * 64,
        "episode_count": 500,
        "step_count": 62250,
        "action_count": 62250,
        "image_bytes": 123,
        "action_bytes": 456,
    }
    manifest = build_completion_manifest(
        config=MATCHED_TRAIN_CONFIG,
        fingerprint=fingerprint,
        source_revision="source-rev",
    )
    manifest_path = tmp_path / "completion.json"
    write_completion_manifest(manifest_path, manifest)
    checkpoint = tmp_path / Path(MATCHED_TRAIN_CONFIG.checkpoint.checkpoint_subpath)
    checkpoint.mkdir(parents=True)
    evidence = run_preflight(
        mode="matched_train",
        dataset_manifest=manifest_path,
        checkpoint_path=checkpoint,
    )
    assert evidence["optimizer_updates"] == compute_optimizer_updates(
        transition_count=62250, epochs=15
    )
    command = build_train_command(
        entrypoint="octo_native_train",
        dataset_manifest=manifest_path,
        checkpoint_path=checkpoint,
        evidence=evidence,
    )
    assert command[command.index(f"--config.num_steps={evidence['optimizer_updates']}")] == (
        f"--config.num_steps={evidence['optimizer_updates']}"
    )
    assert any(arg.endswith(":full,language_conditioned") for arg in command)


def test_preflight_uses_community_counts_and_eval_provenance(tmp_path) -> None:
    checkpoint = tmp_path / Path(COMMUNITY_EVAL_CONFIG.checkpoint.checkpoint_subpath)
    checkpoint.mkdir(parents=True)
    (checkpoint / "params").write_bytes(b"pinned-checkpoint")
    stats_path = checkpoint.parent.parent.parent / "dataset_statistics.json"
    stats_path.write_text(
        '{"dataset": {"num_trajectories": 432, "num_transitions": 52970}}\n',
        encoding="utf-8",
    )
    evidence = run_preflight(
        mode="community_eval",
        dataset_manifest=None,
        checkpoint_path=checkpoint,
    )
    assert evidence["episodes"] == 432
    assert evidence["transitions"] == 52970
    assert evidence["optimizer_updates"] is None
    command = build_eval_command(
        entrypoint="octo_native_eval",
        mode="community_eval",
        dataset_manifest=None,
        checkpoint_path=checkpoint,
        evidence=evidence,
    )
    assert command[command.index("--policy-kind") + 1] == "octo_community_multisuite_190k"
