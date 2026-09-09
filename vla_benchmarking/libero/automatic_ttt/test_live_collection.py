from __future__ import annotations

import json
import hashlib
import sys
import types

import pytest

np = pytest.importorskip("numpy")

from . import live_collection as live_collection_module
from .contracts import SourceState
from .live_collection import (
    canonical_student_observation,
    collect_fresh_arrow_demonstrations,
    collect_task_corrections,
    export_correction_only_lerobot_dataset,
)
from .teacher import ArrowGraspControllerTeacher
from .peft_artifacts import load_arrow_collection_manifest


def _observation(step: int = 0):
    image = np.full((256, 256, 3), step, dtype=np.uint8)
    return {"image": image, "image_wrist": image.copy(), "state": np.zeros(8, dtype=np.float32) + step}


class _LiveEnv:
    def __init__(self):
        self.reset_count = 0
        self.close_count = 0
        self.step_count = 0

    def reset(self, *, seed, task_id, episode_index):
        self.reset_count += 1
        assert seed == 3000 and task_id == 0 and episode_index == 0
        return _observation(0)

    def step(self, action):
        self.step_count += 1
        success = self.step_count == 2
        return _observation(self.step_count), 0.0, success, {"success": success}

    def close(self):
        self.close_count += 1


def test_native_projection_is_exactly_four_student_fields():
    projected = canonical_student_observation(_observation(), instruction="pick up the bowl")
    assert set(projected) == {"agentview", "wrist", "state", "instruction"}
    assert projected["agentview"].shape == (256, 256, 3)
    assert projected["wrist"].shape == (256, 256, 3)
    assert projected["state"].shape == (8,)


def test_native_lerobot_export_treats_task_as_required_frame_metadata_not_a_feature(monkeypatch, tmp_path):
    captured = {"features": None, "frames": []}

    class FakeDataset:
        @classmethod
        def create(cls, *, features, root, **_kwargs):
            captured["features"] = features
            instance = cls()
            instance.root = root
            (root / "meta").mkdir(parents=True)
            return instance

        def add_frame(self, frame):
            assert "task" in frame
            assert "task" not in captured["features"]
            captured["frames"].append(frame)

        def save_episode(self):
            pass

        def finalize(self):
            (self.root / "meta" / "info.json").write_text("{}\n", encoding="utf-8")

    lerobot = types.ModuleType("lerobot")
    datasets = types.ModuleType("lerobot.datasets")
    module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = FakeDataset
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", module)

    accepted = tmp_path / "accepted.jsonl"
    observation = {
        "agentview": np.zeros((256, 256, 3), dtype=np.uint8).tolist(),
        "wrist": np.zeros((256, 256, 3), dtype=np.uint8).tolist(),
        "state": [0.0] * 8,
        "instruction": "pick up the bowl",
    }
    accepted.write_text(json.dumps({
        "episode_id": "episode-0",
        "transitions": [{
            "actor": "arrow_grasp_controller", "observation": observation,
            "action": [0.0] * 7,
        }],
    }) + "\n", encoding="utf-8")

    export_correction_only_lerobot_dataset(accepted, tmp_path / "dataset")

    assert captured["frames"][0]["task"] == "pick up the bowl"


def test_collection_uses_one_live_episode_for_vla_prefix_and_arrow_suffix(tmp_path):
    raw = _LiveEnv()

    def teacher_factory(_episode, _output):
        def recover(view, _request):
            with pytest.raises(Exception):
                view.reset()
            view.step((0.0,) * 7)
            return {
                "transitions": list(view.executed_transitions),
                "success": True,
                "metadata": {"evaluator_success": True, "evaluator_receipt_id": "receipt-0"},
            }

        return ArrowGraspControllerTeacher(recover)

    def fake_dataset_exporter(_accepted_path, dataset_root):
        (dataset_root / "meta").mkdir(parents=True)
        info = dataset_root / "meta" / "info.json"
        info.write_text('{"total_frames": 1}\n', encoding="utf-8")
        dataset_manifest = dataset_root.parent / "dataset_manifest.json"
        dataset_manifest.write_text(json.dumps({
            "schema_version": 1,
            "dataset_root": str(dataset_root.resolve()),
            "source_accepted_episodes": str(_accepted_path.resolve()),
            "source_accepted_episodes_sha256": hashlib.sha256(_accepted_path.read_bytes()).hexdigest(),
            "files": [{
                "path": "meta/info.json",
                "sha256": hashlib.sha256(info.read_bytes()).hexdigest(),
                "bytes": info.stat().st_size,
            }],
        }) + "\n", encoding="utf-8")
        return {
            "dataset_root": str(dataset_root),
            "dataset_manifest_path": str(dataset_manifest),
            "dataset_manifest_sha256": hashlib.sha256(dataset_manifest.read_bytes()).hexdigest(),
        }

    result = collect_task_corrections(
        task_id=0,
        task_description="pick up the bowl",
        policy_id="smolvla",
        environment_factory=lambda _episode: raw,
        reset_environment=lambda env, episode: env.reset(
            seed=episode.seed, task_id=episode.task_id, episode_index=0
        ),
        close_environment=lambda env: env.close(),
        vla_action=lambda _observation, _step: (0.0,) * 7,
        teacher_factory=teacher_factory,
        output_root=tmp_path,
        accepted_target=1,
        vla_step_budget=1,
        source_state_fn=lambda _env, _obs: SourceState.SOURCE_UNHELD,
        controller_config_hash="a" * 64,
        dataset_exporter=fake_dataset_exporter,
    )

    assert result.accepted_count == 1
    assert raw.reset_count == 1
    assert raw.close_count == 1
    assert raw.step_count == 2
    accepted = json.loads(result.accepted_path.read_text(encoding="utf-8").splitlines()[0])
    assert [row["actor"] for row in accepted["transitions"]] == ["vla", "arrow_grasp_controller"]
    assert accepted["task_id"] == 0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_kind"] == "arrow_grasp_controller_trajectory"
    assert manifest["task_ids"] == [0]
    assert manifest["accepted_count"] == 1
    assert manifest["accepted_seeds"] == [3000]
    normalized = load_arrow_collection_manifest(result.manifest_path, task_id=0, expected_successes=1)
    assert normalized["evaluator_confirmed_successes"] == 1
    assert normalized["dataset_root"].endswith("lerobot_dataset")


def test_fresh_collection_never_calls_vla_and_admits_arrow_only_success(tmp_path, monkeypatch):
    class FreshEnv(_LiveEnv):
        def reset(self, *, seed, task_id, episode_index):
            self.reset_count += 1
            return _observation(0)

        def step(self, action):
            self.step_count += 1
            return _observation(self.step_count), 0.0, True, {"success": True}

    environments = []
    vla_called = []

    def make(_episode):
        env = FreshEnv(); environments.append(env); return env

    def export(_accepted, dataset_root):
        (dataset_root / "meta").mkdir(parents=True)
        (dataset_root / "meta" / "info.json").write_text('{"total_frames": 1}\n', encoding="utf-8")
        manifest = dataset_root.parent / "dataset_manifest.json"
        manifest.write_text(json.dumps({"dataset_root": str(dataset_root), "source_accepted_episodes": str(_accepted),
            "source_accepted_episodes_sha256": hashlib.sha256(_accepted.read_bytes()).hexdigest(),
            "files": [{"path": "meta/info.json", "sha256": hashlib.sha256((dataset_root / "meta" / "info.json").read_bytes()).hexdigest()}]}) + "\n", encoding="utf-8")
        return {"dataset_root": str(dataset_root), "dataset_manifest_path": str(manifest),
                "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}

    def teacher_factory(_episode, _output):
        def recover(view, _request):
            view.step((0.0,) * 7)
            return {"transitions": list(view.executed_transitions), "success": True,
                    "metadata": {"evaluator_success": True}}
        return ArrowGraspControllerTeacher(recover)

    original_write = live_collection_module._write_jsonl_immutable

    def assert_cached_before_final_write(path, rows):
        cached = tmp_path / ".accepted_episode_cache" / "seed-3000.json"
        assert cached.is_file()
        assert json.loads(cached.read_text(encoding="utf-8"))["seed"] == 3000
        return original_write(path, rows)

    monkeypatch.setattr(live_collection_module, "_write_jsonl_immutable", assert_cached_before_final_write)

    result = collect_fresh_arrow_demonstrations(
        task_id=0, task_description="pick up the bowl", policy_id="smolvla", environment_factory=make,
        reset_environment=lambda env, ep: env.reset(seed=ep.seed, task_id=ep.task_id, episode_index=0),
        close_environment=lambda env: env.close(), teacher_factory=teacher_factory, output_root=tmp_path,
        accepted_target=1, adaptation_seed_start=3000, max_attempts=1, teacher_step_budget=2,
        source_state_fn=lambda _env, _obs: SourceState.SOURCE_UNHELD, controller_config_hash="a" * 64,
        dataset_exporter=export,
        reset_identity_fn=lambda _env, task_id: {
            "task_id": task_id,
            "selected_init_state_index": 10,
            "init_state_sha256": "b" * 64,
        },
    )
    assert result.accepted_count == 1
    assert all(env.reset_count == 1 and env.step_count == 1 for env in environments)
    accepted = json.loads(result.accepted_path.read_text(encoding="utf-8"))
    assert accepted["collection_mode"] == "fresh_arrow"
    assert [row["actor"] for row in accepted["transitions"]] == ["arrow_grasp_controller"]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_kind"] == "arrow_grasp_controller_fresh_demonstration"
    assert manifest["vla_called"] is False
    cached = json.loads((tmp_path / ".accepted_episode_cache" / "seed-3000.json").read_text(encoding="utf-8"))
    assert cached == accepted


def test_fresh_collection_resets_each_attempt_and_discards_failed_trace(tmp_path):
    environments = []

    class FreshEnv(_LiveEnv):
        def reset(self, *, seed, task_id, episode_index):
            self.seed = seed
            self.reset_count += 1
            return _observation(0)

        def step(self, action):
            self.step_count += 1
            return _observation(self.step_count), 0.0, False, {"success": False}

    def make(_episode):
        env = FreshEnv()
        environments.append(env)
        return env

    def teacher_factory(episode, _output):
        def recover(view, _request):
            view.step((0.0,) * 7)
            success = episode.seed == 3001
            return {
                "transitions": list(view.executed_transitions),
                "success": success,
                "status": "teacher_success" if success else "teacher_failed",
                    "metadata": {"evaluator_success": success, "evaluator_phase": "post_retreat"},
            }

        return ArrowGraspControllerTeacher(recover)

    def export(accepted, dataset_root):
        (dataset_root / "meta").mkdir(parents=True)
        info = dataset_root / "meta" / "info.json"
        info.write_text('{"total_frames": 1}\n', encoding="utf-8")
        manifest = dataset_root.parent / "dataset_manifest.json"
        manifest.write_text(json.dumps({
            "dataset_root": str(dataset_root),
            "source_accepted_episodes": str(accepted),
            "source_accepted_episodes_sha256": hashlib.sha256(accepted.read_bytes()).hexdigest(),
            "files": [{"path": "meta/info.json", "sha256": hashlib.sha256(info.read_bytes()).hexdigest()}],
        }) + "\n", encoding="utf-8")
        return {
            "dataset_root": str(dataset_root),
            "dataset_manifest_path": str(manifest),
            "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }

    result = collect_fresh_arrow_demonstrations(
        task_id=0, task_description="pick up the bowl", policy_id="smolvla",
        environment_factory=make,
        reset_environment=lambda env, ep: env.reset(seed=ep.seed, task_id=ep.task_id, episode_index=0),
        close_environment=lambda env: env.close(), teacher_factory=teacher_factory,
        output_root=tmp_path, accepted_target=1, adaptation_seed_start=3000,
        max_attempts=2, teacher_step_budget=2,
        source_state_fn=lambda _env, _obs: SourceState.SOURCE_UNHELD,
        controller_config_hash="a" * 64, dataset_exporter=export,
        reset_identity_fn=lambda env, task_id: {
            "task_id": task_id,
            "selected_init_state_index": 10 + (env.seed - 3000),
            "init_state_sha256": f"{env.seed:064x}",
        },
    )
    assert result.attempted_count == 2
    assert result.failed_path is None
    assert len(environments) == 2
    assert all(env.reset_count == 1 and env.close_count == 1 for env in environments)
    accepted = json.loads(result.accepted_path.read_text(encoding="utf-8"))
    assert accepted["seed"] == 3001
    assert all(row["actor"] == "arrow_grasp_controller" for row in accepted["transitions"])
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["discarded_failure_count"] == 1
    assert manifest["discarded_failure_categories"] == {"teacher_failed": 1}
    assert not (tmp_path / "failed_episodes.jsonl").exists()
    assert not (tmp_path / ".accepted_episode_cache" / "seed-3000.json").exists()
    assert (tmp_path / ".accepted_episode_cache" / "seed-3001.json").is_file()


def test_fresh_collection_discards_controller_timeout_and_tries_next_seed(tmp_path):
    environments = []

    class FreshEnv(_LiveEnv):
        def reset(self, *, seed, task_id, episode_index):
            self.seed = seed
            self.reset_count += 1
            return _observation(0)

        def step(self, action):
            self.step_count += 1
            success = self.seed == 3001
            return _observation(self.step_count), 0.0, success, {"success": success}

    def make(_episode):
        env = FreshEnv()
        environments.append(env)
        return env

    def teacher_factory(episode, _output):
        def recover(view, _request):
            view.step((0.0,) * 7)
            if episode.seed == 3000:
                raise TimeoutError("phase descend_place exceeded 160 steps")
            return {
                "transitions": list(view.executed_transitions),
                "success": True,
                "status": "teacher_success",
                "metadata": {"evaluator_success": True, "evaluator_phase": "post_retreat"},
            }

        return ArrowGraspControllerTeacher(recover)

    def export(accepted, dataset_root):
        (dataset_root / "meta").mkdir(parents=True)
        info = dataset_root / "meta" / "info.json"
        info.write_text('{"total_frames": 1}\n', encoding="utf-8")
        manifest = dataset_root.parent / "dataset_manifest.json"
        manifest.write_text(json.dumps({
            "dataset_root": str(dataset_root),
            "source_accepted_episodes": str(accepted),
            "source_accepted_episodes_sha256": hashlib.sha256(accepted.read_bytes()).hexdigest(),
            "files": [{"path": "meta/info.json", "sha256": hashlib.sha256(info.read_bytes()).hexdigest()}],
        }) + "\n", encoding="utf-8")
        return {
            "dataset_root": str(dataset_root),
            "dataset_manifest_path": str(manifest),
            "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }

    result = collect_fresh_arrow_demonstrations(
        task_id=0, task_description="pick up the bowl", policy_id="smolvla",
        environment_factory=make,
        reset_environment=lambda env, ep: env.reset(seed=ep.seed, task_id=ep.task_id, episode_index=0),
        close_environment=lambda env: env.close(), teacher_factory=teacher_factory,
        output_root=tmp_path, accepted_target=1, adaptation_seed_start=3000,
        max_attempts=2, teacher_step_budget=2,
        source_state_fn=lambda _env, _obs: SourceState.SOURCE_UNHELD,
        controller_config_hash="a" * 64, dataset_exporter=export,
        reset_identity_fn=lambda env, task_id: {
            "task_id": task_id,
            "selected_init_state_index": 10 + (env.seed - 3000),
            "init_state_sha256": f"{env.seed:064x}",
        },
    )

    assert result.accepted_count == 1
    assert result.attempted_count == 2
    assert len(environments) == 2
    assert all(env.reset_count == 1 and env.close_count == 1 for env in environments)
    accepted = json.loads(result.accepted_path.read_text(encoding="utf-8"))
    assert accepted["seed"] == 3001
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["discarded_failure_count"] == 1
    assert manifest["discarded_failure_categories"] == {"controller_motion_timeout": 1}
    assert not (tmp_path / "failed_episodes.jsonl").exists()
    assert not (tmp_path / ".accepted_episode_cache" / "seed-3000.json").exists()
