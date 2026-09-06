from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from vla_benchmarking.libero.finetuned_vlas.pi05 import (
    PI05_IO,
    PI05_TRAINING,
    Pi05Adapter,
    validate_action_chunk as validate_pi05_action,
)
from vla_benchmarking.libero.finetuned_vlas.pi05.dataset import (
    build_manifest as build_pi05_manifest,
    canonical_frame,
    serialize_frame as serialize_pi05_frame,
)
from vla_benchmarking.libero.finetuned_vlas.openvla_oft import (
    OPENVLA_IO,
    OPENVLA_TRAINING,
    OpenVLAOFTAdapter,
    OpenVLAOFTNativeRuntime,
    validate_action_chunk as validate_openvla_action,
)
from vla_benchmarking.libero.finetuned_vlas.openvla_oft.dataset import (
    NOOP_FILTER_NAME,
    NOOP_FILTER_THRESHOLD,
    filter_noop_transitions,
    build_manifest as build_openvla_manifest,
    serialize_transition,
)


def _verified_openvla_receipt(frame_count: int) -> dict[str, object]:
    return {
        "dataset_name": "libero_spatial_no_noops",
        "filter": NOOP_FILTER_NAME,
        "threshold": NOOP_FILTER_THRESHOLD,
        "source_frames": frame_count,
        "output_frames": frame_count,
        "dropped_frames": 0,
        "source_sha256": "a" * 64,
        "output_sha256": "b" * 64,
    }
from vla_benchmarking.libero.finetuned_vlas.openvla_oft.rlds_builder import (
    build_tfds_dataset,
    write_rlds_source_jsonl,
)
from vla_benchmarking.libero.finetuned_vlas.pi05.eval import build_command as build_pi05_eval_command
from vla_benchmarking.libero.finetuned_vlas.pi05.train import two_update_smoke as pi_two_update_smoke
from vla_benchmarking.libero.finetuned_vlas.openvla_oft.train import two_update_smoke as oft_two_update_smoke
from vla_benchmarking.libero.evaluation.native_vla_eval import _canonical_observation
from vla_benchmarking.libero.finetuned_vlas.openvla_oft.eval import build_command as build_openvla_eval_command


def _frame() -> dict:
    return {
        "observation.images.image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.images.image2": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.state": np.zeros(8, dtype=np.float32),
        "action": np.zeros(7, dtype=np.float32),
        "task": "pick up the bowl and place it on the plate",
    }


def test_pi05_contract_is_sealed_and_import_safe() -> None:
    assert PI05_IO.action_horizon == 50
    assert PI05_IO.state_dim == 8
    assert PI05_TRAINING.effective_batch_size == 32
    assert PI05_TRAINING.optimizer_updates == 29180
    assert serialize_pi05_frame(_frame())["arrow_condition"] == "none"


def test_openvla_contract_and_rlds_serializer() -> None:
    assert OPENVLA_IO.action_horizon == 8
    assert OPENVLA_TRAINING.lora_rank == 32
    record = serialize_transition(_frame(), episode_id="task0-demo0", step_index=0, episode_length=2)
    assert record["is_first"] is True
    assert record["is_last"] is False
    assert record["observation"]["proprio"] == [0.0] * 8
    assert record["no_arrow_condition"] is True


def test_adapters_accept_injected_policies_without_optional_dependencies() -> None:
    class FakePi:
        def predict_action_chunk(self, payload):
            return np.zeros((50, 7), dtype=np.float32)

    class FakeOpenVLA:
        def predict_action(self, payload):
            return np.zeros((8, 7), dtype=np.float32)

    pi = Pi05Adapter(FakePi())
    oft = OpenVLAOFTAdapter(FakeOpenVLA())
    assert pi.predict(_frame()).shape == (50, 7)
    assert oft.predict(_frame()).shape == (8, 7)
    pi.reset("pick up the bowl", 1000)
    oft.reset("pick up the bowl", 1000)
    assert pi.act(_frame()).shape == (50, 7)
    assert oft.act(_frame()).shape == (8, 7)
    assert pi.metadata.native_action_horizon == 50
    assert oft.metadata.native_action_horizon == 8


def test_action_validators_reject_wrong_native_horizon() -> None:
    with pytest.raises(ValueError, match="shape"):
        validate_pi05_action(np.zeros((8, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        validate_openvla_action(np.zeros((10, 7), dtype=np.float32))


def test_pi05_does_not_repeat_single_step_select_action() -> None:
    class SingleStep:
        def select_action(self, payload):
            return np.zeros(7, dtype=np.float32)

    with pytest.raises(RuntimeError, match="predict_action_chunk"):
        Pi05Adapter(SingleStep()).predict(_frame())


def test_pi05_loader_passes_current_factory_contract() -> None:
    seen = {}

    def factory(cfg, ds_meta, env_cfg):
        seen.update(cfg=cfg, ds_meta=ds_meta, env_cfg=env_cfg)
        return object()

    adapter = Pi05Adapter.load(
        factory=factory,
        policy_config="cfg",
        dataset_meta="meta",
        env_config="env",
    )
    assert adapter.policy is not None
    assert seen == {"cfg": "cfg", "ds_meta": "meta", "env_cfg": "env"}


def test_pi05_loader_passes_local_tokenizer_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "paligemma"
    tokenizer_path.mkdir()
    seen: dict[str, object] = {}

    class FakePolicy:
        config = object()

        @classmethod
        def from_pretrained(cls, checkpoint, revision=None):
            seen["checkpoint"] = checkpoint
            seen["revision"] = revision
            return cls()

    def make_pre_post_processors(**kwargs):
        seen.update(kwargs)
        return object(), object()

    lerobot = types.ModuleType("lerobot")
    policies = types.ModuleType("lerobot.policies")
    pi05 = types.ModuleType("lerobot.policies.pi05")
    policies.make_pre_post_processors = make_pre_post_processors
    pi05.PI05Policy = FakePolicy
    lerobot.policies = policies
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.policies", policies)
    monkeypatch.setitem(sys.modules, "lerobot.policies.pi05", pi05)
    monkeypatch.setenv("PI05_TOKENIZER_PATH", str(tokenizer_path))

    adapter = Pi05Adapter.load()

    assert adapter.policy is not None
    assert seen["preprocessor_overrides"] == {
        "tokenizer_processor": {"tokenizer_name": str(tokenizer_path.resolve())}
    }


def test_pi05_loader_rejects_missing_local_tokenizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PI05_TOKENIZER_PATH", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="PI05_TOKENIZER_PATH"):
        Pi05Adapter.load()


def test_openvla_native_hooks_and_gripper_inversion() -> None:
    seen = {}

    def get_vla_action(**kwargs):
        seen.update(kwargs)
        values = np.zeros((8, 7), dtype=np.float32)
        values[:, 6] = 1.0
        return values

    adapter = OpenVLAOFTAdapter(
        object(),
        native_runtime=OpenVLAOFTNativeRuntime(
            config="config",
            processor="processor",
            action_head="action_head",
            proprio_projector="proprio_projector",
            get_vla_action=get_vla_action,
        ),
    )
    result = adapter.predict(_frame(), task="task")
    assert result.shape == (8, 7)
    assert np.all(result[:, 6] == -1.0)
    assert seen["cfg"] == "config"
    assert seen["vla"] is not None
    assert seen["processor"] == "processor"
    assert seen["action_head"] == "action_head"
    assert seen["proprio_projector"] == "proprio_projector"


def test_openvla_noop_filter_matches_upstream_rule_and_reindexes() -> None:
    zero = {**_frame(), "episode_id": "e0", "step_index": 0, "episode_length": 3}
    moving = {**zero, "step_index": 1, "action": np.asarray([0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32)}
    repeated_zero = {**zero, "step_index": 2, "action": np.zeros(7, dtype=np.float32)}
    retained = filter_noop_transitions([zero, moving, repeated_zero])
    assert len(retained) == 1
    assert retained[0]["step_index"] == 0
    assert retained[0]["episode_length"] == 1
    assert retained[0]["native_noop_filter_applied"] is True


def test_manifests_hash_image_bytes_and_derive_counts() -> None:
    frame0 = {**_frame(), "episode_id": "e0", "step_index": 0}
    frame1 = {**_frame(), "episode_id": "e0", "step_index": 1}
    pi_manifest = build_pi05_manifest([frame0, frame1])
    assert pi_manifest["episodes"] == 1
    assert pi_manifest["frames"] == 2
    changed = {**frame0, "observation.images.image": np.ones((256, 256, 3), dtype=np.uint8)}
    assert build_pi05_manifest([changed, frame1])["content_sha256"] != pi_manifest["content_sha256"]
    # OpenVLA's production manifest is fail-closed: frames must carry the
    # verified no-op-filter marker.  Use non-noop actions here so the fixture
    # represents a source that passed the pinned filter.
    moving_action = np.asarray([0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    open_frame0 = {**frame0, "episode_length": 2, "action": moving_action, "native_noop_filter_applied": True}
    open_frame1 = {**frame1, "episode_length": 2, "action": moving_action, "native_noop_filter_applied": True}
    open_manifest = build_openvla_manifest(
        [open_frame0, open_frame1], noop_filter_receipt=_verified_openvla_receipt(2)
    )
    assert open_manifest["episodes"] == 1
    assert open_manifest["timesteps"] == 2
    with pytest.raises(ValueError, match="arrow"):
        serialize_pi05_frame({**frame0, "has_arrows": True})


def test_pi05_writer_canonicalizes_and_drops_source_metadata() -> None:
    frame = {
        **_frame(),
        "episode_id": "e0",
        "step_index": 0,
        "source_only": "must-not-reach-lerobot",
    }
    canonical = canonical_frame(frame)
    assert set(canonical) == {
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
        "action",
        "task",
    }


def test_dry_run_outputs_are_json_serializable() -> None:
    json.dumps(Pi05Adapter().dry_run())
    json.dumps(OpenVLAOFTAdapter().dry_run())


def test_rlds_source_writer_retains_image_bytes_and_tfds_fails_closed(tmp_path: Path) -> None:
    frames = [
        {
            **_frame(),
            "episode_id": "e0",
            "step_index": 0,
            "episode_length": 1,
            "action": np.asarray([0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        },
    ]
    source = write_rlds_source_jsonl(frames, tmp_path / "source.jsonl")
    line = source.read_text(encoding="utf-8").strip()
    assert "data_b64" in line
    assert "fields_sha256" in line
    try:
        import tensorflow_datasets  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="tensorflow-datasets"):
            build_tfds_dataset(source, tmp_path / "tfds")


def test_two_update_smoke_seams_require_finite_losses() -> None:
    assert pi_two_update_smoke(lambda index: {"loss": 1.0 - index * 0.1})["updates"] == 2
    assert oft_two_update_smoke(lambda index: {"loss": 1.0 - index * 0.1})["updates"] == 2
    with pytest.raises(ValueError, match="non-finite"):
        pi_two_update_smoke(lambda index: {"loss": float("nan")})


def test_guarded_entrypoint_help_and_print_command(tmp_path: Path) -> None:
    modules = (
        "vla_benchmarking.libero.finetuned_vlas.pi05.preflight",
        "vla_benchmarking.libero.finetuned_vlas.pi05.train",
        "vla_benchmarking.libero.finetuned_vlas.pi05.eval",
        "vla_benchmarking.libero.finetuned_vlas.openvla_oft.preflight",
        "vla_benchmarking.libero.finetuned_vlas.openvla_oft.train",
        "vla_benchmarking.libero.finetuned_vlas.openvla_oft.eval",
    )
    for module in modules:
        help_result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True)
        assert help_result.returncode == 0, help_result.stderr
    frame0 = {**_frame(), "episode_id": "e0", "step_index": 0}
    frame1 = {**_frame(), "episode_id": "e0", "step_index": 1}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(build_pi05_manifest([frame0, frame1])), encoding="utf-8")
    oft_manifest_path = tmp_path / "oft_manifest.json"
    moving_action = np.asarray([0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    oft_manifest_path.write_text(
        json.dumps(
            build_openvla_manifest(
                [
                    {**frame0, "episode_length": 2, "action": moving_action, "native_noop_filter_applied": True},
                    {**frame1, "episode_length": 2, "action": moving_action, "native_noop_filter_applied": True},
                ],
                noop_filter_receipt=_verified_openvla_receipt(2),
            )
        ),
        encoding="utf-8",
    )
    pi_command = subprocess.run(
        [
            sys.executable,
            "-m",
            "vla_benchmarking.libero.finetuned_vlas.pi05.train",
            "--print-command",
            "--dataset-root",
            "data/pi",
            "--output-root",
            "runs/pi",
            "--manifest",
            str(manifest_path),
            "--checkpoint-revision",
            "a" * 40,
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "vla_benchmarking.libero.finetuned_vlas.pi05.trainer" in pi_command
    assert "--steps=32" in pi_command
    assert "--gradient_accumulation_steps=32" in pi_command
    assert "--policy.path=lerobot/pi05_libero_finetuned_v044" in pi_command
    oft_command = subprocess.run(
        [
            sys.executable,
            "-m",
            "vla_benchmarking.libero.finetuned_vlas.openvla_oft.train",
            "--print-command",
            "--dataset-root",
            "data/oft",
            "--output-root",
            "runs/oft",
            "--manifest",
            str(oft_manifest_path),
            "--checkpoint-revision",
            "a" * 40,
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "vla-scripts/finetune.py" in oft_command
    assert "--lora_rank 32" in oft_command
    # The OpenVLA trainer's max_steps is optimizer updates, not micro-steps.
    assert "--max_steps 1" in oft_command
    assert "--adapter_tmp_dir" not in oft_command


def test_execute_guard_runs_no_subprocess_when_preflight_fails(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vla_benchmarking.libero.finetuned_vlas.pi05.train",
            "--execute",
            "--manifest",
            str(tmp_path / "missing.json"),
            "--checkpoint-revision",
            "a" * 40,
            "--dataset-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "out"),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "manifest does not exist" in (result.stderr + result.stdout)


def test_preflight_receipt_validates_revision_and_counts(tmp_path: Path) -> None:
    frame0 = {**_frame(), "episode_id": "e0", "step_index": 0}
    frame1 = {**_frame(), "episode_id": "e0", "step_index": 1}
    manifest_path = tmp_path / "pi_manifest.json"
    manifest_path.write_text(
        json.dumps(build_pi05_manifest([frame0, frame1])), encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vla_benchmarking.libero.finetuned_vlas.pi05.preflight",
            "--manifest",
            str(manifest_path),
            "--checkpoint-revision",
            "b" * 40,
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "preflight_ok"
    assert receipt["dataset"] == {"episodes": 1, "timesteps": 2}


def test_eval_command_preserves_requested_device_and_local_checkpoint() -> None:
    command = build_pi05_eval_command(
        python_executable="python",
        checkpoint="runs/pi/checkpoint-123",
        output_root="runs/eval",
        device="cpu",
    )
    assert "--policy.path=runs/pi/checkpoint-123" in command
    assert "--policy.device=cpu" in command


def test_shared_plan_switches_model_evaluators_to_native_bridge(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    native_pi = build_pi05_eval_command(
        python_executable="python", checkpoint="runs/pi/checkpoint-123",
        output_root="runs/eval", device="cpu", plan=plan,
    )
    native_oft = build_openvla_eval_command(
        python_executable="python", checkpoint="runs/oft/checkpoint-123",
        output_root="runs/eval", plan=plan,
    )
    assert native_pi[1:3] == ["-m", "vla_benchmarking.libero.evaluation.native_vla_eval"]
    assert native_oft[1:3] == ["-m", "vla_benchmarking.libero.evaluation.native_vla_eval"]
    assert "--model" in native_pi and "pi05" in native_pi
    assert "--model" in native_oft and "openvla_oft" in native_oft


def test_shared_vla_observation_normalizer_uses_axis_angle_state() -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    observation = _canonical_observation(
        {
            "agentview_image": image,
            "robot0_eye_in_hand_image": image,
            "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.asarray([0.4, 0.5], dtype=np.float32),
        }
    )
    assert observation["observation.images.image"].shape == (256, 256, 3)
    assert observation["observation.images.image2"].shape == (256, 256, 3)
    assert np.allclose(observation["observation.state"], [0.1, 0.2, 0.3, 0, 0, 0, 0.4, 0.5])
