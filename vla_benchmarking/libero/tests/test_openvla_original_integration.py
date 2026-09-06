from __future__ import annotations

from pathlib import Path

import numpy as np

from vla_benchmarking.libero.evaluation.native_vla_eval import _canonical_observation
from vla_benchmarking.libero.evaluation.registry import get_policy_capabilities
from vla_benchmarking.libero.finetuned_vlas.openvla import OpenVLAAdapter


def test_original_openvla_adapter_uses_official_prompt_crop_and_action_semantics() -> None:
    calls: dict[str, object] = {}

    class Processor:
        def __call__(self, prompt, image):
            calls["prompt"] = prompt
            calls["size"] = image.size
            return {}

    class Model:
        def predict_action(self, **kwargs):
            calls["kwargs"] = kwargs
            return np.asarray([0, 0, 0, 0, 0, 0, 0.8], dtype=np.float32)

    adapter = OpenVLAAdapter(Model(), processor=Processor())
    adapter.reset("Pick Up The Bowl", 0)
    action = adapter.act({"agentview": np.zeros((256, 256, 3), dtype=np.uint8)})
    assert action.shape == (1, 7)
    assert action[0, -1] == -1.0
    assert calls["prompt"] == "In: What action should the robot take to pick up the bowl?\nOut:"
    assert calls["size"] == (224, 224)
    assert calls["kwargs"] == {"unnorm_key": "libero_spatial", "do_sample": False}


def test_original_openvla_environment_contract_has_only_agentview() -> None:
    observation = _canonical_observation(
        {
            "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.asarray([0, 0, 0, 1]),
            "robot0_gripper_qpos": np.zeros(2),
        },
        original_openvla=True,
    )
    assert sorted(observation) == ["agentview", "frame_provenance"]


def test_original_openvla_is_a_distinct_native_policy_kind() -> None:
    capabilities = get_policy_capabilities("openvla")
    assert capabilities.backend.name == "native_policy"
    assert capabilities.visual_inputs == ("none",)


def test_legion_job_isolates_original_openvla_transformers_pin() -> None:
    root = Path(__file__).resolve().parents[3]
    requirements = (root / "vla_benchmarking/libero/finetuned_vlas/openvla/requirements.txt").read_text()
    job = (root / "vla_benchmarking/libero/finetuned_vlas/legion/run_vla_eval_matrix.sbatch").read_text()
    assert "transformers==4.40.1" in requirements
    assert '"$RUN_ROOT/openvla_venv"' in job
    assert 'local python_bin="$1"' in job
    assert 'run_model "$OPENVLA_PYTHON" openvla' in job
    assert 'run_model "$PYTHON" pi05' in job
    assert 'init_states: $LIBERO_INIT_TARGET' in job
    assert 'conditions=(sealed)' in job
    assert 'PRESERVED_OPENVLA_VANILLA' in job


def test_legion_pi05_repair_skips_openvla_and_records_tokenizer_snapshot() -> None:
    root = Path(__file__).resolve().parents[3]
    job = (root / "vla_benchmarking/libero/finetuned_vlas/legion/run_vla_eval_matrix.sbatch").read_text()

    assert 'pi05|pi05_only|pi05-only' in job
    assert 'if [[ "$PI05_ONLY" != "1" ]]; then' in job
    assert '"$BASE_PYTHON" -m venv --system-site-packages "$RUN_ROOT/openvla_venv"' in job
    assert "'setuptools<81'" in job
    assert 'PI05_TOKENIZER_REPO="google/paligemma-3b-pt-224"' in job
    assert 'PI05_TOKENIZER_REVISION" =~ ^[0-9a-fA-F]{40}$' in job
    assert 'resolve_snapshot "$PI05_TOKENIZER_REPO" pi05_tokenizer' in job
    assert 'export PI05_TOKENIZER_PATH' in job
    assert '"pi05_tokenizer": {"repo_id": "$PI05_TOKENIZER_REPO"' in job
    assert 'mode == "pi05_only"' in job
    assert 'source "$OMNIS_SECRETS_ENV"' in job
    assert 'eval "$hf_token_assignment"' not in job
    assert 'set -a' not in job
    assert 'PI05_CANARY_ONLY=1 requires VLA_REPAIR=pi05_only' in job
    assert 'Pi05Adapter.load(artifact=artifact)' in job
    assert 'adapter.predict(observation' in job
    assert '"schema": "pi05_canary.v1"' in job
    assert '"action_shape": list(action.shape)' in job

    # The OpenVLA environment, snapshot, plan, and execution are all behind
    # the Pi0.5-only guard rather than merely omitted from the final summary.
    openvla_setup = job.index('"$BASE_PYTHON" -m venv --system-site-packages "$RUN_ROOT/openvla_venv"')
    openvla_plan = job.index('--model openvla --checkpoint-path')
    openvla_run = job.index('run_model "$OPENVLA_PYTHON" openvla')
    assert job.rfind('if [[ "$PI05_ONLY" != "1" ]]; then', 0, openvla_setup) != -1
    assert job.rfind('if [[ "$PI05_ONLY" != "1" ]]; then', 0, openvla_plan) != -1
    assert job.rfind('if [[ "$PI05_ONLY" != "1" ]]; then', 0, openvla_run) != -1

    tokenizer_snapshot = job.index('resolve_snapshot "$PI05_TOKENIZER_REPO" pi05_tokenizer')
    pi05_snapshot = job.index('resolve_snapshot "$PI05_REPO" pi05')
    token_clear = job.rindex('unset HF_TOKEN')
    provenance = job.index('"pi05_tokenizer": {"repo_id": "$PI05_TOKENIZER_REPO"')
    canary = job.index('if [[ "$PI05_CANARY_ONLY" == "1" ]]; then', provenance)
    assert tokenizer_snapshot < pi05_snapshot < token_clear < provenance < canary
