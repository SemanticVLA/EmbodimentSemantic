from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vla_benchmarking.arrow_finetuned_vla.workflows import run_no_arrow_sealed_eval as eval_runner


def _artifacts(tmp_path: Path) -> tuple[str, str]:
    adapter = tmp_path / "checkpoints" / "029190" / "pretrained_model"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_bytes(b"{}")
    (adapter / "train_config.json").write_bytes(b"{}")
    pair_manifest = tmp_path / "sealed_lora_pair_manifest.json"
    pair_manifest.write_text(
        json.dumps(
            {
                "pair_kind": "sealed_lora_control_treatment",
                "full_experiment_ready": True,
                "launch_eligibility": "full_experiment_ready",
            }
        ) + "\n",
        encoding="utf-8",
    )
    pair_manifest_sha256 = hashlib.sha256(pair_manifest.read_bytes()).hexdigest()
    pair_sentinel = tmp_path / "sealed_lora_pair_verified.json"
    pair_sentinel.write_text(
        json.dumps(
            {
                "pair_kind": "sealed_lora_control_treatment",
                "full_experiment_ready": True,
                "launch_eligibility": "full_experiment_ready",
                "manifest_sha256": pair_manifest_sha256,
            }
        ) + "\n",
        encoding="utf-8",
    )
    training_manifest = tmp_path / "training_manifest.json"
    training_manifest.write_text(
        json.dumps(
            {
                "pair_manifest": str(pair_manifest),
                "pair_manifest_sha256": pair_manifest_sha256,
                "pair_sentinel": str(pair_sentinel),
                "pair_sentinel_sha256": hashlib.sha256(pair_sentinel.read_bytes()).hexdigest(),
            }
        ) + "\n",
        encoding="utf-8",
    )
    return str(adapter), str(training_manifest)


def test_full_manifest_is_one_no_arrow_cell_with_500_planned_episodes(tmp_path: Path):
    adapter, training_manifest = _artifacts(tmp_path)
    manifest = eval_runner.build_manifest(
        adapter_checkpoint=adapter,
        training_manifest=training_manifest,
        output_root=tmp_path / "outputs",
        protocol="full",
        episodes=50,
    )

    assert manifest["protocol"] == "full"
    assert manifest["episodes"] == 50
    assert manifest["planned_episodes"] == 500
    assert manifest["seed"] == 1000
    assert manifest["seed_base"] == 1000
    assert manifest["episode_seeds"] == list(range(1000, 1050))
    assert manifest["episode_seed_policy"] == "seed=seed_base+episode_index"
    assert manifest["evaluation_visual_condition"] == "none"
    assert manifest["adapter_config_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert manifest["train_config_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert manifest["randomize_scenes"] is True
    assert manifest["batch_size"] == 1
    assert len(manifest["cells"]) == 1
    assert manifest["cells"][0]["cell_id"] == eval_runner.CELL_ID
    assert manifest["cells"][0]["live_arrows"] is False


def test_protocol_and_episode_count_must_match(tmp_path: Path):
    adapter, training_manifest = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="requires exactly 50"):
        eval_runner.build_manifest(
            adapter_checkpoint=adapter,
            training_manifest=training_manifest,
            output_root=tmp_path / "outputs",
            protocol="full",
            episodes=1,
        )
    with pytest.raises(ValueError, match="base seed exactly 1000"):
        eval_runner.build_manifest(
            adapter_checkpoint=adapter,
            training_manifest=training_manifest,
            output_root=tmp_path / "outputs2",
            protocol="smoke",
            episodes=1,
            seed=1001,
        )


def test_current_pair_sentinel_may_be_semantically_revalidated_after_file_drift(tmp_path: Path):
    adapter, training_manifest = _artifacts(tmp_path)
    training = json.loads(Path(training_manifest).read_text(encoding="utf-8"))
    sentinel = Path(training["pair_sentinel"])
    payload = json.loads(sentinel.read_text(encoding="utf-8"))
    payload["revalidated_at"] = "later"
    sentinel.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    manifest = eval_runner.build_manifest(
        adapter_checkpoint=adapter,
        training_manifest=training_manifest,
        output_root=tmp_path / "outputs",
        protocol="smoke",
        episodes=1,
    )

    provenance = manifest["provenance"]
    assert provenance["pair_sentinel_status"] == "semantic_revalidation_after_file_drift"
    assert provenance["pair_sentinel_training_sha256"] != provenance["pair_sentinel_observed_sha256"]


def test_pair_sentinel_revalidation_rejects_a_different_pair_manifest(tmp_path: Path):
    adapter, training_manifest = _artifacts(tmp_path)
    training = json.loads(Path(training_manifest).read_text(encoding="utf-8"))
    sentinel = Path(training["pair_sentinel"])
    payload = json.loads(sentinel.read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "0" * 64
    sentinel.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot revalidate"):
        eval_runner.build_manifest(
            adapter_checkpoint=adapter,
            training_manifest=training_manifest,
            output_root=tmp_path / "outputs",
            protocol="smoke",
            episodes=1,
        )


def test_immutable_manifest_rejects_changed_condition(tmp_path: Path):
    adapter, training_manifest = _artifacts(tmp_path)
    manifest = eval_runner.build_manifest(
        adapter_checkpoint=adapter,
        training_manifest=training_manifest,
        output_root=tmp_path / "outputs",
        protocol="smoke",
        episodes=1,
    )
    path = tmp_path / "outputs" / eval_runner.MANIFEST_FILENAME
    digest = eval_runner.write_immutable_manifest(path, manifest)
    assert digest == hashlib.sha256(
        eval_runner._canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    changed = dict(manifest)
    changed["evaluation_visual_condition"] = "visual_arrows"
    with pytest.raises(ValueError, match="VISUAL_CONDITION"):
        eval_runner.write_immutable_manifest(tmp_path / "changed.json", changed)


def test_cli_requires_explicit_protocol_and_supports_no_videos():
    args = eval_runner.parse_args(
        [
            "--adapter-checkpoint", "adapter",
            "--training-manifest", "manifest.json",
            "--output-root", "outputs",
            "--episodes", "50",
            "--protocol", "full",
            "--no-videos",
        ]
    )
    assert args.episodes == 50
    assert args.protocol == "full"
    assert args.no_videos is True


def test_main_rejects_unsealed_video_mode_before_filesystem_access(capsys):
    result = eval_runner.main(
        [
            "--adapter-checkpoint", "missing-adapter",
            "--training-manifest", "missing-manifest.json",
            "--output-root", "outputs",
            "--episodes", "1",
            "--protocol", "smoke",
        ]
    )
    assert result == 1
    assert "requires --no-videos" in capsys.readouterr().out
