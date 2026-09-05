from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vla_benchmarking.arrow_finetuned_vla.workflows import run_no_arrow_sealed_eval as eval_runner


def _artifacts(tmp_path: Path) -> tuple[str, str]:
    adapter = tmp_path / "checkpoints" / "029190" / "pretrained_model"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    training_manifest = tmp_path / "training_manifest.json"
    training_manifest.write_text("{}\n", encoding="utf-8")
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
