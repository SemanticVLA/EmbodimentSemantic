from __future__ import annotations

import csv
import json
import os

from run_scene_graph_visual_ablation import _find_baseline, _write_visual_outputs


def _eval_info(pc_success: float, successes: list[bool]) -> dict:
    return {
        "per_task": [
            {
                "task_group": "libero_spatial",
                "task_id": 0,
                "metrics": {
                    "sum_rewards": [float(item) for item in successes],
                    "max_rewards": [float(item) for item in successes],
                    "successes": successes,
                    "video_paths": [f"episode_{index}.mp4" for index in range(len(successes))],
                },
            }
        ],
        "overall": {
            "pc_success": pc_success,
            "n_episodes": len(successes),
        },
    }


def test_visual_outputs_and_baseline_comparison(tmp_path):
    output_root = tmp_path / "sg_visual_ablation_test"
    condition_dir = output_root / "visual_arrows"
    condition_dir.mkdir(parents=True)
    (condition_dir / "eval_info.json").write_text(
        json.dumps(_eval_info(50.0, [True, False])),
        encoding="utf-8",
    )
    (condition_dir / "visual_relation_audit.jsonl").write_text(
        '{"condition":"visual_arrows"}\n',
        encoding="utf-8",
    )

    baseline_dir = tmp_path / "sg_format_ablation_previous" / "standard"
    baseline_dir.mkdir(parents=True)
    baseline_eval_info = baseline_dir / "eval_info.json"
    baseline_eval_info.write_text(
        json.dumps(_eval_info(25.0, [True, False, False, False])),
        encoding="utf-8",
    )

    _write_visual_outputs(
        output_root,
        "visual_arrows",
        condition_dir,
        1000,
        baseline_eval_info,
    )

    for filename in (
        "visual_summary.csv",
        "visual_task_summary.csv",
        "visual_episodes.csv",
        "visual_vs_baseline.csv",
    ):
        assert (output_root / filename).is_file()

    with (output_root / "visual_vs_baseline.csv").open(
        encoding="utf-8",
        newline="",
    ) as fh:
        comparison = list(csv.DictReader(fh))
    assert [row["condition"] for row in comparison] == ["standard", "visual_arrows"]
    assert [row["pc_success"] for row in comparison] == ["25.0000", "50.0000"]

    with (output_root / "visual_episodes.csv").open(
        encoding="utf-8",
        newline="",
    ) as fh:
        episodes = list(csv.DictReader(fh))
    assert len(episodes) == 2
    assert [row["seed"] for row in episodes] == ["1000", "1001"]


def test_find_baseline_uses_latest_standard_eval_info(tmp_path):
    older = tmp_path / "older" / "standard"
    newer = tmp_path / "newer" / "standard"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    older_eval = older / "eval_info.json"
    newer_eval = newer / "eval_info.json"
    older_eval.write_text("{}", encoding="utf-8")
    newer_eval.write_text("{}", encoding="utf-8")
    older_eval.touch()
    newer_eval.touch()
    older_eval_time = newer_eval.stat().st_mtime - 10
    os.utime(older_eval, (older_eval_time, older_eval_time))

    result = _find_baseline(tmp_path, None, tmp_path / "current")

    assert result == newer_eval.resolve()
