from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "arrow_failure_forensics.py"


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (value, value, value)).save(path)


def test_forensics_classifies_controller_phase_and_keeps_evaluator_offline(tmp_path):
    archive = tmp_path / "canonical"
    episode = archive / "sealed_randomized" / "task_4" / "episode_2"
    phase = episode / "phase_frames" / "03_lift.png"
    clean = episode / "clean_agentview.png"
    depth = episode / "agentview_depth_metric_m.npy"
    _write_image(phase, 30)
    _write_image(clean, 220)
    np.save(depth, np.asarray([[0.5, np.nan], [0.7, 0.6]], dtype=np.float32))
    audit = episode / "arrow_pick_place_audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": "arrow_pick_place_mvp.v1",
                "task_id": 4,
                "seed": 1002,
                "suite_mode": "sealed_randomized",
                "motion_executed": True,
                "phase_frames": ["phase_frames/03_lift.png"],
                "frames": ["clean_agentview.png"],
                "metric_depth": "agentview_depth_metric_m.npy",
                "phases": [
                    {"phase": "pregrasp", "status": "reached"},
                    {"phase": "lift", "status": "stall", "steps": 20},
                ],
                "evaluator_success": None,
                "endpoint_depths_m": {"source_tail": 0.5},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "forensics"
    result = subprocess.run(
        [sys.executable, str(TOOL), str(archive), "--output-dir", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["record_count"] == 1
    assert summary["controller_failure_count"] == 1
    assert summary["controller_failure_by_stage"] == {"lift": 1}
    assert summary["evaluator_outcomes_offline"] == {"success": 0, "failure": 0, "not_available": 1}
    row = summary["episodes"][0]
    assert row["controller_confidence"] == "high"
    assert row["failed_phase"] == "lift"
    assert row["valid_depth_file_count"] == 1
    assert row["existing_frame_count"] == 2
    assert Path(summary["contact_sheet"]).is_file()
    assert json.loads((output / "image_references.json").read_text(encoding="utf-8"))["references"]
    with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["controller_stage"] == "lift"


def test_forensics_does_not_turn_evaluator_failure_into_controller_failure(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "task_id": 9,
                "seed": 1000,
                "phases": [{"phase": "retreat", "status": "reached"}],
                "evaluator_success": False,
                "motion_executed": True,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    assert subprocess.run([sys.executable, str(TOOL), str(audit), "--output-dir", str(output)], check=False).returncode == 0
    row = json.loads((output / "summary.json").read_text(encoding="utf-8"))["episodes"][0]
    assert row["controller_failure"] is False
    assert row["controller_stage"] is None
    assert row["evaluator_outcome"] == "failure"


def test_forensics_outputs_are_deterministic_and_never_import_motion_stack():
    source = Path(TOOL).read_text(encoding="utf-8")
    assert "libero" not in source.lower()
    assert "mujoco" not in source.lower()
    assert "robosuite" not in source.lower()
    assert "subprocess" not in source

    # The implementation's stable ordering is tested through two equivalent
    # input files, avoiding any dependency on archive timestamps.
    first = Path.cwd() / ".pytest-forensics-a.json"
    second = Path.cwd() / ".pytest-forensics-b.json"
    try:
        payload = {"task_id": 1, "seed": 1, "phases": [{"phase": "close", "status": "timeout"}]}
        first.write_text(json.dumps(payload), encoding="utf-8")
        second.write_text(json.dumps(payload), encoding="utf-8")
        out_a = Path.cwd() / ".pytest-forensics-out-a"
        out_b = Path.cwd() / ".pytest-forensics-out-b"
        subprocess.run([sys.executable, str(TOOL), str(first), "--output-dir", str(out_a)], check=True)
        subprocess.run([sys.executable, str(TOOL), str(second), "--output-dir", str(out_b)], check=True)
        # Source paths are intentionally part of provenance, so compare the
        # deterministic schema/ordering rather than absolute path strings.
        a = json.loads((out_a / "summary.json").read_text(encoding="utf-8"))
        b = json.loads((out_b / "summary.json").read_text(encoding="utf-8"))
        for value in (a, b):
            value["input_files"] = []
            value["episodes"][0]["source_path"] = ""
        assert a == b
    finally:
        for path in (first, second):
            path.unlink(missing_ok=True)
        for path in (Path.cwd() / ".pytest-forensics-out-a", Path.cwd() / ".pytest-forensics-out-b"):
            if path.exists():
                for child in sorted(path.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                path.rmdir()
