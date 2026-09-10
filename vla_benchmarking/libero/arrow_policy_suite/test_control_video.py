from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np

from arrow_policy_suite.control_video import ownership_segments, write_control_video


def _records():
    records = []
    for timestep, owner in enumerate(("vla", "vla", "arrow", "arrow", "vla")):
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        image[..., 1] = timestep * 20
        frame = SimpleNamespace(timestep=timestep, observation={"agentview": image})
        records.append(SimpleNamespace(frame=frame, executed_by=owner, success=timestep == 4, terminal=timestep == 4))
    return tuple(records)


def test_segments_and_annotated_video_mark_takeover_and_handback(tmp_path: Path):
    records = _records()
    assert [segment.to_dict() for segment in ownership_segments(records)] == [
        {"owner": "vla", "start_step": 0, "end_step": 1, "frames": 2},
        {"owner": "arrow", "start_step": 2, "end_step": 3, "frames": 2},
        {"owner": "vla", "start_step": 4, "end_step": 4, "frames": 1},
    ]
    receipt = {"schema": "receipt", "status": "COMPLETED", "steps": 5}
    path = tmp_path / "task_00_episode_00.mp4"
    sidecar = write_control_video(records, path, task_id=0, episode_index=0, receipt=receipt, fps=5)
    assert path.is_file() and path.stat().st_size > 0
    assert sidecar["format"] == "H264/yuv420p"
    assert sidecar["frames"] == 5
    assert sidecar["owner_counts"] == {"vla": 3, "arrow": 2, "hybrid": 0}
    assert sidecar["transition_count"] == 2
    stored = json.loads(Path(str(path) + ".json").read_text())
    assert stored["video_sha256"] == sidecar["video_sha256"]
    frames = imageio.mimread(path)
    assert len(frames) == 5
    # The top-left pixel is in the opaque owner banner: blue VLA first,
    # orange Arrow during takeover, then blue after handback.
    assert frames[0][4, 4, 2] > frames[0][4, 4, 0]
    assert frames[2][4, 4, 0] > frames[2][4, 4, 2]
    assert frames[4][4, 4, 2] > frames[4][4, 4, 0]


def test_control_video_is_create_only(tmp_path: Path):
    records = _records()
    path = tmp_path / "video.mp4"
    write_control_video(records, path, task_id=0, episode_index=0, receipt={"x": 1}, fps=5)
    try:
        write_control_video(records, path, task_id=0, episode_index=0, receipt={"x": 1}, fps=5)
    except FileExistsError:
        pass
    else:
        raise AssertionError("video writer overwrote an existing artifact")
