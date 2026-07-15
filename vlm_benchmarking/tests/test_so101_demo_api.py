from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

from demo.so101_backend import DemoRepository, ProxyDemoServer
from demo.so101_proxy_demo.proxy.artifacts import ArtifactStore
from demo.so101_proxy_demo.proxy.schemas import (
    BBoxFrame,
    BBoxObject,
    EpisodeRecord,
    ProxyFrame,
    SampledFrameRecord,
)


def test_standalone_api_serves_nonblank_image_and_range_video(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    image_path = dataset / "task" / "frames" / "agent_view" / "episode_0" / "frame_000000.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12), (40, 120, 180)).save(image_path)
    video_path = dataset / "task" / "videos" / "observation.images.agent_view" / "chunk-000" / "file-000.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"0123456789abcdef")

    artifacts = ArtifactStore(tmp_path / "artifacts")
    artifacts.ensure_layout()
    episode = EpisodeRecord(
        task="task",
        task_text="task",
        episode="episode_0",
        episode_index=0,
        length=12,
        fps=30,
        dataset_from_index=0,
        dataset_to_index=12,
        data_chunk_index=0,
        data_file_index=0,
        videos={
            "agent_view": {
                "path": str(video_path.resolve()),
                "from_timestamp": 0,
                "to_timestamp": .4,
                "exists": True,
            }
        },
    )
    sample = SampledFrameRecord(
        task="task",
        episode="episode_0",
        episode_index=0,
        frame=0,
        timestamp=0,
        camera="agent_view",
        width=16,
        height=12,
        image_path=str(image_path.resolve()),
    )
    artifacts.write_jsonl("index/episodes.jsonl", [episode.to_dict()])
    artifacts.write_jsonl("index/sampled_frames.jsonl", [sample.to_dict()])
    bbox = BBoxFrame(
        task="task",
        episode="episode_0",
        frame=0,
        timestamp=0,
        camera="agent_view",
        width=16,
        height=12,
        objects={
            "black_bowl": BBoxObject((1, 1, 5, 5), 1, 1, True, "detected"),
            "red_drawer": BBoxObject((2, 2, 6, 6), 1, 1, True, "detected"),
        },
        detector="test",
    )
    artifacts.write_jsonl("bboxes/agent_view.jsonl", [bbox.to_dict()])
    proxy = ProxyFrame(
        task="task",
        episode="episode_0",
        frame=0,
        timestamp=0,
        camera="agent_view",
        mode="gt",
        visible_objects=("black_bowl", "red_drawer"),
        bboxes={
            "black_bowl": BBoxObject((9, 9, 12, 12), 1, 1, True, "smoothed"),
            "red_drawer": BBoxObject((8, 8, 11, 11), 1, 1, True, "smoothed"),
        },
        relations=(),
        gripper_phase="held",
        metadata_reliable=True,
        model_version="test",
    )
    artifacts.write_jsonl("proxy_graphs/gt/agent_view.jsonl", [proxy.to_dict()])
    config = {
        "paths": {
            "so101_dataset": str(dataset),
            "gemini_predictions": str(tmp_path / "predictions"),
        }
    }
    repository = DemoRepository(config, artifacts)
    server = ProxyDemoServer(("127.0.0.1", 0), repository)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/api/health") as response:
            health = json.loads(response.read())
        assert health["episodes"] == 1
        assert health["bbox_source"] == "agent_view.jsonl"

        with urlopen(base + "/api/frame?task=task&episode=episode_0&frame=0&camera=agent_view&mode=gt") as response:
            payload = json.loads(response.read())
        assert payload["bboxes"]["black_bowl"]["bbox"] == [1.0, 1.0, 5.0, 5.0]
        assert payload["bboxes"]["red_drawer"]["bbox"] == [2.0, 2.0, 6.0, 6.0]

        with urlopen(base + "/api/image?task=task&episode=episode_0&frame=0&camera=agent_view") as response:
            image_bytes = response.read()
        assert image_bytes.startswith(b"\xff\xd8")
        assert len(image_bytes) > 100

        request = Request(
            base + "/api/video?task=task&episode=episode_0&camera=agent_view",
            headers={"Range": "bytes=2-5"},
        )
        with urlopen(request) as response:
            assert response.status == 206
            assert response.read() == b"2345"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
