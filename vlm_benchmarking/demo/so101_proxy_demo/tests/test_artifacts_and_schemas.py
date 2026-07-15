from __future__ import annotations

from pathlib import Path

import pytest

from demo.so101_proxy_demo.proxy.artifacts import ArtifactPathError, ArtifactStore
from demo.so101_proxy_demo.proxy.schemas import BBoxFrame, BBoxObject, ProxyFrame, RelationRecord


def test_artifact_store_refuses_external_writes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    assert store.path("reports/result.json").is_relative_to(store.root)
    with pytest.raises(ArtifactPathError):
        store.path("../outside.json")
    with pytest.raises(ArtifactPathError):
        store.path(store.root.parent / "outside.json")


def test_bbox_and_proxy_schema_round_trip() -> None:
    bbox = BBoxFrame(
        task="task",
        episode="episode_0",
        frame=30,
        timestamp=1.0,
        camera="agent_view",
        width=640,
        height=480,
        objects={"black_bowl": BBoxObject((10, 20, 30, 40), .9, .8)},
    )
    assert BBoxFrame.from_dict(bbox.to_dict()) == bbox

    proxy = ProxyFrame(
        task="task",
        episode="episode_0",
        frame=30,
        timestamp=1.0,
        camera="agent_view",
        mode="gt",
        visible_objects=("black_bowl", "black_stove"),
        bboxes={
            "black_bowl": BBoxObject((10, 20, 30, 40)),
            "black_stove": BBoxObject((0, 0, 100, 100)),
        },
        relations=(
            RelationRecord("black_bowl", "is_on_top_of", "black_stove", "bbox_support", .9),
            RelationRecord("black_stove", "is_below_of", "black_bowl", "bbox_support", .9),
        ),
        gripper_phase="released",
        metadata_reliable=True,
        model_version="none",
    )
    assert ProxyFrame.from_dict(proxy.to_dict()) == proxy
