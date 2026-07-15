from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from demo.so101_proxy_demo.proxy.dataset import index_dataset
from demo.so101_proxy_demo.proxy.schemas import BBoxObject, ProxyFrame, RelationRecord
from demo.so101_proxy_demo.proxy.validation import validate_proxy_frame


DATASET_ROOT = Path(__file__).resolve().parents[3] / "data" / "SO1001_dataset"


@pytest.mark.skipif(not DATASET_ROOT.exists(), reason="SO101 dataset is not available")
def test_real_dataset_index_and_missing_episode_detection() -> None:
    episodes, sampled, report = index_dataset(DATASET_ROOT)
    assert len(episodes) == 257
    assert len(sampled) == 8252
    assert report["sampled_per_camera"] == {"agent_view": 4126, "wrist": 4126}
    right_cookie = next(
        item for item in report["tasks"]
        if item["task"].endswith("cookie-at-the-ri")
    )
    assert right_cookie["missing_metadata_episode_indices"] == [12, 13, 14, 15]


def test_proxy_validation_checks_pair_completeness() -> None:
    proxy = ProxyFrame(
        task="place-the-black-bowl-from-the-left-to-the-top-of-the-stove",
        episode="episode_0",
        frame=0,
        timestamp=0,
        camera="agent_view",
        mode="bbox",
        visible_objects=("black_bowl", "black_stove"),
        bboxes={
            "black_bowl": BBoxObject((0, 0, 10, 10)),
            "black_stove": BBoxObject((0, 0, 20, 20)),
        },
        relations=(
            RelationRecord("black_bowl", "is_on_top_of", "black_stove", "bbox_support", 1),
            RelationRecord("black_stove", "is_below_of", "black_bowl", "bbox_support", 1),
        ),
        gripper_phase="released",
        metadata_reliable=True,
        model_version="none",
    )
    assert validate_proxy_frame(proxy) == []
    assert validate_proxy_frame(replace(proxy, relations=proxy.relations[:1]))
