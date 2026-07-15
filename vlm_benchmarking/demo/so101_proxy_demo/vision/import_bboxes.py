from __future__ import annotations

from pathlib import Path
from typing import Any

from ..proxy.artifacts import ArtifactStore
from ..proxy.schemas import BBoxFrame, BBoxObject, read_jsonl
from ..proxy.task_priors import canonical_object_name


def _canonicalize_frame(value: dict[str, Any]) -> BBoxFrame:
    frame = BBoxFrame.from_dict(value)
    objects: dict[str, BBoxObject] = {}
    for name, item in frame.objects.items():
        canonical = canonical_object_name(name)
        if canonical is not None:
            previous = objects.get(canonical)
            if previous is None or item.confidence > previous.confidence:
                objects[canonical] = item
    return BBoxFrame(
        task=frame.task,
        episode=frame.episode,
        frame=frame.frame,
        timestamp=frame.timestamp,
        camera=frame.camera,
        width=frame.width,
        height=frame.height,
        objects=objects,
        detector=frame.detector,
        schema_version=frame.schema_version,
    )


def import_bbox_jsonl(source: str | Path, artifacts: ArtifactStore) -> dict[str, Any]:
    source_path = Path(source).resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    records = [_canonicalize_frame(value) for value in read_jsonl(source_path)]
    records.sort(key=lambda item: (item.task, item.episode, item.camera, item.frame))
    output = artifacts.write_jsonl("bboxes/imported.jsonl", (item.to_dict() for item in records))
    report = {
        "source": str(source_path),
        "output": str(output),
        "frames": len(records),
        "agent_view_frames": sum(1 for item in records if item.camera == "agent_view"),
        "wrist_frames": sum(1 for item in records if item.camera == "wrist"),
        "objects": sorted({name for item in records for name in item.objects}),
    }
    artifacts.write_json("reports/bbox_import_report.json", report)
    return report
