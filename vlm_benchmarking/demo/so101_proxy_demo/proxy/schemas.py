from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .. import SCHEMA_VERSION


BBox = tuple[float, float, float, float]
Triplet = tuple[str, str, str]


@dataclass(frozen=True)
class EpisodeRecord:
    task: str
    task_text: str
    episode: str
    episode_index: int
    length: int
    fps: float
    dataset_from_index: int
    dataset_to_index: int
    data_chunk_index: int
    data_file_index: int
    videos: dict[str, dict[str, Any]]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EpisodeRecord":
        return cls(**value)


@dataclass(frozen=True)
class SampledFrameRecord:
    task: str
    episode: str
    episode_index: int
    frame: int
    timestamp: float
    camera: str
    width: int
    height: int
    image_path: str
    schema_version: str = SCHEMA_VERSION

    def key(self) -> tuple[str, str, int, str]:
        return self.task, self.episode, self.frame, self.camera

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SampledFrameRecord":
        return cls(**value)


@dataclass(frozen=True)
class BBoxObject:
    bbox: BBox
    confidence: float = 1.0
    tracking_confidence: float = 1.0
    visible: bool = True
    source: str = "detected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [float(item) for item in self.bbox],
            "confidence": float(self.confidence),
            "tracking_confidence": float(self.tracking_confidence),
            "visible": bool(self.visible),
            "source": str(self.source),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BBoxObject":
        bbox = value.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("BBox objects require bbox=[x1,y1,x2,y2]")
        return cls(
            bbox=tuple(float(item) for item in bbox),
            confidence=float(value.get("confidence", 1.0)),
            tracking_confidence=float(value.get("tracking_confidence", 1.0)),
            visible=bool(value.get("visible", True)),
            source=str(value.get("source", "detected")),
        )


@dataclass(frozen=True)
class BBoxFrame:
    task: str
    episode: str
    frame: int
    timestamp: float
    camera: str
    width: int
    height: int
    objects: dict[str, BBoxObject]
    detector: str = "imported"
    schema_version: str = SCHEMA_VERSION

    def key(self) -> tuple[str, str, int, str]:
        return self.task, self.episode, self.frame, self.camera

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["objects"] = {name: item.to_dict() for name, item in self.objects.items()}
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BBoxFrame":
        objects = {
            str(name): BBoxObject.from_dict(item)
            for name, item in dict(value.get("objects", {})).items()
        }
        return cls(
            task=str(value["task"]),
            episode=str(value["episode"]),
            frame=int(value["frame"]),
            timestamp=float(value.get("timestamp", int(value["frame"]) / 30.0)),
            camera=str(value["camera"]),
            width=int(value.get("width", 640)),
            height=int(value.get("height", 480)),
            objects=objects,
            detector=str(value.get("detector", "imported")),
            schema_version=str(value.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class MetadataFrame:
    task: str
    episode: str
    frame: int
    timestamp: float
    phase: str
    gripper_open: bool
    held: bool
    lifted: bool
    released: bool
    metadata_reliable: bool
    state_xyz: tuple[float, float, float]
    state_rotation: tuple[float, float, float]
    state_gripper: float
    action_gripper: float
    ee_speed: float
    angular_speed: float
    gates: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def key(self) -> tuple[str, str, int]:
        return self.task, self.episode, self.frame

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state_xyz"] = list(self.state_xyz)
        payload["state_rotation"] = list(self.state_rotation)
        payload["gates"] = list(self.gates)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MetadataFrame":
        return cls(
            task=str(value["task"]),
            episode=str(value["episode"]),
            frame=int(value["frame"]),
            timestamp=float(value["timestamp"]),
            phase=str(value["phase"]),
            gripper_open=bool(value["gripper_open"]),
            held=bool(value["held"]),
            lifted=bool(value["lifted"]),
            released=bool(value["released"]),
            metadata_reliable=bool(value["metadata_reliable"]),
            state_xyz=tuple(float(item) for item in value["state_xyz"]),
            state_rotation=tuple(float(item) for item in value["state_rotation"]),
            state_gripper=float(value["state_gripper"]),
            action_gripper=float(value["action_gripper"]),
            ee_speed=float(value["ee_speed"]),
            angular_speed=float(value["angular_speed"]),
            gates=tuple(str(item) for item in value.get("gates", [])),
            schema_version=str(value.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class RelationRecord:
    subject: str
    relation: str
    object: str
    source: str
    confidence: float
    metadata_gates: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def triplet(self) -> Triplet:
        return self.subject, self.relation, self.object

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata_gates"] = list(self.metadata_gates)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RelationRecord":
        return cls(
            subject=str(value["subject"]),
            relation=str(value["relation"]),
            object=str(value["object"]),
            source=str(value["source"]),
            confidence=float(value["confidence"]),
            metadata_gates=tuple(str(item) for item in value.get("metadata_gates", [])),
            evidence=dict(value.get("evidence", {})),
        )


@dataclass(frozen=True)
class ProxyFrame:
    task: str
    episode: str
    frame: int
    timestamp: float
    camera: str
    mode: str
    visible_objects: tuple[str, ...]
    bboxes: dict[str, BBoxObject]
    relations: tuple[RelationRecord, ...]
    gripper_phase: str
    metadata_reliable: bool
    model_version: str
    schema_version: str = SCHEMA_VERSION

    def key(self) -> tuple[str, str, int, str, str]:
        return self.task, self.episode, self.frame, self.camera, self.mode

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "episode": self.episode,
            "frame": self.frame,
            "timestamp": self.timestamp,
            "camera": self.camera,
            "mode": self.mode,
            "visible_objects": list(self.visible_objects),
            "bboxes": {name: item.to_dict() for name, item in self.bboxes.items()},
            "relations": [item.to_dict() for item in self.relations],
            "gripper_phase": self.gripper_phase,
            "metadata_reliable": self.metadata_reliable,
            "model_version": self.model_version,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProxyFrame":
        return cls(
            task=str(value["task"]),
            episode=str(value["episode"]),
            frame=int(value["frame"]),
            timestamp=float(value["timestamp"]),
            camera=str(value["camera"]),
            mode=str(value["mode"]),
            visible_objects=tuple(str(item) for item in value.get("visible_objects", [])),
            bboxes={
                str(name): BBoxObject.from_dict(item)
                for name, item in dict(value.get("bboxes", {})).items()
            },
            relations=tuple(RelationRecord.from_dict(item) for item in value.get("relations", [])),
            gripper_phase=str(value.get("gripper_phase", "unknown")),
            metadata_reliable=bool(value.get("metadata_reliable", False)),
            model_version=str(value.get("model_version", "none")),
            schema_version=str(value.get("schema_version", SCHEMA_VERSION)),
        )


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    import json

    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
