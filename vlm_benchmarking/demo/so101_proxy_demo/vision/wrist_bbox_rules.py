from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
from typing import Any

from ..proxy.artifacts import ArtifactStore
from ..proxy.bbox_geometry import bbox_iou
from ..proxy.dataset import load_episode_index, load_sampled_index
from ..proxy.metadata_signals import load_metadata_frames
from ..proxy.schemas import BBoxFrame, BBoxObject, MetadataFrame, SampledFrameRecord, read_jsonl
from .detection_cache import DetectionCache
from .grounded_sam2 import GroundedSam2Detector, _load_batch
from .optical_flow import track_bowl_sequence


WRIST_CANDIDATE_VERSION = "wrist-dino-candidates-v1"
WRIST_OUTPUT_VERSION = "wrist-bbox-rules-v1"


def _settings(config: dict[str, Any]) -> dict[str, float | int | str]:
    vision = config["vision"]
    return {
        "detector": GroundedSam2Detector.cache_name(config),
        "direct_threshold": float(vision.get("wrist_box_threshold", 0.50)),
        "candidate_threshold": float(vision.get("wrist_candidate_box_threshold", 0.15)),
        "text_threshold": float(vision.get("wrist_text_threshold", 0.20)),
        "minimum_area": float(vision.get("wrist_minimum_box_area_fraction", 0.001)),
        "maximum_area": float(vision.get("wrist_maximum_box_area_fraction", 0.90)),
        "maximum_bowl_area": float(vision.get("wrist_maximum_bowl_area_fraction", 0.80)),
        "flow_minimum_iou": float(
            vision.get("wrist_flow_minimum_bidirectional_iou", 0.25)
        ),
        "maximum_flow_angular_speed": float(
            vision.get("wrist_maximum_flow_angular_speed", 0.08)
        ),
        "held_anchor_minimum_iou": float(
            vision.get("wrist_held_anchor_minimum_iou", 0.50)
        ),
    }


def _fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(_settings(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def wrist_candidate_version(config: dict[str, Any]) -> str:
    return f"{WRIST_CANDIDATE_VERSION}-{_fingerprint(config)}"


def wrist_output_version(config: dict[str, Any]) -> str:
    return f"{WRIST_OUTPUT_VERSION}-{_fingerprint(config)}"


def _candidate_record(
    sample: SampledFrameRecord,
    objects: dict[str, BBoxObject],
    detector: str,
) -> BBoxFrame:
    return BBoxFrame(
        task=sample.task,
        episode=sample.episode,
        frame=sample.frame,
        timestamp=sample.timestamp,
        camera="wrist",
        width=sample.width,
        height=sample.height,
        objects={
            name: replace(item, source="wrist_dino_candidate")
            for name, item in objects.items()
        },
        detector=detector,
    )


def materialize_direct_wrist_frame(
    sample: SampledFrameRecord,
    candidates: BBoxFrame | None,
    *,
    direct_threshold: float,
    detector: str,
) -> BBoxFrame:
    objects = {
        name: replace(item, source="wrist_dino_direct")
        for name, item in (candidates.objects.items() if candidates else ())
        if item.confidence >= direct_threshold
    }
    return BBoxFrame(
        task=sample.task,
        episode=sample.episode,
        frame=sample.frame,
        timestamp=sample.timestamp,
        camera="wrist",
        width=sample.width,
        height=sample.height,
        objects=objects,
        detector=detector,
    )


def select_flow_recovery(
    candidate: BBoxObject | None,
    tracked: BBoxObject | None,
    *,
    minimum_iou: float,
) -> BBoxObject | None:
    if tracked is None:
        return None
    if candidate is None:
        return replace(tracked, source="wrist_optical_flow_fill")
    if bbox_iou(candidate.bbox, tracked.bbox) < minimum_iou:
        return None
    return replace(
        candidate,
        tracking_confidence=min(candidate.tracking_confidence, tracked.tracking_confidence),
        source="wrist_dino_temporal_confirmed",
    )


def flow_window_is_stable(
    signals: list[MetadataFrame],
    *,
    maximum_angular_speed: float,
) -> bool:
    return not signals or max(item.angular_speed for item in signals) <= maximum_angular_speed


def _median_box(items: list[BBoxObject]) -> tuple[float, float, float, float]:
    return tuple(float(median(values)) for values in zip(*(item.bbox for item in items)))


def apply_held_bowl_tracking(
    records: list[BBoxFrame],
    candidates_by_frame: dict[int, BBoxFrame],
    metadata_index: dict[tuple[str, str, int], MetadataFrame],
    *,
    minimum_anchor_iou: float,
) -> tuple[list[BBoxFrame], int]:
    if not records:
        return records, 0
    ordered = sorted(records, key=lambda item: item.frame)
    held_positions = []
    for index, record in enumerate(ordered):
        signal = metadata_index.get((record.task, record.episode, record.frame))
        if (
            signal is not None
            and signal.metadata_reliable
            and (signal.held or signal.lifted)
            and not signal.released
        ):
            held_positions.append(index)

    runs: list[list[int]] = []
    for position in held_positions:
        if not runs or position != runs[-1][-1] + 1:
            runs.append([position])
        else:
            runs[-1].append(position)

    filled = 0
    for run in runs:
        anchors = [
            (position, ordered[position].objects["black_bowl"])
            for position in run
            if "black_bowl" in ordered[position].objects
        ]
        if len(anchors) < 2:
            continue
        center = _median_box([item for _, item in anchors])
        consistent = [
            (position, item)
            for position, item in anchors
            if bbox_iou(item.bbox, center) >= minimum_anchor_iou
        ]
        if len(consistent) < 2:
            continue
        center = _median_box([item for _, item in consistent])
        start, end = consistent[0][0], consistent[-1][0]
        anchor_confidence = min(item.confidence for _, item in consistent)
        anchor_tracking = min(item.tracking_confidence for _, item in consistent)
        for position in run:
            if position <= start or position >= end:
                continue
            record = ordered[position]
            if "black_bowl" in record.objects:
                continue
            candidate_frame = candidates_by_frame.get(record.frame)
            candidate = (
                candidate_frame.objects.get("black_bowl")
                if candidate_frame is not None
                else None
            )
            if candidate is not None and bbox_iou(candidate.bbox, center) >= minimum_anchor_iou / 2:
                bowl = replace(
                    candidate,
                    tracking_confidence=min(candidate.tracking_confidence, 0.65),
                    source="wrist_dino_temporal_confirmed",
                )
            else:
                bowl = BBoxObject(
                    bbox=center,
                    confidence=0.70 * anchor_confidence,
                    tracking_confidence=0.60 * anchor_tracking,
                    visible=True,
                    source="wrist_held_bowl_track",
                )
            objects = dict(record.objects)
            objects["black_bowl"] = bowl
            ordered[position] = replace(record, objects=objects)
            filled += 1
    return ordered, filled


def _recover_isolated_flow_gaps(
    samples: list[SampledFrameRecord],
    records: list[BBoxFrame],
    candidates_by_frame: dict[int, BBoxFrame],
    signals: list[MetadataFrame],
    video: dict[str, Any],
    fps: float,
    config: dict[str, Any],
) -> tuple[list[BBoxFrame], Counter[str]]:
    stats: Counter[str] = Counter()
    if len(records) < 3 or not video.get("exists"):
        return records, stats
    settings = _settings(config)
    by_frame = {item.frame: item for item in records}
    direct_names = {item.frame: frozenset(item.objects) for item in records}
    signals_in_window = sorted(signals, key=lambda item: item.frame)

    for index in range(1, len(samples) - 1):
        previous_sample, sample, next_sample = samples[index - 1:index + 2]
        window_signals = [
            item
            for item in signals_in_window
            if previous_sample.frame <= item.frame <= next_sample.frame
        ]
        if not flow_window_is_stable(
            window_signals,
            maximum_angular_speed=float(settings["maximum_flow_angular_speed"]),
        ):
            stats["rapid_motion_rejections"] += 1
            continue
        previous = by_frame[previous_sample.frame]
        current = by_frame[sample.frame]
        following = by_frame[next_sample.frame]
        candidate_frame = candidates_by_frame.get(sample.frame)
        for name in sorted(direct_names[previous.frame] & direct_names[following.frame]):
            if name in direct_names[current.frame]:
                continue
            tracked = track_bowl_sequence(
                video_path=video["path"],
                video_start_timestamp=float(video.get("from_timestamp", 0.0)),
                fps=fps,
                previous_frame=previous.frame,
                target_frames=[current.frame],
                next_frame=following.frame,
                previous=previous.objects[name],
                following=following.objects[name],
                minimum_bidirectional_iou=float(settings["flow_minimum_iou"]),
            ).get(current.frame)
            candidate = candidate_frame.objects.get(name) if candidate_frame else None
            recovered = select_flow_recovery(
                candidate,
                tracked,
                minimum_iou=float(settings["flow_minimum_iou"]),
            )
            if recovered is None:
                continue
            objects = dict(current.objects)
            objects[name] = recovered
            current = replace(current, objects=objects)
            by_frame[current.frame] = current
            stats[recovered.source] += 1

    return [by_frame[item.frame] for item in records], stats


def _read_existing(path: Path) -> list[BBoxFrame]:
    if not path.exists():
        return []
    return [BBoxFrame.from_dict(value) for value in read_jsonl(path)]


def generate_wrist_bboxes(
    sampled_index_path: str | Path,
    metadata_path: str | Path,
    episode_index_path: str | Path,
    artifacts: ArtifactStore,
    config: dict[str, Any],
    *,
    task: str | None = None,
    episode: str | None = None,
    limit_frames: int | None = None,
) -> dict[str, Any]:
    all_wrist_samples = sorted(
        (item for item in load_sampled_index(sampled_index_path) if item.camera == "wrist"),
        key=lambda item: (item.task, item.episode_index, item.frame),
    )
    samples = [
        item
        for item in all_wrist_samples
        if (task is None or item.task == task)
        and (episode is None or item.episode == episode)
    ]
    if limit_frames is not None:
        samples = samples[:max(0, limit_frames)]
    if not samples:
        raise ValueError("No wrist samples matched the requested filters")

    candidate_version = wrist_candidate_version(config)
    output_version = wrist_output_version(config)
    settings = _settings(config)
    cache_path = artifacts.path("cache/wrist_bbox_candidates.sqlite3")
    output_path = artifacts.path("bboxes/wrist.jsonl")
    metadata_index = (
        load_metadata_frames(str(metadata_path)) if Path(metadata_path).exists() else {}
    )
    episodes = {
        (item.task, item.episode): item for item in load_episode_index(episode_index_path)
    }
    batch_size = max(1, int(config["vision"].get("batch_size", 8)))
    candidate_inferences = 0
    detector: GroundedSam2Detector | None = None

    with DetectionCache(cache_path) as cache:
        cached = cache.load(candidate_version)
        missing = [item for item in samples if item.key() not in cached]
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            if detector is None:
                detector = GroundedSam2Detector(config)
            images, paths = _load_batch(batch)
            detected = detector.detect_batch(
                images,
                paths,
                threshold=float(settings["candidate_threshold"]),
                text_threshold=float(settings["text_threshold"]),
                filter_minimum=float(settings["candidate_threshold"]),
            )
            records = [
                _candidate_record(sample, objects, candidate_version)
                for sample, objects in zip(batch, detected)
            ]
            cache.upsert(records)
            for record in records:
                cached[record.key()] = record
            candidate_inferences += len(records)
            completed = start + len(records)
            print(
                f"Wrist candidates: {completed} / {len(missing)} new frames; "
                f"{len(cached)} cached",
                file=sys.stderr,
                flush=True,
            )
        cache_counts = cache.detector_counts()

    grouped_samples: dict[tuple[str, str], list[SampledFrameRecord]] = defaultdict(list)
    for sample in samples:
        grouped_samples[(sample.task, sample.episode)].append(sample)

    generated: list[BBoxFrame] = []
    recovery_stats: Counter[str] = Counter()
    held_fills = 0
    for key, episode_samples in sorted(grouped_samples.items()):
        episode_samples.sort(key=lambda item: item.frame)
        candidates_by_frame = {
            sample.frame: cached[sample.key()]
            for sample in episode_samples
            if sample.key() in cached
        }
        records = [
            materialize_direct_wrist_frame(
                sample,
                candidates_by_frame.get(sample.frame),
                direct_threshold=float(settings["direct_threshold"]),
                detector=output_version,
            )
            for sample in episode_samples
        ]
        episode_record = episodes.get(key)
        episode_signals = [
            value
            for metadata_key, value in metadata_index.items()
            if metadata_key[:2] == key
        ]
        if episode_record is not None:
            records, flow_stats = _recover_isolated_flow_gaps(
                episode_samples,
                records,
                candidates_by_frame,
                episode_signals,
                episode_record.videos.get("wrist", {}),
                float(episode_record.fps),
                config,
            )
            recovery_stats.update(flow_stats)
        records, episode_held_fills = apply_held_bowl_tracking(
            records,
            candidates_by_frame,
            metadata_index,
            minimum_anchor_iou=float(settings["held_anchor_minimum_iou"]),
        )
        held_fills += episode_held_fills
        generated.extend(records)

    selected_keys = {item.key() for item in generated}
    preserved = [
        item
        for item in _read_existing(output_path)
        if item.detector == output_version and item.key() not in selected_keys
    ]
    merged = sorted(
        (*preserved, *generated),
        key=lambda item: (item.task, item.episode, item.frame, item.camera),
    )
    artifacts.write_jsonl("bboxes/wrist.jsonl", (item.to_dict() for item in merged))

    source_counts = Counter(
        item.source for frame in merged for item in frame.objects.values()
    )
    object_counts = Counter(name for frame in merged for name in frame.objects)
    visible_counts = Counter(len(frame.objects) for frame in merged)
    expected_keys = {item.key() for item in all_wrist_samples}
    output_keys = {item.key() for item in merged}
    report = {
        "version": output_version,
        "candidate_version": candidate_version,
        "requested_frames": len(samples),
        "candidate_inferences": candidate_inferences,
        "output_frames": len(merged),
        "expected_wrist_frames": len(all_wrist_samples),
        "complete_wrist_coverage": output_keys == expected_keys,
        "missing_wrist_frames": len(expected_keys - output_keys),
        "cache": str(cache_path),
        "cache_counts": cache_counts,
        "settings": settings,
        "filters": {"task": task, "episode": episode, "limit_frames": limit_frames},
        "object_counts": dict(sorted(object_counts.items())),
        "objects_per_frame": {str(key): value for key, value in sorted(visible_counts.items())},
        "provenance": dict(sorted(source_counts.items())),
        "flow_recovery": dict(sorted(recovery_stats.items())),
        "held_bowl_fills": held_fills,
        "sam2_refinement": detector is not None and detector.sam_predictor is not None,
    }
    artifacts.write_json("reports/wrist_bbox_report.json", report)
    return report
