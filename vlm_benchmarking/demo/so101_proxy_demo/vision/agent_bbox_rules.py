from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from math import exp, hypot, log
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

from ..proxy.artifacts import ArtifactStore
from ..proxy.bbox_geometry import bbox_area, bbox_center, bbox_iomin
from ..proxy.dataset import load_episode_index, load_sampled_index
from ..proxy.metadata_signals import load_metadata_frames
from ..proxy.schemas import BBox, BBoxFrame, BBoxObject, SampledFrameRecord, read_jsonl
from ..proxy.task_priors import task_prior
from .detection_cache import DetectionCache
from .grounded_sam2 import (
    RECOVERY_PROMPTS,
    STATIC_OBJECTS,
    GroundedSam2Detector,
    plan_agent_episode,
)
from .optical_flow import track_bowl_sequence


AGENT_RULES_VERSION = "agent-bbox-rules-v7"
CANDIDATE_VERSION = "agent-bowl-candidates-v1"
STATIC_VERSION = "agent-static-anchors-v1"


def _settings(config: dict[str, Any]) -> dict[str, float | int]:
    vision = config["vision"]
    return {
        "threshold": float(vision.get("bowl_candidate_box_threshold", 0.05)),
        "minimum_area": float(vision.get("minimum_bowl_area_fraction", 0.002)),
        "maximum_area": float(vision.get("maximum_bowl_candidate_area_fraction", 0.04)),
        "ideal_area": float(vision.get("ideal_bowl_area_fraction", 0.018)),
        "batch_size": max(1, int(vision.get("batch_size", 8))),
        "minimum_flow_iou": float(vision.get("optical_flow_minimum_bidirectional_iou", 0.15)),
    }


def _candidate_fingerprint(config: dict[str, Any]) -> str:
    vision = config["vision"]
    settings = _settings(config)
    values = {
        "model": vision["grounding_dino_model"],
        "model_revision": str(vision.get("grounding_dino_revision", "")),
        "processor_use_fast": bool(vision.get("processor_use_fast", True)),
        "threshold": settings["threshold"],
        "minimum_area": settings["minimum_area"],
        "maximum_area": settings["maximum_area"],
        "text_threshold": float(vision.get("text_threshold", 0.25)),
        "sam2_config": str(vision.get("sam2_config", "")),
        "sam2_checkpoint": str(vision.get("sam2_checkpoint", "")),
    }
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return f"{CANDIDATE_VERSION}-{digest}"


def _fingerprint(config: dict[str, Any]) -> str:
    values = {
        "candidate_version": _candidate_fingerprint(config),
        "static_version": _static_cache_name(config),
        "ideal_area": _settings(config)["ideal_area"],
        "minimum_flow_iou": _settings(config)["minimum_flow_iou"],
    }
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return f"{AGENT_RULES_VERSION}-{digest}"


def _candidate_cache_name(config: dict[str, Any]) -> str:
    return _candidate_fingerprint(config)


def _static_cache_name(config: dict[str, Any]) -> str:
    vision = config["vision"]
    values = {
        "model": vision["grounding_dino_model"],
        "model_revision": str(vision.get("grounding_dino_revision", "")),
        "processor_use_fast": bool(vision.get("processor_use_fast", True)),
        "box_threshold": float(vision.get("box_threshold", 0.30)),
        "text_threshold": float(vision.get("text_threshold", 0.25)),
        "recovery_threshold": float(vision.get("targeted_recovery_box_threshold", 0.12)),
        "maximum_area": float(vision.get("maximum_box_area_fraction", 0.65)),
        "sam2_config": str(vision.get("sam2_config", "")),
        "sam2_checkpoint": str(vision.get("sam2_checkpoint", "")),
    }
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return f"{STATIC_VERSION}-{digest}"


def _candidate_objects(frame: BBoxFrame | None) -> list[BBoxObject]:
    if frame is None:
        return []
    return [frame.objects[name] for name in sorted(frame.objects)]


def _area_fraction(item: BBoxObject, width: int, height: int) -> float:
    return bbox_area(item.bbox) / max(1.0, float(width * height))


def _box_distance(a: BBox, b: BBox, width: int, height: int) -> float:
    ac = bbox_center(a, width, height)
    bc = bbox_center(b, width, height)
    return hypot(ac[0] - bc[0], ac[1] - bc[1])


def _support_affinity(candidate: BBox, support: BBox, width: int, height: int) -> float:
    overlap = bbox_iomin(candidate, support)
    if overlap > 0:
        return overlap
    x_gap = max(0.0, support[0] - candidate[2], candidate[0] - support[2]) / width
    y_gap = max(0.0, support[1] - candidate[3], candidate[1] - support[3]) / height
    return 0.35 * exp(-10.0 * hypot(x_gap, y_gap))


def _change_score(box: BBox, first: np.ndarray, second: np.ndarray) -> float:
    height, width = first.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    x1, x2 = max(0, min(x1, width - 1)), max(1, min(x2, width))
    y1, y2 = max(0, min(y1, height - 1)), max(1, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    delta = np.abs(first[y1:y2, x1:x2].astype(np.float32) - second[y1:y2, x1:x2])
    return min(2.0, float(delta.mean()) / 48.0)


def select_endpoint_candidate(
    candidates: list[BBoxObject],
    *,
    support: BBoxObject | None,
    role: str,
    task: str,
    width: int,
    height: int,
    ideal_area_fraction: float,
    change_scores: dict[BBox, float] | None = None,
) -> tuple[BBoxObject | None, float]:
    """Select a source or destination bowl using task geometry, not score alone."""
    if not candidates:
        return None, float("-inf")
    changes = change_scores or {}
    direction = "left" if "from-the-left" in task else (
        "right" if "from-the-right" in task else None
    )
    support_center = bbox_center(support.bbox, width, height) if support else None

    def score(item: BBoxObject) -> float:
        area = max(1e-6, _area_fraction(item, width, height))
        area_penalty = abs(log(area / max(ideal_area_fraction, 1e-6)))
        support_score = (
            _support_affinity(item.bbox, support.bbox, width, height) if support else 0.0
        )
        relation_score = 0.0
        if role == "source" and direction and support_center:
            # A left/right source is beside the destination support, not on it.
            support_score = -bbox_iomin(item.bbox, support.bbox)
            x, _ = bbox_center(item.bbox, width, height)
            relation_score = 1.0 if (
                (direction == "left" and x < support_center[0])
                or (direction == "right" and x > support_center[0])
            ) else -1.0
        return (
            4.0 * support_score
            + 1.5 * relation_score
            + 0.8 * changes.get(item.bbox, 0.0)
            + 0.25 * item.confidence
            - 0.65 * area_penalty
        )

    selected = max(candidates, key=score)
    selected_score = score(selected)
    source = (
        "agent_bowl_source_prior"
        if role == "source"
        else "agent_bowl_target_prior"
    )
    return (
        replace(
            selected,
            confidence=max(0.55, selected.confidence),
            tracking_confidence=max(0.60, selected.tracking_confidence),
            source=source,
        ),
        selected_score,
    )


def _interpolate_box(left: BBox, right: BBox, alpha: float) -> BBox:
    return tuple((1.0 - alpha) * a + alpha * b for a, b in zip(left, right))  # type: ignore[return-value]


def _needs_flow_candidate(item: BBoxObject, width: int, height: int) -> bool:
    if item.source.endswith("linear_fallback"):
        return True
    x1, y1, x2, y2 = item.bbox
    margin = 1.0
    return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin


def select_candidate_path(
    frames: list[int],
    candidates: dict[int, list[BBoxObject]],
    *,
    source_frame: int,
    source: BBoxObject,
    target_frame: int,
    target: BBoxObject,
    width: int,
    height: int,
    ideal_area_fraction: float,
    change_scores: dict[tuple[int, BBox], float] | None = None,
) -> dict[int, BBoxObject]:
    if not frames:
        return {}
    changes = change_scores or {}
    diagonal = hypot(width, height)
    expected_area = max(
        1e-6,
        (_area_fraction(source, width, height) + _area_fraction(target, width, height)) / 2,
    )
    # Endpoint area is a better per-episode prior than the dataset-wide default.
    ideal = expected_area if expected_area > 0 else ideal_area_fraction
    states: dict[int, list[BBoxObject]] = {}
    for frame in frames:
        values = list(candidates.get(frame, []))
        alpha = (frame - source_frame) / max(1, target_frame - source_frame)
        fallback = BBoxObject(
            bbox=_interpolate_box(source.bbox, target.bbox, max(0.0, min(1.0, alpha))),
            confidence=0.40,
            tracking_confidence=0.35,
            visible=True,
            source="agent_bowl_linear_fallback",
        )
        values.append(fallback)
        states[frame] = values

    costs: list[dict[int, tuple[float, int | None]]] = []
    previous_items = [source]
    previous_costs = {0: (0.0, None)}
    for frame in frames:
        alpha = (frame - source_frame) / max(1, target_frame - source_frame)
        expected = _interpolate_box(source.bbox, target.bbox, max(0.0, min(1.0, alpha)))
        layer: dict[int, tuple[float, int | None]] = {}
        for index, item in enumerate(states[frame]):
            area = max(1e-6, _area_fraction(item, width, height))
            node = (
                0.90 * abs(log(area / ideal))
                + 0.35 * _box_distance(item.bbox, expected, width, height)
                - 0.65 * changes.get((frame, item.bbox), 0.0)
                - 0.50 * item.confidence
                + (0.9 if item.source.endswith("linear_fallback") else 0.0)
            )
            best = (float("inf"), None)
            for previous_index, previous in enumerate(previous_items):
                prior_cost = previous_costs[previous_index][0]
                transition = (
                    2.5
                    * hypot(
                        (item.bbox[0] + item.bbox[2] - previous.bbox[0] - previous.bbox[2]) / 2,
                        (item.bbox[1] + item.bbox[3] - previous.bbox[1] - previous.bbox[3]) / 2,
                    )
                    / diagonal
                    + 0.12 * abs(log(area / max(1e-6, _area_fraction(previous, width, height))))
                )
                value = prior_cost + transition + node
                if value < best[0]:
                    best = (value, previous_index)
            layer[index] = best
        costs.append(layer)
        previous_items = states[frame]
        previous_costs = layer

    last_frame = frames[-1]
    final_index = min(
        costs[-1],
        key=lambda index: costs[-1][index][0]
        + 2.5 * _box_distance(states[last_frame][index].bbox, target.bbox, width, height),
    )
    output: dict[int, BBoxObject] = {}
    index: int | None = final_index
    for layer_index in range(len(frames) - 1, -1, -1):
        assert index is not None
        frame = frames[layer_index]
        selected = states[frame][index]
        if not selected.source.endswith("linear_fallback"):
            source = (
                "agent_bowl_optical_flow_path"
                if selected.source == "agent_bowl_dynamic_optical_flow"
                else "agent_bowl_dino_temporal_path"
            )
            selected = replace(
                selected,
                confidence=max(0.50, selected.confidence),
                tracking_confidence=max(0.55, selected.tracking_confidence),
                source=source,
            )
        output[frame] = selected
        index = costs[layer_index][index][1]
    return output


def _image_array(path: str) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"))


def _open_rgb(path: str) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def _progress(phase: int, label: str, completed: int, total: int) -> None:
    percent = 100.0 if total == 0 else 100.0 * completed / total
    print(
        f"[agent-bbox {phase}/4] {label}: {completed} / {total} ({percent:.1f}%)",
        file=sys.stderr,
        flush=True,
    )


def _load_base_frames(path: Path) -> dict[tuple[str, str, int, str], BBoxFrame]:
    return {
        item.key(): item
        for item in (BBoxFrame.from_dict(value) for value in read_jsonl(path))
    }


def generate_agent_bboxes(
    sampled_index_path: str | Path,
    metadata_path: str | Path,
    episode_index_path: str | Path,
    artifacts: ArtifactStore,
    config: dict[str, Any],
    *,
    task: str | None = None,
    episode: str | None = None,
    limit_episodes: int | None = None,
) -> dict[str, Any]:
    """Build agent-view bboxes directly from images, metadata, and explicit rules.

    The legacy detected bbox artifact is intentionally not an input. Static
    objects and low-threshold bowl proposals have separate versioned caches.
    """
    settings = _settings(config)
    version = _fingerprint(config)
    candidate_version = _candidate_cache_name(config)
    static_version = _static_cache_name(config)
    samples = [item for item in load_sampled_index(sampled_index_path) if item.camera == "agent_view"]
    grouped_samples: dict[tuple[str, str], list[SampledFrameRecord]] = defaultdict(list)
    for sample in samples:
        grouped_samples[(sample.task, sample.episode)].append(sample)
    for values in grouped_samples.values():
        values.sort(key=lambda item: item.frame)
    selected_keys = [
        key for key in sorted(grouped_samples)
        if (task is None or key[0] == task) and (episode is None or key[1] == episode)
    ]
    if limit_episodes is not None:
        selected_keys = selected_keys[: max(0, limit_episodes)]
    if not selected_keys:
        raise ValueError("No agent-view episodes matched the requested filters")
    selected_frame_count = sum(len(grouped_samples[key]) for key in selected_keys)
    _progress(1, "Planned episodes", len(selected_keys), len(selected_keys))
    print(
        f"[agent-bbox 1/4] Selected {selected_frame_count} sampled frames; "
        "source images and videos remain read-only",
        file=sys.stderr,
        flush=True,
    )

    metadata = load_metadata_frames(str(metadata_path))
    metadata_by_episode: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for item in metadata.values():
        metadata_by_episode[(item.task, item.episode)].append(item)
    episodes = {(item.task, item.episode): item for item in load_episode_index(episode_index_path)}
    output_path = artifacts.path("bboxes/agent_view.jsonl")
    output: dict[tuple[str, str, int, str], BBoxFrame] = {}
    if output_path.exists():
        previous = _load_base_frames(output_path)
        output.update(
            (key, value) for key, value in previous.items() if value.detector == version
        )
    selected_key_set = set(selected_keys)
    output = {
        record_key: value
        for record_key, value in output.items()
        if record_key[:2] not in selected_key_set
    }

    plans = {
        key: plan_agent_episode(grouped_samples[key], metadata_by_episode.get(key, []))
        for key in selected_keys
    }
    required_samples: dict[tuple[str, str, int, str], SampledFrameRecord] = {}
    for key in selected_keys:
        values = grouped_samples[key]
        plan = plans[key]
        required_frames = {values[0].frame, values[-1].frame, *plan.bowl_detection_frames}
        for sample in values:
            if sample.frame in required_frames:
                required_samples[sample.key()] = sample

    cache_path = artifacts.path("cache/agent_bbox_candidates.sqlite3")
    detector: GroundedSam2Detector | None = None
    static_inferences = 0
    static_recovery_inferences = 0

    def get_detector() -> GroundedSam2Detector:
        nonlocal detector
        if detector is None:
            print(
                "[agent-bbox] Loading pinned Grounding DINO model...",
                file=sys.stderr,
                flush=True,
            )
            detector = GroundedSam2Detector(config)
            print("[agent-bbox] Grounding DINO ready", file=sys.stderr, flush=True)
        return detector

    with DetectionCache(cache_path) as cache:
        static_cached = cache.load(static_version)
        missing_static_samples = [
            grouped_samples[key][0]
            for key in selected_keys
            if grouped_samples[key][0].key() not in static_cached
        ]
        cached_static_count = len(selected_keys) - len(missing_static_samples)
        _progress(2, "Static anchors (cache)", cached_static_count, len(selected_keys))
        batch_size = int(settings["batch_size"])
        for start in range(0, len(missing_static_samples), batch_size):
            batch = missing_static_samples[start:start + batch_size]
            images = [_open_rgb(item.image_path) for item in batch]
            detections = get_detector().detect_batch(
                images,
                [Path(item.image_path) for item in batch],
            )
            records: list[BBoxFrame] = []
            for sample, objects in zip(batch, detections):
                static_objects = {
                    name: replace(item, source="agent_static_first_frame_dino")
                    for name, item in objects.items()
                    if name in STATIC_OBJECTS
                }
                record = BBoxFrame(
                    task=sample.task,
                    episode=sample.episode,
                    frame=sample.frame,
                    timestamp=sample.timestamp,
                    camera=sample.camera,
                    width=sample.width,
                    height=sample.height,
                    objects=static_objects,
                    detector=static_version,
                )
                records.append(record)
                static_cached[record.key()] = record
            cache.upsert(records)
            static_inferences += len(batch)
            _progress(
                2,
                "Static anchors",
                cached_static_count + min(start + len(batch), len(missing_static_samples)),
                len(selected_keys),
            )

        recovery_threshold = float(
            config["vision"].get("targeted_recovery_box_threshold", 0.12)
        )
        fallback_positions = (-1, "middle", "quarter")
        for attempt, position in enumerate(fallback_positions):
            for target_name in sorted(STATIC_OBJECTS):
                requests: list[SampledFrameRecord] = []
                request_keys: list[tuple[str, str]] = []
                for key in selected_keys:
                    anchor_sample = grouped_samples[key][0]
                    anchor = static_cached[anchor_sample.key()]
                    if target_name in anchor.objects:
                        continue
                    sequence = grouped_samples[key]
                    index = (
                        -1
                        if position == -1
                        else len(sequence) // (2 if position == "middle" else 4)
                    )
                    requests.append(sequence[index])
                    request_keys.append(key)
                for start in range(0, len(requests), batch_size):
                    batch = requests[start:start + batch_size]
                    key_batch = request_keys[start:start + batch_size]
                    images = [_open_rgb(item.image_path) for item in batch]
                    detections = get_detector().detect_batch(
                        images,
                        [Path(item.image_path) for item in batch],
                        prompt=RECOVERY_PROMPTS[target_name],
                        allowed_objects=frozenset({target_name}),
                        threshold=recovery_threshold,
                        filter_minimum=recovery_threshold,
                    )
                    updated: list[BBoxFrame] = []
                    for key, objects in zip(key_batch, detections):
                        item = objects.get(target_name)
                        if item is None:
                            continue
                        anchor_sample = grouped_samples[key][0]
                        anchor = static_cached[anchor_sample.key()]
                        anchor_objects = dict(anchor.objects)
                        anchor_objects[target_name] = replace(
                            item,
                            source=f"agent_static_targeted_dino_attempt_{attempt + 1}",
                        )
                        record = replace(anchor, objects=anchor_objects)
                        static_cached[record.key()] = record
                        updated.append(record)
                    cache.upsert(updated)
                    static_recovery_inferences += len(batch)
                    _progress(
                        2,
                        f"Recover {target_name} (attempt {attempt + 1})",
                        min(start + len(batch), len(requests)),
                        len(requests),
                    )

        cached = cache.load(candidate_version)
        missing = [sample for key, sample in required_samples.items() if key not in cached]
        cached_candidate_count = len(required_samples) - len(missing)
        _progress(3, "Bowl proposals (cache)", cached_candidate_count, len(required_samples))
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            images = [_open_rgb(item.image_path) for item in batch]
            detected_candidates = get_detector().detect_bowl_candidates_batch(
                images,
                threshold=float(settings["threshold"]),
                minimum_area_fraction=float(settings["minimum_area"]),
                maximum_area_fraction=float(settings["maximum_area"]),
            )
            records = []
            for sample, candidates in zip(batch, detected_candidates):
                record = BBoxFrame(
                    task=sample.task,
                    episode=sample.episode,
                    frame=sample.frame,
                    timestamp=sample.timestamp,
                    camera=sample.camera,
                    width=sample.width,
                    height=sample.height,
                    objects={f"candidate_{index:03d}": item for index, item in enumerate(candidates)},
                    detector=candidate_version,
                )
                records.append(record)
                cached[record.key()] = record
            cache.upsert(records)
            _progress(
                3,
                "Bowl proposals",
                cached_candidate_count + min(start + len(batch), len(missing)),
                len(required_samples),
            )

    provenance: Counter[str] = Counter()
    episode_reports: list[dict[str, Any]] = []
    _progress(4, "Building canonical tracks", 0, len(selected_keys))
    for sequence_number, key in enumerate(selected_keys, start=1):
        sequence = grouped_samples[key]
        plan = plans[key]
        first_sample, last_sample = sequence[0], sequence[-1]
        static_anchor = static_cached[first_sample.key()]
        missing_static = sorted(STATIC_OBJECTS - set(static_anchor.objects))
        if missing_static:
            episode_reports.append(
                {
                    "task": key[0],
                    "episode": key[1],
                    "status": "missing_static_objects",
                    "objects": missing_static,
                }
            )
            continue
        first_array = _image_array(first_sample.image_path)
        last_array = _image_array(last_sample.image_path)
        prior = task_prior(key[0], episodes[key].task_text)
        source_support_name = prior.source_support or prior.target_support
        source_support = (
            static_anchor.objects.get(source_support_name) if source_support_name else None
        )
        target_support = (
            static_anchor.objects.get(prior.target_support) if prior.target_support else None
        )
        source_candidates = _candidate_objects(cached.get(first_sample.key()))
        target_candidates = _candidate_objects(cached.get(last_sample.key()))
        source_changes = {
            item.bbox: _change_score(item.bbox, first_array, last_array)
            for item in source_candidates
        }
        target_changes = {
            item.bbox: _change_score(item.bbox, first_array, last_array)
            for item in target_candidates
        }
        source, source_score = select_endpoint_candidate(
            source_candidates,
            support=source_support,
            role="source",
            task=key[0],
            width=first_sample.width,
            height=first_sample.height,
            ideal_area_fraction=float(settings["ideal_area"]),
            change_scores=source_changes,
        )
        target, target_score = select_endpoint_candidate(
            target_candidates,
            support=target_support,
            role="target",
            task=key[0],
            width=last_sample.width,
            height=last_sample.height,
            ideal_area_fraction=float(settings["ideal_area"]),
            change_scores=target_changes,
        )
        if source is None or target is None:
            episode_reports.append({"task": key[0], "episode": key[1], "status": "missing_endpoint"})
            continue

        release_frame = plan.release_frame or last_sample.frame
        source_anchor = max(
            (
                sample.frame
                for sample in sequence
                if plan.movement_start is not None and sample.frame < plan.movement_start
            ),
            default=first_sample.frame,
        )
        dynamic_frames = [] if plan.metadata_reliable and plan.movement_start is None else [
            sample.frame for sample in sequence
            if (plan.movement_start is None or sample.frame >= plan.movement_start)
            and sample.frame < release_frame
            and sample.frame != first_sample.frame
        ]
        frame_candidates: dict[int, list[BBoxObject]] = {}
        for sample in sequence:
            if sample.frame not in dynamic_frames:
                continue
            frame_candidates[sample.frame] = _candidate_objects(cached.get(sample.key()))

        dynamic_changes: dict[tuple[int, BBox], float] = {}
        for sample in sequence:
            values = frame_candidates.get(sample.frame)
            if values is None:
                continue
            current_array = _image_array(sample.image_path)
            for item in values:
                dynamic_changes[(sample.frame, item.bbox)] = _change_score(
                    item.bbox, first_array, current_array
                )

        path = select_candidate_path(
            dynamic_frames,
            frame_candidates,
            source_frame=source_anchor,
            source=source,
            target_frame=release_frame,
            target=target,
            width=first_sample.width,
            height=first_sample.height,
            ideal_area_fraction=float(settings["ideal_area"]),
            change_scores=dynamic_changes,
        )
        flow_frames = [
            frame
            for frame, item in path.items()
            if _needs_flow_candidate(item, first_sample.width, first_sample.height)
        ]
        video = episodes[key].videos.get("agent_view", {})
        if flow_frames and video.get("exists"):
            flow_candidates = track_bowl_sequence(
                video_path=video["path"],
                video_start_timestamp=float(video.get("from_timestamp", 0.0)),
                fps=float(episodes[key].fps),
                previous_frame=source_anchor,
                target_frames=flow_frames,
                next_frame=release_frame,
                previous=source,
                following=target,
                minimum_bidirectional_iou=float(settings["minimum_flow_iou"]),
            )
            for frame, item in flow_candidates.items():
                path[frame] = replace(
                    item,
                    confidence=max(0.50, item.confidence),
                    tracking_confidence=max(0.55, item.tracking_confidence),
                    source="agent_bowl_optical_flow_path",
                )

        for sample in sequence:
            if plan.metadata_reliable and (
                plan.movement_start is None or sample.frame < plan.movement_start
            ):
                bowl = source
            elif sample.frame >= release_frame:
                bowl = target
            else:
                bowl = path.get(sample.frame, source)
            objects = dict(static_anchor.objects)
            objects["black_bowl"] = bowl
            record = BBoxFrame(
                task=sample.task,
                episode=sample.episode,
                frame=sample.frame,
                timestamp=sample.timestamp,
                camera=sample.camera,
                width=sample.width,
                height=sample.height,
                objects=objects,
                detector=version,
            )
            output[record.key()] = record
            provenance[bowl.source] += 1
        episode_reports.append(
            {
                "task": key[0],
                "episode": key[1],
                "status": "generated",
                "source_score": source_score,
                "target_score": target_score,
                "source_anchor_frame": source_anchor,
                "release_frame": release_frame,
                "dynamic_frames": len(dynamic_frames),
            }
        )
        _progress(4, "Building canonical tracks", sequence_number, len(selected_keys))

    artifacts.write_jsonl(
        "bboxes/agent_view.jsonl",
        (item.to_dict() for item in sorted(output.values(), key=lambda item: item.key())),
    )
    report = {
        "version": version,
        "candidate_version": candidate_version,
        "static_version": static_version,
        "legacy_bbox_input": False,
        "selected_episodes": len(selected_keys),
        "generated_episodes": sum(item["status"] == "generated" for item in episode_reports),
        "static_anchor_inferences": static_inferences,
        "static_recovery_inferences": static_recovery_inferences,
        "candidate_frames_required": len(required_samples),
        "candidate_frames_inferred": len(missing),
        "output_frames": len(output),
        "expected_agent_frames": len(samples),
        "complete_agent_coverage": len(output) == len(samples),
        "provenance": dict(provenance),
        "candidate_cache": str(cache_path),
        "output": str(output_path),
        "filters": {"task": task, "episode": episode, "limit_episodes": limit_episodes},
        "failures": [item for item in episode_reports if item["status"] != "generated"],
    }
    artifacts.write_json("reports/agent_bbox_report.json", report)
    print(
        f"[agent-bbox] Wrote {len(output)} frames to {output_path}",
        file=sys.stderr,
        flush=True,
    )
    return report
