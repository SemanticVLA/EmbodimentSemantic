from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .artifacts import ArtifactStore
from .bbox_geometry import (
    INVERSE_RELATIONS,
    RELATION_FAMILY,
    axis_margin,
    bbox_center,
    bbox_iomin,
    deterministic_axis_relation,
    midpoint_arrow_evidence,
    relation_from_family,
    smooth_bbox_frames,
    unordered_visible_pairs,
    visible_objects,
)
from .metadata_signals import load_metadata_frames
from .schemas import (
    BBoxFrame,
    MetadataFrame,
    ProxyFrame,
    RelationRecord,
    read_jsonl,
)
from .task_priors import object_role, task_prior


MODES = ("gt",)


@dataclass(frozen=True)
class FusionOptions:
    use_metadata: bool


MODE_OPTIONS = {
    "gt": FusionOptions(True),
}


def _empty_metadata(frame: BBoxFrame) -> MetadataFrame:
    return MetadataFrame(
        task=frame.task,
        episode=frame.episode,
        frame=frame.frame,
        timestamp=frame.timestamp,
        phase="unknown",
        gripper_open=False,
        held=False,
        lifted=False,
        released=False,
        metadata_reliable=False,
        state_xyz=(0.0, 0.0, 0.0),
        state_rotation=(0.0, 0.0, 0.0),
        state_gripper=0.0,
        action_gripper=0.0,
        ee_speed=0.0,
        angular_speed=0.0,
        gates=("metadata_unavailable",),
    )


def _directional_records(
    a_name: str,
    b_name: str,
    relation: str,
    source: str,
    confidence: float,
    gates: tuple[str, ...],
    a_box: tuple[float, float, float, float],
    b_box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[RelationRecord, RelationRecord]:
    inverse = INVERSE_RELATIONS[relation]
    return (
        RelationRecord(
            a_name,
            relation,
            b_name,
            source,
            confidence,
            gates,
            midpoint_arrow_evidence(a_box, b_box, width, height),
        ),
        RelationRecord(
            b_name,
            inverse,
            a_name,
            source,
            confidence,
            gates,
            midpoint_arrow_evidence(b_box, a_box, width, height),
        ),
    )


def _support_pair(a_name: str, b_name: str) -> tuple[str, str] | None:
    if object_role(a_name) == "bowl":
        return a_name, b_name
    if object_role(b_name) == "bowl":
        return b_name, a_name
    return None


def _raw_pair_family(
    frame: BBoxFrame,
    a_name: str,
    b_name: str,
    metadata: MetadataFrame,
    options: FusionOptions,
    config: dict[str, Any],
    task_text: str,
) -> tuple[str, str, float, tuple[str, ...]]:
    a, b = frame.objects[a_name].bbox, frame.objects[b_name].bbox
    prior = task_prior(frame.task, task_text)
    support_pair = _support_pair(a_name, b_name)
    support_allowed = bool(support_pair and support_pair[1] in prior.allowed_supports)
    gates = tuple(metadata.gates if options.use_metadata else ())
    support_suppressed = bool(
        options.use_metadata
        and metadata.metadata_reliable
        and (metadata.held or metadata.lifted)
    )
    iomin = bbox_iomin(a, b)
    support_threshold = float(config["geometry"]["support_iomin_threshold"])
    soft_support_threshold = float(config["geometry"].get("support_soft_iomin_threshold", support_threshold))
    center_distance_threshold = float(config["geometry"].get("support_center_distance_threshold", 0.0))

    if support_allowed and not support_suppressed:
        ua, va = bbox_center(a, frame.width, frame.height)
        ub, vb = bbox_center(b, frame.width, frame.height)
        center_distance = ((ua - ub) ** 2 + (va - vb) ** 2) ** 0.5
        strong_support = iomin >= support_threshold
        soft_centered_support = (
            iomin >= soft_support_threshold
            and center_distance <= center_distance_threshold
        )
        if strong_support or soft_centered_support:
            family = "containment" if prior.containment else "support"
            return family, "bbox_support", max(0.5, min(1.0, iomin)), gates

    deterministic = deterministic_axis_relation(a, b, frame.width, frame.height)
    family = RELATION_FAMILY[deterministic]
    margin = axis_margin(a, b, frame.width, frame.height)
    return family, "bbox_axis", 0.5 + 0.5 * margin, gates


def fuse_agent_sequence(
    frames: list[BBoxFrame],
    metadata_index: dict[tuple[str, str, int], MetadataFrame],
    mode: str,
    config: dict[str, Any],
    task_text: str,
) -> list[ProxyFrame]:
    options = MODE_OPTIONS[mode]
    window = int(config["geometry"]["center_median_window"])
    frames = smooth_bbox_frames(frames, window)
    persistence = int(config["geometry"]["relation_persistence_frames"])
    state: dict[tuple[str, str], dict[str, Any]] = {}
    output: list[ProxyFrame] = []

    for frame in frames:
        metadata = metadata_index.get((frame.task, frame.episode, frame.frame), _empty_metadata(frame))
        relations: list[RelationRecord] = []
        for a_name, b_name in unordered_visible_pairs(frame):
            raw_family, raw_source, raw_confidence, gates = _raw_pair_family(
                frame,
                a_name,
                b_name,
                metadata,
                options,
                config,
                task_text,
            )
            key = (a_name, b_name)
            pair_state = state.setdefault(
                key,
                {"stable": raw_family, "candidate": raw_family, "count": persistence, "source": raw_source, "confidence": raw_confidence},
            )
            immediate_axis = options.use_metadata and (metadata.held or metadata.lifted) and raw_family in {"axis_lr", "axis_fb"}
            immediate_released_support = (
                options.use_metadata
                and metadata.metadata_reliable
                and metadata.released
                and raw_family in {"support", "containment"}
            )
            if raw_family == pair_state["stable"] or immediate_axis or immediate_released_support:
                pair_state.update(stable=raw_family, candidate=raw_family, count=persistence, source=raw_source, confidence=raw_confidence)
            else:
                if raw_family == pair_state["candidate"]:
                    pair_state["count"] += 1
                else:
                    pair_state.update(candidate=raw_family, count=1)
                if pair_state["count"] >= persistence:
                    pair_state.update(stable=raw_family, source=raw_source, confidence=raw_confidence)

            family = str(pair_state["stable"])
            relation = relation_from_family(
                family,
                a_name,
                b_name,
                frame.objects[a_name].bbox,
                frame.objects[b_name].bbox,
                frame.width,
                frame.height,
            )
            tracking = min(
                frame.objects[a_name].tracking_confidence,
                frame.objects[b_name].tracking_confidence,
            )
            confidence = max(0.0, min(1.0, float(pair_state["confidence"]) * tracking))
            relations.extend(
                _directional_records(
                    a_name,
                    b_name,
                    relation,
                    str(pair_state["source"]),
                    confidence,
                    gates,
                    frame.objects[a_name].bbox,
                    frame.objects[b_name].bbox,
                    frame.width,
                    frame.height,
                )
            )

        objects = visible_objects(frame)
        output.append(
            ProxyFrame(
                task=frame.task,
                episode=frame.episode,
                frame=frame.frame,
                timestamp=frame.timestamp,
                camera="agent_view",
                mode=mode,
                visible_objects=tuple(sorted(objects)),
                bboxes=objects,
                relations=tuple(relations),
                gripper_phase=metadata.phase if options.use_metadata else "not_used",
                metadata_reliable=metadata.metadata_reliable if options.use_metadata else False,
                model_version="none",
            )
        )
    return output


def filter_wrist_frame(agent: ProxyFrame, wrist: BBoxFrame) -> ProxyFrame:
    visible = visible_objects(wrist)
    active = set(visible) & set(agent.visible_objects)
    visible = {name: item for name, item in visible.items() if name in active}
    relations: list[RelationRecord] = []
    for item in agent.relations:
        if item.subject not in active or item.object not in active:
            continue
        subject = visible[item.subject]
        obj = visible[item.object]
        display = midpoint_arrow_evidence(
            subject.bbox,
            obj.bbox,
            wrist.width,
            wrist.height,
        )
        evidence = dict(item.evidence)
        evidence.update(
            {
                "semantic_camera": "agent_view",
                "visibility_camera": "wrist",
                "display_basis": "wrist_bbox_midpoint_arrow",
                "display_subject_center": display["subject_center"],
                "display_object_center": display["object_center"],
                "display_arrow_dx_norm": display["arrow_dx_norm"],
                "display_arrow_dy_norm": display["arrow_dy_norm"],
            }
        )
        relations.append(
            replace(
                item,
                confidence=min(
                    item.confidence,
                    subject.tracking_confidence,
                    obj.tracking_confidence,
                ),
                evidence=evidence,
            )
        )
    return ProxyFrame(
        task=agent.task,
        episode=agent.episode,
        frame=agent.frame,
        timestamp=agent.timestamp,
        camera="wrist",
        mode=agent.mode,
        visible_objects=tuple(sorted(active)),
        bboxes=visible,
        relations=tuple(relations),
        gripper_phase=agent.gripper_phase,
        metadata_reliable=agent.metadata_reliable,
        model_version=agent.model_version,
    )


def load_bbox_frames(path: str | Path) -> list[BBoxFrame]:
    return [BBoxFrame.from_dict(value) for value in read_jsonl(path)]


def generate_proxy_artifacts(
    bbox_path: str | Path,
    metadata_path: str | Path,
    episode_index_path: str | Path,
    artifacts: ArtifactStore,
    config: dict[str, Any],
    *,
    wrist_bbox_path: str | Path | None = None,
) -> dict[str, Any]:
    from .dataset import load_episode_index

    bbox_frames = load_bbox_frames(bbox_path)
    if wrist_bbox_path is not None and Path(wrist_bbox_path).resolve() != Path(bbox_path).resolve():
        bbox_frames.extend(load_bbox_frames(wrist_bbox_path))
    metadata_index = load_metadata_frames(str(metadata_path)) if Path(metadata_path).exists() else {}
    episodes = load_episode_index(episode_index_path)
    task_text = {(item.task, item.episode): item.task_text for item in episodes}

    grouped: dict[tuple[str, str, str], list[BBoxFrame]] = defaultdict(list)
    for frame in bbox_frames:
        grouped[(frame.task, frame.episode, frame.camera)].append(frame)

    report: dict[str, Any] = {
        "bbox_source": str(Path(bbox_path).resolve()),
        "bbox_sources": {
            "agent_view": str(Path(bbox_path).resolve()),
            "wrist": str(Path(wrist_bbox_path).resolve()) if wrist_bbox_path is not None else None,
        },
        "modes": {},
    }
    for mode in MODES:
        agent_records: list[ProxyFrame] = []
        wrist_records: list[ProxyFrame] = []
        for (task, episode, camera), frames in sorted(grouped.items()):
            if camera != "agent_view":
                continue
            agent_sequence = fuse_agent_sequence(
                sorted(frames, key=lambda item: item.frame),
                metadata_index,
                mode,
                config,
                task_text.get((task, episode), task.replace("-", " ")),
            )
            agent_records.extend(agent_sequence)
            wrist_by_frame = {
                item.frame: item
                for item in grouped.get((task, episode, "wrist"), [])
            }
            wrist_records.extend(
                filter_wrist_frame(agent, wrist_by_frame[agent.frame])
                for agent in agent_sequence if agent.frame in wrist_by_frame
            )

        artifacts.write_jsonl(
            f"proxy_graphs/{mode}/agent_view.jsonl",
            (item.to_dict() for item in agent_records),
        )
        artifacts.write_jsonl(
            f"proxy_graphs/{mode}/wrist.jsonl",
            (item.to_dict() for item in wrist_records),
        )
        source_counts: dict[str, int] = defaultdict(int)
        for record in (*agent_records, *wrist_records):
            for relation in record.relations:
                source_counts[relation.source] += 1
        report["modes"][mode] = {
            "agent_view_frames": len(agent_records),
            "wrist_frames": len(wrist_records),
            "directed_relations": sum(len(item.relations) for item in (*agent_records, *wrist_records)),
            "provenance": dict(sorted(source_counts.items())),
        }
    artifacts.write_json("reports/proxy_generation_report.json", report)
    return report
