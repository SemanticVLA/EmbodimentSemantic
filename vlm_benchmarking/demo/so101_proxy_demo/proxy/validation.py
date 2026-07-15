from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .artifacts import ArtifactStore
from .bbox_geometry import INVERSE_RELATIONS
from .schemas import ProxyFrame, read_jsonl
from .task_priors import task_prior


def validate_proxy_frame(frame: ProxyFrame, task_text: str = "") -> list[str]:
    errors: list[str] = []
    visible = set(frame.visible_objects)
    expected_relations = len(visible) * max(0, len(visible) - 1)
    if len(frame.relations) != expected_relations:
        errors.append(
            f"expected {expected_relations} directed relations for {len(visible)} objects, got {len(frame.relations)}"
        )

    triplets = [item.triplet() for item in frame.relations]
    if len(set(triplets)) != len(triplets):
        errors.append("duplicate directed triplets")
    for relation in frame.relations:
        if relation.subject not in visible or relation.object not in visible:
            errors.append(f"relation endpoint is not visible: {relation.triplet()}")
        inverse = INVERSE_RELATIONS.get(relation.relation)
        if inverse is None:
            errors.append(f"unknown relation: {relation.relation}")
        elif (relation.object, inverse, relation.subject) not in set(triplets):
            errors.append(f"missing inverse for {relation.triplet()}")

    prior = task_prior(frame.task, task_text)
    for relation in frame.relations:
        if relation.relation not in {"is_on_top_of", "is_inside"}:
            continue
        if relation.subject != "black_bowl" or relation.object not in prior.allowed_supports:
            errors.append(f"support relation violates task whitelist: {relation.triplet()}")
    return sorted(set(errors))


def validate_wrist_contract(frame: ProxyFrame, agent: ProxyFrame | None) -> list[str]:
    errors: list[str] = []
    if agent is None:
        return ["missing synchronized agent GT frame"]
    visible = set(frame.visible_objects)
    if visible != set(frame.bboxes):
        errors.append("wrist visible_objects do not match wrist bbox keys")
    expected = {
        relation.triplet()
        for relation in agent.relations
        if relation.subject in visible and relation.object in visible
    }
    actual = {relation.triplet() for relation in frame.relations}
    if actual != expected:
        errors.append("wrist relations are not the exact visibility-filtered agent graph")
    for relation in frame.relations:
        if relation.source.startswith("wrist"):
            errors.append(f"wrist relation was independently inferred: {relation.triplet()}")
        if relation.evidence.get("semantic_camera") != "agent_view":
            errors.append(f"wrist relation lacks agent semantic provenance: {relation.triplet()}")
        if relation.evidence.get("visibility_camera") != "wrist":
            errors.append(f"wrist relation lacks wrist visibility provenance: {relation.triplet()}")
    return sorted(set(errors))


def validate_proxy_files(
    artifact_root: str | Path,
    artifacts: ArtifactStore,
    *,
    expected_sampled_per_camera: int = 4126,
    cameras: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    root = Path(artifact_root)
    files = sorted((root / "proxy_graphs").glob("*/*.jsonl"))
    if cameras is not None:
        files = [path for path in files if path.stem in cameras]
    file_reports: list[dict[str, Any]] = []
    total_errors = 0
    for path in files:
        records = [ProxyFrame.from_dict(value) for value in read_jsonl(path)]
        agent_by_key: dict[tuple[str, str, int], ProxyFrame] = {}
        if path.stem == "wrist":
            agent_path = path.with_name("agent_view.jsonl")
            if agent_path.exists():
                agent_by_key = {
                    (item.task, item.episode, item.frame): item
                    for item in (
                        ProxyFrame.from_dict(value) for value in read_jsonl(agent_path)
                    )
                }
        errors: list[dict[str, Any]] = []
        for record in records:
            frame_errors = validate_proxy_frame(record)
            if record.camera == "wrist":
                frame_errors.extend(
                    validate_wrist_contract(
                        record,
                        agent_by_key.get((record.task, record.episode, record.frame)),
                    )
                )
                frame_errors = sorted(set(frame_errors))
            if frame_errors:
                errors.append(
                    {
                        "key": [record.task, record.episode, record.frame, record.camera, record.mode],
                        "errors": frame_errors,
                    }
                )
        total_errors += len(errors)
        counts = Counter(relation.source for record in records for relation in record.relations)
        file_reports.append(
            {
                "path": str(path.resolve()),
                "frames": len(records),
                "expected_complete_sampled_frames": expected_sampled_per_camera,
                "complete_sampled_coverage": len(records) == expected_sampled_per_camera,
                "invalid_frames": len(errors),
                "first_errors": errors[:20],
                "provenance": dict(sorted(counts.items())),
            }
        )

    report = {
        "files": file_reports,
        "files_checked": len(files),
        "cameras": list(cameras) if cameras is not None else "all",
        "invalid_frames": total_errors,
        "valid": bool(files) and total_errors == 0,
        "complete_sampled_coverage": bool(file_reports)
        and all(item["complete_sampled_coverage"] for item in file_reports),
    }
    report["ready"] = bool(report["valid"] and report["complete_sampled_coverage"])
    artifacts.write_json("reports/validation_report.json", report)
    return report
