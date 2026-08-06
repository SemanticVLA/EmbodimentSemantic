"""Scene-graph prompt formatters for target-filtered ablations."""

from __future__ import annotations

from collections.abc import Iterable


TARGET_OBJECT = "akita_black_bowl_1"

STAGE_A_FORMATS = (
    "standard",
    "triplet_published",
    "triplet_human",
    "natural_separate",
    "natural_compact",
)

LEGACY_FORMAT = "legacy_scene_graph"

EXTENDED_FORMATS = (
    "natural_compact_current",
    "natural_compact_no_label",
    "natural_compact_context_first",
    "triplet_human_bar",
    "triplet_human_shared_subject",
)

DEFAULT_ABLATION_FORMATS = (
    "standard",
    LEGACY_FORMAT,
    "triplet_published",
    "triplet_human",
    "triplet_human_bar",
    "triplet_human_shared_subject",
    "natural_separate",
    "natural_compact",
    "natural_compact_current",
    "natural_compact_no_label",
    "natural_compact_context_first",
)

SUPPORTED_CONTEXT_FORMATS = STAGE_A_FORMATS + (LEGACY_FORMAT,) + EXTENDED_FORMATS

FORMAT_ALIASES = {
    "existing_triplets": LEGACY_FORMAT,
    "legacy_triplets": LEGACY_FORMAT,
}

OBJECT_NAME_MAP = {
    "akita_black_bowl_1": "black bowl",
    "akita_black_bowl_2": "black bowl",
    "plate_1": "plate",
    "cookies_1": "cookie box",
    "glazed_rim_porcelain_ramekin_1": "ramekin",
    "wooden_cabinet_1": "wooden cabinet",
    "flat_stove_1": "stove",
}

RELATION_NAME_MAP = {
    "is_left_of": "left of",
    "is_right_of": "right of",
    "is_in_front_of": "in front of",
    "is_behind": "behind",
    "is_on_top_of": "on top of",
    "is_below_of": "below",
    "is_inside": "inside",
    "contains": "contains",
}


def normalize_context_format(format_name: str | None) -> str:
    normalized = (format_name or LEGACY_FORMAT).strip().lower()
    normalized = FORMAT_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_CONTEXT_FORMATS:
        allowed = ", ".join(SUPPORTED_CONTEXT_FORMATS)
        raise ValueError(f"Unknown CONTEXT_FORMAT '{format_name}'. Expected one of: {allowed}.")
    return normalized


def human_object_name(object_id: str) -> str:
    if object_id in OBJECT_NAME_MAP:
        return OBJECT_NAME_MAP[object_id]
    parts = object_id.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts)


def human_relation_name(relation_id: str) -> str:
    return RELATION_NAME_MAP.get(relation_id, relation_id.replace("_", " "))


def dedupe_relations(relations: Iterable[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[tuple[str, str, str]] = []
    for relation in relations:
        if relation in seen:
            continue
        seen.add(relation)
        deduped.append(relation)
    return deduped


def _machine_triplet(subject: str, relation: str, obj: str) -> str:
    return f"({subject}, {relation}, {obj})"


def _human_triplet(subject: str, relation: str, obj: str) -> str:
    return (
        f"({human_object_name(subject)}, "
        f"{human_relation_name(relation)}, "
        f"{human_object_name(obj)})"
    )


def _human_bar_triplet(subject: str, relation: str, obj: str) -> str:
    return (
        f"{human_object_name(subject)} | "
        f"{human_relation_name(relation)} | "
        f"{human_object_name(obj)}"
    )


def _natural_fact(subject: str, relation: str, obj: str) -> str:
    subject_name = human_object_name(subject)
    relation_name = human_relation_name(relation)
    object_name = human_object_name(obj)
    if relation == "contains":
        return f"The {subject_name} contains the {object_name}."
    return f"The {subject_name} is {relation_name} the {object_name}."


def _compact_relation_phrase(relation: str, obj: str, *, first: bool = False) -> str:
    object_name = human_object_name(obj)
    relation_name = human_relation_name(relation)
    if relation == "contains":
        return f"contains the {object_name}"
    if first:
        return f"is {relation_name} the {object_name}"
    return f"{relation_name} the {object_name}"


def _compact_relation_text(relations: list[tuple[str, str, str]]) -> tuple[str, str]:
    subject_name = human_object_name(relations[0][0] if relations else TARGET_OBJECT)
    relation_text = "; ".join(
        _compact_relation_phrase(relation, obj, first=index == 0)
        for index, (_, relation, obj) in enumerate(relations)
    )
    return subject_name, relation_text


def _compact_location_line(relations: list[tuple[str, str, str]], label: str) -> str:
    if not relations:
        return f"{label}: no target relations detected."
    subject_name, relation_text = _compact_relation_text(relations)
    return f"{label}: the {subject_name} {relation_text}."


def _compact_unlabeled_line(relations: list[tuple[str, str, str]]) -> str:
    if not relations:
        return "No target relations detected."
    subject_name, relation_text = _compact_relation_text(relations)
    return f"The {subject_name} {relation_text}."


def format_scene_context(
    task_text: str,
    target_relations: list[tuple[str, str, str]],
    format_name: str,
) -> str:
    """Serialize one fixed relation list into an ablation prompt.

    The relation extractor is responsible for target filtering and ordering. This
    function only changes surface form, so A1-A4 compare prompt representation
    without changing the underlying facts.
    """

    normalized = normalize_context_format(format_name)
    relations = dedupe_relations(target_relations)

    if normalized == "standard":
        return task_text

    if normalized == "triplet_published":
        lines = ["Current Scene graph:", "", f"Task: {task_text}"]
        lines.extend(_machine_triplet(*relation) for relation in relations)
        return "\n".join(lines)

    if normalized == "triplet_human":
        lines = [f"Task: {task_text}", "Relations:"]
        lines.extend(_human_triplet(*relation) for relation in relations)
        return "\n".join(lines)

    if normalized == "triplet_human_bar":
        lines = [f"Task: {task_text}", "Relations:"]
        lines.extend(_human_bar_triplet(*relation) for relation in relations)
        return "\n".join(lines)

    if normalized == "triplet_human_shared_subject":
        if not relations:
            return f"Task: {task_text}\nRelations: no target relations detected."
        subject_name = human_object_name(relations[0][0])
        relation_text = "; ".join(
            f"{human_relation_name(relation)} {human_object_name(obj)}"
            for _, relation, obj in relations
        )
        return f"Task: {task_text}\nRelations:\n{subject_name}: {relation_text}."

    if normalized == "natural_separate":
        lines = [f"Task: {task_text}"]
        lines.extend(_natural_fact(*relation) for relation in relations)
        return "\n".join(lines)

    if normalized == "natural_compact":
        return f"Task: {task_text}\n{_compact_location_line(relations, 'Location')}"

    if normalized == "natural_compact_current":
        return f"Task: {task_text}\n{_compact_location_line(relations, 'Current location')}"

    if normalized == "natural_compact_no_label":
        return f"Task: {task_text}\n{_compact_unlabeled_line(relations)}"

    if normalized == "natural_compact_context_first":
        return f"{_compact_location_line(relations, 'Location')}\nTask: {task_text}"

    raise ValueError(f"Format '{format_name}' is not implemented by the scene-graph formatter.")


def legacy_scene_graph_suffix(
    scene_graph: dict[str, list[tuple[str, str, str]]],
    *,
    target_subject: str | None = None,
) -> str:
    """Keep the pre-ablation scene-graph prompt available for old runs."""

    lines = ["Scene graph:"]
    for view_name, triplets in scene_graph.items():
        lines.append(f"View: {view_name}")
        if not triplets:
            lines.append("(none)")
            continue
        for subject, relation, obj in triplets:
            subject_name = subject
            object_name = obj
            if target_subject is not None and subject == target_subject:
                subject_name = f"target_{subject_name}"
            lines.append(f"({subject_name}, {relation}, {object_name})")
    return "\n".join(lines)
