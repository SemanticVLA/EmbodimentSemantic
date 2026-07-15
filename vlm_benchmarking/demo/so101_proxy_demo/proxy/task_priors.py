from __future__ import annotations

from dataclasses import dataclass


CANONICAL_OBJECTS = (
    "black_bowl",
    "red_drawer",
    "black_stove",
    "cookie",
    "white_plate",
)

ALIASES = {
    "black_bowl": "black_bowl",
    "bowl": "black_bowl",
    "akita_black_bowl_1": "black_bowl",
    "akita_black_bowl_2": "black_bowl",
    "red_drawer": "red_drawer",
    "drawer": "red_drawer",
    "wooden_cabinet_1": "red_drawer",
    "black_stove": "black_stove",
    "stove": "black_stove",
    "flat_stove_1": "black_stove",
    "cookie": "cookie",
    "cookies_1": "cookie",
    "white_plate": "white_plate",
    "plate": "white_plate",
    "plate_1": "white_plate",
}


@dataclass(frozen=True)
class TaskPrior:
    source_support: str | None
    target_support: str | None
    allowed_supports: frozenset[str]
    containment: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "source_support": self.source_support,
            "target_support": self.target_support,
            "allowed_supports": sorted(self.allowed_supports),
            "containment": self.containment,
        }


def canonical_object_name(name: str) -> str | None:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(normalized)


def task_prior(task_name: str, task_text: str = "") -> TaskPrior:
    text = f"{task_name} {task_text}".lower().replace("-", " ")
    source: str | None = None
    target: str | None = None
    containment = False

    if "from the top of the drawer" in text or "from on top of the drawer" in text:
        source = "red_drawer"
    if "cookie" in text:
        target = "cookie"
    elif "stove" in text:
        target = "black_stove"

    if "in the drawer" in text and "on top" not in text:
        containment = True

    allowed = frozenset(item for item in (source, target) if item is not None)
    return TaskPrior(
        source_support=source,
        target_support=target,
        allowed_supports=allowed,
        containment=containment,
    )


def object_role(name: str) -> str:
    canonical = canonical_object_name(name) or name.lower()
    if "bowl" in canonical:
        return "bowl"
    if "drawer" in canonical or "cabinet" in canonical:
        return "container_support"
    if "stove" in canonical:
        return "surface_support"
    if "cookie" in canonical:
        return "flat_support"
    if "plate" in canonical:
        return "flat_object"
    if "ramekin" in canonical:
        return "container"
    return "other"
