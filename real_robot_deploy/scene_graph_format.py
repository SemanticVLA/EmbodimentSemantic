"""Format (subject, relation, object) triplets into a prompt-ready string.

Same three formats as vla_benchmarking/libero_live_semantic_context.py
(triplet / natural_language / json), but operating on triplets that come
from a VLM's text response (vlm_bench.io_utils.parse_triplets) instead of
privileged MuJoCo simulator state. No robosuite dependency, so this is
safe to import from a real-robot deployment process.
"""

from __future__ import annotations

import json

Triplet = tuple[str, str, str]

_RELATION_PHRASES: dict[str, str] = {
    "is_left_of": "is to the left of",
    "is_right_of": "is to the right of",
    "is_in_front_of": "is in front of",
    "is_behind": "is behind",
    "is_on_top_of": "is on top of",
    "is_below_of": "is below",
    "is_inside": "is inside",
    "contains": "contains",
}


def format_triplet_line(subject: str, relation: str, obj: str) -> str:
    return f"({subject}, {relation}, {obj})"


def format_sentence(subject: str, relation: str, obj: str) -> str:
    subject_name = subject.replace("_", " ")
    object_name = obj.replace("_", " ")
    relation_phrase = _RELATION_PHRASES.get(relation, relation.replace("_", " "))
    return f"The {subject_name} {relation_phrase} the {object_name}."


def format_triplets(triplets: list[Triplet], fmt: str = "triplet") -> str:
    """Render triplets as `fmt` in {"triplet", "natural_language", "json"}.

    Object names (e.g. "black_cube_1" / "black_cube_2") are kept as-is, not
    collapsed to a shared category name -- with two same-category objects in
    a scene (the two-cubes-arrange task), stripping the instance suffix would
    make both sides of a relation read identically, e.g.
    "(black_cube, is_left_of, black_cube)", losing exactly the distinction
    the policy needs.
    """
    fmt = fmt.strip().lower()

    if not triplets:
        return "(no relations detected)" if fmt != "natural_language" else "No relations detected."

    if fmt == "json":
        items = [{"subject": s, "relation": r, "object": o} for s, r, o in triplets]
        return json.dumps(items, separators=(",", ":"))

    if fmt == "natural_language":
        return "\n".join(format_sentence(s, r, o) for s, r, o in triplets)

    # default: triplet
    return "\n".join(format_triplet_line(s, r, o) for s, r, o in triplets)


def context_label(fmt: str) -> str:
    return "Context (JSON)" if fmt.strip().lower() == "json" else "Context"




