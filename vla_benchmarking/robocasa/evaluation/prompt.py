"""Minimal RoboCasa prompt adaptation for the frozen LIBERO policy."""

from __future__ import annotations

from typing import Final


_FALLBACK_CANONICAL_PROMPT: Final = (
    "Point to exposed parts of the rim of the bowl highlighted in red where "
    "robot fingers could grasp it without touching nearby objects."
)

_OBJECT_CONTACT_PROMPT_TEMPLATE: Final = (
    "Point to visible grasp contact locations on the {noun} at the tail, or "
    "start, of the green arrow. Ignore the arrowhead and destination. Choose "
    "locations where a parallel-jaw gripper can descend from above, "
    "straddle the object, and close without touching nearby objects or support "
    "surfaces. Point only on the {noun}."
)


def canonical_prompt() -> str:
    """Return the frozen source prompt without importing another benchmark."""

    return _FALLBACK_CANONICAL_PROMPT


def adapt_source_noun(source_noun: str, *, template: str | None = None) -> str:
    """Replace only the canonical source noun in the Molmo prompt.

    The deliberately bowl-oriented ``rim`` wording remains unchanged for the
    first RoboCasa portability run.  This is the only semantic policy change
    permitted by the benchmark adapter.
    """

    noun = " ".join(str(source_noun).strip().split())
    if not noun:
        raise ValueError("source_noun must be non-empty")
    prompt = canonical_prompt() if template is None else str(template)
    if not prompt.strip():
        raise ValueError("template must be non-empty")
    occurrences = prompt.count("the bowl")
    if occurrences != 1:
        raise ValueError(
            "canonical prompt must contain exactly one literal 'the bowl' "
            f"source phrase; found {occurrences}"
        )
    return prompt.replace("the bowl", f"the {noun}", 1)


def object_contact_prompt(source_noun: str) -> str:
    """Return the explicit green-arrow contact prompt for a source noun."""

    noun = " ".join(str(source_noun).strip().split())
    if not noun:
        raise ValueError("source_noun must be non-empty")
    return _OBJECT_CONTACT_PROMPT_TEMPLATE.format(noun=noun)


__all__ = ["adapt_source_noun", "canonical_prompt", "object_contact_prompt"]
