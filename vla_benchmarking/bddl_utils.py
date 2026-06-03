"""BDDL file rewriting utilities for object-removal randomization."""

from __future__ import annotations

import atexit
import os
import re
import tempfile

_tmp_files: list[str] = []


@atexit.register
def _cleanup_tmp_files() -> None:
    for path in _tmp_files:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass


def check_removal_safety(bddl_text: str, obj_name: str) -> None:
    """Raise ValueError if removing obj_name would orphan akita_black_bowl_1's placement.

    This catches the task 3 / task 5 trap: bowl_1 starts On a distractor object,
    so removing that object leaves bowl_1 with no placement sampler and it spawns
    at the origin.
    """
    pattern = re.compile(
        r"\(\s*On\s+akita_black_bowl_1\s+" + re.escape(obj_name) + r"\s*\)",
        re.IGNORECASE,
    )
    if pattern.search(bddl_text):
        raise ValueError(
            f"Cannot remove '{obj_name}': akita_black_bowl_1 starts On {obj_name} "
            f"in (:init). Removing it would leave bowl_1 with no placement sampler "
            f"and it would spawn at the origin. "
            f"Known unsafe combinations: task 3 + 'cookies_1', task 5 + 'glazed_rim_porcelain_ramekin_1'."
        )


def remove_object_from_bddl(bddl_text: str, obj_name: str) -> str:
    """Return bddl_text with all references to obj_name stripped.

    Edits only (:objects ...) and (:init ...). Does not touch (:regions),
    (:obj_of_interest), (:goal), or (:fixtures) — distractors never appear there
    in libero_spatial tasks.

    Raises ValueError if obj_name appears in (:goal ...) as a safety guard.
    """
    # Guard: obj_name must not be a goal object
    goal_match = re.search(r"\(:goal(.*?)\)\s*\)", bddl_text, re.DOTALL)
    if goal_match and re.search(r"\b" + re.escape(obj_name) + r"\b", goal_match.group(1)):
        raise ValueError(
            f"'{obj_name}' appears in (:goal ...) — removing it would make the task unreachable."
        )

    lines = bddl_text.split("\n")
    out = []
    for line in lines:
        stripped = line.strip()

        # (:objects ...) — remove the object token; drop line if it becomes empty or type-only
        # Format is: "  name1 name2 - type" — obj_name appears as a space-separated token
        # before the " - type" part.
        if re.search(r"\b" + re.escape(obj_name) + r"\b", stripped) and " - " in stripped and not stripped.startswith("("):
            new_line = re.sub(r"\b" + re.escape(obj_name) + r"\b\s*", "", line)
            # If only whitespace and "- type" remain (no object names left), drop the line
            remainder = new_line.strip()
            if re.fullmatch(r"-\s*\w+", remainder):
                continue
            out.append(new_line.rstrip())
            continue

        # (:init ...) — drop any predicate line that mentions obj_name
        # Matches lines like: (On obj_name ...) or (On something obj_name)
        if re.search(r"\b" + re.escape(obj_name) + r"\b", stripped) and stripped.startswith("("):
            continue

        out.append(line)

    return "\n".join(out)


def make_filtered_bddl(source_path: str, remove_objects: list[str]) -> str:
    """Write a filtered copy of source_path with remove_objects stripped out.

    Performs safety checks before rewriting. Returns the path to the temp file.
    The temp file is registered for deletion on process exit.
    """
    with open(source_path) as f:
        bddl = f.read()

    for obj in remove_objects:
        check_removal_safety(bddl, obj)
        bddl = remove_object_from_bddl(bddl, obj)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".bddl", delete=False)
    tmp.write(bddl)
    tmp.close()
    _tmp_files.append(tmp.name)
    return tmp.name
