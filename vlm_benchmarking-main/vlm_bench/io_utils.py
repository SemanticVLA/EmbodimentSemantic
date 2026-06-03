import csv
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .prompts import BIDIRECTIONAL_RELATIONS

RELATION_ALIASES = {
    "behind": "is_behind",
    "is_behind_of": "is_behind",
    "in_front_of": "is_in_front_of",
    "front_of": "is_in_front_of",
    "is_front_of": "is_in_front_of",
    "is_in_front": "is_in_front_of",
    "left_of": "is_left_of",
    "is_to_left_of": "is_left_of",
    "is_to_the_left_of": "is_left_of",
    "right_of": "is_right_of",
    "is_to_right_of": "is_right_of",
    "is_to_the_right_of": "is_right_of",
    "on_top_of": "is_on_top_of",
    "is_on_top": "is_on_top_of",
    "below_of": "is_below_of",
    "is_below": "is_below_of",
    "inside": "is_inside",
    "is_inside_of": "is_inside",
    "is_in": "is_inside",
    "is_contained_by": "is_inside",
    "is_contained_in": "is_inside",
    "is_contains": "contains",
    "is_contains_of": "contains",
    "contains_of": "contains",
}

_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*]\s*|\d+[\).\s-]+)")


def list_hdf5_files(input_dir: str, max_tasks: Optional[int] = None) -> list[Path]:
    input_path = Path(input_dir)
    if input_path.is_file():
        files = [input_path] if input_path.suffix.lower() in {".h5", ".hdf5"} else []
    else:
        files = sorted([*input_path.glob("**/*.h5"), *input_path.glob("**/*.hdf5")])
    if max_tasks is not None:
        files = files[:max_tasks]
    return files


def task_name_from_path(path: Path) -> tuple[str, str]:
    """Returns (task_name, task_hint). Strips _demo suffix; hint is human-readable."""
    stem = path.stem
    task_name = stem.replace("_demo", "")
    task_hint = task_name.replace("_", " ")
    return task_name, task_hint


def frame_to_pil(frame_rgb: np.ndarray, rotate180: bool = False) -> Image.Image:
    if isinstance(frame_rgb, np.ndarray) and frame_rgb.ndim == 1:
        image = Image.open(BytesIO(frame_rgb.tobytes())).convert("RGB")
        return image.rotate(180) if rotate180 else image

    if frame_rgb.dtype != np.uint8:
        frame_rgb = (frame_rgb * 255).clip(0, 255).astype(np.uint8)
    if rotate180:
        frame_rgb = np.rot90(frame_rgb, 2).copy()
    return Image.fromarray(frame_rgb)


def image_key_for_camera(camera_name: str) -> str:
    if camera_name == "eye_in_hand":
        return "eye_in_hand_rgb"
    if camera_name == "agentview":
        return "agentview_rgb"
    raise ValueError(f"Unknown camera '{camera_name}'. Expected one of: ['agentview', 'eye_in_hand']")


def canonicalize_relation(relation: str) -> str | None:
    rel = _clean_token(relation)
    rel = RELATION_ALIASES.get(rel, rel)
    return rel if rel in BIDIRECTIONAL_RELATIONS else None


def _clean_token(value) -> str:
    text = str(value).strip()
    text = text.strip("`'\"")
    text = text.strip("[](){}")
    text = text.strip("`'\"")
    return text.strip()


def _add_triplet(
    triplets: list[tuple[str, str, str]],
    seen: set[tuple[str, str, str]],
    a,
    relation,
    b,
) -> None:
    obj_a = _clean_token(a)
    obj_b = _clean_token(b)
    rel = canonicalize_relation(str(relation))
    if not rel or not obj_a or not obj_b:
        return
    if not _TOKEN_RE.match(obj_a) or not _TOKEN_RE.match(obj_b):
        return
    triplet = (obj_a, rel, obj_b)
    if triplet in seen:
        return
    seen.add(triplet)
    triplets.append(triplet)


def _add_json_triplets(value, triplets: list[tuple[str, str, str]], seen: set[tuple[str, str, str]]) -> None:
    if isinstance(value, dict):
        key_sets = [
            ("objectA", "relation", "objectB"),
            ("subject", "predicate", "object"),
            ("subj", "rel", "obj"),
        ]
        for a_key, rel_key, b_key in key_sets:
            if a_key in value and rel_key in value and b_key in value:
                _add_triplet(triplets, seen, value[a_key], value[rel_key], value[b_key])
                break
        for child in value.values():
            _add_json_triplets(child, triplets, seen)
        return

    if isinstance(value, list):
        if len(value) == 3 and all(not isinstance(v, (dict, list)) for v in value):
            _add_triplet(triplets, seen, value[0], value[1], value[2])
            return
        for child in value:
            _add_json_triplets(child, triplets, seen)


def parse_triplets(text: str) -> list[tuple[str, str, str]]:
    triplets = []
    seen = set()

    text = text or ""
    json_candidate = text.strip().strip("`")
    try:
        parsed = json.loads(json_candidate)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        _add_json_triplets(parsed, triplets, seen)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        line = _LIST_PREFIX_RE.sub("", line).strip()
        line = line.rstrip(",;")
        line = line.strip().strip("`")
        line = line.strip("[]()")
        line = line.strip()
        parts = [_clean_token(p) for p in line.split(",")]
        if len(parts) != 3 or not all(parts):
            continue
        _add_triplet(triplets, seen, parts[0], parts[1], parts[2])
    return triplets


def write_csv(path: str, rows: list[dict]):
    fieldnames = ["task", "demo", "frame", "camera", "objectA", "relation", "objectB"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_from_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for a, rel, b in parse_triplets(rec.get("response", "")):
                rows.append(
                    {
                        "task": rec.get("task", ""),
                        "demo": rec.get("demo", ""),
                        "frame": rec.get("frame", ""),
                        "camera": rec.get("camera", ""),
                        "objectA": a,
                        "relation": rel,
                        "objectB": b,
                    }
                )
    return rows


def append_jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
