"""Shared validation contracts for sealed SmolVLA evaluation outputs.

The retired 2x2 treatment runner used to own these checks.  They are kept in
this small module because both retained profiles need the same raw-output and
randomization validation, while no retired policy runner should remain in the
operator surface.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

TASK_IDS = tuple(range(10))


def parse_int_list(value: str) -> list[int]:
    """Parse comma-separated or JSON integer lists, rejecting duplicates."""
    text = value.strip()
    try:
        parsed = json.loads(text) if text.startswith("[") else text.split(",")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid integer list: {value!r}") from exc
    if isinstance(parsed, (int, float)) or not isinstance(parsed, (list, tuple)):
        parsed = [parsed]
    try:
        result = [int(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer list: {value!r}") from exc
    if not result:
        raise ValueError("integer list must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"integer list contains duplicates: {result}")
    return result


def validate_randomization_audit(cell_output: Path, manifest: dict[str, Any]) -> None:
    """Require one complete observed reset record per task episode."""
    audit_path = cell_output / "randomization_audit.jsonl"
    if not audit_path.exists():
        raise ValueError(f"missing randomization audit: {audit_path}")
    records: list[dict[str, Any]] = []
    try:
        lines = audit_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"unreadable randomization audit: {audit_path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed randomization audit line {line_number}: {audit_path}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"randomization audit line {line_number} is not an object")
        records.append(record)
    expected_tasks = set(manifest["tasks"])
    seen_keys: set[tuple[int, int, int]] = set()
    per_task: dict[int, int] = {task_id: 0 for task_id in expected_tasks}
    for record in records:
        required = {"task_id", "env_index", "reset_sequence", "dimensions_enabled", "dimensions_realized", "details", "status"}
        if not required <= set(record):
            raise ValueError("randomization audit record is incomplete")
        try:
            task_id, env_index, reset_sequence = (int(record[key]) for key in ("task_id", "env_index", "reset_sequence"))
        except (TypeError, ValueError) as exc:
            raise ValueError("randomization audit key fields are invalid") from exc
        key = (task_id, env_index, reset_sequence)
        if task_id not in expected_tasks or env_index < 0 or reset_sequence <= 0:
            raise ValueError(f"randomization audit has invalid reset key: {key}")
        if key in seen_keys:
            raise ValueError(f"duplicate randomization audit reset record: {key}")
        seen_keys.add(key)
        per_task[task_id] += 1
        expected_dimensions = manifest["randomization_dimensions"][str(task_id)]
        if record["dimensions_enabled"] != expected_dimensions:
            raise ValueError(f"randomization audit enabled dimensions mismatch for task {task_id}")
        realized = record["dimensions_realized"]
        if set(realized) != set(expected_dimensions) or any(not isinstance(value, bool) for value in realized.values()):
            raise ValueError(f"randomization audit realized dimensions malformed for task {task_id}")
        if record["status"] != "ok" or any(bool(enabled) != bool(realized[name]) for name, enabled in expected_dimensions.items()):
            raise ValueError(f"randomization audit dimensions were not fully realized for task {task_id}")
        details = record["details"]
        if not isinstance(details, dict) or "removed" not in details or "projection" not in details:
            raise ValueError(f"randomization audit lacks observed removal/projection evidence for task {task_id}")
        expected_config = manifest["randomization_config"]
        if details["removed"] != expected_config["remove"][str(task_id)]:
            raise ValueError(f"randomization audit removal evidence mismatch for task {task_id}")
        projection = details["projection"]
        if not isinstance(projection, dict) or projection.get("success") is not True or (projection.get("required") and projection.get("projected") is not True):
            raise ValueError(f"randomization audit projection evidence failed for task {task_id}")
        if details.get("protected") != {"akita_black_bowl_1": True, "plate_1": True}:
            raise ValueError(f"randomization audit protected-object evidence failed for task {task_id}")
        if expected_dimensions.get("scene_layout"):
            layout = details.get("layout")
            expected_layout = expected_config["layout"][str(task_id)]
            expected_applied = sorted(label for operation in expected_layout for label in operation)
            if not isinstance(layout, dict) or layout.get("configured") != expected_layout or sorted(layout.get("applied", [])) != expected_applied or layout.get("skipped"):
                raise ValueError(f"randomization audit layout application evidence failed for task {task_id}")
        elif "layout" in details:
            raise ValueError(f"no-layout task {task_id} must not claim layout evidence")
    wrong_counts = [task_id for task_id, count in per_task.items() if count != int(manifest["episodes"])]
    if wrong_counts:
        raise ValueError(f"randomization audit must contain exactly {manifest['episodes']} resets per task: {wrong_counts}")


def validate_eval_info(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate one successful cell's raw LeRobot result without rewriting it."""
    if not path.is_file():
        raise ValueError(f"missing eval_info.json: {path}")
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable eval_info.json: {path}") from exc
    configured_tasks = manifest.get("tasks", list(TASK_IDS))
    if not isinstance(configured_tasks, list) or not configured_tasks or any(isinstance(task_id, bool) or not isinstance(task_id, int) for task_id in configured_tasks) or len(set(configured_tasks)) != len(configured_tasks):
        raise ValueError(f"manifest task schedule is invalid: {path}")
    expected_tasks = tuple(configured_tasks)
    per_task = info.get("per_task")
    if not isinstance(per_task, list) or len(per_task) != len(expected_tasks):
        raise ValueError(f"eval_info must contain exactly {len(expected_tasks)} task records: {path}")
    seen: set[int] = set()
    expected_episodes = int(manifest["episodes"])
    for record in per_task:
        if not isinstance(record, dict) or not isinstance(record.get("task_id"), int):
            raise ValueError(f"eval_info has an invalid task record: {path}")
        task_id = record["task_id"]
        if task_id in seen or task_id not in expected_tasks:
            raise ValueError(f"eval_info has duplicate or unexpected task {task_id}: {path}")
        seen.add(task_id)
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"eval_info task {task_id} lacks metrics: {path}")
        successes, sum_rewards, max_rewards = (metrics.get(key) for key in ("successes", "sum_rewards", "max_rewards"))
        if not all(isinstance(values, list) for values in (successes, sum_rewards, max_rewards)) or not all(len(values) == expected_episodes for values in (successes, sum_rewards, max_rewards)):
            raise ValueError(f"eval_info task {task_id} has incomplete metric arrays: {path}")
        if not all(isinstance(value, bool) for value in successes) or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for values in (sum_rewards, max_rewards) for value in values):
            raise ValueError(f"eval_info task {task_id} has invalid metric values: {path}")
    if seen != set(expected_tasks):
        raise ValueError(f"eval_info task IDs are incomplete: {path}")
    overall = info.get("overall")
    if overall is not None:
        if not isinstance(overall, dict) or overall.get("n_episodes") != len(expected_tasks) * expected_episodes:
            raise ValueError(f"eval_info overall episode count is incomplete: {path}")
        pc_success = overall.get("pc_success")
        if not isinstance(pc_success, (int, float)) or not math.isfinite(float(pc_success)) or not 0 <= float(pc_success) <= 100:
            raise ValueError(f"eval_info overall pc_success is invalid: {path}")
    return info
