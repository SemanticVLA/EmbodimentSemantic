"""Offline geometry-rule tuning from cached SAM masks (no model rerun)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import product
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .ground_truth import GroundTruthIndex
from .metrics import EvaluationReport, evaluate_predictions, ground_truth_triplet_set
from samgraph_core.geometry_profiles import (
    SAMGRAPH_SPATIAL_MASK_GEOMETRY_PROFILE,
    samgraph_spatial_mask_geometry,
)


SUPPORTED_GEOMETRY_KEYS = frozenset({
    "profile",
    "rule_profile",
    "vertical_scale",
    "hull_coverage_min",
    "hull_coverage_near_min",
    "near_distance_max_normalized",
    "cookies_reverse_coverage_min",
    "cabinet_top_layer_quantile",
    "bbox_padding_x",
    "bbox_padding_y",
    "bbox_padding_x_fraction",
    "bbox_padding_y_fraction",
    "padding_x_fraction",
    "padding_y_fraction",
    "padding_x",
    "padding_y",
    "padding_x_px",
    "padding_y_px",
    "containment_iomin_threshold",
    "iomin_threshold",
    "direction_horizontal_axis",
    "horizontal_axis",
    "direction_horizontal_sign",
    "horizontal_sign",
    "direction_vertical_axis",
    "vertical_axis",
    "direction_vertical_sign",
    "vertical_sign",
    "direction_vertical_scale",
    "scale",
    "direction_tie_axis",
    "tie_axis",
    "direction_front_coefficients",
    "direction_left_coefficients",
    "coordinate_adjustment",
    "stack_order",
    "drawer_stack_order",
    "drawer_mode",
})

# These knobs belong to the old generic SamGraph geometry contract.  They are
# accepted by that contract but have no effect once a SamGraph proxy profile
# is selected; accepting them here would create a misleading candidate whose
# hash does not represent the requested override.  ``vertical_scale`` is the
# one historical alias retained below because it maps to the active proxy
# direction scale.
IGNORED_LEGACY_GEOMETRY_KEYS = frozenset({
    "hull_coverage_min",
    "hull_coverage_near_min",
    "near_distance_max_normalized",
    "cookies_reverse_coverage_min",
    "cabinet_top_layer_quantile",
})

# These spellings are understood by the low-level geometry manifest parser for
# backwards compatibility, but they are not safe sparse tuner inputs: after
# merging with the canonical preset they would be shadowed by the canonical
# field of the same setting.  Reject them here rather than accepting a
# candidate whose requested value is not represented by its hash.
NONCANONICAL_SAMGRAPH_ALIASES = frozenset({
    "bbox_padding_x_fraction",
    "bbox_padding_y_fraction",
    "padding_x_fraction",
    "padding_y_fraction",
    "padding_x",
    "padding_x_px",
    "padding_y",
    "padding_y_px",
    "iomin_threshold",
    "horizontal_axis",
    "horizontal_sign",
    "vertical_axis",
    "vertical_sign",
    "scale",
    "tie_axis",
})

_AFFINE_COORDINATE_KEYS = frozenset({
    "direction_front_coefficients",
    "direction_left_coefficients",
})

# A thread pool is deliberately used for cached replay.  On Windows a process
# pool would spawn fresh interpreters and pickle/copy the complete mask cache
# into each worker, which is both slow and potentially very memory hungry.
# Threads share the read-only cache and keep serial evaluation as the default.
MAX_TUNING_WORKERS = 32


def validate_geometry_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise ValueError("geometry candidate must be a JSON object")
    unsupported = sorted(set(candidate) - SUPPORTED_GEOMETRY_KEYS)
    if unsupported:
        raise ValueError(
            "candidate contains non-replayable or unsupported geometry keys: "
            + ", ".join(unsupported)
        )
    ignored_legacy = sorted(set(candidate) & IGNORED_LEGACY_GEOMETRY_KEYS)
    if ignored_legacy:
        raise ValueError(
            "candidate contains ignored legacy geometry keys: "
            + ", ".join(ignored_legacy)
        )
    aliases = sorted(set(candidate) & NONCANONICAL_SAMGRAPH_ALIASES)
    if aliases:
        raise ValueError(
            "candidate contains non-canonical SamGraph geometry aliases; use "
            "the direction/bbox/containment canonical keys: "
            + ", ".join(aliases)
        )
    value = dict(candidate)
    nested_coordinate = value.get("coordinate_adjustment")
    if nested_coordinate is not None:
        if not isinstance(nested_coordinate, Mapping):
            raise ValueError("coordinate_adjustment must be a JSON object")
        unsupported_nested = sorted(set(nested_coordinate) - _AFFINE_COORDINATE_KEYS)
        if unsupported_nested:
            raise ValueError(
                "candidate contains unsupported coordinate_adjustment keys: "
                + ", ".join(unsupported_nested)
            )
    # A bare candidate means "the SamGraph default" in this offline tuner.
    # The generic SamGraph default remains available to the live resolver through
    # geometry_rules=None and is not changed here.
    if not value:
        return samgraph_spatial_mask_geometry()
    profile = value.get("profile", value.get("rule_profile"))
    proxy_keys = {
        key for key in value
        if key not in {
            "profile", "rule_profile", "vertical_scale", "hull_coverage_min",
            "hull_coverage_near_min", "near_distance_max_normalized",
            "cookies_reverse_coverage_min", "cabinet_top_layer_quantile",
        }
    }
    if proxy_keys and profile is None:
        raise ValueError(
            "SamGraph geometry parameters require "
            f"profile={SAMGRAPH_SPATIAL_MASK_GEOMETRY_PROFILE}"
        )
    if profile is not None and str(profile) != SAMGRAPH_SPATIAL_MASK_GEOMETRY_PROFILE:
        raise ValueError(f"unsupported geometry profile: {profile!r}")
    # Candidate files may tune numeric/axis geometry knobs, but every candidate
    # is resolved from the same canonical profile and therefore shares its
    # fixed all-qualified support-pair semantics.  Hashes are computed from
    # the fully resolved graph rules, not from sparse input dictionaries.
    merged = samgraph_spatial_mask_geometry()
    merged.update(value)
    # Resolve the retained historical alias after merging the preset.  The
    # canonical key is present in the preset, so leaving the alias untouched
    # would otherwise make an explicit ``vertical_scale`` override inert.
    if "vertical_scale" in value and "direction_vertical_scale" not in value:
        merged["direction_vertical_scale"] = value["vertical_scale"]
    _resolved_config(merged)
    return merged


def _resolved_config(value: Mapping[str, Any] | None) -> dict[str, Any]:
    from samgraph_core.geometric_graph import geometric_relation_rules
    return geometric_relation_rules(value)


def config_hash(config: Mapping[str, Any] | None) -> str:
    from samgraph_core.geometric_graph import geometric_relation_rules_sha256
    return geometric_relation_rules_sha256(config)


def load_candidate_configs(path: str | Path, *, max_candidates: int = 256) -> list[dict[str, Any]]:
    """Load an explicit list or a bounded Cartesian grid from JSON."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, list):
        candidates = value
    elif isinstance(value, Mapping) and "candidates" in value:
        candidates = value["candidates"]
    elif isinstance(value, Mapping) and "grid" in value:
        grid = value["grid"]
        if not isinstance(grid, Mapping) or not grid:
            raise ValueError("candidate grid must be a non-empty mapping")
        names = list(grid)
        choices = [grid[name] if isinstance(grid[name], list) else [grid[name]] for name in names]
        candidates = [dict(zip(names, values)) for values in product(*choices)]
    else:
        candidates = [value]
    if not isinstance(candidates, list) or not candidates or len(candidates) > max_candidates:
        raise ValueError(f"candidate count must be between 1 and {max_candidates}")
    if not all(isinstance(item, Mapping) for item in candidates):
        raise ValueError("each geometry candidate must be a JSON object")
    return [validate_geometry_candidate(item) for item in candidates]


def _role(instance_id: str) -> tuple[str, str]:
    if instance_id.startswith("black_bowl_"):
        return "black_bowl", "bowl"
    if instance_id.startswith("white_ramekin_"):
        return "white_ramekin", "support"
    if instance_id.startswith("cookies_"):
        return "cookies", "support"
    if instance_id.startswith("plate_"):
        return "plate", "support"
    if instance_id.startswith("flat_stove_"):
        return "flat_stove", "support"
    if instance_id.startswith("wooden_cabinet_"):
        return "wooden_cabinet", "cabinet"
    return instance_id.rsplit("_", 1)[0], "support"


def _cached_masks(predictions_path: str | Path) -> dict[tuple[str, str, int], tuple[dict[str, np.ndarray], tuple[int, int]]]:
    result = {}
    with Path(predictions_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cache = row.get("mask_cache")
            if not cache:
                continue
            path = Path(cache)
            if not path.is_absolute():
                path = Path(predictions_path).parent / path
            if not path.is_file():
                raise FileNotFoundError(f"declared mask cache is missing: {path}")
            expected_digest = row.get("mask_cache_sha256")
            if not isinstance(expected_digest, str) or not expected_digest:
                raise ValueError(f"declared mask cache has no SHA-256: {path}")
            actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_digest != expected_digest:
                raise ValueError(f"mask cache SHA-256 mismatch: {path}")
            with np.load(path) as loaded:
                masks = {str(key): np.ascontiguousarray(loaded[key], dtype=bool) for key in loaded.files}
            if not masks:
                raise ValueError(f"declared mask cache is empty: {path}")
            shape = next(iter(masks.values())).shape
            if any(mask.shape != shape or not mask.any() for mask in masks.values()):
                raise ValueError(f"declared mask cache has invalid mask arrays: {path}")
            key = (str(row["task"]), str(row.get("demo", "demo_0")), int(row["frame"]))
            result[key] = (masks, shape)
    return result


def _tuning_input_provenance(
    predictions_path: str | Path,
    *,
    expected_keys: Iterable[tuple[str, str, int]],
    cached_keys: Iterable[tuple[str, str, int]],
) -> dict[str, Any]:
    """Return stable, candidate-independent provenance for the frozen mask set."""

    path = Path(predictions_path)
    expected = sorted(expected_keys)
    cached = set(cached_keys)
    rows_by_key: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    inventory = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["task"]), str(row.get("demo", "demo_0")), int(row["frame"]))
            if key not in cached:
                continue
            # The cache loader uses the last valid row for a repeated frame;
            # keep provenance aligned with that same deterministic selection
            # instead of double-counting duplicate JSONL records.
            if row.get("mask_cache") and row.get("mask_cache_sha256"):
                rows_by_key[key] = row

    cache_entries: list[dict[str, Any]] = []
    for key in sorted(cached):
        row = rows_by_key.get(key)
        if row is None:
            continue
        if row.get("error"):
            inventory["error"] += 1
        elif isinstance(row.get("inventory_coverage"), Mapping):
            coverage = row["inventory_coverage"]
            complete = (
                coverage.get("class_fraction") == 1.0
                and coverage.get("instance_fraction") == 1.0
            )
            inventory["complete" if complete else "partial"] += 1
        elif row.get("allow_partial_inventory") is True:
            missing = row.get("missing_class_ids") or []
            inventory["partial" if missing else "complete"] += 1
        else:
            inventory["unknown"] += 1
        cache = row["mask_cache"]
        digest = row["mask_cache_sha256"]
        cache_path = Path(cache)
        if not cache_path.is_absolute():
            cache_path = path.parent / cache_path
        try:
            display_path = os.path.relpath(cache_path.resolve(), path.parent.resolve())
        except ValueError:
            display_path = str(cache_path.resolve())
        cache_entries.append({
            "key": list(key),
            "path": display_path,
            "sha256": str(digest),
        })
    cache_digest = hashlib.sha256(
        json.dumps(cache_entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "predictions_jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expected_keys": [list(key) for key in expected],
        "cached_keys": [list(key) for key in sorted(cached)],
        "cache_file_count": len({(item["path"], item["sha256"]) for item in cache_entries}),
        "cache_entry_count": len(cache_entries),
        "cache_digest": cache_digest,
        "cache_entries": cache_entries,
        "inventory_counts": dict(sorted(inventory.items())),
    }


def _evaluate_cached_predictions(
    sampled_ground_truth: Mapping[tuple[str, str, int], set[tuple[str, str, str]]],
    predictions: Mapping[tuple[str, str, int], Iterable[object]],
    *,
    frame_stride: int,
) -> EvaluationReport:
    """Evaluate cached candidates without re-canonicalizing GT per candidate.

    This is intentionally the same accounting contract as
    ``metrics.evaluate_predictions``: sampled GT frames remain the denominator,
    missing prediction keys are empty predictions/FNs, and both primary
    duplicate-bowl-invariant and strict totals are retained.
    """

    # Reuse the same metric implementation as live scoring.  This keeps the
    # exact and VLM-comparator fields mathematically identical and prevents
    # offline tuning from silently drifting from the benchmark evaluator.
    sampled_index = GroundTruthIndex({
        key: frozenset(values) for key, values in sampled_ground_truth.items()
    })
    return evaluate_predictions(sampled_index, predictions, frame_stride=frame_stride)


def evaluate_cached_geometry(
    ground_truth: GroundTruthIndex,
    predictions_path: str | Path,
    candidates: Iterable[Mapping[str, Any]],
    *,
    frame_stride: int = 1,
    workers: int = 1,
    tasks: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Rebuild each frame's graph for each config using only cached masks.

    ``workers`` opts into a bounded thread pool over candidates.  The default
    remains serial for reproducibility and low-resource environments.  Threads
    are used instead of processes so Windows does not duplicate the complete
    cached-mask set through spawn/pickle; the cache is read-only during replay.
    Candidate reports are collected in input order before the existing metric
    sort, so parallel and serial runs have identical per-candidate results.
    """

    if not isinstance(frame_stride, int) or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise ValueError("workers must be a positive integer")
    if workers < 1 or workers > MAX_TUNING_WORKERS:
        raise ValueError(
            f"workers must be between 1 and {MAX_TUNING_WORKERS}"
        )

    from samgraph_core.geometric_graph import build_directed_relations
    requested_tasks = None if tasks is None else tuple(sorted({str(task) for task in tasks}))
    if requested_tasks is not None and not requested_tasks:
        raise ValueError("tasks must contain at least one task")
    available_tasks = {str(key[0]) for key in ground_truth.keys()}
    if requested_tasks is not None:
        unknown_tasks = sorted(set(requested_tasks) - available_tasks)
        if unknown_tasks:
            raise ValueError("tasks contain unknown task names: " + ", ".join(unknown_tasks))
    sampled_ground_truth = {
        key: ground_truth_triplet_set(ground_truth.frames[key])
        for key in ground_truth.keys()
        if key[2] % frame_stride == 0
        and (requested_tasks is None or str(key[0]) in requested_tasks)
    }
    cached = {
        key: value for key, value in _cached_masks(predictions_path).items()
        if key in sampled_ground_truth
    }
    if not cached:
        raise ValueError("selected tuning data has zero cached-mask coverage")
    input_provenance = _tuning_input_provenance(
        predictions_path,
        expected_keys=sampled_ground_truth,
        cached_keys=cached,
    )
    # Resolve and de-duplicate candidates before scheduling them.  Besides
    # preserving the historical config-hash contract, this prevents a large
    # duplicate grid from consuming unnecessary worker slots.
    unique_candidates: list[tuple[dict[str, Any], str]] = []
    seen_hashes: set[str] = set()
    for candidate in candidates:
        candidate = validate_geometry_candidate(candidate)
        candidate_digest = config_hash(candidate)
        if candidate_digest in seen_hashes:
            continue
        seen_hashes.add(candidate_digest)
        unique_candidates.append((candidate, candidate_digest))

    def evaluate_one(item: tuple[dict[str, Any], str]) -> dict[str, Any]:
        candidate, candidate_digest = item
        predictions = {}
        errors = 0
        for key, (masks, shape) in cached.items():
            # Live inference intentionally treats a frame with fewer than two
            # visible entities as a valid empty graph.  Mirror that contract
            # during offline replay: this is not a geometry failure and must
            # not invalidate an otherwise complete tuning candidate.
            if len(masks) < 2:
                predictions[key] = []
                continue
            instances = []
            for instance_id in masks:
                class_id, role = _role(instance_id)
                instances.append({"instance_id": instance_id, "class_id": class_id, "role": role})
            try:
                _relations, triplets, _evidence = build_directed_relations(
                    instances, masks, suite="spatial",
                    instruction=" ".join(key[0].replace("_", " ").split()),
                    shape=shape, geometry_rules=candidate,
                    allow_partial_visibility=True,
                )
                predictions[key] = [
                    [str(item["subject"]), str(item["relation"]), str(item["object"])]
                    for item in triplets
                ]
            except Exception:
                errors += 1
                predictions[key] = []
        report = _evaluate_cached_predictions(
            sampled_ground_truth, predictions, frame_stride=frame_stride
        )
        resolved = _resolved_config(candidate)
        return {
            "config": resolved,
            "config_hash": candidate_digest,
            # Explicit v2 metric names.  ``exact`` is the tuning objective;
            # ``vlm_comparator`` preserves the duplicate-bowl convention.
            "exact": report.exact,
            "vlm_comparator": report.vlm_comparator,
            "exact_per_task": report.exact_per_task,
            "vlm_per_task": report.vlm_per_task,
            "exact_mean_task_f1": report.exact_mean_task_f1,
            "vlm_mean_task_f1": report.vlm_mean_task_f1,
            # v1 aliases retained for the queued run/receiver contract.
            "primary": report.primary,
            "strict": report.strict,
            "per_task": report.per_task,
            "mean_task_f1": report.mean_task_f1,
            "frames_expected": report.frames_expected,
            "frame_stride": frame_stride,
            "frames_with_cached_masks": len(cached),
            "frames_scored_from_masks": len(predictions),
            "geometry_errors": errors,
            "mask_coverage": min(1.0, len(cached) / report.frames_expected) if report.frames_expected else 0.0,
            "frames_missing_mask_cache": max(0, report.frames_expected - len(cached)),
            "predictions_jsonl_sha256": input_provenance["predictions_jsonl_sha256"],
            "mask_cache_digest": input_provenance["cache_digest"],
            "input_provenance_sha256": input_provenance["cache_digest"],
            "exploratory": True,
        }

    effective_workers = min(workers, len(unique_candidates)) if unique_candidates else 1
    if effective_workers == 1:
        candidate_reports = [evaluate_one(item) for item in unique_candidates]
        execution_backend = "serial"
    else:
        # executor.map preserves the input order even though candidates finish
        # in different orders.  The final ranking remains the same stable sort
        # used by the serial implementation.
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            candidate_reports = list(executor.map(evaluate_one, unique_candidates))
        execution_backend = "thread_pool"

    candidate_reports.sort(
        key=lambda item: (
            -float(item["exact"]["f1"]),
            -float(item["exact_mean_task_f1"]),
            str(item["config_hash"]),
        )
    )
    return {"schema": "samgraph.geometry_tuning.v1", "exploratory": True,
            "frame_stride": frame_stride,
            "tasks": list(requested_tasks) if requested_tasks is not None else None,
            "input_provenance": input_provenance,
            "candidate_count": len(candidate_reports), "results": candidate_reports,
            "execution": {
                "backend": execution_backend,
                "workers": effective_workers,
                "requested_workers": workers,
            },
            "best": candidate_reports[0] if candidate_reports else None}


def _cached_metric_rank_key(result: Mapping[str, Any]) -> tuple[float, float, str]:
    """Stable exact-triplet ranking used by exploratory task CV."""

    return (
        -float(result["exact"]["f1"]),
        -float(result["exact_mean_task_f1"]),
        str(result["config_hash"]),
    )


def _sum_metric_counts(results: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "fn": 0}
    for result in results:
        metric = result.get(field)
        if not isinstance(metric, Mapping):
            raise ValueError(f"cached CV result is missing metric {field!r}")
        for name in counts:
            counts[name] += int(metric.get(name, 0))
    return counts


def _aggregate_cached_metric(counts: Mapping[str, int]) -> dict[str, Any]:
    tp, fp, fn = (int(counts[name]) for name in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_cached_geometry_leave_one_task_out(
    ground_truth: GroundTruthIndex,
    predictions_path: str | Path,
    candidates: Iterable[Mapping[str, Any]],
    *,
    frame_stride: int = 1,
    workers: int = 1,
    tasks: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Rank on nine tasks and score the selected candidate on the tenth.

    This is a leakage-aware exploratory check over the frozen mask cache.  The
    HDF5 labels are consulted only by ``evaluate_cached_geometry`` for scoring;
    they are never passed to graph construction or used to infer masks.  The
    returned fold score is therefore a held-out task score, while the
    all-ten score remains explicitly exploratory/resubstitution.
    """

    if not isinstance(frame_stride, int) or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer")
    candidate_list = [validate_geometry_candidate(candidate) for candidate in candidates]
    if not candidate_list:
        raise ValueError("candidates must contain at least one geometry candidate")
    sampled_task_names = sorted({
        str(key[0]) for key in ground_truth.keys() if key[2] % frame_stride == 0
    })
    if tasks is not None:
        requested_tasks = tuple(sorted({str(task) for task in tasks}))
        if not requested_tasks:
            raise ValueError("tasks must contain at least one task")
        unknown = sorted(set(requested_tasks) - set(sampled_task_names))
        if unknown:
            raise ValueError("tasks contain unknown task names: " + ", ".join(unknown))
        sampled_task_names = list(requested_tasks)
    if len(sampled_task_names) < 2:
        raise ValueError("leave-one-task-out evaluation requires at least two tasks")

    folds: list[dict[str, Any]] = []
    held_out_results: list[Mapping[str, Any]] = []
    for held_out in sampled_task_names:
        train_tasks = [task for task in sampled_task_names if task != held_out]
        train = evaluate_cached_geometry(
            ground_truth,
            predictions_path,
            candidate_list,
            frame_stride=frame_stride,
            workers=workers,
            tasks=train_tasks,
        )
        ranked = sorted(train["results"], key=_cached_metric_rank_key)
        if not ranked:
            raise ValueError(f"no candidates were evaluated for held-out task {held_out!r}")
        winner = ranked[0]
        winner_hash = str(winner["config_hash"])
        winner_candidates = [
            candidate for candidate in candidate_list
            if config_hash(candidate) == winner_hash
        ]
        if not winner_candidates:
            raise ValueError(f"CV winner {winner_hash} is absent from candidate list")
        held_out_eval = evaluate_cached_geometry(
            ground_truth,
            predictions_path,
            [winner_candidates[0]],
            frame_stride=frame_stride,
            workers=1,
            tasks=[held_out],
        )
        held_out_result = held_out_eval["results"][0]
        held_out_results.append(held_out_result)
        folds.append({
            "held_out_task": held_out,
            "train_tasks": train_tasks,
            "candidate_count": train["candidate_count"],
            "selected": {
                "config_hash": winner_hash,
                "config": winner_candidates[0],
                "resolved_config": winner.get("config"),
                "exact": winner.get("exact"),
                "exact_mean_task_f1": winner.get("exact_mean_task_f1"),
            },
            "held_out": {
                "exact": held_out_result["exact"],
                "vlm_comparator": held_out_result["vlm_comparator"],
                "exact_mean_task_f1": held_out_result["exact_mean_task_f1"],
                "vlm_mean_task_f1": held_out_result["vlm_mean_task_f1"],
                "frames_expected": held_out_result["frames_expected"],
                "frames_scored_from_masks": held_out_result["frames_scored_from_masks"],
                "frames_missing_mask_cache": held_out_result["frames_missing_mask_cache"],
                "geometry_errors": held_out_result["geometry_errors"],
                "predictions_jsonl_sha256": held_out_eval["input_provenance"]["predictions_jsonl_sha256"],
                "mask_cache_digest": held_out_eval["input_provenance"]["cache_digest"],
            },
        })

    exact_counts = _sum_metric_counts(held_out_results, "exact")
    comparator_counts = _sum_metric_counts(held_out_results, "vlm_comparator")
    return {
        "schema": "samgraph.geometry_tuning.leave_one_task_out.v1",
        "exploratory": True,
        "resubstitution": False,
        "frame_stride": frame_stride,
        "tasks": sampled_task_names,
        "candidate_count": len({config_hash(candidate) for candidate in candidate_list}),
        "selection": {
            "training_tasks_per_fold": len(sampled_task_names) - 1,
            "metric": "exact.f1",
            "tie_break": ["exact_mean_task_f1 descending", "config_hash ascending"],
            "labels_used_for_inference": False,
        },
        "folds": folds,
        "holdout_exact": _aggregate_cached_metric(exact_counts),
        "holdout_vlm_comparator": _aggregate_cached_metric(comparator_counts),
        "holdout_task_mean_exact_f1": round(
            sum(float(item["exact"]["f1"]) for item in held_out_results)
            / len(held_out_results), 4
        ),
        "input_provenance": {
            "predictions_jsonl_sha256": (
                folds[0]["held_out"]["predictions_jsonl_sha256"] if folds else None
            ),
            "mask_cache_digests_by_task": {
                fold["held_out_task"]: fold["held_out"]["mask_cache_digest"]
                for fold in folds
            },
            "note": "Each fold is replayed from the same frozen mask-cache JSONL; HDF5 labels are evaluation-only.",
        },
    }
