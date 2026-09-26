"""Prediction-only runner and JSONL persistence for SamGraph."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .frames import FrameRecord, discover_archives, iter_zip_frames


def _is_count(value: Any) -> bool:
    """Return whether ``value`` is an explicit non-negative inventory count."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _inventory_from_masks(inventory: Any, masks: Any) -> dict[str, Any]:
    """Build frame-local inventory metadata from the masks being persisted.

    The resolver's inventory describes the episode's initial scene.  It is
    useful as expected-inventory provenance, but it must not be reported as
    the set of masks observed in the current frame.  This function therefore
    uses the same stringified keys that ``_cache_masks`` writes and only uses
    resolver metadata for expected counts/class names.
    """
    source = inventory if isinstance(inventory, Mapping) else {}
    source_coverage = source.get("coverage") if isinstance(source.get("coverage"), Mapping) else None
    source_completeness = (
        source.get("completeness") if isinstance(source.get("completeness"), Mapping) else None
    )
    source_missing = source.get("missing_class_ids") if isinstance(source.get("missing_class_ids"), list) else None

    # An empty/absent resolver inventory means expected counts are unknown;
    # do not turn that absence into fabricated zero denominators.
    expected_classes = (
        [str(class_id) for class_id in source_completeness]
        if source_completeness is not None and source_completeness
        else None
    )
    explicit_expected_class_count = (
        source_coverage.get("expected_class_count")
        if source_coverage is not None and _is_count(source_coverage.get("expected_class_count"))
        else None
    )
    expected_class_count = (
        explicit_expected_class_count
        if explicit_expected_class_count is not None
        else len(expected_classes) if expected_classes is not None else None
    )

    expected_instance_count = None
    if source_coverage is not None and _is_count(source_coverage.get("expected_instance_count")):
        expected_instance_count = int(source_coverage["expected_instance_count"])
    elif source_completeness is not None:
        expected_counts = [
            value.get("expected_count")
            for value in source_completeness.values()
            if isinstance(value, Mapping)
        ]
        if expected_counts and all(_is_count(value) for value in expected_counts):
            expected_instance_count = sum(int(value) for value in expected_counts)

    # ``_cache_masks`` stringifies keys into a dict before writing NPZ, so
    # mirror that normalization and avoid counting a colliding raw key twice.
    mask_keys = sorted({str(key) for key in masks}) if isinstance(masks, Mapping) else []

    def class_for_instance(instance_id: str) -> str:
        if expected_classes:
            matches = [
                class_id for class_id in expected_classes
                if instance_id == class_id or instance_id.startswith(class_id + "_")
            ]
            if matches:
                return max(matches, key=len)
        prefix, separator, suffix = instance_id.rpartition("_")
        return prefix if separator and suffix.isdigit() else instance_id

    observed_class_counts: dict[str, int] = {}
    for instance_id in mask_keys:
        class_id = class_for_instance(instance_id)
        observed_class_counts[class_id] = observed_class_counts.get(class_id, 0) + 1
    observed_classes = sorted(observed_class_counts)

    expected_instance_ids: list[str] | None = None
    if expected_classes is not None and source_completeness is not None:
        generated: list[str] = []
        can_generate = True
        for class_id in expected_classes:
            status = source_completeness.get(class_id)
            count = status.get("expected_count") if isinstance(status, Mapping) else None
            if not _is_count(count):
                can_generate = False
                break
            generated.extend(f"{class_id}_{index}" for index in range(1, int(count) + 1))
        if can_generate:
            expected_instance_ids = generated

    completeness: dict[str, dict[str, Any]] | None = None
    missing_class_ids: list[str] | None = None
    incomplete_class_ids: list[str] | None = None
    missing_instance_ids: list[str] | None = None
    matched_observed_class_count = None
    matched_observed_instance_count = None
    if expected_classes is not None:
        completeness = {}
        missing_class_ids = []
        incomplete_class_ids = []
        matched_observed_class_count = 0
        matched_observed_instance_count = 0
        for class_id in expected_classes:
            status = source_completeness.get(class_id, {})
            expected_count = status.get("expected_count") if isinstance(status, Mapping) else None
            observed_count = observed_class_counts.get(class_id, 0)
            matched_observed_class_count += int(observed_count > 0)
            matched_observed_instance_count += observed_count
            known_count = _is_count(expected_count)
            missing_class = observed_count == 0
            incomplete = known_count and observed_count < int(expected_count)
            if missing_class:
                missing_class_ids.append(class_id)
            if incomplete:
                incomplete_class_ids.append(class_id)
            completeness[class_id] = {
                "expected_count": int(expected_count) if known_count else expected_count,
                "observed_count": observed_count,
                "complete": (not incomplete) if known_count else None,
                "missing": incomplete if known_count else None,
            }
        if expected_instance_ids is not None:
            observed_set = set(mask_keys)
            missing_instance_ids = [
                instance_id for instance_id in expected_instance_ids if instance_id not in observed_set
            ]

    coverage = {
        "expected_class_count": expected_class_count,
        "observed_class_count": len(observed_classes),
        "class_fraction": (
            matched_observed_class_count / expected_class_count
            if expected_class_count not in (None, 0) and matched_observed_class_count is not None
            else (1.0 if expected_class_count == 0 else None)
        ),
        "expected_instance_count": expected_instance_count,
        "observed_instance_count": len(mask_keys),
        "instance_fraction": (
            matched_observed_instance_count / expected_instance_count
            if expected_instance_count not in (None, 0) and matched_observed_instance_count is not None
            else (1.0 if expected_instance_count == 0 else None)
        ),
        "observed_class_ids": observed_classes,
        "observed_instance_ids": mask_keys,
        "unrecognized_class_ids": sorted(
            set(observed_classes) - set(expected_classes or [])
        ),
    }
    return {
        "coverage": coverage,
        "completeness": completeness,
        "missing_class_ids": missing_class_ids,
        "incomplete_class_ids": incomplete_class_ids,
        "missing_instance_ids": missing_instance_ids,
        "initial_coverage": source_coverage,
        "initial_completeness": source_completeness,
        "initial_missing_class_ids": source_missing,
    }


class PredictionRunner:
    """Run a callable on sampled ZIP frames and preserve failures as empty outputs.

    The callable receives ``(rgb, task, frame)``.  It may return a graph
    mapping with ``triplets`` or an iterable of triplets.  It never receives
    ground truth and this class has no HDF5 dependency.
    """

    def __init__(self, predictor: Callable[[Any, str, int], Any], *, resolution: int | None = None,
                 undo_rotation: bool = True, mask_cache_dir: str | Path | None = None,
                 frame_stride: int = 1, tracking_stride: int | None = None,
                 no_clobber: bool = False, max_raw_frames: int | None = None,
                 episodes: tuple[int, ...] | None = (0,)):
        if not isinstance(frame_stride, int) or frame_stride < 1:
            raise ValueError("frame_stride must be a positive integer")
        if tracking_stride is None:
            tracking_stride = frame_stride
        if not isinstance(tracking_stride, int) or tracking_stride < 1:
            raise ValueError("tracking_stride must be a positive integer")
        if frame_stride % tracking_stride:
            raise ValueError("evaluation frame stride must be divisible by tracking stride")
        if max_raw_frames is not None and (
            not isinstance(max_raw_frames, int) or max_raw_frames < 1
        ):
            raise ValueError("max_raw_frames must be a positive integer")
        self.predictor = predictor
        self.resolution = resolution
        self.undo_rotation = undo_rotation
        self.mask_cache_dir = Path(mask_cache_dir) if mask_cache_dir is not None else None
        self.frame_stride = frame_stride
        self.tracking_stride = tracking_stride
        self.no_clobber = bool(no_clobber)
        self.max_raw_frames = max_raw_frames
        self.episodes = episodes

    def _cache_masks(self, task: str, frame: int, output: Any,
                     *, key: str = "masks", demo: str = "demo_0") -> tuple[str, str] | None:
        if self.mask_cache_dir is None or not isinstance(output, Mapping):
            return None
        raw = output.get(key)
        if not isinstance(raw, Mapping) or not raw:
            return None
        import numpy as np
        arrays = {str(key): np.asarray(value, dtype=bool) for key, value in raw.items()}
        suffix = "" if key == "masks" else f".{key}"
        directory = self.mask_cache_dir / task
        if demo != "demo_0":
            directory = directory / demo
        target = directory / f"{frame:06d}{suffix}.npz"
        if self.no_clobber and target.exists():
            raise FileExistsError(f"mask cache already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return str(target), digest

    @staticmethod
    def _triplets(value: Any) -> list:
        if isinstance(value, Mapping):
            value = value.get("triplets", [])
        if value is None:
            return []
        return [list(item) for item in value if isinstance(item, (list, tuple)) and len(item) == 3]

    def iter_archive(self, archive: str | Path) -> Iterable[dict]:
        on_start = getattr(self.predictor, "on_archive_start", None)
        on_end = getattr(self.predictor, "on_archive_end", None)
        archive_path = Path(archive)
        on_episode = getattr(self.predictor, "on_episode_start", None)
        if callable(on_episode):
            on_episode(archive_path.parent.name, archive_path.stem)
        elif callable(on_start):
            on_start(archive_path.parent.name)
        try:
            fatal_tracking_error: str | None = None
            for frame in iter_zip_frames(archive, resolution=self.resolution, undo_rotation=self.undo_rotation):
                if self.max_raw_frames is not None and frame.frame >= self.max_raw_frames:
                    break
                if frame.frame % self.tracking_stride:
                    continue
                error = None
                output = None
                if fatal_tracking_error is None:
                    try:
                        output = self.predictor(frame.rgb, frame.task, frame.frame)
                        if frame.frame % self.frame_stride:
                            continue
                        triplets = self._triplets(output)
                        mask_cache_info = self._cache_masks(frame.task, frame.frame, output, demo=frame.demo)
                        effective_cache_info = self._cache_masks(
                            frame.task, frame.frame, output, key="effective_masks", demo=frame.demo,
                        )
                    except Exception as exc:  # model failures become explicit FN frames
                        error = f"{type(exc).__name__}: {exc}"
                        if frame.frame % self.frame_stride:
                            fatal_tracking_error = f"raw frame {frame.frame}: {error}"
                            continue
                        if self.tracking_stride < self.frame_stride:
                            fatal_tracking_error = f"raw frame {frame.frame}: {error}"
                        triplets = []
                        mask_cache_info = None
                        effective_cache_info = None
                else:
                    if frame.frame % self.frame_stride:
                        continue
                    triplets = []
                    error = fatal_tracking_error
                    mask_cache_info = None
                    effective_cache_info = None
                inventory = output.get("inventory", {}) if isinstance(output, Mapping) else {}
                current_inventory = _inventory_from_masks(
                    inventory,
                    output.get("masks") if isinstance(output, Mapping) else None,
                )
                tracking_pair = output.get("tracking_pair", {}) if isinstance(output, Mapping) else {}
                yield {
                    "task": frame.task, "demo": frame.demo, "frame": frame.frame,
                    "triplets": triplets, "error": error,
                    "source_name": frame.source_name, "source_sha256": frame.source_sha256,
                    "width": int(frame.rgb.shape[1]), "height": int(frame.rgb.shape[0]),
                    "mask_cache": mask_cache_info[0] if mask_cache_info else None,
                    "mask_cache_sha256": mask_cache_info[1] if mask_cache_info else None,
                    "effective_masks_cache": (effective_cache_info[0]
                                               if effective_cache_info else None),
                    "effective_masks_cache_sha256": (effective_cache_info[1]
                                                      if effective_cache_info else None),
                    "frame_stride": self.frame_stride,
                    "tracking_stride": self.tracking_stride,
                    "geometry_rules": output.get("geometry_rules") if isinstance(output, Mapping) else None,
                    "geometry_rules_sha256": output.get("geometry_rules_sha256") if isinstance(output, Mapping) else None,
                    "relation_revision": output.get("relation_revision") if isinstance(output, Mapping) else None,
                    # These fields describe the finalized masks for this frame.
                    "inventory_coverage": current_inventory["coverage"],
                    "inventory_completeness": current_inventory["completeness"],
                    "missing_class_ids": current_inventory["missing_class_ids"],
                    "incomplete_class_ids": current_inventory["incomplete_class_ids"],
                    "missing_instance_ids": current_inventory["missing_instance_ids"],
                    # Preserve the resolver's episode-initial metadata as
                    # provenance, without presenting it as current visibility.
                    "initial_inventory_coverage": current_inventory["initial_coverage"],
                    "initial_inventory_completeness": current_inventory["initial_completeness"],
                    "initial_missing_class_ids": current_inventory["initial_missing_class_ids"],
                    "allow_partial_inventory": inventory.get("allow_partial_inventory") if isinstance(inventory, Mapping) else None,
                    "tracking_pair": tracking_pair,
                    "object_states": output.get("states") if isinstance(output, Mapping) else None,
                    "acquisition_diagnostics": (output.get("acquisition_diagnostics")
                                                if isinstance(output, Mapping) else None),
                    "observed_triplets": (output.get("observed_triplets")
                                           if isinstance(output, Mapping) else None),
                }
        finally:
            if callable(on_end):
                on_end()

    def run_archive(self, archive: str | Path) -> list[dict]:
        return list(self.iter_archive(archive))

    def run_root(self, frames_root: str | Path) -> list[dict]:
        rows = []
        for archive in discover_archives(frames_root, episodes=self.episodes):
            rows.extend(self.run_archive(archive))
        return rows

    def run_root_to_jsonl(self, frames_root: str | Path, output: str | Path) -> Path:
        """Stream rows to a recoverable partial file, then atomically finalize."""
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial = output_path.with_name(output_path.name + ".partial")
        if self.no_clobber and (output_path.exists() or partial.exists()):
            raise FileExistsError(f"prediction output or partial already exists: {output_path}")
        try:
            with partial.open("x" if self.no_clobber else "w",
                              encoding="utf-8", newline="\n") as handle:
                for archive in discover_archives(frames_root, episodes=self.episodes):
                    for row in self.iter_archive(archive):
                        for cache_key in ("mask_cache", "effective_masks_cache"):
                            cache = row.get(cache_key)
                            if not cache:
                                continue
                            # Keep references portable when the prediction and
                            # mask trees are copied together to another host.
                            cache_path = Path(str(cache)).resolve()
                            output_root = output_path.parent.resolve()
                            if output_root != cache_path and output_root not in cache_path.parents:
                                raise ValueError("mask cache must be under the prediction output root")
                            row[cache_key] = os.path.relpath(str(cache_path), str(output_root))
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                        handle.flush()
            if self.no_clobber and output_path.exists():
                raise FileExistsError(f"prediction output already exists: {output_path}")
            os.replace(partial, output_path)
        except Exception:
            # Leave completed rows on disk for inspection/restart.
            raise
        return output_path


def write_predictions_jsonl(rows: Iterable[Mapping[str, Any]], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".jsonl", mode="w", encoding="utf-8", newline="\n", delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path


def read_predictions_jsonl(path: str | Path) -> dict[tuple[str, str, int], list]:
    result = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["task"]), str(row.get("demo", "demo_0")), int(row["frame"]))
            result.setdefault(key, []).extend(row.get("triplets") or [])
    return result
