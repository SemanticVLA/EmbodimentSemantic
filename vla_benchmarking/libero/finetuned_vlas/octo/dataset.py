"""Octo native RLDS serialization boundary.

This module deliberately emits plain Python/NumPy step dictionaries.  A
compute-node wrapper may feed them to TensorFlow Datasets/RLDS, but source
conversion and lineage checks remain testable without installing TensorFlow.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import base64
import hashlib
import json
from typing import Any

from pathlib import Path

from .contracts import (
    FRAME_ORIENTATION_LIBERO_CANONICAL,
    FRAME_ORIENTATION_STORED_RAW,
    OCTO_IMAGE_KEY,
    OCTO_LANGUAGE_KEY,
    OCTO_ACTION_HORIZON,
    serialize_rlds_step,
)
from .config import OCTO_DATASET_NAME


_BUILDER_CLASS: type | None = None
_TFDS_BUILDER_PATCHED = False


def _image_hwc(value: Any) -> Any:
    """Validate the canonical Octo image shape without importing TensorFlow."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - compute-node boundary
        raise RuntimeError("numpy is required for Octo dataset materialization") from exc
    image = np.asarray(value)
    if image.ndim != 3 or image.shape != (256, 256, 3):
        raise ValueError(f"Octo image_primary must be 256x256 RGB HWC, got {image.shape}")
    if image.dtype != np.uint8:
        if not np.isfinite(image).all():
            raise ValueError("Octo image_primary contains non-finite values")
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _encoded_image(value: Any) -> dict[str, Any]:
    image = _image_hwc(value)
    return {
        "dtype": str(image.dtype),
        "shape": list(image.shape),
        "data_b64": base64.b64encode(image.tobytes(order="C")).decode("ascii"),
    }


def _decode_image(value: Mapping[str, Any]) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - compute-node boundary
        raise RuntimeError("numpy is required for Octo dataset materialization") from exc
    try:
        raw = base64.b64decode(str(value["data_b64"]), validate=True)
        shape = tuple(int(item) for item in value["shape"])
        image = np.frombuffer(raw, dtype=np.dtype(str(value["dtype"]))).reshape(shape)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid encoded Octo image") from exc
    return _image_hwc(image)


def write_rlds_source_jsonl(steps: Iterable[Mapping[str, Any]], output: str | Path) -> Path:
    """Write a lossless, deterministic intermediate consumed by the TFDS builder.

    The JSONL file is intentionally not advertised as the trainer dataset.  It
    carries image bytes and source lineage so the subsequent TFDS/RLDS
    materialization can be regenerated and audited without HDF5 or TensorFlow
    imports in the caller.
    """

    records = list(steps)
    if not records:
        raise ValueError("cannot write an Octo RLDS source from zero steps")
    by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for index, step in enumerate(records):
        if not isinstance(step, Mapping):
            raise ValueError(f"Octo step {index} is not a mapping")
        observation = step.get("observation")
        task = step.get("task")
        if not isinstance(observation, Mapping) or OCTO_IMAGE_KEY not in observation:
            raise ValueError(f"Octo step {index} lacks observation.image_primary")
        if not isinstance(task, Mapping) or not str(task.get("language_instruction", "")).strip():
            raise ValueError(f"Octo step {index} lacks language_instruction")
        episode_id = str(step.get("episode_id", ""))
        if not episode_id:
            raise ValueError(f"Octo step {index} lacks episode_id")
        # Reuse the canonical validator, including arrow-free and action-space
        # checks, before bytes leave the in-memory boundary.
        fingerprint_steps([step])
        by_episode.setdefault(episode_id, []).append(step)

    serialized: list[dict[str, Any]] = []
    for episode_id in sorted(by_episode):
        episode = sorted(by_episode[episode_id], key=lambda item: int(item.get("frame_id", -1)))
        validate_episode_steps(episode)
        expected_frames = list(range(len(episode)))
        observed_frames = [int(item.get("frame_id", -1)) for item in episode]
        if observed_frames != expected_frames:
            raise ValueError(f"Octo episode {episode_id!r} has non-contiguous frame IDs")
        for step in episode:
            serialized.append(
                {
                    "episode_id": episode_id,
                    "frame_id": int(step["frame_id"]),
                    "image": _encoded_image(step["observation"][OCTO_IMAGE_KEY]),
                    "action": [float(value) for value in step["action"]],
                    "language_instruction": str(step["task"][OCTO_LANGUAGE_KEY]),
                    "is_first": bool(step.get("is_first", False)),
                    "is_last": bool(step.get("is_last", False)),
                    "is_terminal": bool(step.get("is_terminal", False)),
                    "frame_provenance": dict(step.get("frame_provenance") or {}),
                    "action_space": str(step.get("action_space", "octo_libero_dataset_v1")),
                }
            )
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in serialized:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return destination


def make_builder_class(tfds: Any, *, dataset_name: str = OCTO_DATASET_NAME) -> type:
    """Create the registered TFDS builder consumed by Octo ``make_single_dataset``.

    The class is constructed only after TensorFlow Datasets is installed.  A
    concrete subclass of ``GeneratorBasedBuilder`` registers itself with TFDS,
    allowing the pinned Octo loader's ``tfds.builder(name, data_dir=...)`` call
    to resolve the materialized dataset in a fresh training process.
    """

    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be non-empty")
    global _BUILDER_CLASS
    if _BUILDER_CLASS is not None and getattr(_BUILDER_CLASS, "name", None) == dataset_name:
        return _BUILDER_CLASS
    if _BUILDER_CLASS is not None and getattr(_BUILDER_CLASS, "name", None) != dataset_name:
        raise RuntimeError("only one Octo TFDS builder may be registered per Python process")

    class OctoLiberoRLDS(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        name = dataset_name

        def __init__(self, *, source_path: str | Path | None = None, **kwargs: Any) -> None:
            self._source_path = Path(source_path).expanduser().resolve() if source_path is not None else None
            super().__init__(**kwargs)

        def _info(self) -> Any:
            import numpy as np

            return tfds.core.DatasetInfo(
                builder=self,
                description="Canonical LIBERO no-arrow trajectories for Octo-Base 1.5.",
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": tfds.features.FeaturesDict(
                                    {
                                        "image_primary": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8),
                                    }
                                ),
                                "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                "language_instruction": tfds.features.Text(),
                                "is_first": tfds.features.Tensor(shape=(), dtype=np.bool_),
                                "is_last": tfds.features.Tensor(shape=(), dtype=np.bool_),
                                "is_terminal": tfds.features.Tensor(shape=(), dtype=np.bool_),
                            }
                        ),
                        "episode_metadata": tfds.features.FeaturesDict(
                            {"episode_id": tfds.features.Text()}
                        ),
                    }
                ),
                supervised_keys=None,
            )

        def _split_generators(self, dl_manager: Any) -> list[Any]:
            del dl_manager
            if self._source_path is None:
                raise ValueError("source_path is required only when building the local Octo RLDS dataset")
            train_name = getattr(getattr(tfds, "Split", None), "TRAIN", "train")
            return [
                tfds.core.SplitGenerator(name=train_name, gen_kwargs={"source_path": self._source_path})
            ]

        def _generate_examples(self, source_path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
            episodes: dict[str, list[dict[str, Any]]] = {}
            with Path(source_path).open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid Octo RLDS source JSON at line {line_number}") from exc
                    episodes.setdefault(str(record["episode_id"]), []).append(record)
            for episode_id in sorted(episodes):
                records = sorted(episodes[episode_id], key=lambda item: int(item["frame_id"]))
                if [int(item["frame_id"]) for item in records] != list(range(len(records))):
                    raise ValueError(f"Octo episode {episode_id!r} has non-contiguous frame IDs")
                steps = []
                for record in records:
                    try:
                        import numpy as np
                        action = np.asarray(record["action"], dtype=np.float32)
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f"invalid Octo action in episode {episode_id!r}") from exc
                    if action.shape != (7,) or not np.isfinite(action).all() or not 0.0 <= float(action[6]) <= 1.0:
                        raise ValueError(f"invalid Octo action in episode {episode_id!r}")
                    steps.append(
                        {
                            "observation": {"image_primary": _decode_image(record["image"])},
                            "action": action,
                            "language_instruction": str(record["language_instruction"]),
                            "is_first": bool(record["is_first"]),
                            "is_last": bool(record["is_last"]),
                            "is_terminal": bool(record["is_terminal"]),
                        }
                    )
                yield episode_id, {
                    "steps": steps,
                    "episode_metadata": {"episode_id": episode_id},
                }

    _BUILDER_CLASS = OctoLiberoRLDS
    return OctoLiberoRLDS


def register_tfds_builder(*, dataset_name: str = OCTO_DATASET_NAME) -> type:
    """Register the Octo builder in the current TFDS process."""

    try:
        import tensorflow_datasets as tfds  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime boundary
        raise RuntimeError("tensorflow-datasets is required to register the Octo RLDS builder") from exc
    builder_cls = make_builder_class(tfds, dataset_name=dataset_name)
    # TFDS only discovers builders shipped inside its package by default.  The
    # Octo trainer calls ``tfds.builder(name, data_dir=...)`` in a fresh
    # process, so install a narrow local dispatch for this checked-in builder.
    # Other dataset names continue through TFDS unchanged.
    global _TFDS_BUILDER_PATCHED
    if not _TFDS_BUILDER_PATCHED:
        original_builder = getattr(tfds, "builder", None)
        if not callable(original_builder):
            raise RuntimeError("tensorflow-datasets does not expose builder()")

        def _octo_builder(name: Any, *args: Any, **kwargs: Any) -> Any:
            if str(name) == str(dataset_name):
                return builder_cls(**kwargs)
            return original_builder(name, *args, **kwargs)

        setattr(tfds, "builder", _octo_builder)
        _TFDS_BUILDER_PATCHED = True
    return builder_cls


def build_tfds_dataset(
    source_jsonl: str | Path,
    output_dir: str | Path,
    *,
    dataset_name: str = OCTO_DATASET_NAME,
) -> Path:
    """Materialize a real episode-structured TFDS/RLDS dataset for Octo."""

    source = Path(source_jsonl).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Octo RLDS source JSONL does not exist: {source}")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    try:
        import tensorflow_datasets as tfds  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime boundary
        raise RuntimeError("tensorflow-datasets is required to materialize the Octo RLDS builder") from exc
    builder_cls = make_builder_class(tfds, dataset_name=dataset_name)
    builder = builder_cls(source_path=source, data_dir=str(destination))
    builder.download_and_prepare()
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    receipt = {
        "schema": "octo_tfds_materialization.v1",
        "dataset_name": dataset_name,
        "dataset_version": "1.0.0",
        "source_jsonl": str(source),
        "source_sha256": source_digest,
        "output_dir": str(destination),
        "rlds": {"episode_key": "steps", "image_key": "observation.image_primary", "language_key": "language_instruction"},
        "frame_orientation": {
            "source": FRAME_ORIENTATION_STORED_RAW,
            "canonical": FRAME_ORIENTATION_LIBERO_CANONICAL,
            "rotation_owner": "octo.dataset.serialize_episode",
        },
        "action_space": "octo_libero_dataset_v1",
        "action_conversion": {
            "source": "libero",
            "target": "octo_dataset",
            "gaussian_dimensions": [0, 1, 2, 3, 4, 5],
            "gripper": "octo_open=(1-libero_gripper)/2",
        },
        "action_normalization_mask": [True, True, True, True, True, True, False],
    }
    (destination / "octo_materialization.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def smoke_test_make_single_dataset(
    data_dir: str | Path,
    *,
    make_single_dataset_fn: Any | None = None,
    dataset_kwargs: Mapping[str, Any] | None = None,
    traj_transform_kwargs: Mapping[str, Any] | None = None,
    frame_transform_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read one batch through the pinned Octo ``make_single_dataset`` API.

    This is deliberately a dependency-light smoke boundary: callers may inject
    the pinned function in tests, while compute nodes import Octo lazily.  It
    verifies that the registered TFDS/RLDS materialization is consumable by the
    exact trainer data path rather than merely existing on disk.
    """

    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Octo dataset root does not exist: {root}")
    if make_single_dataset_fn is None:
        try:
            from octo.data.dataset import make_single_dataset as make_single_dataset_fn
        except ImportError as exc:  # pragma: no cover - compute-node boundary
            raise RuntimeError("pinned Octo make_single_dataset is required for the smoke test") from exc
    if not callable(make_single_dataset_fn):
        raise TypeError("make_single_dataset_fn must be callable")
    kwargs = dict(dataset_kwargs or {})
    kwargs.setdefault("name", OCTO_DATASET_NAME)
    kwargs.setdefault("data_dir", str(root))
    traj_kwargs = dict(traj_transform_kwargs or {"window_size": 1, "action_horizon": OCTO_ACTION_HORIZON})
    frame_kwargs = dict(frame_transform_kwargs or {})
    try:
        dataset = make_single_dataset_fn(
            dataset_kwargs=kwargs,
            traj_transform_kwargs=traj_kwargs,
            frame_transform_kwargs=frame_kwargs,
            train=True,
        )
    except TypeError:
        # Older pinned Octo revisions use positional dictionaries while newer
        # revisions expose the same names as keyword-only arguments.
        dataset = make_single_dataset_fn(kwargs, traj_kwargs, frame_kwargs, train=True)
    try:
        batch = next(iter(dataset))
    except (StopIteration, TypeError) as exc:
        raise ValueError("Octo make_single_dataset returned no readable batch") from exc
    if not isinstance(batch, Mapping):
        raise ValueError("Octo make_single_dataset smoke batch must be a mapping")
    if "observation" not in batch or "action" not in batch:
        raise ValueError("Octo smoke batch lacks observation/action fields")
    return {
        "status": "PASS",
        "dataset_name": str(kwargs["name"]),
        "data_dir": str(root),
        "window_size": int(traj_kwargs.get("window_size", 1)),
        "action_horizon": int(traj_kwargs.get("action_horizon", OCTO_ACTION_HORIZON)),
        "batch_keys": sorted(str(key) for key in batch),
    }


def serialize_canonical_episode(
    frames: Iterable[Any],
    actions: Iterable[Any],
    *,
    instruction: str,
    episode_id: str,
) -> Iterator[dict[str, Any]]:
    """Serialize frames from ``common.libero_hdf5`` without a second flip."""

    return serialize_episode(
        frames,
        actions,
        instruction=instruction,
        episode_id=episode_id,
        rotate_stored_frames=False,
        source_frame_orientation=FRAME_ORIENTATION_LIBERO_CANONICAL,
    )


def _libero_to_octo_dataset_action(action: Any) -> Any:
    """Convert only the gripper convention for Octo dataset storage.

    Octo computes Gaussian statistics for dimensions 0--5 from the stored
    dataset.  Those dimensions therefore remain in the source LIBERO units;
    only the final LIBERO ``[-1, 1]`` gripper command is converted to Octo's
    unscaled open-gripper ``[0, 1]`` convention.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - compute-node boundary
        raise RuntimeError("numpy is required for Octo dataset conversion") from exc
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError(f"LIBERO source action must have shape (7,) and finite values, got {value.shape}")
    converted = value.copy()
    converted[6] = (1.0 - converted[6]) / 2.0
    if not 0.0 <= float(converted[6]) <= 1.0:
        raise ValueError("converted Octo gripper command must be in [0, 1]")
    return converted


def serialize_episode(
    frames: Iterable[Any],
    actions: Iterable[Any],
    *,
    instruction: str,
    episode_id: str,
    rotate_stored_frames: bool = True,
    source_frame_orientation: str = FRAME_ORIENTATION_STORED_RAW,
) -> Iterator[dict[str, Any]]:
    """Yield one Octo step per source frame/action pair with source IDs.

    The final action horizon is a policy-time contract, not a requirement that
    source episodes have a multiple-of-four length.  Padding/chunking belongs
    to the native Octo trainer and is not silently performed here.
    """

    frame_iter, action_iter = iter(frames), iter(actions)
    try:
        frame = next(frame_iter)
        action = next(action_iter)
    except StopIteration as exc:
        raise ValueError("Octo episode cannot be empty") from exc
    frame_index = 0
    while True:
        try:
            next_frame = next(frame_iter)
        except StopIteration:
            next_frame = None
            frame_done = True
        else:
            frame_done = False
        try:
            next_action = next(action_iter)
        except StopIteration:
            next_action = None
            action_done = True
        else:
            action_done = False
        if frame_done != action_done:
            raise ValueError("Octo episode has mismatched frame/action counts")
        final = frame_done
        step = serialize_rlds_step(
            image_primary=frame,
            language_instruction=instruction,
            action=_libero_to_octo_dataset_action(action),
            episode_id=episode_id,
            frame_id=frame_index,
            is_first=frame_index == 0,
            is_last=final,
            is_terminal=final,
            rotate_stored_frame=rotate_stored_frames,
            source_frame_orientation=source_frame_orientation,
        )
        step["action_space"] = "octo_libero_dataset_v1"
        yield step
        if final:
            return
        frame, action = next_frame, next_action
        frame_index += 1


def validate_episode_steps(steps: Iterable[Mapping[str, Any]]) -> int:
    """Validate basic RLDS flags and return the number of serialized steps."""

    records = list(steps)
    if not records:
        raise ValueError("Octo episode cannot be empty")
    if records[0].get("is_first") is not True:
        raise ValueError("first Octo step must set is_first")
    if any(record.get("is_first") is True for record in records[1:]):
        raise ValueError("only the first Octo step may set is_first")
    if any(record.get("is_last") or record.get("is_terminal") for record in records[:-1]):
        raise ValueError("terminal flags may only appear on the final Octo step")
    return len(records)


def _array_bytes(value: Any, *, name: str) -> tuple[bytes, tuple[int, ...], str]:
    """Return deterministic dtype/shape metadata and contiguous bytes."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - compute-node boundary
        raise RuntimeError("numpy is required for Octo dataset hashing") from exc
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    if not array.dtype.isnative:
        array = array.byteswap().newbyteorder()
    return array.tobytes(order="C"), tuple(int(dim) for dim in array.shape), array.dtype.str


def fingerprint_steps(steps: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Hash serialized image/action bytes and return derived dataset counts.

    The digest includes the full image payload, action payload, source lineage,
    language, and terminal flags.  The returned counts are derived from the
    same records, so a manifest cannot claim 500 episodes while hashing a
    different stream.
    """

    digest = hashlib.sha256()
    episode_ids: set[str] = set()
    step_count = 0
    image_bytes = 0
    action_bytes = 0
    terminal_count = 0
    for index, record in enumerate(steps):
        if not isinstance(record, Mapping):
            raise ValueError(f"Octo step {index} is not a mapping")
        for marker in ("arrow_overlay", "visual_arrow", "has_arrows", "arrow_mask", "arrows"):
            value = record.get(marker)
            if value is None:
                value = (record.get("observation") or {}).get(marker)
            if value is not None:
                try:
                    import numpy as np
                    present = bool(np.asarray(value).any())
                except Exception:
                    present = bool(value)
                if present:
                    raise ValueError(f"Octo dataset step {index} contains arrow marker {marker}")
        observation = record.get("observation")
        if not isinstance(observation, Mapping) or "image_primary" not in observation:
            raise ValueError(f"Octo step {index} lacks observation.image_primary")
        if "action" not in record:
            raise ValueError(f"Octo step {index} lacks action")
        image_payload, image_shape, image_dtype = _array_bytes(observation["image_primary"], name="image_primary")
        action_payload, action_shape, action_dtype = _array_bytes(record["action"], name="action")
        if action_shape != (7,):
            raise ValueError(f"Octo dataset action must have shape (7,), got {action_shape}")
        try:
            import numpy as np
            action_value = np.frombuffer(action_payload, dtype=np.dtype(action_dtype)).reshape(action_shape)
            if not 0.0 <= float(action_value[6]) <= 1.0:
                raise ValueError("Octo dataset gripper action must use the [0, 1] open convention")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("unable to validate Octo dataset action bytes") from exc
        image_bytes += len(image_payload)
        action_bytes += len(action_payload)
        episode_id = str(record.get("episode_id", ""))
        if not episode_id:
            raise ValueError(f"Octo step {index} lacks episode_id")
        episode_ids.add(episode_id)
        task_value = record.get("task")
        if isinstance(task_value, Mapping):
            language = str(task_value.get("language_instruction", ""))
        else:
            language = str(task_value or "")
        metadata = {
            "index": index,
            "episode_id": episode_id,
            "frame_id": int(record.get("frame_id", -1)),
            "image_shape": image_shape,
            "image_dtype": image_dtype,
            "action_shape": action_shape,
            "action_dtype": action_dtype,
            "language": language,
            "action_space": str(record.get("action_space", "unspecified")),
            "is_first": bool(record.get("is_first", False)),
            "is_last": bool(record.get("is_last", False)),
            "is_terminal": bool(record.get("is_terminal", False)),
            "frame_provenance": dict(record.get("frame_provenance") or {}),
            "image_nbytes": len(image_payload),
            "action_nbytes": len(action_payload),
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(image_payload).to_bytes(8, "big"))
        digest.update(image_payload)
        digest.update(len(action_payload).to_bytes(8, "big"))
        digest.update(action_payload)
        step_count += 1
        terminal_count += int(bool(record.get("is_terminal", False)))
    if step_count == 0:
        raise ValueError("cannot fingerprint an empty Octo dataset")
    return {
        "schema_version": "octo_dataset_fingerprint.v1",
        "sha256": digest.hexdigest(),
        "episode_count": len(episode_ids),
        "step_count": step_count,
        "action_count": step_count,
        "image_bytes": image_bytes,
        "action_bytes": action_bytes,
        "terminal_count": terminal_count,
    }


def dataset_fingerprint(steps: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Named alias used by dataset launchers and experiment manifests."""

    return fingerprint_steps(steps)
