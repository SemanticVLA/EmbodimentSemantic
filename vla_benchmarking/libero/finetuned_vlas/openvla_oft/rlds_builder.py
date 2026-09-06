"""Import-safe RLDS/TFDS source writer and registered builder for OpenVLA-OFT.

The upstream trainer consumes an Open X-Embodiment-style RLDS dataset.  A
plain TFRecord stream is not sufficient because it has no episode/step
structure or TFDS feature contract.  This module keeps serialization usable in
minimal environments and imports TensorFlow Datasets only when building.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import OPENVLA_DATASET_NAME, OPENVLA_IO
from .dataset import (
    NOOP_FILTER_NAME,
    NOOP_FILTER_THRESHOLD,
    _lineage_digest,
    filter_noop_transitions,
    noop_filter_receipt_path,
)
from .dataset import serialize_transition


def _image_hwc(value: Any, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.shape == (3, 256, 256):
        image = np.transpose(image, (1, 2, 0))
    if image.shape != (256, 256, 3):
        raise ValueError(f"{key} must be 256x256 RGB, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _encoded_image(value: Any, key: str) -> dict[str, Any]:
    image = _image_hwc(value, key)
    return {
        "dtype": str(image.dtype),
        "shape": list(image.shape),
        "data_b64": base64.b64encode(image.tobytes(order="C")).decode("ascii"),
    }


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rlds_source_jsonl(frames: Iterable[Mapping[str, Any]], output: str | Path) -> Path:
    """Write deterministic, lossless source records for the TFDS builder.

    The JSONL source is not itself advertised as the trainer dataset.  It is a
    reproducible intermediate consumed by :func:`build_tfds_dataset`, and it
    retains image bytes plus every canonical scalar field needed to reconstruct
    the RLDS episode.
    """
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("cannot write RLDS source from zero frames")
    source_count, source_sha256 = _lineage_digest(frame_list)
    source_episodes = len({str(frame["episode_id"]) for frame in frame_list})
    # This is the production boundary: the generated *_no_noops source cannot
    # be written without the exact pinned upstream filter and a receipt.
    frame_list = filter_noop_transitions(frame_list, threshold=NOOP_FILTER_THRESHOLD)
    output_count, output_sha256 = _lineage_digest(frame_list)
    records: list[dict[str, Any]] = []
    for frame in frame_list:
        required = {"episode_id", "step_index", "episode_length"}
        if not required.issubset(frame):
            raise ValueError("RLDS source requires episode_id, step_index, and episode_length")
        episode_id = str(frame["episode_id"])
        step_index = int(frame["step_index"])
        episode_length = int(frame["episode_length"])
        # Validation includes no-arrow markers, shape, finite state/action, and
        # all-field provenance hashing before bytes are serialized.
        contract = serialize_transition(
            frame, episode_id=episode_id, step_index=step_index, episode_length=episode_length
        )
        records.append(
            {
                "episode_id": episode_id,
                "step_index": step_index,
                "episode_length": episode_length,
                "image": _encoded_image(frame[OPENVLA_IO.camera_keys[0]], "image"),
                "wrist_image": _encoded_image(frame[OPENVLA_IO.camera_keys[1]], "wrist_image"),
                "state": np.asarray(frame["observation.state"], dtype=np.float32).tolist(),
                "action": np.asarray(frame["action"], dtype=np.float32).tolist(),
                "language_instruction": str(frame["task"]),
                "is_first": bool(contract["is_first"]),
                "is_last": bool(contract["is_last"]),
                "is_terminal": bool(contract["is_terminal"]),
                "native_noop_filter_applied": bool(frame.get("native_noop_filter_applied", False)),
                "fields_sha256": contract["fields_sha256"],
            }
        )
    records.sort(key=lambda item: (item["episode_id"], item["step_index"]))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    receipt = {
        "schema_version": 1,
        "filter": NOOP_FILTER_NAME,
        "threshold": NOOP_FILTER_THRESHOLD,
        "dataset_name": OPENVLA_DATASET_NAME,
        "source_frames": source_count,
        "output_frames": output_count,
        "dropped_frames": source_count - output_count,
        "source_sha256": source_sha256,
        "output_sha256": _file_sha256(output_path),
        "output_lineage_sha256": output_sha256,
        "source_episodes": source_episodes,
        "output_episodes": len({str(frame["episode_id"]) for frame in frame_list}),
    }
    receipt_path = noop_filter_receipt_path(output_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def make_builder_class(tfds: Any) -> type:
    """Create the TFDS builder class against an injected TFDS module.

    Injection keeps package imports and unit tests independent of TensorFlow.
    """
    class LiberoSpatialNoNoops(tfds.core.GeneratorBasedBuilder):
        # Explicitly pin the registered TFDS name.  This is the name already
        # supported by the pinned OpenVLA-OFT OXE registry and its
        # libero_dataset_transform; a new local name would be rejected by
        # RLDSDataset before it ever reaches the data directory.
        name = OPENVLA_DATASET_NAME
        VERSION = tfds.core.Version("1.0.0")

        def __init__(self, *, source_path: str | Path | None = None, **kwargs: Any) -> None:
            self._source_path = Path(source_path).expanduser() if source_path is not None else None
            super().__init__(**kwargs)

        def _info(self) -> Any:
            image = tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8)
            return tfds.core.DatasetInfo(
                builder=self,
                description="LIBERO spatial no-arrow trajectories for OpenVLA-OFT.",
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": {
                                    "image": image,
                                    "wrist_image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8),
                                    "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                                },
                                "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                "language_instruction": tfds.features.Text(),
                                "is_first": tfds.features.Tensor(shape=(), dtype=np.bool_),
                                "is_last": tfds.features.Tensor(shape=(), dtype=np.bool_),
                                "is_terminal": tfds.features.Tensor(shape=(), dtype=np.bool_),
                            }
                        ),
                        "episode_metadata": tfds.features.FeaturesDict({"episode_id": tfds.features.Text()}),
                    }
                ),
                supervised_keys=None,
            )

        def _split_generators(self, dl_manager: Any) -> list[Any]:
            if self._source_path is None:
                raise ValueError("source_path is required to build the local RLDS dataset")
            split = getattr(tfds, "Split", None)
            train_name = getattr(split, "TRAIN", "train")
            return [
                tfds.core.SplitGenerator(
                    name=train_name,
                    gen_kwargs={"source_path": self._source_path},
                )
            ]

        def _generate_examples(self, source_path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
            episodes: dict[str, list[dict[str, Any]]] = {}
            with source_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid RLDS source JSON at line {line_number}") from exc
                    episodes.setdefault(str(record["episode_id"]), []).append(record)
            for episode_id in sorted(episodes):
                records = sorted(episodes[episode_id], key=lambda item: int(item["step_index"]))
                steps = []
                for record in records:
                    def decode_image(encoded: Mapping[str, Any]) -> np.ndarray:
                        raw = base64.b64decode(str(encoded["data_b64"]))
                        array = np.frombuffer(raw, dtype=np.dtype(encoded["dtype"]))
                        return array.reshape(tuple(int(v) for v in encoded["shape"]))

                    steps.append(
                        {
                            "observation": {
                                "image": decode_image(record["image"]),
                                "wrist_image": decode_image(record["wrist_image"]),
                                "state": np.asarray(record["state"], dtype=np.float32),
                            },
                            "action": np.asarray(record["action"], dtype=np.float32),
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

    return LiberoSpatialNoNoops


def register_tfds_builder(tfds: Any | None = None) -> type:
    """Register and return the pinned-fork-compatible local TFDS builder.

    The upstream ``RLDSDataset`` calls ``tfds.builder(name, data_dir=...)`` in
    the training process.  Materializing the dataset in a separate process is
    not enough: the builder must be imported into TFDS's registry before that
    lookup.  The training launcher calls this hook immediately before running
    the pinned ``vla-scripts/finetune.py``.

    ``tfds`` is injectable so a dependency-light test can verify registration
    and name selection without TensorFlow or a dataset download.
    """
    if tfds is None:
        try:
            import tensorflow_datasets as tfds  # type: ignore
        except ImportError as exc:
            raise RuntimeError("tensorflow-datasets is required to register the OpenVLA-OFT RLDS builder") from exc

    # Avoid a duplicate-registration error when a caller imports this hook
    # more than once in the same process (for example, an interactive run).
    registered = getattr(getattr(tfds, "core", None), "registered", None)
    if registered is not None:
        try:
            if OPENVLA_DATASET_NAME in registered.list_imported_builders():
                return registered.imported_builder_cls(OPENVLA_DATASET_NAME)
        except (AttributeError, TypeError):
            # Older TFDS releases may not expose the registry helpers.  The
            # normal class registration below remains compatible with them.
            pass
    return make_builder_class(tfds)


def build_tfds_dataset(source_jsonl: str | Path, output_dir: str | Path) -> Path:
    """Materialize the source as a TFDS RLDS dataset or fail closed."""
    source = Path(source_jsonl).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"RLDS source JSONL does not exist: {source}")
    receipt_path = noop_filter_receipt_path(source)
    if not receipt_path.is_file():
        raise ValueError(
            f"refusing to materialize {OPENVLA_DATASET_NAME}: missing verified no-op filter receipt {receipt_path}"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid no-op filter receipt: {receipt_path}") from exc
    if receipt.get("dataset_name") != OPENVLA_DATASET_NAME or receipt.get("filter") != NOOP_FILTER_NAME:
        raise ValueError("RLDS source receipt does not prove the pinned OpenVLA-OFT no-op filter")
    if float(receipt.get("threshold", -1)) != NOOP_FILTER_THRESHOLD:
        raise ValueError("RLDS source receipt has an unexpected no-op threshold")
    if not all(isinstance(receipt.get(key), str) and len(receipt[key]) == 64 for key in ("source_sha256", "output_sha256")):
        raise ValueError("RLDS source receipt is missing source/output hashes")
    source_count = int(receipt.get("source_frames", -1))
    output_count = int(receipt.get("output_frames", -1))
    if source_count < output_count or int(receipt.get("dropped_frames", -1)) != source_count - output_count:
        raise ValueError("RLDS source receipt has inconsistent before/after counts")
    output_sha256 = _file_sha256(source)
    if receipt.get("output_sha256") != output_sha256:
        raise ValueError("RLDS source changed after its no-op filter receipt was written")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or int(receipt.get("output_frames", -1)) != len(records):
        raise ValueError("RLDS source receipt frame count does not match the source")
    if not all(bool(record.get("native_noop_filter_applied")) for record in records):
        raise ValueError("RLDS source contains records without verified no-op filtering")
    destination = Path(output_dir).expanduser().resolve()
    try:
        import tensorflow_datasets as tfds  # type: ignore
    except ImportError as exc:
        raise RuntimeError("tensorflow-datasets is required to materialize the RLDS builder") from exc
    builder_cls = register_tfds_builder(tfds)
    builder = builder_cls(source_path=source, data_dir=str(destination))
    builder.download_and_prepare()
    return destination
