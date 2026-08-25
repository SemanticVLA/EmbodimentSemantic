"""Convert LIBERO-Spatial demonstration HDF5 files into LeRobot-format datasets.

Two variants of the same 500 demonstrations:
    control   - frames as recorded, unmodified.
    treatment - the main (agentview) camera frame has ground-truth scene-graph
                 arrows baked in, subject-filtered to SCENE_GRAPH_SUBJECT_FILTER.
    target_arrow_treatment - the same frame pair, but with exactly one synthetic
                 subject-to-goal arrow (the bowl to the task goal).
                The wrist (eye_in_hand) frame is never modified.

Both variants share identical actions/states/task text/episode boundaries, so the
baked-in arrows are the only difference between the two datasets a policy can train on.

Modes (--mode):
    preview  - render a handful of frames to PNG and exit. No lerobot.datasets
               dependency, no dataset written. Use this first.
    convert  - write a legacy-compatible LeRobotDataset for one --variant.
     convert-pair - write the sealed control+treatment pair in one HDF5 source pass.
     convert-target-arrow-pair - write the distinct one-arrow pair in one HDF5 pass.
     verify   - source-grounded verification of a sealed pair; writes the sentinel
                 required by training launchers that enforce the sealed contract.

Usage:
    python hdf5_to_lerobot_dataset.py --mode preview --tasks 0 3 --demos-per-task 1
    python hdf5_to_lerobot_dataset.py --mode convert-pair
     python hdf5_to_lerobot_dataset.py --mode verify
     python hdf5_to_lerobot_dataset.py --mode convert-target-arrow-pair
     python hdf5_to_lerobot_dataset.py --mode verify-target-arrow
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from config import (
    SCENE_GRAPH_SUBJECT_FILTER,
    TASK_GOAL_OBJECT_CONFIG,
    TASK_NAMES,
    TASK_PROMPT_OVERRIDE,
)
from visual_scene_graph import (
    DEFAULT_GOAL_OBJECT,
    SEALED_LORA_ARROW_HEAD_LENGTH,
    SEALED_LORA_ARROW_WIDTH,
    SEALED_LORA_IMAGE_SIZE,
    SEALED_LORA_VISUAL_CONTRACT,
    VISUAL_GOAL_ARROW_CONDITION,
    drawable_relations,
    draw_scene_graph_arrows,
    select_visual_relations,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT.parent / "vlm_benchmarking" / "data" / "libero_spatial_v5"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "lora_datasets"

VARIANTS = ("control", "treatment")
TARGET_ARROW_VARIANT = "target_arrow_treatment"
FRAME_VARIANTS = (*VARIANTS, TARGET_ARROW_VARIANT)
SEALED_PAIR_MANIFEST_NAME = "sealed_lora_pair_manifest.json"
SEALED_PAIR_SENTINEL_NAME = "sealed_lora_pair_verified.json"
TARGET_ARROW_PAIR_MANIFEST_NAME = "sealed_lora_target_arrow_pair_manifest.json"
TARGET_ARROW_PAIR_SENTINEL_NAME = "sealed_lora_target_arrow_pair_verified.json"
TARGET_ARROW_PAIR_KIND = "sealed_lora_control_target_arrow_treatment"
TARGET_ARROW_VISUAL_CONTRACT = {
    **SEALED_LORA_VISUAL_CONTRACT,
    "name": "sealed_lora_target_arrow_v2",
    "relation_selection": "single_subject_to_task_goal",
    "subject": SCENE_GRAPH_SUBJECT_FILTER,
    "goal_object_default": DEFAULT_GOAL_OBJECT,
}
SEALED_BBOX_SOURCE_SIZE = 128
FULL_EXPERIMENT_TASK_IDS = tuple(sorted(TASK_NAMES))
FULL_EXPERIMENT_DEMOS_PER_TASK = 50
FULL_EXPERIMENT_EPISODES = len(FULL_EXPERIMENT_TASK_IDS) * FULL_EXPERIMENT_DEMOS_PER_TASK

# lerobot's LiberoProcessorStep flips live MuJoCo renders 180deg to match the
# HuggingFaceVLA/libero training-data convention that smolvla_libero expects.
# Our HDF5s store raw (unflipped) renders (verified: matches live env raw render,
# r=0.94; flipped does not, r=-0.25) -- so offline training data must be flipped
# here, once, since lerobot only applies that flip to *live* gym rollouts.
#
# The sealed pair intentionally stores *lossless image tensors* (not videos): video
# encoding can perturb pixels differently across the control and treatment copies,
# invalidating the one-variable LoRA comparison.
FEATURES = {
    "observation.images.image": {
        "dtype": "image",
        "shape": (SEALED_LORA_IMAGE_SIZE, SEALED_LORA_IMAGE_SIZE, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.image2": {
        "dtype": "image",
        "shape": (SEALED_LORA_IMAGE_SIZE, SEALED_LORA_IMAGE_SIZE, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz", "gripper_1", "gripper_2"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["d_x", "d_y", "d_z", "d_rx", "d_ry", "d_rz", "gripper"],
    },
}
FPS = 20  # HDF5 env_args.env_kwargs.control_freq, confirmed identical across all 10 task files.


def hdf5_path_for_task(task_id: int, data_dir: Path) -> Path:
    return data_dir / f"{TASK_NAMES[task_id]}_demo.hdf5"


def _decode_json_blob(dataset: h5py.Dataset) -> list:
    """Read a whole-demo JSON blob stored as a scalar object dataset.

    obs/agentview_bboxes, obs/agentview_scene_graph, obs/agentview_world_coords
    (and their robot0_eye_in_hand_* counterparts) are shape=() scalar object
    datasets holding ONE JSON string for the whole demo -- not per-frame
    indexable arrays. Read with [()], not [i].
    """
    raw = dataset[()]
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return json.loads(text)


def task_text_for(task_id: int, hdf5_file: h5py.File) -> str:
    problem_info = json.loads(hdf5_file["data"].attrs["problem_info"])
    base_text = problem_info["language_instruction"]
    return TASK_PROMPT_OVERRIDE.get(task_id, base_text)


def flip180(frame: np.ndarray) -> np.ndarray:
    """Rotate a HxWxC frame 180 degrees (flip both height and width)."""
    return np.ascontiguousarray(frame[::-1, ::-1])


def resize_rgb_image(frame: np.ndarray, size: int = SEALED_LORA_IMAGE_SIZE) -> np.ndarray:
    """Resize an RGB frame to the sealed square image size, preserving uint8 pixels."""
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise TypeError("frame must be a uint8 numpy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must have shape HxWx3, got {frame.shape}")
    if size <= 0:
        raise ValueError("size must be positive")
    if frame.shape[:2] == (size, size):
        return np.ascontiguousarray(frame.copy())

    # cv2's interpolation is deterministic and keeps the downstream overlay in
    # exactly the same 256x256 coordinate system used by evaluation.
    import cv2

    return np.ascontiguousarray(cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR))


def scale_and_clamp_bboxes(
    bboxes: dict[str, list[int] | tuple[int, int, int, int]],
    *,
    source_size: int = SEALED_BBOX_SOURCE_SIZE,
    target_size: int = SEALED_LORA_IMAGE_SIZE,
) -> dict[str, list[int]]:
    """Map source 128px bbox coordinates into the sealed image coordinate space.

    HDF5 bboxes use a 128px raster convention. Values outside that raster are
    tolerated but clamped before drawing so OpenCV cannot place an arrow beyond
    the image that training/evaluation actually receive.
    """
    if source_size <= 0 or target_size <= 0:
        raise ValueError("source_size and target_size must be positive")
    scale = target_size / source_size
    maximum = target_size - 1
    result: dict[str, list[int]] = {}
    for name, bbox in bboxes.items():
        if len(bbox) != 4:
            raise ValueError(f"bbox for {name!r} must contain four values, got {bbox!r}")
        scaled = [round(float(value) * scale) for value in bbox]
        result[name] = [min(max(value, 0), maximum) for value in scaled]
    return result


def main_image_change_mask(control_image: np.ndarray, treatment_image: np.ndarray) -> np.ndarray:
    """Return the exact spatial mask where a treatment main image differs."""
    control_array = np.asarray(control_image)
    treatment_array = np.asarray(treatment_image)
    # LeRobot returns image features as CHW torch tensors, while build_frame and
    # the provenance mask use HWC numpy arrays. Normalize only at this boundary.
    if control_array.ndim == 3 and control_array.shape[0] == 3 and control_array.shape[-1] != 3:
        control_array = np.moveaxis(control_array, 0, -1)
    if treatment_array.ndim == 3 and treatment_array.shape[0] == 3 and treatment_array.shape[-1] != 3:
        treatment_array = np.moveaxis(treatment_array, 0, -1)
    if control_array.shape != treatment_array.shape:
        raise AssertionError(
            f"main image shape mismatch: {control_array.shape} != {treatment_array.shape}"
        )
    return np.any(control_array != treatment_array, axis=2)


def filter_by_subject(
    relations: list,
    subject: str | None,
) -> list[tuple[str, str, str]]:
    """Keep only relations whose subject matches, mirroring the live eval-time filter.

    Stored scene graphs are the full unfiltered graph (all 7 objects as subjects,
    ~42 relations/frame). libero_live_semantic_context.py achieves the same result
    live by restricting its generation loop to one subject; select_visual_relations()
    does NOT filter by subject for the visual_arrows condition, so this filtering
    step has to happen here explicitly.
    """
    if subject is None:
        return [tuple(r) for r in relations]
    return [tuple(r) for r in relations if r[0] == subject]


def _relations_for_variant(
    *,
    bboxes: dict[str, list[int]],
    relations: list[tuple[str, str, str]],
    variant: str,
    goal_object: str = DEFAULT_GOAL_OBJECT,
) -> list[tuple[str, str, str]]:
    """Select the exact relation set rendered for a training variant."""
    filtered = filter_by_subject(relations, SCENE_GRAPH_SUBJECT_FILTER)
    if variant == TARGET_ARROW_VARIANT:
        # This intentionally synthesizes the task goal edge instead of relying on
        # the geometric relation ontology to contain a bowl→plate edge.  It is the
        # same selector used by live ``visual_goal_arrow`` evaluation.
        return select_visual_relations(
            bboxes,
            filtered,
            condition=VISUAL_GOAL_ARROW_CONDITION,
            subject=SCENE_GRAPH_SUBJECT_FILTER,
            goal_object=goal_object,
        )
    return filtered


def build_frame(
    *,
    agentview_rgb: np.ndarray,
    eye_in_hand_rgb: np.ndarray,
    bboxes: dict[str, list[int]],
    relations: list[tuple[str, str, str]],
    ee_pos: np.ndarray,
    ee_ori: np.ndarray,
    gripper_states: np.ndarray,
    action: np.ndarray,
    task_text: str,
    variant: str,
    goal_object: str = DEFAULT_GOAL_OBJECT,
    image_size: int = SEALED_LORA_IMAGE_SIZE,
) -> dict:
    """Pure function: build one LeRobot-dataset frame dict from raw HDF5 fields.

    No file or simulator I/O -- takes already-decoded per-frame arrays/values.
    Covered directly by unit tests since this is where the flip/state/subject-filter
    logic actually lives.
    """
    if variant not in FRAME_VARIANTS:
        raise ValueError(f"variant must be one of {FRAME_VARIANTS}, got {variant!r}")

    resized_main = resize_rgb_image(agentview_rgb, image_size)
    resized_wrist = resize_rgb_image(eye_in_hand_rgb, image_size)
    scaled_bboxes = scale_and_clamp_bboxes(
        bboxes,
        source_size=SEALED_BBOX_SOURCE_SIZE,
        target_size=image_size,
    )
    image = resized_main
    if variant in ("treatment", TARGET_ARROW_VARIANT):
        selected_relations = _relations_for_variant(
            bboxes=bboxes,
            relations=relations,
            variant=variant,
            goal_object=goal_object,
        )
    else:
        selected_relations = []
    if selected_relations:
        # Draw in the pre-flip training orientation. flip180 below applies equally
        # to arrows and camera pixels, matching the policy's LIBERO convention.
        image = draw_scene_graph_arrows(
            image,
            scaled_bboxes,
            selected_relations,
            line_width=SEALED_LORA_ARROW_WIDTH,
            head_length=SEALED_LORA_ARROW_HEAD_LENGTH,
        )

    state = np.concatenate(
        [np.asarray(ee_pos), np.asarray(ee_ori), np.asarray(gripper_states)]
    ).astype(np.float32)

    return {
        "observation.images.image": flip180(image),
        "observation.images.image2": flip180(resized_wrist),
        "observation.state": state,
        "action": np.asarray(action, dtype=np.float32),
        "task": task_text,
    }


def build_paired_frames(
    *,
    task_text: str,
    treatment_variant: str = "treatment",
    goal_object: str = DEFAULT_GOAL_OBJECT,
    **frame_kwargs: object,
) -> tuple[dict, dict, np.ndarray]:
    """Build and exhaustively assert one sealed control/treatment frame pair.

    The returned mask is in the *stored*, post-flip 256px main-image coordinate
    system. It is the only permitted location of pair differences.
    """
    control = build_frame(**frame_kwargs, task_text=task_text, variant="control")
    treatment = build_frame(
        **frame_kwargs,
        task_text=task_text,
        variant=treatment_variant,
        goal_object=goal_object,
    )
    expected_mask = expected_arrow_mask_from_source_frame(
        frame_kwargs,
        treatment_variant=treatment_variant,
        goal_object=goal_object,
    )
    assert_paired_frame_invariants(
        control,
        treatment,
        expected_arrow_mask=expected_mask,
    )
    return control, treatment, expected_mask


def assert_paired_frame_invariants(
    control: dict,
    treatment: dict,
    *,
    expected_arrow_mask: np.ndarray | None = None,
    frame_label: str = "frame",
) -> np.ndarray:
    """Assert all non-main values are byte-identical and arrow changes are localized."""
    for key in ("task", "action", "observation.state", "observation.images.image2"):
        c_value = control[key]
        t_value = treatment[key]
        equal = c_value == t_value if isinstance(c_value, str) else np.array_equal(c_value, t_value)
        if not equal:
            raise AssertionError(f"{frame_label}: {key} differs between control and treatment")

    actual_mask = main_image_change_mask(
        np.asarray(control["observation.images.image"]),
        np.asarray(treatment["observation.images.image"]),
    )
    if expected_arrow_mask is not None:
        unexpected = int(np.count_nonzero(actual_mask & ~expected_arrow_mask))
        if unexpected:
            raise AssertionError(
                f"{frame_label}: main-image changes are not localized to the expected arrow mask "
                f"(unexpected_pixels={unexpected})"
            )
        if np.any(expected_arrow_mask) and not np.any(actual_mask):
            raise AssertionError(f"{frame_label}: expected arrows but found zero arrow pixels")
    return actual_mask


def expected_drawable_relations(
    frame_kwargs: dict,
    *,
    treatment_variant: str = "treatment",
    goal_object: str = DEFAULT_GOAL_OBJECT,
) -> list[tuple[str, str, str]]:
    """Return the source relations expected to produce a sealed treatment arrow."""
    bboxes = scale_and_clamp_bboxes(frame_kwargs["bboxes"])
    relations = _relations_for_variant(
        bboxes=bboxes,
        relations=frame_kwargs["relations"],
        variant=treatment_variant,
        goal_object=goal_object,
    )
    drawn, _ = drawable_relations(bboxes, relations)
    return drawn


def expected_arrow_mask_from_source_frame(
    frame_kwargs: dict,
    *,
    treatment_variant: str = "treatment",
    goal_object: str = DEFAULT_GOAL_OBJECT,
) -> np.ndarray:
    """Render the canonical arrow footprint independently of source image pixels."""
    bboxes = scale_and_clamp_bboxes(frame_kwargs["bboxes"])
    relations = expected_drawable_relations(
        frame_kwargs,
        treatment_variant=treatment_variant,
        goal_object=goal_object,
    )
    canvas = np.zeros((SEALED_LORA_IMAGE_SIZE, SEALED_LORA_IMAGE_SIZE, 3), dtype=np.uint8)
    rendered = draw_scene_graph_arrows(
        canvas,
        bboxes,
        relations,
        line_width=SEALED_LORA_ARROW_WIDTH,
        head_length=SEALED_LORA_ARROW_HEAD_LENGTH,
    )
    # The stored images are rotated after overlay, so the expected spatial mask
    # must be in exactly that post-flip coordinate system before comparison.
    return np.any(flip180(rendered) != 0, axis=2)


def iter_demo_frames(hdf5_file: h5py.File, demo_key: str):
    """Yield build_frame(...) kwargs (minus variant) for every frame of one demo."""
    demo = hdf5_file["data"][demo_key]
    obs = demo["obs"]

    bboxes_by_frame = _decode_json_blob(obs["agentview_bboxes"])
    scene_graph_by_frame = _decode_json_blob(obs["agentview_scene_graph"])
    n_frames = obs["agentview_rgb"].shape[0]

    for i in range(n_frames):
        yield dict(
            agentview_rgb=obs["agentview_rgb"][i],
            eye_in_hand_rgb=obs["eye_in_hand_rgb"][i],
            bboxes=bboxes_by_frame[i],
            relations=scene_graph_by_frame[i],
            ee_pos=obs["ee_pos"][i],
            ee_ori=obs["ee_ori"][i],
            gripper_states=obs["gripper_states"][i],
            action=demo["actions"][i],
        )


def resolve_task_ids(tasks: list[int] | None) -> list[int]:
    return sorted(TASK_NAMES) if not tasks else list(tasks)


# ---------------------------------------------------------------------------
# preview mode
# ---------------------------------------------------------------------------


def run_preview(args: argparse.Namespace) -> None:
    from PIL import Image

    out_dir = Path(args.preview_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for task_id in resolve_task_ids(args.tasks):
        path = hdf5_path_for_task(task_id, args.data_dir)
        with h5py.File(path, "r") as f:
            task_text = task_text_for(task_id, f)
            for demo_idx in range(args.demos_per_task):
                demo_key = f"demo_{demo_idx}"
                if demo_key not in f["data"]:
                    break
                frames = list(iter_demo_frames(f, demo_key))
                frame_kwargs = frames[0]
                for variant in VARIANTS:
                    built = build_frame(**frame_kwargs, task_text=task_text, variant=variant)
                    img = built["observation.images.image"]
                    Image.fromarray(img).resize((512, 512), Image.NEAREST).save(
                        out_dir / f"task{task_id}_{demo_key}_{variant}_frame0.png"
                    )
                print(f"task {task_id} ({task_text!r}), {demo_key}: wrote control+treatment previews")

    print(f"\nPreview images written to {out_dir}")


# ---------------------------------------------------------------------------
# convert mode
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return a deterministic content digest without trusting filename or mtime."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def manifest_sha256(path: Path) -> str:
    """Digest exact manifest bytes: whitespace edits are meaningful provenance edits."""
    return sha256_file(path)


def dataset_tree_fingerprint(root: Path) -> dict:
    """Hash every regular file in a dataset tree with paths, sizes, and content.

    This is intentionally stronger than a directory-name or metadata check: a
    launcher can recompute it to reject any changed PNG/parquet/metadata byte.
    """
    root = root.resolve()
    if not root.is_dir():
        raise AssertionError(f"dataset root is missing or not a directory: {root}")
    digest = hashlib.sha256()
    file_count = total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise AssertionError(f"dataset tree must not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(len(relative_path).to_bytes(4, "big"))
        digest.update(relative_path)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as file_obj:
            while chunk := file_obj.read(8 * 1024 * 1024):
                digest.update(chunk)
        file_count += 1
        total_bytes += size
    if file_count == 0:
        raise AssertionError(f"dataset tree contains no files: {root}")
    return {
        "algorithm": "sha256-tree-v1",
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _source_identity(path: Path, task_id: int, hdf5_file: h5py.File) -> dict:
    stat = path.stat()
    return {
        "task_id": task_id,
        "task_name": TASK_NAMES[task_id],
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
        "num_demos": int(hdf5_file["data"].attrs["num_demos"]),
        "problem_info": str(hdf5_file["data"].attrs["problem_info"]),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sealed_pair_manifest_path(output_root: Path) -> Path:
    return output_root / SEALED_PAIR_MANIFEST_NAME


def sealed_pair_sentinel_path(output_root: Path) -> Path:
    return output_root / SEALED_PAIR_SENTINEL_NAME


def sealed_target_arrow_pair_manifest_path(output_root: Path) -> Path:
    return output_root / TARGET_ARROW_PAIR_MANIFEST_NAME


def sealed_target_arrow_pair_sentinel_path(output_root: Path) -> Path:
    return output_root / TARGET_ARROW_PAIR_SENTINEL_NAME


def _sealed_profile(target_arrow: bool) -> dict:
    """Return immutable naming/contract details for one sealed dataset profile."""
    if target_arrow:
        return {
            "manifest_name": TARGET_ARROW_PAIR_MANIFEST_NAME,
            "sentinel_name": TARGET_ARROW_PAIR_SENTINEL_NAME,
            "pair_kind": TARGET_ARROW_PAIR_KIND,
            "visual_contract": TARGET_ARROW_VISUAL_CONTRACT,
            "treatment_variant": TARGET_ARROW_VARIANT,
            "variants": ("control", TARGET_ARROW_VARIANT),
        }
    return {
        "manifest_name": SEALED_PAIR_MANIFEST_NAME,
        "sentinel_name": SEALED_PAIR_SENTINEL_NAME,
        "pair_kind": "sealed_lora_control_treatment",
        "visual_contract": SEALED_LORA_VISUAL_CONTRACT,
        "treatment_variant": "treatment",
        "variants": VARIANTS,
    }


def _validate_task_ids(task_ids: list[int]) -> None:
    unknown = sorted(set(task_ids) - set(TASK_NAMES))
    if unknown:
        raise ValueError(f"unknown LIBERO task ids: {unknown}; expected ids {sorted(TASK_NAMES)}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"duplicate task ids are not allowed: {task_ids}")


def _full_experiment_ready(manifest: dict) -> bool:
    task_ids = [int(task_id) for task_id in manifest.get("task_ids", [])]
    task_records = manifest.get("tasks", [])
    recorded_task_ids = [int(record.get("task_id", -1)) for record in task_records]
    return (
        task_ids == list(FULL_EXPERIMENT_TASK_IDS)
        and recorded_task_ids == list(FULL_EXPERIMENT_TASK_IDS)
        and len(task_records) == len(FULL_EXPERIMENT_TASK_IDS)
        and all(len(record.get("demos", [])) == FULL_EXPERIMENT_DEMOS_PER_TASK for record in task_records)
        and int(manifest.get("total_episodes", -1)) == FULL_EXPERIMENT_EPISODES
    )


def _source_snapshot_identity(tasks: list[dict]) -> dict:
    sources = [task["source_identity"] for task in tasks]
    return {
        "algorithm": "sha256-canonical-json-v1",
        "sha256": canonical_json_sha256(sources),
        "source_count": len(sources),
    }


def _create_dataset(*, output_root: Path, variant: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=f"local/libero_spatial_{variant}",
        fps=FPS,
        features=FEATURES,
        root=output_root / variant,
        robot_type="panda",
        use_videos=False,
    )


def run_convert(args: argparse.Namespace) -> None:
    """Legacy-compatible one-variant conversion (not eligible for sealed training)."""
    dataset = _create_dataset(output_root=Path(args.output_root), variant=args.variant)

    task_ids = resolve_task_ids(args.tasks)
    _validate_task_ids(task_ids)
    total_frames = 0
    total_episodes = 0
    for task_id in task_ids:
        path = hdf5_path_for_task(task_id, args.data_dir)
        with h5py.File(path, "r") as f:
            task_text = task_text_for(task_id, f)
            num_demos = int(f["data"].attrs["num_demos"])
            n_demos = min(args.demos_per_task, num_demos) if args.demos_per_task else num_demos

            for demo_idx in range(n_demos):
                demo_key = f"demo_{demo_idx}"
                for frame_kwargs in iter_demo_frames(f, demo_key):
                    frame = build_frame(**frame_kwargs, task_text=task_text, variant=args.variant)
                    dataset.add_frame(frame)
                    total_frames += 1
                dataset.save_episode()
                total_episodes += 1
            print(f"task {task_id} ({task_text!r}): converted {n_demos} demos")

    dataset.finalize()
    print(
        f"\n[{args.variant}] wrote {total_episodes} episodes, {total_frames} frames "
        f"to {Path(args.output_root) / args.variant} (unverified single variant)"
    )


def run_convert_pair(args: argparse.Namespace, *, target_arrow: bool = False) -> None:
    """Convert one sealed control/treatment profile from each HDF5 frame once."""
    output_root = Path(args.output_root)
    profile = _sealed_profile(target_arrow)
    sentinel = output_root / profile["sentinel_name"]
    if sentinel.exists():
        sentinel.unlink()

    task_ids = resolve_task_ids(args.tasks)
    _validate_task_ids(task_ids)
    requested_full = (
        task_ids == list(FULL_EXPERIMENT_TASK_IDS)
        and args.demos_per_task == FULL_EXPERIMENT_DEMOS_PER_TASK
    )
    if not requested_full and not getattr(args, "allow_subset", False):
        raise ValueError(
            "subset conversion is smoke/test-only; pass --allow-subset explicitly. "
            "The launchable sealed dataset requires tasks 0..9 and 50 demos per task."
        )
    control_dataset = _create_dataset(output_root=output_root, variant="control")
    treatment_variant = profile["treatment_variant"]
    treatment_dataset = _create_dataset(output_root=output_root, variant=treatment_variant)
    manifest: dict = {
        "schema_version": 1,
        "pair_kind": profile["pair_kind"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "task_ids": task_ids,
        "visual_contract": profile["visual_contract"],
        "storage_contract": {"image_dtype": "image", "use_videos": False, "fps": FPS},
        "tasks": [],
        "total_episodes": 0,
        "total_frames": 0,
    }

    for task_id in task_ids:
        path = hdf5_path_for_task(task_id, args.data_dir)
        with h5py.File(path, "r") as hdf5_file:
            task_text = task_text_for(task_id, hdf5_file)
            source_identity = _source_identity(path, task_id, hdf5_file)
            available_demos = source_identity["num_demos"]
            n_demos = min(args.demos_per_task, available_demos) if args.demos_per_task else available_demos
            task_record = {
                "task_id": task_id,
                "task_name": TASK_NAMES[task_id],
                "task_text": task_text,
                "source_identity": source_identity,
                "demos": [],
                "frame_count": 0,
            }
            for demo_idx in range(n_demos):
                demo_key = f"demo_{demo_idx}"
                demo_frames = 0
                for frame_kwargs in iter_demo_frames(hdf5_file, demo_key):
                    control, treatment, arrow_mask = build_paired_frames(
                        **frame_kwargs,
                        task_text=task_text,
                        treatment_variant=treatment_variant,
                        goal_object=TASK_GOAL_OBJECT_CONFIG.get(task_id, DEFAULT_GOAL_OBJECT),
                    )
                    # This assertion executes before either writer sees a frame. It
                    # makes accidental non-image divergence fail at its source.
                    assert_paired_frame_invariants(
                        control,
                        treatment,
                        expected_arrow_mask=arrow_mask,
                        frame_label=f"task={task_id} demo={demo_key} frame={demo_frames}",
                    )
                    control_dataset.add_frame(control)
                    treatment_dataset.add_frame(treatment)
                    demo_frames += 1
                control_dataset.save_episode()
                treatment_dataset.save_episode()
                task_record["demos"].append({"demo_key": demo_key, "frame_count": demo_frames})
                task_record["frame_count"] += demo_frames
                manifest["total_episodes"] += 1
                manifest["total_frames"] += demo_frames
            manifest["tasks"].append(task_record)
            print(f"task {task_id} ({task_text!r}): paired {n_demos} demos")

    control_dataset.finalize()
    treatment_dataset.finalize()
    manifest["source_snapshot_identity"] = _source_snapshot_identity(manifest["tasks"])
    manifest["full_experiment_ready"] = _full_experiment_ready(manifest)
    manifest["launch_eligibility"] = (
        "full_experiment_ready" if manifest["full_experiment_ready"] else "subset_smoke_not_launchable"
    )
    if requested_full and not manifest["full_experiment_ready"]:
        raise AssertionError(
            "full conversion did not produce the required 10 tasks × 50 demos = 500 episodes; "
            "the pair remains unlaunchable"
        )
    _write_json(output_root / profile["manifest_name"], manifest)
    print(
        f"\n[sealed pair] wrote {manifest['total_episodes']} episodes and "
        f"{manifest['total_frames']} frames per variant to {output_root}. "
        f"Run --mode {'verify-target-arrow' if target_arrow else 'verify'} to create "
        f"{profile['sentinel_name']}."
    )


# ---------------------------------------------------------------------------
# verify mode
# ---------------------------------------------------------------------------


def _load_sealed_manifest(output_root: Path, *, target_arrow: bool = False) -> dict:
    profile = _sealed_profile(target_arrow)
    path = output_root / profile["manifest_name"]
    if not path.is_file():
        raise AssertionError(
            f"sealed pair manifest is missing: {path}. Run --mode convert-pair first."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("pair_kind") != profile["pair_kind"]:
        raise AssertionError(f"{path}: unexpected pair_kind")
    if manifest.get("visual_contract") != profile["visual_contract"]:
        raise AssertionError(f"{path}: visual contract does not match this converter")
    if manifest.get("storage_contract") != {"image_dtype": "image", "use_videos": False, "fps": FPS}:
        raise AssertionError(f"{path}: storage contract is not lossless image-backed")
    if manifest.get("source_snapshot_identity") != _source_snapshot_identity(manifest.get("tasks", [])):
        raise AssertionError(f"{path}: source snapshot identity does not match task source records")
    full_ready = _full_experiment_ready(manifest)
    if bool(manifest.get("full_experiment_ready")) != full_ready:
        raise AssertionError(f"{path}: full_experiment_ready is inconsistent with task/demo/episode counts")
    expected_eligibility = "full_experiment_ready" if full_ready else "subset_smoke_not_launchable"
    if manifest.get("launch_eligibility") != expected_eligibility:
        raise AssertionError(f"{path}: launch eligibility is inconsistent with full-experiment status")
    return manifest


def validate_verified_pair(
    output_root: Path,
    *,
    require_full_experiment: bool = True,
    target_arrow: bool = False,
) -> dict:
    """Recompute launch-gate fingerprints for a verified sealed pair.

    This helper is intentionally independent of dataset/repository names in the
    sentinel: it resolves the canonical roots, hashes their live trees, and
    binds those bytes to the exact manifest bytes recorded at verification.
    """
    output_root = Path(output_root)
    profile = _sealed_profile(target_arrow)
    sentinel_path = output_root / profile["sentinel_name"]
    if not sentinel_path.is_file():
        raise AssertionError(f"verified sentinel is missing: {sentinel_path}")
    sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    manifest_path = output_root / profile["manifest_name"]
    if sentinel.get("manifest_path") != profile["manifest_name"]:
        raise AssertionError("sentinel does not point to the canonical pair manifest")
    if sentinel.get("manifest_sha256") != manifest_sha256(manifest_path):
        raise AssertionError("pair manifest bytes differ from the verified manifest digest")

    manifest = _load_sealed_manifest(output_root, target_arrow=target_arrow)
    required_fields = {
        "pair_kind": manifest["pair_kind"],
        "visual_contract": manifest["visual_contract"],
        "storage_contract": manifest["storage_contract"],
        "task_ids": manifest["task_ids"],
        "total_episodes": manifest["total_episodes"],
        "total_frames": manifest["total_frames"],
        "source_snapshot_identity": manifest["source_snapshot_identity"],
        "full_experiment_ready": manifest["full_experiment_ready"],
        "launch_eligibility": manifest["launch_eligibility"],
    }
    for field, expected in required_fields.items():
        if sentinel.get(field) != expected:
            raise AssertionError(f"sentinel {field!r} does not match the pair manifest")
    arrow_frames = sentinel.get("arrow_frames")
    if not isinstance(arrow_frames, int) or not 0 <= arrow_frames <= int(manifest["total_frames"]):
        raise AssertionError("sentinel arrow_frames is invalid")

    for variant in profile["variants"]:
        expected = sentinel.get("dataset_fingerprints", {}).get(variant)
        actual = dataset_tree_fingerprint(output_root / variant)
        if expected != actual:
            raise AssertionError(
                f"{variant} dataset tree/content fingerprint differs from the verified pair"
            )

    if require_full_experiment and not manifest["full_experiment_ready"]:
        raise AssertionError(
            "sealed pair is a subset smoke conversion, not a launchable full experiment "
            "(requires tasks 0..9, 50 demos/task, 500 episodes)"
        )
    return sentinel


def _assert_source_identity(expected: dict, path: Path, task_id: int, hdf5_file: h5py.File) -> None:
    actual = _source_identity(path, task_id, hdf5_file)
    if actual != expected:
        raise AssertionError(
            f"task {task_id}: source HDF5 identity changed since conversion; "
            "regenerate the pair rather than verifying against a different source"
        )


def _assert_loaded_frame_matches(
    actual: dict,
    expected: dict,
    *,
    variant: str,
    frame_label: str,
) -> None:
    for key in ("task", "action", "observation.state", "observation.images.image2", "observation.images.image"):
        if key not in actual:
            raise AssertionError(f"{frame_label}: {variant} dataset omitted required field {key!r}")
        actual_value = actual[key]
        expected_value = expected[key]
        if isinstance(expected_value, str):
            equal = actual_value == expected_value
        else:
            actual_array = np.asarray(actual_value)
            expected_array = np.asarray(expected_value)
            if key.startswith("observation.images.") and actual_array.ndim == 3:
                if actual_array.shape[0] == 3 and actual_array.shape[-1] != 3:
                    actual_array = np.moveaxis(actual_array, 0, -1)
            # LeRobot normalizes decoded image observations to float32 [0, 1]
            # in __getitem__. PNG storage itself remains lossless; compare in
            # that documented loader representation without tolerances.
            if key.startswith("observation.images.") and actual_array.dtype.kind == "f":
                expected_array = expected_array.astype(actual_array.dtype) / 255
            equal = np.array_equal(actual_array, expected_array)
        if not equal:
            raise AssertionError(
                f"{frame_label}: {variant} {key} differs from source expectation "
                f"(actual_shape={getattr(actual_array, 'shape', None)}, "
                f"actual_dtype={getattr(actual_array, 'dtype', None)}, "
                f"expected_shape={getattr(expected_array, 'shape', None)}, "
                f"expected_dtype={getattr(expected_array, 'dtype', None)})"
            )


def _assert_episode_expectations(
    control_frame: dict,
    treatment_frame: dict,
    *,
    episode_index: int,
    frame_index: int,
    frame_label: str,
) -> None:
    for key, expected in (("episode_index", episode_index), ("frame_index", frame_index)):
        if key not in control_frame or key not in treatment_frame:
            raise AssertionError(f"{frame_label}: dataset omitted required {key} metadata")
        control_value = int(np.asarray(control_frame[key]).item())
        treatment_value = int(np.asarray(treatment_frame[key]).item())
        if control_value != expected or treatment_value != expected:
            raise AssertionError(
                f"{frame_label}: {key} mismatch: expected {expected}, "
                f"control={control_value}, treatment={treatment_value}"
            )


def run_verify(args: argparse.Namespace, *, target_arrow: bool = False) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_root = Path(args.output_root)
    profile = _sealed_profile(target_arrow)
    # A sentinel is evidence of a specific successful verification pass, never a
    # reusable approval after the datasets or source have changed.
    stale_sentinel = output_root / profile["sentinel_name"]
    if stale_sentinel.exists():
        stale_sentinel.unlink()
    manifest = _load_sealed_manifest(output_root, target_arrow=target_arrow)
    manifest_path = output_root / profile["manifest_name"]
    control_root = output_root / "control"
    treatment_variant = profile["treatment_variant"]
    treatment_root = output_root / treatment_variant

    control = LeRobotDataset("local/libero_spatial_control", root=control_root)
    treatment = LeRobotDataset(f"local/libero_spatial_{treatment_variant}", root=treatment_root)

    expected_total = int(manifest["total_frames"])
    if len(control) != len(treatment) or len(control) != expected_total:
        raise AssertionError(
            f"frame count mismatch: expected={expected_total} control={len(control)} treatment={len(treatment)}"
        )

    checked = arrow_frames = 0
    global_episode = 0
    for task_record in manifest["tasks"]:
        task_id = int(task_record["task_id"])
        path = hdf5_path_for_task(task_id, args.data_dir)
        with h5py.File(path, "r") as hdf5_file:
            _assert_source_identity(task_record["source_identity"], path, task_id, hdf5_file)
            if task_text_for(task_id, hdf5_file) != task_record["task_text"]:
                raise AssertionError(f"task {task_id}: task text no longer matches pair manifest")
            for demo_record in task_record["demos"]:
                demo_key = demo_record["demo_key"]
                source_frames = iter_demo_frames(hdf5_file, demo_key)
                frame_count = 0
                for frame_index, frame_kwargs in enumerate(source_frames):
                    frame_label = f"task={task_id} demo={demo_key} frame={frame_index}"
                    expected_control, expected_treatment, expected_mask = build_paired_frames(
                        **frame_kwargs,
                        task_text=task_record["task_text"],
                        treatment_variant=treatment_variant,
                        goal_object=TASK_GOAL_OBJECT_CONFIG.get(task_id, DEFAULT_GOAL_OBJECT),
                    )
                    c = control[checked]
                    t = treatment[checked]
                    _assert_episode_expectations(
                        c,
                        t,
                        episode_index=global_episode,
                        frame_index=frame_index,
                        frame_label=frame_label,
                    )
                    _assert_loaded_frame_matches(c, expected_control, variant="control", frame_label=frame_label)
                    _assert_loaded_frame_matches(
                        t,
                        expected_treatment,
                        variant=treatment_variant,
                        frame_label=frame_label,
                    )
                    actual_mask = assert_paired_frame_invariants(
                        c,
                        t,
                        expected_arrow_mask=expected_mask,
                        frame_label=frame_label,
                    )
                    if np.any(expected_mask) and not np.any(actual_mask):
                        raise AssertionError(f"{frame_label}: expected arrows but found zero arrow pixels")
                    if not np.any(expected_mask) and np.any(actual_mask):
                        raise AssertionError(f"{frame_label}: unexpected arrows on a no-arrow source frame")
                    arrow_frames += int(np.any(actual_mask))
                    checked += 1
                    frame_count += 1
                if frame_count != int(demo_record["frame_count"]):
                    raise AssertionError(
                        f"task {task_id} {demo_key}: frame count changed: "
                        f"expected {demo_record['frame_count']} got {frame_count}"
                    )
                global_episode += 1

    if checked != expected_total or global_episode != int(manifest["total_episodes"]):
        raise AssertionError(
            f"manifest traversal mismatch: frames={checked}/{expected_total}, "
            f"episodes={global_episode}/{manifest['total_episodes']}"
        )

    sentinel = {
        "schema_version": 2,
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_path": profile["manifest_name"],
        "manifest_sha256": manifest_sha256(manifest_path),
        "pair_kind": manifest["pair_kind"],
        "visual_contract": manifest["visual_contract"],
        "storage_contract": manifest["storage_contract"],
        "task_ids": manifest["task_ids"],
        "source_snapshot_identity": manifest["source_snapshot_identity"],
        "full_experiment_ready": manifest["full_experiment_ready"],
        "launch_eligibility": manifest["launch_eligibility"],
        "total_episodes": global_episode,
        "total_frames": checked,
        "arrow_frames": arrow_frames,
        "dataset_fingerprints": {
            variant: dataset_tree_fingerprint(output_root / variant)
            for variant in profile["variants"]
        },
    }
    _write_json(output_root / profile["sentinel_name"], sentinel)

    print(
        f"verify OK: {checked} source-grounded frames and {global_episode} episodes checked; "
        f"task/action/state/wrist/main expectations match, {arrow_frames} frames contain arrows. "
        f"Wrote training sentinel: {output_root / profile['sentinel_name']}"
    )


def run_preflight(args: argparse.Namespace, *, target_arrow: bool = False) -> None:
    """Recompute the sealed launch gate without trusting filenames or old checks."""
    sentinel = validate_verified_pair(
        Path(args.output_root),
        require_full_experiment=True,
        target_arrow=target_arrow,
    )
    print(
        "preflight OK: full sealed pair is byte-bound to its manifest and both dataset trees "
        f"(control={sentinel['dataset_fingerprints']['control']['sha256']}, "
        f"treatment={sentinel['dataset_fingerprints'][_sealed_profile(target_arrow)['treatment_variant']]['sha256']})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "preview",
            "convert",
            "convert-pair",
            "convert-target-arrow-pair",
            "verify",
            "verify-target-arrow",
            "preflight",
            "preflight-target-arrow",
        ],
        default="convert-pair",
    )
    parser.add_argument("--variant", choices=VARIANTS, help="required for --mode convert")
    parser.add_argument("--tasks", type=int, nargs="*", default=None, help="task ids, default all 10")
    parser.add_argument("--demos-per-task", type=int, default=50)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="allow non-launchable subset conversion for an explicit smoke/test only",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preview-dir", type=Path, default=REPO_ROOT / "hdf5_conversion_previews")
    args = parser.parse_args(argv)

    if args.mode == "convert" and args.variant is None:
        parser.error("--variant is required for --mode convert")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "preview":
        run_preview(args)
    elif args.mode == "convert":
        run_convert(args)
    elif args.mode == "convert-pair":
        run_convert_pair(args)
    elif args.mode == "convert-target-arrow-pair":
        run_convert_pair(args, target_arrow=True)
    elif args.mode == "verify":
        run_verify(args)
    elif args.mode == "verify-target-arrow":
        run_verify(args, target_arrow=True)
    elif args.mode == "preflight":
        run_preflight(args)
    elif args.mode == "preflight-target-arrow":
        run_preflight(args, target_arrow=True)


if __name__ == "__main__":
    main()
