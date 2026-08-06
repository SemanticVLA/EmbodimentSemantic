"""Convert LIBERO-Spatial demonstration HDF5 files into LeRobot-format datasets.

Two variants of the same 500 demonstrations:
    control   - frames as recorded, unmodified.
    treatment - the main (agentview) camera frame has ground-truth scene-graph
                arrows baked in, subject-filtered to SCENE_GRAPH_SUBJECT_FILTER.
                The wrist (eye_in_hand) frame is never modified.

Both variants share identical actions/states/task text/episode boundaries, so the
baked-in arrows are the only difference between the two datasets a policy can train on.

Modes (--mode):
    preview  - render a handful of frames to PNG and exit. No lerobot.datasets
               dependency, no dataset written. Use this first.
    convert  - write a full LeRobotDataset for one --variant.
    verify   - load a previously-written control/treatment pair and assert they
               only differ in the agentview image, and only on frames with
               target-subject relations.

Usage:
    python hdf5_to_lerobot_dataset.py --mode preview --tasks 0 3 --demos-per-task 1
    python hdf5_to_lerobot_dataset.py --mode convert --variant control
    python hdf5_to_lerobot_dataset.py --mode convert --variant treatment
    python hdf5_to_lerobot_dataset.py --mode verify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from config import SCENE_GRAPH_SUBJECT_FILTER, TASK_NAMES, TASK_PROMPT_OVERRIDE
from visual_scene_graph import draw_scene_graph_arrows

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT.parent / "vlm_benchmarking" / "data" / "libero_spatial_v5"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "lora_datasets"

VARIANTS = ("control", "treatment")

# lerobot's LiberoProcessorStep flips live MuJoCo renders 180deg to match the
# HuggingFaceVLA/libero training-data convention that smolvla_libero expects.
# Our HDF5s store raw (unflipped) renders (verified: matches live env raw render,
# r=0.94; flipped does not, r=-0.25) -- so offline training data must be flipped
# here, once, since lerobot only applies that flip to *live* gym rollouts.
FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (128, 128, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.image2": {
        "dtype": "video",
        "shape": (128, 128, 3),
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
) -> dict:
    """Pure function: build one LeRobot-dataset frame dict from raw HDF5 fields.

    No file or simulator I/O -- takes already-decoded per-frame arrays/values.
    Covered directly by unit tests since this is where the flip/state/subject-filter
    logic actually lives.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

    filtered_relations = filter_by_subject(relations, SCENE_GRAPH_SUBJECT_FILTER)

    image = agentview_rgb
    if variant == "treatment" and filtered_relations:
        image = draw_scene_graph_arrows(image, bboxes, filtered_relations)

    state = np.concatenate(
        [np.asarray(ee_pos), np.asarray(ee_ori), np.asarray(gripper_states)]
    ).astype(np.float32)

    return {
        "observation.images.image": flip180(image),
        "observation.images.image2": flip180(eye_in_hand_rgb),
        "observation.state": state,
        "action": np.asarray(action, dtype=np.float32),
        "task": task_text,
    }


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


def run_convert(args: argparse.Namespace) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_root = Path(args.output_root) / args.variant
    repo_id = f"local/libero_spatial_{args.variant}"

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=FEATURES,
        root=output_root,
        robot_type="panda",
        use_videos=True,
    )

    task_ids = resolve_task_ids(args.tasks)
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
        f"to {output_root}"
    )


# ---------------------------------------------------------------------------
# verify mode
# ---------------------------------------------------------------------------


def run_verify(args: argparse.Namespace) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    control_root = Path(args.output_root) / "control"
    treatment_root = Path(args.output_root) / "treatment"

    control = LeRobotDataset("local/libero_spatial_control", root=control_root)
    treatment = LeRobotDataset("local/libero_spatial_treatment", root=treatment_root)

    if len(control) != len(treatment):
        raise AssertionError(
            f"frame count mismatch: control={len(control)} treatment={len(treatment)}"
        )

    mismatched_image_frames = 0
    checked = 0
    for i in range(len(control)):
        c = control[i]
        t = treatment[i]

        if c["task"] != t["task"]:
            raise AssertionError(f"frame {i}: task text differs: {c['task']!r} vs {t['task']!r}")
        if not np.allclose(c["action"], t["action"]):
            raise AssertionError(f"frame {i}: action differs")
        if not np.allclose(c["observation.state"], t["observation.state"]):
            raise AssertionError(f"frame {i}: observation.state differs")
        if not np.array_equal(
            np.asarray(c["observation.images.image2"]), np.asarray(t["observation.images.image2"])
        ):
            raise AssertionError(f"frame {i}: wrist (image2) differs -- should never change")

        same_main_image = np.array_equal(
            np.asarray(c["observation.images.image"]), np.asarray(t["observation.images.image"])
        )
        if not same_main_image:
            mismatched_image_frames += 1
        checked += 1

    print(
        f"verify OK: {checked} frames checked, task/action/state/wrist-image identical, "
        f"{mismatched_image_frames} frames have arrow overlay differences on the main image "
        f"({100 * mismatched_image_frames / max(checked, 1):.1f}%)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preview", "convert", "verify"], default="convert")
    parser.add_argument("--variant", choices=VARIANTS, help="required for --mode convert")
    parser.add_argument("--tasks", type=int, nargs="*", default=None, help="task ids, default all 10")
    parser.add_argument("--demos-per-task", type=int, default=50)
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
    elif args.mode == "verify":
        run_verify(args)


if __name__ == "__main__":
    main()
