import os
import sys

# Must be set before any robosuite/mujoco import — robosuite defaults to "egl" which is Linux-only
os.environ.setdefault("MUJOCO_GL", "wgl")

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_libero_root = os.path.join(os.path.dirname(_repo_root), "LIBERO")

# Prefer a project-local LIBERO config only when one actually exists. Otherwise
# keep the user's configured ~/.libero paths, which point at the local clone.
_project_libero_config = os.path.join(_repo_root, ".libero")
if os.path.isdir(_project_libero_config):
    os.environ["LIBERO_CONFIG_PATH"] = _project_libero_config

# Insert at 0 so our LIBERO clone wins over any other libero on sys.path/PYTHONPATH
sys.path.insert(0, _repo_root)
sys.path.insert(0, _libero_root)

from radomize_scenes import (
    discover_objects,
    get_object_pose,
    set_object_pose,
    swap_objects,
)
from bddl_utils import (
    extract_joint_schema,
    make_filtered_bddl,
    project_init_states_by_joint_name,
)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from lerobot.envs.libero import get_libero_dummy_action, get_task_init_states
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from config import (
    BENCHMARK_NAME, CAMERA_HEIGHT, CAMERA_WIDTH,
    SETTLE_STEPS_SWAP, TASK_SWAP_CONFIG,
    TASK_NAMES, OUTPUT_DIR, DEFAULT_CAMERAS,
    TASK_REMOVE_CONFIG, TASK_PROMPT_OVERRIDE,
)


def _render(env, cameras: list[str]) -> list:
    """Render all cameras and return list of (camera_name, image) tuples."""
    return [
        (cam, env.sim.render(camera_name=cam, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)[::-1])
        for cam in cameras
    ]


def _make_env(bddl_path: str, cameras: list[str], seed: int) -> OffScreenRenderEnv:
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_names=cameras,
        camera_heights=CAMERA_HEIGHT,
        camera_widths=CAMERA_WIDTH,
    )
    env.seed(seed)
    env.reset()
    return env


def _set_first_frame_state(env: OffScreenRenderEnv, state: np.ndarray) -> None:
    """Match LeRobot's first-frame reset behavior for a stored LIBERO state."""
    env.set_init_state(state)
    for _ in range(10):
        env.step(get_libero_dummy_action())


def _snapshot_protected(env: OffScreenRenderEnv) -> dict[str, np.ndarray]:
    snapshot = {}
    for label in ("akita_black_bowl_1", "plate_1"):
        pose = get_object_pose(env, label)
        if pose is None:
            raise RuntimeError(f"protected object missing from demo scene: {label}")
        snapshot[label] = pose
    return snapshot


def _restore_and_verify_protected(
    env: OffScreenRenderEnv,
    snapshot: dict[str, np.ndarray],
) -> None:
    for label, expected in snapshot.items():
        if not set_object_pose(env, label, expected):
            raise RuntimeError(f"could not restore protected object in demo: {label}")
    env.sim.forward()
    for label, expected in snapshot.items():
        actual = get_object_pose(env, label)
        if actual is None or not np.allclose(actual, expected, atol=1e-7, rtol=0):
            raise RuntimeError(f"protected object moved during demo randomization: {label}")


def run_task_demo(task_id: int, output_dir: str = OUTPUT_DIR):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[BENCHMARK_NAME]()
    task = task_suite.get_task(task_id)

    base_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    remove_objects = TASK_REMOVE_CONFIG.get(task_id, [])
    cameras = DEFAULT_CAMERAS
    swaps = TASK_SWAP_CONFIG.get(task_id, [])
    canonical_state = np.asarray(get_task_init_states(task_suite, task_id)[0]).copy()

    # --- Use the exact same stored first state on both sides. ---
    if remove_objects:
        filtered_bddl = make_filtered_bddl(base_bddl, remove_objects)
        print(f"  [REMOVE] Removed {remove_objects} from BDDL")
    else:
        filtered_bddl = base_bddl

    render_seed = 1000 + task_id
    env_before = _make_env(base_bddl, cameras, render_seed)
    try:
        source_schema = extract_joint_schema(env_before.sim.model)
        _set_first_frame_state(env_before, canonical_state)
        canonical_protected = _snapshot_protected(env_before)
        frames_before = _render(env_before, cameras)
    finally:
        # MuJoCo's offscreen renderer is process-global in this runtime. Render
        # and close the canonical model before constructing the filtered model,
        # otherwise one context can display the other model's scene.
        env_before.close()

    env_after = _make_env(filtered_bddl, cameras, render_seed)
    try:
        target_schema = extract_joint_schema(env_after.sim.model)
        filtered_state = project_init_states_by_joint_name(
            canonical_state,
            source_schema,
            target_schema,
        )
        _set_first_frame_state(env_after, filtered_state)
        removed_still_present = [
            label for label in remove_objects if label in discover_objects(env_after)
        ]
        if removed_still_present:
            raise RuntimeError(f"configured removals still visible: {removed_still_present}")

        protected = _snapshot_protected(env_after)
        for label, expected in canonical_protected.items():
            if not np.allclose(protected[label], expected, atol=1e-7, rtol=0):
                raise RuntimeError(f"protected object changed before task {task_id} layout: {label}")
        for obj_a, obj_b in swaps:
            result = swap_objects(
                env_after,
                obj_a,
                obj_b,
                settle_steps=SETTLE_STEPS_SWAP,
                verbose=False,
            )
            if result.get("skipped") or sorted(result.get("applied", [])) != sorted((obj_a, obj_b)):
                raise RuntimeError(f"task {task_id} swap failed: {result}")
        _restore_and_verify_protected(env_after, protected)
        frames_after = _render(env_after, cameras)
    finally:
        env_after.close()

    # --- Prompt ---
    base_prompt = TASK_NAMES.get(task_id, f"task_{task_id}")
    final_prompt = TASK_PROMPT_OVERRIDE.get(task_id, base_prompt)
    prompt_changed = task_id in TASK_PROMPT_OVERRIDE
    if prompt_changed:
        print(f"  [PROMPT] Override: {final_prompt!r}")
    else:
        print(f"  [PROMPT] Default:  {final_prompt!r}")

    # --- Build figure ---
    # Before: one row per DEFAULT_CAMERAS (agentview + robot0_eye_in_hand).
    # Both sides use the fixed training camera pair.
    n_before = len(frames_before)
    n_after  = len(frames_after)
    n_rows   = max(n_before, n_after)
    has_prompt_banner = prompt_changed or bool(remove_objects)

    fig = plt.figure(figsize=(10, n_rows * 4 + (1.2 if has_prompt_banner else 0)))
    gs = gridspec.GridSpec(
        n_rows + (1 if has_prompt_banner else 0),
        2,
        height_ratios=([0.3] + [1] * n_rows if has_prompt_banner else [1] * n_rows),
        hspace=0.4,
    )

    row_offset = 0
    if has_prompt_banner:
        ax_text = fig.add_subplot(gs[0, :])
        ax_text.axis("off")
        info_lines = []
        if remove_objects:
            info_lines.append(f"REMOVED: {', '.join(remove_objects)}")
        if swaps:
            info_lines.append(
                "SWAPPED: " + ", ".join(f"{left} ↔ {right}" for left, right in swaps)
            )
        if prompt_changed:
            info_lines.append(f"PROMPT: {final_prompt}")
        ax_text.text(0.01, 0.5, "\n".join(info_lines), transform=ax_text.transAxes,
                     fontsize=8, va="center", wrap=True,
                     bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#ffc107"))
        row_offset = 1

    before_label = "Before" + (" Swap" if swaps else "") + (" / Remove" if remove_objects else "")
    after_label  = "After"  + (" Swap" if swaps else "") + (" / Remove" if remove_objects else "")

    for row_idx in range(n_rows):
        row = row_idx + row_offset

        if row_idx < n_before:
            cam_name_b, img_b = frames_before[row_idx]
            ax_b = fig.add_subplot(gs[row, 0])
            ax_b.imshow(img_b)
            ax_b.set_title(f"{before_label}\n[{cam_name_b}]", fontsize=8)
            ax_b.axis("off")

        if row_idx < n_after:
            cam_name_a, img_a = frames_after[row_idx]
            ax_a = fig.add_subplot(gs[row, 1])
            ax_a.imshow(img_a)
            ax_a.set_title(f"{after_label}\n[{cam_name_a}]", fontsize=8)
            ax_a.axis("off")

    task_name = TASK_NAMES.get(task_id, f"task_{task_id}")
    fig.suptitle(f"Task {task_id}: {task_name}", fontsize=9, y=1.01)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"task_{task_id:02d}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    import argparse

    all_tasks = sorted(set(
        list(TASK_SWAP_CONFIG.keys())
        + list(TASK_REMOVE_CONFIG.keys())
        + list(TASK_PROMPT_OVERRIDE.keys())
    ))
    if not all_tasks:
        all_tasks = list(range(10))

    parser = argparse.ArgumentParser(description="Visualize all LIBERO randomizations for a task.")
    parser.add_argument(
        "tasks",
        nargs="*",
        type=int,
        metavar="TASK_ID",
        help=f"Task IDs to run (default: all configured). Configured: {all_tasks}",
    )
    args = parser.parse_args()

    task_ids = args.tasks if args.tasks else all_tasks

    for t_id in task_ids:
        print(f"\n--- Task {t_id} ---")
        run_task_demo(t_id)
