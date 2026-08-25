import os
import re
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swap_outputs")

RANDOMIZE_SCENES = True

BENCHMARK_NAME = "libero_spatial"

TASK_NAMES = {
    0: "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    1: "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    2: "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    3: "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    4: "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    5: "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    6: "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    7: "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    8: "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    9: "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
}

TASK_GOAL_OBJECT_CONFIG: dict[int, str] = {
    0: "plate_1",
    1: "plate_1",
    2: "plate_1",
    3: "plate_1",
    4: "plate_1",
    5: "plate_1",
    6: "plate_1",
    7: "plate_1",
    8: "plate_1",
    9: "plate_1",
}

SCENE_GRAPH_SUBJECT_FILTER = "akita_black_bowl_1"  # set to None to include all subjects

# Default cameras lerobot passes to every LIBERO env (LiberoEnvConfig.camera_name default).
# These are the MuJoCo camera names (no _image suffix).
DEFAULT_CAMERAS = ["agentview", "robot0_eye_in_hand"]
# LeRobot's policy observation keys include the image suffix; these are not
# MuJoCo camera names and are stripped only at OffScreenRenderEnv creation.
LEROBOT_CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256
SETTLE_STEPS_INIT = 100
SETTLE_STEPS_SWAP = 200

# Every task receives exactly one safe distractor removal before the LIBERO env
# is constructed.  Never remove the target bowl, goal plate, or task support.
TASK_REMOVE_CONFIG: dict[int, list[str]] = {
    0: ["cookies_1"],
    1: ["cookies_1"],
    2: ["glazed_rim_porcelain_ramekin_1"],
    3: ["glazed_rim_porcelain_ramekin_1"],
    4: ["cookies_1"],
    5: ["cookies_1"],
    6: ["glazed_rim_porcelain_ramekin_1"],
    7: ["cookies_1"],
    8: ["cookies_1"],
    9: ["glazed_rim_porcelain_ramekin_1"],
}

# Per-task prompt overrides. Reward is independent of the prompt string (it reads
# the BDDL goal condition directly), so any rephrasing is safe as long as the
# bowl→plate intent is preserved.
TASK_PROMPT_OVERRIDE: dict[int, str] = {
    0: "pick up the black bowl in front of the ramekin and place it on the plate",
    1: "pick up the black bowl to the right of the ramekin and place it on the plate",
    2: "pick up the black bowl next to the stove and place it on the plate",
    3: "pick up the black bowl to the right of the wooden cabinet and place it on the plate",
    4: "pick up the black bowl inside the open drawer of the wooden cabinet and place it on the plate",
    5: "pick up the black bowl above the ramekin and place it on the plate",
    6: "pick up the black bowl directly to the left of the cookie box and place it on the plate",
    7: "pick up the black bowl behind the wooden cabinet and place it on the plate",
    8: "pick up the black bowl to the right of the plate and place it on the plate",
    9: "pick up the black bowl behind the stove and place it on the plate",
}

# Distractor-only pose swaps are limited to tasks whose instruction-defining
# landmarks remain unchanged. Tasks 0, 1, 3, 5, and 6 are removal-only.
TASK_SWAP_CONFIG: dict[int, list[tuple[str, str]]] = {
    2: [("akita_black_bowl_2", "cookies_1")],
    4: [("akita_black_bowl_2", "glazed_rim_porcelain_ramekin_1")],
    7: [("akita_black_bowl_2", "glazed_rim_porcelain_ramekin_1")],
    8: [("akita_black_bowl_2", "glazed_rim_porcelain_ramekin_1")],
    9: [("akita_black_bowl_2", "cookies_1")],
}

# Every task uses one explicit prompt paraphrase.  The scene randomizer and the
# HDF5/LeRobot converter both consume this map, so the prompt condition remains
# sealed and identical across visual inspection, training, and evaluation.
RANDOMIZATION_DIMENSIONS = (
    "scene_layout",
    "object_removal",
    "prompt_variant",
)


def task_randomization_dimensions(
    task_id: int,
) -> dict[str, bool]:
    """Return enabled randomization dimensions for a task.

    This describes the configured condition.  Runtime audit records separately
    report which enabled dimensions were realized in an episode.
    """
    task_id = int(task_id)
    return {
        "scene_layout": bool(TASK_SWAP_CONFIG.get(task_id)),
        "object_removal": bool(TASK_REMOVE_CONFIG.get(task_id)),
        "prompt_variant": bool(task_id in TASK_PROMPT_OVERRIDE),
    }


def validate_randomization_config() -> None:
    """Guard the sealed hybrid condition against unsafe or incomplete changes."""
    expected_tasks = set(range(10))
    layout_tasks = {2, 4, 7, 8, 9}
    if set(TASK_PROMPT_OVERRIDE) != expected_tasks:
        raise ValueError("every task must have exactly one prompt override")
    for task_id in sorted(expected_tasks):
        prompt = TASK_PROMPT_OVERRIDE[task_id]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"task {task_id} prompt override must be a non-empty string")
        normalized_prompt = re.sub(r"\s+", " ", prompt.replace("_", " ").strip()).casefold()
        normalized_canonical = re.sub(
            r"\s+", " ", TASK_NAMES[task_id].replace("_", " ").strip()
        ).casefold()
        if normalized_prompt == normalized_canonical:
            raise ValueError(f"task {task_id} prompt override must differ from canonical task text")
    if set(TASK_REMOVE_CONFIG) != expected_tasks or set(TASK_SWAP_CONFIG) != layout_tasks:
        raise ValueError(
            "every task must have one removal; only tasks 2, 4, 7, 8, and 9 may use layout"
        )
    forbidden = {"akita_black_bowl_1", "plate_1"}
    removal_forbidden = forbidden | {"akita_black_bowl_2"}
    unsafe_support_removals = {3: "cookies_1", 5: "glazed_rim_porcelain_ramekin_1"}
    for task_id in expected_tasks:
        removals = TASK_REMOVE_CONFIG[task_id]
        if len(removals) != 1 or removals[0] in removal_forbidden:
            raise ValueError(f"task {task_id} must have exactly one removal")
        if removals[0] == unsafe_support_removals.get(task_id):
            raise ValueError(f"task {task_id} removal would delete the target support")
        if task_id not in TASK_SWAP_CONFIG:
            continue
        swaps = TASK_SWAP_CONFIG[task_id]
        if len(swaps) != 1:
            raise ValueError(f"task {task_id} must have exactly one distractor swap")
        left, right = swaps[0]
        if not isinstance(left, str) or not isinstance(right, str):
            raise ValueError(
                f"task {task_id} swap entries must be string object-name pairs"
            )
        if any(obj in forbidden or obj == removals[0] for obj in (left, right)) or left == right:
            raise ValueError(f"task {task_id} swap targets a semantic, shared, or removed object")


validate_randomization_config()
