import os
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
SCENE_GRAPH_SUBJECT_FILTER = "akita_black_bowl_1"  # set to None to include all subjects

# Default cameras lerobot passes to every LIBERO env (LiberoEnvConfig.camera_name default).
# These are the MuJoCo camera names (no _image suffix).
# TASK_CAMERA_OVERRIDE replaces this list for specific tasks.
DEFAULT_CAMERAS = ["agentview", "robot0_eye_in_hand"]
CAMERA_NAME = "agentview"  # legacy single-camera reference used by the demo
CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256
SETTLE_STEPS_INIT = 100
SETTLE_STEPS_SWAP = 200

# Per-task distractor objects to strip from the BDDL before the env loads.
# Format: { task_id: ["obj_a", "obj_b", ...] }
# UNSAFE combinations (raises ValueError at startup, do not use):
#   Task 3 + "cookies_1"                       (bowl_1 starts On cookies_1)
#   Task 5 + "glazed_rim_porcelain_ramekin_1"  (bowl_1 starts On the ramekin)
# Safe to remove on all other tasks: "cookies_1", "akita_black_bowl_2",
#                                    "glazed_rim_porcelain_ramekin_1"
TASK_REMOVE_CONFIG: dict[int, list[str]] = {
    # UNSAFE combinations (raises ValueError at startup, do not use):
    #   Task 3 + "cookies_1"                       (bowl_1 starts On cookies_1)
    #   Task 5 + "glazed_rim_porcelain_ramekin_1"  (bowl_1 starts On the ramekin)
    4: ["cookies_1", "glazed_rim_porcelain_ramekin_1"],           # bowl_1 in cabinet drawer — clear the table
    8: ["glazed_rim_porcelain_ramekin_1", "cookies_1"],                       # bowl_1 next to plate — remove decoys
}

# Per-task prompt overrides. Reward is independent of the prompt string (it reads
# the BDDL goal condition directly), so any rephrasing is safe as long as the
# bowl→plate intent is preserved.
TASK_PROMPT_OVERRIDE: dict[int, str] = {
    0: "pick up the black bowl in front of the ramekin and place it on the plate",
    7: "pick up the black bowl behind the wooden cabinet and place it on the plate",
}

# Per-task camera override. Value is a comma-separated camera name string, matching
# the lerobot LiberoEnvConfig.camera_name format.
# Available cameras: agentview, robot0_eye_in_hand, frontview, birdview, sideview, robot0_robotview
# WARNING: switching away from training cameras (agentview, robot0_eye_in_hand) puts the model OOD.
TASK_CAMERA_OVERRIDE: dict[int, str] = {
    2: "frontview,robot0_robotview",  # bowl_1 at table center — front + robot-mounted view
    6: "frontview,robot0_robotview",           # bowl_1 next to cookie box — side + top view
}

TASK_SWAP_CONFIG = {  # 4 tasks with pose swaps

    # change the target object
    1: [("akita_black_bowl_1", "akita_black_bowl_2"), ("glazed_rim_porcelain_ramekin_1", "cookies_1"), ("cookies_1", "plate_1"), ("akita_black_bowl_1", "glazed_rim_porcelain_ramekin_1")],
    5: [("cookies_1", "glazed_rim_porcelain_ramekin_1"), ("akita_black_bowl_1", "akita_black_bowl_2"), ("akita_black_bowl_1", (0.0, 0.0, 0.05)), ("plate_1", (-0.05, -0.45, 0.5))],

    # change the other objects
    3: [("akita_black_bowl_2", "plate_1"), ("glazed_rim_porcelain_ramekin_1", "akita_black_bowl_2")],
    9: [("akita_black_bowl_2", "plate_1"), ("cookies_1", "akita_black_bowl_2"), ("akita_black_bowl_2", (0.0, -0.1, 0))],
}

# TASK_SWAP_CONFIG = {
#     0: [("cookies_1", "akita_black_bowl_2"), ("glazed_rim_porcelain_ramekin_1", "plate_1")],
#     1: [("akita_black_bowl_1", "akita_black_bowl_2"), ("glazed_rim_porcelain_ramekin_1", "cookies_1"), ("cookies_1", "plate_1")],
#     2: [("cookies_1", "glazed_rim_porcelain_ramekin_1"), ("akita_black_bowl_2", "plate_1")],
#     3: [("akita_black_bowl_2", "plate_1"), ("glazed_rim_porcelain_ramekin_1", "akita_black_bowl_2")],
#     4: [("plate_1", "akita_black_bowl_2"), ("akita_black_bowl_2", "cookies_1"), ("glazed_rim_porcelain_ramekin_1", "cookies_1")],
#     5: [("cookies_1", "glazed_rim_porcelain_ramekin_1"),("akita_black_bowl_1", "akita_black_bowl_2"), ("akita_black_bowl_1", (0.0, 0.0, 0.05)), ("plate_1", (-0.05, -0.45, 0.5))],
#     6: [("akita_black_bowl_1", "akita_black_bowl_2"), ("glazed_rim_porcelain_ramekin_1", (-0.1, -0.2, 0)), ("glazed_rim_porcelain_ramekin_1", "cookies_1")],
#     7: [("akita_black_bowl_2", "glazed_rim_porcelain_ramekin_1"), ("akita_black_bowl_2", (-0.1, -0.1, 0)), ("plate_1", "cookies_1")],
#     8: [("akita_black_bowl_1", "akita_black_bowl_2"), ("glazed_rim_porcelain_ramekin_1", "cookies_1"), ("cookies_1", "plate_1")],
#     9: [("akita_black_bowl_2", "plate_1"), ("cookies_1", "akita_black_bowl_2"), ("akita_black_bowl_2", (0.0, -0.1, 0))],
# }
