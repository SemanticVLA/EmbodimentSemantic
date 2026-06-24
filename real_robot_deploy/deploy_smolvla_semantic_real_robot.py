#!/usr/bin/env python3
"""Deploy a trained SmolVLA checkpoint on the real SO101 robot, with optional
live VLM-derived scene-graph context injected into the task prompt.

This is deploy_ee_real_robot.py (lerobot-so101-data) plus a semantic-context
layer: on every control step (or every --context-refresh-interval steps), the
current agent_view frame is sent to a VLM (Gemini by default, same adapters as
vlm_benchmarking/vlm_bench/models/) with the same spatial scene-graph prompt
used for the VLM benchmark, the response is parsed into (subject, relation,
object) triplets, formatted, and appended to the task instruction the policy
sees for its next inference step. --context-mode switches between this and
the plain "standard" prompt with no scene graph.

Usage:
    python deploy_smolvla_semantic_real_robot.py \\
        --policy-path /home/r84368868/lerobot-so101-data/outputs/train/smolvla_so101_multitask/checkpoints/last/pretrained_model \\
        --task "Pick up the black cube on top of the blue box and place it inside the drawer" \\
        --urdf /home/r84368868/lerobot-so101-data/SO101/so101_new_calib.urdf \\
        --context-mode scene_graph \\
        --scene-graph-format triplet \\
        --duration 60

    # No scene graph (plain SmolVLA, same behaviour as deploy_ee_real_robot.py):
    python deploy_smolvla_semantic_real_robot.py \\
        --policy-path .../pretrained_model --task "..." --urdf .../so101_new_calib.urdf --context-mode standard

Note: --urdf has no default -- it must point at the same so101_new_calib.urdf
your data-collection scripts use, and that path is inherently machine-specific.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBODIMENT_SEMANTIC_DIR = os.path.dirname(ROOT_DIR)
VLM_BENCHMARKING_DIR = os.path.join(EMBODIMENT_SEMANTIC_DIR, "vlm_benchmarking")

sys.path.insert(0, ROOT_DIR)  # scene_graph_format, live_vlm_scene_graph
sys.path.insert(0, VLM_BENCHMARKING_DIR)  # vlm_bench package

# Default object vocabulary: union of objects across the 5 trained SO101 tasks
# (bowl-to-table, bowl-to-yellow-rectangle, cube-to-drawer, two-cubes-arrange,
# push-cube). Override with --objects for other tasks.
DEFAULT_OBJECTS = [
    "black_bowl",
    "drawer",
    "table",
    "yellow_rectangle",
    "black_cube_1",
    "black_cube_2",
    "blue_box",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", required=True, help="Local SmolVLA checkpoint directory.")
    parser.add_argument("--task", required=True, help="Natural-language task description.")
    parser.add_argument("--duration", type=float, default=60.0, help="Episode duration in seconds.")
    parser.add_argument("--fps", type=int, default=30, help="Control frequency.")
    parser.add_argument("--follower-port", default="/dev/ttyACM0", help="Follower arm USB port.")
    parser.add_argument("--follower-id", default="follower_arm", help="Follower arm calibration ID.")
    parser.add_argument("--wrist-cam", default="/dev/video2", help="Wrist camera device path.")
    parser.add_argument("--agent-cam", default="/dev/video8", help="Agent-view camera device path.")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument(
        "--urdf",
        required=True,
        help="Path to SO101 URDF for kinematics (e.g. the so101_new_calib.urdf used by your "
        "lerobot-so101-data data-collection scripts). EmbodimentSemantic has no copy of its own and "
        "this path is inherently machine-specific, so there is no safe default -- always pass it explicitly.",
    )

    # Semantic-context options.
    parser.add_argument(
        "--context-mode",
        choices=["standard", "scene_graph"],
        default="scene_graph",
        help="standard: plain task prompt, no VLM call. scene_graph: append a live VLM-derived scene graph.",
    )
    parser.add_argument(
        "--scene-graph-format",
        choices=["triplet", "natural_language", "json"],
        default="triplet",
        help="How the scene graph is rendered into text before being appended to the task prompt.",
    )
    parser.add_argument(
        "--context-refresh-interval",
        type=int,
        default=1,
        help="Query the VLM every N control steps (default 1 = every step). Each VLM call blocks the "
        "control loop for its full latency, so raising this trades context freshness for loop speed.",
    )
    parser.add_argument("--vlm-model", default="gemini-3.1-pro-preview", help="Model id passed to GeminiVLM.")
    parser.add_argument(
        "--objects",
        nargs="+",
        default=DEFAULT_OBJECTS,
        help="Object vocabulary the VLM is asked to name relations between. Override per task.",
    )
    parser.add_argument(
        "--vlm-max-new-tokens",
        type=int,
        default=4096,
        help="Gemini's 'thinking' tokens count against this budget too (see GeminiVLM's "
        "thinking_budget=1024 default) -- too low and the triplet response gets cut off mid-token.",
    )
    parser.add_argument("--vlm-temperature", type=float, default=None)
    parser.add_argument("--vlm-max-retries", type=int, default=8)
    parser.add_argument(
        "--debug-context", action="store_true", help="Print the VLM response and augmented prompt each refresh."
    )
    return parser.parse_args()


def _load_prompt_tokenizer(policy_config) -> tuple[object | None, int]:
    """Tokenizer + budget used to keep `task + scene-graph suffix` within the
    policy's prompt length (SmolVLAConfig.tokenizer_max_length, default 48).
    Uses the policy's own VLM backbone tokenizer (vlm_model_name) rather than
    assuming PaliGemma, since SmolVLA's backbone (SmolVLM2) has a different one.
    """
    tokenizer_name = getattr(policy_config, "vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    max_length = getattr(policy_config, "tokenizer_max_length", 48)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"  [warn] could not load tokenizer '{tokenizer_name}' ({e}); falling back to a char-length cap.")
        tokenizer = None
    return tokenizer, max_length


def main():
    args = parse_args()

    from dotenv import load_dotenv

    load_dotenv(os.path.join(VLM_BENCHMARKING_DIR, ".env"))

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.configs import PreTrainedConfig
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.processor import (
        RobotProcessorPipeline,
        observation_to_transition,
        robot_action_observation_to_transition,
        transition_to_observation,
        transition_to_robot_action,
    )
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.robots.so_follower.robot_kinematic_processor import (
        ForwardKinematicsJointsToEE,
        InverseKinematicsEEToJoints,
    )
    from lerobot.rollout import BaseStrategyConfig, RolloutConfig, build_rollout_context
    from lerobot.rollout.inference import SyncInferenceConfig
    from lerobot.rollout.strategies.base import BaseStrategy
    from lerobot.rollout.strategies.core import send_next_action
    from lerobot.types import RobotAction, RobotObservation
    from lerobot.utils.constants import ACTION
    from lerobot.utils.process import ProcessSignalHandler
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import init_logging
    from PIL import Image

    from live_vlm_scene_graph import LiveVLMSceneGraphGenerator
    from vlm_bench.models.gemini import GeminiVLM

    init_logging()

    camera_config = {
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_cam,
            width=args.cam_width,
            height=args.cam_height,
            fps=args.fps,
            fourcc="MJPG",
        ),
        "agent_view": OpenCVCameraConfig(
            index_or_path=args.agent_cam,
            width=args.cam_width,
            height=args.cam_height,
            fps=args.fps,
            fourcc="MJPG",
        ),
    }

    robot_config = SO101FollowerConfig(
        port=args.follower_port,
        id=args.follower_id,
        cameras=camera_config,
        use_degrees=True,
    )

    # Peek at motor names without connecting (used to configure kinematics).
    temp_robot = SO101Follower(robot_config)
    motor_names = list(temp_robot.bus.motors.keys())

    kinematics = RobotKinematics(
        urdf_path=args.urdf,
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )

    # Joint obs -> EE obs (what the policy was trained on).
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=motor_names)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # See deploy_ee_real_robot.py for why this explicit teleop_action_processor
    # is required (works around a feature-aggregation bug in
    # ForwardKinematicsJointsToEE.transform_features when it's left implicit).
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=motor_names)],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # EE action -> joint action (what the motors accept).
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=motor_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    policy_config = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_config.pretrained_path = args.policy_path

    cfg = RolloutConfig(
        robot=robot_config,
        policy=policy_config,
        strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(),
        fps=args.fps,
        duration=args.duration,
        task=args.task,
    )

    live_generator = None
    if args.context_mode != "standard":
        vlm = GeminiVLM(
            args.vlm_model,
            temperature=args.vlm_temperature,
            max_new_tokens=args.vlm_max_new_tokens,
            max_retries=args.vlm_max_retries,
        )
        tokenizer, max_prompt_tokens = _load_prompt_tokenizer(policy_config)
        live_generator = LiveVLMSceneGraphGenerator(
            vlm=vlm,
            objects=args.objects,
            scene_graph_format=args.scene_graph_format,
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            debug=args.debug_context,
        )

    print(f"Policy:          {args.policy_path}")
    print(f"Robot:           so101_follower on {args.follower_port} ({args.follower_id})")
    print(f"Cameras:         wrist={args.wrist_cam}  agent={args.agent_cam}")
    print(f"Task:            {args.task}")
    print(f"Context mode:    {args.context_mode}" + (f"  (format={args.scene_graph_format})" if live_generator else ""))
    if live_generator:
        print(f"VLM:             {args.vlm_model}  (refresh every {args.context_refresh_interval} step(s))")
        print(f"Objects:         {args.objects}")
    print(f"Duration:        {args.duration}s @ {args.fps} FPS")
    confirm = input("Place the robot in a safe start pose, clear the workspace, then type RUN: ")
    if confirm.strip() != "RUN":
        print("Aborted.")
        sys.exit(0)

    signal_handler = ProcessSignalHandler(use_threads=True)
    ctx = build_rollout_context(
        cfg,
        signal_handler.shutdown_event,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    # See deploy_ee_real_robot.py for why this realignment is required: for an
    # EE-space policy with no action_feature_names set, build_rollout_context
    # resolves ordered_action_keys against the robot's raw joint names instead
    # of the actual EE action names make_robot_action keys its output by.
    correct_action_keys = list(ctx.data.dataset_features[ACTION]["names"])
    ctx.data.ordered_action_keys[:] = correct_action_keys
    ctx.policy.inference._ordered_action_keys = correct_action_keys

    class SemanticBaseStrategy(BaseStrategy):
        """BaseStrategy.run(), plus a live VLM scene-graph refresh before each
        action is requested. ctx.policy.inference._task is a plain instance
        attribute the engine reads fresh on every get_action() call (see
        lerobot/rollout/inference/sync.py) -- updating it here is the
        intended hook for per-step prompt augmentation, no patching needed.
        """

        def run(self, ctx) -> None:
            engine = self._engine
            cfg = ctx.runtime.cfg
            robot = ctx.hardware.robot_wrapper
            interpolator = self._interpolator
            control_interval = interpolator.get_control_interval(cfg.fps)

            base_task = cfg.task
            step = 0
            start_time = time.perf_counter()
            engine.resume()
            print("Semantic strategy control loop started")

            while not ctx.runtime.shutdown_event.is_set():
                loop_start = time.perf_counter()

                if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                    print(f"Duration limit reached ({cfg.duration:.0f}s)")
                    break

                obs = robot.get_observation()
                obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                    continue

                if live_generator is not None and step % args.context_refresh_interval == 0:
                    frame = Image.fromarray(obs["agent_view"])
                    t0 = time.perf_counter()
                    suffix = live_generator.suffix_for_frame(frame, base_task, args.context_mode)
                    vlm_latency = time.perf_counter() - t0
                    engine._task = f"{base_task}{suffix}"
                    if args.debug_context:
                        print(f"\n[step {step}] VLM call took {vlm_latency:.2f}s")
                        print(f"  raw response: {live_generator._last_response!r}")
                        print(f"  prompt sent to policy: {engine._task!r}\n")

                action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
                self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                step += 1

                dt = time.perf_counter() - loop_start
                if (sleep_t := control_interval - dt) > 0:
                    precise_sleep(sleep_t)
                elif live_generator is None:
                    print(
                        f"[warn] control loop running slower ({1 / dt:.1f} Hz) than target FPS ({cfg.fps} Hz)."
                    )
                # When live_generator is set, running below target FPS is expected
                # (every refreshed step blocks on the VLM call) -- no warning spam.

    strategy = SemanticBaseStrategy(cfg.strategy)
    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    finally:
        strategy.teardown(ctx)


if __name__ == "__main__":
    main()
