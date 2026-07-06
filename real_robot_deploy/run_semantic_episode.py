#!/usr/bin/env python3
"""Run a single semantic-context episode on the real SO101 robot and record it.

Wraps deploy_smolvla_semantic_real_robot.py with:
  - a dedicated documentation camera (--record-cam) that captures a start
    snapshot + full episode video, independent of the policy's wrist/agent
    cams (same pattern as
    lerobot-so101-data/language_ablation_study/experiments/run_ablation_episode.py)
  - when --context-mode scene_graph, every VLM scene-graph refresh during
    the episode (raw VLM response + augmented prompt + latency) is recorded
    and saved as scene_graph_log.json next to the video

Output layout (mirrors run_ablation_episode.py):
    results/media/<trial-label>/<context_mode>/trial_<timestamp>/
        snapshot_start.png        (only with --record-cam)
        episode.mp4               (only with --record-cam)
        scene_graph_log.json      (only when --context-mode scene_graph)

After the episode ends, prints the log_trial.py command to record the
trial's outcome.

Usage:
    python run_semantic_episode.py \\
        --policy-path /home/refinath/lerobot-so101-data/outputs/train/smolvla_so101_multitask/checkpoints/last/pretrained_model \\
        --task "Pick up the black cube on top of the blue box and place it inside the drawer" \\
        --objects black_cube_1 blue_box drawer \\
        --urdf /home/refinath/lerobot-so101-data/SO101/so101_new_calib.urdf \\
        --context-mode scene_graph \\
        --context-refresh-interval 50 \\
        --record-cam /dev/video14 \\
        --max-steps 500

    # Plain SmolVLA, no VLM call, no scene-graph log, run for a fixed duration instead of steps:
    python run_semantic_episode.py \\
        --policy-path <path> --task "..." --context-mode standard --record-cam /dev/video14 --duration 60
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBODIMENT_SEMANTIC_DIR = os.path.dirname(ROOT_DIR)
VLM_BENCHMARKING_DIR = os.path.join(EMBODIMENT_SEMANTIC_DIR, "vlm_benchmarking")

sys.path.insert(0, ROOT_DIR)  # scene_graph_format, live_vlm_scene_graph
sys.path.insert(0, VLM_BENCHMARKING_DIR)  # vlm_bench package

RESULTS_DIR = Path(ROOT_DIR) / "results"
MEDIA_DIR = RESULTS_DIR / "media"

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


class EpisodeRecorder:
    """Captures a high-quality start snapshot and a video for one trial
    from a dedicated documentation camera, independent of the policy's
    observation cameras."""

    def __init__(self, device, width, height, fps, codec):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.cap = None
        self.writer = None
        self.thread = None
        self.stop_event = threading.Event()

    def open(self):
        import cv2
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open recording camera at {self.device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Warm up so auto-exposure/white-balance settle before we snapshot.
        for _ in range(15):
            self.cap.read()

    def capture_snapshot(self, path):
        import cv2
        ok, frame = self.cap.read()
        if not ok:
            print(f"  WARNING: failed to capture snapshot from {self.device}")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        print(f"  Saved snapshot: {path}")

    def start_recording(self, path):
        import cv2
        path.parent.mkdir(parents=True, exist_ok=True)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(str(path), fourcc, self.fps, (actual_w, actual_h))
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        print(f"  Recording started ({actual_w}x{actual_h} @ {self.fps}fps): {path}")

    def _record_loop(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if ok and self.writer is not None:
                self.writer.write(frame)

    def stop_recording(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        if self.writer:
            self.writer.release()
            self.writer = None

    def close(self):
        self.stop_recording()
        if self.cap:
            self.cap.release()
            self.cap = None


def slugify(text: str, max_words: int = 4) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())[:max_words]
    return "_".join(words) or "trial"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-path", required=True, help="Local SmolVLA checkpoint directory.")
    p.add_argument("--task", required=True, help="Natural-language task description.")
    p.add_argument("--duration", type=float, default=60.0,
                    help="Episode duration in seconds. Ignored if --max-steps is set.")
    p.add_argument("--max-steps", type=int, default=None,
                    help="Stop after this many control steps instead of by wall-clock duration. "
                    "Takes precedence over --duration when set.")
    p.add_argument("--fps", type=int, default=30, help="Control frequency.")
    p.add_argument("--follower-port", default="/dev/ttyACM0", help="Follower arm USB port.")
    p.add_argument("--follower-id", default="follower_arm", help="Follower arm calibration ID.")
    p.add_argument("--wrist-cam", default="/dev/video2", help="Wrist camera device path.")
    p.add_argument("--agent-cam", default="/dev/video8", help="Agent-view camera device path.")
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument(
        "--urdf",
        default="/home/refinath/lerobot-so101-data/SO101/so101_new_calib.urdf",
        help="Path to SO101 URDF for kinematics. EmbodimentSemantic has no copy of its own, so this "
        "defaults to the one in lerobot-so101-data; override if running on a different machine.",
    )

    # Semantic-context options (same as deploy_smolvla_semantic_real_robot.py).
    p.add_argument(
        "--context-mode",
        choices=["standard", "scene_graph"],
        default="scene_graph",
        help="standard: plain task prompt, no VLM call. scene_graph: append a live VLM-derived scene graph.",
    )
    p.add_argument(
        "--scene-graph-format",
        choices=["triplet", "natural_language", "json"],
        default="triplet",
        help="How the scene graph is rendered into text before being appended to the task prompt.",
    )
    p.add_argument(
        "--context-refresh-interval",
        type=int,
        default=50,
        help="Query the VLM every N control steps (default 50). Each VLM call blocks the control loop "
        "for its full latency, so lower values trade loop speed for context freshness.",
    )
    p.add_argument("--vlm-model", default="gemini-3.1-pro-preview", help="Model id passed to GeminiVLM.")
    p.add_argument(
        "--objects",
        nargs="+",
        default=DEFAULT_OBJECTS,
        help="Object vocabulary the VLM is asked to name relations between. Override per task.",
    )
    p.add_argument("--vlm-max-new-tokens", type=int, default=4096)
    p.add_argument("--vlm-temperature", type=float, default=None)
    p.add_argument("--vlm-max-retries", type=int, default=8)
    p.add_argument(
        "--debug-context", action="store_true", help="Print the VLM response and augmented prompt each refresh."
    )

    # Recording + bookkeeping options (mirrors run_ablation_episode.py).
    p.add_argument(
        "--trial-label",
        default=None,
        help="Folder name this trial groups under (results/media/<trial-label>/...). "
        "Defaults to a slug of --task, e.g. 'pick_up_the_black'.",
    )
    p.add_argument(
        "--record-cam",
        default=None,
        help="Device path of a dedicated documentation camera (e.g. /dev/video14). If set, captures a "
        "start snapshot + full episode video for this trial. Independent of --wrist-cam/--agent-cam.",
    )
    p.add_argument("--record-width", type=int, default=1920, help="Recording camera capture width")
    p.add_argument("--record-height", type=int, default=1080, help="Recording camera capture height")
    p.add_argument("--record-fps", type=int, default=30, help="Recording camera capture fps")
    p.add_argument("--record-codec", default="mp4v", help="FourCC video codec for the .mp4 output")
    p.add_argument(
        "--dry-run", action="store_true", help="Print trial details and exit without running the robot."
    )
    return p.parse_args()


def _load_prompt_tokenizer(policy_config) -> tuple[object | None, int]:
    """Tokenizer + budget used to keep `task + scene-graph suffix` within the
    policy's prompt length (SmolVLAConfig.tokenizer_max_length, default 48).
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


def _print_log_command(trial_label, task, context_mode, media_dir):
    print("\n--- Log this trial result ---")
    print(f"python log_trial.py \\")
    print(f"    --trial-label {trial_label} \\")
    print(f'    --task "{task}" \\')
    print(f"    --context-mode {context_mode} \\")
    print(f"    --grasp <0|1> \\")
    print(f"    --success <0|1> \\")
    print(f"    --ttc <seconds_or_nan> \\")
    print(f"    --failure-mode <grasp_fail|transport_fail|placement_fail|wrong_object|no_motion|none>" +
          (" \\" if media_dir else ""))
    if media_dir:
        print(f"    --media-dir {media_dir}")


def main():
    args = parse_args()

    trial_label = args.trial_label or slugify(args.task)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    media_dir = MEDIA_DIR / trial_label / args.context_mode / f"trial_{timestamp}"

    print("=" * 70)
    print("  SEMANTIC EPISODE")
    print(f"  Trial label:     {trial_label}")
    print(f"  Task:            {args.task}")
    context_suffix = f" (format={args.scene_graph_format})" if args.context_mode != "standard" else ""
    print(f"  Context mode:    {args.context_mode}{context_suffix}")
    print(f"  Policy:          {args.policy_path}")
    if args.max_steps is not None:
        print(f"  Stop after:      {args.max_steps} steps  FPS: {args.fps}")
    else:
        print(f"  Duration:        {args.duration}s  FPS: {args.fps}")
    print(f"  Output folder:   {media_dir}")
    print("=" * 70)

    if args.dry_run:
        print("\n[dry-run] Exiting without running robot.")
        _print_log_command(trial_label, args.task, args.context_mode, None)
        return

    recorder = None
    if args.record_cam:
        recorder = EpisodeRecorder(args.record_cam, args.record_width, args.record_height,
                                    args.record_fps, args.record_codec)
        print(f"\nOpening documentation camera {args.record_cam} ...")
        recorder.open()

    input("\nPress ENTER when scene is set up and robot is ready...")

    if recorder:
        recorder.capture_snapshot(media_dir / "snapshot_start.png")

    start = time.time()

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
            index_or_path=args.wrist_cam, width=args.cam_width, height=args.cam_height,
            fps=args.fps, fourcc="MJPG",
        ),
        "agent_view": OpenCVCameraConfig(
            index_or_path=args.agent_cam, width=args.cam_width, height=args.cam_height,
            fps=args.fps, fourcc="MJPG",
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

    scene_graph_log: list[dict] = []
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

    class RecordingSemanticStrategy(BaseStrategy):
        """Same per-step VLM scene-graph refresh as
        deploy_smolvla_semantic_real_robot.SemanticBaseStrategy, plus
        appending each refresh to scene_graph_log for later inspection.
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

                if args.max_steps is not None:
                    if step >= args.max_steps:
                        print(f"Step limit reached ({args.max_steps} steps)")
                        break
                elif cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
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
                    scene_graph_log.append({
                        "step": step,
                        "time_s": round(time.perf_counter() - start_time, 3),
                        "vlm_latency_s": round(vlm_latency, 3),
                        "raw_response": live_generator._last_response,
                        "prompt_sent": engine._task,
                    })
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

    strategy = RecordingSemanticStrategy(cfg.strategy)
    try:
        strategy.setup(ctx)
        if recorder:
            recorder.start_recording(media_dir / "episode.mp4")
        strategy.run(ctx)
    finally:
        if recorder:
            recorder.close()
        strategy.teardown(ctx)

    if scene_graph_log:
        media_dir.mkdir(parents=True, exist_ok=True)
        log_path = media_dir / "scene_graph_log.json"
        with open(log_path, "w") as f:
            json.dump(
                {
                    "task": args.task,
                    "objects": args.objects,
                    "context_mode": args.context_mode,
                    "scene_graph_format": args.scene_graph_format,
                    "vlm_model": args.vlm_model,
                    "context_refresh_interval": args.context_refresh_interval,
                    "entries": scene_graph_log,
                },
                f,
                indent=2,
            )
        print(f"Scene graph log saved: {log_path}")

    elapsed = time.time() - start
    print(f"\nEpisode finished in {elapsed:.1f}s")
    has_output = recorder is not None or bool(scene_graph_log)
    if has_output:
        print(f"Media saved to: {media_dir}")
    _print_log_command(trial_label, args.task, args.context_mode, media_dir if has_output else None)


if __name__ == "__main__":
    main()
