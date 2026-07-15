from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .artifacts import ArtifactStore
from .dataset import load_episode_index, load_state_rows
from .schemas import EpisodeRecord, MetadataFrame, read_jsonl
from .task_priors import task_prior


@dataclass(frozen=True)
class GripperAnalysis:
    closed_center: float
    open_center: float
    threshold: float
    transitions: tuple[tuple[int, bool, bool], ...]
    grasp_frame: int | None
    release_frame: int | None
    first_open_frame: int | None
    return_close_frame: int | None
    action_lead_frames: int | None
    reliable: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "closed_center": self.closed_center,
            "open_center": self.open_center,
            "threshold": self.threshold,
            "transitions": [
                {"frame": frame, "from_open": before, "to_open": after}
                for frame, before, after in self.transitions
            ],
            "grasp_frame": self.grasp_frame,
            "release_frame": self.release_frame,
            "first_open_frame": self.first_open_frame,
            "return_close_frame": self.return_close_frame,
            "action_lead_frames": self.action_lead_frames,
            "reliable": self.reliable,
            "reasons": list(self.reasons),
        }


def moving_median(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) <= 1:
        return values.copy()
    radius = window // 2
    return np.array(
        [np.median(values[max(0, index - radius): min(len(values), index + radius + 1)]) for index in range(len(values))],
        dtype=float,
    )


def two_cluster_threshold(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot cluster an empty gripper sequence")
    centers = np.array([np.quantile(values, 0.25), np.quantile(values, 0.90)], dtype=float)
    for _ in range(50):
        labels = np.abs(values[:, None] - centers[None, :]).argmin(axis=1)
        updated = np.array(
            [values[labels == index].mean() if np.any(labels == index) else centers[index] for index in range(2)]
        )
        if np.allclose(updated, centers):
            break
        centers = updated
    centers.sort()
    return float(centers[0]), float(centers[1]), float(centers.mean())


def debounce_boolean(values: np.ndarray, minimum_dwell: int) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    if len(values) == 0 or minimum_dwell <= 1:
        return values.copy()
    result = values.copy()
    current = bool(result[0])
    index = 1
    while index < len(result):
        if bool(result[index]) == current:
            index += 1
            continue
        end = index
        while end < len(result) and bool(result[end]) != current:
            end += 1
        if end - index < minimum_dwell:
            result[index:end] = current
        else:
            current = bool(result[index])
        index = end
    return result


def transitions(values: np.ndarray) -> tuple[tuple[int, bool, bool], ...]:
    return tuple(
        (index, bool(values[index - 1]), bool(values[index]))
        for index in range(1, len(values))
        if bool(values[index]) != bool(values[index - 1])
    )


def action_lead_frames(action: np.ndarray, state: np.ndarray, maximum_lag: int = 6) -> int | None:
    if len(action) < 4 or len(state) != len(action):
        return None
    da, ds = np.diff(action), np.diff(state)
    scores: list[float] = []
    for lag in range(maximum_lag + 1):
        x, y = (da, ds) if lag == 0 else (da[:-lag], ds[lag:])
        if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
            scores.append(float("-inf"))
        else:
            scores.append(float(np.corrcoef(x, y)[0, 1]))
    best = int(np.argmax(scores))
    return best if np.isfinite(scores[best]) else None


def _select_events(
    open_state: np.ndarray,
    z: np.ndarray,
    lift_delta: float,
) -> tuple[int | None, int | None, int | None, int | None]:
    changes = transitions(open_state)
    first_open = next((frame for frame, before, after in changes if not before and after), None)
    close_candidates = [
        frame for frame, before, after in changes
        if before and not after and (first_open is None or frame > first_open)
    ]
    grasp: int | None = None
    for candidate in close_candidates:
        stop = min(len(z), candidate + 91)
        if stop > candidate and float(np.max(z[candidate:stop]) - z[candidate]) >= lift_delta:
            grasp = candidate
            break
    if grasp is None and close_candidates:
        grasp = close_candidates[0]

    release = next(
        (frame for frame, before, after in changes if not before and after and grasp is not None and frame > grasp + 4),
        None,
    )
    return_close = next(
        (frame for frame, before, after in changes if before and not after and release is not None and frame > release),
        None,
    )
    return first_open, grasp, release, return_close


def analyze_episode(
    episode: EpisodeRecord,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[GripperAnalysis, list[MetadataFrame]]:
    if not rows:
        raise ValueError(f"No state rows found for {episode.task}/{episode.episode}")
    state = np.asarray([row["observation.state"] for row in rows], dtype=float)
    action = np.asarray([row["action"] for row in rows], dtype=float)
    if state.ndim != 2 or state.shape[1] < 7 or action.shape != state.shape:
        raise ValueError(f"Invalid state/action shape for {episode.task}/{episode.episode}")

    settings = config["metadata"]
    state_gripper = moving_median(state[:, 6], int(settings["smoothing_window"]))
    action_gripper = moving_median(action[:, 6], int(settings["smoothing_window"]))
    closed_center, open_center, threshold = two_cluster_threshold(state_gripper)
    open_state = debounce_boolean(
        state_gripper > threshold,
        int(settings["minimum_transition_dwell"]),
    )
    changes = transitions(open_state)
    first_open, grasp, release, return_close = _select_events(
        open_state,
        state[:, 2],
        float(settings["lift_delta_m"]),
    )
    lead = action_lead_frames(
        action_gripper,
        state_gripper,
        int(settings["action_confirmation_max_lag_frames"]),
    )

    reasons: list[str] = []
    if open_center - closed_center < float(settings["minimum_gripper_cluster_separation"]):
        reasons.append("weak_gripper_cluster_separation")
    if len(changes) < int(settings["minimum_transitions"]):
        reasons.append("too_few_gripper_transitions")
    if len(changes) > int(settings["maximum_transitions"]):
        reasons.append("too_many_gripper_transitions")
    if grasp is None:
        reasons.append("grasp_not_found")
    if release is None:
        reasons.append("release_not_found")
    reliable = not reasons

    analysis = GripperAnalysis(
        closed_center=closed_center,
        open_center=open_center,
        threshold=threshold,
        transitions=changes,
        grasp_frame=grasp,
        release_frame=release,
        first_open_frame=first_open,
        return_close_frame=return_close,
        action_lead_frames=lead,
        reliable=reliable,
        reasons=tuple(reasons),
    )

    xyz_delta = np.diff(state[:, :3], axis=0, prepend=state[:1, :3])
    rotation_delta = np.diff(state[:, 3:6], axis=0, prepend=state[:1, 3:6])
    ee_speed = np.linalg.norm(xyz_delta, axis=1)
    angular_speed = np.linalg.norm(rotation_delta, axis=1)
    grasp_z = float(state[grasp, 2]) if grasp is not None else float("inf")
    prior = task_prior(episode.task, episode.task_text)
    frame_signals: list[MetadataFrame] = []

    for offset, row in enumerate(rows):
        frame = int(row["frame_index"])
        held = bool(reliable and grasp is not None and release is not None and grasp <= offset < release)
        lifted = bool(held and state[offset, 2] >= grasp_z + float(settings["lift_delta_m"]))
        released = bool(reliable and release is not None and offset >= release and (return_close is None or offset < return_close))

        if first_open is not None and offset < first_open:
            phase = "home"
        elif grasp is not None and offset < grasp:
            phase = "approach"
        elif held:
            phase = "lifted" if lifted else "held"
        elif released:
            phase = "released"
        elif release is not None and offset >= release:
            phase = "return"
        else:
            phase = "unknown"

        gates: list[str] = []
        if held:
            gates.append("held")
        if lifted:
            gates.append("lifted")
        if released:
            gates.append("released")
        if prior.allowed_supports:
            gates.append("task_support_allowed")
        if not reliable:
            gates.append("metadata_unreliable")

        frame_signals.append(
            MetadataFrame(
                task=episode.task,
                episode=episode.episode,
                frame=frame,
                timestamp=float(row.get("timestamp", frame / episode.fps)),
                phase=phase,
                gripper_open=bool(open_state[offset]),
                held=held,
                lifted=lifted,
                released=released,
                metadata_reliable=reliable,
                state_xyz=tuple(float(item) for item in state[offset, :3]),
                state_rotation=tuple(float(item) for item in state[offset, 3:6]),
                state_gripper=float(state[offset, 6]),
                action_gripper=float(action[offset, 6]),
                ee_speed=float(ee_speed[offset]),
                angular_speed=float(angular_speed[offset]),
                gates=tuple(gates),
            )
        )
    return analysis, frame_signals


def extract_metadata(
    dataset_root: str,
    episode_index_path: str,
    artifacts: ArtifactStore,
    config: dict[str, Any],
) -> dict[str, Any]:
    episodes = load_episode_index(episode_index_path)
    rows_by_episode = load_state_rows(dataset_root, episodes)
    summaries: list[dict[str, Any]] = []
    all_frames: list[MetadataFrame] = []
    failed: list[str] = []

    for episode in episodes:
        rows = rows_by_episode.get((episode.task, episode.episode_index), [])
        try:
            analysis, frames = analyze_episode(episode, rows, config)
        except (ValueError, KeyError, TypeError) as exc:
            failed.append(f"{episode.task}/{episode.episode}: {exc}")
            continue
        summaries.append(
            {
                "task": episode.task,
                "task_text": episode.task_text,
                "episode": episode.episode,
                "episode_index": episode.episode_index,
                "task_prior": task_prior(episode.task, episode.task_text).to_dict(),
                **analysis.to_dict(),
            }
        )
        all_frames.extend(frames)

    artifacts.write_jsonl("metadata/episode_signals.jsonl", summaries)
    artifacts.write_jsonl("metadata/frame_signals.jsonl", (frame.to_dict() for frame in all_frames))
    report = {
        "episodes_requested": len(episodes),
        "episodes_analyzed": len(summaries),
        "reliable_episodes": sum(1 for item in summaries if item["reliable"]),
        "unreliable_episodes": sum(1 for item in summaries if not item["reliable"]),
        "frame_signals": len(all_frames),
        "failures": failed,
        "median_action_lead_frames": float(
            np.median([item["action_lead_frames"] for item in summaries if item["action_lead_frames"] is not None])
        ) if summaries else None,
    }
    artifacts.write_json("reports/metadata_report.json", report)
    return report


def load_metadata_frames(path: str) -> dict[tuple[str, str, int], MetadataFrame]:
    return {
        frame.key(): frame
        for frame in (MetadataFrame.from_dict(value) for value in read_jsonl(path))
    }
