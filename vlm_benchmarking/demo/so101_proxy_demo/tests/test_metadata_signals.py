from __future__ import annotations

import numpy as np

from demo.so101_proxy_demo.proxy.metadata_signals import (
    action_lead_frames,
    analyze_episode,
    debounce_boolean,
    two_cluster_threshold,
)
from demo.so101_proxy_demo.proxy.schemas import EpisodeRecord


def _config() -> dict:
    return {
        "metadata": {
            "smoothing_window": 5,
            "minimum_transition_dwell": 5,
            "minimum_gripper_cluster_separation": 5.0,
            "minimum_transitions": 2,
            "maximum_transitions": 6,
            "lift_delta_m": .02,
            "action_confirmation_max_lag_frames": 6,
        }
    }


def _episode() -> EpisodeRecord:
    return EpisodeRecord(
        task="place-the-black-bowl-from-the-left-to-the-top-of-the-stove",
        task_text="Place the black bowl from the left to the top of the stove",
        episode="episode_0",
        episode_index=0,
        length=100,
        fps=30,
        dataset_from_index=0,
        dataset_to_index=100,
        data_chunk_index=0,
        data_file_index=0,
        videos={},
    )


def _rows() -> list[dict]:
    gripper = np.concatenate([
        np.full(15, 2.0),
        np.full(20, 20.0),
        np.full(30, 2.0),
        np.full(20, 20.0),
        np.full(15, 2.0),
    ])
    z = np.full(100, .01)
    z[40:65] = np.linspace(.01, .08, 25)
    action = np.roll(gripper, -3)
    action[-3:] = gripper[-1]
    return [
        {
            "observation.state": [0, 0, float(z[index]), 0, 0, 0, float(gripper[index])],
            "action": [0, 0, float(z[index]), 0, 0, 0, float(action[index])],
            "timestamp": index / 30,
            "frame_index": index,
            "episode_index": 0,
        }
        for index in range(100)
    ]


def test_adaptive_gripper_clustering_and_phase_gates() -> None:
    analysis, frames = analyze_episode(_episode(), _rows(), _config())
    assert analysis.reliable
    assert analysis.grasp_frame is not None
    assert analysis.release_frame is not None
    assert analysis.action_lead_frames == 3
    assert any(frame.held for frame in frames)
    assert any(frame.lifted for frame in frames)
    assert all(not frame.held for frame in frames[analysis.release_frame:])


def test_cluster_threshold_is_episode_adaptive() -> None:
    low, high, threshold = two_cluster_threshold(np.array([1, 1, 2, 2, 20, 21, 22]))
    assert low < threshold < high
    assert threshold != 11.0


def test_debounce_removes_short_transition() -> None:
    values = np.array([False] * 8 + [True] * 2 + [False] * 8)
    assert not debounce_boolean(values, 5).any()


def test_action_lead_detection() -> None:
    state = np.zeros(50)
    state[20:] = 10
    action = np.zeros(50)
    action[17:] = 10
    assert action_lead_frames(action, state, 6) == 3
