"""Causal cross-view identity evidence for the two visually identical bowls."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from samgraph_core.automatic_scene import _appearance, _center


def filter_agent_triplets(triplets: Iterable[Iterable[str]], members: set[str]) -> list[list[str]]:
    """Retain exact ordered agent relations whose two endpoints are wrist members."""
    result: list[list[str]] = []
    for item in triplets:
        row = list(item)
        if len(row) != 3 or not all(isinstance(value, str) for value in row):
            raise ValueError("malformed frozen agent triplet")
        if row[0] in members and row[2] in members:
            result.append(row)
    return result


def _state_by_track(states: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(state["track_id"]): state for state in states
            if str(state.get("class_id")) == "black_bowl"}


class CrossViewBowlResolver:
    """Resolve wrist bowl tracks to frozen agent tracks without spatial numbering."""

    def __init__(self, *, appearance_margin: float = 0.08,
                 motion_margin: float = 0.012, motion_streak: int = 3):
        self.appearance_margin = float(appearance_margin)
        self.motion_margin = float(motion_margin)
        self.required_motion_streak = int(motion_streak)
        self.mapping: dict[str, str] = {}
        self.assignment_frame: int | None = None
        self.assignment_method: str | None = None
        self._history: dict[str, dict[str, list[tuple[int, np.ndarray]]]] = {
            "wrist": defaultdict(list), "agent": defaultdict(list),
        }
        self._classes: dict[str, dict[str, str]] = {"wrist": {}, "agent": {}}
        self._motion_winner: str | None = None
        self._motion_streak = 0

    @staticmethod
    def _current_masks(states: Iterable[Mapping[str, Any]], masks: Mapping[str, np.ndarray],
                       *, wrist: bool) -> dict[str, np.ndarray]:
        result = {}
        for state in states:
            if state.get("class_id") != "black_bowl":
                continue
            if wrist and state.get("observation_state") != "current_observation":
                continue
            if not wrist and state.get("status") != "observed":
                continue
            key = str(state.get("output_id") or state.get("semantic_id") or state["track_id"])
            if key in masks and np.asarray(masks[key], dtype=bool).any():
                result[str(state["track_id"])] = np.asarray(masks[key], dtype=bool)
        return result

    def _append(self, view: str, frame: int, masks: Mapping[str, np.ndarray]) -> None:
        for track, mask in masks.items():
            history = self._history[view][track]
            value = _center(mask).astype(np.float64)
            if not history or history[-1][0] != frame:
                history.append((frame, value))
            else:
                history[-1] = (frame, value)
            if len(history) > 12:
                del history[:-12]

    def _motion_scores(self, view: str) -> dict[str, float]:
        anchor_deltas = []
        for track, history in self._history[view].items():
            if self._classes[view].get(track) == "black_bowl" or len(history) < 2:
                continue
            anchor_deltas.append(history[-1][1] - history[-2][1])
        camera_delta = (np.median(np.stack(anchor_deltas), axis=0)
                        if anchor_deltas else np.zeros(2, dtype=np.float64))
        scores = {}
        for track, history in self._history[view].items():
            if self._classes[view].get(track) != "black_bowl" or len(history) < 2:
                continue
            (_, old), (_, new) = history[-2], history[-1]
            scores[track] = float(np.linalg.norm((new - old) - camera_delta)) / 1024.0
        return scores

    def _append_view(self, view: str, frame: int, states: Iterable[Mapping[str, Any]],
                     masks: Mapping[str, np.ndarray], *, wrist: bool) -> None:
        current = {}
        for state in states:
            if wrist and state.get("observation_state") != "current_observation":
                continue
            if not wrist and state.get("status") != "observed":
                continue
            key = str(state.get("output_id") or state.get("semantic_id") or state["track_id"])
            if key not in masks or not np.asarray(masks[key], dtype=bool).any():
                continue
            track = str(state["track_id"])
            current[track] = np.asarray(masks[key], dtype=bool)
            self._classes[view][track] = str(state.get("class_id"))
        self._append(view, frame, current)

    @staticmethod
    def _agent_moving_track(states: Iterable[Mapping[str, Any]]) -> str | None:
        bowls = _state_by_track(states)
        named = [track for track, state in bowls.items()
                 if state.get("semantic_id") == "black_bowl_1"
                 and state.get("identity_method") == "causal_motion"]
        return named[0] if len(named) == 1 else None

    def update(self, *, frame: int, wrist_rgb: np.ndarray,
               wrist_masks: Mapping[str, np.ndarray], wrist_states: list[Mapping[str, Any]],
               agent_rgb: np.ndarray, agent_masks: Mapping[str, np.ndarray],
               agent_states: list[Mapping[str, Any]]) -> dict[str, Any]:
        current_wrist = self._current_masks(wrist_states, wrist_masks, wrist=True)
        current_agent = self._current_masks(agent_states, agent_masks, wrist=False)
        self._append_view("wrist", frame, wrist_states, wrist_masks, wrist=True)
        self._append_view("agent", frame, agent_states, agent_masks, wrist=False)
        evidence: dict[str, Any] = {
            "frame": int(frame),
            "policy": "cross_view_rgb_appearance_then_causal_motion",
            "wrist_tracks_observed": sorted(current_wrist),
            "agent_tracks_observed": sorted(current_agent),
            "mapping_before": dict(self.mapping),
        }
        if self.mapping:
            evidence.update(decision="retained_prior_causal_assignment",
                            mapping_after=dict(self.mapping),
                            assignment_frame=self.assignment_frame,
                            assignment_method=self.assignment_method)
            return evidence

        if len(current_wrist) == len(current_agent) == 2:
            wrist_ids = sorted(current_wrist)
            agent_ids = sorted(current_agent)
            descriptors_w = {track: _appearance(wrist_rgb, mask)
                             for track, mask in current_wrist.items()}
            descriptors_a = {track: _appearance(agent_rgb, mask)
                             for track, mask in current_agent.items()}
            direct = sum(float(np.abs(descriptors_w[wrist_ids[i]]
                                      - descriptors_a[agent_ids[i]]).sum()) for i in range(2))
            swapped = sum(float(np.abs(descriptors_w[wrist_ids[i]]
                                       - descriptors_a[agent_ids[1 - i]]).sum()) for i in range(2))
            margin = abs(direct - swapped)
            evidence["appearance"] = {
                "direct_cost": direct, "swapped_cost": swapped,
                "assignment_margin": margin,
                "threshold": self.appearance_margin,
                "descriptor": "per-channel_8-bin_rgb_histogram",
            }
            if margin >= self.appearance_margin:
                order = agent_ids if direct < swapped else list(reversed(agent_ids))
                self.mapping = dict(zip(wrist_ids, order))
                self.assignment_frame = int(frame)
                self.assignment_method = "cross_view_rgb_appearance"

        agent_moving = self._agent_moving_track(agent_states)
        wrist_scores = {key: value for key, value in self._motion_scores("wrist").items()
                        if key in _state_by_track(wrist_states)}
        agent_scores = {key: value for key, value in self._motion_scores("agent").items()
                        if key in _state_by_track(agent_states)}
        evidence["causal_motion"] = {
            "wrist_anchor_residual_scores": wrist_scores,
            "agent_anchor_residual_scores": agent_scores,
            "agent_causal_moving_track": agent_moving,
            "required_wrist_margin": self.motion_margin,
            "required_consecutive_samples": self.required_motion_streak,
        }
        if not self.mapping and agent_moving and len(wrist_scores) == 2:
            ordered = sorted(wrist_scores.items(), key=lambda item: (-item[1], item[0]))
            winner, winner_score = ordered[0]
            margin = winner_score - ordered[1][1]
            if margin >= self.motion_margin:
                self._motion_streak = self._motion_streak + 1 if winner == self._motion_winner else 1
                self._motion_winner = winner
            else:
                self._motion_streak = 0
                self._motion_winner = None
            evidence["causal_motion"].update(
                wrist_candidate=winner, wrist_margin=margin,
                current_streak=self._motion_streak,
            )
            if self._motion_streak >= self.required_motion_streak:
                agent_ids = sorted(_state_by_track(agent_states))
                other_agent = next((track for track in agent_ids if track != agent_moving), None)
                other_wrist = next((track for track in wrist_scores if track != winner), None)
                if other_agent and other_wrist:
                    self.mapping = {winner: agent_moving, other_wrist: other_agent}
                    self.assignment_frame = int(frame)
                    self.assignment_method = "cross_view_causal_motion"

        evidence.update(
            decision=("assigned" if self.mapping else "unresolved_correspondence"),
            mapping_after=dict(self.mapping),
            assignment_frame=self.assignment_frame,
            assignment_method=self.assignment_method,
        )
        return evidence
