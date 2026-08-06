from __future__ import annotations

import numpy as np

from visual_scene_graph import (
    VisualGraphVecEnvWrapper,
    draw_scene_graph_arrows,
    goal_arrow_prompt_hint,
    select_visual_relations,
)


def test_renderer_draws_arrow_and_preserves_shape_and_dtype():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    bboxes = {
        "target": [4, 28, 12, 36],
        "object": [52, 28, 60, 36],
    }
    relations = [("target", "is_left_of", "object")]

    rendered = draw_scene_graph_arrows(image, bboxes, relations)

    assert rendered.shape == image.shape
    assert rendered.dtype == np.uint8
    assert not np.array_equal(rendered, image)
    assert np.array_equal(image, np.zeros_like(image))
    assert rendered[32, 32].any()


def test_renderer_skips_relation_with_missing_bbox():
    image = np.full((32, 32, 3), 17, dtype=np.uint8)

    rendered = draw_scene_graph_arrows(
        image,
        {"target": [2, 2, 8, 8]},
        [("target", "is_left_of", "missing")],
    )

    assert rendered.shape == image.shape
    assert rendered.dtype == image.dtype
    assert np.array_equal(rendered, image)


def test_goal_arrow_selector_keeps_only_target_to_goal():
    bboxes = {
        "target": [2, 2, 8, 8],
        "object": [12, 2, 18, 8],
        "plate_1": [24, 2, 30, 8],
    }
    selected = select_visual_relations(
        bboxes,
        [
            ("target", "is_left_of", "object"),
            ("target", "is_left_of", "plate_1"),
        ],
        condition="visual_goal_arrow",
        subject="target",
        goal_object="plate_1",
    )

    assert selected == [("target", "goal", "plate_1")]


def test_goal_arrow_prompt_hint_uses_goal_object_name():
    assert (
        goal_arrow_prompt_hint("wooden_cabinet_1")
        == "The green arrow in the image points from the black bowl to the wooden cabinet where it should be placed."
    )


class _FakeSubEnv:
    task_id = 3
    camera_name_mapping = {
        "agentview_image": "image",
        "robot0_eye_in_hand_image": "image2",
    }


class _FakeVecEnv:
    def __init__(self):
        self.num_envs = 1
        self.envs = [_FakeSubEnv()]
        self.main = np.zeros((1, 64, 64, 3), dtype=np.uint8)
        self.wrist = np.full((1, 64, 64, 3), 31, dtype=np.uint8)
        self.raw_render = np.full((64, 64, 3), 99, dtype=np.uint8)

    def reset(self, **kwargs):
        return {
            "pixels": {
                "image": self.main.copy(),
                "image2": self.wrist.copy(),
            }
        }, {"reset": True}

    def step(self, actions):
        observation, _ = self.reset()
        return observation, np.array([0.0]), np.array([False]), np.array([False]), {}

    def call(self, name, *args, **kwargs):
        if name == "render":
            return [self.raw_render.copy()]
        return [getattr(self.envs[0], name)]


class _FakeGenerator:
    scene_graph_subject_filter = "target"

    def observe_visual_graph(self, env, *, camera):
        assert env.task_id == 3
        assert camera == "agentview"
        return {
            "camera": camera,
            "bboxes": {
                "target": [4, 28, 12, 36],
                "object": [52, 28, 60, 36],
                "plate_1": [28, 4, 36, 12],
            },
            "relations": [("target", "is_left_of", "object")],
        }


def test_vec_env_wrapper_changes_only_main_image_and_caches_policy_render():
    base_env = _FakeVecEnv()
    wrapper = VisualGraphVecEnvWrapper(base_env, _FakeGenerator())

    observation, info = wrapper.reset()

    assert info == {"reset": True}
    assert not np.array_equal(observation["pixels"]["image"], base_env.main)
    assert np.array_equal(observation["pixels"]["image2"], base_env.wrist)
    cached_render = wrapper.call("render")
    assert len(cached_render) == 1
    assert np.array_equal(cached_render[0], observation["pixels"]["image"][0])
    assert not np.array_equal(cached_render[0], base_env.raw_render)


def test_vec_env_wrapper_overlays_step_observation():
    wrapper = VisualGraphVecEnvWrapper(_FakeVecEnv(), _FakeGenerator())

    observation, reward, terminated, truncated, info = wrapper.step(
        np.zeros((1, 7), dtype=np.float32)
    )

    assert observation["pixels"]["image"].any()
    assert np.array_equal(
        observation["pixels"]["image2"],
        np.full((1, 64, 64, 3), 31, dtype=np.uint8),
    )
    assert reward.tolist() == [0.0]
    assert not terminated.any()
    assert not truncated.any()
    assert info == {}
