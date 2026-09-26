"""CPU checks for the SO101 adapter and agent-view scene specialization."""
from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samgraph_so101.cli import _episode_selection, _tasks  # noqa: E402
from samgraph_so101.config import load_so101_config  # noqa: E402
from samgraph_so101.scene import (  # noqa: E402
    SO101_SINGLE_CLASS_RETRY_MIN_SCORE,
    SO101AgentSceneController,
    SO101RolloverSceneTracker,
    _SO101SingleClassScoreGate,
    install_so101_catalog,
)


CONFIG = Path(__file__).resolve().parents[1] / "config" / "so101_objects.json"


def test_so101_config_declares_real_inventory_and_2d_camera_contract():
    config = load_so101_config(CONFIG)
    assert config.object_ids == (
        "black_bowl", "cookie_packet", "white_plate", "red_drawer", "black_stove",
    )
    assert len(config.tasks) == 5
    assert config.camera_convention["camera"] == "agent_view"
    assert config.camera_convention["depth_claim"] is False
    assert config.camera_convention["physical_contact_claim"] is False
    assert config.geometry["direction_horizontal_sign"] == -1
    assert config.persistence["manipulated_object_ids"] == ["black_bowl"]
    assert config.persistence["max_remembered_native_frames"] == 5
    prompts = config.prompts_for_tasks(list(config.tasks))
    drawer_task = next(iter(config.tasks))
    assert "open black bowl resting on a pink drawer organizer" in (
        prompts[drawer_task]["black_bowl"]
    )
    assert "white saucer" in prompts[drawer_task]["white_plate"]
    assert "small white and orange snack packet" in prompts[drawer_task]["cookie_packet"]


def test_task_and_episode_selection_reject_missing_or_duplicate_values():
    available = ["a", "b", "c"]
    assert _tasks("all", available) == available
    assert _tasks("2,0", available) == ["c", "a"]
    assert _episode_selection("1:3", (0, 1, 2, 4)) == (1, 2)
    with pytest.raises(FileNotFoundError, match="3"):
        _episode_selection("1:4", (0, 1, 2, 4))
    with pytest.raises(ValueError, match="duplicate"):
        _tasks("a,a", available)


class Segmenter:
    def __init__(self, masks, prompts):
        self.masks = masks
        self.prompts = prompts

    def segment_candidates(self, *, prompt, jpeg):
        class_id = next((key for key, values in self.prompts.items() if prompt in values), None)
        if class_id is None:
            return SimpleNamespace(masks=[])
        stream = io.BytesIO()
        Image.fromarray(self.masks[class_id].astype(np.uint8) * 255).save(stream, format="PNG")
        return SimpleNamespace(masks=[SimpleNamespace(png=stream.getvalue(), score=0.9)])


def test_so101_single_class_retry_score_gate_preserves_joint_acquisition():
    from samgraph_core.automatic_scene import MaskProposal

    mask = np.ones((4, 4), dtype=bool)

    class Acquirer:
        diagnostics = {}

        def acquire(self, rgb, *, classes=None, multiscale=True):
            del rgb, multiscale
            proposals = [
                MaskProposal("white_plate", mask, "white saucer", 0.508),
                MaskProposal("white_plate", mask, "white plate", 0.75),
            ]
            return {"white_plate": proposals}

    gate = _SO101SingleClassScoreGate(Acquirer())
    joint = gate.acquire(np.zeros((4, 4, 3), dtype=np.uint8))
    assert len(joint["white_plate"]) == 2
    isolated = gate.acquire(
        np.zeros((4, 4, 3), dtype=np.uint8), classes=("white_plate",),
    )
    assert [proposal.score for proposal in isolated["white_plate"]] == [0.75]
    rejection = gate.last_rejected_candidates["white_plate"][0]
    assert rejection["sam_score"] == 0.508
    assert rejection["min_sam_score"] == SO101_SINGLE_CLASS_RETRY_MIN_SCORE


class Tracker:
    def __init__(self):
        self.masks = {}
        self.frame = 0

    def start(self, *, env_id, seed_mask, **kwargs):
        self.masks[env_id] = np.asarray(seed_mask, dtype=bool).copy()

    def bind_scene(self, members, *, seed_conditioning):
        assert seed_conditioning == "full_mask"

    def submit_frame(self, *, frame_ts, **kwargs):
        self.frame = int(frame_ts)

    def snapshot(self, env_id, *, camera_id):
        return {"native_evidence": {}, "source_frame_ts": self.frame}

    def latest_mask(self, env_id, *, camera_id):
        return self.masks[env_id]

    def remove_object(self, env_id, *, camera_id):
        self.masks.pop(env_id, None)

    def add_current_object(self, *, env_id, seed_mask, **kwargs):
        self.masks[env_id] = np.asarray(seed_mask, dtype=bool).copy()

    def stop(self, env_id, *, camera_id):
        self.masks.pop(env_id, None)


def test_so101_scene_uses_canonical_ids_and_native_dense_tracking():
    from samgraph_core import automatic_scene

    old_catalog = automatic_scene.AUTOMATIC_CATALOG
    old_by_name = automatic_scene.CATALOG_BY_NAME
    try:
        config = load_so101_config(CONFIG)
        install_so101_catalog(config)
        prompts = config.prompts_for_tasks(list(config.tasks))
        rgb = np.zeros((80, 100, 3), dtype=np.uint8)
        masks = {}
        for index, object_id in enumerate(config.object_ids):
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            x0 = 3 + index * 18
            mask[10:20, x0:x0 + 10] = True
            masks[object_id] = mask
            rgb[mask] = (30 + index * 30, 20 + index * 20, 10 + index * 10)
        task = next(iter(config.tasks))
        controller = SO101AgentSceneController(
            Segmenter(masks, prompts[task]), Tracker(), config=config,
            task_prompts={task: prompts[task]},
        )
        first = controller.start_episode(
            task=f"{task}/episode_0", instruction=config.tasks[task].instruction,
            frame=0, rgb=rgb,
        )
        assert set(first["masks"]) == set(config.object_ids)
        assert {item["output_id"] for item in first["states"]} == set(config.object_ids)
        assert all(item["identity_method"] == "class_name" for item in first["states"])
        second = controller.step(frame=1, rgb=rgb)
        assert set(second["masks"]) == set(config.object_ids)
        assert all(item["method"] == "native_tracking" for item in second["states"])
        controller.close_episode()
    finally:
        automatic_scene.AUTOMATIC_CATALOG = old_catalog
        automatic_scene.CATALOG_BY_NAME = old_by_name


def test_config_rejects_depth_claim(tmp_path):
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["camera_convention"]["depth_claim"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="disable depth claims"):
        load_so101_config(path)


def test_so101_manipulated_object_memory_expires_without_stale_geometry():
    from samgraph_core.automatic_scene import ObjectMemory

    config = load_so101_config(CONFIG)
    controller = SO101AgentSceneController.__new__(SO101AgentSceneController)
    controller.so101_config = config
    controller.manipulated_object_ids = frozenset(
        config.persistence["manipulated_object_ids"]
    )
    controller.max_remembered_native_frames = int(
        config.persistence["max_remembered_native_frames"]
    )
    recent = np.zeros((20, 20), dtype=bool)
    recent[10:14, 12:16] = True
    old = np.zeros_like(recent)
    old[2:8, 2:8] = True
    item = ObjectMemory(
        "black_bowl_track_1", "black_bowl", "bowl", "black_bowl",
        last_mask=recent, last_observed_frame=10,
        persistence_mask=old, persistence_frame=0,
        status="persisted",
    )

    mask, source, status, method = controller._effective_mask(item, frame=15)
    assert np.array_equal(mask, recent)
    assert source == 10
    assert status == "persisted"
    assert method == "short_horizon_latest_observation"

    mask, source, status, method = controller._effective_mask(item, frame=16)
    assert mask is None
    assert source is None
    assert status == "unresolved"
    assert method == "expired_manipulated_object_memory"


def test_so101_adapter_separates_tracking_and_reacquisition_references():
    from samgraph_core.automatic_scene import ObjectMemory

    config = load_so101_config(CONFIG)
    controller = SO101AgentSceneController.__new__(SO101AgentSceneController)
    latest = np.zeros((40, 40), dtype=bool)
    latest[20:24, 20:24] = True
    anchor = np.zeros_like(latest)
    anchor[2:10, 2:10] = True
    item = ObjectMemory(
        "black_bowl_track_1", "black_bowl", "bowl", "black_bowl",
        last_mask=latest, persistence_mask=anchor,
        descriptor=np.zeros(24, dtype=np.float32),
        persistence_descriptor=np.ones(24, dtype=np.float32),
    )
    controller.manipulated_object_ids = frozenset({"black_bowl"})
    reference, descriptor = controller._temporal_reference(item)
    assert np.array_equal(reference, latest)
    assert np.array_equal(descriptor, item.descriptor)

    reference, descriptor = controller._reacquisition_reference(item)
    assert np.array_equal(reference, anchor)
    assert np.array_equal(descriptor, item.persistence_descriptor)


def test_so101_reacquisition_accepts_clean_bowl_after_occluded_latest_template():
    from samgraph_core.automatic_scene import ObjectMemory, _appearance

    controller = SO101AgentSceneController.__new__(SO101AgentSceneController)
    controller.manipulated_object_ids = frozenset({"black_bowl"})
    controller.objects = {}
    controller._last_frame = 20
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    rgb[4:14, 4:14] = (20, 20, 20)
    rgb[20:30, 20:30] = (20, 20, 20)
    clean = np.zeros((40, 40), dtype=bool)
    clean[4:14, 4:14] = True
    reacquired = np.zeros_like(clean)
    reacquired[20:30, 20:30] = True
    occluded = np.zeros_like(clean)
    occluded[4:14, 4:9] = True
    rgb[4:14, 4:9] = (240, 20, 20)
    item = ObjectMemory(
        "black_bowl_track_1", "black_bowl", "bowl", "black_bowl",
        last_mask=occluded,
        persistence_mask=clean,
        descriptor=_appearance(rgb, occluded),
        persistence_descriptor=np.concatenate([
            np.array([100, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            np.array([100, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            np.array([100, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        ]) / 300.0,
    )
    controller.objects[item.track_id] = item

    assert controller._quality(item, reacquired, rgb, {}) == "appearance_jump"
    assert controller._quality(
        item, reacquired, rgb, {}, reacquisition=True,
    ) is None


def test_so101_rollover_preserves_members_and_advances_current_frame(monkeypatch):
    """Exercise the SO101 rollover boundary without loading SAM or a GPU."""
    from samgraph_core.sam31_live_scene import LocalSam31SceneTracker, _Scene

    def fake_start(self, *, env_id, query, frame_ts, jpeg, seed_mask, camera_id, **kwargs):
        self._sessions[(env_id, camera_id)] = SimpleNamespace(
            query=query,
            frame_ts=float(frame_ts),
            previous_rgb=bytes(jpeg),
            last_mask=np.array(seed_mask, dtype=bool, copy=True),
            snapshot={"status": "observed", "mask_present": True},
        )

    serial = {"value": 0}

    def fake_bind(self, members, *, seed_conditioning):
        serial["value"] += 1
        first = self._sessions[(members[0], self.camera_id)]
        scene = _Scene(
            f"session-{serial['value']}", list(members), first.frame_ts,
            hashlib.sha256(first.previous_rgb).hexdigest(),
            seed_conditioning=seed_conditioning,
            object_ids={member: index for index, member in enumerate(members, 1)},
            max_object_id=len(members),
        )
        for member in members:
            self._scenes[member] = scene

    def fake_stop(self, env_id=None, camera_id=None):
        affected = list(self._scenes) if env_id is None else [str(env_id)]
        for member in affected:
            scene = self._scenes.get(member)
            if scene is None:
                continue
            for peer in list(scene.members):
                self._scenes.pop(peer, None)
                self._sessions.pop((peer, self.camera_id), None)

    def fake_submit(self, *, env_id, camera_id, frame_ts, jpeg):
        scene = self._scenes[env_id]
        scene.frame_ts = float(frame_ts)
        scene.digest = hashlib.sha256(jpeg).hexdigest()
        for member in scene.members:
            session = self._sessions[(member, camera_id)]
            session.frame_ts = float(frame_ts)
            session.last_mask = np.array(session.last_mask, copy=True)
            session.snapshot.update(
                native_session_id=scene.session_id,
                native_frame_index=1,
                status="observed",
                mask_present=True,
            )
        return True

    monkeypatch.setattr(LocalSam31SceneTracker, "start", fake_start)
    monkeypatch.setattr(LocalSam31SceneTracker, "bind_scene", fake_bind)
    monkeypatch.setattr(LocalSam31SceneTracker, "stop", fake_stop)
    monkeypatch.setattr(LocalSam31SceneTracker, "submit_frame", fake_submit)

    tracker = SO101RolloverSceneTracker(
        object(), camera_id="agent_view", max_native_session_frames=2,
    )
    mask_a = np.zeros((8, 8), dtype=bool)
    mask_b = np.zeros((8, 8), dtype=bool)
    mask_a[1:3, 1:3] = True
    mask_b[5:7, 5:7] = True
    for member, mask in (("a", mask_a), ("b", mask_b)):
        tracker.start(
            env_id=member, query=member, frame_ts=0.0, jpeg=b"frame-0",
            seed_mask=mask, camera_id="agent_view",
        )
    tracker.bind_scene(["a", "b"], seed_conditioning="full_mask")
    tracker.submit_frame(
        env_id="a", camera_id="agent_view", frame_ts=1.0, jpeg=b"frame-1",
    )
    tracker.submit_frame(
        env_id="a", camera_id="agent_view", frame_ts=2.0, jpeg=b"frame-2",
    )

    provenance = tracker.rollover_provenance()
    assert provenance["native_session_rollover_count"] == 1
    assert provenance["frames_in_native_session"] == 2
    assert provenance["total_episode_frames_processed"] == 3
    assert provenance["last_rollover"]["previous_native_session_id"] == "session-1"
    assert provenance["last_rollover"]["new_native_session_id"] == "session-2"
    assert provenance["last_rollover"]["at_previous_frame_ts"] == 1.0
    assert set(tracker._scenes["a"].members) == {"a", "b"}
    assert tracker._sessions[("a", "agent_view")].frame_ts == 2.0
