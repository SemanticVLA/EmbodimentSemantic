"""Small causal contract checks without SAM weights or reference masks."""
from __future__ import annotations

from pathlib import Path
import io
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samgraph_core.automatic_scene import (  # noqa: E402
    AutomaticMaskAcquirer, AutomaticSceneController, CATALOG_BY_NAME,
)
from samgraph_core.geometry_profiles import samgraph_spatial_mask_geometry  # noqa: E402
from samgraph_benchmark.runner import PredictionRunner  # noqa: E402


def _scene():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = {
        "black_bowl": [(4, 4, 10, 10), (15, 4, 21, 10)],
        "cookies": [(4, 22, 11, 29)],
        "plate": [(15, 22, 22, 29)],
        "white_ramekin": [(26, 4, 33, 11)],
        "flat_stove": [(26, 22, 34, 30)],
        "wooden_cabinet": [(44, 22, 56, 35)],
    }
    masks = {}
    for index, (name, areas) in enumerate(boxes.items(), 1):
        masks[name] = []
        for x0, y0, x1, y1 in areas:
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[y0:y1, x0:x1] = True
            rgb[mask] = ((120, 121, 123) if name == "white_ramekin"
                         else (index * 27, index * 19, index * 13))
            masks[name].append(mask)
    return rgb, masks


class _Segmenter:
    def __init__(self, masks):
        self.masks = masks
        self.calls = []

    def segment_candidates(self, *, prompt, jpeg):
        self.calls.append(prompt)
        names = [name for name, spec in CATALOG_BY_NAME.items() if prompt in spec.prompts]
        proposals = []
        for name in names:
            for mask in self.masks[name]:
                stream = io.BytesIO()
                Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(stream, format="PNG")
                proposals.append(SimpleNamespace(png=stream.getvalue(), score=0.9))
        return SimpleNamespace(masks=proposals)


def test_cluster_keeps_strong_score_instead_of_larger_weak_mask():
    rgb, masks = _scene()
    correct = masks["white_ramekin"][0]
    weak = correct.copy()
    weak[11, 26:33] = True
    distractor = np.zeros_like(correct)
    distractor[42:51, 3:12] = True

    class Proposals(_Segmenter):
        def segment_candidates(self, *, prompt, jpeg):
            if prompt not in CATALOG_BY_NAME["white_ramekin"].prompts:
                return super().segment_candidates(prompt=prompt, jpeg=jpeg)
            mask, score = ((correct, .67) if prompt == "silver cylindrical cup" else (weak, .22))
            results = []
            for candidate, confidence in ((mask, score), (distractor, .27)):
                stream = io.BytesIO()
                Image.fromarray(candidate.astype(np.uint8) * 255).save(stream, format="PNG")
                results.append(SimpleNamespace(png=stream.getvalue(), score=confidence))
            return SimpleNamespace(masks=results)

    acquired = AutomaticMaskAcquirer(Proposals(masks)).acquire(rgb)
    selected = acquired["white_ramekin"][0]
    assert np.array_equal(selected.mask, correct)
    assert selected.score == .67
    assert selected.prompt == "silver cylindrical cup"


class _Tracker:
    def __init__(self):
        self.sessions = {}
        self.frame = 0
        self.added = []

    def start(self, *, env_id, seed_mask, **kwargs):
        self.sessions[env_id] = np.array(seed_mask, copy=True)

    def bind_scene(self, members, *, seed_conditioning):
        assert seed_conditioning == "full_mask"
        assert all(member in self.sessions for member in members)

    def submit_frame(self, *, frame_ts, **kwargs):
        self.frame = int(frame_ts)

    def snapshot(self, env_id, *, camera_id):
        return {"native_evidence": {}, "source_frame_ts": self.frame}

    def latest_mask(self, env_id, *, camera_id):
        if "white_ramekin" in env_id and 1 <= self.frame <= 5:
            return None
        return self.sessions[env_id]

    def remove_object(self, env_id, *, camera_id):
        self.sessions.pop(env_id)

    def add_current_object(self, *, env_id, seed_mask, **kwargs):
        self.added.append((env_id, self.frame))
        self.sessions[env_id] = np.array(seed_mask, copy=True)

    def stop(self, env_id, *, camera_id):
        self.sessions.pop(env_id, None)


def test_initial_inventory_and_past_only_persistence():
    rgb, masks = _scene()
    segmenter = _Segmenter(masks)
    tracker = _Tracker()
    controller = AutomaticSceneController(
        segmenter, tracker, geometry_rules=samgraph_spatial_mask_geometry(),
    )
    task = "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"
    first = controller.start_episode(task=task, instruction=task, frame=0, rgb=rgb)
    assert len(first["masks"]) == 7
    assert len(first["states"]) == 7
    assert all(item["status"] == "observed" for item in first["states"])
    assert next(item for item in first["states"]
                if item["class_id"] == "flat_stove")["mask_scope"] == "burner_proxy"
    assert len(segmenter.calls) == sum(len(spec.prompts) for spec in CATALOG_BY_NAME.values())

    second = controller.step(frame=1, rgb=rgb)
    ramekin = next(item for item in second["states"] if item["class_id"] == "white_ramekin")
    assert ramekin["status"] == "persisted"
    assert ramekin["last_observed_frame"] == 0
    assert "white_ramekin_1" not in second["masks"]
    assert "white_ramekin_1" in second["effective_masks"]
    assert len(segmenter.calls) == sum(len(spec.prompts) for spec in CATALOG_BY_NAME.values())

    for frame in range(2, 5):
        controller.step(frame=frame, rgb=rgb)
    sixth = controller.step(frame=5, rgb=rgb)
    ramekin = next(item for item in sixth["states"] if item["class_id"] == "white_ramekin")
    assert ramekin["status"] == "observed"
    assert ramekin["last_observed_frame"] == 5
    assert tracker.added and tracker.added[0][1] == 5
    assert len(segmenter.calls) == (sum(len(spec.prompts) for spec in CATALOG_BY_NAME.values())
                                  + len(CATALOG_BY_NAME["white_ramekin"].prompts))


def test_dense_tracking_emits_only_stride_five_rows(tmp_path):
    archive = tmp_path / "task" / "demo_0.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as output:
        for frame in range(11):
            image = Image.fromarray(
                np.full((16, 16, 3), frame, dtype=np.uint8), mode="RGB",
            )
            stream = io.BytesIO()
            image.save(stream, format="JPEG")
            output.writestr(f"{frame}.jpg", stream.getvalue())
    seen = []

    def predict(rgb, task, frame):
        seen.append(frame)
        return {"triplets": [], "masks": {}}

    rows = PredictionRunner(
        predict, frame_stride=5, tracking_stride=1,
    ).run_archive(archive)
    assert seen == list(range(11))
    assert [row["frame"] for row in rows] == [0, 5, 10]
    assert all(row["error"] is None for row in rows)


def test_dense_tracking_preserves_first_unsampled_error(tmp_path):
    archive = tmp_path / "task" / "demo_0.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as output:
        for frame in range(6):
            stream = io.BytesIO()
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
                stream, format="JPEG",
            )
            output.writestr(f"{frame}.jpg", stream.getvalue())
    seen = []

    def predict(rgb, task, frame):
        seen.append(frame)
        if frame == 1:
            raise RuntimeError("native tracker root cause")
        return {"triplets": [], "masks": {}}

    rows = PredictionRunner(
        predict, frame_stride=5, tracking_stride=1,
    ).run_archive(archive)
    assert seen == [0, 1]
    assert [row["frame"] for row in rows] == [0, 5]
    assert rows[1]["error"] == "raw frame 1: RuntimeError: native tracker root cause"


def test_wrong_object_track_is_quarantined_before_reacquisition():
    rgb, masks = _scene()

    class WrongTracker(_Tracker):
        def latest_mask(self, env_id, *, camera_id):
            if "white_ramekin" in env_id and self.frame == 1:
                return masks["cookies"][0]
            return self.sessions[env_id]

    tracker = WrongTracker()
    controller = AutomaticSceneController(
        _Segmenter(masks), tracker, geometry_rules=samgraph_spatial_mask_geometry(),
    )
    task = "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"
    controller.start_episode(task=task, instruction=task, frame=0, rgb=rgb)
    result = controller.step(frame=1, rgb=rgb)
    ramekin = next(item for item in result["states"] if item["class_id"] == "white_ramekin")
    assert ramekin["status"] == "persisted"
    assert ramekin["native_member"] is None
    assert ramekin["rejection"] in {"appearance_jump", "duplicate_of:cookies_track_1"}
    assert "white_ramekin_1" not in result["masks"]
    assert "white_ramekin_1" in result["effective_masks"]


def test_uniform_fine_tiles_recover_text_mask_without_pixel_hints():
    class FineOnlySegmenter:
        def segment_many(self, *, prompts, jpeg):
            image = Image.open(io.BytesIO(jpeg))
            results = {}
            for prompt in prompts:
                masks = []
                if image.size == (128, 128) and prompt in {"burner", "cooking burner"}:
                    mask = np.zeros((128, 128), dtype=np.uint8)
                    mask[25:55, 25:55] = 255
                    stream = io.BytesIO()
                    Image.fromarray(mask).save(stream, format="PNG")
                    masks.append(SimpleNamespace(png=stream.getvalue(), score=0.8))
                results[prompt] = SimpleNamespace(masks=masks)
            return results

    acquirer = AutomaticMaskAcquirer(FineOnlySegmenter())
    found = acquirer.acquire(np.zeros((256, 256, 3), dtype=np.uint8),
                             classes=("flat_stove",))
    assert len(found["flat_stove"]) == 1
    assert acquirer.diagnostics["flat_stove"]["tile_window_count"] == 20


def test_material_seed_collisions_are_rejected_independent_of_class_order():
    cookie = np.zeros((64, 64), dtype=bool)
    plate = np.zeros_like(cookie)
    cookie[10:30, 10:30] = True
    plate[28:48, 10:30] = True
    segmenter = _Segmenter({"cookies": [cookie], "plate": [plate]})
    for classes in (("cookies", "plate"), ("plate", "cookies")):
        found = AutomaticMaskAcquirer(segmenter).acquire(
            np.zeros((64, 64, 3), dtype=np.uint8), classes=classes,
        )
        assert all(not masks for masks in found.values())


def test_both_bowls_emit_before_motion_then_receive_causal_names():
    rgb, masks = _scene()

    class MovingTracker(_Tracker):
        def latest_mask(self, env_id, *, camera_id):
            if "black_bowl_track_2" in env_id:
                return np.roll(self.sessions[env_id], self.frame + 2, axis=0)
            return self.sessions[env_id]

    tracker = MovingTracker()
    controller = AutomaticSceneController(
        _Segmenter(masks), tracker, geometry_rules=samgraph_spatial_mask_geometry(),
    )
    first = controller.start_episode(task="scene", instruction="pick up a bowl",
                                     frame=0, rgb=rgb)
    assert len(first["masks"]) == 7
    assert {"black_bowl_track_1", "black_bowl_track_2"} <= first["masks"].keys()
    initial = controller.objects["black_bowl_track_2"].last_mask.copy()
    for frame in range(1, 4):
        current = rgb.copy()
        current[initial] = 0
        current[np.roll(initial, frame + 2, axis=0)] = rgb[initial][0]
        result = controller.step(frame=frame, rgb=current)
        if frame < 3:
            assert "black_bowl_track_2" in result["masks"]
    assert {"black_bowl_1", "black_bowl_2"} <= result["masks"].keys()
    moving = next(s for s in result["states"] if s["track_id"] == "black_bowl_track_2")
    assert moving["semantic_id"] == "black_bowl_1"
    assert moving["identity_assigned_frame"] == 3
    assert all(s["semantic_id"] is None for s in first["states"] if s["class_id"] == "black_bowl")


def test_wrong_class_full_frame_candidate_does_not_suppress_tile_search():
    class Segmenter:
        def segment_many(self, *, prompts, jpeg):
            size = Image.open(io.BytesIO(jpeg)).size[0]
            results = {}
            for prompt in prompts:
                masks = []
                if prompt in {"burner", "white dinner plate"}:
                    mask = np.zeros((size, size), dtype=np.uint8)
                    if size == 256:
                        mask[10:40, 10:40] = 255
                    elif prompt == "burner":
                        mask[70:100, 70:100] = 255
                    if mask.any():
                        stream = io.BytesIO()
                        Image.fromarray(mask).save(stream, format="PNG")
                        masks.append(SimpleNamespace(png=stream.getvalue(),
                            score=0.95 if prompt == "white dinner plate" else 0.7))
                results[prompt] = SimpleNamespace(masks=masks)
            return results
    acquirer = AutomaticMaskAcquirer(Segmenter())
    found = acquirer.acquire(np.zeros((256, 256, 3), dtype=np.uint8),
                             classes=("plate", "flat_stove"))
    assert len(found["plate"]) == len(found["flat_stove"]) == 1
    assert acquirer.diagnostics["flat_stove"]["tile_window_count"] == 4


def test_occlusion_fragment_does_not_replace_past_shape_memory():
    from samgraph_core.automatic_scene import ObjectMemory, MaskProposal
    rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
    item = ObjectMemory("cup", "white_ramekin", "support", "white_ramekin_1")
    for frame, width in enumerate((20, 15, 10, 2)):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 10:10+width] = True
        AutomaticSceneController._remember(
            item, MaskProposal("white_ramekin", mask, "ribbed silver cup", 0.9),
            rgb, frame, "native_tracking")
    assert item.last_mask.sum() == 40
    assert item.last_observed_frame == 3
    assert item.persistence_mask.sum() == 400
    assert item.persistence_frame == 0
    controller = AutomaticSceneController.__new__(AutomaticSceneController)
    controller.objects = {item.track_id: item}
    controller._last_frame = 4
    assert controller._quality(item, item.persistence_mask, rgb, {}) is None
    assert controller._quality(item, item.last_mask, rgb, {}) == "area_jump"


def test_explicit_names_are_used_without_pixel_hints():
    queried = []
    class Segmenter:
        def segment_many(self, *, prompts, jpeg):
            queried.extend(prompts)
            return {name: SimpleNamespace(masks=[]) for name in prompts}
    names = {name: [f"description of {name}"] for name in CATALOG_BY_NAME}
    acquirer = AutomaticMaskAcquirer(Segmenter(), prompts_by_class=names)
    acquirer.acquire(np.zeros((32, 32, 3), dtype=np.uint8), multiscale=False)
    assert queried == [values[0] for values in names.values()]
    assert acquirer.diagnostics["black_bowl"]["prompt_list"] == names["black_bowl"]
    assert CATALOG_BY_NAME["black_bowl"].prompts[0] == "ceramic bowl with patterned interior"


def test_default_task_configuration_preserves_verified_prompt_order():
    import json
    path = Path(__file__).resolve().parents[1] / "config" / "libero_spatial_object_prompts.json"
    configured = json.loads(path.read_text())
    assert len(configured) == 10
    task_specific = "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
    for task, names in configured.items():
        acquirer = AutomaticMaskAcquirer(None, prompts_by_class=names)
        assert set(acquirer.catalog) == set(CATALOG_BY_NAME)
        for class_id, concept in CATALOG_BY_NAME.items():
            configured_concept = acquirer.catalog[class_id]
            assert (configured_concept.name, configured_concept.role, configured_concept.count) == (
                concept.name, concept.role, concept.count
            )
            if task != task_specific or class_id != "white_ramekin":
                assert configured_concept.prompts == concept.prompts


def test_gray_ribbed_cup_prompt_is_scoped_to_on_ramekin_task():
    import json
    path = Path(__file__).resolve().parents[1] / "config" / "libero_spatial_object_prompts.json"
    configured = json.loads(path.read_text())
    target = "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
    assert configured[target]["white_ramekin"] == [
        "ribbed silver cup", "ribbed metal container", "silver cylindrical cup",
        "gray ribbed cup",
    ]
    assert all(
        "gray ribbed cup" not in names["white_ramekin"]
        for task, names in configured.items() if task != target
    )


def test_observed_moving_bowl_can_be_named_when_other_bowl_is_occluded():
    from samgraph_core.automatic_scene import ObjectMemory, MaskProposal
    rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
    controller = AutomaticSceneController.__new__(AutomaticSceneController)
    controller.objects = {}
    for index, x in enumerate((5, 35), 1):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:20, x:x+10] = True
        item = ObjectMemory(f"black_bowl_track_{index}", "black_bowl", "bowl", None)
        controller._remember(item, MaskProposal("black_bowl", mask, "bowl", .9), rgb, 0, "text_initialization")
        controller.objects[item.track_id] = item
    stationary, moving = controller.objects.values()
    stationary.status = "persisted"
    moving.last_mask = np.roll(moving.last_mask, 10, axis=0)
    for frame in (1, 2, 3):
        moving.last_observed_frame = frame
        controller._name_moving_bowl(frame)
        if frame < 3:
            assert moving.semantic_id is None
    assert moving.semantic_id == "black_bowl_1"
    assert stationary.semantic_id == "black_bowl_2"
    assert moving.identity_assigned_frame == 3


def test_lone_reacquisition_cannot_steal_another_occluded_bowls_identity():
    from samgraph_core.automatic_scene import ObjectMemory, MaskProposal
    rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
    controller = AutomaticSceneController.__new__(AutomaticSceneController)
    controller.objects = {}
    for index, x in enumerate((10, 70), 1):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:30, x:x+10] = True
        item = ObjectMemory(f"bowl_track_{index}", "black_bowl", "bowl", None)
        controller._remember(item, MaskProposal("black_bowl", mask, "bowl", .9),
                             rgb, 0, "text_initialization")
        item.status = "persisted"
        controller.objects[item.track_id] = item
    first, second = controller.objects.values()
    candidate = MaskProposal("black_bowl", np.roll(second.last_mask, 3, axis=0), "bowl", .9)
    # Both bowls are occluded in memory; one proposal exists. Only its causal
    # owner may claim it, regardless of iteration order or semantic bowl names.
    assert not controller._owns_reacquisition(first, candidate, rgb)
    assert controller._owns_reacquisition(second, candidate, rgb)
    middle = np.zeros((100, 100), dtype=bool)
    middle[20:30, 40:50] = True
    ambiguous = MaskProposal("black_bowl", middle, "bowl", .9)
    assert not controller._owns_reacquisition(first, ambiguous, rgb)
    assert not controller._owns_reacquisition(second, ambiguous, rgb)


def test_recovery_binds_lone_candidate_to_its_owner_not_first_lost_track():
    from samgraph_core.automatic_scene import MaskProposal, _encode_rgb
    rgb, masks = _scene()
    tracker = _Tracker()
    controller = AutomaticSceneController(
        _Segmenter(masks), tracker, geometry_rules=samgraph_spatial_mask_geometry())
    controller.start_episode(task="scene", instruction="pick up a bowl", frame=0, rgb=rgb)
    first = controller.objects["black_bowl_track_1"]
    second = controller.objects["black_bowl_track_2"]
    for item in (first, second):
        controller._remove_native(item)
        item.status = "persisted"
    proposal = MaskProposal("black_bowl", second.last_mask.copy(), "bowl", .9)
    controller.acquirer = SimpleNamespace(
        diagnostics={}, acquire=lambda *args, **kwargs: {"black_bowl": [proposal]})
    controller._last_frame = tracker.frame = 5
    controller._recover_one(rgb, 5, _encode_rgb(rgb))
    assert first.status == "persisted"
    assert first.native_member is None
    assert first.rejection == "ambiguous_reacquisition_identity"
    assert second.status == "observed"
    assert second.method == "text_reacquisition"
    assert len(tracker.added) == 1
    assert tracker.added[0][0].endswith("black_bowl_track_2")
