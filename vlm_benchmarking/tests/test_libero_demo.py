import json
import threading
from urllib.parse import quote
from urllib.request import urlopen

import h5py
import numpy as np
from PIL import Image

from demo.libero_backend import (
    Hdf5SceneGraphStore,
    Hdf5SemanticCache,
    PredictionStore,
    RendererManager,
    SceneGraphDemoServer,
    SimFrameRenderer,
    duplicate_bowl_invariant_prediction,
    prediction_metrics,
    relation_edges,
    relation_label,
    relation_triplets,
    rotate_bbox_180,
    scale_bbox,
)


def test_bbox_scaling_without_rotation():
    assert scale_bbox([10, 20, 30, 40], (100, 100), (200, 300)) == [20, 60, 60, 120]


def test_bbox_rotation_and_scaling():
    rotated = rotate_bbox_180([10, 20, 30, 40], (100, 80))
    assert rotated == [70, 40, 90, 60]
    assert scale_bbox([10, 20, 30, 40], (100, 80), (200, 160), rotate180=True) == [140, 80, 180, 120]


def test_relation_label_shortening():
    assert relation_label("is_left_of") == "left"
    assert relation_label("is_in_front_of") == "front"
    assert relation_label("is_on_top_of") == "top"
    assert relation_label("contains") == "contains"


def test_relation_edges_filters_subject_and_requires_visible_boxes():
    triplets = [
        ["akita_black_bowl_1", "is_left_of", "plate_1"],
        ["akita_black_bowl_2", "is_right_of", "plate_1"],
        ["akita_black_bowl_1", "is_behind", "missing_1"],
    ]
    bboxes = {
        "akita_black_bowl_1": [0, 0, 10, 10],
        "akita_black_bowl_2": [20, 0, 30, 10],
        "plate_1": [40, 0, 60, 20],
    }

    edges = relation_edges(triplets, bboxes, "akita_black_bowl_1")

    assert len(edges) == 1
    assert edges[0]["label"] == "left"
    assert edges[0]["start"] == [5.0, 5.0]
    assert edges[0]["end"] == [50.0, 10.0]


def test_relation_triplets_marks_visible_and_selected():
    triplets = [
        ["akita_black_bowl_1", "is_left_of", "plate_1"],
        ["akita_black_bowl_2", "is_behind", "missing_1"],
    ]
    bboxes = {
        "akita_black_bowl_1": [0, 0, 10, 10],
        "plate_1": [40, 0, 60, 20],
    }

    items = relation_triplets(triplets, bboxes, "akita_black_bowl_1")

    assert items[0] == {
        "subject": "akita_black_bowl_1",
        "relation": "is_left_of",
        "label": "left",
        "object": "plate_1",
        "visible": True,
        "selected": True,
        "correct": True,
    }
    assert items[1]["visible"] is False
    assert items[1]["selected"] is False


def test_hdf5_metadata_and_overlay_loading(tmp_path):
    data_dir = _write_tiny_hdf5(tmp_path)
    store = Hdf5SceneGraphStore(data_dir, "agentview", rotate_agentview=True)

    assert store.list_tasks()[0]["id"] == "task"
    assert store.list_demos("task") == [{"id": "demo_0", "frame_count": 1}]

    payload = store.frame_payload("task", "demo_0", 0, (512, 512), "akita_black_bowl_1")

    assert payload["objects"] == ["akita_black_bowl_1", "plate_1"]
    assert payload["visible_triplet_count"] == 1
    assert payload["edges"][0]["label"] == "left"
    assert payload["bboxes"][0]["bbox"] == [392.0, 352.0, 472.0, 432.0]
    assert payload["triplets"][0]["label"] == "left"
    assert payload["triplets"][0]["visible"] is True
    assert payload["triplets"][0]["correct"] is True


def test_hdf5_semantic_cache_reuses_decoded_payload_across_store_instances(tmp_path):
    data_dir = _write_tiny_hdf5(tmp_path)
    cache_dir = tmp_path / "semantic-cache"
    first = Hdf5SceneGraphStore(
        data_dir,
        "agentview",
        rotate_agentview=True,
        semantic_cache_dir=cache_dir,
    )

    first.frame_payload("task", "demo_0", 0, (512, 512), "akita_black_bowl_1")
    Hdf5SceneGraphStore._shared_demo_cache.clear()
    second = Hdf5SceneGraphStore(
        data_dir,
        "agentview",
        rotate_agentview=True,
        semantic_cache_dir=cache_dir,
    )
    second._load_demo_data_from_hdf5 = lambda path, demo: (_ for _ in ()).throw(
        AssertionError("semantic cache miss")
    )

    payload = second.frame_payload("task", "demo_0", 0, (512, 512), "akita_black_bowl_1")

    assert payload["visible_triplet_count"] == 1
    assert list(cache_dir.glob("*.pkl"))


def test_hdf5_semantic_cache_invalidates_when_source_hdf5_changes(tmp_path):
    data_dir = _write_tiny_hdf5(tmp_path)
    hdf5_path = data_dir / "task_demo.hdf5"
    cache = Hdf5SemanticCache(tmp_path / "semantic-cache")
    store = Hdf5SceneGraphStore(data_dir, "agentview", semantic_cache_dir=None)
    data = store.demo_data("task", "demo_0")

    cache.put(hdf5_path, "demo_0", {"agentview": data, "eye_in_hand": data})
    assert cache.get(hdf5_path, "demo_0") is not None
    with h5py.File(hdf5_path, "a") as f:
        f.create_dataset("cache_bust", data=np.ones((256,), dtype=np.uint8))

    assert cache.get(hdf5_path, "demo_0") is None


def test_prediction_store_lists_and_loads_jsonl(tmp_path):
    output_dir = tmp_path / "output"
    json_dir = output_dir / "model-a" / "agentview" / "json"
    json_dir.mkdir(parents=True)
    path = json_dir / "task_agentview_v4.jsonl"
    path.write_text(
        json.dumps(
            {
                "task": "task",
                "demo": "demo_0",
                "frame": 0,
                "camera": "agentview",
                "response": "akita_black_bowl_1,is_left_of,plate_1\nplate_1,is_right_of,akita_black_bowl_1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store = PredictionStore(output_dir)

    runs = store.list_runs("task", "agentview")

    assert len(runs) == 1
    assert runs[0]["id"] == "model-a/agentview/json/task_agentview_v4.jsonl"
    assert store.triplets_for_frame(runs[0]["id"], "demo_0", "agentview", 0) == [
        ("akita_black_bowl_1", "is_left_of", "plate_1"),
        ("plate_1", "is_right_of", "akita_black_bowl_1"),
    ]


def test_prediction_frame_payload_marks_wrong_triplets(tmp_path):
    data_dir = _write_tiny_hdf5(tmp_path)
    store = Hdf5SceneGraphStore(data_dir, "agentview", rotate_agentview=True)

    payload = store.frame_payload(
        "task",
        "demo_0",
        0,
        (512, 512),
        "akita_black_bowl_1",
        mode="prediction",
        prediction_triplets=[
            ("akita_black_bowl_1", "is_left_of", "plate_1"),
            ("akita_black_bowl_1", "is_right_of", "plate_1"),
        ],
        prediction_run="model/agentview/json/task_agentview_v4.jsonl",
    )

    assert payload["mode"] == "prediction"
    assert payload["triplet_count"] == 2
    assert payload["gt_triplet_count"] == 1
    assert [edge["correct"] for edge in payload["edges"]] == [True, False]
    assert [triplet["correct"] for triplet in payload["triplets"]] == [True, False]


def test_duplicate_bowl_invariant_prediction_matches_eval_convention():
    gt = {("akita_black_bowl_1", "is_left_of", "plate_1")}
    pred = [("akita_black_bowl_2", "is_left_of", "plate_1")]

    normalized, swapped = duplicate_bowl_invariant_prediction(gt, pred)

    assert swapped is True
    assert normalized == [("akita_black_bowl_1", "is_left_of", "plate_1")]
    assert prediction_metrics(gt, set(normalized)) == {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


def test_prediction_payload_uses_duplicate_bowl_invariant_coloring(tmp_path):
    data_dir = _write_tiny_hdf5(tmp_path, include_second_bowl=True)
    store = Hdf5SceneGraphStore(data_dir, "agentview", rotate_agentview=True)

    payload = store.frame_payload(
        "task",
        "demo_0",
        0,
        (512, 512),
        "akita_black_bowl_1",
        mode="prediction",
        prediction_triplets=[("akita_black_bowl_2", "is_left_of", "plate_1")],
    )

    assert payload["duplicate_bowl_swap_applied"] is True
    assert payload["metrics"]["f1"] == 1.0
    assert payload["triplets"][0]["subject"] == "akita_black_bowl_1"
    assert payload["triplets"][0]["correct"] is True


def test_compare_payload_includes_gt_and_prediction_layers(tmp_path):
    data_dir = _write_tiny_hdf5(tmp_path)
    store = Hdf5SceneGraphStore(data_dir, "agentview", rotate_agentview=True)

    payload = store.frame_payload(
        "task",
        "demo_0",
        0,
        (512, 512),
        "akita_black_bowl_1",
        mode="compare",
        prediction_triplets=[("akita_black_bowl_1", "is_right_of", "plate_1")],
        prediction_available=True,
    )

    assert payload["mode"] == "compare"
    assert payload["gt_edges"][0]["label"] == "left"
    assert payload["edges"][0]["label"] == "right"
    assert payload["edges"][0]["correct"] is False
    assert payload["metrics"]["fp"] == 1


def test_eye_in_hand_overlay_loading_does_not_rotate_bboxes(tmp_path):
    data_dir = _write_tiny_hdf5(tmp_path)
    store = Hdf5SceneGraphStore(data_dir, "eye_in_hand", rotate_agentview=True)

    payload = store.frame_payload("task", "demo_0", 0, (512, 512), "akita_black_bowl_1")

    assert payload["camera"] == "eye_in_hand"
    assert payload["bboxes"][0]["bbox"] == [40.0, 80.0, 120.0, 160.0]
    assert payload["visible_triplet_count"] == 1


def test_api_frame_endpoint_with_mock_renderer(tmp_path):
    data_dir = _write_tiny_hdf5(tmp_path)
    store = Hdf5SceneGraphStore(data_dir, "agentview", rotate_agentview=True)
    server = SceneGraphDemoServer(("127.0.0.1", 0), store, _MockRenderers(), "akita_black_bowl_1")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        with urlopen(
            f"http://127.0.0.1:{port}/api/frame?task=task&demo=demo_0&frame=0&subject=akita_black_bowl_1"
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert payload["image_url"].startswith("/api/image?")
    assert payload["overlay"]["visible_triplet_count"] == 1
    assert payload["overlay"]["target_size"] == {"width": 8, "height": 8}


def test_api_frame_endpoint_can_select_eye_in_hand_camera(tmp_path):
    data_dir = _write_tiny_hdf5(tmp_path)
    stores = {
        "agentview": Hdf5SceneGraphStore(data_dir, "agentview", rotate_agentview=True),
        "eye_in_hand": Hdf5SceneGraphStore(data_dir, "eye_in_hand", rotate_agentview=True),
    }
    renderers = {"agentview": _MockRenderers(), "eye_in_hand": _MockRenderers()}
    server = SceneGraphDemoServer(
        ("127.0.0.1", 0),
        stores,
        renderers,
        "akita_black_bowl_1",
        default_camera="agentview",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        with urlopen(
            f"http://127.0.0.1:{port}/api/frame?camera=eye_in_hand&task=task&demo=demo_0&frame=0&subject=akita_black_bowl_1"
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert payload["overlay"]["camera"] == "eye_in_hand"
    assert payload["overlay"]["bboxes"][0]["bbox"] == [0.625, 1.25, 1.875, 2.5]


def test_api_compare_endpoint_loads_prediction_triplets(tmp_path):
    data_dir = _write_tiny_hdf5(tmp_path)
    output_dir = _write_prediction_jsonl(
        tmp_path,
        "task",
        "agentview",
        "demo_0",
        0,
        "akita_black_bowl_1,is_right_of,plate_1",
    )
    server = SceneGraphDemoServer(
        ("127.0.0.1", 0),
        Hdf5SceneGraphStore(data_dir, "agentview", rotate_agentview=True),
        _MockRenderers(),
        "akita_black_bowl_1",
        predictions=PredictionStore(output_dir),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        run_id = quote("model-a/agentview/json/task_agentview_v4.jsonl", safe="")
        with urlopen(
            f"http://127.0.0.1:{port}/api/frame?task=task&demo=demo_0&frame=0&subject=akita_black_bowl_1&mode=compare&prediction={run_id}"
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert payload["overlay"]["mode"] == "compare"
    assert payload["overlay"]["prediction_available"] is True
    assert payload["overlay"]["edges"][0]["label"] == "right"
    assert payload["overlay"]["edges"][0]["correct"] is False
    assert payload["overlay"]["gt_edges"][0]["label"] == "left"


def test_api_preload_endpoint_with_mock_renderer(tmp_path):
    data_dir = _write_tiny_hdf5(tmp_path, frame_count=3)
    store = Hdf5SceneGraphStore(data_dir, "agentview", rotate_agentview=True)
    renderers = _MockRenderers()
    server = SceneGraphDemoServer(("127.0.0.1", 0), store, renderers, "akita_black_bowl_1")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/api/preload?task=task&demo=demo_0&start=0&end=2&subject=akita_black_bowl_1"
        with urlopen(url) as response:
            first_payload = json.loads(response.read().decode("utf-8"))
        with urlopen(url) as response:
            second_payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert renderers.render_many_calls == 1
    assert len(first_payload["frames"]) == 3
    assert len(second_payload["frames"]) == 3
    assert first_payload["rendered"] == 3
    assert second_payload["cached"] == 3
    assert all(frame["image_url"].startswith("/api/image?") for frame in first_payload["frames"])
    assert [frame["overlay"]["frame"] for frame in first_payload["frames"]] == [0, 1, 2]


def test_api_frame_uses_persistent_disk_render_cache(tmp_path):
    data_dir = _write_tiny_hdf5(tmp_path)
    cache_dir = tmp_path / "cache"
    first_renderer = _MockRenderers()
    first_server = SceneGraphDemoServer(
        ("127.0.0.1", 0),
        Hdf5SceneGraphStore(data_dir, "agentview", rotate_agentview=True),
        first_renderer,
        "akita_black_bowl_1",
        cache_dir=cache_dir,
    )
    first_thread = threading.Thread(target=first_server.serve_forever, daemon=True)
    first_thread.start()
    try:
        port = first_server.server_address[1]
        with urlopen(
            f"http://127.0.0.1:{port}/api/frame?task=task&demo=demo_0&frame=0&subject=akita_black_bowl_1"
        ) as response:
            json.loads(response.read().decode("utf-8"))
    finally:
        first_server.shutdown()
        first_server.server_close()
        first_thread.join(timeout=2)

    second_renderer = _MockRenderers()
    second_server = SceneGraphDemoServer(
        ("127.0.0.1", 0),
        Hdf5SceneGraphStore(data_dir, "agentview", rotate_agentview=True),
        second_renderer,
        "akita_black_bowl_1",
        cache_dir=cache_dir,
    )
    second_thread = threading.Thread(target=second_server.serve_forever, daemon=True)
    second_thread.start()
    try:
        port = second_server.server_address[1]
        with urlopen(
            f"http://127.0.0.1:{port}/api/frame?task=task&demo=demo_0&frame=0&subject=akita_black_bowl_1"
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        second_server.shutdown()
        second_server.server_close()
        second_thread.join(timeout=2)

    assert first_renderer.render_calls == 1
    assert second_renderer.render_calls == 0
    assert payload["image_url"].startswith("/api/image?")


def test_api_image_endpoint_returns_png_and_cache_headers(tmp_path):
    data_dir = _write_tiny_hdf5(tmp_path)
    server = SceneGraphDemoServer(
        ("127.0.0.1", 0),
        Hdf5SceneGraphStore(data_dir, "agentview", rotate_agentview=True),
        _MockRenderers(),
        "akita_black_bowl_1",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        with urlopen(
            f"http://127.0.0.1:{port}/api/frame?task=task&demo=demo_0&frame=0&subject=akita_black_bowl_1"
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        with urlopen(f"http://127.0.0.1:{port}{payload['image_url']}") as response:
            image_bytes = response.read()
            content_type = response.headers.get("Content-Type")
            cache_control = response.headers.get("Cache-Control")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert content_type == "image/png"
    assert "immutable" in cache_control


def test_overlay_changes_reuse_same_image_url_without_rerender(tmp_path):
    data_dir = _write_tiny_hdf5(tmp_path)
    renderer = _MockRenderers()
    server = SceneGraphDemoServer(
        ("127.0.0.1", 0),
        Hdf5SceneGraphStore(data_dir, "agentview", rotate_agentview=True),
        renderer,
        "akita_black_bowl_1",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        with urlopen(
            f"http://127.0.0.1:{port}/api/frame?task=task&demo=demo_0&frame=0&subject=akita_black_bowl_1"
        ) as response:
            first_payload = json.loads(response.read().decode("utf-8"))
        with urlopen(
            f"http://127.0.0.1:{port}/api/frame?task=task&demo=demo_0&frame=0&subject=plate_1&mode=prediction"
        ) as response:
            second_payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert first_payload["image_url"] == second_payload["image_url"]
    assert renderer.render_calls == 1


def test_renderer_manager_prioritizes_interactive_render_over_background_queue(tmp_path):
    manager = RendererManager(res=8)
    fake = _BlockingRenderer()
    manager._renderer = fake
    background_done = threading.Event()
    background_images = []

    def run_background():
        background_images.extend(
            manager.render_many(
                tmp_path / "task_demo.hdf5",
                [np.array([1]), np.array([2])],
                [{}, {}],
                camera="agentview",
                res=8,
                priority=10,
            )
        )
        background_done.set()

    thread = threading.Thread(target=run_background, daemon=True)
    thread.start()
    assert fake.first_background_started.wait(timeout=2)

    interactive_done = threading.Event()
    interactive_result = []

    def run_interactive():
        interactive_result.append(
            manager.render(
                tmp_path / "task_demo.hdf5",
                np.array([9]),
                {},
                camera="agentview",
                res=8,
                priority=0,
            )
        )
        interactive_done.set()

    interactive_thread = threading.Thread(target=run_interactive, daemon=True)
    interactive_thread.start()
    fake.release_first_background.set()

    assert interactive_done.wait(timeout=2)
    assert background_done.wait(timeout=2)
    manager.close_all()

    assert fake.order == [1, 9, 2]
    assert len(background_images) == 2
    assert len(interactive_result) == 1


def test_browser_cache_code_has_decoded_lru_and_no_broad_frame_clear():
    from demo import libero_backend

    html = libero_backend.INDEX_HTML

    assert "decodedImagePixelBudget" in html
    assert "loadImageCached(data.image_url)" in html
    assert "frameCache.clear()" not in html
    assert "Starting simulator" in html
    assert "Preparing playback ${percent}%" in html


def test_prediction_mode_defaults_to_five_fps_and_gemini_pro():
    from demo import libero_backend

    html = libero_backend.INDEX_HTML

    assert 'const preferredPredictionModel = "gemini 3 1 pro";' in html
    assert "function applyPredictionModeDefaults()" in html
    assert "setPlaybackFps(5)" in html
    assert "selectPreferredPredictionRun(true)" in html


def test_demo_defaults_to_1024_resolution():
    from demo import libero_backend

    parser = libero_backend.build_arg_parser()
    args = parser.parse_args([])

    assert args.res == 1024
    assert all(preset["res"] == 1024 for preset in libero_backend.ICLR_PRESETS)
    assert "|| 1024" in libero_backend.INDEX_HTML


def test_sim_frame_renderer_sets_state_without_reset_for_out_of_order_cameras(tmp_path):
    renderer = SimFrameRenderer(tmp_path / "task_demo.hdf5", 8, rotate_agentview=True)
    fake_env = _FakeRenderEnv()
    renderer._get_env = lambda hdf5_path: fake_env

    first = renderer.render_state("agentview", 4, tmp_path / "task_demo.hdf5", np.array([3]))
    second = renderer.render_state("eye_in_hand", 6, tmp_path / "task_demo.hdf5", np.array([1]))

    assert fake_env.reset_calls == 0
    assert fake_env.sim.states == [3, 1]
    assert first.size == (4, 4)
    assert second.size == (6, 6)
    assert np.asarray(first)[0, 0, 0] == 3
    assert np.asarray(second)[0, 0, 0] == 1


def test_sim_frame_renderer_rebuilds_environment_when_task_changes(tmp_path):
    first_env = _FakeRenderEnv()
    second_env = _FakeRenderEnv()
    environments = iter((first_env, second_env))
    renderer = SimFrameRenderer(None, 8, rotate_agentview=True)
    renderer._create_env = lambda hdf5_path: next(environments)
    first_path = tmp_path / "first_task_demo.hdf5"
    second_path = tmp_path / "second_task_demo.hdf5"

    assert renderer._get_env(first_path) is first_env
    renderer._fixed_body_ids["object"] = 1
    assert renderer._get_env(first_path) is first_env
    assert renderer._get_env(second_path) is second_env

    assert first_env.close_calls == 1
    assert renderer.hdf5_path == second_path
    assert renderer._fixed_body_ids == {}
    renderer.close()
    assert second_env.close_calls == 1


class _MockRenderers:
    def __init__(self):
        self.res = 8
        self.render_calls = 0
        self.render_many_calls = 0

    def render(self, hdf5_path, state, fixed_body_positions=None):
        self.render_calls += 1
        return Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

    def render_many(self, hdf5_path, states, fixed_body_positions=None):
        self.render_many_calls += 1
        return [self.render(hdf5_path, state) for state in states]


class _BlockingRenderer:
    def __init__(self):
        self.order = []
        self.first_background_started = threading.Event()
        self.release_first_background = threading.Event()
        self._blocked_once = False

    def render_state(self, camera, res, hdf5_path, state, fixed_body_transforms=None):
        value = int(state[0])
        self.order.append(value)
        if value == 1 and not self._blocked_once:
            self._blocked_once = True
            self.first_background_started.set()
            assert self.release_first_background.wait(timeout=2)
        return Image.fromarray(np.zeros((res, res, 3), dtype=np.uint8))

    def close(self):
        pass


class _FakeRenderEnv:
    def __init__(self):
        self.reset_calls = 0
        self.close_calls = 0
        self.sim = _FakeSim()

    def reset(self):
        self.reset_calls += 1

    def close(self):
        self.close_calls += 1


class _FakeSim:
    def __init__(self):
        self.states = []
        self.current = 0

    def set_state_from_flattened(self, state):
        self.current = int(state[0])
        self.states.append(self.current)

    def forward(self):
        pass

    def render(self, camera_name, height, width, depth=False):
        return np.full((height, width, 3), self.current, dtype=np.uint8)


def _write_tiny_hdf5(tmp_path, frame_count=1, include_second_bowl=False):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "task_demo.hdf5"

    bboxes = [
        {
            "akita_black_bowl_1": [10, 20, 30, 40],
            "plate_1": [50, 60, 70, 80],
        }
        for _ in range(frame_count)
    ]
    if include_second_bowl:
        for frame_bboxes in bboxes:
            frame_bboxes["akita_black_bowl_2"] = [15, 25, 35, 45]
    graph = [[["akita_black_bowl_1", "is_left_of", "plate_1"]] for _ in range(frame_count)]
    world_coords = [
        {
            "akita_black_bowl_1": {"pos": [0.1, 0.2, 0.3]},
            "plate_1": {"pos": [0.4, 0.5, 0.6]},
        }
        for _ in range(frame_count)
    ]

    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("states", data=np.zeros((frame_count, 2)))
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((frame_count, 128, 128, 3), dtype=np.uint8))
        obs.create_dataset("agentview_bboxes", data=json.dumps(bboxes))
        obs.create_dataset("agentview_scene_graph", data=json.dumps(graph))
        obs.create_dataset("agentview_world_coords", data=json.dumps(world_coords))
        obs.create_dataset("eye_in_hand_rgb", data=np.zeros((frame_count, 128, 128, 3), dtype=np.uint8))
        obs.create_dataset("robot0_eye_in_hand_bboxes", data=json.dumps(bboxes))
        obs.create_dataset("robot0_eye_in_hand_scene_graph", data=json.dumps(graph))
        obs.create_dataset("robot0_eye_in_hand_world_coords", data=json.dumps(world_coords))

    return data_dir


def _write_prediction_jsonl(tmp_path, task, camera, demo, frame, response_text):
    output_dir = tmp_path / "output"
    json_dir = output_dir / "model-a" / camera / "json"
    json_dir.mkdir(parents=True)
    path = json_dir / f"{task}_{camera}_v4.jsonl"
    path.write_text(
        json.dumps(
            {
                "task": task,
                "demo": demo,
                "frame": frame,
                "camera": camera,
                "response": response_text,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir
