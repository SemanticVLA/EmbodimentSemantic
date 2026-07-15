import json
import threading
from pathlib import Path
from urllib.request import urlopen

import h5py

from demo.libero_backend import PredictionStore
from demo.server import UnifiedDemoServer
from demo.so101_backend import DemoRepository
from demo.so101_proxy_demo.proxy.artifacts import ArtifactStore
from demo.so101_proxy_demo.proxy.config import load_config


class _LiberoStore:
    camera = "agentview"

    def list_tasks(self):
        return [{"id": "task", "name": "task"}]


class _Renderer:
    res = 8

    def close_all(self):
        return None


class _So101Repository:
    def health(self):
        return {"name": "SO101", "proxy_frames": 1}


def test_unified_demo_serves_selector_apps_and_both_api_namespaces(tmp_path):
    server = UnifiedDemoServer(
        ("127.0.0.1", 0),
        _LiberoStore(),
        _Renderer(),
        "black_bowl",
        cache_dir=tmp_path / "cache",
        so101_repository=_So101Repository(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urlopen(base + "/") as response:
            portal = response.read().decode("utf-8")
        with urlopen(base + "/libero/") as response:
            libero = response.read().decode("utf-8")
        with urlopen(base + "/so101/") as response:
            so101 = response.read().decode("utf-8")
        with urlopen(base + "/api/tasks") as response:
            libero_api = json.loads(response.read().decode("utf-8"))
        with urlopen(base + "/so101/api/health") as response:
            so101_api = json.loads(response.read().decode("utf-8"))
        with urlopen(base + "/common/demo_common.js") as response:
            common_js = response.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert "liberoTab" in portal and "so101Tab" in portal
    assert "Static" not in portal and "sample" not in portal.lower()
    assert "LIBERO Demo" in libero
    assert "SO101 Demo" in so101
    assert "window.DemoCommon" in common_js
    assert "common.scopedApiRoot" in (Path(__file__).parents[1] / "demo" / "so101" / "app.js").read_text(encoding="utf-8")
    assert libero_api["tasks"][0]["id"] == "task"
    assert so101_api["proxy_frames"] == 1


def test_deployment_cache_contains_no_libero_images():
    root = Path(__file__).parents[1] / "demo"
    assert not (root / "data").exists()
    assert not (root / "static_server.py").exists()
    assert not (root / "build_static_bundle.py").exists()
    cache = root / "libero_demo_cache"
    files = list(cache.glob("*.hdf5"))
    assert len(files) == 10
    assert not any(path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for path in cache.rglob("*"))
    for path in files:
        with h5py.File(path, "r") as handle:
            assert list(handle["data"]) == ["demo_0"]
            assert handle["data/demo_0/states"].shape[0] > 0
            assert handle["data/demo_0/obs/agentview_rgb"].id.get_storage_size() == 0
            assert handle["data/demo_0/obs/eye_in_hand_rgb"].id.get_storage_size() == 0


def test_deployment_caches_keep_predictions_and_so101_metrics():
    root = Path(__file__).parents[1] / "demo"
    task = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
    predictions = PredictionStore(root / "libero_prediction_cache")
    runs = predictions.list_runs(task, "agentview")
    assert len(runs) == 11
    assert predictions.has_frame(runs[0]["id"], "demo_0", "agentview", 0)

    config = load_config(root / "so101_demo_cache/config.yaml")
    repository = DemoRepository(config, ArtifactStore(config["paths"]["artifacts"]))
    so_task = repository.tasks()[0]["id"]
    frame = repository.frames_for(so_task, "episode_0", "agent_view")[0]
    payload = repository.frame_payload(so_task, "episode_0", frame, "agent_view", "gt")
    assert payload["metrics"]["tp"] == 16
    assert payload["metrics"]["fp"] == 4
    assert payload["metrics"]["fn"] == 4
    assert round(payload["metrics"]["f1"], 2) == 0.80
