import json
import threading
import tomllib
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
        with urlopen(base + "/index.html") as response:
            legacy_portal = response.read().decode("utf-8")
            legacy_portal_url = response.geturl()
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
    assert legacy_portal == portal
    assert legacy_portal_url == base + "/"
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


def test_bundled_libero_cache_is_complete():
    cache_dir = Path(__file__).parents[1] / "demo" / "libero_frame_cache"
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["complete"] is True
    assert manifest["resolution"] == 1024
    assert manifest["cached_frames"] == manifest["expected_frames"] == 2542
    assert len(manifest["archives"]) == 20
    assert sum(record["frame_count"] for record in manifest["archives"].values()) == 2542
    assert all((cache_dir / record["path"]).is_file() for record in manifest["archives"].values())
    assert sum(key.startswith("agentview/") for key in manifest["archives"]) == 10
    assert sum(key.startswith("eye_in_hand/") for key in manifest["archives"]) == 10


def test_docker_defaults_to_cached_production_and_keeps_optional_libero_runtime():
    dockerfile = (Path(__file__).parents[2] / "Dockerfile").read_text(encoding="utf-8")
    requirements = (Path(__file__).parents[1] / "requirements-demo.txt").read_text(encoding="utf-8")
    cached_requirements = (Path(__file__).parents[1] / "requirements-demo-cached.txt").read_text(
        encoding="utf-8"
    )
    assert any(
        line.startswith("FROM python:3.12-slim-trixie")
        for line in dockerfile.splitlines()[:2]
    )
    assert "FROM cached-deps AS app-base" in dockerfile
    assert "FROM libero-deps AS libero-runtime" in dockerfile
    assert "FROM app-base AS production" in dockerfile
    assert "libglib2.0-0t64" in dockerfile
    assert "libgomp1" in dockerfile
    assert "libice6" in dockerfile
    assert "libsm6" in dockerfile
    assert "PYTHONPATH=/opt/LIBERO" in dockerfile
    assert "pip install --no-deps --editable /opt/LIBERO" in dockerfile
    assert "--mount=type=cache,target=/root/.cache/pip" in dockerfile
    assert "--constraint /tmp/constraints-demo.txt" in dockerfile
    assert "python -m pip check" in dockerfile
    assert "from libero.libero import get_libero_path" in dockerfile
    assert "from libero.libero.envs import OffScreenRenderEnv" in dockerfile
    assert "RUN python -m demo.deployment_smoke" in dockerfile
    assert "python -m demo.deployment_smoke --cached-only" in dockerfile
    assert '"--bundled-cache-dir", "demo/libero_frame_cache", "--cached-only"' in dockerfile
    assert "termcolor==3.3.0" in requirements
    assert "future==0.18.2" in requirements
    assert "hydra-core==1.3.2" in requirements
    assert "opencv-python==4.13.0.92" in requirements
    assert "opencv-python-headless" not in requirements
    assert "h5py==3.16.0" in cached_requirements
    assert "mujoco" not in cached_requirements.lower()
    assert "robosuite" not in cached_requirements.lower()


def test_fly_uses_lightweight_cached_production_target():
    with (Path(__file__).parents[2] / "fly.toml").open("rb") as handle:
        config = tomllib.load(handle)

    assert config["build"]["target"] == "production"
    vm = config["vm"][0]
    assert vm["cpu_kind"] == "shared"
    assert vm["cpus"] == 1
    assert vm["memory"] == "1gb"


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
