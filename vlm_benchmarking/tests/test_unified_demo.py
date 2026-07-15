import json
import threading
from pathlib import Path
from urllib.request import urlopen

from demo.server import UnifiedDemoServer
from demo.static_server import DemoStaticRequestHandler


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


def test_dependency_free_server_uses_browser_safe_static_mime_types():
    assert DemoStaticRequestHandler.extensions_map[".webp"] == "image/webp"
    assert DemoStaticRequestHandler.extensions_map[".json"].startswith("application/json")


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
    assert "LIBERO Demo" in libero
    assert "SO101 Demo" in so101
    assert "window.DemoCommon" in common_js
    assert "common.scopedApiRoot" in (Path(__file__).parents[1] / "demo" / "so101" / "app.js").read_text(encoding="utf-8")
    assert libero_api["tasks"][0]["id"] == "task"
    assert so101_api["proxy_frames"] == 1


def test_github_pages_bundles_reference_existing_images_and_visible_endpoints():
    demo_root = Path(__file__).parents[1] / "demo"
    expected = {
        "libero": {"tasks": 10, "cameras": {"agentview", "eye_in_hand"}, "sequences": 20},
        "so101": {"tasks": 5, "cameras": {"agent_view", "wrist"}, "sequences": 10},
    }
    for name in ("libero", "so101"):
        bundle = json.loads((demo_root / "data" / f"{name}.json").read_text(encoding="utf-8"))
        assert bundle["dataset"] == name
        assert bundle["schema_version"] == 2
        assert len({item["task"] for item in bundle["sequences"]}) == expected[name]["tasks"]
        assert {item["camera"] for item in bundle["sequences"]} == expected[name]["cameras"]
        assert len(bundle["sequences"]) == expected[name]["sequences"]
        assert bundle["frame_count"] == sum(
            len(item["frames"]) for item in bundle["sequences"]
        )
        for sequence in bundle["sequences"]:
            assert 1 <= len(sequence["frames"]) <= 12
            if name == "libero":
                assert sequence["sequence"] == "demo_0"
            else:
                assert sequence["sequence"] == "episode_0"
            for frame in sequence["frames"]:
                image = demo_root / frame["image"]
                assert image.is_file()
                assert image.suffix == ".webp"
                assert image.is_relative_to(demo_root / "data" / name)
                bbox_names = {item["object"] for item in frame["bboxes"]}
                assert all(
                    relation["subject"] in bbox_names and relation["object"] in bbox_names
                    for relation in frame["relations"]
                )

    manifest = json.loads((demo_root / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["image_format"] == "WebP"
    assert manifest["total_bytes"] == sum(
        path.stat().st_size for path in (demo_root / "data").rglob("*") if path.is_file()
    )
    assert manifest["total_bytes"] < 50 * 1024 * 1024
    assert not (demo_root / "assets").exists()
