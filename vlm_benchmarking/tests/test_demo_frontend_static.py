from pathlib import Path


DEMO_ROOT = Path(__file__).parents[1] / "demo"


def test_portal_switches_cancel_child_demo_work():
    portal_js = (DEMO_ROOT / "portal.js").read_text(encoding="utf-8")
    libero_html = (DEMO_ROOT / "libero" / "index.html").read_text(encoding="utf-8")
    so101_js = (DEMO_ROOT / "so101" / "app.js").read_text(encoding="utf-8")

    assert 'postMessage({type: "demo:stop"}' in portal_js
    assert 'event.data?.type === "demo:stop"' in libero_html
    assert 'event.data?.type === "demo:stop"' in so101_js
    assert "window.addEventListener(\"pagehide\", cancelActiveWork)" in libero_html
    assert "window.addEventListener(\"pagehide\", cancelActiveWork)" in so101_js
    assert "workToken" in libero_html
    assert "workToken" in so101_js
    assert "Promise.all(\n    state.frames.map" not in so101_js
