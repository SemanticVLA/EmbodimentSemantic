from __future__ import annotations

import ast
from pathlib import Path


def test_native_factory_requests_wrist_camera_for_canonical_vla_observation():
    source = Path(__file__).with_name("native_legion_factory.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    build_calls = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "build_libero_env"]
    assert len(build_calls) == 1
    camera = next(keyword for keyword in build_calls[0].keywords if keyword.arg == "extra_camera_names")
    assert ast.literal_eval(camera.value) == ("robot0_eye_in_hand",)
