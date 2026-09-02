"""Independent tests for the arrow-controller data boundary.

The controller is intentionally a pure visual/RGB-D seam.  These tests use
static inspection and strict fakes so that an accidental dependency on
simulator state cannot be hidden behind a permissive ``**kwargs`` fake.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np


def _source_path(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    path = getattr(module, "__file__", None)
    assert path, f"could not locate source for {module_name}"
    return Path(path).resolve()


def test_arrow_controller_has_no_simulator_or_scene_graph_dependency():
    """The pure controller must not import or reference runtime scene state."""
    path = _source_path("arrow_controller")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0].lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0].lower())

    forbidden_imports = {"libero", "mujoco", "robosuite", "scene_graph"}
    assert not imported_roots.intersection(forbidden_imports)
    # hashlib is used only for durable RGB-D provenance digests; it does not
    # create a runtime/simulator dependency.
    assert imported_roots <= {"__future__", "dataclasses", "typing", "numpy", "hashlib"}

    # Check executable identifiers and attribute names, not prose in the
    # module docstring.  ``world`` and ``camera`` are valid geometry terms;
    # object/simulator-specific names are not valid controller inputs.
    forbidden_names = {
        "libero", "mujoco", "robosuite", "scene_graph", "bboxes", "bbox",
        "env", "environment", "sim", "model", "evaluator", "success",
        "body", "geom", "body_xpos", "geom_xpos", "object_pose", "object_state",
    }
    references = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            references.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            references.add(node.attr.lower())
    assert not references.intersection(forbidden_names)


def test_runner_passes_only_explicit_visual_geometry_to_controller(tmp_path, monkeypatch):
    """Strict fake signatures reject bboxes, names, env, and evaluator leakage."""
    runner = importlib.import_module("run_arrow_pick_place_eval")
    calls = {"decode": [], "deproject": [], "waypoints": [], "action": []}

    class Arrow:
        source_xy = (8.0, 8.0)
        target_xy = (24.0, 24.0)

    def fake_decode(*, clean_rgb, arrow_rgb):
        calls["decode"].append((clean_rgb, arrow_rgb))
        return Arrow()

    def fake_deproject(point, depth, K, T):
        calls["deproject"].append((point, depth, K, T))
        return np.asarray([float(point[0]) / 10.0, float(point[1]) / 10.0, 0.5])

    def fake_waypoints(source, target, rotation, config):
        calls["waypoints"].append((source, target, rotation, config))
        return np.zeros((6, 3), dtype=np.float64)

    def fake_action(*, current_pos, current_rot, target_pos, target_rot, gripper, scales):
        calls["action"].append(
            (current_pos, current_rot, target_pos, target_rot, gripper, scales)
        )
        return np.zeros(7, dtype=np.float32)

    monkeypatch.setattr(runner, "decode_arrow", fake_decode)
    monkeypatch.setattr(runner, "deproject_endpoint", fake_deproject)
    monkeypatch.setattr(runner, "build_bowl_waypoints", fake_waypoints)
    monkeypatch.setattr(runner, "normalized_osc_action", fake_action)

    calibration = runner.CameraCalibration(
        "agentview", 256, 256,
        [[10.0, 0.0, 128.0], [0.0, -10.0, 128.0], [0.0, 0.0, 1.0]],
        np.eye(4).tolist(),
    )
    capture = runner.CapturedRGBD(
        np.zeros((256, 256, 3), dtype=np.uint8),
        np.ones((256, 256), dtype=np.float32),
        np.ones((256, 256), dtype=np.float32),
        calibration,
        {
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        },
    )

    # No simulator methods are needed by a dry-run; this opaque object makes
    # accidental access to the environment from a controller call observable.
    class OpaqueEnv:
        pass

    audit = runner.run_episode(
        env=OpaqueEnv(),
        task_id=0,
        seed=1000,
        output_dir=tmp_path,
        arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        capture=capture,
        dry_run=True,
        stop_after_phase="pregrasp",
    )

    assert audit["controller_input"] == "pixels_depth_calibration"
    assert len(calls["decode"]) == 1
    assert len(calls["deproject"]) == 2
    assert len(calls["waypoints"]) == 1
    assert len(calls["action"]) == 1

    # Strict function signatures are the primary assertion: if the runner
    # starts passing any object names, bboxes, env, or evaluator to these
    # seams, this test fails before these value-level checks run.
    clean, arrow = calls["decode"][0]
    assert clean.shape == arrow.shape == (256, 256, 3)
    for point, depth, K, T in calls["deproject"]:
        assert np.asarray(point).shape == (2,)
        # The runner computes and audits the robust local depth first, then
        # passes that exact scalar into deprojection.  This prevents the
        # controller/helper from silently selecting a different depth patch.
        assert np.asarray(depth).ndim == 0
        assert float(depth) == 1.0
        assert np.asarray(K).shape == (3, 3)
        assert np.asarray(T).shape == (4, 4)
