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


def test_native_factory_wires_fast_and_trace_artifact_builders_without_fallbacks():
    source = Path(__file__).with_name("native_legion_factory.py").read_text(encoding="utf-8")
    assert "build_fast_native_components" in source
    assert "build_trace_native_fields" in source
    assert "cleanup_attempt=session.cleanup_attempt" in source
    for required in (
        "ARROW_SUITE_TRACE_ROUTE_ARTIFACT",
        "ARROW_SUITE_TRACE_CALIBRATION_ARTIFACT",
        "ARROW_SUITE_GRAPH_CONTEXT_REVISION",
    ):
        assert required in source


def test_trace_geometry_routes_simulator_assisted_rgbd_through_runtime_callbacks():
    source = Path(__file__).with_name("native_legion_factory.py").read_text(encoding="utf-8")
    # RGB-D and simulator-assisted RGB-D both use the live camera/arrow
    # callback seam. Only simulator-assisted Arrow needs the external anchor
    # factory; RGB-D receives simulator hints through its endpoint callback.
    assert 'geometry_variant in {"rgbd", "simulator_assisted_rgbd"}' in source
    assert "simulator_assisted_arrow" in source
    rgbd_branch = source.split('if geometry_variant in {"rgbd", "simulator_assisted_rgbd"}', 1)[1]
    arrow_branch = rgbd_branch.split('else:  # simulator_assisted_arrow', 1)
    assert len(arrow_branch) == 2
    assert "ARROW_SUITE_TRACE_SIMULATOR_ANCHORS_FACTORY" not in arrow_branch[0]
    assert "ARROW_SUITE_TRACE_SIMULATOR_ANCHORS_FACTORY" in arrow_branch[1]


def test_proposal_arrow_input_callbacks_disable_environment_recording():
    legion_source = Path(__file__).with_name("native_legion_factory.py").read_text(encoding="utf-8")
    teacher_source = Path(__file__).with_name("native_arrow_teacher.py").read_text(encoding="utf-8")
    assert "record_on_env=False" in legion_source
    assert "record_on_env=False" in teacher_source
