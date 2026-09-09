from __future__ import annotations

import ast
from pathlib import Path

import pytest

from arrow_policy_suite.contracts import ActionProposal, ObservationFrame
from arrow_policy_suite.native_legion_factory import (
    _artifact_sha256,
    _build_minimal_branch_runner,
    _identity_int,
    _LiveMinimalPolicy,
    _learned_artifact,
    _runtime_trace_callbacks,
    _vision_endpoint_detector,
)


@pytest.mark.parametrize("operation", ("collect", "evaluate"))
def test_native_factory_requires_explicit_task_and_seed_for_runs(monkeypatch, operation):
    monkeypatch.delenv("ARROW_SUITE_TASK_ID", raising=False)
    monkeypatch.delenv("ARROW_SUITE_SEED", raising=False)
    with pytest.raises(Exception, match="TASK_ID.*required explicitly"):
        _identity_int("ARROW_SUITE_TASK_ID", operation=operation, default=7)
    with pytest.raises(Exception, match="SEED.*required explicitly"):
        _identity_int("ARROW_SUITE_SEED", operation=operation, default=11)


def test_native_factory_identity_default_is_canary_only(monkeypatch):
    monkeypatch.delenv("ARROW_SUITE_TASK_ID", raising=False)
    assert _identity_int("ARROW_SUITE_TASK_ID", operation="canary", default=7) == 7
    monkeypatch.setenv("ARROW_SUITE_TASK_ID", "3")
    assert _identity_int("ARROW_SUITE_TASK_ID", operation="collect", default=7) == 3


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


def test_trace_rgbd_callback_rejects_implicit_simulator_endpoint():
    with pytest.raises(Exception, match="vision-only.*endpoint detector"):
        _runtime_trace_callbacks(
            object(), task_id=0, resolution=256,
            frame_name="world", calibration_revision="calibration-v1",
        )


def test_trace_factory_requires_injected_vision_detector_and_labels_simulator_path():
    source = Path(__file__).with_name("native_legion_factory.py").read_text(encoding="utf-8")
    assert "ARROW_SUITE_TRACE_VISION_ENDPOINT_FACTORY" in source
    assert "endpoint_detector=_vision_endpoint_detector" in source
    assert "simulator_assisted=True" in source
    assert '"source": "canonical_simulator_endpoint"' in source
    assert '"vision_only": False' in source


def test_vision_endpoint_factory_cannot_receive_simulator_object(monkeypatch):
    import importlib
    legion = importlib.import_module("arrow_policy_suite.native_legion_factory")

    calls = []

    def adversarial_factory(raw_environment):
        calls.append(raw_environment)
        return lambda _frame, _capture: {
            "source_xy": (1.0, 1.0), "destination_xy": (2.0, 2.0),
            "provenance": {"vision_only": True},
        }

    monkeypatch.setattr(legion, "_callable", lambda _spec, *, label: adversarial_factory)
    with pytest.raises(Exception, match="may not receive a simulator/environment"):
        _vision_endpoint_detector(
            "test:adversarial_factory", frame_name="world",
            calibration_revision="cal-v1", run_dir="/tmp/run",
        )
    assert calls == []


def test_proposal_arrow_input_callbacks_disable_environment_recording():
    legion_source = Path(__file__).with_name("native_legion_factory.py").read_text(encoding="utf-8")
    teacher_source = Path(__file__).with_name("native_arrow_teacher.py").read_text(encoding="utf-8")
    assert "record_on_env=False" in legion_source
    assert "record_on_env=False" in teacher_source


def test_learned_factory_requires_exactly_one_existing_artifact(tmp_path):
    artifact = tmp_path / "adapter.pt"
    artifact.write_bytes(b"immutable")
    with pytest.raises(Exception, match="exactly one"):
        _learned_artifact("arrow_editor", ())
    with pytest.raises(Exception, match="exactly one"):
        _learned_artifact("arrow_editor", (artifact, artifact))
    assert _learned_artifact("arrow_editor", (artifact,)) == artifact.resolve()
    with pytest.raises(Exception, match="only valid"):
        _learned_artifact("frozen_base", (artifact,))


def test_directory_learned_artifact_uses_recursive_executor_hash(tmp_path):
    bundle = tmp_path / "adapter_bundle"
    nested = bundle / "subdir"
    nested.mkdir(parents=True)
    (bundle / "adapter_config.json").write_bytes(b'{"r":16}')
    (nested / "adapter_model.safetensors").write_bytes(b"adapter-bytes")
    artifact = _learned_artifact("arrow_apprentice", (bundle,))
    import hashlib
    import json
    entries = [
        ("adapter_config.json", hashlib.sha256(b'{"r":16}').hexdigest()),
        ("subdir/adapter_model.safetensors", hashlib.sha256(b"adapter-bytes").hexdigest()),
    ]
    expected = hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert _artifact_sha256(artifact) == expected


def test_minimal_runtime_factory_builds_real_eight_mask_runner():
    class Environment:
        def __init__(self):
            self.steps = 0

        def observe(self):
            return {"state": [0.0] * 8}

        def snapshot(self):
            return self.steps

        def restore(self, state):
            self.steps = int(state)

        def step(self, _action):
            self.steps += 1
            return {"reward": 0.0}

    class Producer:
        def __init__(self, action):
            self.action = action
            self.pending = None

        def propose(self, frame):
            self.pending = ActionProposal(self.action, policy_id="producer", timestep=frame.timestep,
                                          observation_digest=frame.digest)
            return self.pending

        def commit(self, _record):
            self.pending = None

        def invalidate_pending(self, **_kwargs):
            self.pending = None

        def snapshot_state(self):
            return self.pending

        def restore_state(self, state):
            self.pending = state

    class Teacher(Producer):
        def interrupt(self):
            self.pending = None

    environment = Environment()
    vla = Producer((0.1,) * 7)
    teacher = Teacher((0.2,) * 7)
    policy = _LiveMinimalPolicy()
    runner = _build_minimal_branch_runner(environment, vla, teacher, policy)
    policy.branch_runner = runner
    frame = ObservationFrame({"state": [0.0] * 8}, timestep=0)
    base = vla.propose(frame)
    arrow = teacher.propose(frame)
    decision = policy.decide(frame, base, arrow)
    assert runner.is_real is True
    assert runner.horizon == 20
    assert runner.masks == tuple(range(8))
    assert decision.metadata["branch_masks_evaluated"] == 8
    assert environment.steps == 0
