from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

import arrow_policy_suite.oncall_matrix as matrix
from arrow_policy_suite.config import StudyConfig
from arrow_policy_suite.contracts import ActionProposal
from arrow_policy_suite.native_executor import execute_native
from arrow_policy_suite.native_factory import NativeHostSpec, build_native_host
from arrow_policy_suite.control_video import write_control_video

CONFIG = Path(__file__).parent / "configs" / "exploratory_oncall_10x10.json"


def _config() -> StudyConfig:
    return matrix._load_matrix_config(CONFIG)


def _provenance(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.bin"; checkpoint.write_bytes(b"checkpoint")
    controller = tmp_path / "controller.json"; controller.write_text("{}")
    return "tests.factory:build_host", checkpoint, controller


def test_matrix_schedule_is_sealed_and_video_selection_is_fixed():
    schedule = matrix.matrix_schedule()
    assert len(schedule) == 100
    assert (schedule[0].task_id, schedule[0].episode_index, schedule[0].seed, schedule[0].init_state_index) == (0, 0, 1000, 0)
    assert (schedule[-1].task_id, schedule[-1].episode_index, schedule[-1].seed, schedule[-1].init_state_index) == (9, 9, 1009, 9)
    assert matrix.video_selection() == tuple((task, 0) for task in range(10))


def test_matrix_plan_race_and_provenance_disagreement(tmp_path: Path):
    spec, checkpoint, controller = _provenance(tmp_path)
    root = tmp_path / "matrix"
    plans = []
    errors = []
    def worker():
        try: plans.append(matrix._ensure_plan(root, _config(), factory_spec=spec, checkpoint=checkpoint, controller=controller))
        except Exception as exc: errors.append(exc)
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert not errors and len(plans) == 4
    assert json.loads((root / "matrix_plan.json").read_text()) == plans[0]
    tampered = json.loads((root / "matrix_plan.json").read_text()); tampered["factory"] = "other.factory:build"
    (root / "matrix_plan.json").write_text(json.dumps(tampered))
    with pytest.raises(Exception, match="provenance disagreement"):
        matrix.finalize(config_path=CONFIG, output_root=root, factory_spec=spec, checkpoint=checkpoint, controller=controller)


def _write_matrix(root: Path, tmp_path: Path):
    spec, checkpoint, controller = _provenance(tmp_path)
    config = _config()
    plan = matrix._ensure_plan(root, config, factory_spec=spec, checkpoint=checkpoint, controller=controller)
    for task in matrix.TASK_IDS:
        task_root = root / "tasks" / f"task_{task:02d}"
        task_root.mkdir(parents=True)
        for episode in matrix.EPISODE_INDICES:
            episode_root = task_root / "episodes" / f"episode_{episode:02d}"
            episode_root.mkdir(parents=True)
            expected = matrix.matrix_schedule()[task * 10 + episode]
            manifest = {"schema": "arrow_policy_suite.native_run.v1", "experiment_evidence": False,
                "git_revision": plan["git_revision"], "config_sha256": plan["config_sha256"],
                "identity_seal_sha256": plan["identity_seal_sha256"], "policy_id": "arrow_on_call",
                "max_steps": 280, "task_id": task, "seed": expected.seed,
                "init_state_index": expected.init_state_index, "checkpoint_sha256": plan["checkpoint"]["sha256"],
                "controller_sha256": plan["controller"]["sha256"], "steps": 1, "step_digests": [], "success": episode == 0, "terminal": True}
            receipt = {"schema": "arrow_policy_suite.native_execution.v1", "experiment_evidence": False,
                       "status": "COMPLETED", "operation": "evaluate", "policy_id": "arrow_on_call",
                       "steps": 1, "success": episode == 0, "terminal": True,
                       "manifest": manifest}
            receipt_path = episode_root / "execution_receipt.json"
            receipt_path.write_text(json.dumps(receipt, sort_keys=True))
            result = matrix._canonical_episode_row(expected, receipt, receipt_path)
            (episode_root / "episode_result.json").write_text(json.dumps(result, sort_keys=True))
        (task_root / "task_summary.json").write_text(json.dumps({"task_id": task, "status": "COMPLETED"}))
    worker_root = root / "worker_status"; worker_root.mkdir()
    for task in matrix.TASK_IDS:
        (worker_root / f"task_{task:02d}.json").write_text(json.dumps({
            "schema": "arrow_policy_suite.oncall_worker_terminal.v1", "experiment_evidence": False,
            "task_id": task, "task_status": "COMPLETED", "task_exit_code": 0, "workload_exit_code": 0,
        }))
    videos = root / "videos"; videos.mkdir()
    for task in matrix.TASK_IDS:
        video = videos / f"task_{task:02d}_episode_00.mp4"
        receipt_path = root / "tasks" / f"task_{task:02d}" / "episodes" / "episode_00" / "execution_receipt.json"
        receipt_value = json.loads(receipt_path.read_text())
        frame = SimpleNamespace(timestep=0, observation={"agentview": np.zeros((32, 48, 3), dtype=np.uint8)})
        record = SimpleNamespace(frame=frame, executed_by="vla", success=False, terminal=True)
        sidecar = write_control_video((record,), video, task_id=task, episode_index=0, receipt=receipt_value, fps=10)
        spec = matrix.EpisodeSpec(task, 0, 1000, 0)
        result_path = receipt_path.parent / "episode_result.json"
        result_path.write_text(json.dumps(matrix._canonical_episode_row(spec, json.loads(receipt_path.read_text()), receipt_path,
            video={**sidecar, "path": str(video)}), sort_keys=True))
    return spec, checkpoint, controller


def test_pending_finalize_is_successful_and_does_not_seal_failure(tmp_path: Path):
    spec, checkpoint, controller = _provenance(tmp_path); root = tmp_path / "pending"
    matrix._ensure_plan(root, _config(), factory_spec=spec, checkpoint=checkpoint, controller=controller)
    pending = matrix.finalize(config_path=CONFIG, output_root=root)
    assert pending["status"] == "PENDING"
    assert not (root / "finalization_failure.json").exists()


def test_worker_terminal_receipts_gate_pending_and_failure_sealing(tmp_path: Path):
    spec, checkpoint, controller = _provenance(tmp_path); root = tmp_path / "worker_gate"
    matrix._ensure_plan(root, _config(), factory_spec=spec, checkpoint=checkpoint, controller=controller)
    status_root = root / "worker_status"; status_root.mkdir()
    for task in range(9):
        (status_root / f"task_{task:02d}.json").write_text(json.dumps({
            "schema": "arrow_policy_suite.oncall_worker_terminal.v1", "experiment_evidence": False,
            "task_id": task, "task_status": "COMPLETED", "task_exit_code": 0, "workload_exit_code": 0,
        }))
    pending = matrix.finalize(config_path=CONFIG, output_root=root)
    assert pending["status"] == "PENDING" and not (root / "FAILED").exists()
    (status_root / "task_09.json").write_text(json.dumps({
        "schema": "arrow_policy_suite.oncall_worker_terminal.v1", "experiment_evidence": False,
        "task_id": 9, "task_status": "COMPLETED", "task_exit_code": 0, "workload_exit_code": 0,
    }))
    with pytest.raises(Exception, match="finalization failed"):
        matrix.finalize(config_path=CONFIG, output_root=root, archive_root=tmp_path / "worker_archive")
    assert (root / "FAILED").is_file() and (root / "finalization_failure.json").is_file()
    assert (tmp_path / "worker_archive" / "matrix_status.json").is_file()


def test_complete_summaries_without_all_terminals_remain_pending(tmp_path: Path):
    root = tmp_path / "summary_early"; _write_matrix(root, tmp_path)
    for path in (root / "worker_status").glob("task_*.json"):
        path.unlink()
    (root / "worker_status" / "task_00.json").write_text(json.dumps({
        "schema": "arrow_policy_suite.oncall_worker_terminal.v1", "experiment_evidence": False,
        "task_id": 0, "task_status": "COMPLETED", "task_exit_code": 0, "workload_exit_code": 0,
    }))
    result = matrix.finalize(config_path=CONFIG, output_root=root)
    assert result["status"] == "PENDING" and not (root / "COMPLETED").exists()


def test_ten_terminal_workers_without_matrix_plan_seal_failure(tmp_path: Path):
    root = tmp_path / "no_plan_terminal"; root.mkdir(); status_root = root / "worker_status"; status_root.mkdir()
    for task in range(10):
        (status_root / f"task_{task:02d}.json").write_text(json.dumps({
            "schema": "arrow_policy_suite.oncall_worker_terminal.v1", "experiment_evidence": False,
            "task_id": task, "task_status": "COMPLETED", "task_exit_code": 0, "workload_exit_code": 0,
        }))
    archive = tmp_path / "no_plan_archive"
    with pytest.raises(Exception, match="missing matrix plan"):
        matrix.finalize(config_path=CONFIG, output_root=root, archive_root=archive)
    assert (root / "FAILED").is_file() and (archive / "matrix_status.json").is_file()
    assert json.loads((archive / "matrix_status.json").read_text())["status"] == "FAILED_INFRASTRUCTURE"


def test_finalize_layout_aggregates_takeover_and_archive_is_idempotent(tmp_path: Path):
    root = tmp_path / "matrix"; spec, checkpoint, controller = _write_matrix(root, tmp_path)
    archive = tmp_path / "archive"
    summary = matrix.finalize(config_path=CONFIG, output_root=root, archive_root=archive)
    assert summary["status"] == "COMPLETED" and summary["takeover_count"] == 0
    assert (root / "tasks/task_00/episodes/episode_00/execution_receipt.json").is_file()
    assert (archive / "matrix_status.json").is_file()
    archived_receipts = list((archive / "tasks").glob("task_*/episodes/episode_*/execution_receipt.json"))
    archived_workers = list((archive / "worker_status").glob("task_*.json"))
    assert len(archived_receipts) == 100 and len(archived_workers) == 10
    archive_status = json.loads((archive / "matrix_status.json").read_text())
    assert archive_status["status"] == "VERIFIED" and archive_status["required_counts"]["execution_receipts"] == 100
    assert archive_status["inventory_sha256"] == matrix._canonical_digest(archive_status["artifacts"])
    assert matrix.finalize(config_path=CONFIG, output_root=root, archive_root=archive)["status"] == "COMPLETED"
    (archive / "matrix_status.json").unlink()
    # Simulate a process dying after the run-root COMPLETED marker but before
    # the archive status write. A later finalizer must repair the archive.
    repaired = matrix.finalize(config_path=CONFIG, output_root=root, archive_root=archive)
    assert repaired["status"] == "COMPLETED" and (archive / "matrix_status.json").is_file()


def test_archive_copy_failure_cannot_publish_verified_status(tmp_path: Path, monkeypatch):
    root = tmp_path / "copy_failure"; _write_matrix(root, tmp_path)
    matrix.finalize(config_path=CONFIG, output_root=root)
    archive = tmp_path / "copy_failure_archive"
    original_write = matrix.write_artifact
    def fail_task_copy(path, data, **kwargs):
        if "tasks" in str(path):
            raise OSError("simulated archive copy failure")
        return original_write(path, data, **kwargs)
    monkeypatch.setattr(matrix, "write_artifact", fail_task_copy)
    with pytest.raises(Exception, match="archive copy failure"):
        matrix.finalize(config_path=CONFIG, output_root=root, archive_root=archive)
    assert not (archive / "matrix_status.json").exists()


def test_finalize_rejects_receipt_or_video_digest_mismatch(tmp_path: Path):
    root = tmp_path / "broken"; _write_matrix(root, tmp_path)
    receipt = root / "tasks/task_00/episodes/episode_00/execution_receipt.json"
    value = json.loads(receipt.read_text()); value["manifest"]["seed"] = 9999; receipt.write_text(json.dumps(value, sort_keys=True))
    with pytest.raises(Exception, match="finalization failed"):
        matrix.finalize(config_path=CONFIG, output_root=root)
    assert (root / "finalization_failure.json").is_file()


@pytest.mark.parametrize("mutation", ["corrupt", "frame_count", "fps"])
def test_finalize_rejects_corrupt_or_mismatched_video(tmp_path: Path, mutation: str):
    root = tmp_path / f"video_{mutation}"; _write_matrix(root, tmp_path)
    video = root / "videos/task_00_episode_00.mp4"; sidecar = Path(str(video) + ".json")
    if mutation == "corrupt":
        video.write_bytes(video.read_bytes() + b"corrupt")
    else:
        metadata = json.loads(sidecar.read_text())
        metadata["frames" if mutation == "frame_count" else "fps"] = 99
        sidecar.write_text(json.dumps(metadata, sort_keys=True))
    with pytest.raises(Exception, match="finalization failed"):
        matrix.finalize(config_path=CONFIG, output_root=root)


@pytest.mark.parametrize("field,value", [
    ("success", False), ("steps", 999), ("status", "FAILED"),
    ("owner_counts", {"vla": 99, "arrow": 0, "hybrid": 0}),
    ("takeover_count", 99), ("takeover_duration", 99),
])
def test_finalize_rejects_tampered_episode_result_metrics(tmp_path: Path, field, value):
    root = tmp_path / f"tampered_{field}"; _write_matrix(root, tmp_path)
    path = root / "tasks/task_00/episodes/episode_00/episode_result.json"
    result = json.loads(path.read_text()); result[field] = value; path.write_text(json.dumps(result, sort_keys=True))
    with pytest.raises(Exception, match="finalization failed"):
        matrix.finalize(config_path=CONFIG, output_root=root)


def test_episode0_failure_short_circuits_task(monkeypatch, tmp_path: Path):
    root = tmp_path / "short"
    def fake_execute(factory, *, output, run_dir, **kwargs):
        run_dir.mkdir(parents=True)
        receipt = SimpleNamespace(status="FAILED", success=False, terminal=False, steps=0, error="video failed", manifest={})
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps({"status": "FAILED", "experiment_evidence": False, "manifest": {}}))
        return receipt
    monkeypatch.setattr(matrix, "execute_native", fake_execute)
    monkeypatch.setattr(matrix, "import_callable", lambda _spec: object())
    summary = matrix.run_task(config_path=CONFIG, output_root=root, task_id=0, factory_spec="tests.factory:build_host")
    assert summary["status"] == "FAILED_INFRASTRUCTURE" and summary["skipped_episode_indices"] == list(range(1, 10))
    assert not (root / "tasks/task_00/episodes/episode_01").exists()


def test_cli_exit_codes_distinguish_failed_task_and_pending_finalize(monkeypatch):
    monkeypatch.setattr(matrix, "run_task", lambda **_kwargs: {"status": "FAILED_INFRASTRUCTURE"})
    assert matrix.main(["run-task", "--config", str(CONFIG), "--output-root", "out", "--task-id", "0", "--factory", "x:y"]) == 2
    monkeypatch.setattr(matrix, "finalize", lambda **_kwargs: {"status": "PENDING"})
    assert matrix.main(["finalize", "--config", str(CONFIG), "--output-root", "out"]) == 0


def test_records_consumer_failure_has_no_success_manifest(tmp_path: Path, monkeypatch):
    config = _config(); monkeypatch.chdir(Path(__file__).resolve().parents[2])
    class Env:
        def observe(self): return {"agentview": [[[0, 0, 0]]], "wrist": [[[0, 0, 0]]], "state": [0.0] * 8, "instruction": "pick"}
        def step(self, _action): return {"success": True, "terminal": True}
        def snapshot_state(self): return 0
        def restore_state(self, _state): return None
    class VLA:
        n_action_steps = 1
        def propose(self, frame): return ActionProposal((0.1,) + (0.0,) * 6, "vla", frame.timestep, observation_digest=frame.digest)
        def snapshot_state(self): return 0
        def restore_state(self, _state): return None
    class Teacher(VLA):
        def propose(self, frame): return ActionProposal((0.2,) + (0.0,) * 6, "arrow", frame.timestep, observation_digest=frame.digest)
        def commit(self, _record): pass
        def interrupt(self): pass
    def factory(**_kwargs): return build_native_host(NativeHostSpec(Env(), VLA(), Teacher(), policy_id="teacher_only"))
    receipt = execute_native(factory, config=config, operation="canary", policy_id="teacher_only", run_dir=tmp_path / "run", max_steps=1, records_consumer=lambda *_: (_ for _ in ()).throw(RuntimeError("video failed")))
    assert receipt.status == "FAILED" and not (tmp_path / "run/run_manifest.json").exists() and (tmp_path / "run/failure_manifest.json").exists()
