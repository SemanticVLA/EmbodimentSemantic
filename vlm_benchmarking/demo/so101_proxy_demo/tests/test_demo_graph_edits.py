from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from demo.so101_backend import (
    DemoRepository,
    EDIT_SCHEMA_VERSION,
    ProxyDemoServer,
    StaleGraphEditError,
    StaleReviewStatusError,
    REVIEW_SCHEMA_VERSION,
)
from demo.so101_proxy_demo.proxy.artifacts import ArtifactStore
from demo.so101_proxy_demo.proxy.schemas import (
    BBoxFrame,
    BBoxObject,
    EpisodeRecord,
    ProxyFrame,
    RelationRecord,
    SampledFrameRecord,
)


TASK = "pick-black-bowl"
EPISODE = "episode_0"
CAMERA = "agent_view"
MODE = "gt"


def _episode_record() -> EpisodeRecord:
    return EpisodeRecord(
        task=TASK,
        task_text="Pick black bowl",
        episode=EPISODE,
        episode_index=0,
        length=20,
        fps=30.0,
        dataset_from_index=0,
        dataset_to_index=20,
        data_chunk_index=0,
        data_file_index=0,
        videos={
            "agent_view": {"exists": False},
            "wrist": {"exists": False},
        },
    )


def _sample(frame: int) -> SampledFrameRecord:
    return SampledFrameRecord(
        task=TASK,
        episode=EPISODE,
        episode_index=0,
        frame=frame,
        timestamp=frame / 30.0,
        camera=CAMERA,
        width=640,
        height=480,
        image_path=f"images/{frame}.jpg",
    )


def _bboxes(frame: int) -> BBoxFrame:
    return BBoxFrame(
        task=TASK,
        episode=EPISODE,
        frame=frame,
        timestamp=frame / 30.0,
        camera=CAMERA,
        width=640,
        height=480,
        objects={
            "black_bowl": BBoxObject((20, 40, 120, 160)),
            "black_stove": BBoxObject((220, 40, 340, 160)),
        },
    )


def _proxy(frame: int) -> ProxyFrame:
    return ProxyFrame(
        task=TASK,
        episode=EPISODE,
        frame=frame,
        timestamp=frame / 30.0,
        camera=CAMERA,
        mode=MODE,
        visible_objects=("black_bowl", "black_stove"),
        bboxes=_bboxes(frame).objects,
        relations=(
            RelationRecord("black_bowl", "is_left_of", "black_stove", "bbox_geometry", 1.0),
            RelationRecord("black_stove", "is_right_of", "black_bowl", "bbox_geometry", 1.0),
        ),
        gripper_phase="released",
        metadata_reliable=True,
        model_version="test",
    )


def _repository(tmp_path: Path, *, read_only: bool = False, old_edit: dict | None = None) -> DemoRepository:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    artifacts.write_jsonl("index/episodes.jsonl", [_episode_record().to_dict()])
    artifacts.write_jsonl("index/sampled_frames.jsonl", [_sample(0).to_dict(), _sample(10).to_dict()])
    artifacts.write_jsonl("bboxes/agent_view.jsonl", [_bboxes(0).to_dict(), _bboxes(10).to_dict()])
    artifacts.write_jsonl("proxy_graphs/gt/agent_view.jsonl", [_proxy(0).to_dict(), _proxy(10).to_dict()])
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    output = tmp_path / "output"
    if old_edit is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "so101_graph_edits.jsonl").write_text(
            json.dumps(old_edit, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    config = {
        "paths": {
            "so101_dataset": str(tmp_path / "dataset"),
            "gemini_predictions": str(predictions),
            "artifacts": str(artifacts.root),
        }
    }
    return DemoRepository(config, artifacts, graph_output_dir=output, read_only=read_only)


def _frame(repo: DemoRepository) -> dict:
    return repo.frame_payload(TASK, EPISODE, 0, CAMERA, MODE)


def _pair(payload: dict, relation: str = "contains") -> dict[str, str]:
    pair = payload["graph_pairs"][0]
    return {"subject": pair["subject"], "relation": relation, "object": pair["object"]}


def test_so101_app_fetch_wrapper_forwards_post_options() -> None:
    source = (Path(__file__).resolve().parents[2] / "so101" / "app.js").read_text(encoding="utf-8")
    assert "async function fetchJson(url, options = {})" in source
    assert "return common.fetchJson(url, options);" in source


def test_online_mode_rejects_edit_and_export(tmp_path: Path) -> None:
    repo = _repository(tmp_path, read_only=True)
    payload = _frame(repo)
    assert payload["editable"] is False

    with pytest.raises(PermissionError):
        repo.save_graph_edit(
            TASK,
            EPISODE,
            0,
            CAMERA,
            MODE,
            base_graph_hash=payload["base_graph_hash"],
            pairs=[_pair(payload)],
        )
    with pytest.raises(PermissionError):
        repo.save_review_status(
            TASK,
            EPISODE,
            0,
            CAMERA,
            MODE,
            base_graph_hash=payload["base_graph_hash"],
            review_status="reviewed",
        )
    with pytest.raises(PermissionError):
        repo.export_graph_csvs()


def test_online_mode_http_edit_and_export_endpoints_return_403(tmp_path: Path) -> None:
    repo = _repository(tmp_path, read_only=True)
    payload = _frame(repo)
    server = ProxyDemoServer(("127.0.0.1", 0), repo)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/api"
    edit_body = json.dumps(
        {
            "task": TASK,
            "episode": EPISODE,
            "frame": 0,
            "camera": CAMERA,
            "mode": MODE,
            "base_graph_hash": payload["base_graph_hash"],
            "pairs": [_pair(payload)],
        }
    ).encode("utf-8")
    export_body = b"{}"
    try:
        for url, body in (
            (f"{base_url}/graph-edits", edit_body),
            (f"{base_url}/review-status", json.dumps(
                {
                    "task": TASK,
                    "episode": EPISODE,
                    "frame": 0,
                    "camera": CAMERA,
                    "mode": MODE,
                    "base_graph_hash": payload["base_graph_hash"],
                    "review_status": "reviewed",
                }
            ).encode("utf-8")),
            (f"{base_url}/export-csv", export_body),
        ):
            request = Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(HTTPError) as exc_info:
                urlopen(request, timeout=5)
            assert exc_info.value.code == 403
        with pytest.raises(HTTPError) as exc_info:
            urlopen(
                f"{base_url}/review-status?task={TASK}&episode={EPISODE}&frame=0&camera={CAMERA}&mode={MODE}",
                timeout=5,
            )
        assert exc_info.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_reset_requires_post_and_restores_generated_graph(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    payload = _frame(repo)
    repo.save_graph_edit(
        TASK,
        EPISODE,
        0,
        CAMERA,
        MODE,
        base_graph_hash=payload["base_graph_hash"],
        pairs=[_pair(payload, "contains")],
    )
    server = ProxyDemoServer(("127.0.0.1", 0), repo)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/api"
    body = json.dumps(
        {
            "task": TASK,
            "episode": EPISODE,
            "frame": 0,
            "camera": CAMERA,
            "mode": MODE,
        }
    ).encode("utf-8")
    try:
        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"{base_url}/graph-edits/reset", timeout=5)
        assert exc_info.value.code == 405

        request = Request(
            f"{base_url}/graph-edits/reset",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        reset = json.loads(urlopen(request, timeout=5).read().decode("utf-8"))
        assert reset["manual_edit"] is False
        assert reset["graph_pairs"][0]["relation"] == "is_left_of"
        assert any(
            item["subject"] == "black_bowl"
            and item["relation"] == "is_left_of"
            and item["object"] == "black_stove"
            for item in reset["proxy_relations"]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_save_accepts_post_on_root_and_so101_api_mounts(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    payload = _frame(repo)
    server = ProxyDemoServer(("127.0.0.1", 0), repo)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"{base_url}/api/graph-edits", timeout=5)
        assert exc_info.value.code == 405

        first_body = json.dumps(
            {
                "task": TASK,
                "episode": EPISODE,
                "frame": 0,
                "camera": CAMERA,
                "mode": MODE,
                "base_graph_hash": payload["base_graph_hash"],
                "pairs": [_pair(payload, "contains")],
            }
        ).encode("utf-8")
        request = Request(
            f"{base_url}/api/graph-edits",
            data=first_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        first_saved = json.loads(urlopen(request, timeout=5).read().decode("utf-8"))
        assert first_saved["manual_edit"] is True
        assert first_saved["graph_pairs"][0]["relation"] == "contains"

        second_payload = repo.frame_payload(TASK, EPISODE, 10, CAMERA, MODE)
        second_body = json.dumps(
            {
                "task": TASK,
                "episode": EPISODE,
                "frame": 10,
                "camera": CAMERA,
                "mode": MODE,
                "base_graph_hash": second_payload["base_graph_hash"],
                "pairs": [_pair(second_payload, "is_on_top_of")],
            }
        ).encode("utf-8")
        request = Request(
            f"{base_url}/so101/api/graph-edits",
            data=second_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        second_saved = json.loads(urlopen(request, timeout=5).read().decode("utf-8"))
        assert second_saved["manual_edit"] is True
        assert second_saved["graph_pairs"][0]["relation"] == "is_on_top_of"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_save_and_reset_edit_lifecycle(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    payload = _frame(repo)

    saved = repo.save_graph_edit(
        TASK,
        EPISODE,
        0,
        CAMERA,
        MODE,
        base_graph_hash=payload["base_graph_hash"],
        pairs=[_pair(payload, "is_on_top_of")],
    )
    assert saved["manual_edit"] is True
    assert saved["edit_revision"] == 1
    record = json.loads(repo.graph_edit_path.read_text(encoding="utf-8").strip())
    assert record["schema_version"] == EDIT_SCHEMA_VERSION
    assert record["validation_status"] == "valid"

    reset = repo.reset_graph_edit(TASK, EPISODE, 0, CAMERA, MODE)
    assert reset["manual_edit"] is False
    assert repo.graph_edit_path.read_text(encoding="utf-8") == ""


def test_review_status_save_and_reset_lifecycle(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    payload = _frame(repo)

    saved = repo.save_review_status(
        TASK,
        EPISODE,
        0,
        CAMERA,
        MODE,
        base_graph_hash=payload["base_graph_hash"],
        review_status="reviewed",
        reviewer="reviewer-a",
        note="looks correct",
    )
    assert saved["review_status"] == "reviewed"
    assert saved["reviewer"] == "reviewer-a"
    assert saved["review_note"] == "looks correct"
    record = json.loads(repo.review_status_path.read_text(encoding="utf-8").strip())
    assert record["schema_version"] == REVIEW_SCHEMA_VERSION
    assert record["review_status"] == "reviewed"
    assert record["base_graph_hash"] == payload["base_graph_hash"]

    reset = repo.reset_review_status(TASK, EPISODE, 0, CAMERA, MODE)
    assert reset["review_status"] == "unreviewed"
    assert repo.review_status_path.read_text(encoding="utf-8") == ""


def test_stale_review_base_graph_hash_is_rejected(tmp_path: Path) -> None:
    repo = _repository(tmp_path)

    with pytest.raises(StaleReviewStatusError):
        repo.save_review_status(
            TASK,
            EPISODE,
            0,
            CAMERA,
            MODE,
            base_graph_hash="stale",
            review_status="reviewed",
        )


def test_worklist_filters_edit_review_and_validation_status(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    payload = _frame(repo)
    repo.save_graph_edit(
        TASK,
        EPISODE,
        0,
        CAMERA,
        MODE,
        base_graph_hash=payload["base_graph_hash"],
        pairs=[_pair(payload, "contains")],
    )
    frame_10 = repo.frame_payload(TASK, EPISODE, 10, CAMERA, MODE)
    repo.save_review_status(
        TASK,
        EPISODE,
        10,
        CAMERA,
        MODE,
        base_graph_hash=frame_10["base_graph_hash"],
        review_status="reviewed",
    )

    edited = repo.worklist(task=TASK, episode=EPISODE, camera=CAMERA, mode=MODE, edit_status="edited")
    assert edited["count"] == 1
    assert edited["items"][0]["frame"] == 0
    reviewed = repo.worklist(task=TASK, episode=EPISODE, camera=CAMERA, mode=MODE, review_status="reviewed")
    assert reviewed["count"] == 1
    assert reviewed["items"][0]["frame"] == 10

    key = (TASK, EPISODE, 0, CAMERA, MODE)
    repo.edits[key]["relations"] = [
        {"subject": "black_bowl", "relation": "contains", "object": "black_stove"}
    ]
    invalid = repo.worklist(task=TASK, episode=EPISODE, camera=CAMERA, mode=MODE, validation_status="invalid")
    assert invalid["count"] == 1
    assert invalid["items"][0]["frame"] == 0
    assert invalid["summary"]["invalid_edit_frames"] == 1


def test_pair_save_generates_inverse_directed_triplet(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    payload = _frame(repo)
    saved = repo.save_graph_edit(
        TASK,
        EPISODE,
        0,
        CAMERA,
        MODE,
        base_graph_hash=payload["base_graph_hash"],
        pairs=[_pair(payload, "contains")],
    )

    triplets = {
        (item["subject"], item["relation"], item["object"])
        for item in saved["proxy_relations"]
    }
    assert ("black_bowl", "contains", "black_stove") in triplets
    assert ("black_stove", "is_inside", "black_bowl") in triplets


def test_saved_edit_updates_frame_payload_for_canvas_and_export_csv(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    payload = _frame(repo)
    pair = payload["graph_pairs"][0]
    assert pair["relation"] == "is_left_of"

    saved = repo.save_graph_edit(
        TASK,
        EPISODE,
        0,
        CAMERA,
        MODE,
        base_graph_hash=payload["base_graph_hash"],
        pairs=[{"subject": pair["subject"], "relation": "contains", "object": pair["object"]}],
    )

    saved_pair = saved["graph_pairs"][0]
    assert saved_pair["subject"] == "black_bowl"
    assert saved_pair["relation"] == "contains"
    assert saved_pair["inverse_relation"] == "is_inside"
    assert saved_pair["original_relation"] == "is_left_of"
    assert any(
        item["subject"] == "black_bowl"
        and item["relation"] == "contains"
        and item["object"] == "black_stove"
        for item in saved["proxy_relations"]
    )

    reloaded = repo.frame_payload(TASK, EPISODE, 0, CAMERA, MODE)
    assert reloaded["manual_edit"] is True
    assert reloaded["graph_pairs"][0]["relation"] == "contains"
    assert any(
        item["subject"] == "black_bowl"
        and item["relation"] == "contains"
        and item["object"] == "black_stove"
        for item in reloaded["proxy_relations"]
    )

    manifest = repo.export_graph_csvs()
    rows = list(
        csv.DictReader(
            (Path(manifest["output_dir"]) / "agent_view.csv").open(
                encoding="utf-8",
                newline="",
            )
        )
    )
    edited_forward = [
        row for row in rows
        if row["task"] == TASK
        and row["episode"] == EPISODE
        and row["frame"] == "0"
        and row["camera"] == CAMERA
        and row["subject"] == "black_bowl"
        and row["object"] == "black_stove"
    ]
    assert edited_forward
    assert edited_forward[0]["relation"] == "contains"
    assert edited_forward[0]["edited"] == "yes"
    assert edited_forward[0]["original_relation"] == "is_left_of"


def test_export_marks_generated_rows_as_not_edited_without_saved_overlay(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    manifest = repo.export_graph_csvs()
    rows = list(
        csv.DictReader(
            (Path(manifest["output_dir"]) / "agent_view.csv").open(
                encoding="utf-8",
                newline="",
            )
        )
    )
    assert rows
    assert {row["edited"] for row in rows} == {"no"}
    assert {row["original_relation"] for row in rows} == {""}


def test_stale_base_graph_hash_is_rejected(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    payload = _frame(repo)

    with pytest.raises(StaleGraphEditError):
        repo.save_graph_edit(
            TASK,
            EPISODE,
            0,
            CAMERA,
            MODE,
            base_graph_hash="stale",
            pairs=[_pair(payload)],
        )


def test_invalid_predicate_and_object_are_rejected(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    payload = _frame(repo)

    with pytest.raises(ValueError, match="unknown predicate"):
        repo.save_graph_edit(
            TASK,
            EPISODE,
            0,
            CAMERA,
            MODE,
            base_graph_hash=payload["base_graph_hash"],
            pairs=[_pair(payload, "touches")],
        )
    with pytest.raises(ValueError, match="not visible"):
        repo.save_graph_edit(
            TASK,
            EPISODE,
            0,
            CAMERA,
            MODE,
            base_graph_hash=payload["base_graph_hash"],
            pairs=[{"subject": "black_bowl", "relation": "contains", "object": "missing_object"}],
        )


def test_export_writes_clean_timestamped_camera_csvs(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    payload = _frame(repo)
    saved_edit = repo.save_graph_edit(
        TASK,
        EPISODE,
        0,
        CAMERA,
        MODE,
        base_graph_hash=payload["base_graph_hash"],
        pairs=[_pair(payload, "contains")],
    )
    repo.save_review_status(
        TASK,
        EPISODE,
        0,
        CAMERA,
        MODE,
        base_graph_hash=saved_edit["base_graph_hash"],
        review_status="reviewed",
        reviewer="reviewer-a",
        note="paper artifact row",
    )

    manifest = repo.export_graph_csvs()
    export_dir = Path(manifest["output_dir"])
    assert export_dir.parent.name == "annotated_graphs"
    assert export_dir.name != "latest"
    assert "latest_dir" not in manifest
    files = sorted(path.name for path in export_dir.iterdir())
    assert files == ["agent_view.csv"]

    rows = list(csv.DictReader((export_dir / "agent_view.csv").open(encoding="utf-8", newline="")))
    edited_rows = [row for row in rows if row["edited"] == "yes"]
    assert rows
    assert edited_rows
    assert set(rows[0]) == {
        "task",
        "episode",
        "frame",
        "timestamp",
        "camera",
        "mode",
        "subject",
        "relation",
        "object",
        "edited",
        "original_relation",
    }
    assert "edited" in rows[0]
    assert any(row["relation"] == "contains" and row["original_relation"] == "is_left_of" for row in edited_rows)

    assert manifest["rows"] == len(rows)
    assert manifest["reviewed_frames"] == 1
    assert manifest["files"][0]["name"] == "agent_view.csv"
    assert manifest["schema"] == list(rows[0].keys())


def test_export_fails_when_saved_edit_is_invalid(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    payload = _frame(repo)
    repo.save_graph_edit(
        TASK,
        EPISODE,
        0,
        CAMERA,
        MODE,
        base_graph_hash=payload["base_graph_hash"],
        pairs=[_pair(payload)],
    )
    key = (TASK, EPISODE, 0, CAMERA, MODE)
    repo.edits[key]["relations"] = [{"subject": "black_bowl", "relation": "contains", "object": "black_stove"}]

    with pytest.raises(ValueError, match="Cannot export CSVs"):
        repo.export_graph_csvs()


def test_export_fails_when_saved_review_is_stale(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    payload = _frame(repo)
    repo.save_review_status(
        TASK,
        EPISODE,
        0,
        CAMERA,
        MODE,
        base_graph_hash=payload["base_graph_hash"],
        review_status="reviewed",
    )
    key = (TASK, EPISODE, 0, CAMERA, MODE)
    repo.reviews[key]["base_graph_hash"] = "stale"

    with pytest.raises(ValueError, match="saved review status is stale"):
        repo.export_graph_csvs()


def test_pipeline_status_reports_artifacts_progress_and_coverage(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    payload = _frame(repo)
    repo.save_review_status(
        TASK,
        EPISODE,
        0,
        CAMERA,
        MODE,
        base_graph_hash=payload["base_graph_hash"],
        review_status="needs_attention",
        note="check occlusion",
    )

    status = repo.pipeline_status(include_hashes=True)
    assert status["coverage_by_camera"]["agent_view"]["status"] == "complete"
    assert status["progress"]["needs_attention_frames"] == 1
    assert status["source_artifacts"]["index/sampled_frames.jsonl"]["exists"] is True
    assert status["source_artifacts"]["index/sampled_frames.jsonl"]["sha256"]
    health = repo.health()
    assert health["review_records"] == 1
    assert health["needs_attention_frames"] == 1
    assert "wrist_status" in health["pipeline_status"]


def test_legacy_edit_log_loads_and_upgrades_on_next_save(tmp_path: Path) -> None:
    old_edit = {
        "task": TASK,
        "episode": EPISODE,
        "frame": 0,
        "camera": CAMERA,
        "mode": MODE,
        "updated_at": "2026-01-01T00:00:00Z",
        "relations": [
            {"subject": "black_bowl", "relation": "is_on_top_of", "object": "black_stove"},
            {"subject": "black_stove", "relation": "is_below_of", "object": "black_bowl"},
        ],
    }
    repo = _repository(tmp_path, old_edit=old_edit)
    payload = _frame(repo)
    assert payload["manual_edit"] is True
    assert payload["validation_errors"] == []

    repo.save_graph_edit(
        TASK,
        EPISODE,
        0,
        CAMERA,
        MODE,
        base_graph_hash=payload["base_graph_hash"],
        pairs=[_pair(payload, "contains")],
    )
    upgraded = json.loads(repo.graph_edit_path.read_text(encoding="utf-8").strip())
    assert upgraded["schema_version"] == EDIT_SCHEMA_VERSION
    assert upgraded["base_graph_hash"] == payload["base_graph_hash"]
    assert upgraded["edit_revision"] == 1
