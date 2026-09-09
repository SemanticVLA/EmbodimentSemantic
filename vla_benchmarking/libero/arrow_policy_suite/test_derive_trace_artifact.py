from __future__ import annotations

import json
from types import ModuleType, SimpleNamespace

import pytest

from arrow_policy_suite.contracts import ContractError
from arrow_policy_suite.derive_trace_artifact import derive_trace_artifact, main
from arrow_policy_suite.trace_native_factory import TRACE_ROUTE_ARTIFACT_SCHEMA


class _FakeTable:
    def __init__(self, rows):
        self._rows = rows

    def to_pylist(self):
        return list(self._rows)


class _FakeParquetFile:
    rows = []
    columns = ()
    physical_columns = None
    records = {}

    def __init__(self, path):
        record = self.records.get(__import__("pathlib").Path(path).name, {})
        rows = record.get("rows", self.rows)
        columns = record.get("columns", self.columns)
        physical_columns = record.get("physical_columns", self.physical_columns)
        self._rows = rows
        self.schema = SimpleNamespace(names=list(physical_columns or columns))
        self.schema_arrow = SimpleNamespace(names=list(columns))
        self.metadata = SimpleNamespace(num_rows=len(rows))

    def read(self, columns):
        return _FakeTable([{key: row.get(key) for key in columns} for row in self._rows])


@pytest.fixture()
def fake_pyarrow(monkeypatch):
    pyarrow = ModuleType("pyarrow")
    pyarrow.__path__ = []
    parquet = ModuleType("pyarrow.parquet")
    parquet.ParquetFile = _FakeParquetFile
    monkeypatch.setitem(__import__("sys").modules, "pyarrow", pyarrow)
    monkeypatch.setitem(__import__("sys").modules, "pyarrow.parquet", parquet)


def _dataset(tmp_path, rows, columns=None):
    root = tmp_path / "dataset"
    root.mkdir(exist_ok=True)
    path = root / "data-000.parquet"
    path.write_bytes(b"fake-parquet-input")
    _FakeParquetFile.rows = rows
    _FakeParquetFile.columns = columns or ("task_index", "episode_index", "frame_index", "observation.state", "action", "agentview_image")
    _FakeParquetFile.physical_columns = None
    _FakeParquetFile.records = {}
    return root


def _rows():
    return [
        {"task_index": 2, "episode_index": 9, "frame_index": 0, "observation.state": [0.0, 0.0, 0.2, 0, 0, 0, 0.1, 0.1], "action": [1] * 7, "agentview_image": b"image"},
        {"task_index": 2, "episode_index": 9, "frame_index": 1, "observation.state": [0.2, 0.0, 0.2, 0, 0, 0, 0.0, 0.0], "action": [1] * 7, "agentview_image": b"image"},
        {"task_index": 2, "episode_index": 9, "frame_index": 2, "observation.state": [0.3, 0.0, 0.2, 0, 0, 0, 0.0, 0.0], "action": [1] * 7, "agentview_image": b"image"},
        {"task_index": 2, "episode_index": 9, "frame_index": 3, "observation.state": [0.8, 0.0, 0.2, 0, 0, 0, 0.1, 0.1], "action": [1] * 7, "agentview_image": b"image"},
    ]


def _derive(tmp_path):
    output = tmp_path / "trace.json"
    result = derive_trace_artifact(
        _dataset(tmp_path, _rows()), output, task_id=2, episode_ids=[9],
        graph_triplet=("cup", "inside", "bowl"), coordinate_frame="world",
        parent_collection_manifest="a" * 64,
    )
    return result, output


def test_derivation_is_create_only_state_only_and_lineage_complete(tmp_path, fake_pyarrow):
    result, output = _derive(tmp_path)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["schema"] == TRACE_ROUTE_ARTIFACT_SCHEMA
    assert artifact["create_only"] is True
    assert artifact["transformation_revision"]
    assert artifact["parent_collection_manifest_sha256"] == "a" * 64
    assert artifact["input"]["parquet_file_count"] == 1
    assert artifact["input"]["row_count"] == 4
    assert artifact["counts"]["accepted_routes"] == 1
    route = artifact["routes"][0]
    assert route["source_anchor"] == [0.2, 0.0, 0.2]
    assert route["destination_anchor"] == [0.8, 0.0, 0.2]
    assert route["source_role"] == "cup"
    assert route["destination_role"] == "bowl"
    assert route["states"][1]["event"] == "close"
    assert route["states"][3]["event"] == "reopen"
    serialized = json.dumps(artifact).lower()
    assert "agentview_image" not in serialized
    assert '"action"' not in serialized
    assert result.output_sha256
    with pytest.raises(ContractError, match="overwrite"):
        _derive(tmp_path)


def test_explicit_anchor_override_is_recorded_without_actions(tmp_path, fake_pyarrow):
    output = tmp_path / "trace_override.json"
    derive_trace_artifact(
        _dataset(tmp_path, _rows()), output, task_id=2, episode_ids=[9],
        graph_triplet=("cup", "inside", "bowl"), coordinate_frame="world",
        parent_collection_manifest="b" * 64,
        source_anchor=(1.0, 2.0, 3.0), destination_anchor=(4.0, 5.0, 6.0),
    )
    route = json.loads(output.read_text(encoding="utf-8"))["routes"][0]
    assert route["source_anchor"] == [1.0, 2.0, 3.0]
    assert route["destination_anchor"] == [4.0, 5.0, 6.0]


def test_derivation_fails_closed_without_close_reopen_or_required_columns(tmp_path, fake_pyarrow):
    no_reopen = _rows()[:2] + [{**_rows()[2], "observation.state": [0.3, 0, 0.2, 0, 0, 0, 0.0, 0.0]}]
    output = tmp_path / "bad.json"
    with pytest.raises(ContractError, match="no requested episodes"):
        derive_trace_artifact(
            _dataset(tmp_path, no_reopen), output, task_id=2, episode_ids=[9],
            graph_triplet=("cup", "inside", "bowl"), coordinate_frame="world",
            parent_collection_manifest="c" * 64,
        )
    missing = tmp_path / "missing_dataset"
    missing.mkdir()
    (missing / "bad.parquet").write_bytes(b"bad")
    _FakeParquetFile.rows = []
    _FakeParquetFile.columns = ("task_index", "episode_index", "frame_index", "action")
    with pytest.raises(ContractError, match="state column"):
        derive_trace_artifact(
            missing, tmp_path / "missing.json", task_id=2, episode_ids=[9],
            graph_triplet=("cup", "inside", "bowl"), coordinate_frame="world",
            parent_collection_manifest="d" * 64,
        )


def test_derivation_uses_logical_arrow_schema_for_fixed_size_list_state(tmp_path, fake_pyarrow):
    dataset = _dataset(tmp_path, _rows())
    _FakeParquetFile.physical_columns = (
        "task_index", "episode_index", "frame_index", "element", "element", "element"
    )
    result = derive_trace_artifact(
        dataset, tmp_path / "logical-schema.json", task_id=2, episode_ids=[9],
        graph_triplet=("cup", "inside", "bowl"), coordinate_frame="world",
        parent_collection_manifest="f" * 64,
    )
    assert result.artifact["counts"]["accepted_routes"] == 1


def test_derivation_skips_metadata_parquets_and_audits_the_selection(tmp_path, fake_pyarrow):
    dataset = _dataset(tmp_path, _rows())
    metadata = dataset / "meta"
    metadata.mkdir()
    (metadata / "episodes.parquet").write_bytes(b"fake-episodes-metadata")
    (metadata / "tasks.parquet").write_bytes(b"fake-tasks-metadata")
    _FakeParquetFile.records = {
        "episodes.parquet": {
            "rows": [{"episode_index": 9, "length": 4}],
            "columns": ("episode_index", "length"),
        },
        "tasks.parquet": {
            "rows": [{"task_index": 2, "task": "pick"}],
            "columns": ("task_index", "task"),
        },
    }
    result = derive_trace_artifact(
        dataset, tmp_path / "metadata-audited.json", task_id=2, episode_ids=[9],
        graph_triplet=("cup", "inside", "bowl"), coordinate_frame="world",
        parent_collection_manifest="a" * 64,
    )
    artifact = result.artifact
    assert artifact["input"]["parquet_file_count"] == 3
    assert artifact["input"]["parquet_audit"] == {
        "scanned_parquet_files": 3,
        "data_parquet_files": 1,
        "skipped_non_data_parquet_files": 2,
        "skipped_non_data_reasons": {"missing_required_columns": 2},
    }
    audits = {entry["path"].split("/")[-1]: entry for entry in artifact["input"]["files"]}
    assert audits["data-000.parquet"]["status"] == "data"
    assert audits["episodes.parquet"]["status"] == "skipped"
    assert audits["tasks.parquet"]["skip_reason"] == "missing_required_columns"
    assert artifact["counts"]["accepted_routes"] == 1


def test_derivation_fails_closed_when_only_metadata_parquets_exist(tmp_path, fake_pyarrow):
    dataset = tmp_path / "metadata-only"
    dataset.mkdir()
    (dataset / "episodes.parquet").write_bytes(b"fake-episodes-metadata")
    _FakeParquetFile.records = {
        "episodes.parquet": {
            "rows": [{"episode_index": 9, "length": 4}],
            "columns": ("episode_index", "length"),
        },
    }
    with pytest.raises(ContractError, match="no parquet file with required task, episode, and state column"):
        derive_trace_artifact(
            dataset, tmp_path / "metadata-only.json", task_id=2, episode_ids=[9],
            graph_triplet=("cup", "inside", "bowl"), coordinate_frame="world",
            parent_collection_manifest="a" * 64,
        )


def test_cli_emits_reproducible_completion_receipt(tmp_path, fake_pyarrow, capsys):
    dataset = _dataset(tmp_path, _rows())
    output = tmp_path / "cli-trace.json"
    assert main([
        "--dataset-dir", str(dataset), "--output", str(output), "--task-id", "2",
        "--episode-id", "9", "--graph-triplet", '["cup", "inside", "bowl"]',
        "--coordinate-frame", "world", "--parent-manifest", "e" * 64,
        "--close-delta", "0.01", "--reopen-delta", "0.01",
    ]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "COMPLETED"
    assert receipt["counts"]["accepted_routes"] == 1
    assert receipt["output_sha256"]
    assert output.is_file()
