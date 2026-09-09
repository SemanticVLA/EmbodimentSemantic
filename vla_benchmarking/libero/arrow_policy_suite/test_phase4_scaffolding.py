from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arrow_policy_suite.apprentice_training import (  # noqa: E402
    ApprenticeTrainingConfig,
    export_apprentice_dataset,
    make_export_request,
    train_apprentice,
)
from arrow_policy_suite.fast_training import (  # noqa: E402
    FAST_PARAMETER_COUNT,
    FastSupportExample,
    GraphFastModel,
    fast_parameter_count,
    train_fast_support,
)
from arrow_policy_suite.contracts import ObservationFrame  # noqa: E402
from arrow_policy_suite.policies import make_policy  # noqa: E402
from arrow_policy_suite.runtime import make_proposal  # noqa: E402
from arrow_policy_suite.learning import DatasetManifest, InterventionRow  # noqa: E402
from arrow_policy_suite.residual_model import (  # noqa: E402
    ResidualBatch,
    ResidualModel,
    ResidualModelConfig,
    fit_residual_model,
)


def _parent_manifest(rows: int = 2) -> DatasetManifest:
    return DatasetManifest("arrow_policy_suite.interventions.v1", rows, ("ep-1",), "teacher.jsonl", "parent-hash", "success")


def _rows() -> tuple[InterventionRow, ...]:
    return tuple(
        InterventionRow(
            "ep-1", 0, timestep, {"observation.state": [0.0] * 8, "image": "synthetic"},
            (0.0,) * 7, (0.25,) + (0.0,) * 6, True,
        )
        for timestep in range(2)
    )


def test_residual_model_shape_and_lineage_are_explicit():
    batch = ResidualBatch(
        features=((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
        base_actions=((0.0,) * 7, (0.0,) * 7),
        teacher_actions=((0.25,) + (0.0,) * 6, (0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0)),
        episode_ids=("ep-1", "ep-1"),
        observation_digests=("digest-0", "digest-1"),
        source_manifest_sha256="parent-hash",
    )
    model = ResidualModel(ResidualModelConfig(input_dim=4, max_abs_residual=0.5))
    seen = {}

    def trainer(received_model, received_batch, config):
        seen.update(rows=received_batch.rows, seed=config.seed, model=received_model)
        return {"status": "configured"}

    receipt = fit_residual_model(model, batch, trainer)
    assert model.predict(((1.0, 0.0, 0.0, 0.0),)) == ((0.0,) * 7,)
    assert seen["rows"] == 2 and seen["seed"] == 1000
    assert receipt.data_lineage["source_manifest_sha256"] == "parent-hash"
    assert len(receipt.data_lineage["content_sha256"]) == 64


def test_apprentice_export_and_training_preserve_teacher_lineage():
    rows = _rows()
    request = make_export_request(rows, _parent_manifest(), destination="synthetic/export")
    assert request.native_rows()[0]["action"] == [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert request.native_rows()[0]["observation"]["image"] == "synthetic"

    def exporter(received):
        assert received.parent_manifest.content_sha256 == "parent-hash"
        return {"dataset_path": "synthetic/export", "dataset_manifest_sha256": "dataset-hash", "frames": 2}

    receipt = export_apprentice_dataset(request, exporter)
    result = train_apprentice(
        receipt,
        lambda exported, *, lora, training: {
            "model_id": lora.model_id, "steps": training.derived_steps(exported.frames)
        },
        training=ApprenticeTrainingConfig(epochs=5, effective_batch_size=8),
    )
    assert result["dataset"]["parent_manifest_sha256"] == "parent-hash"
    assert result["training"]["optimizer_steps"] == 2
    assert result["native_result"]["model_id"] == "HuggingFaceVLA/smolvla_libero"


def test_fast_graph_update_has_exact_448_parameters_and_shape_contract():
    model = GraphFastModel()
    assert fast_parameter_count() == FAST_PARAMETER_COUNT == 448
    support = [
        FastSupportExample(
            graph={"graph_features": [1.0] + [0.0] * 31, "router": {"hand_to_source": 1.0, "hand_to_destination": 0.0}},
            base_action=(0.0,) * 7,
            teacher_action=(0.5,) + (0.0,) * 6,
            episode_id="ep-1",
        )
    ]
    receipt = train_fast_support(model, support, success=True, source_manifest_sha256="parent-hash")
    assert receipt.parameter_count == 448
    assert receipt.adapted is True and receipt.fallback is False
    assert receipt.source_manifest_sha256 == "parent-hash"
    assert any(value != 0.0 for row in model.weights["hand_to_source"] for value in row)
    assert len(model.correction(support[0].graph)) == 7


def test_fast_failure_is_explicit_fallback_and_empty_support_is_rejected_by_shape_contract():
    model = GraphFastModel()
    receipt = train_fast_support(model, [], success=False)
    assert receipt.parameter_count == 448 and receipt.fallback is True
    with pytest.raises(ValueError):
        FastSupportExample({}, (0.0,) * 6, (0.0,) * 7)


def test_fast_fit_artifact_is_consumable_by_fast_policy_factory():
    model = GraphFastModel()
    graph = {"graph_features": [1.0] + [0.0] * 31,
             "router": {"hand_to_source": 1.0, "hand_to_destination": 0.0}}
    train_fast_support(
        model,
        [FastSupportExample(graph, (0.0,) * 7, (0.5,) + (0.0,) * 6, "ep-1", 0)],
        success=True,
    )
    assert model.last_receipt is not None
    assert model.metadata.parameter_count == 448 and model.metadata.adapted is True
    frame = ObservationFrame({"state": [0.0] * 8}, timestep=0, episode_id="eval-1")
    base = make_proposal((0.0,) * 7, "vla", frame)
    policy = make_policy("arrow_fast", corrector=model, graph_fn=lambda _frame: graph)
    decision = policy.decide(frame, base, None)
    assert decision.action[0] == pytest.approx(0.25, abs=1e-5)
    assert decision.metadata["fast_parameter_count"] == 448
    assert decision.metadata["fast_receipt"] is True


def test_fast_support_is_one_chronological_episode_and_one_shot():
    model = GraphFastModel()
    graph = [1.0] + [0.0] * 31
    with pytest.raises(ValueError, match="one chronological episode"):
        train_fast_support(
            model,
            [FastSupportExample(graph, (0.0,) * 7, (0.1,) + (0.0,) * 6, "ep-a", 0),
             FastSupportExample(graph, (0.0,) * 7, (0.1,) + (0.0,) * 6, "ep-b", 1)],
            success=True,
        )
    support = [FastSupportExample(graph, (0.0,) * 7, (0.1,) + (0.0,) * 6, "ep-a", 0)]
    train_fast_support(model, support, success=True)
    with pytest.raises(ValueError, match="one-shot"):
        train_fast_support(model, support, success=True)
