"""Standalone, read-only LIBERO Spatial benchmark harness for SamGraph.

The package deliberately keeps image prediction and HDF5 scoring separate.  A
prediction runner only receives ZIP frames; the evaluator is the only code
that opens the HDF5 ground-truth files.
"""

from .artifacts import ArtifactStore
from .frames import FrameRecord, iter_zip_frames, resize_rgb, undo_agentview_rotation
from .runner import PredictionRunner, write_predictions_jsonl
from .runner import read_predictions_jsonl
from .predictor import SamGraphSamPredictor


def __getattr__(name: str):
    """Load evaluation APIs only when a caller explicitly requests them."""
    from importlib import import_module

    modules = {
        "GroundTruthIndex": "ground_truth",
        "load_ground_truth": "ground_truth",
        "EvaluationReport": "metrics",
        "evaluate_predictions": "metrics",
        "triplet_set": "metrics",
        "evaluate_cached_geometry": "tuning",
        "load_candidate_configs": "tuning",
    }
    module = modules.get(name)
    if module is None:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{module}"), name)

__all__ = [
    "ArtifactStore",
    "EvaluationReport",
    "FrameRecord",
    "GroundTruthIndex",
    "PredictionRunner",
    "evaluate_predictions",
    "iter_zip_frames",
    "load_ground_truth",
    "resize_rgb",
    "triplet_set",
    "undo_agentview_rotation",
    "write_predictions_jsonl",
    "read_predictions_jsonl",
    "SamGraphSamPredictor",
    "evaluate_cached_geometry",
    "load_candidate_configs",
]
