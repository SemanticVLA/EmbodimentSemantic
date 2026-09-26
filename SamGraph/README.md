# SamGraph

SamGraph extracts object masks from robot observations and converts mask geometry into directed spatial scene graphs. This directory contains the reusable pipeline used for the LIBERO and SO101 experiments reported in the repository README.

## What is included

- `src/samgraph_core/`: segmentation, tracking, mask encoding, and geometric relation rules.
- `src/samgraph_benchmark/`: LIBERO frame loading, prediction, export, evaluation, and tuning utilities.
- `src/samgraph_wrist/`: LIBERO wrist-camera extension.
- `src/samgraph_so101/`: SO101 dataset and export support.
- `config/`: object prompt vocabularies and the published SO101 episode selection.
- `scripts/`: compact export and evaluation utilities.
- `results/`: aggregate result JSON files and one qualitative image.
- `tests/`: CPU tests for graph rules, formats, and pipeline boundaries.

Large generated artifacts are not versioned here. In particular, the source repository excludes raw masks, extracted frames, HDF5 inputs, videos, prediction shards, scheduler logs, and machine-specific provenance. Those files are inputs or run artifacts rather than source code.

## Installation

Python 3.12 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e "./SamGraph[benchmark,so101,test]"
```

SAM inference additionally requires a compatible SAM 3.1 checkpoint and its upstream runtime dependencies. The checkpoint is supplied explicitly on the command line and is not bundled in this repository.

## LIBERO external camera

Run inference and export graph predictions:

```bash
python SamGraph/run_libero_agentview.py \
  --frames-root PATH_TO_AGENTVIEW_FRAME_ARCHIVES \
  --checkpoint PATH_TO_SAM_CHECKPOINT \
  --run-id libero-agentview \
  --episodes 0:50
```

To export previously generated predictions without rerunning the model, add `--predictions PATH_TO_PREDICTIONS_JSONL` and omit `--checkpoint`.

Evaluate an exported prediction set against LIBERO HDF5 scene-graph annotations:

```bash
python SamGraph/scripts/evaluate_libero_agentview.py \
  --prediction-root PATH_TO_PREDICTION_SHARDS \
  --hdf5-root PATH_TO_LIBERO_HDF5_FILES \
  --output evaluation.json
```

## LIBERO wrist camera

```bash
python SamGraph/run_libero_wrist.py \
  --wrist-frames-root PATH_TO_WRIST_FRAME_ARCHIVES \
  --agent-frames-root PATH_TO_AGENTVIEW_FRAME_ARCHIVES \
  --agent-predictions PATH_TO_AGENTVIEW_PREDICTIONS \
  --checkpoint PATH_TO_SAM_CHECKPOINT \
  --run-id libero-wrist \
  --episodes 0 1 2 3 4 5 6 7 8 9
```

The wrist extension uses external-camera predictions to maintain object identity and filters the frozen graph to objects visible from the wrist camera.

## SO101 external camera

```bash
python SamGraph/run_so101_agentview.py \
  --dataset-root PATH_TO_LEROBOT_DATASET \
  --objects-config SamGraph/config/so101_objects.json \
  --checkpoint PATH_TO_SAM_CHECKPOINT \
  --output-dir PATH_TO_NEW_OUTPUT_DIRECTORY \
  --tasks all \
  --episodes 0
```

The published 50-episode subset is defined in `config/so101_agentview_50_selection.json`. SO101 currently has no verified triplet-level ground truth, so the corresponding result is an export summary rather than an accuracy claim.

## Published aggregate results

| Evaluation | Scope | Precision | Recall | Micro F1 | Mean per-task F1 |
|---|---:|---:|---:|---:|---:|
| LIBERO `agentview` | 500 task-episodes / 12,648 sampled frames | 0.8209 | 0.8116 | 0.8162 | **0.8196** |
| LIBERO wrist | 100 task-episodes / 2,551 sampled frames | 0.5321 | 0.3863 | 0.4476 | **0.3017** |
| SO101 `agent_view` | 50 task-episodes / 835 sampled frames | — | — | — | — |

The LIBERO `agentview` evaluation preserves the original per-frame bowl-identity swap policy; 3,980 sampled frames use evaluation-only aliases. The wrist evaluation is smaller and should not be compared as though it had the same coverage as the 500-episode external-camera result. See `results/README.md` for the machine-readable records.

## Tests

```bash
python -m pytest SamGraph/tests
```

The default test suite is CPU-only. Tests that exercise checkpoint-backed inference require the optional upstream SAM runtime and model weights.
