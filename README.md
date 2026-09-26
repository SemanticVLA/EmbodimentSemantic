<div align="center">

# EmbodimentSemantic

### A Spatial Scene-Graph Dataset and Benchmark for Vision-Language Models on Embodied Manipulation Trajectories

**Anonymous Authors**

![LIBERO scene-graph comparison: ground-truth (top) vs. Gemini predictions (bottom). Green edges are correct triplets, red edges are errors.](assets/banner.png)

</div>

---

## Overview

Spatial grounding remains a key limitation for vision-language models operating on embodied manipulation observations. EmbodimentSemantic evaluates whether VLMs can recover exact object–relation–object triplets from robot manipulation observations, using automatically derived scene-graph supervision across simulated and real-robot data.

The dataset has two components:

- **LIBERO benchmark** — 500 simulator demonstrations (62,250 paired timesteps, 124,500 RGB frames) across 10 LIBERO-Spatial tasks. Ground-truth scene graphs are derived automatically from MuJoCo geometry, world coordinates, and camera projections, giving exact triplet-level supervision without manual annotation.
- **SO101 real-robot dataset** — 257 teleoperated episodes across 5 tabletop bowl-placement tasks collected with the low-cost SO101 arm. Includes external-camera, wrist-camera, and depth streams in LeRobot format.

![SO101 data collection setup](assets/so101_data_collection_setup.png)

---

## Dataset Statistics

### LIBERO Simulator Benchmark

| Attribute | Value |
| --- | --- |
| Tasks | 10 |
| Demonstrations | 500 (50 per task) |
| Recorded frames per camera | 62,250 |
| Total recorded frames (both cameras) | 124,500 |
| Frames per demo | 75–197 (mean 124.5) |
| Cameras | 2 (`agentview`, `eye_in_hand`) |
| RGB resolution | 128 × 128 |
| Canonical objects | 7 (`akita_black_bowl_1`, `akita_black_bowl_2`, `cookies_1`, `glazed_rim_porcelain_ramekin_1`, `plate_1`, `wooden_cabinet_1`, `flat_stove_1`) |
| Directed relations | 8 |
| Mean triplets / frame (agentview) | 42.0 |
| Mean triplets / frame (eye_in_hand) | 16.73 |

### SO101 Real-Robot Dataset

| Attribute | Value |
| --- | --- |
| Tasks | 5 |
| Demonstrations | 257 (47–53 per task) |
| Total recorded frames (both cameras) | 240,598 |
| Recorded frames per camera | 120,299 |
| Total frames used for VLM eval (both cameras) | 8,252 |
| VLM eval frames per camera | 4,126 |
| Cameras | 2 (`agent_view`, `wrist`) |
| Frame rate | 30 FPS (1 frame/sec sampled, step 30) |
| Canonical objects | 5 (`black_bowl`, `red_drawer`, `black_stove`, `cookie`, `white_plate`) |
| Format | LeRobot |

### LIBERO-Spatial Tasks

| Task ID | Initial spatial context of `akita_black_bowl_1` |
|---|---|
| T0 | Between the plate and the ramekin |
| T1 | From table center |
| T2 | In the top drawer of the wooden cabinet |
| T3 | Next to the cookie box |
| T4 | Next to the plate |
| T5 | Next to the ramekin |
| T6 | On the cookie box |
| T7 | On the ramekin |
| T8 | On the stove |
| T9 | On the wooden cabinet |

### Object and Relation Ontology

| Type | Name | Description |
|---|---|---|
| Object | `akita_black_bowl_1` | Manipulated bowl |
| Object | `akita_black_bowl_2` | Distractor bowl |
| Object | `cookies_1` | Cookie box |
| Object | `glazed_rim_porcelain_ramekin_1` | Ramekin reference object |
| Object | `plate_1` | Target placement object |
| Object | `wooden_cabinet_1` | Cabinet / drawer |
| Object | `flat_stove_1` | Stove reference surface |
| Relation | `is_left_of` / `is_right_of` | Lateral world-frame ordering |
| Relation | `is_in_front_of` / `is_behind` | Depth world-frame ordering |
| Relation | `is_on_top_of` / `is_below_of` | Vertical support / stacking |
| Relation | `is_inside` / `contains` | Containment |

---

## Repository Structure

```
EmbodimentSemantic/
├── LIBERO_Semantic_Generation.ipynb   # Offline scene-graph annotation pipeline
├── SamGraph/                          # SAM-based scene-graph extraction
│   ├── src/                           # Reusable pipeline packages
│   ├── config/                        # Portable object-prompt configurations
│   ├── scripts/                       # Export and evaluation utilities
│   ├── results/                       # Compact aggregate results and example
│   └── tests/                         # CPU unit tests
└── vlm_benchmarking/                  # Offline VLM benchmark
    ├── run.py                         # Inference entry point
    ├── evaluate.py                    # Standalone evaluation
    ├── evaluate_all.py                # Batch evaluation across models
    ├── config.yaml                    # Camera, frame, and object configuration
    ├── models.yaml                    # Catalogue of tested model IDs
    ├── vlm_bench/                     # Core benchmark library
    │   ├── runner.py                  # Inference pipeline
    │   ├── eval.py                    # Triplet-level evaluation metrics
    │   └── prompts.py                 # Camera-specific prompt templates
    ├── run_job.sh                     # SLURM inference job
    └── eval_job.sh                    # SLURM evaluation job
```

---

## VLM Benchmark

Evaluates whether VLMs can recover directed spatial scene graphs from robot camera observations. Models are prompted with the RGB frame, task description, object vocabulary, and fixed relation set. Predictions are scored at the exact triplet level against simulator-derived ground truth.

### Setup

```bash
cd vlm_benchmarking
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# API models (Gemini, OpenAI, Anthropic, NVIDIA)
pip install -r requirements-api.txt

# Local models via vLLM
pip install vllm
pip install -r requirements.txt

cp .env.example .env
# Fill in: GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, NVIDIA_API_KEY, HF_TOKEN
```

### Running Inference

```bash
# API model
python run.py --model gemini-3.1-pro-preview --type gemini --input-dir data/libero_spatial_v5/

# Local model (vLLM)
python run.py --model Qwen/Qwen2.5-VL-7B-Instruct --type vllm --name qwen2.5-7b \
    --input-dir data/libero_spatial_v5/

# Limit scope for testing
python run.py --model Qwen/Qwen2.5-VL-7B-Instruct --type vllm \
    --input-dir data/libero_spatial_v5/ --tasks 2 --demos 5 --frames 1

# SO101 real-robot dataset
python run.py --dataset-type so1001 --model gemini-3.1-pro-preview --type gemini \
    --input-dir data/SO1001_dataset/

# List all known model IDs
python run.py --list-models
```

**Supported backends:**

| `--type` | Description | Required env var |
|---|---|---|
| `vllm` | Local inference via vLLM | `HF_TOKEN` |
| `hf` | HuggingFace Transformers (fallback) | `HF_TOKEN` |
| `gemini` | Google Gemini API | `GEMINI_API_KEY` |
| `openai` | OpenAI API | `OPENAI_API_KEY` |
| `anthropic` | Anthropic Claude API | `ANTHROPIC_API_KEY` |
| `nvidia` | NVIDIA NIM API | `NVIDIA_API_KEY` |

### Evaluation

```bash
# Evaluate a model output folder
python evaluate.py --model-dir output/qwen2.5-7b/ --input-dir data/libero_spatial_v5/

# Compare two models
python evaluate.py --model-dir output/qwen2.5-7b/ output/gemini-3.1-pro-preview/ \
    --input-dir data/libero_spatial_v5/ --save-csv results/comparison.csv

# Unidirectional evaluation (collapse inverse pairs)
python evaluate.py --model-dir output/qwen2.5-7b/ --input-dir data/libero_spatial_v5/ --direction u
```

**Reported metrics:**

| Metric | Description |
|---|---|
| Mean per-task F1 | Main score — equal weight per task |
| Macro F1 | Mean of per-frame F1 scores |
| Micro F1 | Pooled TP/FP/FN across all frames |
| Per-relation F1 | Precision/recall/F1 per relation type |
| Coverage | Fraction of GT relation types present in predictions |
| Hallucination rate | FP / (TP + FP) |
| Direction consistency | Fraction of inverse pairs both predicted correctly |
| Per-object recall | Recall per object across all frames |

### Results

| Model | agentview mF1 | eye_in_hand mF1 |
|---|---|---|
| gemini-3.1-pro | **0.5674** | **0.3773** |
| InternVL3-78B | 0.3519 | 0.2101 |
| Qwen3-VL-8B-Instruct | 0.2837 | 0.2108 |
| InternVL3_5-14B | 0.2561 | 0.1265 |
| gemma-4-E4B-it | 0.2209 | 0.1472 |
| Molmo2-8B | 0.1888 | 0.1459 |
| InternVL3_5-8B | 0.1769 | 0.1465 |
| nemotron-nano-12b-v2-vl | 0.1599 | 0.1270 |
| Qwen2.5-VL-7B-Instruct | 0.1405 | 0.1147 |
| MomaGraph-R1 | 0.1023 | 0.0926 |

A key finding: models achieve high relation-type coverage (up to 0.98) but low exact triplet F1, revealing that current VLMs often produce plausible spatial predicates while failing to bind them to the correct ordered object pairs. Depth, support, and containment relations are systematically harder than lateral relations.

---

## SamGraph

SamGraph converts segmentation masks into directed spatial scene graphs. It supports the LIBERO external and wrist cameras and an SO101 external-camera export. The repository contains the pipeline, portable configurations, aggregate evaluation records, and one qualitative example; raw masks, frame archives, videos, scheduler logs, and per-run provenance bundles are deliberately kept outside the source tree.

### Results

| Evaluation | Scope | Precision | Recall | Micro F1 | Mean per-task F1 |
|---|---:|---:|---:|---:|---:|
| LIBERO `agentview` | 500 task-episodes / 12,648 sampled frames | 0.8209 | 0.8116 | 0.8162 | **0.8196** |
| LIBERO wrist | 100 task-episodes / 2,551 sampled frames | 0.5321 | 0.3863 | 0.4476 | **0.3017** |
| SO101 `agent_view` | 50 task-episodes / 835 sampled frames | — | — | — | — |

The LIBERO `agentview` score uses the original per-frame bowl-identity swap policy and evaluation-only aliases; 3,980 frames required a swap. This policy is reported because bowl identity is a material source of uncertainty, not an implementation detail. The SO101 export contains 16,268 predicted triplets but does not have verified triplet ground truth, so reporting precision, recall, or F1 for it would be misleading.

See [SamGraph](SamGraph/README.md) for installation and reproduction commands and [the compact result records](SamGraph/results/README.md) for schemas and limitations.

---

## Offline Scene-Graph Generation

`LIBERO_Semantic_Generation.ipynb` documents the full pipeline for generating frame-level scene graphs from LIBERO HDF5 files. For each frame it:

1. Extracts MuJoCo object geometry (world position, rotation, half-sizes) and projects 8 bounding-box corners into camera views via the pinhole model.
2. Checks pairwise 2D bounding-box overlap to assign vertical (`is_on_top_of` / `is_below_of`) and containment (`is_inside` / `contains`) relations.
3. Falls back to dominant-axis world-frame ordering for lateral (`is_left_of` / `is_right_of`) and depth (`is_in_front_of` / `is_behind`) relations.
4. Writes annotations back to the HDF5 under `obs/agentview_scene_graph` and `obs/robot0_eye_in_hand_scene_graph`.
