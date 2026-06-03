# VLM Benchmarking

Benchmark vision-language models (local and API-based) on spatial relation extraction from robot manipulation HDF5 datasets. Each model is queried per-frame, per-camera, and results are saved as triplet CSVs and raw JSONL logs. After inference, evaluation runs automatically and reports macro/micro F1, per-relation F1, relation-type coverage, hallucination rate, direction consistency, and per-object recall.

## How it works

### Data format

```text
data/libero_spatial_v5/
└── pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5
    └── data/
        ├── demo_0/obs/agentview_rgb                  ← [N, H, W, 3] uint8 frames
        ├── demo_0/obs/eye_in_hand_rgb
        ├── demo_0/obs/agentview_scene_graph           ← JSON-encoded GT triplets per frame
        ├── demo_0/obs/robot0_eye_in_hand_scene_graph
        ├── demo_1/...
        └── ...
```

- **Task**: one HDF5 file, derived from the filename.
- **Demo**: one trajectory/episode inside the file (`demo_0`, `demo_1`, …).
- **Frame**: one timestep image, indexed by position in the dataset.
- **Scene graph**: ground-truth spatial triplets stored as JSON in the HDF5 under `obs/{camera}_scene_graph`.

### Inference pipeline (`run.py` → `vlm_bench/runner.py`)

1. Load HDF5 files (optionally limited by `--tasks`).
2. For each task, build a prompt listing all objects and spatial rules.
3. Iterate demos × cameras × frame indices.
4. Query the VLM with the frame image and prompt.
5. Parse the response into `(objectA, relation, objectB)` triplets.
6. Write per-task CSVs and JSONL logs to `output/<name>/<camera>/{csv,json}/`.

### Prompts (`vlm_bench/prompts.py`)

Two camera-specific templates (agentview overhead, wrist eye-in-hand) wrap a shared rules block.
Extraction is exclusively bidirectional — 8 relations are enforced:

- `is_left_of` / `is_right_of`
- `is_in_front_of` / `is_behind`
- `is_on_top_of` / `is_below_of`
- `is_inside` / `contains`

Every ordered pair (A, B) gets exactly one relation; inverses are always emitted together.

### Evaluation (`vlm_bench/eval.py`)

Raw JSONL responses are re-parsed and compared against HDF5 ground-truth scene graphs at the triplet level. CSVs are still written for inspection, but they are not used as the metric source:

- **TP** = predicted ∩ GT
- **FP** = predicted − GT
- **FN** = GT − predicted

Evaluation is bidirectional over the full 8-relation ontology used in the prompt. For the visually identical LIBERO black bowls, scoring is frame-level permutation-invariant over `akita_black_bowl_1` and `akita_black_bowl_2`: each frame is scored with the predicted bowl IDs unchanged and swapped, and the higher-F1 assignment is used. The two bowl instances remain distinct; unsuffixed bowl names do not receive this leniency.

**Reported metrics:**

| Metric | Description |
| --- | --- |
| Macro F1 | Mean of per-frame F1 scores |
| Micro F1 | Pooled TP/FP/FN across all frames |
| Per-relation F1 | P/R/F1 per relation type |
| Relation-type coverage | Fraction of GT relation types appearing in predictions |
| Hallucination rate | FP / (TP + FP) |
| Direction consistency | Fraction of inverse pairs both predicted |
| Per-object recall | Recall per object across all frames |
| Mean per-task F1 | Average per-task F1 across all evaluated tasks |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**API models only (Gemini, OpenAI, Anthropic, NVIDIA):**

```bash
pip install -r requirements-api.txt
```

**Local models (vLLM):**

```bash
pip install vllm
pip install -r requirements.txt
```

**Environment variables:**

```bash
cp .env.example .env
# Fill in the keys you need:
# GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, NVIDIA_API_KEY, HF_TOKEN
```

---

## Configuration

`config.yaml` controls cameras, frames, and objects:

```yaml
extraction:
  prompt_version: v4          # appended to output filenames
  frame_step: 10              # extract every Nth frame
  frame_max: 10000            # upper bound on frames
  cameras:
    - agentview               # overhead camera
    - eye_in_hand             # wrist camera
  rotate_agentview_180: true  # flip overhead image to canonical orientation

objects:                      # exact object names used in prompts
  - akita_black_bowl_1
  - akita_black_bowl_2
  - cookies_1
  - glazed_rim_porcelain_ramekin_1
  - plate_1
  - wooden_cabinet_1
  - flat_stove_1

so101:
  prompt_version: v1
  frame_step: 30              # SO101: every Nth video frame; 30 FPS means 1 sample/sec
  frame_max: 10000
  cameras:
    - agent_view
    - wrist
  objects:                    # exact SO101 object names used in prompts
    - black_bowl
    - drawer
    - stove
    - cookie
    - plate
```

`models.yaml` is a reference catalogue of tested model IDs grouped by backend type. It is not used for routing — see below.

---

## Running inference

Pass the model ID and backend type directly on the command line. No YAML changes needed to run a new model.

**Minimal run:**

```bash
python run.py --model Qwen/Qwen2.5-VL-7B-Instruct --type vllm --input-dir data/libero_spatial_v5/
```

**With a custom output name (recommended — keeps folder names short):**

```bash
python run.py --model Qwen/Qwen2.5-VL-7B-Instruct --type vllm --name qwen2.5-7b \
    --input-dir data/libero_spatial_v5/
```

**API models:**

```bash
python run.py --model gemini-3.1-pro-preview --type gemini --input-dir data/libero_spatial_v5/
python run.py --model gpt-5.1 --type openai --input-dir data/libero_spatial_v5/
python run.py --model claude-opus-4-7 --type anthropic --input-dir data/libero_spatial_v5/
python run.py --model mistralai/mistral-large-3-675b-instruct-2512 --type nvidia --input-dir data/libero_spatial_v5/
```

**HuggingFace local (no vLLM):**

```bash
python run.py --model allenai/Molmo2-8B --type hf --input-dir data/libero_spatial_v5/
```

**Limit scope:**

```bash
python run.py --model Qwen/Qwen2.5-VL-7B-Instruct --type vllm \
    --input-dir data/libero_spatial_v5/ --tasks 2 --demos 5 --frames 1 --cameras eye_in_hand
```

**SO101 inference:**

```bash
python run.py --dataset-type so101 --model Qwen/Qwen2.5-VL-7B-Instruct --type vllm \
    --input-dir data/SO101_dataset/ --name qwen-so101

python run.py --dataset-type so101 --model Qwen/Qwen2.5-VL-7B-Instruct --type vllm \
    --input-dir data/SO101_dataset/ --task-id 0 --demos 1 --frames 5 \
    --cameras agent_view --save-frames debug_frames/so101-smoke
```

**vLLM tuning flags:**

```bash
python run.py --model Qwen/Qwen3-VL-32B-Instruct-AWQ --type vllm \
    --quantization awq --max-pixels 1048576 --batch-size 8 \
    --input-dir data/libero_spatial_v5/
```

**List known model IDs:**

```bash
python run.py --list-models
```

**Full CLI reference:**

| Argument | Default | Description |
| --- | --- | --- |
| `--model` | required | Model ID (HuggingFace path or API model name) |
| `--type` | required | Backend: `vllm`, `nvidia`, `gemini`, `openai`, `anthropic`, `hf` |
| `--name` | model ID (/ → --) | Output folder name under `output/` |
| `--input-dir` | required | Directory containing HDF5 files or SO101 task directories |
| `--output-dir` | `output` | Root output directory |
| `--batch-size` | `8` | Batch size |
| `--max-new-tokens` | `4096` | Max output tokens |
| `--temperature` | `0.2` | Sampling temperature |
| `--max-retries` | `5` | API retry attempts |
| `--max-model-len` | `8192` | vLLM max context length |
| `--gpu-memory-utilization` | `0.90` | vLLM GPU memory fraction |
| `--max-pixels` | none | vLLM vision max pixels (e.g. `1048576` for Qwen) |
| `--quantization` | none | vLLM quantization (`awq`, `fp8`) |
| `--thinking-budget` | none | Gemini thinking budget tokens |
| `--tasks` | all | Limit to first N HDF5 files or SO101 task directories |
| `--task-id` | none | Run only the task at this 0-based index |
| `--demos` | all | Limit to first N demos/episodes per task |
| `--frames` | all | Use only first N frame indices from config |
| `--cameras` | config value | Override cameras (`agentview`, `eye_in_hand`; SO101: `agent_view`, `wrist`) |
| `--dataset-type` | auto | Force `hdf5` or `so101` |
| `--save-frames` | off | Save sampled input frames to a folder for inspection |
| `--config` | `config.yaml` | Config file path |
| `--list-models` | — | Print known model IDs from `models.yaml` and exit |
| `--verbose` | off | Print per-file triplet counts |

---

## Standalone evaluation

Evaluate existing JSONL logs without re-running inference:

```bash
# entire model output folder
python evaluate.py --model-dir output/qwen2.5-7b/ --input-dir data/libero_spatial_v5/

# single JSONL
python evaluate.py --jsonl output/qwen2.5-7b/agentview/json/task_agentview_v4.jsonl \
    --input-dir data/libero_spatial_v5/

# compare two models, save per-frame breakdown
python evaluate.py --model-dir output/qwen2.5-7b/ output/gemini-3.1-pro-preview/ \
    --input-dir data/libero_spatial_v5/ --save-csv results/comparison.csv

# regenerate parsed CSV artifacts from JSONL logs
python evaluate.py --model-dir output/qwen2.5-7b/ --input-dir data/libero_spatial_v5/ \
    --reparse-jsonl-to-csv
```

**CLI reference:**

| Argument | Default | Description |
| --- | --- | --- |
| `--jsonl` | — | One or more prediction JSONL log files |
| `--model-dir` | — | One or more model output folders (auto-globs JSONLs) |
| `--input-dir` | required | Directory containing HDF5 files |
| `--frames` | all | Use only first N frame indices from config |
| `--cameras` / `--camera` | all in file | Restrict to selected cameras |
| `--verbose` / `-v` | off | Print per-frame TP/FP/FN breakdown |
| `--save-csv` | none | Save per-frame results to a CSV |

> Exactly one of `--jsonl` or `--model-dir` is required.

### Paper-wide results and plots

`evaluate_all.py` is the paper-results entry point. It evaluates all model folders from JSONL, rewrites `paper_results.csv`, regenerates aggregate plots under `figures/paper_results/`, and replays the existing qualitative Gemini frame specs under `figures/gemini-3.1-pro-preview/` with `plot_frame.py`.

```bash
python evaluate_all.py --input-dir data/libero_spatial_v5/ --output-dir output/ --out paper_results.csv

# evaluation only, no plot regeneration
python evaluate_all.py --input-dir data/libero_spatial_v5/ --no-plots
```

Useful plotting flags:

| Argument | Default | Description |
| --- | --- | --- |
| `--no-plots` | off | Skip all figure regeneration |
| `--no-paper-plots` | off | Skip `figures/paper_results/` regeneration |
| `--no-frame-plots` | off | Skip qualitative frame plot regeneration |
| `--paper-figures-dir` | `figures/paper_results` | Aggregate plot output directory |
| `--frame-figure-model` | `gemini-3.1-pro-preview` | Model folder under `figures/` to replay with `plot_frame.py` |
| `--frame-hires` | none | Optional simulator render resolution for qualitative frames |
| `--workers` | `4` | Parallel model/camera evaluation workers |

---

## Output format

For each model × task × camera, two files are written under `output/<name>/<camera>/`:

- `csv/<task>_<camera>_<prompt_ver>.csv` — columns: `task, demo, frame, camera, objectA, relation, objectB`
- `json/<task>_<camera>_<prompt_ver>.jsonl` — one record per query: `task, demo, frame, camera, model, input_hash, response, latency_s`

The `<name>` folder defaults to the model ID with `/` replaced by `--` (e.g. `Qwen--Qwen2.5-VL-7B-Instruct`), or whatever was passed as `--name`.

---

## SLURM batch jobs

Two scripts are provided — both forward all extra arguments to their respective Python scripts.

**Inference (`run_job.sh`):**

```bash
# vLLM model
sbatch run_job.sh --model Qwen/Qwen2.5-VL-7B-Instruct --type vllm --name qwen2.5-7b

# vLLM, specific camera only
sbatch run_job.sh --model allenai/Molmo2-8B --type vllm --name molmo2-8b --cameras eye_in_hand

# NVIDIA NIM
sbatch run_job.sh --model mistralai/mistral-large-3-675b-instruct-2512 --type nvidia --name mistral-large-3

# single task by index
sbatch run_job.sh --model Qwen/Qwen2.5-VL-7B-Instruct --type vllm --name qwen2.5-7b --task-id 5
```

**Evaluation (`eval_job.sh`):**

```bash
# entire model folder
sbatch eval_job.sh --model-dir output/qwen2.5-7b/

# compare two models
sbatch eval_job.sh --model-dir output/qwen2.5-7b/ output/mistral-large-3/

# save per-frame metric rows
sbatch eval_job.sh --model-dir output/qwen2.5-7b/ --save-csv results/qwen_eval.csv
```

Logs are written to `logs/slurm_<job_id>.out` (inference) and `logs/slurm_eval_<job_id>.out` (evaluation).

---

## Supported backends

| `--type` | Description | Required env var |
| --- | --- | --- |
| `vllm` | Local inference via vLLM (continuous batching, GPU) | `HF_TOKEN` |
| `hf` | Local inference via HuggingFace Transformers (fallback) | `HF_TOKEN` |
| `gemini` | Google Gemini API | `GEMINI_API_KEY` |
| `openai` | OpenAI API | `OPENAI_API_KEY` |
| `anthropic` | Anthropic Claude API | `ANTHROPIC_API_KEY` |
| `nvidia` | NVIDIA NIM API | `NVIDIA_API_KEY` |

Run `python run.py --list-models` to see the catalogue of tested model IDs for each backend.
