# VLA Benchmarking — dataset_eval

Evaluate vision-language-action (VLA) policies on the LIBERO benchmark with live semantic context augmentation and optional scene randomization. Policies are queried through lerobot eval with per-episode scene-graph or bounding-box context appended to the task prompt.

---

## Requirements

- **Python 3.12+** — lerobot hard-requires `>=3.12`. Python 3.10 will not work.
- **CUDA GPU recommended** — pi0 and most VLA models are 2–8 GB. CPU inference is extremely slow and may OOM.
- **Git** — used by the setup script to clone LIBERO.
- **Windows**: Enable Developer Mode for HuggingFace Hub symlinks (Settings → System → For Developers → Developer Mode), or run your terminal as Administrator.

---

## Setup

This repo (`dataset_eval/`) is the entire project. Clone it and run from inside it.

### 1. Clone this repo

```bash
git clone <this-repo-url>
cd dataset_eval
```

### 2. Create a Python 3.12 conda environment

```bash
conda create -n vla_bench python=3.12 -y
conda activate vla_bench
```

### 3. Run the one-shot setup script

```bash
python setup_env.py
```

This does everything in order:

1. Clones LIBERO into `LIBERO/` inside this repo
2. Installs `lerobot[pi]` (torch, transformers, gymnasium, huggingface-hub, etc.)
3. Installs remaining deps from `requirements.txt` (robosuite, bddl, hydra, etc.)
4. Installs LIBERO as an editable package
5. Writes a `.pth` file so `from libero.libero import ...` resolves correctly
6. **(Windows only)** Patches `MUJOCO_GPU_RENDERING=False` in robosuite's `macros.py` — robosuite forces `MUJOCO_GL=egl` when this is True, but `egl` is Linux-only and crashes on Windows
7. Writes `~/.libero/config.yaml` pointing LIBERO asset/init paths at the cloned `LIBERO/` folder
8. Runs a smoke test to verify all imports work

### Windows: mujoco.dll (manual step if setup fails)

robosuite 1.4.x expects a MuJoCo 2.3 DLL at `robosuite/utils/mujoco.dll`. This is **not** shipped in the PyPI wheel and cannot be substituted from the standalone `mujoco` package (different binary version). If the smoke test fails with a `mujoco.dll not found` error, you must copy it from a conda env that already has a working robosuite 1.4.x installation:

```powershell
Copy-Item "<other_env>\Lib\site-packages\robosuite\utils\mujoco.dll" `
          "$env:CONDA_PREFIX\Lib\site-packages\robosuite\utils\mujoco.dll"
```

### HuggingFace token (recommended)

Avoids rate limits and enables faster model downloads. Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**PowerShell:**

```powershell
$env:HF_TOKEN="hf_your_token_here"
```

**bash/zsh:**

```bash
export HF_TOKEN="hf_your_token_here"
```

---

## Running

All commands are run from inside the `dataset_eval/` directory.

### PowerShell (Windows)

```powershell
$env:CONTEXT_MODE="scene_graph"; $env:TASK_IDS="[3]"; python run_lerobot_eval_with_context.py
```

### bash / zsh (Linux / macOS)

```bash
CONTEXT_MODE=scene_graph TASK_IDS=[3] python run_lerobot_eval_with_context.py
```

### Key environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `CONTEXT_MODE` | *(required)* | `standard`, `scene_graph`, `bounding_boxes`, `scene_graph_bounding_boxes` |
| `CONTEXT_FORMAT` | `legacy_scene_graph` | Scene-graph serialization: `standard`, `legacy_scene_graph`, `triplet_published`, `triplet_human`, `triplet_human_bar`, `triplet_human_shared_subject`, `natural_separate`, `natural_compact`, `natural_compact_current`, `natural_compact_no_label`, `natural_compact_context_first` |
| `TASK_IDS` | `[0]` | Task index, e.g. `[3]` or `[0,1,2]` |
| `MODELS` | `lerobot/pi0_libero_base` | HuggingFace policy checkpoint path |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `N_EPISODES` | `1` | Episodes per task |
| `BATCH_SIZE` | `1` | Parallel episodes |
| `SEED` | `1000` | Random seed |
| `PROMPT_AUDIT` | `0` | Write `prompt_audit.jsonl` with raw prompt metadata |
| `TOKEN_AUDIT` | `0` | Add tokenizer counts and truncation fields to `prompt_audit.jsonl` when possible |
| `MAX_EPISODES_RENDERED` | LeRobot default `10` | Override rendered videos per task |
| `HF_TOKEN` | — | HuggingFace token for authenticated downloads |

**Visualize scene swaps only (no policy, no GPU needed):**

```bash
python randomize_scenes_demo.py        # all tasks
python randomize_scenes_demo.py 3 7    # specific tasks
```

Output images are saved to `swap_outputs/`.

### Scene-graph format ablation

Run the default big SmolVLA matrix:

```powershell
python run_scene_graph_format_ablation.py
```

Default conditions:

```text
standard
legacy_scene_graph
triplet_published
triplet_human
triplet_human_bar
triplet_human_shared_subject
natural_separate
natural_compact
natural_compact_current
natural_compact_no_label
natural_compact_context_first
```

By default this runs all 10 LIBERO Spatial tasks, 4 episodes per task/format, and saves one video per evaluated episode.

Outputs are written under `ablation_outputs/<run_id>/`:

- `ablation_summary.csv`: one row per format with overall success and output paths
- `ablation_task_summary.csv`: one row per format/task with task-level success
- `ablation_episodes.csv`: one row per format/task/episode with seed, reward, success, and video path
- `<format>/prompt_audit.jsonl`: raw prompts plus token counts/truncation info when tokenizer loading succeeds
- `<format>/videos/`: LeRobot rendered episode videos, grouped by task

Useful variants:

```powershell
python run_scene_graph_format_ablation.py --episodes 4 --tasks "[0,1,2,3,4,5,6,7,8,9]"
python run_scene_graph_format_ablation.py --episodes 20 --formats "standard,legacy_scene_graph,triplet_published,natural_compact"
python run_scene_graph_format_ablation.py --episodes 4 --no-token-audit
```

### Visual arrow ablation

The visual condition keeps the normal task instruction as the complete text
prompt and draws live ground-truth scene-graph edges directly on the main
policy image. The overlay contains green subject-to-object arrows only: no
relation labels and no bounding boxes. The wrist image remains unchanged.

Preview the first policy frame for all ten tasks without loading a policy:

```powershell
python preview_visual_arrows.py
```

This writes individual raw/overlaid images, `preview_audit.json`, and the
`visual_arrows_first_frames.png` contact sheet under
`visual_arrow_previews/`. Every task uses the fixed training camera pair:
`agentview` and `robot0_eye_in_hand`; arrows are drawn on the main `agentview`
policy image while the wrist image remains unchanged.

Run the default SmolVLA visual ablation (ten tasks, four episodes each):

```powershell
python run_scene_graph_visual_ablation.py
```

The runner does not rerun the raw standard condition. It auto-detects the
newest prior `standard/eval_info.json`, or accepts an explicit baseline:

```powershell
python run_scene_graph_visual_ablation.py --baseline-run .\ablation_outputs\sg_format_ablation_<run-id>
```

Outputs are written under
`ablation_outputs/sg_visual_ablation_<timestamp>/`:

- `visual_summary.csv`: overall visual-condition result
- `visual_task_summary.csv`: task-level results
- `visual_episodes.csv`: episode-level rewards, success, seeds, and videos
- `visual_manifest.json`: run configuration and resolved baseline
- `visual_vs_baseline.csv`: prior raw standard versus visual arrows
- `visual_arrows/visual_relation_audit.jsonl`: live relations and drawable-edge audit
- `visual_arrows/eval_stdout_stderr.log`: complete evaluation log
- `visual_arrows/videos/`: videos made from the exact overlaid policy frames

One-task CPU smoke run:

```powershell
python run_scene_graph_visual_ablation.py --tasks "[0]" --episodes 1 --max-videos 1 --device cpu
```

### Optional ZeroGrasp grasp-proposal runtime

ZeroGrasp is an isolated, opt-in grasp/reconstruction provider for the
arrow-only pick/place runner. The official checkout is pinned to
`152f67c27269ff3f089783bd2f041d67641fa506`; it is kept outside the LIBERO
environment. Bootstrap never downloads a checkpoint:

```bash
bash vla_benchmarking/legion/bootstrap_zerograsp.sh \
  --root /absolute/path/to/ZeroGrasp \
  --create-venv
```

Use `--install-deps` only when intentionally installing the pinned repository
requirements into the isolated `ROOT.venv` (or an explicit `--venv` path); it is
not the complete official CUDA runtime. For a reproducible Legion compute-node
environment, submit the setup-and-smoke job (never execute it on a login node):

```bash
# REPO_ROOT is this isolated evaluation checkout; ZERO_GRASP_ROOT may be a
# not-yet-created absolute path on persistent storage.
export REPO_ROOT=/absolute/path/to/evaluation-checkout
export ZERO_GRASP_ROOT=/absolute/path/to/ZeroGrasp
export ARROW_MATRIX_EXPECTED_COMMIT=<exact-40-character-evaluation-checkout-sha>
export ZERO_GRASP_SETUP_MODE=acquire
sbatch vla_benchmarking/legion/setup_zerograsp_runtime.sbatch
```

That job loads the pinned `nvhpc-nompi/25.1` CUDA compiler module,
`gcc/11.5.0` host compiler, and `miniforge/24.3.0-0`; it explicitly pins
`CC`, `CXX`, and `CUDAHOSTCXX` before creating a persistent Python 3.11 venv
outside the checkout, installs the official CUDA 12.1 stack (Torch 2.2.0,
torchvision 0.17.0, PyG wheels, xformers 0.0.24), the pinned current
`ocnn`/`dwconv`/`graspnetAPI` refs, and the octree submodule. The upstream
requirements leave `ocnn` floating, so the setup fails closed unless the
script's reviewed immutable ref is used. The upstream `dwconv` VCS requirement
is likewise filtered and installed at its pinned ref after Torch with build
isolation disabled, because its build imports Torch and compiles a CUDA
extension. The runtime lock records and verifies the compiler module names and
the resolved paths, versions, and executable hashes for `nvcc`, `gcc`, and
`g++`. Because Legion compute-node DNS does not reliably resolve the PyG wheel
host, pinned `torch_cluster==1.6.3` is built once in that persistent venv with
build isolation disabled. It installs `gdown==5.2.0` only on
the compute node and fetches the official Drive file id
`1xUmFdgT_Ozu4zIPIsh_1SJMcegeQUWqQ` only when the checkpoint path is absent;
existing checkpoint files are never overwritten. Acquisition exits before any
runtime/model import, worker process, or checkpoint deserialization. The
resulting checkpoint SHA-256 and byte size, config hash, Python/CUDA/GPU identity, source refs, and
`pip freeze` are written to an external lock and persistent archive. The first
acquisition records the observed checkpoint hash in
`checkpoint-acquisition.json`; operators must preserve it as
`ZERO_GRASP_CHECKPOINT_SHA256` for subsequent fail-closed execution:

```bash
export ZERO_GRASP_SETUP_MODE=execute
export ZERO_GRASP_CHECKPOINT_SHA256=<sha256-from-checkpoint-acquisition.json>
export ARROW_MATRIX_EXPECTED_COMMIT=<exact-40-character-evaluation-checkout-sha>
sbatch vla_benchmarking/legion/setup_zerograsp_runtime.sbatch
```

Only execute mode performs imports, a true official checkpoint load, and a
bounded worker process ready/close preflight. The preflight verifies the
explicit lowercase checkpoint SHA before starting the worker, so a missing or
mismatched digest cannot reach model deserialization. The SHA must match both
the artifact and acquisition/runtime lock. The model-load smoke does
not construct LIBERO/MuJoCo, move a robot, or query an evaluator. Demo-image
candidate inference remains a separate later smoke once an operator supplies
the exact input artifact and calibration.

The matrix launcher performs ZeroGrasp
preflight only when the fully expanded external controller JSON selects a
ZeroGrasp provider. That preflight requires all of the following environment
variables: `ZERO_GRASP_ROOT`, `ZERO_GRASP_PYTHON`, `ZERO_GRASP_CHECKPOINT`,
`ZERO_GRASP_CHECKPOINT_SHA256`, `ZERO_GRASP_CONFIG`, and
`ZERO_GRASP_CONFIG_SHA256`. The adapter is the repository file
`vla_benchmarking/zerograsp_worker.py`; it uses the built-in official backend
when no optional operator entrypoint is configured. `ZERO_GRASP_ENV_LOCK` and its
`ZERO_GRASP_ENV_LOCK_SHA256` are mandatory for every ZeroGrasp matrix; the
launcher verifies the lock's selected source revision, checkpoint/config paths
and hashes, canonical venv/interpreter path, interpreter executable hash,
current `pip freeze`, exact NumPy 1.26.4 ABI-compatible runtime, and
Torch/CUDA identity before starting the worker. A
different `ZERO_GRASP_PYTHON` or stale environment therefore fails closed.

That worker must implement the `zerograsp-jsonl-v1` JSONL handshake; the
launcher verifies the pinned source revision and every supplied artifact hash,
then performs that handshake before the first LIBERO/MuJoCo environment
construction. The worker receives only
the allowed RGB-D/calibration/proprioception path and arrow-derived source
selection; task bboxes, object poses, simulator state, and evaluator results
are not passed to it. Resolved revision and hashes are recorded in
`job_context.env` and the combined dual-matrix summary. Classical and existing
v1--v9 controller configurations do not invoke or import ZeroGrasp.

### SmolVLA LoRA fine-tuning (Lambda GPU)

The operator workflow is one command with explicit actions and profiles. The
frozen `smolvla_libero` snapshot is the baseline; only the selected treatment
adapter is trained. `treatment` uses baked-in arrows and `no-arrow` uses the
arrow-free frames. These are distinct profiles, not a control/treatment adapter
pair.

The training environment is Python 3.12 with the pinned LeRobot source commit
`d656da8ccca5989ff0a2207e81fbfa2c2d5bafb1` and its dataset, training, PEFT,
SmolVLA, and LIBERO extras. Bootstrap also requires the LIBERO checkout at
reviewed commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`. On a clean Lambda
GPU, from this directory:

```bash
bash run_smolvla_pipeline.sh setup --profile treatment
```

`prepare_lambda_data.sh` downloads the pinned `libero_spatial_v5.zip` source
at dataset commit `a0dded49581dcbf5a109f8350305411d345c5d99`, checks the
SHA256 `560fae66b5b41d2e383e6876b15052bbc105f4aae906cf1f630a2310b87a1fa9`, requires exactly ten expected LIBERO HDF5 files, then
runs the profile-specific sealed conversion and verification:

Use `--profile no-arrow` to prepare the arrow-free dataset.

`prepare_base_snapshot.sh` stores the immutable local base snapshot at the
revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`. Do not train directly
from a floating Hub name. The launcher requires both
`lora_datasets/sealed_lora_pair_manifest.json` and the converter's
`lora_datasets/sealed_lora_pair_verified.json` sentinel.

Run the real preflight without writing a run directory, then an explicitly
authorized two-step smoke run, then the sealed full run:

```bash
bash run_smolvla_pipeline.sh dry --profile treatment
bash run_smolvla_pipeline.sh smoke --profile treatment --run-dir lora_runs/treatment_smoke
bash run_smolvla_pipeline.sh full --profile treatment --run-dir lora_runs/treatment_full
```

`dry` performs the same preflight as a real launch and prints the one training
command; it does not create a run directory or plan. `smoke` is exactly two
steps. `full` is sealed to `EPOCHS=15`, `UPDATES_PER_EPOCH=1946`,
`STEPS=29190`, and `SAVE_FREQ=1946`; ambient conflicting training variables
are ignored. Every run writes immutable provenance and requires a non-empty
adapter artifact. A safe resume requires an existing run directory, a
contained checkpoint config, and compatible preserved provenance:

```bash
bash run_smolvla_pipeline.sh resume --profile treatment --run-dir lora_runs/treatment_full
```

Evaluation is explicit and treatment-only. It derives the frozen base,
treatment adapter, and training manifest from the selected run; no evaluation
is launched automatically after training. Evaluation uses the fixed training
cameras (`agentview` and `robot0_eye_in_hand`):

```bash
bash run_smolvla_pipeline.sh eval --profile treatment \
  --run-dir lora_runs/treatment_full --seeds 1000,1001 --episodes 50 \
  --output-root eval_outputs/treatment_full
```

---

## Configuration

All settings are in `config.py`:

```python
RANDOMIZE_SCENES = True                        # set False for standard (non-randomized) eval
SCENE_GRAPH_SUBJECT_FILTER = "akita_black_bowl_1"  # set None to include all objects in scene graph

BENCHMARK_NAME = "libero_spatial"
SETTLE_STEPS_INIT = 100                        # physics steps after initial env reset
SETTLE_STEPS_SWAP = 200                        # physics steps after each swap operation
```

Randomization is multidimensional. A task is not considered unchanged merely
because it has no swap operation. The sealed evaluator records and validates
these dimensions per task and episode:

- scene layout/object pose swaps (`TASK_SWAP_CONFIG`);
- distractor removal from the task scene (`TASK_REMOVE_CONFIG`);
- prompt variants (`TASK_PROMPT_OVERRIDE`) and any live semantic prompt suffix.

Tasks 2, 4, 7, 8, and 9 each have one distractor-only pose swap and one safe
distractor removal. Tasks 0, 1, 3, 5, and 6 are removal-only: their
instruction-defining landmarks or target-support constraints leave no safe
swap. All ten tasks have one validated prompt override, and every task
realizes the `prompt_variant` dimension.
Removal entries never target the target bowl, goal plate, or secondary black
bowl (`akita_black_bowl_2`).
LeRobot receives `agentview_image,robot0_eye_in_hand_image`; the underlying
MuJoCo renderer is constructed with raw `agentview,robot0_eye_in_hand` names.

`TASK_SWAP_CONFIG` is one component of the per-task scene randomization. Each
configured entry is one safe swap operation; removal-only tasks intentionally
have no layout entry:

- **Swap** `("obj_a", "obj_b")` — exchange full poses of two distractor objects

---

## How it works

### Environment

10 tasks from `libero_spatial`, each involving picking up a black bowl and placing it on a plate. Objects: `akita_black_bowl_1`, `akita_black_bowl_2`, `cookies_1`, `glazed_rim_porcelain_ramekin_1`, `plate_1`.

### Scene randomization (`radomize_scenes.py`)

Before each episode, enabled object-pose operations are applied via
`TASK_SWAP_CONFIG`, after task-specific distractors have been removed from the
BDDL. Physics is settled after each pose operation. Controlled by
`RANDOMIZE_SCENES` in `config.py` and audited in
`randomization_audit.jsonl`.
The standalone `randomize_scenes_demo.py` replays the same first stored LIBERO
init state on both sides of each before/after comparison and uses the same
named-joint projection path as sealed evaluation for the filtered scene.
For filtered tasks, evaluation briefly loads the canonical and filtered BDDL
models, projects each `[time, qpos, qvel]` init state by retained joint name,
and records projection evidence in the per-reset audit.
The sealed evaluator requires `--batch-size 1` so each reset can be observed
and audited independently.

### Semantic context (`libero_live_semantic_context.py`)

At each policy query, the live MuJoCo simulator state is read to compute:

- **Bounding boxes** — per-object pixel bounding boxes projected from 3D geom corners
- **Scene graph** — spatial relation triplets (`is_left_of`, `is_right_of`, `is_on_top_of`, `is_inside`, etc.)

Context is appended as a JSON suffix to the task description string seen by the policy. The subject filter (`SCENE_GRAPH_SUBJECT_FILTER`) restricts which object's relations are included.

### Eval wrapper (`run_lerobot_eval_with_context.py`)

Patches `lerobot_eval.make_env` to inject two wrappers around the vectorized environment:

1. `TaskContextVecEnv` — intercepts `.call("task_description")` and appends the live semantic context suffix
2. `SceneRandomizerVecEnvWrapper` — intercepts every `reset()` and applies the swap config

---

## Supported models

See `models.yaml` for the full list of tested HuggingFace checkpoints. Pass any checkpoint via `MODELS`:

**PowerShell:**

```powershell
$env:MODELS="openvla/openvla-7b-finetuned-libero-spatial"; $env:CONTEXT_MODE="standard"; $env:TASK_IDS="[3]"; python run_lerobot_eval_with_context.py
```

**bash/zsh:**

```bash
MODELS=openvla/openvla-7b-finetuned-libero-spatial CONTEXT_MODE=standard TASK_IDS=[3] python run_lerobot_eval_with_context.py
```

---

## Project structure

After setup, the directory looks like:

```text
dataset_eval/              ← repo root (cd here before running anything)
├── LIBERO/                ← cloned by setup_env.py
├── config.py              ← all settings and swap configs
├── radomize_scenes.py     ← object pose manipulation + SceneRandomizerVecEnvWrapper
├── libero_live_semantic_context.py  ← live bounding box and scene graph computation
├── run_lerobot_eval_with_context.py ← lerobot eval entry point
├── randomize_scenes_demo.py         ← visualize scene swaps (no policy needed)
├── models.yaml            ← catalogue of tested HuggingFace checkpoints
├── requirements.txt       ← pip dependencies
├── setup_env.py           ← one-shot environment setup
└── swap_outputs/          ← demo images written here
```

`~/.libero/config.yaml` is written to your home directory by `setup_env.py` and tells lerobot where to find LIBERO's assets and init states.
