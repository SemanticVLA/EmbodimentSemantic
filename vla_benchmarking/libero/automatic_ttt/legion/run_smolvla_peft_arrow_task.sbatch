#!/usr/bin/env bash
# One-job exploratory PEFT experiment for SmolVLA on one LIBERO task.
#
# This reuses the existing LeRobot/PEFT trainer and source-grounded HDF5
# converter. It is deliberately labelled PEFT, not RoboTTT TTT: the treatment
# consists of verified ground-truth scene-graph-arrow demonstration frames.
# The default ARROW_DEMOS is 50 and is an explicit, recorded variable.

#SBATCH --job-name=smolvla_peft_arrow_t0
#SBATCH --partition=gpu_a40
#SBATCH --exclude=compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-23:59:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.err

set -Eeuo pipefail
umask 027
export PYTHONDONTWRITEBYTECODE=1

die() { printf 'smolvla PEFT task-0: %s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || die 'submit with sbatch'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
EXPECTED_COMMIT="${PEFT_EXPECTED_COMMIT:-}"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'PEFT_EXPECTED_COMMIT must be a 40-character SHA'
command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'
command -v flock >/dev/null 2>&1 || die 'flock is required'

REPO_ROOT="$(realpath -e -- "$REPO_ROOT")" || die 'cannot resolve REPO_ROOT'
[[ -d "$REPO_ROOT/.git" ]] || die 'REPO_ROOT is not a Git checkout'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] || die 'shared HOME checkout is forbidden'
ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)" || die 'cannot read checkout SHA'
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || die "checkout SHA differs: $ACTUAL_COMMIT"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'REPO_ROOT is dirty'

VLA_NAME="${PEFT_VLA:-smolvla}"
TASK_ID="${PEFT_TASK_ID:-0}"
[[ "$VLA_NAME" == smolvla ]] || die 'this launcher supports PEFT_VLA=smolvla only'
[[ "$TASK_ID" == 0 ]] || die 'this launcher is sealed to LIBERO task 0'
ARROW_DEMOS="${PEFT_ARROW_DEMOS:-50}"
EVAL_EPISODES="${PEFT_EVAL_EPISODES:-10}"
EVAL_SEED="${PEFT_EVAL_SEED:-1000}"
PEFT_EPOCHS="${PEFT_EPOCHS:-15}"
PEFT_BATCH_SIZE="${PEFT_BATCH_SIZE:-32}"
[[ "$ARROW_DEMOS" =~ ^[1-9][0-9]*$ ]] || die 'PEFT_ARROW_DEMOS must be positive'
[[ "$EVAL_EPISODES" == 10 && "$EVAL_SEED" == 1000 ]] || die 'evaluation is sealed to 10 episodes and seed 1000'
[[ "$PEFT_EPOCHS" =~ ^[1-9][0-9]*$ && "$PEFT_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die 'invalid PEFT schedule'

BASE_POLICY_REVISION="${PEFT_BASE_POLICY_REVISION:-6721902bc4d61e50a3bfdb11dfb4cb626f05d102}"
BASE_POLICY="${PEFT_BASE_POLICY:-$HOME/EmbodimentSemantic/vla_benchmarking/base_models/smolvla_libero-$BASE_POLICY_REVISION}"
LIBERO_DIR="${PEFT_LIBERO_DIR:-$HOME/EmbodimentSemantic/vla_benchmarking/LIBERO}"
LIBERO_DATA_DIR="${PEFT_LIBERO_DATA_DIR:-$HOME/EmbodimentSemantic/vlm_benchmarking/data/libero_spatial_v5}"
LIBERO_CONFIG="${PEFT_LIBERO_CONFIG:-$HOME/EmbodimentSemantic_runtime/config/config.yaml}"
PYTHON="${PEFT_PYTHON:-$HOME/.conda/envs/embodiment-smolvla-py312/bin/python}"
LIBERO_COMMIT="${PEFT_LIBERO_COMMIT:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
CONVERTER="$REPO_ROOT/vla_benchmarking/libero/finetuned_vlas/smolvla/workflows/hdf5_to_lerobot_dataset.py"
TRAINER="$REPO_ROOT/vla_benchmarking/libero/finetuned_vlas/smolvla/workflows/run_lerobot_train.py"
AUDITOR="$REPO_ROOT/vla_benchmarking/libero/finetuned_vlas/smolvla/workflows/adapter_audit.py"
EVALUATOR="$REPO_ROOT/vla_benchmarking/libero/evaluation/run_lerobot_eval_with_context.py"
[[ -x "$PYTHON" ]] || die "Python is missing: $PYTHON"
for path in "$BASE_POLICY" "$LIBERO_DIR" "$LIBERO_DATA_DIR" "$LIBERO_CONFIG" "$CONVERTER" "$TRAINER" "$AUDITOR" "$EVALUATOR"; do [[ -e "$path" ]] || die "required path is missing: $path"; done
[[ -d "$LIBERO_DIR/.git" ]] || die 'LIBERO_DIR is not a Git checkout'
[[ "$(git -C "$LIBERO_DIR" rev-parse HEAD 2>/dev/null || true)" == "$LIBERO_COMMIT" ]] || die "LIBERO is not pinned to $LIBERO_COMMIT"
[[ -z "$(git -C "$LIBERO_DIR" status --porcelain --untracked-files=no 2>/dev/null || true)" ]] || die 'LIBERO checkout is dirty'
[[ -f "$BASE_POLICY/config.json" && -f "$BASE_POLICY/base_snapshot_manifest.json" ]] || die 'base snapshot is incomplete'
"$PYTHON" - "$BASE_POLICY/base_snapshot_manifest.json" "$BASE_POLICY" "$BASE_POLICY_REVISION" <<'PY'
import hashlib, json, pathlib, sys
manifest_path, root, revision = map(pathlib.Path, sys.argv[1:])
data = json.loads(manifest_path.read_text(encoding='utf-8'))
if data.get('revision') != str(revision):
    raise SystemExit(f"base revision mismatch: {data.get('revision')}")
for name, expected in data.get('files', {}).items():
    path = root / name
    if not path.is_file():
        raise SystemExit(f"base manifest file missing: {name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"base manifest hash mismatch: {name}")
PY

LABEL="${PEFT_LABEL:-smolvla_task0_arrow_peft}"
[[ "$LABEL" =~ ^[A-Za-z0-9_.-]{1,96}$ ]] || die 'PEFT_LABEL is not safe'
SCRATCH_ROOT="${PEFT_SCRATCH_ROOT:-${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/peft_arrow}"
RUN_ROOT="${PEFT_RUN_ROOT:-$SCRATCH_ROOT/${LABEL}_${SLURM_JOB_ID}}"
ARCHIVE_ROOT="${PEFT_ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/peft_arrow/$VLA_NAME/task_${TASK_ID}/${LABEL}_${SLURM_JOB_ID}}"
for path in "$RUN_ROOT" "$ARCHIVE_ROOT"; do [[ "$path" = /* && "$path" != / && ! -L "$path" ]] || die "unsafe output path: $path"; done
RUN_ROOT="$(realpath -m -- "$RUN_ROOT")"; ARCHIVE_ROOT="$(realpath -m -- "$ARCHIVE_ROOT")"
case "$RUN_ROOT" in "$REPO_ROOT"|"$REPO_ROOT"/*) die 'RUN_ROOT overlaps REPO_ROOT' ;; esac
case "$ARCHIVE_ROOT" in "$REPO_ROOT"|"$REPO_ROOT"/*) die 'ARCHIVE_ROOT overlaps REPO_ROOT' ;; esac
case "$ARCHIVE_ROOT" in "$RUN_ROOT"|"$RUN_ROOT"/*) die 'ARCHIVE_ROOT is inside RUN_ROOT' ;; esac
case "$RUN_ROOT" in "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'RUN_ROOT is inside ARCHIVE_ROOT' ;; esac
[[ ! -e "$RUN_ROOT" && ! -e "$ARCHIVE_ROOT" ]] || die 'output path already exists'

TMP_ROOT="${PEFT_TMP_ROOT:-/tmp/peft-${SLURM_JOB_ID}}"
[[ "$TMP_ROOT" == "/tmp/peft-${SLURM_JOB_ID}" && ! -e "$TMP_ROOT" && ! -L "$TMP_ROOT" ]] || die 'unsafe/stale TMP_ROOT'
mkdir -p "$RUN_ROOT" "$TMP_ROOT"
export TMPDIR="$TMP_ROOT" XDG_CACHE_HOME="$RUN_ROOT/xdg"
mkdir -p "$XDG_CACHE_HOME"

DATA_ROOT="$RUN_ROOT/dataset"
BASELINE_EVAL_ROOT="$RUN_ROOT/eval_baseline"
TRAIN_ROOT="$RUN_ROOT/training"
TRAIN_META_ROOT="$RUN_ROOT/training_metadata"
ADAPTED_EVAL_ROOT="$RUN_ROOT/eval_adapted"
mkdir -p "$DATA_ROOT" "$BASELINE_EVAL_ROOT" "$TRAIN_META_ROOT" "$ADAPTED_EVAL_ROOT"
export REPO_ROOT VLA_ROOT="$REPO_ROOT/vla_benchmarking/libero" LIBERO_DIR LIBERO_CONFIG LIBERO_DATA_DIR
export PYTHONPATH="$REPO_ROOT:$LIBERO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=1000 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISTIC=1

copy_tree() { local source="$1" destination="$2"; [[ -e "$source" ]] || return 0; mkdir -p "$destination"; if command -v rsync >/dev/null 2>&1; then rsync -a -- "$source/" "$destination/"; else cp -a -- "$source/." "$destination/"; fi; }
archive_on_exit() {
  local workload_rc=$? archive_rc=0; trap - EXIT; set +e
  mkdir -p "$ARCHIVE_ROOT" || archive_rc=$?
  copy_tree "$RUN_ROOT" "$ARCHIVE_ROOT/run" || archive_rc=$?
  log_root="${SLURM_LOG_ROOT:-$HOME/EmbodimentSemantic_runtime/operator/logs}"; mkdir -p "$ARCHIVE_ROOT/slurm_logs" || archive_rc=$?
  for suffix in out err; do log_path="$log_root/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${suffix}"; [[ -f "$log_path" ]] && cp -p -- "$log_path" "$ARCHIVE_ROOT/slurm_logs/" || true; done
  if [[ -d "$ARCHIVE_ROOT" ]]; then
    printf 'schema=smolvla_peft_arrow_task_job.v1\njob_id=%s\nstatus=%s\nworkload_exit_code=%s\nrepo_commit=%s\nvla=%s\ntask_id=%s\narrow_demos=%s\nrun_root=%s\narchive_root=%s\n' "$SLURM_JOB_ID" "$([[ "$workload_rc" -eq 0 && "$archive_rc" -eq 0 ]] && echo VERIFIED || echo PRESERVED_FAILURE)" "$workload_rc" "$ACTUAL_COMMIT" "$VLA_NAME" "$TASK_ID" "$ARROW_DEMOS" "$RUN_ROOT" "$ARCHIVE_ROOT" > "$ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
    find "$ARCHIVE_ROOT" -type f ! -name inventory.sha256 ! -name tree_sha256 -print0 | sort -z | xargs -0 -r sha256sum > "$ARCHIVE_ROOT/inventory.sha256" || archive_rc=$?
    sha256sum "$ARCHIVE_ROOT/inventory.sha256" | awk '{print tolower($1)}' > "$ARCHIVE_ROOT/tree_sha256" || archive_rc=$?
  fi
  if [[ "$TMP_ROOT" == "/tmp/peft-${SLURM_JOB_ID}" && -e "$TMP_ROOT" ]]; then rm -rf -- "$TMP_ROOT" || archive_rc=$?; fi
  [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]] && workload_rc=$archive_rc; exit "$workload_rc"
}
trap archive_on_exit EXIT

export RUN_ROOT VLA_NAME TASK_ID ARROW_DEMOS EVAL_EPISODES EVAL_SEED BASE_POLICY BASE_POLICY_REVISION ACTUAL_COMMIT LIBERO_COMMIT PEFT_EPOCHS PEFT_BATCH_SIZE
"$PYTHON" - "$RUN_ROOT/run_context.json" <<'PY'
import json, os, pathlib, platform
from importlib.metadata import PackageNotFoundError, version
def pkg(name):
    try: return version(name)
    except PackageNotFoundError: return None
payload = {
    "schema": "smolvla_peft_arrow_task.v1", "method": "peft_lora_exploratory_control",
    "vla": os.environ["VLA_NAME"], "task_id": int(os.environ["TASK_ID"]), "arrow_demos": int(os.environ["ARROW_DEMOS"]),
    "eval_episodes": int(os.environ["EVAL_EPISODES"]), "eval_seed": int(os.environ["EVAL_SEED"]),
    "base_policy": os.environ["BASE_POLICY"], "base_policy_revision": os.environ["BASE_POLICY_REVISION"],
    "repo_commit": os.environ["ACTUAL_COMMIT"], "libero_commit": os.environ["LIBERO_COMMIT"],
    "python": platform.python_version(), "runtime_versions": {n: pkg(n) for n in ("torch", "lerobot", "peft", "accelerate")},
    "data_contract": "source-grounded treatment frames with ground-truth scene-graph arrows; not runtime Arrow-controller corrections",
    "train_epochs": int(os.environ["PEFT_EPOCHS"]), "batch_size": int(os.environ["PEFT_BATCH_SIZE"]),
}
pathlib.Path(os.sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

printf '[%s] runtime import preflight\n' "$(date +'%F %T')"
"$PYTHON" -c 'import torch, lerobot, peft, accelerate, h5py, cv2, PIL, numpy, robosuite; print("imports OK", torch.__version__, lerobot.__version__, peft.__version__, accelerate.__version__)'
"$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' || die 'CUDA is unavailable on the allocated node'

printf '[%s] baseline evaluation (task 0, 10 sealed episodes)\n' "$(date +'%F %T')"
MODELS="$BASE_POLICY" CONTEXT_MODE=standard TASK_IDS='[0]' N_EPISODES="$EVAL_EPISODES" BATCH_SIZE=1 DEVICE=cuda SEED="$EVAL_SEED" SUITE_MODE=sealed_randomized OUTPUT_DIR="$BASELINE_EVAL_ROOT" \
  "$PYTHON" "$EVALUATOR" --eval.use_async_envs=false

printf '[%s] converting and verifying %s arrow demonstrations\n' "$(date +'%F %T')" "$ARROW_DEMOS"
"$PYTHON" "$CONVERTER" --mode convert-pair --tasks "$TASK_ID" --demos-per-task "$ARROW_DEMOS" --allow-subset --data-dir "$LIBERO_DATA_DIR" --output-root "$DATA_ROOT"
"$PYTHON" "$CONVERTER" --mode verify --data-dir "$LIBERO_DATA_DIR" --output-root "$DATA_ROOT"
"$PYTHON" - "$DATA_ROOT/sealed_lora_pair_manifest.json" "$DATA_ROOT/sealed_lora_pair_verified.json" "$ARROW_DEMOS" <<'PY'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
sentinel = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding='utf-8'))
expected = int(sys.argv[3])
if manifest.get('pair_kind') != 'sealed_lora_control_treatment': raise SystemExit('wrong pair kind')
if manifest.get('task_ids') != [0] or manifest.get('total_episodes') != expected: raise SystemExit('arrow dataset task/count mismatch')
if manifest.get('launch_eligibility') != 'subset_smoke_not_launchable': raise SystemExit('subset unexpectedly marked full launchable')
if sentinel.get('total_episodes') != expected or sentinel.get('task_ids') != [0]: raise SystemExit('verification sentinel mismatch')
if not sentinel.get('dataset_fingerprints', {}).get('treatment', {}).get('sha256'): raise SystemExit('treatment fingerprint missing')
PY

TOTAL_FRAMES="$($PYTHON - "$DATA_ROOT/treatment/meta/info.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1], encoding='utf-8')).get('total_frames')
if not isinstance(value,int) or value <= 0: raise SystemExit('invalid dataset total_frames')
print(value)
PY
)"
UPDATES_PER_EPOCH=$(( (TOTAL_FRAMES + PEFT_BATCH_SIZE - 1) / PEFT_BATCH_SIZE ))
STEPS="${PEFT_STEPS:-$((UPDATES_PER_EPOCH * PEFT_EPOCHS))}"
SAVE_FREQ="${PEFT_SAVE_FREQ:-$UPDATES_PER_EPOCH}"
[[ "$STEPS" =~ ^[1-9][0-9]*$ && "$SAVE_FREQ" =~ ^[1-9][0-9]*$ ]] || die 'PEFT_STEPS/PEFT_SAVE_FREQ must be positive integers'
export DATA_ROOT TRAIN_ROOT TRAIN_META_ROOT BASELINE_EVAL_ROOT ADAPTED_EVAL_ROOT STEPS SAVE_FREQ UPDATES_PER_EPOCH

"$PYTHON" "$AUDITOR" --generate-expected --base-policy "$BASE_POLICY" --output "$TRAIN_META_ROOT/expected_adapter_inventory.json"
"$PYTHON" - "$TRAIN_META_ROOT/training_plan.json" <<'PY'
import json, os, pathlib
out = pathlib.Path(os.environ['TRAIN_META_ROOT']) / 'training_plan.json'
payload = {
    'schema': 'smolvla_peft_arrow_task_training.v1',
    'method': 'peft_lora',
    'vla': os.environ['VLA_NAME'],
    'task_id': int(os.environ['TASK_ID']),
    'base_policy': os.environ['BASE_POLICY'],
    'base_policy_revision': os.environ['BASE_POLICY_REVISION'],
    'dataset_root': str((pathlib.Path(os.environ['DATA_ROOT']) / 'treatment').resolve()),
    'dataset_manifest': str((pathlib.Path(os.environ['DATA_ROOT']) / 'sealed_lora_pair_manifest.json').resolve()),
    'dataset_variant': 'treatment',
    'arrow_demos': int(os.environ['ARROW_DEMOS']),
    'steps': int(os.environ['STEPS']),
    'save_freq': int(os.environ['SAVE_FREQ']),
    'epochs': int(os.environ['PEFT_EPOCHS']),
    'updates_per_epoch': int(os.environ['UPDATES_PER_EPOCH']),
    'batch_size': int(os.environ['PEFT_BATCH_SIZE']),
    'seed': int(os.environ['EVAL_SEED']),
    'peft_r': 16,
    'trainable_scope': 'native_LeRobot_SmolVLA_PEFT',
    'repo_commit': os.environ['ACTUAL_COMMIT'],
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

printf '[%s] native SmolVLA PEFT training (%s steps)\n' "$(date +'%F %T')" "$STEPS"
# LeRobot refuses to train into an existing output directory when resume=false.
# Keep provenance metadata in a sibling directory, then create the trainer output
# directory immediately before invoking the native trainer.
[[ ! -e "$TRAIN_ROOT" ]] || die 'trainer output directory unexpectedly exists'
mkdir "$TRAIN_ROOT"
"$PYTHON" "$TRAINER" \
  --policy.push_to_hub=false --peft.r=16 \
  --dataset.repo_id=local/libero_spatial_treatment \
  --dataset.root="$DATA_ROOT/treatment" --output_dir="$TRAIN_ROOT" \
  --steps="$STEPS" --save_freq="$SAVE_FREQ" --eval_freq=0 \
  --batch_size="$PEFT_BATCH_SIZE" --policy.device=cuda --seed="$EVAL_SEED" \
  --policy.path="$BASE_POLICY"

CHECKPOINT_DIR="$($PYTHON - "$TRAIN_ROOT/checkpoints" <<'PY'
import pathlib,sys
root=pathlib.Path(sys.argv[1])
candidates=[]
for path in root.glob('*/pretrained_model/adapter_model.safetensors'):
    try: step=int(path.parent.parent.name)
    except ValueError: continue
    if path.is_file() and path.stat().st_size > 0: candidates.append((step,path.parent))
if not candidates: raise SystemExit('no adapter checkpoint produced')
print(max(candidates, key=lambda item: item[0])[1].resolve())
PY
)"
cp -p "$TRAIN_META_ROOT/expected_adapter_inventory.json" "$TRAIN_ROOT/expected_adapter_inventory.json"
cp -p "$TRAIN_META_ROOT/training_plan.json" "$TRAIN_ROOT/training_plan.json"
"$PYTHON" "$AUDITOR" --checkpoint "$CHECKPOINT_DIR" --expected-inventory "$TRAIN_ROOT/expected_adapter_inventory.json"
printf 'adapter_checkpoint=%s\n' "$CHECKPOINT_DIR" >> "$RUN_ROOT/run_context.env"

printf '[%s] adapted evaluation (task 0, same 10 sealed episodes)\n' "$(date +'%F %T')"
MODELS="$CHECKPOINT_DIR" CONTEXT_MODE=standard TASK_IDS='[0]' N_EPISODES="$EVAL_EPISODES" BATCH_SIZE=1 DEVICE=cuda SEED="$EVAL_SEED" SUITE_MODE=sealed_randomized OUTPUT_DIR="$ADAPTED_EVAL_ROOT" \
  "$PYTHON" "$EVALUATOR" --eval.use_async_envs=false

BASELINE_EVAL_FILE="$($PYTHON - "$BASELINE_EVAL_ROOT" <<'PY'
import pathlib,sys
items=sorted(pathlib.Path(sys.argv[1]).rglob('eval_info.json'))
if len(items) != 1: raise SystemExit(f'expected one baseline eval_info.json, found {len(items)}')
print(items[0])
PY
)"
ADAPTED_EVAL_FILE="$($PYTHON - "$ADAPTED_EVAL_ROOT" <<'PY'
import pathlib,sys
items=sorted(pathlib.Path(sys.argv[1]).rglob('eval_info.json'))
if len(items) != 1: raise SystemExit(f'expected one adapted eval_info.json, found {len(items)}')
print(items[0])
PY
)"
export CHECKPOINT_DIR BASELINE_EVAL_FILE ADAPTED_EVAL_FILE ARCHIVE_ROOT

"$PYTHON" - <<'PY'
import json, os, pathlib
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from vla_benchmarking.libero.automatic_ttt.peft_artifacts import save_peft_adapter

def pkg(name):
    try: return version(name)
    except PackageNotFoundError: return None

def stats(path):
    data = json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    overall = data.get('overall', data)
    successes = overall.get('n_successes', overall.get('successes'))
    episodes = overall.get('n_episodes', overall.get('episodes'))
    rate = float(successes) / float(episodes) if episodes else None
    return {'successes': successes, 'episodes': episodes, 'success_rate': rate, 'path': str(pathlib.Path(path).resolve())}

baseline = stats(os.environ['BASELINE_EVAL_FILE'])
adapted = stats(os.environ['ADAPTED_EVAL_FILE'])
run_id = f"{os.environ['VLA_NAME']}_task{os.environ['TASK_ID']}_arrow{os.environ['ARROW_DEMOS']}_job{os.environ.get('SLURM_JOB_ID','unknown')}"
manifest = save_peft_adapter(
    os.environ['CHECKPOINT_DIR'], os.environ['ARCHIVE_ROOT'],
    vla=os.environ['VLA_NAME'], task_id=int(os.environ['TASK_ID']), run_id=run_id,
    base_checkpoint=os.environ['BASE_POLICY'],
    dataset_manifest=pathlib.Path(os.environ['DATA_ROOT']) / 'sealed_lora_pair_manifest.json',
    seed=int(os.environ['EVAL_SEED']),
    train_counts={'arrow_demos': int(os.environ['ARROW_DEMOS']), 'steps': int(os.environ['STEPS']), 'save_freq': int(os.environ['SAVE_FREQ']), 'batch_size': int(os.environ['PEFT_BATCH_SIZE'])},
    eval_counts={'baseline': baseline, 'adapted': adapted},
    git_commit=os.environ['ACTUAL_COMMIT'],
    runtime_versions={name: pkg(name) for name in ('python','torch','lerobot','peft','accelerate')},
)
summary = {
    'schema': 'smolvla_peft_arrow_task_summary.v1', 'method': 'peft_lora_exploratory_control',
    'vla': os.environ['VLA_NAME'], 'task_id': int(os.environ['TASK_ID']), 'arrow_demos': int(os.environ['ARROW_DEMOS']),
    'baseline': baseline, 'adapted': adapted,
    'success_delta': (adapted['success_rate'] - baseline['success_rate']) if baseline['success_rate'] is not None and adapted['success_rate'] is not None else None,
    'adapter_artifact': asdict(manifest),
}
root = pathlib.Path(os.environ['RUN_ROOT'])
(root / 'experiment_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(root / 'ADAPTER_ARTIFACT_PATH').write_text(manifest.artifact_path + '\n', encoding='utf-8')
PY

touch "$RUN_ROOT/COMPLETED"
printf '[%s] PEFT task-0 experiment completed; archive is written on exit\n' "$(date +'%F %T')"
