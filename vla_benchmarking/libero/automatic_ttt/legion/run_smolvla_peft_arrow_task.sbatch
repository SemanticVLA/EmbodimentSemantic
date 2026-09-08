#!/usr/bin/env bash
# One task-specific SmolVLA PEFT run. The only dataset input is the
# ArrowGraspController collection manifest. There is no HDF5 converter or
# multi-task mode. Submit one array element per task (0..9).
# This pilot trains for five dataset-equivalent epochs; the exact optimizer
# step count is computed after the accepted Arrow dataset is materialized.

#SBATCH --job-name=smolvla_peft_arrow_task
#SBATCH --partition=gpu_a40
#SBATCH --exclude=compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-23:59:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%A_%a.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%A_%a.err

set -Eeuo pipefail
umask 027
export PYTHONDONTWRITEBYTECODE=1
die() { printf 'smolvla PEFT task-specific: %s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || die 'submit with sbatch'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
EXPECTED_COMMIT="${PEFT_EXPECTED_COMMIT:-}"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'PEFT_EXPECTED_COMMIT must be a 40-character SHA'
EXPECTED_CONTROLLER_HASH="${PEFT_EXPECTED_CONTROLLER_HASH:-}"
[[ "$EXPECTED_CONTROLLER_HASH" =~ ^[0-9a-f]{64}$ ]] || die 'PEFT_EXPECTED_CONTROLLER_HASH must be a 64-character SHA-256'
command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'

# Scalar task identity only. PEFT_TASK_ID defaults to SLURM_ARRAY_TASK_ID.
PEFT_TASK_ID="${PEFT_TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}"
[[ "$PEFT_TASK_ID" =~ ^[0-9]+$ && "$PEFT_TASK_ID" -ge 0 && "$PEFT_TASK_ID" -le 9 ]] || die 'PEFT_TASK_ID must be one integer in 0..9'
TASK_ID="$PEFT_TASK_ID"; TASK_IDS_JSON="[$TASK_ID]"; TASK_SCOPE="task_${TASK_ID}"
[[ "${PEFT_VLA:-smolvla}" == smolvla ]] || die 'this launcher supports PEFT_VLA=smolvla only'
VLA_NAME=smolvla

# Explicit sealed condition: 50 demos, 10 paired evaluation episodes, 5 epochs.
ARROW_DEMOS=50; EVAL_EPISODES=10; EVAL_SEED=1000; REQUESTED_EPOCHS=5; STEPS=0
MAX_ATTEMPTS=500; VLA_STEP_BUDGET=280; ARROW_STEP_BUDGET=1200
EPISODE_HORIZON=280; RESOLUTION=256; EVAL_CAMERAS='agentview_image,robot0_eye_in_hand_image'
[[ -z "${PEFT_BATCH_SIZE+x}" || "$PEFT_BATCH_SIZE" == 8 ]] || die 'PEFT_BATCH_SIZE is sealed to the paper-equivalent global batch 8'
PEFT_BATCH_SIZE=8; PEFT_R=16
LORA_ALPHA=8; LORA_DROPOUT=0.0; LORA_BIAS=none; LORA_INIT=true; LORA_RSLORA=false; LORA_FAN_IN_FAN_OUT=false
ACTION_SIDE_TARGET_REGEX='(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
OPTIMIZER_NAME=AdamW; WEIGHT_DECAY=1e-5; PEAK_LR=5e-5; OPTIMIZER_BETAS='[0.9,0.95]'; OPTIMIZER_EPS=1e-8; GRAD_CLIP=10.0
SCHEDULER_NAME=cosine_decay_with_warmup; SCHEDULER_WARMUP_STEPS=0; SCHEDULER_DECAY_STEPS=0; SCHEDULER_DECAY_LR=2.5e-6
SAVE_FREQ=1; NUM_WORKERS=4; PREFETCH_FACTOR=4; PERSISTENT_WORKERS=true; LOG_FREQ=200; TOLERANCE_S=0.0001
[[ "$PEFT_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die 'PEFT_BATCH_SIZE must be a positive integer'
[[ -z "${PEFT_EPOCHS+x}" ]] || die 'PEFT_EPOCHS is forbidden; the budget is optimizer steps'
[[ -z "${PEFT_STEPS+x}" ]] || die 'PEFT_STEPS override is forbidden; steps are derived from five epochs'
# Optional replay input. If absent, this job collects fresh-reset successful
# Arrow demonstrations after the baseline evaluation.
COLLECTION_MANIFEST="${PEFT_ARROW_COLLECTION_MANIFEST:-}"

BASE_POLICY_REVISION="${PEFT_BASE_POLICY_REVISION:-6721902bc4d61e50a3bfdb11dfb4cb626f05d102}"
BASE_POLICY="${PEFT_BASE_POLICY:-$HOME/EmbodimentSemantic/vla_benchmarking/base_models/smolvla_libero-$BASE_POLICY_REVISION}"
LIBERO_DIR="${PEFT_LIBERO_DIR:-$HOME/EmbodimentSemantic/vla_benchmarking/LIBERO}"
LIBERO_CONFIG="${PEFT_LIBERO_CONFIG:-$HOME/EmbodimentSemantic_runtime/config/config.yaml}"
# Use the canonical Arrow/LeRobot runtime: MolmoPoint is pinned to
# Transformers 4.57.1, while the generic SmolVLA environment may be 5.x.
PYTHON="${PEFT_PYTHON:-$HOME/EmbodimentSemantic_runtime/EmbodimentSemantic/grasp_controller/venv-py312-eacdc54a9663db0a/bin/python}"
LIBERO_COMMIT="${PEFT_LIBERO_COMMIT:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
TRAINER="$REPO_ROOT/vla_benchmarking/libero/finetuned_vlas/smolvla/workflows/run_lerobot_train.py"
AUDITOR="$REPO_ROOT/vla_benchmarking/libero/finetuned_vlas/smolvla/workflows/adapter_audit.py"
EVALUATOR="$REPO_ROOT/vla_benchmarking/libero/evaluation/run_lerobot_eval_with_context.py"
[[ -x "$PYTHON" ]] || die "Python is missing: $PYTHON"
for path in "$BASE_POLICY" "$LIBERO_DIR" "$LIBERO_CONFIG" "$TRAINER" "$AUDITOR" "$EVALUATOR"; do [[ -e "$path" ]] || die "required path is missing: $path"; done
"$PYTHON" - <<'PY'
from importlib.metadata import version
if version("transformers") != "4.57.1":
    raise SystemExit(f"Arrow MolmoPoint requires transformers==4.57.1, found {version('transformers')}")
print("Arrow runtime Transformers 4.57.1 verified")
PY
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
if [[ -n "$COLLECTION_MANIFEST" ]]; then COLLECTION_MANIFEST="$(realpath -e -- "$COLLECTION_MANIFEST")"; fi
[[ -d "$REPO_ROOT/.git" ]] || die 'REPO_ROOT is not a Git checkout'
ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)" || die 'cannot read checkout SHA'
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || die "checkout SHA differs: $ACTUAL_COMMIT"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'REPO_ROOT is dirty'
[[ -d "$LIBERO_DIR/.git" ]] || die 'LIBERO_DIR is not a Git checkout'
[[ "$(git -C "$LIBERO_DIR" rev-parse HEAD 2>/dev/null || true)" == "$LIBERO_COMMIT" ]] || die "LIBERO is not pinned to $LIBERO_COMMIT"
[[ -z "$(git -C "$LIBERO_DIR" status --porcelain --untracked-files=no 2>/dev/null || true)" ]] || die 'LIBERO checkout is dirty'
[[ -f "$BASE_POLICY/config.json" && -f "$BASE_POLICY/base_snapshot_manifest.json" ]] || die 'base snapshot is incomplete'
for forbidden in RESUME RESUME_CONFIG_PATH PEFT_RESUME PEFT_ADAPTER ADAPTER_PATH PEFT_BASE_ADAPTER; do [[ -z "${!forbidden+x}" ]] || die "$forbidden is forbidden"; done

# Match the already-validated canary runtime instead of depending on the
# submission shell's inherited module state.
module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"

"$PYTHON" - "$BASE_POLICY/base_snapshot_manifest.json" "$BASE_POLICY" "$BASE_POLICY_REVISION" <<'PY'
import hashlib, json, pathlib, sys
manifest_path, root, revision = map(pathlib.Path, sys.argv[1:])
data = json.loads(manifest_path.read_text(encoding="utf-8"))
if data.get("revision") != str(revision): raise SystemExit("base revision mismatch")
for name, expected in data.get("files", {}).items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected: raise SystemExit(f"base manifest hash mismatch: {name}")
PY

LABEL="${PEFT_LABEL:-smolvla_task${TASK_ID}_arrow_peft}"
[[ "$LABEL" =~ ^[A-Za-z0-9_.-]{1,96}$ ]] || die 'PEFT_LABEL is not safe'
SCRATCH_ROOT="${PEFT_SCRATCH_ROOT:-${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/peft_arrow}"
RUN_ROOT="${PEFT_RUN_ROOT:-$SCRATCH_ROOT/${LABEL}_${SLURM_JOB_ID}}"
ARCHIVE_ROOT="${PEFT_ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/peft_arrow/$VLA_NAME/$TASK_SCOPE/${LABEL}_${SLURM_JOB_ID}}"
for path in "$RUN_ROOT" "$ARCHIVE_ROOT"; do [[ "$path" = /* && "$path" != / && ! -L "$path" ]] || die "unsafe output path: $path"; done
RUN_ROOT="$(realpath -m -- "$RUN_ROOT")"; ARCHIVE_ROOT="$(realpath -m -- "$ARCHIVE_ROOT")"
case "$RUN_ROOT" in "$REPO_ROOT"|"$REPO_ROOT"/*) die 'RUN_ROOT overlaps REPO_ROOT' ;; esac
case "$ARCHIVE_ROOT" in "$REPO_ROOT"|"$REPO_ROOT"/*) die 'ARCHIVE_ROOT overlaps REPO_ROOT' ;; esac
case "$ARCHIVE_ROOT" in "$RUN_ROOT"|"$RUN_ROOT"/*) die 'ARCHIVE_ROOT is inside RUN_ROOT' ;; esac
case "$RUN_ROOT" in "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'RUN_ROOT is inside ARCHIVE_ROOT' ;; esac
[[ ! -e "$RUN_ROOT" && ! -e "$ARCHIVE_ROOT" ]] || die 'output path already exists'
TMP_ROOT="${PEFT_TMP_ROOT:-/tmp/peft-${SLURM_JOB_ID}}"
[[ "$TMP_ROOT" == "/tmp/peft-${SLURM_JOB_ID}" && ! -e "$TMP_ROOT" && ! -L "$TMP_ROOT" ]] || die 'unsafe/stale TMP_ROOT'
mkdir -p "$RUN_ROOT" "$TMP_ROOT"; export TMPDIR="$TMP_ROOT" XDG_CACHE_HOME="$RUN_ROOT/xdg"; mkdir -p "$XDG_CACHE_HOME"
DATA_ROOT="$RUN_ROOT/dataset"; BASELINE_EVAL_ROOT="$RUN_ROOT/eval_baseline"; TRAIN_ROOT="$RUN_ROOT/training"; TRAIN_META_ROOT="$RUN_ROOT/training_metadata"; ADAPTED_EVAL_ROOT="$RUN_ROOT/eval_adapted"
mkdir -p "$DATA_ROOT" "$BASELINE_EVAL_ROOT" "$TRAIN_META_ROOT" "$ADAPTED_EVAL_ROOT"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$(dirname "$LIBERO_CONFIG")}"
LIBERO_CONFIG="$LIBERO_CONFIG_PATH/config.yaml"
[[ -d "$LIBERO_CONFIG_PATH" && -f "$LIBERO_CONFIG" ]] || die 'LIBERO_CONFIG_PATH/config.yaml is missing'
ARROW_CONTROLLER_CONFIG="${ARROW_CONTROLLER_CONFIG:-$REPO_ROOT/vla_benchmarking/libero/arrow_grasp_controller/configs/canonical_molmo_rgbd_grasp.json}"
[[ -f "$ARROW_CONTROLLER_CONFIG" ]] || die 'canonical Arrow controller config is missing'
export PYTHONPATH="$REPO_ROOT:$LIBERO_DIR${PYTHONPATH:+:$PYTHONPATH}"
CONTROLLER_CONFIG_HASH="$($PYTHON - "$ARROW_CONTROLLER_CONFIG" <<'PY'
import sys
from vla_benchmarking.libero.arrow_grasp_controller.configs import load_controller_config
print(load_controller_config(sys.argv[1])["config_hash"])
PY
)"
[[ "$CONTROLLER_CONFIG_HASH" == "$EXPECTED_CONTROLLER_HASH" ]] || die 'canonical Arrow controller semantic hash differs from submitted contract'
TASK_DESCRIPTION="$($PYTHON - "$TASK_ID" <<'PY'
import sys
from vla_benchmarking.libero.evaluation.native_vla_eval import _task_description
print(_task_description(int(sys.argv[1]), {"suite_mode": "sealed_randomized"}))
PY
)"
export REPO_ROOT VLA_ROOT="$REPO_ROOT/vla_benchmarking/libero" LIBERO_DIR LIBERO_CONFIG_PATH LIBERO_CONFIG ARROW_CONTROLLER_CONFIG CONTROLLER_CONFIG_HASH TASK_DESCRIPTION PYTHONPATH="$REPO_ROOT:$LIBERO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export SMOLVLA_BASE_POLICY="$BASE_POLICY"
ARROW_HF_CACHE="${ARROW_HF_CACHE:-$HOME/EmbodimentSemantic_runtime/EmbodimentSemantic/grasp_controller/cache/huggingface}"
[[ -d "$ARROW_HF_CACHE/transformers/models--allenai--MolmoPoint-8B" ]] || die "pinned MolmoPoint cache is missing: $ARROW_HF_CACHE"
export HF_HOME="$ARROW_HF_CACHE" HF_HUB_CACHE="$ARROW_HF_CACHE/hub" TRANSFORMERS_CACHE="$ARROW_HF_CACHE/transformers" HF_MODULES_CACHE="$ARROW_HF_CACHE/modules"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=1000 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISTIC=1 ACCELERATE_MIXED_PRECISION=no
copy_tree() { local source="$1" destination="$2"; [[ -e "$source" ]] || return 0; mkdir -p "$destination"; if command -v rsync >/dev/null 2>&1; then rsync -a -- "$source/" "$destination/"; else cp -a -- "$source/." "$destination/"; fi; }
archive_on_exit() {
  local workload_rc=$? archive_rc=0; trap - EXIT; set +e
  mkdir -p "$ARCHIVE_ROOT" || archive_rc=$?; copy_tree "$RUN_ROOT" "$ARCHIVE_ROOT/run" || archive_rc=$?
  log_root="${SLURM_LOG_ROOT:-$HOME/EmbodimentSemantic_runtime/operator/logs}"; mkdir -p "$ARCHIVE_ROOT/slurm_logs" || archive_rc=$?
  for suffix in out err; do log_path="$log_root/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${suffix}"; [[ -f "$log_path" ]] && cp -p -- "$log_path" "$ARCHIVE_ROOT/slurm_logs/" || true; done
  if [[ -d "$ARCHIVE_ROOT" ]]; then
    find "$ARCHIVE_ROOT" -type f ! -name archive_status.env ! -name inventory.sha256 ! -name tree_sha256 -print0 | sort -z | xargs -0 -r sha256sum > "$ARCHIVE_ROOT/inventory.sha256" || archive_rc=$?
    sha256sum "$ARCHIVE_ROOT/inventory.sha256" | awk '{print tolower($1)}' > "$ARCHIVE_ROOT/tree_sha256" || archive_rc=$?
    printf 'schema=smolvla_peft_arrow_task_job.v3\njob_id=%s\nstatus=%s\nworkload_exit_code=%s\nrepo_commit=%s\nvla=%s\ntask_scope=%s\ntask_id=%s\narrow_demos=%s\nsteps=%s\nbatch_size=%s\nrun_root=%s\narchive_root=%s\n' "$SLURM_JOB_ID" "$([[ "$workload_rc" -eq 0 && "$archive_rc" -eq 0 ]] && echo VERIFIED || echo PRESERVED_FAILURE)" "$workload_rc" "$ACTUAL_COMMIT" "$VLA_NAME" "$TASK_SCOPE" "$TASK_ID" "$ARROW_DEMOS" "$STEPS" "$PEFT_BATCH_SIZE" "$RUN_ROOT" "$ARCHIVE_ROOT" > "$ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
  fi
  if [[ "$TMP_ROOT" == "/tmp/peft-${SLURM_JOB_ID}" && -e "$TMP_ROOT" ]]; then rm -rf -- "$TMP_ROOT" || archive_rc=$?; fi
  [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]] && workload_rc=$archive_rc; exit "$workload_rc"
}
trap archive_on_exit EXIT

export RUN_ROOT VLA_NAME TASK_ID TASK_IDS_JSON TASK_SCOPE ARROW_DEMOS EVAL_EPISODES EVAL_SEED STEPS MAX_ATTEMPTS VLA_STEP_BUDGET ARROW_STEP_BUDGET EPISODE_HORIZON RESOLUTION EVAL_CAMERAS PEFT_BATCH_SIZE PEFT_R LORA_ALPHA LORA_DROPOUT LORA_BIAS LORA_INIT LORA_RSLORA LORA_FAN_IN_FAN_OUT ACTION_SIDE_TARGET_REGEX OPTIMIZER_NAME WEIGHT_DECAY PEAK_LR OPTIMIZER_BETAS OPTIMIZER_EPS GRAD_CLIP SCHEDULER_NAME SCHEDULER_WARMUP_STEPS SCHEDULER_DECAY_STEPS SCHEDULER_DECAY_LR SAVE_FREQ NUM_WORKERS PREFETCH_FACTOR PERSISTENT_WORKERS LOG_FREQ TOLERANCE_S BASE_POLICY BASE_POLICY_REVISION ACTUAL_COMMIT LIBERO_COMMIT COLLECTION_MANIFEST
export TRAIN_META_ROOT
printf '[%s] preflight task %s\n' "$(date +'%F %T')" "$TASK_ID"
"$PYTHON" -c 'import torch, lerobot, peft, accelerate, h5py, cv2, PIL, numpy, robosuite; from importlib.metadata import version; from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig; assert version("lerobot") == "0.5.2", version("lerobot"); assert version("peft") == "0.18.0", version("peft"); c=SmolVLAConfig(); required=("optimizer_lr","optimizer_betas","optimizer_eps","optimizer_weight_decay","optimizer_grad_clip_norm","scheduler_warmup_steps","scheduler_decay_steps","scheduler_decay_lr","chunk_size","n_action_steps","num_steps"); assert all(hasattr(c,n) for n in required); print("imports/native optimizer fields OK", torch.__version__, lerobot.__version__, peft.__version__)'
"$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' || die 'CUDA is unavailable on allocated node'
printf '[%s] baseline evaluation task %s, sealed seeds 1000..1009\n' "$(date +'%F %T')" "$TASK_ID"
PEFT_PAIRED_EVAL=1 N_ACTION_STEPS=checkpoint LIBERO_EPISODE_LENGTH="$EPISODE_HORIZON" EVAL_RESOLUTION="$RESOLUTION" EVAL_CAMERAS="$EVAL_CAMERAS" MODELS="$BASE_POLICY" CONTEXT_MODE=standard TASK_IDS="$TASK_IDS_JSON" N_EPISODES="$EVAL_EPISODES" BATCH_SIZE=1 DEVICE=cuda SEED="$EVAL_SEED" SUITE_MODE=sealed_randomized OUTPUT_DIR="$BASELINE_EVAL_ROOT" "$PYTHON" "$EVALUATOR" --eval.use_async_envs=false

BASELINE_EVAL_FILE="$($PYTHON - "$BASELINE_EVAL_ROOT" <<'PY'
import pathlib,sys
from vla_benchmarking.libero.evaluation.randomization_contract import randomization_config_payload
from vla_benchmarking.libero.shared.config import task_randomization_dimensions
from vla_benchmarking.libero.finetuned_vlas.smolvla.workflows.evaluation_contracts import validate_eval_info, validate_randomization_audit
items=sorted(pathlib.Path(sys.argv[1]).rglob("eval_info.json"))
if len(items)!=1: raise SystemExit(f"expected one baseline eval_info.json, found {len(items)}")
task=int(__import__('os').environ['TASK_ID'])
manifest={"tasks":[task], "episodes":10, "randomization_dimensions":{str(task):task_randomization_dimensions(task)}, "randomization_config":randomization_config_payload()}
validate_eval_info(items[0], manifest)
validate_randomization_audit(items[0].parent, manifest)
print(items[0])
PY
)"
export BASELINE_EVAL_FILE

# Freeze the exact ten init-state identities used by baseline evaluation. A
# different numeric seed namespace is not enough: LIBERO can map both seeds to
# the same init-state row. Fresh Arrow collection is allowed only on rows
# outside this sealed contract.
EVAL_RESET_CONTRACT="$RUN_ROOT/eval_reset_contract.json"
"$PYTHON" - "$BASELINE_EVAL_FILE" "$EVAL_RESET_CONTRACT" "$TASK_ID" <<'PY'
import json, pathlib, sys
eval_info, output, task_id = map(pathlib.Path, sys.argv[1:])
audit_path = eval_info.parent / "randomization_audit.jsonl"
rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
identities = []
for row in rows:
    identity = row.get("details", {}).get("reset_identity")
    if not isinstance(identity, dict):
        raise SystemExit("baseline audit lacks reset_identity")
    if int(identity.get("task_id", -1)) != int(task_id):
        raise SystemExit("baseline reset identity task mismatch")
    identities.append(identity)
identities.sort(key=lambda item: int(item.get("selected_init_state_index", -1)))
if [item.get("selected_init_state_index") for item in identities] != list(range(10)):
    raise SystemExit("baseline eval must reserve init-state indices 0..9 exactly")
hashes = [item.get("init_state_sha256") for item in identities]
if any(not isinstance(value, str) or len(value) != 64 for value in hashes):
    raise SystemExit("baseline eval reset identities lack init_state_sha256")
payload = {"schema_version": 1, "task_id": int(task_id),
           "reserved_eval_init_state_indices": list(range(10)),
           "reserved_eval_init_state_hashes": [value.lower() for value in hashes]}
tmp = output.with_name("." + output.name + ".partial")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(output)
PY
export EVAL_RESET_CONTRACT

# Live collection is the normal path. A supplied manifest is a replay mode
# for a previously completed collector run and is still loader-validated.
if [[ -z "$COLLECTION_MANIFEST" ]]; then
  COLLECTION_ROOT="$RUN_ROOT/collection"
  mkdir -p "$COLLECTION_ROOT"
  printf '[%s] collecting 50 fresh Arrow demonstrations for task %s\n' "$(date +'%F %T')" "$TASK_ID"
  "$PYTHON" -m vla_benchmarking.libero.automatic_ttt.collect_smolvla_arrow_corrections \
    --task-id "$TASK_ID" --task-description "$TASK_DESCRIPTION" --output-root "$COLLECTION_ROOT" \
    --accepted-target "$ARROW_DEMOS" --adaptation-seed-start 3000 --max-attempts "$MAX_ATTEMPTS" \
    --vla-step-budget "$VLA_STEP_BUDGET" --arrow-step-budget "$ARROW_STEP_BUDGET" --resolution "$RESOLUTION" \
    --controller-config-hash "$CONTROLLER_CONFIG_HASH" --collection-mode fresh_arrow \
    --reserved-eval-contract "$EVAL_RESET_CONTRACT" \
    --factory vla_benchmarking.libero.automatic_ttt.smolvla_arrow_factory:collect
  COLLECTION_MANIFEST="$COLLECTION_ROOT/collection_manifest.json"
fi
COLLECTION_MANIFEST="$(realpath -e -- "$COLLECTION_MANIFEST")" || die 'cannot resolve collection manifest after collection'
export COLLECTION_MANIFEST COLLECTION_INFO="$RUN_ROOT/collection_normalized.json"
"$PYTHON" - "$COLLECTION_MANIFEST" "$TASK_ID" "$COLLECTION_INFO" "$CONTROLLER_CONFIG_HASH" <<'PY'
import json, pathlib, sys
from vla_benchmarking.libero.automatic_ttt.peft_artifacts import load_arrow_collection_manifest
value = load_arrow_collection_manifest(sys.argv[1], task_id=int(sys.argv[2]), expected_successes=50)
if value["controller_config_hash"] != sys.argv[4].lower(): raise SystemExit("collection controller hash differs from active config")
pathlib.Path(sys.argv[3]).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
DATASET_ROOT="$($PYTHON - "$COLLECTION_INFO" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["dataset_root"])
PY
)"
DATASET_FRAMES="$($PYTHON - "$DATASET_ROOT/meta/info.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1], encoding="utf-8")).get("total_frames")
if isinstance(value,bool) or not isinstance(value,int) or value <= 0: raise SystemExit("invalid dataset total_frames")
print(value)
PY
)"
STEPS="$($PYTHON - "$REQUESTED_EPOCHS" "$PEFT_BATCH_SIZE" "$DATASET_FRAMES" <<'PY'
import math,sys
frames=int(sys.argv[3]); batch=int(sys.argv[2]); epochs=int(sys.argv[1])
print(math.ceil(epochs*frames/batch))
PY
)"
EPOCH_EQUIVALENT="$($PYTHON - "$STEPS" "$PEFT_BATCH_SIZE" "$DATASET_FRAMES" <<'PY'
import sys
print(format(int(sys.argv[1])*int(sys.argv[2])/int(sys.argv[3]), ".8f"))
PY
)"
SCHEDULER_WARMUP_STEPS="$($PYTHON - "$STEPS" <<'PY'
import sys
print(int(sys.argv[1]) // 30)
PY
)"
SCHEDULER_DECAY_STEPS="$STEPS"
SAVE_FREQ="$($PYTHON - "$STEPS" <<'PY'
import sys
print(min(2000, int(sys.argv[1])))
PY
)"
export TRAIN_EXPECTED_UPDATES="$STEPS" TRAIN_EXPECTED_BASE_LR="$PEAK_LR" TRAIN_EXPECTED_WEIGHT_DECAY="$WEIGHT_DECAY" TRAIN_EXPECTED_BETAS="$OPTIMIZER_BETAS" TRAIN_EXPECTED_EPS="$OPTIMIZER_EPS" TRAIN_EXPECTED_GRAD_CLIP="$GRAD_CLIP" TRAIN_EXPECTED_SCHEDULER="$SCHEDULER_NAME" TRAIN_EXPECTED_WARMUP="$SCHEDULER_WARMUP_STEPS" TRAIN_EXPECTED_DECAY_STEPS="$SCHEDULER_DECAY_STEPS" TRAIN_EXPECTED_DECAY_LR="$SCHEDULER_DECAY_LR"
COLLECTION_SUCCESS_COUNT=50
export DATASET_ROOT DATASET_FRAMES EPOCH_EQUIVALENT COLLECTION_SUCCESS_COUNT REQUESTED_EPOCHS

# Arrow collection needs the isolated Molmo cache; SmolVLA audit, training,
# and evaluation need the separately populated SmolVLM2 cache.  Switch only
# after collection so every process resolves its pinned local model offline.
SMOLVLA_HF_CACHE="${SMOLVLA_HF_CACHE:-$HOME/.cache/huggingface}"
[[ -d "$SMOLVLA_HF_CACHE/hub/models--HuggingFaceTB--SmolVLM2-500M-Instruct/snapshots" ]] || die "pinned SmolVLM2 cache is missing: $SMOLVLA_HF_CACHE"
export HF_HOME="$SMOLVLA_HF_CACHE" HF_HUB_CACHE="$SMOLVLA_HF_CACHE/hub" TRANSFORMERS_CACHE="$SMOLVLA_HF_CACHE/hub" HF_MODULES_CACHE="$SMOLVLA_HF_CACHE/modules"

"$PYTHON" - "$RUN_ROOT/run_context.json" <<'PY'
import json, os, pathlib, platform
from importlib.metadata import PackageNotFoundError, version
def pkg(name):
    try: return version(name)
    except PackageNotFoundError: return None
collection = json.loads(pathlib.Path(os.environ["COLLECTION_INFO"]).read_text(encoding="utf-8"))
payload = {
    "schema":"smolvla_peft_arrow_task.v4", "method":"fresh_arrow_behavior_cloning_peft",
    "vla":"smolvla", "task_id":int(os.environ["TASK_ID"]), "task_ids":[int(os.environ["TASK_ID"])],
    "training_scope":"task_specific", "arrow_demos":50, "collection_mode":"fresh_arrow", "collection_manifest":os.environ["COLLECTION_MANIFEST"],
    "collection_success_count":50, "controller_config_hash":os.environ["CONTROLLER_CONFIG_HASH"],
    "source_kind":"arrow_grasp_controller_fresh_demonstration", "adaptation_seeds":collection["adaptation_seeds"],
    "eval_seeds":list(range(1000,1010)), "base_policy":os.environ["BASE_POLICY"],
    "base_policy_revision":os.environ["BASE_POLICY_REVISION"], "repo_commit":os.environ["ACTUAL_COMMIT"],
    "libero_commit":os.environ["LIBERO_COMMIT"], "python":platform.python_version(),
    "runtime_versions":{n:pkg(n) for n in ("torch","lerobot","peft","accelerate")},
    "steps":int(os.environ["STEPS"]), "requested_epochs":int(os.environ["REQUESTED_EPOCHS"]), "batch_size":int(os.environ["PEFT_BATCH_SIZE"]), "dataset_frames":int(os.environ["DATASET_FRAMES"]),
    "epoch_equivalent":float(os.environ["EPOCH_EQUIVALENT"]), "save_freq":int(os.environ["SAVE_FREQ"]),
    "episode_horizon":int(os.environ["EPISODE_HORIZON"]), "resolution":int(os.environ["RESOLUTION"]), "eval_cameras":os.environ["EVAL_CAMERAS"].split(","),
    "action_semantics":{"chunk_size":50,"n_action_steps":1,"n_obs_steps":1,"num_steps":10},
    "dataloader":{"num_workers":4,"prefetch_factor":4,"persistent_workers":True,"batch_size":int(os.environ["PEFT_BATCH_SIZE"])},
    "precision":{"use_amp":False,"accelerate_mixed_precision":"no"},
    "peft":{"r":16,"lora_alpha":8,"lora_dropout":0.0,"bias":"none","init_lora_weights":True,"use_rslora":False,"fan_in_fan_out":False,"target_modules":os.environ["ACTION_SIDE_TARGET_REGEX"],"modules_to_save":[]},
    "optimizer":{"optimizer":"AdamW","weight_decay":1e-5,"peak_learning_rate":5e-5,"betas":[0.9,0.95],"epsilon":1e-8,"gradient_clip_norm":10.0,
                  "scheduler":"cosine_decay_with_warmup","warmup_steps":int(os.environ["SCHEDULER_WARMUP_STEPS"]),"decay_steps":int(os.environ["SCHEDULER_DECAY_STEPS"]),"decay_lr":2.5e-6},
    "provenance_categories":{"steps":"USER_PILOT_DERIVED_CEIL_5_EPOCHS","requested_epochs":"USER_PILOT","batch_size":"PAPER_DERIVED","optimizer":"PAPER_EXPLICIT","peft":"SMOLVLA_PINNED","episode_horizon":"SMOLVLA_PINNED","resolution":"SMOLVLA_PINNED","eval_seeds":"USER_STUDY_DESIGN","controller":"CONTROLLER_PINNED","save_freq":"OPERATIONAL"},
}
pathlib.Path(os.sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"$PYTHON" - "$TRAIN_META_ROOT/training_plan.json" <<'PY'
import hashlib, json, os, pathlib
payload={"schema":"smolvla_peft_arrow_task_training.v6","method":"fresh_arrow_behavior_cloning_peft","training_scope":"task_specific","vla":"smolvla","task_id":int(os.environ["TASK_ID"]),"trained_task_ids":[int(os.environ["TASK_ID"])],"base_policy":os.environ["BASE_POLICY"],"base_policy_revision":os.environ["BASE_POLICY_REVISION"],"collection_manifest":os.environ["COLLECTION_MANIFEST"],"collection_manifest_sha256":hashlib.sha256(pathlib.Path(os.environ["COLLECTION_MANIFEST"]).read_bytes()).hexdigest(),"source_kind":"arrow_grasp_controller_fresh_demonstration","collection_mode":"fresh_arrow","controller_config_hash":os.environ["CONTROLLER_CONFIG_HASH"],"collection_success_count":50,"dataset_root":os.environ["DATASET_ROOT"],"dataset_frames":int(os.environ["DATASET_FRAMES"]),"steps":int(os.environ["STEPS"]),"requested_epochs":int(os.environ["REQUESTED_EPOCHS"]),"batch_size":int(os.environ["PEFT_BATCH_SIZE"]),"global_batch_size":8,"epoch_equivalent":float(os.environ["EPOCH_EQUIVALENT"]),"save_freq":int(os.environ["SAVE_FREQ"]),"seed":1000,"peft": {"r":16,"lora_alpha":8,"lora_dropout":0.0,"bias":"none","init_lora_weights":True,"use_rslora":False,"fan_in_fan_out":False,"target_modules":os.environ["ACTION_SIDE_TARGET_REGEX"],"modules_to_save":[]},"action_semantics":{"chunk_size":50,"n_action_steps":1,"n_obs_steps":1,"num_steps":10},"dataloader":{"num_workers":4,"prefetch_factor":4,"persistent_workers":True,"tolerance_s":0.0001},"precision":{"use_amp":False,"accelerate_mixed_precision":"no"},"episode_horizon":280,"resolution":256,"eval_cameras":os.environ["EVAL_CAMERAS"].split(","),"optimizer":{"optimizer":"AdamW","weight_decay":1e-5,"peak_learning_rate":5e-5,"betas":[0.9,0.95],"epsilon":1e-8,"gradient_clip_norm":10.0,"scheduler":"cosine_decay_with_warmup","warmup_steps":int(os.environ["SCHEDULER_WARMUP_STEPS"]),"decay_steps":int(os.environ["SCHEDULER_DECAY_STEPS"]),"decay_lr":2.5e-6},"eval_seeds":list(range(1000,1010)),"repo_commit":os.environ["ACTUAL_COMMIT"]}
pathlib.Path(os.sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
"$PYTHON" "$AUDITOR" --generate-expected --base-policy "$BASE_POLICY" --output "$TRAIN_META_ROOT/expected_adapter_inventory.json"

printf '[%s] training task %s for %s optimizer steps (requested_epochs=%s)\n' "$(date +'%F %T')" "$TASK_ID" "$STEPS" "$REQUESTED_EPOCHS"
[[ ! -e "$TRAIN_ROOT" ]] || die 'trainer output directory unexpectedly exists'
TRAINING_RUNTIME_EVIDENCE="$TRAIN_META_ROOT/runtime_evidence.json" "$PYTHON" "$TRAINER" \
  --policy.push_to_hub=false --peft.r=16 --peft.lora_alpha=8 --peft.lora_dropout=0.0 --peft.bias=none \
  --peft.init_lora_weights=true --peft.use_rslora=false --peft.fan_in_fan_out=false \
  --peft.target_modules="$ACTION_SIDE_TARGET_REGEX" --peft.modules_to_save="[]" \
  --dataset.repo_id="local/arrow_grasp_task_${TASK_ID}" --dataset.root="$DATASET_ROOT" --output_dir="$TRAIN_ROOT" \
  --steps="$STEPS" --save_freq="$SAVE_FREQ" --eval_freq=0 --batch_size="$PEFT_BATCH_SIZE" --seed=1000 \
  --num_workers="$NUM_WORKERS" --prefetch_factor="$PREFETCH_FACTOR" --persistent_workers="$PERSISTENT_WORKERS" \
  --log_freq="$LOG_FREQ" --tolerance_s="$TOLERANCE_S" --cudnn_deterministic=true --policy.device=cuda --policy.path="$BASE_POLICY" \
  --policy.use_amp=false --policy.n_obs_steps=1 --policy.chunk_size=50 --policy.n_action_steps=1 --policy.num_steps=10 \
  --policy.resize_imgs_with_padding='[512,512]' --policy.tokenizer_max_length=48 --policy.freeze_vision_encoder=true --policy.train_expert_only=true --policy.train_state_proj=true \
  --policy.optimizer_lr="$PEAK_LR" \
  --policy.optimizer_betas="$OPTIMIZER_BETAS" --policy.optimizer_eps="$OPTIMIZER_EPS" \
  --policy.optimizer_weight_decay="$WEIGHT_DECAY" --policy.optimizer_grad_clip_norm="$GRAD_CLIP" \
  --policy.scheduler_warmup_steps="$SCHEDULER_WARMUP_STEPS" --policy.scheduler_decay_steps="$SCHEDULER_DECAY_STEPS" --policy.scheduler_decay_lr="$SCHEDULER_DECAY_LR"

RUNTIME_EVIDENCE_SHA256="$($PYTHON - "$TRAIN_META_ROOT/runtime_evidence.json" <<'PY'
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
export RUNTIME_EVIDENCE_SHA256
"$PYTHON" - "$TRAIN_META_ROOT/runtime_evidence.json" <<'PY'
import json, math, os, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "updates": int(os.environ["STEPS"]), "base_lr": float(os.environ["PEAK_LR"]),
    "weight_decay": float(os.environ["WEIGHT_DECAY"]),
    "betas": [0.9, 0.95], "eps": float(os.environ["OPTIMIZER_EPS"]),
    "grad_clip_norm": float(os.environ["GRAD_CLIP"]), "scheduler": os.environ["SCHEDULER_NAME"],
    "warmup_steps": int(os.environ["SCHEDULER_WARMUP_STEPS"]),
    "decay_steps": int(os.environ["SCHEDULER_DECAY_STEPS"]),
    "decay_lr": float(os.environ["SCHEDULER_DECAY_LR"]),
}
if value.get("attestation_status") != "VERIFIED": raise SystemExit("training runtime evidence is not VERIFIED")
if value.get("updates_observed") != expected["updates"]: raise SystemExit("runtime evidence update count differs from derived five-epoch step count")
if value.get("optimizer_class") != "torch.optim.adamw.AdamW": raise SystemExit("runtime optimizer class is not AdamW")
if value.get("scheduler_class", "").split(".")[-1] not in {"LambdaLR", "SequentialLR", "CosineAnnealingLR"}: raise SystemExit("runtime scheduler class is not approved")
if value.get("expected_contract") != expected: raise SystemExit(f"runtime evidence contract mismatch: {value.get('expected_contract')!r}")
opt = value.get("optimizer_hyperparameters", {})
if any(abs(float(opt.get(k, "nan")) - expected[k]) > 1e-12 for k in ("lr", "weight_decay", "eps", "grad_clip_norm")): raise SystemExit("runtime optimizer evidence mismatch")
if [float(x) for x in opt.get("betas", [])] != expected["betas"]: raise SystemExit("runtime AdamW betas mismatch")
if not value.get("optimizer_scheduler_objects_checked") or not value.get("lr_milestones_consistent"): raise SystemExit("runtime object/milestone attestation missing")
if not value.get("all_losses_finite") or not value.get("all_grad_norms_finite"): raise SystemExit("non-finite runtime metrics")
for key in ("min_learning_rate", "max_learning_rate", "last_learning_rate"):
    if not math.isfinite(float(value.get(key, "nan"))): raise SystemExit(f"non-finite runtime LR: {key}")
if abs(float(value["last_learning_rate"]) - expected["decay_lr"]) > max(1e-8, expected["base_lr"] * 2e-3): raise SystemExit("final LR is not the sealed scheduler floor")
print("VERIFIED runtime training evidence", value["updates_observed"], value["last_learning_rate"])
PY

CHECKPOINT_DIR="$($PYTHON - "$TRAIN_ROOT/checkpoints" <<'PY'
import pathlib,sys
items=[]
for path in pathlib.Path(sys.argv[1]).glob("*/pretrained_model/adapter_model.safetensors"):
    try: step=int(path.parent.parent.name)
    except ValueError: continue
    if path.is_file() and path.stat().st_size > 0: items.append((step,path.parent))
if not items: raise SystemExit("no adapter checkpoint produced")
step,path=max(items,key=lambda item:item[0])
if step != int(__import__('os').environ['STEPS']): raise SystemExit(f"expected derived-step checkpoint, got {step}")
print(path.resolve())
PY
)"
cp -p "$TRAIN_META_ROOT/expected_adapter_inventory.json" "$TRAIN_ROOT/expected_adapter_inventory.json"
cp -p "$TRAIN_META_ROOT/training_plan.json" "$TRAIN_ROOT/training_plan.json"
"$PYTHON" "$AUDITOR" --checkpoint "$CHECKPOINT_DIR" --expected-inventory "$TRAIN_ROOT/expected_adapter_inventory.json"
printf 'adapter_checkpoint=%s\ncheckpoint_step=%s\n' "$CHECKPOINT_DIR" "$STEPS" >> "$RUN_ROOT/run_context.env"

printf '[%s] adapted evaluation task %s on same sealed seeds\n' "$(date +'%F %T')" "$TASK_ID"
PEFT_PAIRED_EVAL=1 N_ACTION_STEPS=checkpoint LIBERO_EPISODE_LENGTH="$EPISODE_HORIZON" EVAL_RESOLUTION="$RESOLUTION" EVAL_CAMERAS="$EVAL_CAMERAS" MODELS="$CHECKPOINT_DIR" CONTEXT_MODE=standard TASK_IDS="$TASK_IDS_JSON" N_EPISODES="$EVAL_EPISODES" BATCH_SIZE=1 DEVICE=cuda SEED="$EVAL_SEED" SUITE_MODE=sealed_randomized OUTPUT_DIR="$ADAPTED_EVAL_ROOT" "$PYTHON" "$EVALUATOR" --eval.use_async_envs=false
ADAPTED_EVAL_FILE="$($PYTHON - "$ADAPTED_EVAL_ROOT" <<'PY'
import pathlib,sys
from vla_benchmarking.libero.evaluation.randomization_contract import randomization_config_payload
from vla_benchmarking.libero.shared.config import task_randomization_dimensions
from vla_benchmarking.libero.finetuned_vlas.smolvla.workflows.evaluation_contracts import validate_eval_info, validate_randomization_audit
items=sorted(pathlib.Path(sys.argv[1]).rglob("eval_info.json"))
if len(items)!=1: raise SystemExit(f"expected one adapted eval_info.json, found {len(items)}")
task=int(__import__('os').environ['TASK_ID'])
manifest={"tasks":[task], "episodes":10, "randomization_dimensions":{str(task):task_randomization_dimensions(task)}, "randomization_config":randomization_config_payload()}
validate_eval_info(items[0], manifest)
validate_randomization_audit(items[0].parent, manifest)
print(items[0])
PY
)"
export CHECKPOINT_DIR BASELINE_EVAL_FILE ADAPTED_EVAL_FILE ARCHIVE_ROOT

"$PYTHON" - "$BASELINE_EVAL_FILE" "$ADAPTED_EVAL_FILE" <<'PY'
import json, pathlib, sys
def identities(eval_info):
    audit = pathlib.Path(eval_info).parent / "randomization_audit.jsonl"
    rows = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = []
    for row in rows:
        identity = row.get("details", {}).get("reset_identity")
        if not isinstance(identity, dict):
            raise SystemExit(f"paired evaluation audit lacks reset_identity: {audit}")
        result.append(identity)
    if len(result) != 10:
        raise SystemExit(f"paired evaluation expected exactly 10 reset identities, got {len(result)}: {audit}")
    return result
baseline = identities(sys.argv[1]); adapted = identities(sys.argv[2])
if baseline != adapted:
    raise SystemExit("baseline/adapted evaluations did not use identical sealed reset identities")
print("VERIFIED paired sealed reset identities", len(baseline))
PY

"$PYTHON" - <<'PY'
import hashlib, json, os, pathlib
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from vla_benchmarking.libero.automatic_ttt.peft_artifacts import save_peft_adapter
def pkg(name):
    try: return version(name)
    except PackageNotFoundError: return None
def stats(path):
    data=json.loads(pathlib.Path(path).read_text(encoding="utf-8")); overall=data.get("overall",data)
    successes=overall.get("n_successes",overall.get("successes")); episodes=overall.get("n_episodes",overall.get("episodes"))
    if successes is None: successes=sum(bool(v) for t in data.get("per_task",[]) for v in t.get("metrics",{}).get("successes",[]))
    if episodes is None: episodes=sum(len(t.get("metrics",{}).get("successes",[])) for t in data.get("per_task",[]))
    return {"successes":int(successes),"episodes":int(episodes),"success_rate":float(successes)/float(episodes) if episodes else None,"seeds":list(range(1000,1010)),"path":str(pathlib.Path(path).resolve())}
baseline=stats(os.environ["BASELINE_EVAL_FILE"]); adapted=stats(os.environ["ADAPTED_EVAL_FILE"])
manifest=save_peft_adapter(
    os.environ["CHECKPOINT_DIR"],os.environ["ARCHIVE_ROOT"],vla="smolvla",task_id=int(os.environ["TASK_ID"]),
    run_id=f"smolvla_task{os.environ['TASK_ID']}_arrow50_steps{os.environ['STEPS']}_job{os.environ.get('SLURM_JOB_ID','unknown')}",
    base_checkpoint=os.environ["BASE_POLICY"],base_checkpoint_revision=os.environ["BASE_POLICY_REVISION"],
    collection_manifest=os.environ["COLLECTION_MANIFEST"],collection_success_count=50,controller_config_hash=os.environ["CONTROLLER_CONFIG_HASH"],checkpoint_step=int(os.environ["STEPS"]),seed=1000,
    train_counts={"successful_trajectories":50,"dataset_frames":int(os.environ["DATASET_FRAMES"]),"steps":int(os.environ["STEPS"]),"requested_epochs":int(os.environ["REQUESTED_EPOCHS"]),"batch_size":int(os.environ["PEFT_BATCH_SIZE"]),"global_batch_size":8,"epoch_equivalent":float(os.environ["EPOCH_EQUIVALENT"]),"save_freq":int(os.environ["SAVE_FREQ"]),"seed":1000,"training_scope":"task_specific","runtime_evidence_source_path":os.environ["TRAIN_META_ROOT"] + "/runtime_evidence.json","runtime_evidence_source_sha256":os.environ["RUNTIME_EVIDENCE_SHA256"]},
    eval_counts={"baseline":baseline,"adapted":adapted,"seeds":list(range(1000,1010))},
    optimizer={"optimizer":"AdamW","weight_decay":1e-5,"peak_learning_rate":5e-5,"betas":[0.9,0.95],"epsilon":1e-8,"gradient_clip_norm":10.0,"scheduler":"cosine_decay_with_warmup","warmup_steps":int(os.environ["SCHEDULER_WARMUP_STEPS"]),"decay_steps":int(os.environ["SCHEDULER_DECAY_STEPS"]),"decay_lr":2.5e-6},
    runtime_evidence=os.environ["TRAIN_META_ROOT"] + "/runtime_evidence.json",
    git_commit=os.environ["ACTUAL_COMMIT"],runtime_versions={n:pkg(n) for n in ("python","torch","lerobot","peft","accelerate")})
summary={"schema":"smolvla_peft_arrow_task_summary.v4","training_scope":"task_specific","vla":"smolvla","task_id":int(os.environ["TASK_ID"]),"trained_task_ids":[int(os.environ["TASK_ID"])] ,"source_kind":"arrow_grasp_controller_fresh_demonstration","collection_mode":"fresh_arrow","collection_manifest_sha256":hashlib.sha256(pathlib.Path(os.environ["COLLECTION_MANIFEST"]).read_bytes()).hexdigest(),"controller_config_hash":os.environ["CONTROLLER_CONFIG_HASH"],"dataset_frames":int(os.environ["DATASET_FRAMES"]),"optimizer_steps":int(os.environ["STEPS"]),"requested_epochs":int(os.environ["REQUESTED_EPOCHS"]),"global_batch_size":int(os.environ["PEFT_BATCH_SIZE"]),"epoch_equivalent":float(os.environ["EPOCH_EQUIVALENT"]),"baseline":baseline,"adapted":adapted,"success_delta":adapted["success_rate"]-baseline["success_rate"],"adapter_artifact":asdict(manifest)}
root=pathlib.Path(os.environ["RUN_ROOT"]); (root/"experiment_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"); (root/"ADAPTER_ARTIFACT_PATH.json").write_text(json.dumps(manifest.artifact_path)+"\n",encoding="utf-8")
PY
touch "$RUN_ROOT/COMPLETED"
printf '[%s] task-specific PEFT completed; archive is written on exit\n' "$(date +'%F %T')" "$TASK_ID"
