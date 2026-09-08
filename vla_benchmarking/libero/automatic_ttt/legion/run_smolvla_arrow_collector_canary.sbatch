#!/usr/bin/env bash
# One-GPU task-0 end-to-end canary. It must complete one genuine fresh-reset
# Arrow demonstration, two real PEFT updates, adapter audit/reload,
# and a paired one-seed evaluation before the full array is released.

# Required exports: REPO_ROOT, PEFT_EXPECTED_COMMIT, PEFT_CANARY_RUN_ROOT,
# PEFT_CANARY_ARCHIVE_ROOT, PEFT_CANARY_CONTROLLER_HASH.

#SBATCH --job-name=smolvla_arrow_canary
#SBATCH --partition=gpu_a40
#SBATCH --exclude=compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%A.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%A.err

set -Eeuo pipefail
umask 027
export PYTHONDONTWRITEBYTECODE=1
die() { printf 'smolvla Arrow collector canary: %s\n' "$*" >&2; exit 2; }

[[ -n "${SLURM_JOB_ID:-}" && "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || die 'submit with sbatch'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
[[ "${PEFT_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die 'PEFT_EXPECTED_COMMIT must be a 40-character SHA'
[[ "${PEFT_CANARY_CONTROLLER_HASH:-}" =~ ^[0-9a-f]{64}$ ]] || die 'PEFT_CANARY_CONTROLLER_HASH must be a 64-character SHA-256'
for name in PEFT_CANARY_RUN_ROOT PEFT_CANARY_ARCHIVE_ROOT; do
  value="${!name:-}"
  [[ "$value" = /* && "$value" != / && ! -L "$value" ]] || die "$name must be a non-root, non-symlink absolute path"
done
command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v git >/dev/null 2>&1 || die 'git is required'

REPO_ROOT="$(realpath -e -- "$REPO_ROOT")" || die 'cannot resolve REPO_ROOT'
[[ -d "$REPO_ROOT/.git" ]] || die 'REPO_ROOT is not a Git checkout'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] || die 'shared HOME checkout is forbidden'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)" || die 'cannot read repository SHA'
[[ "$actual_commit" == "$PEFT_EXPECTED_COMMIT" ]] || die "checkout SHA differs: $actual_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'REPO_ROOT is dirty'

RUN_ROOT="$(realpath -m -- "$PEFT_CANARY_RUN_ROOT")"
ARCHIVE_ROOT="$(realpath -m -- "$PEFT_CANARY_ARCHIVE_ROOT")"
case "$RUN_ROOT" in "$REPO_ROOT"|"$REPO_ROOT"/*) die 'run root overlaps repository' ;; esac
case "$ARCHIVE_ROOT" in "$REPO_ROOT"|"$REPO_ROOT"/*) die 'archive root overlaps repository' ;; esac
case "$RUN_ROOT" in "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'run/archive roots overlap' ;; esac
case "$ARCHIVE_ROOT" in "$RUN_ROOT"|"$RUN_ROOT"/*) die 'archive/run roots overlap' ;; esac
[[ ! -e "$RUN_ROOT" && ! -e "$ARCHIVE_ROOT" ]] || die 'canary output already exists'
mkdir -p "$RUN_ROOT"

copy_tree() { local source="$1" destination="$2"; mkdir -p "$destination"; if command -v rsync >/dev/null 2>&1; then rsync -a -- "$source/" "$destination/"; else cp -a -- "$source/." "$destination/"; fi; }
archive_on_exit() {
  local workload_rc=$? archive_rc=0
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT" || archive_rc=$?
  copy_tree "$RUN_ROOT" "$ARCHIVE_ROOT/run" || archive_rc=$?
  mkdir -p "$ARCHIVE_ROOT/slurm_logs" || archive_rc=$?
  for suffix in out err; do
    path="$HOME/EmbodimentSemantic_runtime/operator/logs/"$SLURM_JOB_NAME"_"$SLURM_JOB_ID".$suffix"
    [[ -f "$path" ]] && cp -p "$path" "$ARCHIVE_ROOT/slurm_logs/" || true
  done
  if [[ -d "$ARCHIVE_ROOT" ]]; then
    find "$ARCHIVE_ROOT" -type f ! -name inventory.sha256 ! -name tree_sha256 -print0 |
      sort -z | xargs -0 -r sha256sum > "$ARCHIVE_ROOT/inventory.sha256" || archive_rc=$?
    sha256sum "$ARCHIVE_ROOT/inventory.sha256" | awk '{print tolower($1)}' > "$ARCHIVE_ROOT/tree_sha256" || archive_rc=$?
    status=PRESERVED_FAILURE
    if [[ "$workload_rc" -eq 0 && "$archive_rc" -eq 0 && -s "$RUN_ROOT/COMPLETED" &&
          -s "$ARCHIVE_ROOT/inventory.sha256" && -s "$ARCHIVE_ROOT/tree_sha256" ]]; then
      status=VERIFIED
    fi
    printf 'schema=smolvla_arrow_collector_canary_job.v1\nstatus=%s\njob_id=%s\nrepo_commit=%s\ncontroller_config_hash=%s\nrun_root=%s\narchive_root=%s\n' \
      "$status" "$SLURM_JOB_ID" "$actual_commit" "$PEFT_CANARY_CONTROLLER_HASH" "$RUN_ROOT" "$ARCHIVE_ROOT" > "$ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
  fi
  [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]] && workload_rc=$archive_rc
  exit "$workload_rc"
}
trap archive_on_exit EXIT

PYTHON="${PEFT_PYTHON:-$HOME/.conda/envs/embodiment-smolvla-py312/bin/python}"
BASE_POLICY="${PEFT_BASE_POLICY:-$HOME/EmbodimentSemantic/vla_benchmarking/base_models/smolvla_libero-6721902bc4d61e50a3bfdb11dfb4cb626f05d102}"
LIBERO_DIR="${PEFT_LIBERO_DIR:-$HOME/EmbodimentSemantic/vla_benchmarking/LIBERO}"
LIBERO_CONFIG="${PEFT_LIBERO_CONFIG:-$HOME/EmbodimentSemantic_runtime/config/config.yaml}"
LIBERO_COMMIT="${PEFT_LIBERO_COMMIT:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
CONTROLLER_CONFIG="$REPO_ROOT/vla_benchmarking/libero/arrow_grasp_controller/configs/canonical_molmo_rgbd_grasp.json"
COLLECTOR="$REPO_ROOT/vla_benchmarking/libero/automatic_ttt/collect_smolvla_arrow_corrections.py"
TRAINER="$REPO_ROOT/vla_benchmarking/libero/finetuned_vlas/smolvla/workflows/run_lerobot_train.py"
AUDITOR="$REPO_ROOT/vla_benchmarking/libero/finetuned_vlas/smolvla/workflows/adapter_audit.py"
EVALUATOR="$REPO_ROOT/vla_benchmarking/libero/evaluation/run_lerobot_eval_with_context.py"
[[ -x "$PYTHON" ]] || die "Python is missing: $PYTHON"
for path in "$COLLECTOR" "$CONTROLLER_CONFIG" "$TRAINER" "$AUDITOR" "$EVALUATOR"; do
  [[ -f "$path" ]] || die "required canary component is missing: $path"
done
[[ -d "$BASE_POLICY" && -f "$BASE_POLICY/base_snapshot_manifest.json" ]] || die 'pinned SmolVLA base is missing'
[[ -d "$LIBERO_DIR/.git" && -f "$LIBERO_CONFIG" ]] || die 'pinned LIBERO runtime is missing'
[[ "$(git -C "$LIBERO_DIR" rev-parse HEAD 2>/dev/null || true)" == "$LIBERO_COMMIT" ]] || die 'LIBERO checkout is not pinned'
[[ -z "$(git -C "$LIBERO_DIR" status --porcelain --untracked-files=no 2>/dev/null || true)" ]] || die 'LIBERO checkout is dirty'

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
export PYTHONPATH="$REPO_ROOT:$LIBERO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$(dirname "$LIBERO_CONFIG")"
# Reuse the verified cache populated by the canonical Arrow controller job.
# The pinned MolmoPoint snapshot is already present on Legion; pointing both
# Hugging Face and Transformers at this tree avoids falling back to an empty
# per-job/default cache while offline mode is enabled.
ARROW_HF_CACHE="${ARROW_HF_CACHE:-$HOME/EmbodimentSemantic_runtime/EmbodimentSemantic/grasp_controller/cache/huggingface}"
[[ -d "$ARROW_HF_CACHE/transformers/models--allenai--MolmoPoint-8B" ]] || die "pinned MolmoPoint cache is missing: $ARROW_HF_CACHE"
export HF_HOME="$ARROW_HF_CACHE" HF_HUB_CACHE="$ARROW_HF_CACHE/hub" TRANSFORMERS_CACHE="$ARROW_HF_CACHE/transformers" HF_MODULES_CACHE="$ARROW_HF_CACHE/modules"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export SMOLVLA_BASE_POLICY="$BASE_POLICY" ARROW_CONTROLLER_CONFIG="$CONTROLLER_CONFIG"
export TMPDIR="$RUN_ROOT/tmp" XDG_CACHE_HOME="$RUN_ROOT/xdg"
export PYTHONHASHSEED=1000 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISTIC=1 ACCELERATE_MIXED_PRECISION=no
export TRAIN_EXPECTED_UPDATES=2 TRAIN_EXPECTED_BASE_LR=5e-5 TRAIN_EXPECTED_WEIGHT_DECAY=1e-5 TRAIN_EXPECTED_BETAS='[0.9,0.95]' TRAIN_EXPECTED_EPS=1e-8 TRAIN_EXPECTED_GRAD_CLIP=10.0 TRAIN_EXPECTED_SCHEDULER=cosine_decay_with_warmup TRAIN_EXPECTED_WARMUP=0 TRAIN_EXPECTED_DECAY_STEPS=2 TRAIN_EXPECTED_DECAY_LR=2.5e-6
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

"$PYTHON" - "$CONTROLLER_CONFIG" "$PEFT_CANARY_CONTROLLER_HASH" "$BASE_POLICY" <<'PY'
import json, pathlib, sys
from vla_benchmarking.libero.arrow_grasp_controller.configs import load_controller_config
config_path, expected_hash, base = sys.argv[1:]
resolved = load_controller_config(config_path)
if resolved.get("config_hash") != expected_hash:
    raise SystemExit("canonical controller semantic hash mismatch")
base_manifest = json.loads((pathlib.Path(base) / "base_snapshot_manifest.json").read_text())
if base_manifest.get("revision") != "6721902bc4d61e50a3bfdb11dfb4cb626f05d102":
    raise SystemExit("base checkpoint revision mismatch")
print("canonical controller/base preflight OK")
PY
"$PYTHON" -c 'import torch, lerobot, peft; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print("CUDA/native imports OK")'

COLLECTION_ROOT="$RUN_ROOT/collection"
TASK_DESCRIPTION="$("$PYTHON" - <<'PY'
from vla_benchmarking.libero.evaluation.native_vla_eval import _task_description
print(_task_description(0, {"suite_mode": "sealed_randomized"}))
PY
 )"
EVAL_RESET_CONTRACT="$RUN_ROOT/eval_reset_contract.json"
"$PYTHON" - "$EVAL_RESET_CONTRACT" <<'PY'
import json, pathlib, sys
from vla_benchmarking.libero.evaluation.run_arrow_pick_place_eval import build_libero_env

output = pathlib.Path(sys.argv[1])
identities = []
for index in range(10):
    env = build_libero_env(
        0, 1000 + index, 256, suite_mode="sealed_randomized",
        extra_camera_names=("robot0_eye_in_hand",), init_state_index=index,
    )
    try:
        diagnostics = getattr(env, "_arrow_init_state_diagnostics", None)
        if not isinstance(diagnostics, dict):
            raise SystemExit("canary reset-contract probe lacks init-state diagnostics")
        digest = diagnostics.get("selected_row_sha256")
        if diagnostics.get("selected_index") != index or not isinstance(digest, str) or len(digest) != 64:
            raise SystemExit(f"invalid canary reset identity for index {index}: {diagnostics}")
        identities.append(digest.lower())
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
if len(set(identities)) != 10:
    raise SystemExit("canary evaluation reset hashes are not unique")
payload = {
    "schema_version": 1, "task_id": 0,
    "reserved_eval_init_state_indices": list(range(10)),
    "reserved_eval_init_state_hashes": identities,
}
tmp = output.with_name("." + output.name + ".partial")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(output)
PY
"$PYTHON" -m vla_benchmarking.libero.automatic_ttt.collect_smolvla_arrow_corrections \
  --task-id 0 --task-description "$TASK_DESCRIPTION" --output-root "$COLLECTION_ROOT" \
  --accepted-target 1 --adaptation-seed-start 3000 --max-attempts 10 --vla-step-budget 280 --arrow-step-budget 1200 --resolution 256 \
  --controller-config-hash "$PEFT_CANARY_CONTROLLER_HASH" --collection-mode fresh_arrow \
  --reserved-eval-contract "$EVAL_RESET_CONTRACT" \
  --factory vla_benchmarking.libero.automatic_ttt.smolvla_arrow_factory:collect

MANIFEST="$COLLECTION_ROOT/collection_manifest.json"
[[ -s "$MANIFEST" && ! -L "$MANIFEST" ]] || die 'collector did not produce an immutable manifest'
"$PYTHON" - "$MANIFEST" "$PEFT_CANARY_CONTROLLER_HASH" "$PEFT_EXPECTED_COMMIT" "$COLLECTION_ROOT" <<'PY'
import hashlib, json, pathlib, sys
from vla_benchmarking.libero.automatic_ttt.peft_artifacts import load_arrow_collection_manifest
manifest, controller_hash, commit, root = sys.argv[1:]
value = load_arrow_collection_manifest(manifest, task_id=0, expected_successes=1)
if value.get("controller_config_hash") != controller_hash:
    raise SystemExit("validated manifest controller hash mismatch")
if value.get("collection_mode") != "fresh_arrow":
    raise SystemExit("canary collection mode is not fresh_arrow")
raw = json.loads(pathlib.Path(manifest).read_text(encoding="utf-8"))
if raw.get("starts_from_reset") is not True or raw.get("vla_called") is not False:
    raise SystemExit("canary did not prove fresh-reset Arrow-only collection")
payload = {
    "schema": "arrow_grasp_controller_collector_canary.v2", "status": "collection_passed",
    "repo_commit": commit, "controller_contract": "arrow_grasp_controller.v1",
    "controller_contract_sha256": controller_hash, "task_id": 0, "accepted_successes": 1,
    "collection_manifest": str(pathlib.Path(manifest).resolve()),
    "collection_manifest_sha256": hashlib.sha256(pathlib.Path(manifest).read_bytes()).hexdigest(),
}
path = pathlib.Path(root).parent / "collection_receipt.json"
encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
tmp = path.with_name("." + path.name + ".partial")
tmp.write_bytes(encoded)
tmp.replace(path)
print(path)
PY

DATASET_ROOT="$("$PYTHON" - "$MANIFEST" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["dataset_root"])
PY
)"
[[ -f "$DATASET_ROOT/meta/info.json" ]] || die 'canary native dataset is incomplete'

# Exercise the exact production PEFT configuration for two updates.  The
# shorter step count is a runtime smoke budget, not a five-epoch result;
# production jobs derive their exact steps from the accepted dataset.
TRAIN_ROOT="$RUN_ROOT/training_smoke"
TRAIN_META="$RUN_ROOT/training_smoke_metadata"
mkdir -p "$TRAIN_META"
"$PYTHON" "$AUDITOR" --generate-expected --base-policy "$BASE_POLICY" --output "$TRAIN_META/expected_adapter_inventory.json"
TRAINING_RUNTIME_EVIDENCE="$TRAIN_META/runtime_evidence.json" "$PYTHON" "$TRAINER" \
  --policy.push_to_hub=false --peft.r=16 --peft.lora_alpha=8 --peft.lora_dropout=0.0 --peft.bias=none \
  --peft.init_lora_weights=true --peft.use_rslora=false --peft.fan_in_fan_out=false \
  --peft.target_modules='(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))' --peft.modules_to_save='[]' \
  --dataset.repo_id=local/arrow_grasp_task_0_canary --dataset.root="$DATASET_ROOT" --output_dir="$TRAIN_ROOT" \
  --steps=2 --save_freq=2 --eval_freq=0 --batch_size=8 --seed=1000 \
  --num_workers=4 --prefetch_factor=4 --persistent_workers=true --log_freq=1 --tolerance_s=0.0001 --cudnn_deterministic=true \
  --policy.device=cuda --policy.path="$BASE_POLICY" --policy.use_amp=false --policy.n_obs_steps=1 --policy.chunk_size=50 --policy.n_action_steps=1 --policy.num_steps=10 \
  --policy.resize_imgs_with_padding='[512,512]' --policy.tokenizer_max_length=48 --policy.freeze_vision_encoder=true --policy.train_expert_only=true --policy.train_state_proj=true \
  --policy.optimizer_lr=5e-5 --policy.optimizer_betas='[0.9,0.95]' --policy.optimizer_eps=1e-8 \
  --policy.optimizer_weight_decay=1e-5 --policy.optimizer_grad_clip_norm=10.0 \
  --policy.scheduler_warmup_steps=0 --policy.scheduler_decay_steps=2 --policy.scheduler_decay_lr=2.5e-6

CHECKPOINT_DIR="$("$PYTHON" - "$TRAIN_ROOT/checkpoints" <<'PY'
import pathlib,sys
items=[]
for path in pathlib.Path(sys.argv[1]).glob("*/pretrained_model/adapter_model.safetensors"):
    try: step=int(path.parent.parent.name)
    except ValueError: continue
    if path.is_file() and path.stat().st_size: items.append((step,path.parent))
if not items or max(items)[0] != 2: raise SystemExit("canary did not produce the two-update adapter")
print(max(items)[1].resolve())
PY
)"
"$PYTHON" "$AUDITOR" --checkpoint "$CHECKPOINT_DIR" --expected-inventory "$TRAIN_META/expected_adapter_inventory.json"
"$PYTHON" - "$TRAIN_META/runtime_evidence.json" <<'PY'
import json,math,sys
v=json.load(open(sys.argv[1], encoding="utf-8"))
if v.get("attestation_status") != "VERIFIED" or v.get("updates_observed") != 2 or not v.get("all_losses_finite") or not v.get("all_grad_norms_finite"):
    raise SystemExit(f"invalid two-update runtime evidence: {v}")
if not all(math.isfinite(float(v[k])) for k in ("min_learning_rate","max_learning_rate","last_learning_rate")):
    raise SystemExit("canary learning-rate evidence is non-finite")
if not v.get("optimizer_scheduler_objects_checked") or not v.get("lr_milestones_consistent"):
    raise SystemExit("canary optimizer/scheduler attestation is incomplete")
PY

for phase in baseline adapted; do
  policy="$BASE_POLICY"; [[ "$phase" == adapted ]] && policy="$CHECKPOINT_DIR"
  out="$RUN_ROOT/eval_${phase}"
  PEFT_PAIRED_EVAL=1 N_ACTION_STEPS=checkpoint LIBERO_EPISODE_LENGTH=280 EVAL_RESOLUTION=256 \
    EVAL_CAMERAS=agentview_image,robot0_eye_in_hand_image MODELS="$policy" CONTEXT_MODE=standard \
    TASK_IDS='[0]' N_EPISODES=1 BATCH_SIZE=1 DEVICE=cuda SEED=1000 SUITE_MODE=sealed_randomized \
    OUTPUT_DIR="$out" "$PYTHON" "$EVALUATOR" --eval.use_async_envs=false
  [[ "$(find "$out" -name eval_info.json -type f | wc -l)" -eq 1 ]] || die "${phase} canary evaluation did not emit one eval_info.json"
done

"$PYTHON" - "$RUN_ROOT/eval_baseline" "$RUN_ROOT/eval_adapted" <<'PY'
import json, pathlib, sys
from vla_benchmarking.libero.evaluation.randomization_contract import randomization_config_payload
from vla_benchmarking.libero.shared.config import task_randomization_dimensions
from vla_benchmarking.libero.finetuned_vlas.smolvla.workflows.evaluation_contracts import validate_eval_info, validate_randomization_audit

def validate(root):
    items = sorted(pathlib.Path(root).rglob("eval_info.json"))
    if len(items) != 1:
        raise SystemExit(f"expected one canary eval_info.json under {root}, found {len(items)}")
    contract = {
        "tasks": [0], "episodes": 1,
        "randomization_dimensions": {"0": task_randomization_dimensions(0)},
        "randomization_config": randomization_config_payload(),
    }
    validate_eval_info(items[0], contract)
    validate_randomization_audit(items[0].parent, contract)
    audit = items[0].parent / "randomization_audit.jsonl"
    rows = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines() if line.strip()]
    identities = [row.get("details", {}).get("reset_identity") for row in rows]
    if len(identities) != 1 or not isinstance(identities[0], dict):
        raise SystemExit(f"canary evaluation lacks one reset_identity: {audit}")
    return identities

baseline = validate(sys.argv[1])
adapted = validate(sys.argv[2])
if baseline != adapted:
    raise SystemExit("canary baseline/adapted reset identities differ")
print("VERIFIED canary paired sealed reset identity")
PY

"$PYTHON" - "$RUN_ROOT/collector_canary_marker.json" "$PEFT_EXPECTED_COMMIT" "$PEFT_CANARY_CONTROLLER_HASH" "$CHECKPOINT_DIR" <<'PY'
import json,pathlib,sys
path,commit,controller,checkpoint=sys.argv[1:]
payload={"schema":"smolvla_arrow_peft_e2e_canary.v1","status":"passed","repo_commit":commit,
         "controller_contract_sha256":controller,"task_id":0,"accepted_successes":1,
         "optimizer_updates":2,"adapter_checkpoint":str(pathlib.Path(checkpoint).resolve()),
         "paired_eval_seeds":[1000],"production_requested_epochs":5}
target=pathlib.Path(path); tmp=target.with_name("."+target.name+".partial")
tmp.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n",encoding="utf-8"); tmp.replace(target)
PY

touch "$RUN_ROOT/COMPLETED"
printf '[%s] genuine Arrow + PEFT + paired-eval canary passed; marker written\n' "$(date +'%F %T')"
