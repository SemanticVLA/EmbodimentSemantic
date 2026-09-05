#!/usr/bin/env bash
# This launcher must retain LF endings for Bash/SLURM portability.
# Run once with NO_ARROW_EVAL_SCOPE=smoke, then submit the full scope with
# --dependency=afterok:<smoke-job-id>.

#SBATCH --job-name=noarrow_vla_eval
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

die() {
  printf 'no-arrow sealed evaluation: %s\n' "$*" >&2
  exit 2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit with sbatch; direct execution is refused'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
[[ "${NO_ARROW_EVAL_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || die 'NO_ARROW_EVAL_EXPECTED_COMMIT must be a lowercase 40-character SHA'
[[ "${NO_ARROW_EVAL_LABEL:-}" =~ ^[A-Za-z0-9_.-]{1,96}$ ]] \
  || die 'NO_ARROW_EVAL_LABEL is required and must be safe'

SCOPE="${NO_ARROW_EVAL_SCOPE:-}"
case "$SCOPE" in
  smoke) EPISODES=1 ;;
  full) EPISODES=50 ;;
  *) die 'NO_ARROW_EVAL_SCOPE must be smoke or full' ;;
esac

command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")" || die 'cannot resolve REPO_ROOT'
actual_root="$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null)" \
  || die 'REPO_ROOT is not a Git checkout'
actual_root="$(realpath -e -- "$actual_root")" || die 'cannot resolve Git root'
[[ "$actual_root" == "$REPO_ROOT" ]] || die 'REPO_ROOT is not the worktree root'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] \
  || die 'shared HOME checkout is forbidden; use an isolated immutable release'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)" \
  || die 'cannot read REPO_ROOT commit'
[[ "$actual_commit" == "$NO_ARROW_EVAL_EXPECTED_COMMIT" ]] \
  || die "checkout SHA differs: $actual_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] \
  || die 'REPO_ROOT is dirty'

PYTHON="/home/hjaber/.conda/envs/embodiment-smolvla-py312/bin/python"
ADAPTER_DIR="/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/runs/legion_no_arrow_lora_full_s1000_v1_no_arrow_treatment_1910197/checkpoints/029190/pretrained_model"
TRAINING_MANIFEST="/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/runs/legion_no_arrow_lora_full_s1000_v1_no_arrow_treatment_1910197/training_manifest.json"
EXPECTED_ADAPTER_SHA256="80b3c23fc3987530d57766ab45ed33db918f08983739139c1ff0397184cc7092"
EXPECTED_ADAPTER_CONFIG_SHA256="6382c39c1df2e4ccfc43946b36aa014abb73e1767366f009999415981b01efb1"
EXPECTED_TRAIN_CONFIG_SHA256="89a570fb1d07e93ec16adde1e78f18b0f0f7c0148da7896a7b5ba257797658f8"
EXPECTED_TRAINING_MANIFEST_SHA256="95e376aff504265bea2bb53e63cc221fb42d7baa01dd6c3810317de85875c391"
ENTRYPOINT="$REPO_ROOT/vla_benchmarking/arrow_finetuned_vla/workflows/run_no_arrow_sealed_eval.py"
LAUNCHER="$REPO_ROOT/vla_benchmarking/arrow_finetuned_vla/smolvla_no_arrows/legion/run_sealed_randomized_eval.sbatch"
SCRATCH_ROOT="${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/no_arrow_vla_eval"
RUN_ROOT="$SCRATCH_ROOT/runs/${NO_ARROW_EVAL_LABEL}_${SCOPE}_${SLURM_JOB_ID}"
OUTPUT_ROOT="$RUN_ROOT/results"
ARCHIVE_ROOT="$HOME/EmbodimentSemantic_archive/no_arrow_vla_eval/${NO_ARROW_EVAL_LABEL}_${SCOPE}_${SLURM_JOB_ID}"

[[ -x "$PYTHON" ]] || die "Python environment is missing: $PYTHON"
[[ -f "$ENTRYPOINT" ]] || die "entrypoint is missing: $ENTRYPOINT"
[[ -f "$LAUNCHER" ]] || die "launcher is missing: $LAUNCHER"
[[ -f "$ADAPTER_DIR/adapter_model.safetensors" ]] || die 'adapter artifact is missing'
[[ -f "$ADAPTER_DIR/adapter_config.json" ]] || die 'adapter config is missing'
[[ -f "$ADAPTER_DIR/train_config.json" ]] || die 'adapter train config is missing'
[[ -f "$TRAINING_MANIFEST" ]] || die 'training manifest is missing'
[[ "$(sha256sum -- "$ADAPTER_DIR/adapter_model.safetensors" | awk '{print tolower($1)}')" == "$EXPECTED_ADAPTER_SHA256" ]] \
  || die 'adapter SHA-256 does not match the protected 43% model'
[[ "$(sha256sum -- "$ADAPTER_DIR/adapter_config.json" | awk '{print tolower($1)}')" == "$EXPECTED_ADAPTER_CONFIG_SHA256" ]] \
  || die 'adapter-config SHA-256 does not match the protected 43% model'
[[ "$(sha256sum -- "$ADAPTER_DIR/train_config.json" | awk '{print tolower($1)}')" == "$EXPECTED_TRAIN_CONFIG_SHA256" ]] \
  || die 'train-config SHA-256 does not match the protected 43% model'
[[ "$(sha256sum -- "$TRAINING_MANIFEST" | awk '{print tolower($1)}')" == "$EXPECTED_TRAINING_MANIFEST_SHA256" ]] \
  || die 'training-manifest SHA-256 does not match the protected run'
[[ ! -e "$RUN_ROOT" ]] || die "run root already exists: $RUN_ROOT"
[[ ! -e "$ARCHIVE_ROOT" ]] || die "archive root already exists: $ARCHIVE_ROOT"
mkdir -p "$RUN_ROOT" "$OUTPUT_ROOT"

copy_tree() {
  local source="$1" destination="$2"
  [[ -e "$source" ]] || return 0
  mkdir -p "$destination"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a -- "$source/" "$destination/"
  else
    cp -a -- "$source/." "$destination/"
  fi
}

archive_on_exit() {
  local workload_rc=$? archive_rc=0 status=INCOMPLETE
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT" || archive_rc=$?
  if [[ "$archive_rc" -eq 0 ]]; then
    copy_tree "$RUN_ROOT" "$ARCHIVE_ROOT/run" || archive_rc=$?
    log_root="$HOME/EmbodimentSemantic_runtime/operator/logs"
    mkdir -p "$ARCHIVE_ROOT/slurm_logs" || archive_rc=$?
    for suffix in out err; do
      log_path="$log_root/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${suffix}"
      [[ -f "$log_path" ]] && cp -p -- "$log_path" "$ARCHIVE_ROOT/slurm_logs/" || true
    done
  fi
  if [[ "$archive_rc" -ne 0 ]]; then
    status=FAILED
  elif [[ "$workload_rc" -eq 0 && -s "$ARCHIVE_ROOT/run/results/no_arrow_trained_sealed_randomized_summary.csv" && -s "$ARCHIVE_ROOT/run/results/no_arrow_trained_sealed_randomized_manifest.json" ]]; then
    status=VERIFIED
  elif [[ "$workload_rc" -ne 0 ]]; then
    status=PRESERVED_FAILURE
  fi
  if [[ -d "$ARCHIVE_ROOT" ]]; then
    {
      printf 'schema=no_arrow_vla_sealed_eval_job.v1\n'
      printf 'archive_status=%s\njob_exit_code=%s\njob_id=%s\n' "$status" "$workload_rc" "$SLURM_JOB_ID"
      printf 'scope=%s\nepisodes_per_task=%s\nplanned_episodes=%s\n' "$SCOPE" "$EPISODES" "$((10 * EPISODES))"
      printf 'expected_commit=%s\nactual_commit=%s\n' "$NO_ARROW_EVAL_EXPECTED_COMMIT" "$actual_commit"
      printf 'adapter_sha256=%s\ntraining_manifest_sha256=%s\n' "$EXPECTED_ADAPTER_SHA256" "$EXPECTED_TRAINING_MANIFEST_SHA256"
      printf 'adapter_config_sha256=%s\ntrain_config_sha256=%s\n' "$EXPECTED_ADAPTER_CONFIG_SHA256" "$EXPECTED_TRAIN_CONFIG_SHA256"
      printf 'run_root=%s\noutput_root=%s\narchive_root=%s\n' "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT"
    } > "$ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
    find "$ARCHIVE_ROOT" -type f ! -name inventory.sha256 ! -name tree_sha256 -print0 \
      | sort -z | xargs -0 -r sha256sum > "$ARCHIVE_ROOT/inventory.sha256" || archive_rc=$?
    sha256sum "$ARCHIVE_ROOT/inventory.sha256" | awk '{print tolower($1)}' \
      > "$ARCHIVE_ROOT/tree_sha256" || archive_rc=$?
  fi
  if [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]]; then workload_rc=$archive_rc; fi
  exit "$workload_rc"
}
trap archive_on_exit EXIT

ENTRYPOINT_SHA256="$(sha256sum -- "$ENTRYPOINT" | awk '{print tolower($1)}')"
LAUNCHER_SHA256="$(sha256sum -- "$LAUNCHER" | awk '{print tolower($1)}')"
cat > "$RUN_ROOT/job_context.env" <<EOF
schema=no_arrow_vla_sealed_eval_job.v1
job_id=$SLURM_JOB_ID
job_name=$SLURM_JOB_NAME
label=$NO_ARROW_EVAL_LABEL
scope=$SCOPE
task_ids=0,1,2,3,4,5,6,7,8,9
episodes_per_task=$EPISODES
planned_episodes=$((10 * EPISODES))
episode_seed_base=1000
episode_seed_last=$((999 + EPISODES))
suite_mode=sealed_randomized
visual_condition=none
expected_commit=$NO_ARROW_EVAL_EXPECTED_COMMIT
actual_commit=$actual_commit
entrypoint=$ENTRYPOINT
entrypoint_sha256=$ENTRYPOINT_SHA256
launcher=$LAUNCHER
launcher_sha256=$LAUNCHER_SHA256
adapter_dir=$ADAPTER_DIR
adapter_sha256=$EXPECTED_ADAPTER_SHA256
adapter_config_sha256=$EXPECTED_ADAPTER_CONFIG_SHA256
train_config_sha256=$EXPECTED_TRAIN_CONFIG_SHA256
training_job_id=1910197
training_manifest=$TRAINING_MANIFEST
training_manifest_sha256=$EXPECTED_TRAINING_MANIFEST_SHA256
historical_evaluation_job_id=1910198
historical_no_arrow_result=43/100
run_root=$RUN_ROOT
output_root=$OUTPUT_ROOT
archive_root=$ARCHIVE_ROOT
EOF

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$HOME/EmbodimentSemantic_runtime/config"
export LIBERO_CONFIG="$LIBERO_CONFIG_PATH/config.yaml"
[[ -f "$LIBERO_CONFIG" ]] || die 'LIBERO configuration is missing'
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export NO_ARROW_EVAL_EXPECTED_COMMIT

"$PYTHON" - "$RUN_ROOT/runtime_versions.json" <<'PY'
import importlib.metadata as md
import json
import pathlib
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one visible CUDA device")
versions = {name: md.version(name) for name in ("lerobot", "torch", "transformers", "peft")}
payload = {
    "python": sys.executable,
    "python_version": sys.version,
    "packages": versions,
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_name": torch.cuda.get_device_name(0),
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

"$PYTHON" "$ENTRYPOINT" \
  --adapter-checkpoint "$ADAPTER_DIR" \
  --training-manifest "$TRAINING_MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --protocol "$SCOPE" \
  --episodes "$EPISODES" \
  --device cuda \
  --no-videos

[[ -s "$OUTPUT_ROOT/no_arrow_trained_sealed_randomized_manifest.json" ]] \
  || die 'evaluation manifest is missing'
[[ -s "$OUTPUT_ROOT/no_arrow_trained_sealed_randomized_summary.csv" ]] \
  || die 'evaluation summary is missing'
cell_root="$OUTPUT_ROOT/seed_1000/no_arrow_trained_no_arrows"
[[ -s "$cell_root/eval_info.json" ]] || die 'raw eval_info.json is missing'
[[ -s "$cell_root/randomization_audit.jsonl" ]] || die 'randomization audit is missing'
[[ ! -s "$cell_root/visual_relation_audit.jsonl" ]] \
  || die 'no-arrow condition unexpectedly produced a visual-arrow audit'
touch "$RUN_ROOT/COMPLETED"
