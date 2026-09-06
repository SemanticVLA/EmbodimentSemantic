#!/usr/bin/env bash
# Immutable Legion evaluator for the target-arrow-only SmolVLA adapter.
# Submit from an isolated, clean release with explicit scope and commit.
# smoke: checkpoint 000002, tasks 0 and 4, 1 episode; full: checkpoint 029190.
#SBATCH --job-name=smolvla_target_arrow_eval
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
die() { printf 'target-arrow evaluation: %s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit this launcher directly with sbatch'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
[[ "${TARGET_ARROW_EVAL_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die 'TARGET_ARROW_EVAL_EXPECTED_COMMIT must be a lowercase 40-character SHA'
[[ "${TARGET_ARROW_EVAL_LABEL:-}" =~ ^[A-Za-z0-9_.-]{1,96}$ ]] || die 'TARGET_ARROW_EVAL_LABEL is required and must be safe'
[[ -n "${TARGET_ARROW_ADAPTER:-}" && -n "${TARGET_ARROW_TRAINING_MANIFEST:-}" ]] || die 'TARGET_ARROW_ADAPTER and TARGET_ARROW_TRAINING_MANIFEST are required'
EXPECTED_COMMIT="$TARGET_ARROW_EVAL_EXPECTED_COMMIT"
LABEL="$TARGET_ARROW_EVAL_LABEL"
EVAL_SCOPE="${TARGET_ARROW_EVAL_SCOPE:-}"
case "$EVAL_SCOPE" in
  smoke) EPISODES=1; TASK_IDS="0,4" ;;
  full) EPISODES=10; TASK_IDS="0,1,2,3,4,5,6,7,8,9" ;;
  *) die 'TARGET_ARROW_EVAL_SCOPE must be explicitly set to smoke or full' ;;
esac
SEEDS="${TARGET_ARROW_SEEDS:-1000}"
[[ "$SEEDS" == 1000 ]] || die 'TARGET_ARROW_SEEDS must be exactly 1000'
command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'
command -v git >/dev/null 2>&1 || die 'git is required'
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")" || die 'cannot resolve REPO_ROOT'
actual_root="$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null)" || die 'REPO_ROOT is not a Git checkout'
actual_root="$(realpath -e -- "$actual_root")" || die 'cannot resolve Git root'
[[ "$actual_root" == "$REPO_ROOT" ]] || die 'REPO_ROOT is not the worktree root'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] || die 'shared HOME checkout is forbidden; use an isolated immutable release'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)" || die 'cannot read REPO_ROOT commit'
[[ "$actual_commit" == "$EXPECTED_COMMIT" ]] || die "checkout SHA differs: $actual_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'REPO_ROOT is dirty'
PYTHON="${TARGET_ARROW_PYTHON:-/home/hjaber/.conda/envs/embodiment-smolvla-py312/bin/python}"
DEVICE="${TARGET_ARROW_DEVICE:-cuda}"
[[ "$PYTHON" = /* && "$PYTHON" != / && -x "$PYTHON" ]] || die "pinned Legion Python is missing or not absolute: $PYTHON"
[[ "$DEVICE" == cuda ]] || die 'TARGET_ARROW_DEVICE must be cuda'
ADAPTER_DIR="$(realpath -e -- "$TARGET_ARROW_ADAPTER")" || die 'cannot resolve target-arrow adapter directory'
TRAINING_MANIFEST="$(realpath -e -- "$TARGET_ARROW_TRAINING_MANIFEST")" || die 'cannot resolve target-arrow training manifest'
[[ -d "$ADAPTER_DIR" && "${ADAPTER_DIR##*/}" == pretrained_model ]] || die 'TARGET_ARROW_ADAPTER must be a pretrained_model directory'
for artifact in adapter_model.safetensors adapter_config.json train_config.json; do [[ -s "$ADAPTER_DIR/$artifact" ]] || die "adapter artifact is missing: $artifact"; done
[[ -s "$TRAINING_MANIFEST" ]] || die 'training manifest is missing or empty'
SCRATCH_ROOT="${TARGET_ARROW_EVAL_SCRATCH_ROOT:-${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/smolvla_target_arrow_eval}"
RUN_ROOT="${TARGET_ARROW_EVAL_RUN_ROOT:-$SCRATCH_ROOT/runs/${LABEL}_${EVAL_SCOPE}_${SLURM_JOB_ID}}"
OUTPUT_ROOT="${TARGET_ARROW_OUTPUT_ROOT:-$RUN_ROOT/results}"
ARCHIVE_ROOT="${TARGET_ARROW_EVAL_ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/smolvla_target_arrow_eval/${LABEL}_${EVAL_SCOPE}_${SLURM_JOB_ID}}"

canonical_external_root() {
  local name="$1" value="$2" canonical
  [[ "$value" = /* && "$value" != / ]] || die "$name must be an absolute non-root path: $value"
  canonical="$(realpath -m -- "$value")" || die "cannot canonicalize $name: $value"
  [[ "$canonical" != / ]] || die "$name resolves to the filesystem root"
  case "$canonical" in
    "$REPO_ROOT"|"$REPO_ROOT"/*) die "$name must be outside REPO_ROOT: $canonical" ;;
  esac
  printf '%s\n' "$canonical"
}

assert_disjoint_root() {
  local left_name="$1" left="$2" right_name="$3" right="$4"
  case "$left" in
    "$right"|"$right"/*) die "$left_name overlaps $right_name: $left / $right" ;;
  esac
  case "$right" in
    "$left"|"$left"/*) die "$right_name overlaps $left_name: $right / $left" ;;
  esac
}

RUN_ROOT="$(canonical_external_root TARGET_ARROW_EVAL_RUN_ROOT "$RUN_ROOT")"
OUTPUT_ROOT="$(canonical_external_root TARGET_ARROW_OUTPUT_ROOT "$OUTPUT_ROOT")"
ARCHIVE_ROOT="$(canonical_external_root TARGET_ARROW_EVAL_ARCHIVE_ROOT "$ARCHIVE_ROOT")"
assert_disjoint_root TARGET_ARROW_EVAL_ARCHIVE_ROOT "$ARCHIVE_ROOT" TARGET_ARROW_EVAL_RUN_ROOT "$RUN_ROOT"
assert_disjoint_root TARGET_ARROW_EVAL_ARCHIVE_ROOT "$ARCHIVE_ROOT" TARGET_ARROW_OUTPUT_ROOT "$OUTPUT_ROOT"
case "$RUN_ROOT" in
  "$OUTPUT_ROOT"|"$OUTPUT_ROOT"/*) die 'OUTPUT_ROOT cannot contain RUN_ROOT' ;;
esac
[[ ! -e "$RUN_ROOT" ]] || die "RUN_ROOT already exists: $RUN_ROOT"
[[ ! -e "$OUTPUT_ROOT" ]] || die "OUTPUT_ROOT already exists: $OUTPUT_ROOT"
[[ ! -e "$ARCHIVE_ROOT" ]] || die "ARCHIVE_ROOT already exists: $ARCHIVE_ROOT"
mkdir -p "$RUN_ROOT" "$OUTPUT_ROOT"

ENTRYPOINT="$REPO_ROOT/vla_benchmarking/libero/finetuned_vlas/smolvla/workflows/run_lora_no_arrow_pair_eval.py"
LAUNCHER="$REPO_ROOT/vla_benchmarking/libero/finetuned_vlas/smolvla/target_arrow_only/legion/run_target_arrow_pair_eval.sbatch"
[[ -s "$ENTRYPOINT" ]] || die "evaluator entrypoint is missing: $ENTRYPOINT"
[[ -s "$LAUNCHER" ]] || die "launcher source is missing: $LAUNCHER"
ENTRYPOINT_SHA256="$(sha256sum -- "$ENTRYPOINT" | awk '{print tolower($1)}')"
LAUNCHER_SHA256="$(sha256sum -- "$LAUNCHER" | awk '{print tolower($1)}')"
ADAPTER_SHA256="$(sha256sum -- "$ADAPTER_DIR/adapter_model.safetensors" | awk '{print tolower($1)}')"
ADAPTER_CONFIG_SHA256="$(sha256sum -- "$ADAPTER_DIR/adapter_config.json" | awk '{print tolower($1)}')"
TRAIN_CONFIG_SHA256="$(sha256sum -- "$ADAPTER_DIR/train_config.json" | awk '{print tolower($1)}')"
TRAINING_MANIFEST_SHA256="$(sha256sum -- "$TRAINING_MANIFEST" | awk '{print tolower($1)}')"
PYTHON_SHA256="$(sha256sum -- "$PYTHON" | awk '{print tolower($1)}')"

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
  local workload_rc=$? archive_rc=0 status=PRESERVED_FAILURE
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT" || archive_rc=$?
  copy_tree "$RUN_ROOT" "$ARCHIVE_ROOT/run" || archive_rc=$?
  if [[ "$OUTPUT_ROOT" != "$RUN_ROOT" && "$OUTPUT_ROOT" != "$RUN_ROOT"/* ]]; then
    copy_tree "$OUTPUT_ROOT" "$ARCHIVE_ROOT/output" || archive_rc=$?
  fi
  log_root="${SLURM_LOG_ROOT:-$HOME/EmbodimentSemantic_runtime/operator/logs}"
  mkdir -p "$ARCHIVE_ROOT/slurm_logs" || archive_rc=$?
  for suffix in out err; do
    log_path="$log_root/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${suffix}"
    [[ -f "$log_path" ]] && cp -p -- "$log_path" "$ARCHIVE_ROOT/slurm_logs/" || true
  done
  if [[ "$workload_rc" -eq 0 && "$archive_rc" -eq 0 &&
        -s "$OUTPUT_ROOT/target_arrow_trained_arrow_pair_manifest.json" &&
        -s "$OUTPUT_ROOT/target_arrow_trained_arrow_pair_summary.csv" ]]; then
    status=VERIFIED
  elif [[ "$archive_rc" -ne 0 ]]; then
    status=ARCHIVE_FAILURE
  fi
  if [[ -d "$ARCHIVE_ROOT" ]]; then
    {
      printf 'schema=smolvla_target_arrow_pair_eval_job.v1\n'
      printf 'archive_status=%s\njob_exit_code=%s\njob_id=%s\n' "$status" "$workload_rc" "$SLURM_JOB_ID"
      printf 'label=%s\nscope=%s\ntask_ids=%s\nepisodes_per_task=%s\n' "$LABEL" "$EVAL_SCOPE" "$TASK_IDS" "$EPISODES"
      printf 'expected_commit=%s\nactual_commit=%s\n' "$EXPECTED_COMMIT" "$actual_commit"
      printf 'entrypoint=%s\nentrypoint_sha256=%s\nlauncher=%s\nlauncher_sha256=%s\n' "$ENTRYPOINT" "$ENTRYPOINT_SHA256" "$LAUNCHER" "$LAUNCHER_SHA256"
      printf 'adapter_dir=%s\nadapter_sha256=%s\nadapter_config_sha256=%s\ntrain_config_sha256=%s\n' "$ADAPTER_DIR" "$ADAPTER_SHA256" "$ADAPTER_CONFIG_SHA256" "$TRAIN_CONFIG_SHA256"
      printf 'training_manifest=%s\ntraining_manifest_sha256=%s\n' "$TRAINING_MANIFEST" "$TRAINING_MANIFEST_SHA256"
      printf 'python=%s\npython_sha256=%s\ndevice=%s\nrun_root=%s\noutput_root=%s\narchive_root=%s\n' "$PYTHON" "$PYTHON_SHA256" "$DEVICE" "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT"
    } > "$ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
    find "$ARCHIVE_ROOT" -type f ! -name inventory.sha256 ! -name tree_sha256 -print0 |
      sort -z | xargs -0 -r sha256sum > "$ARCHIVE_ROOT/inventory.sha256" || archive_rc=$?
    sha256sum "$ARCHIVE_ROOT/inventory.sha256" | awk '{print tolower($1)}' > "$ARCHIVE_ROOT/tree_sha256" || archive_rc=$?
  fi
  if [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]]; then workload_rc=$archive_rc; fi
  exit "$workload_rc"
}
trap archive_on_exit EXIT

cat > "$RUN_ROOT/job_context.env" <<EOF
schema=smolvla_target_arrow_pair_eval_job.v1
job_id=$SLURM_JOB_ID
job_name=${SLURM_JOB_NAME:-smolvla_target_arrow_eval}
label=$LABEL
scope=$EVAL_SCOPE
task_ids=$TASK_IDS
episodes_per_task=$EPISODES
seeds=$SEEDS
expected_commit=$EXPECTED_COMMIT
actual_commit=$actual_commit
repo_root=$REPO_ROOT
python=$PYTHON
python_sha256=$PYTHON_SHA256
device=$DEVICE
entrypoint=$ENTRYPOINT
entrypoint_sha256=$ENTRYPOINT_SHA256
launcher=$LAUNCHER
launcher_sha256=$LAUNCHER_SHA256
adapter_dir=$ADAPTER_DIR
adapter_sha256=$ADAPTER_SHA256
adapter_config_sha256=$ADAPTER_CONFIG_SHA256
train_config_sha256=$TRAIN_CONFIG_SHA256
training_manifest=$TRAINING_MANIFEST
training_manifest_sha256=$TRAINING_MANIFEST_SHA256
run_root=$RUN_ROOT
output_root=$OUTPUT_ROOT
archive_root=$ARCHIVE_ROOT
EOF

export REPO_ROOT VLA_ROOT="$REPO_ROOT/vla_benchmarking/libero"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="${TARGET_ARROW_LIBERO_CONFIG_PATH:-$HOME/EmbodimentSemantic_runtime/config}"
export LIBERO_CONFIG="$LIBERO_CONFIG_PATH/config.yaml"
[[ -f "$LIBERO_CONFIG" ]] || die "LIBERO configuration is missing: $LIBERO_CONFIG"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

"$PYTHON" - "$RUN_ROOT/runtime_versions.json" <<'PY'
import importlib.metadata as md
import json
import pathlib
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one visible CUDA device")
names = ("lerobot", "torch", "transformers", "peft")
versions = {name: md.version(name) for name in names}
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
  --profile target_arrow_treatment \
  --evaluation-scope "$EVAL_SCOPE" \
  --adapter-checkpoint "$ADAPTER_DIR" \
  --training-manifest "$TRAINING_MANIFEST" \
  --seeds "$SEEDS" --task-ids "$TASK_IDS" --episodes "$EPISODES" --batch-size 1 --device "$DEVICE" \
  --output-root "$OUTPUT_ROOT" --no-videos

[[ -s "$OUTPUT_ROOT/target_arrow_trained_arrow_pair_manifest.json" ]] || die 'evaluation manifest is missing'
[[ -s "$OUTPUT_ROOT/target_arrow_trained_arrow_pair_summary.csv" ]] || die 'evaluation summary is missing'
touch "$RUN_ROOT/COMPLETED"
