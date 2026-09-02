#!/usr/bin/env bash
# Run the complete 10-task x 10-episode arrow-only LIBERO matrix on a compute node.
#
# Required submission environment:
#   REPO_ROOT                   isolated, clean checkout of this repository
#   ARROW_MATRIX_EXPECTED_COMMIT exact 40-character commit at REPO_ROOT
#
# Optional submission environment:
#   RUN_ROOT       scratch run directory; defaults below when SCRATCH_FLASH exists
#   OUTPUT_ROOT    matrix output directory; defaults to RUN_ROOT/results
#   ARCHIVE_ROOT   persistent archive directory; defaults under HOME
#   LIBERO_CONFIG_PATH directory containing config.yaml; defaults to the validated
#                    Legion runtime config directory
#   MATRIX_TASK_IDS comma-separated non-negative task IDs; defaults to 0..9
#   MATRIX_EPISODES_PER_TASK positive episode count; defaults to 10
#   MATRIX_SEED_BASE non-negative seed base; defaults to 1000
#   MATRIX_RESOLUTION positive square capture resolution; defaults to 256
#   ARROW_CONTROLLER_VARIANT legacy controller label; defaults to rgbd_arrow_v1
#   CONTROLLER_CONFIG or ARROW_CONTROLLER_CONFIG external JSON controller config;
#                    mutually exclusive with a non-default controller variant
#
# Submit this file with sbatch from the Legion login node. Do not execute it
# directly on the login node or pass an active shared checkout as REPO_ROOT.

#SBATCH --job-name=arrow_pick_place_matrix
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

die() {
  printf 'arrow matrix job: %s\n' "$*" >&2
  exit 2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit with sbatch; direct execution is refused'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT must name an isolated exact-commit checkout'
[[ -n "${ARROW_MATRIX_EXPECTED_COMMIT:-}" ]] || die 'ARROW_MATRIX_EXPECTED_COMMIT is required'

REPO_ROOT="$(cd "$REPO_ROOT" 2>/dev/null && pwd -P)" || die "cannot resolve REPO_ROOT"
[[ -d "$REPO_ROOT" ]] || die "repository does not exist: $REPO_ROOT"
EXPECTED_COMMIT="$ARROW_MATRIX_EXPECTED_COMMIT"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'ARROW_MATRIX_EXPECTED_COMMIT must be a lowercase 40-character SHA'

actual_root="$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null)" || die 'REPO_ROOT is not a Git checkout'
actual_root="$(cd "$actual_root" && pwd -P)"
[[ "$actual_root" == "$REPO_ROOT" ]] || die 'REPO_ROOT does not resolve to its Git worktree root'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] || die 'active shared checkout is not an isolated run checkout'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)" || die 'cannot read checkout commit'
[[ "$actual_commit" == "$EXPECTED_COMMIT" ]] || die "checkout commit $actual_commit differs from expected $EXPECTED_COMMIT"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'REPO_ROOT is dirty'

PYTHON="${PYTHON:-$HOME/.conda/envs/embodiment-smolvla-py312/bin/python}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/results}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/EmbodimentSemantic_runtime/config}"
MATRIX_TASK_IDS="${MATRIX_TASK_IDS:-0,1,2,3,4,5,6,7,8,9}"
MATRIX_EPISODES_PER_TASK="${MATRIX_EPISODES_PER_TASK:-10}"
MATRIX_SEED_BASE="${MATRIX_SEED_BASE:-1000}"
MATRIX_RESOLUTION="${MATRIX_RESOLUTION:-256}"
MATRIX_CONTINUE_ON_MOTION_FAILURE="${MATRIX_CONTINUE_ON_MOTION_FAILURE:-0}"
CONTROLLER_VARIANT="${ARROW_CONTROLLER_VARIANT:-rgbd_arrow_v1}"
CONTROLLER_CONFIG_INPUT="${CONTROLLER_CONFIG:-${ARROW_CONTROLLER_CONFIG:-}}"
CONTROLLER_CONFIG_PATH=""
CONTROLLER_CONFIG_FILE_SHA256=""
CONTROLLER_CONFIG_HASH=""
CONTROLLER_CONFIG_CANONICAL_PATH=""

if [[ -n "${CONTROLLER_CONFIG:-}" && -n "${ARROW_CONTROLLER_CONFIG:-}" && "$CONTROLLER_CONFIG" != "$ARROW_CONTROLLER_CONFIG" ]]; then
  die 'CONTROLLER_CONFIG and ARROW_CONTROLLER_CONFIG disagree'
fi

[[ "$MATRIX_TASK_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die 'MATRIX_TASK_IDS must be comma-separated non-negative integers'
[[ "$MATRIX_EPISODES_PER_TASK" =~ ^[1-9][0-9]*$ ]] || die 'MATRIX_EPISODES_PER_TASK must be a positive integer'
[[ "$MATRIX_SEED_BASE" =~ ^[0-9]+$ ]] || die 'MATRIX_SEED_BASE must be a non-negative integer'
[[ "$MATRIX_RESOLUTION" =~ ^[1-9][0-9]*$ ]] || die 'MATRIX_RESOLUTION must be a positive integer'
[[ "$MATRIX_CONTINUE_ON_MOTION_FAILURE" =~ ^[01]$ ]] || die 'MATRIX_CONTINUE_ON_MOTION_FAILURE must be 0 or 1'
[[ "$CONTROLLER_VARIANT" =~ ^[A-Za-z0-9_.-]+$ ]] || die 'ARROW_CONTROLLER_VARIANT is invalid'
if [[ -n "$CONTROLLER_CONFIG_INPUT" && "$CONTROLLER_VARIANT" != rgbd_arrow_v1 ]]; then
  die 'CONTROLLER_CONFIG cannot be combined with a non-default ARROW_CONTROLLER_VARIANT'
fi

[[ "$RUN_ROOT" = /* && "$OUTPUT_ROOT" = /* && "$ARCHIVE_ROOT" = /* ]] || die 'RUN_ROOT, OUTPUT_ROOT, and ARCHIVE_ROOT must be absolute paths'
command -v realpath >/dev/null 2>&1 || die 'realpath is required for safe run/archive path validation'
if [[ -n "$CONTROLLER_CONFIG_INPUT" ]]; then
  [[ -f "$CONTROLLER_CONFIG_INPUT" ]] || die "controller config is not a regular file: $CONTROLLER_CONFIG_INPUT"
  CONTROLLER_CONFIG_PATH="$(realpath -e -- "$CONTROLLER_CONFIG_INPUT")" || die 'cannot canonicalize CONTROLLER_CONFIG'
  command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required for controller config provenance'
  CONTROLLER_CONFIG_FILE_SHA256="$(sha256sum -- "$CONTROLLER_CONFIG_PATH" | awk '{print $1}')" || die 'cannot hash CONTROLLER_CONFIG'
  [[ "$CONTROLLER_CONFIG_FILE_SHA256" =~ ^[0-9a-f]{64}$ ]] || die 'controller config SHA-256 is invalid'
fi
RUN_ROOT="$(realpath -m -- "$RUN_ROOT")" || die 'cannot canonicalize RUN_ROOT'
OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")" || die 'cannot canonicalize OUTPUT_ROOT'
ARCHIVE_ROOT="$(realpath -m -- "$ARCHIVE_ROOT")" || die 'cannot canonicalize ARCHIVE_ROOT'
[[ "$RUN_ROOT" != "$ARCHIVE_ROOT" ]] || die 'RUN_ROOT and ARCHIVE_ROOT must be different'
case "$ARCHIVE_ROOT" in
  "$RUN_ROOT"|"$RUN_ROOT"/*) die 'ARCHIVE_ROOT must not be inside RUN_ROOT' ;;
esac
case "$RUN_ROOT" in
  "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'RUN_ROOT must not be inside ARCHIVE_ROOT' ;;
esac
case "$OUTPUT_ROOT" in
  "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'OUTPUT_ROOT must not be inside ARCHIVE_ROOT' ;;
esac
case "$ARCHIVE_ROOT" in
  "$OUTPUT_ROOT"|"$OUTPUT_ROOT"/*) die 'ARCHIVE_ROOT must not be inside OUTPUT_ROOT' ;;
esac
[[ ! -e "$RUN_ROOT" ]] || die "RUN_ROOT already exists; refusing to overwrite: $RUN_ROOT"
[[ ! -e "$OUTPUT_ROOT" ]] || die "OUTPUT_ROOT already exists; refusing to overwrite: $OUTPUT_ROOT"
[[ ! -e "$ARCHIVE_ROOT" ]] || die "ARCHIVE_ROOT already exists; refusing to overwrite: $ARCHIVE_ROOT"
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
  local workload_rc=$? archive_rc=0 log_root archived_summary_path
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT"
  if [[ -e "$ARCHIVE_ROOT/run" ]]; then
    printf 'archive destination already exists: %s\n' "$ARCHIVE_ROOT/run" >&2
    archive_rc=1
  else
    copy_tree "$RUN_ROOT" "$ARCHIVE_ROOT/run" || archive_rc=$?
  fi
  case "$OUTPUT_ROOT" in
    "$RUN_ROOT") archived_summary_path="$ARCHIVE_ROOT/run/arrow_pick_place_matrix_summary.json" ;;
    "$RUN_ROOT"/*)
      archived_summary_path="$ARCHIVE_ROOT/run${OUTPUT_ROOT#"$RUN_ROOT"}/arrow_pick_place_matrix_summary.json"
      ;;
    *)
      copy_tree "$OUTPUT_ROOT" "$ARCHIVE_ROOT/output" || archive_rc=$?
      archived_summary_path="$ARCHIVE_ROOT/output/arrow_pick_place_matrix_summary.json"
      ;;
  esac
  log_root="${SLURM_LOG_ROOT:-$HOME/EmbodimentSemantic_runtime/operator/logs}"
  mkdir -p "$ARCHIVE_ROOT/slurm_logs" || archive_rc=$?
  for suffix in out err; do
    log_path="$log_root/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${suffix}"
    [[ -f "$log_path" ]] && cp -p -- "$log_path" "$ARCHIVE_ROOT/slurm_logs/" || true
  done
  {
    printf 'archive_status=%s\n' "$([[ "$archive_rc" -eq 0 ]] && printf VERIFIED || printf FAILED)"
    printf 'job_exit_code=%s\n' "$workload_rc"
    printf 'job_id=%s\n' "$SLURM_JOB_ID"
    printf 'expected_commit=%s\n' "$EXPECTED_COMMIT"
    printf 'controller_variant=%s\n' "$CONTROLLER_VARIANT"
    printf 'controller_config_path=%s\n' "$CONTROLLER_CONFIG_PATH"
    printf 'controller_config_file_sha256=%s\n' "$CONTROLLER_CONFIG_FILE_SHA256"
    printf 'controller_config_hash=%s\n' "$CONTROLLER_CONFIG_HASH"
    printf 'controller_config_canonical_path=%s\n' "$CONTROLLER_CONFIG_CANONICAL_PATH"
    printf 'run_root=%s\n' "$RUN_ROOT"
    printf 'output_root=%s\n' "$OUTPUT_ROOT"
    printf 'archive_root=%s\n' "$ARCHIVE_ROOT"
    if [[ -f "$OUTPUT_ROOT/arrow_pick_place_matrix_summary.json" ]]; then
      printf 'summary_status=present\n'
      printf 'summary_path=%s\n' "$archived_summary_path"
    else
      printf 'summary_status=missing\n'
    fi
  } > "$ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
  if [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]]; then
    workload_rc=$archive_rc
  fi
  exit "$workload_rc"
}
trap archive_on_exit EXIT

cat > "$RUN_ROOT/job_context.env" <<EOF
job_id=$SLURM_JOB_ID
job_name=$SLURM_JOB_NAME
expected_commit=$EXPECTED_COMMIT
actual_commit=$actual_commit
repo_root=$REPO_ROOT
run_root=$RUN_ROOT
output_root=$OUTPUT_ROOT
archive_root=$ARCHIVE_ROOT
python=$PYTHON
libero_config_path=$LIBERO_CONFIG_PATH
matrix_tasks=$MATRIX_TASK_IDS
matrix_episodes_per_task=$MATRIX_EPISODES_PER_TASK
matrix_seed_base=$MATRIX_SEED_BASE
matrix_resolution=$MATRIX_RESOLUTION
controller_variant=$CONTROLLER_VARIANT
controller_config_path=$CONTROLLER_CONFIG_PATH
controller_config_file_sha256=$CONTROLLER_CONFIG_FILE_SHA256
continue_on_motion_failure=$MATRIX_CONTINUE_ON_MOTION_FAILURE
motion_mode=execute_motion
arrow_input_policy=hardcoded_subject_goal_bbox_renderer
language_translation=false
EOF

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"

[[ -x "$PYTHON" ]] || die "Python executable is missing: $PYTHON"
[[ -d "$LIBERO_CONFIG_PATH" && -f "$LIBERO_CONFIG_PATH/config.yaml" ]] || die "LIBERO_CONFIG_PATH must contain config.yaml: $LIBERO_CONFIG_PATH"
export PATH="$(dirname "$PYTHON"):$PATH"
# Keep imports tied to the exact checkout; ambient PYTHONPATH can otherwise
# silently resolve a different runner/configuration.
export PYTHONPATH="$REPO_ROOT/vla_benchmarking"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH
export LIBERO_CONFIG="$LIBERO_CONFIG_PATH/config.yaml"
export ARROW_CONTROLLER_VARIANT="$CONTROLLER_VARIANT"
if [[ -n "$CONTROLLER_CONFIG_PATH" ]]; then
  CONTROLLER_CONFIG_CANONICAL_PATH="$RUN_ROOT/controller_config_canonical.json"
  CONTROLLER_CONFIG_HASH="$($PYTHON - "$CONTROLLER_CONFIG_PATH" "$CONTROLLER_CONFIG_CANONICAL_PATH" <<'PY'
import json
import sys
from pathlib import Path

from controller_configs import load_controller_config

config_path, canonical_path = sys.argv[1:]
expanded = load_controller_config(config_path)
Path(canonical_path).write_text(
    json.dumps(expanded, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(expanded["config_hash"])
PY
  )" || die 'controller config failed semantic validation'
  [[ "$CONTROLLER_CONFIG_HASH" =~ ^[0-9a-f]{64}$ ]] || die 'controller config semantic hash is invalid'
  export ARROW_CONTROLLER_CONFIG="$CONTROLLER_CONFIG_PATH"
  {
    printf 'controller_config_hash=%s\n' "$CONTROLLER_CONFIG_HASH"
    printf 'controller_config_canonical_path=%s\n' "$CONTROLLER_CONFIG_CANONICAL_PATH"
  } >> "$RUN_ROOT/job_context.env"
fi
export TMPDIR="$RUN_ROOT/tmp"
export HF_HOME="$RUN_ROOT/cache/hf"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export TORCH_HOME="$RUN_ROOT/cache/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"

command -v nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi is unavailable inside the compute allocation'
nvidia-smi -L
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA smoke OK", torch.cuda.get_device_name(0))'
"$PYTHON" -c 'import libero, mujoco, robosuite; print("LIBERO/MuJoCo/robosuite imports OK")'

matrix_args=( \
  --task-ids "$MATRIX_TASK_IDS" \
  --episodes-per-task "$MATRIX_EPISODES_PER_TASK" \
  --seed-base "$MATRIX_SEED_BASE" \
  --resolution "$MATRIX_RESOLUTION" \
  --output-root "$OUTPUT_ROOT" \
  --controller-variant "$CONTROLLER_VARIANT" \
  --execute-motion \
  --allow-unvalidated-profile
)
if [[ "$MATRIX_CONTINUE_ON_MOTION_FAILURE" == 1 ]]; then
  matrix_args+=(--continue-on-motion-failure)
fi
if [[ -n "$CONTROLLER_CONFIG_PATH" ]]; then
  matrix_args+=(--controller-config "$CONTROLLER_CONFIG_PATH")
fi
"$PYTHON" "$REPO_ROOT/vla_benchmarking/run_arrow_pick_place_matrix.py" "${matrix_args[@]}"
