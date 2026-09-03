#!/usr/bin/env bash
# Project-local MolmoPoint + SAM3.1 canary launcher.
# Submit through the documented Legion wrapper from an isolated exact-commit
# checkout. This experimental launcher never uses the Omnis tree.
#
# Required: REPO_ROOT, CANARY_EXPECTED_COMMIT, CANARY_LABEL,
# CANARY_VARIANT, SAM3_CHECKPOINT_SHA256 (full SHA beginning with 0567de).
# Optional: CANARY_PHASE=prefix|full60, CANARY_PYTHON, SAM3_SOURCE_ROOT,
# SAM3_CHECKPOINT, CANARY_SETUP_SOURCE=1, CANARY_ENTRYPOINT.

#SBATCH --job-name=molmo_sam3_canary
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
die() { printf 'molmo/sam3 canary: %s\n' "$*" >&2; exit 2; }

[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit with sbatch; direct execution is refused'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
[[ -n "${CANARY_EXPECTED_COMMIT:-}" ]] || die 'CANARY_EXPECTED_COMMIT is required'
[[ -n "${CANARY_LABEL:-${1:-}}" ]] || die 'CANARY_LABEL is required'
[[ -n "${CANARY_VARIANT:-${2:-}}" ]] || die 'CANARY_VARIANT is required'
CANARY_LABEL="${CANARY_LABEL:-$1}"
CANARY_VARIANT="${CANARY_VARIANT:-$2}"
CANARY_PHASE="${CANARY_PHASE:-${3:-prefix}}"
EXPECTED_COMMIT="$CANARY_EXPECTED_COMMIT"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'CANARY_EXPECTED_COMMIT must be a lowercase 40-character SHA'
[[ "$CANARY_LABEL" =~ ^[A-Za-z0-9_.-]{1,96}$ ]] || die 'CANARY_LABEL contains unsafe characters'
case "$CANARY_VARIANT" in
  sam3_geometry_agentview|molmo_local_agentview|molmo_dense_agentview|molmo_dense_wrist) ;;
  *) die "unsupported CANARY_VARIANT: $CANARY_VARIANT" ;;
esac
case "$CANARY_PHASE" in prefix|full60) ;; *) die 'CANARY_PHASE must be prefix or full60' ;; esac

REPO_ROOT="$(cd "$REPO_ROOT" 2>/dev/null && pwd -P)" || die 'cannot resolve REPO_ROOT'
actual_root="$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null)" || die 'REPO_ROOT is not a Git checkout'
actual_root="$(cd "$actual_root" && pwd -P)"
[[ "$actual_root" == "$REPO_ROOT" ]] || die 'REPO_ROOT is not the worktree root'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] || die 'shared HOME checkout is forbidden; use an isolated release'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)" || die 'cannot read REPO_ROOT commit'
[[ "$actual_commit" == "$EXPECTED_COMMIT" ]] || die "checkout SHA differs: $actual_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'REPO_ROOT is dirty'

# Slurm may execute a spooled copy of this script, so BASH_SOURCE is not a
# reliable sibling path. Resolve the manifest only from the validated release.
MANIFEST_PATH="$REPO_ROOT/vla_benchmarking/legion/molmo_sam3_canary_pins.env"
[[ -f "$MANIFEST_PATH" ]] || die "pin manifest is missing: $MANIFEST_PATH"
# shellcheck disable=SC1091
source "$MANIFEST_PATH"

readonly SOURCE_COMMIT="$SAM3_SOURCE_COMMIT"
readonly CHECKPOINT_PREFIX="$SAM3_CHECKPOINT_SHA256_PREFIX"
readonly MODEL_REVISION="$MOLMOPOINT_MODEL_REVISION"
readonly PROMPT_ID="$MOLMOPOINT_PROMPT_ID"
[[ "$SOURCE_COMMIT" == 96914d2425f90a64f45ca977c2b5165418099543 ]] || die 'unexpected SAM3 source pin'
[[ "$MODEL_REVISION" == 188130f961c8e0888a34e11121a1423c461a01ba ]] || die 'unexpected MolmoPoint revision pin'
case "$PROMPT_ID" in rim_contact|rim_downward_approach|rim_clearance) ;; *) die 'unexpected MolmoPoint prompt pin' ;; esac
EXPECTED_CHECKPOINT_SHA="${SAM3_CHECKPOINT_SHA256:-}"
[[ "$EXPECTED_CHECKPOINT_SHA" =~ ^[0-9a-fA-F]{64}$ ]] || die 'SAM3_CHECKPOINT_SHA256 must be a full 64-character SHA-256'
[[ "${EXPECTED_CHECKPOINT_SHA,,}" == "$CHECKPOINT_PREFIX"* ]] || die 'SAM3_CHECKPOINT_SHA256 does not match 0567de prefix'

PYTHON="${CANARY_PYTHON:-$HOME/.conda/envs/embodiment-smolvla-py312/bin/python}"
CANARY_TASK_IDS="${CANARY_TASK_IDS:-4,6,9}"
CANARY_SEED_BASE="${CANARY_SEED_BASE:-1000}"
if [[ -z "${CANARY_EPISODES_PER_TASK:-}" ]]; then
  CANARY_EPISODES_PER_TASK=$([[ "$CANARY_PHASE" == prefix ]] && printf 2 || printf 10)
fi
CANARY_SUITE_MODES="${CANARY_SUITE_MODES:-vanilla,sealed_randomized}"
CANARY_ENTRYPOINT="${CANARY_ENTRYPOINT:-$REPO_ROOT/vla_benchmarking/run_molmo_sam3_canary.py}"
[[ -x "$PYTHON" ]] || die "Python executable is missing: $PYTHON"
[[ -f "$CANARY_ENTRYPOINT" ]] || die "canary entrypoint is missing: $CANARY_ENTRYPOINT"
[[ "$CANARY_TASK_IDS" == 4,6,9 ]] || die 'launcher only supports planned task set 4,6,9'
[[ "$CANARY_SEED_BASE" == 1000 ]] || die 'launcher only supports seed base 1000'
[[ "$CANARY_SUITE_MODES" == vanilla,sealed_randomized ]] || die 'suite modes must be vanilla,sealed_randomized'
if [[ "$CANARY_PHASE" == prefix ]]; then
  [[ "$CANARY_EPISODES_PER_TASK" == 2 ]] || die 'prefix phase requires 2 episodes per task'
else
  [[ "$CANARY_EPISODES_PER_TASK" == 10 ]] || die 'full60 phase requires 10 episodes per task'
fi
[[ "$CANARY_EPISODES_PER_TASK" =~ ^[0-9]+$ && "$CANARY_EPISODES_PER_TASK" -gt 0 ]] || die 'invalid episode count'

PROJECT_RUNTIME_ROOT="${PROJECT_RUNTIME_ROOT:-$HOME/EmbodimentSemantic_runtime/EmbodimentSemantic/molmo_sam3}"
SCRATCH_ROOT="${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/molmo_sam3"
RUN_ROOT="${RUN_ROOT:-$SCRATCH_ROOT/runs/${CANARY_LABEL}_${SLURM_JOB_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/results}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/molmo_sam3/${CANARY_LABEL}_${SLURM_JOB_ID}}"
SAM3_SOURCE_ROOT="${SAM3_SOURCE_ROOT:-$PROJECT_RUNTIME_ROOT/source/sam3-$SOURCE_COMMIT}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-$PROJECT_RUNTIME_ROOT/models/sam3.1.pt}"
HF_HOME_ROOT="${HF_HOME_ROOT:-$PROJECT_RUNTIME_ROOT/models/huggingface}"
for path in "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT"; do [[ "$path" = /* ]] || die "path must be absolute: $path"; done
command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'
RUN_ROOT="$(realpath -m -- "$RUN_ROOT")" || die 'cannot canonicalize RUN_ROOT'
OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")" || die 'cannot canonicalize OUTPUT_ROOT'
ARCHIVE_ROOT="$(realpath -m -- "$ARCHIVE_ROOT")" || die 'cannot canonicalize ARCHIVE_ROOT'
[[ "$RUN_ROOT" != "$ARCHIVE_ROOT" ]] || die 'RUN_ROOT and ARCHIVE_ROOT must differ'
case "$ARCHIVE_ROOT" in "$RUN_ROOT"|"$RUN_ROOT"/*) die 'ARCHIVE_ROOT cannot be inside RUN_ROOT' ;; esac
case "$RUN_ROOT" in "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'RUN_ROOT cannot be inside ARCHIVE_ROOT' ;; esac
[[ ! -e "$RUN_ROOT" ]] || die "RUN_ROOT already exists: $RUN_ROOT"
[[ ! -e "$ARCHIVE_ROOT" ]] || die "ARCHIVE_ROOT already exists: $ARCHIVE_ROOT"
mkdir -p "$RUN_ROOT" "$OUTPUT_ROOT"

copy_tree() {
  local source="$1" destination="$2"
  [[ -e "$source" ]] || return 0
  mkdir -p "$destination"
  if command -v rsync >/dev/null 2>&1; then rsync -a -- "$source/" "$destination/"; else cp -a -- "$source/." "$destination/"; fi
}
archive_on_exit() {
  local workload_rc=$? archive_rc=0
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT"
  copy_tree "$RUN_ROOT" "$ARCHIVE_ROOT/run" || archive_rc=$?
  if [[ "$OUTPUT_ROOT" != "$RUN_ROOT" && "$OUTPUT_ROOT" != "$RUN_ROOT"/* ]]; then copy_tree "$OUTPUT_ROOT" "$ARCHIVE_ROOT/output" || archive_rc=$?; fi
  log_root="${SLURM_LOG_ROOT:-$HOME/EmbodimentSemantic_runtime/operator/logs}"
  mkdir -p "$ARCHIVE_ROOT/slurm_logs" || archive_rc=$?
  for suffix in out err; do
    log_path="$log_root/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${suffix}"
    [[ -f "$log_path" ]] && cp -p -- "$log_path" "$ARCHIVE_ROOT/slurm_logs/" || true
  done
  {
    printf 'archive_status=%s\n' "$([[ "$archive_rc" -eq 0 ]] && printf VERIFIED || printf FAILED)"
    printf 'job_exit_code=%s\njob_id=%s\n' "$workload_rc" "$SLURM_JOB_ID"
    printf 'expected_commit=%s\nactual_commit=%s\n' "$EXPECTED_COMMIT" "$actual_commit"
    printf 'label=%s\nvariant=%s\nphase=%s\n' "$CANARY_LABEL" "$CANARY_VARIANT" "$CANARY_PHASE"
    printf 'sam3_source_commit=%s\nsam3_checkpoint_sha256=%s\nmolmopoint_model=%s\nmolmopoint_revision=%s\n' "$SOURCE_COMMIT" "${ACTUAL_CHECKPOINT_SHA:-}" "$MOLMOPOINT_MODEL" "$MODEL_REVISION"
    printf 'run_root=%s\noutput_root=%s\narchive_root=%s\n' "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT"
  } > "$ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
  if [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]]; then workload_rc=$archive_rc; fi
  exit "$workload_rc"
}
trap archive_on_exit EXIT

if [[ "${SAM3_SOURCE_ROOT,,}" == *'/omnis'* || "${SAM3_CHECKPOINT,,}" == *'/omnis'* ]]; then die 'Omnis paths are forbidden'; fi
SAM3_SOURCE_ARCHIVE_SHA256=""
source_marker_valid=0
if [[ -f "$SAM3_SOURCE_ROOT/.canary_source_commit" ]]; then
  source_marker_value="$(tr -d '[:space:]' < "$SAM3_SOURCE_ROOT/.canary_source_commit")"
  if [[ "$source_marker_value" == "$SOURCE_COMMIT" ]]; then
    source_marker_valid=1
  fi
fi
if [[ ! -d "$SAM3_SOURCE_ROOT/.git" && "$source_marker_valid" != 1 ]]; then
  [[ "${CANARY_SETUP_SOURCE:-0}" == 1 ]] || die "project-local SAM3 source is unavailable: $SAM3_SOURCE_ROOT"
  command -v curl >/dev/null 2>&1 || die 'curl is required to acquire SAM3 source'
  command -v tar >/dev/null 2>&1 || die 'tar is required to acquire SAM3 source'
  [[ ! -e "$SAM3_SOURCE_ROOT" ]] || die "SAM3 source target already exists but is not a Git checkout: $SAM3_SOURCE_ROOT"
  mkdir -p "$(dirname "$SAM3_SOURCE_ROOT")"
  source_stage="$RUN_ROOT/tmp/sam3-source-stage"
  mkdir -p "$source_stage"
  archive_path="$source_stage/sam3-${SOURCE_COMMIT}.tar.gz"
  curl --fail --location --silent --show-error \
    "${SAM3_REPOSITORY%.git}/archive/${SOURCE_COMMIT}.tar.gz" --output "$archive_path" \
    || die 'pinned SAM3 source archive download failed'
  SAM3_SOURCE_ARCHIVE_SHA256="$(sha256sum -- "$archive_path" | awk '{print tolower($1)}')" || die 'cannot hash SAM3 source archive'
  tar -xzf "$archive_path" -C "$source_stage" || die 'pinned SAM3 source archive extraction failed'
  extracted_root="$(find "$source_stage" -mindepth 1 -maxdepth 1 -type d -name "sam3-*" -print -quit)"
  [[ -n "$extracted_root" && -d "$extracted_root/sam3" ]] || die 'pinned SAM3 archive has no sam3/ package'
  mv -- "$extracted_root" "$SAM3_SOURCE_ROOT" || die 'cannot install project-local SAM3 source'
  printf '%s\n' "$SOURCE_COMMIT" > "$SAM3_SOURCE_ROOT/.canary_source_commit"
fi
source_root_real="$(realpath -e -- "$SAM3_SOURCE_ROOT")" || die 'cannot resolve project-local SAM3 source'
[[ "${source_root_real,,}" != *'/omnis'* ]] || die 'resolved SAM3 source points into Omnis'
if [[ -f "$source_root_real/.canary_source_commit" ]]; then
  source_actual_commit="$(tr -d '[:space:]' < "$source_root_real/.canary_source_commit")"
else
  source_actual_commit="$(git -C "$source_root_real" rev-parse HEAD 2>/dev/null)" || die 'cannot read SAM3 source commit'
fi
[[ "$source_actual_commit" == "$SOURCE_COMMIT" ]] || die "SAM3 source SHA differs: $source_actual_commit"
if [[ -d "$source_root_real/.git" ]]; then
  [[ -z "$(git -C "$source_root_real" status --porcelain --untracked-files=all)" ]] || die 'project-local SAM3 source is dirty'
  [[ "$(git -C "$source_root_real" remote get-url origin 2>/dev/null)" == "$SAM3_REPOSITORY" ]] || die 'SAM3 source origin is unexpected'
fi

[[ -f "$SAM3_CHECKPOINT" ]] || die "SAM3 checkpoint is unavailable: $SAM3_CHECKPOINT"
[[ -s "$SAM3_CHECKPOINT" ]] || die 'SAM3 checkpoint is empty'
checkpoint_real="$(realpath -e -- "$SAM3_CHECKPOINT")" || die 'cannot resolve SAM3 checkpoint'
ACTUAL_CHECKPOINT_SHA="$(sha256sum -- "$checkpoint_real" | awk '{print tolower($1)}')" || die 'cannot hash SAM3 checkpoint'
[[ "$ACTUAL_CHECKPOINT_SHA" == "$EXPECTED_CHECKPOINT_SHA" ]] || die 'SAM3 checkpoint SHA-256 mismatch'
[[ "$ACTUAL_CHECKPOINT_SHA" == "$CHECKPOINT_PREFIX"* ]] || die 'SAM3 checkpoint SHA-256 prefix mismatch'
MANIFEST_SHA256="$(sha256sum -- "$MANIFEST_PATH" | awk '{print $1}')" || die 'cannot hash pin manifest'
CONFIG_PATH="${CANARY_CONFIG_PATH:-$REPO_ROOT/vla_benchmarking/controller_configs/v9d_rgbd_region_grasp_search.json}"
[[ -f "$CONFIG_PATH" ]] || die "canary config is unavailable: $CONFIG_PATH"
CONFIG_SHA256="$(sha256sum -- "$CONFIG_PATH" | awk '{print $1}')" || die 'cannot hash canary config'

mkdir -p "$RUN_ROOT/tmp" "$RUN_ROOT/cache" "$HF_HOME_ROOT"
cat > "$RUN_ROOT/job_context.env" <<EOF
job_id=$SLURM_JOB_ID
job_name=$SLURM_JOB_NAME
label=$CANARY_LABEL
variant=$CANARY_VARIANT
phase=$CANARY_PHASE
task_ids=$CANARY_TASK_IDS
episodes_per_task=$CANARY_EPISODES_PER_TASK
seed_base=$CANARY_SEED_BASE
suite_modes=$CANARY_SUITE_MODES
expected_commit=$EXPECTED_COMMIT
actual_commit=$actual_commit
repo_root=$REPO_ROOT
config_path=$CONFIG_PATH
config_sha256=$CONFIG_SHA256
manifest_path=$MANIFEST_PATH
manifest_sha256=$MANIFEST_SHA256
sam3_repository=$SAM3_REPOSITORY
sam3_source_root=$source_root_real
sam3_source_commit=$SOURCE_COMMIT
sam3_source_archive_sha256=$SAM3_SOURCE_ARCHIVE_SHA256
sam3_checkpoint=$checkpoint_real
sam3_checkpoint_sha256=$ACTUAL_CHECKPOINT_SHA
molmopoint_model=$MOLMOPOINT_MODEL
molmopoint_revision=$MODEL_REVISION
molmopoint_prompt_id=$PROMPT_ID
runtime_root=$PROJECT_RUNTIME_ROOT
run_root=$RUN_ROOT
output_root=$OUTPUT_ROOT
archive_root=$ARCHIVE_ROOT
EOF

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$source_root_real:$REPO_ROOT/vla_benchmarking"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/EmbodimentSemantic_runtime/config}"
export TMPDIR="$RUN_ROOT/tmp" XDG_CACHE_HOME="$RUN_ROOT/cache/xdg" TORCH_HOME="$RUN_ROOT/cache/torch" HF_HOME="$HF_HOME_ROOT"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME"
command -v nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi is unavailable in the allocation'
gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
[[ "$gpu_count" == 1 ]] || die "expected exactly one visible GPU, found $gpu_count"
"$PYTHON" - "$source_root_real" "$checkpoint_real" <<'PY'
import pathlib
import sys
source_root, checkpoint = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(source_root))
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one CUDA device, found {torch.cuda.device_count()}")
import sam3  # noqa: F401
import transformers  # noqa: F401
from PIL import Image  # noqa: F401
if checkpoint.stat().st_size <= 0:
    raise SystemExit("checkpoint is empty")
print(f"runtime_preflight=passed gpu={torch.cuda.get_device_name(0)}")
PY

runner_help="$("$PYTHON" "$CANARY_ENTRYPOINT" --help)" || die 'canary entrypoint help failed'
grep -Fq -- '--variant' <<<"$runner_help" || die 'canary entrypoint lacks --variant'
grep -Fq -- '--output-dir' <<<"$runner_help" || die 'canary entrypoint lacks --output-dir'

# The environment is the stable provenance contract. Optional flags are added
# only when the checked-out runner advertises them, allowing the operational
# wrapper to remain usable while the live adapter evolves.
export MOLMO_SAM3_CANARY_LABEL="$CANARY_LABEL"
export MOLMO_SAM3_CANARY_PHASE="$CANARY_PHASE"
export MOLMO_SAM3_CANARY_TASK_IDS="$CANARY_TASK_IDS"
export MOLMO_SAM3_CANARY_EPISODES_PER_TASK="$CANARY_EPISODES_PER_TASK"
export MOLMO_SAM3_CANARY_SEED_BASE="$CANARY_SEED_BASE"
export MOLMO_SAM3_CANARY_SUITE_MODES="$CANARY_SUITE_MODES"
export MOLMO_SAM3_CANARY_CONFIG="$CONFIG_PATH"
export MOLMO_SAM3_CANARY_SAM3_SOURCE="$source_root_real"
export MOLMO_SAM3_CANARY_SAM3_CHECKPOINT="$checkpoint_real"
export MOLMO_SAM3_CANARY_SAM3_SOURCE_COMMIT="$SOURCE_COMMIT"
export MOLMO_SAM3_CANARY_SAM3_CHECKPOINT_SHA256="$ACTUAL_CHECKPOINT_SHA"
export MOLMO_SAM3_CANARY_MOLMOPOINT_MODEL="$MOLMOPOINT_MODEL"
export MOLMO_SAM3_CANARY_MOLMOPOINT_REVISION="$MODEL_REVISION"
export MOLMO_SAM3_CANARY_MOLMOPOINT_PROMPT_ID="$PROMPT_ID"
export MOLMO_SAM3_CANARY_RUNTIME_ROOT="$PROJECT_RUNTIME_ROOT"

runner_args=(--variant "$CANARY_VARIANT" --output-dir "$OUTPUT_ROOT")
if grep -Fq -- '--label' <<<"$runner_help"; then runner_args+=(--label "$CANARY_LABEL"); fi
if grep -Fq -- '--phase' <<<"$runner_help"; then runner_args+=(--phase "$CANARY_PHASE"); fi
if grep -Fq -- '--task-ids' <<<"$runner_help"; then runner_args+=(--task-ids "$CANARY_TASK_IDS"); fi
if grep -Fq -- '--episodes-per-task' <<<"$runner_help"; then runner_args+=(--episodes-per-task "$CANARY_EPISODES_PER_TASK"); fi
if grep -Fq -- '--seed-base' <<<"$runner_help"; then runner_args+=(--seed-base "$CANARY_SEED_BASE"); fi
if grep -Fq -- '--suite-modes' <<<"$runner_help"; then runner_args+=(--suite-modes "$CANARY_SUITE_MODES"); fi
if grep -Fq -- '--config' <<<"$runner_help"; then runner_args+=(--config "$CONFIG_PATH"); fi
if grep -Fq -- '--sam3-source' <<<"$runner_help"; then runner_args+=(--sam3-source "$source_root_real"); fi
if grep -Fq -- '--sam3-checkpoint' <<<"$runner_help"; then runner_args+=(--sam3-checkpoint "$checkpoint_real"); fi
if grep -Fq -- '--sam3-source-commit' <<<"$runner_help"; then runner_args+=(--sam3-source-commit "$SOURCE_COMMIT"); fi
if grep -Fq -- '--sam3-checkpoint-sha256' <<<"$runner_help"; then runner_args+=(--sam3-checkpoint-sha256 "$ACTUAL_CHECKPOINT_SHA"); fi
if grep -Fq -- '--molmopoint-model' <<<"$runner_help"; then runner_args+=(--molmopoint-model "$MOLMOPOINT_MODEL"); fi
if grep -Fq -- '--molmopoint-revision' <<<"$runner_help"; then runner_args+=(--molmopoint-revision "$MODEL_REVISION"); fi
if grep -Fq -- '--molmopoint-prompt-id' <<<"$runner_help"; then runner_args+=(--molmopoint-prompt-id "$PROMPT_ID"); fi
if grep -Fq -- '--runtime-root' <<<"$runner_help"; then runner_args+=(--runtime-root "$PROJECT_RUNTIME_ROOT"); fi
if [[ "${CANARY_DRY_RUN:-0}" == 1 ]]; then
  grep -Fq -- '--dry-run' <<<"$runner_help" || die 'CANARY_DRY_RUN=1 but entrypoint lacks --dry-run'
  runner_args+=(--dry-run)
fi
set +e
"$PYTHON" "$CANARY_ENTRYPOINT" "${runner_args[@]}"
workload_rc=$?
set -e
exit "$workload_rc"
