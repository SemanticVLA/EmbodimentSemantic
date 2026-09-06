#!/usr/bin/env bash
# Canonical Legion launcher for the retained target-arrow-only SmolVLA profile.
#
# This file is submitted directly with sbatch.  It intentionally invokes the
# repository helpers with absolute paths; submitting those helpers as separate
# Slurm jobs would spool them and break their BASH_SOURCE-relative paths.
#
# Required environment:
#   REPO_ROOT                                isolated clean checkout
#   TARGET_ARROW_TRAINING_EXPECTED_COMMIT   exact lowercase 40-char SHA
#   TARGET_ARROW_TRAINING_LABEL             safe run label
#   TARGET_ARROW_TRAINING_SCOPE             smoke, reuse_verified, or full
#
# Optional pinned Legion roots may be overridden explicitly for a site-specific
# release, but are never inferred from the submitted script's working directory.

#SBATCH --job-name=smolvla_target_arrow_train
#SBATCH --partition=gpu_a40
#SBATCH --exclude=compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Legion's gpu_a40 partition is capped below 24 hours.  Keep the full sealed
# training run schedulable; resume support preserves progress if it ever needs
# another allocation.
#SBATCH --time=0-23:59:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.err

set -Eeuo pipefail
umask 027
export PYTHONDONTWRITEBYTECODE=1

die() {
  printf 'target-arrow training: %s\n' "$*" >&2
  exit 2
}

[[ -n "${SLURM_JOB_ID:-}" && "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] ||
  die 'submit this launcher directly with sbatch; SLURM_JOB_ID must be numeric'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
EXPECTED_COMMIT="${TARGET_ARROW_TRAINING_EXPECTED_COMMIT:-${TARGET_ARROW_EXPECTED_COMMIT:-}}"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
  die 'TARGET_ARROW_TRAINING_EXPECTED_COMMIT must be a lowercase 40-character SHA'
LABEL="${TARGET_ARROW_TRAINING_LABEL:-${TARGET_ARROW_LABEL:-${2:-}}}"
[[ "$LABEL" =~ ^[A-Za-z0-9_.-]{1,96}$ ]] ||
  die 'TARGET_ARROW_TRAINING_LABEL is required and must be safe'
SCOPE="${TARGET_ARROW_TRAINING_SCOPE:-${TARGET_ARROW_SCOPE:-${1:-}}}"
case "$SCOPE" in
  smoke) STEPS=2; SAVE_FREQ=2; PIPELINE_SCOPE=smoke; REUSE_VERIFIED=false ;;
  reuse_verified) STEPS=2; SAVE_FREQ=2; PIPELINE_SCOPE=smoke; REUSE_VERIFIED=true ;;
  full) STEPS=29190; SAVE_FREQ=1946; PIPELINE_SCOPE=full; REUSE_VERIFIED=false ;;
  *) die 'TARGET_ARROW_TRAINING_SCOPE must be smoke or full' ;;
esac

command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'
command -v flock >/dev/null 2>&1 || die 'flock is required'

REPO_ROOT="$(realpath -e -- "$REPO_ROOT")" || die 'cannot resolve REPO_ROOT'
actual_root="$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null)" ||
  die 'REPO_ROOT is not a Git checkout'
actual_root="$(realpath -e -- "$actual_root")" || die 'cannot resolve Git root'
[[ "$actual_root" == "$REPO_ROOT" ]] || die 'REPO_ROOT is not the worktree root'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] ||
  die 'shared HOME checkout is forbidden; use an isolated immutable release'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)" ||
  die 'cannot read REPO_ROOT commit'
[[ "$actual_commit" == "$EXPECTED_COMMIT" ]] ||
  die "checkout SHA differs: $actual_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] ||
  die 'REPO_ROOT is dirty'

BASE_POLICY_REVISION="${TARGET_ARROW_BASE_POLICY_REVISION:-6721902bc4d61e50a3bfdb11dfb4cb626f05d102}"
LIBERO_COMMIT="${TARGET_ARROW_LIBERO_COMMIT:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
PYTHON="${TARGET_ARROW_PYTHON:-/home/hjaber/.conda/envs/embodiment-smolvla-py312/bin/python}"
RUNTIME_ROOT="${TARGET_ARROW_RUNTIME_ROOT:-$HOME/EmbodimentSemantic_runtime/EmbodimentSemantic/smolvla}"
LIBERO_DIR="${TARGET_ARROW_LIBERO_DIR:-$RUNTIME_ROOT/LIBERO}"
BASE_POLICY="${TARGET_ARROW_BASE_POLICY:-$RUNTIME_ROOT/base_models/smolvla_libero-$BASE_POLICY_REVISION}"
HDF5_ROOT="${TARGET_ARROW_HDF5_ROOT:-$RUNTIME_ROOT/libero_spatial_v5}"
DATA_ROOT="${TARGET_ARROW_DATA_ROOT:-$RUNTIME_ROOT/lora_datasets_target_arrow}"
DATA_CACHE="${TARGET_ARROW_DATA_CACHE:-$RUNTIME_ROOT/dataset_cache}"
LIBERO_CONFIG="${TARGET_ARROW_LIBERO_CONFIG:-$HOME/EmbodimentSemantic_runtime/config/config.yaml}"
SCRATCH_ROOT="${TARGET_ARROW_SCRATCH_ROOT:-${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/smolvla_target_arrow}"
RUN_ROOT="${TARGET_ARROW_RUN_ROOT:-$SCRATCH_ROOT/runs/${LABEL}_${SCOPE}_${SLURM_JOB_ID}}"
OUTPUT_ROOT="$RUN_ROOT/training"
ARCHIVE_ROOT="${TARGET_ARROW_ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/smolvla_target_arrow/${LABEL}_${SCOPE}_${SLURM_JOB_ID}}"

WORKFLOW_ROOT="$REPO_ROOT/vla_benchmarking/libero/finetuned_vlas/smolvla/workflows"
PREPARE_DATA_SCRIPT="$WORKFLOW_ROOT/prepare_lambda_data.sh"
PIPELINE_SCRIPT="$WORKFLOW_ROOT/run_smolvla_pipeline.sh"

canonical_external_root() {
  local name="$1" value="$2" canonical
  [[ "$value" = /* && "$value" != / ]] || die "$name must be an absolute non-root path: $value"
  [[ ! -L "$value" ]] || die "$name must not be a symlink: $value"
  canonical="$(realpath -m -- "$value")" || die "cannot canonicalize $name: $value"
  [[ "$canonical" != / ]] || die "$name resolves to the filesystem root"
  case "$canonical" in
    "$REPO_ROOT"|"$REPO_ROOT"/*) die "$name must be outside REPO_ROOT: $canonical" ;;
  esac
  printf '%s\n' "$canonical"
}

HDF5_ROOT="$(canonical_external_root TARGET_ARROW_HDF5_ROOT "$HDF5_ROOT")"
DATA_ROOT="$(canonical_external_root TARGET_ARROW_DATA_ROOT "$DATA_ROOT")"
DATA_CACHE="$(canonical_external_root TARGET_ARROW_DATA_CACHE "$DATA_CACHE")"

assert_disjoint_root() {
  local left_name="$1" left="$2" right_name="$3" right="$4"
  case "$left" in
    "$right"|"$right"/*) die "$left_name overlaps $right_name: $left / $right" ;;
  esac
  case "$right" in
    "$left"|"$left"/*) die "$right_name overlaps $left_name: $right / $left" ;;
  esac
}

assert_disjoint_root TARGET_ARROW_HDF5_ROOT "$HDF5_ROOT" TARGET_ARROW_DATA_ROOT "$DATA_ROOT"
assert_disjoint_root TARGET_ARROW_HDF5_ROOT "$HDF5_ROOT" TARGET_ARROW_DATA_CACHE "$DATA_CACHE"
assert_disjoint_root TARGET_ARROW_DATA_ROOT "$DATA_ROOT" TARGET_ARROW_DATA_CACHE "$DATA_CACHE"
for path in "$PYTHON" "$LIBERO_DIR" "$BASE_POLICY" "$LIBERO_CONFIG" "$PREPARE_DATA_SCRIPT" "$PIPELINE_SCRIPT"; do
  [[ "$path" = /* && "$path" != / ]] || die "path must be an absolute non-root path: $path"
done
[[ -x "$PYTHON" ]] || die "pinned Legion Python is missing: $PYTHON"
[[ -d "$LIBERO_DIR/.git" ]] || die "pinned LIBERO checkout is missing: $LIBERO_DIR"
[[ "$(git -C "$LIBERO_DIR" rev-parse HEAD 2>/dev/null || true)" == "$LIBERO_COMMIT" ]] ||
  die "LIBERO checkout is not pinned to $LIBERO_COMMIT"
[[ -z "$(git -C "$LIBERO_DIR" status --porcelain --untracked-files=no 2>/dev/null || true)" ]] ||
  die 'LIBERO checkout has tracked/staged changes'
[[ -f "$BASE_POLICY/config.json" && -f "$BASE_POLICY/base_snapshot_manifest.json" ]] ||
  die "pinned SmolVLA base snapshot is missing or incomplete: $BASE_POLICY"
[[ -f "$LIBERO_CONFIG" ]] || die "LIBERO config is missing: $LIBERO_CONFIG"
[[ -f "$PREPARE_DATA_SCRIPT" && -f "$PIPELINE_SCRIPT" ]] || die 'required workflow scripts are missing'

for path in "$REPO_ROOT" "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT"; do
  [[ "$path" = /* && "$path" != / ]] || die "path must be absolute and non-root: $path"
done
[[ ! -L "$RUN_ROOT" ]] || die "RUN_ROOT must not be a symlink: $RUN_ROOT"
[[ ! -L "$ARCHIVE_ROOT" ]] || die "ARCHIVE_ROOT must not be a symlink: $ARCHIVE_ROOT"
RUN_ROOT="$(realpath -m -- "$RUN_ROOT")" || die 'cannot canonicalize RUN_ROOT'
OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")" || die 'cannot canonicalize OUTPUT_ROOT'
ARCHIVE_ROOT="$(realpath -m -- "$ARCHIVE_ROOT")" || die 'cannot canonicalize ARCHIVE_ROOT'
case "$RUN_ROOT" in "$REPO_ROOT"|"$REPO_ROOT"/*) die 'run outputs must be outside REPO_ROOT' ;; esac
case "$ARCHIVE_ROOT" in "$REPO_ROOT"|"$REPO_ROOT"/*) die 'archive must be outside REPO_ROOT' ;; esac
case "$ARCHIVE_ROOT" in "$RUN_ROOT"|"$RUN_ROOT"/*) die 'archive cannot be inside RUN_ROOT' ;; esac
case "$RUN_ROOT" in "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'RUN_ROOT cannot be inside archive' ;; esac
for data_name_path in \
  "TARGET_ARROW_HDF5_ROOT:$HDF5_ROOT" \
  "TARGET_ARROW_DATA_ROOT:$DATA_ROOT" \
  "TARGET_ARROW_DATA_CACHE:$DATA_CACHE"; do
  data_name="${data_name_path%%:*}"
  data_path="${data_name_path#*:}"
  assert_disjoint_root "$data_name" "$data_path" RUN_ROOT "$RUN_ROOT"
  assert_disjoint_root "$data_name" "$data_path" ARCHIVE_ROOT "$ARCHIVE_ROOT"
done
[[ ! -e "$RUN_ROOT" ]] || die "RUN_ROOT already exists: $RUN_ROOT"
[[ ! -e "$ARCHIVE_ROOT" ]] || die "ARCHIVE_ROOT already exists: $ARCHIVE_ROOT"

TMP_ROOT="${TARGET_ARROW_TMP_ROOT:-/tmp/ta-${SLURM_JOB_ID}}"
[[ "$TMP_ROOT" == "/tmp/ta-${SLURM_JOB_ID}" ]] ||
  die 'TARGET_ARROW_TMP_ROOT must be the exact job-unique /tmp/ta-$SLURM_JOB_ID path'
[[ "$TMP_ROOT" =~ ^/tmp/ta-[0-9]+$ ]] || die "unsafe TMP_ROOT: $TMP_ROOT"
(( ${#TMP_ROOT} <= 64 )) || die "TMP_ROOT is too long for AF_UNIX paths: $TMP_ROOT"
[[ ! -L "$TMP_ROOT" ]] || die "TMP_ROOT must not be a symlink: $TMP_ROOT"
[[ ! -e "$TMP_ROOT" ]] || die "TMP_ROOT already exists: $TMP_ROOT"

mkdir -p "$RUN_ROOT" "$HDF5_ROOT" "$DATA_ROOT" "$DATA_CACHE"
export REPO_ROOT VLA_ROOT="$REPO_ROOT/vla_benchmarking/libero"
export PYTHON LIBERO_DIR LIBERO_CONFIG LIBERO_DATA_DIR="$HDF5_ROOT" DATA_ROOT
export BASE_POLICY BASE_POLICY_REVISION LIBERO_COMMIT
export LIBERO_DATASET_CACHE="$DATA_CACHE"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$TMP_ROOT"
export TMPDIR="$TMP_ROOT" XDG_CACHE_HOME="$RUN_ROOT/xdg"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

cat > "$RUN_ROOT/job_context.env" <<EOF
schema=smolvla_target_arrow_training_job.v1
job_id=$SLURM_JOB_ID
job_name=${SLURM_JOB_NAME:-smolvla_target_arrow_train}
label=$LABEL
scope=$SCOPE
pipeline_scope=$PIPELINE_SCOPE
reuse_verified=$REUSE_VERIFIED
steps=$STEPS
save_freq=$SAVE_FREQ
expected_commit=$EXPECTED_COMMIT
actual_commit=$actual_commit
repo_root=$REPO_ROOT
python=$PYTHON
base_policy=$BASE_POLICY
base_policy_revision=$BASE_POLICY_REVISION
libero_dir=$LIBERO_DIR
libero_commit=$LIBERO_COMMIT
libero_config=$LIBERO_CONFIG
hdf5_root=$HDF5_ROOT
dataset_root=$DATA_ROOT
run_root=$RUN_ROOT
output_root=$OUTPUT_ROOT
archive_root=$ARCHIVE_ROOT
tmp_root=$TMP_ROOT
profile=target_arrow_treatment
EOF

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
  local workload_rc=$? archive_rc=0
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT" || archive_rc=$?
  copy_tree "$RUN_ROOT" "$ARCHIVE_ROOT/run" || archive_rc=$?
  log_root="${SLURM_LOG_ROOT:-$HOME/EmbodimentSemantic_runtime/operator/logs}"
  mkdir -p "$ARCHIVE_ROOT/slurm_logs" || archive_rc=$?
  for suffix in out err; do
    log_path="$log_root/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${suffix}"
    [[ -f "$log_path" ]] && cp -p -- "$log_path" "$ARCHIVE_ROOT/slurm_logs/" || true
  done
  if [[ -d "$ARCHIVE_ROOT" ]]; then
    {
      printf 'schema=smolvla_target_arrow_training_job.v1\narchive_status=%s\njob_exit_code=%s\n' \
        "$([[ "$workload_rc" -eq 0 && "$archive_rc" -eq 0 ]] && echo VERIFIED || echo PRESERVED_FAILURE)" "$workload_rc"
      printf 'job_id=%s\nlabel=%s\nscope=%s\nsteps=%s\nsave_freq=%s\n' "$SLURM_JOB_ID" "$LABEL" "$SCOPE" "$STEPS" "$SAVE_FREQ"
      printf 'expected_commit=%s\nactual_commit=%s\n' "$EXPECTED_COMMIT" "$actual_commit"
      printf 'run_root=%s\noutput_root=%s\narchive_root=%s\n' "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT"
    } > "$ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
    find "$ARCHIVE_ROOT" -type f ! -name inventory.sha256 ! -name tree_sha256 -print0 |
      sort -z | xargs -0 -r sha256sum > "$ARCHIVE_ROOT/inventory.sha256" || archive_rc=$?
    sha256sum "$ARCHIVE_ROOT/inventory.sha256" | awk '{print tolower($1)}' > "$ARCHIVE_ROOT/tree_sha256" || archive_rc=$?
  fi
  if [[ "$TMP_ROOT" == "/tmp/ta-${SLURM_JOB_ID}" && "$TMP_ROOT" =~ ^/tmp/ta-[0-9]+$ && -e "$TMP_ROOT" ]]; then
    rm -rf -- "$TMP_ROOT" || archive_rc=$?
  fi
  if [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]]; then workload_rc=$archive_rc; fi
  exit "$workload_rc"
}
trap archive_on_exit EXIT

printf '[%s] preparing verified target-arrow dataset\n' "$(date +'%F %T')"
DATA_LOCK="$DATA_CACHE/.sealed-target-arrow-${BASE_POLICY_REVISION}-target_arrow_treatment.lock"
[[ ! -L "$DATA_LOCK" ]] || die "dataset lock must not be a symlink: $DATA_LOCK"
exec 8>>"$DATA_LOCK"
flock -x 8
printf 'dataset_lock=%s\n' "$DATA_LOCK" >> "$RUN_ROOT/job_context.env"
if [[ "$REUSE_VERIFIED" == true ]]; then
  CONVERTER_SCRIPT="$WORKFLOW_ROOT/hdf5_to_lerobot_dataset.py"
  [[ -f "$CONVERTER_SCRIPT" ]] || die "dataset converter is missing: $CONVERTER_SCRIPT"
  "$PYTHON" "$CONVERTER_SCRIPT" --mode verify-target-arrow \
    --data-dir "$HDF5_ROOT" --output-root "$DATA_ROOT"
  "$PYTHON" "$CONVERTER_SCRIPT" --mode preflight-target-arrow \
    --data-dir "$HDF5_ROOT" --output-root "$DATA_ROOT"
else
  bash "$PREPARE_DATA_SCRIPT" target_arrow_treatment
fi
flock -u 8
exec 8>&-

printf '[%s] launching sealed target-arrow training (%s, %s steps)\n' "$(date +'%F %T')" "$SCOPE" "$STEPS"
bash "$PIPELINE_SCRIPT" "$PIPELINE_SCOPE" --profile target-arrow --run-dir "$OUTPUT_ROOT" --python "$PYTHON"

[[ -f "$OUTPUT_ROOT/training_manifest.json" ]] || die 'training manifest is missing after pipeline completion'
touch "$RUN_ROOT/COMPLETED"
