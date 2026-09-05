#!/usr/bin/env bash
# Canonical Legion launcher for the retained no-arrow SmolVLA profile.
#
# Submit this file directly with sbatch.  The shared workflow scripts are
# invoked by absolute path so Slurm never spools them as independent jobs and
# breaks their BASH_SOURCE-relative imports.
#
# Required environment:
#   REPO_ROOT                              isolated clean checkout
#   NO_ARROW_TRAINING_EXPECTED_COMMIT     exact lowercase 40-character SHA
#   NO_ARROW_TRAINING_LABEL               safe run label
#   NO_ARROW_TRAINING_SCOPE               smoke or full
#
# Dataset mode defaults to prepare.  A smoke can explicitly reuse an existing
# fully verified pair with:
#   NO_ARROW_TRAINING_DATA_MODE=reuse_verified
#   NO_ARROW_TRAINING_VERIFIED_DATA_ROOT=/absolute/path/to/pair
# Reuse never trusts the old sentinel alone: the exact verify and preflight
# byte-bound checks run against the pinned source HDF5 tree before training.

#SBATCH --job-name=smolvla_no_arrow_train
#SBATCH --partition=gpu_a40
#SBATCH --exclude=compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Legion's gpu_a40 partition is capped below 24 hours.  The full run is kept
# schedulable; the pipeline's resume support preserves progress if needed.
#SBATCH --time=0-23:59:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.err

set -Eeuo pipefail
umask 027
export PYTHONDONTWRITEBYTECODE=1

die() {
  printf 'no-arrow training: %s\n' "$*" >&2
  exit 2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit this launcher directly with sbatch'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
EXPECTED_COMMIT="${NO_ARROW_TRAINING_EXPECTED_COMMIT:-${NO_ARROW_EXPECTED_COMMIT:-}}"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
  die 'NO_ARROW_TRAINING_EXPECTED_COMMIT must be a lowercase 40-character SHA'
LABEL="${NO_ARROW_TRAINING_LABEL:-${NO_ARROW_LABEL:-${2:-}}}"
[[ "$LABEL" =~ ^[A-Za-z0-9_.-]{1,96}$ ]] ||
  die 'NO_ARROW_TRAINING_LABEL is required and must be safe'
SCOPE="${NO_ARROW_TRAINING_SCOPE:-${NO_ARROW_SCOPE:-${1:-}}}"
case "$SCOPE" in
  smoke) STEPS=2; SAVE_FREQ=2 ;;
  full) STEPS=29190; SAVE_FREQ=1946 ;;
  *) die 'NO_ARROW_TRAINING_SCOPE must be smoke or full' ;;
esac

DATA_MODE="${NO_ARROW_TRAINING_DATA_MODE:-prepare}"
case "$DATA_MODE" in
  prepare|reuse_verified) ;;
  *) die 'NO_ARROW_TRAINING_DATA_MODE must be prepare or reuse_verified' ;;
esac
if [[ "$DATA_MODE" == reuse_verified && "$SCOPE" != smoke ]]; then
  die 'reuse_verified is permitted only for an explicit smoke run'
fi

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

BASE_POLICY_REVISION="${NO_ARROW_BASE_POLICY_REVISION:-6721902bc4d61e50a3bfdb11dfb4cb626f05d102}"
LIBERO_COMMIT="${NO_ARROW_LIBERO_COMMIT:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
PYTHON="${NO_ARROW_PYTHON:-/home/hjaber/.conda/envs/embodiment-smolvla-py312/bin/python}"
RUNTIME_ROOT="${NO_ARROW_RUNTIME_ROOT:-$HOME/EmbodimentSemantic_runtime/EmbodimentSemantic/smolvla}"
LIBERO_DIR="${NO_ARROW_LIBERO_DIR:-$RUNTIME_ROOT/LIBERO}"
BASE_POLICY="${NO_ARROW_BASE_POLICY:-$RUNTIME_ROOT/base_models/smolvla_libero-$BASE_POLICY_REVISION}"
HDF5_ROOT="${NO_ARROW_HDF5_ROOT:-$RUNTIME_ROOT/libero_spatial_v5}"
if [[ "$DATA_MODE" == reuse_verified ]]; then
  DATA_ROOT="${NO_ARROW_TRAINING_VERIFIED_DATA_ROOT:-}"
  [[ -n "$DATA_ROOT" ]] ||
    die 'NO_ARROW_TRAINING_VERIFIED_DATA_ROOT is required in reuse_verified mode'
else
  DATA_ROOT="${NO_ARROW_DATA_ROOT:-$RUNTIME_ROOT/lora_datasets_no_arrow}"
fi
DATA_CACHE="${NO_ARROW_DATA_CACHE:-$RUNTIME_ROOT/dataset_cache}"
LIBERO_CONFIG="${NO_ARROW_LIBERO_CONFIG:-$HOME/EmbodimentSemantic_runtime/config/config.yaml}"
SCRATCH_ROOT="${NO_ARROW_SCRATCH_ROOT:-${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/smolvla_no_arrow}"
RUN_ROOT="${NO_ARROW_RUN_ROOT:-$SCRATCH_ROOT/runs/${LABEL}_${SCOPE}_${SLURM_JOB_ID}}"
OUTPUT_ROOT="$RUN_ROOT/training"
ARCHIVE_ROOT="${NO_ARROW_ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/smolvla_no_arrow/${LABEL}_${SCOPE}_${SLURM_JOB_ID}}"

WORKFLOW_ROOT="$REPO_ROOT/vla_benchmarking/libero/finetuned_vlas/smolvla/workflows"
PREPARE_DATA_SCRIPT="$WORKFLOW_ROOT/prepare_lambda_data.sh"
PAIR_SCRIPT="$WORKFLOW_ROOT/hdf5_to_lerobot_dataset.py"
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

HDF5_ROOT="$(canonical_external_root NO_ARROW_HDF5_ROOT "$HDF5_ROOT")"
DATA_ROOT="$(canonical_external_root NO_ARROW_DATA_ROOT "$DATA_ROOT")"
DATA_CACHE="$(canonical_external_root NO_ARROW_DATA_CACHE "$DATA_CACHE")"

assert_disjoint_root() {
  local left_name="$1" left="$2" right_name="$3" right="$4"
  case "$left" in
    "$right"|"$right"/*) die "$left_name overlaps $right_name: $left / $right" ;;
  esac
  case "$right" in
    "$left"|"$left"/*) die "$right_name overlaps $left_name: $right / $left" ;;
  esac
}

assert_disjoint_root NO_ARROW_HDF5_ROOT "$HDF5_ROOT" NO_ARROW_DATA_ROOT "$DATA_ROOT"
assert_disjoint_root NO_ARROW_HDF5_ROOT "$HDF5_ROOT" NO_ARROW_DATA_CACHE "$DATA_CACHE"
assert_disjoint_root NO_ARROW_DATA_ROOT "$DATA_ROOT" NO_ARROW_DATA_CACHE "$DATA_CACHE"
for path in "$PYTHON" "$LIBERO_DIR" "$BASE_POLICY" "$LIBERO_CONFIG" "$PAIR_SCRIPT" "$PREPARE_DATA_SCRIPT" "$PIPELINE_SCRIPT"; do
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
[[ -f "$PREPARE_DATA_SCRIPT" && -f "$PAIR_SCRIPT" && -f "$PIPELINE_SCRIPT" ]] ||
  die 'required workflow scripts are missing'

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
  "NO_ARROW_HDF5_ROOT:$HDF5_ROOT" \
  "NO_ARROW_DATA_ROOT:$DATA_ROOT" \
  "NO_ARROW_DATA_CACHE:$DATA_CACHE"; do
  data_name="${data_name_path%%:*}"
  data_path="${data_name_path#*:}"
  assert_disjoint_root "$data_name" "$data_path" RUN_ROOT "$RUN_ROOT"
  assert_disjoint_root "$data_name" "$data_path" ARCHIVE_ROOT "$ARCHIVE_ROOT"
done
[[ ! -e "$RUN_ROOT" ]] || die "RUN_ROOT already exists: $RUN_ROOT"
[[ ! -e "$ARCHIVE_ROOT" ]] || die "ARCHIVE_ROOT already exists: $ARCHIVE_ROOT"

mkdir -p "$RUN_ROOT" "$HDF5_ROOT" "$DATA_CACHE"
if [[ "$DATA_MODE" == prepare ]]; then
  mkdir -p "$DATA_ROOT"
fi
export REPO_ROOT VLA_ROOT="$REPO_ROOT/vla_benchmarking/libero"
export PYTHON LIBERO_DIR LIBERO_CONFIG LIBERO_DATA_DIR="$HDF5_ROOT" DATA_ROOT
export BASE_POLICY BASE_POLICY_REVISION LIBERO_COMMIT
export LIBERO_DATASET_CACHE="$DATA_CACHE"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR="$RUN_ROOT/tmp" XDG_CACHE_HOME="$RUN_ROOT/xdg"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

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

cat > "$RUN_ROOT/job_context.env" <<EOF
schema=smolvla_no_arrow_training_job.v1
job_id=$SLURM_JOB_ID
job_name=${SLURM_JOB_NAME:-smolvla_no_arrow_train}
label=$LABEL
scope=$SCOPE
steps=$STEPS
save_freq=$SAVE_FREQ
data_mode=$DATA_MODE
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
profile=no_arrow_treatment
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
      printf 'schema=smolvla_no_arrow_training_job.v1\narchive_status=%s\njob_exit_code=%s\n' \
        "$([[ "$workload_rc" -eq 0 && "$archive_rc" -eq 0 ]] && echo VERIFIED || echo PRESERVED_FAILURE)" "$workload_rc"
      printf 'job_id=%s\nlabel=%s\nscope=%s\nsteps=%s\nsave_freq=%s\ndata_mode=%s\n' \
        "$SLURM_JOB_ID" "$LABEL" "$SCOPE" "$STEPS" "$SAVE_FREQ" "$DATA_MODE"
      printf 'expected_commit=%s\nactual_commit=%s\n' "$EXPECTED_COMMIT" "$actual_commit"
      printf 'run_root=%s\noutput_root=%s\narchive_root=%s\n' "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT"
    } > "$ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
    find "$ARCHIVE_ROOT" -type f ! -name inventory.sha256 ! -name tree_sha256 -print0 |
      sort -z | xargs -0 -r sha256sum > "$ARCHIVE_ROOT/inventory.sha256" || archive_rc=$?
    sha256sum "$ARCHIVE_ROOT/inventory.sha256" | awk '{print tolower($1)}' > "$ARCHIVE_ROOT/tree_sha256" || archive_rc=$?
  fi
  if [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]]; then workload_rc=$archive_rc; fi
  exit "$workload_rc"
}
trap archive_on_exit EXIT

DATA_LOCK="$DATA_CACHE/.sealed-no-arrow-${BASE_POLICY_REVISION}-no_arrow_treatment.lock"
[[ ! -L "$DATA_LOCK" ]] || die "dataset lock must not be a symlink: $DATA_LOCK"
exec 8>>"$DATA_LOCK"
flock -x 8
if [[ "$DATA_MODE" == prepare ]]; then
  printf '[%s] preparing verified no-arrow dataset\n' "$(date +'%F %T')"
  bash "$PREPARE_DATA_SCRIPT" no_arrow_treatment
else
  printf '[%s] reusing explicit verified no-arrow dataset root\n' "$(date +'%F %T')"
  [[ -d "$DATA_ROOT" ]] || die "verified dataset root is missing: $DATA_ROOT"
  [[ -s "$DATA_ROOT/sealed_lora_pair_manifest.json" ]] || die 'verified dataset manifest is missing'
  [[ -s "$DATA_ROOT/sealed_lora_pair_verified.json" ]] || die 'verified dataset sentinel is missing'
  "$PYTHON" "$PAIR_SCRIPT" --mode verify --data-dir "$HDF5_ROOT" --output-root "$DATA_ROOT"
  "$PYTHON" "$PAIR_SCRIPT" --mode preflight --data-dir "$HDF5_ROOT" --output-root "$DATA_ROOT"
fi
flock -u 8
exec 8>&-

printf '[%s] launching sealed no-arrow training (%s, %s steps)\n' "$(date +'%F %T')" "$SCOPE" "$STEPS"
bash "$PIPELINE_SCRIPT" "$SCOPE" --profile no-arrow --run-dir "$OUTPUT_ROOT" --python "$PYTHON"

[[ -f "$OUTPUT_ROOT/training_manifest.json" ]] || die 'training manifest is missing after pipeline completion'
touch "$RUN_ROOT/COMPLETED"
