#!/usr/bin/env bash
# Arrow OnCall exploratory matrix: ten LIBERO tasks, ten fixed episodes per task.
# One array element owns one task and invokes the sealed Python matrix runner;
# the runner is responsible for the episode schedule and annotated task video.
# This wrapper owns only cluster/runtime setup, provenance, and archival.

#SBATCH --job-name=arrow_oncall_matrix
#SBATCH --array=0-9%2
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%A_%a.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%A_%a.err

set -Eeuo pipefail
umask 027
task_rc=125

# The submit front door uses --export=ALL for the sealed ARROW_ONCALL_* input
# contract.  Fail closed against policy-affecting values inherited from the
# login shell or scheduler environment before constructing the host.
while IFS= read -r name; do
  case "$name" in
    ARROW_SUITE_*|ARROW_CONTROLLER_*) unset "$name" ;;
  esac
done < <(compgen -v)
# These legacy teacher-constructor knobs are outside the ARROW_SUITE_ prefix
# and must not override the sealed Arrow/SmolVLA inputs.
unset ARROW_MOLMOPOINT_MODEL ARROW_MOLMOPOINT_REVISION \
  ARROW_GRIPPER_CLOSED_THRESHOLD ARROW_GRIPPER_OPEN_THRESHOLD \
  SMOLVLA_BASE_POLICY

die() { printf 'arrow oncall matrix: %s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]] || die 'submit with sbatch as an array'
[[ "$SLURM_ARRAY_TASK_ID" -ge 0 && "$SLURM_ARRAY_TASK_ID" -le 9 ]] || die 'array task must be in 0..9'

for name in ARROW_ONCALL_REPO_ROOT ARROW_ONCALL_EXPECTED_COMMIT \
  ARROW_ONCALL_RUN_ROOT ARROW_ONCALL_ARCHIVE_ROOT ARROW_ONCALL_CONFIG \
  ARROW_ONCALL_CHECKPOINT ARROW_ONCALL_CONTROLLER ARROW_ONCALL_RUNTIME_PYTHON \
  ARROW_ONCALL_HF_CACHE; do
  [[ -n "${!name:-}" ]] || die "$name is required"
done
[[ "$ARROW_ONCALL_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'expected commit must be a full lowercase SHA'
for name in ARROW_ONCALL_REPO_ROOT ARROW_ONCALL_RUN_ROOT ARROW_ONCALL_ARCHIVE_ROOT \
  ARROW_ONCALL_CONFIG ARROW_ONCALL_CHECKPOINT ARROW_ONCALL_CONTROLLER \
  ARROW_ONCALL_RUNTIME_PYTHON ARROW_ONCALL_HF_CACHE; do
  value="${!name}"
  [[ "$value" = /* && "$value" != *'/../'* && "$value" != */.. ]] || die "$name must be a safe absolute Linux path"
done
command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v git >/dev/null 2>&1 || die 'git is required'

REPO_ROOT="$(realpath -e -- "$ARROW_ONCALL_REPO_ROOT")" || die 'repository cannot be resolved'
[[ -d "$REPO_ROOT/.git" || -f "$REPO_ROOT/.git" ]] || die 'repository is not a Git checkout'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)" || die 'cannot resolve repository SHA'
[[ "$actual_commit" == "$ARROW_ONCALL_EXPECTED_COMMIT" ]] || die "repository SHA mismatch: $actual_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'repository checkout is dirty'
cd -- "$REPO_ROOT"

for name in ARROW_ONCALL_CONFIG ARROW_ONCALL_CONTROLLER ARROW_ONCALL_RUNTIME_PYTHON; do
  [[ -f "${!name}" ]] || die "$name is not a regular file: ${!name}"
done
[[ -e "$ARROW_ONCALL_CHECKPOINT" && ! -L "$ARROW_ONCALL_CHECKPOINT" ]] ||
  die 'checkpoint must be an existing immutable file or directory'
[[ -x "$ARROW_ONCALL_RUNTIME_PYTHON" ]] || die 'runtime Python is not executable'
[[ -d "$ARROW_ONCALL_HF_CACHE" && ! -L "$ARROW_ONCALL_HF_CACHE" ]] || die 'HF cache must be an existing non-symlink directory'

RUN_ROOT="$(realpath -m -- "$ARROW_ONCALL_RUN_ROOT")"
ARCHIVE_ROOT="$(realpath -m -- "$ARROW_ONCALL_ARCHIVE_ROOT")"
case "$RUN_ROOT" in "$REPO_ROOT"|"$REPO_ROOT"/*) die 'run root overlaps repository' ;; esac
case "$ARCHIVE_ROOT" in "$REPO_ROOT"|"$REPO_ROOT"/*) die 'archive root overlaps repository' ;; esac
case "$RUN_ROOT" in "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'run/archive roots overlap' ;; esac
case "$ARCHIVE_ROOT" in "$RUN_ROOT"|"$RUN_ROOT"/*) die 'archive/run roots overlap' ;; esac

TASK_ID="$SLURM_ARRAY_TASK_ID"
TASK_NAME="task_$(printf '%02d' "$TASK_ID")"
TASK_RUN_ROOT="$RUN_ROOT/tasks/$TASK_NAME"
TASK_ARCHIVE_ROOT="$ARCHIVE_ROOT/tasks/$TASK_NAME"
PYTHON="${ARROW_ONCALL_RUNTIME_PYTHON:-}"
mkdir -p "$RUN_ROOT/tasks" "$ARCHIVE_ROOT/tasks"
[[ ! -e "$TASK_RUN_ROOT" ]] || die "task output already exists: $TASK_NAME"
[[ ! -e "$TASK_ARCHIVE_ROOT" ]] || die "task archive already exists: $TASK_NAME"

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

  # Record a terminal worker receipt before copying anything.  This is the
  # create-only failure ledger used by the matrix finalizer when a compute
  # preflight or Python worker crashes before task_summary.json exists.  The
  # receipt is written/fsynced to a unique temporary file, then published by
  # hard-link so readers never observe a partially written target.
  status_root="$RUN_ROOT/worker_status"
  status_path="$status_root/$TASK_NAME.json"
  mkdir -p "$status_root" || archive_rc=$?
  publish_worker_status() {
    local task_status=FAILED_INFRASTRUCTURE temp_path validator
    [[ "${task_rc:-125}" -eq 0 ]] && task_status=COMPLETED
    validator="${PYTHON:-}"
    if [[ ! -x "$validator" ]]; then validator="$(command -v python3 2>/dev/null || true)"; fi

    if [[ -e "$status_path" ]]; then
      [[ -x "$validator" ]] || return 1
      "$validator" - "$status_path" "$TASK_ID" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("schema") != "arrow_policy_suite.oncall_worker_terminal.v1":
    raise SystemExit("worker status schema mismatch")
if payload.get("experiment_evidence") is not False:
    raise SystemExit("worker status evidence flag mismatch")
if payload.get("task_id") != int(sys.argv[2]):
    raise SystemExit("worker status task mismatch")
if payload.get("task_status") not in {"COMPLETED", "FAILED_INFRASTRUCTURE"}:
    raise SystemExit("worker status terminal value mismatch")
PY
      return $?
    fi

    temp_path="$(mktemp "$status_root/.${TASK_NAME}.terminal.XXXXXX")" || return 1
    if [[ -x "${PYTHON:-}" ]]; then
      "$PYTHON" - "$temp_path" "$TASK_ID" "$task_status" "${task_rc:-125}" "$workload_rc" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "arrow_policy_suite.oncall_worker_terminal.v1",
    "experiment_evidence": False,
    "task_id": int(sys.argv[2]),
    "task_status": sys.argv[3],
    "task_exit_code": int(sys.argv[4]),
    "workload_exit_code": int(sys.argv[5]),
}
with path.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
      writer_rc=$?
    else
      # The explicit values are numeric/shell-safe, so retain the terminal
      # ledger even if the module setup failed before Python was usable.  The
      # temp file is flushed before create-only hard-link publication.
      printf '{\n  "experiment_evidence": false,\n  "schema": "arrow_policy_suite.oncall_worker_terminal.v1",\n  "task_exit_code": %s,\n  "task_id": %s,\n  "task_status": "%s",\n  "workload_exit_code": %s\n}\n' \
        "${task_rc:-125}" "$TASK_ID" "$task_status" "$workload_rc" > "$temp_path"
      writer_rc=$?
      sync -d "$temp_path" 2>/dev/null || sync
    fi
    if [[ "${writer_rc:-1}" -ne 0 ]]; then
      rm -f -- "$temp_path"
      return "$writer_rc"
    fi
    if ln -- "$temp_path" "$status_path" 2>/dev/null; then
      rm -f -- "$temp_path"
      return $?
    fi
    rm -f -- "$temp_path"
    # Another writer may have won the create-only race; accept it only after
    # validating the complete JSON receipt against this task identity.
    [[ -x "$validator" ]] || return 1
    "$validator" - "$status_path" "$TASK_ID" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("schema") != "arrow_policy_suite.oncall_worker_terminal.v1":
    raise SystemExit("worker status schema mismatch")
if payload.get("experiment_evidence") is not False:
    raise SystemExit("worker status evidence flag mismatch")
if payload.get("task_id") != int(sys.argv[2]):
    raise SystemExit("worker status task mismatch")
if payload.get("task_status") not in {"COMPLETED", "FAILED_INFRASTRUCTURE"}:
    raise SystemExit("worker status terminal value mismatch")
PY
  }
  publish_worker_status || archive_rc=$?

  # A failed preflight can bypass the normal post-run finalizer.  Let the
  # terminal array element still attempt global sealing, but never replace the
  # original workload exit code with this best-effort call's result.
  finalize_log="$RUN_ROOT/finalize_logs/$TASK_NAME.log"
  mkdir -p "$RUN_ROOT/finalize_logs" || archive_rc=$?
  if [[ -x "${PYTHON:-}" && -n "${ARROW_ONCALL_CONFIG:-}" && -n "${RUN_ROOT:-}" && -n "${ARCHIVE_ROOT:-}" ]]; then
    "$PYTHON" -m vla_benchmarking.libero.arrow_policy_suite.oncall_matrix finalize \
      --config "$ARROW_ONCALL_CONFIG" --output-root "$RUN_ROOT" --archive-root "$ARCHIVE_ROOT" \
      --factory vla_benchmarking.libero.arrow_policy_suite.native_legion_factory:build_host \
      --checkpoint "$ARROW_ONCALL_CHECKPOINT" --controller "$ARROW_ONCALL_CONTROLLER" \
      >> "$finalize_log" 2>&1 || true
  fi

  copy_tree "$TASK_RUN_ROOT" "$TASK_ARCHIVE_ROOT" || archive_rc=$?
  if [[ -s "$status_path" ]]; then
    mkdir -p "$TASK_ARCHIVE_ROOT/worker_status" || archive_rc=$?
    cp -p -- "$status_path" "$TASK_ARCHIVE_ROOT/worker_status/" || archive_rc=$?
  fi
  if [[ -s "$finalize_log" ]]; then
    cp -p -- "$finalize_log" "$TASK_ARCHIVE_ROOT/" || archive_rc=$?
  fi
  log_root="${SLURM_LOG_ROOT:-$HOME/EmbodimentSemantic_runtime/operator/logs}"
  mkdir -p "$TASK_ARCHIVE_ROOT/slurm_logs" || archive_rc=$?
  for suffix in out err; do
    log_path="$log_root/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TASK_ID}.$suffix"
    [[ -f "$log_path" ]] && cp -p -- "$log_path" "$TASK_ARCHIVE_ROOT/slurm_logs/" || true
  done
  if [[ -d "$TASK_ARCHIVE_ROOT" ]]; then
    find "$TASK_ARCHIVE_ROOT" -type f ! -name inventory.sha256 ! -name tree_sha256 -print0 |
      sort -z | xargs -0 -r sha256sum > "$TASK_ARCHIVE_ROOT/inventory.sha256" || archive_rc=$?
    sha256sum "$TASK_ARCHIVE_ROOT/inventory.sha256" | awk '{print tolower($1)}' > "$TASK_ARCHIVE_ROOT/tree_sha256" || archive_rc=$?
    archive_status=PRESERVED_FAILURE
    if [[ "$workload_rc" -eq 0 && "$archive_rc" -eq 0 &&
      -s "$TASK_ARCHIVE_ROOT/task_summary.json" &&
      -s "$TASK_ARCHIVE_ROOT/worker_status/$TASK_NAME.json" &&
      -s "$TASK_ARCHIVE_ROOT/inventory.sha256" ]]; then
      archive_status=VERIFIED
    fi
    printf 'schema=arrow_policy_suite_oncall_task_archive.v1\nexperiment_evidence=false\nstatus=%s\njob_id=%s\narray_task_id=%s\nrepo_commit=%s\nrun_root=%s\narchive_root=%s\n' \
      "$archive_status" \
      "$SLURM_JOB_ID" "$TASK_ID" "$actual_commit" "$TASK_RUN_ROOT" "$TASK_ARCHIVE_ROOT" > "$TASK_ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
  fi
  [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]] && workload_rc=$archive_rc
  exit "$workload_rc"
}
trap archive_on_exit EXIT

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
PYTHON="$ARROW_ONCALL_RUNTIME_PYTHON"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TMPDIR="${SCRATCH_FLASH:-$HOME}/EmbodimentSemantic_runtime/tmp"
mkdir -p "$TMPDIR"

# The standard HF hub contains complete SmolVLM metadata but the Molmo entry
# used by the Arrow teacher is weights/config-only.  Reproduce the validated
# canary overlay exactly, while keeping every overlay task-local and read-only
# with respect to both source caches.
HF_CACHE="$(realpath -e -- "$ARROW_ONCALL_HF_CACHE")" || die 'HF cache cannot be resolved'
[[ "$HF_CACHE" != / && ! -L "$HF_CACHE" ]] || die 'HF cache must be a non-root non-symlink directory'
MOLMO_REVISION='188130f961c8e0888a34e11121a1423c461a01ba'
SMOLVLM_REVISION='7b375e1b73b11138ff12fe22c8f2822d8fe03467'
MOLMO_MODEL_DIR=''
SMOLVLM_MODEL_DIR=''
MOLMO_LAYOUT_ROOT=''
SMOLVLM_LAYOUT_ROOT=''
for cache_layout in hub transformers; do
  layout_root="$HF_CACHE/$cache_layout"
  molmo_snapshot="$layout_root/models--allenai--MolmoPoint-8B/snapshots/$MOLMO_REVISION"
  smolvlm_snapshot="$layout_root/models--HuggingFaceTB--SmolVLM2-500M-Instruct/snapshots/$SMOLVLM_REVISION"
  [[ -f "$smolvlm_snapshot/config.json" && -f "$smolvlm_snapshot/processor_config.json" ]] && {
    SMOLVLM_MODEL_DIR="$layout_root/models--HuggingFaceTB--SmolVLM2-500M-Instruct"
    SMOLVLM_LAYOUT_ROOT="$layout_root"
  }
  if [[ -d "$molmo_snapshot" && -f "$molmo_snapshot/config.json" &&
    -f "$molmo_snapshot/processor_config.json" &&
    -f "$molmo_snapshot/preprocessor_config.json" &&
    -f "$molmo_snapshot/processing_molmo2.py" &&
    -f "$molmo_snapshot/video_processing_molmo2.py" &&
    -f "$molmo_snapshot/tokenizer_config.json" &&
    -f "$molmo_snapshot/tokenizer.json" &&
    -f "$molmo_snapshot/chat_template.jinja" ]]; then
    MOLMO_MODEL_DIR="$layout_root/models--allenai--MolmoPoint-8B"
    MOLMO_LAYOUT_ROOT="$layout_root"
  fi
done
if [[ -z "$MOLMO_MODEL_DIR" ]]; then
  legacy_cache="$HOME/EmbodimentSemantic_runtime/EmbodimentSemantic/grasp_controller/cache/huggingface"
  for cache_layout in hub transformers; do
    layout_root="$legacy_cache/$cache_layout"
    molmo_snapshot="$layout_root/models--allenai--MolmoPoint-8B/snapshots/$MOLMO_REVISION"
    if [[ -f "$molmo_snapshot/config.json" &&
      -f "$molmo_snapshot/processor_config.json" &&
      -f "$molmo_snapshot/preprocessor_config.json" &&
      -f "$molmo_snapshot/processing_molmo2.py" &&
      -f "$molmo_snapshot/video_processing_molmo2.py" &&
      -f "$molmo_snapshot/tokenizer_config.json" &&
      -f "$molmo_snapshot/tokenizer.json" &&
      -f "$molmo_snapshot/chat_template.jinja" ]]; then
      MOLMO_MODEL_DIR="$layout_root/models--allenai--MolmoPoint-8B"
      MOLMO_LAYOUT_ROOT="$layout_root"
      break
    fi
  done
fi
[[ -n "$SMOLVLM_MODEL_DIR" ]] || die 'pinned SmolVLM snapshot is missing complete processor metadata'
[[ -n "$MOLMO_MODEL_DIR" ]] || die 'pinned MolmoPoint snapshot is missing complete processor metadata'
MODEL_CACHE_ROOT="$RUN_ROOT/model_cache/$TASK_NAME/hub"
mkdir -p "$MODEL_CACHE_ROOT"
ln -s -- "$MOLMO_MODEL_DIR" "$MODEL_CACHE_ROOT/models--allenai--MolmoPoint-8B"
ln -s -- "$SMOLVLM_MODEL_DIR" "$MODEL_CACHE_ROOT/models--HuggingFaceTB--SmolVLM2-500M-Instruct"
export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$MODEL_CACHE_ROOT"
export TRANSFORMERS_CACHE="$MODEL_CACHE_ROOT"
export HF_MODULES_CACHE="$HF_CACHE/modules"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

nvidia-smi -L >/dev/null || die 'CUDA device is unavailable'
"$PYTHON" - <<'PY' || die 'compute-node Python preflight failed'
import importlib
import pathlib
import tempfile
import numpy as np
import torch

if not torch.cuda.is_available():
    raise SystemExit('torch.cuda.is_available() is false')
if torch.cuda.device_count() < 1:
    raise SystemExit('no CUDA device visible')
importlib.import_module('vla_benchmarking.libero.arrow_policy_suite.native_legion_factory')
from vla_benchmarking.libero.arrow_policy_suite.native_legion_factory import build_host
if not callable(build_host):
    raise SystemExit('native_legion_factory:build_host is not callable')

# Match the annotated-video dependency contract without running an episode.
import imageio.v2 as imageio
with tempfile.TemporaryDirectory(dir=None) as tmp:
    path = pathlib.Path(tmp) / 'preflight.mp4'
    frames = [np.zeros((32, 32, 3), dtype=np.uint8), np.full((32, 32, 3), 127, dtype=np.uint8)]
    with imageio.get_writer(path, format='FFMPEG', mode='I', fps=5, codec='libx264', pixelformat='yuv420p') as writer:
        for frame in frames:
            writer.append_data(frame)
    decoded = imageio.mimread(path, format='FFMPEG')
    if len(decoded) < 2:
        raise SystemExit(f'video preflight decoded only {len(decoded)} frames')
print('ONCALL_COMPUTE_PREFLIGHT_VERIFIED', torch.cuda.get_device_name(0), flush=True)
PY

# Stable factory inputs.  The matrix worker changes ARROW_SUITE_TASK_ID,
# ARROW_SUITE_SEED, and ARROW_SUITE_INIT_STATE_INDEX itself for each episode;
# this wrapper intentionally does not pin those values to the array element.
export ARROW_SUITE_CHECKPOINT="$ARROW_ONCALL_CHECKPOINT"
export ARROW_SUITE_CONTROLLER="$ARROW_ONCALL_CONTROLLER"
export ARROW_CONTROLLER_CONFIG="$ARROW_ONCALL_CONTROLLER"
export ARROW_SUITE_DEVICE=cuda
export ARROW_SUITE_HF_CACHE="$HF_CACHE"
export ARROW_SUITE_PYTHON="$PYTHON"
export ARROW_SUITE_SUITE_MODE=vanilla
export ARROW_SUITE_RESOLUTION=256
export ARROW_SUITE_CONTROLLER_HASH="$($PYTHON -c 'import sys; from vla_benchmarking.libero.arrow_grasp_controller.configs import load_controller_config; print(load_controller_config(sys.argv[1])["config_hash"])' "$ARROW_ONCALL_CONTROLLER")"
[[ "$ARROW_SUITE_CONTROLLER_HASH" =~ ^[0-9a-f]{64}$ ]] || die 'resolved controller config hash is invalid'

export ARROW_ONCALL_TASK_ID="$TASK_ID"
export ARROW_ONCALL_TASK_RUN_ROOT="$TASK_RUN_ROOT"
export ARROW_ONCALL_TASK_ARCHIVE_ROOT="$TASK_ARCHIVE_ROOT"
export ARROW_ONCALL_EPISODES=10
export ARROW_ONCALL_EPISODE_START=0
export ARROW_ONCALL_SEED_BASE=1000
export ARROW_ONCALL_INIT_STATE_BASE=0
export ARROW_ONCALL_HORIZON=280
export ARROW_ONCALL_POLICY=arrow_on_call
export ARROW_ONCALL_VIDEO_EPISODE=0
export ARROW_ONCALL_ARCHIVE_STATUS="$ARCHIVE_ROOT/matrix_status.json"

set +e
"$PYTHON" -m vla_benchmarking.libero.arrow_policy_suite.oncall_matrix run-task \
  --config "$ARROW_ONCALL_CONFIG" \
  --output-root "$RUN_ROOT" \
  --task-id "$TASK_ID" \
  --factory vla_benchmarking.libero.arrow_policy_suite.native_legion_factory:build_host \
  --checkpoint "$ARROW_ONCALL_CHECKPOINT" \
  --controller "$ARROW_ONCALL_CONTROLLER"
task_rc=$?
set -e

# Finalization is deliberately attempted even after a task-side infrastructure
# failure, so the matrix retains a sealed partial ledger and the task archive.
set +e
"$PYTHON" -m vla_benchmarking.libero.arrow_policy_suite.oncall_matrix finalize \
  --config "$ARROW_ONCALL_CONFIG" \
  --output-root "$RUN_ROOT" \
  --archive-root "$ARCHIVE_ROOT" \
  --factory vla_benchmarking.libero.arrow_policy_suite.native_legion_factory:build_host \
  --checkpoint "$ARROW_ONCALL_CHECKPOINT" \
  --controller "$ARROW_ONCALL_CONTROLLER"
final_rc=$?
set -e

# The finalizer is allowed to return before the other array elements finish.
# Once all ten task manifests are present, however, a successful finalizer must
# have emitted the durable archive-level seal/status requested by the matrix
# contract.  This catches a Python finalizer that only writes scratch output.
task_manifest_count=0
for manifest in "$RUN_ROOT"/tasks/task_*/task_summary.json; do
  [[ -s "$manifest" ]] && task_manifest_count=$((task_manifest_count + 1)) || true
done
if [[ "$task_manifest_count" -eq 10 && "$final_rc" -eq 0 &&
  ! -s "$ARROW_ONCALL_ARCHIVE_STATUS" ]]; then
  printf 'arrow oncall matrix: all task manifests exist but archive seal is missing: %s\n' \
    "$ARROW_ONCALL_ARCHIVE_STATUS" >&2
  final_rc=3
fi

if [[ "$task_rc" -ne 0 ]]; then exit "$task_rc"; fi
exit "$final_rc"
