#!/usr/bin/env bash
# Evaluate the frozen e8 Molmo-dense RGB-D/no-SAM policy on the complete
# sealed-randomized 100-cell inventory (tasks 0..9 x episodes 1000..1009).
#
# This is a separate immutable experiment. It intentionally does not import,
# download, or initialize SAM3: the e8 winner uses the v9d RGB-D region path.
# Required submit-time values: REPO_ROOT, CANARY_EXPECTED_COMMIT,
# SEALED100_LABEL, SEALED100_OPERATOR_SCRIPT.

#SBATCH --job-name=molmo_sealed100
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

die() { printf 'molmo sealed100: %s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit with sbatch; direct execution is refused'
REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
EXPECTED_COMMIT="${CANARY_EXPECTED_COMMIT:?CANARY_EXPECTED_COMMIT is required}"
LABEL="${SEALED100_LABEL:?SEALED100_LABEL is required}"
OPERATOR_SCRIPT="${SEALED100_OPERATOR_SCRIPT:?SEALED100_OPERATOR_SCRIPT is required}"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'CANARY_EXPECTED_COMMIT must be a lowercase 40-character SHA'
[[ "$LABEL" =~ ^[A-Za-z0-9_.-]{1,96}$ ]] || die 'SEALED100_LABEL contains unsafe characters'

command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")" || die 'cannot resolve REPO_ROOT'
OPERATOR_SCRIPT="$(realpath -e -- "$OPERATOR_SCRIPT")" || die 'cannot resolve SEALED100_OPERATOR_SCRIPT'
[[ -f "$OPERATOR_SCRIPT" ]] || die 'SEALED100_OPERATOR_SCRIPT is not a file'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] || die 'shared HOME checkout is forbidden; use an isolated release'
actual_root="$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null)" || die 'REPO_ROOT is not a Git checkout'
actual_root="$(realpath -e -- "$actual_root")" || die 'cannot resolve Git worktree root'
[[ "$actual_root" == "$REPO_ROOT" ]] || die 'REPO_ROOT is not the worktree root'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)" || die 'cannot read REPO_ROOT commit'
[[ "$actual_commit" == "$EXPECTED_COMMIT" ]] || die "checkout SHA differs: $actual_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'REPO_ROOT is dirty'

RUNTIME_ROOT="${V9D_MOLMO_RUNTIME_ROOT:-$HOME/EmbodimentSemantic_runtime/EmbodimentSemantic/v9d_molmo}"
PYTHON="${SEALED100_PYTHON:-$RUNTIME_ROOT/venv-py312/bin/python}"
CONFIG_PATH="${SEALED100_CONFIG:-$REPO_ROOT/vla_benchmarking/controller_configs/v9d_rgbd_region_grasp_search.json}"
LIBERO_CONFIG_ROOT="${LIBERO_CONFIG_PATH:-$HOME/EmbodimentSemantic_runtime/config}"
SCRATCH_ROOT="${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/molmo_sealed100"
RUN_ROOT="${SEALED100_RUN_ROOT:-$SCRATCH_ROOT/runs/${LABEL}_${SLURM_JOB_ID}}"
ARCHIVE_ROOT="${SEALED100_ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/molmo_sealed100/${LABEL}_${SLURM_JOB_ID}}"
for path in "$RUN_ROOT" "$ARCHIVE_ROOT" "$CONFIG_PATH" "$LIBERO_CONFIG_ROOT"; do
  [[ "$path" = /* && "$path" != / ]] || die "path must be an absolute non-root path: $path"
done
RUN_ROOT="$(realpath -m -- "$RUN_ROOT")" || die 'cannot canonicalize RUN_ROOT'
ARCHIVE_ROOT="$(realpath -m -- "$ARCHIVE_ROOT")" || die 'cannot canonicalize ARCHIVE_ROOT'
[[ "$RUN_ROOT" != "$ARCHIVE_ROOT" ]] || die 'RUN_ROOT and ARCHIVE_ROOT must differ'
case "$ARCHIVE_ROOT" in "$RUN_ROOT"|"$RUN_ROOT"/*) die 'ARCHIVE_ROOT cannot be inside RUN_ROOT' ;; esac
case "$RUN_ROOT" in "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'RUN_ROOT cannot be inside ARCHIVE_ROOT' ;; esac
[[ ! -e "$RUN_ROOT" ]] || die "RUN_ROOT already exists: $RUN_ROOT"
[[ ! -e "$ARCHIVE_ROOT" ]] || die "ARCHIVE_ROOT already exists: $ARCHIVE_ROOT"
[[ -x "$PYTHON" ]] || die "MolmoPoint Python is missing: $PYTHON"
[[ -f "$CONFIG_PATH" ]] || die "controller config is missing: $CONFIG_PATH"
[[ -d "$LIBERO_CONFIG_ROOT" && -f "$LIBERO_CONFIG_ROOT/config.yaml" ]] || die 'LIBERO_CONFIG_PATH/config.yaml is missing'

mkdir -p "$RUN_ROOT/tmp" "$RUN_ROOT/cache"
archive_on_exit() {
  local workload_rc=$? archive_rc=0 result_status=UNVERIFIED
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT" || archive_rc=$?
  if [[ "$archive_rc" -eq 0 ]]; then
    if command -v rsync >/dev/null 2>&1; then
      rsync -a -- "$RUN_ROOT/" "$ARCHIVE_ROOT/run/" || archive_rc=$?
    else
      mkdir -p "$ARCHIVE_ROOT/run" || archive_rc=$?
      cp -a -- "$RUN_ROOT/." "$ARCHIVE_ROOT/run/" || archive_rc=$?
    fi
  fi
  log_root="${SLURM_LOG_ROOT:-$HOME/EmbodimentSemantic_runtime/operator/logs}"
  mkdir -p "$ARCHIVE_ROOT/slurm_logs" || archive_rc=$?
  for suffix in out err; do
    log_path="$log_root/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${suffix}"
    if [[ -f "$log_path" ]]; then cp -p -- "$log_path" "$ARCHIVE_ROOT/slurm_logs/" || archive_rc=$?; fi
  done
  if [[ "$archive_rc" -eq 0 && -f "$ARCHIVE_ROOT/run/job_context.env" &&
        ( -f "$ARCHIVE_ROOT/run/results/molmo_sam3_canary_summary.json" ||
          -f "$ARCHIVE_ROOT/run/results/molmo_sam3_canary_failed.json" ) ]]; then
    result_status=VERIFIED
  elif [[ "$archive_rc" -ne 0 ]]; then
    result_status=FAILED
  fi
  if [[ -d "$ARCHIVE_ROOT" ]]; then
    {
      printf 'schema=molmo_dense_sealed100_job.v1\n'
      printf 'archive_status=%s\njob_exit_code=%s\njob_id=%s\n' "$result_status" "$workload_rc" "$SLURM_JOB_ID"
      printf 'expected_commit=%s\nactual_commit=%s\nlabel=%s\n' "$EXPECTED_COMMIT" "$actual_commit" "$LABEL"
      printf 'variant=molmo_dense_agentview\nregion_backend=rgbd\nsam3_used=false\n'
      printf 'motion_profile=release_plus20mm\nobservation_profile=parked\nopening_profile=full_open\n'
      printf 'molmopoint_model=allenai/MolmoPoint-8B\nmolmopoint_revision=188130f961c8e0888a34e11121a1423c461a01ba\nmolmopoint_prompt_id=rim_clearance\n'
      printf 'task_ids=0,1,2,3,4,5,6,7,8,9\nepisodes_per_task=10\nseed_base=1000\nsuite_modes=sealed_randomized\n'
      printf 'operator_script=%s\noperator_script_sha256=%s\npython=%s\nruntime_root=%s\n' "$OPERATOR_SCRIPT" "${OPERATOR_SCRIPT_SHA256:-}" "$PYTHON" "$RUNTIME_ROOT"
      printf 'run_root=%s\narchive_root=%s\n' "$RUN_ROOT" "$ARCHIVE_ROOT"
    } > "$ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
  fi
  if [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]]; then workload_rc=$archive_rc; fi
  exit "$workload_rc"
}
trap archive_on_exit EXIT

OPERATOR_SCRIPT_SHA256="$(sha256sum -- "$OPERATOR_SCRIPT" | awk '{print tolower($1)}')" || die 'cannot hash operator script'
CONFIG_SHA256="$(sha256sum -- "$CONFIG_PATH" | awk '{print tolower($1)}')" || die 'cannot hash controller config'
[[ "$CONFIG_SHA256" =~ ^[0-9a-f]{64}$ ]] || die 'invalid controller config hash'

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_ROOT" LIBERO_CONFIG="$LIBERO_CONFIG_ROOT/config.yaml"
export PYTHONPATH="$REPO_ROOT/vla_benchmarking"
export TMPDIR="$RUN_ROOT/tmp" XDG_CACHE_HOME="$RUNTIME_ROOT/cache/xdg" TORCH_HOME="$RUNTIME_ROOT/cache/torch"
export HF_HOME="$RUNTIME_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

command -v nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi is unavailable in the allocation'
gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
[[ "$gpu_count" == 1 ]] || die "expected exactly one visible GPU, found $gpu_count"

"$PYTHON" - "$RUN_ROOT" <<'PY'
import importlib.metadata as metadata
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
required = {
    "torch": "2.10.0", "transformers": "4.57.1", "huggingface-hub": "0.36.0",
    "tokenizers": "0.22.1", "einops": "0.8.1", "decord2": "3.4.0",
}
observed = {name: metadata.version(name) for name in required}
for name, expected in required.items():
    if observed[name] != expected:
        raise SystemExit(f"{name} version mismatch: {observed[name]!r} != {expected!r}")
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one CUDA device")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA device does not support BF16")
import transformers  # noqa: F401
from PIL import Image  # noqa: F401
runtime = {
    "python": sys.executable, "python_version": sys.version,
    "packages": observed, "cuda_device_count": int(torch.cuda.device_count()),
    "cuda_device_name": torch.cuda.get_device_name(0), "bf16_supported": True,
}
(run_root / "runtime_versions.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(runtime, sort_keys=True))
PY

cat > "$RUN_ROOT/job_context.env" <<EOF
schema=molmo_dense_sealed100_job.v1
job_id=$SLURM_JOB_ID
expected_commit=$EXPECTED_COMMIT
actual_commit=$actual_commit
repo_root=$REPO_ROOT
label=$LABEL
variant=molmo_dense_agentview
region_backend=rgbd
sam3_used=false
motion_profile=release_plus20mm
observation_profile=parked
opening_profile=full_open
molmopoint_model=allenai/MolmoPoint-8B
molmopoint_revision=188130f961c8e0888a34e11121a1423c461a01ba
molmopoint_prompt_id=rim_clearance
task_ids=0,1,2,3,4,5,6,7,8,9
episodes_per_task=10
seed_base=1000
suite_modes=sealed_randomized
evaluation_scope=sealed_100_cells
config_path=$CONFIG_PATH
config_sha256=$CONFIG_SHA256
python=$PYTHON
runtime_root=$RUNTIME_ROOT
operator_script=$OPERATOR_SCRIPT
operator_script_sha256=$OPERATOR_SCRIPT_SHA256
run_root=$RUN_ROOT
archive_root=$ARCHIVE_ROOT
EOF

RUNNER="$REPO_ROOT/vla_benchmarking/run_molmo_sam3_canary.py"
[[ -f "$RUNNER" ]] || die "runner is missing: $RUNNER"
set +e
"$PYTHON" "$RUNNER" \
  --sealed-100 --phase full60 \
  --variant molmo_dense_agentview \
  --region-backend rgbd \
  --config "$CONFIG_PATH" \
  --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --episodes-per-task 10 \
  --seed-base 1000 \
  --suite-modes sealed_randomized \
  --motion-profile release_plus20mm \
  --observation-profile parked \
  --opening-profile full_open \
  --molmopoint-model allenai/MolmoPoint-8B \
  --molmopoint-revision 188130f961c8e0888a34e11121a1423c461a01ba \
  --molmopoint-prompt-id rim_clearance \
  --runtime-root "$RUNTIME_ROOT" \
  --output-dir "$RUN_ROOT/results" \
  --label "$LABEL"
workload_rc=$?
set -e
exit "$workload_rc"
