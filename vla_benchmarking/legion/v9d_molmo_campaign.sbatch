#!/usr/bin/env bash
# One isolated, single-GPU campaign for all SAM-free v9d+Molmo variants.
#
# Submit this file through the Legion SSH wrapper from an immutable, clean
# release checkout. The campaign entrypoint owns the ordered variant list and
# loads the pinned public MolmoPoint model once, reusing it for every variant.
# This wrapper owns the allocation, compatibility venv, persistent caches,
# provenance, and failure-safe archival.
#
# Required submission environment:
#   REPO_ROOT                 isolated clean checkout used for the campaign
#   CANARY_EXPECTED_COMMIT    exact lowercase 40-character checkout SHA
#   CANARY_LABEL              safe run label (or pass it as the first argument)
#
# Optional:
#   V9D_MOLMO_OUTPUT_ROOT     absolute output path; defaults to scratch/results
#   V9D_MOLMO_RUNTIME_ROOT    persistent runtime root under $HOME by default
#   V9D_MOLMO_ARCHIVE_ROOT    persistent archive root under $HOME by default

#SBATCH --job-name=v9d_molmo_campaign
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
  printf 'v9d+Molmo campaign: %s\n' "$*" >&2
  exit 2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit with sbatch; direct execution is refused'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
[[ "${CANARY_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die 'CANARY_EXPECTED_COMMIT must be a lowercase 40-character SHA'
CANARY_LABEL="${CANARY_LABEL:-${1:-}}"
[[ -n "$CANARY_LABEL" ]] || die 'CANARY_LABEL is required (or pass a label as argv[1])'
[[ "$CANARY_LABEL" =~ ^[A-Za-z0-9_.-]{1,96}$ ]] || die 'CANARY_LABEL contains unsafe characters'

command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")" || die 'cannot resolve REPO_ROOT'
actual_root="$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null)" || die 'REPO_ROOT is not a Git checkout'
actual_root="$(realpath -e -- "$actual_root")" || die 'cannot resolve Git checkout root'
[[ "$actual_root" == "$REPO_ROOT" ]] || die 'REPO_ROOT is not the worktree root'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] || die 'shared HOME checkout is forbidden; use an isolated release'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)" || die 'cannot read REPO_ROOT commit'
[[ "$actual_commit" == "$CANARY_EXPECTED_COMMIT" ]] || die "checkout SHA differs: $actual_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'REPO_ROOT is dirty'

BASE_PYTHON="${V9D_MOLMO_BASE_PYTHON:-/home/hjaber/.conda/envs/embodiment-smolvla-py312/bin/python}"
[[ -x "$BASE_PYTHON" ]] || die "validated base Python is missing: $BASE_PYTHON"
CAMPAIGN_SCRIPT="$REPO_ROOT/vla_benchmarking/run_v9d_molmo_campaign.py"
REQUIREMENTS_FILE="$REPO_ROOT/vla_benchmarking/legion/molmo_requirements.txt"
[[ -f "$CAMPAIGN_SCRIPT" ]] || die "campaign entrypoint is missing: $CAMPAIGN_SCRIPT"
[[ -f "$REQUIREMENTS_FILE" ]] || die "Molmo requirements are missing: $REQUIREMENTS_FILE"

SCRATCH_ROOT="${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/v9d_molmo"
RUN_ROOT="${V9D_MOLMO_RUN_ROOT:-$SCRATCH_ROOT/runs/${CANARY_LABEL}_${SLURM_JOB_ID}}"
OUTPUT_ROOT="${V9D_MOLMO_OUTPUT_ROOT:-$RUN_ROOT/results}"
RUNTIME_ROOT="${V9D_MOLMO_RUNTIME_ROOT:-$HOME/EmbodimentSemantic_runtime/EmbodimentSemantic/v9d_molmo}"
ARCHIVE_ROOT="${V9D_MOLMO_ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/v9d_molmo/${CANARY_LABEL}_${SLURM_JOB_ID}}"
for path in "$RUN_ROOT" "$OUTPUT_ROOT" "$RUNTIME_ROOT" "$ARCHIVE_ROOT"; do
  [[ "$path" = /* && "$path" != / ]] || die "path must be an absolute non-root path: $path"
done
RUN_ROOT="$(realpath -m -- "$RUN_ROOT")" || die 'cannot canonicalize RUN_ROOT'
OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")" || die 'cannot canonicalize OUTPUT_ROOT'
RUNTIME_ROOT="$(realpath -m -- "$RUNTIME_ROOT")" || die 'cannot canonicalize RUNTIME_ROOT'
ARCHIVE_ROOT="$(realpath -m -- "$ARCHIVE_ROOT")" || die 'cannot canonicalize ARCHIVE_ROOT'
[[ ! -e "$RUN_ROOT" ]] || die "RUN_ROOT already exists: $RUN_ROOT"
[[ ! -e "$ARCHIVE_ROOT" ]] || die "ARCHIVE_ROOT already exists: $ARCHIVE_ROOT"
case "$ARCHIVE_ROOT" in "$RUN_ROOT"|"$RUN_ROOT"/*) die 'ARCHIVE_ROOT cannot be inside RUN_ROOT' ;; esac
case "$RUN_ROOT" in "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'RUN_ROOT cannot be inside ARCHIVE_ROOT' ;; esac
mkdir -p "$RUN_ROOT" "$OUTPUT_ROOT" "$RUNTIME_ROOT"

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
  mkdir -p "$ARCHIVE_ROOT"
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
  {
    printf 'archive_status=%s\n' "$([[ "$archive_rc" -eq 0 ]] && printf VERIFIED || printf FAILED)"
    printf 'job_exit_code=%s\njob_id=%s\n' "$workload_rc" "$SLURM_JOB_ID"
    printf 'expected_commit=%s\nactual_commit=%s\nlabel=%s\n' "$CANARY_EXPECTED_COMMIT" "$actual_commit" "$CANARY_LABEL"
    printf 'run_root=%s\noutput_root=%s\nruntime_root=%s\narchive_root=%s\n' "$RUN_ROOT" "$OUTPUT_ROOT" "$RUNTIME_ROOT" "$ARCHIVE_ROOT"
  } > "$ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
  if [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]]; then workload_rc=$archive_rc; fi
  exit "$workload_rc"
}
trap archive_on_exit EXIT

module purge
module load miniforge/24.3.0-0

export TMPDIR="$RUN_ROOT/tmp"
export XDG_CACHE_HOME="$RUNTIME_ROOT/cache/xdg"
export TORCH_HOME="$RUNTIME_ROOT/cache/torch"
export HF_HOME="$RUNTIME_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$REPO_ROOT/vla_benchmarking"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/EmbodimentSemantic_runtime/config}"
export LIBERO_CONFIG="$LIBERO_CONFIG_PATH/config.yaml"
[[ -d "$LIBERO_CONFIG_PATH" ]] || die "LIBERO_CONFIG_PATH is missing: $LIBERO_CONFIG_PATH"
[[ -f "$LIBERO_CONFIG" ]] || die "LIBERO_CONFIG is missing: $LIBERO_CONFIG"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$OUTPUT_ROOT"

VENV="$RUNTIME_ROOT/venv-py312"
setup_started_epoch="$(date +%s)"
if [[ -e "$VENV" ]]; then
  [[ -x "$VENV/bin/python" ]] || die "existing runtime venv is incomplete: $VENV"
else
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV" || die 'could not create Molmo runtime venv'
fi
[[ -x "$VENV/bin/python" ]] || die "runtime venv Python is missing: $VENV/bin/python"
# venv/bin/python is normally a symlink to the base interpreter. Keep the
# venv path literal: realpath would bypass the venv and let pip mutate the
# validated base environment instead of the isolated overlay.
PYTHON="$VENV/bin/python"
"$PYTHON" -c 'import sys; assert sys.prefix != sys.base_prefix, (sys.prefix, sys.base_prefix)' \
  || die 'runtime Python is not an isolated venv interpreter'

# --no-deps is intentional. The validated base environment owns torch and
# simulation packages; this compatibility layer must not resolve a second
# torch stack or upgrade the simulator. The manifest pins the packages
# intentionally overlaid in the project-local venv.
"$PYTHON" -m pip install --disable-pip-version-check --no-input --no-deps -r "$REQUIREMENTS_FILE" \
  || die 'Molmo compatibility package installation failed'

setup_finished_epoch="$(date +%s)"
"$PYTHON" - "$BASE_PYTHON" "$REQUIREMENTS_FILE" "$RUN_ROOT" "$CANARY_EXPECTED_COMMIT" "$actual_commit" <<'PY'
import importlib.metadata as md
import json
import pathlib
import subprocess
import sys

base_python, requirements, run_root, expected_commit, actual_commit = sys.argv[1:]
root = pathlib.Path(run_root)
def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None

required = {
    "transformers": "4.57.1",
    "huggingface-hub": "0.36.0",
    "tokenizers": "0.22.1",
    "einops": "0.8.1",
    "decord2": "3.4.0",
}
observed = {name: version(name) for name in required}
for name, expected in required.items():
    if observed[name] != expected:
        raise SystemExit(f"{name} version mismatch: {observed[name]!r} != {expected!r}")
import torch
if version("torch") != "2.10.0":
    raise SystemExit(f"torch version changed or is unavailable: {version('torch')!r}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one CUDA device for the campaign")
import transformers
if transformers.__version__ != "4.57.1":
    raise SystemExit("runtime Transformers version mismatch")
import libero  # noqa: F401
import mujoco  # noqa: F401
import robosuite  # noqa: F401
runtime = {
    "python": sys.executable,
    "python_version": sys.version,
    "base_python": base_python,
    "requirements_file": requirements,
    "expected_commit": expected_commit,
    "actual_commit": actual_commit,
    "packages": {**observed, "torch": version("torch"), "accelerate": version("accelerate"), "safetensors": version("safetensors")},
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()),
    "cuda_device_name": torch.cuda.get_device_name(0),
    "dtype": str(torch.bfloat16),
}
(root / "runtime_versions.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(root / "pip_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
print(json.dumps(runtime, sort_keys=True))
PY

requirements_sha256="$(sha256sum -- "$REQUIREMENTS_FILE" | awk '{print $1}')"
campaign_sha256="$(sha256sum -- "$CAMPAIGN_SCRIPT" | awk '{print $1}')"
cat > "$RUN_ROOT/job_context.env" <<EOF
schema=v9d_molmo_campaign_job.v1
job_id=$SLURM_JOB_ID
job_name=$SLURM_JOB_NAME
label=$CANARY_LABEL
expected_commit=$CANARY_EXPECTED_COMMIT
actual_commit=$actual_commit
repo_root=$REPO_ROOT
campaign_script=$CAMPAIGN_SCRIPT
campaign_script_sha256=$campaign_sha256
requirements_file=$REQUIREMENTS_FILE
requirements_sha256=$requirements_sha256
base_python=$BASE_PYTHON
runtime_python=$PYTHON
runtime_root=$RUNTIME_ROOT
run_root=$RUN_ROOT
output_root=$OUTPUT_ROOT
archive_root=$ARCHIVE_ROOT
hf_home=$HF_HOME
model_id=allenai/MolmoPoint-8B
model_revision=188130f961c8e0888a34e11121a1423c461a01ba
dtype=bfloat16
setup_started_epoch=$setup_started_epoch
setup_finished_epoch=$setup_finished_epoch
setup_elapsed_seconds=$((setup_finished_epoch - setup_started_epoch))
model_load_owner=run_v9d_molmo_campaign.py
model_load_count_expected=1
EOF

"$PYTHON" "$CAMPAIGN_SCRIPT" --output-dir "$OUTPUT_ROOT" --label "$CANARY_LABEL"
