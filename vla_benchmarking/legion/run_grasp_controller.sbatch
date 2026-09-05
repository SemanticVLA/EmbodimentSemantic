#!/usr/bin/env bash
# This launcher must retain LF endings for Bash/SLURM portability.
# Canonical grasp-controller evaluation launcher.
#
# This is the sole active Legion launcher for the frozen default treatment.
# The treatment, model, prompt, camera/backend, grasp geometry, opening,
# motion, retry, and evaluator policies are fixed by
# vla_benchmarking/run_grasp_controller.py. This wrapper intentionally
# exposes only evaluation-scope settings.
#
# Required at submission time:
#   REPO_ROOT                         isolated clean checkout
#   GRASP_CONTROLLER_EXPECTED_COMMIT exact lowercase 40-character SHA
#
# Scope settings (environment variables only):
#   GRASP_CONTROLLER_LABEL            safe run label (or argv[1])
#   GRASP_TASK_IDS                    comma-separated task IDs (default 0..9)
#   GRASP_EPISODES_PER_TASK          positive episode count (default 10)
#   GRASP_SEED_BASE                  non-negative seed base (default 1000)
#   GRASP_SUITE_MODES                suite list (default sealed_randomized)
#   GRASP_OUTPUT_ROOT                absolute output path (default: run/results)
#
# Do not add treatment knobs here. A stale experimental knob is rejected
# rather than silently changing the canonical experiment.

#SBATCH --job-name=grasp_controller
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
  printf 'canonical grasp-controller: %s\n' "$*" >&2
  exit 2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit with sbatch; direct execution is refused'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
[[ "${GRASP_CONTROLLER_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || die 'GRASP_CONTROLLER_EXPECTED_COMMIT must be a lowercase 40-character SHA'

command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'
command -v flock >/dev/null 2>&1 || die 'flock is required'

REPO_ROOT="$(realpath -e -- "$REPO_ROOT")" || die 'cannot resolve REPO_ROOT'
actual_root="$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null)" \
  || die 'REPO_ROOT is not a Git checkout'
actual_root="$(realpath -e -- "$actual_root")" || die 'cannot resolve Git root'
[[ "$actual_root" == "$REPO_ROOT" ]] || die 'REPO_ROOT is not the worktree root'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] \
  || die 'shared HOME checkout is forbidden; use an isolated immutable release'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)" \
  || die 'cannot read REPO_ROOT commit'
[[ "$actual_commit" == "$GRASP_CONTROLLER_EXPECTED_COMMIT" ]] \
  || die "checkout SHA differs: $actual_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] \
  || die 'REPO_ROOT is dirty'

LABEL="${GRASP_CONTROLLER_LABEL:-${1:-}}"
[[ "$LABEL" =~ ^[A-Za-z0-9_.-]{1,96}$ ]] \
  || die 'GRASP_CONTROLLER_LABEL is required and must be safe (or pass it as argv[1])'
TASK_IDS="${GRASP_TASK_IDS:-0,1,2,3,4,5,6,7,8,9}"
EPISODES="${GRASP_EPISODES_PER_TASK:-10}"
SEED_BASE="${GRASP_SEED_BASE:-1000}"
SUITE_MODES="${GRASP_SUITE_MODES:-sealed_randomized}"
[[ "$TASK_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die 'GRASP_TASK_IDS is invalid'
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die 'GRASP_EPISODES_PER_TASK must be positive'
[[ "$SEED_BASE" =~ ^[0-9]+$ ]] || die 'GRASP_SEED_BASE must be non-negative'
[[ "$SUITE_MODES" =~ ^(vanilla|sealed_randomized)(,(vanilla|sealed_randomized))*$ ]] \
  || die 'GRASP_SUITE_MODES must contain vanilla and/or sealed_randomized'

BASE_PYTHON="/home/hjaber/.conda/envs/embodiment-smolvla-py312/bin/python"
RUNTIME_ROOT="${HOME}/EmbodimentSemantic_runtime/EmbodimentSemantic/grasp_controller"
SCRATCH_ROOT="${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/grasp_controller"
RUN_ROOT="${GRASP_RUN_ROOT:-$SCRATCH_ROOT/runs/${LABEL}_${SLURM_JOB_ID}}"
OUTPUT_ROOT="${GRASP_OUTPUT_ROOT:-$RUN_ROOT/results}"
ARCHIVE_ROOT="${HOME}/EmbodimentSemantic_archive/grasp_controller/${LABEL}_${SLURM_JOB_ID}"
REQUIREMENTS_FILE="$REPO_ROOT/vla_benchmarking/legion/grasp_controller_requirements.txt"
ENTRYPOINT="$REPO_ROOT/vla_benchmarking/run_grasp_controller.py"
CANONICAL_CONFIG="$REPO_ROOT/vla_benchmarking/controller_configs/canonical_molmo_rgbd_grasp.json"
POLICY_LOCK="$REPO_ROOT/vla_benchmarking/controller_configs/active_policy.lock.json"

for path in "$RUNTIME_ROOT" "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT"; do
  [[ "$path" = /* && "$path" != / ]] || die "path must be an absolute non-root path: $path"
done
RUN_ROOT="$(realpath -m -- "$RUN_ROOT")" || die 'cannot canonicalize RUN_ROOT'
OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")" || die 'cannot canonicalize OUTPUT_ROOT'
RUNTIME_ROOT="$(realpath -m -- "$RUNTIME_ROOT")" || die 'cannot canonicalize RUNTIME_ROOT'
ARCHIVE_ROOT="$(realpath -m -- "$ARCHIVE_ROOT")" || die 'cannot canonicalize ARCHIVE_ROOT'
[[ ! -e "$RUN_ROOT" ]] || die "RUN_ROOT already exists: $RUN_ROOT"
[[ ! -e "$OUTPUT_ROOT" ]] || die "OUTPUT_ROOT already exists: $OUTPUT_ROOT"
[[ ! -e "$ARCHIVE_ROOT" ]] || die "ARCHIVE_ROOT already exists: $ARCHIVE_ROOT"
case "$ARCHIVE_ROOT" in "$RUN_ROOT"|"$RUN_ROOT"/*) die 'archive cannot be inside run root' ;; esac
case "$RUN_ROOT" in "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'run root cannot be inside archive' ;; esac
[[ -f "$REQUIREMENTS_FILE" ]] || die "requirements file is missing: $REQUIREMENTS_FILE"
[[ -f "$ENTRYPOINT" ]] || die "canonical entrypoint is missing: $ENTRYPOINT"
[[ -f "$CANONICAL_CONFIG" ]] || die "canonical config is missing: $CANONICAL_CONFIG"
[[ -f "$POLICY_LOCK" ]] || die "policy lock is missing: $POLICY_LOCK"

IFS=$'\t' read -r CANONICAL_CONFIG_SHA256 LOCKED_CONFIG_SHA256 < <(
  "$BASE_PYTHON" - "$REPO_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "vla_benchmarking"))
from controller_configs import load_controller_config

config = load_controller_config()
lock = json.loads(
    (root / "vla_benchmarking/controller_configs/active_policy.lock.json").read_text(encoding="utf-8")
)
print(f"{config['config_hash']}\t{lock['canonical_config_sha256']}")
PY
)
[[ "$CANONICAL_CONFIG_SHA256" =~ ^[0-9a-f]{64}$ ]] || die 'canonical config hash is invalid'
[[ "$CANONICAL_CONFIG_SHA256" == "$LOCKED_CONFIG_SHA256" ]] \
  || die "canonical config does not match policy lock: $CANONICAL_CONFIG_SHA256"

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
  local workload_rc=$? archive_rc=0 status=INCOMPLETE archived_output_root=''
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT" || archive_rc=$?
  if [[ "$archive_rc" -eq 0 ]]; then
    copy_tree "$RUN_ROOT" "$ARCHIVE_ROOT/run" || archive_rc=$?
    if [[ "$OUTPUT_ROOT" != "$RUN_ROOT" && "$OUTPUT_ROOT" != "$RUN_ROOT"/* ]]; then
      copy_tree "$OUTPUT_ROOT" "$ARCHIVE_ROOT/output" || archive_rc=$?
    fi
  fi
  log_root="${SLURM_LOG_ROOT:-$HOME/EmbodimentSemantic_runtime/operator/logs}"
  mkdir -p "$ARCHIVE_ROOT/slurm_logs" || archive_rc=$?
  for suffix in out err; do
    log_path="$log_root/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${suffix}"
    [[ -f "$log_path" ]] && cp -p -- "$log_path" "$ARCHIVE_ROOT/slurm_logs/" || true
  done
  if [[ "$OUTPUT_ROOT" == "$RUN_ROOT" ]]; then
    archived_output_root="$ARCHIVE_ROOT/run"
  elif [[ "$OUTPUT_ROOT" == "$RUN_ROOT"/* ]]; then
    archived_output_root="$ARCHIVE_ROOT/run/${OUTPUT_ROOT#"$RUN_ROOT"/}"
  else
    archived_output_root="$ARCHIVE_ROOT/output"
  fi
  if [[ "$archive_rc" -ne 0 ]]; then
    status=FAILED
  elif [[ "$workload_rc" -eq 0 && -f "$archived_output_root/canonical_grasp_summary.json" ]]; then
    status=VERIFIED
  elif [[ -f "$archived_output_root/canonical_grasp_failed.json" ]]; then
    status=PRESERVED_FAILURE
  fi
  if [[ -d "$ARCHIVE_ROOT" ]]; then
    {
      printf 'schema=canonical_grasp_controller_job.v1\n'
      printf 'archive_status=%s\njob_exit_code=%s\njob_id=%s\n' "$status" "$workload_rc" "$SLURM_JOB_ID"
      printf 'expected_commit=%s\nactual_commit=%s\nlabel=%s\n' \
        "$GRASP_CONTROLLER_EXPECTED_COMMIT" "$actual_commit" "$LABEL"
      printf 'task_ids=%s\nepisodes_per_task=%s\nseed_base=%s\nsuite_modes=%s\n' \
        "$TASK_IDS" "$EPISODES" "$SEED_BASE" "$SUITE_MODES"
      printf 'entrypoint=%s\nentrypoint_sha256=%s\nrequirements=%s\nrequirements_sha256=%s\n' \
        "$ENTRYPOINT" "${ENTRYPOINT_SHA256:-}" "$REQUIREMENTS_FILE" "${REQUIREMENTS_SHA256:-}"
      printf 'canonical_config=%s\ncanonical_config_sha256=%s\npolicy_lock=%s\npolicy_lock_sha256=%s\n' \
        "$CANONICAL_CONFIG" "$CANONICAL_CONFIG_SHA256" "$POLICY_LOCK" "${POLICY_LOCK_SHA256:-}"
      printf 'run_root=%s\noutput_root=%s\narchive_root=%s\nruntime_root=%s\n' \
        "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT" "$RUNTIME_ROOT"
      printf 'model_id=allenai/MolmoPoint-8B\nmodel_revision=188130f961c8e0888a34e11121a1423c461a01ba\n'
      printf 'prompt_id=rim_clearance\nregion_backend=rgbd\ncamera=agentview\nsam3_used=false\n'
      printf 'opening_profile=preshape40mm\nmotion_profile=release20_retreat80mm\nobservation_profile=parked\n'
    } > "$ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
  fi
  if [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]]; then workload_rc=$archive_rc; fi
  exit "$workload_rc"
}
trap archive_on_exit EXIT

ENTRYPOINT_SHA256="$(sha256sum -- "$ENTRYPOINT" | awk '{print tolower($1)}')" \
  || die 'cannot hash canonical entrypoint'
REQUIREMENTS_SHA256="$(sha256sum -- "$REQUIREMENTS_FILE" | awk '{print tolower($1)}')" \
  || die 'cannot hash requirements'
POLICY_LOCK_SHA256="$(sha256sum -- "$POLICY_LOCK" | awk '{print tolower($1)}')" \
  || die 'cannot hash policy lock'

mkdir -p "$RUN_ROOT/tmp" "$RUNTIME_ROOT/cache/huggingface" "$RUNTIME_ROOT/cache/xdg"
cat > "$RUN_ROOT/job_context.env" <<EOF
schema=canonical_grasp_controller_job.v1
job_id=$SLURM_JOB_ID
job_name=$SLURM_JOB_NAME
expected_commit=$GRASP_CONTROLLER_EXPECTED_COMMIT
actual_commit=$actual_commit
repo_root=$REPO_ROOT
label=$LABEL
task_ids=$TASK_IDS
episodes_per_task=$EPISODES
seed_base=$SEED_BASE
suite_modes=$SUITE_MODES
entrypoint=$ENTRYPOINT
entrypoint_sha256=$ENTRYPOINT_SHA256
requirements_file=$REQUIREMENTS_FILE
requirements_sha256=$REQUIREMENTS_SHA256
canonical_config=$CANONICAL_CONFIG
canonical_config_sha256=$CANONICAL_CONFIG_SHA256
policy_lock=$POLICY_LOCK
policy_lock_sha256=$POLICY_LOCK_SHA256
run_root=$RUN_ROOT
output_root=$OUTPUT_ROOT
archive_root=$ARCHIVE_ROOT
runtime_root=$RUNTIME_ROOT
model_id=allenai/MolmoPoint-8B
model_revision=188130f961c8e0888a34e11121a1423c461a01ba
prompt_id=rim_clearance
region_backend=rgbd
camera=agentview
sam3_used=false
opening_profile=preshape40mm
motion_profile=release20_retreat80mm
observation_profile=parked
EOF

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
[[ -x "$BASE_PYTHON" ]] || die "validated base Python is missing: $BASE_PYTHON"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/EmbodimentSemantic_runtime/config}"
export LIBERO_CONFIG="$LIBERO_CONFIG_PATH/config.yaml"
[[ -d "$LIBERO_CONFIG_PATH" && -f "$LIBERO_CONFIG" ]] || die 'LIBERO_CONFIG_PATH/config.yaml is missing'
export PYTHONPATH="$REPO_ROOT/vla_benchmarking"
export TMPDIR="$RUN_ROOT/tmp"
export XDG_CACHE_HOME="$RUNTIME_ROOT/cache/xdg"
export HF_HOME="$RUNTIME_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

VENV="$RUNTIME_ROOT/venv-py312-${REQUIREMENTS_SHA256:0:16}"
VENV_LOCK="$RUNTIME_ROOT/venv-py312-${REQUIREMENTS_SHA256:0:16}.lock"
exec 9>"$VENV_LOCK"
flock 9
if [[ -e "$VENV" ]]; then
  [[ -x "$VENV/bin/python" ]] || die "existing runtime venv is incomplete: $VENV"
else
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV" \
    || die 'could not create canonical runtime venv'
fi
PYTHON="$VENV/bin/python"
[[ -x "$PYTHON" ]] || die "runtime Python is missing: $PYTHON"
"$PYTHON" -c 'import sys; assert sys.prefix != sys.base_prefix'
"$PYTHON" -m pip install --disable-pip-version-check --no-input --no-deps \
  -r "$REQUIREMENTS_FILE" || die 'canonical Molmo dependency installation failed'
flock -u 9

"$PYTHON" - "$RUN_ROOT" <<'PY'
import importlib.metadata as md
import json
import pathlib
import sys
import torch

root = pathlib.Path(sys.argv[1])
required = {
    "torch": "2.10.0", "transformers": "4.57.1", "huggingface-hub": "0.36.0",
    "tokenizers": "0.22.1", "einops": "0.8.1", "decord2": "3.4.0",
}
observed = {}
for name in required:
    observed[name] = md.version(name)
for name, expected in required.items():
    if observed[name] != expected:
        raise SystemExit(f"{name} version mismatch: {observed[name]!r} != {expected!r}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one visible CUDA device")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("canonical Molmo runtime requires BF16 support")
import libero  # noqa: F401
import mujoco  # noqa: F401
import robosuite  # noqa: F401
runtime = {
    "python": sys.executable, "python_version": sys.version,
    "packages": observed, "cuda_device_count": int(torch.cuda.device_count()),
    "cuda_device_name": torch.cuda.get_device_name(0), "bf16_supported": True,
    "dtype": "bfloat16", "model_id": "allenai/MolmoPoint-8B",
    "model_revision": "188130f961c8e0888a34e11121a1423c461a01ba",
}
(root / "runtime_versions.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(runtime, sort_keys=True))
PY

# The thin entrypoint owns the frozen treatment. Only scope arguments cross
# this boundary, so changing treatment behavior requires a new reviewed code
# commit instead of an ambient scheduler variable.
set +e
"$PYTHON" "$ENTRYPOINT" \
  --task-ids "$TASK_IDS" \
  --episodes-per-task "$EPISODES" \
  --seed-base "$SEED_BASE" \
  --suite-modes "$SUITE_MODES" \
  --output-dir "$OUTPUT_ROOT" \
  --label "$LABEL"
workload_rc=$?
set -e
exit "$workload_rc"
