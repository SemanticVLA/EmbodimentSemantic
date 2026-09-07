#!/usr/bin/env bash
# Automatic TTT task-0 runner for SmolVLA on Legion.
#
# One job executes: baseline (10 sealed episodes), dynamic Arrow collection
# (100 accepted corrections with a bounded attempt budget), task-0-only
# training from the original checkpoint, and adapted evaluation on the same
# ten sealed seeds. Failed Arrow attempts remain in the run directory but
# are never admitted as training demonstrations.
#
# Required environment: REPO_ROOT, AUTOMATIC_TTT_EXPECTED_COMMIT,
# AUTOMATIC_TTT_LABEL, AUTOMATIC_TTT_BASE_POLICY, AUTOMATIC_TTT_LIBERO_DIR,
# AUTOMATIC_TTT_LIBERO_CONFIG. The reviewed Python runner must be supplied by
# the ML/runtime integration at automatic_ttt/task0_dynamic.py.

#SBATCH --job-name=automatic_ttt_smolvla_t0
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

die() { printf 'automatic_ttt task-0: %s\n' "$*" >&2; exit 2; }

[[ -n "${SLURM_JOB_ID:-}" && "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] ||
  die 'submit this launcher with sbatch; direct execution is refused'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
[[ "${AUTOMATIC_TTT_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] ||
  die 'AUTOMATIC_TTT_EXPECTED_COMMIT must be a lowercase 40-character SHA'
LABEL="${AUTOMATIC_TTT_LABEL:-}"
[[ "$LABEL" =~ ^[A-Za-z0-9_.-]{1,96}$ ]] || die 'AUTOMATIC_TTT_LABEL is required and must be safe'

command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'
command -v git >/dev/null 2>&1 || die 'git is required'
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")" || die 'cannot resolve REPO_ROOT'
actual_root="$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null)" || die 'REPO_ROOT is not a Git checkout'
actual_root="$(realpath -e -- "$actual_root")" || die 'cannot resolve Git root'
[[ "$actual_root" == "$REPO_ROOT" ]] || die 'REPO_ROOT is not the checkout root'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] || die 'shared HOME checkout is forbidden'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)" || die 'cannot read REPO_ROOT commit'
[[ "$actual_commit" == "$AUTOMATIC_TTT_EXPECTED_COMMIT" ]] || die "checkout SHA differs: $actual_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'REPO_ROOT is dirty'

TASK_ID=0
BASELINE_EPISODES="${AUTOMATIC_TTT_BASELINE_EPISODES:-10}"
BASELINE_SEEDS="${AUTOMATIC_TTT_BASELINE_SEEDS:-1000,1001,1002,1003,1004,1005,1006,1007,1008,1009}"
ADAPTATION_SEED_BASE="${AUTOMATIC_TTT_ADAPTATION_SEED_BASE:-3000}"
ACCEPTED_TARGET="${AUTOMATIC_TTT_ACCEPTED_TARGET:-100}"
MAX_ATTEMPTS="${AUTOMATIC_TTT_MAX_ATTEMPTS:-1000}"
[[ "$BASELINE_EPISODES" == 10 ]] || die 'baseline is sealed to exactly 10 episodes'
[[ "$BASELINE_SEEDS" == 1000,1001,1002,1003,1004,1005,1006,1007,1008,1009 ]] || die 'baseline seeds are sealed to 1000..1009'
[[ "$ADAPTATION_SEED_BASE" =~ ^[0-9]+$ ]] || die 'adaptation seed base must be numeric'
[[ "$ACCEPTED_TARGET" == 100 ]] || die 'accepted Arrow target is sealed to 100'
[[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || die 'max attempts must be positive'
(( MAX_ATTEMPTS >= ACCEPTED_TARGET )) || die 'max attempts must cover accepted target'

PYTHON="${AUTOMATIC_TTT_PYTHON:-/home/hjaber/.conda/envs/embodiment-smolvla-py312/bin/python}"
BASE_POLICY="${AUTOMATIC_TTT_BASE_POLICY:-}"
LIBERO_DIR="${AUTOMATIC_TTT_LIBERO_DIR:-}"
LIBERO_CONFIG="${AUTOMATIC_TTT_LIBERO_CONFIG:-}"
ARROW_CONFIG="${AUTOMATIC_TTT_ARROW_CONFIG:-$REPO_ROOT/vla_benchmarking/libero/arrow_grasp_controller/configs/canonical_molmo_rgbd_grasp.json}"
RUNNER="$REPO_ROOT/vla_benchmarking/libero/automatic_ttt/task0_dynamic.py"
SCRATCH_ROOT="${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/automatic_ttt_smolvla_t0"
RUN_ROOT="${AUTOMATIC_TTT_RUN_ROOT:-$SCRATCH_ROOT/runs/${LABEL}_${SLURM_JOB_ID}}"
ARCHIVE_ROOT="${AUTOMATIC_TTT_ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/automatic_ttt_smolvla_t0/${LABEL}_${SLURM_JOB_ID}}"

for path_name in PYTHON BASE_POLICY LIBERO_DIR LIBERO_CONFIG ARROW_CONFIG RUNNER RUN_ROOT ARCHIVE_ROOT; do
  path="${!path_name}"
  [[ "$path" = /* && "$path" != / ]] || die "$path_name must be an absolute non-root path: $path"
done
[[ -x "$PYTHON" ]] || die "Python executable is missing: $PYTHON"
[[ -f "$RUNNER" ]] || die "reviewed dynamic runner is missing: $RUNNER"
[[ -d "$BASE_POLICY" && -f "$BASE_POLICY/config.json" && -f "$BASE_POLICY/model.safetensors" ]] || die 'SmolVLA base policy is missing/incomplete'
[[ -d "$LIBERO_DIR/.git" ]] || die "LIBERO checkout is missing: $LIBERO_DIR"
[[ -f "$LIBERO_CONFIG" ]] || die "LIBERO config.yaml is missing: $LIBERO_CONFIG"
[[ -f "$ARROW_CONFIG" ]] || die "Arrow controller config is missing: $ARROW_CONFIG"
LIBERO_COMMIT="${AUTOMATIC_TTT_LIBERO_COMMIT:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
[[ "$LIBERO_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'LIBERO commit must be a lowercase 40-character SHA'
[[ "$(git -C "$LIBERO_DIR" rev-parse HEAD 2>/dev/null)" == "$LIBERO_COMMIT" ]] || die 'LIBERO checkout is not pinned'
[[ -z "$(git -C "$LIBERO_DIR" status --porcelain --untracked-files=all)" ]] || die 'LIBERO checkout is dirty'

canonical_external_root() {
  local name="$1" value="$2" canonical
  [[ ! -L "$value" ]] || die "$name must not be a symlink: $value"
  canonical="$(realpath -m -- "$value")" || die "cannot canonicalize $name"
  [[ "$canonical" != / ]] || die "$name resolves to filesystem root"
  case "$canonical" in "$REPO_ROOT"|"$REPO_ROOT"/*) die "$name must be outside REPO_ROOT" ;; esac
  printf '%s\n' "$canonical"
}
RUN_ROOT="$(canonical_external_root AUTOMATIC_TTT_RUN_ROOT "$RUN_ROOT")"
ARCHIVE_ROOT="$(canonical_external_root AUTOMATIC_TTT_ARCHIVE_ROOT "$ARCHIVE_ROOT")"
case "$RUN_ROOT" in "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'RUN_ROOT overlaps ARCHIVE_ROOT' ;; esac
case "$ARCHIVE_ROOT" in "$RUN_ROOT"|"$RUN_ROOT"/*) die 'ARCHIVE_ROOT overlaps RUN_ROOT' ;; esac
[[ ! -e "$RUN_ROOT" ]] || die "RUN_ROOT already exists: $RUN_ROOT"
[[ ! -e "$ARCHIVE_ROOT" ]] || die "ARCHIVE_ROOT already exists: $ARCHIVE_ROOT"
mkdir -p "$RUN_ROOT"

sha256() { sha256sum -- "$1" | awk '{print tolower($1)}'; }
RUNNER_SHA256="$(sha256 "$RUNNER")"
BASE_CONFIG_SHA256="$(sha256 "$BASE_POLICY/config.json")"
BASE_MODEL_SHA256="$(sha256 "$BASE_POLICY/model.safetensors")"
LIBERO_CONFIG_SHA256="$(sha256 "$LIBERO_CONFIG")"
ARROW_CONFIG_SHA256="$(sha256 "$ARROW_CONFIG")"

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$(dirname "$LIBERO_CONFIG")"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR="$RUN_ROOT/tmp" XDG_CACHE_HOME="$RUN_ROOT/xdg"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

"$PYTHON" - "$RUN_ROOT" <<'PY'
import importlib.metadata as md
import json
import pathlib
import sys
import torch
root = pathlib.Path(sys.argv[1])
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('expected exactly one visible CUDA device')
packages = {name: md.version(name) for name in ('lerobot', 'torch', 'transformers', 'peft', 'robosuite', 'h5py')}
(root / 'runtime_versions.json').write_text(json.dumps({
    'python': sys.executable, 'python_version': sys.version,
    'packages': packages, 'cuda_device_name': torch.cuda.get_device_name(0),
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

cat > "$RUN_ROOT/job_context.env" <<EOF
schema=automatic_ttt_smolvla_task0_dynamic_job.v1
job_id=$SLURM_JOB_ID
label=$LABEL
task_id=$TASK_ID
expected_commit=$AUTOMATIC_TTT_EXPECTED_COMMIT
actual_commit=$actual_commit
libero_commit=$LIBERO_COMMIT
python=$PYTHON
base_policy=$BASE_POLICY
base_config_sha256=$BASE_CONFIG_SHA256
base_model_sha256=$BASE_MODEL_SHA256
libero_dir=$LIBERO_DIR
libero_config=$LIBERO_CONFIG
libero_config_sha256=$LIBERO_CONFIG_SHA256
arrow_config=$ARROW_CONFIG
arrow_config_sha256=$ARROW_CONFIG_SHA256
runner=$RUNNER
runner_sha256=$RUNNER_SHA256
baseline_episodes=$BASELINE_EPISODES
baseline_seeds=$BASELINE_SEEDS
adaptation_seed_base=$ADAPTATION_SEED_BASE
accepted_arrow_target=$ACCEPTED_TARGET
max_attempts=$MAX_ATTEMPTS
task_mode=task_specific
run_root=$RUN_ROOT
archive_root=$ARCHIVE_ROOT
EOF

archive_on_exit() {
  local workload_rc=$? archive_rc=0 status=PRESERVED_FAILURE
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT" || archive_rc=$?
  if command -v rsync >/dev/null 2>&1; then
    rsync -a -- "$RUN_ROOT/" "$ARCHIVE_ROOT/run/" || archive_rc=$?
  else
    mkdir -p "$ARCHIVE_ROOT/run"
    cp -a -- "$RUN_ROOT/." "$ARCHIVE_ROOT/run/" || archive_rc=$?
  fi
  if [[ "$workload_rc" -eq 0 && "$archive_rc" -eq 0 && -s "$RUN_ROOT/adapted_eval/summary.json" && -s "$RUN_ROOT/run_manifest.json" ]]; then
    status=VERIFIED
  fi
  [[ "$archive_rc" -eq 0 ]] || status=ARCHIVE_FAILURE
  printf 'schema=automatic_ttt_smolvla_task0_dynamic_archive.v1\narchive_status=%s\njob_exit_code=%s\njob_id=%s\nexpected_commit=%s\nactual_commit=%s\nrun_root=%s\narchive_root=%s\n' "$status" "$workload_rc" "$SLURM_JOB_ID" "$AUTOMATIC_TTT_EXPECTED_COMMIT" "$actual_commit" "$RUN_ROOT" "$ARCHIVE_ROOT" > "$ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
  if [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]]; then workload_rc=$archive_rc; fi
  exit "$workload_rc"
}
trap archive_on_exit EXIT

"$PYTHON" "$RUNNER" \
  --task-id "$TASK_ID" --base-policy "$BASE_POLICY" --arrow-config "$ARROW_CONFIG" \
  --libero-dir "$LIBERO_DIR" --libero-config "$LIBERO_CONFIG" \
  --baseline-seeds "$BASELINE_SEEDS" --adaptation-seed-base "$ADAPTATION_SEED_BASE" \
  --baseline-episodes "$BASELINE_EPISODES" --accepted-arrow-target "$ACCEPTED_TARGET" \
  --max-attempts "$MAX_ATTEMPTS" --task-specific --output-root "$RUN_ROOT"

for required in "$RUN_ROOT/run_manifest.json" "$RUN_ROOT/baseline/summary.json" "$RUN_ROOT/collection/accepted_manifest.json" "$RUN_ROOT/training/training_manifest.json" "$RUN_ROOT/adapted_eval/summary.json"; do
  [[ -s "$required" ]] || die "dynamic runner did not emit required artifact: $required"
done

"$PYTHON" - "$RUN_ROOT/collection/accepted_manifest.json" "$ACCEPTED_TARGET" <<'PY'
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
target = int(sys.argv[2])
if type(payload.get('accepted_count')) is not int or payload['accepted_count'] != target:
    raise SystemExit(f"accepted Arrow count {payload.get('accepted_count')!r} does not equal target {target}")
if payload.get('task_id') != 0 or payload.get('task_mode') != 'task_specific':
    raise SystemExit('accepted manifest is not task-0-specific')
PY

touch "$RUN_ROOT/COMPLETED"

