#!/usr/bin/env bash
# This launcher must retain LF endings for Bash/SLURM portability.
# One isolated RoboCasa target-split evaluation on one A40.
#
# ROBOCASA_TASKS is either `all` or a comma-separated list of registered task
# names.  It is split into argv entries below; it is never evaluated as shell.

#SBATCH --job-name=robocasa_arrow21
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.err

set -Eeuo pipefail
umask 027
export PYTHONDONTWRITEBYTECODE=1

die() {
  printf 'robocasa pick-place: %s\n' "$*" >&2
  exit 2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit with sbatch; direct execution is refused'
[[ "${ROBOCASA_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || die 'ROBOCASA_EXPECTED_COMMIT must be a lowercase 40-character SHA'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
command -v flock >/dev/null 2>&1 || die 'flock is required'

LABEL="${ROBOCASA_RUN_LABEL:-robocasa_pickplace21_21x1}"
TASKS="${ROBOCASA_TASKS:-all}"
EPISODES="${ROBOCASA_EPISODES_PER_TASK:-1}"
SEED_BASE="${ROBOCASA_SEED_BASE:-1000}"
SETUP_RUNTIME="${ROBOCASA_SETUP_RUNTIME:-0}"
MODE="${ROBOCASA_MODE:-}"
[[ "$LABEL" =~ ^[A-Za-z0-9_.-]{1,96}$ ]] || die 'invalid ROBOCASA_RUN_LABEL'
[[ "$TASKS" == all || "$TASKS" =~ ^[A-Za-z][A-Za-z0-9_]*(,[A-Za-z][A-Za-z0-9_]*)*$ ]] \
  || die 'ROBOCASA_TASKS must be all or a comma-separated registered task list'
[[ "$TASKS" == all || ! "$TASKS" =~ (^|,)all(,|$) ]] \
  || die 'ROBOCASA_TASKS must use all by itself, not mixed with task names'
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die 'ROBOCASA_EPISODES_PER_TASK must be positive'
[[ "$SEED_BASE" =~ ^[0-9]+$ ]] || die 'ROBOCASA_SEED_BASE must be non-negative'
[[ "$SETUP_RUNTIME" == 0 || "$SETUP_RUNTIME" == 1 ]] \
  || die 'ROBOCASA_SETUP_RUNTIME must be 0 (reuse) or 1 (bootstrap)'
[[ -z "$MODE" || "$MODE" == smoke || "$MODE" == full ]] \
  || die 'ROBOCASA_MODE must be smoke or full'
if [[ -z "$MODE" ]]; then
  if [[ "$TASKS" == all ]]; then
    MODE=full
  else
    MODE=smoke
  fi
fi
if [[ "$MODE" == full ]]; then
  [[ "$TASKS" == all ]] || die 'ROBOCASA_MODE=full requires ROBOCASA_TASKS=all'
  [[ "$EPISODES" == 1 ]] || die 'ROBOCASA_MODE=full requires ROBOCASA_EPISODES_PER_TASK=1'
  [[ "$SEED_BASE" == 1000 ]] || die 'ROBOCASA_MODE=full requires ROBOCASA_SEED_BASE=1000'
fi

TASK_ARGS=()
if [[ "$TASKS" == all ]]; then
  TASK_ARGS=(all)
else
  IFS=',' read -r -a task_names <<< "$TASKS"
  for task_name in "${task_names[@]}"; do
    TASK_ARGS+=("$task_name")
  done
fi

command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")" || die 'cannot resolve REPO_ROOT'
actual_root="$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null)" \
  || die 'REPO_ROOT is not a Git checkout'
actual_root="$(realpath -e -- "$actual_root")" || die 'cannot resolve Git root'
[[ "$actual_root" == "$REPO_ROOT" ]] || die 'REPO_ROOT is not the worktree root'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] \
  || die 'shared HOME checkout is forbidden; use an isolated immutable checkout'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)" \
  || die 'cannot read REPO_ROOT commit'
[[ "$actual_commit" == "$ROBOCASA_EXPECTED_COMMIT" ]] \
  || die "checkout SHA differs: $actual_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] \
  || die 'REPO_ROOT is dirty'

RUN_ROOT="${ROBOCASA_RUN_ROOT:-${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/robocasa/runs/${LABEL}_${SLURM_JOB_ID}}"
OUTPUT_ROOT="${ROBOCASA_OUTPUT_ROOT:-$RUN_ROOT/results}"
ARCHIVE_ROOT="${ROBOCASA_ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/robocasa/${LABEL}_${SLURM_JOB_ID}}"
ENV_ROOT="${ROBOCASA_ENV_ROOT:-$HOME/.conda/envs/robocasa-pickplace-py311}"
RUNTIME_ROOT="${ROBOCASA_RUNTIME_ROOT:-$HOME/EmbodimentSemantic_runtime/robocasa}"
LOCK_PATH="${ROBOCASA_LOCK_PATH:-$HOME/EmbodimentSemantic_runtime/operator/robocasa_pickplace.lock}"
for path in "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT" "$ENV_ROOT" "$RUNTIME_ROOT" "$LOCK_PATH"; do
  [[ "$path" = /* && "$path" != / ]] || die "path must be an absolute non-root path: $path"
done
[[ ! -e "$RUN_ROOT" ]] || die "RUN_ROOT already exists: $RUN_ROOT"
[[ ! -e "$ARCHIVE_ROOT" ]] || die "ARCHIVE_ROOT already exists: $ARCHIVE_ROOT"
mkdir -p "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT" "$RUNTIME_ROOT" "$(dirname "$LOCK_PATH")"
exec 9>"$LOCK_PATH"
flock -n 9 || die 'another RoboCasa evaluation job is already running'

REQUIREMENTS_FILE="$REPO_ROOT/vla_benchmarking/robocasa/requirements.txt"
ENTRYPOINT_MODULE="vla_benchmarking.robocasa.arrow_grasp_controller.controller.entrypoint"
ENTRYPOINT_FILE="$REPO_ROOT/vla_benchmarking/robocasa/arrow_grasp_controller/controller/entrypoint.py"
CONTROLLER_ROOT="$REPO_ROOT/vla_benchmarking/robocasa/arrow_grasp_controller"
CONTROLLER_CONFIG="$CONTROLLER_ROOT/configs/canonical_molmo_rgbd_grasp.json"
POLICY_LOCK="$CONTROLLER_ROOT/configs/active_policy.lock.json"
MODEL_ID="allenai/MolmoPoint-8B"
MODEL_REVISION="188130f961c8e0888a34e11121a1423c461a01ba"
ROBOSUITE_REVISION="5ce6643f3092639d08f7b0f90ed1c6a84f50552c"
ROBOCASA_REVISION="4f8a2980def75a55dff96b990745b83540425f09"
PROMPT_ID="rim_clearance"
GRASP_PROFILE="${ROBOCASA_GRASP_PROFILE:-canonical_rim}"
case "$GRASP_PROFILE" in
  canonical_rim) GRASP_CONFIG="$CONTROLLER_CONFIG" ;;
  object_contact_v1) GRASP_CONFIG="$CONTROLLER_ROOT/configs/object_contact_v1.json" ;;
  object_contact_v2) GRASP_CONFIG="$CONTROLLER_ROOT/configs/object_contact_v2.json" ;;
  object_contact_v3) GRASP_CONFIG="$CONTROLLER_ROOT/configs/object_contact_v3.json" ;;
  object_contact_v4) GRASP_CONFIG="$CONTROLLER_ROOT/configs/object_contact_v4.json" ;;
  object_contact_v5) GRASP_CONFIG="$CONTROLLER_ROOT/configs/object_contact_v5.json" ;;
  *) die 'ROBOCASA_GRASP_PROFILE must be canonical_rim, object_contact_v1, object_contact_v2, object_contact_v3, object_contact_v4, or object_contact_v5' ;;
esac
[[ -f "$REQUIREMENTS_FILE" ]] || die 'RoboCasa requirements file is missing'
[[ -f "$ENTRYPOINT_FILE" ]] || die 'RoboCasa runner is missing'
[[ -f "$GRASP_CONFIG" ]] || die "RoboCasa grasp profile config is missing: $GRASP_CONFIG"
[[ -f "$POLICY_LOCK" ]] || die 'RoboCasa policy lock is missing'

ENTRYPOINT_SHA256=''
REQUIREMENTS_SHA256=''
CONTROLLER_SOURCE_SHA256=''
CONTROLLER_CONFIG_SHA256=''
POLICY_LOCK_SHA256=''
GRASP_CONFIG_SHA256=''
RUNTIME_PROBE_PATH="$RUN_ROOT/runtime_probe.json"

controller_source_sha256() {
  find "$CONTROLLER_ROOT" -type f \
    ! -path '*/legion/*' ! -path '*/__pycache__/*' -print0 \
    | sort -z \
    | while IFS= read -r -d '' source_path; do
        printf '%s  %s\n' \
          "$(sha256sum -- "$source_path" | awk '{print tolower($1)}')" \
          "${source_path#"$REPO_ROOT/"}"
      done \
    | sha256sum | awk '{print tolower($1)}'
}

archive_on_exit() {
  rc=$?
  archive_rc=0
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$RUN_ROOT/" "$ARCHIVE_ROOT/run/" || archive_rc=$?
  else
    cp -a "$RUN_ROOT/." "$ARCHIVE_ROOT/run/" || archive_rc=$?
  fi
  for suffix in out err; do
    log_path="${SLURM_LOG_ROOT:-$HOME/EmbodimentSemantic_runtime/operator/logs}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${suffix}"
    [[ -f "$log_path" ]] && cp -p "$log_path" "$ARCHIVE_ROOT/" || true
  done
  {
    printf 'schema=robocasa_pickplace21_job.v2\n'
    printf 'job_id=%s\njob_name=%s\nlabel=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NAME" "$LABEL"
    printf 'expected_commit=%s\nactual_commit=%s\nrepo_root=%s\n' "$ROBOCASA_EXPECTED_COMMIT" "$actual_commit" "$REPO_ROOT"
    printf 'tasks=%s\nepisodes_per_task=%s\nseed_base=%s\nsplit=target\nmode=%s\n' "$TASKS" "$EPISODES" "$SEED_BASE" "$MODE"
    printf 'run_root=%s\noutput_root=%s\narchive_root=%s\nenv_root=%s\nruntime_root=%s\nruntime_setup=%s\nruntime_probe=%s\n' \
      "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT" "$ENV_ROOT" "$RUNTIME_ROOT" "$SETUP_RUNTIME" "$RUNTIME_PROBE_PATH"
    printf 'runtime_versions=%s\n' "$RUN_ROOT/runtime_versions.json"
    printf 'requirements_sha256=%s\nentrypoint_sha256=%s\ncontroller_source_sha256=%s\n' \
      "$REQUIREMENTS_SHA256" "$ENTRYPOINT_SHA256" "$CONTROLLER_SOURCE_SHA256"
    printf 'robosuite_revision=%s\nrobocasa_revision=%s\n' "$ROBOSUITE_REVISION" "$ROBOCASA_REVISION"
    printf 'grasp_profile=%s\ngrasp_config=%s\ngrasp_config_sha256=%s\n' "$GRASP_PROFILE" "$GRASP_CONFIG" "$GRASP_CONFIG_SHA256"
    printf 'controller_config=%s\ncontroller_config_sha256=%s\npolicy_lock=%s\npolicy_lock_sha256=%s\n' \
      "$CONTROLLER_CONFIG" "$CONTROLLER_CONFIG_SHA256" "$POLICY_LOCK" "$POLICY_LOCK_SHA256"
    printf 'model_id=%s\nmodel_revision=%s\nprompt_id=%s\nregion_backend=rgbd\ncamera=agentview\n' \
      "$MODEL_ID" "$MODEL_REVISION" "$PROMPT_ID"
    printf 'hf_home=%s\ntransformers_cache=%s\nxdg_cache_home=%s\njob_exit_code=%s\n' \
      "${HF_HOME:-}" "${TRANSFORMERS_CACHE:-}" "${XDG_CACHE_HOME:-}" "$rc"
  } > "$ARCHIVE_ROOT/run_manifest.env" || archive_rc=$?
  if [[ "$rc" -eq 0 && "$archive_rc" -ne 0 ]]; then
    rc="$archive_rc"
  fi
  exit "$rc"
}
trap archive_on_exit EXIT

ENTRYPOINT_SHA256="$(sha256sum -- "$ENTRYPOINT_FILE" | awk '{print tolower($1)}')"
REQUIREMENTS_SHA256="$(sha256sum -- "$REQUIREMENTS_FILE" | awk '{print tolower($1)}')"
CONTROLLER_SOURCE_SHA256="$(controller_source_sha256)"
CONTROLLER_CONFIG_SHA256="$(sha256sum -- "$CONTROLLER_CONFIG" | awk '{print tolower($1)}')"
POLICY_LOCK_SHA256="$(sha256sum -- "$POLICY_LOCK" | awk '{print tolower($1)}')"
GRASP_CONFIG_SHA256="$(sha256sum -- "$GRASP_CONFIG" | awk '{print tolower($1)}')"

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "$SETUP_RUNTIME" == 1 ]]; then
  if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
    conda create -y -p "$ENV_ROOT" python=3.11 pip
  fi
else
  [[ -x "$ENV_ROOT/bin/python" ]] \
    || die "reusable RoboCasa runtime is missing: $ENV_ROOT (set ROBOCASA_SETUP_RUNTIME=1 for bootstrap)"
fi
PYTHON="$ENV_ROOT/bin/python"
if [[ "$SETUP_RUNTIME" == 1 ]]; then
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install --requirement "$REQUIREMENTS_FILE"
  # RoboCasa and robosuite declare opencv-python while LeRobot declares
  # opencv-python-headless.  The cluster runtime is headless; leave exactly one
  # provider installed after dependency resolution so import order cannot change
  # the pixels supplied to the controller.
  "$PYTHON" -m pip uninstall --yes opencv-python opencv-contrib-python opencv-contrib-python-headless \
    >/dev/null 2>&1 || true
  "$PYTHON" -m pip install --no-deps --force-reinstall "opencv-python-headless==4.11.0.86"
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TMPDIR="$RUN_ROOT/tmp"
export XDG_CACHE_HOME="$RUNTIME_ROOT/cache/xdg"
export HF_HOME="$RUNTIME_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="${ROBOCASA_TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
[[ "$TRANSFORMERS_CACHE" = /* && "$TRANSFORMERS_CACHE" != / ]] \
  || die 'ROBOCASA_TRANSFORMERS_CACHE must be an absolute non-root path'
export HF_HUB_DISABLE_TELEMETRY=1
export RUN_ROOT MODEL_ID MODEL_REVISION PROMPT_ID SETUP_RUNTIME REQUIREMENTS_FILE
export ROBOSUITE_REVISION ROBOCASA_REVISION
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import json
import os
import pathlib
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"RoboCasa runtime requires Python 3.11, got {sys.version_info.major}.{sys.version_info.minor}")
required = {
    "numpy": "2.2.5",
    "numba": "0.61.2",
    "scipy": "1.15.3",
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "transformers": "4.57.1",
    "huggingface-hub": "0.36.0",
    "tokenizers": "0.22.1",
    "einops": "0.8.1",
    "decord2": "3.4.0",
    "accelerate": "1.10.1",
    "safetensors": "0.8.0",
    "opencv-python-headless": "4.11.0.86",
    "mujoco": "3.3.1",
    "lerobot": "0.3.3",
    "robosuite": "1.5.2",
    "robocasa": "1.0.1",
}
source_revisions = {
    "robosuite": os.environ["ROBOSUITE_REVISION"],
    "robocasa": os.environ["ROBOCASA_REVISION"],
}
requirements_text = pathlib.Path(os.environ["REQUIREMENTS_FILE"]).read_text(encoding="utf-8")
for package, revision in source_revisions.items():
    if revision not in requirements_text:
        raise SystemExit(f"{package} source revision is absent from requirements.txt: {revision}")
for package, expected in required.items():
    actual = metadata.version(package)
    if actual != expected:
        raise SystemExit(f"{package} version mismatch: {actual!r} != {expected!r}")
for package in ("opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"):
    try:
        metadata.version(package)
    except metadata.PackageNotFoundError:
        continue
    raise SystemExit(f"conflicting OpenCV distribution is installed: {package}")
for package, expected_revision in source_revisions.items():
    direct_url = metadata.distribution(package).read_text("direct_url.json")
    if not direct_url:
        raise SystemExit(f"{package} has no direct_url.json; expected source revision {expected_revision}")
    try:
        direct_url_data = json.loads(direct_url)
        vcs_info = direct_url_data["vcs_info"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{package} direct_url.json is not a VCS install: {exc}") from exc
    if (
        vcs_info.get("vcs") != "git"
        or vcs_info.get("requested_revision") != expected_revision
        or vcs_info.get("commit_id") != expected_revision
    ):
        raise SystemExit(
            f"{package} source revision mismatch: requested={vcs_info.get('requested_revision')!r}, "
            f"commit={vcs_info.get('commit_id')!r}, expected={expected_revision!r}"
        )
import cv2, mujoco, robocasa, robosuite
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one visible CUDA device")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("MolmoPoint runtime requires BF16 support")
observed = {package: metadata.version(package) for package in required}
runtime = {
    "python": sys.executable,
    "python_version": sys.version,
    "packages": observed,
    "torch": metadata.version("torch"),
    "cuda_device_count": int(torch.cuda.device_count()),
    "cuda_device_name": torch.cuda.get_device_name(0),
    "bf16_supported": True,
    "dtype": "bfloat16",
    "runtime_setup": os.environ["SETUP_RUNTIME"],
    "source_revisions": source_revisions,
    "model_id": os.environ["MODEL_ID"],
    "model_revision": os.environ["MODEL_REVISION"],
    "prompt_id": os.environ["PROMPT_ID"],
    "hf_home": os.environ["HF_HOME"],
    "transformers_cache": os.environ["TRANSFORMERS_CACHE"],
    "xdg_cache_home": os.environ["XDG_CACHE_HOME"],
}
pathlib.Path(os.environ["RUN_ROOT"]).joinpath("runtime_versions.json").write_text(
    json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("RoboCasa runtime imports OK")
print("robocasa", getattr(robocasa, "__version__", "unknown"))
print("robosuite", getattr(robosuite, "__version__", "unknown"))
print("mujoco", getattr(mujoco, "__version__", "unknown"))
print("opencv", cv2.__version__)
print("transformers", metadata.version("transformers"))
print("model", "allenai/MolmoPoint-8B", "revision", "188130f961c8e0888a34e11121a1423c461a01ba")
PY

ASSET_MARKER="$RUNTIME_ROOT/assets_ready_robocasa_${ROBOCASA_REVISION}"
assets_marker_matches() {
  [[ -f "$ASSET_MARKER" ]] && grep -Fxq "robocasa_revision=$ROBOCASA_REVISION" "$ASSET_MARKER"
}
verify_assets() {
  "$PYTHON" - <<'PY'
import robocasa
from pathlib import Path
root = Path(robocasa.__path__[0]) / "models" / "assets"
required = [root / "textures", root / "fixtures", root / "objects"]
if not all(path.is_dir() and any(path.iterdir()) for path in required):
    raise SystemExit(f"RoboCasa assets are incomplete under {root}")
print("RoboCasa assets ready", root)
PY
}
if [[ "$SETUP_RUNTIME" == 0 ]]; then
  verify_assets
  if ! assets_marker_matches; then
    mkdir -p "$(dirname "$ASSET_MARKER")"
    printf 'robocasa_revision=%s\nsource_commit=%s\n' "$ROBOCASA_REVISION" "$ROBOCASA_EXPECTED_COMMIT" > "$ASSET_MARKER"
  fi
else
  if ! assets_marker_matches; then
    printf 'y\n' | "$PYTHON" -m robocasa.scripts.setup_macros || true
    printf 'y\n' | "$PYTHON" -m robocasa.scripts.download_kitchen_assets --type all
  fi
  verify_assets
  mkdir -p "$(dirname "$ASSET_MARKER")"
  printf 'robocasa_revision=%s\nsource_commit=%s\n' "$ROBOCASA_REVISION" "$ROBOCASA_EXPECTED_COMMIT" > "$ASSET_MARKER"
fi

cat > "$RUN_ROOT/job_context.env" <<EOF
schema=robocasa_pickplace21_job.v2
job_id=$SLURM_JOB_ID
expected_commit=$ROBOCASA_EXPECTED_COMMIT
actual_commit=$actual_commit
repo_root=$REPO_ROOT
label=$LABEL
tasks=$TASKS
episodes_per_task=$EPISODES
seed_base=$SEED_BASE
split=target
mode=$MODE
runtime_setup=$SETUP_RUNTIME
runtime_probe=$RUNTIME_PROBE_PATH
run_root=$RUN_ROOT
output_root=$OUTPUT_ROOT
archive_root=$ARCHIVE_ROOT
env_root=$ENV_ROOT
runtime_root=$RUNTIME_ROOT
runtime_versions=$RUN_ROOT/runtime_versions.json
requirements_sha256=$REQUIREMENTS_SHA256
entrypoint_sha256=$ENTRYPOINT_SHA256
controller_source_sha256=$CONTROLLER_SOURCE_SHA256
grasp_profile=$GRASP_PROFILE
grasp_config=$GRASP_CONFIG
grasp_config_sha256=$GRASP_CONFIG_SHA256
robosuite_revision=$ROBOSUITE_REVISION
robocasa_revision=$ROBOCASA_REVISION
controller_config=$CONTROLLER_CONFIG
controller_config_sha256=$CONTROLLER_CONFIG_SHA256
policy_lock=$POLICY_LOCK
policy_lock_sha256=$POLICY_LOCK_SHA256
model_id=$MODEL_ID
model_revision=$MODEL_REVISION
prompt_id=$PROMPT_ID
region_backend=rgbd
camera=agentview
hf_home=$HF_HOME
transformers_cache=$TRANSFORMERS_CACHE
xdg_cache_home=$XDG_CACHE_HOME
EOF

cd "$REPO_ROOT"
if HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PYTHON" -m vla_benchmarking.robocasa.tools.probe_runtime \
    --task PickPlaceCounterToStove \
    --seed 1000 \
    --output "$RUN_ROOT/runtime_probe.json" \
    --execute-motion; then
  :
else
  probe_rc=$?
  die "RoboCasa runtime probe failed with exit code $probe_rc"
fi
[[ -s "$RUNTIME_PROBE_PATH" ]] || die "RoboCasa runtime probe did not write $RUNTIME_PROBE_PATH"

"$PYTHON" -m "$ENTRYPOINT_MODULE" \
  --mode "$MODE" \
  --grasp-profile "$GRASP_PROFILE" \
  --execute-motion \
  --tasks "${TASK_ARGS[@]}" \
  --output-dir "$OUTPUT_ROOT" \
  --episodes-per-task "$EPISODES" \
  --seed-base "$SEED_BASE" \
  --split target
