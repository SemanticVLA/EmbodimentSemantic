#!/usr/bin/env bash
# This launcher must retain LF endings for Bash/SLURM portability.
# One isolated RoboCasa target-split evaluation: 21 tasks x one episode.

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

LABEL="${ROBOCASA_RUN_LABEL:-robocasa_pickplace21_21x1}"
TASKS="${ROBOCASA_TASKS:-all}"
EPISODES="${ROBOCASA_EPISODES_PER_TASK:-1}"
SEED_BASE="${ROBOCASA_SEED_BASE:-1000}"
[[ "$LABEL" =~ ^[A-Za-z0-9_.-]{1,96}$ ]] || die 'invalid ROBOCASA_RUN_LABEL'
[[ "$TASKS" == all ]] || die 'this frozen baseline runs all 21 registered tasks'
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die 'ROBOCASA_EPISODES_PER_TASK must be positive'
[[ "$SEED_BASE" =~ ^[0-9]+$ ]] || die 'ROBOCASA_SEED_BASE must be non-negative'

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
for path in "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT" "$ENV_ROOT"; do
  [[ "$path" = /* && "$path" != / ]] || die "path must be an absolute non-root path: $path"
done
[[ ! -e "$RUN_ROOT" ]] || die "RUN_ROOT already exists: $RUN_ROOT"
[[ ! -e "$ARCHIVE_ROOT" ]] || die "ARCHIVE_ROOT already exists: $ARCHIVE_ROOT"
mkdir -p "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT"

REQUIREMENTS_FILE="$REPO_ROOT/vla_benchmarking/robocasa/requirements.txt"
ENTRYPOINT_MODULE="vla_benchmarking.robocasa.evaluation.runner"
ENTRYPOINT_FILE="$REPO_ROOT/vla_benchmarking/robocasa/evaluation/runner.py"
[[ -f "$REQUIREMENTS_FILE" ]] || die 'RoboCasa requirements file is missing'
[[ -f "$ENTRYPOINT_FILE" ]] || die 'RoboCasa runner is missing'

archive_on_exit() {
  rc=$?
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$RUN_ROOT/" "$ARCHIVE_ROOT/run/"
  else
    cp -a "$RUN_ROOT/." "$ARCHIVE_ROOT/run/"
  fi
  for suffix in out err; do
    log_path="${SLURM_LOG_ROOT:-$HOME/EmbodimentSemantic_runtime/operator/logs}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${suffix}"
    [[ -f "$log_path" ]] && cp -p "$log_path" "$ARCHIVE_ROOT/"
  done
  {
    printf 'schema=robocasa_pickplace21_job.v1\n'
    printf 'job_id=%s\njob_name=%s\nlabel=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NAME" "$LABEL"
    printf 'expected_commit=%s\nactual_commit=%s\nrepo_root=%s\n' "$ROBOCASA_EXPECTED_COMMIT" "$actual_commit" "$REPO_ROOT"
    printf 'tasks=%s\nepisodes_per_task=%s\nseed_base=%s\nsplit=target\n' "$TASKS" "$EPISODES" "$SEED_BASE"
    printf 'run_root=%s\noutput_root=%s\narchive_root=%s\nenv_root=%s\n' "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT" "$ENV_ROOT"
    printf 'requirements_sha256=%s\nentrypoint_sha256=%s\njob_exit_code=%s\n' \
      "$(sha256sum "$REQUIREMENTS_FILE" | awk '{print tolower($1)}')" \
      "$(sha256sum "$ENTRYPOINT_FILE" | awk '{print tolower($1)}')" "$rc"
  } > "$ARCHIVE_ROOT/run_manifest.env"
  exit "$rc"
}
trap archive_on_exit EXIT

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  conda create -y -p "$ENV_ROOT" python=3.11 pip
fi
PYTHON="$ENV_ROOT/bin/python"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install --requirement "$REQUIREMENTS_FILE"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" - <<'PY'
import cv2, mujoco, robocasa, robosuite
print("RoboCasa runtime imports OK")
print("robocasa", getattr(robocasa, "__version__", "unknown"))
print("robosuite", getattr(robosuite, "__version__", "unknown"))
print("mujoco", getattr(mujoco, "__version__", "unknown"))
print("opencv", cv2.__version__)
PY

ASSET_MARKER="$HOME/EmbodimentSemantic_runtime/robocasa/assets_ready_${ROBOCASA_EXPECTED_COMMIT}"
if [[ ! -f "$ASSET_MARKER" ]]; then
  printf 'y\n' | "$PYTHON" -m robocasa.scripts.setup_macros || true
  printf 'y\n' | "$PYTHON" -m robocasa.scripts.download_kitchen_assets --type all
  "$PYTHON" - <<'PY'
import robocasa
from pathlib import Path
root = Path(robocasa.__path__[0]) / "models" / "assets"
required = [root / "textures", root / "fixtures", root / "objects"]
if not all(path.exists() and any(path.iterdir()) for path in required):
    raise SystemExit(f"RoboCasa assets are incomplete under {root}")
print("RoboCasa assets ready", root)
PY
  mkdir -p "$(dirname "$ASSET_MARKER")"
  printf 'commit=%s\n' "$ROBOCASA_EXPECTED_COMMIT" > "$ASSET_MARKER"
fi

cat > "$RUN_ROOT/job_context.env" <<EOF
schema=robocasa_pickplace21_job.v1
job_id=$SLURM_JOB_ID
expected_commit=$ROBOCASA_EXPECTED_COMMIT
actual_commit=$actual_commit
repo_root=$REPO_ROOT
label=$LABEL
tasks=$TASKS
episodes_per_task=$EPISODES
seed_base=$SEED_BASE
split=target
run_root=$RUN_ROOT
output_root=$OUTPUT_ROOT
archive_root=$ARCHIVE_ROOT
env_root=$ENV_ROOT
EOF

cd "$REPO_ROOT"
"$PYTHON" -m "$ENTRYPOINT_MODULE" \
  --mode full \
  --execute-motion \
  --output-dir "$OUTPUT_ROOT" \
  --episodes-per-task "$EPISODES" \
  --seed-base "$SEED_BASE" \
  --split target
