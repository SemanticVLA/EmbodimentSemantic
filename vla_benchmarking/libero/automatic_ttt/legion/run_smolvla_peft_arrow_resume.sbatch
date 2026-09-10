#!/usr/bin/env bash
# Publication-only recovery for the preserved task-0 SmolVLA run.
# This job intentionally invokes the scalar runner with an explicit source;
# the runner exits before its collection/training/evaluation branches.

#SBATCH --job-name=smolvla_peft_arrow_resume
#SBATCH --partition=gpu_a40_ext
#SBATCH --exclude=compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.err

set -Eeuo pipefail
umask 027
die() { printf 'SmolVLA PEFT resume: %s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || die 'submit with sbatch'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
[[ "${PEFT_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die 'PEFT_EXPECTED_COMMIT must be a 40-character SHA'
[[ "${PEFT_EXPECTED_CONTROLLER_HASH:-}" =~ ^[0-9a-f]{64}$ ]] || die 'PEFT_EXPECTED_CONTROLLER_HASH must be a 64-character SHA-256'
[[ -n "${PEFT_RESUME_SOURCE_RUN_ROOT:-}" ]] || die 'PEFT_RESUME_SOURCE_RUN_ROOT is required and must be explicit'
[[ -d "$PEFT_RESUME_SOURCE_RUN_ROOT" && ! -L "$PEFT_RESUME_SOURCE_RUN_ROOT" ]] || die 'resume source run root is missing or symlinked'
[[ "${PEFT_RESUME_SOURCE_JOB_ID:-}" =~ ^[0-9]+$ ]] || die 'PEFT_RESUME_SOURCE_JOB_ID must be an integer'
[[ "${PEFT_RESUME_TRAINING_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die 'PEFT_RESUME_TRAINING_COMMIT must be a 40-character SHA'
[[ "${PEFT_REQUESTED_EPOCHS:-}" =~ ^[1-9][0-9]*$ ]] || die 'PEFT_REQUESTED_EPOCHS must be a positive integer'

REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
[[ -d "$REPO_ROOT/.git" ]] || die 'REPO_ROOT is not a Git checkout'
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$PEFT_EXPECTED_COMMIT" ]] || die 'repository SHA differs'
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'repository is dirty'

BASE_POLICY_REVISION="${PEFT_BASE_POLICY_REVISION:-6721902bc4d61e50a3bfdb11dfb4cb626f05d102}"
BASE_POLICY="${PEFT_BASE_POLICY:-$HOME/EmbodimentSemantic/vla_benchmarking/base_models/smolvla_libero-$BASE_POLICY_REVISION}"
LIBERO_DIR="${PEFT_LIBERO_DIR:-$HOME/EmbodimentSemantic/vla_benchmarking/LIBERO}"
LIBERO_CONFIG="${PEFT_LIBERO_CONFIG:-$HOME/EmbodimentSemantic_runtime/config/config.yaml}"
PYTHON="${PEFT_PYTHON:-$HOME/EmbodimentSemantic_runtime/EmbodimentSemantic/grasp_controller/venv-py312-eacdc54a9663db0a/bin/python}"
ARROW_CONTROLLER_CONFIG="${ARROW_CONTROLLER_CONFIG:-$REPO_ROOT/vla_benchmarking/libero/arrow_grasp_controller/configs/canonical_molmo_rgbd_grasp.json}"
for path in "$BASE_POLICY" "$LIBERO_DIR" "$LIBERO_CONFIG" "$PYTHON" "$ARROW_CONTROLLER_CONFIG"; do [[ -e "$path" ]] || die "required path is missing: $path"; done

RUN_PARENT="${PEFT_RESUME_RUN_PARENT:-${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/peft_arrow}"
ARCHIVE_PARENT="${PEFT_RESUME_ARCHIVE_PARENT:-$HOME/EmbodimentSemantic_archive/peft_arrow/smolvla}"
RUN_ROOT="${PEFT_RUN_ROOT:-$RUN_PARENT/smolvla_task0_resume_${SLURM_JOB_ID}/run}"
ARCHIVE_ROOT="${PEFT_ARCHIVE_ROOT:-$ARCHIVE_PARENT/smolvla_task0_resume_${SLURM_JOB_ID}}"
for path in "$RUN_ROOT" "$ARCHIVE_ROOT"; do
  [[ "$path" = /* && "$path" != / && ! -L "$path" ]] || die "unsafe output path: $path"
  [[ ! -e "$path" ]] || die "refusing to reuse existing output path: $path"
done

export REPO_ROOT PEFT_EXPECTED_COMMIT PEFT_EXPECTED_CONTROLLER_HASH
export PEFT_TASK_ID=0 PEFT_ARROW_DEMOS=1 PEFT_SKIP_BASELINE=1 PEFT_VLA=smolvla
export PEFT_BASE_POLICY="$BASE_POLICY" PEFT_BASE_POLICY_REVISION="$BASE_POLICY_REVISION"
export PEFT_RUN_ROOT="$RUN_ROOT" PEFT_ARCHIVE_ROOT="$ARCHIVE_ROOT"
export PEFT_RESUME_SOURCE_RUN_ROOT PEFT_RESUME_SOURCE_JOB_ID PEFT_RESUME_TRAINING_COMMIT PEFT_REQUESTED_EPOCHS
export ARROW_CONTROLLER_CONFIG

RUNNER="$REPO_ROOT/vla_benchmarking/libero/automatic_ttt/legion/run_smolvla_peft_arrow_task.sbatch"
[[ -f "$RUNNER" ]] || die 'scalar runner is missing'
bash "$RUNNER"
