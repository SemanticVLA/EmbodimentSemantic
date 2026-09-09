#!/usr/bin/env bash
# Run all ten task-specific SmolVLA adaptations sequentially inside one GPU
# allocation. Each child starts from the same frozen base checkpoint.

#SBATCH --job-name=smolvla_fresh_arrow_all
#SBATCH --partition=gpu_a40_ext
#SBATCH --exclude=compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.err

set -Eeuo pipefail
umask 027

die() { printf 'SmolVLA all-task PEFT job: %s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit with sbatch'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
[[ "${PEFT_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die 'PEFT_EXPECTED_COMMIT is invalid'
[[ "${PEFT_EXPECTED_CONTROLLER_HASH:-}" =~ ^[0-9a-f]{64}$ ]] || die 'controller hash is invalid'
[[ -n "${PEFT_ALL_TASK_RUN_ROOT:-}" && -n "${PEFT_ALL_TASK_ARCHIVE_ROOT:-}" ]] || die 'all-task roots are required'
# This all-task pilot is intentionally sealed to one Arrow demonstration and
# adapted-only evaluation.  Reject inherited values that would silently make
# the ten task adapters use a different protocol; the task runner's baseline
# branch remains available for standalone/default runs.
if [[ -n "${PEFT_ARROW_DEMOS:-}" && "${PEFT_ARROW_DEMOS:-}" != 1 ]]; then die 'PEFT_ARROW_DEMOS must be 1 for this all-task pilot'; fi
if [[ -n "${PEFT_SKIP_BASELINE:-}" && "${PEFT_SKIP_BASELINE:-}" != 1 ]]; then die 'PEFT_SKIP_BASELINE must be 1 for this all-task pilot'; fi
export PEFT_ARROW_DEMOS=1 PEFT_SKIP_BASELINE=1

REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
[[ -d "$REPO_ROOT/.git" ]] || die 'REPO_ROOT is not a Git checkout'
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$PEFT_EXPECTED_COMMIT" ]] || die 'repository SHA differs'
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'repository is dirty'
RUNNER="$REPO_ROOT/vla_benchmarking/libero/automatic_ttt/legion/run_smolvla_peft_arrow_task.sbatch"
[[ -f "$RUNNER" ]] || die 'single-task runner is missing'

for task_id in 0 1 2 3 4 5 6 7 8 9; do
  export PEFT_TASK_ID="$task_id"
  unset PEFT_TASK_IDS PEFT_ARROW_COLLECTION_MANIFEST
  export PEFT_LABEL="${PEFT_ALL_TASK_LABEL:-smolvla_fresh_arrow_bc5}_task_${task_id}"
  export PEFT_RUN_ROOT="$PEFT_ALL_TASK_RUN_ROOT/task_${task_id}/run"
  export PEFT_ARCHIVE_ROOT="$PEFT_ALL_TASK_ARCHIVE_ROOT/task_${task_id}"
  export PEFT_SCRATCH_ROOT="$PEFT_ALL_TASK_RUN_ROOT/task_${task_id}"
  printf '[%s] starting task %s/9 in job %s\n' "$(date +'%F %T')" "$task_id" "$SLURM_JOB_ID"
  bash "$RUNNER"
  printf '[%s] completed task %s/9 in job %s\n' "$(date +'%F %T')" "$task_id" "$SLURM_JOB_ID"
done

printf 'VERIFIED all ten task-specific runs completed sequentially in job %s\n' "$SLURM_JOB_ID"
