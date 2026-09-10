#!/usr/bin/env bash
# One allocation, ten independent task-specific SmolVLA visual-arrow adapters.
# Every task starts from the same frozen base through the common task runner.

#SBATCH --job-name=smolvla_visual_arrow_all
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

die() { printf 'SmolVLA visual-arrow all-task job: %s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit with sbatch'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
[[ "${PEFT_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die 'PEFT_EXPECTED_COMMIT is invalid'
[[ "${PEFT_EXPECTED_CONTROLLER_HASH:-}" =~ ^[0-9a-f]{64}$ ]] || die 'controller hash is invalid'
[[ -n "${PEFT_ALL_TASK_RUN_ROOT:-}" && -n "${PEFT_ALL_TASK_ARCHIVE_ROOT:-}" ]] || die 'all-task roots are required'

for path in "$PEFT_ALL_TASK_RUN_ROOT" "$PEFT_ALL_TASK_ARCHIVE_ROOT"; do
  [[ "$path" = /* && "$path" != / && ! -L "$path" ]] || die "unsafe all-task root: $path"
done
case "$PEFT_ALL_TASK_RUN_ROOT" in /mnt/beegfs/hjaber/*) ;; *) die 'run root must be under /mnt/beegfs/hjaber' ;; esac
case "$PEFT_ALL_TASK_ARCHIVE_ROOT" in /mnt/beegfs/hjaber/*) ;; *) die 'archive root must be under /mnt/beegfs/hjaber for visual-arrow runs' ;; esac

if [[ -n "${PEFT_ARROW_DEMOS:-}" && "$PEFT_ARROW_DEMOS" != 1 ]]; then die 'PEFT_ARROW_DEMOS must be 1'; fi
if [[ -n "${PEFT_SKIP_BASELINE:-}" && "$PEFT_SKIP_BASELINE" != 1 ]]; then die 'PEFT_SKIP_BASELINE must be 1'; fi
if [[ -n "${PEFT_STUDENT_VISUAL_CONDITION:-}" && "$PEFT_STUDENT_VISUAL_CONDITION" != visual_goal_arrow ]]; then
  die 'PEFT_STUDENT_VISUAL_CONDITION must be visual_goal_arrow'
fi
export PEFT_ARROW_DEMOS=1 PEFT_SKIP_BASELINE=1 PEFT_STUDENT_VISUAL_CONDITION=visual_goal_arrow

PEFT_START_TASK_ID="${PEFT_START_TASK_ID:-0}"
[[ "$PEFT_START_TASK_ID" =~ ^[0-9]+$ ]] || die 'PEFT_START_TASK_ID must be an integer in [0,9]'
(( PEFT_START_TASK_ID <= 9 )) || die 'PEFT_START_TASK_ID must be in [0,9]'

REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
[[ -d "$REPO_ROOT/.git" ]] || die 'REPO_ROOT is not a Git checkout'
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$PEFT_EXPECTED_COMMIT" ]] || die 'repository SHA differs'
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'repository is dirty'
RUNNER="$REPO_ROOT/vla_benchmarking/libero/automatic_ttt/legion/run_smolvla_peft_arrow_task.sbatch"
[[ -f "$RUNNER" ]] || die 'common task runner is missing'

for task_id in $(seq "$PEFT_START_TASK_ID" 9); do
  export PEFT_TASK_ID="$task_id"
  unset PEFT_TASK_IDS PEFT_ARROW_COLLECTION_MANIFEST PEFT_RESUME_SOURCE_RUN_ROOT
  export PEFT_LABEL="${PEFT_ALL_TASK_LABEL:-smolvla_visual_goal_arrow_bc5}_task_${task_id}"
  export PEFT_RUN_ROOT="$PEFT_ALL_TASK_RUN_ROOT/task_${task_id}/run"
  export PEFT_ARCHIVE_ROOT="$PEFT_ALL_TASK_ARCHIVE_ROOT/task_${task_id}"
  export PEFT_SCRATCH_ROOT="$PEFT_ALL_TASK_RUN_ROOT/task_${task_id}"
  printf '[%s] starting visual-arrow task %s/9 in job %s\n' "$(date +'%F %T')" "$task_id" "$SLURM_JOB_ID"
  bash "$RUNNER"
  printf '[%s] completed visual-arrow task %s/9 in job %s\n' "$(date +'%F %T')" "$task_id" "$SLURM_JOB_ID"
done

printf 'VERIFIED visual-arrow task-specific runs %s..9 completed sequentially in job %s\n' "$PEFT_START_TASK_ID" "$SLURM_JOB_ID"
