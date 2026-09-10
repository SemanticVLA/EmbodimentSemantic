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
for path in "$PEFT_ALL_TASK_RUN_ROOT" "$PEFT_ALL_TASK_ARCHIVE_ROOT"; do
  [[ "$path" = /* && "$path" != / && ! -L "$path" ]] || die "unsafe all-task root: $path"
done
case "$PEFT_ALL_TASK_RUN_ROOT" in /mnt/beegfs/hjaber/*) ;; *) die 'run root must be under /mnt/beegfs/hjaber' ;; esac
case "$PEFT_ALL_TASK_ARCHIVE_ROOT" in /mnt/beegfs/hjaber/*) ;; *) die 'archive root must be under /mnt/beegfs/hjaber' ;; esac
VALIDATION_PYTHON="${PEFT_PYTHON:-$HOME/EmbodimentSemantic_runtime/EmbodimentSemantic/grasp_controller/venv-py312-eacdc54a9663db0a/bin/python}"
[[ -x "$VALIDATION_PYTHON" ]] || die "Python is missing: $VALIDATION_PYTHON"
# A normal run starts at task 0 and executes every task once.  Operational
# recovery may start at a later task after earlier task artifacts have been
# preserved, or may publish task 0 from a preserved, already-trained run before
# continuing with fresh task 1..9 runs in this same allocation.  Never let a
# resume source be silently ignored or leak into a later fresh task.
PEFT_START_TASK_ID="${PEFT_START_TASK_ID:-0}"
[[ "$PEFT_START_TASK_ID" =~ ^[0-9]+$ ]] || die 'PEFT_START_TASK_ID must be an integer in [0,9]'
(( PEFT_START_TASK_ID <= 9 )) || die 'PEFT_START_TASK_ID must be in [0,9]'
PEFT_END_TASK_ID="${PEFT_END_TASK_ID:-9}"
[[ "$PEFT_END_TASK_ID" =~ ^[0-9]+$ ]] || die 'PEFT_END_TASK_ID must be an integer in [0,9]'
(( PEFT_END_TASK_ID <= 9 )) || die 'PEFT_END_TASK_ID must be in [0,9]'
(( PEFT_START_TASK_ID <= PEFT_END_TASK_ID )) || die 'PEFT_START_TASK_ID must not exceed PEFT_END_TASK_ID'
if [[ -n "${PEFT_RESUME_SOURCE_RUN_ROOT:-}" && "$PEFT_START_TASK_ID" -ne 0 ]]; then
  die 'PEFT_RESUME_SOURCE_RUN_ROOT is permitted only when PEFT_START_TASK_ID=0'
fi
RESUME_RUNNER_RELATIVE="${PEFT_RESUME_RUNNER_RELATIVE:-vla_benchmarking/libero/automatic_ttt/legion/run_smolvla_peft_arrow_resume.sbatch}"
if [[ "$RESUME_RUNNER_RELATIVE" = /* || "$RESUME_RUNNER_RELATIVE" == *'..'* || "$RESUME_RUNNER_RELATIVE" == *$'\n'* || "$RESUME_RUNNER_RELATIVE" == *$'\r'* || "$RESUME_RUNNER_RELATIVE" == *' '* ]]; then
  die 'PEFT_RESUME_RUNNER_RELATIVE must be a relative shell-safe path'
fi
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
if [[ -n "${PEFT_RESUME_SOURCE_RUN_ROOT:-}" ]]; then
  RESUME_RUNNER="$REPO_ROOT/$RESUME_RUNNER_RELATIVE"
  [[ -f "$RESUME_RUNNER" ]] || die "resume runner is missing: $RESUME_RUNNER_RELATIVE"
  export PEFT_RESUME_RUNNER="$RESUME_RUNNER"
  export PEFT_TASK_ID=0
  export PEFT_LABEL="${PEFT_ALL_TASK_LABEL:-smolvla_fresh_arrow_bc5}_task_0_resume"
  export PEFT_RUN_ROOT="$PEFT_ALL_TASK_RUN_ROOT/task_0/run"
  export PEFT_ARCHIVE_ROOT="$PEFT_ALL_TASK_ARCHIVE_ROOT/task_0"
  export PEFT_SCRATCH_ROOT="$PEFT_ALL_TASK_RUN_ROOT/task_0"
  printf '[%s] resuming task 0 publication from %s in job %s\n' "$(date +'%F %T')" "$PEFT_RESUME_SOURCE_RUN_ROOT" "$SLURM_JOB_ID"
  bash "$RESUME_RUNNER"
  # The publication-only runner must prove that the new run has a completed
  # receipt pointing at an existing immutable adapter before task 1 starts.
  # COMPLETED is an empty marker created with `touch`; existence, not size,
  # is the completion contract.  The recovery receipt is deliberately a
  # separate, non-empty publication record so a stale marker cannot authorize
  # task 1.
  [[ -f "$PEFT_RUN_ROOT/COMPLETED" ]] || die 'task 0 resume did not produce COMPLETED'
  RECOVERY_RECEIPT="$PEFT_RUN_ROOT/publication_recovery_receipt.json"
  [[ -s "$RECOVERY_RECEIPT" ]] || die 'task 0 resume did not produce publication_recovery_receipt.json'
  "$VALIDATION_PYTHON" - "$RECOVERY_RECEIPT" <<'PY'
import json
import pathlib
import sys

receipt_path = pathlib.Path(sys.argv[1])
try:
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"invalid task 0 adapter receipt: {exc}")
if not isinstance(value, dict):
    raise SystemExit("invalid task 0 publication recovery receipt: expected an object")
candidate = next(
    (value.get(key) for key in ("artifact_path", "published_artifact_path", "adapter_artifact_path")
     if isinstance(value.get(key), str)),
    None,
)
artifact = pathlib.Path(candidate) if candidate else pathlib.Path()
if not artifact.is_absolute() or not artifact.exists():
    raise SystemExit(f"task 0 publication receipt does not point to an existing absolute artifact: {artifact}")
PY
  printf '[%s] task 0 publication verified; continuing with fresh tasks 1..%s\n' "$(date +'%F %T')" "$PEFT_END_TASK_ID"
  unset PEFT_RESUME_SOURCE_RUN_ROOT PEFT_RESUME_SOURCE_JOB_ID PEFT_RESUME_TRAINING_COMMIT PEFT_RESUME_RUNNER
  PEFT_START_TASK_ID=1
fi

FIRST_TASK_ID="$PEFT_START_TASK_ID"
for task_id in $(seq "$FIRST_TASK_ID" "$PEFT_END_TASK_ID"); do
  export PEFT_TASK_ID="$task_id"
  unset PEFT_TASK_IDS PEFT_ARROW_COLLECTION_MANIFEST
  export PEFT_LABEL="${PEFT_ALL_TASK_LABEL:-smolvla_fresh_arrow_bc5}_task_${task_id}"
  export PEFT_RUN_ROOT="$PEFT_ALL_TASK_RUN_ROOT/task_${task_id}/run"
  export PEFT_ARCHIVE_ROOT="$PEFT_ALL_TASK_ARCHIVE_ROOT/task_${task_id}"
  export PEFT_SCRATCH_ROOT="$PEFT_ALL_TASK_RUN_ROOT/task_${task_id}"
  printf '[%s] starting task %s/%s in job %s\n' "$(date +'%F %T')" "$task_id" "$PEFT_END_TASK_ID" "$SLURM_JOB_ID"
  bash "$RUNNER"
  printf '[%s] completed task %s/%s in job %s\n' "$(date +'%F %T')" "$task_id" "$PEFT_END_TASK_ID" "$SLURM_JOB_ID"
done

printf 'VERIFIED task-specific runs %s..%s completed sequentially in job %s\n' "$FIRST_TASK_ID" "$PEFT_END_TASK_ID" "$SLURM_JOB_ID"
