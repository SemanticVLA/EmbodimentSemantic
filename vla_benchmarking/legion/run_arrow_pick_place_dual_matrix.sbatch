#!/usr/bin/env bash
# Sequential vanilla and sealed-randomized arrow-only LIBERO matrix.
# Requires REPO_ROOT (isolated clean checkout) and
# ARROW_MATRIX_EXPECTED_COMMIT (exact 40-character checkout SHA).
# Controller selection is frozen to the checked-in v9d policy by default:
#   v9d_rgbd_region_grasp_search.json + its full active label.
# Historical and experimental controller documents are archival only and are
# not executable through this standard launcher.
# ZeroGrasp and adjacent controller experiments are retired from this
# standard launcher; their source, runtime, and results remain archival.

#SBATCH --job-name=arrow_pick_place_dual_matrix
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
die() { printf 'arrow dual-matrix job: %s\n' "$*" >&2; exit 2; }

[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit with sbatch; direct execution is refused'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
[[ -n "${ARROW_MATRIX_EXPECTED_COMMIT:-}" ]] || die 'ARROW_MATRIX_EXPECTED_COMMIT is required'

REPO_ROOT="$(cd "$REPO_ROOT" 2>/dev/null && pwd -P)" || die 'cannot resolve REPO_ROOT'
EXPECTED_COMMIT="$ARROW_MATRIX_EXPECTED_COMMIT"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'expected commit must be a lowercase 40-character SHA'
actual_root="$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null)" || die 'REPO_ROOT is not a Git checkout'
actual_root="$(cd "$actual_root" && pwd -P)"
[[ "$actual_root" == "$REPO_ROOT" ]] || die 'REPO_ROOT is not the worktree root'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] || die 'active shared checkout is not isolated'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)" || die 'cannot read checkout commit'
[[ "$actual_commit" == "$EXPECTED_COMMIT" ]] || die "checkout commit differs: $actual_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'REPO_ROOT is dirty'

PYTHON="${PYTHON:-$HOME/.conda/envs/embodiment-smolvla-py312/bin/python}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/results}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/EmbodimentSemantic_runtime/config}"
MATRIX_TASK_IDS="${MATRIX_TASK_IDS:-0,1,2,3,4,5,6,7,8,9}"
MATRIX_EPISODES_PER_TASK="${MATRIX_EPISODES_PER_TASK:-10}"
MATRIX_SEED_BASE="${MATRIX_SEED_BASE:-1000}"
MATRIX_RESOLUTION="${MATRIX_RESOLUTION:-256}"
FINAL_CONTROLLER_LABEL="v9d_rgbd_region_grasp_search"
FINAL_CONTROLLER_VARIANT="libero_spatial_akita_bowl_agentview_v9d_rgbd_region_grasp_search"
FINAL_CONTROLLER_CONFIG_PATH="$REPO_ROOT/vla_benchmarking/controller_configs/v9d_rgbd_region_grasp_search.json"
[[ -f "$FINAL_CONTROLLER_CONFIG_PATH" ]] || die "frozen v9d controller config is missing: $FINAL_CONTROLLER_CONFIG_PATH"
FINAL_CONTROLLER_CONFIG_PATH="$(realpath -e -- "$FINAL_CONTROLLER_CONFIG_PATH")" || die 'cannot canonicalize frozen v9d controller config'
if [[ -n "${ARROW_CONTROLLER_VARIANT:-}" && "$ARROW_CONTROLLER_VARIANT" != "$FINAL_CONTROLLER_VARIANT" ]]; then
  die 'ambient ARROW_CONTROLLER_VARIANT override refused; v9d is the only active arrow controller'
fi
for ambient_config in "${CONTROLLER_CONFIG:-}" "${ARROW_CONTROLLER_CONFIG:-}"; do
  if [[ -n "$ambient_config" ]]; then
    [[ -f "$ambient_config" ]] || die "ambient controller config override is not a file: $ambient_config"
    ambient_config_path="$(realpath -e -- "$ambient_config")" || die 'cannot canonicalize ambient controller config override'
    [[ "$ambient_config_path" == "$FINAL_CONTROLLER_CONFIG_PATH" ]] || die 'ambient controller config override refused; v9d is the only active arrow controller'
  fi
done
CONTROLLER_VARIANT="$FINAL_CONTROLLER_VARIANT"
CONTROLLER_CONFIG_INPUT="$FINAL_CONTROLLER_CONFIG_PATH"
CONTROLLER_CONFIG_PATH=""
CONTROLLER_CONFIG_FILE_SHA256=""
CONTROLLER_CONFIG_HASH=""
CONTROLLER_CONFIG_CANONICAL_PATH=""
MATRIX_CONTINUE_ON_MOTION_FAILURE="${MATRIX_CONTINUE_ON_MOTION_FAILURE:-1}"
MATRIX_EXECUTION_MODE="${MATRIX_EXECUTION_MODE:-execute_motion}"

if [[ -n "${CONTROLLER_CONFIG:-}" && -n "${ARROW_CONTROLLER_CONFIG:-}" && "$CONTROLLER_CONFIG" != "$ARROW_CONTROLLER_CONFIG" ]]; then
  die 'CONTROLLER_CONFIG and ARROW_CONTROLLER_CONFIG disagree'
fi

[[ "$MATRIX_TASK_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die 'MATRIX_TASK_IDS is invalid'
[[ "$MATRIX_EPISODES_PER_TASK" =~ ^[1-9][0-9]*$ ]] || die 'MATRIX_EPISODES_PER_TASK is invalid'
[[ "$MATRIX_SEED_BASE" =~ ^[0-9]+$ ]] || die 'MATRIX_SEED_BASE is invalid'
[[ "$MATRIX_RESOLUTION" =~ ^[1-9][0-9]*$ ]] || die 'MATRIX_RESOLUTION is invalid'
[[ "$CONTROLLER_VARIANT" =~ ^[A-Za-z0-9_.-]+$ ]] || die 'ARROW_CONTROLLER_VARIANT is invalid'
if [[ -n "$CONTROLLER_CONFIG_INPUT" && "$CONTROLLER_VARIANT" != "$FINAL_CONTROLLER_VARIANT" ]]; then
  die 'CONTROLLER_CONFIG cannot be combined with a non-default ARROW_CONTROLLER_VARIANT'
fi
[[ "$MATRIX_CONTINUE_ON_MOTION_FAILURE" =~ ^[01]$ ]] || die 'MATRIX_CONTINUE_ON_MOTION_FAILURE is invalid'
[[ "$MATRIX_EXECUTION_MODE" == execute_motion || "$MATRIX_EXECUTION_MODE" == dry_run ]] || die 'MATRIX_EXECUTION_MODE must be execute_motion or dry_run'

[[ "$RUN_ROOT" = /* && "$OUTPUT_ROOT" = /* && "$ARCHIVE_ROOT" = /* ]] || die 'paths must be absolute'
command -v realpath >/dev/null 2>&1 || die 'realpath is required for safe path validation'
if [[ -n "$CONTROLLER_CONFIG_INPUT" ]]; then
  [[ -f "$CONTROLLER_CONFIG_INPUT" ]] || die "controller config is not a regular file: $CONTROLLER_CONFIG_INPUT"
  CONTROLLER_CONFIG_PATH="$(realpath -e -- "$CONTROLLER_CONFIG_INPUT")" || die 'cannot canonicalize CONTROLLER_CONFIG'
  command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required for controller config provenance'
  CONTROLLER_CONFIG_FILE_SHA256="$(sha256sum -- "$CONTROLLER_CONFIG_PATH" | awk '{print $1}')" || die 'cannot hash CONTROLLER_CONFIG'
  [[ "$CONTROLLER_CONFIG_FILE_SHA256" =~ ^[0-9a-f]{64}$ ]] || die 'controller config SHA-256 is invalid'
fi
RUN_ROOT="$(realpath -m -- "$RUN_ROOT")" || die 'cannot canonicalize RUN_ROOT'
OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")" || die 'cannot canonicalize OUTPUT_ROOT'
ARCHIVE_ROOT="$(realpath -m -- "$ARCHIVE_ROOT")" || die 'cannot canonicalize ARCHIVE_ROOT'
[[ "$RUN_ROOT" != "$ARCHIVE_ROOT" ]] || die 'RUN_ROOT and ARCHIVE_ROOT must differ'
case "$ARCHIVE_ROOT" in
  "$RUN_ROOT"|"$RUN_ROOT"/*) die 'ARCHIVE_ROOT must not be inside RUN_ROOT' ;;
esac
case "$RUN_ROOT" in
  "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'RUN_ROOT must not be inside ARCHIVE_ROOT' ;;
esac
case "$OUTPUT_ROOT" in
  "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'OUTPUT_ROOT must not be inside ARCHIVE_ROOT' ;;
esac
case "$ARCHIVE_ROOT" in
  "$OUTPUT_ROOT"|"$OUTPUT_ROOT"/*) die 'ARCHIVE_ROOT must not be inside OUTPUT_ROOT' ;;
esac
[[ ! -e "$RUN_ROOT" ]] || die "RUN_ROOT already exists: $RUN_ROOT"
[[ ! -e "$OUTPUT_ROOT" ]] || die "OUTPUT_ROOT already exists: $OUTPUT_ROOT"
[[ ! -e "$ARCHIVE_ROOT" ]] || die "ARCHIVE_ROOT already exists: $ARCHIVE_ROOT"
mkdir -p "$RUN_ROOT" "$OUTPUT_ROOT"

VANILLA_ROOT="$OUTPUT_ROOT/vanilla"
RANDOMIZED_ROOT="$OUTPUT_ROOT/sealed_randomized"
COMBINED_SUMMARY="$OUTPUT_ROOT/arrow_pick_place_dual_matrix_summary.json"
MATRIX_LAUNCHER="$REPO_ROOT/vla_benchmarking/run_arrow_pick_place_matrix.py"

copy_tree() {
  local source="$1" destination="$2"
  [[ -e "$source" ]] || return 0
  mkdir -p "$destination"
  if command -v rsync >/dev/null 2>&1; then rsync -a -- "$source/" "$destination/"; else cp -a -- "$source/." "$destination/"; fi
}

archive_on_exit() {
  local workload_rc=$? archive_rc=0 log_root archived_output
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT"
  copy_tree "$RUN_ROOT" "$ARCHIVE_ROOT/run" || archive_rc=$?
  case "$OUTPUT_ROOT" in
    "$RUN_ROOT") archived_output="$ARCHIVE_ROOT/run" ;;
    "$RUN_ROOT"/*) archived_output="$ARCHIVE_ROOT/run${OUTPUT_ROOT#"$RUN_ROOT"}" ;;
    *)
      copy_tree "$OUTPUT_ROOT" "$ARCHIVE_ROOT/output" || archive_rc=$?
      archived_output="$ARCHIVE_ROOT/output"
      ;;
  esac
  log_root="${SLURM_LOG_ROOT:-$HOME/EmbodimentSemantic_runtime/operator/logs}"
  mkdir -p "$ARCHIVE_ROOT/slurm_logs" || archive_rc=$?
  for suffix in out err; do
    log_path="$log_root/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${suffix}"
    [[ -f "$log_path" ]] && cp -p -- "$log_path" "$ARCHIVE_ROOT/slurm_logs/" || true
  done
  for relative in vanilla/arrow_pick_place_matrix_summary.json sealed_randomized/arrow_pick_place_matrix_summary.json arrow_pick_place_dual_matrix_summary.json; do
    [[ -s "$archived_output/$relative" ]] || archive_rc=1
  done
  {
    printf 'archive_status=%s\n' "$([[ "$archive_rc" -eq 0 ]] && printf VERIFIED || printf FAILED)"
    printf 'job_exit_code=%s\n' "$workload_rc"
    printf 'job_id=%s\n' "$SLURM_JOB_ID"
    printf 'expected_commit=%s\n' "$EXPECTED_COMMIT"
    printf 'controller_variant=%s\n' "$CONTROLLER_VARIANT"
    printf 'controller_label=%s\n' "$FINAL_CONTROLLER_LABEL"
    printf 'controller_config_path=%s\n' "$CONTROLLER_CONFIG_PATH"
    printf 'controller_config_file_sha256=%s\n' "$CONTROLLER_CONFIG_FILE_SHA256"
    printf 'controller_config_hash=%s\n' "$CONTROLLER_CONFIG_HASH"
    printf 'controller_config_canonical_path=%s\n' "$CONTROLLER_CONFIG_CANONICAL_PATH"
    printf 'run_root=%s\n' "$RUN_ROOT"
    printf 'output_root=%s\n' "$OUTPUT_ROOT"
    printf 'archive_root=%s\n' "$ARCHIVE_ROOT"
    printf 'summary_status=%s\n' "$([[ -s "$COMBINED_SUMMARY" ]] && printf present || printf missing)"
    printf 'summary_path=%s\n' "$archived_output/arrow_pick_place_dual_matrix_summary.json"
  } > "$ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
  if [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]]; then workload_rc=$archive_rc; fi
  exit "$workload_rc"
}
trap archive_on_exit EXIT

cat > "$RUN_ROOT/job_context.env" <<EOF
job_id=$SLURM_JOB_ID
job_name=$SLURM_JOB_NAME
expected_commit=$EXPECTED_COMMIT
actual_commit=$actual_commit
repo_root=$REPO_ROOT
run_root=$RUN_ROOT
output_root=$OUTPUT_ROOT
archive_root=$ARCHIVE_ROOT
python=$PYTHON
libero_config_path=$LIBERO_CONFIG_PATH
matrix_tasks=$MATRIX_TASK_IDS
matrix_episodes_per_task=$MATRIX_EPISODES_PER_TASK
matrix_seed_base=$MATRIX_SEED_BASE
matrix_resolution=$MATRIX_RESOLUTION
controller_variant=$CONTROLLER_VARIANT
controller_config_path=$CONTROLLER_CONFIG_PATH
controller_config_file_sha256=$CONTROLLER_CONFIG_FILE_SHA256
suite_order=vanilla,sealed_randomized
continue_on_motion_failure=$MATRIX_CONTINUE_ON_MOTION_FAILURE
motion_mode=$MATRIX_EXECUTION_MODE
arrow_input_policy=hardcoded_subject_goal_bbox_renderer
language_translation=false
EOF

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
[[ -x "$PYTHON" ]] || die "Python executable is missing: $PYTHON"
[[ -d "$LIBERO_CONFIG_PATH" && -f "$LIBERO_CONFIG_PATH/config.yaml" ]] || die "LIBERO_CONFIG_PATH lacks config.yaml: $LIBERO_CONFIG_PATH"
[[ -f "$MATRIX_LAUNCHER" ]] || die "matrix launcher is missing: $MATRIX_LAUNCHER"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$REPO_ROOT/vla_benchmarking"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH
export LIBERO_CONFIG="$LIBERO_CONFIG_PATH/config.yaml"
export TMPDIR="$RUN_ROOT/tmp" HF_HOME="$RUN_ROOT/cache/hf" XDG_CACHE_HOME="$RUN_ROOT/cache/xdg" TORCH_HOME="$RUN_ROOT/cache/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"
export ARROW_CONTROLLER_VARIANT="$CONTROLLER_VARIANT"
if [[ -n "$CONTROLLER_CONFIG_PATH" ]]; then
  CONTROLLER_CONFIG_CANONICAL_PATH="$RUN_ROOT/controller_config_canonical.json"
  CONTROLLER_CONFIG_HASH="$($PYTHON - "$CONTROLLER_CONFIG_PATH" "$CONTROLLER_CONFIG_CANONICAL_PATH" <<'PY'
import json
import sys
from pathlib import Path

from controller_configs import load_controller_config

config_path, canonical_path = sys.argv[1:]
expanded = load_controller_config(config_path)
Path(canonical_path).write_text(
    json.dumps(expanded, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(expanded["config_hash"])
PY
  )" || die 'controller config failed semantic validation'
  [[ "$CONTROLLER_CONFIG_HASH" =~ ^[0-9a-f]{64}$ ]] || die 'controller config semantic hash is invalid'
  export ARROW_CONTROLLER_CONFIG="$CONTROLLER_CONFIG_PATH"
  {
    printf 'controller_config_hash=%s\n' "$CONTROLLER_CONFIG_HASH"
    printf 'controller_config_canonical_path=%s\n' "$CONTROLLER_CONFIG_CANONICAL_PATH"
  } >> "$RUN_ROOT/job_context.env"
fi

command -v nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi is unavailable inside the allocation'
nvidia-smi -L
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA smoke OK", torch.cuda.get_device_name(0))'
"$PYTHON" -c 'import libero, mujoco, robosuite; print("LIBERO/MuJoCo/robosuite imports OK")'
help_text="$("$PYTHON" "$MATRIX_LAUNCHER" --help)" || die 'matrix launcher help failed'
grep -Fq -- '--suite-mode' <<<"$help_text" || die 'matrix launcher lacks --suite-mode'
grep -Fq -- '--controller-variant' <<<"$help_text" || die 'matrix launcher lacks --controller-variant'
grep -Fq -- '--controller-config' <<<"$help_text" || die 'matrix launcher lacks --controller-config'

IFS=, read -r -a task_id_array <<<"$MATRIX_TASK_IDS"
expected_cells_per_suite=$(( ${#task_id_array[@]} * MATRIX_EPISODES_PER_TASK ))
expected_cells_total=$(( expected_cells_per_suite * 2 ))

write_suite_context() {
  local suite="$1" suite_root="$2" randomize="$3"
  mkdir -p "$suite_root"
  cat > "$suite_root/suite_context.env" <<EOF
suite_mode=$suite
controller_variant=$CONTROLLER_VARIANT
controller_config_path=$CONTROLLER_CONFIG_PATH
controller_config_file_sha256=$CONTROLLER_CONFIG_FILE_SHA256
controller_config_hash=$CONTROLLER_CONFIG_HASH
controller_config_canonical_path=$CONTROLLER_CONFIG_CANONICAL_PATH
randomize_scenes=$randomize
task_ids=$MATRIX_TASK_IDS
episodes_per_task=$MATRIX_EPISODES_PER_TASK
seed_base=$MATRIX_SEED_BASE
resolution=$MATRIX_RESOLUTION
output_root=$suite_root
EOF
}

run_suite() {
  local suite="$1" suite_root="$2" randomize="$3"
  write_suite_context "$suite" "$suite_root" "$randomize"
  export ARROW_SUITE_MODE="$suite" LIBERO_SUITE_MODE="$suite" ARROW_CONTROLLER_VARIANT="$CONTROLLER_VARIANT" RANDOMIZE_SCENES="$randomize"
  matrix_args=(
    --task-ids "$MATRIX_TASK_IDS"
    --episodes-per-task "$MATRIX_EPISODES_PER_TASK"
    --seed-base "$MATRIX_SEED_BASE"
    --resolution "$MATRIX_RESOLUTION"
    --output-root "$suite_root"
    --suite-mode "$suite"
    --controller-variant "$CONTROLLER_VARIANT"
    --allow-unvalidated-profile
  )
  if [[ "$MATRIX_EXECUTION_MODE" == execute_motion ]]; then
    matrix_args+=(--execute-motion)
  else
    matrix_args+=(--dry-run)
  fi
  if [[ "$MATRIX_CONTINUE_ON_MOTION_FAILURE" == 1 ]]; then matrix_args+=(--continue-on-motion-failure); fi
  if [[ -n "$CONTROLLER_CONFIG_PATH" ]]; then matrix_args+=(--controller-config "$CONTROLLER_CONFIG_PATH"); fi
  "$PYTHON" "$MATRIX_LAUNCHER" "${matrix_args[@]}"
}

run_suite vanilla "$VANILLA_ROOT" 0
run_suite sealed_randomized "$RANDOMIZED_ROOT" 1
[[ -s "$VANILLA_ROOT/arrow_pick_place_matrix_summary.json" ]] || die 'vanilla summary is missing'
[[ -s "$RANDOMIZED_ROOT/arrow_pick_place_matrix_summary.json" ]] || die 'sealed-randomized summary is missing'

"$PYTHON" - "$VANILLA_ROOT/arrow_pick_place_matrix_summary.json" "$RANDOMIZED_ROOT/arrow_pick_place_matrix_summary.json" "$COMBINED_SUMMARY" "$expected_cells_per_suite" "$expected_cells_total" "$CONTROLLER_VARIANT" "$CONTROLLER_CONFIG_PATH" "$CONTROLLER_CONFIG_FILE_SHA256" "$CONTROLLER_CONFIG_HASH" <<'PY'
import json
import sys
from pathlib import Path

(
    vanilla_path,
    randomized_path,
    combined_path,
    expected_each,
    expected_total,
    variant,
    config_path,
    config_file_sha256,
    config_hash,
) = sys.argv[1:]
expected_each = int(expected_each)
expected_total = int(expected_total)
summaries = {}
for mode, path in (("vanilla", vanilla_path), ("sealed_randomized", randomized_path)):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("total_cells", -1)) != expected_each:
        raise SystemExit(f"{mode} total_cells does not match planned count")
    if int(payload.get("completed_cells", 0)) + int(payload.get("failed_cells_count", 0)) != expected_each:
        raise SystemExit(f"{mode} summary does not account for all planned cells")
    summaries[mode] = payload

observed_variants = {str(payload.get("controller_variant")) for payload in summaries.values()}
if len(observed_variants) != 1:
    raise SystemExit("suite controller variants disagree")
variant = observed_variants.pop()
if config_hash:
    observed_config_hashes = {
        str((payload.get("controller_config") or {}).get("config_hash"))
        for payload in summaries.values()
    }
    if observed_config_hashes != {config_hash}:
        raise SystemExit("suite controller config hashes disagree with launcher preflight")

def total(field):
    return sum(int(payload.get(field, 0) or 0) for payload in summaries.values())

completed = total("completed_cells")
failed = total("failed_cells_count")
evaluated = total("evaluated_cells")
successes = total("successes")
if completed + failed != expected_total:
    raise SystemExit("combined summary does not account for all planned cells")
combined = {
    "schema_version": "arrow_pick_place_dual_matrix.v1",
    "controller_variant": variant,
    "controller_config": {
        "path": config_path or None,
        "file_sha256": config_file_sha256 or None,
        "config_hash": config_hash or None,
    },
    "suite_order": ["vanilla", "sealed_randomized"],
    "expected_cells_per_suite": expected_each,
    "expected_total_cells": expected_total,
    "total_cells": completed + failed,
    "completed_cells": completed,
    "failed_cells_count": failed,
    "evaluated_cells": evaluated,
    "successes": successes,
    "planned_success_rate": successes / (completed + failed) if evaluated else None,
    "evaluable_success_rate": successes / evaluated if evaluated else None,
    "per_suite": {
        mode: {
            "summary_path": path,
            "contract_hash": summaries[mode].get("contract_hash"),
            "total_cells": summaries[mode].get("total_cells"),
            "completed_cells": summaries[mode].get("completed_cells"),
            "failed_cells_count": summaries[mode].get("failed_cells_count"),
            "evaluated_cells": summaries[mode].get("evaluated_cells"),
            "successes": summaries[mode].get("successes"),
            "planned_success_rate": summaries[mode].get("planned_success_rate"),
            "evaluable_success_rate": summaries[mode].get("evaluable_success_rate"),
            "failure_by_class": summaries[mode].get("failure_by_class", {}),
            "failure_by_stage": summaries[mode].get("failure_by_stage", {}),
            "per_task": summaries[mode].get("per_task", {}),
        }
        for mode, path in (("vanilla", vanilla_path), ("sealed_randomized", randomized_path))
    },
}
Path(combined_path).write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

printf 'dual matrix combined summary: %s\n' "$COMBINED_SUMMARY"
