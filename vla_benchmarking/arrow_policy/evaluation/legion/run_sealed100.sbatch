#!/usr/bin/env bash
# One resumable SLURM job for ArrowStudent's sealed-randomized 100-cell eval.
# The evaluator owns all environment, arrow, policy, action, and result logic.
# This wrapper owns pins, runtime setup, run placement, and archival.
#SBATCH --job-name=arrow_student_sealed100
#SBATCH --partition=gpu_a40_ext
#SBATCH --exclude=compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4-23:00:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.err

set -Eeuo pipefail
umask 027
export PYTHONDONTWRITEBYTECODE=1
die() { printf 'arrow student sealed100: %s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit with sbatch; direct execution is refused'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
[[ "${ARROW_STUDENT_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die 'expected commit must be a 40-character SHA'
[[ "${ARROW_STUDENT_RUN_ID:-}" =~ ^[A-Za-z0-9_.-]{1,96}$ ]] || die 'run ID is required and must be safe'
command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'
command -v flock >/dev/null 2>&1 || die 'flock is required'

REPO_ROOT="$(realpath -e -- "$REPO_ROOT")" || die 'cannot resolve REPO_ROOT'
[[ -d "$REPO_ROOT/.git" ]] || die 'REPO_ROOT is not a Git checkout'
actual_root="$(realpath -e -- "$(git -C "$REPO_ROOT" rev-parse --show-toplevel)")" || die 'cannot resolve Git root'
[[ "$actual_root" == "$REPO_ROOT" ]] || die 'REPO_ROOT is not the worktree root'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] || die 'shared checkout is forbidden'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)" || die 'cannot read checkout commit'
[[ "$actual_commit" == "$ARROW_STUDENT_EXPECTED_COMMIT" ]] || die "checkout SHA differs: $actual_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'REPO_ROOT is dirty'

PYTHON="${PYTHON:-$HOME/.conda/envs/embodiment-smolvla-py312/bin/python}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/EmbodimentSemantic_runtime/config}"
CHECKPOINT="${ARROW_STUDENT_CHECKPOINT:-/home/hjaber/EmbodimentSemantic_archive/arrow_policy/arrow_visual_language_free_v1_20260904_fix1/checkpoints/D_full_student/update_0010000}"
SOURCE_CHECKPOINT="${ARROW_STUDENT_SOURCE_CHECKPOINT:-/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/runs/legion_action_visual_lora_no_arrow_s1000_v7_20260831T101333Z_candidate/checkpoints/029190/pretrained_model}"
BASE_POLICY="${ARROW_STUDENT_BASE_POLICY:-/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/vla_benchmarking/base_models/smolvla_libero-6721902bc4d61e50a3bfdb11dfb4cb626f05d102}"
REFERENCE_ROOT="${ARROW_STUDENT_REFERENCE_ROOT:-/home/hjaber/EmbodimentSemantic_archive/molmo_failure_sealed100/molmo_failure_sealed100_fa1ae83_1920556/run/results/sealed_randomized}"
REFERENCE_MANIFEST_SHA256=eb0788ce41e10215f4ee730d53d52360cd1731a6e6a91999af385babd127b369
CHECKPOINT_MODEL_SHA256=5d7670368c1eec80eb7855454e77d69d4e177d693155246f6a68409646a85bcc
CHECKPOINT_MANIFEST_SHA256=827ebb564e6b6e9371e7a30c2d953d6e9d6a0af9e793f99dc7af569b85ab3cab
BASE_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102
[[ -n "${SCRATCH_FLASH:-}" ]] || die 'SCRATCH_FLASH is required'
[[ "$CHECKPOINT" = /* && "$SOURCE_CHECKPOINT" = /* && "$BASE_POLICY" = /* && "$REFERENCE_ROOT" = /* && "$LIBERO_CONFIG_PATH" = /* ]] || die 'paths must be absolute'

RUN_ROOT="${ARROW_STUDENT_RUN_ROOT:-$SCRATCH_FLASH/arrow_student_sealed100/$ARROW_STUDENT_RUN_ID}"
ARCHIVE_ROOT="${ARROW_STUDENT_ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/arrow_student_sealed100/$ARROW_STUDENT_RUN_ID}"
RUN_ROOT="$(realpath -m -- "$RUN_ROOT")" || die 'cannot canonicalize RUN_ROOT'
ARCHIVE_ROOT="$(realpath -m -- "$ARCHIVE_ROOT")" || die 'cannot canonicalize ARCHIVE_ROOT'
[[ "$RUN_ROOT" != "$ARCHIVE_ROOT" ]] || die 'run and archive roots must differ'
case "$RUN_ROOT" in "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'run root is inside archive' ;; esac
case "$ARCHIVE_ROOT" in "$RUN_ROOT"|"$RUN_ROOT"/*) die 'archive root is inside run' ;; esac
OUTPUT_ROOT="$RUN_ROOT/run/results/sealed_randomized"
PREFLIGHT_ROOT="$RUN_ROOT/run/preflight/results/sealed_randomized"
STATUS_PATH="$OUTPUT_ROOT/arrow_pick_place_matrix_status.json"
SUMMARY_PATH="$OUTPUT_ROOT/arrow_pick_place_matrix_summary.json"
[[ -x "$PYTHON" ]] || die "Python executable is missing: $PYTHON"
[[ -d "$LIBERO_CONFIG_PATH" && -f "$LIBERO_CONFIG_PATH/config.yaml" ]] || die 'LIBERO_CONFIG_PATH must contain config.yaml'
[[ -d "$CHECKPOINT" && -f "$CHECKPOINT/model.pt" ]] || die 'student model is missing'
[[ -f "$CHECKPOINT/manifest.json" ]] || die 'student checkpoint manifest is missing'
[[ -d "$SOURCE_CHECKPOINT" ]] || die 'source checkpoint is missing'
[[ -f "$BASE_POLICY/config.json" ]] || die 'base policy config is missing'
[[ -f "$REFERENCE_ROOT/arrow_pick_place_matrix_manifest.json" ]] || die 'reference manifest is missing'

mkdir -p "$RUN_ROOT" "$ARCHIVE_ROOT" "$RUN_ROOT/run" "$RUN_ROOT/logs" "$RUN_ROOT/cache" "$PREFLIGHT_ROOT"
exec 9>"$RUN_ROOT/.run.lock"
flock -n 9 || die "run ID is already active: $ARROW_STUDENT_RUN_ID"
snapshot_inputs() {
  local source target
  mkdir -p "$RUN_ROOT/provenance/input_snapshots"
  for source in "$REPO_ROOT/vla_benchmarking/EXPERIMENT_LEDGER.md" "$REPO_ROOT/vla_benchmarking/evaluation_results_tracker.json"; do
    [[ -f "$source" ]] || continue
    target="$RUN_ROOT/provenance/input_snapshots/$(basename "$source")"
    [[ -e "$target" ]] || cp -p -- "$source" "$target"
  done
}
copy_tree() { local source="$1" destination="$2"; [[ -e "$source" ]] || return 0; mkdir -p "$destination"; if command -v rsync >/dev/null 2>&1; then rsync -a -- "$source/" "$destination/"; else cp -a -- "$source/." "$destination/"; fi; }
restore_archive() {
  [[ -d "$ARCHIVE_ROOT/run" ]] || return 0
  if command -v rsync >/dev/null 2>&1; then rsync -a --update --exclude='.run.lock' -- "$ARCHIVE_ROOT/run/" "$RUN_ROOT/run/"; else copy_tree "$ARCHIVE_ROOT/run" "$RUN_ROOT/run"; fi
  if [[ -f "$ARCHIVE_ROOT/run/run_contract.json" && -f "$RUN_ROOT/run/run_contract.json" ]]; then cmp -s "$ARCHIVE_ROOT/run/run_contract.json" "$RUN_ROOT/run/run_contract.json" || die 'scratch/archive contracts differ'; fi
}
copy_slurm_logs() { local suffix source; for suffix in out err; do source="$HOME/EmbodimentSemantic_runtime/operator/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${suffix}"; [[ -f "$source" ]] && cp -p -- "$source" "$RUN_ROOT/logs/slurm_${SLURM_JOB_ID}.${suffix}" || true; done; }
archive_snapshot() {
  local reason="${1:-snapshot}" rc=0
  copy_slurm_logs || rc=$?; mkdir -p "$ARCHIVE_ROOT" || rc=$?
  if [[ "$rc" -eq 0 ]]; then if command -v rsync >/dev/null 2>&1; then rsync -a --exclude='.run.lock' -- "$RUN_ROOT/" "$ARCHIVE_ROOT/" || rc=$?; else copy_tree "$RUN_ROOT" "$ARCHIVE_ROOT" || rc=$?; fi; fi
  if [[ "$rc" -eq 0 ]]; then { printf 'schema=arrow_student_sealed100_archive.v1\n'; printf 'archive_status=%s\njob_exit_code=%s\njob_id=%s\n' "$reason" "${SLURM_JOB_EXIT_CODE:-}" "$SLURM_JOB_ID"; printf 'run_id=%s\nexpected_commit=%s\nactual_commit=%s\n' "$ARROW_STUDENT_RUN_ID" "$ARROW_STUDENT_EXPECTED_COMMIT" "$actual_commit"; printf 'run_root=%s\narchive_root=%s\noutput_root=%s\nstatus_path=%s\nsummary_path=%s\n' "$RUN_ROOT" "$ARCHIVE_ROOT" "$OUTPUT_ROOT" "$STATUS_PATH" "$SUMMARY_PATH"; printf 'archived_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$ARCHIVE_ROOT/archive_status.env" || rc=$?; fi
  return "$rc"
}
archive_on_exit() { local workload_rc=$? archive_rc=0; trap - EXIT INT TERM HUP; set +e; archive_snapshot "job_exit_${workload_rc}" || archive_rc=$?; if [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]]; then workload_rc="$archive_rc"; fi; exit "$workload_rc"; }
trap archive_on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

restore_archive
module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/vla_benchmarking${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH
export TMPDIR="$RUN_ROOT/cache/tmp" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="${ARROW_STUDENT_HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg" TORCH_HOME="$RUN_ROOT/cache/torch"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME"
nvidia-smi -L
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
snapshot_inputs

model_sha="$(sha256sum -- "$CHECKPOINT/model.pt" | awk '{print tolower($1)}')"
manifest_sha="$(sha256sum -- "$CHECKPOINT/manifest.json" | awk '{print tolower($1)}')"
reference_sha="$(sha256sum -- "$REFERENCE_ROOT/arrow_pick_place_matrix_manifest.json" | awk '{print tolower($1)}')"
[[ "$model_sha" == "$CHECKPOINT_MODEL_SHA256" ]] || die "student model hash mismatch: $model_sha"
[[ "$manifest_sha" == "$CHECKPOINT_MANIFEST_SHA256" ]] || die "student manifest hash mismatch: $manifest_sha"
[[ "$reference_sha" == "$REFERENCE_MANIFEST_SHA256" ]] || die "reference manifest hash mismatch: $reference_sha"
"$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("revision")==sys.argv[2], (p.get("revision"),sys.argv[2])' "$BASE_POLICY/base_snapshot_manifest.json" "$BASE_REVISION"

LAUNCHER_PATH="$REPO_ROOT/vla_benchmarking/arrow_policy/evaluation/legion/run_sealed100.sbatch"
LAUNCHER_SHA256="$(sha256sum -- "$LAUNCHER_PATH" | awk '{print tolower($1)}')"
RUN_CONTRACT="$RUN_ROOT/run/run_contract.json"
export CHECKPOINT SOURCE_CHECKPOINT BASE_POLICY REFERENCE_ROOT RUN_ROOT ARCHIVE_ROOT OUTPUT_ROOT PREFLIGHT_ROOT STATUS_PATH SUMMARY_PATH LAUNCHER_PATH LAUNCHER_SHA256
export REFERENCE_MANIFEST_SHA256 CHECKPOINT_MODEL_SHA256 CHECKPOINT_MANIFEST_SHA256 BASE_REVISION
if [[ -f "$RUN_CONTRACT" ]]; then
  "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); checks={"schema":"arrow_student_sealed100_run.v1","expected_commit":sys.argv[2],"actual_commit":sys.argv[2],"run_root":sys.argv[3],"archive_root":sys.argv[4],"output_root":sys.argv[5],"checkpoint":sys.argv[6],"reference_root":sys.argv[7],"cell_count":100,"seed_base":1000}; bad=[(k,p.get(k),v) for k,v in checks.items() if p.get(k)!=v]; assert not bad,bad' "$RUN_CONTRACT" "$actual_commit" "$RUN_ROOT" "$ARCHIVE_ROOT" "$OUTPUT_ROOT" "$CHECKPOINT" "$REFERENCE_ROOT"
else
  "$PYTHON" -c 'import json,os,sys,time; keys=("ARROW_STUDENT_RUN_ID","SLURM_JOB_ID","ARROW_STUDENT_EXPECTED_COMMIT","CHECKPOINT","CHECKPOINT_MODEL_SHA256","CHECKPOINT_MANIFEST_SHA256","SOURCE_CHECKPOINT","BASE_POLICY","BASE_REVISION","REFERENCE_ROOT","REFERENCE_MANIFEST_SHA256","RUN_ROOT","ARCHIVE_ROOT","OUTPUT_ROOT","PREFLIGHT_ROOT","STATUS_PATH","SUMMARY_PATH","LAUNCHER_PATH","LAUNCHER_SHA256"); p={k.lower():os.environ.get(k) for k in keys}; p.update(schema="arrow_student_sealed100_run.v1",run_id=os.environ["ARROW_STUDENT_RUN_ID"],job_id=os.environ.get("SLURM_JOB_ID"),expected_commit=os.environ["ARROW_STUDENT_EXPECTED_COMMIT"],actual_commit=os.environ["ARROW_STUDENT_EXPECTED_COMMIT"],task_ids=list(range(10)),episodes_per_task=10,seed_base=1000,suite_mode="sealed_randomized",camera="agentview",resolution=256,cell_count=100,motion="execute_motion",created_at_unix=time.time()); out=sys.argv[1]; tmp=out+".tmp"; open(tmp,"w").write(json.dumps(p,indent=2,sort_keys=True)+"\n"); os.replace(tmp,out)' "$RUN_CONTRACT"
fi

# The no-motion preflight runs in this same allocation before any action. It
# writes only below the run's preflight subtree, leaving the authoritative
# 100-cell result root untouched.
"$PYTHON" -m vla_benchmarking.arrow_policy.evaluation.run_sealed100 \
  --output-root "$PREFLIGHT_ROOT" --checkpoint "$CHECKPOINT" \
  --source-checkpoint "$SOURCE_CHECKPOINT" --base-policy "$BASE_POLICY" \
  --reference-root "$REFERENCE_ROOT" --preflight-only
resume_args=()
if [[ -e "$STATUS_PATH" || -e "$OUTPUT_ROOT/arrow_pick_place_matrix_manifest.json" || -e "$OUTPUT_ROOT/arrow_pick_place_matrix_manifest.jsonl" ]]; then resume_args=(--resume); fi
"$PYTHON" -m vla_benchmarking.arrow_policy.evaluation.run_sealed100 \
  --output-root "$OUTPUT_ROOT" --checkpoint "$CHECKPOINT" \
  --source-checkpoint "$SOURCE_CHECKPOINT" --base-policy "$BASE_POLICY" \
  --reference-root "$REFERENCE_ROOT" --execute-motion "${resume_args[@]}"
[[ -f "$STATUS_PATH" ]] || die "status manifest is missing: $STATUS_PATH"
[[ -f "$SUMMARY_PATH" ]] || die "final summary is missing: $SUMMARY_PATH"
"$PYTHON" -c 'import json,sys; c=json.load(open(sys.argv[1])).get("cells"); assert isinstance(c,list) and len(c)==100, len(c) if isinstance(c,list) else None' "$STATUS_PATH"
