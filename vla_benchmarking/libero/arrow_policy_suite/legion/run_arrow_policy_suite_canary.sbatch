#!/usr/bin/env bash
# Parameterized short native canary.  The job never chooses a checkpoint,
# controller, factory, or study config implicitly; every identity comes from
# the caller's sealed run contract.
#
# Required exports:
#   ARROW_SUITE_REPO_ROOT, ARROW_SUITE_EXPECTED_COMMIT,
#   ARROW_SUITE_CONFIG, ARROW_SUITE_FACTORY, ARROW_SUITE_POLICY (normal mode)
#   ARROW_SUITE_RUN_ROOT, ARROW_SUITE_ARCHIVE_ROOT
# Optional:
#   ARROW_SUITE_OPERATION (canary), ARROW_SUITE_STEPS, ARROW_SUITE_OUTPUT,
#   ARROW_SUITE_CHECKPOINT, ARROW_SUITE_CONTROLLER, ARROW_SUITE_PYTHON,
#   ARROW_SUITE_HF_CACHE (provide both explicitly for teacher-dependent policies),
#   ARROW_SUITE_GRAPH_CONTEXT_REVISION, ARROW_SUITE_TRACE_GEOMETRY_VARIANT,
#   ARROW_SUITE_GRAPH_FACTORY, and policy-specific Fast/Trace artifact inputs.
# Engineering-only environment smoke (no config/factory required):
#   ARROW_SUITE_ENGINEERING_SMOKE=1

#SBATCH --job-name=arrow_policy_canary
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-00:30:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%A.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%A.err

set -Eeuo pipefail
umask 027
die() { printf 'arrow policy suite canary: %s\n' "$*" >&2; exit 2; }

[[ -n "${SLURM_JOB_ID:-}" && "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || die 'submit with sbatch'
for name in ARROW_SUITE_REPO_ROOT ARROW_SUITE_RUN_ROOT ARROW_SUITE_ARCHIVE_ROOT; do
  [[ -n "${!name:-}" ]] || die "$name is required"
done
[[ "${ARROW_SUITE_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die 'ARROW_SUITE_EXPECTED_COMMIT must be a full SHA'
for name in ARROW_SUITE_RUN_ROOT ARROW_SUITE_ARCHIVE_ROOT; do
  value="${!name}"
  [[ "$value" = /* && "$value" != / && ! -L "$value" ]] || die "$name must be a non-root absolute non-symlink path"
done
command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v git >/dev/null 2>&1 || die 'git is required'

REPO_ROOT="$(realpath -e -- "$ARROW_SUITE_REPO_ROOT")" || die 'repository cannot be resolved'
[[ -e "$REPO_ROOT/.git" ]] || die 'repository is not a Git checkout'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)" || die 'cannot resolve repository SHA'
[[ "$actual_commit" == "$ARROW_SUITE_EXPECTED_COMMIT" ]] || die "repository SHA mismatch: $actual_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'repository checkout is dirty'
cd -- "$REPO_ROOT" || die 'cannot enter repository checkout'

if [[ -n "${ARROW_SUITE_PYTHON:-}" ]]; then
  [[ "$ARROW_SUITE_PYTHON" =~ ^/[A-Za-z0-9_./-]+$ && "$ARROW_SUITE_PYTHON" != *'/../'* && "$ARROW_SUITE_PYTHON" != */.. ]] || \
    die 'ARROW_SUITE_PYTHON must be a safe absolute Linux path'
  ARROW_SUITE_PYTHON_RESOLVED="$(realpath -e -- "$ARROW_SUITE_PYTHON")" || die 'ARROW_SUITE_PYTHON cannot be resolved'
  [[ "$ARROW_SUITE_PYTHON_RESOLVED" =~ ^/[A-Za-z0-9_./-]+$ && "$ARROW_SUITE_PYTHON_RESOLVED" != *'/../'* && "$ARROW_SUITE_PYTHON_RESOLVED" != */.. && ! -L "$ARROW_SUITE_PYTHON_RESOLVED" && -x "$ARROW_SUITE_PYTHON_RESOLVED" ]] || \
    die "ARROW_SUITE_PYTHON target is not a safe executable regular path: $ARROW_SUITE_PYTHON_RESOLVED"
fi
if [[ -n "${ARROW_SUITE_HF_CACHE:-}" ]]; then
  [[ "$ARROW_SUITE_HF_CACHE" =~ ^/[A-Za-z0-9_./-]+$ && "$ARROW_SUITE_HF_CACHE" != *'/../'* && "$ARROW_SUITE_HF_CACHE" != */.. ]] || \
    die 'ARROW_SUITE_HF_CACHE must be a safe absolute Linux path'
  [[ ! -L "$ARROW_SUITE_HF_CACHE" && -d "$ARROW_SUITE_HF_CACHE" ]] || die "ARROW_SUITE_HF_CACHE is not an existing regular directory: $ARROW_SUITE_HF_CACHE"
fi

ENGINEERING_SMOKE="${ARROW_SUITE_ENGINEERING_SMOKE:-0}"
operation="${ARROW_SUITE_OPERATION:-canary}"
if [[ "$ENGINEERING_SMOKE" != 1 ]]; then
  for name in ARROW_SUITE_CONFIG ARROW_SUITE_FACTORY ARROW_SUITE_POLICY; do
    [[ -n "${!name:-}" ]] || die "$name is required outside engineering smoke"
  done
  case "$ARROW_SUITE_POLICY" in
    teacher_only|arrow_together|arrow_on_call|arrow_apprentice|arrow_editor|arrow_minimal|arrow_minimal_runtime|arrow_minimal_learned|arrow_fast)
      [[ -n "${ARROW_SUITE_PYTHON:-}" && -n "${ARROW_SUITE_HF_CACHE:-}" ]] || \
        die 'teacher-dependent policies require explicit ARROW_SUITE_PYTHON and ARROW_SUITE_HF_CACHE'
      ;;
  esac
  case "$ARROW_SUITE_POLICY" in
    arrow_fast|arrow_trace)
      for name in ARROW_SUITE_GRAPH_FACTORY ARROW_SUITE_GRAPH_CONTEXT_REVISION \
        ARROW_SUITE_TEXT_GRAPH_TRIPLET ARROW_SUITE_VISUAL_ARROW; do
        [[ -n "${!name:-}" ]] || die "$name is required for $ARROW_SUITE_POLICY"
      done
      for name in ARROW_SUITE_TEXT_GRAPH_TRIPLET ARROW_SUITE_VISUAL_ARROW; do
        value="${!name}"
        [[ "$value" == @/* ]] || die "$name must be supplied as @/absolute/artifact.json"
        artifact_path="${value#@}"
        [[ "$artifact_path" != *'/../'* && "$artifact_path" != */.. && -f "$artifact_path" ]] || \
          die "$name artifact is unsafe or missing"
      done
      ;;
  esac
  case "$ARROW_SUITE_POLICY" in
    arrow_fast)
      for name in ARROW_SUITE_GRAPH_FACTORY ARROW_SUITE_FAST_ARTIFACT_KIND \
        ARROW_SUITE_FAST_ENCODER_REVISION ARROW_SUITE_FAST_ENCODER_SHA256 \
        ARROW_SUITE_FAST_ROUTER_REVISION ARROW_SUITE_FAST_ROUTER_SHA256 \
        ARROW_SUITE_FAST_SOURCE_MANIFEST_SHA256 ARROW_SUITE_FAST_VLA_MANIFEST_SHA256 \
        ARROW_SUITE_GRAPH_CONTEXT_REVISION; do
        [[ -n "${!name:-}" ]] || die "$name is required for arrow_fast"
      done
      [[ "$ARROW_SUITE_GRAPH_FACTORY" =~ ^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$ ]] || die 'ARROW_SUITE_GRAPH_FACTORY is invalid'
      [[ "$ARROW_SUITE_FAST_ARTIFACT_KIND" =~ ^[A-Za-z0-9_.:-]+$ ]] || die 'ARROW_SUITE_FAST_ARTIFACT_KIND is invalid'
      [[ "$ARROW_SUITE_FAST_ENCODER_REVISION" =~ ^[A-Za-z0-9_.:/-]+$ ]] || die 'ARROW_SUITE_FAST_ENCODER_REVISION is invalid'
      [[ "$ARROW_SUITE_FAST_ROUTER_REVISION" =~ ^[A-Za-z0-9_.:/-]+$ ]] || die 'ARROW_SUITE_FAST_ROUTER_REVISION is invalid'
      for name in ARROW_SUITE_FAST_ENCODER_SHA256 ARROW_SUITE_FAST_ROUTER_SHA256 \
        ARROW_SUITE_FAST_SOURCE_MANIFEST_SHA256 ARROW_SUITE_FAST_VLA_MANIFEST_SHA256; do
        [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || die "$name must be a full lowercase SHA-256"
      done
      [[ "$ARROW_SUITE_GRAPH_CONTEXT_REVISION" =~ ^[A-Za-z0-9_.:/-]+$ ]] || die 'ARROW_SUITE_GRAPH_CONTEXT_REVISION is invalid'
      ;;
    arrow_trace)
      for name in ARROW_SUITE_GRAPH_FACTORY ARROW_SUITE_TRACE_ROUTE_ARTIFACT \
        ARROW_SUITE_TRACE_CALIBRATION_ARTIFACT ARROW_SUITE_TRACE_GEOMETRY_VARIANT \
        ARROW_SUITE_GRAPH_CONTEXT_REVISION; do
        [[ -n "${!name:-}" ]] || die "$name is required for arrow_trace"
      done
      [[ "$ARROW_SUITE_GRAPH_FACTORY" =~ ^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$ ]] || die 'ARROW_SUITE_GRAPH_FACTORY is invalid'
      for name in ARROW_SUITE_TRACE_ROUTE_ARTIFACT ARROW_SUITE_TRACE_CALIBRATION_ARTIFACT; do
        value="${!name}"
        [[ "$value" =~ ^/[A-Za-z0-9_./-]+$ && "$value" != *'/../'* && "$value" != */.. ]] || die "$name must be a safe absolute Linux path"
      done
      [[ "$ARROW_SUITE_GRAPH_CONTEXT_REVISION" =~ ^[A-Za-z0-9_.:/-]+$ ]] || die 'ARROW_SUITE_GRAPH_CONTEXT_REVISION is invalid'
      ;;
  esac
  CONFIG="$(realpath -e -- "$ARROW_SUITE_CONFIG")" || die 'sealed config cannot be resolved'
  [[ -f "$CONFIG" ]] || die 'sealed config is not a file'
else
  operation=engineering_smoke
fi
RUN_ROOT="$(realpath -m -- "$ARROW_SUITE_RUN_ROOT")"
ARCHIVE_ROOT="$(realpath -m -- "$ARROW_SUITE_ARCHIVE_ROOT")"
case "$RUN_ROOT" in "$REPO_ROOT"|"$REPO_ROOT"/*) die 'run root overlaps repository' ;; esac
case "$ARCHIVE_ROOT" in "$REPO_ROOT"|"$REPO_ROOT"/*) die 'archive root overlaps repository' ;; esac
case "$RUN_ROOT" in "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'run/archive roots overlap' ;; esac
case "$ARCHIVE_ROOT" in "$RUN_ROOT"|"$RUN_ROOT"/*) die 'archive/run roots overlap' ;; esac
[[ ! -e "$RUN_ROOT" && ! -e "$ARCHIVE_ROOT" ]] || die 'run or archive directory already exists'
mkdir -p "$RUN_ROOT"

copy_tree() { local source="$1" destination="$2"; [[ -e "$source" ]] || return 0; mkdir -p "$destination"; if command -v rsync >/dev/null 2>&1; then rsync -a -- "$source/" "$destination/"; else cp -a -- "$source/." "$destination/"; fi; }
archive_on_exit() {
  local workload_rc=$? archive_rc=0
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT" || archive_rc=$?
  copy_tree "$RUN_ROOT" "$ARCHIVE_ROOT/run" || archive_rc=$?
  log_root="${SLURM_LOG_ROOT:-$HOME/EmbodimentSemantic_runtime/operator/logs}"
  mkdir -p "$ARCHIVE_ROOT/slurm_logs" || archive_rc=$?
  for suffix in out err; do
    log_job_id="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
    log_path="$log_root/${SLURM_JOB_NAME}_${log_job_id}.$suffix"
    [[ -f "$log_path" ]] && cp -p -- "$log_path" "$ARCHIVE_ROOT/slurm_logs/" || true
  done
  if [[ -d "$ARCHIVE_ROOT" ]]; then
    find "$ARCHIVE_ROOT" -type f ! -name inventory.sha256 ! -name tree_sha256 -print0 | sort -z | xargs -0 -r sha256sum > "$ARCHIVE_ROOT/inventory.sha256" || archive_rc=$?
    sha256sum "$ARCHIVE_ROOT/inventory.sha256" | awk '{print tolower($1)}' > "$ARCHIVE_ROOT/tree_sha256" || archive_rc=$?
    status=PRESERVED_FAILURE
    if [[ "$workload_rc" -eq 0 && -s "$ARCHIVE_ROOT/run/run_manifest.json" && -s "$ARCHIVE_ROOT/inventory.sha256" ]]; then status=VERIFIED; fi
    printf 'schema=arrow_policy_suite_canary_archive.v1\nstatus=%s\njob_id=%s\nrepo_commit=%s\nrun_root=%s\narchive_root=%s\n' "$status" "$SLURM_JOB_ID" "$actual_commit" "$RUN_ROOT" "$ARCHIVE_ROOT" > "$ARCHIVE_ROOT/archive_status.env" || archive_rc=$?
  fi
  [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]] && workload_rc=$archive_rc
  exit "$workload_rc"
}
trap archive_on_exit EXIT

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
PYTHON="${ARROW_SUITE_PYTHON_RESOLVED:-${ARROW_SUITE_PYTHON:-$HOME/.conda/envs/embodiment-smolvla-py312/bin/python}}"
[[ -x "$PYTHON" ]] || die "Python is missing: $PYTHON"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
if [[ -n "${ARROW_SUITE_HF_CACHE:-}" ]]; then
  HF_CACHE="$(realpath -e -- "$ARROW_SUITE_HF_CACHE")" || die 'HF cache cannot be resolved'
  [[ "$HF_CACHE" != / && ! -L "$HF_CACHE" ]] || die 'HF cache must resolve to a non-root non-symlink directory'
  HF_MODEL_ROOT="$HF_CACHE/transformers/models--allenai--MolmoPoint-8B"
  [[ -d "$HF_MODEL_ROOT/snapshots" ]] || die "pinned MolmoPoint cache is missing snapshots: $HF_MODEL_ROOT"
  find "$HF_MODEL_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit | grep -q . || \
    die "pinned MolmoPoint cache has no snapshot: $HF_MODEL_ROOT/snapshots"
  export HF_HOME="$HF_CACHE"
  export HF_HUB_CACHE="$HF_CACHE/hub"
  export TRANSFORMERS_CACHE="$HF_CACHE/transformers"
  export HF_MODULES_CACHE="$HF_CACHE/modules"
  export HF_HUB_DISABLE_TELEMETRY=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
else
  export HF_HUB_DISABLE_TELEMETRY=1
fi
nvidia-smi -L >/dev/null || die 'CUDA device is unavailable'
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))' || die 'Torch CUDA preflight failed'
if [[ -n "${ARROW_SUITE_PYTHON:-}" ]]; then
  "$PYTHON" - <<'PY'
from importlib.metadata import version
expected = {"transformers": "4.57.1"}
for package, wanted in expected.items():
    observed = version(package)
    if observed != wanted:
        raise SystemExit(f"explicit Arrow teacher runtime requires {package}=={wanted}, found {observed}")
print("verified explicit Arrow teacher runtime", version("transformers"))
PY
fi

if [[ "$ENGINEERING_SMOKE" != 1 && -n "${ARROW_SUITE_CONTROLLER:-}" ]]; then
  [[ -f "$ARROW_SUITE_CONTROLLER" ]] || die "controller config is not a file: $ARROW_SUITE_CONTROLLER"
  # The controller contract is keyed by the hash of the fully expanded
  # semantic configuration, not by the raw JSON file bytes.
  export ARROW_SUITE_CONTROLLER_HASH="$("$PYTHON" -c 'import sys; from vla_benchmarking.libero.arrow_grasp_controller.configs import load_controller_config; print(load_controller_config(sys.argv[1])["config_hash"])' "$ARROW_SUITE_CONTROLLER")"
  [[ "$ARROW_SUITE_CONTROLLER_HASH" =~ ^[0-9a-f]{64}$ ]] || die 'resolved controller config hash is invalid'
fi

steps="${ARROW_SUITE_STEPS:-3}"
output="${ARROW_SUITE_OUTPUT:-$RUN_ROOT/execution_receipt.json}"
if [[ "$ENGINEERING_SMOKE" == 1 ]]; then
  "$PYTHON" - "$RUN_ROOT/run_manifest.json" <<'PY'
import importlib, json, pathlib, sys
import torch
required = (
    "vla_benchmarking.libero.arrow_policy_suite.contracts",
    "vla_benchmarking.libero.arrow_policy_suite.native_factory",
    "vla_benchmarking.libero.arrow_policy_suite.native_executor",
)
for name in required:
    importlib.import_module(name)
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "schema": "arrow_policy_suite.engineering_smoke.v1",
    "status": "ENGINEERING_SMOKE_VERIFIED",
    "experiment_evidence": False,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()),
    "imports": list(required),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("ENGINEERING_SMOKE_VERIFIED")
PY
  touch "$RUN_ROOT/COMPLETED"
  exit 0
fi
args=( -m vla_benchmarking.libero.arrow_policy_suite.cli "$operation"
  --config "$CONFIG" --factory "$ARROW_SUITE_FACTORY" --policy "$ARROW_SUITE_POLICY"
  --run-dir "$RUN_ROOT/native_run" --output "$output" --steps "$steps" --execute )
[[ -n "${ARROW_SUITE_CHECKPOINT:-}" ]] && args+=(--checkpoint "$ARROW_SUITE_CHECKPOINT")
[[ -n "${ARROW_SUITE_CONTROLLER:-}" ]] && args+=(--controller "$ARROW_SUITE_CONTROLLER")
[[ -n "${ARROW_SUITE_GRAPH_CONTEXT_REVISION:-}" ]] && args+=(--graph-context-revision "$ARROW_SUITE_GRAPH_CONTEXT_REVISION")
[[ -n "${ARROW_SUITE_TRACE_GEOMETRY_VARIANT:-}" ]] && args+=(--trace-geometry-variant "$ARROW_SUITE_TRACE_GEOMETRY_VARIANT")
"$PYTHON" "${args[@]}" | tee "$RUN_ROOT/cli_receipt.json"

"$PYTHON" - "$RUN_ROOT/cli_receipt.json" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("status") != "COMPLETED":
    raise SystemExit("native canary did not complete")
print("VERIFIED native Arrow policy canary")
PY
touch "$RUN_ROOT/COMPLETED"
