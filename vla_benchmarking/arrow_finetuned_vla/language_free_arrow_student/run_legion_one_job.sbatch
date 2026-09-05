#!/usr/bin/env bash
# One-job, resumable Legion launcher for the language-free arrow policy.
# All checkpoints, logs, metadata, smoke output, and results live below one
# RUN_ROOT on SCRATCH_FLASH and are mirrored to ARCHIVE_ROOT on HOME.
#
# Required: REPO_ROOT, ARROW_POLICY_EXPECTED_COMMIT, ARROW_POLICY_RUN_ID,
# ARROW_DATASET_ROOT, ARROW_TEACHER_DATASET_ROOT, ARROW_TEACHER_CHECKPOINT, ARROW_INITIAL_CHECKPOINT,
# ARROW_BASE_POLICY.  The checkpoint arguments may be full snapshots or PEFT
# adapter directories; the latter are merged into this pinned local base.
# Resume by submitting another job with the same ARROW_POLICY_RUN_ID.

#SBATCH --job-name=arrow_policy_full
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

die() { printf 'arrow policy job: %s\n' "$*" >&2; exit 2; }

[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit with sbatch; direct execution is refused'
[[ -n "${REPO_ROOT:-}" ]] || die 'REPO_ROOT is required'
[[ "${ARROW_POLICY_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die 'ARROW_POLICY_EXPECTED_COMMIT must be a lowercase 40-character SHA'
[[ "${ARROW_POLICY_RUN_ID:-}" =~ ^[A-Za-z0-9_.-]+$ ]] || die 'ARROW_POLICY_RUN_ID is required and contains unsafe characters'
[[ -n "${ARROW_DATASET_ROOT:-}" ]] || die 'ARROW_DATASET_ROOT is required'
[[ -n "${ARROW_TEACHER_DATASET_ROOT:-}" ]] || die 'ARROW_TEACHER_DATASET_ROOT is required'
[[ -n "${ARROW_TEACHER_CHECKPOINT:-}" ]] || die 'ARROW_TEACHER_CHECKPOINT is required'
[[ -n "${ARROW_INITIAL_CHECKPOINT:-}" ]] || die 'ARROW_INITIAL_CHECKPOINT is required'
[[ -n "${ARROW_BASE_POLICY:-}" ]] || die 'ARROW_BASE_POLICY is required'
[[ "$ARROW_DATASET_ROOT" = /* && "$ARROW_TEACHER_DATASET_ROOT" = /* && "$ARROW_TEACHER_CHECKPOINT" = /* && "$ARROW_INITIAL_CHECKPOINT" = /* && "$ARROW_BASE_POLICY" = /* ]] || die 'dataset and checkpoint paths must be absolute'

REPO_ROOT="$(cd "$REPO_ROOT" 2>/dev/null && pwd -P)" || die 'cannot resolve REPO_ROOT'
[[ -d "$REPO_ROOT/.git" ]] || die 'REPO_ROOT is not a Git checkout'
actual_root="$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null)" || die 'cannot read checkout root'
actual_root="$(cd "$actual_root" && pwd -P)"
[[ "$actual_root" == "$REPO_ROOT" ]] || die 'REPO_ROOT does not resolve to its worktree root'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] || die 'shared checkout is not an isolated run checkout'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)" || die 'cannot read checkout commit'
[[ "$actual_commit" == "$ARROW_POLICY_EXPECTED_COMMIT" ]] || die "checkout commit $actual_commit differs from expected"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'REPO_ROOT is dirty'

PYTHON="${PYTHON:-$HOME/.conda/envs/embodiment-smolvla-py312/bin/python}"
TRAINER_MODULE="${ARROW_TRAINER_MODULE:-vla_benchmarking.arrow_finetuned_vla.language_free_arrow_student.arrow_lerobot_impl}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/EmbodimentSemantic_runtime/config}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/arrow_finetuned_vla/language_free_arrow_student/${ARROW_POLICY_RUN_ID}}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/arrow_finetuned_vla/language_free_arrow_student/${ARROW_POLICY_RUN_ID}}"
[[ "$RUN_ROOT" = /* && "$ARCHIVE_ROOT" = /* ]] || die 'run/archive roots must be absolute paths'
command -v realpath >/dev/null 2>&1 || die 'realpath is required'
RUN_ROOT="$(realpath -m -- "$RUN_ROOT")" || die 'cannot canonicalize RUN_ROOT'
ARCHIVE_ROOT="$(realpath -m -- "$ARCHIVE_ROOT")" || die 'cannot canonicalize ARCHIVE_ROOT'
[[ "$RUN_ROOT" != "$ARCHIVE_ROOT" ]] || die 'RUN_ROOT and ARCHIVE_ROOT must differ'
case "$RUN_ROOT" in "$ARCHIVE_ROOT"|"$ARCHIVE_ROOT"/*) die 'RUN_ROOT cannot be inside ARCHIVE_ROOT' ;; esac
case "$ARCHIVE_ROOT" in "$RUN_ROOT"|"$RUN_ROOT"/*) die 'ARCHIVE_ROOT cannot be inside RUN_ROOT' ;; esac
[[ -d "$ARROW_DATASET_ROOT" ]] || die "ARROW_DATASET_ROOT is not a directory: $ARROW_DATASET_ROOT"
[[ -d "$ARROW_TEACHER_DATASET_ROOT" ]] || die "ARROW_TEACHER_DATASET_ROOT is not a directory: $ARROW_TEACHER_DATASET_ROOT"
[[ -e "$ARROW_TEACHER_CHECKPOINT" ]] || die "teacher checkpoint is missing: $ARROW_TEACHER_CHECKPOINT"
[[ -e "$ARROW_INITIAL_CHECKPOINT" ]] || die "initial checkpoint is missing: $ARROW_INITIAL_CHECKPOINT"
[[ -d "$ARROW_BASE_POLICY" && -f "$ARROW_BASE_POLICY/config.json" ]] || die "base policy snapshot is missing config.json: $ARROW_BASE_POLICY"
[[ -d "$LIBERO_CONFIG_PATH" && -f "$LIBERO_CONFIG_PATH/config.yaml" ]] || die 'LIBERO_CONFIG_PATH must contain config.yaml'
ARROW_DATASET_ROOT="$(realpath -e -- "$ARROW_DATASET_ROOT")" || die 'cannot canonicalize ARROW_DATASET_ROOT'
ARROW_TEACHER_DATASET_ROOT="$(realpath -e -- "$ARROW_TEACHER_DATASET_ROOT")" || die 'cannot canonicalize ARROW_TEACHER_DATASET_ROOT'
ARROW_TEACHER_CHECKPOINT="$(realpath -e -- "$ARROW_TEACHER_CHECKPOINT")" || die 'cannot canonicalize ARROW_TEACHER_CHECKPOINT'
ARROW_INITIAL_CHECKPOINT="$(realpath -e -- "$ARROW_INITIAL_CHECKPOINT")" || die 'cannot canonicalize ARROW_INITIAL_CHECKPOINT'
ARROW_BASE_POLICY="$(realpath -e -- "$ARROW_BASE_POLICY")" || die 'cannot canonicalize ARROW_BASE_POLICY'
LIBERO_CONFIG_PATH="$(realpath -e -- "$LIBERO_CONFIG_PATH")" || die 'cannot canonicalize LIBERO_CONFIG_PATH'

mkdir -p "$RUN_ROOT" "$ARCHIVE_ROOT" "$RUN_ROOT/logs" "$RUN_ROOT/stages" "$RUN_ROOT/checkpoints" "$RUN_ROOT/cache"

# A stable run ID maps to one mutable checkpoint tree. Serialize every restore,
# stage marker, checkpoint, and archive operation so a repeated submission
# cannot corrupt latest.json or mix two workers' snapshots.
command -v flock >/dev/null 2>&1 || die 'flock is required for run serialization'
exec 9>"$RUN_ROOT/.run.lock"
flock -n 9 || die "run ID is already active: $ARROW_POLICY_RUN_ID"

restore_archive() {
  [[ -d "$ARCHIVE_ROOT" ]] || return 0
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --update --exclude='.run.lock' -- "$ARCHIVE_ROOT/" "$RUN_ROOT/"
  else
    copy_tree_without_run_lock "$ARCHIVE_ROOT" "$RUN_ROOT"
  fi
}

# ``cp -a source/. destination/`` has no portable exclude flag. Keep the
# fallback lock-safe so a host without rsync cannot overwrite the lock file
# held by this process during resume or archival.
copy_tree_without_run_lock() {
  local source="$1" destination="$2" item relative target
  while IFS= read -r -d '' item; do
    relative="${item#"$source"/}"
    [[ "$relative" == ".run.lock" || "$relative" == */.run.lock ]] && continue
    target="$destination/$relative"
    if [[ -d "$item" && ! -L "$item" ]]; then
      mkdir -p "$target"
    else
      mkdir -p "$(dirname "$target")"
      cp -a -- "$item" "$target"
    fi
  done < <(find "$source" -mindepth 1 -print0)
}

copy_slurm_logs() {
  local source base extension
  for extension in out err; do
    source="$HOME/EmbodimentSemantic_runtime/operator/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${extension}"
    if [[ -f "$source" ]]; then
      base="$RUN_ROOT/logs/slurm_${SLURM_JOB_ID}.${extension}"
      cp -f -- "$source" "$base" || return 1
    fi
  done
}

archive_snapshot() {
  local reason="${1:-snapshot}" rc=0
  copy_slurm_logs || rc=$?
  mkdir -p "$ARCHIVE_ROOT" || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    if command -v rsync >/dev/null 2>&1; then rsync -a --exclude='.run.lock' -- "$RUN_ROOT/" "$ARCHIVE_ROOT/" || rc=$?
    else copy_tree_without_run_lock "$RUN_ROOT" "$ARCHIVE_ROOT" || rc=$?; fi
  fi
  if [[ "$rc" -eq 0 ]]; then
    { printf 'schema=arrow-policy-archive-v1\n'; printf 'reason=%s\n' "$reason"; printf 'job_id=%s\n' "$SLURM_JOB_ID"; printf 'job_name=%s\n' "$SLURM_JOB_NAME"; printf 'run_root=%s\n' "$RUN_ROOT"; printf 'archive_root=%s\n' "$ARCHIVE_ROOT"; printf 'archived_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$ARCHIVE_ROOT/archive_status.env" || rc=$?
  fi
  return "$rc"
}

archive_on_exit() {
  local workload_rc=$? archive_rc=0
  trap - EXIT INT TERM HUP
  set +e
  archive_snapshot "job_exit_${workload_rc}" || archive_rc=$?
  if [[ "$workload_rc" -eq 0 && "$archive_rc" -ne 0 ]]; then workload_rc="$archive_rc"; fi
  exit "$workload_rc"
}
trap archive_on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

restore_archive || die 'could not restore the durable archive into scratch'

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
[[ -x "$PYTHON" ]] || die "Python executable is missing: $PYTHON"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
# The shared Legion environment contains optional pytest entry-point plugins
# whose dependency versions are not compatible with Python 3.12 (Hydra is one
# example).  The experiment only needs the package's own tests, so keep the
# preflight deterministic and isolated from unrelated site-wide plugins.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH
export TMPDIR="$RUN_ROOT/tmp"
# The nested SmolVLM constructor resolves HuggingFaceTB/SmolVLM2-500M-Instruct
# by model ID rather than through the outer policy snapshot. Legion already has
# a durable shared Hub cache; reuse it read-only and force offline resolution so
# an interrupted run cannot float to a newly-published model revision. New
# experiment outputs and checkpoints still remain under RUN_ROOT.
export HF_HOME="${ARROW_HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export TORCH_HOME="$RUN_ROOT/cache/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"

# Bind the run to the same immutable base revision used by the existing
# SmolVLA/LoRA experiments. This prevents an adapter from being merged into a
# look-alike snapshot with silently different VLM weights.
"$PYTHON" - "$ARROW_BASE_POLICY/base_snapshot_manifest.json" <<'PY'
import json, sys
from pathlib import Path
manifest = Path(sys.argv[1])
if not manifest.is_file():
    raise SystemExit(f"base snapshot manifest is missing: {manifest}")
data = json.loads(manifest.read_text(encoding="utf-8"))
expected = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
if data.get("revision") != expected:
    raise SystemExit(f"base snapshot revision {data.get('revision')!r} != {expected}")
PY

TRAINER_FLAGS_RAW="${ARROW_TRAINER_FLAGS:-}"
CONTRACT_PATH="$RUN_ROOT/run_contract.json"
"$PYTHON" - "$CONTRACT_PATH" "$ARROW_POLICY_RUN_ID" "$actual_commit" "$TRAINER_MODULE" "$ARROW_DATASET_ROOT" "$ARROW_TEACHER_DATASET_ROOT" "$ARROW_TEACHER_CHECKPOINT" "$ARROW_INITIAL_CHECKPOINT" "$ARROW_BASE_POLICY" "$LIBERO_CONFIG_PATH" "$TRAINER_FLAGS_RAW" <<'PY'
import hashlib, json, os, sys, tempfile
from pathlib import Path
out, run_id, commit, trainer, dataset, teacher_dataset, teacher, initial, base, libero, flags = sys.argv[1:]

def digest(path: Path, role: str):
    if not path.exists():
        return None
    if path.is_file():
        return {"kind": "file", "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}

    # The inventory records every dataset file, and payload hashes detect
    # same-size mutations. The full dataset is intentionally hashed at contract
    # creation/resume startup because the paired LIBERO inputs are part of the
    # experiment identity. Immutable base manifests still avoid rereading the
    # multi-GB model files.
    files = [p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts]
    inventory = []
    for item in sorted(files):
        relative = item.relative_to(path).as_posix()
        inventory.append({"path": relative, "size": item.stat().st_size})

    manifest_path = path / "base_snapshot_manifest.json"
    manifest_hashes = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest.get("files"), dict):
            manifest_hashes = dict(manifest["files"])
            actual_names = {entry["path"] for entry in inventory if entry["path"] != "base_snapshot_manifest.json"}
            if actual_names != set(manifest_hashes):
                raise SystemExit(f"{role} manifest file inventory differs from the directory")

    metadata_names = {
        "meta/info.json", "meta/stats.json", "meta/tasks.parquet", "config.json",
        "adapter_config.json", "train_config.json", "base_snapshot_manifest.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
    }
    hashed_files = []
    for item in sorted(files):
        relative = item.relative_to(path).as_posix()
        if relative in manifest_hashes:
            # The immutable base manifest authenticates multi-GB model files;
            # reuse its digest instead of rereading them during every resume.
            hashed_files.append({"path": relative, "sha256": manifest_hashes[relative], "source": "authenticated_manifest"})
        elif role == "dataset" or role == "checkpoint" or role == "base_policy" or relative in metadata_names:
            hashed_files.append({"path": relative, "sha256": hashlib.sha256(item.read_bytes()).hexdigest(), "source": "local_hash"})
    return {"kind": "directory", "inventory": inventory, "hashed_files": hashed_files}

contract = {
    "schema": "arrow-policy-run-contract-v1",
    "run_id": run_id,
    "commit": commit,
    "trainer_module": trainer,
    "trainer_flags": flags,
    "base_revision": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
    "paths": {
        "dataset": dataset,
        "teacher_dataset": teacher_dataset,
        "teacher_checkpoint": teacher,
        "initial_checkpoint": initial,
        "base_policy": base,
        "libero_config": libero,
    },
    "identity_files": {
        "dataset": digest(Path(dataset), "dataset"),
        "teacher_dataset": digest(Path(teacher_dataset), "dataset"),
        "teacher_checkpoint": digest(Path(teacher), "checkpoint"),
        "initial_checkpoint": digest(Path(initial), "checkpoint"),
        "base_policy": digest(Path(base), "base_policy"),
        "libero_config": digest(Path(libero), "config"),
    },
    "stages": ["A_distill_bridge", "B_action_bridge", "C_joint_action", "D_full_student"],
}
target = Path(out)
encoded = json.dumps(contract, indent=2, sort_keys=True) + "\n"
if target.exists():
    existing = json.loads(target.read_text(encoding="utf-8"))
    if existing != contract:
        raise SystemExit("run contract mismatch; use a new ARROW_POLICY_RUN_ID for changed code, inputs, or flags")
else:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
Path(str(target) + ".sha256").write_text(hashlib.sha256(encoded.encode()).hexdigest() + "\n", encoding="utf-8")
PY
RUN_CONTRACT_SHA="$(sha256sum "$CONTRACT_PATH" | awk '{print $1}')"

command -v nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi is unavailable in the compute allocation'
nvidia-smi -L | tee "$RUN_ROOT/gpu_inventory.txt"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))' | tee "$RUN_ROOT/cuda_smoke.txt"

# Persist exact code, data, checkpoint, host, and job provenance before work.
# The first attempt remains immutable; subsequent retries append a new attempt
# record under the same run folder instead of erasing the original evidence.
PROVENANCE_ATTEMPT="$RUN_ROOT/provenance_attempt_${SLURM_JOB_ID}.json"
"$PYTHON" - "$PROVENANCE_ATTEMPT" "$RUN_ROOT/provenance.json" "$RUN_ROOT/run_contract.json" "$REPO_ROOT" "$actual_commit" "$PYTHON" "$TRAINER_MODULE" "$RUN_ROOT" "$ARCHIVE_ROOT" "$ARROW_DATASET_ROOT" "$ARROW_TEACHER_DATASET_ROOT" "$ARROW_TEACHER_CHECKPOINT" "$ARROW_INITIAL_CHECKPOINT" "$ARROW_BASE_POLICY" "$LIBERO_CONFIG_PATH" <<'PY'
import json, os, platform, socket, subprocess, sys
from pathlib import Path
out, initial_out, contract_path, repo, commit, python, trainer, run_root, archive_root, dataset, teacher_dataset, teacher, initial, base, libero = sys.argv[1:]
def run(*args):
    try: return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc: return f"UNAVAILABLE: {exc}"
payload = {
    "schema": "arrow-policy-legion-provenance-v1",
    "job": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST", "SLURM_SUBMIT_DIR")},
    "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version},
    "gpu": run("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
    "repo": {"root": repo, "commit": commit, "status": run("git", "-C", repo, "status", "--short", "--branch")},
    "entrypoint": trainer,
    "paths": {"run_root": run_root, "archive_root": archive_root, "dataset_root": dataset, "teacher_dataset_root": teacher_dataset, "teacher_checkpoint": teacher, "initial_checkpoint": initial, "base_policy": base, "libero_config_path": libero},
    "stages": ["A_distill_bridge", "B_action_bridge", "C_joint_action", "D_full_student"],
    "resume_contract": "same run ID restores archive into scratch and skips completed stage markers",
    "run_contract": str(Path(contract_path).resolve()),
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not Path(initial_out).exists():
    Path(initial_out).write_text(Path(out).read_text(encoding="utf-8"), encoding="utf-8")
PY
sha256sum "$PROVENANCE_ATTEMPT" > "$PROVENANCE_ATTEMPT.sha256"
{
  sha256sum "$REPO_ROOT/vla_benchmarking/arrow_finetuned_vla/language_free_arrow_student/run_legion_one_job.sbatch"
  sha256sum "$REPO_ROOT/vla_benchmarking/arrow_finetuned_vla/language_free_arrow_student/config.py"
  sha256sum "$REPO_ROOT/vla_benchmarking/arrow_finetuned_vla/language_free_arrow_student/bridge.py"
  if [[ -f "$REPO_ROOT/vla_benchmarking/arrow_finetuned_vla/language_free_arrow_student/arrow_lerobot_impl.py" ]]; then
    sha256sum "$REPO_ROOT/vla_benchmarking/arrow_finetuned_vla/language_free_arrow_student/arrow_lerobot_impl.py"
  fi
} > "$RUN_ROOT/code_sha256.txt"
archive_snapshot preflight_start || die 'initial archive snapshot failed'

run_logged() {
  local label="$1"; shift
  local log="$RUN_ROOT/logs/${label}.log" rc
  printf '\n===== %s (%s) =====\n' "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log"
  set +e
  "$@" 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}

# Cheap package checks always run before the trainer, including on resume.
run_logged bridge_tests "$PYTHON" -m pytest -q "$REPO_ROOT/vla_benchmarking/arrow_finetuned_vla/language_free_arrow_student/tests"
archive_snapshot bridge_tests || die 'bridge-test archive failed'
run_logged bridge_smoke "$PYTHON" -m vla_benchmarking.arrow_finetuned_vla.language_free_arrow_student.smoke
archive_snapshot bridge_smoke || die 'bridge-smoke archive failed'
"$PYTHON" -c "import importlib.util; assert importlib.util.find_spec('$TRAINER_MODULE') is not None, 'trainer module is missing: $TRAINER_MODULE'"

trainer_flags=()
if [[ -n "${ARROW_TRAINER_FLAGS:-}" ]]; then read -r -a trainer_flags <<< "$ARROW_TRAINER_FLAGS"; fi
printf '%s\n' "${ARROW_TRAINER_FLAGS:-}" > "$RUN_ROOT/trainer_flags.txt"

# Real checkpoint/data smoke before any long stage. It loads both merged
# policies, consumes one paired 50-step batch, and performs one finite update
# through A/B/C/D. A failure exits before expensive training and is archived.
smoke_args=(
  -m "$TRAINER_MODULE" --smoke --run-root "$RUN_ROOT" --output-root "$RUN_ROOT"
  --checkpoint-root "$RUN_ROOT/checkpoints" --dataset-root "$ARROW_DATASET_ROOT" --teacher-dataset-root "$ARROW_TEACHER_DATASET_ROOT"
  --teacher-checkpoint "$ARROW_TEACHER_CHECKPOINT" --initial-checkpoint "$ARROW_INITIAL_CHECKPOINT" --base-policy "$ARROW_BASE_POLICY"
  --libero-config-path "$LIBERO_CONFIG_PATH" --device cuda
)
run_logged integration_smoke "$PYTHON" "${smoke_args[@]}" "${trainer_flags[@]}" || die 'real checkpoint/data integration smoke failed; no long stage was started'
[[ -s "$RUN_ROOT/integration_smoke.json" ]] || die 'integration smoke did not write its result marker'
archive_snapshot integration_smoke || die 'integration-smoke archive failed'

stages=(A_distill_bridge B_action_bridge C_joint_action D_full_student)
stage_targets=(5000 5000 10000 10000)
for stage_index in "${!stages[@]}"; do
  stage="${stages[$stage_index]}"
  target_updates="${stage_targets[$stage_index]}"
  marker="$RUN_ROOT/stages/${stage}.complete.json"
  if [[ -s "$marker" ]]; then
    "$PYTHON" - "$marker" "$stage" "$target_updates" "$RUN_ROOT/checkpoints" "$RUN_CONTRACT_SHA" <<'PY'
import hashlib, json, sys
from pathlib import Path
marker, expected_stage, target_updates, checkpoint_root, contract_sha = sys.argv[1:]
target_updates = int(target_updates)
data = json.loads(Path(marker).read_text(encoding="utf-8"))
if data.get("schema") != "arrow-policy-stage-v1" or data.get("stage") != expected_stage:
    raise SystemExit("invalid or mismatched stage marker")
if data.get("contract_sha256") != contract_sha:
    raise SystemExit("stage marker belongs to a different run contract")
if Path(data.get("checkpoint_root", "")).resolve() != Path(checkpoint_root).resolve():
    raise SystemExit("stage marker checkpoint root differs from this run")
manifest_path = Path(data.get("checkpoint_manifest", "")).resolve()
stage_root = Path(checkpoint_root).resolve() / expected_stage
update_dir = manifest_path.parent
if (
    update_dir.parent != stage_root
    or manifest_path.name != "manifest.json"
    or not update_dir.name.startswith("update_")
    or not update_dir.name[7:].isdigit()
):
    raise SystemExit("stage marker checkpoint manifest is outside its stage directory")
if not manifest_path.is_file():
    raise SystemExit("stage marker checkpoint manifest is missing")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("stage") != expected_stage or int(manifest.get("update", -1)) < target_updates:
    raise SystemExit("stage marker checkpoint has the wrong stage or insufficient updates")
if update_dir.name != f"update_{int(manifest['update']):07d}":
    raise SystemExit("stage marker update directory does not match checkpoint manifest")
files = manifest.get("files")
if not isinstance(files, dict) or not files:
    raise SystemExit("stage checkpoint manifest has no file hashes")
for name, expected in files.items():
    child = manifest_path.parent / name
    if Path(name).is_absolute() or ".." in Path(name).parts or not child.is_file():
        raise SystemExit(f"stage checkpoint file is invalid or missing: {name}")
    if hashlib.sha256(child.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"stage checkpoint checksum mismatch: {name}")
if data.get("update") != int(manifest["update"]):
    raise SystemExit("stage marker update does not match checkpoint manifest")
if data.get("manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
    raise SystemExit("stage marker manifest hash does not match checkpoint manifest")
PY
    printf 'resuming: completed stage marker exists: %s\n' "$marker"
    archive_snapshot "resume_${stage}" || die "archive failed while resuming $stage"
    continue
  fi
  stage_args=(
    -m "$TRAINER_MODULE" --stage "$stage" --run-root "$RUN_ROOT" --output-root "$RUN_ROOT"
    --checkpoint-root "$RUN_ROOT/checkpoints" --dataset-root "$ARROW_DATASET_ROOT" --teacher-dataset-root "$ARROW_TEACHER_DATASET_ROOT"
    --teacher-checkpoint "$ARROW_TEACHER_CHECKPOINT" --initial-checkpoint "$ARROW_INITIAL_CHECKPOINT" --base-policy "$ARROW_BASE_POLICY"
    --libero-config-path "$LIBERO_CONFIG_PATH" --device cuda --updates "$target_updates" --resume
  )
  run_logged "train_${stage}" "$PYTHON" "${stage_args[@]}" "${trainer_flags[@]}" || die "trainer failed in stage $stage; partial state was archived"
  "$PYTHON" - "$marker" "$stage" "$target_updates" "$RUN_ROOT/checkpoints" "$RUN_CONTRACT_SHA" <<'PY'
import hashlib, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
marker, stage, target_updates, checkpoint_root, contract_sha = sys.argv[1:]
target_updates = int(target_updates)
stage_root = Path(checkpoint_root).resolve() / stage
valid = []
for manifest_path in stage_root.glob("update_*/manifest.json"):
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        update = int(manifest.get("update", -1))
        files = manifest.get("files")
        if manifest.get("stage") != stage or update < target_updates or not isinstance(files, dict) or not files:
            continue
        for name, expected in files.items():
            child = manifest_path.parent / name
            if Path(name).is_absolute() or ".." in Path(name).parts or not child.is_file():
                raise ValueError(f"invalid checkpoint file {name}")
            if hashlib.sha256(child.read_bytes()).hexdigest() != expected:
                raise ValueError(f"checkpoint checksum mismatch {name}")
        valid.append((update, manifest_path, manifest))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        continue
if not valid:
    raise SystemExit(f"no valid {stage} checkpoint manifest reached target update {target_updates}")
update, manifest_path, manifest = max(valid, key=lambda item: item[0])
payload = {"schema": "arrow-policy-stage-v1", "stage": stage, "target_updates": target_updates, "update": update, "checkpoint_root": str(Path(checkpoint_root).resolve()), "checkpoint_manifest": str(manifest_path.resolve()), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "contract_sha256": contract_sha, "completed_job_id": os.environ.get("SLURM_JOB_ID"), "completed_at_utc": datetime.now(timezone.utc).isoformat()}
target = Path(marker)
fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY
  archive_snapshot "completed_${stage}" || die "archive failed after stage $stage"
done

printf 'arrow policy experiment completed; run_root=%s archive_root=%s\n' "$RUN_ROOT" "$ARCHIVE_ROOT"
