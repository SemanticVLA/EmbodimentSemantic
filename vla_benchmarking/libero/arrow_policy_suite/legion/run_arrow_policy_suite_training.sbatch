#!/usr/bin/env bash
# Sealed Legion training front door. It accepts only an exact release,
# immutable source/base artifacts, and fresh output/archive pairs.
#SBATCH --job-name=arrow_policy_training
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%A.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%A.err

set -Eeuo pipefail
umask 027
die() { printf 'arrow policy suite training: %s\n' "$*" >&2; exit 2; }

[[ -n "${SLURM_JOB_ID:-}" && "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || die 'submit with sbatch'
for name in ARROW_TRAIN_REPO_ROOT ARROW_TRAIN_EXPECTED_COMMIT ARROW_TRAIN_KIND \
  ARROW_TRAIN_SOURCE_ARTIFACT ARROW_TRAIN_SOURCE_SHA256 ARROW_TRAIN_BASE_CHECKPOINT \
  ARROW_TRAIN_BASE_SHA256 ARROW_TRAIN_OUTPUT_ROOT ARROW_TRAIN_ARCHIVE_ROOT; do
  [[ -n "${!name:-}" ]] || die "$name is required"
done
[[ "$ARROW_TRAIN_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'ARROW_TRAIN_EXPECTED_COMMIT must be a full lowercase SHA'
[[ "$ARROW_TRAIN_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || die 'ARROW_TRAIN_SOURCE_SHA256 must be a full lowercase SHA-256'
[[ "$ARROW_TRAIN_BASE_SHA256" =~ ^[0-9a-f]{64}$ ]] || die 'ARROW_TRAIN_BASE_SHA256 must be a full lowercase SHA-256'
[[ "$ARROW_TRAIN_KIND" =~ ^(editor|minimal_learned|apprentice)$ ]] || die 'ARROW_TRAIN_KIND is invalid'

safe_abs_path() {
  local label="$1" value="$2"
  [[ "$value" =~ ^/[A-Za-z0-9_./-]+$ && "$value" != / && "$value" != *'/../'* && "$value" != */.. ]] || die "$label must be a safe non-root absolute Linux path"
}
for name in ARROW_TRAIN_REPO_ROOT ARROW_TRAIN_SOURCE_ARTIFACT ARROW_TRAIN_BASE_CHECKPOINT \
  ARROW_TRAIN_OUTPUT_ROOT ARROW_TRAIN_ARCHIVE_ROOT; do
  safe_abs_path "$name" "${!name}"
done
[[ "$ARROW_TRAIN_OUTPUT_ROOT" != "$ARROW_TRAIN_ARCHIVE_ROOT" ]] || die 'output and archive roots must be distinct'
case "$ARROW_TRAIN_ARCHIVE_ROOT/" in
  "$ARROW_TRAIN_OUTPUT_ROOT/"* ) die 'archive root cannot be inside output root' ;;
esac
case "$ARROW_TRAIN_OUTPUT_ROOT/" in
  "$ARROW_TRAIN_ARCHIVE_ROOT/"* ) die 'output root cannot be inside archive root' ;;
esac
for artifact_name in ARROW_TRAIN_SOURCE_ARTIFACT ARROW_TRAIN_BASE_CHECKPOINT; do
  artifact_path="${!artifact_name}"
  case "$artifact_path/" in
    "$ARROW_TRAIN_OUTPUT_ROOT/"*|"$ARROW_TRAIN_ARCHIVE_ROOT/"*) die "$artifact_name cannot be inside an output or archive root" ;;
  esac
  case "$ARROW_TRAIN_OUTPUT_ROOT/" in
    "$artifact_path/"*) die "$artifact_name cannot contain an output root" ;;
  esac
  case "$ARROW_TRAIN_ARCHIVE_ROOT/" in
    "$artifact_path/"*) die "$artifact_name cannot contain an archive root" ;;
  esac
done

for name in ARROW_TRAIN_TASK_IDS ARROW_TRAIN_RESET_IDS ARROW_TRAIN_EPISODE_IDS; do
  value="${!name:-}"
  if [[ -n "$value" ]]; then
    IFS=: read -r -a values <<< "$value"
    for item in "${values[@]}"; do
      [[ "$item" =~ ^[A-Za-z0-9_.:/-]+$ ]] || die "$name contains an unsafe filter identity"
    done
  fi
done
for name in ARROW_TRAIN_SEED ARROW_TRAIN_INIT_STATE_INDEX; do
  if [[ -n "${!name:-}" ]]; then [[ "${!name}" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer"; fi
done
[[ "${ARROW_TRAIN_EPOCHS:-1}" =~ ^[1-9][0-9]{0,2}$ ]] || die 'ARROW_TRAIN_EPOCHS is invalid'
[[ "${ARROW_TRAIN_BATCH_SIZE:-32}" =~ ^[1-9][0-9]{0,3}$ ]] || die 'ARROW_TRAIN_BATCH_SIZE is invalid'
[[ "${ARROW_TRAIN_DEVICE:-cuda}" =~ ^(cpu|cuda|mps)$ ]] || die 'ARROW_TRAIN_DEVICE is invalid'
if [[ "$ARROW_TRAIN_KIND" == apprentice ]]; then
  [[ -n "${ARROW_TRAIN_MODEL_REVISION:-}" && "$ARROW_TRAIN_MODEL_REVISION" =~ ^[A-Za-z0-9_.:/-]+$ ]] || die 'ARROW_TRAIN_MODEL_REVISION is required and unsafe'
  [[ -n "${ARROW_TRAIN_PROCESSOR_REVISION:-}" && "$ARROW_TRAIN_PROCESSOR_REVISION" =~ ^[A-Za-z0-9_.:/-]+$ ]] || die 'ARROW_TRAIN_PROCESSOR_REVISION is required and unsafe'
fi

command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v git >/dev/null 2>&1 || die 'git is required'
[[ ! -e "$ARROW_TRAIN_OUTPUT_ROOT" && ! -L "$ARROW_TRAIN_OUTPUT_ROOT" ]] || die "output root already exists or is a symlink: $ARROW_TRAIN_OUTPUT_ROOT"
[[ ! -e "$ARROW_TRAIN_ARCHIVE_ROOT" && ! -L "$ARROW_TRAIN_ARCHIVE_ROOT" ]] || die "archive root already exists or is a symlink: $ARROW_TRAIN_ARCHIVE_ROOT"

RUN_ROOT="$ARROW_TRAIN_OUTPUT_ROOT"
ARCHIVE_ROOT="$ARROW_TRAIN_ARCHIVE_ROOT"
mkdir -p -- "$RUN_ROOT"
printf 'RUNNING\n' > "$RUN_ROOT/STARTED"
archive_exit() {
  local status=$?
  if (( status == 0 )); then printf 'COMPLETED\n' > "$RUN_ROOT/COMPLETED"; else printf 'FAILED:%s\n' "$status" > "$RUN_ROOT/FAILED"; fi
  mkdir -p -- "$ARCHIVE_ROOT/run"
  cp -a -- "$RUN_ROOT/." "$ARCHIVE_ROOT/run/"
  if (( status == 0 )); then printf 'COMPLETED\n' > "$ARCHIVE_ROOT/terminal_status"; else printf 'FAILED:%s\n' "$status" > "$ARCHIVE_ROOT/terminal_status"; fi
  exit "$status"
}
trap archive_exit EXIT

REPO_ROOT="$(realpath -e -- "$ARROW_TRAIN_REPO_ROOT")" || die 'repository cannot be resolved'
[[ -d "$REPO_ROOT/.git" || -f "$REPO_ROOT/.git" ]] || die 'repository is not a Git checkout'
actual_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)" || die 'cannot resolve repository SHA'
[[ "$actual_commit" == "$ARROW_TRAIN_EXPECTED_COMMIT" ]] || die "repository SHA mismatch: $actual_commit"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'repository checkout is dirty'

validate_immutable_tree() {
  local label="$1" path="$2" entry
  [[ ! -L "$path" && ( -f "$path" || -d "$path" ) ]] || die "$label must be an existing immutable file or directory: $path"
  if [[ -d "$path" ]]; then
    while IFS= read -r -d '' entry; do
      [[ ! -L "$entry" ]] || die "$label contains a symlink: $entry"
      [[ -d "$entry" || -f "$entry" ]] || die "$label contains a non-regular entry: $entry"
    done < <(find -P -- "$path" -mindepth 1 -print0)
  fi
}
validate_immutable_tree 'source artifact' "$ARROW_TRAIN_SOURCE_ARTIFACT"
validate_immutable_tree 'base checkpoint' "$ARROW_TRAIN_BASE_CHECKPOINT"

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
PYTHON="${ARROW_TRAIN_PYTHON:-$HOME/.conda/envs/embodiment-smolvla-py312/bin/python}"
[[ "$PYTHON" =~ ^/[A-Za-z0-9_./-]+$ && ! -L "$PYTHON" && -x "$PYTHON" ]] || die 'ARROW_TRAIN_PYTHON must be a safe executable path'
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$REPO_ROOT" || die 'cannot enter repository checkout'

artifact_sha256() {
  "$PYTHON" - "$1" <<'PY'
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if path.is_file():
    print(hashlib.sha256(path.read_bytes()).hexdigest())
elif path.is_dir():
    entries = []
    for child in sorted(item for item in path.rglob('*') if item.is_file()):
        if child.is_symlink():
            raise SystemExit(f'symlink in immutable tree: {child}')
        entries.append((child.relative_to(path).as_posix(), hashlib.sha256(child.read_bytes()).hexdigest()))
    print(hashlib.sha256(json.dumps(entries, separators=(',', ':')).encode()).hexdigest())
else:
    raise SystemExit(f'missing artifact: {path}')
PY
}
base_checkpoint_sha256() {
  "$PYTHON" - "$1" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
if not root.is_dir():
    raise SystemExit(f'base checkpoint tree is not a directory: {root}')
digest = hashlib.sha256()
for child in sorted(item for item in root.rglob('*') if item.is_file()):
    relative = child.relative_to(root)
    if relative.name == 'base_snapshot_manifest.json' or '.cache' in relative.parts:
        continue
    encoded = relative.as_posix().encode('utf-8')
    digest.update(len(encoded).to_bytes(8, 'big'))
    digest.update(encoded)
    with child.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
print(digest.hexdigest())
PY
}
source_actual="$(artifact_sha256 "$ARROW_TRAIN_SOURCE_ARTIFACT")" || die 'cannot hash source artifact'
base_actual="$(base_checkpoint_sha256 "$ARROW_TRAIN_BASE_CHECKPOINT")" || die 'cannot hash canonical base checkpoint tree'
[[ "$source_actual" == "$ARROW_TRAIN_SOURCE_SHA256" ]] || die "source artifact SHA mismatch: $source_actual"
[[ "$base_actual" == "$ARROW_TRAIN_BASE_SHA256" ]] || die "base checkpoint SHA mismatch: $base_actual"

"$PYTHON" - "$RUN_ROOT/launch_manifest.json" <<'PY'
import json, os, pathlib, sys
out = pathlib.Path(sys.argv[1])
payload = {
    'schema': 'arrow_policy_training_launch.v1',
    'status': 'RUNNING',
    'slurm_job_id': os.environ['SLURM_JOB_ID'],
    'git_commit': os.environ['ARROW_TRAIN_EXPECTED_COMMIT'],
    'train_kind': os.environ['ARROW_TRAIN_KIND'],
    'source_artifact': os.environ['ARROW_TRAIN_SOURCE_ARTIFACT'],
    'source_sha256': os.environ['ARROW_TRAIN_SOURCE_SHA256'],
    'base_checkpoint': os.environ['ARROW_TRAIN_BASE_CHECKPOINT'],
    'base_sha256': os.environ['ARROW_TRAIN_BASE_SHA256'],
    'task_ids': [x for x in os.environ.get('ARROW_TRAIN_TASK_IDS', '').split(':') if x],
    'reset_ids': [x for x in os.environ.get('ARROW_TRAIN_RESET_IDS', '').split(':') if x],
    'episode_ids': [x for x in os.environ.get('ARROW_TRAIN_EPISODE_IDS', '').split(':') if x],
    'seed': os.environ.get('ARROW_TRAIN_SEED'),
    'init_state_index': os.environ.get('ARROW_TRAIN_INIT_STATE_INDEX'),
}
out.write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n', encoding='utf-8')
PY

args=()
if [[ "$ARROW_TRAIN_KIND" == editor ]]; then
  args+=(--variant editor)
elif [[ "$ARROW_TRAIN_KIND" == minimal_learned ]]; then
  args+=(--variant minimal-learned)
fi
if [[ "$ARROW_TRAIN_KIND" == editor || "$ARROW_TRAIN_KIND" == minimal_learned ]]; then
  args+=(--training-source "$ARROW_TRAIN_SOURCE_ARTIFACT" --output-checkpoint "$RUN_ROOT/checkpoint.pt" --base-vla-sha256 "$ARROW_TRAIN_BASE_SHA256")
  IFS=: read -r -a values <<< "${ARROW_TRAIN_TASK_IDS:-}"; for value in "${values[@]}"; do [[ -n "$value" ]] && args+=(--task-id "$value"); done
  IFS=: read -r -a values <<< "${ARROW_TRAIN_RESET_IDS:-}"; for value in "${values[@]}"; do [[ -n "$value" ]] && args+=(--reset-id "$value"); done
  IFS=: read -r -a values <<< "${ARROW_TRAIN_EPISODE_IDS:-}"; for value in "${values[@]}"; do [[ -n "$value" ]] && args+=(--episode-id "$value"); done
  [[ -n "${ARROW_TRAIN_SEED:-}" ]] && args+=(--seed "$ARROW_TRAIN_SEED")
  args+=(--success-only)
  args+=(--epochs "${ARROW_TRAIN_EPOCHS:-1}" --batch-size "${ARROW_TRAIN_BATCH_SIZE:-32}")
  module_name='vla_benchmarking.libero.arrow_policy_suite.residual_job'
else
  module_name='vla_benchmarking.libero.arrow_policy_suite.apprentice_job'
  args+=(--transitions "$ARROW_TRAIN_SOURCE_ARTIFACT" --parent-artifact "$ARROW_TRAIN_SOURCE_SHA256" \
    --dataset-root "$RUN_ROOT/dataset" --base-checkpoint "$ARROW_TRAIN_BASE_CHECKPOINT" \
    --base-sha256 "$ARROW_TRAIN_BASE_SHA256" --model-revision "$ARROW_TRAIN_MODEL_REVISION" \
    --processor-revision "$ARROW_TRAIN_PROCESSOR_REVISION" --output-dir "$RUN_ROOT/training" \
    --epochs "${ARROW_TRAIN_EPOCHS:-5}" --batch-size "${ARROW_TRAIN_BATCH_SIZE:-8}" --device "${ARROW_TRAIN_DEVICE:-cuda}")
  IFS=: read -r -a values <<< "${ARROW_TRAIN_TASK_IDS:-}"; for value in "${values[@]}"; do [[ -n "$value" ]] && args+=(--task-id "$value"); done
  IFS=: read -r -a values <<< "${ARROW_TRAIN_RESET_IDS:-}"; for value in "${values[@]}"; do [[ -n "$value" ]] && args+=(--reset-id "$value"); done
  IFS=: read -r -a values <<< "${ARROW_TRAIN_EPISODE_IDS:-}"; for value in "${values[@]}"; do [[ -n "$value" ]] && args+=(--episode-id "$value"); done
  [[ -n "${ARROW_TRAIN_SEED:-}" ]] && args+=(--seed "$ARROW_TRAIN_SEED")
fi
printf '%q ' "$PYTHON" -m "$module_name" "${args[@]}" > "$RUN_ROOT/command.txt"
printf '\n' >> "$RUN_ROOT/command.txt"
"$PYTHON" -m "$module_name" "${args[@]}" 2>&1 | tee "$RUN_ROOT/training.log"

execution_receipt="$RUN_ROOT/execution_receipt.json"
[[ ! -e "$execution_receipt" && ! -L "$execution_receipt" ]] || die 'refusing to overwrite execution_receipt.json'
"$PYTHON" - "$RUN_ROOT/training.log" "$execution_receipt" <<'PY'
import json
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
decoder = json.JSONDecoder()
best = None
best_end = -1
for index, character in enumerate(source):
    if character != "{":
        continue
    try:
        value, end = decoder.raw_decode(source[index:])
    except json.JSONDecodeError:
        continue
    if isinstance(value, dict) and index + end > best_end:
        best = value
        best_end = index + end
if not isinstance(best, dict):
    raise SystemExit("training output did not contain a JSON execution receipt")
Path(sys.argv[2]).write_text(json.dumps(best, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

artifact_receipt="$RUN_ROOT/artifact_receipt.json"
[[ ! -e "$artifact_receipt" && ! -L "$artifact_receipt" ]] || die 'refusing to overwrite artifact_receipt.json'
export ARROW_TRAIN_ARTIFACT_RECEIPT="$artifact_receipt"
if [[ "$ARROW_TRAIN_KIND" == editor || "$ARROW_TRAIN_KIND" == minimal_learned ]]; then
  runtime_artifact="$RUN_ROOT/checkpoint.pt"
  runtime_sidecar="$runtime_artifact.json"
  [[ ! -L "$runtime_artifact" && -f "$runtime_artifact" ]] || die 'residual training did not create checkpoint.pt'
  [[ ! -L "$runtime_sidecar" && -f "$runtime_sidecar" ]] || die 'residual training did not create checkpoint.pt.json'
  export ARROW_TRAIN_RUNTIME_ARTIFACT="$runtime_artifact"
  export ARROW_TRAIN_RUNTIME_SIDECAR="$runtime_sidecar"
  export ARROW_TRAIN_RUNTIME_KIND='residual'
else
  mapfile -d '' manifests < <(find -P -- "$RUN_ROOT/training" -type f -name apprentice_manifest.json -print0)
  ((${#manifests[@]} == 1)) || die 'Apprentice training must create exactly one apprentice_manifest.json'
  runtime_manifest="${manifests[0]}"
  runtime_artifact="$(dirname -- "$runtime_manifest")"
  [[ ! -L "$runtime_artifact" && -d "$runtime_artifact" ]] || die 'Apprentice manifest parent is not a regular bundle directory'
  export ARROW_TRAIN_RUNTIME_ARTIFACT="$runtime_artifact"
  export ARROW_TRAIN_RUNTIME_MANIFEST="$runtime_manifest"
  export ARROW_TRAIN_RUNTIME_KIND='apprentice'
fi
if [[ "$ARROW_TRAIN_RUNTIME_KIND" == residual ]]; then
  "$PYTHON" - "$ARROW_TRAIN_RUNTIME_ARTIFACT" "$ARROW_TRAIN_BASE_SHA256" "$ARROW_TRAIN_KIND" <<'PY'
import sys
from vla_benchmarking.libero.arrow_policy_suite.native_learned_training import load_native_residual
variant = "arrow_minimal_learned" if sys.argv[3] == "minimal_learned" else "arrow_editor"
load_native_residual(sys.argv[1], base_vla_sha256=sys.argv[2], variant=variant)
PY
else
  "$PYTHON" - "$ARROW_TRAIN_RUNTIME_ARTIFACT" "$ARROW_TRAIN_BASE_SHA256" <<'PY'
import sys
from vla_benchmarking.libero.arrow_policy_suite.apprentice_training import verify_adapter_reload
verify_adapter_reload(sys.argv[1], base_vla_sha256=sys.argv[2], loader=lambda _path: {})
PY
fi
"$PYTHON" - "$artifact_receipt" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

receipt_path = Path(sys.argv[1])
if receipt_path.exists() or receipt_path.is_symlink():
    raise SystemExit("refusing to overwrite artifact_receipt.json")
kind = os.environ["ARROW_TRAIN_RUNTIME_KIND"]
runtime = Path(os.environ["ARROW_TRAIN_RUNTIME_ARTIFACT"])
run_root = Path(os.environ["ARROW_TRAIN_OUTPUT_ROOT"])
archive_root = Path(os.environ["ARROW_TRAIN_ARCHIVE_ROOT"])

def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def inventory_hash(inventory: dict[str, str]) -> str:
    encoded = (json.dumps(dict(sorted(inventory.items())), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

if kind == "residual":
    sidecar = Path(os.environ["ARROW_TRAIN_RUNTIME_SIDECAR"])
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    artifact_sha256 = file_hash(runtime)
    if payload.get("checkpoint_sha256") != artifact_sha256:
        raise SystemExit("residual sidecar checkpoint hash does not match checkpoint.pt")
    inventory = {runtime.name: artifact_sha256}
    inventory_sha256 = inventory_hash(inventory)
    sidecar_record = {"path": str(sidecar), "sha256": file_hash(sidecar)}
    manifest_record = None
else:
    manifest = Path(os.environ["ARROW_TRAIN_RUNTIME_MANIFEST"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest.name != "apprentice_manifest.json" or manifest.parent != runtime:
        raise SystemExit("Apprentice manifest does not belong to its runtime bundle")
    inventory = {}
    for item in sorted(runtime.rglob("*")):
        if item.is_symlink():
            raise SystemExit("symlink in Apprentice runtime bundle")
        if item.is_file() and item != manifest:
            inventory[item.relative_to(runtime).as_posix()] = file_hash(item)
    if not inventory:
        raise SystemExit("Apprentice runtime bundle payload is empty")
    expected = payload.get("checkpoint_inventory", payload.get("inventory"))
    expected_hash = payload.get("checkpoint_sha256", payload.get("inventory_sha256"))
    if dict(expected or {}) != inventory or expected_hash != inventory_hash(inventory):
        raise SystemExit("Apprentice manifest inventory does not match runtime bundle")
    entries = [(item.relative_to(runtime).as_posix(), file_hash(item)) for item in sorted(runtime.rglob("*")) if item.is_file()]
    artifact_sha256 = hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode("utf-8")).hexdigest()
    inventory_sha256 = expected_hash
    sidecar_record = None
    manifest_record = {"path": str(manifest), "sha256": file_hash(manifest)}

relative = runtime.relative_to(run_root).as_posix()
archive_runtime = archive_root / "run" / relative
execution_receipt = run_root / "execution_receipt.json"
archive_execution_receipt = archive_root / "run" / "execution_receipt.json"
if sidecar_record is not None:
    sidecar_record["archive_path"] = str(archive_runtime) + ".json"
if manifest_record is not None:
    manifest_record["archive_path"] = str(archive_runtime / "apprentice_manifest.json")
receipt = {
    "schema": "arrow_policy_suite.training_artifact_receipt.v1",
    "train_kind": os.environ["ARROW_TRAIN_KIND"],
    "source_sha256": os.environ["ARROW_TRAIN_SOURCE_SHA256"],
    "base_sha256": os.environ["ARROW_TRAIN_BASE_SHA256"],
    "runtime_artifact_path": str(runtime),
    "archive_artifact_path": str(archive_runtime),
    "learned_artifact_runtime_path": str(runtime),
    "learned_artifact_archive_path": str(archive_runtime),
    "execution_receipt_path": str(execution_receipt),
    "archive_execution_receipt_path": str(archive_execution_receipt),
    "artifact_sha256": artifact_sha256,
    "inventory_sha256": inventory_sha256,
    "sidecar": sidecar_record,
    "manifest": manifest_record,
    "experiment_evidence": False,
}
receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
