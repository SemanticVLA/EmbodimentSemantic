#!/usr/bin/env bash
# Sequential OpenVLA then Pi0.5 evaluation; each model receives 100 vanilla
# and 100 sealed-randomized episodes (10 tasks x 10 episodes per condition).
# The pi05_smolvla_matrix repair mode keeps the same allocation and runs five
# ordered cells: Pi0.5 vanilla/sealed, base SmolVLA vanilla/sealed, and the
# pinned no-arrow LoRA SmolVLA vanilla cell.
#SBATCH --partition=gpu_a40_ext
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=4-23:00:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/vla_eval_matrix_%j.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/vla_eval_matrix_%j.err

set -Eeuo pipefail

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
export BASE_PYTHON="$HOME/.conda/envs/embodiment-smolvla-py312/bin/python"
test -x "$BASE_PYTHON"

export RUN_ROOT="$SCRATCH_FLASH/EmbodimentSemantic_runtime/runs/vla_eval_matrix_${SLURM_JOB_ID}"
export ARCHIVE_ROOT="$HOME/EmbodimentSemantic_archive/runs/vla_eval_matrix_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$ARCHIVE_ROOT" "$HOME/EmbodimentSemantic_runtime/operator/logs"

archive_on_exit() {
  rc=$?
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$RUN_ROOT/" "$ARCHIVE_ROOT/"
  else
    cp -a "$RUN_ROOT/." "$ARCHIVE_ROOT/"
  fi
  printf '{"job_id":"%s","return_code":%s,"archive_root":"%s"}\n' "$SLURM_JOB_ID" "$rc" "$ARCHIVE_ROOT" > "$ARCHIVE_ROOT/terminal.json"
  exit "$rc"
}
trap archive_on_exit EXIT

export REPO_ROOT="${REPO_ROOT:-$HOME/EmbodimentSemantic}"
test -d "$REPO_ROOT/.git"
cd "$REPO_ROOT"
export LIBERO_CONFIG_PATH="$RUN_ROOT/libero_config"
export MUJOCO_GL=egl
export HF_HOME="${HF_HOME:-$SCRATCH_FLASH/EmbodimentSemantic_runtime/cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export PYTHONNOUSERSITE=1

# Parse the repair mode before environment setup so Pi0.5-only reruns never
# create or touch the isolated original-OpenVLA runtime.
VLA_REPAIR="${VLA_REPAIR:-0}"
PRESERVED_OPENVLA_VANILLA="${PRESERVED_OPENVLA_VANILLA:-}"
case "$VLA_REPAIR" in
  0|1) ;;
  pi05|pi05_only|pi05-only) VLA_REPAIR="pi05_only" ;;
  pi05_smolvla_matrix|pi05-smolvla-matrix) VLA_REPAIR="pi05_smolvla_matrix" ;;
  *) echo "unsupported VLA_REPAIR=$VLA_REPAIR (expected 0, 1, pi05_only, or pi05_smolvla_matrix)" >&2; exit 2 ;;
esac
PI05_ONLY="0"
MATRIX_MODE="0"
if [[ "$VLA_REPAIR" == "pi05_only" || "$VLA_REPAIR" == "pi05_smolvla_matrix" ]]; then
  PI05_ONLY="1"
fi
if [[ "$VLA_REPAIR" == "pi05_smolvla_matrix" ]]; then
  MATRIX_MODE="1"
fi
PI05_CANARY_ONLY="${PI05_CANARY_ONLY:-0}"
case "$PI05_CANARY_ONLY" in
  0|1) ;;
  *) echo "unsupported PI05_CANARY_ONLY=$PI05_CANARY_ONLY (expected 0 or 1)" >&2; exit 2 ;;
esac
if [[ "$PI05_CANARY_ONLY" == "1" && "$PI05_ONLY" != "1" ]]; then
  echo "PI05_CANARY_ONLY=1 requires VLA_REPAIR=pi05_only or pi05_smolvla_matrix" >&2
  exit 2
fi
if [[ "$VLA_REPAIR" == "1" ]]; then
  test -s "$PRESERVED_OPENVLA_VANILLA"
  [[ "$(awk 'NF{n++} END{print n+0}' "$PRESERVED_OPENVLA_VANILLA")" -eq 100 ]]
fi

# Each concurrent job gets its own LIBERO config and BDDL tree.  Never mutate
# the shared ~/.libero/config.yaml while another SLURM job may be rolling out.
export LIBERO_SOURCE_ROOT="$HOME/EmbodimentSemantic/vla_benchmarking/LIBERO"
export LIBERO_SOURCE_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"
export LIBERO_BDDL_SOURCE="$LIBERO_SOURCE_ROOT/libero/libero/bddl_files"
export LIBERO_BDDL_TARGET="$RUN_ROOT/libero_cache/bddl_files"
export LIBERO_INIT_SOURCE="$LIBERO_SOURCE_ROOT/libero/libero/init_files"
export LIBERO_INIT_TARGET="$RUN_ROOT/libero_cache/init_files"
test -d "$LIBERO_BDDL_SOURCE"
test -d "$LIBERO_INIT_SOURCE"
[[ "$(git -C "$LIBERO_SOURCE_ROOT" rev-parse HEAD)" == "$LIBERO_SOURCE_COMMIT" ]]
[[ -z "$(git -C "$LIBERO_SOURCE_ROOT" status --porcelain --untracked-files=all)" ]]
mkdir -p "$LIBERO_CONFIG_PATH" "$LIBERO_BDDL_TARGET" "$LIBERO_INIT_TARGET"
rsync -a "$LIBERO_BDDL_SOURCE/" "$LIBERO_BDDL_TARGET/"
rsync -a "$LIBERO_INIT_SOURCE/" "$LIBERO_INIT_TARGET/"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $HOME/.cache/libero/benchmark
bddl_files: $LIBERO_BDDL_TARGET
init_states: $LIBERO_INIT_TARGET
datasets: $HOME/.cache/libero/datasets
assets: $HOME/.cache/libero/assets
YAML
export LIBERO_BDDL_TREE_SHA256="$($BASE_PYTHON -c 'from vla_benchmarking.libero.evaluation.plan import libero_source_hashes; print(libero_source_hashes()["libero_bddl_tree_sha256"])')"

nvidia-smi -L
"$BASE_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

"$BASE_PYTHON" -m venv --system-site-packages "$RUN_ROOT/venv"
export PYTHON="$RUN_ROOT/venv/bin/python"
"$PYTHON" -m pip install --disable-pip-version-check --upgrade pip 'setuptools<81' wheel

"$PYTHON" -m pip install --disable-pip-version-check ml-collections sentencepiece timm

# Original OpenVLA requires the pre-5.x AutoModelForVision2Seq API.  LeRobot's
# pinned Pi05 extra intentionally stays on transformers 5.3.0, so keep the
# original OpenVLA runtime in a separate provenance-bound environment.
if [[ "$PI05_ONLY" != "1" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$RUN_ROOT/openvla_venv"
  export OPENVLA_PYTHON="$RUN_ROOT/openvla_venv/bin/python"
  "$OPENVLA_PYTHON" -m pip install --disable-pip-version-check --upgrade pip 'setuptools<81' wheel
  "$OPENVLA_PYTHON" -m pip install --disable-pip-version-check \
    -r vla_benchmarking/libero/finetuned_vlas/openvla/requirements.txt
  "$OPENVLA_PYTHON" -c 'import transformers; from transformers import AutoModelForVision2Seq; assert transformers.__version__ == "4.40.1"; print("original OpenVLA transformers", transformers.__version__)'
fi
"$PYTHON" -c 'import transformers; print("Pi05 transformers", transformers.__version__)'

SNAPSHOT_SCRIPT="$RUN_ROOT/resolve_hf_snapshot.py"
cat > "$SNAPSHOT_SCRIPT" <<'PY'
from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
import json, os, sys
repo, out, revision = sys.argv[1], Path(sys.argv[2]).resolve(), sys.argv[3]
token = os.environ.get("HF_TOKEN") or None
info = HfApi(token=token).model_info(repo, revision=revision)
if info.sha != revision:
    raise SystemExit(f"resolved Hub revision mismatch for {repo}: {info.sha} != {revision}")
path = Path(snapshot_download(repo_id=repo, revision=revision, local_dir=str(out), local_dir_use_symlinks=False)).resolve()
print(json.dumps({"repo_id": repo, "revision": revision, "path": str(path)}))
PY

resolve_snapshot() {
  local repo="$1"; local name="$2"; local revision="$3"; local result
  result="$($PYTHON "$SNAPSHOT_SCRIPT" "$repo" "$RUN_ROOT/checkpoints/$name" "$revision")"
  printf '%s\n' "$result" | tee "$RUN_ROOT/${name}_snapshot.json"
  printf '%s\n' "$result" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["revision"])'
}

PI05_REPO="lerobot/pi05_libero_finetuned_v044"
OPENVLA_REPO_ID="openvla/openvla-7b-finetuned-libero-spatial"
PI05_REVISION="8e174154ef5f6c60a8da12ae99c303d8963138c1"
OPENVLA_REVISION="962318cec55ac10993ff0f5f43eda9a270b4c873"
PI05_CHECKPOINT="$RUN_ROOT/checkpoints/pi05"
OPENVLA_CHECKPOINT="$RUN_ROOT/checkpoints/openvla"

# Exact, immutable SmolVLA artifacts for the combined matrix.
SMOLVLA_BASE_PATH="/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/vla_benchmarking/base_models/smolvla_libero-6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
SMOLVLA_BASE_REVISION="6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
SMOLVLA_ADAPTER_PATH="/home/hjaber/EmbodimentSemantic_archive/runs/legion_no_arrow_lora_full_s1000_v1_no_arrow_treatment_1910197/checkpoints/029190/pretrained_model"
SMOLVLA_TRAINING_ROOT="/home/hjaber/EmbodimentSemantic_archive/runs/legion_no_arrow_lora_full_s1000_v1_no_arrow_treatment_1910197"
SMOLVLA_TRAINING_MANIFEST="$SMOLVLA_TRAINING_ROOT/training_manifest.json"
SMOLVLA_ADAPTER_SHA256="80b3c23fc3987530d57766ab45ed33db918f08983739139c1ff0397184cc7092"
SMOLVLA_BASE_MANIFEST_SHA256="e4bcf9b4481cae4b523ef6c3af6a6a9fd5e9886215f48f911e7226221761486b"
SMOLVLA_BASE_TREE_SHA256="d086e041f3f6bfb919f335265106fee1db8b8a3c386af357cb42a89735f74bf1"
export SMOLVLA_BASE_PATH SMOLVLA_BASE_REVISION SMOLVLA_ADAPTER_PATH
export SMOLVLA_TRAINING_MANIFEST SMOLVLA_ADAPTER_SHA256
export SMOLVLA_BASE_MANIFEST_SHA256 SMOLVLA_BASE_TREE_SHA256

validate_smolvla_artifacts() {
  "$BASE_PYTHON" - "$SMOLVLA_BASE_PATH" "$SMOLVLA_BASE_REVISION" \
    "$SMOLVLA_BASE_MANIFEST_SHA256" "$SMOLVLA_BASE_TREE_SHA256" \
    "$SMOLVLA_ADAPTER_PATH" "$SMOLVLA_ADAPTER_SHA256" "$SMOLVLA_TRAINING_MANIFEST" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

base, revision, manifest_sha, tree_sha, adapter, adapter_sha, training = sys.argv[1:]
base = pathlib.Path(base).resolve()
manifest_path = base / "base_snapshot_manifest.json"
adapter = pathlib.Path(adapter).resolve()
training = pathlib.Path(training).resolve()

def sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

if not base.is_dir() or base.name != f"smolvla_libero-{revision}":
    raise SystemExit(f"invalid SmolVLA base snapshot path: {base}")
if not manifest_path.is_file() or sha(manifest_path) != manifest_sha:
    raise SystemExit("SmolVLA base snapshot manifest hash mismatch")
snapshot = json.loads(manifest_path.read_text(encoding="utf-8"))
if snapshot.get("revision") != revision or not isinstance(snapshot.get("files"), dict):
    raise SystemExit("SmolVLA base snapshot manifest revision/inventory mismatch")
actual = {
    path.relative_to(base).as_posix()
    for path in base.rglob("*")
    if path.is_file() and path.name != "base_snapshot_manifest.json" and ".cache" not in path.parts
}
if actual != set(snapshot["files"]):
    raise SystemExit("SmolVLA base snapshot file inventory mismatch")
for name, expected in snapshot["files"].items():
    path = base / name
    if not path.is_file() or sha(path) != expected:
        raise SystemExit(f"SmolVLA base snapshot file hash mismatch: {name}")
inventory = {name: snapshot["files"][name] for name in sorted(snapshot["files"])}
observed_tree = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if observed_tree != tree_sha:
    raise SystemExit("SmolVLA base snapshot tree hash mismatch")
artifact = adapter / "adapter_model.safetensors"
if not adapter.is_dir() or adapter.name != "pretrained_model" or not artifact.is_file() or sha(artifact) != adapter_sha:
    raise SystemExit("SmolVLA adapter path/hash mismatch")
if not training.is_file() or training.parent != pathlib.Path("/home/hjaber/EmbodimentSemantic_archive/runs/legion_no_arrow_lora_full_s1000_v1_no_arrow_treatment_1910197"):
    raise SystemExit("SmolVLA training manifest path mismatch")
data = json.loads(training.read_text(encoding="utf-8"))
if data.get("base_policy_revision") != revision or pathlib.Path(data.get("base_policy", "")).resolve() != base:
    raise SystemExit("SmolVLA training manifest base lineage mismatch")
record = data.get("no_arrow_treatment_adapter")
if not isinstance(record, dict):
    raise SystemExit("SmolVLA training manifest lacks no-arrow adapter record")
recorded = pathlib.Path(record.get("path", "")).resolve()
recorded = recorded.parent if recorded.name == "adapter_model.safetensors" else recorded
archive_runs = pathlib.Path("/home/hjaber/EmbodimentSemantic_archive/runs")
scratch_runs = pathlib.Path("/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/runs")
try:
    adapter_rel = adapter.relative_to(archive_runs)
except ValueError:
    raise SystemExit("SmolVLA adapter is outside the expected archive runs root")
allowed_adapter_paths = {adapter, scratch_runs / adapter_rel}
if recorded not in allowed_adapter_paths or record.get("sha256") != adapter_sha:
    raise SystemExit("SmolVLA training manifest adapter lineage mismatch")
print(json.dumps({
    "base_path": str(base), "base_revision": revision,
    "base_manifest_sha256": manifest_sha, "base_tree_sha256": observed_tree,
    "adapter_path": str(adapter), "adapter_sha256": adapter_sha,
    "training_manifest": str(training), "training_manifest_sha256": sha(training),
}, sort_keys=True))
PY
}

# The Pi0.5-only repair path is the only path that needs the gated PaliGemma
# tokenizer.  Read exactly HF_TOKEN from the protected dotenv file without
# importing unrelated secrets into the evaluation environment.
PI05_TOKENIZER_REPO="google/paligemma-3b-pt-224"
PI05_TOKENIZER_PATH=""
PI05_TOKENIZER_REVISION="${PI05_TOKENIZER_REVISION:-}"
if [[ "$PI05_ONLY" == "1" ]]; then
  OMNIS_SECRETS_ENV="$HOME/.config/omnis/secrets.env"
  test -f "$OMNIS_SECRETS_ENV" || { echo "missing protected secrets file: $OMNIS_SECRETS_ENV" >&2; exit 2; }
  test -r "$OMNIS_SECRETS_ENV" || { echo "secrets file is not readable: $OMNIS_SECRETS_ENV" >&2; exit 2; }
  unset HF_TOKEN
  # Source the protected dotenv only in a short-lived subshell and capture
  # exactly HF_TOKEN.  Unrelated variables therefore cannot enter this job's
  # environment, and the token value is never printed or persisted.
  HF_TOKEN="$(
    set +u
    # shellcheck disable=SC1090
    source "$OMNIS_SECRETS_ENV"
    printf '%s' "${HF_TOKEN:-}"
  )"
  export HF_TOKEN
  test -n "$HF_TOKEN" || { echo "HF_TOKEN is empty in protected secrets.env" >&2; exit 2; }
  [[ "$PI05_TOKENIZER_REVISION" =~ ^[0-9a-fA-F]{40}$ ]] || {
    echo "PI05_TOKENIZER_REVISION must be an exact 40-character commit hash" >&2
    exit 2
  }
  PI05_TOKENIZER_REVISION="${PI05_TOKENIZER_REVISION,,}"
  export PI05_TOKENIZER_REVISION
  resolve_snapshot "$PI05_TOKENIZER_REPO" pi05_tokenizer "$PI05_TOKENIZER_REVISION" >/dev/null
  PI05_TOKENIZER_PATH="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$RUN_ROOT/pi05_tokenizer_snapshot.json")"
  test -d "$PI05_TOKENIZER_PATH" || { echo "Pi0.5 tokenizer snapshot directory is unavailable" >&2; exit 2; }
  [[ -n "$(find "$PI05_TOKENIZER_PATH" -type f -print -quit)" ]] || {
    echo "Pi0.5 tokenizer snapshot is empty" >&2
    exit 2
  }
  export PI05_TOKENIZER_PATH
fi
resolve_snapshot "$PI05_REPO" pi05 "$PI05_REVISION" >/dev/null
if [[ "$PI05_ONLY" != "1" ]]; then
  resolve_snapshot "$OPENVLA_REPO_ID" openvla "$OPENVLA_REVISION" >/dev/null
else
  # The protected token is needed only for authenticated Hub resolution.
  # Remove it before writing provenance or entering canary/evaluation code.
  unset HF_TOKEN
fi

if [[ "$MATRIX_MODE" == "1" ]]; then
  validate_smolvla_artifacts | tee "$RUN_ROOT/smolvla_artifact_provenance.json"
fi

DATASET_MANIFEST="$RUN_ROOT/libero_spatial_eval_manifest.json"
"$PYTHON" - "$DATASET_MANIFEST" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
source = pathlib.Path("vlm_benchmarking/data/libero_spatial_v5/libero_spatial_v5_source_manifest.json")
payload = json.loads(source.read_text()) if source.is_file() else {}
payload.update({"schema": "libero_vla_eval_manifest.v1", "suite": "libero_spatial", "tasks": list(range(10)), "episodes_per_task": 10, "conditions": ["vanilla", "sealed_randomized"], "total_episodes": 200})
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

PLAN_DIR="$RUN_ROOT/plans"
if [[ "$PI05_ONLY" != "1" ]]; then
  "$OPENVLA_PYTHON" -m vla_benchmarking.libero.finetuned_vlas.legion.prepare_eval_plans \
    --model openvla --checkpoint-path "$OPENVLA_CHECKPOINT" \
    --checkpoint-revision "$OPENVLA_REVISION" --dataset-manifest "$DATASET_MANIFEST" --output-dir "$PLAN_DIR"
fi
"$PYTHON" -m vla_benchmarking.libero.finetuned_vlas.legion.prepare_eval_plans \
  --model pi05 --checkpoint-path "$PI05_CHECKPOINT" \
  --checkpoint-revision "$PI05_REVISION" --dataset-manifest "$DATASET_MANIFEST" --output-dir "$PLAN_DIR"

cat > "$RUN_ROOT/run_provenance.json" <<JSON
{
  "schema": "vla_eval_matrix_run.v1",
  "job_id": "${SLURM_JOB_ID}",
  "source_commit": "$(git rev-parse HEAD)",
  "repair_mode": $([[ "$VLA_REPAIR" == "1" ]] && echo true || echo false),
  "vla_repair_mode": "$VLA_REPAIR",
  "preserved_openvla_vanilla": "${PRESERVED_OPENVLA_VANILLA}",
  "pi05_tokenizer": {"repo_id": "$PI05_TOKENIZER_REPO", "revision": "$PI05_TOKENIZER_REVISION", "path": "$PI05_TOKENIZER_PATH"},
  "models": {
    "openvla": {"repo_id": "$OPENVLA_REPO_ID", "revision": "$OPENVLA_REVISION", "path": "$OPENVLA_CHECKPOINT", "fine_tuned_again": false},
    "pi05": {"repo_id": "$PI05_REPO", "revision": "$PI05_REVISION", "path": "$PI05_CHECKPOINT", "fine_tuned_again": false}
  },
  "schedule": {"tasks": 10, "episodes_per_task": 10, "conditions": ["vanilla", "sealed_randomized"], "episodes_per_model": 200},
  "libero_source": {"commit": "$LIBERO_SOURCE_COMMIT", "bddl_tree_sha256": "$LIBERO_BDDL_TREE_SHA256"},
  "dataset_manifest": "$DATASET_MANIFEST"
}
JSON

if [[ "$MATRIX_MODE" == "1" ]]; then
"$BASE_PYTHON" - "$RUN_ROOT/run_provenance.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
training = pathlib.Path("/home/hjaber/EmbodimentSemantic_archive/runs/legion_no_arrow_lora_full_s1000_v1_no_arrow_treatment_1910197/training_manifest.json")
training_sha = hashlib.sha256(training.read_bytes()).hexdigest()
data["repair_mode"] = True
data["vla_repair_mode"] = "pi05_smolvla_matrix"
data["matrix"] = {
    "cells": [
        {"cell_id": "pi05_vanilla", "model": "pi05", "suite_mode": "vanilla", "episodes": 100},
        {"cell_id": "pi05_sealed", "model": "pi05", "suite_mode": "sealed_randomized", "episodes": 100},
        {"cell_id": "smolvla_base_vanilla", "model": "smolvla", "suite_mode": "vanilla", "episodes": 100},
        {"cell_id": "smolvla_base_sealed_randomized", "model": "smolvla", "suite_mode": "sealed_randomized", "episodes": 100},
        {"cell_id": "smolvla_no_arrow_ft_vanilla", "model": "smolvla", "suite_mode": "vanilla", "fine_tuned": True, "episodes": 100},
    ],
    "total_episodes": 500,
    "execution": "sequential_same_allocation",
}
data["models"]["smolvla"] = {
    "base_checkpoint": "/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/vla_benchmarking/base_models/smolvla_libero-6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
    "base_revision": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
    "adapter_checkpoint": "/home/hjaber/EmbodimentSemantic_archive/runs/legion_no_arrow_lora_full_s1000_v1_no_arrow_treatment_1910197/checkpoints/029190/pretrained_model",
    "adapter_sha256": "80b3c23fc3987530d57766ab45ed33db918f08983739139c1ff0397184cc7092",
    "training_manifest": str(training), "training_manifest_sha256": training_sha,
    "base_manifest_sha256": os.environ["SMOLVLA_BASE_MANIFEST_SHA256"],
    "base_tree_sha256": os.environ["SMOLVLA_BASE_TREE_SHA256"],
}
data["schedule"] = {
    "tasks": 10, "episodes_per_task": 10, "cells": 5,
    "conditions": ["vanilla", "sealed_randomized", "vanilla", "sealed_randomized", "vanilla"],
    "episodes_total": 500, "execution": "sequential_same_allocation",
}
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

MATRIX_STAGE_RESULTS="$RUN_ROOT/matrix_stage_results.jsonl"
MATRIX_FAILURES="0"

run_matrix_stage() {
  local label="$1"; shift
  local started ended rc
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "START stage=$label job=$SLURM_JOB_ID"
  set +e
  "$@"
  rc=$?
  set -e
  ended="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"label":"%s","return_code":%s,"started":"%s","ended":"%s"}\n' \
    "$label" "$rc" "$started" "$ended" >> "$MATRIX_STAGE_RESULTS"
  echo "DONE stage=$label return_code=$rc job=$SLURM_JOB_ID"
  if [[ "$rc" -ne 0 ]]; then MATRIX_FAILURES="1"; fi
  return 0
}

run_pi05_matrix_cell() {
  local condition="$1"
  local output="$RUN_ROOT/pi05_${condition}.jsonl"
  "$PYTHON" -m vla_benchmarking.libero.evaluation.native_vla_eval \
    --model pi05 --plan "$PLAN_DIR/pi05_${condition}_100.json" \
    --checkpoint-path "$PI05_CHECKPOINT" --checkpoint-revision "$PI05_REVISION" \
    --dataset-manifest "$DATASET_MANIFEST" --output-jsonl "$output"
  [[ "$(awk 'NF{n++} END{print n+0}' "$output")" -eq 100 ]] || {
    echo "Pi0.5 $condition cell must contain exactly 100 rows: $output" >&2
    return 1
  }
}

run_smolvla_matrix() {
  local protocol="$1"
  local output="$RUN_ROOT/smolvla_matrix_${protocol}"
  mkdir -p "$output"
  "$PYTHON" -m vla_benchmarking.libero.finetuned_vlas.smolvla.workflows.run_smolvla_eval_matrix \
    --adapter-checkpoint "$SMOLVLA_ADAPTER_PATH" \
    --training-manifest "$SMOLVLA_TRAINING_MANIFEST" \
    --base-checkpoint "$SMOLVLA_BASE_PATH" \
    --output-root "$output" --protocol "$protocol" --device cuda --no-videos
}

validate_smolvla_matrix_result() {
  local protocol="$1"; local episodes_per_task="$2"
  local output="$RUN_ROOT/smolvla_matrix_${protocol}"
  "$BASE_PYTHON" - "$output" "$protocol" "$episodes_per_task" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
protocol = sys.argv[2]
episodes_per_task = int(sys.argv[3])
expected_cells = [
    "smolvla_base_vanilla",
    "smolvla_base_sealed_randomized",
    "smolvla_no_arrow_ft_vanilla",
]
manifest_path = root / "smolvla_eval_matrix_manifest.json"
schedule_path = root / "smolvla_eval_matrix_schedule.json"
summary_path = root / "smolvla_eval_matrix_summary.csv"
if not manifest_path.is_file() or not schedule_path.is_file() or not summary_path.is_file():
    raise SystemExit(f"SmolVLA {protocol} matrix artifacts are incomplete")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
tasks = manifest.get("tasks")
if not isinstance(tasks, list) or not tasks:
    raise SystemExit("SmolVLA matrix task schedule is missing")
episodes_per_cell = len(tasks) * episodes_per_task
if manifest.get("protocol") != protocol or manifest.get("episodes") != episodes_per_task:
    raise SystemExit("SmolVLA matrix manifest protocol/episode contract mismatch")
if manifest.get("planned_episodes_per_cell") != episodes_per_cell or manifest.get("planned_episodes_total") != episodes_per_cell * 3:
    raise SystemExit("SmolVLA matrix planned episode counts are invalid")
if [cell.get("cell_id") for cell in manifest.get("cells", [])] != expected_cells:
    raise SystemExit("SmolVLA matrix manifest cell IDs are not exact")
if schedule.get("protocol") != protocol or len(schedule.get("cells", [])) != episodes_per_cell * 3:
    raise SystemExit("SmolVLA matrix schedule is incomplete")
if {cell.get("cell_id") for cell in schedule["cells"]} != set(expected_cells):
    raise SystemExit("SmolVLA matrix schedule cell IDs are incomplete")
rows = list(csv.DictReader(summary_path.open(newline="", encoding="utf-8")))
cell_rows = [row for row in rows if row.get("row_type") == "cell"]
if [row.get("cell_id") for row in cell_rows] != expected_cells:
    raise SystemExit("SmolVLA matrix summary cell rows are incomplete or out of order")
if any(row.get("status") != "complete" or int(row.get("episodes", -1)) != episodes_per_cell for row in cell_rows):
    raise SystemExit("SmolVLA matrix summary contains incomplete cells")
cells = []
for cell in manifest["cells"]:
    output_dir = pathlib.Path(cell["output_dir"])
    info_path = output_dir / "eval_info.json"
    if not info_path.is_file():
        raise SystemExit(f"missing SmolVLA eval_info.json: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("overall", {}).get("n_episodes") != episodes_per_cell:
        raise SystemExit(f"{cell['cell_id']} does not contain {episodes_per_cell} episodes")
    if cell["suite_mode"] == "sealed_randomized":
        audit_path = output_dir / "randomization_audit.jsonl"
        records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()] if audit_path.is_file() else []
        if len(records) != episodes_per_cell or any(record.get("status") != "ok" for record in records):
            raise SystemExit(f"{cell['cell_id']} sealed randomization audit is incomplete")
    cells.append({
        "cell_id": cell["cell_id"], "suite_mode": cell["suite_mode"],
        "episodes": episodes_per_cell, "output_dir": str(output_dir),
        "eval_info": str(info_path), "successes": sum(sum(item["metrics"]["successes"]) for item in info["per_task"]),
        "pc_success": info.get("overall", {}).get("pc_success"),
    })
(root / "matrix_postcondition.json").write_text(json.dumps({
    "protocol": protocol, "cell_ids": expected_cells, "episodes_per_task": episodes_per_task,
    "episodes_per_cell": episodes_per_cell, "total_episodes": episodes_per_cell * 3, "cells": cells,
}, indent=2, sort_keys=True) + "\n")
PY
}

if [[ "$PI05_CANARY_ONLY" == "1" ]]; then
  # Reuse the exact resolved Pi0.5 checkpoint and local gated tokenizer.  The
  # adapter's predict() path invokes one native predict_action_chunk call and
  # enforces the unchanged (50, 7) contract before this bounded canary exits.
  "$PYTHON" - "$PI05_CHECKPOINT" "$PI05_REVISION" "$PI05_TOKENIZER_PATH" "$PI05_TOKENIZER_REVISION" "$RUN_ROOT/pi05_canary.json" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

from vla_benchmarking.libero.finetuned_vlas.pi05 import Pi05Adapter
from vla_benchmarking.libero.finetuned_vlas.pi05.contracts import PolicyArtifact

checkpoint, revision, tokenizer_path, tokenizer_revision, receipt_path = sys.argv[1:]
artifact = PolicyArtifact(
    model_id="lerobot/pi05_libero_finetuned_v044",
    revision=revision,
    checkpoint_path=checkpoint,
    provenance="legion_pi05_canary",
)
adapter = Pi05Adapter.load(artifact=artifact)
adapter.reset("Canary: inspect the scene and pick up the bowl.", 0)
observation = {
    "observation.images.image": np.zeros((256, 256, 3), dtype=np.uint8),
    "observation.images.image2": np.zeros((256, 256, 3), dtype=np.uint8),
    "observation.state": np.zeros((8,), dtype=np.float32),
}
action = adapter.predict(observation, task="Canary: inspect the scene and pick up the bowl.")
if action.shape != (50, 7):
    raise SystemExit(f"Pi0.5 canary action shape mismatch: {action.shape}")
if not np.isfinite(action).all():
    raise SystemExit("Pi0.5 canary action contains non-finite values")
receipt = {
    "schema": "pi05_canary.v1",
    "status": "passed",
    "model_repo": artifact.model_id,
    "model_revision": revision,
    "checkpoint_path": str(Path(checkpoint).resolve()),
    "tokenizer_repo": "google/paligemma-3b-pt-224",
    "tokenizer_revision": tokenizer_revision,
    "tokenizer_path": str(Path(tokenizer_path).resolve()),
    "action_shape": list(action.shape),
    "finite": True,
}
Path(receipt_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, sort_keys=True))
PY
  if [[ "$MATRIX_MODE" == "1" ]]; then
    run_matrix_stage smolvla_matrix_smoke run_smolvla_matrix smoke
    run_matrix_stage smolvla_matrix_smoke_postcondition validate_smolvla_matrix_result smoke 1
    "$BASE_PYTHON" - "$RUN_ROOT" "$MATRIX_STAGE_RESULTS" <<'PY'
import json
import pathlib
import sys
root, stages = map(pathlib.Path, sys.argv[1:])
rows = [json.loads(line) for line in stages.read_text().splitlines() if line.strip()]
if len(rows) != 2 or any(row["return_code"] != 0 for row in rows):
    raise SystemExit("combined SmolVLA canary failed")
postcondition = root / "smolvla_matrix_smoke" / "matrix_postcondition.json"
if not postcondition.is_file() or json.loads(postcondition.read_text()).get("total_episodes") != 6:
    raise SystemExit("combined SmolVLA canary postcondition is incomplete")
(root / "matrix_canary_summary.json").write_text(json.dumps({
    "status": "passed", "pi05_canary": str(root / "pi05_canary.json"),
    "smolvla_matrix": str(postcondition),
}, indent=2, sort_keys=True) + "\n")
PY
  fi
  exit 0
fi

run_model() {
  local python_bin="$1"; local model="$2"; local ckpt="$3"; local revision="$4"; local factory="${5:-}"
  local extra=()
  if [[ -n "$factory" ]]; then extra+=(--policy-config-factory "$factory"); fi
  local conditions=(vanilla sealed)
  if [[ "$VLA_REPAIR" == "1" && "$model" == "openvla" ]]; then
    conditions=(sealed)
  fi
  for condition in "${conditions[@]}"; do
    echo "START model=$model condition=$condition job=$SLURM_JOB_ID"
    "$python_bin" -m vla_benchmarking.libero.evaluation.native_vla_eval \
      --model "$model" --plan "$PLAN_DIR/${model}_${condition}_100.json" \
      --checkpoint-path "$ckpt" --checkpoint-revision "$revision" \
      --dataset-manifest "$DATASET_MANIFEST" \
      --output-jsonl "$RUN_ROOT/${model}_${condition}.jsonl" "${extra[@]}"
    echo "DONE model=$model condition=$condition job=$SLURM_JOB_ID"
  done
}

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
if [[ "$MATRIX_MODE" == "1" ]]; then
  : > "$MATRIX_STAGE_RESULTS"
  # Preserve the user-requested sequential order within one GPU allocation.
  run_matrix_stage pi05_vanilla run_pi05_matrix_cell vanilla
  run_matrix_stage pi05_sealed run_pi05_matrix_cell sealed
  run_matrix_stage smolvla_matrix_full run_smolvla_matrix full
  run_matrix_stage smolvla_matrix_full_postcondition validate_smolvla_matrix_result full 10
  "$BASE_PYTHON" - "$RUN_ROOT" "$MATRIX_STAGE_RESULTS" <<'PY'
import hashlib
import os
import json
import pathlib
import sys

root, stages = map(pathlib.Path, sys.argv[1:])
expected = [
    "pi05_vanilla", "pi05_sealed", "smolvla_matrix_full",
    "smolvla_matrix_full_postcondition",
]
stage_rows = [json.loads(line) for line in stages.read_text(encoding="utf-8").splitlines() if line.strip()]
if [row.get("label") for row in stage_rows] != expected:
    raise SystemExit(f"combined matrix stage order/count mismatch: {stage_rows}")
if any(int(row.get("return_code", 1)) != 0 for row in stage_rows):
    raise SystemExit("combined matrix has a failed stage")
cells = {}
combined_rows = []
for label in expected[:2]:
    path = root / f"{label}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []
    count = len(records)
    if count != 100:
        raise SystemExit(f"{label} must contain exactly 100 rows, got {count}")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SystemExit(f"{label} contains a non-object row")
        combined_rows.append({**record, "cell": label, "matrix_row_index": len(combined_rows)})
    cells[label] = {"episodes": count, "output": str(path)}
matrix_root = root / "smolvla_matrix_full"
postcondition = json.loads((matrix_root / "matrix_postcondition.json").read_text(encoding="utf-8"))
if postcondition.get("protocol") != "full" or postcondition.get("total_episodes") != 300:
    raise SystemExit("SmolVLA full matrix postcondition is incomplete")
manifest = json.loads((matrix_root / "smolvla_eval_matrix_manifest.json").read_text(encoding="utf-8"))
manifest_cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
for cell in postcondition["cells"]:
    label = cell["cell_id"]
    if cell.get("episodes") != 100:
        raise SystemExit(f"{label} must contain exactly 100 episodes")
    info = json.loads(pathlib.Path(cell["eval_info"]).read_text(encoding="utf-8"))
    episode_index = 0
    metadata = manifest_cells[label]
    for task in info["per_task"]:
        for task_episode_index, success in enumerate(task["metrics"]["successes"]):
            combined_rows.append({
                "cell": label, "cell_id": label, "suite_mode": metadata["suite_mode"],
                "checkpoint": metadata["checkpoint"], "seed": 1000 + task_episode_index,
                "task_id": task["task_id"], "episode_index": task_episode_index,
                "matrix_row_index": len(combined_rows), "success": bool(success),
            })
            episode_index += 1
    if episode_index != 100:
        raise SystemExit(f"{label} could not be flattened to 100 rows")
    cells[label] = cell
if len(combined_rows) != 500 or sum(int(cell["episodes"]) for cell in cells.values()) != 500:
    raise SystemExit("combined matrix must contain exactly 500 episodes")
(root / "combined_episodes.jsonl").write_text(
    "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined_rows), encoding="utf-8"
)
(root / "combined_summary.json").write_text(json.dumps({
    "schema": "pi05_smolvla_matrix.v1", "status": "passed",
    "cells": cells, "total_episodes": 500, "total_rows": len(combined_rows),
    "combined_episodes": str(root / "combined_episodes.jsonl"), "stage_results": str(stages),
    "smolvla_matrix": str(matrix_root),
    "smolvla": {
        "base_path": os.environ.get("SMOLVLA_BASE_PATH", ""),
        "base_revision": os.environ.get("SMOLVLA_BASE_REVISION", ""),
        "base_manifest_sha256": os.environ.get("SMOLVLA_BASE_MANIFEST_SHA256", ""),
        "base_tree_sha256": os.environ.get("SMOLVLA_BASE_TREE_SHA256", ""),
        "adapter_path": os.environ.get("SMOLVLA_ADAPTER_PATH", ""),
        "adapter_sha256": os.environ.get("SMOLVLA_ADAPTER_SHA256", ""),
        "training_manifest": os.environ.get("SMOLVLA_TRAINING_MANIFEST", ""),
        "training_manifest_sha256": hashlib.sha256(pathlib.Path(os.environ["SMOLVLA_TRAINING_MANIFEST"]).read_bytes()).hexdigest(),
    },
}, indent=2, sort_keys=True) + "\n")
PY
  exit 0
fi
if [[ "$PI05_ONLY" != "1" ]]; then
  run_model "$OPENVLA_PYTHON" openvla "$OPENVLA_CHECKPOINT" "$OPENVLA_REVISION"
fi
run_model "$PYTHON" pi05 "$PI05_CHECKPOINT" "$PI05_REVISION"

"$PYTHON" - "$RUN_ROOT" <<'PY'
import json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = {"episodes": {}}
for path in sorted(root.glob("*_*.jsonl")):
    summary["episodes"][path.stem] = sum(1 for line in path.read_text().splitlines() if line.strip())
mode = os.environ.get("VLA_REPAIR", "0")
repair = mode == "1"
pi05_only = mode == "pi05_only"
expected = ({"pi05_vanilla": 100, "pi05_sealed": 100}
            if pi05_only else
            {"openvla_sealed": 100, "pi05_vanilla": 100, "pi05_sealed": 100}
            if repair else
            {"openvla_vanilla": 100, "openvla_sealed": 100, "pi05_vanilla": 100, "pi05_sealed": 100})
if summary["episodes"] != expected:
    raise SystemExit(f"evaluation completeness failure: expected {expected}, got {summary['episodes']}")
if repair:
    preserved = pathlib.Path(os.environ["PRESERVED_OPENVLA_VANILLA"])
    preserved_count = sum(1 for line in preserved.read_text().splitlines() if line.strip())
    if preserved_count != 100:
        raise SystemExit(f"preserved OpenVLA vanilla artifact must contain 100 records, got {preserved_count}")
    summary["preserved_episodes"] = {"openvla_vanilla": preserved_count}
summary["total_episodes"] = sum(summary["episodes"].values())
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
