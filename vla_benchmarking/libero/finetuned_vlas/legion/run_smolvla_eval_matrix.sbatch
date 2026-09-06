#!/usr/bin/env bash
# Standalone SmolVLA matrix: base vanilla, base sealed-randomized, then the
# protected no-arrow LoRA vanilla cell, all sequentially on one GPU.
#
# Set SMOLVLA_CANARY_ONLY=1 for the bounded 2-task/1-episode-per-cell smoke.
# The normal run uses all ten tasks and ten episodes per task per cell.
#SBATCH --partition=gpu_a40_ext
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=4-23:00:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/smolvla_eval_matrix_%j.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/smolvla_eval_matrix_%j.err

set -Eeuo pipefail

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
export BASE_PYTHON="$HOME/.conda/envs/embodiment-smolvla-py312/bin/python"
test -x "$BASE_PYTHON"

export REPO_ROOT="${REPO_ROOT:-$HOME/EmbodimentSemantic}"
test -d "$REPO_ROOT/.git"
cd "$REPO_ROOT"
export SMOLVLA_MATRIX_EXPECTED_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=all)"

export RUN_ROOT="$SCRATCH_FLASH/EmbodimentSemantic_runtime/runs/smolvla_eval_matrix_${SLURM_JOB_ID}"
export ARCHIVE_ROOT="$HOME/EmbodimentSemantic_archive/runs/smolvla_eval_matrix_${SLURM_JOB_ID}"
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
  printf '{"job_id":"%s","return_code":%s,"archive_root":"%s"}\n' \
    "$SLURM_JOB_ID" "$rc" "$ARCHIVE_ROOT" > "$ARCHIVE_ROOT/terminal.json"
  exit "$rc"
}
trap archive_on_exit EXIT

export LIBERO_CONFIG_PATH="$RUN_ROOT/libero_config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$SCRATCH_FLASH/EmbodimentSemantic_runtime/cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

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

nvidia-smi -L
"$BASE_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
"$BASE_PYTHON" -m venv --system-site-packages "$RUN_ROOT/venv"
export PYTHON="$RUN_ROOT/venv/bin/python"
"$PYTHON" -m pip install --disable-pip-version-check --upgrade pip 'setuptools<81' wheel
"$PYTHON" -m pip install --disable-pip-version-check ml-collections sentencepiece timm
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

export SMOLVLA_BASE_PATH="/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/vla_benchmarking/base_models/smolvla_libero-6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
export SMOLVLA_BASE_REVISION="6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
export SMOLVLA_ADAPTER_PATH="/home/hjaber/EmbodimentSemantic_archive/runs/legion_no_arrow_lora_full_s1000_v1_no_arrow_treatment_1910197/checkpoints/029190/pretrained_model"
export SMOLVLA_TRAINING_MANIFEST="/home/hjaber/EmbodimentSemantic_archive/runs/legion_no_arrow_lora_full_s1000_v1_no_arrow_treatment_1910197/training_manifest.json"
export SMOLVLA_ADAPTER_SHA256="80b3c23fc3987530d57766ab45ed33db918f08983739139c1ff0397184cc7092"
export SMOLVLA_BASE_MANIFEST_SHA256="e4bcf9b4481cae4b523ef6c3af6a6a9fd5e9886215f48f911e7226221761486b"
export SMOLVLA_BASE_TREE_SHA256="d086e041f3f6bfb919f335265106fee1db8b8a3c386af357cb42a89735f74bf1"
test -d "$SMOLVLA_BASE_PATH"
test -d "$SMOLVLA_ADAPTER_PATH"
test -f "$SMOLVLA_TRAINING_MANIFEST"

SMOLVLA_CANARY_ONLY="${SMOLVLA_CANARY_ONLY:-0}"
case "$SMOLVLA_CANARY_ONLY" in
  0) PROTOCOL=full ;;
  1) PROTOCOL=smoke ;;
  *) echo "SMOLVLA_CANARY_ONLY must be 0 or 1" >&2; exit 2 ;;
esac

cat > "$RUN_ROOT/run_provenance.json" <<JSON
{
  "schema": "smolvla_eval_matrix_run.v1",
  "job_id": "${SLURM_JOB_ID}",
  "source_commit": "${SMOLVLA_MATRIX_EXPECTED_COMMIT}",
  "protocol": "${PROTOCOL}",
  "execution": "sequential_same_allocation",
  "base_policy": "${SMOLVLA_BASE_PATH}",
  "base_revision": "${SMOLVLA_BASE_REVISION}",
  "base_manifest_sha256": "${SMOLVLA_BASE_MANIFEST_SHA256}",
  "base_tree_sha256": "${SMOLVLA_BASE_TREE_SHA256}",
  "adapter_path": "${SMOLVLA_ADAPTER_PATH}",
  "adapter_sha256": "${SMOLVLA_ADAPTER_SHA256}",
  "training_manifest": "${SMOLVLA_TRAINING_MANIFEST}"
}
JSON

OUTPUT_ROOT="$RUN_ROOT/matrix_${PROTOCOL}"
"$PYTHON" -m vla_benchmarking.libero.finetuned_vlas.smolvla.workflows.run_smolvla_eval_matrix \
  --adapter-checkpoint "$SMOLVLA_ADAPTER_PATH" \
  --training-manifest "$SMOLVLA_TRAINING_MANIFEST" \
  --base-checkpoint "$SMOLVLA_BASE_PATH" \
  --output-root "$OUTPUT_ROOT" --protocol "$PROTOCOL" --device cuda --no-videos

"$BASE_PYTHON" - "$OUTPUT_ROOT" "$PROTOCOL" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
protocol = sys.argv[2]
expected_episodes = 2 if protocol == "smoke" else 100
expected_total = expected_episodes * 3
manifest = json.loads((root / "smolvla_eval_matrix_manifest.json").read_text())
schedule = json.loads((root / "smolvla_eval_matrix_schedule.json").read_text())
rows = list(csv.DictReader((root / "smolvla_eval_matrix_summary.csv").open(newline="")))
expected_cells = [
    "smolvla_base_vanilla",
    "smolvla_base_sealed_randomized",
    "smolvla_no_arrow_ft_vanilla",
]
cell_rows = [row for row in rows if row.get("row_type") == "cell"]
if [row.get("cell_id") for row in cell_rows] != expected_cells:
    raise SystemExit("SmolVLA standalone cell order is incomplete")
if any(row.get("status") != "complete" or int(row.get("episodes", -1)) != expected_episodes for row in cell_rows):
    raise SystemExit("SmolVLA standalone cell counts are incomplete")
if manifest.get("planned_episodes_total") != expected_total or len(schedule.get("cells", [])) != expected_total:
    raise SystemExit("SmolVLA standalone schedule total is invalid")
for cell in manifest["cells"]:
    info_path = pathlib.Path(cell["output_dir"]) / "eval_info.json"
    if not info_path.is_file():
        raise SystemExit(f"missing eval_info.json: {info_path}")
    info = json.loads(info_path.read_text())
    if info.get("overall", {}).get("n_episodes") != expected_episodes:
        raise SystemExit(f"invalid episode count for {cell['cell_id']}")
    audit = pathlib.Path(cell["output_dir"]) / "randomization_audit.jsonl"
    if cell["suite_mode"] == "sealed_randomized":
        records = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
        if len(records) != expected_episodes or any(item.get("status") != "ok" for item in records):
            raise SystemExit("sealed SmolVLA randomization audit is incomplete")
    elif audit.exists():
        raise SystemExit("vanilla SmolVLA cell unexpectedly emitted a randomization audit")
postcondition = {
    "schema": "smolvla_standalone_postcondition.v1",
    "protocol": protocol,
    "cell_ids": expected_cells,
    "episodes_per_cell": expected_episodes,
    "total_episodes": expected_total,
    "status": "passed",
}
(root / "matrix_postcondition.json").write_text(json.dumps(postcondition, indent=2, sort_keys=True) + "\n")
PY

echo "SmolVLA standalone matrix ${PROTOCOL} postcondition passed: ${OUTPUT_ROOT}"
