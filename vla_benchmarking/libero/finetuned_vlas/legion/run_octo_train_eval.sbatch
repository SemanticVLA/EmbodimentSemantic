#!/usr/bin/env bash
# Fine-tune rail-berkeley/octo-base-1.5, then evaluate the resulting
# checkpoint sequentially on 100 vanilla + 100 sealed-randomized episodes.
#SBATCH --partition=gpu_a40_ext
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=4-23:00:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/octo_train_eval_%j.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/octo_train_eval_%j.err

set -Eeuo pipefail
module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
export BASE_PYTHON="$HOME/.conda/envs/embodiment-smolvla-py312/bin/python"
test -x "$BASE_PYTHON"

export RUN_ROOT="$SCRATCH_FLASH/EmbodimentSemantic_runtime/runs/octo_train_eval_${SLURM_JOB_ID}"
export ARCHIVE_ROOT="$HOME/EmbodimentSemantic_archive/runs/octo_train_eval_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$ARCHIVE_ROOT" "$HOME/EmbodimentSemantic_runtime/operator/logs"
archive_on_exit() {
  rc=$?
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT"
  if command -v rsync >/dev/null 2>&1; then rsync -a "$RUN_ROOT/" "$ARCHIVE_ROOT/"; else cp -a "$RUN_ROOT/." "$ARCHIVE_ROOT/"; fi
  printf '{"job_id":"%s","return_code":%s,"archive_root":"%s"}\n' "$SLURM_JOB_ID" "$rc" "$ARCHIVE_ROOT" > "$ARCHIVE_ROOT/terminal.json"
  exit "$rc"
}
trap archive_on_exit EXIT

cd "$HOME/EmbodimentSemantic"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
export MUJOCO_GL=egl
export HF_HOME="${HF_HOME:-$SCRATCH_FLASH/EmbodimentSemantic_runtime/cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export PYTHONNOUSERSITE=1
nvidia-smi -L
"$BASE_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

export OCTO_ENV="$RUN_ROOT/octo_env_py310"
conda create --prefix "$OCTO_ENV" python=3.10 -y
export PYTHON="$OCTO_ENV/bin/python"
"$PYTHON" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel

OCTO_COMMIT="241fb3514b7c40957a86d869fecb7c7fc353f540"
OCTO_REPO="$RUN_ROOT/octo"
export OCTO_REPO OCTO_COMMIT
if [[ ! -d "$OCTO_REPO/.git" ]]; then git clone https://github.com/octo-models/octo.git "$OCTO_REPO"; fi
git -C "$OCTO_REPO" fetch --quiet origin "$OCTO_COMMIT"
git -C "$OCTO_REPO" checkout --detach "$OCTO_COMMIT"
[[ "$(git -C "$OCTO_REPO" rev-parse HEAD)" == "$OCTO_COMMIT" ]]
"$PYTHON" -m pip install --disable-pip-version-check -r "$OCTO_REPO/requirements.txt"
"$PYTHON" -m pip install --disable-pip-version-check h5py pillow
"$PYTHON" -m pip install --disable-pip-version-check -e "$OCTO_REPO" --no-deps
"$PYTHON" -m pip install --disable-pip-version-check \
  "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

export PYTHONPATH="$HOME/EmbodimentSemantic:$OCTO_REPO:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

SNAPSHOT_SCRIPT="$RUN_ROOT/resolve_hf_snapshot.py"
cat > "$SNAPSHOT_SCRIPT" <<'PY'
from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
import json, sys
repo, out = sys.argv[1], Path(sys.argv[2]).resolve()
info = HfApi().model_info(repo, revision=sys.argv[3])
path = Path(snapshot_download(repo_id=repo, revision=info.sha, local_dir=str(out), local_dir_use_symlinks=False)).resolve()
print(json.dumps({"repo_id": repo, "revision": info.sha, "path": str(path)}))
PY
BASE_REPO="rail-berkeley/octo-base-1.5"
BASE_REVISION="ee3c10e8edd6ce2e8b1e8744d3c6fba4097bed48"
BASE_ROOT="$RUN_ROOT/checkpoints/octo-base-1.5"
mkdir -p "$BASE_ROOT"
"$PYTHON" "$SNAPSHOT_SCRIPT" "$BASE_REPO" "$BASE_ROOT" "$BASE_REVISION" | tee "$RUN_ROOT/base_snapshot.json"

DATA_ROOT="$HOME/EmbodimentSemantic/vlm_benchmarking/data/libero_spatial_v5"
DATASET_ROOT="$RUN_ROOT/octo_tfds"
"$PYTHON" -m vla_benchmarking.libero.finetuned_vlas.legion.prepare_octo_dataset \
  --data-root "$DATA_ROOT" --run-root "$RUN_ROOT/dataset" --dataset-root "$DATASET_ROOT"
MANIFEST="$RUN_ROOT/dataset/octo_completion_manifest.json"
CONFIG_PATH="$HOME/EmbodimentSemantic/vla_benchmarking/libero/finetuned_vlas/octo/native_finetune_config.py"

# Measure every declared A40 candidate in a fresh two-update process.  OOM or
# unsupported recipes are recorded as non-fitting; the smallest successful
# candidate is selected by the checked-in receipt validator.
declare -A MICRO=( [32]=32 [16]=16 [8]=8 )
declare -A ACCUM=( [32]=1 [16]=2 [8]=4 )
declare -A PEAK
declare -A STATUS
SMOKE_OK=0
for mb in 32 16 8; do
  smoke_root="$RUN_ROOT/memory_smoke/mb${mb}"
  mkdir -p "$smoke_root"
  monitor="$smoke_root/memory_mib.txt"
  : > "$smoke_root/pid"
  (
    while [[ -f "$smoke_root/pid" ]]; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$monitor" 2>/dev/null || true
      sleep 1
    done
  ) &
  monitor_pid=$!
  microsteps=$((2 * ${ACCUM[$mb]}))
  set +e
  "$PYTHON" "$OCTO_REPO/scripts/finetune.py" \
    --config="$CONFIG_PATH:full,language_conditioned" \
    --config.pretrained_path="$BASE_ROOT" --config.pretrained_step=300000 \
    --config.dataset_kwargs.data_dir="$DATASET_ROOT" \
    --config.save_dir="$smoke_root/output" --config.num_steps="$microsteps" \
    --config.batch_size="$mb" --config.optimizer.grad_accumulation_steps="${ACCUM[$mb]}" \
    --config.seed=1000 --config.window_size=1 --config.traj_transform_kwargs.action_horizon=4 \
    > "$smoke_root/train.log" 2>&1
  rc=$?
  set -e
  rm -f "$smoke_root/pid"
  wait "$monitor_pid" || true
  peak_mib="$(awk 'BEGIN{m=0} {gsub(/[^0-9.]/,""); if ($1+0>m) m=$1+0} END{printf "%.0f",m}' "$monitor" 2>/dev/null || echo 0)"
  if [[ "$peak_mib" -le 0 ]]; then peak_mib=999999; fi
  PEAK[$mb]="$(awk -v x="$peak_mib" 'BEGIN{printf "%.6f",x/1024.0}')"
  if [[ "$rc" -eq 0 ]]; then STATUS[$mb]=PASS; SMOKE_OK=1; else STATUS[$mb]=FAILED; fi
  printf '{"microbatch":%s,"return_code":%s,"microsteps":%s,"status":"%s","peak_vram_gb":%s}\n' "$mb" "$rc" "$microsteps" "${STATUS[$mb]}" "${PEAK[$mb]}" | tee "$smoke_root/receipt.json"
done
if [[ "$SMOKE_OK" -ne 1 ]]; then
  echo "all Octo memory smoke candidates failed" >&2
  exit 1
fi

DATASET_DIGEST="$($PYTHON -c 'import json,sys; from vla_benchmarking.libero.finetuned_vlas.octo.manifest import completion_manifest_sha256; print(completion_manifest_sha256(json.load(open(sys.argv[1]))))' "$MANIFEST")"
CONFIG_DIGEST="$($PYTHON -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$CONFIG_PATH")"
OCTO_RUNTIME_SHA="$($PYTHON -c 'from vla_benchmarking.libero.finetuned_vlas.octo.preflight import runtime_closure_paths; from vla_benchmarking.libero.evaluation.policy_adapter import derive_runtime_receipt; print(derive_runtime_receipt(closure_paths=runtime_closure_paths(), require_clean=True)["sha256"])')"
"$PYTHON" -m vla_benchmarking.libero.finetuned_vlas.legion.write_octo_memory_receipt \
  --output "$RUN_ROOT/a40_memory_receipt.json" --checkpoint-revision "$BASE_REVISION" \
  --dataset-manifest-sha256 "$DATASET_DIGEST" --config-sha256 "$CONFIG_DIGEST" \
  --runtime-sha256 "$OCTO_RUNTIME_SHA" --octo-commit "$OCTO_COMMIT" \
  --peaks "${PEAK[32]}" "${PEAK[16]}" "${PEAK[8]}" \
  --statuses "${STATUS[32]}" "${STATUS[16]}" "${STATUS[8]}"

TRAIN_ROOT="$RUN_ROOT/training"
"$PYTHON" -m vla_benchmarking.libero.finetuned_vlas.octo.train \
  --execute --dataset-manifest "$MANIFEST" --checkpoint-path "$BASE_ROOT" \
  --dataset-root "$DATASET_ROOT" --memory-measurements "$RUN_ROOT/a40_memory_receipt.json" \
  --entrypoint "$HOME/EmbodimentSemantic/vla_benchmarking/libero/finetuned_vlas/octo/native_train.py" --config-path "$CONFIG_PATH" --output-dir "$TRAIN_ROOT"

# Normalize the native trainer's output into the pinned 300000/default/checkpoint
# layout expected by the strict matched-mode loader, while preserving the
# original output tree and a receipt that records the source location.
FINAL_ROOT="$RUN_ROOT/final_octo_base15_finetuned"
FINAL_CKPT="$FINAL_ROOT/300000/default/checkpoint"
mkdir -p "$FINAL_CKPT"
FINAL_RECEIPT="$TRAIN_ROOT/octo_final_checkpoint_receipt.json"
[[ -f "$FINAL_RECEIPT" ]]
FOUND="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$FINAL_RECEIPT")"
[[ -n "$FOUND" && -e "$FOUND" ]]
cp -a "$FOUND/." "$FINAL_CKPT/"
cp -a "$TRAIN_ROOT/." "$FINAL_ROOT/trainer_output/"

"$PYTHON" -m vla_benchmarking.libero.finetuned_vlas.octo.preflight \
  --mode matched_train --dataset-manifest "$MANIFEST" --checkpoint-path "$FINAL_ROOT" \
  > "$RUN_ROOT/final_preflight.json"
CHECKPOINT_SHA="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' "$RUN_ROOT/final_preflight.json")"
RUNTIME_SHA="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_sha256"])' "$RUN_ROOT/final_preflight.json")"
IO_SHA="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["io_sha256"])' "$RUN_ROOT/final_preflight.json")"
"$PYTHON" -m vla_benchmarking.libero.finetuned_vlas.legion.prepare_octo_eval_plans \
  --checkpoint-path "$FINAL_CKPT" --checkpoint-revision "$BASE_REVISION" \
  --dataset-manifest "$MANIFEST" --output-dir "$RUN_ROOT/plans"

cat > "$RUN_ROOT/run_provenance.json" <<JSON
{
  "schema": "octo_train_eval_run.v1",
  "job_id": "${SLURM_JOB_ID}",
  "source_commit": "$(git rev-parse HEAD)",
  "base": {"repo_id": "$BASE_REPO", "revision": "$BASE_REVISION", "path": "$BASE_ROOT"},
  "fine_tuned_checkpoint": {"path": "$FINAL_CKPT", "checkpoint_sha256": "$CHECKPOINT_SHA"},
  "dataset_manifest": "$MANIFEST",
  "schedule": {"tasks": 10, "episodes_per_task": 10, "conditions": ["vanilla", "sealed_randomized"], "episodes": 200},
  "runtime_sha256": "$RUNTIME_SHA",
  "io_sha256": "$IO_SHA"
}
JSON

for condition in vanilla sealed; do
  "$PYTHON" -m vla_benchmarking.libero.finetuned_vlas.octo.native_eval \
    --mode matched_train --dataset-manifest "$MANIFEST" --dataset-manifest-sha256 "$DATASET_DIGEST" \
    --checkpoint-path "$FINAL_CKPT" --checkpoint-root "$FINAL_ROOT" --checkpoint-step 300000 \
    --policy-kind octo_base15_spatial_no_arrow_matched --checkpoint-revision "$BASE_REVISION" \
    --checkpoint-sha256 "$CHECKPOINT_SHA" --runtime-sha256 "$RUNTIME_SHA" --io-sha256 "$IO_SHA" \
    --action-horizon 4 --execute-horizon 4 --plan "$RUN_ROOT/plans/octo_${condition}_100.json" \
    --output-jsonl "$RUN_ROOT/octo_${condition}.jsonl"
done

"$PYTHON" - "$RUN_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = {"episodes": {}}
for path in sorted(root.glob("octo_*.jsonl")):
    summary["episodes"][path.stem] = sum(1 for line in path.read_text().splitlines() if line.strip())
expected = {"octo_vanilla": 100, "octo_sealed": 100}
if summary["episodes"] != expected:
    raise SystemExit(f"evaluation completeness failure: expected {expected}, got {summary['episodes']}")
summary["total_episodes"] = sum(summary["episodes"].values())
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
