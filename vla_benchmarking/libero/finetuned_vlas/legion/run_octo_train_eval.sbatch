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

# LeRobot's pinned LIBERO API requires Python >=3.12, while the Octo training
# environment is intentionally Python 3.10.  Evaluation therefore runs as a
# separate Python 3.12 process with system-site packages (the policy/env bridge
# remains one process), and fails closed if both runtimes are not importable.
export PYTHONPATH="$HOME/EmbodimentSemantic:$OCTO_REPO:${PYTHONPATH:-}"
export OCTO_EVAL_ENV="$RUN_ROOT/octo_eval_py312"
"$BASE_PYTHON" -m venv --system-site-packages "$OCTO_EVAL_ENV"
export OCTO_EVAL_PYTHON="$OCTO_EVAL_ENV/bin/python"
"$OCTO_EVAL_PYTHON" -m pip install --disable-pip-version-check -e "$OCTO_REPO" --no-deps
# Python 3.12 has no wheels for the Octo paper's original JAX/TensorFlow
# pins.  Keep the training environment on those exact pins, and use this
# separately recorded, API-compatible inference environment for the required
# LIBERO rollout process.
"$OCTO_EVAL_PYTHON" -m pip install --disable-pip-version-check \
  "jax[cuda12]==0.4.38" \
  "flax==0.10.2" "orbax-checkpoint==0.11.28" \
  "tensorflow==2.20.0" "tensorflow-probability==0.25.0" \
  "tensorflow-datasets==4.9.9" "tensorflow-hub==0.16.1" \
  "tensorflow-text==2.20.0" "tensorflow-graphics==2021.12.3" \
  "optax==0.2.4" "chex==0.1.89" "distrax==0.1.5" \
  "ml-collections>=0.1.0" "einops>=0.6.1" "transformers==4.40.1" \
  "sentencepiece" "scipy>=1.6.0" "imageio>=2.31.1" \
  "tqdm>=4.60.0" "absl-py>=0.12.0"
"$OCTO_EVAL_PYTHON" -c 'import jax, flax, tensorflow, octo, lerobot.envs.libero, libero.libero; import sys; assert sys.version_info >= (3,12); print({"python": sys.version, "jax": jax.__version__, "flax": flax.__version__, "tensorflow": tensorflow.__version__})'
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
TRAIN_RUNTIME_SHA="$($PYTHON -c 'from vla_benchmarking.libero.finetuned_vlas.octo.preflight import runtime_closure_paths; from vla_benchmarking.libero.evaluation.policy_adapter import derive_runtime_receipt; print(derive_runtime_receipt(closure_paths=runtime_closure_paths(), require_clean=True)["sha256"])')"
"$PYTHON" -m vla_benchmarking.libero.finetuned_vlas.legion.write_octo_memory_receipt \
  --output "$RUN_ROOT/a40_memory_receipt.json" --checkpoint-revision "$BASE_REVISION" \
  --dataset-manifest-sha256 "$DATASET_DIGEST" --config-sha256 "$CONFIG_DIGEST" \
  --runtime-sha256 "$TRAIN_RUNTIME_SHA" --octo-commit "$OCTO_COMMIT" \
  --peaks "${PEAK[32]}" "${PEAK[16]}" "${PEAK[8]}" \
  --statuses "${STATUS[32]}" "${STATUS[16]}" "${STATUS[8]}"

TRAIN_ROOT="$RUN_ROOT/training"
"$PYTHON" -m vla_benchmarking.libero.finetuned_vlas.octo.train \
  --execute --dataset-manifest "$MANIFEST" --checkpoint-path "$BASE_ROOT" \
  --dataset-root "$DATASET_ROOT" --memory-measurements "$RUN_ROOT/a40_memory_receipt.json" \
  --entrypoint "$HOME/EmbodimentSemantic/vla_benchmarking/libero/finetuned_vlas/octo/native_train.py" --config-path "$CONFIG_PATH" --output-dir "$TRAIN_ROOT"

# Preserve the native Octo experiment root.  OctoModel.load_pretrained reads
# config.json, example_batch.msgpack, and dataset_statistics.json from this
# root in addition to the selected parameter checkpoint.
FINAL_RECEIPT="$TRAIN_ROOT/octo_final_checkpoint_receipt.json"
[[ -f "$FINAL_RECEIPT" ]]
FOUND="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$FINAL_RECEIPT")"
[[ -n "$FOUND" && -e "$FOUND" ]]
FOUND_ROOT="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment_root"])' "$FINAL_RECEIPT")"
FINAL_STEP="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$FINAL_RECEIPT")"
FINAL_ROOT="$FOUND_ROOT"
FINAL_CKPT="$FOUND"
[[ -f "$FINAL_ROOT/config.json" ]]
[[ -f "$FINAL_ROOT/example_batch.msgpack" ]]
[[ -f "$FINAL_ROOT/dataset_statistics.json" ]]

# Prove the final parameter tree is loadable before plans are generated.
"$PYTHON" - "$FINAL_ROOT" "$FINAL_STEP" <<'PY'
import sys
from octo.model.octo_model import OctoModel
model = OctoModel.load_pretrained(sys.argv[1], step=int(sys.argv[2]))
assert model.example_batch is not None and model.dataset_statistics is not None
print({"final_root": sys.argv[1], "final_step": int(sys.argv[2]), "reload": "PASS"})
PY

"$PYTHON" -m vla_benchmarking.libero.finetuned_vlas.octo.preflight \
  --mode matched_train --dataset-manifest "$MANIFEST" --checkpoint-path "$FINAL_ROOT" \
  > "$RUN_ROOT/final_preflight.json"
CHECKPOINT_SHA="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' "$RUN_ROOT/final_preflight.json")"
TRAIN_RUNTIME_SHA="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_sha256"])' "$RUN_ROOT/final_preflight.json")"
IO_SHA="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["io_sha256"])' "$RUN_ROOT/final_preflight.json")"
"$OCTO_EVAL_PYTHON" -m vla_benchmarking.libero.finetuned_vlas.legion.prepare_octo_eval_plans \
  --checkpoint-path "$FINAL_CKPT" --checkpoint-revision "$BASE_REVISION" \
  --dataset-manifest "$MANIFEST" --output-dir "$RUN_ROOT/plans"
EVAL_RUNTIME_SHA="$($OCTO_EVAL_PYTHON -c 'from vla_benchmarking.libero.finetuned_vlas.octo.preflight import runtime_closure_paths; from vla_benchmarking.libero.evaluation.policy_adapter import derive_runtime_receipt; print(derive_runtime_receipt(closure_paths=runtime_closure_paths(), require_clean=True)["sha256"])')"

cat > "$RUN_ROOT/run_provenance.json" <<JSON
{
  "schema": "octo_train_eval_run.v1",
  "job_id": "${SLURM_JOB_ID}",
  "source_commit": "$(git rev-parse HEAD)",
  "base": {"repo_id": "$BASE_REPO", "revision": "$BASE_REVISION", "path": "$BASE_ROOT"},
  "fine_tuned_checkpoint": {"path": "$FINAL_CKPT", "checkpoint_sha256": "$CHECKPOINT_SHA"},
  "dataset_manifest": "$MANIFEST",
  "schedule": {"tasks": 10, "episodes_per_task": 10, "conditions": ["vanilla", "sealed_randomized"], "episodes": 200},
  "runtime_sha256": "$EVAL_RUNTIME_SHA",
  "training_runtime_sha256": "$TRAIN_RUNTIME_SHA",
  "io_sha256": "$IO_SHA",
  "octo_commit": "$OCTO_COMMIT",
  "eval_python": "$OCTO_EVAL_PYTHON"
}
JSON

for condition in vanilla sealed; do
  "$OCTO_EVAL_PYTHON" -m vla_benchmarking.libero.finetuned_vlas.octo.native_eval \
    --mode matched_train --dataset-manifest "$MANIFEST" --dataset-manifest-sha256 "$DATASET_DIGEST" \
    --checkpoint-path "$FINAL_ROOT" --checkpoint-root "$FINAL_ROOT" --checkpoint-step "$FINAL_STEP" \
    --policy-kind octo_base15_spatial_no_arrow_matched --checkpoint-revision "$BASE_REVISION" \
    --checkpoint-sha256 "$CHECKPOINT_SHA" --runtime-sha256 "$EVAL_RUNTIME_SHA" --io-sha256 "$IO_SHA" \
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
