#!/usr/bin/env bash
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-23:59:00
#SBATCH --output=logs/slurm_train_%j.out
#SBATCH --error=logs/slurm_train_%j.err

set -euo pipefail

# LoRA-fine-tune HuggingFaceVLA/smolvla_libero on the control/treatment LeRobot
# datasets produced by hdf5_to_lerobot_dataset.py (control = stock frames,
# treatment = ground-truth scene-graph arrows baked into the agentview frame).
#
# Run BOTH variants with identical flags -- only --dataset.root/--output_dir
# differ -- so arrows are the only thing that can explain a difference in the
# resulting policies. Before submitting the real (--steps=8000) job, do a short
# dry run first (STEPS=100) and confirm pretrained_model/adapter_config.json +
# adapter_model.safetensors exist in the output dir -- that's the concrete check
# that --peft.r actually triggered LoRA rather than silently falling through to
# smolvla_libero's own frozen-vision/dense-expert-only recipe.
#
# NOTE: --partition here matches this repo's existing run_eval.sh (gpu_a40).
# If your allocation uses a different partition (e.g. agent-long), override it
# with `sbatch --partition=... train_lora.sh ...` or edit the line above.
#
# Requires: pip install "lerobot[peft]"   (not installed by setup_env.py/requirements.txt)
#
# Usage:
#   sbatch train_lora.sh control
#   sbatch train_lora.sh treatment
#   STEPS=100 SAVE_FREQ=50 sbatch train_lora.sh control     # dry run

module purge
module load miniforge/24.3.0-0
# vla_bench_py312 is the only local env with lerobot[dataset]/[peft] installed
# (Python 3.12; lerobot requires >=3.12). Confirm the equivalent env name on
# this cluster -- run_eval.sh's `vla_bench` may or may not be the same env.
conda activate "${CONDA_ENV:-vla_bench_py312}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

VARIANT="${1:?usage: sbatch train_lora.sh <control|treatment> [dry]}"
if [[ "$VARIANT" != "control" && "$VARIANT" != "treatment" ]]; then
  echo "VARIANT must be 'control' or 'treatment', got '$VARIANT'" >&2
  exit 1
fi

DATASET_ROOT="${DATASET_ROOT:-$(pwd)/lora_datasets/$VARIANT}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/lora_runs/${VARIANT}_$(date +%Y_%m_%d_%H_%M_%S)}"
BASE_POLICY="${BASE_POLICY:-HuggingFaceVLA/smolvla_libero}"
PEFT_R="${PEFT_R:-16}"
STEPS="${STEPS:-8000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-1000}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Dataset root does not exist: $DATASET_ROOT" >&2
  echo "Run hdf5_to_lerobot_dataset.py --mode convert --variant $VARIANT first." >&2
  exit 1
fi

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Starting LoRA fine-tune"
echo "  variant       : $VARIANT"
echo "  base policy   : $BASE_POLICY"
echo "  dataset root  : $DATASET_ROOT"
echo "  output dir    : $OUTPUT_DIR"
echo "  peft.r        : $PEFT_R"
echo "  steps         : $STEPS (save every $SAVE_FREQ)"
echo "========================================================"

lerobot-train \
  --policy.path="$BASE_POLICY" \
  --policy.push_to_hub=false \
  --peft.r="$PEFT_R" \
  --dataset.repo_id="local/libero_spatial_${VARIANT}" \
  --dataset.root="$DATASET_ROOT" \
  --output_dir="$OUTPUT_DIR" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --eval_freq=0 \
  --batch_size="$BATCH_SIZE" \
  --policy.device="$DEVICE" \
  --seed="$SEED"

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Done. Checkpoints under $OUTPUT_DIR/checkpoints/"
echo "Sanity check before trusting this run:"
echo "  ls $OUTPUT_DIR/checkpoints/*/pretrained_model/adapter_config.json"
echo "========================================================"
