#!/usr/bin/env bash
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-23:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s338920@studenti.polito.it
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

module purge
module load miniforge/24.3.0-0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm

cd ~/vlm_benchmarking

# Usage:
# sbatch run_job.sh --model <model_id> --type <type> [--cameras <camera_name>] [other args...]
#
# Examples:
# sbatch run_job.sh --model google/gemma-4-E4B-it --type vllm --cameras eye_in_hand
# sbatch run_job.sh --model Qwen/Qwen2.5-VL-7B-Instruct --type vllm --cameras agentview --task-id 5
# sbatch run_job.sh --model mistralai/mistral-large-3-675b-instruct-2512 --type nvidia

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

INPUT_DIR="data/libero_spatial_v5/"
OUTPUT_DIR="output/"
DEMOS=50

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Starting run"
echo "  Args    : $@"
echo "  Demos   : $DEMOS"
echo "========================================================"

python run.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --demos "$DEMOS" \
    --verbose \
    "$@"

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Done."
echo "========================================================"