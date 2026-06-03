#!/usr/bin/env bash
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-23:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s338920@studenti.polito.it
#SBATCH --output=logs/slurm_eval_%j.out
#SBATCH --error=logs/slurm_eval_%j.err

module purge
module load miniforge/24.3.0-0
conda activate vlm

cd ~/vlm_benchmarking

# Usage:
# sbatch eval_job.sh --model-dir output/<name>/
# sbatch eval_job.sh --model-dir output/<name1>/ output/<name2>/
# sbatch eval_job.sh --jsonl output/<name>/<camera>/json/<file>.jsonl
# sbatch eval_job.sh --model-dir output/<name>/ --save-csv results/<name>.csv
#
# <name> is whatever was passed as --name during the run (or model ID with / replaced by --)
# All flags are passed through to evaluate.py — see python evaluate.py --help

INPUT_DIR="data/libero_spatial_v5/"

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Starting evaluation"
echo "  Args : $@"
echo "========================================================"

python evaluate.py \
    --input-dir "$INPUT_DIR" \
    "$@"

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Done."
echo "========================================================"
