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
conda activate vla_bench

cd ~/vla_benchmarking

mkdir -p logs

# Usage:
# sbatch run_eval.sh                                      # task 1, standard mode, SmolVLA
# sbatch run_eval.sh standard "[1]"                      # task 1 only, standard mode
# sbatch run_eval.sh scene_graph "[1,3,5,9]"             # swap tasks, scene graph mode
#
# Args: $1 = CONTEXT_MODE (default: standard)
#       $2 = TASK_IDS     (default: [1])
#       $3 = MODELS       (default: HuggingFaceVLA/smolvla_libero)

export MUJOCO_GL=egl
export CONTEXT_MODE="${1:-standard}"
export TASK_IDS="${2:-[1]}"
export MODELS="${3:-HuggingFaceVLA/smolvla_libero}"
export N_EPISODES=1
export BATCH_SIZE=1
export DEVICE=cuda
export SEED=1000

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Starting VLA eval"
echo "  CONTEXT_MODE : $CONTEXT_MODE"
echo "  TASK_IDS     : $TASK_IDS"
echo "  MODELS       : $MODELS"
echo "  DEVICE       : $DEVICE"
echo "========================================================"

python dataset_eval/run_lerobot_eval_with_context.py --eval.use_async_envs=false

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Done."
echo "========================================================"
