#!/usr/bin/env bash
# Evaluate a completed target-arrow-only adapter through the shared SmolVLA
# pair evaluator. This launcher contains no model-specific checkpoint hash;
# operators must provide the immutable checkpoint and training manifest.
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-23:59:00

set -Eeuo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'submit with sbatch' >&2; exit 2; }
[[ -n "${REPO_ROOT:-}" && -n "${TARGET_ARROW_ADAPTER:-}" && -n "${TARGET_ARROW_TRAINING_MANIFEST:-}" ]] || {
  echo 'REPO_ROOT, TARGET_ARROW_ADAPTER, and TARGET_ARROW_TRAINING_MANIFEST are required' >&2; exit 2;
}
EXPECTED_COMMIT="${TARGET_ARROW_EXPECTED_COMMIT:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo 'repository commit drift' >&2; exit 2; }
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || { echo 'REPO_ROOT is dirty' >&2; exit 2; }
PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${TARGET_ARROW_OUTPUT_ROOT:-$REPO_ROOT/vla_benchmarking/libero/eval_outputs/target_arrow_only_${SLURM_JOB_ID}}"
EPISODES="${TARGET_ARROW_EPISODES:-10}"
SEEDS="${TARGET_ARROW_SEEDS:-1000}"
exec "$PYTHON" "$REPO_ROOT/vla_benchmarking/libero/finetuned_vlas/smolvla/workflows/run_lora_no_arrow_pair_eval.py" \
  --profile target_arrow_treatment \
  --adapter-checkpoint "$TARGET_ARROW_ADAPTER" \
  --training-manifest "$TARGET_ARROW_TRAINING_MANIFEST" \
  --seeds "$SEEDS" --episodes "$EPISODES" --batch-size 1 --device "${DEVICE:-cuda}" \
  --output-root "$OUTPUT_ROOT" --no-videos
