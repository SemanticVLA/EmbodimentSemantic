#!/usr/bin/env bash
set -euo pipefail

# This wrapper is intentionally PRECHECK_ONLY=1 by default.  It is suitable
# for an A40/SLURM allocation but never submits a job on its own.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"
: "${MANIFEST:?set MANIFEST to the validated Pi0.5 dataset manifest}"
: "${CHECKPOINT_REVISION:?set CHECKPOINT_REVISION to the immutable base revision}"
: "${DATASET_ROOT:?set DATASET_ROOT to the LeRobot dataset root}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to the run directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-lerobot/pi05_libero_base}"
CHECKPOINT_PATH_ARG=()
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then CHECKPOINT_PATH_ARG=(--checkpoint-path "$CHECKPOINT_PATH"); fi

MODE="--preflight-only"
if [[ "${PRECHECK_ONLY:-1}" == "0" ]]; then MODE="--execute"; fi
exec "$PYTHON_BIN" -m vla_benchmarking.libero.finetuned_vlas.pi05.train \
  "$MODE" --manifest "$MANIFEST" --checkpoint-revision "$CHECKPOINT_REVISION" \
  --checkpoint "$CHECKPOINT" "${CHECKPOINT_PATH_ARG[@]}" \
  --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT" --require-cuda
