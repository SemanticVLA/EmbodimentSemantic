#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"
: "${MANIFEST:?set MANIFEST to the completed matched Octo manifest}"
: "${CHECKPOINT_PATH:?set CHECKPOINT_PATH to the pinned Octo checkpoint directory}"
: "${DATASET_ROOT:?set DATASET_ROOT to the materialized Octo TFDS/RLDS root}"
: "${A40_MEMORY_MEASUREMENTS:?set A40_MEMORY_MEASUREMENTS to the two-update A40 VRAM receipt}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="--preflight-only"
if [[ "${PRECHECK_ONLY:-1}" == "0" ]]; then MODE="--execute"; fi
exec "$PYTHON_BIN" -m vla_benchmarking.libero.finetuned_vlas.octo.train \
  "$MODE" --dataset-manifest "$MANIFEST" --checkpoint-path "$CHECKPOINT_PATH" \
  --dataset-root "$DATASET_ROOT" \
  --memory-measurements "$A40_MEMORY_MEASUREMENTS"
