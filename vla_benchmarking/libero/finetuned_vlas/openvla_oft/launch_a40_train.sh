#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"
: "${MANIFEST:?set MANIFEST to the validated OpenVLA-OFT RLDS manifest}"
: "${CHECKPOINT_REVISION:?set CHECKPOINT_REVISION to the immutable base revision}"
: "${DATASET_ROOT:?set DATASET_ROOT to the RLDS dataset root}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to the run directory}"
: "${OPENVLA_OFT_REPO:?set OPENVLA_OFT_REPO to the pinned OpenVLA-OFT checkout}"
: "${OPENVLA_OFT_COMMIT:?set OPENVLA_OFT_COMMIT to the pinned OpenVLA-OFT commit}"
[[ -f "$OPENVLA_OFT_REPO/vla-scripts/finetune.py" ]] || { echo "missing pinned finetune entrypoint under OPENVLA_OFT_REPO" >&2; exit 2; }
ACTUAL_OPENVLA_OFT_COMMIT="$(git -C "$OPENVLA_OFT_REPO" rev-parse HEAD 2>/dev/null || true)"
[[ "$ACTUAL_OPENVLA_OFT_COMMIT" == "$OPENVLA_OFT_COMMIT" ]] || { echo "OpenVLA-OFT checkout commit mismatch: expected $OPENVLA_OFT_COMMIT, got $ACTUAL_OPENVLA_OFT_COMMIT" >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-openvla/openvla-7b}"
CHECKPOINT_PATH_ARG=()
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then CHECKPOINT_PATH_ARG=(--checkpoint-path "$CHECKPOINT_PATH"); fi
MODE="--preflight-only"
if [[ "${PRECHECK_ONLY:-1}" == "0" ]]; then MODE="--execute"; fi
exec "$PYTHON_BIN" -m vla_benchmarking.libero.finetuned_vlas.openvla_oft.train \
  "$MODE" --manifest "$MANIFEST" --checkpoint-revision "$CHECKPOINT_REVISION" \
  --checkpoint "$CHECKPOINT" "${CHECKPOINT_PATH_ARG[@]}" \
  --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT" --require-cuda \
  --upstream-repo "$OPENVLA_OFT_REPO" --upstream-commit "$OPENVLA_OFT_COMMIT"
