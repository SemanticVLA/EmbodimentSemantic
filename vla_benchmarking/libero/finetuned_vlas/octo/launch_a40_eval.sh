#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"
: "${CHECKPOINT_PATH:?set CHECKPOINT_PATH to the pinned Octo checkpoint directory}"
: "${OCTO_MODE:?set OCTO_MODE to community_eval or matched_train}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="--preflight-only"
if [[ "${PRECHECK_ONLY:-1}" == "0" ]]; then MODE="--execute"; fi
ARGS=("$MODE" --mode "$OCTO_MODE" --checkpoint-path "$CHECKPOINT_PATH")
if [[ "$OCTO_MODE" == "matched_train" ]]; then
  : "${MANIFEST:?set MANIFEST to the completed matched Octo manifest}"
  ARGS+=(--dataset-manifest "$MANIFEST")
fi
if [[ "${PRECHECK_ONLY:-1}" == "0" ]]; then
  : "${PLAN:?set PLAN to the sealed shared evaluation plan for native execution}"
fi
if [[ -n "${PLAN:-}" ]]; then ARGS+=(--plan "$PLAN"); fi
exec "$PYTHON_BIN" -m vla_benchmarking.libero.finetuned_vlas.octo.eval "${ARGS[@]}"
