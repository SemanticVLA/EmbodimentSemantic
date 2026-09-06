#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"
: "${MANIFEST:?set MANIFEST to the validated Pi0.5 dataset manifest}"
: "${CHECKPOINT_REVISION:?set CHECKPOINT_REVISION to the immutable checkpoint revision}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to the evaluation output directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-lerobot/pi05_libero_finetuned_v044}"
CHECKPOINT_PATH_ARG=()
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then CHECKPOINT_PATH_ARG=(--checkpoint-path "$CHECKPOINT_PATH"); fi
MODE="--preflight-only"
if [[ "${PRECHECK_ONLY:-1}" == "0" ]]; then MODE="--execute"; fi
PLAN_ARG=()
FACTORY_ARG=()
if [[ "${PRECHECK_ONLY:-1}" == "0" ]]; then
  : "${PLAN:?set PLAN to the sealed shared evaluation plan for native execution}"
  : "${PI05_POLICY_CONFIG_FACTORY:?set PI05_POLICY_CONFIG_FACTORY to the pinned LeRobot config factory}"
  PLAN_ARG=(--plan "$PLAN")
  FACTORY_ARG=(--policy-config-factory "$PI05_POLICY_CONFIG_FACTORY")
fi
exec "$PYTHON_BIN" -m vla_benchmarking.libero.finetuned_vlas.pi05.eval \
  "$MODE" --manifest "$MANIFEST" --checkpoint-revision "$CHECKPOINT_REVISION" \
  --checkpoint "$CHECKPOINT" "${CHECKPOINT_PATH_ARG[@]}" \
  --output-root "$OUTPUT_ROOT" --require-cuda "${PLAN_ARG[@]}" "${FACTORY_ARG[@]}"
