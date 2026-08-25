#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point for older handoff notes.  The experiment is not a
# control/treatment adapter pair: the base checkpoint is frozen and only one
# treatment adapter is trained.  Require the explicit unified CLI/profile so a
# stale pair invocation cannot silently launch the wrong condition.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "WARNING: launch_lora_pair.sh is deprecated and does not train a control/treatment pair; delegating to the unified SmolVLA CLI." >&2
exec bash "$SCRIPT_DIR/run_smolvla_pipeline.sh" "$@"
