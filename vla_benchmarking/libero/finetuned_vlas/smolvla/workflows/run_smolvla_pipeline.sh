#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../../../" && pwd)}"
VLA_ROOT="${VLA_ROOT:-$REPO_ROOT/vla_benchmarking/libero}"
# Keep organized roots in the environment of nested launchers.  This is
# explicit because the Windows bash bridge used by local contract tests does
# not reliably preserve non-exported shell variables across `env ... bash`.
export REPO_ROOT VLA_ROOT
PYTHON="${LAMBDA_VENV:-$VLA_ROOT/.venv-lora}/bin/python"
ACTION="${1:-}"
shift || true
PROFILE=""
RUN_DIR=""
OUTPUT_ROOT=""
SEEDS=""
EPISODES=1
BATCH_SIZE=1
DEVICE=cuda
VIDEOS=false
MAX_VIDEOS=1
RESUME_CONFIG_PATH=""
declare -A SEEN_OPTIONS=()
mark_seen() {
  local option="$1"
  if [[ -n "${SEEN_OPTIONS[$option]+seen}" ]]; then
    echo "option may be provided only once: $option" >&2
    exit 2
  fi
  SEEN_OPTIONS[$option]=1
}
abs_path() {
  local value="$1"
  if [[ "$value" == /* ]]; then
    realpath -m -- "$value"
  else
    realpath -m -- "$PWD/$value"
  fi
}
abs_executable_path() {
  local value="$1"
  local directory filename
  if [[ "$value" == /* ]]; then
    directory="${value%/*}"
    filename="${value##*/}"
  else
    value="$PWD/$value"
    directory="${value%/*}"
    filename="${value##*/}"
  fi
  [[ -n "$directory" ]] || directory=/
  # Canonicalize only the containing directory.  The final component may be
  # an executable symlink (notably .venv-lora/bin/python), and resolving it
  # here would bypass that environment's site-packages.
  printf '%s/%s\n' "$(realpath -m -- "$directory")" "$filename"
}

usage() {
  echo "Usage: $0 <setup|dry|smoke|full|resume|eval> --profile <no-arrow|target-arrow> [--run-dir PATH] [--seeds LIST] [--output-root PATH]"
}
[[ -n "$ACTION" ]] || { usage; exit 2; }
case "$ACTION" in setup|dry|smoke|full|resume|eval) ;; -h|--help) usage; exit 0 ;; *) usage >&2; exit 2 ;; esac
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) mark_seen profile; PROFILE="${2:?--profile requires a value}"; shift 2 ;;
    --run-dir) mark_seen run-dir; RUN_DIR="${2:?--run-dir requires a value}"; shift 2 ;;
    --output-root) mark_seen output-root; OUTPUT_ROOT="${2:?--output-root requires a value}"; shift 2 ;;
    --seeds) mark_seen seeds; SEEDS="${2:?--seeds requires a value}"; shift 2 ;;
    --episodes) mark_seen episodes; EPISODES="${2:?--episodes requires a value}"; shift 2 ;;
    --batch-size) mark_seen batch-size; BATCH_SIZE="${2:?--batch-size requires a value}"; shift 2 ;;
    --device) mark_seen device; DEVICE="${2:?--device requires a value}"; shift 2 ;;
    --python) mark_seen python; PYTHON="${2:?--python requires a value}"; shift 2 ;;
    --videos|--no-videos) mark_seen videos; [[ "$1" == --videos ]] && VIDEOS=true || VIDEOS=false; shift ;;
    --max-videos) mark_seen max-videos; MAX_VIDEOS="${2:?--max-videos requires a value}"; shift 2 ;;
    --resume-config) mark_seen resume-config; RESUME_CONFIG_PATH="${2:?--resume-config requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$PROFILE" in
  no-arrow|no_arrow_treatment) PROFILE_CANONICAL=no_arrow_treatment ;;
  target-arrow|target_arrow_treatment|target) PROFILE_CANONICAL=target_arrow_treatment ;;
  *) echo "--profile must be no-arrow or target-arrow" >&2; exit 2 ;;
esac
export PROFILE_CANONICAL
if [[ "$ACTION" != resume && -n "$RESUME_CONFIG_PATH" ]]; then
  echo "--resume-config is valid only with resume" >&2; exit 2
fi
if [[ "$ACTION" != eval ]]; then
  for option in episodes batch-size device videos max-videos output-root seeds; do
    if [[ -n "${SEEN_OPTIONS[$option]+seen}" ]]; then
      echo "--$option is valid only with eval" >&2; exit 2
    fi
  done
fi
if [[ "$ACTION" == setup && -n "${SEEN_OPTIONS[run-dir]+seen}" ]]; then
  echo "--run-dir is not used by setup" >&2; exit 2
fi
if [[ "$ACTION" == resume && -z "$RUN_DIR" ]]; then
  echo "resume requires --run-dir" >&2; exit 2
fi
if [[ -n "${SEEN_OPTIONS[python]+seen}" ]]; then
  if [[ "$PYTHON" == */* ]]; then
    PYTHON="$(abs_executable_path "$PYTHON")"
  else
    PYTHON="$(command -v "$PYTHON" 2>/dev/null || true)"
  fi
  [[ -n "$PYTHON" && -x "$PYTHON" ]] || { echo "explicit --python is not an executable: $PYTHON" >&2; exit 2; }
else
  PYTHON="$(abs_executable_path "$PYTHON")"
fi
if [[ -n "$RUN_DIR" ]]; then RUN_DIR="$(abs_path "$RUN_DIR")"; fi
if [[ -n "$OUTPUT_ROOT" ]]; then OUTPUT_ROOT="$(abs_path "$OUTPUT_ROOT")"; fi
if [[ -n "$RESUME_CONFIG_PATH" ]]; then RESUME_CONFIG_PATH="$(abs_path "$RESUME_CONFIG_PATH")"; fi
if [[ -n "${SEEN_OPTIONS[max-videos]+seen}" ]]; then
  [[ "$MAX_VIDEOS" =~ ^[0-9]+$ ]] || { echo "--max-videos must be a non-negative integer" >&2; exit 2; }
  if [[ "$VIDEOS" == true && "$MAX_VIDEOS" -le 0 ]]; then
    echo "--videos requires a positive --max-videos" >&2; exit 2
  fi
fi
export PYTHON

case "$ACTION" in
  setup)
    bash "$VLA_ROOT/environment/bootstrap_lambda_runtime.sh"
    # Bootstrap creates the canonical runtime under VLA_ROOT.  Refresh the
    # default interpreter after it completes so setup stages cannot retain the
    # workflow-local path that was computed before bootstrap ran.  An explicit
    # --python remains authoritative for test doubles and operator overrides.
    if [[ -z "${SEEN_OPTIONS[python]+seen}" ]]; then
      bootstrapped_python="$(abs_executable_path "${LAMBDA_VENV:-$VLA_ROOT/.venv-lora}/bin/python")"
      [[ -x "$bootstrapped_python" ]] && PYTHON="$bootstrapped_python"
      export PYTHON
    fi
    PYTHON="$PYTHON" bash "$SCRIPT_DIR/prepare_lambda_data.sh" "$PROFILE_CANONICAL"
    BASE_POLICY_REVISION="${BASE_POLICY_REVISION:-6721902bc4d61e50a3bfdb11dfb4cb626f05d102}"
    BASE_POLICY_PATH="${BASE_POLICY:-$VLA_ROOT/base_models/smolvla_libero-$BASE_POLICY_REVISION}"
    PYTHON="$PYTHON" BASE_POLICY_REVISION="$BASE_POLICY_REVISION" BASE_POLICY_SNAPSHOT="$BASE_POLICY_PATH" \
      bash "$SCRIPT_DIR/prepare_base_snapshot.sh"
    BASE_POLICY="$BASE_POLICY_PATH" PYTHON="$PYTHON" bash "$SCRIPT_DIR/lambda_preflight.sh" "$PROFILE_CANONICAL"
    ;;
  dry|smoke|full|resume)
    if [[ "$ACTION" == dry ]]; then
      [[ -n "$RUN_DIR" ]] || RUN_DIR="$SCRIPT_DIR/lora_runs/dry-run-$PROFILE_CANONICAL"
    else
      [[ -n "$RUN_DIR" ]] || RUN_DIR="$SCRIPT_DIR/lora_runs/${PROFILE_CANONICAL}_$(date +%Y_%m_%d_%H_%M_%S)"
    fi
    RUN_DIR="$(abs_path "$RUN_DIR")"
    RESUME_VALUE=false; [[ "$ACTION" == resume ]] && RESUME_VALUE=true
    TRAIN_BATCH_SIZE=32
    env -u EPOCHS -u STEPS -u SAVE_FREQ -u UPDATES_PER_EPOCH \
      REPO_ROOT="$REPO_ROOT" VLA_ROOT="$VLA_ROOT" \
      PYTHON="$PYTHON" TRAINING_PROFILE="$PROFILE_CANONICAL" TRAINING_MODE="$ACTION" \
      RUN_ROOT="$RUN_DIR" RESUME="$RESUME_VALUE" BATCH_SIZE="$TRAIN_BATCH_SIZE" SEED=1000 PEFT_R=16 DEVICE="$DEVICE" \
      DATA_ROOT="${DATA_ROOT:-}" PAIR_MANIFEST="${PAIR_MANIFEST:-}" PAIR_SENTINEL="${PAIR_SENTINEL:-}" \
      LIBERO_DATA_DIR="${LIBERO_DATA_DIR:-}" LIBERO_DIR="${LIBERO_DIR:-}" BASE_POLICY="${BASE_POLICY:-}" \
      BASE_POLICY_REVISION="${BASE_POLICY_REVISION:-}" \
      RESUME_CONFIG_PATH="$RESUME_CONFIG_PATH" \
      bash "$SCRIPT_DIR/launch_lora_treatment.sh" "$ACTION"
    ;;
  eval)
    [[ -n "$RUN_DIR" ]] || { echo "eval requires --run-dir" >&2; exit 2; }
    [[ -n "$SEEDS" ]] || { echo "eval requires explicit --seeds" >&2; exit 2; }
    [[ -f "$RUN_DIR/training_manifest.json" ]] || { echo "training_manifest.json missing: $RUN_DIR" >&2; exit 1; }
    [[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="$RUN_DIR/eval"
    mapfile -t adapter_values < <("$PYTHON" - "$RUN_DIR/training_manifest.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
profile = __import__("os").environ.get("PROFILE_CANONICAL", "no_arrow_treatment")
expected = {
    "no_arrow_treatment": ("smolvla_lora_no_arrow_treatment_training", "no_arrow_treatment", "control", "no_arrows", "no_arrow_treatment_adapter"),
    "target_arrow_treatment": ("smolvla_lora_target_arrow_treatment_training", "target_arrow_treatment", "target_arrow_treatment", "target_arrow", "target_arrow_treatment_adapter"),
}[profile]
if data.get("experiment") != expected[0] or data.get("training_variant") != expected[1] or data.get("dataset_variant") != expected[2]:
    raise SystemExit("evaluation requires a training manifest for the selected SmolVLA profile")
if data.get("trained_on_visual_condition") != expected[3]:
    raise SystemExit("training manifest visual condition does not match selected profile")
adapter = data.get(expected[4], {}).get("path")
if not adapter:
    raise SystemExit("manifest lacks the selected profile adapter path")
adapter_path = pathlib.Path(adapter).expanduser()
adapter_dir = adapter_path.parent if adapter_path.name == "adapter_model.safetensors" else adapter_path
artifact = adapter_dir / "adapter_model.safetensors"
if not adapter_dir.is_dir() or adapter_dir.name != "pretrained_model" or not artifact.is_file():
    raise SystemExit(f"selected pretrained_model adapter directory is missing: {adapter_dir}")
print(str(adapter_dir.resolve()))
PY
    )
    [[ "${#adapter_values[@]}" -eq 1 ]] || { echo "could not derive selected-profile adapter checkpoint" >&2; exit 1; }
    eval_args=(--adapter-checkpoint "${adapter_values[0]}" --training-manifest "$RUN_DIR/training_manifest.json" --profile "$PROFILE_CANONICAL" --seeds "$SEEDS" --episodes "$EPISODES" --batch-size "$BATCH_SIZE" --device "$DEVICE" --output-root "$OUTPUT_ROOT" --max-videos "$MAX_VIDEOS")
    [[ "$VIDEOS" == true ]] && eval_args+=(--videos) || eval_args+=(--no-videos)
    "$PYTHON" "$SCRIPT_DIR/run_lora_no_arrow_pair_eval.py" "${eval_args[@]}"
     ;;
esac
