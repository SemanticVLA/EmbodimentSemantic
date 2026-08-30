#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${LAMBDA_VENV:-$SCRIPT_DIR/.venv-lora}/bin/python"
ACTION="${1:-}"
shift || true
PROFILE=""
RUN_DIR=""
OUTPUT_ROOT=""
SEEDS=""
FINETUNING_POLICY_ID="action_only_lora_v1"
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
  echo "Usage: $0 <setup|dry|smoke|full|resume|eval> --profile <treatment|no-arrow|graph|arrow-graph> [--finetuning-policy action_only_lora_v1|action_visual_lora_v1] [--run-dir PATH] [--seeds LIST] [--output-root PATH]"
}
[[ -n "$ACTION" ]] || { usage; exit 2; }
case "$ACTION" in setup|dry|smoke|full|resume|eval) ;; -h|--help) usage; exit 0 ;; *) usage >&2; exit 2 ;; esac
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) mark_seen profile; PROFILE="${2:?--profile requires a value}"; shift 2 ;;
    --run-dir) mark_seen run-dir; RUN_DIR="${2:?--run-dir requires a value}"; shift 2 ;;
    --output-root) mark_seen output-root; OUTPUT_ROOT="${2:?--output-root requires a value}"; shift 2 ;;
    --seeds) mark_seen seeds; SEEDS="${2:?--seeds requires a value}"; shift 2 ;;
    --finetuning-policy) mark_seen finetuning-policy; FINETUNING_POLICY_ID="${2:?--finetuning-policy requires a value}"; shift 2 ;;
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
  treatment) PROFILE_CANONICAL=treatment ;;
  no-arrow) PROFILE_CANONICAL=no_arrow_treatment ;;
  graph) PROFILE_CANONICAL=graph_treatment ;;
  arrow-graph) PROFILE_CANONICAL=arrow_graph_treatment ;;
  graph_treatment) PROFILE_CANONICAL=graph_treatment ;;
  arrow_graph_treatment) PROFILE_CANONICAL=arrow_graph_treatment ;;
  *) echo "--profile must be treatment, no-arrow, graph, or arrow-graph" >&2; exit 2 ;;
esac
if [[ "$PROFILE_CANONICAL" == arrow_graph_treatment && "$ACTION" != setup ]]; then
  echo "arrow_graph_treatment is prepare-only; training/evaluation is intentionally blocked" >&2
  exit 2
fi
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
    bash "$SCRIPT_DIR/bootstrap_lambda_runtime.sh"
    if [[ "$PROFILE_CANONICAL" == graph_treatment || "$PROFILE_CANONICAL" == arrow_graph_treatment ]]; then
      # Graph preparation is downstream of the historical pair: it must carry
      # the same source HDF5/provenance contract before adding graph context.
      PYTHON="$PYTHON" bash "$SCRIPT_DIR/prepare_lambda_data.sh" treatment
      PYTHON="$PYTHON" bash "$SCRIPT_DIR/prepare_lambda_data.sh" "$PROFILE_CANONICAL"
    else
      PYTHON="$PYTHON" bash "$SCRIPT_DIR/prepare_lambda_data.sh" "$PROFILE_CANONICAL"
    fi
    BASE_POLICY_REVISION="${BASE_POLICY_REVISION:-6721902bc4d61e50a3bfdb11dfb4cb626f05d102}"
    BASE_POLICY_PATH="${BASE_POLICY:-$SCRIPT_DIR/base_models/smolvla_libero-$BASE_POLICY_REVISION}"
    PYTHON="$PYTHON" BASE_POLICY_REVISION="$BASE_POLICY_REVISION" BASE_POLICY_SNAPSHOT="$BASE_POLICY_PATH" \
      bash "$SCRIPT_DIR/prepare_base_snapshot.sh"
    if [[ "$PROFILE_CANONICAL" == graph_treatment || "$PROFILE_CANONICAL" == arrow_graph_treatment ]]; then
      GRAPH_BASE_POLICY_PATH="${GRAPH_BASE_POLICY:-$SCRIPT_DIR/base_models/smolvla_libero-$BASE_POLICY_REVISION-graph96}"
      "$PYTHON" "$SCRIPT_DIR/prompt_audit.py" --prepare-graph-policy "$BASE_POLICY_PATH" "$GRAPH_BASE_POLICY_PATH"
      "$PYTHON" "$SCRIPT_DIR/prompt_audit.py" --verify-graph-policy "$GRAPH_BASE_POLICY_PATH"
      BASE_POLICY="$GRAPH_BASE_POLICY_PATH" PYTHON="$PYTHON" bash "$SCRIPT_DIR/lambda_preflight.sh" "$PROFILE_CANONICAL"
    else
      BASE_POLICY="$BASE_POLICY_PATH" PYTHON="$PYTHON" bash "$SCRIPT_DIR/lambda_preflight.sh" "$PROFILE_CANONICAL"
    fi
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
      PYTHON="$PYTHON" TRAINING_PROFILE="$PROFILE_CANONICAL" TRAINING_MODE="$ACTION" \
      FINETUNING_POLICY_ID="$FINETUNING_POLICY_ID" \
      RUN_ROOT="$RUN_DIR" RESUME="$RESUME_VALUE" BATCH_SIZE="$TRAIN_BATCH_SIZE" SEED=1000 PEFT_R=16 DEVICE="$DEVICE" \
      RESUME_CONFIG_PATH="$RESUME_CONFIG_PATH" \
      bash "$SCRIPT_DIR/launch_lora_treatment.sh" "$ACTION"
    ;;
  eval)
    [[ -n "$RUN_DIR" ]] || { echo "eval requires --run-dir" >&2; exit 2; }
    [[ -n "$SEEDS" ]] || { echo "eval requires explicit --seeds" >&2; exit 2; }
    [[ -f "$RUN_DIR/training_manifest.json" ]] || { echo "training_manifest.json missing: $RUN_DIR" >&2; exit 1; }
    [[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="$RUN_DIR/eval"
    if [[ "$PROFILE_CANONICAL" == arrow_graph_treatment ]]; then
      echo "arrow_graph_treatment is prepare-only and has no evaluation cell" >&2
      exit 2
    elif [[ "$PROFILE_CANONICAL" == graph_treatment ]]; then
    [[ "$SEEDS" == "1000" || "$SEEDS" == "[1000]" ]] || { echo "graph pilot evaluation is sealed to --seeds 1000" >&2; exit 2; }
    mapfile -t graph_values < <("$PYTHON" - "$RUN_DIR/training_manifest.json" <<'PY'
import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if data.get("experiment") != "smolvla_lora_graph_treatment_training":
    raise SystemExit("evaluation requires a graph_treatment training manifest")
if data.get("training_variant") != "graph_treatment" or data.get("dataset_variant") != "graph_treatment":
    raise SystemExit("evaluation requires graph_treatment dataset lineage")
if data.get("trained_on_visual_condition") != "no_arrows":
    raise SystemExit("graph evaluation requires no-arrows training provenance")
adapter = data.get("graph_treatment_adapter", {}).get("path")
if not adapter:
    raise SystemExit("manifest lacks graph_treatment_adapter.path")
adapter_path = pathlib.Path(adapter).expanduser()
adapter_dir = adapter_path.parent if adapter_path.name == "adapter_model.safetensors" else adapter_path
artifact = adapter_dir / "adapter_model.safetensors"
if not adapter_dir.is_dir() or adapter_dir.name != "pretrained_model" or not artifact.is_file():
    raise SystemExit(f"graph pretrained_model adapter directory is missing: {adapter_dir}")
print(str(adapter_dir.resolve()))
PY
    )
    [[ "${#graph_values[@]}" -eq 1 ]] || { echo "could not derive graph adapter checkpoint" >&2; exit 1; }
    graph_eval_args=(--adapter-checkpoint "${graph_values[0]}" --training-manifest "$RUN_DIR/training_manifest.json" --output-root "$OUTPUT_ROOT" --seeds "$SEEDS" --episodes "$EPISODES" --batch-size "$BATCH_SIZE" --device "$DEVICE" --max-videos "$MAX_VIDEOS")
    [[ "$VIDEOS" == true ]] && graph_eval_args+=(--videos) || graph_eval_args+=(--no-videos)
    "$PYTHON" "$SCRIPT_DIR/run_lora_graph_pair_eval.py" "${graph_eval_args[@]}"
    elif [[ "$PROFILE_CANONICAL" == treatment ]]; then
    mapfile -t values < <("$PYTHON" - "$RUN_DIR/training_manifest.json" <<'PY'
import hashlib, json, pathlib, sys
d=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
if d.get('experiment') != 'smolvla_lora_treatment_training': raise SystemExit('unexpected training manifest')
if d.get('training_variant') != 'treatment': raise SystemExit('evaluation requires a treatment training manifest')
base=d.get('base_policy'); adapter=d.get('treatment_adapter',{}).get('path')
if not base or not adapter: raise SystemExit('manifest lacks base_policy or treatment_adapter.path')
base_path=pathlib.Path(base).expanduser()
adapter_path=pathlib.Path(adapter).expanduser()
if not base_path.is_dir(): raise SystemExit(f'base checkpoint directory is missing: {base_path}')
if not adapter_path.is_file() or adapter_path.name != 'adapter_model.safetensors':
    raise SystemExit(f'adapter_model.safetensors is missing: {adapter_path}')
revision=d.get('base_policy_revision')
snapshot_manifest=base_path/'base_snapshot_manifest.json'
if not revision or not snapshot_manifest.is_file(): raise SystemExit('base snapshot manifest/revision is missing')
snapshot=json.loads(snapshot_manifest.read_text(encoding='utf-8'))
if snapshot.get('revision') != revision: raise SystemExit('base snapshot revision does not match training manifest')
files=snapshot.get('files')
if not isinstance(files,dict) or not files: raise SystemExit('base snapshot manifest has no recorded files')
actual={p.relative_to(base_path).as_posix() for p in base_path.rglob('*') if p.is_file() and p.name != 'base_snapshot_manifest.json' and '.cache' not in p.parts}
if actual != set(files): raise SystemExit('base snapshot file set differs from its immutable manifest')
for name, expected in files.items():
    path=base_path/name
    if not path.is_file(): raise SystemExit(f'base snapshot file missing: {name}')
    h=hashlib.sha256(path.read_bytes()).hexdigest()
    if h != expected: raise SystemExit(f'base snapshot hash mismatch: {name}')
print(str(base_path.resolve())); print(str(adapter_path.parent.resolve()))
PY
    )
    [[ "${#values[@]}" -eq 2 ]] || { echo "could not derive eval checkpoints" >&2; exit 1; }
    eval_args=(--base-checkpoint "${values[0]}" --treatment-checkpoint "${values[1]}" --training-manifest "$RUN_DIR/training_manifest.json" --seeds "$SEEDS" --episodes "$EPISODES" --batch-size "$BATCH_SIZE" --device "$DEVICE" --output-root "$OUTPUT_ROOT" --max-videos "$MAX_VIDEOS")
    [[ "$VIDEOS" == true ]] && eval_args+=(--videos) || eval_args+=(--no-videos)
    "$PYTHON" "$SCRIPT_DIR/run_lora_2x2_eval.py" "${eval_args[@]}"
    else
    mapfile -t adapter_values < <("$PYTHON" - "$RUN_DIR/training_manifest.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("experiment") != "smolvla_lora_no_arrow_treatment_training":
    raise SystemExit("evaluation requires a no-arrow training manifest")
if data.get("training_variant") != "no_arrow_treatment" or data.get("dataset_variant") != "control":
    raise SystemExit("evaluation requires the no-arrow control dataset lineage")
if data.get("trained_on_visual_condition") != "no_arrows":
    raise SystemExit("evaluation requires no_arrows training provenance")
if "treatment_adapter" in data:
    raise SystemExit("no-arrow manifest must not contain treatment_adapter")
adapter = data.get("no_arrow_treatment_adapter", {}).get("path")
if not adapter:
    raise SystemExit("manifest lacks no_arrow_treatment_adapter.path")
adapter_path = pathlib.Path(adapter).expanduser()
adapter_dir = adapter_path.parent if adapter_path.name == "adapter_model.safetensors" else adapter_path
artifact = adapter_dir / "adapter_model.safetensors"
if not adapter_dir.is_dir() or adapter_dir.name != "pretrained_model" or not artifact.is_file():
    raise SystemExit(f"no-arrow pretrained_model adapter directory is missing: {adapter_dir}")
print(str(adapter_dir.resolve()))
PY
    )
    [[ "${#adapter_values[@]}" -eq 1 ]] || { echo "could not derive no-arrow adapter checkpoint" >&2; exit 1; }
    eval_args=(--adapter-checkpoint "${adapter_values[0]}" --training-manifest "$RUN_DIR/training_manifest.json" --seeds "$SEEDS" --episodes "$EPISODES" --batch-size "$BATCH_SIZE" --device "$DEVICE" --output-root "$OUTPUT_ROOT" --max-videos "$MAX_VIDEOS")
    [[ "$VIDEOS" == true ]] && eval_args+=(--videos) || eval_args+=(--no-videos)
    "$PYTHON" "$SCRIPT_DIR/run_lora_no_arrow_pair_eval.py" "${eval_args[@]}"
    fi
    ;;
esac
