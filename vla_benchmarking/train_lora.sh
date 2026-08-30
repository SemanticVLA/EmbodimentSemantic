#!/usr/bin/env bash
# Optional Slurm directives. Output paths are files in the submission working
# directory so Slurm can open them before this script starts.
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-23:59:00
#SBATCH --output=slurm_train_%j.out
#SBATCH --error=slurm_train_%j.err

set -euo pipefail

# Lambda is headless; keep the same renderer contract used by LIBERO evaluation.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Fine-tune a local, revision-pinned SmolVLA snapshot on one verified LIBERO
# profile. The frozen smolvla_libero snapshot remains the baseline; the
# no_arrow_treatment profile is an explicit arrow-free adapter ablation. Graph
# profiles add the canonical target-centric natural-language graph. This works as bash
# on Lambda and as sbatch on Slurm. It does not
# activate a guessed module or conda environment: put the intended Python
# environment on PATH, or set PYTHON/CONDA_RUN explicitly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
VARIANT="${1:-treatment}"
case "$VARIANT" in
  treatment)
    DATASET_VARIANT="treatment"
    DATASET_REPO_ID="local/libero_spatial_treatment"
    PAIR_MANIFEST_NAME="sealed_lora_pair_manifest.json"
    PAIR_SENTINEL_NAME="sealed_lora_pair_verified.json"
    PAIR_KIND="sealed_lora_control_treatment"
    PREFLIGHT_MODE="preflight"
    ;;
  no_arrow_treatment)
    DATASET_VARIANT="control"
    DATASET_REPO_ID="local/libero_spatial_control"
    PAIR_MANIFEST_NAME="sealed_lora_pair_manifest.json"
    PAIR_SENTINEL_NAME="sealed_lora_pair_verified.json"
    PAIR_KIND="sealed_lora_control_treatment"
    PREFLIGHT_MODE="preflight"
    ;;
  graph_treatment)
    DATASET_VARIANT="graph_treatment"
    DATASET_REPO_ID="local/libero_spatial_graph_treatment"
    PAIR_MANIFEST_NAME="sealed_lora_graph_pair_manifest.json"
    PAIR_SENTINEL_NAME="sealed_lora_graph_pair_verified.json"
    PAIR_KIND="sealed_lora_graph_treatment_arrow_graph_treatment"
    PREFLIGHT_MODE="preflight-graph"
    ;;
  arrow_graph_treatment)
    DATASET_VARIANT="arrow_graph_treatment"
    DATASET_REPO_ID="local/libero_spatial_arrow_graph_treatment"
    PAIR_MANIFEST_NAME="sealed_lora_graph_pair_manifest.json"
    PAIR_SENTINEL_NAME="sealed_lora_graph_pair_verified.json"
    PAIR_KIND="sealed_lora_graph_treatment_arrow_graph_treatment"
    PREFLIGHT_MODE="preflight-graph"
    ;;
  *)
    echo "usage: $0 <treatment|no_arrow_treatment>" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_DATA_ROOT="${DEFAULT_DATA_ROOT:-$SCRIPT_DIR/lora_datasets}"
DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
DATASET_ROOT="${DATASET_ROOT:-$DATA_ROOT/$DATASET_VARIANT}"
LIBERO_DATA_DIR="${LIBERO_DATA_DIR:-$REPO_ROOT/vlm_benchmarking/data/libero_spatial_v5}"
LIBERO_DIR="${LIBERO_DIR:-$SCRIPT_DIR/LIBERO}"
LIBERO_COMMIT="${LIBERO_COMMIT:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
LIBERO_CONFIG="${LIBERO_CONFIG:-${HOME:-}/.libero/config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/lora_runs/${VARIANT}_$(date +%Y_%m_%d_%H_%M_%S)}"
PAIR_MANIFEST="${PAIR_MANIFEST:-$DATA_ROOT/$PAIR_MANIFEST_NAME}"
PAIR_SENTINEL="${PAIR_SENTINEL:-$DATA_ROOT/$PAIR_SENTINEL_NAME}"
BASE_POLICY_REVISION="${BASE_POLICY_REVISION:-6721902bc4d61e50a3bfdb11dfb4cb626f05d102}"
BASE_POLICY_SNAPSHOT="${BASE_POLICY_SNAPSHOT:-$SCRIPT_DIR/base_models/smolvla_libero-$BASE_POLICY_REVISION}"
BASE_POLICY="${BASE_POLICY:-$BASE_POLICY_SNAPSHOT}"
PEFT_R="${PEFT_R:-16}"
# These are sealed experiment constants.  In particular, do not let a stale
# EPOCHS/STEPS/SAVE_FREQ exported by a shell or scheduler silently change the
# treatment condition.
TRAINING_MODE="${TRAINING_MODE:-full}"
SEALED_EPOCHS=15
SEALED_UPDATES_PER_EPOCH=1946
SEALED_STEPS=$((SEALED_EPOCHS * SEALED_UPDATES_PER_EPOCH))
case "$TRAINING_MODE" in
  smoke) STEPS_VALUE=2; SAVE_FREQ_VALUE=2 ;;
  full|resume) STEPS_VALUE="$SEALED_STEPS"; SAVE_FREQ_VALUE="$SEALED_UPDATES_PER_EPOCH" ;;
  *) echo "train_lora: unsupported TRAINING_MODE=$TRAINING_MODE" >&2; exit 2 ;;
esac
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-1000}"
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python}"
RESUME="${RESUME:-false}"
RESUME_CONFIG_PATH="${RESUME_CONFIG_PATH:-}"

# Graph profiles are sealed experimental conditions.  Ignore ambient shell
# overrides so direct invocation cannot silently change the paper cell.
if [[ "$VARIANT" == graph_treatment || "$VARIANT" == arrow_graph_treatment ]]; then
  PEFT_R=16
  BATCH_SIZE=32
  SEED=1000
fi

if [[ -n "${CONDA_RUN:-}" ]]; then
  # Optional explicit launcher, e.g. CONDA_RUN='conda run -n lambda_vla'.
  # shellcheck disable=SC2206
  PYTHON_CMD=(${CONDA_RUN} "$PYTHON")
else
  PYTHON_CMD=("$PYTHON")
fi

die() { echo "train_lora preflight: $*" >&2; exit 1; }
run_python() { "${PYTHON_CMD[@]}" "$@"; }

# LeRobot loads policy_preprocessor.json from policy.path.  A CLI
# tokenizer_max_length override alone does not rewrite that serialized step,
# so graph profiles use a derived immutable snapshot whose effective
# TokenizerProcessorStep is sealed to 96.  Historical profiles retain the
# original 48-token snapshot byte-for-byte.
if [[ "$VARIANT" == graph_treatment || "$VARIANT" == arrow_graph_treatment ]]; then
  GRAPH_BASE_POLICY="${GRAPH_BASE_POLICY:-$SCRIPT_DIR/base_models/smolvla_libero-$BASE_POLICY_REVISION-graph96}"
  if [[ "$BASE_POLICY" == "$BASE_POLICY_SNAPSHOT" ]]; then
    run_python "$SCRIPT_DIR/prompt_audit.py" --prepare-graph-policy "$BASE_POLICY" "$GRAPH_BASE_POLICY" \
      || die "could not prepare the sealed 96-token graph policy snapshot"
    BASE_POLICY="$GRAPH_BASE_POLICY"
  fi
fi

[[ -d "$LIBERO_DIR/.git" ]] || die "LIBERO checkout is not a git repository: $LIBERO_DIR"
[[ "$(git -C "$LIBERO_DIR" rev-parse HEAD 2>/dev/null || true)" == "$LIBERO_COMMIT" ]] || die "LIBERO checkout is not pinned to $LIBERO_COMMIT"
[[ -z "$(git -C "$LIBERO_DIR" status --porcelain --untracked-files=no 2>/dev/null || true)" ]] || die "LIBERO checkout has tracked/staged changes"
command -v "${PYTHON_CMD[0]}" >/dev/null 2>&1 || die "Python launcher not found: ${PYTHON_CMD[0]}"
run_python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' ||
  die "Python 3.12 is required (set PYTHON or CONDA_RUN explicitly)"
[[ -d "$DATASET_ROOT" ]] || die "dataset root does not exist: $DATASET_ROOT"
[[ -d "$LIBERO_DATA_DIR" ]] || die "LIBERO HDF5 source directory does not exist: $LIBERO_DATA_DIR"
[[ -d "$LIBERO_DIR/libero/libero/assets" ]] || die "LIBERO assets are missing: $LIBERO_DIR/libero/libero/assets"
[[ -d "$LIBERO_DIR/libero/libero/bddl_files" ]] || die "LIBERO BDDL assets are missing: $LIBERO_DIR/libero/libero/bddl_files"
[[ -d "$LIBERO_DIR/libero/libero/init_files" ]] || die "LIBERO init-state assets are missing: $LIBERO_DIR/libero/libero/init_files"
[[ -f "$LIBERO_CONFIG" ]] || die "LIBERO config is missing: $LIBERO_CONFIG"
[[ -d "$BASE_POLICY" ]] || die "local base snapshot does not exist: $BASE_POLICY (run prepare_base_snapshot.sh)"
[[ -f "$BASE_POLICY/config.json" ]] || die "base snapshot is missing config.json: $BASE_POLICY"
[[ -f "$BASE_POLICY/base_snapshot_manifest.json" ]] || die "base snapshot manifest is missing: $BASE_POLICY"
if [[ "$VARIANT" == graph_treatment || "$VARIANT" == arrow_graph_treatment ]]; then
  [[ -f "$BASE_POLICY/README.md" || -f "$BASE_POLICY/preprocessor_config.json" || -f "$BASE_POLICY/policy_preprocessor.json" ]] ||
    die "base snapshot does not look like a Hugging Face/LeRobot model: $BASE_POLICY"
else
  [[ -f "$BASE_POLICY/README.md" || -f "$BASE_POLICY/preprocessor_config.json" ]] ||
    die "base snapshot does not look like a Hugging Face model: $BASE_POLICY"
fi
run_python - "$BASE_POLICY/base_snapshot_manifest.json" "$BASE_POLICY" "$BASE_POLICY_REVISION" <<'PY' || die "base snapshot revision or file hashes are not pinned to the required commit"
import hashlib, json, pathlib, sys
manifest_path, root, revision = map(pathlib.Path, sys.argv[1:])
data = json.loads(manifest_path.read_text(encoding="utf-8"))
if data.get("revision") != str(revision):
    raise SystemExit(f"expected revision {revision}, got {data.get('revision')}")
for name, expected in data.get("files", {}).items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"base snapshot file hash mismatch: {name}")
PY

if [[ ! -f "$PAIR_SENTINEL" || ! -f "$PAIR_MANIFEST" ]]; then
  die "sealed pair manifest and verified sentinel are both required; run the matching converter verify mode"
fi
run_python - "$PAIR_SENTINEL" "$PAIR_MANIFEST" "$PAIR_KIND" <<'PY' || die "pair sentinel is not a verified source pair for this training profile"
import json, pathlib, sys
sentinel_path = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
expected_kind = sys.argv[3]
sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
if sentinel.get("pair_kind") != expected_kind:
    raise SystemExit(f"unexpected sentinel pair_kind: {sentinel.get('pair_kind')!r}")
if sentinel.get("full_experiment_ready") is not True or sentinel.get("launch_eligibility") != "full_experiment_ready":
    raise SystemExit("sealed pair is not marked full-experiment launchable")
if not manifest_path.is_file():
    raise SystemExit(f"sentinel references a missing pair manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("pair_kind") != expected_kind:
    raise SystemExit("pair manifest is not the sealed converter manifest")
if manifest.get("full_experiment_ready") is not True or manifest.get("launch_eligibility") != "full_experiment_ready":
    raise SystemExit("sealed pair manifest is not full-experiment launchable")
PY
echo "Using verified LIBERO pair sentinel (control frames are provenance-only): $PAIR_SENTINEL"
if [[ "$VARIANT" == graph_treatment || "$VARIANT" == arrow_graph_treatment ]]; then
  run_python - "$PAIR_MANIFEST" "$STEPS_VALUE" "$SAVE_FREQ_VALUE" "$BATCH_SIZE" "$SEED" "$PEFT_R" "$TRAINING_MODE" <<'PY' || die "graph training contract is not sealed to the canonical LoRA condition"
import json, pathlib, sys
path, steps, save_freq, batch, seed, peft_r, mode = sys.argv[1:]
contract = json.loads(pathlib.Path(path).read_text(encoding="utf-8")).get("training_contract")
expected = {"peft": "lora", "peft_r": 16, "batch_size": 32, "seed": 1000, "steps": 29190, "save_freq": 1946, "action_side_only": True}
if contract != expected:
    raise SystemExit(f"unexpected graph training_contract: {contract!r}")
if (int(batch), int(seed), int(peft_r)) != (32, 1000, 16):
    raise SystemExit("ambient graph training constants were not canonicalized")
if mode != "smoke" and (int(steps) != 29190 or int(save_freq) != 1946):
    raise SystemExit("graph full schedule is not canonical")
PY
fi

export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export CUDNN_DETERMINISTIC=1
export TOKENIZERS_PARALLELISM=false

run_python -c 'import torch; import lerobot; import peft; import accelerate; print("runtime imports OK", lerobot.__version__, peft.__version__, accelerate.__version__, torch.__version__)' ||
  die "missing/incompatible lerobot training dependencies (install requirements-lora.txt)"
run_python -c 'import h5py, cv2, PIL, numpy, robosuite; print("data/runtime imports OK")' ||
  die "missing converter/runtime dependencies (h5py, cv2, Pillow, numpy, robosuite)"
run_python -c 'from libero.libero import benchmark, get_libero_path; print("LIBERO import OK", get_libero_path("assets"))' ||
  die "LIBERO package/path import failed"
if [[ "$DEVICE" == cuda ]]; then
  run_python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' ||
    die "CUDA GPU is unavailable; use a Lambda GPU runtime or set DEVICE=cpu only for a deliberate smoke test"
fi

run_python "$SCRIPT_DIR/hdf5_to_lerobot_dataset.py" \
  --mode "$PREFLIGHT_MODE" --data-dir "$LIBERO_DATA_DIR" --output-root "$DATA_ROOT" \
  || die "sealed converter preflight failed; pair is not full-experiment launchable"
if [[ "$VARIANT" == graph_treatment || "$VARIANT" == arrow_graph_treatment ]]; then
  if [[ "$RESUME" == true ]]; then
    GRAPH_AUDIT_PENDING="$OUTPUT_DIR/graph_tokenizer_audit.json"
  else
    GRAPH_AUDIT_PENDING="${OUTPUT_DIR}.graph_tokenizer_audit.pending.json"
    [[ ! -e "$GRAPH_AUDIT_PENDING" ]] || die "stale graph tokenizer audit sidecar exists: $GRAPH_AUDIT_PENDING"
  fi
  run_python "$SCRIPT_DIR/prompt_audit.py" \
    --graph-manifest "$PAIR_MANIFEST" \
    --base-policy "$BASE_POLICY" \
    --dataset-root "$DATA_ROOT/graph_treatment" \
    --dataset-root "$DATA_ROOT/arrow_graph_treatment" \
    --audit-output "$GRAPH_AUDIT_PENDING" \
    || die "strict graph tokenizer preflight failed"
fi

# Seal the exact LoRA module/trainable-parameter inventory from the local
# LeRobot SmolVLA before training starts.  This is intentionally generated on
# the compute node (where LeRobot/PEFT are installed), never inferred from a
# checkpoint after the fact.
EXPECTED_INVENTORY="$OUTPUT_DIR/expected_adapter_inventory.json"
EXPECTED_INVENTORY_PENDING="${OUTPUT_DIR}.expected_adapter_inventory.pending.json"
if [[ "$RESUME" == true ]]; then
  if [[ -f "$EXPECTED_INVENTORY" && -f "$EXPECTED_INVENTORY_PENDING" ]]; then
    run_python - "$EXPECTED_INVENTORY" "$EXPECTED_INVENTORY_PENDING" <<'PY' || die "final and pending expected LoRA inventories differ"
import sys
from pathlib import Path
if Path(sys.argv[1]).read_bytes() != Path(sys.argv[2]).read_bytes():
    raise SystemExit(1)
PY
  fi
  if [[ ! -f "$EXPECTED_INVENTORY" && -f "$EXPECTED_INVENTORY_PENDING" ]]; then
    run_python - "$EXPECTED_INVENTORY_PENDING" "$BASE_POLICY" "$BASE_POLICY_REVISION" <<'PY' || die "pending expected LoRA inventory is not authenticated to the requested base"
import sys
from pathlib import Path
from adapter_audit import load_expected_inventory
pending, base, revision = sys.argv[1:]
value = load_expected_inventory(pending)
if value.get("base_policy") != str(Path(base).expanduser().resolve()) or value.get("base_policy_revision") != revision:
    raise SystemExit("pending expected inventory base policy/revision mismatch")
PY
    mv "$EXPECTED_INVENTORY_PENDING" "$EXPECTED_INVENTORY"
  fi
  [[ -f "$EXPECTED_INVENTORY" ]] || die "RESUME=true requires expected_adapter_inventory.json from the original run"
  run_python - "$EXPECTED_INVENTORY" <<'PY' || die "sealed expected LoRA inventory is invalid"
import sys
from adapter_audit import load_expected_inventory
load_expected_inventory(sys.argv[1])
PY
else
  [[ ! -e "$EXPECTED_INVENTORY_PENDING" ]] || die "stale expected LoRA inventory sidecar exists: $EXPECTED_INVENTORY_PENDING"
  run_python "$SCRIPT_DIR/adapter_audit.py" --generate-expected --base-policy "$BASE_POLICY" --output "$EXPECTED_INVENTORY_PENDING" \
    || die "could not build expected live-policy LoRA inventory"
fi

if [[ "$RESUME" == true ]]; then
  [[ -d "$OUTPUT_DIR" ]] || die "RESUME=true requires an existing output directory: $OUTPUT_DIR"
  if [[ -z "$RESUME_CONFIG_PATH" ]]; then
    RESUME_CONFIG_PATH="$(find "$OUTPUT_DIR/checkpoints" -type f -name train_config.json -print 2>/dev/null | sort | tail -n 1)"
  fi
  [[ -f "$RESUME_CONFIG_PATH" ]] || die "RESUME=true requires a checkpoint train_config.json under $OUTPUT_DIR/checkpoints"
  resume_root="$({ cd "$OUTPUT_DIR" && pwd -P; })"
  resume_config_abs="$({ cd "$(dirname "$RESUME_CONFIG_PATH")" && printf '%s/%s\n' "$PWD" "$(basename "$RESUME_CONFIG_PATH")"; })"
  case "$resume_config_abs" in
    "$resume_root"/*) ;;
    *) die "resume checkpoint config must be contained by OUTPUT_DIR: $RESUME_CONFIG_PATH" ;;
  esac
  [[ "$(basename "$RESUME_CONFIG_PATH")" == "train_config.json" ]] || die "resume config must be checkpoints/<step>/pretrained_model/train_config.json"
  resume_checkpoint_dir="$(cd "$(dirname "$RESUME_CONFIG_PATH")" && pwd -P)"
  [[ "$(basename "$resume_checkpoint_dir")" == "pretrained_model" ]] || die "resume config must be under a pretrained_model checkpoint directory"
  resume_checkpoint_root="$(cd "$resume_checkpoint_dir/.." && pwd -P)"
  resume_checkpoint_name="$(basename "$resume_checkpoint_root")"
  [[ "$resume_checkpoint_name" =~ ^[0-9]+$ ]] || die "resume checkpoint directory must be numeric"
  resume_checkpoint_step=$((10#$resume_checkpoint_name))
  (( resume_checkpoint_step > 0 && resume_checkpoint_step < 29190 && resume_checkpoint_step % 1946 == 0 )) || die "resume checkpoint must be an ordinary saved step in (0,29190) on the 1946-step schedule"
  [[ "$resume_checkpoint_root" == "$resume_root/checkpoints/$resume_checkpoint_name" ]] || die "resume checkpoint must resolve exactly to $resume_root/checkpoints/$resume_checkpoint_name"
  [[ -d "$resume_checkpoint_root/training_state" ]] || die "resume checkpoint lacks training_state required for optimizer/scheduler/RNG restore"
  run_python - "$resume_checkpoint_root/training_state" <<'PY' || die "resume checkpoint training_state schema is incomplete"
import pathlib, re, sys
root = pathlib.Path(sys.argv[1]).resolve()
files = [path for path in root.rglob('*') if path.is_file()]
if not files or any(path.is_symlink() for path in files):
    raise SystemExit('training_state must contain regular files only')
names = {path.name.lower() for path in files}
if not any(re.search(r'(^|[._-])optimizer([._-]|$)', name) for name in names): raise SystemExit('optimizer state is missing')
if not any(re.search(r'(^|[._-])scheduler([._-]|$)', name) for name in names): raise SystemExit('scheduler state is missing')
if not any('random_state' in name or 'rng' in name for name in names): raise SystemExit('RNG state is missing')
PY
  if [[ "$VARIANT" == graph_treatment || "$VARIANT" == arrow_graph_treatment ]]; then
    [[ -d "$resume_checkpoint_dir/tokenizer" ]] || cp -a "$BASE_POLICY/tokenizer" "$resume_checkpoint_dir/tokenizer" || die "could not restore graph checkpoint tokenizer"
    [[ -f "$resume_checkpoint_dir/tokenizer_provenance.json" ]] || cp "$BASE_POLICY/tokenizer_provenance.json" "$resume_checkpoint_dir/tokenizer_provenance.json" || die "could not restore graph checkpoint tokenizer provenance"
    run_python "$SCRIPT_DIR/prompt_audit.py" --retarget-graph-checkpoint "$resume_checkpoint_dir" || die "could not retarget graph resume preprocessor"
  fi
  echo "Resuming from checkpoint config: $RESUME_CONFIG_PATH"
else
  [[ ! -e "$OUTPUT_DIR" ]] || die "output directory already exists; choose a fresh OUTPUT_DIR: $OUTPUT_DIR"
  mkdir -p "$(dirname "$OUTPUT_DIR")"
fi
PROVENANCE_PENDING="${OUTPUT_DIR}.run_provenance.pending.json"
if [[ "$RESUME" == true ]]; then
  [[ -f "$OUTPUT_DIR/run_provenance.json" || -f "$OUTPUT_DIR/training_plan.json" ]] ||
    die "RESUME=true requires preserved run_provenance.json or training_plan.json"
else
  [[ ! -e "$PROVENANCE_PENDING" ]] || die "stale pending provenance exists; choose a fresh OUTPUT_DIR: $OUTPUT_DIR"
fi
if [[ "$RESUME" != true ]]; then
manifest_sha256="$("${PYTHON_CMD[@]}" - "$PAIR_MANIFEST" 2>/dev/null <<'PY'
import hashlib, pathlib, sys
p=pathlib.Path(sys.argv[1])
print(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "sentinel-only")
PY
)"
git_commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
git_dirty="$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | tr '\n' ' ' || true)"
run_python - "$PROVENANCE_PENDING" "$VARIANT" "$DATASET_ROOT" "$BASE_POLICY" "$BASE_POLICY_REVISION" "$manifest_sha256" "$git_commit" "$git_dirty" "$DATASET_REPO_ID" "$PAIR_KIND" "$STEPS_VALUE" "$SAVE_FREQ_VALUE" <<'PY'
import json, pathlib, platform, sys
out, variant, dataset, base, revision, pair_sha, commit, dirty, dataset_repo_id, pair_kind, steps, save_freq = sys.argv[1:]
try:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    cuda = torch.version.cuda
except Exception:
    gpu = cuda = None
import os
payload = {
    "variant": variant, "dataset_variant": pathlib.Path(dataset).name, "dataset_root": str(pathlib.Path(dataset).resolve()),
    "dataset_repo_id": dataset_repo_id, "pair_kind": pair_kind,
    "base_policy": str(pathlib.Path(base).resolve()), "base_policy_revision": revision,
    "libero_dir": str(pathlib.Path(os.environ.get("LIBERO_DIR", "")).resolve()),
    "libero_commit": os.environ.get("LIBERO_COMMIT", "8f1084e3132a39270c3a13ebe37270a43ece2a01"),
    "libero_worktree_status": "clean", "libero_tracked_clean": True,
    "pair_manifest_sha256": pair_sha, "git_commit": commit, "git_dirty": dirty,
    "python": platform.python_version(), "platform": platform.platform(),
    "gpu": gpu, "torch_cuda": cuda,
    "flags": {"steps": int(steps), "save_freq": int(save_freq), "seed": int(os.environ["PYTHONHASHSEED"]),
              "peft_r": int(os.environ.get("PEFT_R", "16")), "batch_size": int(os.environ.get("BATCH_SIZE", "32")),
              "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
              "cudnn_deterministic": True},
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
fi

echo "[$(date +'%F %T')] starting LoRA fine-tune ($VARIANT)"
echo "base=$BASE_POLICY revision=$BASE_POLICY_REVISION dataset=$DATASET_ROOT output=$OUTPUT_DIR"

TRAIN_ARGS=(
  --policy.push_to_hub=false \
  --peft.r="$PEFT_R" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --output_dir="$OUTPUT_DIR" \
  --steps="$STEPS_VALUE" \
  --save_freq="$SAVE_FREQ_VALUE" \
  --eval_freq=0 \
  --batch_size="$BATCH_SIZE" \
  --policy.device="$DEVICE" \
  --seed="$SEED"
)
if [[ "$RESUME" == true ]]; then
  TRAIN_ARGS+=(--config_path="$RESUME_CONFIG_PATH" --resume=true)
else
  TRAIN_ARGS+=(--policy.path="$BASE_POLICY")
fi
if [[ "$VARIANT" == graph_treatment || "$VARIANT" == arrow_graph_treatment ]]; then
  # Graph prompts have a larger but still sealed language budget.  Historical
  # profiles intentionally inherit SmolVLA's original 48-token setting.
  TRAIN_ARGS+=(--policy.tokenizer_max_length=96)
fi
run_python "$SCRIPT_DIR/run_lerobot_train.py" "${TRAIN_ARGS[@]}"

[[ -d "$OUTPUT_DIR" ]] || die "LeRobot did not create the output directory: $OUTPUT_DIR"
if [[ "$RESUME" != true ]]; then
  mv "$PROVENANCE_PENDING" "$OUTPUT_DIR/run_provenance.json"
  mv "$EXPECTED_INVENTORY_PENDING" "$EXPECTED_INVENTORY"
  if [[ "$VARIANT" == graph_treatment || "$VARIANT" == arrow_graph_treatment ]]; then
    [[ -f "$GRAPH_AUDIT_PENDING" ]] || die "graph tokenizer audit evidence was not produced"
    mv "$GRAPH_AUDIT_PENDING" "$OUTPUT_DIR/graph_tokenizer_audit.json"
  fi
elif [[ ! -f "$OUTPUT_DIR/run_provenance.json" && -f "$PROVENANCE_PENDING" ]]; then
  # An interrupted first attempt may have left the immutable provenance sidecar.
  # Promote that original evidence only when no finalized record exists.
  mv "$PROVENANCE_PENDING" "$OUTPUT_DIR/run_provenance.json"
fi

mapfile -t adapters < <(find "$OUTPUT_DIR/checkpoints" -type f -path '*/pretrained_model/adapter_config.json' -print 2>/dev/null)
(( ${#adapters[@]} > 0 )) || die "training completed without adapter_config.json under $OUTPUT_DIR/checkpoints"
for config in "${adapters[@]}"; do
  model="$(dirname "$config")/adapter_model.safetensors"
  [[ -s "$model" ]] || die "adapter checkpoint is missing or empty: $model"
done
if [[ "$VARIANT" == graph_treatment || "$VARIANT" == arrow_graph_treatment ]]; then
  [[ -d "$BASE_POLICY/tokenizer" && -f "$BASE_POLICY/tokenizer_provenance.json" ]] ||
    die "graph base policy lacks the sealed local tokenizer asset"
  base_tokenizer_provenance="$BASE_POLICY/tokenizer_provenance.json"
  graph_preprocessors=0
  while IFS= read -r preprocessor; do
    [[ -n "$preprocessor" ]] || continue
    checkpoint_dir="$(dirname "$preprocessor")"
    # LeRobot checkpoints may omit auxiliary processor assets.  Copy the
    # already-sealed local tokenizer into each checkpoint so reload/smoke
    # validation cannot resolve a floating Hub tokenizer.
    if [[ ! -d "$checkpoint_dir/tokenizer" ]]; then
      cp -a "$BASE_POLICY/tokenizer" "$checkpoint_dir/tokenizer" || die "could not copy graph tokenizer asset"
    fi
    if [[ ! -f "$checkpoint_dir/tokenizer_provenance.json" ]]; then
      cp "$base_tokenizer_provenance" "$checkpoint_dir/tokenizer_provenance.json" || die "could not copy graph tokenizer provenance"
    fi
    run_python "$SCRIPT_DIR/prompt_audit.py" --retarget-graph-checkpoint "$checkpoint_dir" \
      || die "could not bind graph checkpoint preprocessor to its local tokenizer"
    run_python "$SCRIPT_DIR/prompt_audit.py" --verify-graph-checkpoint "$(dirname "$preprocessor")" \
      || die "trained graph checkpoint has an invalid effective LeRobot preprocessor: $preprocessor"
    graph_preprocessors=$((graph_preprocessors + 1))
  done < <(find "$OUTPUT_DIR/checkpoints" -type f -path '*/pretrained_model/policy_preprocessor.json' -print 2>/dev/null | sort)
  (( graph_preprocessors > 0 )) || die "graph training produced no serialized policy_preprocessor evidence"
fi
# This must be the final checkpoint mutation/audit stage: graph tokenizer and
# provenance assets are copied above and are part of the sealed consumed-file
# inventory.  Historical checkpoints have no copy stage but use the same gate.
for config in "${adapters[@]}"; do
  run_python "$SCRIPT_DIR/adapter_audit.py" --checkpoint "$(dirname "$config")" \
    --expected-inventory "$EXPECTED_INVENTORY" \
    || die "checkpoint failed the action-side LoRA audit: $(dirname "$config")"
done
printf '[%s] adapter postcondition OK (%s checkpoints)\n' "$(date +'%F %T')" "${#adapters[@]}"
