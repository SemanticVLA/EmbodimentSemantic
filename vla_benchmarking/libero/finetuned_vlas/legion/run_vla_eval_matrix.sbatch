#!/usr/bin/env bash
# Sequential OpenVLA then Pi0.5 evaluation; each model receives 100 vanilla
# and 100 sealed-randomized episodes (10 tasks x 10 episodes per condition).
#SBATCH --partition=gpu_a40_ext
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=4-23:00:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/vla_eval_matrix_%j.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/vla_eval_matrix_%j.err

set -Eeuo pipefail

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
export BASE_PYTHON="$HOME/.conda/envs/embodiment-smolvla-py312/bin/python"
test -x "$BASE_PYTHON"

export RUN_ROOT="$SCRATCH_FLASH/EmbodimentSemantic_runtime/runs/vla_eval_matrix_${SLURM_JOB_ID}"
export ARCHIVE_ROOT="$HOME/EmbodimentSemantic_archive/runs/vla_eval_matrix_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$ARCHIVE_ROOT" "$HOME/EmbodimentSemantic_runtime/operator/logs"

archive_on_exit() {
  rc=$?
  trap - EXIT
  set +e
  mkdir -p "$ARCHIVE_ROOT"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$RUN_ROOT/" "$ARCHIVE_ROOT/"
  else
    cp -a "$RUN_ROOT/." "$ARCHIVE_ROOT/"
  fi
  printf '{"job_id":"%s","return_code":%s,"archive_root":"%s"}\n' "$SLURM_JOB_ID" "$rc" "$ARCHIVE_ROOT" > "$ARCHIVE_ROOT/terminal.json"
  exit "$rc"
}
trap archive_on_exit EXIT

export REPO_ROOT="${REPO_ROOT:-$HOME/EmbodimentSemantic}"
test -d "$REPO_ROOT/.git"
cd "$REPO_ROOT"
export LIBERO_CONFIG_PATH="$RUN_ROOT/libero_config"
export MUJOCO_GL=egl
export HF_HOME="${HF_HOME:-$SCRATCH_FLASH/EmbodimentSemantic_runtime/cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export PYTHONNOUSERSITE=1

# Parse the repair mode before environment setup so Pi0.5-only reruns never
# create or touch the isolated original-OpenVLA runtime.
VLA_REPAIR="${VLA_REPAIR:-0}"
PRESERVED_OPENVLA_VANILLA="${PRESERVED_OPENVLA_VANILLA:-}"
case "$VLA_REPAIR" in
  0|1) ;;
  pi05|pi05_only|pi05-only) VLA_REPAIR="pi05_only" ;;
  *) echo "unsupported VLA_REPAIR=$VLA_REPAIR (expected 0, 1, or pi05_only)" >&2; exit 2 ;;
esac
PI05_ONLY="0"
if [[ "$VLA_REPAIR" == "pi05_only" ]]; then
  PI05_ONLY="1"
fi
PI05_CANARY_ONLY="${PI05_CANARY_ONLY:-0}"
case "$PI05_CANARY_ONLY" in
  0|1) ;;
  *) echo "unsupported PI05_CANARY_ONLY=$PI05_CANARY_ONLY (expected 0 or 1)" >&2; exit 2 ;;
esac
if [[ "$PI05_CANARY_ONLY" == "1" && "$PI05_ONLY" != "1" ]]; then
  echo "PI05_CANARY_ONLY=1 requires VLA_REPAIR=pi05_only" >&2
  exit 2
fi
if [[ "$VLA_REPAIR" == "1" ]]; then
  test -s "$PRESERVED_OPENVLA_VANILLA"
  [[ "$(awk 'NF{n++} END{print n+0}' "$PRESERVED_OPENVLA_VANILLA")" -eq 100 ]]
fi

# Each concurrent job gets its own LIBERO config and BDDL tree.  Never mutate
# the shared ~/.libero/config.yaml while another SLURM job may be rolling out.
export LIBERO_SOURCE_ROOT="$HOME/EmbodimentSemantic/vla_benchmarking/LIBERO"
export LIBERO_SOURCE_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"
export LIBERO_BDDL_SOURCE="$LIBERO_SOURCE_ROOT/libero/libero/bddl_files"
export LIBERO_BDDL_TARGET="$RUN_ROOT/libero_cache/bddl_files"
export LIBERO_INIT_SOURCE="$LIBERO_SOURCE_ROOT/libero/libero/init_files"
export LIBERO_INIT_TARGET="$RUN_ROOT/libero_cache/init_files"
test -d "$LIBERO_BDDL_SOURCE"
test -d "$LIBERO_INIT_SOURCE"
[[ "$(git -C "$LIBERO_SOURCE_ROOT" rev-parse HEAD)" == "$LIBERO_SOURCE_COMMIT" ]]
[[ -z "$(git -C "$LIBERO_SOURCE_ROOT" status --porcelain --untracked-files=all)" ]]
mkdir -p "$LIBERO_CONFIG_PATH" "$LIBERO_BDDL_TARGET" "$LIBERO_INIT_TARGET"
rsync -a "$LIBERO_BDDL_SOURCE/" "$LIBERO_BDDL_TARGET/"
rsync -a "$LIBERO_INIT_SOURCE/" "$LIBERO_INIT_TARGET/"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $HOME/.cache/libero/benchmark
bddl_files: $LIBERO_BDDL_TARGET
init_states: $LIBERO_INIT_TARGET
datasets: $HOME/.cache/libero/datasets
assets: $HOME/.cache/libero/assets
YAML
export LIBERO_BDDL_TREE_SHA256="$($BASE_PYTHON -c 'from vla_benchmarking.libero.evaluation.plan import libero_source_hashes; print(libero_source_hashes()["libero_bddl_tree_sha256"])')"

nvidia-smi -L
"$BASE_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

"$BASE_PYTHON" -m venv --system-site-packages "$RUN_ROOT/venv"
export PYTHON="$RUN_ROOT/venv/bin/python"
"$PYTHON" -m pip install --disable-pip-version-check --upgrade pip 'setuptools<81' wheel

"$PYTHON" -m pip install --disable-pip-version-check ml-collections sentencepiece timm

# Original OpenVLA requires the pre-5.x AutoModelForVision2Seq API.  LeRobot's
# pinned Pi05 extra intentionally stays on transformers 5.3.0, so keep the
# original OpenVLA runtime in a separate provenance-bound environment.
if [[ "$PI05_ONLY" != "1" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$RUN_ROOT/openvla_venv"
  export OPENVLA_PYTHON="$RUN_ROOT/openvla_venv/bin/python"
  "$OPENVLA_PYTHON" -m pip install --disable-pip-version-check --upgrade pip 'setuptools<81' wheel
  "$OPENVLA_PYTHON" -m pip install --disable-pip-version-check \
    -r vla_benchmarking/libero/finetuned_vlas/openvla/requirements.txt
  "$OPENVLA_PYTHON" -c 'import transformers; from transformers import AutoModelForVision2Seq; assert transformers.__version__ == "4.40.1"; print("original OpenVLA transformers", transformers.__version__)'
fi
"$PYTHON" -c 'import transformers; print("Pi05 transformers", transformers.__version__)'

SNAPSHOT_SCRIPT="$RUN_ROOT/resolve_hf_snapshot.py"
cat > "$SNAPSHOT_SCRIPT" <<'PY'
from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
import json, os, sys
repo, out, revision = sys.argv[1], Path(sys.argv[2]).resolve(), sys.argv[3]
token = os.environ.get("HF_TOKEN") or None
info = HfApi(token=token).model_info(repo, revision=revision)
if info.sha != revision:
    raise SystemExit(f"resolved Hub revision mismatch for {repo}: {info.sha} != {revision}")
path = Path(snapshot_download(repo_id=repo, revision=revision, local_dir=str(out), local_dir_use_symlinks=False)).resolve()
print(json.dumps({"repo_id": repo, "revision": revision, "path": str(path)}))
PY

resolve_snapshot() {
  local repo="$1"; local name="$2"; local revision="$3"; local result
  result="$($PYTHON "$SNAPSHOT_SCRIPT" "$repo" "$RUN_ROOT/checkpoints/$name" "$revision")"
  printf '%s\n' "$result" | tee "$RUN_ROOT/${name}_snapshot.json"
  printf '%s\n' "$result" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["revision"])'
}

PI05_REPO="lerobot/pi05_libero_finetuned_v044"
OPENVLA_REPO_ID="openvla/openvla-7b-finetuned-libero-spatial"
PI05_REVISION="8e174154ef5f6c60a8da12ae99c303d8963138c1"
OPENVLA_REVISION="962318cec55ac10993ff0f5f43eda9a270b4c873"
PI05_CHECKPOINT="$RUN_ROOT/checkpoints/pi05"
OPENVLA_CHECKPOINT="$RUN_ROOT/checkpoints/openvla"

# The Pi0.5-only repair path is the only path that needs the gated PaliGemma
# tokenizer.  Read exactly HF_TOKEN from the protected dotenv file without
# importing unrelated secrets into the evaluation environment.
PI05_TOKENIZER_REPO="google/paligemma-3b-pt-224"
PI05_TOKENIZER_PATH=""
PI05_TOKENIZER_REVISION="${PI05_TOKENIZER_REVISION:-}"
if [[ "$PI05_ONLY" == "1" ]]; then
  OMNIS_SECRETS_ENV="$HOME/.config/omnis/secrets.env"
  test -f "$OMNIS_SECRETS_ENV" || { echo "missing protected secrets file: $OMNIS_SECRETS_ENV" >&2; exit 2; }
  test -r "$OMNIS_SECRETS_ENV" || { echo "secrets file is not readable: $OMNIS_SECRETS_ENV" >&2; exit 2; }
  unset HF_TOKEN
  # Source the protected dotenv only in a short-lived subshell and capture
  # exactly HF_TOKEN.  Unrelated variables therefore cannot enter this job's
  # environment, and the token value is never printed or persisted.
  HF_TOKEN="$(
    set +u
    # shellcheck disable=SC1090
    source "$OMNIS_SECRETS_ENV"
    printf '%s' "${HF_TOKEN:-}"
  )"
  export HF_TOKEN
  test -n "$HF_TOKEN" || { echo "HF_TOKEN is empty in protected secrets.env" >&2; exit 2; }
  [[ "$PI05_TOKENIZER_REVISION" =~ ^[0-9a-fA-F]{40}$ ]] || {
    echo "PI05_TOKENIZER_REVISION must be an exact 40-character commit hash" >&2
    exit 2
  }
  PI05_TOKENIZER_REVISION="${PI05_TOKENIZER_REVISION,,}"
  export PI05_TOKENIZER_REVISION
  resolve_snapshot "$PI05_TOKENIZER_REPO" pi05_tokenizer "$PI05_TOKENIZER_REVISION" >/dev/null
  PI05_TOKENIZER_PATH="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$RUN_ROOT/pi05_tokenizer_snapshot.json")"
  test -d "$PI05_TOKENIZER_PATH" || { echo "Pi0.5 tokenizer snapshot directory is unavailable" >&2; exit 2; }
  [[ -n "$(find "$PI05_TOKENIZER_PATH" -type f -print -quit)" ]] || {
    echo "Pi0.5 tokenizer snapshot is empty" >&2
    exit 2
  }
  export PI05_TOKENIZER_PATH
fi
resolve_snapshot "$PI05_REPO" pi05 "$PI05_REVISION" >/dev/null
if [[ "$PI05_ONLY" != "1" ]]; then
  resolve_snapshot "$OPENVLA_REPO_ID" openvla "$OPENVLA_REVISION" >/dev/null
else
  # The protected token is needed only for authenticated Hub resolution.
  # Remove it before writing provenance or entering canary/evaluation code.
  unset HF_TOKEN
fi

DATASET_MANIFEST="$RUN_ROOT/libero_spatial_eval_manifest.json"
"$PYTHON" - "$DATASET_MANIFEST" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
source = pathlib.Path("vlm_benchmarking/data/libero_spatial_v5/libero_spatial_v5_source_manifest.json")
payload = json.loads(source.read_text()) if source.is_file() else {}
payload.update({"schema": "libero_vla_eval_manifest.v1", "suite": "libero_spatial", "tasks": list(range(10)), "episodes_per_task": 10, "conditions": ["vanilla", "sealed_randomized"], "total_episodes": 200})
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

PLAN_DIR="$RUN_ROOT/plans"
if [[ "$PI05_ONLY" != "1" ]]; then
  "$OPENVLA_PYTHON" -m vla_benchmarking.libero.finetuned_vlas.legion.prepare_eval_plans \
    --model openvla --checkpoint-path "$OPENVLA_CHECKPOINT" \
    --checkpoint-revision "$OPENVLA_REVISION" --dataset-manifest "$DATASET_MANIFEST" --output-dir "$PLAN_DIR"
fi
"$PYTHON" -m vla_benchmarking.libero.finetuned_vlas.legion.prepare_eval_plans \
  --model pi05 --checkpoint-path "$PI05_CHECKPOINT" \
  --checkpoint-revision "$PI05_REVISION" --dataset-manifest "$DATASET_MANIFEST" --output-dir "$PLAN_DIR"

cat > "$RUN_ROOT/run_provenance.json" <<JSON
{
  "schema": "vla_eval_matrix_run.v1",
  "job_id": "${SLURM_JOB_ID}",
  "source_commit": "$(git rev-parse HEAD)",
  "repair_mode": $([[ "$VLA_REPAIR" == "1" ]] && echo true || echo false),
  "vla_repair_mode": "$VLA_REPAIR",
  "preserved_openvla_vanilla": "${PRESERVED_OPENVLA_VANILLA}",
  "pi05_tokenizer": {"repo_id": "$PI05_TOKENIZER_REPO", "revision": "$PI05_TOKENIZER_REVISION", "path": "$PI05_TOKENIZER_PATH"},
  "models": {
    "openvla": {"repo_id": "$OPENVLA_REPO_ID", "revision": "$OPENVLA_REVISION", "path": "$OPENVLA_CHECKPOINT", "fine_tuned_again": false},
    "pi05": {"repo_id": "$PI05_REPO", "revision": "$PI05_REVISION", "path": "$PI05_CHECKPOINT", "fine_tuned_again": false}
  },
  "schedule": {"tasks": 10, "episodes_per_task": 10, "conditions": ["vanilla", "sealed_randomized"], "episodes_per_model": 200},
  "libero_source": {"commit": "$LIBERO_SOURCE_COMMIT", "bddl_tree_sha256": "$LIBERO_BDDL_TREE_SHA256"},
  "dataset_manifest": "$DATASET_MANIFEST"
}
JSON

if [[ "$PI05_CANARY_ONLY" == "1" ]]; then
  # Reuse the exact resolved Pi0.5 checkpoint and local gated tokenizer.  The
  # adapter's predict() path invokes one native predict_action_chunk call and
  # enforces the unchanged (50, 7) contract before this bounded canary exits.
  "$PYTHON" - "$PI05_CHECKPOINT" "$PI05_REVISION" "$PI05_TOKENIZER_PATH" "$PI05_TOKENIZER_REVISION" "$RUN_ROOT/pi05_canary.json" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

from vla_benchmarking.libero.finetuned_vlas.pi05 import Pi05Adapter
from vla_benchmarking.libero.finetuned_vlas.pi05.contracts import PolicyArtifact

checkpoint, revision, tokenizer_path, tokenizer_revision, receipt_path = sys.argv[1:]
artifact = PolicyArtifact(
    model_id="lerobot/pi05_libero_finetuned_v044",
    revision=revision,
    checkpoint_path=checkpoint,
    provenance="legion_pi05_canary",
)
adapter = Pi05Adapter.load(artifact=artifact)
adapter.reset("Canary: inspect the scene and pick up the bowl.", 0)
observation = {
    "observation.images.image": np.zeros((256, 256, 3), dtype=np.uint8),
    "observation.images.image2": np.zeros((256, 256, 3), dtype=np.uint8),
    "observation.state": np.zeros((8,), dtype=np.float32),
}
action = adapter.predict(observation, task="Canary: inspect the scene and pick up the bowl.")
if action.shape != (50, 7):
    raise SystemExit(f"Pi0.5 canary action shape mismatch: {action.shape}")
if not np.isfinite(action).all():
    raise SystemExit("Pi0.5 canary action contains non-finite values")
receipt = {
    "schema": "pi05_canary.v1",
    "status": "passed",
    "model_repo": artifact.model_id,
    "model_revision": revision,
    "checkpoint_path": str(Path(checkpoint).resolve()),
    "tokenizer_repo": "google/paligemma-3b-pt-224",
    "tokenizer_revision": tokenizer_revision,
    "tokenizer_path": str(Path(tokenizer_path).resolve()),
    "action_shape": list(action.shape),
    "finite": True,
}
Path(receipt_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, sort_keys=True))
PY
  exit 0
fi

run_model() {
  local python_bin="$1"; local model="$2"; local ckpt="$3"; local revision="$4"; local factory="${5:-}"
  local extra=()
  if [[ -n "$factory" ]]; then extra+=(--policy-config-factory "$factory"); fi
  local conditions=(vanilla sealed)
  if [[ "$VLA_REPAIR" == "1" && "$model" == "openvla" ]]; then
    conditions=(sealed)
  fi
  for condition in "${conditions[@]}"; do
    echo "START model=$model condition=$condition job=$SLURM_JOB_ID"
    "$python_bin" -m vla_benchmarking.libero.evaluation.native_vla_eval \
      --model "$model" --plan "$PLAN_DIR/${model}_${condition}_100.json" \
      --checkpoint-path "$ckpt" --checkpoint-revision "$revision" \
      --dataset-manifest "$DATASET_MANIFEST" \
      --output-jsonl "$RUN_ROOT/${model}_${condition}.jsonl" "${extra[@]}"
    echo "DONE model=$model condition=$condition job=$SLURM_JOB_ID"
  done
}

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
if [[ "$PI05_ONLY" != "1" ]]; then
  run_model "$OPENVLA_PYTHON" openvla "$OPENVLA_CHECKPOINT" "$OPENVLA_REVISION"
fi
run_model "$PYTHON" pi05 "$PI05_CHECKPOINT" "$PI05_REVISION"

"$PYTHON" - "$RUN_ROOT" <<'PY'
import json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = {"episodes": {}}
for path in sorted(root.glob("*_*.jsonl")):
    summary["episodes"][path.stem] = sum(1 for line in path.read_text().splitlines() if line.strip())
mode = os.environ.get("VLA_REPAIR", "0")
repair = mode == "1"
pi05_only = mode == "pi05_only"
expected = ({"pi05_vanilla": 100, "pi05_sealed": 100}
            if pi05_only else
            {"openvla_sealed": 100, "pi05_vanilla": 100, "pi05_sealed": 100}
            if repair else
            {"openvla_vanilla": 100, "openvla_sealed": 100, "pi05_vanilla": 100, "pi05_sealed": 100})
if summary["episodes"] != expected:
    raise SystemExit(f"evaluation completeness failure: expected {expected}, got {summary['episodes']}")
if repair:
    preserved = pathlib.Path(os.environ["PRESERVED_OPENVLA_VANILLA"])
    preserved_count = sum(1 for line in preserved.read_text().splitlines() if line.strip())
    if preserved_count != 100:
        raise SystemExit(f"preserved OpenVLA vanilla artifact must contain 100 records, got {preserved_count}")
    summary["preserved_episodes"] = {"openvla_vanilla": preserved_count}
summary["total_episodes"] = sum(summary["episodes"].values())
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
