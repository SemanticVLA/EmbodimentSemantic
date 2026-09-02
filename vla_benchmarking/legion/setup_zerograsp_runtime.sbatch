#!/usr/bin/env bash
# Compute-node-only ZeroGrasp runtime setup and no-motion model smoke.
# Submit with sbatch; direct execution is refused.
#SBATCH --job-name=zerograsp_setup_smoke
#SBATCH --partition=gpu_a40
#SBATCH --exclude=compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.err

set -Eeuo pipefail
umask 027
die() { printf 'ZeroGrasp setup job: %s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit with sbatch; direct execution is refused'
[[ -n "${REPO_ROOT:-}" && -d "$REPO_ROOT" ]] || die 'REPO_ROOT evaluation checkout is required'
EXPECTED_EVAL_COMMIT="${ARROW_MATRIX_EXPECTED_COMMIT:-}"
[[ "$EXPECTED_EVAL_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'ARROW_MATRIX_EXPECTED_COMMIT must be a lowercase 40-character SHA'
SETUP_MODE="${ZERO_GRASP_SETUP_MODE:-acquire}"
[[ "$SETUP_MODE" == "acquire" || "$SETUP_MODE" == "execute" ]] || die 'ZERO_GRASP_SETUP_MODE must be acquire or execute'
if [[ "$SETUP_MODE" == "execute" ]]; then
  [[ "${ZERO_GRASP_CHECKPOINT_SHA256:-}" =~ ^[0-9a-fA-F]{64}$ ]] || die 'execute mode requires explicit ZERO_GRASP_CHECKPOINT_SHA256'
fi

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ZERO_GRASP_ROOT="${ZERO_GRASP_ROOT:?ZERO_GRASP_ROOT is required}"
[[ "$ZERO_GRASP_ROOT" = /* && "$ZERO_GRASP_ROOT" != / ]] || die 'ZERO_GRASP_ROOT must be a non-root absolute path'
RUNTIME_ROOT="${ZERO_GRASP_RUNTIME_ROOT:-${SCRATCH_FLASH:?SCRATCH_FLASH is required}/ZeroGrasp_runtime}"
ARCHIVE_ROOT="${ZERO_GRASP_ARCHIVE_ROOT:-${HOME}/EmbodimentSemantic_archive/zerograsp_setup/${SLURM_JOB_ID}}"
mkdir -p "$RUNTIME_ROOT" "$ARCHIVE_ROOT"
exec > >(tee -a "$ARCHIVE_ROOT/setup.log") 2>&1
archive_on_exit() {
  rc=$?
  printf 'setup_exit_code=%s\n' "$rc" > "$ARCHIVE_ROOT/exit_status"
  {
    printf 'evaluation_commit=%s\n' "${EVAL_COMMIT:-}"
    printf 'expected_evaluation_commit=%s\n' "${EXPECTED_EVAL_COMMIT:-}"
    printf 'setup_script=%s\nsetup_script_sha256=%s\n' "${SETUP_SCRIPT:-}" "${SETUP_SCRIPT_SHA256:-}"
    printf 'worker_script=%s\nworker_script_sha256=%s\n' "${WORKER_SCRIPT:-}" "${WORKER_SCRIPT_SHA256:-}"
    printf 'setup_mode=%s\n' "${SETUP_MODE:-}"
  } > "$ARCHIVE_ROOT/setup_provenance.env" || true
  if [[ -n "${LOCK:-}" && -f "$LOCK" ]]; then cp -- "$LOCK" "$ARCHIVE_ROOT/zerograsp-runtime.lock.partial.json" || true; fi
  exit "$rc"
}
trap archive_on_exit EXIT
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")" || die 'cannot resolve REPO_ROOT'
SCRIPT_DIR="$(realpath -e -- "$SCRIPT_DIR")" || die 'cannot resolve setup script directory'
[[ "$SCRIPT_DIR" == "$REPO_ROOT/vla_benchmarking/legion" ]] || die 'setup script is not from the expected evaluation checkout'
EVAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)" || die 'cannot read evaluation checkout commit'
[[ "$EVAL_COMMIT" == "$EXPECTED_EVAL_COMMIT" ]] || die "evaluation checkout differs: $EVAL_COMMIT"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'evaluation checkout is dirty'
SETUP_SCRIPT="$REPO_ROOT/vla_benchmarking/legion/setup_zerograsp_runtime.sbatch"
WORKER_SCRIPT="$REPO_ROOT/vla_benchmarking/zerograsp_worker.py"
SETUP_SCRIPT_SHA256="$(sha256sum -- "$SETUP_SCRIPT" | awk '{print $1}')" || die 'cannot hash setup script'
WORKER_SCRIPT_SHA256="$(sha256sum -- "$WORKER_SCRIPT" | awk '{print $1}')" || die 'cannot hash worker script'
# Clone/pin the source before resolving any files beneath it. This makes a
# first run with an absent checkout follow the same order as later reruns.
bash "$SCRIPT_DIR/bootstrap_zerograsp.sh" --root "$ZERO_GRASP_ROOT" || die 'ZeroGrasp checkout bootstrap failed'
ZERO_GRASP_ROOT="$(realpath -e -- "$ZERO_GRASP_ROOT")" || die 'cannot resolve ZeroGrasp checkout'
VENV="${ZERO_GRASP_VENV:-${RUNTIME_ROOT}/venv-py311}"
CHECKPOINT="${ZERO_GRASP_CHECKPOINT:-${RUNTIME_ROOT}/epoch=1-step=80000.ckpt}"
CONFIG="${ZERO_GRASP_CONFIG:-${ZERO_GRASP_ROOT}/configs/demo.yaml}"
LOCK="${ZERO_GRASP_ENV_LOCK:-${RUNTIME_ROOT}/zerograsp-runtime.lock.json}"
ACQUISITION_MANIFEST="${ZERO_GRASP_CHECKPOINT_ACQUISITION_MANIFEST:-${RUNTIME_ROOT}/checkpoint-acquisition.json}"
mkdir -p "$RUNTIME_ROOT" "$ARCHIVE_ROOT" "$(dirname -- "$CHECKPOINT")" "$(dirname -- "$LOCK")"
case "$VENV" in "$ZERO_GRASP_ROOT"|"$ZERO_GRASP_ROOT"/*) die 'venv must be outside the ZeroGrasp checkout' ;; esac
case "$LOCK" in "$ZERO_GRASP_ROOT"|"$ZERO_GRASP_ROOT"/*) die 'lock must be outside the ZeroGrasp checkout' ;; esac
if [[ -e "$VENV" ]]; then
  [[ -x "$VENV/bin/python" ]] || die 'existing venv is incomplete; refusing to overwrite it'
else
  conda create -y -p "$VENV" python=3.11 pip || die 'could not create explicit Python 3.11 conda environment'
fi
PYTHON="$(realpath -e -- "$VENV/bin/python")" || die 'cannot resolve isolated Python'
PYTHON_MM="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$PYTHON_MM" == 3.11 ]] || die "official runtime requires Python 3.11 (found $PYTHON_MM)"

readonly DRIVE_ID="1xUmFdgT_Ozu4zIPIsh_1SJMcegeQUWqQ"
if [[ -e "$CHECKPOINT" ]]; then
  [[ -f "$CHECKPOINT" ]] || die "checkpoint path is not a regular file: $CHECKPOINT"
else
  "$PYTHON" -m pip install --disable-pip-version-check 'gdown==5.2.0' || die 'pinned gdown install failed'
  DOWNLOAD_TMP="${CHECKPOINT}.partial"
  [[ ! -e "$DOWNLOAD_TMP" ]] || die "partial checkpoint exists; refusing to overwrite: $DOWNLOAD_TMP"
  "$PYTHON" -m gdown --id "$DRIVE_ID" --output "$DOWNLOAD_TMP" --quiet || die 'official Drive checkpoint download failed'
  [[ -f "$DOWNLOAD_TMP" && -s "$DOWNLOAD_TMP" ]] || die 'downloaded checkpoint is empty or missing'
  [[ ! -e "$CHECKPOINT" ]] || die 'checkpoint appeared during download; refusing replacement'
  mv -- "$DOWNLOAD_TMP" "$CHECKPOINT" || die 'could not publish downloaded checkpoint'
fi
CHECKPOINT="$(realpath -e -- "$CHECKPOINT")" || die 'cannot resolve checkpoint'
chmod a-w "$CHECKPOINT" || die 'could not mark checkpoint immutable'
CHECKPOINT_MODE="$(stat -c %a "$CHECKPOINT")"
[[ "$CHECKPOINT_MODE" =~ ^[0145]{3}$ ]] || die "checkpoint retains write bits: mode $CHECKPOINT_MODE"
CHECKPOINT_SHA="$(sha256sum "$CHECKPOINT" | awk '{print tolower($1)}')"
CHECKPOINT_SIZE="$(stat -c %s "$CHECKPOINT")"
if [[ "$SETUP_MODE" == "execute" ]]; then
  CONFIG="$(realpath -e -- "$CONFIG")" || die 'cannot resolve model config'
  [[ -f "$CONFIG" ]] || die "model config is not a regular file: $CONFIG"
  CONFIG_SHA="$(sha256sum "$CONFIG" | awk '{print tolower($1)}')"
  [[ "$CHECKPOINT_SHA" == "${ZERO_GRASP_CHECKPOINT_SHA256,,}" ]] || die 'checkpoint hash differs from supplied expectation'
fi
if [[ -f "$LOCK" ]]; then
  OLD_CHECKPOINT_SHA="$($PYTHON - "$LOCK" <<'PY'
import json, pathlib, sys
try:
    data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    value = data.get("checkpoint_sha256", "")
    if not value:
        raise ValueError("checkpoint hash is missing")
    print(value)
except Exception as exc:
    raise SystemExit(f"invalid existing runtime lock: {exc}")
PY
  )"
  [[ "$OLD_CHECKPOINT_SHA" == "$CHECKPOINT_SHA" ]] || die 'checkpoint differs from existing immutable lock'
fi

if [[ -f "$ACQUISITION_MANIFEST" ]]; then
  OLD_ACQUISITION_SHA="$($PYTHON - "$ACQUISITION_MANIFEST" <<'PY'
import json, pathlib, sys
try:
    value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")).get("checkpoint_sha256", "")
    if not value:
        raise ValueError("checkpoint hash is missing")
    print(value)
except Exception as exc:
    raise SystemExit(f"invalid checkpoint acquisition manifest: {exc}")
PY
  )"
  [[ "$OLD_ACQUISITION_SHA" == "$CHECKPOINT_SHA" ]] || die 'checkpoint differs from acquisition manifest'
fi

if [[ "$SETUP_MODE" == "acquire" ]]; then
  "$PYTHON" - "$ACQUISITION_MANIFEST" "$ZERO_GRASP_ROOT" "$CHECKPOINT" "$CHECKPOINT_SHA" "$CHECKPOINT_SIZE" <<'PY'
import json, pathlib, sys
path, root, checkpoint, checkpoint_sha, checkpoint_size = sys.argv[1:]
data = {
    "format": "zerograsp-checkpoint-acquisition-v1",
    "mode": "acquire",
    "official_url": "https://github.com/sh8/ZeroGrasp.git",
    "official_revision": "152f67c27269ff3f089783bd2f041d67641fa506",
    "official_checkpoint_drive_id": "1xUmFdgT_Ozu4zIPIsh_1SJMcegeQUWqQ",
    "checkpoint": pathlib.Path(checkpoint).resolve().as_posix(),
    "checkpoint_sha256": checkpoint_sha,
    "checkpoint_size_bytes": int(checkpoint_size),
    "source_checkout": pathlib.Path(root).resolve().as_posix(),
}
target = pathlib.Path(path).resolve()
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(target)
PY
  cp -- "$ACQUISITION_MANIFEST" "$ARCHIVE_ROOT/checkpoint-acquisition.json"
  printf 'zerograsp_setup_mode=acquire\nzerograsp_model_load=false\nzerograsp_worker_handshake=false\n'
  printf 'zerograsp_checkpoint_sha256=%s\nzerograsp_checkpoint_size_bytes=%s\n' "$CHECKPOINT_SHA" "$CHECKPOINT_SIZE"
  exit 0
fi

[[ -f "$ACQUISITION_MANIFEST" ]] || die 'execute mode requires a prior checkpoint acquisition manifest'

bash "$SCRIPT_DIR/setup_zerograsp_runtime.sh" \
  --root "$ZERO_GRASP_ROOT" --venv "$VENV" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --config "$CONFIG" --config-sha256 "$CONFIG_SHA" \
  --env-lock "$LOCK" --install --smoke

WORKER="$REPO_ROOT/vla_benchmarking/zerograsp_worker.py"
PREFLIGHT="$REPO_ROOT/vla_benchmarking/legion/preflight_zerograsp_process.py"
[[ -f "$WORKER" ]] || die "worker adapter is missing: $WORKER"
[[ -f "$PREFLIGHT" ]] || die "process preflight is missing: $PREFLIGHT"
# Start the real bounded worker process, require its protocol-ready reply, and
# close it. This is the same adapter used by the matrix launcher; unlike the
# legacy worker --handshake flag it exercises process startup and readiness.
preflight_reply="$($PYTHON "$PREFLIGHT" \
  --python "$PYTHON" --worker "$WORKER" --repo "$ZERO_GRASP_ROOT" \
  --checkpoint "$CHECKPOINT" --config "$CONFIG" --timeout-s 20)" \
  || die 'ZeroGrasp process ready/close preflight failed'
printf '%s\n' "$preflight_reply" > "$ARCHIVE_ROOT/zerograsp_process_preflight.json"
"$PYTHON" - "$preflight_reply" <<'PY'
import json
import sys

reply = json.loads(sys.argv[1])
if reply.get("protocol") != "zerograsp-jsonl-v1":
    raise SystemExit("unexpected ZeroGrasp process protocol")
if reply.get("protocol_ready") is not True or reply.get("closed") is not True:
    raise SystemExit("ZeroGrasp process did not complete ready/close")
PY

# Load the actual checkpoint through the official backend. This is a no-motion
# check; it does not construct LIBERO/MuJoCo or evaluate a task.
"$PYTHON" - "$REPO_ROOT" "$ZERO_GRASP_ROOT" "$CHECKPOINT" "$CONFIG" <<'PY'
import pathlib, sys
evaluation_root, repo, checkpoint, config = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(evaluation_root / "vla_benchmarking"))
from zerograsp_worker import _load_official_backend
_load_official_backend(repo=str(repo), checkpoint=str(checkpoint), config=str(config))
print("zerograsp_checkpoint_load=passed")
PY

cp -- "$LOCK" "$ARCHIVE_ROOT/zerograsp-runtime.lock.json"
printf 'zerograsp_motion=false\nzerograsp_evaluator=false\n'
printf 'zerograsp_protocol_ready=true\nzerograsp_process_closed=true\n'
printf 'zerograsp_checkpoint_downloaded=present\n'
printf 'zerograsp_checkpoint_sha256=%s\nzerograsp_checkpoint_size_bytes=%s\n' "$CHECKPOINT_SHA" "$(stat -c %s "$CHECKPOINT")"
printf 'zerograsp_config_sha256=%s\nzerograsp_archive=%s\n' "$CONFIG_SHA" "$ARCHIVE_ROOT"
