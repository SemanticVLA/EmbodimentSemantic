#!/usr/bin/env bash
# Build the immutable one-demonstration Trace route and live camera calibration.
# This is an artifact-only compute job: it never loads a teacher or executes an action.
#SBATCH --job-name=arrow_trace_artifacts
#SBATCH --partition=gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:20:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%A.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%A.err

set -Eeuo pipefail
umask 027
die() { printf 'arrow Trace artifacts: %s\n' "$*" >&2; exit 2; }

required=(
  ARROW_SUITE_REPO_ROOT ARROW_SUITE_EXPECTED_COMMIT
  ARROW_TRACE_DATASET_DIR ARROW_TRACE_PARENT_MANIFEST
  ARROW_TRACE_RUN_ROOT ARROW_TRACE_ARCHIVE_ROOT
  ARROW_TRACE_GRAPH_TRIPLET ARROW_TRACE_GRAPH_REVISION
  ARROW_TRACE_DATASET_TASK_ID ARROW_TRACE_EPISODE_ID
  ARROW_TRACE_LIBERO_TASK_ID ARROW_TRACE_SEED ARROW_TRACE_INIT_STATE_INDEX
)
for name in "${required[@]}"; do [[ -n "${!name:-}" ]] || die "$name is required"; done
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || die 'submit with sbatch'
[[ "$ARROW_SUITE_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'expected commit must be a full SHA'
for name in ARROW_SUITE_REPO_ROOT ARROW_TRACE_DATASET_DIR ARROW_TRACE_PARENT_MANIFEST ARROW_TRACE_RUN_ROOT ARROW_TRACE_ARCHIVE_ROOT; do
  value="${!name}"
  [[ "$value" = /* && "$value" != / && "$value" != *'/../'* && "$value" != */.. ]] || die "$name is not a safe absolute path"
done
for name in ARROW_TRACE_DATASET_TASK_ID ARROW_TRACE_EPISODE_ID ARROW_TRACE_LIBERO_TASK_ID ARROW_TRACE_SEED ARROW_TRACE_INIT_STATE_INDEX; do
  [[ "${!name}" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer"
done
[[ "$ARROW_TRACE_GRAPH_REVISION" =~ ^[A-Za-z0-9_.:/-]+$ ]] || die 'graph revision is invalid'

REPO_ROOT="$(realpath -e -- "$ARROW_SUITE_REPO_ROOT")"
[[ -e "$REPO_ROOT/.git" ]] || die 'repository is not a Git checkout'
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$ARROW_SUITE_EXPECTED_COMMIT" ]] || die 'repository SHA mismatch'
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'repository checkout is dirty'
[[ -d "$ARROW_TRACE_DATASET_DIR" && -f "$ARROW_TRACE_PARENT_MANIFEST" ]] || die 'source dataset or manifest is missing'
[[ ! -e "$ARROW_TRACE_RUN_ROOT" && ! -e "$ARROW_TRACE_ARCHIVE_ROOT" ]] || die 'artifact output already exists'
mkdir -p "$ARROW_TRACE_RUN_ROOT"

archive_on_exit() {
  rc=$?
  trap - EXIT
  set +e
  mkdir -p "$ARROW_TRACE_ARCHIVE_ROOT"
  cp -a "$ARROW_TRACE_RUN_ROOT/." "$ARROW_TRACE_ARCHIVE_ROOT/"
  archive_rc=$?
  status=PRESERVED_FAILURE
  [[ "$rc" -eq 0 && -s "$ARROW_TRACE_ARCHIVE_ROOT/artifact_manifest.json" ]] && status=VERIFIED
  printf 'schema=arrow_trace_artifact_archive.v1\nstatus=%s\njob_id=%s\nrepo_commit=%s\n' \
    "$status" "$SLURM_JOB_ID" "$ARROW_SUITE_EXPECTED_COMMIT" > "$ARROW_TRACE_ARCHIVE_ROOT/archive_status.env"
  [[ "$rc" -eq 0 && "$archive_rc" -ne 0 ]] && rc=$archive_rc
  exit "$rc"
}
trap archive_on_exit EXIT

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
PYTHON="$HOME/.conda/envs/embodiment-smolvla-py312/bin/python"
[[ -x "$PYTHON" ]] || die 'validated Python is missing'
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$REPO_ROOT"
nvidia-smi -L

route="$ARROW_TRACE_RUN_ROOT/trace_route.json"
calibration="$ARROW_TRACE_RUN_ROOT/trace_calibration.json"
"$PYTHON" -m vla_benchmarking.libero.arrow_policy_suite.derive_trace_artifact \
  --dataset-dir "$ARROW_TRACE_DATASET_DIR" \
  --output "$route" \
  --task-id "$ARROW_TRACE_DATASET_TASK_ID" \
  --episode-id "$ARROW_TRACE_EPISODE_ID" \
  --graph-triplet "$ARROW_TRACE_GRAPH_TRIPLET" \
  --coordinate-frame libero_mujoco_world \
  --parent-manifest "$ARROW_TRACE_PARENT_MANIFEST" \
  --source-role hand_to_source --destination-role hand_to_destination \
  --transformation-revision trace-state-route-from-lerobot-parquet-v1 \
  > "$ARROW_TRACE_RUN_ROOT/route_receipt.json"

"$PYTHON" -m vla_benchmarking.libero.arrow_policy_suite.capture_trace_calibration \
  --output "$calibration" \
  --expected-commit "$ARROW_SUITE_EXPECTED_COMMIT" \
  --revision "$ARROW_TRACE_GRAPH_REVISION" \
  --task-id "$ARROW_TRACE_LIBERO_TASK_ID" \
  --seed "$ARROW_TRACE_SEED" \
  --init-state-index "$ARROW_TRACE_INIT_STATE_INDEX" \
  --resolution 256 --suite-mode vanilla \
  > "$ARROW_TRACE_RUN_ROOT/calibration_receipt.json"

"$PYTHON" - "$route" "$calibration" "$ARROW_TRACE_RUN_ROOT/artifact_manifest.json" <<'PY'
import hashlib, json, pathlib, sys
route, calibration, output = map(pathlib.Path, sys.argv[1:])
payload = {
    "schema": "arrow_policy_suite.trace_artifact_bundle.v1",
    "create_only": True,
    "repo_commit": __import__("os").environ["ARROW_SUITE_EXPECTED_COMMIT"],
    "source_dataset": __import__("os").environ["ARROW_TRACE_DATASET_DIR"],
    "source_manifest_sha256": hashlib.sha256(pathlib.Path(__import__("os").environ["ARROW_TRACE_PARENT_MANIFEST"]).read_bytes()).hexdigest(),
    "route": str(route),
    "route_sha256": hashlib.sha256(route.read_bytes()).hexdigest(),
    "calibration": str(calibration),
    "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
    "experiment_evidence": False,
    "diagnostic_only": True,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
