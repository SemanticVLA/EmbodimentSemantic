#!/usr/bin/env bash
#SBATCH --job-name=zerograsp_live_frame_probe
#SBATCH --partition=gpu_a40
#SBATCH --exclude=compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.err

set -Eeuo pipefail
umask 027
export PYTHONDONTWRITEBYTECODE=1
die() { printf 'ZeroGrasp live frame probe: %s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit with sbatch; direct execution is refused'
[[ -n "${REPO_ROOT:-}" && -d "$REPO_ROOT" ]] || die 'REPO_ROOT is required'
[[ "${ARROW_MATRIX_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die 'exact expected commit is required'

REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$ARROW_MATRIX_EXPECTED_COMMIT" ]] || die 'evaluation commit differs'
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'evaluation checkout is dirty'
PYTHON="${PYTHON:-$HOME/.conda/envs/embodiment-smolvla-py312/bin/python}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/EmbodimentSemantic_runtime/config}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/probes/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/zerograsp_frame_probe/${SLURM_JOB_ID}}"
[[ ! -e "$RUN_ROOT" && ! -e "$ARCHIVE_ROOT" ]] || die 'probe output already exists'
mkdir -p "$RUN_ROOT" "$ARCHIVE_ROOT"
archive_on_exit() {
  rc=$?
  trap - EXIT
  cp -a -- "$RUN_ROOT/." "$ARCHIVE_ROOT/" || rc=2
  printf 'job_exit_code=%s\ncommit=%s\n' "$rc" "$ARROW_MATRIX_EXPECTED_COMMIT" > "$ARCHIVE_ROOT/status.env"
  exit "$rc"
}
trap archive_on_exit EXIT

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
export PATH="$(dirname -- "$PYTHON"):$PATH"
export PYTHONPATH="$REPO_ROOT/vla_benchmarking"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH
export LIBERO_CONFIG="$LIBERO_CONFIG_PATH/config.yaml"
export TMPDIR="$RUN_ROOT/tmp" XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

PROBE="$REPO_ROOT/vla_benchmarking/sanity_checks/run_live_panda_frame_probe.py"
CONTROLLER_CONFIG="$REPO_ROOT/vla_benchmarking/controller_configs/v10_zg_grasp_only.json"
CALIBRATION_INPUT="$REPO_ROOT/vla_benchmarking/calibrations/panda_graspnet_eef_v1.json"
for suite in vanilla sealed_randomized; do
  "$PYTHON" "$PROBE" --task-id 0 --seed 1000 --resolution 256 \
    --suite-mode "$suite" --controller-config "$CONTROLLER_CONFIG" \
    --calibration-input "$CALIBRATION_INPUT" --output "$RUN_ROOT/${suite}.json"
done

"$PYTHON" - "$RUN_ROOT" "$ARROW_MATRIX_EXPECTED_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
records = [
    json.loads((root / f"{suite}.json").read_text())
    for suite in ("vanilla", "sealed_randomized")
]
for record in records:
    if record.get("passed") is not True:
        raise SystemExit("frame probe did not pass")
    execution = record.get("execution", {})
    if (
        execution.get("commanded_robot_motion") is not False
        or execution.get("evaluator_queried") is not False
    ):
        raise SystemExit("probe motion/evaluator contract failed")
manifest = {
    "schema_version": "zerograsp_live_frame_probe_manifest.v1",
    "evaluation_commit": sys.argv[2],
    "records": records,
}
encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
(root / "probe_manifest.json").write_bytes(encoded)
digest = hashlib.sha256(encoded).hexdigest()
(root / "probe_manifest.sha256").write_text(digest + "\n")
print(f"zerograsp_live_frame_probe_sha256={digest}")
print("zerograsp_commanded_motion=false")
print("zerograsp_evaluator=false")
PY
