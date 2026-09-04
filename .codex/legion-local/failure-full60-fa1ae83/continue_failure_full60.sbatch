#!/usr/bin/env bash
# Immutable continuation of failure_opening40_retreat80: 12-cell prefix -> 60.
# No SAM3 is loaded; this uses the same RGB-D/no-SAM Molmo path as job 1920258.
#SBATCH --job-name=molmo_failure_full60
#SBATCH --partition=gpu_a40
#SBATCH --exclude=compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-23:59:00
#SBATCH --output=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.out
#SBATCH --error=/home/hjaber/EmbodimentSemantic_runtime/operator/logs/%x_%j.err
set -Eeuo pipefail
umask 027
die(){ printf 'failure full60: %s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die 'submit with sbatch; direct execution is refused'
REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
PREFIX_ROOT="${PREFIX_ROOT:?PREFIX_ROOT is required}"
OPERATOR_SCRIPT="${OPERATOR_SCRIPT:?OPERATOR_SCRIPT is required}"
EXPECTED_COMMIT="${CANARY_EXPECTED_COMMIT:?CANARY_EXPECTED_COMMIT is required}"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'commit pin must be a lowercase 40-character SHA'
[[ "$REPO_ROOT" != "$HOME/EmbodimentSemantic" ]] || die 'shared checkout forbidden'
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")" || die 'cannot resolve checkout'
PREFIX_ROOT="$(realpath -e -- "$PREFIX_ROOT")" || die 'cannot resolve prefix'
OPERATOR_SCRIPT="$(realpath -e -- "$OPERATOR_SCRIPT")" || die 'cannot resolve operator'
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]] || die 'release checkout is dirty'
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die 'release SHA differs'
RUNTIME_ROOT="${V9D_MOLMO_RUNTIME_ROOT:-$HOME/EmbodimentSemantic_runtime/EmbodimentSemantic/v9d_molmo}"
PYTHON="${CONTINUATION_PYTHON:-$RUNTIME_ROOT/venv-py312/bin/python}"
SCRATCH_ROOT="${SCRATCH_FLASH:?SCRATCH_FLASH is required}/EmbodimentSemantic_runtime/v9d_molmo"
LABEL="${CONTINUATION_LABEL:?CONTINUATION_LABEL is required}"
RUN_ROOT="${CONTINUATION_RUN_ROOT:-$SCRATCH_ROOT/full60/${LABEL}_${SLURM_JOB_ID}}"
OUTPUT_ROOT="${CONTINUATION_OUTPUT_ROOT:-$RUN_ROOT/results}"
ARCHIVE_ROOT="${CONTINUATION_ARCHIVE_ROOT:-$HOME/EmbodimentSemantic_archive/v9d_molmo_failure_full60/${LABEL}_${SLURM_JOB_ID}}"
for p in "$RUN_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT"; do [[ "$p" = /* && "$p" != / ]] || die "unsafe path: $p"; done
[[ ! -e "$RUN_ROOT" && ! -e "$ARCHIVE_ROOT" && ! -e "$OUTPUT_ROOT" ]] || die 'output path already exists'
mkdir -p "$RUN_ROOT"
archive(){ local rc=$?; trap - EXIT; set +e; mkdir -p "$ARCHIVE_ROOT"; rsync -a -- "$RUN_ROOT/" "$ARCHIVE_ROOT/run/" 2>/dev/null || cp -a -- "$RUN_ROOT/." "$ARCHIVE_ROOT/run/"; [[ ! -d "$OUTPUT_ROOT" ]] || rsync -a -- "$OUTPUT_ROOT/" "$ARCHIVE_ROOT/output/" 2>/dev/null || cp -a -- "$OUTPUT_ROOT/." "$ARCHIVE_ROOT/output/"; mkdir -p "$ARCHIVE_ROOT/slurm_logs"; for s in out err; do cp -p "/home/hjaber/EmbodimentSemantic_runtime/operator/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${s}" "$ARCHIVE_ROOT/slurm_logs/" 2>/dev/null || true; done; printf 'schema=molmo_failure_full60_job.v1\njob_id=%s\njob_exit_code=%s\nexpected_commit=%s\nlabel=%s\nprefix_root=%s\noutput_root=%s\narchive_root=%s\nbackend=rgbd\nsam3_used=false\n' "$SLURM_JOB_ID" "$rc" "$EXPECTED_COMMIT" "$LABEL" "$PREFIX_ROOT" "$OUTPUT_ROOT" "$ARCHIVE_ROOT" > "$ARCHIVE_ROOT/archive_status.env"; exit "$rc"; }
trap archive EXIT
[[ -x "$PYTHON" ]] || die "Python missing: $PYTHON"
[[ -d "${LIBERO_CONFIG_PATH:-$HOME/EmbodimentSemantic_runtime/config}" ]] || die 'LIBERO_CONFIG_PATH missing'
 LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/EmbodimentSemantic_runtime/config}"
 export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH
 export LIBERO_CONFIG="$LIBERO_CONFIG_PATH/config.yaml"
export PYTHONPATH="$REPO_ROOT/vla_benchmarking" TMPDIR="$RUN_ROOT/tmp" XDG_CACHE_HOME="$RUNTIME_ROOT/cache/xdg" TORCH_HOME="$RUNTIME_ROOT/cache/torch" HF_HOME="$RUNTIME_ROOT/cache/huggingface" HF_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$RUNTIME_ROOT/cache/transformers"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME"
"$PYTHON" -c 'import torch, transformers, libero, mujoco, robosuite; assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported(); assert transformers.__version__=="4.57.1"; print(torch.cuda.get_device_name(0))' || die 'runtime preflight failed'
"$PYTHON" "$OPERATOR_SCRIPT" --repo-root "$REPO_ROOT" --prefix-root "$PREFIX_ROOT" --output-root "$OUTPUT_ROOT" --operator-label "$LABEL"
