#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a user-local Python 3.12 runtime on a clean Lambda Ubuntu image.
# This script does not start training or evaluation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$VLA_ROOT/.." && pwd)"
ENV_DIR="${LAMBDA_VENV:-$SCRIPT_DIR/.venv-lora}"
PYTHON310="${PYTHON310:-python3}"
LIBERO_DIR="${LIBERO_DIR:-$VLA_ROOT/LIBERO}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$VLA_ROOT/arrow_finetuned_vla/workflows/requirements-lora.txt}"
LIBERO_CONFIG="${LIBERO_CONFIG:-$HOME/.libero/config.yaml}"
LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"

die() { echo "bootstrap_lambda_runtime: $*" >&2; exit 1; }
command -v "$PYTHON310" >/dev/null 2>&1 || die "system Python is missing: $PYTHON310"

if ! command -v uv >/dev/null 2>&1; then
  "$PYTHON310" -m pip install --user uv || die "could not install uv for the current user"
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv is not on PATH; add ~/.local/bin and rerun"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  uv venv --python 3.12 "$ENV_DIR"
fi
PYTHON="$ENV_DIR/bin/python"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12), sys.version'
uv pip install --python "$PYTHON" --upgrade pip
[[ -f "$REQUIREMENTS_FILE" ]] || die "requirements file is missing: $REQUIREMENTS_FILE"
uv pip install --python "$PYTHON" -r "$REQUIREMENTS_FILE"

if [[ ! -d "$LIBERO_DIR/.git" ]]; then
  command -v git >/dev/null 2>&1 || die "git is required to clone LIBERO"
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
  git -C "$LIBERO_DIR" checkout --detach "$LIBERO_COMMIT" || die "LIBERO reviewed commit is unavailable: $LIBERO_COMMIT"
else
  actual_libero_commit="$(git -C "$LIBERO_DIR" rev-parse HEAD 2>/dev/null || true)"
  [[ "$actual_libero_commit" == "$LIBERO_COMMIT" ]] || die "existing LIBERO checkout is not the reviewed commit $LIBERO_COMMIT (found $actual_libero_commit); refusing drift"
fi
actual_libero_commit="$(git -C "$LIBERO_DIR" rev-parse HEAD 2>/dev/null || true)"
[[ "$actual_libero_commit" == "$LIBERO_COMMIT" ]] || die "LIBERO checkout verification failed: expected $LIBERO_COMMIT, got $actual_libero_commit"
libero_tracked_status="$(git -C "$LIBERO_DIR" status --porcelain --untracked-files=no 2>/dev/null || true)"
[[ -z "$libero_tracked_status" ]] || die "LIBERO checkout has tracked/staged changes; refusing to install from a dirty checkout"
uv pip install --python "$PYTHON" --no-deps -e "$LIBERO_DIR"

mkdir -p "$(dirname "$LIBERO_CONFIG")"
cat > "$LIBERO_CONFIG" <<EOF
assets: $LIBERO_DIR/libero/libero/assets
bddl_files: $LIBERO_DIR/libero/libero/bddl_files
benchmark_root: $LIBERO_DIR/libero/libero
datasets: $LIBERO_DIR/libero/datasets
init_states: $LIBERO_DIR/libero/libero/init_files
EOF

echo "Lambda runtime ready: $PYTHON"
echo "LIBERO config ready: $LIBERO_CONFIG"
echo "Next (after setup completes): bash $VLA_ROOT/arrow_finetuned_vla/workflows/run_smolvla_pipeline.sh dry --profile treatment"
