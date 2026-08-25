#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3.12}"
VENV="${LAMBDA_VENV:-$SCRIPT_DIR/.venv-lora}"
LIBERO_DIR="${LIBERO_DIR:-$SCRIPT_DIR/LIBERO}"
LIBERO_REPO="${LIBERO_REPO:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"

fail() { echo "bootstrap_lambda: $*" >&2; exit 1; }
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  if [[ "${ALLOW_APT_BOOTSTRAP:-0}" != 1 ]]; then
    fail "Python 3.12 not found. On Ubuntu Lambda, rerun with ALLOW_APT_BOOTSTRAP=1 to install python3.12/python3.12-venv."
  fi
  command -v sudo >/dev/null 2>&1 || fail "sudo is required for apt Python bootstrap"
  sudo apt-get update
  sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
cd "$SCRIPT_DIR"
python -m pip install -r requirements-lora.txt

if [[ ! -d "$LIBERO_DIR/libero/libero/assets" ]]; then
  command -v git >/dev/null 2>&1 || fail "git is required to provision LIBERO"
  git clone --depth=1 "$LIBERO_REPO" "$LIBERO_DIR"
  python -m pip install --no-deps -e "$LIBERO_DIR"
fi
[[ -d "$LIBERO_DIR/libero/libero/assets" ]] || fail "LIBERO assets are missing: $LIBERO_DIR/libero/libero/assets"
echo "bootstrap_lambda: ready; activate with source $VENV/bin/activate"
