#!/usr/bin/env bash
# Install/verify the official ZeroGrasp runtime in an isolated environment.
# Never downloads a checkpoint; the operator must provide the official Drive
# artifact and its exact SHA-256.
set -Eeuo pipefail
umask 027

readonly ZERO_GRASP_OFFICIAL_URL="https://github.com/sh8/ZeroGrasp.git"
readonly ZERO_GRASP_PIN="152f67c27269ff3f089783bd2f041d67641fa506"
readonly ZERO_GRASP_CHECKPOINT_URL="https://drive.google.com/file/d/1xUmFdgT_Ozu4zIPIsh_1SJMcegeQUWqQ/view?usp=sharing"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly BOOTSTRAP="${SCRIPT_DIR}/bootstrap_zerograsp.sh"
readonly OCNN_URL="git+https://github.com/octree-nn/ocnn-pytorch.git"
readonly OCNN_PIN="7521c22e2921a0bd8e9285044c842ff6fa2042e0"
readonly DWCONV_URL="git+https://github.com/octree-nn/dwconv.git"
readonly DWCONV_PIN="ae53057eaf36dab01aa2727fcc93a749fd995af5"
readonly GRASPNETAPI_PIN="eb57dd2092d8dbe05312a29c3d0c22f3226efbfc"

die() { printf 'ZeroGrasp runtime setup: %s\n' "$*" >&2; exit 2; }
usage() {
  cat <<'EOF'
Usage: setup_zerograsp_runtime.sh --root ABS --venv ABS \
  --checkpoint ABS --checkpoint-sha256 HEX --config ABS --config-sha256 HEX [options]

  --root PATH             pinned ZeroGrasp checkout
  --venv PATH             isolated Python 3.11 virtualenv, outside --root
  --checkpoint PATH       operator-supplied official Drive checkpoint
  --checkpoint-sha256 HEX expected checkpoint SHA-256 (required)
  --config PATH           model config used by the adapter
  --config-sha256 HEX     expected config SHA-256 (required)
  --env-lock PATH         lock path outside --root (default: VENV.zerograsp-runtime.lock.json)
  --install               install official CUDA 12.1 runtime into --venv
  --smoke                 import runtime, require CUDA, and validate artifacts (no motion)
  --verify-only           verify only; perform no writes
  -h, --help              show this help

The checkpoint URL is provenance only and is never fetched by this script.
EOF
}

ROOT_INPUT="${ZERO_GRASP_ROOT:-}"
VENV_INPUT="${ZERO_GRASP_VENV:-}"
CHECKPOINT_INPUT="${ZERO_GRASP_CHECKPOINT:-}"
CHECKPOINT_SHA_INPUT="${ZERO_GRASP_CHECKPOINT_SHA256:-}"
CONFIG_INPUT="${ZERO_GRASP_CONFIG:-}"
CONFIG_SHA_INPUT="${ZERO_GRASP_CONFIG_SHA256:-}"
LOCK_INPUT="${ZERO_GRASP_ENV_LOCK:-}"
INSTALL=0; SMOKE=0; VERIFY_ONLY=0
while (($#)); do
  case "$1" in
    --root) (($# >= 2)) || die '--root requires a value'; ROOT_INPUT="$2"; shift 2 ;;
    --venv) (($# >= 2)) || die '--venv requires a value'; VENV_INPUT="$2"; shift 2 ;;
    --checkpoint) (($# >= 2)) || die '--checkpoint requires a value'; CHECKPOINT_INPUT="$2"; shift 2 ;;
    --checkpoint-sha256) (($# >= 2)) || die '--checkpoint-sha256 requires a value'; CHECKPOINT_SHA_INPUT="$2"; shift 2 ;;
    --config) (($# >= 2)) || die '--config requires a value'; CONFIG_INPUT="$2"; shift 2 ;;
    --config-sha256) (($# >= 2)) || die '--config-sha256 requires a value'; CONFIG_SHA_INPUT="$2"; shift 2 ;;
    --env-lock) (($# >= 2)) || die '--env-lock requires a value'; LOCK_INPUT="$2"; shift 2 ;;
    --install) INSTALL=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$ROOT_INPUT" = /* && "$ROOT_INPUT" != / ]] || die 'root must be a non-root absolute path'
[[ "$VENV_INPUT" = /* && "$VENV_INPUT" != / ]] || die 'venv must be a non-root absolute path'
[[ "$INSTALL" == 0 || "$VERIFY_ONLY" == 0 ]] || die '--install and --verify-only are mutually exclusive'
command -v git >/dev/null 2>&1 || die 'git is required'
command -v realpath >/dev/null 2>&1 || die 'realpath is required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'
[[ -f "$BOOTSTRAP" ]] || die "bootstrap helper is missing: $BOOTSTRAP"
ROOT="$(realpath -m -- "$ROOT_INPUT")" || die 'cannot canonicalize root'
VENV="$(realpath -m -- "$VENV_INPUT")" || die 'cannot canonicalize venv'
case "$VENV" in "$ROOT"|"$ROOT"/*) die 'venv must be outside the checkout' ;; esac
if [[ -z "$LOCK_INPUT" ]]; then LOCK="${VENV}.zerograsp-runtime.lock.json"; else
  [[ "$LOCK_INPUT" = /* ]] || die 'env lock must be absolute'; LOCK="$(realpath -m -- "$LOCK_INPUT")" || die 'cannot canonicalize lock'
fi
case "$LOCK" in "$ROOT"|"$ROOT"/*) die 'env lock must be outside the checkout' ;; esac

if ((VERIFY_ONLY)); then
  [[ -d "$ROOT" ]] || die "checkout is missing: $ROOT"
else
  bash "$BOOTSTRAP" --root "$ROOT" --venv "$VENV" --create-venv || die 'checkout bootstrap failed'
fi
[[ -d "$ROOT/.git" || -f "$ROOT/.git" ]] || die 'root is not a Git checkout'
[[ "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)" == "$ZERO_GRASP_PIN" ]] || die "checkout must be pinned to $ZERO_GRASP_PIN"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'ZeroGrasp checkout is dirty'

[[ -f "$CHECKPOINT_INPUT" ]] || die "checkpoint is not a regular file: $CHECKPOINT_INPUT"
[[ -f "$CONFIG_INPUT" ]] || die "config is not a regular file: $CONFIG_INPUT"
[[ "$CHECKPOINT_SHA_INPUT" =~ ^[0-9a-fA-F]{64}$ ]] || die 'checkpoint SHA-256 must be 64 hex characters'
[[ "$CONFIG_SHA_INPUT" =~ ^[0-9a-fA-F]{64}$ ]] || die 'config SHA-256 must be 64 hex characters'
CHECKPOINT="$(realpath -e -- "$CHECKPOINT_INPUT")" || die 'cannot canonicalize checkpoint'
CONFIG="$(realpath -e -- "$CONFIG_INPUT")" || die 'cannot canonicalize config'
CHECKPOINT_SHA="$(sha256sum "$CHECKPOINT" | awk '{print tolower($1)}')"
CONFIG_SHA="$(sha256sum "$CONFIG" | awk '{print tolower($1)}')"
[[ "$CHECKPOINT_SHA" == "${CHECKPOINT_SHA_INPUT,,}" ]] || die 'checkpoint SHA-256 mismatch'
[[ "$CONFIG_SHA" == "${CONFIG_SHA_INPUT,,}" ]] || die 'config SHA-256 mismatch'

if ((INSTALL)); then
  [[ -d "$(dirname -- "$VENV")" ]] || die "venv parent does not exist: $(dirname -- "$VENV")"
  [[ -x "$VENV/bin/python" ]] || {
    BASE_PYTHON="${ZERO_GRASP_BASE_PYTHON:-python3}"
    command -v "$BASE_PYTHON" >/dev/null 2>&1 || die "base Python unavailable: $BASE_PYTHON"
    "$BASE_PYTHON" -m venv "$VENV" || die 'could not create isolated venv'
  }
fi
PYTHON_INPUT="${ZERO_GRASP_PYTHON:-${VENV}/bin/python}"
[[ "$PYTHON_INPUT" = /* ]] || die 'ZeroGrasp Python must be absolute'
PYTHON="$(realpath -e -- "$PYTHON_INPUT")" || die 'cannot canonicalize Python'
[[ -x "$PYTHON" ]] || die 'ZeroGrasp Python is not executable'
[[ "$PYTHON" != "$(realpath -m -- "${PYTHON_LIBERO:-/___missing_libero_python}")" ]] || die 'ZeroGrasp Python matches LIBERO Python'
PYTHON_SHA256="$(sha256sum -- "$PYTHON" | awk '{print tolower($1)}')" || die 'cannot hash ZeroGrasp Python executable'
[[ "$PYTHON_SHA256" =~ ^[0-9a-f]{64}$ ]] || die 'ZeroGrasp Python executable SHA-256 is invalid'
PYTHON_VERSION="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" || die 'cannot query Python version'
[[ "$PYTHON_VERSION" == "3.11" ]] || die "official runtime requires Python 3.11 (found $PYTHON_VERSION)"

SUBMODULE="$ROOT/submodules/octree_feature_extractor"
[[ -d "$SUBMODULE" ]] || die 'octree_feature_extractor submodule is missing'
if ((INSTALL)); then
  git -C "$ROOT" submodule update --init --recursive || die 'could not initialize submodules'
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout became dirty while initializing submodules'
  [[ -z "$(git -C "$SUBMODULE" status --porcelain --untracked-files=all)" ]] || die 'octree_feature_extractor submodule is dirty'
fi
SUBMODULE_REVISION="$(git -C "$SUBMODULE" rev-parse HEAD 2>/dev/null)" || die 'cannot read octree submodule revision'
[[ "$SUBMODULE_REVISION" =~ ^[0-9a-f]{40}$ ]] || die 'octree submodule revision is not a full commit hash'
if [[ -f "$LOCK" ]]; then
  "$PYTHON" - "$LOCK" "$VENV" "$PYTHON" "$PYTHON_SHA256" "$CHECKPOINT_SHA" "$CONFIG_SHA" "$ZERO_GRASP_PIN" "$SUBMODULE_REVISION" "$PYTHON_VERSION" <<'PY'
import json, pathlib, subprocess, sys
path, venv, python, python_sha, checkpoint_sha, config_sha, revision, submodule_revision, python_version = sys.argv[1:]
try:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"invalid existing runtime lock: {exc}")
for key, expected in (("venv", pathlib.Path(venv).resolve().as_posix()), ("python", pathlib.Path(python).resolve().as_posix()), ("python_executable_sha256", python_sha), ("checkpoint_sha256", checkpoint_sha), ("config_sha256", config_sha), ("official_revision", revision), ("octree_feature_extractor_revision", submodule_revision), ("python_version", python_version)):
    if data.get(key) != expected:
        raise SystemExit(f"existing runtime lock {key} differs")
expected_freeze = sorted(data.get("pip_freeze", []))
actual_freeze = sorted(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines())
if not expected_freeze or expected_freeze != actual_freeze:
    raise SystemExit("existing runtime lock pip_freeze differs")
try:
    import torch
except Exception as exc:
    raise SystemExit(f"existing runtime lock Torch import failed: {exc}")
torch_info = data.get("torch", {})
if torch.__version__.split("+")[0] != torch_info.get("version") or torch.version.cuda != torch_info.get("cuda"):
    raise SystemExit("existing runtime lock Torch identity differs")
try:
    import numpy
except Exception as exc:
    raise SystemExit(f"existing runtime lock NumPy import failed: {exc}")
if numpy.__version__ != data.get("numpy_version"):
    raise SystemExit("existing runtime lock NumPy identity differs")
PY
  # A lock is immutable provenance: an existing environment is verification
  # only, never a reason to reinstall packages or rewrite the lock.
  INSTALL=0
  VERIFY_ONLY=1
fi
if ((INSTALL)); then
  REQ_TMP="$(mktemp)"
  CONSTRAINT_TMP="$(mktemp)"
  cleanup() { rm -f -- "$REQ_TMP" "$CONSTRAINT_TMP"; }
  trap cleanup EXIT
  # Upstream VCS entries are built in isolated environments where torch is not
  # visible. Omit both and install their reviewed immutable refs explicitly
  # after Torch is present.
  awk '!/^ocnn[[:space:]]*@/ && !/^dwconv[[:space:]]*@/' "$ROOT/requirements.txt" > "$REQ_TMP"
  printf 'numpy==1.26.4\n' > "$CONSTRAINT_TMP"
  export PIP_CONSTRAINT="$CONSTRAINT_TMP"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0+PTX}"
  "$PYTHON" -m pip install --upgrade pip setuptools wheel ninja || die 'build tools install failed'
  "$PYTHON" -m pip install --force-reinstall --no-deps numpy==1.26.4 || die 'NumPy install failed'
  "$PYTHON" -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121 || die 'Torch install failed'
  "$PYTHON" -m pip install --no-build-isolation --no-deps "dwconv @ ${DWCONV_URL}@${DWCONV_PIN}" || die 'dwconv install failed'
  "$PYTHON" -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html || die 'torch-scatter install failed'
  "$PYTHON" -m pip install --upgrade xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121 || die 'xformers install failed'
  "$PYTHON" -m pip install -r "$REQ_TMP" || die 'ZeroGrasp requirements install failed'
  "$PYTHON" -m pip install --force-reinstall --no-deps "ocnn @ ${OCNN_URL}@${OCNN_PIN}" || die 'ocnn install failed'
  (cd -- "$SUBMODULE" && "$PYTHON" setup.py install) || die 'octree_feature_extractor install failed'
  "$PYTHON" -m pip install "graspnetAPI @ git+https://github.com/graspnet/graspnetAPI.git@${GRASPNETAPI_PIN}" --no-deps || die 'graspnetAPI install failed'
  "$PYTHON" -m pip install transforms3d autolab_core cvxopt grasp_nms || die 'grasp utility install failed'
  "$PYTHON" -m pip install torch_cluster -f https://data.pyg.org/whl/torch-2.2.0+cu121.html || die 'torch_cluster install failed'
  NUMPY_VERSION="$($PYTHON -c 'import numpy; print(numpy.__version__)')" || die 'cannot import NumPy after installation'
  [[ "$NUMPY_VERSION" == "1.26.4" ]] || die "NumPy version drifted during installation: $NUMPY_VERSION"
  unset PIP_CONSTRAINT
fi

NUMPY_VERSION="$($PYTHON -c 'import numpy; print(numpy.__version__)')" || die 'cannot query NumPy version'
[[ "$NUMPY_VERSION" == "1.26.4" ]] || die "official runtime requires NumPy 1.26.4 (found $NUMPY_VERSION)"

if ((SMOKE)); then
  "$PYTHON" - "$ROOT" "$CHECKPOINT" "$CONFIG" <<'PY'
import importlib, pathlib, sys
root, checkpoint, config = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(root))
import torch
import numpy
assert torch.__version__.split("+")[0] == "2.2.0", torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert numpy.__version__ == "1.26.4", numpy.__version__
assert torch.cuda.is_available(), "CUDA unavailable; run on a GPU compute node"
for module in ("torchvision", "torch_scatter", "torch_cluster", "xformers", "ocnn", "graspnetAPI", "zerograsp"):
    importlib.import_module(module)
assert checkpoint.is_file() and config.is_file()
print("zerograsp_runtime_smoke=passed")
PY
fi

if ((VERIFY_ONLY)); then
  [[ -f "$LOCK" ]] || die "runtime lock is missing: $LOCK"
else
  "$PYTHON" - "$LOCK" "$ROOT" "$VENV" "$PYTHON" "$PYTHON_SHA256" "$SUBMODULE_REVISION" "$CHECKPOINT" "$CHECKPOINT_SHA" "$CONFIG" "$CONFIG_SHA" "$PYTHON_VERSION" <<'PY'
import json, pathlib, subprocess, sys
lock, root, venv, python, python_sha, submodule_revision, checkpoint, checkpoint_sha, config, config_sha, python_version = sys.argv[1:]
packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
gpu = {"cuda_available": False, "name": None, "torch_cuda": None}
try:
    import torch
    gpu["cuda_available"] = bool(torch.cuda.is_available())
    gpu["torch_cuda"] = torch.version.cuda
    if gpu["cuda_available"]:
        gpu["name"] = torch.cuda.get_device_name(0)
except Exception as exc:
    gpu["error"] = f"{type(exc).__name__}: {exc}"
data = {
    "format": "zerograsp-runtime-lock-v1",
    "official_url": "https://github.com/sh8/ZeroGrasp.git",
    "official_revision": "152f67c27269ff3f089783bd2f041d67641fa506",
    "octree_feature_extractor_revision": submodule_revision,
    "venv": pathlib.Path(venv).resolve().as_posix(),
    "python": pathlib.Path(python).resolve().as_posix(),
    "python_executable_sha256": python_sha,
    "python_version": python_version,
    "numpy_version": "1.26.4",
    "gpu": gpu,
    "torch": {"version": "2.2.0", "torchvision": "0.17.0", "cuda": "12.1"},
    "ocnn_commit": "7521c22e2921a0bd8e9285044c842ff6fa2042e0",
    "dwconv_commit": "ae53057eaf36dab01aa2727fcc93a749fd995af5",
    "graspnetAPI_commit": "eb57dd2092d8dbe05312a29c3d0c22f3226efbfc",
    "checkpoint_url": "https://drive.google.com/file/d/1xUmFdgT_Ozu4zIPIsh_1SJMcegeQUWqQ/view?usp=sharing",
    "checkpoint": pathlib.Path(checkpoint).resolve().as_posix(),
    "checkpoint_sha256": checkpoint_sha,
    "checkpoint_size_bytes": pathlib.Path(checkpoint).stat().st_size,
    "config": pathlib.Path(config).resolve().as_posix(),
    "config_sha256": config_sha,
    "pip_freeze": packages,
}
path = pathlib.Path(lock).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
PY
fi

printf 'zero_grasp_root=%s\n' "$ROOT"
printf 'zero_grasp_revision=%s\n' "$ZERO_GRASP_PIN"
printf 'zero_grasp_python=%s\n' "$PYTHON"
printf 'zero_grasp_checkpoint=%s\n' "$CHECKPOINT"
printf 'zero_grasp_checkpoint_sha256=%s\n' "$CHECKPOINT_SHA"
printf 'zero_grasp_config=%s\n' "$CONFIG"
printf 'zero_grasp_config_sha256=%s\n' "$CONFIG_SHA"
printf 'zero_grasp_numpy_version=%s\n' "$NUMPY_VERSION"
printf 'zero_grasp_env_lock=%s\n' "$LOCK"
[[ -f "$LOCK" ]] && printf 'zero_grasp_env_lock_sha256=%s\n' "$(sha256sum "$LOCK" | awk '{print tolower($1)}')"
printf 'zero_grasp_checkpoint_downloaded=false\n'
