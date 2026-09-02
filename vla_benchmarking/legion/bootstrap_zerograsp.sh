#!/usr/bin/env bash
# Provision/verify the isolated ZeroGrasp source checkout.
#
# This script never downloads a model checkpoint and never installs into the
# LIBERO environment. Dependency installation is an explicit opt-in action
# using --install-deps and an isolated virtual environment.
set -Eeuo pipefail
umask 027

readonly ZERO_GRASP_OFFICIAL_URL="https://github.com/sh8/ZeroGrasp.git"
readonly ZERO_GRASP_PIN="152f67c27269ff3f089783bd2f041d67641fa506"
readonly OCTREE_SUBMODULE_PATH="submodules/octree_feature_extractor"
readonly OCTREE_SUBMODULE_URL="https://github.com/TRI-ML/octree_feature_extractor.git"

die() {
  printf 'ZeroGrasp bootstrap: %s\n' "$*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: bootstrap_zerograsp.sh --root ABSOLUTE_PATH [options]

Options:
  --root PATH             clean ZeroGrasp checkout destination (or
                          ZERO_GRASP_ROOT)
  --python PATH           isolated ZeroGrasp Python executable
  --venv PATH             isolated virtualenv path (default: ROOT.venv)
  --create-venv           create the isolated virtualenv if it does not exist
  --install-deps          install pinned repository requirements into the
                          isolated Python (implies --create-venv when --python
                          is not supplied); never writes into ROOT
  --verify-only           verify the existing checkout and runtime, no writes
  -h, --help              show this help

The checkpoint is intentionally never downloaded by this script.
EOF
}

ROOT_INPUT="${ZERO_GRASP_ROOT:-}"
PYTHON_INPUT="${ZERO_GRASP_PYTHON:-}"
VENV_INPUT="${ZERO_GRASP_VENV:-}"
CREATE_VENV=0
INSTALL_DEPS=0
VERIFY_ONLY=0

while (($#)); do
  case "$1" in
    --root)
      (($# >= 2)) || die '--root requires a value'
      ROOT_INPUT="$2"
      shift 2
      ;;
    --python)
      (($# >= 2)) || die '--python requires a value'
      PYTHON_INPUT="$2"
      shift 2
      ;;
    --venv)
      (($# >= 2)) || die '--venv requires a value'
      VENV_INPUT="$2"
      shift 2
      ;;
    --create-venv)
      CREATE_VENV=1
      shift
      ;;
    --install-deps)
      INSTALL_DEPS=1
      CREATE_VENV=1
      shift
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$ROOT_INPUT" && "$ROOT_INPUT" = /* ]] || die 'root must be an absolute path'
[[ "$ROOT_INPUT" != / ]] || die 'refusing to use filesystem root as checkout'
command -v git >/dev/null 2>&1 || die 'git is required'
command -v realpath >/dev/null 2>&1 || die 'realpath is required'

ROOT_PARENT="$(dirname -- "$ROOT_INPUT")"
[[ -d "$ROOT_PARENT" ]] || die "parent directory does not exist: $ROOT_PARENT"

if [[ ! -e "$ROOT_INPUT" ]]; then
  ((VERIFY_ONLY == 0)) || die "checkout is missing in verify-only mode: $ROOT_INPUT"
  git clone "$ZERO_GRASP_OFFICIAL_URL" "$ROOT_INPUT" || die 'official checkout clone failed'
fi

ROOT="$(realpath -e -- "$ROOT_INPUT")" || die 'cannot canonicalize checkout root'
[[ -d "$ROOT" ]] || die 'checkout root is not a directory'
[[ -d "$ROOT/.git" || -f "$ROOT/.git" ]] || die 'root is not a Git checkout'

[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'existing checkout is dirty; refusing to overwrite user changes'

# A local mirror may not have the pinned object yet. Fetching the one pinned
# commit is safe; no branch or floating revision is accepted.
if ! git -C "$ROOT" cat-file -e "$ZERO_GRASP_PIN^{commit}" 2>/dev/null; then
  ((VERIFY_ONLY == 0)) || die "pinned commit is absent in verify-only mode"
  git -C "$ROOT" fetch --no-tags "$ZERO_GRASP_OFFICIAL_URL" "$ZERO_GRASP_PIN" || die 'could not fetch pinned official commit'
fi

if [[ "$(git -C "$ROOT" rev-parse HEAD)" != "$ZERO_GRASP_PIN" ]]; then
  ((VERIFY_ONLY == 0)) || die "checkout is not pinned to $ZERO_GRASP_PIN"
  # This changes only the clean checkout's selected revision; it never resets
  # files or deletes branches, and dirty trees were rejected above.
  git -C "$ROOT" checkout --detach "$ZERO_GRASP_PIN" || die 'could not select pinned revision'
fi

[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$ZERO_GRASP_PIN" ]] || die 'pinned revision verification failed'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout became dirty during pinning'

# The official repository records this public submodule with an SSH URL.
# Compute nodes cannot service interactive SSH authentication, so pin the
# repository-local submodule URL to the equivalent public HTTPS endpoint.
if ((VERIFY_ONLY == 0)); then
  git -C "$ROOT" config "submodule.${OCTREE_SUBMODULE_PATH}.url" "$OCTREE_SUBMODULE_URL" || die 'could not configure HTTPS submodule URL'
  git -C "$ROOT" submodule update --init --recursive || die 'could not initialize pinned octree submodule over HTTPS'
fi
SUBMODULE_STATUS="$(git -C "$ROOT" submodule status --recursive)" || die 'cannot inspect submodule status'
[[ -n "$SUBMODULE_STATUS" ]] || die 'expected octree submodule is absent'
if printf '%s\n' "$SUBMODULE_STATUS" | grep -Eq '^[+-U]'; then
  die 'submodule checkout is missing or differs from the pinned superproject state'
fi
[[ -z "$(git -C "$ROOT/$OCTREE_SUBMODULE_PATH" status --porcelain --untracked-files=all)" ]] || die 'octree submodule is dirty'

if ((CREATE_VENV)) && [[ -z "$PYTHON_INPUT" ]]; then
  if [[ -z "$VENV_INPUT" ]]; then
    VENV_INPUT="${ROOT}.venv"
  fi
  [[ "$VENV_INPUT" = /* ]] || die 'venv must be an absolute path'
  VENV_INPUT="$(realpath -m -- "$VENV_INPUT")" || die 'cannot canonicalize virtualenv path'
  case "$VENV_INPUT" in
    "$ROOT"|"$ROOT"/*) die 'virtualenv must be outside the ZeroGrasp checkout' ;;
  esac
  VENV_PARENT="$(dirname -- "$VENV_INPUT")"
  [[ -d "$VENV_PARENT" ]] || die "virtualenv parent does not exist: $VENV_PARENT"
  VENV_PYTHON="$VENV_INPUT/bin/python"
  if [[ ! -x "$VENV_PYTHON" ]]; then
    ((VERIFY_ONLY == 0)) || die "isolated virtualenv is missing in verify-only mode: $VENV_PYTHON"
    BASE_PYTHON="${ZERO_GRASP_BASE_PYTHON:-python3}"
    command -v "$BASE_PYTHON" >/dev/null 2>&1 || die "base Python is unavailable: $BASE_PYTHON"
    "$BASE_PYTHON" -m venv "$VENV_INPUT" || die 'could not create isolated ZeroGrasp virtualenv'
  fi
  PYTHON_INPUT="$VENV_PYTHON"
fi

if [[ -n "$PYTHON_INPUT" ]]; then
  [[ "$PYTHON_INPUT" = /* ]] || die 'python must be an absolute path'
  PYTHON="$(realpath -e -- "$PYTHON_INPUT")" || die 'cannot canonicalize ZeroGrasp Python'
  [[ -x "$PYTHON" ]] || die 'ZeroGrasp Python is not executable'
  [[ "$PYTHON" != "${PYTHON_LIBERO:-}" ]] || die 'ZeroGrasp Python matches LIBERO Python'
  if ((INSTALL_DEPS)); then
    "$PYTHON" -m pip install -r "$ROOT/requirements.txt" || die 'isolated ZeroGrasp dependency installation failed'
  fi
elif ((INSTALL_DEPS)); then
  die 'dependency installation requires an isolated Python'
fi

printf 'zero_grasp_root=%s\n' "$ROOT"
printf 'zero_grasp_revision=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
printf 'zero_grasp_official_url=%s\n' "$ZERO_GRASP_OFFICIAL_URL"
if [[ -n "${PYTHON:-}" ]]; then
  printf 'zero_grasp_python=%s\n' "$PYTHON"
fi
printf 'zero_grasp_checkpoint_downloaded=false\n'
if ((INSTALL_DEPS)); then
  printf 'zero_grasp_dependencies=installed_in_isolated_python\n'
else
  printf 'zero_grasp_dependencies=unchanged\n'
fi
