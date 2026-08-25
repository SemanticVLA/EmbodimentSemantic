#!/usr/bin/env bash
set -euo pipefail

# Lambda is headless; MuJoCo/robosuite must use EGL rather than an X display.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"
VARIANT="${1:-treatment}"
PYTHON="${PYTHON:-python}"
case "$VARIANT" in
  treatment)
    DATASET_VARIANT="treatment"
    REQUIRED_PAIR_VARIANT="treatment"
    DATASET_REPO_ID="local/libero_spatial_treatment"
    PAIR_MANIFEST_NAME="sealed_lora_pair_manifest.json"
    PAIR_SENTINEL_NAME="sealed_lora_pair_verified.json"
    PAIR_KIND="sealed_lora_control_treatment"
    CONVERTER_MODE="preflight"
    ;;
  no_arrow_treatment)
    # Train the adapter on the arrow-free control frames.  The sealed pair
    # still remains the provenance contract, so both pair variants must exist
    # and pass the normal (all-arrows) preflight.
    DATASET_VARIANT="control"
    REQUIRED_PAIR_VARIANT="treatment"
    DATASET_REPO_ID="local/libero_spatial_control"
    PAIR_MANIFEST_NAME="sealed_lora_pair_manifest.json"
    PAIR_SENTINEL_NAME="sealed_lora_pair_verified.json"
    PAIR_KIND="sealed_lora_control_treatment"
    CONVERTER_MODE="preflight"
    ;;
  *) fail() { echo "LAMBDA PREFLIGHT FAILED: unsupported profile $VARIANT" >&2; exit 2; }; fail ;;
esac
REVISION="${BASE_POLICY_REVISION:-6721902bc4d61e50a3bfdb11dfb4cb626f05d102}"
BASE="${BASE_POLICY:-$SCRIPT_DIR/base_models/smolvla_libero-$REVISION}"
DEFAULT_DATA_ROOT="${DEFAULT_DATA_ROOT:-$SCRIPT_DIR/lora_datasets}"
DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
LIBERO_DATA_DIR="${LIBERO_DATA_DIR:-$SCRIPT_DIR/../vlm_benchmarking/data/libero_spatial_v5}"
LIBERO_DIR="${LIBERO_DIR:-$SCRIPT_DIR/LIBERO}"
LIBERO_CONFIG="${LIBERO_CONFIG:-${HOME:-}/.libero/config.yaml}"
PAIR_MANIFEST="${PAIR_MANIFEST:-$DATA_ROOT/$PAIR_MANIFEST_NAME}"
PAIR_SENTINEL="${PAIR_SENTINEL:-$DATA_ROOT/$PAIR_SENTINEL_NAME}"

fail() { echo "LAMBDA PREFLIGHT FAILED: $*" >&2; exit 1; }
[[ -d "$LIBERO_DIR/.git" ]] || fail "LIBERO checkout is not a git repository: $LIBERO_DIR"
libero_head="$(git -C "$LIBERO_DIR" rev-parse HEAD 2>/dev/null || true)"
[[ "$libero_head" == "$LIBERO_COMMIT" ]] || fail "LIBERO commit drift: expected $LIBERO_COMMIT, got $libero_head"
libero_tracked_status="$(git -C "$LIBERO_DIR" status --porcelain --untracked-files=no 2>/dev/null || true)"
[[ -z "$libero_tracked_status" ]] || fail "LIBERO checkout has tracked/staged changes; refusing launch"
command -v "$PYTHON" >/dev/null 2>&1 || fail "Python not found: $PYTHON"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || fail "Python 3.12 required"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))' || fail "CUDA GPU unavailable"
"$PYTHON" -c 'import lerobot, peft, accelerate; print("lerobot", lerobot.__version__, "peft", peft.__version__, "accelerate", accelerate.__version__)' || fail "training dependencies missing; install requirements-lora.txt"
"$PYTHON" -c 'import h5py, cv2, PIL, numpy, robosuite; print("converter/runtime imports OK")' || fail "converter dependencies missing: h5py, cv2, Pillow, numpy, robosuite"
[[ -f "$LIBERO_CONFIG" ]] || fail "LIBERO config missing: $LIBERO_CONFIG (install LIBERO and configure its assets/init states)"
LIBERO_CONFIG_PATH="$LIBERO_CONFIG" "$PYTHON" -c 'import os, pathlib, sys, yaml; p=pathlib.Path(os.environ["LIBERO_CONFIG_PATH"]); cfg=yaml.safe_load(p.read_text(encoding="utf-8")); required=("assets", "bddl_files", "init_states"); missing=[k for k in required if not cfg.get(k) or not pathlib.Path(cfg[k]).exists()]; sys.exit(f"invalid LIBERO config paths: {missing}") if missing else print("LIBERO config/assets OK")' || fail "LIBERO config does not point to existing assets, BDDL files, and init states"
PYTHONPATH="$LIBERO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c 'from libero.libero import get_libero_path; print("LIBERO import OK", get_libero_path("assets"))' || fail "LIBERO package is not importable from $LIBERO_DIR"
[[ -d "$BASE" && -f "$BASE/config.json" ]] || fail "base snapshot missing: $BASE"
[[ -f "$BASE/base_snapshot_manifest.json" ]] || fail "base snapshot manifest missing; run prepare_base_snapshot.sh"
"$PYTHON" - "$BASE/base_snapshot_manifest.json" "$BASE" "$REVISION" <<'PY' || fail "base snapshot revision or file hashes are not pinned to the required commit"
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
[[ -d "$DATA_ROOT/control" && -d "$DATA_ROOT/$REQUIRED_PAIR_VARIANT" ]] || fail "control and $REQUIRED_PAIR_VARIANT datasets are required under $DATA_ROOT"
[[ -d "$LIBERO_DATA_DIR" ]] || fail "LIBERO HDF5 source directory missing: $LIBERO_DATA_DIR"
[[ -d "$LIBERO_DIR/libero/libero/assets" ]] || fail "LIBERO assets missing: $LIBERO_DIR/libero/libero/assets"
[[ -d "$LIBERO_DIR/libero/libero/bddl_files" ]] || fail "LIBERO BDDL assets missing: $LIBERO_DIR/libero/libero/bddl_files"
[[ -d "$LIBERO_DIR/libero/libero/init_files" ]] || fail "LIBERO init-state assets missing: $LIBERO_DIR/libero/libero/init_files"
[[ -f "$LIBERO_CONFIG" ]] || fail "LIBERO config missing: $LIBERO_CONFIG"
"$PYTHON" -c 'from libero.libero import benchmark, get_libero_path; print("LIBERO import OK", get_libero_path("assets"))' || fail "LIBERO package/path import failed"
[[ -f "$PAIR_MANIFEST" && -f "$PAIR_SENTINEL" ]] ||
  fail "sealed pair manifest and verified sentinel are required under $DATA_ROOT; run converter --mode verify"
DATASET_ROOT="$DATA_ROOT/$DATASET_VARIANT"
[[ -f "$DATASET_ROOT/meta/info.json" ]] || fail "dataset metadata missing: $DATASET_ROOT/meta/info.json"
"$PYTHON" - "$PAIR_MANIFEST" "$PAIR_SENTINEL" "$PAIR_KIND" <<'PY'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
sentinel = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_kind = sys.argv[3]
if manifest.get("pair_kind") != expected_kind:
    raise SystemExit("unexpected sealed pair manifest kind")
if sentinel.get("pair_kind") != expected_kind:
    raise SystemExit("unexpected sealed pair sentinel kind")
if sentinel.get("full_experiment_ready") is not True or sentinel.get("launch_eligibility") != "full_experiment_ready":
    raise SystemExit("sealed pair is not marked full-experiment launchable")
PY
"$PYTHON" "$SCRIPT_DIR/hdf5_to_lerobot_dataset.py" \
  --mode "$CONVERTER_MODE" --data-dir "$LIBERO_DATA_DIR" --output-root "$DATA_ROOT" \
  || fail "sealed converter preflight failed; pair is not full-experiment launchable"
echo "LAMBDA PREFLIGHT OK: GPU, Python, dependencies, base snapshot, LIBERO assets, sealed pair, and $VARIANT dataset"
