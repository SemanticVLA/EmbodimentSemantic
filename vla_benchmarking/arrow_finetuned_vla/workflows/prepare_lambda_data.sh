#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VLA_ROOT="$REPO_ROOT/vla_benchmarking"
PYTHON="${PYTHON:-python}"
PROFILE="${1:-${TRAINING_PROFILE:-treatment}}"
case "$PROFILE" in
  treatment) DATA_MODE="convert-pair"; VERIFY_MODE="verify"; PREFLIGHT_MODE="preflight"; DEFAULT_OUTPUT_ROOT="$VLA_ROOT/lora_datasets" ;;
  no_arrow_treatment) DATA_MODE="convert-pair"; VERIFY_MODE="verify"; PREFLIGHT_MODE="preflight"; DEFAULT_OUTPUT_ROOT="$VLA_ROOT/lora_datasets" ;;
  graph_treatment|arrow_graph_treatment) DATA_MODE="convert-graph-pair"; VERIFY_MODE="verify-graph"; PREFLIGHT_MODE="preflight-graph"; DEFAULT_OUTPUT_ROOT="$VLA_ROOT/lora_datasets" ;;
  *) echo "usage: $0 <treatment|no_arrow_treatment|graph_treatment|arrow_graph_treatment>" >&2; exit 2 ;;
esac
DATA_ROOT="${DATA_ROOT:-$DEFAULT_OUTPUT_ROOT}"
LIBERO_DATA_DIR="${LIBERO_DATA_DIR:-$REPO_ROOT/vlm_benchmarking/data/libero_spatial_v5}"
DATASET_COMMIT="${LIBERO_DATASET_COMMIT:-a0dded49581dcbf5a109f8350305411d345c5d99}"
DATASET_URL="${LIBERO_DATASET_URL:-https://huggingface.co/datasets/SemVLA/EmbodimentSemantic/resolve/$DATASET_COMMIT/libero_spatial_v5.zip?download=true}"
SHA256_EXPECTED="${LIBERO_DATASET_SHA256:-560fae66b5b41d2e383e6876b15052bbc105f4aae906cf1f630a2310b87a1fa9}"
CACHE_DIR="${LIBERO_DATASET_CACHE:-$SCRIPT_DIR/.cache}"
ARCHIVE="$CACHE_DIR/libero_spatial_v5-$DATASET_COMMIT.zip"

fail() { echo "prepare_lambda_data: $*" >&2; exit 1; }
command -v "$PYTHON" >/dev/null 2>&1 || fail "Python not found: $PYTHON"
mkdir -p "$CACHE_DIR" "$LIBERO_DATA_DIR" "$DATA_ROOT"

if [[ ! -s "$ARCHIVE" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --output "$ARCHIVE" "$DATASET_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget --no-verbose --tries=3 --output-document="$ARCHIVE" "$DATASET_URL"
  else
    fail "curl or wget is required to download the pinned dataset"
  fi
fi

"$PYTHON" - "$ARCHIVE" "$LIBERO_DATA_DIR" "$DATASET_COMMIT" "$SHA256_EXPECTED" "$SCRIPT_DIR" <<'PY'
from __future__ import annotations
import hashlib, json, pathlib, sys, zipfile
archive = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
commit = sys.argv[3]
prefix = sys.argv[4]
script_dir = pathlib.Path(sys.argv[5])
actual = hashlib.sha256(archive.read_bytes()).hexdigest()
if actual != prefix:
    raise SystemExit(f"dataset SHA256 mismatch: expected {prefix}, got {actual}")
sys.path.insert(0, str(script_dir))
from vla_benchmarking.shared.config import TASK_NAMES
members = [item for item in zipfile.ZipFile(archive).infolist() if item.filename.lower().endswith(".hdf5")]
if len(members) != 10:
    raise SystemExit(f"expected exactly 10 LIBERO HDF5 files, found {len(members)}")
by_name = {}
for item in members:
    name = pathlib.PurePosixPath(item.filename).name
    if name in by_name:
        raise SystemExit(f"duplicate HDF5 basename in archive: {name}")
    by_name[name] = item
expected = {f"{TASK_NAMES[i]}_demo.hdf5" for i in sorted(TASK_NAMES)}
if set(by_name) != expected:
    raise SystemExit(f"archive HDF5 set mismatch; missing={sorted(expected-set(by_name))}, extra={sorted(set(by_name)-expected)}")
destination.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as zf:
    for name, item in by_name.items():
        target = destination / name
        target.write_bytes(zf.read(item))
manifest = {
    "schema_version": 1, "dataset_repo": "SemVLA/EmbodimentSemantic",
    "revision": commit, "archive_sha256": actual,
    "hdf5_files": {name: hashlib.sha256((destination / name).read_bytes()).hexdigest() for name in sorted(by_name)},
}
(destination / "libero_spatial_v5_source_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2))
PY

"$PYTHON" "$SCRIPT_DIR/hdf5_to_lerobot_dataset.py" --mode "$DATA_MODE" \
  --data-dir "$LIBERO_DATA_DIR" --output-root "$DATA_ROOT"
"$PYTHON" "$SCRIPT_DIR/hdf5_to_lerobot_dataset.py" --mode "$VERIFY_MODE" \
  --data-dir "$LIBERO_DATA_DIR" --output-root "$DATA_ROOT"
"$PYTHON" "$SCRIPT_DIR/hdf5_to_lerobot_dataset.py" --mode "$PREFLIGHT_MODE" \
  --data-dir "$LIBERO_DATA_DIR" --output-root "$DATA_ROOT"
echo "prepare_lambda_data: sealed $PROFILE pair converted under $DATA_ROOT"
