#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VLA_ROOT="$REPO_ROOT/vla_benchmarking"
REVISION="${BASE_POLICY_REVISION:-6721902bc4d61e50a3bfdb11dfb4cb626f05d102}"
REPO_ID="${BASE_POLICY_REPO:-HuggingFaceVLA/smolvla_libero}"
DEST="${BASE_POLICY_SNAPSHOT:-$VLA_ROOT/base_models/smolvla_libero-$REVISION}"
PYTHON="${PYTHON:-python}"

command -v "$PYTHON" >/dev/null 2>&1 || { echo "Python not found: $PYTHON" >&2; exit 1; }
"$PYTHON" - "$REPO_ID" "$REVISION" "$DEST" <<'PY'
from __future__ import annotations
import hashlib, json, pathlib, shutil, sys, tempfile
from huggingface_hub import HfApi, snapshot_download

repo_id, revision, destination = sys.argv[1:]
destination = pathlib.Path(destination)
destination.parent.mkdir(parents=True, exist_ok=True)
resolved = HfApi().model_info(repo_id, revision=revision).sha
if resolved != revision:
    raise SystemExit(f"Hub revision resolved to {resolved}, expected {revision}")
if destination.exists():
    manifest_path = destination / "base_snapshot_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("repo_id") == repo_id and existing.get("revision") == revision:
            actual_names = {
                path.relative_to(destination).as_posix()
                for path in destination.rglob("*")
                if path.is_file() and path.name != "base_snapshot_manifest.json" and ".cache" not in path.parts
            }
            valid = all(
                (destination / name).is_file()
                and hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
                for name, digest in existing.get("files", {}).items()
            ) and actual_names == set(existing.get("files", {}))
            if valid:
                print(json.dumps(existing, indent=2))
                raise SystemExit(0)
    raise SystemExit(f"refusing to overwrite stale base snapshot directory: {destination}")
temporary = pathlib.Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
try:
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=temporary, force_download=True)
except Exception:
    shutil.rmtree(temporary, ignore_errors=True)
    raise
destination = destination.resolve()
working = temporary.resolve()
required = ("config.json",)
missing = [name for name in required if not (working / name).is_file()]
if missing:
    shutil.rmtree(temporary, ignore_errors=True)
    raise SystemExit(f"snapshot is missing required files: {missing}")
files = {}
for path in sorted(working.rglob("*")):
    if not path.is_file() or path.name == "base_snapshot_manifest.json" or ".cache" in path.parts:
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    files[path.relative_to(working).as_posix()] = digest
manifest = {
    "schema_version": 1,
    "repo_id": repo_id,
    "revision": revision,
    "local_snapshot": str(destination.resolve()),
    "files": files,
}
working.joinpath("base_snapshot_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
try:
    working.replace(destination)
except Exception:
    shutil.rmtree(temporary, ignore_errors=True)
    raise
print(json.dumps(manifest, indent=2))
PY
