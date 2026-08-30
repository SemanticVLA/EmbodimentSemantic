#!/usr/bin/env bash
# Submit the sealed graph-text pilot from the Legion login node.  Submission
# is lightweight; every Python/dataset/GPU operation is inside an sbatch job.
set -Eeuo pipefail

ACTION="${1:-}"
shift || true
EXPECTED_REPO_COMMIT=''
STATE_FILE_ARG=''
while (($#)); do
  case "$1" in
    --expected-commit)
      [[ -n "${2:-}" ]] || { echo '--expected-commit requires a 40-character commit hash' >&2; exit 2; }
      EXPECTED_REPO_COMMIT="$2"
      shift 2
      ;;
    --state-file)
      [[ -n "${2:-}" ]] || { echo '--state-file requires the setup state path' >&2; exit 2; }
      STATE_FILE_ARG="$2"
      shift 2
      ;;
    *)
      echo "usage: $0 <setup|launch> --expected-commit <40hex> [--state-file PATH]" >&2
      exit 2
      ;;
  esac
done
[[ "$ACTION" == setup || "$ACTION" == launch ]] || { echo "usage: $0 <setup|launch> --expected-commit <40hex> [--state-file PATH]" >&2; exit 2; }
[[ "$EXPECTED_REPO_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo '--expected-commit must be exactly 40 lowercase hexadecimal characters' >&2; exit 2; }
if [[ "$ACTION" == setup ]]; then
  [[ -z "$STATE_FILE_ARG" ]] || { echo '--state-file is only valid for launch' >&2; exit 2; }
else
  [[ -n "$STATE_FILE_ARG" ]] || { echo 'launch requires --state-file from a completed setup action' >&2; exit 2; }
fi
[[ -z "${SLURM_JOB_ID:-}" ]] || { echo 'run this submitter on the Legion login node' >&2; exit 2; }
command -v sbatch >/dev/null 2>&1 || { echo 'sbatch is unavailable; this is not the Legion login node' >&2; exit 2; }

# The caller must provide the exact reviewed graph-pipeline commit.  Never use
# a moving branch, a repository default, or an ambient override for a run.
LIBERO_COMMIT='8f1084e3132a39270c3a13ebe37270a43ece2a01'
BASE_POLICY_REVISION='6721902bc4d61e50a3bfdb11dfb4cb626f05d102'
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
RUNTIME='/home/hjaber/EmbodimentSemantic_runtime'
SCRATCH_ROOT='/mnt/beegfs/hjaber/EmbodimentSemantic_runtime'
ARCHIVE_ROOT='/home/hjaber/EmbodimentSemantic_archive'
PYTHON='/home/hjaber/.conda/envs/embodiment-smolvla-py312/bin/python'

[[ -d "$REPO/.git" ]] || { echo "repository missing: $REPO" >&2; exit 1; }
[[ "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)" == "$EXPECTED_REPO_COMMIT" ]] || { echo "repository is not clean reviewed commit $EXPECTED_REPO_COMMIT" >&2; exit 1; }
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=all)" ]] || { echo 'repository is dirty; refusing submission' >&2; exit 1; }
[[ -d "$RUNTIME" && -d "$SCRATCH_ROOT" ]] || { echo 'required persistent or scratch runtime is missing' >&2; exit 1; }

state_value() { awk -F= -v key="$1" '$1 == key { value = substr($0, index($0, "=") + 1) } END { print value }' "$state_file"; }
if [[ "$ACTION" == setup ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  label="legion_graph_treatment_lora_full_s1000_v1_${stamp}"
  state_dir="$RUNTIME/graph_pilot/$label"
  job_dir="$RUNTIME/operator/jobs/$label"
  mkdir -p "$state_dir" "$job_dir" "$RUNTIME/operator/logs"
  state_file="$state_dir/state.env"
  setup_job_id=''; train_job_id=''; eval_job_id=''
  setup_template_sha256=''; train_template_sha256=''; eval_template_sha256=''; eval_rendered_sha256=''
else
  state_file="$STATE_FILE_ARG"
  [[ -f "$state_file" ]] || { echo "setup state file missing: $state_file" >&2; exit 1; }
  state_dir="$(dirname "$state_file")"
  label="$(state_value label)"
  [[ "$label" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'setup state has an invalid label' >&2; exit 1; }
  [[ "$state_dir" == "$RUNTIME/graph_pilot/"* ]] || { echo 'setup state is outside the configured runtime' >&2; exit 1; }
  job_dir="$RUNTIME/operator/jobs/$label"
  setup_job_id="$(state_value setup_job_id)"
  train_job_id="$(state_value train_job_id)"; eval_job_id="$(state_value eval_job_id)"; train_status="$(state_value train_status)"; eval_status="$(state_value eval_status)"
  setup_template_sha256="$(state_value setup_template_sha256)"; train_template_sha256="$(state_value train_template_sha256)"; eval_template_sha256="$(state_value eval_template_sha256)"; eval_rendered_sha256="$(state_value eval_rendered_sha256)"
  launch_status="$(state_value launch_status)"; launch_nonce="$(state_value launch_nonce)"
  [[ "$setup_job_id" =~ ^[0-9]+$ ]] || { echo 'setup state has no valid setup job id' >&2; exit 1; }
  [[ -z "$train_job_id" && -z "$eval_job_id" && "$train_status" == PENDING && "$eval_status" == PENDING ]] || { echo 'setup state already contains a training/evaluation submission' >&2; exit 1; }
  [[ "$launch_status" == PENDING ]] || { echo 'setup state has already been claimed for launch' >&2; exit 1; }
  [[ "$(state_value expected_repo_commit)" == "$EXPECTED_REPO_COMMIT" ]] || { echo 'setup state commit does not match --expected-commit' >&2; exit 1; }
  [[ "$(state_value setup_status)" == OK ]] || { echo 'setup job has not completed successfully; inspect its state before launch' >&2; exit 1; }
  [[ "$(state_value input_bundle_status)" == VERIFIED ]] || { echo 'setup input bundle is not verified; refusing launch' >&2; exit 1; }
  input_bundle_path="$(state_value input_bundle_path)"
  input_bundle_tree_sha256="$(state_value input_bundle_tree_sha256)"
  input_bundle_status="$(state_value input_bundle_status)"
  [[ "$input_bundle_path" == "$ARCHIVE_ROOT/input_bundles/"* && -d "$input_bundle_path" ]] || { echo 'setup input bundle path is missing or outside HOME archive' >&2; exit 1; }
  [[ "$input_bundle_tree_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo 'setup input bundle tree hash is invalid' >&2; exit 1; }
  [[ -s "$input_bundle_path/tree_sha256" && "$(tr -d '[:space:]' < "$input_bundle_path/tree_sha256")" == "$input_bundle_tree_sha256" ]] || { echo 'setup input bundle tree hash changed' >&2; exit 1; }
  [[ -s "$input_bundle_path/bundle_metadata.json" ]] || { echo 'setup input bundle metadata is missing' >&2; exit 1; }
  grep -Fq "\"repo_commit\": \"$EXPECTED_REPO_COMMIT\"" "$input_bundle_path/bundle_metadata.json" || { echo 'setup input bundle commit metadata does not match --expected-commit' >&2; exit 1; }
  setup_archive_status="$(state_value setup_archive_status)"
  setup_archive_tree_sha256="$(state_value setup_archive_tree_sha256)"
  [[ "$setup_archive_status" == VERIFIED && "$setup_archive_tree_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo 'setup archive is not verified; refusing launch' >&2; exit 1; }
  setup_archive_path="$ARCHIVE_ROOT/setup/${label}_${setup_job_id}"
  [[ -d "$setup_archive_path" ]] || { echo 'verified setup archive is missing; refusing launch' >&2; exit 1; }
  "$PYTHON" - "$setup_archive_path" "$setup_archive_tree_sha256" <<'PY_SETUP_ARCHIVE'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
expected_tree = sys.argv[2]
inventory = root / "inventory.sha256"
tree = root / "tree_sha256"
if not inventory.is_file() or not tree.is_file():
    raise SystemExit("setup archive inventory/tree hash is missing")
payload = inventory.read_text(encoding="utf-8")
actual_tree = hashlib.sha256(payload.encode()).hexdigest()
if actual_tree != expected_tree or tree.read_text(encoding="utf-8").strip() != expected_tree:
    raise SystemExit("setup archive tree hash changed")
listed = set()
for line in payload.splitlines():
    digest, name = line.split("  ", 1)
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"setup archive file mismatch: {name}")
    listed.add(name)
actual = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path.name not in {"inventory.sha256", "tree_sha256"}
}
if actual != listed:
    raise SystemExit("setup archive inventory is incomplete")
PY_SETUP_ARCHIVE
  [[ -s "$job_dir/setup.sbatch" && -s "$job_dir/train.sbatch" && -s "$job_dir/eval.sbatch" ]] || { echo 'setup job scripts are missing' >&2; exit 1; }
  # Legion accounting may be temporarily unavailable. Prefer sacct when it
  # responds, but fall back to the controller's retained terminal record while
  # it is still available; never treat an empty/failed query as completion.
  setup_slurm_state=''
  setup_state_source=''
  if command -v sacct >/dev/null 2>&1; then
    setup_slurm_state="$(sacct -j "$setup_job_id" -X -n -o State 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ "$setup_slurm_state" == COMPLETED ]]; then
      setup_state_source='sacct'
    fi
  fi
  if [[ "$setup_slurm_state" != COMPLETED ]] && command -v scontrol >/dev/null 2>&1; then
    setup_controller_state="$(scontrol show job -o "$setup_job_id" 2>/dev/null || true)"
    if [[ "$setup_controller_state" == *'JobState=COMPLETED'* && "$setup_controller_state" == *'ExitCode=0:0'* ]]; then
      setup_slurm_state='COMPLETED'
      setup_state_source='scontrol'
    fi
  fi
  [[ "$setup_slurm_state" == COMPLETED ]] || { echo "setup SLURM state is not COMPLETED: $setup_slurm_state" >&2; exit 1; }
  setup_status=OK
fi
write_state() {
  local tmp="${state_file}.tmp.$$"
  {
    printf 'label=%s\n' "$label"
    printf 'expected_repo_commit=%s\nlibero_commit=%s\nbase_policy_revision=%s\n' "$EXPECTED_REPO_COMMIT" "$LIBERO_COMMIT" "$BASE_POLICY_REVISION"
    printf 'repo=%s\nscratch_root=%s\narchive_root=%s\nstate_dir=%s\n' "$REPO" "$SCRATCH_ROOT" "$ARCHIVE_ROOT" "$state_dir"
    printf 'setup_job_id=%s\ntrain_job_id=%s\neval_job_id=%s\n' "$setup_job_id" "$train_job_id" "$eval_job_id"
    printf 'setup_status=%s\ntrain_status=%s\neval_status=%s\nlaunch_status=%s\nlaunch_nonce=%s\n' "${setup_status:-SUBMITTED}" "${train_status:-PENDING}" "${eval_status:-PENDING}" "${launch_status:-PENDING}" "${launch_nonce:-}"
    printf 'setup_template_sha256=%s\ntrain_template_sha256=%s\neval_template_sha256=%s\neval_rendered_sha256=%s\n' "${setup_template_sha256:-}" "${train_template_sha256:-}" "${eval_template_sha256:-}" "${eval_rendered_sha256:-}"
    printf 'input_bundle_path=%s\ninput_bundle_tree_sha256=%s\ninput_bundle_status=%s\n' "${input_bundle_path:-}" "${input_bundle_tree_sha256:-}" "${input_bundle_status:-PENDING}"
    printf 'setup_archive_status=%s\nsetup_archive_tree_sha256=%s\ntrain_archive_status=%s\ntrain_archive_tree_sha256=%s\neval_archive_status=%s\neval_archive_tree_sha256=%s\n' "${setup_archive_status:-PENDING}" "${setup_archive_tree_sha256:-}" "${train_archive_status:-PENDING}" "${train_archive_tree_sha256:-}" "${eval_archive_status:-PENDING}" "${eval_archive_tree_sha256:-}"
    printf 'training_run_root=%s\ntraining_manifest=%s\nevaluation_root=%s\n' "${SCRATCH_ROOT}/runs/${label}_${train_job_id:-TRAIN_JOB_ID}" "${SCRATCH_ROOT}/runs/${label}_${train_job_id:-TRAIN_JOB_ID}/training_manifest.json" "${SCRATCH_ROOT}/eval/${label}_${train_job_id:-TRAIN_JOB_ID}_${eval_job_id:-EVAL_JOB_ID}"
  } > "$tmp"
  mv -f "$tmp" "$state_file"
}
submit_id() { local raw="$1" id; id="${raw%%;*}"; [[ "$id" =~ ^[0-9]+$ ]] || { echo "invalid sbatch job id: $raw" >&2; exit 1; }; printf '%s\n' "$id"; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

if [[ "$ACTION" == setup ]]; then
cat > "$job_dir/setup.sbatch" <<EOF
#!/usr/bin/env bash
# Setup, full pair verification, tokenizer materialization, and exact 2-step smoke.
#SBATCH --job-name=${label}_setup
#SBATCH --partition=gpu_a40
#SBATCH --exclude=compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-23:59:00
#SBATCH --output=${RUNTIME}/operator/logs/%x_%j.out
#SBATCH --error=${RUNTIME}/operator/logs/%x_%j.err
set -Eeuo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
REPO='${REPO}'; RUNTIME='${RUNTIME}'; SCRATCH_ROOT='${SCRATCH_ROOT}'; ARCHIVE_ROOT='${ARCHIVE_ROOT}'; STATE_FILE='${state_file}'
EXPECTED_REPO_COMMIT='${EXPECTED_REPO_COMMIT}'; LIBERO_COMMIT='${LIBERO_COMMIT}'; BASE_POLICY_REVISION='${BASE_POLICY_REVISION}'; PYTHON='${PYTHON}'; LABEL='${label}'
DATA_ROOT="\$SCRATCH_ROOT/vla_benchmarking/lora_datasets"; LIBERO_DATA_DIR="\$SCRATCH_ROOT/vlm_benchmarking/data/libero_spatial_v5"; LIBERO_DIR="\$SCRATCH_ROOT/vla_benchmarking/LIBERO"; BASE_POLICY="\$SCRATCH_ROOT/vla_benchmarking/base_models/smolvla_libero-\$BASE_POLICY_REVISION"; GRAPH_BASE_POLICY="\$SCRATCH_ROOT/vla_benchmarking/base_models/smolvla_libero-\$BASE_POLICY_REVISION-graph96-v2"; LIBERO_CONFIG="\$RUNTIME/config/config.yaml"; EVIDENCE="\$SCRATCH_ROOT/runs/\${LABEL}_setup_\$SLURM_JOB_ID"; ARCHIVE_DIR="\$ARCHIVE_ROOT/setup/\${LABEL}_\$SLURM_JOB_ID"
die() { echo "graph setup: \$*" >&2; exit 1; }
copy_tree() { local s="\$1" d="\$2"; [[ -e "\$s" ]] || return 0; mkdir -p "\$d"; if command -v rsync >/dev/null 2>&1; then rsync -a "\$s/" "\$d/"; else cp -a "\$s/." "\$d/"; fi; }
copy_required() { local s="\$1" d="\$2"; [[ -e "\$s" ]] || die "archive source missing: \$s"; mkdir -p "\$(dirname "\$d")"; if [[ -d "\$s" ]]; then copy_tree "\$s" "\$d"; else cp -a "\$s" "\$d"; fi; }
seal_tree() { local root="\$1" mode="\$2"; "\$PYTHON" - "\$root" "\$mode" <<'PY_TREE'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
mode = sys.argv[2]
inventory_path = root / "inventory.sha256"
tree_path = root / "tree_sha256"
if mode == "build":
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"inventory.sha256", "tree_sha256"}:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append((path.relative_to(root).as_posix(), digest))
    if not entries:
        raise SystemExit("archive tree is empty")
    payload = "".join(f"{digest}  {name}\n" for name, digest in entries)
    inventory_path.write_text(payload, encoding="utf-8")
    tree_path.write_text(hashlib.sha256(payload.encode()).hexdigest() + "\n", encoding="utf-8")
else:
    if not inventory_path.is_file() or not tree_path.is_file():
        raise SystemExit("archive inventory/tree hash is missing")
    payload = inventory_path.read_text(encoding="utf-8")
    expected_tree = tree_path.read_text(encoding="utf-8").strip()
    if hashlib.sha256(payload.encode()).hexdigest() != expected_tree:
        raise SystemExit("archive tree hash does not match inventory")
    listed = set()
    for line in payload.splitlines():
        digest, name = line.split("  ", 1)
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"archive inventory mismatch: {name}")
        listed.add(name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name not in {"inventory.sha256", "tree_sha256"}}
    if actual != listed:
        raise SystemExit("archive contains files absent from its inventory")
PY_TREE
}
finish() {
  local workload_rc=\$? archive_rc=0 archive_status=FAILED archive_tree=''
  trap - EXIT
  set +e
  mkdir -p "\$EVIDENCE"
  printf 'exit_code=%s\\n' "\$workload_rc" > "\$EVIDENCE/status.env"
  copy_tree "\$EVIDENCE" "\$ARCHIVE_DIR" || archive_rc=\$?
  if [[ \$archive_rc -eq 0 ]]; then seal_tree "\$ARCHIVE_DIR" build || archive_rc=\$?; fi
  if [[ \$archive_rc -eq 0 ]]; then seal_tree "\$ARCHIVE_DIR" verify || archive_rc=\$?; fi
  if [[ \$archive_rc -eq 0 ]]; then printf 'archive_status=VERIFIED\\n' > "\$ARCHIVE_DIR/archive_status.env"; seal_tree "\$ARCHIVE_DIR" build || archive_rc=\$?; fi
  if [[ \$archive_rc -eq 0 ]]; then seal_tree "\$ARCHIVE_DIR" verify || archive_rc=\$?; fi
  if [[ \$archive_rc -eq 0 ]]; then archive_status=VERIFIED; archive_tree="\$(tr -d '[:space:]' < "\$ARCHIVE_DIR/tree_sha256")"; fi
  printf 'archive_status=%s\\narchive_tree_sha256=%s\\n' "\$archive_status" "\$archive_tree" > "\$EVIDENCE/archive_status.env"
  printf 'input_bundle_path=%s\\ninput_bundle_tree_sha256=%s\\ninput_bundle_status=%s\\n' "\${input_bundle_path:-}" "\${input_bundle_tree_sha256:-}" "\${input_bundle_status:-FAILED}" >> "\$STATE_FILE"
  printf 'setup_archive_status=%s\\nsetup_archive_tree_sha256=%s\\n' "\$archive_status" "\$archive_tree" >> "\$STATE_FILE"
  if [[ \$workload_rc -eq 0 && \$archive_rc -ne 0 ]]; then workload_rc=90; fi
  printf 'setup_status=%s\\n' "\$( [[ \$workload_rc -eq 0 ]] && echo OK || echo FAILED )" >> "\$STATE_FILE"
  exit "\$workload_rc"
}; trap finish EXIT
[[ -d "\$SCRATCH_ROOT" ]] || die "scratch runtime missing: \$SCRATCH_ROOT"
[[ -d "\$REPO/.git" && "\$(git -C "\$REPO" rev-parse HEAD)" == "\$EXPECTED_REPO_COMMIT" ]] || die 'repository commit drift'
[[ -z "\$(git -C "\$REPO" status --porcelain --untracked-files=all)" ]] || die 'repository is dirty'
[[ -x "\$PYTHON" ]] || die "Python environment missing: \$PYTHON"
[[ -d "\$LIBERO_DIR/.git" && "\$(git -C "\$LIBERO_DIR" rev-parse HEAD)" == "\$LIBERO_COMMIT" ]] || die 'LIBERO checkout is not pinned'
[[ -z "\$(git -C "\$LIBERO_DIR" status --porcelain --untracked-files=no)" ]] || die 'LIBERO checkout is dirty'
mkdir -p "\$EVIDENCE"
for p in "\$LIBERO_DATA_DIR" "\$DATA_ROOT/sealed_lora_pair_manifest.json" "\$BASE_POLICY/config.json" "\$BASE_POLICY/base_snapshot_manifest.json" "\$LIBERO_CONFIG"; do [[ -e "\$p" ]] || die "required staged artifact missing: \$p"; done
module purge; module load miniforge/24.3.0-0; source "\$(conda info --base)/etc/profile.d/conda.sh"
export PATH="\$(dirname "\$PYTHON"):\$PATH" PYTHONPATH="\$REPO/vla_benchmarking:\${PYTHONPATH:-}" LIBERO_CONFIG_PATH="\$RUNTIME/config" LIBERO_CONFIG
export BASE_POLICY_REVISION BASE_POLICY GRAPH_BASE_POLICY DATA_ROOT LIBERO_DATA_DIR LIBERO_DIR LIBERO_COMMIT PAIR_MANIFEST="\$DATA_ROOT/sealed_lora_graph_pair_manifest.json" PAIR_SENTINEL="\$DATA_ROOT/sealed_lora_graph_pair_verified.json"
"\$PYTHON" -m py_compile "\$REPO/vla_benchmarking/run_lora_graph_pair_eval.py" "\$REPO/vla_benchmarking/prompt_audit.py" "\$REPO/vla_benchmarking/hdf5_to_lerobot_dataset.py"
graph_artifact_count=0
for p in "\$DATA_ROOT/graph_treatment" "\$DATA_ROOT/arrow_graph_treatment" "\$DATA_ROOT/sealed_lora_graph_pair_manifest.json" "\$DATA_ROOT/sealed_lora_graph_pair_verified.json"; do [[ -e "\$p" ]] && graph_artifact_count=\$((graph_artifact_count + 1)); done
case "\$graph_artifact_count" in
  0)
    # The first conversion needs to materialize and verify the historical
    # pair before the graph manifest can bind to its sentinel.
    "\$PYTHON" "\$REPO/vla_benchmarking/hdf5_to_lerobot_dataset.py" --mode verify --data-dir "\$LIBERO_DATA_DIR" --output-root "\$DATA_ROOT"
    [[ -s "\$DATA_ROOT/sealed_lora_pair_verified.json" ]] || die 'historical pair verification did not produce its sentinel'
    "\$PYTHON" "\$REPO/vla_benchmarking/hdf5_to_lerobot_dataset.py" --mode convert-graph-pair --data-dir "\$LIBERO_DATA_DIR" --output-root "\$DATA_ROOT"
    "\$PYTHON" "\$REPO/vla_benchmarking/hdf5_to_lerobot_dataset.py" --mode verify-graph --data-dir "\$LIBERO_DATA_DIR" --output-root "\$DATA_ROOT"
    ;;
  4)
    # A historical verify pass rewrites its timestamped sentinel and would
    # invalidate the immutable graph manifest binding.  Preflight validates
    # the existing historical sentinel without changing its bytes.  The graph
    # pair was source-grounded by the standalone verification repair job; its
    # immutable sentinel is rechecked here without another full frame traversal.
    "\$PYTHON" "\$REPO/vla_benchmarking/hdf5_to_lerobot_dataset.py" --mode preflight --data-dir "\$LIBERO_DATA_DIR" --output-root "\$DATA_ROOT"
    "\$PYTHON" "\$REPO/vla_benchmarking/hdf5_to_lerobot_dataset.py" --mode preflight-graph --data-dir "\$LIBERO_DATA_DIR" --output-root "\$DATA_ROOT"
    ;;
  *)
    die 'partial graph dataset/pair artifacts found; refusing to generate or overwrite them'
    ;;
esac
"\$PYTHON" "\$REPO/vla_benchmarking/prompt_audit.py" --prepare-graph-policy "\$BASE_POLICY" "\$GRAPH_BASE_POLICY"
"\$PYTHON" "\$REPO/vla_benchmarking/prompt_audit.py" --verify-graph-policy "\$GRAPH_BASE_POLICY"
"\$PYTHON" "\$REPO/vla_benchmarking/prompt_audit.py" --graph-manifest "\$PAIR_MANIFEST" --base-policy "\$GRAPH_BASE_POLICY" --dataset-root "\$DATA_ROOT/graph_treatment" --dataset-root "\$DATA_ROOT/arrow_graph_treatment" --audit-output "\$EVIDENCE/graph_prompt_audit.json"
# Create an immutable HOME reproducibility bundle before running the smoke.
# The graph datasets are copied exactly; capacity/free-space checks happen before
# any large copy and all content is verified again after the copy.
BUNDLE_PARENT="\$ARCHIVE_ROOT/input_bundles"
BUNDLE_STAGING="\$BUNDLE_PARENT/.staging/\${LABEL}_\$SLURM_JOB_ID"
[[ ! -e "\$BUNDLE_STAGING" ]] || die "stale bundle staging path exists: \$BUNDLE_STAGING"
mkdir -p "\$BUNDLE_PARENT/.staging"
dataset_bytes="\$(du -sb "\$DATA_ROOT/graph_treatment" "\$DATA_ROOT/arrow_graph_treatment" | awk '{sum += \$1} END {print sum + 0}')"
[[ "\$dataset_bytes" =~ ^[0-9]+\$ && \$dataset_bytes -gt 0 ]] || die 'could not determine exact graph dataset size'
reserve_bytes=\$((dataset_bytes + dataset_bytes / 10 + 1073741824))
# df is only a conservative filesystem-capacity precheck; quota enforcement
# can be site-specific.  The atomic HOME copy plus full inventory verification
# below is the authoritative fail-closed durability gate.
free_bytes="\$(df -Pk "\$ARCHIVE_ROOT" | awk 'NR == 2 {print \$4 * 1024}')"
[[ "\$free_bytes" =~ ^[0-9]+\$ && \$free_bytes -ge \$reserve_bytes ]] || die "HOME filesystem free-space precheck failed (need \$reserve_bytes bytes, have \${free_bytes:-unknown})"
mkdir "\$BUNDLE_STAGING"
copy_required "\$PAIR_MANIFEST" "\$BUNDLE_STAGING/pairs/sealed_lora_graph_pair_manifest.json"
copy_required "\$PAIR_SENTINEL" "\$BUNDLE_STAGING/pairs/sealed_lora_graph_pair_verified.json"
copy_required "\$DATA_ROOT/sealed_lora_pair_manifest.json" "\$BUNDLE_STAGING/pairs/sealed_lora_pair_manifest.json"
copy_required "\$DATA_ROOT/sealed_lora_pair_verified.json" "\$BUNDLE_STAGING/pairs/sealed_lora_pair_verified.json"
copy_required "\$BASE_POLICY/base_snapshot_manifest.json" "\$BUNDLE_STAGING/base_policy/base_snapshot_manifest.json"
copy_required "\$GRAPH_BASE_POLICY/base_snapshot_manifest.json" "\$BUNDLE_STAGING/graph_policy/base_snapshot_manifest.json"
copy_required "\$GRAPH_BASE_POLICY/policy_preprocessor.json" "\$BUNDLE_STAGING/graph_policy/policy_preprocessor.json"
copy_required "\$GRAPH_BASE_POLICY/tokenizer_provenance.json" "\$BUNDLE_STAGING/graph_policy/tokenizer_provenance.json"
copy_required "\$GRAPH_BASE_POLICY/tokenizer" "\$BUNDLE_STAGING/graph_policy/tokenizer"
copy_required "\$REPO/vla_benchmarking/submit_legion_graph_pilot.sh" "\$BUNDLE_STAGING/code/submit_legion_graph_pilot.sh"
for code_file in hdf5_to_lerobot_dataset.py prompt_audit.py launch_lora_treatment.sh train_lora.sh run_lora_graph_pair_eval.py scene_graph_formats.py; do
  copy_required "\$REPO/vla_benchmarking/\$code_file" "\$BUNDLE_STAGING/code/\$code_file"
done
git -C "\$REPO" archive --format=tar "\$EXPECTED_REPO_COMMIT" vla_benchmarking > "\$BUNDLE_STAGING/code/vla_benchmarking_source.tar"
[[ -s "\$BUNDLE_STAGING/code/vla_benchmarking_source.tar" ]] || die 'exact-commit VLA source archive is empty'
copy_required "\$job_dir/setup.sbatch" "\$BUNDLE_STAGING/operator/setup.sbatch"
copy_required "\$job_dir/train.sbatch" "\$BUNDLE_STAGING/operator/train.sbatch"
copy_required "\$job_dir/eval.sbatch" "\$BUNDLE_STAGING/operator/eval.sbatch"
printf '%s\\n' "\$EXPECTED_REPO_COMMIT" > "\$BUNDLE_STAGING/code/repo_commit.txt"
printf '%s\\n' "\$LIBERO_COMMIT" > "\$BUNDLE_STAGING/code/libero_commit.txt"
"\$PYTHON" - "\$LIBERO_DATA_DIR" "\$PAIR_MANIFEST" "\$BUNDLE_STAGING/source/hdf5_source_manifest.json" <<'PY_HDF5'
import hashlib, json, pathlib, sys
root, pair_path, output = map(pathlib.Path, sys.argv[1:])
pair = json.loads(pair_path.read_text(encoding="utf-8"))
expected = {}
for task in pair.get("tasks", []):
    identity = task.get("source_identity", {})
    if identity.get("path") and identity.get("sha256"):
        expected[pathlib.Path(identity["path"]).name] = identity["sha256"]
files = []
for path in sorted(root.glob("*.hdf5")):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected.get(path.name) != digest:
        raise SystemExit(f"HDF5 source hash mismatch: {path.name}")
    files.append({"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest})
if set(expected) != {item["name"] for item in files}:
    raise SystemExit("HDF5 source inventory differs from the sealed graph manifest")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"algorithm": "sha256", "root_name": root.name, "files": files}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY_HDF5
"\$PYTHON" - "\$BUNDLE_STAGING/bundle_metadata.json" "\$EXPECTED_REPO_COMMIT" "\$LIBERO_COMMIT" "\$BASE_POLICY_REVISION" "\$dataset_bytes" <<'PY_META'
import json, pathlib, sys
output, repo_commit, libero_commit, base_revision, dataset_bytes = sys.argv[1:]
pathlib.Path(output).write_text(json.dumps({
    "schema_version": 1,
    "bundle_kind": "sealed_graph_text_training_input",
    "repo_commit": repo_commit,
    "libero_commit": libero_commit,
    "base_policy_revision": base_revision,
    "graph_tokenizer_max_length": 96,
    "graph_dataset_bytes": int(dataset_bytes),
    "contents": {
        "graph_pair": ["pairs/sealed_lora_graph_pair_manifest.json", "pairs/sealed_lora_graph_pair_verified.json"],
        "historical_pair": ["pairs/sealed_lora_pair_manifest.json", "pairs/sealed_lora_pair_verified.json"],
        "graph_policy": ["graph_policy/base_snapshot_manifest.json", "graph_policy/policy_preprocessor.json", "graph_policy/tokenizer_provenance.json", "graph_policy/tokenizer/"],
        "source": ["source/hdf5_source_manifest.json"],
        "datasets": ["datasets/graph_treatment/", "datasets/arrow_graph_treatment/"],
        "code": ["code/repo_commit.txt", "code/libero_commit.txt", "code/submit_legion_graph_pilot.sh", "operator/"],
    },
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY_META
copy_tree "\$DATA_ROOT/graph_treatment" "\$BUNDLE_STAGING/datasets/graph_treatment"
copy_tree "\$DATA_ROOT/arrow_graph_treatment" "\$BUNDLE_STAGING/datasets/arrow_graph_treatment"
seal_tree "\$BUNDLE_STAGING" build
seal_tree "\$BUNDLE_STAGING" verify
input_bundle_tree_sha256="\$(tr -d '[:space:]' < "\$BUNDLE_STAGING/tree_sha256")"
[[ "\$input_bundle_tree_sha256" =~ ^[0-9a-f]{64}\$ ]] || die 'bundle tree hash is invalid'
input_bundle_path="\$BUNDLE_PARENT/\${LABEL}_\${input_bundle_tree_sha256}"
if [[ -e "\$input_bundle_path" ]]; then
  seal_tree "\$input_bundle_path" verify || die 'existing content-addressed input bundle failed verification'
else
  mv "\$BUNDLE_STAGING" "\$input_bundle_path"
  seal_tree "\$input_bundle_path" verify || die 'content-addressed input bundle failed post-move verification'
fi
printf '%s\\n' "\$input_bundle_path" > "\$EVIDENCE/input_bundle_path"
printf '%s\\n' "\$input_bundle_tree_sha256" > "\$EVIDENCE/input_bundle_tree_sha256"
input_bundle_status=VERIFIED
export BASE_POLICY="\$GRAPH_BASE_POLICY" TRAINING_PROFILE=graph_treatment TRAINING_MODE=smoke RUN_ROOT="\$EVIDENCE/smoke" BATCH_SIZE=32 SEED=1000 PEFT_R=16 DEVICE=cuda RESUME=false RESUME_CONFIG_PATH=''
export LIBERO_CONFIG_PATH="\$RUNTIME/config"
export LIBERO_CONFIG="\$RUNTIME/config/config.yaml"
"\$PYTHON" "\$REPO/vla_benchmarking/prompt_audit.py" --verify-graph-policy "\$GRAPH_BASE_POLICY"
bash "\$REPO/vla_benchmarking/lambda_preflight.sh" graph_treatment
"\$PYTHON" -m pytest -q "\$REPO/vla_benchmarking/tests/test_terminal_reset_compensation.py" > "\$EVIDENCE/terminal_reset_compensation_test.txt" 2>&1 || {
  cat "\$EVIDENCE/terminal_reset_compensation_test.txt" >&2
  die 'terminal reset compensation runtime test failed'
}
grep -Eq '2 passed' "\$EVIDENCE/terminal_reset_compensation_test.txt" || die 'terminal reset compensation runtime test did not execute both cases'
unset EPOCHS STEPS SAVE_FREQ UPDATES_PER_EPOCH
bash "\$REPO/vla_benchmarking/launch_lora_treatment.sh" smoke
[[ -s "\$EVIDENCE/smoke/checkpoints/000002/pretrained_model/adapter_model.safetensors" ]] || die '2-step GPU smoke produced no adapter'
SMOKE_CHECKPOINT="\$EVIDENCE/smoke/checkpoints/000002/pretrained_model"

# Exercise the real pinned SmolVLA loader, checkpoint-local graph tokenizer,
# preprocessing, CUDA action inference, explicit timeout/reset path, and both
# removal-only (task 0) and removal+swap (task 2) randomizations.  Two control
# steps keep this a setup smoke rather than an evaluation result.
LIVE_SMOKE="\$EVIDENCE/live_checkpoint_smoke"
export MODELS="\$SMOKE_CHECKPOINT" TOKENIZER_MODEL="\$SMOKE_CHECKPOINT/tokenizer" TASK_IDS='[0,2]' N_EPISODES=1 BATCH_SIZE=1 SEED=1000 DEVICE=cuda
export CONTEXT_MODE=scene_graph CONTEXT_FORMAT=target_natural_v1 VISUAL_CONDITION=none VISUAL_ARROWS=0 RANDOMIZE_SCENES=1 N_ACTION_STEPS=checkpoint MAX_EPISODES_RENDERED=0 RENDER_MODE=none
"\$PYTHON" "\$REPO/vla_benchmarking/run_lerobot_eval_with_context.py" \
  --eval.use_async_envs=false \
  --output_dir="\$LIVE_SMOKE" \
  --policy.path="\$SMOKE_CHECKPOINT" \
  --env.task_ids='[0,2]' \
  --env.episode_length=2 \
  --env.camera_name=agentview_image,robot0_eye_in_hand_image \
  --env.observation_height=256 \
  --env.observation_width=256

# Trigger the installed LiberoEnv's genuine terminal-success reset branch on a
# real simulator instance, while using the production compensation patch.  The
# success predicate is forced only to deterministically reach that branch; the
# emitted evidence is a reset-semantics smoke, never a task-success result.
"\$PYTHON" - "\$EVIDENCE/live_terminal_reset_smoke.json" <<'PY_RUNTIME_RESET'
import hashlib, inspect, json, pathlib, sys
from libero.libero import benchmark
from lerobot.envs import libero as lerobot_libero
from config import TASK_REMOVE_CONFIG
from radomize_scenes import init_state_evidence, sim_state_sha256
from run_lerobot_eval_with_context import (
    _patch_libero_env_bddl_selection,
    _patch_libero_env_camera_creation,
    _patch_libero_env_terminal_reset_compensation,
)

output = pathlib.Path(sys.argv[1]).resolve()
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
upstream_step_source = inspect.getsource(lerobot_libero.LiberoEnv.step)
_patch_libero_env_bddl_selection(TASK_REMOVE_CONFIG)
_patch_libero_env_camera_creation()
_patch_libero_env_terminal_reset_compensation()
env = lerobot_libero.LiberoEnv(
    suite,
    0,
    "libero_spatial",
    episode_length=2,
    camera_name="agentview_image,robot0_eye_in_hand_image",
    render_mode=None,
    observation_width=256,
    observation_height=256,
    episode_index=0,
    n_envs=1,
)
selected = init_state_evidence(env)
try:
    env.reset(seed=1000)
    counter_before = int(env.init_state_id)
    reset_state_sha256 = sim_state_sha256(env._env.sim)
    env._env.check_success = lambda: True
    _, _, terminated, truncated, info = env.step(lerobot_libero.get_libero_dummy_action())
    compensation = getattr(env, "_paired_reset_compensation", None)
    if not terminated or truncated or not info.get("is_success"):
        raise RuntimeError("actual LiberoEnv did not take the forced terminal-success branch")
    if int(env.init_state_id) != counter_before:
        raise RuntimeError("terminal-success reset consumed an init-state row after compensation")
    if not isinstance(compensation, dict) or compensation.get("detected") is not True:
        raise RuntimeError("production terminal reset compensation was not observed")
    payload = {
        "status": "PASS",
        "scope": "actual_pinned_libero_terminal_reset_semantics_not_task_success",
        "libero_env_source": inspect.getsourcefile(lerobot_libero.LiberoEnv),
        "upstream_libero_env_step_sha256": hashlib.sha256(upstream_step_source.encode()).hexdigest(),
        "patched_libero_env_step_sha256": hashlib.sha256(inspect.getsource(lerobot_libero.LiberoEnv.step).encode()).hexdigest(),
        "task_id": 0,
        "selected_index": selected["selected_index"],
        "selected_row_sha256": selected["selected_row_sha256"],
        "reset_sim_state_sha256": reset_state_sha256,
        "counter_before_terminal_step": counter_before,
        "counter_after_terminal_step": int(env.init_state_id),
        "terminal_reset_compensation": compensation,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
finally:
    close = getattr(env, "close", None)
    if callable(close):
        close()
PY_RUNTIME_RESET

"\$PYTHON" - "\$LIVE_SMOKE" "\$EVIDENCE/live_checkpoint_smoke_verified.json" <<'PY_RUNTIME_VERIFY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2]).resolve()
required = [root / "eval_info.json", root / "prompt_audit.jsonl", root / "randomization_audit.jsonl"]
for path in required:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"live checkpoint smoke evidence missing: {path}")
records = [json.loads(line) for line in (root / "randomization_audit.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
final = [item for item in records if item.get("details", {}).get("status") == "environment_ok"]
by_task = {int(item["task_id"]): item for item in final if item.get("task_id") in (0, 2)}
if set(by_task) != {0, 2}:
    raise SystemExit("live checkpoint smoke did not realize both task 0 and task 2")
for task_id, record in by_task.items():
    details = record["details"]
    init_state = details.get("init_state")
    if not isinstance(init_state, dict) or not isinstance(init_state.get("selected_index"), int) or len(init_state.get("selected_row_sha256", "")) != 64:
        raise SystemExit(f"task {task_id} lacks paired init-row evidence")
    if len(details.get("sim_state_sha256", "")) != 64:
        raise SystemExit(f"task {task_id} lacks realized simulator-state evidence")
if by_task[0]["dimensions_realized"].get("scene_layout") is not False or by_task[0]["details"].get("swaps") != []:
    raise SystemExit("task 0 did not realize the sealed removal-only condition")
if by_task[2]["dimensions_realized"].get("scene_layout") is not True or not by_task[2]["details"].get("swaps"):
    raise SystemExit("task 2 did not realize the sealed removal+swap condition")
payload = {
    "status": "PASS",
    "scope": "actual_checkpoint_load_preprocess_cuda_inference_timeout_and_randomization",
    "task_ids": [0, 2],
    "episode_length": 2,
    "evidence_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in required},
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY_RUNTIME_VERIFY
EOF

cat > "$job_dir/train.sbatch" <<EOF
#!/usr/bin/env bash
# Full no-arrow graph-text LoRA training, chained after setup/smoke.
#SBATCH --job-name=${label}_train
#SBATCH --partition=gpu_a40
#SBATCH --exclude=compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-23:59:00
#SBATCH --output=${RUNTIME}/operator/logs/%x_%j.out
#SBATCH --error=${RUNTIME}/operator/logs/%x_%j.err
set -Eeuo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
REPO='${REPO}'; RUNTIME='${RUNTIME}'; SCRATCH_ROOT='${SCRATCH_ROOT}'; ARCHIVE_ROOT='${ARCHIVE_ROOT}'; STATE_FILE='${state_file}'; EXPECTED_REPO_COMMIT='${EXPECTED_REPO_COMMIT}'; LIBERO_COMMIT='${LIBERO_COMMIT}'; BASE_POLICY_REVISION='${BASE_POLICY_REVISION}'; PYTHON='${PYTHON}'; LABEL='${label}'
DATA_ROOT="\$SCRATCH_ROOT/vla_benchmarking/lora_datasets"; LIBERO_DATA_DIR="\$SCRATCH_ROOT/vlm_benchmarking/data/libero_spatial_v5"; LIBERO_DIR="\$SCRATCH_ROOT/vla_benchmarking/LIBERO"; BASE_POLICY="\$SCRATCH_ROOT/vla_benchmarking/base_models/smolvla_libero-\$BASE_POLICY_REVISION-graph96-v2"; LIBERO_CONFIG="\$RUNTIME/config/config.yaml"; RUN_ROOT="\$SCRATCH_ROOT/runs/\${LABEL}_\$SLURM_JOB_ID"; ARCHIVE_DIR="\$ARCHIVE_ROOT/runs/\$(basename "\$RUN_ROOT")"; INPUT_BUNDLE_PATH=''
die() { echo "graph train: \$*" >&2; exit 1; }; copy_tree() { local s="\$1" d="\$2"; [[ -e "\$s" ]] || return 0; mkdir -p "\$d"; if command -v rsync >/dev/null 2>&1; then rsync -a "\$s/" "\$d/"; else cp -a "\$s/." "\$d/"; fi; }; copy_required() { local s="\$1" d="\$2"; [[ -e "\$s" ]] || return 1; mkdir -p "\$(dirname "\$d")"; if [[ -d "\$s" ]]; then copy_tree "\$s" "\$d"; else cp -a "\$s" "\$d"; fi; }
seal_tree() { local root="\$1" mode="\$2"; "\$PYTHON" - "\$root" "\$mode" <<'PY_ARCHIVE'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve(); mode = sys.argv[2]
inventory = root / "inventory.sha256"; tree = root / "tree_sha256"
if mode == "build":
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"inventory.sha256", "tree_sha256"}:
            rows.append((path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
    if not rows: raise SystemExit("archive tree is empty")
    payload = "".join(f"{digest}  {name}\n" for name, digest in rows)
    inventory.write_text(payload, encoding="utf-8"); tree.write_text(hashlib.sha256(payload.encode()).hexdigest() + "\n", encoding="utf-8")
else:
    if not inventory.is_file() or not tree.is_file(): raise SystemExit("archive inventory/tree hash missing")
    payload = inventory.read_text(encoding="utf-8")
    if hashlib.sha256(payload.encode()).hexdigest() != tree.read_text(encoding="utf-8").strip(): raise SystemExit("archive tree hash mismatch")
    listed = set()
    for line in payload.splitlines():
        digest, name = line.split("  ", 1); path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest: raise SystemExit(f"archive file mismatch: {name}")
        listed.add(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name not in {"inventory.sha256", "tree_sha256"}}
    if actual != listed: raise SystemExit("archive inventory is incomplete")
PY_ARCHIVE
}
state_value() { awk -F= -v key="\$1" '\$1 == key { value = substr(\$0, index(\$0, "=") + 1) } END { print value }' "\$STATE_FILE"; }
archive_repro_context() { local d="\$1"; mkdir -p "\$d/input_bundle"; for rel in bundle_metadata.json inventory.sha256 tree_sha256 graph_policy/base_snapshot_manifest.json graph_policy/policy_preprocessor.json graph_policy/tokenizer_provenance.json source/hdf5_source_manifest.json pairs/sealed_lora_graph_pair_manifest.json pairs/sealed_lora_graph_pair_verified.json pairs/sealed_lora_pair_manifest.json pairs/sealed_lora_pair_verified.json code/repo_commit.txt code/libero_commit.txt; do copy_required "\$INPUT_BUNDLE_PATH/\$rel" "\$d/input_bundle/\$rel" || return 1; done; copy_required "\$INPUT_BUNDLE_PATH/graph_policy/tokenizer" "\$d/input_bundle/graph_policy/tokenizer" || return 1; for rel in training_manifest.json run_provenance.json graph_tokenizer_audit.json; do copy_required "\$RUN_ROOT/\$rel" "\$d/run/\$rel" || return 1; done; copy_required "\$RUNTIME/operator/jobs/\$LABEL/train.sbatch" "\$d/operator/train.sbatch" || return 1; }
finish() { local workload_rc=\$? archive_rc=0 archive_status=FAILED archive_tree=''; trap - EXIT; set +e; [[ ! -e "\$ARCHIVE_DIR" ]] || archive_rc=1; if [[ \$archive_rc -eq 0 ]]; then copy_tree "\$RUN_ROOT" "\$ARCHIVE_DIR" || archive_rc=\$?; fi; if [[ \$archive_rc -eq 0 ]]; then archive_repro_context "\$ARCHIVE_DIR/repro" || archive_rc=\$?; fi; if [[ \$archive_rc -eq 0 ]]; then seal_tree "\$ARCHIVE_DIR" build || archive_rc=\$?; fi; if [[ \$archive_rc -eq 0 ]]; then seal_tree "\$ARCHIVE_DIR" verify || archive_rc=\$?; fi; if [[ \$archive_rc -eq 0 ]]; then printf 'archive_status=VERIFIED\\n' > "\$ARCHIVE_DIR/archive_status.env"; seal_tree "\$ARCHIVE_DIR" build || archive_rc=\$?; fi; if [[ \$archive_rc -eq 0 ]]; then seal_tree "\$ARCHIVE_DIR" verify || archive_rc=\$?; fi; if [[ \$archive_rc -eq 0 ]]; then archive_status=VERIFIED; archive_tree="\$(tr -d '[:space:]' < "\$ARCHIVE_DIR/tree_sha256")"; fi; printf 'train_archive_status=%s\\ntrain_archive_tree_sha256=%s\\n' "\$archive_status" "\$archive_tree" >> "\$STATE_FILE"; if [[ \$workload_rc -eq 0 && \$archive_rc -ne 0 ]]; then workload_rc=90; fi; printf 'train_status=%s\\n' "\$( [[ \$workload_rc -eq 0 ]] && echo OK || echo FAILED )" >> "\$STATE_FILE"; exit "\$workload_rc"; }; trap finish EXIT
[[ -d "\$SCRATCH_ROOT" && -d "\$REPO/.git" && -x "\$PYTHON" ]] || die 'scratch, repository, or Python runtime missing'; [[ "\$(git -C "\$REPO" rev-parse HEAD)" == "\$EXPECTED_REPO_COMMIT" ]] || die 'repository commit drift'; [[ -z "\$(git -C "\$REPO" status --porcelain --untracked-files=all)" ]] || die 'repository is dirty'; [[ -d "\$BASE_POLICY/policy_preprocessor.json" || -f "\$BASE_POLICY/policy_preprocessor.json" ]] || die 'graph96 policy is missing'; [[ -d "\$DATA_ROOT/graph_treatment" && -f "\$DATA_ROOT/sealed_lora_graph_pair_verified.json" ]] || die 'graph dataset/pair is missing'
module purge; module load miniforge/24.3.0-0; source "\$(conda info --base)/etc/profile.d/conda.sh"
export PATH="\$(dirname "\$PYTHON"):\$PATH" PYTHONPATH="\$REPO/vla_benchmarking:\${PYTHONPATH:-}" LIBERO_CONFIG_PATH="\$RUNTIME/config" LIBERO_CONFIG
export BASE_POLICY BASE_POLICY_REVISION DATA_ROOT LIBERO_DATA_DIR LIBERO_DIR LIBERO_COMMIT PAIR_MANIFEST="\$DATA_ROOT/sealed_lora_graph_pair_manifest.json" PAIR_SENTINEL="\$DATA_ROOT/sealed_lora_graph_pair_verified.json" TRAINING_PROFILE=graph_treatment TRAINING_MODE=full RUN_ROOT BATCH_SIZE=32 SEED=1000 PEFT_R=16 DEVICE=cuda RESUME=false RESUME_CONFIG_PATH=''
INPUT_BUNDLE_PATH="\$(state_value input_bundle_path)"; INPUT_BUNDLE_TREE_SHA256="\$(state_value input_bundle_tree_sha256)"
[[ "\$(state_value input_bundle_status)" == VERIFIED && -d "\$INPUT_BUNDLE_PATH" && "\$INPUT_BUNDLE_TREE_SHA256" =~ ^[0-9a-f]{64}\$ ]] || die 'verified HOME input bundle is missing from setup state'
[[ -s "\$INPUT_BUNDLE_PATH/tree_sha256" && "\$(tr -d '[:space:]' < "\$INPUT_BUNDLE_PATH/tree_sha256")" == "\$INPUT_BUNDLE_TREE_SHA256" ]] || die 'HOME input bundle anchor changed'
seal_tree "\$INPUT_BUNDLE_PATH" verify || die 'HOME input bundle full verification failed'
unset EPOCHS STEPS SAVE_FREQ UPDATES_PER_EPOCH
"\$PYTHON" "\$REPO/vla_benchmarking/prompt_audit.py" --verify-graph-policy "\$BASE_POLICY"
bash "\$REPO/vla_benchmarking/lambda_preflight.sh" graph_treatment
bash "\$REPO/vla_benchmarking/launch_lora_treatment.sh" full
[[ -s "\$RUN_ROOT/training_manifest.json" && -s "\$RUN_ROOT/checkpoints/029190/pretrained_model/adapter_model.safetensors" ]] || die 'sealed epoch-15 training artifacts are missing'
EOF

cat > "$job_dir/eval.sbatch" <<EOF
#!/usr/bin/env bash
# Exactly two no-arrow cells: graph-present text and graph-removed text.
#SBATCH --job-name=${label}_eval
#SBATCH --partition=gpu_a40
#SBATCH --exclude=compute-4-13
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-23:59:00
#SBATCH --output=${RUNTIME}/operator/logs/%x_%j.out
#SBATCH --error=${RUNTIME}/operator/logs/%x_%j.err
set -Eeuo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
REPO='${REPO}'; RUNTIME='${RUNTIME}'; SCRATCH_ROOT='${SCRATCH_ROOT}'; ARCHIVE_ROOT='${ARCHIVE_ROOT}'; STATE_FILE='${state_file}'; EXPECTED_REPO_COMMIT='${EXPECTED_REPO_COMMIT}'; LIBERO_COMMIT='${LIBERO_COMMIT}'; BASE_POLICY_REVISION='${BASE_POLICY_REVISION}'; PYTHON='${PYTHON}'; LABEL='${label}'; TRAIN_JOB_ID='__TRAIN_JOB_ID__'
DATA_ROOT="\$SCRATCH_ROOT/vla_benchmarking/lora_datasets"; LIBERO_DIR="\$SCRATCH_ROOT/vla_benchmarking/LIBERO"; GRAPH_BASE_POLICY="\$SCRATCH_ROOT/vla_benchmarking/base_models/smolvla_libero-\$BASE_POLICY_REVISION-graph96-v2"; TRAIN_ROOT="\$SCRATCH_ROOT/runs/\${LABEL}_\${TRAIN_JOB_ID}"; MANIFEST="\$TRAIN_ROOT/training_manifest.json"; ADAPTER="\$TRAIN_ROOT/checkpoints/029190/pretrained_model"; OUTPUT_ROOT="\$SCRATCH_ROOT/eval/\${LABEL}_\${TRAIN_JOB_ID}_\$SLURM_JOB_ID"; ARCHIVE_DIR="\$ARCHIVE_ROOT/eval/\$(basename "\$OUTPUT_ROOT")"; INPUT_BUNDLE_PATH=''
die() { echo "graph eval: \$*" >&2; exit 1; }; copy_tree() { local s="\$1" d="\$2"; [[ -e "\$s" ]] || return 0; mkdir -p "\$d"; if command -v rsync >/dev/null 2>&1; then rsync -a "\$s/" "\$d/"; else cp -a "\$s/." "\$d/"; fi; }; copy_required() { local s="\$1" d="\$2"; [[ -e "\$s" ]] || return 1; mkdir -p "\$(dirname "\$d")"; if [[ -d "\$s" ]]; then copy_tree "\$s" "\$d"; else cp -a "\$s" "\$d"; fi; }
seal_tree() { local root="\$1" mode="\$2"; "\$PYTHON" - "\$root" "\$mode" <<'PY_ARCHIVE'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve(); mode = sys.argv[2]
inventory = root / "inventory.sha256"; tree = root / "tree_sha256"
if mode == "build":
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"inventory.sha256", "tree_sha256"}:
            rows.append((path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
    if not rows: raise SystemExit("archive tree is empty")
    payload = "".join(f"{digest}  {name}\n" for name, digest in rows)
    inventory.write_text(payload, encoding="utf-8"); tree.write_text(hashlib.sha256(payload.encode()).hexdigest() + "\n", encoding="utf-8")
else:
    if not inventory.is_file() or not tree.is_file(): raise SystemExit("archive inventory/tree hash missing")
    payload = inventory.read_text(encoding="utf-8")
    if hashlib.sha256(payload.encode()).hexdigest() != tree.read_text(encoding="utf-8").strip(): raise SystemExit("archive tree hash mismatch")
    listed = set()
    for line in payload.splitlines():
        digest, name = line.split("  ", 1); path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest: raise SystemExit(f"archive file mismatch: {name}")
        listed.add(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name not in {"inventory.sha256", "tree_sha256"}}
    if actual != listed: raise SystemExit("archive inventory is incomplete")
PY_ARCHIVE
}
state_value() { awk -F= -v key="\$1" '\$1 == key { value = substr(\$0, index(\$0, "=") + 1) } END { print value }' "\$STATE_FILE"; }
archive_repro_context() { local d="\$1"; mkdir -p "\$d/input_bundle"; for rel in bundle_metadata.json inventory.sha256 tree_sha256 graph_policy/base_snapshot_manifest.json graph_policy/policy_preprocessor.json graph_policy/tokenizer_provenance.json source/hdf5_source_manifest.json pairs/sealed_lora_graph_pair_manifest.json pairs/sealed_lora_graph_pair_verified.json pairs/sealed_lora_pair_manifest.json pairs/sealed_lora_pair_verified.json code/repo_commit.txt code/libero_commit.txt; do copy_required "\$INPUT_BUNDLE_PATH/\$rel" "\$d/input_bundle/\$rel" || return 1; done; copy_required "\$INPUT_BUNDLE_PATH/graph_policy/tokenizer" "\$d/input_bundle/graph_policy/tokenizer" || return 1; for rel in graph_trained_text_pair_manifest.json graph_trained_text_pair_summary.csv; do copy_required "\$OUTPUT_ROOT/\$rel" "\$d/eval/\$rel" || return 1; done; copy_required "\$RUNTIME/operator/jobs/\$LABEL/eval.sbatch" "\$d/operator/eval.sbatch" || return 1; copy_required "\$RUNTIME/operator/jobs/\$LABEL/eval.\$TRAIN_JOB_ID.sbatch" "\$d/operator/eval.\$TRAIN_JOB_ID.sbatch" || return 1; }
finish() { local workload_rc=\$? archive_rc=0 archive_status=FAILED archive_tree=''; trap - EXIT; set +e; [[ ! -e "\$ARCHIVE_DIR" ]] || archive_rc=1; if [[ \$archive_rc -eq 0 ]]; then copy_tree "\$OUTPUT_ROOT" "\$ARCHIVE_DIR" || archive_rc=\$?; fi; if [[ \$archive_rc -eq 0 ]]; then archive_repro_context "\$ARCHIVE_DIR/repro" || archive_rc=\$?; fi; if [[ \$archive_rc -eq 0 ]]; then seal_tree "\$ARCHIVE_DIR" build || archive_rc=\$?; fi; if [[ \$archive_rc -eq 0 ]]; then seal_tree "\$ARCHIVE_DIR" verify || archive_rc=\$?; fi; if [[ \$archive_rc -eq 0 ]]; then printf 'archive_status=VERIFIED\\n' > "\$ARCHIVE_DIR/archive_status.env"; seal_tree "\$ARCHIVE_DIR" build || archive_rc=\$?; fi; if [[ \$archive_rc -eq 0 ]]; then seal_tree "\$ARCHIVE_DIR" verify || archive_rc=\$?; fi; if [[ \$archive_rc -eq 0 ]]; then archive_status=VERIFIED; archive_tree="\$(tr -d '[:space:]' < "\$ARCHIVE_DIR/tree_sha256")"; fi; printf 'eval_archive_status=%s\\neval_archive_tree_sha256=%s\\n' "\$archive_status" "\$archive_tree" >> "\$STATE_FILE"; if [[ \$workload_rc -eq 0 && \$archive_rc -ne 0 ]]; then workload_rc=90; fi; printf 'eval_status=%s\\n' "\$( [[ \$workload_rc -eq 0 ]] && echo OK || echo FAILED )" >> "\$STATE_FILE"; exit "\$workload_rc"; }; trap finish EXIT
[[ -d "\$SCRATCH_ROOT" && -d "\$REPO/.git" && -x "\$PYTHON" ]] || die 'scratch, repository, or Python runtime missing'; [[ "\$(git -C "\$REPO" rev-parse HEAD)" == "\$EXPECTED_REPO_COMMIT" ]] || die 'repository commit drift'; [[ -z "\$(git -C "\$REPO" status --porcelain --untracked-files=all)" ]] || die 'repository is dirty'; [[ -s "\$MANIFEST" && -s "\$ADAPTER/adapter_model.safetensors" ]] || die 'training manifest/checkpoint missing'; [[ -d "\$GRAPH_BASE_POLICY" && -f "\$GRAPH_BASE_POLICY/policy_preprocessor.json" ]] || die 'graph96 policy is missing'
module purge; module load miniforge/24.3.0-0; source "\$(conda info --base)/etc/profile.d/conda.sh"
export PATH="\$(dirname "\$PYTHON"):\$PATH" PYTHONPATH="\$REPO/vla_benchmarking:\${PYTHONPATH:-}" LIBERO_CONFIG_PATH="\$RUNTIME/config" LIBERO_CONFIG="\$RUNTIME/config/config.yaml" BASE_POLICY="\$GRAPH_BASE_POLICY" TRAINING_PROFILE=graph_treatment PROFILE=graph_treatment VISUAL_CONDITION=none VISUAL_ARROWS=0 RANDOMIZE_SCENES=1 DEVICE=cuda
export DATA_ROOT LIBERO_DIR LIBERO_COMMIT LIBERO_DATA_DIR="\$SCRATCH_ROOT/vlm_benchmarking/data/libero_spatial_v5"
INPUT_BUNDLE_PATH="\$(state_value input_bundle_path)"; INPUT_BUNDLE_TREE_SHA256="\$(state_value input_bundle_tree_sha256)"
[[ "\$(state_value input_bundle_status)" == VERIFIED && -d "\$INPUT_BUNDLE_PATH" && "\$INPUT_BUNDLE_TREE_SHA256" =~ ^[0-9a-f]{64}\$ ]] || die 'verified HOME input bundle is missing from setup state'
[[ -s "\$INPUT_BUNDLE_PATH/tree_sha256" && "\$(tr -d '[:space:]' < "\$INPUT_BUNDLE_PATH/tree_sha256")" == "\$INPUT_BUNDLE_TREE_SHA256" ]] || die 'HOME input bundle anchor changed'
seal_tree "\$INPUT_BUNDLE_PATH" verify || die 'HOME input bundle full verification failed'
"\$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
"\$PYTHON" "\$REPO/vla_benchmarking/prompt_audit.py" --verify-graph-policy "\$GRAPH_BASE_POLICY"
mkdir -p "\$OUTPUT_ROOT"
"\$PYTHON" "\$REPO/vla_benchmarking/run_lora_graph_pair_eval.py" --adapter-checkpoint "\$ADAPTER" --training-manifest "\$MANIFEST" --output-root "\$OUTPUT_ROOT" --seeds 1000 --episodes 10 --batch-size 1 --device cuda --videos --max-videos 1
[[ -s "\$OUTPUT_ROOT/graph_trained_text_pair_manifest.json" && -s "\$OUTPUT_ROOT/graph_trained_text_pair_summary.csv" ]] || die 'two-cell evaluation outputs are missing'
"\$PYTHON" - "\$OUTPUT_ROOT/graph_trained_text_pair_manifest.json" <<'PY'
import json, pathlib, sys
d=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
if [c.get('cell_id') for c in d.get('cells', [])] != ['graph_trained_graph_context', 'graph_trained_standard']:
    raise SystemExit('evaluation did not produce exactly graph-present and graph-removed cells')
if d.get('visual_condition') != 'none' or d.get('seeds') != [1000] or d.get('episodes') != 10 or d.get('batch_size') != 1:
    raise SystemExit('evaluation manifest violates the sealed pilot condition')
PY
EOF
fi

# Materialize and submit only the requested stage.  Setup deliberately stops
# after its compute-node smoke so the reviewer can inspect it before launch.
if [[ "$ACTION" == setup ]]; then
  chmod 700 "$job_dir"/*.sbatch
  setup_template_sha256="$(sha256_file "$job_dir/setup.sbatch")"
  train_template_sha256="$(sha256_file "$job_dir/train.sbatch")"
  eval_template_sha256="$(sha256_file "$job_dir/eval.sbatch")"
  write_state
  setup_job_id="$(submit_id "$(sbatch --parsable "$job_dir/setup.sbatch")")"; setup_status=SUBMITTED; write_state
  printf 'submitted graph setup/smoke: setup=%s state=%s\n' "$setup_job_id" "$state_file"
else
  bundle_setup_sha256="$(sha256_file "$input_bundle_path/operator/setup.sbatch")"
  bundle_train_sha256="$(sha256_file "$input_bundle_path/operator/train.sbatch")"
  bundle_eval_sha256="$(sha256_file "$input_bundle_path/operator/eval.sbatch")"
  [[ "$setup_template_sha256" == "$bundle_setup_sha256" && "$train_template_sha256" == "$bundle_train_sha256" && "$eval_template_sha256" == "$bundle_eval_sha256" ]] || { echo 'operator job template drifted from the sealed HOME bundle' >&2; exit 1; }
  [[ "$(sha256_file "$job_dir/setup.sbatch")" == "$setup_template_sha256" && "$(sha256_file "$job_dir/train.sbatch")" == "$train_template_sha256" && "$(sha256_file "$job_dir/eval.sbatch")" == "$eval_template_sha256" ]] || { echo 'operator job template was tampered with after setup' >&2; exit 1; }
  chmod 700 "$job_dir"/*.sbatch
  launch_lock="$state_file.launch.lock"
  mkdir "$launch_lock" || { echo 'launch has already been claimed; refusing duplicate submission' >&2; exit 1; }
  launch_nonce="$(date -u +%Y%m%dT%H%M%S%N)_$$"
  launch_status=CLAIMED
  write_state
  train_raw=''; if train_raw="$(sbatch --parsable --dependency=afterok:$setup_job_id "$job_dir/train.sbatch")"; then train_job_id="$(submit_id "$train_raw")"; train_status=QUEUED; write_state; else train_status=SUBMIT_FAILED; write_state; exit 1; fi
  eval_template_sha256="$eval_template_sha256"
  rendered_eval="$job_dir/eval.${train_job_id}.sbatch"
  [[ ! -e "$rendered_eval" ]] || { eval_status=RENDER_FAILED; write_state; exit 1; }
  rendered_tmp="${rendered_eval}.tmp.$$"
  sed "s/__TRAIN_JOB_ID__/${train_job_id}/g" "$job_dir/eval.sbatch" > "$rendered_tmp" && mv "$rendered_tmp" "$rendered_eval" || { rm -f "$rendered_tmp"; eval_status=RENDER_FAILED; write_state; exit 1; }
  eval_rendered_sha256="$(sha256_file "$rendered_eval")"
  eval_raw=''; if eval_raw="$(sbatch --parsable --dependency=afterok:$train_job_id "$rendered_eval")"; then eval_job_id="$(submit_id "$eval_raw")"; eval_status=QUEUED; write_state; else eval_status=SUBMIT_FAILED; write_state; exit 1; fi
  printf 'submitted graph train+eval: setup=%s train=%s eval=%s state=%s\n' "$setup_job_id" "$train_job_id" "$eval_job_id" "$state_file"
fi
