#!/usr/bin/env bash
# Submit the complete no-arrow action+late-visual LoRA pilot on Legion.
#
# This is intentionally a separate chain from the existing graph pilot:
# setup/smoke -> full candidate training -> matched clean two-cell evaluation.
# The submitter itself only performs login-node validation and sbatch calls.
set -Eeuo pipefail

EXPECTED_REPO_COMMIT=''
ACTION_ONLY_CHECKPOINT=''
ACTION_ONLY_TRAINING_MANIFEST=''
DATA_ROOT_ARG=''
LIBERO_DATA_DIR_ARG=''
LIBERO_DIR_ARG=''
BASE_POLICY_ARG=''
LIBERO_CONFIG_ARG=''
RUNTIME_ARG=''
SCRATCH_ARG=''
ARCHIVE_ARG=''
LABEL_ARG=''

usage() {
  echo "usage: $0 --expected-commit SHA --action-only-checkpoint PATH --action-only-training-manifest PATH [options]" >&2
  echo "options: --label NAME --data-root PATH --libero-data-dir PATH --libero-dir PATH --base-policy PATH --libero-config PATH --runtime-root PATH --scratch-root PATH --archive-root PATH" >&2
}
while (($#)); do
  case "$1" in
    --expected-commit) EXPECTED_REPO_COMMIT="${2:-}"; shift 2 ;;
    --action-only-checkpoint) ACTION_ONLY_CHECKPOINT="${2:-}"; shift 2 ;;
    --action-only-training-manifest) ACTION_ONLY_TRAINING_MANIFEST="${2:-}"; shift 2 ;;
    --label) LABEL_ARG="${2:-}"; shift 2 ;;
    --data-root) DATA_ROOT_ARG="${2:-}"; shift 2 ;;
    --libero-data-dir) LIBERO_DATA_DIR_ARG="${2:-}"; shift 2 ;;
    --libero-dir) LIBERO_DIR_ARG="${2:-}"; shift 2 ;;
    --base-policy) BASE_POLICY_ARG="${2:-}"; shift 2 ;;
    --libero-config) LIBERO_CONFIG_ARG="${2:-}"; shift 2 ;;
    --runtime-root) RUNTIME_ARG="${2:-}"; shift 2 ;;
    --scratch-root) SCRATCH_ARG="${2:-}"; shift 2 ;;
    --archive-root) ARCHIVE_ARG="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
[[ "$EXPECTED_REPO_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo '--expected-commit must be exactly 40 lowercase hexadecimal characters' >&2; exit 2; }
[[ -n "$ACTION_ONLY_CHECKPOINT" && -n "$ACTION_ONLY_TRAINING_MANIFEST" ]] || { echo 'historical action-only checkpoint and training manifest are required' >&2; exit 2; }
[[ -z "${SLURM_JOB_ID:-}" ]] || { echo 'run this submitter on the Legion login node, not inside a job' >&2; exit 2; }
command -v sbatch >/dev/null 2>&1 || { echo 'sbatch is unavailable; this is not the Legion login node' >&2; exit 2; }

LIBERO_COMMIT='8f1084e3132a39270c3a13ebe37270a43ece2a01'
BASE_POLICY_REVISION='6721902bc4d61e50a3bfdb11dfb4cb626f05d102'
POLICY_ID='action_visual_lora_v1'
PROFILE='no_arrow_treatment'
SEALED_STEPS=29190
SEALED_SAVE_FREQ=1946
SEALED_BATCH_SIZE=32
SEALED_SEED=1000
SEALED_PEFT_R=16
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
RUNTIME="${RUNTIME_ARG:-/home/hjaber/EmbodimentSemantic_runtime}"
SCRATCH_ROOT="${SCRATCH_ARG:-/mnt/beegfs/hjaber/EmbodimentSemantic_runtime}"
ARCHIVE_ROOT="${ARCHIVE_ARG:-/home/hjaber/EmbodimentSemantic_archive}"
PYTHON='/home/hjaber/.conda/envs/embodiment-smolvla-py312/bin/python'
DATA_ROOT="${DATA_ROOT_ARG:-$SCRATCH_ROOT/vla_benchmarking/lora_datasets}"
LIBERO_DATA_DIR="${LIBERO_DATA_DIR_ARG:-$SCRATCH_ROOT/vlm_benchmarking/data/libero_spatial_v5}"
LIBERO_DIR="${LIBERO_DIR_ARG:-$SCRATCH_ROOT/vla_benchmarking/LIBERO}"
BASE_POLICY="${BASE_POLICY_ARG:-$SCRATCH_ROOT/vla_benchmarking/base_models/smolvla_libero-$BASE_POLICY_REVISION}"
LIBERO_CONFIG="${LIBERO_CONFIG_ARG:-$RUNTIME/config/config.yaml}"

[[ -d "$REPO/.git" ]] || { echo "repository missing: $REPO" >&2; exit 1; }
[[ "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)" == "$EXPECTED_REPO_COMMIT" ]] || { echo "repository is not reviewed commit $EXPECTED_REPO_COMMIT" >&2; exit 1; }
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=all)" ]] || { echo 'repository is dirty; refusing submission' >&2; exit 1; }
[[ -d "$RUNTIME" && -d "$SCRATCH_ROOT" && -d "$ARCHIVE_ROOT" ]] || { echo 'runtime, scratch, and archive roots must already exist on Legion' >&2; exit 1; }

if [[ -n "$LABEL_ARG" ]]; then
  [[ "$LABEL_ARG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo '--label contains unsafe characters' >&2; exit 2; }
  LABEL="$LABEL_ARG"
else
  LABEL="legion_action_visual_lora_no_arrow_s1000_v1_$(date -u +%Y%m%dT%H%M%SZ)"
fi
STATE_DIR="$RUNTIME/action_visual_lora_pilot/$LABEL"
STATE_FILE="$STATE_DIR/state.env"
JOB_DIR="$RUNTIME/operator/jobs/$LABEL"
LAUNCH_LOCK="$STATE_DIR.launch.lock"
[[ ! -e "$STATE_DIR" && ! -e "$LAUNCH_LOCK" ]] || { echo 'duplicate or partial action-visual pilot submission exists; refusing to continue' >&2; exit 1; }
mkdir -p "$STATE_DIR" "$JOB_DIR" "$RUNTIME/operator/logs"
mkdir "$LAUNCH_LOCK"

ACTION_ONLY_CHECKPOINT="$(cd -- "$ACTION_ONLY_CHECKPOINT" && pwd -P)"
ACTION_ONLY_TRAINING_MANIFEST="$(cd -- "$(dirname -- "$ACTION_ONLY_TRAINING_MANIFEST")" && pwd -P)/$(basename -- "$ACTION_ONLY_TRAINING_MANIFEST")"
[[ "$(basename -- "$ACTION_ONLY_CHECKPOINT")" == pretrained_model && "$(basename -- "$(dirname -- "$ACTION_ONLY_CHECKPOINT")")" == 029190 ]] || { echo 'historical action-only checkpoint must be final checkpoint 029190/pretrained_model' >&2; exit 1; }
DATA_ROOT="$(cd -- "$DATA_ROOT" && pwd -P)"
LIBERO_DATA_DIR="$(cd -- "$LIBERO_DATA_DIR" && pwd -P)"
LIBERO_DIR="$(cd -- "$LIBERO_DIR" && pwd -P)"
BASE_POLICY="$(cd -- "$BASE_POLICY" && pwd -P)"
LIBERO_CONFIG="$(cd -- "$(dirname -- "$LIBERO_CONFIG")" && pwd -P)/$(basename -- "$LIBERO_CONFIG")"
TRAIN_ROOT="$SCRATCH_ROOT/runs/${LABEL}_candidate"
SMOKE_ROOT="$SCRATCH_ROOT/runs/${LABEL}_smoke"
EVAL_ROOT="$SCRATCH_ROOT/eval/${LABEL}"

setup_job_id=''; train_job_id=''; eval_job_id=''
setup_status=QUEUED; train_status=BLOCKED; eval_status=BLOCKED
setup_dependency=''; train_dependency='afterok:SETUP_JOB_ID'; eval_dependency='afterok:TRAIN_JOB_ID'
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
submit_id() { local raw="$1"; local id="${raw%%;*}"; [[ "$id" =~ ^[0-9]+$ ]] || { echo "invalid sbatch job id: $raw" >&2; exit 1; }; printf '%s\n' "$id"; }
write_state() {
  local tmp="${STATE_FILE}.tmp.$$"
  {
    printf 'label=%s\nstate_file=%s\nexpected_repo_commit=%s\nlibero_commit=%s\nbase_policy_revision=%s\n' "$LABEL" "$STATE_FILE" "$EXPECTED_REPO_COMMIT" "$LIBERO_COMMIT" "$BASE_POLICY_REVISION"
    printf 'policy_id=%s\ntraining_profile=%s\ntraining_steps=%s\ntraining_save_freq=%s\ntraining_batch_size=%s\ntraining_seed=%s\npeft_r=%s\n' "$POLICY_ID" "$PROFILE" "$SEALED_STEPS" "$SEALED_SAVE_FREQ" "$SEALED_BATCH_SIZE" "$SEALED_SEED" "$SEALED_PEFT_R"
    printf 'repo=%s\nruntime_root=%s\nscratch_root=%s\narchive_root=%s\ndata_root=%s\nlibero_data_dir=%s\nlibero_dir=%s\nbase_policy=%s\nlibero_config=%s\n' "$REPO" "$RUNTIME" "$SCRATCH_ROOT" "$ARCHIVE_ROOT" "$DATA_ROOT" "$LIBERO_DATA_DIR" "$LIBERO_DIR" "$BASE_POLICY" "$LIBERO_CONFIG"
    printf 'action_only_checkpoint=%s\naction_only_training_manifest=%s\nlegacy_evidence_bundle=%s\nsmoke_root=%s\ntraining_root=%s\nevaluation_root=%s\n' "$ACTION_ONLY_CHECKPOINT" "$ACTION_ONLY_TRAINING_MANIFEST" "$STATE_DIR/legacy_action_only_evidence_v1" "$SMOKE_ROOT" "$TRAIN_ROOT" "$EVAL_ROOT"
    printf 'setup_job_id=%s\ntrain_job_id=%s\neval_job_id=%s\nsetup_dependency=%s\ntrain_dependency=%s\neval_dependency=%s\nsetup_status=%s\ntrain_status=%s\neval_status=%s\n' "$setup_job_id" "$train_job_id" "$eval_job_id" "$setup_dependency" "$train_dependency" "$eval_dependency" "$setup_status" "$train_status" "$eval_status"
    printf 'setup_template_sha256=%s\ntrain_template_sha256=%s\neval_template_sha256=%s\n' "${setup_template_sha256:-}" "${train_template_sha256:-}" "${eval_template_sha256:-}"
  } > "$tmp"
  mv -f "$tmp" "$STATE_FILE"
}
write_state

# All three job files are materialized before the first sbatch call.  This
# makes the chain inspectable and ensures setup cannot silently switch inputs.
cat > "$JOB_DIR/setup.sbatch" <<EOF
#!/usr/bin/env bash
# Candidate policy validation, exact 2-step A40 smoke, and one clean inference.
#SBATCH --job-name=${LABEL}_setup
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
REPO='${REPO}'; RUNTIME='${RUNTIME}'; SCRATCH_ROOT='${SCRATCH_ROOT}'; ARCHIVE_ROOT='${ARCHIVE_ROOT}'; STATE_DIR='${STATE_DIR}'; STATE_FILE='${STATE_FILE}'; PYTHON='${PYTHON}'; EXPECTED_REPO_COMMIT='${EXPECTED_REPO_COMMIT}'; LIBERO_COMMIT='${LIBERO_COMMIT}'; BASE_POLICY_REVISION='${BASE_POLICY_REVISION}'; LABEL='${LABEL}'
DATA_ROOT='${DATA_ROOT}'; LIBERO_DATA_DIR='${LIBERO_DATA_DIR}'; LIBERO_DIR='${LIBERO_DIR}'; BASE_POLICY='${BASE_POLICY}'; LIBERO_CONFIG='${LIBERO_CONFIG}'; ACTION_ONLY_CHECKPOINT='${ACTION_ONLY_CHECKPOINT}'; ACTION_ONLY_TRAINING_MANIFEST='${ACTION_ONLY_TRAINING_MANIFEST}'; SMOKE_ROOT='${SMOKE_ROOT}'; LEGACY_EVIDENCE_BUNDLE="\$STATE_DIR/legacy_action_only_evidence_v1"; ARCHIVE_DIR="\$ARCHIVE_ROOT/setup/${LABEL}_\$SLURM_JOB_ID"; EVIDENCE="\$SCRATCH_ROOT/runs/${LABEL}_setup_\$SLURM_JOB_ID"
die() { echo "action-visual setup: \$*" >&2; exit 1; }
copy_tree() { local src="\$1" dst="\$2"; [[ -e "\$src" ]] || return 1; mkdir -p "\$dst"; if command -v rsync >/dev/null 2>&1; then rsync -a "\$src/" "\$dst/"; else cp -a "\$src/." "\$dst/"; fi; }
seal_tree() { \$PYTHON - "\$1" "\$2" <<'PY_SEAL'
import hashlib, pathlib, sys
root=pathlib.Path(sys.argv[1]).resolve(); mode=sys.argv[2]; inv=root/'inventory.sha256'; tree=root/'tree_sha256'
if mode=='build':
 rows=[(p.relative_to(root).as_posix(),hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in {'inventory.sha256','tree_sha256'}]
 if not rows: raise SystemExit('archive evidence is empty')
 payload=''.join(f'{d}  {n}\\n' for n,d in rows); inv.write_text(payload); tree.write_text(hashlib.sha256(payload.encode()).hexdigest()+'\\n')
else:
 payload=inv.read_text(); expected=tree.read_text().strip()
 if hashlib.sha256(payload.encode()).hexdigest()!=expected: raise SystemExit('archive tree hash mismatch')
 listed=set()
 for line in payload.splitlines():
  digest,name=line.split('  ',1); path=root/name
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise SystemExit(f'archive evidence mismatch: {name}')
  listed.add(name)
 actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name not in {'inventory.sha256','tree_sha256'}}
 if actual!=listed: raise SystemExit('archive inventory is incomplete')
PY_SEAL
}
finish() { local rc=\$? arc=0 tree=''; trap - EXIT; set +e; mkdir -p "\$EVIDENCE"; printf 'slurm_job_id=%s\\nslurm_job_name=%s\\nexit_code=%s\\n' "\${SLURM_JOB_ID:-}" "\${SLURM_JOB_NAME:-}" "\$rc" > "\$EVIDENCE/status.env"; mkdir -p "\$ARCHIVE_DIR"; copy_tree "\$EVIDENCE" "\$ARCHIVE_DIR" || arc=\$?; if [[ \$arc -eq 0 ]]; then seal_tree "\$ARCHIVE_DIR" build || arc=\$?; fi; if [[ \$arc -eq 0 ]]; then seal_tree "\$ARCHIVE_DIR" verify || arc=\$?; fi; if [[ \$arc -eq 0 ]]; then tree="\$(tr -d '[:space:]' < "\$ARCHIVE_DIR/tree_sha256")"; fi; printf 'archive_status=%s\\n' "\$([[ \$arc -eq 0 ]] && echo VERIFIED || echo FAILED)" > "\$EVIDENCE/archive_status.env"; printf 'setup_archive_status=%s\\nsetup_archive_tree_sha256=%s\\nsetup_status=%s\\n' "\$([[ \$arc -eq 0 ]] && echo VERIFIED || echo FAILED)" "\$tree" "\$([[ \$rc -eq 0 && \$arc -eq 0 ]] && echo OK || echo FAILED)" >> "\$STATE_FILE"; [[ \$rc -eq 0 && \$arc -eq 0 ]] || rc=90; exit \$rc; }; trap finish EXIT
[[ -d "\$REPO/.git" && "\$(git -C "\$REPO" rev-parse HEAD)" == "\$EXPECTED_REPO_COMMIT" ]] || die 'repository commit drift'; [[ -z "\$(git -C "\$REPO" status --porcelain --untracked-files=all)" ]] || die 'repository is dirty'; [[ -d "\$LIBERO_DIR/.git" && "\$(git -C "\$LIBERO_DIR" rev-parse HEAD)" == "\$LIBERO_COMMIT" ]] || die 'LIBERO checkout is not pinned'; [[ -z "\$(git -C "\$LIBERO_DIR" status --porcelain --untracked-files=no)" ]] || die 'LIBERO checkout is dirty'; [[ -f "\$LIBERO_CONFIG" && -d "\$BASE_POLICY" && -f "\$BASE_POLICY/config.json" && -f "\$BASE_POLICY/base_snapshot_manifest.json" ]] || die 'pinned LIBERO config/base snapshot is missing'; [[ -d "\$LIBERO_DATA_DIR" && -f "\$DATA_ROOT/sealed_lora_pair_manifest.json" && -f "\$DATA_ROOT/sealed_lora_pair_verified.json" ]] || die 'verified no-arrow dataset pair is missing'
module purge; module load miniforge/24.3.0-0; source "\$(conda info --base)/etc/profile.d/conda.sh"; export PATH="\$(dirname "\$PYTHON"):\$PATH" PYTHONPATH="\$REPO/vla_benchmarking:\${PYTHONPATH:-}" LIBERO_CONFIG_PATH="\$(dirname "\$LIBERO_CONFIG")" LIBERO_CONFIG="\$LIBERO_CONFIG" BASE_POLICY="\$BASE_POLICY" BASE_POLICY_REVISION="\$BASE_POLICY_REVISION" DATA_ROOT="\$DATA_ROOT" LIBERO_DATA_DIR="\$LIBERO_DATA_DIR" LIBERO_DIR="\$LIBERO_DIR" LIBERO_COMMIT="\$LIBERO_COMMIT" TRAINING_PROFILE=no_arrow_treatment PROFILE=no_arrow_treatment DEVICE=cuda RANDOMIZE_SCENES=1 VISUAL_CONDITION=none VISUAL_ARROWS=0
[[ ! -e "\$LEGACY_EVIDENCE_BUNDLE" ]] || die 'legacy action-only evidence bundle already exists'
\$PYTHON "\$REPO/vla_benchmarking/legacy_action_only_evidence.py" build --training-manifest "\$ACTION_ONLY_TRAINING_MANIFEST" --checkpoint "\$ACTION_ONLY_CHECKPOINT" --base-policy "\$BASE_POLICY" --data-root "\$DATA_ROOT" --output-dir "\$LEGACY_EVIDENCE_BUNDLE" || die 'legacy action-only evidence bundle build failed'
\$PYTHON "\$REPO/vla_benchmarking/legacy_action_only_evidence.py" validate --training-manifest "\$ACTION_ONLY_TRAINING_MANIFEST" --checkpoint "\$ACTION_ONLY_CHECKPOINT" --base-policy "\$BASE_POLICY" --data-root "\$DATA_ROOT" --output-dir "\$LEGACY_EVIDENCE_BUNDLE" || die 'legacy action-only evidence bundle validation failed'
\$PYTHON - "\$BASE_POLICY/base_snapshot_manifest.json" "\$BASE_POLICY_REVISION" <<'PY_BASE'
import json,sys
if json.load(open(sys.argv[1])).get('revision') != sys.argv[2]: raise SystemExit('base snapshot revision is not pinned')
PY_BASE
mkdir -p "\$EVIDENCE"
base_hash_before="\$(\$PYTHON - \"\$BASE_POLICY\" <<'PY_HASH'
import hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1]); rows=[(p.relative_to(root).as_posix(),hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob('*')) if p.is_file()]; print(hashlib.sha256(repr(rows).encode()).hexdigest())
PY_HASH
)"
\$PYTHON - "\$EVIDENCE/input_provenance.json" "\$LEGACY_EVIDENCE_BUNDLE" "\$DATA_ROOT/sealed_lora_pair_manifest.json" "\$DATA_ROOT/sealed_lora_pair_verified.json" "\$BASE_POLICY/base_snapshot_manifest.json" <<'PY_PROVENANCE'
import hashlib,json,pathlib,sys
out,bundle,*items=sys.argv[1:]
def digest(path):
 p=pathlib.Path(path).resolve()
 if p.is_dir():
  rows=[(q.relative_to(p).as_posix(),hashlib.sha256(q.read_bytes()).hexdigest()) for q in sorted(p.rglob('*')) if q.is_file()]
  value=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 else: value=hashlib.sha256(p.read_bytes()).hexdigest()
 return {'path':str(p),'sha256':value,'kind':'directory' if p.is_dir() else 'file'}
payload={'repo_commit': '${EXPECTED_REPO_COMMIT}', 'libero_commit': '${LIBERO_COMMIT}', 'base_policy_revision': '${BASE_POLICY_REVISION}', 'legacy_action_only_evidence_bundle': digest(bundle), 'files':[digest(p) for p in items]}
json.dump(payload,open(out,'w'),indent=2,sort_keys=True)
PY_PROVENANCE
copy_tree "\$LEGACY_EVIDENCE_BUNDLE" "\$EVIDENCE/legacy_action_only_evidence_v1" || die 'legacy action-only evidence bundle archival copy failed'
\$PYTHON "\$REPO/vla_benchmarking/adapter_audit.py" --generate-expected --base-policy "\$BASE_POLICY" --finetuning-policy action_visual_lora_v1 --output "\$EVIDENCE/expected_adapter_inventory.json" || die 'candidate live inventory generation failed'
\$PYTHON - "\$EVIDENCE/expected_adapter_inventory.json" <<'PY_INV'
import json,sys
d=json.load(open(sys.argv[1]))
if (d.get('finetuning_policy_id'),len(d.get('matched_module_names',[])),int(d.get('trainable_parameter_numel',-1)),len(d.get('trainable_parameter_names',[]))) != ('action_visual_lora_v1',78,1585152,156): raise SystemExit('candidate inventory is not 78 targets/156 tensors/1585152 parameters')
if d.get('no_full_weight_trainables') is not True or d.get('base_parameters_frozen') is not True: raise SystemExit('candidate inventory does not prove frozen base weights')
PY_INV
export TRAINING_RUNTIME_EVIDENCE="\$EVIDENCE/smoke_training_runtime.json"; env FINETUNING_POLICY_ID=action_visual_lora_v1 DATA_ROOT="\$DATA_ROOT" DATASET_ROOT="\$DATA_ROOT/control" LIBERO_DATA_DIR="\$LIBERO_DATA_DIR" LIBERO_DIR="\$LIBERO_DIR" LIBERO_COMMIT="\$LIBERO_COMMIT" BASE_POLICY="\$BASE_POLICY" BASE_POLICY_REVISION="\$BASE_POLICY_REVISION" PAIR_MANIFEST="\$DATA_ROOT/sealed_lora_pair_manifest.json" PAIR_SENTINEL="\$DATA_ROOT/sealed_lora_pair_verified.json" OUTPUT_DIR="\$SMOKE_ROOT" RUN_ROOT="\$SMOKE_ROOT" TRAINING_MODE=smoke BATCH_SIZE=32 SEED=1000 PEFT_R=16 DEVICE=cuda PYTHON="\$PYTHON" bash "\$REPO/vla_benchmarking/launch_lora_treatment.sh" smoke || die '2-step candidate smoke failed'
SMOKE_CHECKPOINT="\$SMOKE_ROOT/checkpoints/000002/pretrained_model"; [[ -s "\$SMOKE_CHECKPOINT/adapter_model.safetensors" && -f "\$SMOKE_CHECKPOINT/adapter_audit.json" && -f "\$SMOKE_ROOT/expected_adapter_inventory.json" ]] || die 'smoke checkpoint/audit is missing'
\$PYTHON - "\$EVIDENCE/smoke_training_runtime.json" <<'PY_RUNTIME'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('updates_observed') != 2 or d.get('all_losses_finite') is not True or d.get('all_grad_norms_finite') is not True: raise SystemExit('smoke training did not prove two finite updates')
if int(d.get('peak_cuda_allocated_bytes',0)) <= 0 or int(d.get('peak_cuda_reserved_bytes',0)) <= 0: raise SystemExit('smoke training did not record positive CUDA peaks')
PY_RUNTIME
\$PYTHON "\$REPO/vla_benchmarking/adapter_audit.py" --checkpoint "\$SMOKE_CHECKPOINT" --expected-inventory "\$SMOKE_ROOT/expected_adapter_inventory.json" --finetuning-policy action_visual_lora_v1 || die 'smoke checkpoint failed candidate audit'
\$PYTHON - "\$SMOKE_CHECKPOINT/adapter_model.safetensors" "\$EVIDENCE/expected_adapter_inventory.json" <<'PY_GRAD'
import json,sys
from safetensors import safe_open
weights=sys.argv[1]; inv=json.load(open(sys.argv[2])); names=set(inv['matched_module_names']); connector={n for n in names if '.connector.' in n}; vision={n for n in names if '.vision_model.encoder.layers.' in n}
seen_connector=set(); seen_vision=set()
with safe_open(weights,framework='pt',device='cpu') as h:
 for key in h.keys():
  if '.lora_B.' in key:
   module=key.split('.lora_B.',1)[0].removeprefix('base_model.model.');
   if module in connector and bool(h.get_tensor(key).detach().abs().max().item() > 0): seen_connector.add(module)
   if module in vision and bool(h.get_tensor(key).detach().abs().max().item() > 0): seen_vision.add(module)
if not seen_connector: raise SystemExit('smoke did not produce nonzero connector LoRA-B updates')
if not seen_vision: raise SystemExit('smoke did not produce nonzero late-vision LoRA-B updates')
json.dump({'connector_lora_b_nonzero': sorted(seen_connector), 'late_vision_lora_b_nonzero': sorted(seen_vision), 'gradient_proxy': 'nonzero_lora_B_after_two_steps'}, open(sys.argv[2].replace('expected_adapter_inventory.json','smoke_gradient_audit.json'),'w'), indent=2)
PY_GRAD
base_hash_after="\$(\$PYTHON - \"\$BASE_POLICY\" <<'PY_HASH2'
import hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1]); rows=[(p.relative_to(root).as_posix(),hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob('*')) if p.is_file()]; print(hashlib.sha256(repr(rows).encode()).hexdigest())
PY_HASH2
)"; [[ "\$base_hash_before" == "\$base_hash_after" ]] || die 'base policy files changed during smoke'; export EVIDENCE
export MODELS="\$SMOKE_CHECKPOINT" CONTEXT_MODE=standard CONTEXT_FORMAT=standard VISUAL_CONDITION=none VISUAL_ARROWS=0 TASK_IDS='[0]' N_EPISODES=1 BATCH_SIZE=1 SEED=1000 DEVICE=cuda OUTPUT_DIR="\$EVIDENCE/inference" N_ACTION_STEPS=checkpoint
\$PYTHON "\$REPO/vla_benchmarking/run_lerobot_eval_with_context.py" --eval.use_async_envs=false --output_dir="\$EVIDENCE/inference" --policy.path="\$SMOKE_CHECKPOINT" --env.task_ids='[0]' --env.camera_name=observation.images.image,observation.images.image2 --env.observation_height=256 --env.observation_width=256 || die 'one-action checkpoint reload/inference smoke failed'
[[ -s "\$EVIDENCE/inference/eval_info.json" ]] || die 'inference smoke did not emit eval_info.json'
copy_tree "\$SMOKE_ROOT" "\$EVIDENCE/smoke"
EOF

cat > "$JOB_DIR/train.sbatch" <<EOF
#!/usr/bin/env bash
# Full sealed no-arrow action+late-visual LoRA training; chained after setup.
#SBATCH --job-name=${LABEL}_train
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
REPO='${REPO}'; RUNTIME='${RUNTIME}'; SCRATCH_ROOT='${SCRATCH_ROOT}'; ARCHIVE_ROOT='${ARCHIVE_ROOT}'; STATE_FILE='${STATE_FILE}'; PYTHON='${PYTHON}'; EXPECTED_REPO_COMMIT='${EXPECTED_REPO_COMMIT}'; LIBERO_COMMIT='${LIBERO_COMMIT}'; BASE_POLICY_REVISION='${BASE_POLICY_REVISION}'; LABEL='${LABEL}'; DATA_ROOT='${DATA_ROOT}'; LIBERO_DATA_DIR='${LIBERO_DATA_DIR}'; LIBERO_DIR='${LIBERO_DIR}'; BASE_POLICY='${BASE_POLICY}'; LIBERO_CONFIG='${LIBERO_CONFIG}'; TRAIN_ROOT='${TRAIN_ROOT}'; ARCHIVE_DIR="\$ARCHIVE_ROOT/train/${LABEL}_\$SLURM_JOB_ID"; EVIDENCE="\$SCRATCH_ROOT/runs/${LABEL}_train_\$SLURM_JOB_ID"
die() { echo "action-visual train: \$*" >&2; exit 1; }; copy_tree() { mkdir -p "\$2"; if command -v rsync >/dev/null 2>&1; then rsync -a "\$1/" "\$2/"; else cp -a "\$1/." "\$2/"; fi; }; seal_tree() { \$PYTHON - "\$1" "\$2" <<'PY_SEAL_TRAIN'
import hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve(); mode=sys.argv[2]; inv=root/'inventory.sha256'; tree=root/'tree_sha256'
if mode=='build':
 rows=[(p.relative_to(root).as_posix(),hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in {'inventory.sha256','tree_sha256'}]
 if not rows: raise SystemExit('train archive is empty')
 payload=''.join(f'{d}  {n}\\n' for n,d in rows); inv.write_text(payload); tree.write_text(hashlib.sha256(payload.encode()).hexdigest()+'\\n')
else:
 payload=inv.read_text(); listed=set()
 if hashlib.sha256(payload.encode()).hexdigest()!=tree.read_text().strip(): raise SystemExit('train archive tree hash mismatch')
 for line in payload.splitlines():
  digest,name=line.split('  ',1); path=root/name
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise SystemExit(f'train archive mismatch: {name}')
  listed.add(name)
 actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name not in {'inventory.sha256','tree_sha256'}}
 if actual!=listed: raise SystemExit('train archive inventory is incomplete')
PY_SEAL_TRAIN
}; finish() { local rc=\$? arc=0 tree=''; trap - EXIT; set +e; mkdir -p "\$EVIDENCE" "\$ARCHIVE_DIR"; printf 'slurm_job_id=%s\\nslurm_job_name=%s\\nexit_code=%s\\n' "\${SLURM_JOB_ID:-}" "\${SLURM_JOB_NAME:-}" "\$rc" > "\$EVIDENCE/status.env"; copy_tree "\$EVIDENCE" "\$ARCHIVE_DIR/evidence" || arc=\$?; if [[ \$arc -eq 0 && -d "\$TRAIN_ROOT" ]]; then copy_tree "\$TRAIN_ROOT" "\$ARCHIVE_DIR/run" || arc=\$?; fi; if [[ \$arc -eq 0 ]]; then seal_tree "\$ARCHIVE_DIR" build || arc=\$?; fi; if [[ \$arc -eq 0 ]]; then seal_tree "\$ARCHIVE_DIR" verify || arc=\$?; fi; [[ \$arc -eq 0 ]] && tree="\$(tr -d '[:space:]' < "\$ARCHIVE_DIR/tree_sha256")"; printf 'train_archive_status=%s\\ntrain_archive_tree_sha256=%s\\ntrain_status=%s\\n' "\$([[ \$arc -eq 0 ]] && echo VERIFIED || echo FAILED)" "\$tree" "\$([[ \$rc -eq 0 && \$arc -eq 0 ]] && echo OK || echo FAILED)" >> "\$STATE_FILE"; [[ \$rc -eq 0 && \$arc -eq 0 ]] || rc=90; exit \$rc; }; trap finish EXIT
[[ -z "\$(git -C "\$REPO" status --porcelain --untracked-files=all)" ]] || die 'repository is dirty'
[[ -d "\$REPO/.git" && "\$(git -C "\$REPO" rev-parse HEAD)" == "\$EXPECTED_REPO_COMMIT" ]] || die 'repository commit drift'; [[ ! -e "\$TRAIN_ROOT" ]] || die 'candidate training output already exists'; mkdir -p "\$EVIDENCE"; module purge; module load miniforge/24.3.0-0; source "\$(conda info --base)/etc/profile.d/conda.sh"; export PATH="\$(dirname "\$PYTHON"):\$PATH" PYTHONPATH="\$REPO/vla_benchmarking:\${PYTHONPATH:-}" LIBERO_CONFIG_PATH="\$(dirname "\$LIBERO_CONFIG")" LIBERO_CONFIG="\$LIBERO_CONFIG" BASE_POLICY="\$BASE_POLICY" BASE_POLICY_REVISION="\$BASE_POLICY_REVISION" DATA_ROOT="\$DATA_ROOT" LIBERO_DATA_DIR="\$LIBERO_DATA_DIR" LIBERO_DIR="\$LIBERO_DIR" LIBERO_COMMIT="\$LIBERO_COMMIT" TRAINING_PROFILE=no_arrow_treatment PROFILE=no_arrow_treatment FINETUNING_POLICY_ID=action_visual_lora_v1 RUN_ROOT="\$TRAIN_ROOT" OUTPUT_DIR="\$TRAIN_ROOT" BATCH_SIZE=32 SEED=1000 PEFT_R=16 DEVICE=cuda TRAINING_MODE=full RESUME=false TRAINING_RUNTIME_EVIDENCE="\$EVIDENCE/full_training_runtime.json"; bash "\$REPO/vla_benchmarking/launch_lora_treatment.sh" full; [[ -s "\$TRAIN_ROOT/training_manifest.json" && -s "\$TRAIN_ROOT/checkpoints/029190/pretrained_model/adapter_model.safetensors" ]] || die 'sealed candidate training artifacts are missing'; \$PYTHON - "\$TRAIN_ROOT/training_manifest.json" <<'PY_TRAIN'
import json,sys
d=json.load(open(sys.argv[1])); f=d.get('flags',{})
if d.get('finetuning_policy_id')!='action_visual_lora_v1' or d.get('training_variant')!='no_arrow_treatment' or d.get('trained_on_visual_condition')!='no_arrows' or any(int(f.get(k,-1))!=v for k,v in {'steps':29190,'save_freq':1946,'batch_size':32,'seed':1000,'peft_r':16}.items()): raise SystemExit('candidate training manifest is not sealed')
PY_TRAIN
\$PYTHON - "\$EVIDENCE/full_training_runtime.json" <<'PY_RUNTIME_TRAIN'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('updates_observed') != 29190 or d.get('all_losses_finite') is not True or d.get('all_grad_norms_finite') is not True: raise SystemExit('full candidate training did not prove 29190 finite updates')
if int(d.get('peak_cuda_allocated_bytes',0)) <= 0 or int(d.get('peak_cuda_reserved_bytes',0)) <= 0: raise SystemExit('full candidate training did not record positive CUDA peaks')
PY_RUNTIME_TRAIN
EOF

cat > "$JOB_DIR/eval.sbatch" <<EOF
#!/usr/bin/env bash
# Matched clean/no-arrow evaluation; exactly canonical cells, chained after train.
#SBATCH --job-name=${LABEL}_eval
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
REPO='${REPO}'; RUNTIME='${RUNTIME}'; SCRATCH_ROOT='${SCRATCH_ROOT}'; ARCHIVE_ROOT='${ARCHIVE_ROOT}'; STATE_FILE='${STATE_FILE}'; PYTHON='${PYTHON}'; EXPECTED_REPO_COMMIT='${EXPECTED_REPO_COMMIT}'; LIBERO_COMMIT='${LIBERO_COMMIT}'; BASE_POLICY='${BASE_POLICY}'; LIBERO_CONFIG='${LIBERO_CONFIG}'; LIBERO_DATA_DIR='${LIBERO_DATA_DIR}'; LIBERO_DIR='${LIBERO_DIR}'; LABEL='${LABEL}'; ACTION_ONLY_CHECKPOINT='${ACTION_ONLY_CHECKPOINT}'; ACTION_ONLY_TRAINING_MANIFEST='${ACTION_ONLY_TRAINING_MANIFEST}'; TRAIN_ROOT='${TRAIN_ROOT}'; EVAL_ROOT='${EVAL_ROOT}'; TRAIN_JOB_ID='__TRAIN_JOB_ID__'; ARCHIVE_DIR="\$ARCHIVE_ROOT/eval/${LABEL}_\$SLURM_JOB_ID"; EVIDENCE="\$SCRATCH_ROOT/eval/${LABEL}_\$SLURM_JOB_ID"
die() { echo "action-visual eval: \$*" >&2; exit 1; }; copy_tree() { mkdir -p "\$2"; if command -v rsync >/dev/null 2>&1; then rsync -a "\$1/" "\$2/"; else cp -a "\$1/." "\$2/"; fi; }; seal_tree() { \$PYTHON - "\$1" "\$2" <<'PY_SEAL_EVAL'
import hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve(); mode=sys.argv[2]; inv=root/'inventory.sha256'; tree=root/'tree_sha256'
if mode=='build':
 rows=[(p.relative_to(root).as_posix(),hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in {'inventory.sha256','tree_sha256'}]
 if not rows: raise SystemExit('eval archive is empty')
 payload=''.join(f'{d}  {n}\\n' for n,d in rows); inv.write_text(payload); tree.write_text(hashlib.sha256(payload.encode()).hexdigest()+'\\n')
else:
 payload=inv.read_text(); listed=set()
 if hashlib.sha256(payload.encode()).hexdigest()!=tree.read_text().strip(): raise SystemExit('eval archive tree hash mismatch')
 for line in payload.splitlines():
  digest,name=line.split('  ',1); path=root/name
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise SystemExit(f'eval archive mismatch: {name}')
  listed.add(name)
 actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name not in {'inventory.sha256','tree_sha256'}}
 if actual!=listed: raise SystemExit('eval archive inventory is incomplete')
PY_SEAL_EVAL
}; finish() { local rc=\$? arc=0 tree=''; trap - EXIT; set +e; mkdir -p "\$EVIDENCE" "\$ARCHIVE_DIR"; printf 'slurm_job_id=%s\\nslurm_job_name=%s\\nslurm_job_dependency=afterok:%s\\nexit_code=%s\\n' "\${SLURM_JOB_ID:-}" "\${SLURM_JOB_NAME:-}" "\$TRAIN_JOB_ID" "\$rc" > "\$EVIDENCE/status.env"; copy_tree "\$EVIDENCE" "\$ARCHIVE_DIR/evidence" || arc=\$?; if [[ \$arc -eq 0 && -d "\$EVAL_ROOT" ]]; then copy_tree "\$EVAL_ROOT" "\$ARCHIVE_DIR/results" || arc=\$?; fi; if [[ \$arc -eq 0 ]]; then seal_tree "\$ARCHIVE_DIR" build || arc=\$?; fi; if [[ \$arc -eq 0 ]]; then seal_tree "\$ARCHIVE_DIR" verify || arc=\$?; fi; [[ \$arc -eq 0 ]] && tree="\$(tr -d '[:space:]' < "\$ARCHIVE_DIR/tree_sha256")"; printf 'eval_archive_status=%s\\neval_archive_tree_sha256=%s\\neval_status=%s\\n' "\$([[ \$arc -eq 0 ]] && echo VERIFIED || echo FAILED)" "\$tree" "\$([[ \$rc -eq 0 && \$arc -eq 0 ]] && echo OK || echo FAILED)" >> "\$STATE_FILE"; [[ \$rc -eq 0 && \$arc -eq 0 ]] || rc=90; exit \$rc; }; trap finish EXIT
STATE_DIR='${STATE_DIR}'; LEGACY_EVIDENCE_BUNDLE="\$STATE_DIR/legacy_action_only_evidence_v1"
export LIBERO_DIR="\$LIBERO_DIR"
DATA_ROOT='${DATA_ROOT}'; BASE_POLICY_REVISION='${BASE_POLICY_REVISION}'
[[ -z "\$(git -C "\$REPO" status --porcelain --untracked-files=all)" ]] || die 'repository is dirty'
[[ -d "\$LIBERO_DIR/.git" && "\$(git -C "\$LIBERO_DIR" rev-parse HEAD)" == "\$LIBERO_COMMIT" ]] || die 'LIBERO checkout is not pinned'; [[ -z "\$(git -C "\$LIBERO_DIR" status --porcelain --untracked-files=no)" ]] || die 'LIBERO checkout is dirty'
[[ -d "\$LEGACY_EVIDENCE_BUNDLE" ]] || die 'legacy evidence bundle is missing'
[[ -d "\$REPO/.git" && "\$(git -C "\$REPO" rev-parse HEAD)" == "\$EXPECTED_REPO_COMMIT" ]] || die 'repository commit drift'; [[ -s "\$TRAIN_ROOT/training_manifest.json" && -s "\$TRAIN_ROOT/checkpoints/029190/pretrained_model/adapter_model.safetensors" ]] || die 'candidate final checkpoint is missing'; module purge; module load miniforge/24.3.0-0; source "\$(conda info --base)/etc/profile.d/conda.sh"; export PATH="\$(dirname "\$PYTHON"):\$PATH" PYTHONPATH="\$REPO/vla_benchmarking:\${PYTHONPATH:-}" LIBERO_CONFIG_PATH="\$(dirname "\$LIBERO_CONFIG")" LIBERO_CONFIG="\$LIBERO_CONFIG" BASE_POLICY="\$BASE_POLICY" LIBERO_DATA_DIR="\$LIBERO_DATA_DIR" LIBERO_DIR="\$LIBERO_DIR" LIBERO_COMMIT="\$LIBERO_COMMIT" TRAINING_PROFILE=no_arrow_treatment PROFILE=no_arrow_treatment CONTEXT_MODE=standard CONTEXT_FORMAT=standard VISUAL_CONDITION=none VISUAL_ARROWS=0 RANDOMIZE_SCENES=1 DEVICE=cuda; mkdir -p "\$EVAL_ROOT"; \$PYTHON "\$REPO/vla_benchmarking/run_lora_policy_pair_eval.py" --action-only-checkpoint "\$ACTION_ONLY_CHECKPOINT" --action-visual-checkpoint "\$TRAIN_ROOT/checkpoints/029190/pretrained_model" --action-only-training-manifest "\$ACTION_ONLY_TRAINING_MANIFEST" --action-visual-training-manifest "\$TRAIN_ROOT/training_manifest.json" --action-only-legacy-evidence-bundle "\$LEGACY_EVIDENCE_BUNDLE" --training-data-root "\$DATA_ROOT" --output-root "\$EVAL_ROOT" --episodes 10 --batch-size 1 --device cuda --no-videos; [[ -s "\$EVAL_ROOT/action_visual_lora_no_arrow_pair_manifest.json" && -s "\$EVAL_ROOT/action_visual_lora_no_arrow_pair_summary.csv" && -s "\$EVAL_ROOT/episode_results.jsonl" ]] || die 'matched per-task evaluation outputs are missing'; \$PYTHON - "\$EVAL_ROOT/action_visual_lora_no_arrow_pair_manifest.json" <<'PY_EVAL'
import json,sys
d=json.load(open(sys.argv[1])); cells=d.get('cells',[])
if [c.get('cell_id') for c in cells] != ['historical_action_only_lora_v1_no_arrows','action_visual_lora_v1_no_arrows']: raise SystemExit('canonical two-cell order is missing')
if cells[0].get('policy_id') != 'action_only_lora_v1' or cells[0].get('live_arrows') is not False or cells[0].get('visual_condition') != 'none': raise SystemExit('historical action-only retrospective contract is not sealed')
if cells[1].get('policy_id') != 'action_visual_lora_v1' or cells[1].get('live_arrows') is not False or cells[1].get('visual_condition') != 'none': raise SystemExit('candidate action-visual retrospective contract is not sealed')
if d.get('tasks') != list(range(10)) or d.get('episodes') != 10 or d.get('batch_size') != 1 or d.get('seeds') != [1000] or d.get('visual_condition') != 'none': raise SystemExit('matched clean evaluation contract is not sealed')
PY_EVAL
EOF

for generated_job in "$JOB_DIR/setup.sbatch" "$JOB_DIR/train.sbatch" "$JOB_DIR/eval.sbatch"; do
  syntax_output="$(bash -n "$generated_job" 2>&1)" || { printf '%s\n' "$syntax_output" >&2; echo "generated SLURM script has invalid syntax: $generated_job" >&2; exit 1; }
  [[ -z "$syntax_output" ]] || { printf '%s\n' "$syntax_output" >&2; echo "generated SLURM script emitted a syntax warning: $generated_job" >&2; exit 1; }
done
chmod 700 "$JOB_DIR"/*.sbatch
setup_template_sha256="$(sha256_file "$JOB_DIR/setup.sbatch")"; train_template_sha256="$(sha256_file "$JOB_DIR/train.sbatch")"; eval_template_sha256="$(sha256_file "$JOB_DIR/eval.sbatch")"; write_state
setup_raw="$(sbatch --parsable "$JOB_DIR/setup.sbatch")"; setup_job_id="$(submit_id "$setup_raw")"; setup_dependency=''; setup_status=QUEUED; train_dependency="afterok:$setup_job_id"; eval_dependency='afterok:TRAIN_JOB_ID'; write_state
rendered_eval="$JOB_DIR/eval.${setup_job_id}.sbatch"; sed "s/__TRAIN_JOB_ID__/TRAIN_JOB_ID_PLACEHOLDER/g" "$JOB_DIR/eval.sbatch" > "$rendered_eval"; train_raw="$(sbatch --parsable --dependency="afterok:$setup_job_id" "$JOB_DIR/train.sbatch")"; train_job_id="$(submit_id "$train_raw")"; sed -i "s/TRAIN_JOB_ID_PLACEHOLDER/$train_job_id/g" "$rendered_eval"; eval_dependency="afterok:$train_job_id"; train_status=QUEUED; write_state
eval_raw="$(sbatch --parsable --dependency="afterok:$train_job_id" "$rendered_eval")"; eval_job_id="$(submit_id "$eval_raw")"; eval_status=QUEUED; write_state
printf 'queued action_visual_lora_v1 no-arrow chain: setup=%s train=%s eval=%s state=%s\n' "$setup_job_id" "$train_job_id" "$eval_job_id" "$STATE_FILE"
