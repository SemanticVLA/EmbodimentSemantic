#!/usr/bin/env bash
set -euo pipefail

# Train exactly one adapter for the selected sealed LIBERO profile. The
# no_arrow_treatment profile intentionally uses the arrow-free control frames
# as an independent finetuning ablation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VLA_ROOT="$REPO_ROOT/vla_benchmarking"
PROFILE="${TRAINING_PROFILE:-treatment}"
MODE="${1:-full}"
PYTHON_VALUE="${PYTHON:-python}"
case "$PROFILE" in
  treatment)
    PAIR_MANIFEST_NAME="sealed_lora_pair_manifest.json"
    PAIR_SENTINEL_NAME="sealed_lora_pair_verified.json"
    PAIR_KIND="sealed_lora_control_treatment"
    DATASET_VARIANT="treatment"
    EXPECTED_DATASET_VARIANT="treatment"
    DATASET_REPO_ID="local/libero_spatial_treatment"
    PREFLIGHT_VARIANT="treatment"
    DEFAULT_DATA_ROOT="$VLA_ROOT/lora_datasets"
    EXPERIMENT_NAME="smolvla_lora_treatment_training"
    TRAINED_VISUAL_CONDITION="arrows"
    ADAPTER_KEY="treatment_adapter"
    ;;
  no_arrow_treatment)
    PAIR_MANIFEST_NAME="sealed_lora_pair_manifest.json"
    PAIR_SENTINEL_NAME="sealed_lora_pair_verified.json"
    PAIR_KIND="sealed_lora_control_treatment"
    DATASET_VARIANT="control"
    EXPECTED_DATASET_VARIANT="control"
    DATASET_REPO_ID="local/libero_spatial_control"
    PREFLIGHT_VARIANT="no_arrow_treatment"
    DEFAULT_DATA_ROOT="$VLA_ROOT/lora_datasets"
    EXPERIMENT_NAME="smolvla_lora_no_arrow_treatment_training"
    TRAINED_VISUAL_CONDITION="no_arrows"
    ADAPTER_KEY="no_arrow_treatment_adapter"
    ;;
  graph_treatment)
    PAIR_MANIFEST_NAME="sealed_lora_graph_pair_manifest.json"
    PAIR_SENTINEL_NAME="sealed_lora_graph_pair_verified.json"
    PAIR_KIND="sealed_lora_graph_treatment_arrow_graph_treatment"
    DATASET_VARIANT="graph_treatment"
    EXPECTED_DATASET_VARIANT="graph_treatment"
    DATASET_REPO_ID="local/libero_spatial_graph_treatment"
    PREFLIGHT_VARIANT="graph_treatment"
    DEFAULT_DATA_ROOT="$VLA_ROOT/lora_datasets"
    EXPERIMENT_NAME="smolvla_lora_graph_treatment_training"
    TRAINED_VISUAL_CONDITION="no_arrows"
    ADAPTER_KEY="graph_treatment_adapter"
    ;;
  arrow_graph_treatment)
    PAIR_MANIFEST_NAME="sealed_lora_graph_pair_manifest.json"
    PAIR_SENTINEL_NAME="sealed_lora_graph_pair_verified.json"
    PAIR_KIND="sealed_lora_graph_treatment_arrow_graph_treatment"
    DATASET_VARIANT="arrow_graph_treatment"
    EXPECTED_DATASET_VARIANT="arrow_graph_treatment"
    DATASET_REPO_ID="local/libero_spatial_arrow_graph_treatment"
    PREFLIGHT_VARIANT="arrow_graph_treatment"
    DEFAULT_DATA_ROOT="$VLA_ROOT/lora_datasets"
    EXPERIMENT_NAME="smolvla_lora_arrow_graph_treatment_training"
    TRAINED_VISUAL_CONDITION="arrows"
    ADAPTER_KEY="arrow_graph_treatment_adapter"
    ;;
  *) echo "unknown TRAINING_PROFILE: $PROFILE" >&2; exit 2 ;;
esac
case "$MODE" in dry|smoke|full|resume) ;; *) echo "usage: $0 [profile] <dry|smoke|full|resume>" >&2; exit 2 ;; esac
# Sealed schedule: ignore ambient EPOCHS/STEPS/SAVE_FREQ values.
SEALED_EPOCHS=15
SEALED_UPDATES_PER_EPOCH=1946
SEALED_STEPS=$((SEALED_EPOCHS * SEALED_UPDATES_PER_EPOCH))
case "$MODE" in
  dry|full|resume) STEPS_VALUE="$SEALED_STEPS"; SAVE_FREQ_VALUE="$SEALED_UPDATES_PER_EPOCH" ;;
  smoke) STEPS_VALUE=2; SAVE_FREQ_VALUE=2 ;;
esac

RUN_ROOT="${RUN_ROOT:-${TREATMENT_RUN_ROOT:-$SCRIPT_DIR/lora_runs/${PROFILE}_$(date +%Y_%m_%d_%H_%M_%S)}}"
DATA_ROOT_VALUE="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
PAIR_MANIFEST_VALUE="${PAIR_MANIFEST:-$DATA_ROOT_VALUE/$PAIR_MANIFEST_NAME}"
PAIR_SENTINEL_VALUE="${PAIR_SENTINEL:-$DATA_ROOT_VALUE/$PAIR_SENTINEL_NAME}"
LIBERO_DATA_DIR_VALUE="${LIBERO_DATA_DIR:-$REPO_ROOT/vlm_benchmarking/data/libero_spatial_v5}"
BASE_POLICY_REVISION_VALUE="${BASE_POLICY_REVISION:-6721902bc4d61e50a3bfdb11dfb4cb626f05d102}"
BASE_POLICY_VALUE="${BASE_POLICY:-$VLA_ROOT/base_models/smolvla_libero-$BASE_POLICY_REVISION_VALUE}"
LIBERO_DIR_VALUE="${LIBERO_DIR:-$VLA_ROOT/LIBERO}"
LIBERO_COMMIT_VALUE="8f1084e3132a39270c3a13ebe37270a43ece2a01"
RESUME_VALUE="${RESUME:-false}"
RESUME_CONFIG_PATH_VALUE="${RESUME_CONFIG_PATH:-}"

if [[ "$PROFILE" == graph_treatment || "$PROFILE" == arrow_graph_treatment ]]; then
  # Graph cells are sealed: ambient values cannot change their training
  # condition when this operator is invoked directly.
  PEFT_R=16
  BATCH_SIZE=32
  SEED=1000
fi

# ``policy.path`` makes LeRobot load the serialized policy_preprocessor.json.
# Prepare the graph-only 96-token snapshot before any preflight/training-plan
# logic; the historical base snapshot remains untouched at 48 tokens.
if [[ "$PROFILE" == graph_treatment || "$PROFILE" == arrow_graph_treatment ]]; then
  GRAPH_BASE_POLICY_VALUE="${GRAPH_BASE_POLICY:-$VLA_ROOT/base_models/smolvla_libero-$BASE_POLICY_REVISION_VALUE-graph96}"
  if [[ "$BASE_POLICY_VALUE" == "$VLA_ROOT/base_models/smolvla_libero-$BASE_POLICY_REVISION_VALUE" ]]; then
    "$PYTHON_VALUE" "$VLA_ROOT/tools/prompt_audit.py" --prepare-graph-policy "$BASE_POLICY_VALUE" "$GRAPH_BASE_POLICY_VALUE"
    BASE_POLICY_VALUE="$GRAPH_BASE_POLICY_VALUE"
  fi
fi
COMMON_ENV=("PAIR_MANIFEST=$PAIR_MANIFEST_VALUE"
  "PAIR_SENTINEL=$PAIR_SENTINEL_VALUE"
  "DATA_ROOT=$DATA_ROOT_VALUE"
  "LIBERO_DATA_DIR=$LIBERO_DATA_DIR_VALUE"
  "LIBERO_DIR=$LIBERO_DIR_VALUE"
  "LIBERO_COMMIT=$LIBERO_COMMIT_VALUE"
  "PYTHON=$PYTHON_VALUE"
  "BASE_POLICY_REVISION=$BASE_POLICY_REVISION_VALUE"
  "BASE_POLICY=$BASE_POLICY_VALUE"
  "PEFT_R=${PEFT_R:-16}" "TRAINING_MODE=$MODE" "STEPS_VALUE=$STEPS_VALUE" "SAVE_FREQ_VALUE=$SAVE_FREQ_VALUE"
  "STEPS=$STEPS_VALUE" "SAVE_FREQ=$SAVE_FREQ_VALUE" "BATCH_SIZE=${BATCH_SIZE:-32}"
  "SEED=${SEED:-1000}" "DEVICE=${DEVICE:-cuda}"
  "RESUME=${RESUME_VALUE:-false}" "RESUME_CONFIG_PATH=${RESUME_CONFIG_PATH_VALUE:-}")

OUTPUT_VALUE="$RUN_ROOT"
env_args=("${COMMON_ENV[@]}" "DATASET_ROOT=$DATA_ROOT_VALUE/$DATASET_VARIANT" "OUTPUT_DIR=$OUTPUT_VALUE")
RUN_PLAN_PENDING="${RUN_ROOT}.training_plan.pending.json"
env "${env_args[@]}" bash "$SCRIPT_DIR/lambda_preflight.sh" "$PREFLIGHT_VARIANT"
if [[ "$MODE" == dry ]]; then
  printf 'command: env'
  printf ' %q' "${env_args[@]}"
  printf ' bash %q %q\n' "$SCRIPT_DIR/train_lora.sh" "$PROFILE"
  echo "dry run complete: preflight passed; no run directory or plan was written"
  exit 0
fi

if [[ "$MODE" == resume ]]; then
  RESUME_VALUE=true
  [[ -d "$RUN_ROOT" ]] || { echo "resume requires existing run directory: $RUN_ROOT" >&2; exit 1; }
  [[ ! -f "$RUN_ROOT/training_manifest.json" ]] || { echo "completed run has training_manifest.json; refusing resume" >&2; exit 1; }
  PLAN_SOURCE="$RUN_ROOT/training_plan.json"
  if [[ ! -f "$RUN_ROOT/training_plan.json" ]]; then
    [[ -f "$RUN_PLAN_PENDING" ]] || { echo "resume requires training_plan.json or a pending plan sidecar" >&2; exit 1; }
    PLAN_SOURCE="$RUN_PLAN_PENDING"
  fi
  if [[ -z "$RESUME_CONFIG_PATH_VALUE" ]]; then
    RESUME_CONFIG_PATH_VALUE="$(find "$RUN_ROOT/checkpoints" -type f -name train_config.json -print 2>/dev/null | sort | tail -n 1)"
  fi
  [[ -f "$RESUME_CONFIG_PATH_VALUE" ]] || { echo "resume checkpoint train_config.json is required" >&2; exit 1; }
  [[ "$(basename "$RESUME_CONFIG_PATH_VALUE")" == "train_config.json" ]] || { echo "resume config must be checkpoints/<step>/pretrained_model/train_config.json" >&2; exit 1; }
  resume_checkpoint_dir="$(cd "$(dirname "$RESUME_CONFIG_PATH_VALUE")" && pwd -P)"
  [[ "$(basename "$resume_checkpoint_dir")" == "pretrained_model" ]] || { echo "resume config must be under a pretrained_model checkpoint directory" >&2; exit 1; }
  resume_checkpoint_root="$(cd "$resume_checkpoint_dir/.." && pwd -P)"
  resume_checkpoint_name="$(basename "$resume_checkpoint_root")"
  [[ "$resume_checkpoint_name" =~ ^[0-9]+$ ]] || { echo "resume checkpoint directory must be numeric" >&2; exit 1; }
  resume_checkpoint_step=$((10#$resume_checkpoint_name))
  (( resume_checkpoint_step > 0 && resume_checkpoint_step < 29190 && resume_checkpoint_step % 1946 == 0 )) || {
    echo "resume checkpoint must be an ordinary saved step in (0,29190) on the 1946-step schedule" >&2; exit 1;
  }
  run_root_abs="$(cd "$RUN_ROOT" && pwd -P)"
  expected_checkpoint_root="$run_root_abs/checkpoints/$resume_checkpoint_name"
  [[ "$resume_checkpoint_root" == "$expected_checkpoint_root" ]] || {
    echo "resume checkpoint must resolve exactly to $expected_checkpoint_root" >&2; exit 1;
  }
  config_abs="$(cd "$(dirname "$RESUME_CONFIG_PATH_VALUE")" && pwd -P)/$(basename "$RESUME_CONFIG_PATH_VALUE")"
  case "$config_abs" in
    "$run_root_abs"/*) ;;
    *) echo "resume config must be contained by run directory" >&2; exit 1 ;;
  esac
  [[ -d "$resume_checkpoint_root/training_state" ]] || { echo "resume checkpoint lacks training_state required for optimizer/scheduler/RNG restore" >&2; exit 1; }
  [[ -n "$(find "$resume_checkpoint_root/training_state" -type f -print -quit 2>/dev/null)" ]] || { echo "resume checkpoint training_state is empty" >&2; exit 1; }
  "$PYTHON_VALUE" - "$resume_checkpoint_root/training_state" <<'PY'
import pathlib, re, sys
root = pathlib.Path(sys.argv[1]).resolve()
files = [path for path in root.rglob('*') if path.is_file()]
if not files or any(path.is_symlink() for path in files):
    raise SystemExit('resume training_state must contain regular files only')
names = {path.name.lower() for path in files}
required = {
    'optimizer': any(re.search(r'(^|[._-])optimizer([._-]|$)', name) for name in names),
    'scheduler': any(re.search(r'(^|[._-])scheduler([._-]|$)', name) for name in names),
    'rng': any(('random_state' in name or 'rng' in name) for name in names),
}
if not all(required.values()):
    missing = ', '.join(key for key, present in required.items() if not present)
    raise SystemExit(f'resume training_state is missing required state artifacts: {missing}')
PY
  expected_inventory_path="$RUN_ROOT/expected_adapter_inventory.json"
  expected_inventory_pending="${RUN_ROOT}.expected_adapter_inventory.pending.json"
  if [[ -f "$expected_inventory_path" && -f "$expected_inventory_pending" ]]; then
    "$PYTHON_VALUE" - "$expected_inventory_path" "$expected_inventory_pending" <<'PY'
import sys
from pathlib import Path
if Path(sys.argv[1]).read_bytes() != Path(sys.argv[2]).read_bytes():
    raise SystemExit('final and pending expected LoRA inventories differ')
PY
  fi
  if [[ ! -f "$expected_inventory_path" && -f "$expected_inventory_pending" ]]; then
    PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_VALUE" - "$PLAN_SOURCE" "$expected_inventory_pending" "$expected_inventory_path" "$BASE_POLICY_VALUE" "$BASE_POLICY_REVISION_VALUE" <<'PY'
import json, os, pathlib, sys
plan_path = pathlib.Path(sys.argv[1]); pending_path = pathlib.Path(sys.argv[2]); final_path = pathlib.Path(sys.argv[3])
base_policy = pathlib.Path(sys.argv[4]); revision = sys.argv[5]
plan = json.loads(plan_path.read_text(encoding='utf-8'))
if pathlib.Path(plan.get('base_policy', '')).expanduser().resolve() != base_policy.expanduser().resolve() or plan.get('base_policy_revision') != revision:
    raise SystemExit('resume plan does not authenticate the pending expected LoRA inventory base')
flags = plan.get('flags', {})
if any(int(flags.get(key, -1)) != value for key, value in {'seed': 1000, 'peft_r': 16, 'batch_size': 32}.items()):
    raise SystemExit('resume plan LoRA constants are not sealed')
try:
    from adapter_audit import load_expected_inventory
    inventory = load_expected_inventory(pending_path)
except Exception as exc:
    raise SystemExit(f'pending expected LoRA inventory is invalid: {exc}')
if inventory.get('base_policy') != str(pathlib.Path(base_policy).expanduser().resolve()):
    raise SystemExit('pending expected LoRA inventory belongs to a different base policy')
if inventory.get('base_policy_revision') != revision:
    raise SystemExit('pending expected LoRA inventory belongs to a different base revision')
if final_path.exists():
    if final_path.read_bytes() != pending_path.read_bytes():
        raise SystemExit('final and pending expected LoRA inventories differ')
else:
    os.replace(pending_path, final_path)
PY
  fi
  [[ -f "$expected_inventory_path" ]] || { echo "resume requires expected_adapter_inventory.json" >&2; exit 1; }
  if [[ "$PROFILE" == graph_treatment || "$PROFILE" == arrow_graph_treatment ]]; then
    [[ -f "$resume_checkpoint_dir/policy_preprocessor.json" ]] || { echo "graph resume checkpoint lacks policy_preprocessor.json" >&2; exit 1; }
    [[ -d "$resume_checkpoint_dir/tokenizer" ]] || cp -a "$BASE_POLICY_VALUE/tokenizer" "$resume_checkpoint_dir/tokenizer" || { echo "could not restore graph checkpoint tokenizer" >&2; exit 1; }
    [[ -f "$resume_checkpoint_dir/tokenizer_provenance.json" ]] || cp "$BASE_POLICY_VALUE/tokenizer_provenance.json" "$resume_checkpoint_dir/tokenizer_provenance.json" || { echo "could not restore graph checkpoint tokenizer provenance" >&2; exit 1; }
    PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_VALUE" "$VLA_ROOT/tools/prompt_audit.py" --retarget-graph-checkpoint "$resume_checkpoint_dir" || { echo "could not retarget graph resume preprocessor" >&2; exit 1; }
  fi
  "$PYTHON_VALUE" "$SCRIPT_DIR/adapter_audit.py" --checkpoint "$resume_checkpoint_dir" \
    --expected-inventory "$expected_inventory_path" \
    || { echo "resume checkpoint failed the action-side LoRA audit" >&2; exit 1; }
  provenance_path="${RUN_ROOT}/run_provenance.json"
  [[ -f "$provenance_path" ]] || provenance_path="${RUN_ROOT}.run_provenance.pending.json"
  [[ -f "$provenance_path" ]] || { echo "resume requires preserved run_provenance.json or pending provenance" >&2; exit 1; }
  "$PYTHON_VALUE" - "$PLAN_SOURCE" "$RUN_ROOT/training_manifest.json" "$provenance_path" "$PROFILE" "$EXPECTED_DATASET_VARIANT" "$BASE_POLICY_VALUE" "$BASE_POLICY_REVISION_VALUE" "$PAIR_MANIFEST_VALUE" "$DATASET_REPO_ID" "$LIBERO_DIR_VALUE" "$LIBERO_COMMIT_VALUE" <<'PY'
import hashlib, json, pathlib, sys
plan_path=pathlib.Path(sys.argv[1]); manifest_path=pathlib.Path(sys.argv[2]); provenance_path=pathlib.Path(sys.argv[3]); profile=sys.argv[4]; expected_dataset=sys.argv[5]; base_policy=pathlib.Path(sys.argv[6]).expanduser().resolve(); revision=sys.argv[7]; pair_path=pathlib.Path(sys.argv[8]); expected_repo=sys.argv[9]; libero_dir=pathlib.Path(sys.argv[10]).expanduser().resolve(); libero_commit=sys.argv[11]
plan=json.loads(plan_path.read_text(encoding='utf-8'))
required_plan=('training_variant','dataset_variant','dataset_repo_id','base_policy','base_policy_revision','pair_manifest_sha256','flags','libero_dir','libero_commit','libero_worktree_status','libero_tracked_clean')
if any(key not in plan for key in required_plan):
    raise SystemExit('resume training plan is missing required provenance keys')
if plan['training_variant'] != profile or plan['dataset_variant'] != expected_dataset or plan['dataset_repo_id'] != expected_repo or plan['base_policy_revision'] != str(revision):
    raise SystemExit('resume provenance is incompatible with the requested profile/base revision')
if pathlib.Path(plan['base_policy']).expanduser().resolve() != base_policy:
    raise SystemExit('resume base checkpoint differs from the requested snapshot')
if pathlib.Path(plan['libero_dir']).expanduser().resolve() != libero_dir or plan['libero_commit'] != libero_commit or plan['libero_worktree_status'] != 'clean' or plan['libero_tracked_clean'] is not True:
    raise SystemExit('resume training plan LIBERO verification is not clean/pinned')
flags=plan['flags']
expected_flags={'seed': 1000, 'peft_r': 16, 'batch_size': 32, 'steps': 29190, 'save_freq': 1946}
if any(key not in flags or int(flags[key]) != value for key, value in expected_flags.items()):
    raise SystemExit('resume training plan flags are not sealed (seed=1000, peft_r=16, batch_size=32, steps=29190, save_freq=1946)')
digest=hashlib.sha256(pair_path.read_bytes()).hexdigest()
if plan['pair_manifest_sha256'] != digest:
    raise SystemExit('resume pair manifest provenance differs from the current sealed input')
provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
required_prov=('variant','dataset_variant','dataset_repo_id','base_policy','base_policy_revision','pair_manifest_sha256','flags','libero_dir','libero_commit','libero_worktree_status','libero_tracked_clean')
if any(key not in provenance for key in required_prov):
    raise SystemExit('preserved run provenance is missing required keys')
if provenance['variant'] != profile or provenance['dataset_variant'] != expected_dataset or provenance['dataset_repo_id'] != expected_repo or pathlib.Path(provenance['base_policy']).expanduser().resolve() != base_policy or provenance['base_policy_revision'] != str(revision):
    raise SystemExit('preserved run provenance is incompatible with the requested profile/base revision')
if pathlib.Path(provenance['libero_dir']).expanduser().resolve() != libero_dir or provenance['libero_commit'] != libero_commit or provenance['libero_worktree_status'] != 'clean' or provenance['libero_tracked_clean'] is not True:
    raise SystemExit('preserved run provenance LIBERO verification is not clean/pinned')
if provenance['pair_manifest_sha256'] != digest:
    raise SystemExit('preserved run provenance pair hash differs from the current sealed input')
prov_flags=provenance['flags']
if any(key not in prov_flags or int(prov_flags[key]) != value for key, value in expected_flags.items()):
    raise SystemExit('preserved run provenance flags are incompatible with sealed resume')
if manifest_path.is_file():
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('pair_manifest_sha256') != digest:
        raise SystemExit('resume training manifest pair provenance differs from the current sealed input')
    if manifest.get('training_variant') != profile or manifest.get('base_policy_revision') != str(revision):
        raise SystemExit('resume training manifest is incompatible with requested profile/base revision')
PY
  "$PYTHON_VALUE" - "$PLAN_SOURCE" "$RESUME_CONFIG_PATH_VALUE" "$RUN_ROOT/resume_audits" "$PROFILE" "$EXPECTED_DATASET_VARIANT" "$DATASET_REPO_ID" "$resume_checkpoint_dir" <<'PY'
import hashlib, json, pathlib, sys
plan_path, config_path, audit_dir = map(pathlib.Path, sys.argv[1:4])
profile, expected_dataset, expected_repo = sys.argv[4:7]
checkpoint_dir = pathlib.Path(sys.argv[7]).resolve()
def load(path):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc: raise SystemExit(f'unreadable resume evidence: {path}') from exc
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1048576), b''): h.update(block)
    return h.hexdigest()
def tree_sha(root):
    entries=[]
    for path in sorted(root.rglob('*')):
        if path.is_file(): entries.append((path.relative_to(root).as_posix(), sha(path)))
    return hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()
plan=load(plan_path); config=load(config_path)
def get(data, *keys):
    cur=data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur: return None
        cur=cur[key]
    return cur
values={
 'dataset.repo_id':get(config,'dataset','repo_id'), 'dataset.root':get(config,'dataset','root'),
 'output_dir':get(config,'output_dir'), 'seed':get(config,'seed'), 'batch_size':get(config,'batch_size'),
 'steps':get(config,'steps'), 'save_freq':get(config,'save_freq'),
 'peft.r':get(config,'peft','r') if get(config,'peft','r') is not None else get(config,'policy','peft','r'),
}
required=('data_root','output_dir')
if any(key not in plan for key in required): raise SystemExit('resume plan lacks data_root/output_dir for train_config binding')
expected={'dataset.repo_id':expected_repo, 'dataset.root':str(pathlib.Path(plan['data_root'], expected_dataset).resolve()), 'output_dir':str(pathlib.Path(plan['output_dir']).resolve()), 'seed':1000, 'batch_size':32, 'steps':29190, 'save_freq':1946, 'peft.r':16}
for key, wanted in expected.items():
    actual=values[key]
    if actual is None or (str(actual) != str(wanted) and not (isinstance(wanted,int) and int(actual) == wanted)):
        raise SystemExit(f'resume train_config mismatch: {key}')
if profile in ('graph_treatment', 'arrow_graph_treatment'):
    policy = config.get('policy', {})
    if int(policy.get('tokenizer_max_length', -1)) != 96:
        raise SystemExit('resume train_config tokenizer_max_length is not sealed to 96 for graph profile')
config_digest=sha(config_path)
audit_dir.mkdir(parents=True, exist_ok=True)
records=[]
def record_digest(record):
    body=dict(record); body.pop('record_sha256',None)
    return hashlib.sha256((json.dumps(body,indent=2,sort_keys=True)+'\n').encode('utf-8')).hexdigest()
for path in sorted(audit_dir.glob('*.json')):
    record=load(path)
    if record.get('record_sha256') != record_digest(record): raise SystemExit(f'resume audit record hash mismatch: {path}')
    records.append((path, record))
for index, (path, record) in enumerate(records, 1):
    if record.get('chain_index') != index: raise SystemExit('resume audit chain index is not contiguous')
    previous=records[index-2][1]['record_sha256'] if index > 1 else None
    if record.get('previous_record_sha256') != previous: raise SystemExit('resume audit chain linkage is broken')
checkpoint_parts=config_path.parts
checkpoint_indexes=[index for index, part in enumerate(checkpoint_parts) if part == 'checkpoints']
if not checkpoint_indexes:
    raise SystemExit('resume train_config path is not under a checkpoints directory')
step_index=checkpoint_indexes[-1]+1
if step_index >= len(checkpoint_parts) or not checkpoint_parts[step_index].isdigit():
    raise SystemExit('resume train_config path lacks a numeric checkpoints/<step> component')
checkpoint_step=int(checkpoint_parts[step_index])
if not (0 < checkpoint_step < 29190 and checkpoint_step % 1946 == 0):
    raise SystemExit('resume checkpoint is outside the sealed ordinary save schedule')
config_sha=config_digest
adapter_evidence={
    'adapter_config_sha256': sha(checkpoint_dir/'adapter_config.json'),
    'adapter_weights_sha256': sha(checkpoint_dir/'adapter_model.safetensors'),
    'adapter_audit_sha256': sha(checkpoint_dir/'adapter_audit.json'),
    'checkpoint_tree_sha256': tree_sha(checkpoint_dir),
    'checkpoint_root_tree_sha256': tree_sha(checkpoint_dir.parent),
    'checkpoint_root_inventory': {
        path.relative_to(checkpoint_dir.parent).as_posix(): sha(path)
        for path in sorted(checkpoint_dir.parent.rglob('*')) if path.is_file()
    },
    'expected_adapter_inventory_sha256': sha(checkpoint_dir.parents[2] / 'expected_adapter_inventory.json'),
}
for path, record in records:
    if record.get('train_config_sha256') == config_sha:
        if record.get('train_config_path') != str(config_path.resolve()): raise SystemExit('same resume config hash has a different path')
        if record.get('adapter_evidence') != adapter_evidence: raise SystemExit('resume adapter/config/preprocessor evidence drifted')
        raise SystemExit(0)
if records and checkpoint_step <= int(records[-1][1].get('checkpoint_step', -1)):
    raise SystemExit('resume rollback or non-newer checkpoint is not allowed')
chain_index=len(records)+1
record={'schema_version':1,'chain_index':chain_index,'train_config_path':str(config_path.resolve()),'train_config_sha256':config_sha,'checkpoint_step':checkpoint_step,'profile':profile,'dataset_variant':expected_dataset,'adapter_evidence':adapter_evidence,'previous_record_sha256':records[-1][1]['record_sha256'] if records else None}
record['record_sha256']=hashlib.sha256((json.dumps(record,indent=2,sort_keys=True)+'\n').encode('utf-8')).hexdigest()
path=audit_dir/f'{chain_index:06d}.json'
if path.exists(): raise SystemExit(f'resume audit record already exists: {path}')
path.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY
  if [[ "$PLAN_SOURCE" == "$RUN_PLAN_PENDING" ]]; then
    mv "$RUN_PLAN_PENDING" "$RUN_ROOT/training_plan.json"
  fi
  env_args=("${COMMON_ENV[@]}" "DATASET_ROOT=$DATA_ROOT_VALUE/$DATASET_VARIANT" "OUTPUT_DIR=$OUTPUT_VALUE" "RESUME=true" "RESUME_CONFIG_PATH=$RESUME_CONFIG_PATH_VALUE")
else
  [[ ! -e "$RUN_ROOT" ]] || { echo "run directory already exists: $RUN_ROOT" >&2; exit 1; }
  mkdir -p "$(dirname "$RUN_ROOT")"
fi

if [[ "$MODE" != resume ]]; then
"$PYTHON_VALUE" - "$RUN_PLAN_PENDING" "$PAIR_MANIFEST_VALUE" "$PAIR_SENTINEL_VALUE" "$BASE_POLICY_VALUE" "$BASE_POLICY_REVISION_VALUE" "$DATA_ROOT_VALUE" "$STEPS_VALUE" "$SAVE_FREQ_VALUE" "${BATCH_SIZE:-32}" "${SEED:-1000}" "${PEFT_R:-16}" "${DEVICE:-cuda}" "$DATASET_REPO_ID" "$DATASET_VARIANT" "$PROFILE" "$PAIR_KIND" "$OUTPUT_VALUE" "$RESUME_VALUE" "$RESUME_CONFIG_PATH_VALUE" "$LIBERO_DIR_VALUE" "$LIBERO_COMMIT_VALUE" "$EXPERIMENT_NAME" "$TRAINED_VISUAL_CONDITION" <<'PY'
import hashlib, json, pathlib, sys
out, pair_manifest, pair_sentinel, base_policy, revision, data_root, steps, save_freq, batch, seed, peft_r, device, dataset_repo_id, dataset_variant, variant, pair_kind, output_dir, resume, resume_config, libero_dir, libero_commit, experiment, trained_visual_condition = sys.argv[1:]
from importlib.metadata import PackageNotFoundError, version
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
def _package_installed(name):
    try:
        version(name)
    except PackageNotFoundError:
        return False
    return True
payload = {
    "schema_version": 1,
    "experiment": experiment,
    "base_policy": str(pathlib.Path(base_policy).resolve()),
    "base_policy_revision": revision,
    "pair_manifest": str(pathlib.Path(pair_manifest).resolve()),
    "pair_manifest_sha256": sha(pair_manifest),
    "pair_sentinel": str(pathlib.Path(pair_sentinel).resolve()),
    "pair_sentinel_sha256": sha(pair_sentinel),
    "data_root": str(pathlib.Path(data_root).resolve()),
    "dataset_repo_id": dataset_repo_id,
    "dataset_variant": dataset_variant,
    "training_variant": variant,
    "pair_kind": pair_kind,
    "training_entrypoint": "run_lerobot_train.py",
    "flags": {"peft_r": int(peft_r), "steps": int(steps), "save_freq": int(save_freq), "batch_size": int(batch), "seed": int(seed), "device": device,
               "epochs": 15 if int(steps) == 29190 else None, "updates_per_epoch": 1946 if int(steps) == 29190 else None},
    "output_dir": str(pathlib.Path(output_dir).resolve()),
    "expected_adapter_inventory": {"path": str((pathlib.Path(output_dir).resolve() / "expected_adapter_inventory.json"))},
    "libero_commit": libero_commit,
    "libero_dir": str(pathlib.Path(libero_dir).resolve()),
    "libero_worktree_status": "clean",
    "libero_tracked_clean": True,
    "package_versions": {
        name: (version(name) if _package_installed(name) else None)
        for name in ("lerobot", "peft", "accelerate", "torch")
    },
}
if variant == "no_arrow_treatment":
    payload["trained_on_visual_condition"] = trained_visual_condition
    payload["evaluation_contract"] = {
        "tasks": list(range(10)),
        "cells": ["no_arrow_trained_live_arrows", "no_arrow_trained_no_arrows"],
        "episodes_per_task": 10,
        "eval_seed": 1000,
        "eval_batch_size": 1,
        "n_action_steps": "checkpoint",
        "evaluation_entrypoint": "vla_benchmarking/arrow_finetuned_vla/workflows/run_lora_no_arrow_pair_eval.py",
        "evaluation_script": "vla_benchmarking/arrow_finetuned_vla/workflows/run_lora_no_arrow_pair_eval.py",
        "evaluation_experiment": "smolvla_lora_no_arrow_trained_live_vs_none_2cell",
        "evaluation_manifest_filename": "no_arrow_trained_arrow_pair_manifest.json",
        "evaluation_summary_filename": "no_arrow_trained_arrow_pair_summary.csv",
        "visual_conditions": {
            "no_arrow_trained_live_arrows": "visual_arrows",
            "no_arrow_trained_no_arrows": "none",
        },
        "context_mode": "standard",
        "context_format": "standard",
        "visual_prompt_hint": "disabled",
        "canonical_libero_init_state_variation": True,
        "custom_interventions": {
            "scene_layout": True,
            "object_removal": True,
        },
        "static_prompt_overrides": True,
        "camera_name": "agentview_image,robot0_eye_in_hand_image",
        "raw_camera_names": "agentview,robot0_eye_in_hand",
        "observation_size": [256, 256],
        "eval_use_async_envs": False,
        "render_mode": "rgb_array",
        "adapter_key": "no_arrow_treatment_adapter",
        "contrast": "live_arrow_effect_pp",
    }
elif variant in ("graph_treatment", "arrow_graph_treatment"):
    payload["trained_on_visual_condition"] = trained_visual_condition
    payload["trained_on_text_condition"] = "target_natural_v1"
    adapter_key = {
        "graph_treatment": "graph_treatment_adapter",
        "arrow_graph_treatment": "arrow_graph_treatment_adapter",
    }[variant]
    payload["evaluation_contract"] = {
        "status": "pilot_prepared_not_launchable",
        "confirmatory_required_for_paper_claims": True,
        "pilot_contract": {
            "paired_reset_states": True,
            "eval_seed": 1000,
            "episodes_per_task": 10,
            "task_stratified_results": True,
            "evaluation_cells": ["graph_context_no_arrows", "standard_no_arrows"],
            "visual_condition": "none",
        },
        "future_confirmatory_requirements": {
            "paired_reset_states": True,
            "training_seeds": [1000, 1001, 1002],
            "eval_master_seeds": list(range(2000, 2050)),
            "minimum_episodes_per_task": 50,
            "minimum_independent_training_seeds": 3,
            "task_stratified_results": True,
            "interaction_effect_and_95_percent_ci": True,
            "launch_requires_explicit_four_cell_evaluator": True,
        },
        "tasks": list(range(10)),
        "episodes_per_task": 10,
        "eval_seed": 1000,
        "eval_batch_size": 1,
        "context_mode": "scene_graph",
        "context_format": "target_natural_v1",
        "tokenizer_max_length": 96,
        "visual_condition": "visual_arrows" if variant == "arrow_graph_treatment" else "none",
        "adapter_key": adapter_key,
        "graph_pair_kind": "sealed_lora_graph_treatment_arrow_graph_treatment",
    }
training_argv = [
    "run_lerobot_train.py",
    "--policy.push_to_hub=false",
    f"--peft.r={int(peft_r)}",
    f"--dataset.repo_id={dataset_repo_id}",
    f"--dataset.root={pathlib.Path(data_root, dataset_variant).resolve()}",
    f"--output_dir={pathlib.Path(output_dir).resolve()}",
    f"--steps={int(steps)}",
    f"--save_freq={int(save_freq)}",
    "--eval_freq=0",
    f"--batch_size={int(batch)}",
    f"--policy.device={device}",
    f"--seed={int(seed)}",
]
if resume == "true":
    training_argv.extend([f"--config_path={pathlib.Path(resume_config).resolve()}", "--resume=true"])
else:
    training_argv.append(f"--policy.path={pathlib.Path(base_policy).resolve()}")
if variant in ("graph_treatment", "arrow_graph_treatment"):
    training_argv.append("--policy.tokenizer_max_length=96")
payload["training_argv"] = training_argv
p = pathlib.Path(out)
encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
if p.exists() and p.read_text(encoding="utf-8") != encoded:
    raise SystemExit(f"immutable treatment training plan already differs: {p}")
if not p.exists():
    p.write_text(encoded, encoding="utf-8")
PY
fi

env "${env_args[@]}" bash "$SCRIPT_DIR/train_lora.sh" "$PROFILE"

if [[ "$MODE" != resume ]]; then
  [[ -f "$RUN_PLAN_PENDING" ]] || { echo "treatment training plan sidecar is missing: $RUN_PLAN_PENDING" >&2; exit 1; }
  mv "$RUN_PLAN_PENDING" "$RUN_ROOT/training_plan.json"
fi

checkpoint_id="$($PYTHON_VALUE - "$STEPS_VALUE" <<'PY'
import sys
steps = int(sys.argv[1])
print(f"{steps:0{max(6, len(str(steps)))}d}")
PY
)"
adapter_path="$RUN_ROOT/checkpoints/$checkpoint_id/pretrained_model/adapter_model.safetensors"
[[ -s "$adapter_path" ]] || { echo "adapter artifact is missing: $adapter_path" >&2; exit 1; }

"$PYTHON_VALUE" - "$RUN_ROOT/training_manifest.json" "$RUN_ROOT/training_plan.json" "$adapter_path" "$PAIR_MANIFEST_VALUE" "$PAIR_SENTINEL_VALUE" "$BASE_POLICY_VALUE" "$BASE_POLICY_REVISION_VALUE" "$STEPS_VALUE" "$SAVE_FREQ_VALUE" "${BATCH_SIZE:-32}" "${SEED:-1000}" "${PEFT_R:-16}" "$checkpoint_id" "$PROFILE" "$PAIR_KIND" "$DATASET_REPO_ID" "$DATASET_VARIANT" "$LIBERO_DIR_VALUE" "$LIBERO_COMMIT_VALUE" "$RUN_ROOT/resume_audits" "$EXPERIMENT_NAME" "$TRAINED_VISUAL_CONDITION" "$ADAPTER_KEY" "$RUN_ROOT/graph_tokenizer_audit.json" "$RUN_ROOT/expected_adapter_inventory.json" <<'PY'
import hashlib, json, pathlib, sys
out, plan, treatment, pair_manifest, pair_sentinel, base_policy, revision, steps, save_freq, batch, seed, peft_r, checkpoint_id, variant, pair_kind, dataset_repo_id, dataset_variant, libero_dir, libero_commit, audit_dir, experiment, trained_visual_condition, adapter_key, graph_audit_path, expected_inventory_path = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
def record_digest(record):
    body=dict(record); body.pop('record_sha256',None)
    return hashlib.sha256((json.dumps(body,indent=2,sort_keys=True)+'\n').encode('utf-8')).hexdigest()
audit_root=pathlib.Path(audit_dir)
audit_records=[]
if audit_root.is_dir():
    for audit_path in sorted(audit_root.glob('*.json')):
        record=json.loads(audit_path.read_text(encoding='utf-8'))
        if record.get('record_sha256') != record_digest(record): raise SystemExit(f'resume audit hash mismatch: {audit_path}')
        audit_records.append(record)
for index, record in enumerate(audit_records, 1):
    if record.get('chain_index') != index: raise SystemExit('resume audit chain index is not contiguous')
    previous=audit_records[index-2]['record_sha256'] if index > 1 else None
    if record.get('previous_record_sha256') != previous: raise SystemExit('resume audit chain linkage is broken')
chain_digest=hashlib.sha256((json.dumps([r['record_sha256'] for r in audit_records],sort_keys=True,separators=(',',':'))).encode('utf-8')).hexdigest()
payload = {
    "schema_version": 1,
    "experiment": experiment,
    "base_policy": str(pathlib.Path(base_policy).resolve()),
    "base_policy_revision": revision,
    "training_plan": str(pathlib.Path(plan).resolve()),
    "training_plan_sha256": sha(plan),
    "pair_manifest": str(pathlib.Path(pair_manifest).resolve()),
    "pair_manifest_sha256": sha(pair_manifest),
    "pair_sentinel": str(pathlib.Path(pair_sentinel).resolve()),
    "pair_sentinel_sha256": sha(pair_sentinel),
    "training_variant": variant,
    "dataset_repo_id": dataset_repo_id,
    "dataset_variant": dataset_variant,
    "libero_dir": str(pathlib.Path(libero_dir).resolve()),
    "libero_commit": libero_commit,
    "libero_worktree_status": "clean",
    "libero_tracked_clean": True,
    "resume_audits": audit_records,
    "resume_chain_digest": chain_digest,
    "pair_kind": pair_kind,
    "final_checkpoint_id": checkpoint_id,
    "flags": {"steps": int(steps), "save_freq": int(save_freq), "batch_size": int(batch), "seed": int(seed)},
}
if variant == "no_arrow_treatment":
    payload["trained_on_visual_condition"] = trained_visual_condition
    payload["flags"]["peft_r"] = int(peft_r)
payload[adapter_key] = {"path": str(pathlib.Path(treatment).resolve()), "sha256": sha(treatment)}
payload["trained_on_visual_condition"] = trained_visual_condition
payload["trained_on_text_condition"] = (
    "target_natural_v1" if variant in ("graph_treatment", "arrow_graph_treatment") else "none"
)
if variant in ("graph_treatment", "arrow_graph_treatment"):
    payload["flags"]["peft_r"] = int(peft_r)
    payload["flags"]["tokenizer_max_length"] = 96
    graph_manifest = json.loads(pathlib.Path(pair_manifest).read_text(encoding="utf-8"))
    payload["graph_contract_sha256"] = graph_manifest.get("graph_contract_sha256")
    payload["graph_formatter_sha256"] = graph_manifest.get("graph_formatter_sha256")
    payload["graph_extractor_sha256"] = graph_manifest.get("graph_extractor_sha256")
    payload["tokenizer_contract_sha256"] = graph_manifest.get("tokenizer_contract_sha256")
    payload["graph_oracle_disclosure"] = (graph_manifest.get("graph_contract") or {}).get("oracle_disclosure")
    payload["comparability_contract"] = graph_manifest.get("comparability_contract")
    audit = pathlib.Path(graph_audit_path)
    if not audit.is_file():
        raise SystemExit(f"graph tokenizer audit evidence is missing: {audit}")
    payload["graph_tokenizer_audit"] = {
        "path": str(audit.resolve()),
        "sha256": sha(audit),
    }
adapter_dir = pathlib.Path(treatment).resolve().parent
adapter_audit = adapter_dir / "adapter_audit.json"
if not adapter_audit.is_file():
    raise SystemExit(f"action-side LoRA audit evidence is missing: {adapter_audit}")
payload["adapter_audit"] = {
    "path": str(adapter_audit),
    "sha256": sha(adapter_audit),
}
adapter_audit_record = json.loads(adapter_audit.read_text(encoding="utf-8"))
checkpoint_root = adapter_dir.parent
if not (checkpoint_root / "training_state").is_dir():
    raise SystemExit(f"checkpoint training_state is missing: {checkpoint_root / 'training_state'}")
state_files = [path for path in (checkpoint_root / "training_state").rglob("*") if path.is_file()]
if not state_files or any(path.is_symlink() for path in state_files):
    raise SystemExit("checkpoint training_state must contain regular files only")
state_names = {path.name.lower() for path in state_files}
required_state = {
    "optimizer": any("optimizer" in name for name in state_names),
    "scheduler": any("scheduler" in name for name in state_names),
    "rng": any("random_state" in name or "rng" in name for name in state_names),
}
if not all(required_state.values()):
    missing = ", ".join(key for key, present in required_state.items() if not present)
    raise SystemExit(f"checkpoint training_state is missing required state artifacts: {missing}")
payload["checkpoint_root_tree_sha256"] = hashlib.sha256(
    json.dumps({path.relative_to(checkpoint_root).as_posix(): sha(path) for path in sorted(checkpoint_root.rglob("*")) if path.is_file()}, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
payload["checkpoint_root_inventory"] = {
    path.relative_to(checkpoint_root).as_posix(): sha(path)
    for path in sorted(checkpoint_root.rglob("*")) if path.is_file()
}
expected_inventory = pathlib.Path(expected_inventory_path)
if not expected_inventory.is_file():
    raise SystemExit(f"expected live-policy LoRA inventory is missing: {expected_inventory}")
payload["expected_adapter_inventory"] = {
    "path": str(expected_inventory.resolve()),
    "sha256": sha(expected_inventory),
}
if adapter_audit_record.get("expected_inventory_sha256") != json.loads(expected_inventory.read_text(encoding="utf-8")).get("inventory_sha256"):
    raise SystemExit("checkpoint adapter audit is not bound to the expected live-policy inventory")
payload["checkpoint_tree_sha256"] = adapter_audit_record.get("checkpoint_tree_sha256")
payload["checkpoint_inventory"] = adapter_audit_record.get("checkpoint_inventory")
payload["adapter_config_sha256"] = sha(adapter_dir / "adapter_config.json")
p = pathlib.Path(out)
encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
if p.exists() and p.read_text(encoding="utf-8") != encoded:
    raise SystemExit(f"immutable treatment training manifest already differs: {p}")
if not p.exists():
    p.write_text(encoded, encoding="utf-8")
PY

"$PYTHON_VALUE" - "$adapter_path" "$ADAPTER_KEY" <<'PY'
import sys
from pathlib import Path
from peft import PeftConfig
from safetensors import safe_open
adapter = Path(sys.argv[1])
adapter_key = sys.argv[2]
PeftConfig.from_pretrained(adapter.parent)
with safe_open(str(adapter), framework="pt", device="cpu") as handle:
    if not list(handle.keys()):
        raise SystemExit("adapter safetensors has no tensors")
print(f"{adapter_key} reload smoke OK: {adapter.parent}")
PY
printf '[%s] %s launcher finished in %s\n' "$(date +'%F %T')" "$PROFILE" "$RUN_ROOT"
