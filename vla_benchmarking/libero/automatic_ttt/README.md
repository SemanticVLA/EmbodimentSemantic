# Automatic RoboTTT on LIBERO

This folder is a self-contained experiment package for the following causal
test-time-training question:

> A VLA attempts a LIBERO task.  If it fails, can the Arrow grasp controller
> recover the same live episode, and can the resulting correction train the
> VLA so that its success rate improves on held-out episodes?

The attached “Randomized LIBERO-Spatial” table is treated as prior benchmark
context only.  It is not a configuration file, target score, or instruction to
copy the reported numbers.  This package does not claim an improvement until
paired evaluation produces the raw episode records.

## How many Arrow demonstrations?

The paper's DAgger Distillation experiment uses a pooled **100-trajectory**
dataset: 50 DAgger trajectories collected with RoboTTT as the base policy and
50 collected with GR00T N1.7. The paper does not prescribe a fixed number of
controller actions inside each trajectory; one trajectory can contain multiple
interleaved robot and correction chunks, so “100 demonstrations” does not mean
100 single actions.

For this LIBERO study, the frozen proposed collection contract targets **100
accepted Arrow-corrected trajectories per adaptation round** (20 attempts on
each of tasks 0, 2, 4, 6, and 8). Attempts Arrow cannot complete remain failed
attempts in the denominator and are not training demonstrations. This is an
explicit LIBERO design choice derived from the paper's pool, not a count claimed
by the paper for LIBERO.

Every accepted demonstration passes `validate_and_build_demonstration` before
training. It checks the interleaved trace, exact VLA-to-Arrow observation handoff
on the same environment, adjacent observation continuity, normalized finite
actions, source-state classification, complete action chunks, evaluator-confirmed
teacher success, and absence of privileged controller fields in student inputs.
`count_accepted_arrow_trajectories` and `require_arrow_target` count only receipts
with both `teacher_success=true` and `evaluator_success=true`, reject duplicate
episode IDs, and fail closed if the round does not reach the target pool.

## Why this experiment is scientifically useful

RoboTTT identifies a deployment-time failure mode: a policy can have useful
visual-language information but its action decoder is misaligned with the
current scene.  Its TTT-KVB layer learns a short-lived key/value mapping from
the current sequence, updates that mapping before producing the current action,
and carries the fast state through the rollout.  Its DAgger-style distillation
then trains the slow model on human corrections while retaining the failed
robot context.

LIBERO gives us a deterministic, resettable manipulation benchmark where the
same task can be evaluated before and after adaptation.  The Arrow controller
is a reproducible teacher in place of a human: it takes over the *existing live
environment* after the VLA failure, completes only recoverable episodes, and
records its observations and actions.  The important causal comparison is:

```
same policy + same held-out episode seeds
    frozen baseline  ->  failed attempt / no update
    adapted policy   ->  fresh attempt after training on disjoint teacher data
```

The controller's simulator privileges (object bboxes, contacts, recovery
classification) are stored as teacher metadata and are never provided as VLA
observations or training targets.  This prevents a hidden privileged state
leak.

## Fidelity boundary

The default `fidelity_mode` is `exact`.  In this mode `preflight`, `collect`,
`train`, and `evaluate` refuse to run unless a manifest from the official
RoboTTT artifact supplies every value that the paper does not specify.  The
gate currently requires:

* official checkpoint and code commit;
* fast-MLP width and TTT projection dimension;
* learned step-size parameterization;
* TBPTT segment length;
* action horizon and denoising-step count;
* token packing rules; and
* AdamW betas, epsilon, and gradient clipping.

The paper values that *are* known are recorded in
`config.PAPER_KNOWN_SETTINGS`: 16 DiT layers, 16 register tokens per timestep,
the tanh gate initialized at 0.001, KVB MSE, the flow-matching path and target,
Beta(1.5, 1) time sampling, 30,000/20,000 steps, and the published learning
rates/weight decays.  A missing value remains `null`; it is never guessed.

`algorithmic_port` is an explicit research condition for a public-checkpoint
port when the official artifact is unavailable.  It is not an exact RoboTTT
replication and must be reported as such.  It does not silently change the
default experiment.

## Temporary PEFT control experiment

While the RoboTTT action-head runtime is unavailable, the Legion launcher
`legion/run_smolvla_peft_arrow_task.sbatch` runs a clearly labelled native
SmolVLA LoRA/PEFT control.  It evaluates the frozen base on task 0 for ten
sealed seeds, converts exactly `PEFT_ARROW_DEMOS` (default **50**) source HDF5
demos with ground-truth arrows, verifies every control/treatment frame pair,
trains the existing LeRobot PEFT path, and repeats the same evaluation seeds.
This is useful as a running engineering experiment, but it is not evidence for
RoboTTT TTT or for runtime Arrow-controller corrections.

Every adapter is published immutably under:

```text
<archive>/smolvla/task_0/peft_adapter/<run-id>/
```

The directory contains the native adapter files plus `artifact_manifest.json`:
per-file and tree SHA-256 hashes, base-checkpoint and dataset-manifest hashes,
task/VLA/run identity, seed, train/evaluation counts, source commit, and runtime
versions.  Existing artifacts are never overwritten.  The same writer is
intended for future VLA/task launchers, so adapters cannot be confused across
tasks or models.

For a joint all-task run, set `PEFT_TASK_IDS="0 1 2 3 4 5 6 7 8 9"` and
`PEFT_EPOCHS=50`.  The converter then requires the full 10-task × 50-demo
sealed pair (500 episodes), evaluation runs ten episodes per task, and the
single jointly trained checkpoint is copied into one immutable artifact
directory under each `task_0` … `task_9` path with the shared `task_ids` scope
recorded in every manifest.  The default remains task 0 for backwards
compatibility.

## Four VLA conditions

The template includes all requested policy names and evaluates each with the
same task IDs, episode seeds, observation preprocessing, action normalization,
teacher budget, and train/eval split:

| Config name | Display name | Required adapter contract |
| --- | --- | --- |
| `openvla` | OpenVLA | `begin_episode`, `act`, fast-state update/reset |
| `pi05` | Pi0.5 | same contract; use its native processor in the adapter |
| `smolvla` | SmolVLA | same contract; preserve its native action chunk semantics |
| `ours` | Ours/Arrow-compatible | adapter for the repository's Arrow-compatible VLA |

The package does not convert one model's processor into another's.  Each
adapter must declare its checkpoint digest, image resize/flip convention,
state layout, action dimension, action range, action chunk horizon, and
instruction formatting in the run manifest.  A run is invalid if any of these
change between baseline and adapted evaluation.

Exact zero-shot runs also require one auditable task-exclusion evidence path per
VLA in `zero_shot_audit_evidence`. The runtime must turn those byte-verified
records into a `ZeroShotAuditAttestation`; a config boolean is never accepted as
proof.

## Run contract

The run has three phases.  Each phase writes immutable JSON/JSONL records under
the run's `output_root`; it does not overwrite prior runs.

### 1. Baseline collection and teacher takeover

For every VLA × task × seed:

1. Reset LIBERO once and retain the environment object.
2. Run the selected VLA for at most `student_step_budget` steps (220 in the
   template), recording every VLA observation, action, action chunk index,
   timestamp, and success/terminal signal.
3. If the task is not successful and the failure is recoverable, invoke the
   Arrow teacher on that same environment.  The teacher must not call reset or
   close.  It records its observation/action sequence and a recovery label.
4. Reject non-finite actions, terminal environments, wrong-object-held states,
   and unrecoverable episodes without creating a training target.
5. Store a cryptographic digest of each episode's raw record and the exact
   config digest in the manifest.
6. Admit the trace to training only after `validate_and_build_demonstration`
   accepts it. A teacher-success flag without a final evaluator-success verdict,
   or a trace with a boundary/gap/chunk/provenance error, is rejected.

Teacher labels distinguish at least `source_unheld` (safe open/preshape/grasp/
place) and `source_held` (placement-only recovery that preserves the grasp).
Calling the canonical controller as though the object were unheld can drop a
successfully grasped object; the takeover branch must therefore be explicit.

### 2. Slow training on correction trajectories

The training backend must implement the paper's order of operations exactly:

* teacher corrections are targets;
* the complete failed VLA context remains before each correction;
* action loss is masked to teacher-correction tokens only;
* robot-only failed actions do not become positive action labels;
* the flow-matching target is `A - epsilon` with
  `A_tau = tau*A + (1-tau)*epsilon`;
* each action chunk samples `tau = 0.999*(1-u)`, `u ~ Beta(1.5, 1)`;
* TTT-KVB updates `W` with
  `L_FW,t = ||f_W(K_t) - V_t||²`, then predicts with `f_Wt(Q_t)`;
* fast state carries over segment boundaries with detached TBPTT gradients;
* the slow initial state `W0` receives gradient through the first segment only;
* the learned gate is `tanh(alpha)*TTT + attention`, with alpha initialized to
  0.001; and
* the paper's optimizer, schedule, batch sizes, steps, and decay are copied
  from the official artifact before exact mode is enabled.

The bookkeeping and masking implementation is shared across all four VLA
adapters, but exact RoboTTT training is not.  Exact mode is restricted to an
artifact whose native action head is the paper's DiT flow-matching head (the
published RoboTTT/GR00T-compatible case).  OpenVLA-OFT's native continuous L1
head, Pi0.5's native objective, and SmolVLA's native objective must each be
reported as architecture-matched algorithmic ports: retain their native-head
baseline, adapt only through an explicitly declared port, and never place its
result in the exact RoboTTT claim.  Adapter metadata includes the native
objective and a compatibility key so baseline and adapted evaluation cannot
silently use different processors or action semantics.

### 3. Paired evaluation

Evaluation uses fresh held-out episodes and runs each policy twice with the
same seed list: frozen baseline and adapted checkpoint.  Report, per VLA and
task, episode count, success rate, Wilson interval, paired success delta,
teacher recovery rate, VLA-only success, and hybrid (VLA + teacher) success.
Also report controls:

* `correction_only`: remove failed robot context;
* `full_failure_context`: expected RoboTTT-style context condition;
* `shuffled_failure_context`: equal-length but wrong failure context;
* `reset_fast_state`: reset KVB state between segments/episodes;
* `gdn`: global/no-test-time-update control; and
* `frozen_baseline`: no adaptation.

The primary claim is only supported if improvement is measured on held-out
episodes, is paired by seed, and survives the equal-data controls.  The
reported table in the user-provided image is not used as a success criterion.

## Explicit commands

Run these from the repository root with Python 3.12 and the project's
`vla_bench_py312` environment.  No command below launches expensive training
unless it is explicitly invoked without `--dry-run`.

```powershell
# Inspect the blocked template. This should fail until the official artifact
# manifest and all null paper settings are filled in.
python -m vla_benchmarking.libero.automatic_ttt.cli preflight `
  --config vla_benchmarking/libero/automatic_ttt/example_config.json

# Create a copy to edit, preserving the template as a reference.
python -m vla_benchmarking.libero.automatic_ttt.cli init-config `
  vla_benchmarking/libero/automatic_ttt/my_exact_config.json

# Cheap contract checks; no LIBERO or GPU import is performed.
python -m vla_benchmarking.libero.automatic_ttt.cli collect `
  --config vla_benchmarking/libero/automatic_ttt/my_exact_config.json --dry-run
python -m vla_benchmarking.libero.automatic_ttt.cli train `
  --config vla_benchmarking/libero/automatic_ttt/my_exact_config.json --dry-run
python -m vla_benchmarking.libero.automatic_ttt.cli evaluate `
  --config vla_benchmarking/libero/automatic_ttt/my_exact_config.json --dry-run

# After preflight passes and the environment is installed:
python -m vla_benchmarking.libero.automatic_ttt.cli collect `
  --config vla_benchmarking/libero/automatic_ttt/my_exact_config.json
python -m vla_benchmarking.libero.automatic_ttt.cli train `
  --config vla_benchmarking/libero/automatic_ttt/my_exact_config.json
python -m vla_benchmarking.libero.automatic_ttt.cli evaluate `
  --config vla_benchmarking/libero/automatic_ttt/my_exact_config.json

# Summarize paired metrics without changing raw records.
python -m vla_benchmarking.libero.automatic_ttt.cli report `
  vla_benchmarking/libero/automatic_ttt/runs/<run-id> `
  --config vla_benchmarking/libero/automatic_ttt/my_exact_config.json
```

## Runtime injection contract

The package does not fabricate a LIBERO environment, model adapter, teacher,
updater, or evaluator.  To execute a real run, supply an import path either in
the JSON `runtime` block or with the operation's `--factory` option:

```json
"runtime": {
  "collection_factory": "my_project.automatic_ttt_host:collect",
  "training_factory": "my_project.automatic_ttt_host:train",
  "evaluation_factory": "my_project.automatic_ttt_host:evaluate"
}
```

Each callable must have this exact signature and return a JSON-serializable
mapping with a non-blocked status:

```python
def collect(*, config, args) -> dict:
    # Build the four real VLA adapters, one live LIBERO env factory, and the
    # Arrow teacher; call automatic_ttt.experiment.run_episode for each split ID.
    # Return the written trace/dataset paths and counts.

def train(*, config, args) -> dict:
    # Load only train/validation traces, construct the exact RoboTTT model
    # adapter, run a verifier-driven probe through all 16 concrete TTT layers,
    # and pass capture_runtime_attestation(adapter, probe=...) to
    # RoboTTTTrainer. A hand-written receipt or boolean exact flag is rejected.

def evaluate(*, config, args) -> dict:
    # Run frozen_baseline/adapted/hybrid on exactly config.split.eval_episode_ids,
    # write evaluation.TrialMetric records, and return metrics.json paths.
```

The CLI passes the typed configuration (including its digest and provenance)
and the parsed `argparse.Namespace` unchanged.  A factory may be selected for a
single invocation without editing JSON:

```powershell
python -m vla_benchmarking.libero.automatic_ttt.cli collect `
  --config vla_benchmarking/libero/automatic_ttt/my_config.json `
  --factory my_project.automatic_ttt_host:collect
```

If no factory is supplied, the package invokes its lower-level backend, which
returns a nonzero blocked status rather than claiming that data or a checkpoint
was produced.  A factory returning a `BLOCKED...` status or no result is also a
failed command.  The report command is independent of runtime injection and
returns nonzero for empty, malformed, duplicate, unpaired, or incomplete
metrics; it emits no improvement claim in those cases.

For an explicitly labelled algorithmic port, copy the template and change only
`fidelity_mode` to `algorithmic_port`.  The resulting paper/report must call it
an algorithmic port, list every unresolved field, and not place its numbers in
the exact RoboTTT table.

## New fail-closed study contracts

The package now includes four dependency-light ledgers used by a real host:

* `protocol.py` freezes counts, task/transfer assignment, query checkpoints,
  bootstrap settings, and authoritative simulator reset identities before the
  first policy action. A scored lock requires all 10 tasks × 3 adaptation seeds
  × 50 query episodes (1,500 reset identities); `pilot=True` is the explicit
  one-canary escape hatch and cannot be used for a scored claim.
* `provenance.py` verifies byte-hashed model/config/processor artifacts and
  task-exclusion audit records. A caller-supplied `is_zero_shot` flag or JSON
  receipt cannot unlock a zero-shot claim.
* `statistics.py` provides deterministic paired hierarchical resampling and an
  equal-weight-per-task delta estimator; pooled rates remain descriptive.
* `costs.py` records phase-level rollout, teacher, serialization, adaptation,
  and evaluation costs on both success and exception paths, with unavailable
  sensors represented by an explicit reason.

`runtime_host.py` is the typed integration boundary for a concrete LIBERO/VLA
deployment. It owns exactly one reset and one close, keeps simulator-state
identity separate from student-observation identity, and rejects arbitrary
success mappings. No real model loader is fabricated here; a host must supply
the actual checkpoint, processor, simulator serializer/replay key, and Arrow
controller implementation.

## Reproducibility and safety checks

Every run must preserve:

* config and artifact-manifest digests;
* git commits for this repository, LIBERO, robosuite, and each VLA adapter;
* Python/CUDA/torch/lerobot versions;
* controller configuration and teacher privilege declaration;
* task IDs, episode IDs, seeds, and split membership;
* model checkpoint digests and processor settings; and
* raw episode traces alongside derived metrics.

The package is intentionally dependency-light at import time.  A preflight can
run on a login node; LIBERO/robosuite/torch are loaded only by execution
backends.  Do not submit a large cluster job until preflight, one canary task,
and the independent validation checks pass.
