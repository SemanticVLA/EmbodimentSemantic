# Arrow Policy Suite: Remaining End-to-End Implementation Plan

## 1. Definition of “full”

The suite is full only when one command can take a frozen study configuration and:

1. load the pinned SmolVLA checkpoint and LIBERO task;
2. run the stepwise Arrow teacher without allowing it to step the environment itself;
3. collect the declared On-Call attempts on frozen training resets;
4. derive every learned dataset with immutable lineage;
5. train Apprentice, Editor, Minimal-Learned, and the slow part of Fast;
6. prepare the one-observation or one-support Fast state at evaluation time;
7. evaluate every concrete policy row on the same sealed test resets; and
8. produce a paired report containing success, teacher cost, branch cost, support cost, latency, failures, and confidence intervals.

The current package implements the policy logic, contracts, dependency-injection adapters, artifact checks, and synthetic tests. It does not yet supply the native components and launchers that perform the workflow above.

## 2. Frozen scientific contract

Complete this before producing new data. The output is an immutable `StudyConfig` plus split manifest used by every later command.

### Required decisions

- Pin the exact SmolVLA model/revision, processor revision, action encoding, and observation schema.
- Pin the LIBERO repository/environment revision and the ten tasks.
- Reserve three disjoint reset sets for each task:
  - collection/training identities;
  - exactly ten validation identities;
  - exactly ten final test identities.
- Define one reset identity as task ID + LIBERO initial-state index/hash + environment revision. A seed alone is insufficient.
- Freeze the Arrow controller revision, camera names, image resolution, depth units, calibration revision, coordinate frames, and visual-arrow renderer revision.
- Freeze `success@1200` as the primary metric and `success@280` as the secondary metric derived from the same rollout.
- Freeze the collection rule as exactly 50 On-Call attempts per task, including failures. Learned rows may be derived only from successful, complete On-Call episodes and only where the teacher action was actually executed.
- Freeze learned training seeds to 1000, 1001, and 1002. Do not duplicate deterministic policies three times.
- Report both Minimal variants separately. Do not select the better variant after observing test performance.

### Code work

- Extend `config.py` so it records all model, environment, geometry, controller, dataset, and metric revisions.
- Extend `splits.py` so it resolves real LIBERO initial-state identities rather than accepting caller-invented IDs.
- Add a `preflight --write-manifest` path in `cli.py` that emits the sealed configuration and refuses an existing destination.

### Acceptance tests

- Re-running preflight over the same inputs produces the same digest.
- Any overlap between collection, validation, and test identities aborts before simulator startup.
- A checkpoint, processor, controller, camera, geometry, or environment revision mismatch aborts artifact loading and evaluation.

## 3. Native execution spine

This is the critical blocker. All policy families depend on it.

### 3.1 Native LIBERO host

Add `native_runtime.py` inside this folder. It may import stable existing utilities, but any utility that must be changed should be copied here rather than modifying `automatic_ttt`.

Responsibilities:

- construct a LIBERO task at an exact frozen reset identity;
- expose only the canonical student observation: agent view, wrist view, eight-value robot state, and instruction;
- retain success and termination information in an evaluator-only sidecar;
- convert a normalized seven-value OSC action into the exact environment action contract;
- snapshot and restore the complete simulator and wrapper state;
- prove that snapshot → arbitrary steps → restore reproduces the same observation digest and success state;
- own cleanup and video/log destinations without overwriting an earlier run.

Do not leak object poses, segmentation, MuJoCo state, contact state, task predicates, or evaluator success into `ObservationFrame`.

### 3.2 Native SmolVLA adapter

Complete the native path behind `SmolVLAAdapter.from_local_checkpoint` in `smolvla_adapter.py`.

Responsibilities:

- load the pinned processor and checkpoint through the repository’s existing loader;
- enforce the canonical two-image + state8 + instruction input schema;
- configure `n_action_steps=1` for causal arbitration;
- bind every proposal to the observation digest and timestep that produced it;
- snapshot and restore the native action queue, model RNG, processor state if mutable, and any episode-local policy state;
- invalidate the queued VLA action after a teacher or hybrid action and re-propose from the new observation;
- record checkpoint, processor, dtype, device, and action-normalization provenance.

### 3.3 Interruptible Arrow teacher

Add `interruptible_arrow.py`. Port the relevant perception, waypoint, and OSC conversion logic from the existing Arrow evaluator, but split it into a non-stepping state machine.

Required interface:

- `reset(task, reset_identity)` initializes controller state without moving the robot;
- `propose(frame)` returns one same-state teacher action, phase error, milestone, gripper dwell/conflict, and validity metadata;
- `commit(record)` advances the Arrow phase using only the action that the coordinator actually executed;
- `snapshot_state()` and `restore_state()` capture the entire controller, perception, plan, RNG, and phase state;
- `interrupt(...)` begins or resumes takeover without executing an action;
- `anchors(frame)` returns typed Arrow RGB-D source/destination anchors with calibration, camera, frame, units, resolution, timestamp, and renderer provenance.

The existing whole-episode `recover()` path cannot be used inside these policies because it steps the environment internally.

### 3.4 Transactional canary

Add a `canary` CLI command and run it on one task/reset before collection.

The canary must prove:

- the VLA proposal, teacher proposal, and arbitration decision refer to the same observation digest;
- proposals cause zero environment transitions;
- exactly one environment transition occurs per committed decision;
- Gym/LIBERO termination and truncation stop further stepping;
- a rejected proposal or failed commit restores simulator, VLA, teacher, policy, and RNG state;
- snapshot/restore produces identical subsequent base and teacher proposals;
- no privileged field enters any student callback.

## 4. Finish the runtime policies

### 4.1 Together

Wire `TogetherPolicy` into the native execution spine.

Final behavior:

- propose VLA and Arrow actions from the same frame;
- mix translation and rotation with a predeclared 0.5 teacher weight;
- use Arrow’s gripper command;
- fall back to the complete VLA action when the Arrow proposal is invalid;
- record the fallback reason and both unexecuted proposals.

Acceptance: a one-reset video and step log show one proposal pair and one executed action per timestep, with no hidden Arrow stepping.

### 4.2 On-Call

Wire the windowed `ProgressTracker` to the native Arrow phase-error signal.

Final behavior:

- let the VLA own the first complete 20-step progress window;
- trigger when phase error improves by less than 10% across the window, or immediately for a declared safety/gripper conflict;
- keep Arrow in control for at least 20 executed teacher steps;
- hand back only at an Arrow milestone and outside gripper dwell;
- keep proposing the VLA in shadow mode so the base action remains auditable;
- never consult evaluator success when deciding takeover or handback.

Acceptance: scripted improving, stalled, gripper-conflict, milestone, and unsafe-handback cases produce the declared transition sequence.

### 4.3 Minimal-Runtime

The current `BranchRunner` is isolated and tested, but it is not wired into a live Minimal decision.

Implement a coordinator-owned branch sandbox:

1. take one composite snapshot of LIBERO, SmolVLA, Arrow, Minimal state, and RNG;
2. evaluate all eight translation/rotation/gripper masks;
3. hold each mask for exactly 20 replanned actions;
4. after every branch restore the identical composite snapshot;
5. call a branch sufficient only when the Arrow phase advances or phase error drops by at least 10% without controller failure;
6. select the fewest teacher-owned groups, then greatest progress, then fixed mask order;
7. restore the original snapshot and execute the selected hybrid burst for real;
8. use a full-teacher burst when no branch is sufficient;
9. cap online branch decisions and record every cloned simulator step.

Acceptance: discarded branches leave the next real VLA/Arrow proposals bitwise or numerically identical to a no-branch control.

## 5. Collection and immutable derived datasets

Turn `collection.py` from an injected-runner contract into a native command.

### Master collection record

For exactly 50 On-Call attempts per task, preserve:

- frozen reset identity and attempt number;
- before/after canonical student observation digests;
- same-state VLA and Arrow proposals;
- executed policy decision and actual executor;
- takeover/handback events and Arrow phase metrics;
- raw environment result, termination, success, first-success step, and failure reason;
- model/controller/config hashes;
- latency, clipping, fallback, and perception failures.

Failures remain in the master archive. They are never silently replaced or deleted.

### Derived views

Produce the following create-only artifacts from the master manifest:

- `on_call_interventions`: successful, complete, contiguous episodes; rows where the On-Call decision actually executed the teacher action;
- `apprentice_native`: the same eligible rows encoded in SmolVLA’s native training format;
- `editor_residuals`: target `teacher_action - same_state_base_action` over the same eligible row identities;
- `trace_state_routes`: canonical state8 trajectory plus close/reopen events, with actions and images structurally absent;
- `minimal_branch_labels`: selected hybrid action, same-state base action, selected mask, and branch-cost diagnostics from offline Minimal branching.

Each derived manifest must contain parent digest, filter revision, accepted/rejected counts by task and episode, rejection reasons, schema revision, and content hash.

Acceptance: Apprentice and Editor resolve to byte-identical eligible row IDs; Trace artifacts fail validation if action or image fields are present.

## 6. Implement the real learned models

### 6.1 Apprentice

Replace the injected trainer callback in `apprentice_training.py` with a native SmolVLA/LeRobot PEFT entry point.

- reload the frozen base checkpoint for each task/seed;
- add LoRA only to the predeclared action-side modules;
- train on `apprentice_native` and validation identities only;
- save adapter weights, optimizer/scheduler configuration, training curves, seed, source manifest, and processor/checkpoint hashes;
- reload the artifact through `SmolVLAAdapter` before declaring it valid.

Do not evaluate a checkpoint chosen using the final test set.

### 6.2 Editor

Implement the actual residual model behind `ResidualModel`.

Suggested first version:

- frozen VLA;
- frozen or lightly trained image/state encoder producing a compact feature;
- two-layer MLP predicting a bounded seven-value residual;
- target `teacher_action - same_state_base_action`;
- Huber action loss plus small residual-norm regularization;
- final action `clip(base_action + residual)`.

Train separate task/seed artifacts, record the residual clipping rate, and verify the VLA parameters never receive gradients.

### 6.3 Minimal-Learned

Use the same residual architecture as Editor but a different target and artifact:

- target `selected_hybrid_action - same_state_base_action` from offline Minimal branch labels;
- no Arrow proposal or branch simulator at runtime;
- masks remain diagnostics and must not become privileged runtime inputs.

This policy must load and run with `teacher=None`.

### 6.4 Fast: the graph-local one-observation corrector

The dependency-light 448-parameter update is implemented. What remains is the real slow encoder and graph/arrow input path.

Implement a frozen slow network that produces two 32-dimensional role features:

- `h_source = encoder(observation, selected_graph_triplet, hand→source visual arrow)`;
- `h_destination = encoder(observation, selected_graph_triplet, hand→destination visual arrow)`;
- a router produces two role gates;
- only two fast matrices are writable online: `W_source` and `W_destination`, each `7 x 32`.

The runtime residual is:

`delta_action = gate_source * W_source * h_source + gate_destination * W_destination * h_destination`.

For one-observation adaptation, get the VLA and teacher proposals from the same support observation and form:

`error = teacher_action - base_action - delta_action`.

Apply one normalized outer-product update to the two fast matrices, then discard the teacher and evaluate the corrected frozen VLA. The slow encoder, router, and VLA remain frozen. Also retain a one-successful-support-episode version as a predeclared ablation; do not mix its results with the strict one-observation result.

Required implementation:

- real Torch graph/vision encoder and router;
- extraction of the chosen textual triplet and contemporaneous visual-arrow endpoints;
- one-observation support command with no retry;
- reset fast matrices before every trial;
- base fallback when the support proposal fails;
- support cost recorded separately from scored rollout cost;
- artifact checks for graph vocabulary, encoder, router, checkpoint, and feature dimensions.

Acceptance: one support observation changes exactly 448 writable values, changes no slow/VLA parameters, and produces a deterministic corrected action after restoring the identical student scene.

## 7. Finish Trace as a real action-producing policy

The state-route extraction and safety contracts exist. The missing pieces are the native geometry provider and waypoint controller.

### Geometry provider

Implement `rgbd_geometry.py` using the copied Arrow perception path:

- capture synchronized RGB and depth;
- record camera intrinsics/extrinsics and transforms;
- deproject the rendered source/destination arrow endpoints;
- return finite metric anchors in the declared robot/world frame;
- reject stale timestamps, missing depth, degenerate anchors, frame mismatch, or oracle/MuJoCo geometry.

If simulator bounding boxes were used to render the arrow, preserve that fact in provenance and describe the condition as simulator-assisted RGB-D, not vision-only.

### Waypoint controller

Implement a callable that converts the next warped `RoutePoint` into the same normalized seven-value OSC convention as Arrow and SmolVLA.

- pose-aware initial route selection;
- monotone nearest-point search;
- two-point lookahead;
- 0.5 pose mixing with the VLA;
- persistent close/reopen gripper state machine;
- declared failure instead of silent base fallback.

Acceptance: replay a known teacher route in its original scene, then a translated/rotated scene; verify frame/unit consistency, monotone route progress, gripper event order, and action bounds.

## 8. Complete CLI orchestration

Expand `cli.py` from `preflight`, contract-only `collect`, and `report` to these non-ambiguous commands:

- `preflight`: seal configuration and reset manifests;
- `canary`: one-reset native runtime proof;
- `collect`: execute exactly 50 On-Call attempts per task;
- `derive`: build immutable learned/Trace/Minimal views;
- `train-apprentice`: task/seed LoRA training;
- `train-residuals`: Editor and Minimal-Learned training;
- `train-fast-slow`: train/freeze the Fast encoder and router;
- `prepare-fast-support`: perform strict one-observation or declared episode-support adaptation;
- `evaluate`: paired evaluation over the frozen test identities;
- `report`: validate completeness and generate final tables/statistics;
- `audit`: reconcile every result to config, reset, checkpoint, and source-data hashes.

Every command needs `--dry-run`, explicit output paths, create-only outputs by default, resumable status manifests, and a clear distinction among configured, running, failed, and completed artifacts.

## 9. Complete the benchmark and paper controls

### Ranked suite rows

Evaluate eight concrete rows representing seven families:

1. Together
2. On-Call
3. Apprentice
4. Editor
5. Minimal-Runtime
6. Minimal-Learned
7. Fast-One-Observation
8. Trace

### Fixed reference controls

Add unranked but mandatory controls:

- frozen SmolVLA alone;
- Arrow teacher alone;
- the existing 50-observation task-specific PEFT method;
- Fast with zero support/update;
- Fast with graph features removed;
- Trace without graph-conditioned route selection, if graph selection is part of the final method.

### Evaluation unit

- deterministic runtime rows: 10 tasks x 10 frozen test resets;
- learned rows: the same test resets for each of three independent training seeds;
- both horizons from one rollout;
- one successful support or one-observation attempt for Fast as predeclared, with no success-conditioned retry;
- failed perception/support/load attempts count as failures rather than disappearing.

### Required report fields

- success at 280 and 1200;
- first-success step;
- teacher proposals and executed teacher actions;
- Minimal cloned branch steps;
- Fast support steps and whether adaptation occurred;
- fallbacks, invalid actions, clipping, perception failures, and load failures;
- wall-clock and policy latency;
- task/reset/training-seed pairing;
- checkpoint, configuration, split, and artifact digests.

Add hierarchical paired bootstrap confidence intervals over tasks and resets, with learned seed variation retained as a separate level. Never treat cloned branch steps or repeated deterministic runs as independent trials.

## 10. Validation ladder and launch order

Do not begin with the full 10-task experiment.

### Gate A: dependency-free suite

- Run compilation and all package tests.
- Current evidence: 44 synthetic tests pass.

### Gate B: native component smoke tests

- Load one SmolVLA checkpoint and produce one action.
- Reset one LIBERO task at a frozen identity and reproduce it after snapshot/restore.
- Produce one Arrow proposal without an environment transition.
- Produce one typed RGB-D geometry record.

### Gate C: one-reset policy canaries

- Run Together, On-Call, Minimal-Runtime, Fast-One-Observation, and Trace on one reset.
- Audit executed actions, rollback, frames, termination, and cost counters.

### Gate D: one-task pipeline

- Collect all 50 declared attempts for one task.
- Derive every dataset.
- Train one seed of each learned artifact.
- Evaluate every row on that task’s ten test resets.
- Generate a complete report and manually inspect videos/logs.

### Gate E: reproducibility

- Re-run the one-task pipeline from an empty output directory.
- Verify manifests and deterministic components reproduce; document expected nondeterminism in GPU training.

### Gate F: full study

- Train all learned task/seed artifacts.
- Evaluate the complete frozen paired matrix.
- Run artifact, leakage, coordinate-frame, and completeness audits.
- Perform an independent experimental-validity review before interpreting results.

## 11. Recommended implementation order

1. Frozen study/reset manifest.
2. Native LIBERO snapshot/restore host.
3. Native one-step SmolVLA adapter.
4. Non-stepping interruptible Arrow teacher.
5. Native transactional canary.
6. Together and On-Call live runs.
7. Minimal-Runtime branch integration.
8. Native collection and derived datasets.
9. Apprentice, Editor, and Minimal-Learned trainers.
10. Fast slow encoder plus strict one-observation adaptation.
11. Trace RGB-D provider and waypoint controller.
12. Full CLI, paired benchmark, confidence intervals, and audits.

The first five items form the critical path. Until they work, training and full benchmark code would be difficult to validate and easy to confound.

## 12. Completion checklist

The implementation can be called full when all of the following are true:

- A fresh machine/environment can reproduce the dependency setup from pinned versions.
- One command seals the experiment before data collection.
- The teacher never calls `env.step`.
- Snapshot/restore includes simulator, wrappers, VLA queue/RNG, Arrow state, policy state, and global RNG.
- The native canary passes for every policy family.
- All 500 declared On-Call attempts are retained and reconciled.
- Every learned artifact reloads with verified lineage.
- Fast can learn from exactly one same-state teacher observation and updates only its 448 fast values.
- Trace produces actual route-following actions from typed RGB-D geometry.
- All concrete rows run on the identical sealed test identities.
- Missing, failed, or invalid trials cannot be omitted from the report.
- The report includes paired uncertainty and all teacher/simulator/support costs.
- Independent tests and review find no unresolved high-severity correctness or experimental-validity issue.
