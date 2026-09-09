# Arrow policy suite

This folder contains the seven separately evaluated Arrow/SmolVLA policies:

1. `arrow_together` — continuous pose co-control with Arrow owning gripper.
2. `arrow_on_call` — progress-triggered teacher takeover and milestone handback.
3. `arrow_apprentice` — native SmolVLA adapter trained from successful On-Call teacher rows.
4. `arrow_editor` — frozen VLA plus an external residual corrector.
5. `arrow_minimal` — eight action-group masks with `runtime_oracle` and teacher-free `learned` variants.
6. `arrow_fast` — the saved graph-local 448-parameter fast-weight corrector.
7. `arrow_trace` — state-only demonstrated route guidance with explicit RGB-D geometry provenance.

## Integration boundary

The package includes dependency-tolerant adapter boundaries in
`libero_adapter.py`, `smolvla_adapter.py`, and `arrow_adapter.py`. They do not
silently construct a model or simulator: a production launcher injects the
already-loaded LIBERO environment, checkpoint-owned SmolVLA processor/model,
and an interruptible Arrow controller. Missing simulator/controller snapshot
hooks fail closed because Minimal branching and paired Fast adaptation would
otherwise be invalid.

`TransactionalCoordinator` is the only component allowed to call `env.step`.
Proposals are side-effect free and are tied to the digest of the authoritative
pre-action observation. The existing whole-recovery `automatic_ttt` teacher is
not used as a per-step teacher; evaluator utilities may be imported read-only
for canonical observations and geometry conversion.

## Data and lineage

Keep existing raw teacher actions and images in the master archive.  The Trace
view is a derived, versioned view containing only actual canonical state8 values;
it omits actions and images by construction.  `learning.py` creates the identical
eligible successful-On-Call transition view for Apprentice and Editor and can
write a content-hashed manifest.  It does not launch training: pass the rows to
the native SmolVLA trainer through `fit_with_callback`.

The current canonical state is EEF xyz, rotation-vector, and two finger-qpos
values.  It is not an object pose.  Trace therefore requires an explicit
perception/calibration provider and rejects ground-truth geometry.  The current
Arrow rendering path may still use simulator-derived bounding-box centers before
RGB-D deprojection; callers must preserve that provenance and must not describe
the resulting experiment as vision-only.

## Evaluation contract

Run one rollout to 1,200 steps and derive both the 280-step and 1,200-step
success checkpoints from it.  Use the same reset identities for every row.  Log
teacher proposals, executed teacher steps, branch steps, support-demonstration
steps, latency, clipping, fallbacks, and perception failures.  Runtime-only rows
run once; learned rows use independent training seeds without duplicating a
deterministic baseline as pseudo-replicates.

`StudyConfig` is the fail-closed configuration object for these choices.  It
requires ten disjoint validation reset identities per task and a frozen Trace
RGB-D geometry/calibration revision before a manifest can be emitted.

Example construction with injected adapters:

```python
from arrow_policy_suite import (
    ArrowAdapter, LiberoEnvironmentAdapter, SmolVLAAdapter,
    TransactionalCoordinator, TogetherPolicy,
)

env = LiberoEnvironmentAdapter(raw_env, snapshot_hook=save_env,
                               restore_hook=load_env)
smolvla = SmolVLAAdapter(inference=native_inference)
arrow = ArrowAdapter(interruptible_controller)
result = TransactionalCoordinator(env, smolvla, arrow, success_fn=success).run(
    TogetherPolicy(), max_steps=1200
)
```

The package tests are dependency-light and synthetic. Native LIBERO/MuJoCo,
SmolVLA, RGB-D geometry, and full training/evaluation runs still require a
separate canary with local checkpoints and runtime dependencies; no expensive
workload is launched by importing this package or its CLI.

## Explicit native execution

Planning remains the default. A real rollout requires the explicit
`--execute --factory module:callable` path:

```text
python -m vla_benchmarking.libero.arrow_policy_suite.cli canary \
  --config <sealed-config-or-protocol-seal.json> \
  --factory <your_module>:build_host --policy frozen_base --steps 3 \
  --run-dir <new-run-dir> --output <new-receipt.json> --execute
```

The injected factory receives `config`, `operation`, `policy_id`, `run_dir`,
and `max_steps`, and must return a `NativeHost` (or `{"host": NativeHost}`).
`native_factory.py` provides common arbitration for `frozen_base`,
`teacher_only`, and all seven policy families. `native_executor.py` performs
the fail-closed native preflight, refuses reused run directories, and records
exact git/config/checkpoint/controller hashes.

The pre-collection protocol is sealed as `arrow_policy_suite.protocol_seal.v1`.
Training and derived-artifact digests are added only after collection, avoiding
the old circular requirement that a pre-collection seal already contain a
training manifest.

For Legion, use `legion/run_arrow_policy_suite_canary.sbatch` with explicit
environment variables. Its intended short canaries are: native spine (3
steps), Together (20), On-Call (60), Minimal (20 with the policy-owned branch
budget), Fast (20 bounded support/evaluation smoke), and Trace (20 route
steps). `legion/submit_arrow_policy_suite_canary.ps1` validates the launcher
and submits it only after the exact reviewed commit/config are supplied. The
PowerShell wrapper takes `-RemoteConfig` for the Linux path (and optional
`-LocalConfig` only for local validation); it never mistakes a Windows path for
a remote file. Before a teacher factory exists, use `-EngineeringSmoke` for a
compute-node-only import/CUDA contract check. Its artifact is explicitly marked
`experiment_evidence: false` and must not be reported as a policy result.
The default factory is the concrete
`vla_benchmarking.libero.arrow_policy_suite.native_legion_factory:build_host`;
normal native runs additionally require `ARROW_SUITE_CHECKPOINT` and a
controller config (passed with `-Controller`, whose SHA is exported to the
compute job). Use `-RemoteRepoRoot` for an immutable staged release; when it
is omitted, the wrapper resolves
`$HOME/EmbodimentSemantic_runtime/releases/<ExpectedCommit>` on Legion rather
than using the shared dirty checkout.
