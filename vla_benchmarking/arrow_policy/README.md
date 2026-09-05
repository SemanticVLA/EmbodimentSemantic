# Experimental arrow policy

This disposable package contains the language-free visual-arrow bridge and its
LeRobot integration. The tensor modules remain dependency-free for workstation
tests; the checkpoint adapter and staged trainer are imported only on Legion
with the pinned LeRobot environment.

The public inference path is image connector tokens plus the 32-dimensional
robot state:

```text
[batch, 64, 960] + [batch, 32]
    -> ArrowConditioningBridge
    -> 32-layer tuple of [batch, 73, 5, 64] K/V tensors
```

The 73 rows are 64 scene tokens, 8 learned visual-goal query tokens and one
state token. The bridge uses a 480-wide, six-head goal cross-attention block,
four shared pre-LN blocks and 32 bias-free 480-to-640 K/V projections. The
cache is unrotated; `route_expert_cache` applies rotary position encoding once
when the LeRobot action-expert adapter supplies matching factors.

`config.py` records the four 5k/5k/10k/10k stage inventories and AdamW
defaults. `teacher.py` extracts detached teacher targets by inverse-RoPEing
teacher key rows, pooling valid text keys into eight slots and preserving the
image/state values. Strict C/D optimizer construction requires the complete
retained student inventory (vision/connector, bridge, expert, and projections).
`training.py` dispatches the stage objectives: A uses feature plus teacher
velocity distillation, while B-D use demonstration flow matching. It also
provides the 5% warmup/cosine schedule and gradient clipping helper. `flow.py`
contains the pinned Beta(1.5, 1) timestep sampler and masked losses.
`smoke.py` runs a CPU dummy forward/backward pass. Focused tests live in
`tests/` and can be run with:

```text
python -m pytest vla_benchmarking/arrow_policy/tests -q
```

`lerobot_integration.py` contains the optional pinned-checkpoint adapter. It
uses the source `vision_model` and `connector` for the supplied agentview
image, reuses the retained action expert and action projections, and extracts
teacher caches through the pinned `SmolVLMWithExpertModel.forward` contract.
`arrow_lerobot_impl.py` is the stage entry point used by the Legion one-job
launcher. It loads the arrow and clean paired datasets, tokenizes only the
teacher instruction, and runs A/B/C/D with the atomic checkpoint store. The
student never receives task text. A full local policy snapshot can be loaded
directly; a 029190 PEFT directory is merged into the explicitly supplied local
`--base-policy`, so the run cannot silently download another revision.
The wrapper first runs a real checkpoint/data integration smoke (one finite
update through every stage) on the allocated GPU; long training starts only if
that smoke writes `integration_smoke.json`.

Every checkpoint directory contains model, optimizer, scheduler and RNG
states plus SHA-256 manifest entries. `latest.json` is replaced atomically;
the launcher mirrors the whole run root to its durable archive after every
stage and on exit. A failed stage can therefore be resumed with the same run
ID, while a completed stage is skipped by its stage marker.

LeRobot integration requires the pinned source revision and local checkpoints;
the module fails closed if those dependencies or paths are missing. The
teacher dataset must be a clean frame-for-frame pair with the arrow dataset;
the Legion wrapper requires it explicitly as `ARROW_TEACHER_DATASET_ROOT` (the
CLI also accepts `--teacher-dataset-root`).
State/action normalization is loaded from the teacher and initial checkpoint
preprocessor when present; both files must be byte-identical. If neither
checkpoint contains those statistics, the explicit `--base-policy` snapshot
is used as a recorded fallback. A non-finite loss writes a diagnostic and
stops before the optimizer step.
Simulator rollouts remain outside this training entry point.

## Legion one-job runner

`run_legion_one_job.sbatch` is the operational boundary for the full staged
experiment. It runs the package tests and bridge smoke first, then invokes
`vla_benchmarking.arrow_policy.arrow_lerobot_impl` once per stage:

```text
A_distill_bridge -> B_action_bridge -> C_joint_action -> D_full_student
```

The trainer entrypoint must accept `--stage`, `--run-root`, `--output-root`,
`--checkpoint-root`, `--dataset-root`, `--teacher-checkpoint`,
`--initial-checkpoint`, `--libero-config-path`, `--device`, and `--resume`.
It must write checkpoints below the supplied checkpoint root and return only
after that stage is durably saved. The wrapper writes a stage marker only after
the trainer returns successfully, then mirrors the whole run tree to HOME.

Set a stable `ARROW_POLICY_RUN_ID` when submitting. A later job with the same
run ID restores the HOME archive into scratch, validates completed stage
markers, and continues from the first incomplete stage. The dataset and
dataset/checkpoint variables (`ARROW_DATASET_ROOT`,
`ARROW_TEACHER_DATASET_ROOT`, `ARROW_TEACHER_CHECKPOINT`,
`ARROW_INITIAL_CHECKPOINT`, `ARROW_BASE_POLICY`) must be absolute paths.
Optional extra trainer flags are
supplied through `ARROW_TRAINER_FLAGS` and recorded in the run folder; do not
put secrets there.

The default locations are:

```text
$SCRATCH_FLASH/EmbodimentSemantic_runtime/arrow_policy/<run-id>/
$HOME/EmbodimentSemantic_archive/arrow_policy/<run-id>/
```

Every snapshot includes raw logs, checkpoints, stage markers, provenance,
GPU/CUDA diagnostics, cache directories, and code hashes. The EXIT trap also
archives interrupted or failed jobs. The wrapper refuses direct login-node
execution and requires an exact clean checkout commit.
