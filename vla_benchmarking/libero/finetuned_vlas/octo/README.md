# Octo-Base 1.5 LIBERO integration

This directory contains the Octo-specific contracts for the no-arrow VLA
comparison.  It has two separate artifacts:

* `community_libero_190k/`: direct evaluation of the largest available
  community LIBERO checkpoint;
* `no_arrows/`: matched retraining from the official `rail-berkeley/octo-base-1.5`
  pretraining checkpoint on the canonical 500 no-arrow demonstrations.

Neither path downloads or trains during import.  Use an isolated Octo/JAX
runtime on the A40 and run the two-update memory smoke test before any full
job.

## Checkpoint provenance

The direct-evaluation artifact is
`cyrusneary/octo-finetuned-libero` at revision
`f8a0888cfa7ef3be072417eb012339464a9bb6dc`, nested checkpoint:

```text
2025-06-21_octo_base_1p5_libero_finetune/octo_finetune/experiment_20250621_094538/190000/default/checkpoint
```

The matched run starts from `rail-berkeley/octo-base-1.5` revision
`ee3c10e8edd6ce2e8b1e8744d3c6fba4097bed48`,
`300000/default/checkpoint`.

The community checkpoint is an unmatched multi-suite artifact.  It is a
reference row only; it is not the controlled Octo baseline.  Community
preflight binds the immutable selected checkpoint subtree and the checkpoint's
own `dataset_statistics.json` (432 trajectories / 52,970 transitions); it does
not accept a caller-provided digest as proof of that provenance.

## Input and action contract

Octo receives one stored agent-view image (`image_primary`) at 256x256 and the
language instruction.  It receives neither the wrist image nor the 8-D
proprioceptive state.  The community checkpoint uses a two-frame observation
window; the first query repeats the first canonical frame with a false padding
mask, then the adapter retains the latest two frames.  The matched training
contract uses a one-frame window.

Offline source frames are rotated exactly once by 180° (`flip` over height and
width) by `dataset.serialize_episode`, which records frame provenance.  Live
evaluation must pass the evaluator's already-canonical frame and must not
rotate it again.

The native Octo output is `[batch=1, horizon=4, action_dim=7]`.  Dimensions 0–5
use Gaussian action mean/std.  The gripper follows:

```text
libero_gripper = 1 - 2 * octo_open
```

The adapter owns PRNG splitting and requires `reset(seed)` plus an instruction
before sampling.  `adapter.py` is intentionally a thin native wrapper and
does not import JAX or Octo until `from_pretrained` is called.

## Matched training contract

* 500 episodes / 62,250 source timesteps;
* 15 epochs; optimizer updates are derived at preflight as
  `ceil(verified_transitions * 15 / effective_batch)`;
* seed 1000;
* full Octo transformer and diffusion action head trainable;
* T5 language encoder frozen;
* cosine learning rate: peak `3e-4`, 2,000-update warmup, decay to zero;
* AdamW weight decay `0.01`, gradient clip `1.0`;
* native image augmentation; no goal image, wrist image, or proprioception.

The sealed A40 microbatch ladder preserves effective batch 32:

```text
microbatch 32 / accumulation 1
microbatch 16 / accumulation 2
microbatch  8 / accumulation 4
```

Choose the first measured option at or below 43.2 GB peak allocation during
the two-update smoke test.  Do not add an unplanned fallback or freeze modules
after observing memory.

## Guarded entrypoints

`preflight.py` validates a completed dataset fingerprint/manifest and the
mode-specific pinned checkpoint suffix without downloading or initializing a
model.  `dataset.write_rlds_source_jsonl` followed by
`dataset.build_tfds_dataset` materializes a real episode-structured TFDS/RLDS
dataset and writes `octo_materialization.json`; `native_finetune_config.py`
registers that builder before the pinned Octo `make_single_dataset` call.

`train.py` and `eval.py` run this preflight by default and print commands that
target the checked-in `native_train.py` / `native_eval.py` bridges.  Actual
execution requires the explicit `--execute` flag.  Training additionally
requires a two-update A40 memory receipt (`--memory-measurements`) and selects
the first fitting entry in the sealed microbatch ladder, preserving effective
batch 32; no unmeasured fallback is permitted.  `--preflight-only` never
launches anything.

Community evaluation is intentionally bound to the selected checkpoint's own
`dataset_statistics.json` and rejects a caller manifest.  Matched training and
evaluation require the completed local no-arrow manifest.  The native adapter
always derives action statistics from the loaded checkpoint unless an explicit
caller value is supplied and verified equal.

For matched training, `train.py` resolves the checked-in
`native_finetune_config.py` overlay and emits the pinned Octo
`scripts/finetune.py` command.  The `--config.num_steps` value is generated
from the verified manifest transition count by
`manifest.compute_optimizer_updates`; no launch wrapper may override it with
a hand-entered schedule constant.  Supply `--entrypoint` as the absolute path
to the pinned Octo repository's `scripts/finetune.py` when printing or
executing that command.
