# Pi0.5 no-arrow integration

This package defines the import-safe contract for the matched Pi0.5 baseline.
It does not launch training.  The evaluation runtime is the already fine-tuned
Hub artifact `lerobot/pi05_libero_finetuned_v044`; this experiment does not
fine-tune it again.
with two 256×256 cameras (`agentview` and wrist), 8-D proprioception, 7-D
actions, and the checkpoint's native 50-action chunk.

The sealed training recipe is action-expert-only, BF16, gradient checkpointing,
microbatch 1 with accumulation 32, 15 epochs / 29,180 optimizer updates,
seed 1000, AdamW at `2.5e-5`, 1,000 warmup updates, cosine decay, and
checkpoint frequency 1,946 updates.  Resolve and record an immutable base
revision before training; the placeholder `main` is intentionally marked
unverified in the dry-run artifact.

Run the dependency-free contract check with:

```powershell
python -m vla_benchmarking.libero.finetuned_vlas.pi05.cli --dry-run
```

The actual LeRobot import occurs only in `Pi05Adapter.load`.

For shared-plan evaluation, pass `--plan` to `pi05.eval`.  The command then
uses `evaluation.native_vla_eval`, which binds the plan receipts and common
environment-step budget.  Because LeRobot's `make_policy` requires the
checkpoint-owned `config.json`, `policy_preprocessor.json`, and
`policy_postprocessor.json` are loaded directly; no policy-config factory is
required.  An explicit `--policy-config-factory module:callable` remains an
optional override for runtimes that need custom dataset metadata.

The pinned LeRobot 0.5.2 trainer counts `--steps` as loop batches and has no
gradient-accumulation CLI field.  `train.py` therefore emits the checked-in
`pi05.trainer` wrapper, which divides each loss by 32 and performs optimizer,
gradient-clip, scheduler, and checkpoint update boundaries every 32
micro-batches.  For the sealed 62,250-timestep source this is 933,760 loop
steps (29,180 optimizer updates × 32 accumulation); the receipt records both
values.  The wrapper rejects a partial final accumulation window.

The supported LeRobot command uses `--policy.path=...` and policy-scoped
Pi05Config fields (`optimizer_lr`, `scheduler_warmup_steps`, and so on).  It
does not use the removed `--policy.pretrained_path` or
`--accelerator.gradient_accumulation.steps` spellings.
