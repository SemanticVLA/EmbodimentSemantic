# Fine-tuned VLA policies

The protected active scope is the verified SmolVLA 2x2. See
[`ACTIVE_EXPERIMENT.md`](ACTIVE_EXPERIMENT.md) for the exact training
identities, checkpoint provenance, four evaluation cells, and their unequal
sample sizes.

Training and 2x2 evaluation commands live in `workflows/`. They call the
shared `vla_benchmarking.evaluation` package for LIBERO conditions, task/seed
matrices, context augmentation, and result contracts. Root-level command
paths remain transition wrappers for existing operators.

Every other fine-tuning policy has its own sibling folder; there is no generic
archive bucket and no policy is silently deleted:

- `smolvla_no_arrows/` — the primary protected model from jobs `1910197` and
  `1910198`;
- `smolvla_all_arrows/` — the all-arrow-trained Lambda checkpoint and its two
  tasks-0/7 evaluations;
- `target_arrow_lora/` — the one-target-arrow training treatment;
- `action_visual_lora/` — the late-visual LoRA policy from branch
  `action-visual-peft-fix`;
- `graph_text_lora/` — the graph-text fine-tuning family;
- `language_free_arrow_student/` — the separate ArrowStudent policy merged
  from `codex/arrow-student-sealed100`.

`workflows/` contains only reusable LoRA dataset, training, and paired
evaluation machinery shared by more than one of these policy folders. A
policy folder's README is the source of truth for whether that family is
active, historical, incomplete, or invalid for a causal comparison.
