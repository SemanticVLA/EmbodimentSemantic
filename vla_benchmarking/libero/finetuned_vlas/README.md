# Fine-tuned VLA policies

This folder retains exactly two action-only SmolVLA profiles:

- [`smolvla/no_arrows/`](smolvla/no_arrows/) — the protected no-arrow-trained
  model from jobs `1910197`/`1910198` (historical 43/100 result).
- [`smolvla/target_arrow_only/`](smolvla/target_arrow_only/) — the matched
  profile trained with exactly one subject-to-task-goal arrow per frame.

Both cases use the centralized [`smolvla/workflows/`](smolvla/workflows/)
implementation and the shared `vla_benchmarking.libero.evaluation` package.
The retired all-arrow, graph-text, visual-LoRA, and ArrowStudent trees are not
operator-reachable in this checkout.
