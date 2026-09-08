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

The matched no-arrow expansion is organized as independent native backends:

- [`pi05/`](pi05/) — Pi0.5, LeRobot-native, two cameras, state-8, action horizon 10.
- [`openvla_oft/`](openvla_oft/) — OpenVLA-OFT, native RLDS backend, action horizon 8.
- [`octo/`](octo/) — Octo-Base 1.5, JAX/RLDS backend, action horizon 4.

Each package is configuration- and contract-complete before training is
launched. Heavy model dependencies are optional at import time. Octo keeps two
separate rows: the pinned community LIBERO checkpoint is an unmatched reference,
while the official Octo-Base initialization is the matched 500-demo baseline.
