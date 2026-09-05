# SmolVLA fine-tuning policy contract

The retained fine-tuning policy is `action_only_lora_v1`: rank-16 LoRA on
the action expert q/v projections and action/state projections. Vision and
VLM text weights remain frozen.

The two active data profiles are separate experimental conditions:

- `no_arrow_treatment`: clean LIBERO frames, evaluated with `none` and
  `visual_arrows`.
- `target_arrow_treatment`: exactly one subject-to-task-goal arrow in each
  agentview frame, evaluated with `none` and `visual_goal_arrow`.

Both profiles share the same base revision, seed, batch size, cameras, task
schedule, and centralized training/evaluation workflow. Their manifests record
the profile, dataset variant, adapter inventory, and paired evaluation
contract. Retired graph, all-arrow, visual-LoRA, and ArrowStudent profiles are
not accepted by the active workflow.
