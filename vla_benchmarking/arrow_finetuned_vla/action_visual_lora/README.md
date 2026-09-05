# Action-visual LoRA (`action_visual_lora_v1`)

This folder contains the policy-specific implementation and retrospective
evaluation for the action-visual LoRA treatment. It is retained as a named
policy family and is not the active SmolVLA 2x2 default.

## Provenance

- Training: Legion job `1911789`, source commit
  `98f9e295fe05400cbbce6d1e9cf500222327dac5`, trained on the no-arrow
  `control` dataset, seed `1000`, batch size `32`, LoRA rank `16`, epoch-15
  checkpoint `029190`.
- Evaluation: Legion job `1911790`, two no-arrow cells, comparing the new
  action-visual adapter with the historical action-only adapter.
- New action-visual cell: `26/100`, per-task successes
  `2, 8, 0, 1, 0, 0, 3, 4, 8, 0`.
- Historical action-only cell: `43/100`, per-task successes
  `9, 9, 6, 4, 1, 1, 6, 4, 3, 0`.

Job `1911790` exited non-zero because the two cells did not use identical
reset/randomization identities. The individual outputs are complete and
archived, but the 26/100 versus 43/100 difference is not a valid paired
comparison and must not be reported as an improvement claim.

The separate intermediate job `1912060` evaluated checkpoint `007784` with
live arrows at `15/100`; its no-arrow cell was cancelled before any episode.

## Entry points

- `lora_finetuning_policy.py`: versioned LoRA target policy registry.
- `run_lora_policy_pair_eval.py`: matched policy evaluator.
- `legacy_action_only_evidence.py`: immutable historical evidence builder.
- `submit_legion_action_visual_lora_pilot.sh`: pinned Legion setup/train/eval
  chain. It uses the shared `vla_benchmarking.evaluation` implementation.

The original branch and durable Legion archives remain the source of truth for
historical artifacts; this folder is a path-aware source copy for handoff.
