# SmolVLA target-arrow-only profile

This is the second retained SmolVLA case. It uses the same revision-pinned
SmolVLA base, action-only rank-16 LoRA policy, seed `1000`, batch size `32`,
15 epochs, 1,946 updates per epoch, cameras, and LIBERO task/demo schedule as
the protected no-arrow case. The only training-data change is
`target_arrow_treatment`: each agentview frame contains exactly one
subject-to-task-goal arrow.

Training and evaluation are centralized in `../workflows/`. Use
`run_smolvla_pipeline.sh <setup|dry|smoke|full|resume|eval> --profile target-arrow`
for the selected operation. The matched evaluation uses the same adapter with
two cells: `visual_goal_arrow` and `none`, with `DISABLE_VISUAL_PROMPT_HINT=1`
for the target-arrow cell.

No checkpoint is claimed by this folder until a fresh training run completes
and its immutable manifest and adapter audit pass.

## Legion launchers

Submit `legion/run_training.sbatch` directly for either the two-step `smoke`
gate or the sealed 29,190-step `full` run. `reuse_verified` is a smoke-only
scope that re-verifies an existing complete target-arrow dataset before
training. The launcher requires an exact clean release commit and preserves
checkpoint, manifest, hashes, and Slurm provenance.

After training, submit `legion/run_target_arrow_pair_eval.sbatch`. Its explicit
`smoke` protocol evaluates tasks 0 and 4 once with and without the target
arrow; `full` requires checkpoint 029190 and evaluates all ten tasks with ten
episodes each. The training manifest authenticates the permitted checkpoint
and schedule in both protocols.
