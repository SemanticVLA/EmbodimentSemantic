# SmolVLA trained with one target arrow

This folder identifies the distinct target-arrow LoRA treatment. It is not one
of the two protected active 2x2 policies, but it remains a separate, named
fine-tuning policy family rather than being mixed into shared workflow code.

## Recorded provenance

- Tracker identity: `target_arrow_lora_15_epochs`.
- Training input: exactly one baked arrow from `akita_black_bowl_1` to the task
  goal, `plate_1`.
- Dataset: `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_datasets_target_arrow/target_arrow_treatment`.
- Checkpoint: `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_runs/target_arrow_treatment_2026_08_18_19_47_55/checkpoints/029190/pretrained_model`.
- Evaluation input: one live bowl-to-plate target arrow.
- Recorded Lambda result: task 0 `4/10`, task 7 `1/10`, total `5/20`.
- Output: `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/eval_outputs/target_arrow_epoch15_tasks_0_7`.

No no-arrow evaluation, source commit, or Legion job is recorded for this
treatment. The reusable dataset conversion implementation remains in
`../workflows/hdf5_to_lerobot_dataset.py`, while active operator launchers fail
closed on the retired target-arrow profile.
