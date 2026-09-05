# SmolVLA trained with all arrows (active protected treatment)

Training provenance is the tracker identity
`all_arrows_lora_epoch_15_checkpoint`, with the recorded Lambda artifact
`treatment_2026_08_17_19_42_03/.../029190`. The source record does not provide
a commit or Legion job. Its tasks-0/7 Lambda evaluation recorded 11/20 with
live arrows and 5/20 without arrows. This 20-cell result is not comparable as
an aggregate to the no-arrow-trained treatment's sealed 100-cell Legion run.

The implementation is shared with the paired workflow under `../workflows/`.

## Recorded evaluation detail

The only recorded result for this policy is a Lambda probe over tasks `0` and
`7`, ten episodes per task. It produced `11/20` with live all-object arrows
and `5/20` without arrows. Per-task counts are `task 0 = 6/10, task 7 = 5/10`
with arrows, and `task 0 = 5/10, task 7 = 0/10` without arrows. No result was
recorded for tasks `1–6` or `8–9`, so this policy must not be described as a
sealed 100-cell result.
