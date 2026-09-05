# SmolVLA trained without arrows (active protected treatment)

Training provenance: source commit `8579b62`, Legion training job `1910197`,
checkpoint `029190`. Evaluation job `1910198` recorded 43/100 without arrows
and 30/100 with live arrows. Adapter SHA-256 is
`80b3c23fc3987530d57766ab45ed33db918f08983739139c1ff0397184cc7092`; the
training manifest SHA-256 is
`95e376aff504265bea2bb53e63cc221fb42d7baa01dd6c3810317de85875c391`.

Training archive:
`/home/hjaber/EmbodimentSemantic_archive/runs/legion_no_arrow_lora_full_s1000_v1_no_arrow_treatment_1910197`.
Evaluation archive:
`/home/hjaber/EmbodimentSemantic_archive/eval/legion_no_arrow_trained_live_vs_none_s1000_ep10_v1_no_arrow_treatment_1910198`.

The implementation is shared with the paired workflow under `../workflows/`.

## Sealed evaluation detail

The final Legion evaluation was job `1910198`, with seed `1000`, ten episodes
per task, and tasks `0` through `9`:

| Evaluation input | Task successes (0–9) | Total |
| --- | --- | --- |
| Live all-object arrows | 8, 4, 3, 4, 0, 0, 7, 1, 3, 0 | 30/100 |
| No arrows | 9, 9, 6, 4, 1, 1, 6, 4, 3, 0 | 43/100 |

These are final archived results, not a claim that the current checkout has
re-run the 100-cell evaluation.
