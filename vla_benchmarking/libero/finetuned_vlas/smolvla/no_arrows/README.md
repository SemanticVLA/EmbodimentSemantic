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

## Legion training launcher

Submit `legion/run_training.sbatch` directly through Slurm. It supports a
two-step `smoke` scope and the sealed 29,190-step `full` scope, requires an
exact clean release commit, and archives the checkpoint, manifest, hashes, and
Slurm logs. A smoke may explicitly use `reuse_verified` data mode after the
launcher re-verifies the byte-bound full dataset pair; full training always
uses the standard preparation contract.

## Sealed evaluation detail

The final Legion evaluation was job `1910198`, with seed `1000`, ten episodes
per task, and tasks `0` through `9`:

| Evaluation input | Task successes (0–9) | Total |
| --- | --- | --- |
| Live all-object arrows | 8, 4, 3, 4, 0, 0, 7, 1, 3, 0 | 30/100 |
| No arrows | 9, 9, 6, 4, 1, 1, 6, 4, 3, 0 | 43/100 |

These are final archived results, not a claim that the current checkout has
re-run the 100-cell evaluation.

## Expanded sealed-randomized regression

`../workflows/run_no_arrow_sealed_eval.py` is the dedicated one-condition
runner for checking this protected adapter with the organized evaluator.  It
keeps visual arrows disabled and supports only a two-cell smoke scope (tasks 0
and 4, one episode each) or a 500-episode full scope (50 episodes for each task,
with deterministic episode seeds 1000 through 1049). The historical paired
100-cell workflow remains sealed and unchanged.

The 500-cell result is an expanded current-code regression, not a replacement
for or exact replication of job `1910198`: the evaluator revision and sample
count differ.  Raw results and their immutable run manifest must be archived
under a new experiment identity.
