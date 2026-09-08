# LIBERO VLA Benchmark: Canonical Experiment Ledger

This ledger tracks the canonical results, their expanded regressions, and the
recovered archived matrix results below. Earlier campaign history remains
recoverable from Git commit `bf1a6d379dde6540825bc2158d5f808a8598361b` and the
immutable Legion archives. A `PARTIAL` or `RUNNING` record is not a completed
benchmark score.

## Arrow grasp controller — final 87/100

- Treatment: `failure_opening40_retreat80`, now the canonical arrow grasp controller.
- Legion job: `1920556` (`FINAL`, 100/100 terminal).
- Suite: `sealed_randomized`; tasks 0-9; 10 episodes per task.
- Result: **87/100 (87%)**.
- Per task: T0 10/10, T1 10/10, T2 10/10, T3 10/10, T4 7/10,
  T5 10/10, T6 8/10, T7 10/10, T8 10/10, T9 2/10.
- Executed source: `b4fb87759ae3a1ea2cd518cd201a1a737bb14e80`.
- Canonical configuration SHA-256:
  `37497fd0b2f60346b9ffd1501ccc743046c7fa2370ef6fa9531a7204f69cc044`.
- Archive:
  `/home/hjaber/EmbodimentSemantic_archive/molmo_failure_sealed100/molmo_failure_sealed100_fa1ae83_1920556`.

## SmolVLA trained without arrows — final 43/100

- Training job: `1910197`; checkpoint step `029190`.
- Evaluation job: `1910198` (`FINAL`, 100/100 terminal for this condition).
- Evaluation input: no arrows.
- Result: **43/100 (43%)**.
- Per task: T0 9/10, T1 9/10, T2 6/10, T3 4/10, T4 1/10,
  T5 1/10, T6 6/10, T7 4/10, T8 3/10, T9 0/10.
- Adapter SHA-256:
  `80b3c23fc3987530d57766ab45ed33db918f08983739139c1ff0397184cc7092`.
- Adapter configuration SHA-256:
  `6382c39c1df2e4ccfc43946b36aa014abb73e1767366f009999415981b01efb1`.
- Training configuration SHA-256:
  `89a570fb1d07e93ec16adde1e78f18b0f0f7c0148da7896a7b5ba257797658f8`.
- Training manifest SHA-256:
  `95e376aff504265bea2bb53e63cc221fb42d7baa01dd6c3810317de85875c391`.
- Training archive:
  `/home/hjaber/EmbodimentSemantic_archive/runs/legion_no_arrow_lora_full_s1000_v1_no_arrow_treatment_1910197`.
- Evaluation archive:
  `/home/hjaber/EmbodimentSemantic_archive/eval/legion_no_arrow_trained_live_vs_none_s1000_ep10_v1_no_arrow_treatment_1910198`.

## Arrow grasp controller — expanded 500-cell regression

- Original Legion job: `1921211` (time-limited before all seeds ran).
- Recovery Legion job: `1922090`; merge job: `1922126`.
- Purpose: confirm that cleanup preserved the controller behavior while
  completing only cells absent from the original terminal matrix record.
- Suite: `sealed_randomized`; tasks 0-9; 50 episodes per task; seeds 1000-1049.
- Planned: 500 cells.
- Final merged result: **429/500 (85.8%)**, 500/500 terminal.
- Per task: T0 50/50, T1 50/50, T2 50/50, T3 48/50, T4 34/50,
  T5 49/50, T6 39/50, T7 49/50, T8 48/50, T9 12/50.
- The recovery executed only the nine cells absent from the original matrix
  JSONL: task 6 seeds 1000, 1032, 1039, 1046; task 9 seeds 1043, 1046,
  1047, 1048, 1049. Task 4, task 8, and task 9 seed 1028 were already
  terminal and were not merged or rerun.
- Immutable execution commit: `d044a5d672092d4322cf7395d91cc1b0085ef496`.
- Canonical configuration SHA-256:
  `37497fd0b2f60346b9ffd1501ccc743046c7fa2370ef6fa9531a7204f69cc044`.
- Archive target:
  `/home/hjaber/EmbodimentSemantic_archive/grasp_controller/canonical_grasp_sealed500_d044a5d_r2_20260905T1752Z_1921211`.
- Merged result root:
  `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/grasp_controller/runs/canonical_grasp_sealed500_d044a5d_r2_20260905T1752Z_1921211/results/sealed_randomized`.

## SmolVLA trained without arrows — expanded 500-cell regression

- Legion job: `1921360` (`RUNNING`).
- Purpose: evaluate the protected 43% adapter through the current shared
  evaluator on the larger seeded sealed-randomized sample.
- Input: no arrows; both training cameras retained.
- Suite: `sealed_randomized`; tasks 0-9; 50 episodes per task; seeds 1000-1049.
- Planned: 500 cells.
- Last audited state: model and adapter loaded; no terminal episode result was
  yet emitted, so no partial score is inferred.
- Immutable execution commit: `43e8cbd5713311a75eb53b396cbdbd36bed85d78`.
- Adapter SHA-256:
  `80b3c23fc3987530d57766ab45ed33db918f08983739139c1ff0397184cc7092`.
- Training manifest SHA-256:
  `95e376aff504265bea2bb53e63cc221fb42d7baa01dd6c3810317de85875c391`.
- Archive target:
  `/home/hjaber/EmbodimentSemantic_archive/no_arrow_vla_eval/noarrow_sealed500_shared_43e8cbd_full_1921360`.

## Recovered archived matrix results

These records were recovered from immutable Legion artifacts on 2026-09-07.
They were absent from the prior active handoff, so they are recorded here with
their terminal state and provenance rather than silently promoted to canonical
results. Smoke/canary-only artifacts are intentionally omitted.

### OpenVLA original checkpoint — recovered cells

- Vanilla cell: job `1921836`, **83/100 (83%)**, 100/100 rows present.
- Per task: T0 9/10, T1 8/10, T2 9/10, T3 10/10, T4 8/10,
  T5 7/10, T6 9/10, T7 8/10, T8 6/10, T9 9/10.
- Sealed-randomized cell: repair job `1921973`, **33/100 (33%)**, 100/100
  rows present.
- Per task: T0 5/10, T1 7/10, T2 0/10, T3 2/10, T4 6/10,
  T5 2/10, T6 7/10, T7 0/10, T8 4/10, T9 0/10.
- Both jobs exited non-zero after emitting their respective cell artifacts,
  so these are `PARTIAL` recovered cells, not one atomic 200-episode matrix.
  Job `1921973` explicitly preserved the vanilla JSONL from `1921836`.
- Model: `openvla/openvla-7b-finetuned-libero-spatial`, revision
  `962318cec55ac10993ff0f5f43eda9a270b4c873`.
- Execution commits: `e9b7f34875d2ebe4d837be3f44cdf9231b370c9b` (vanilla),
  `c1cb71c82df2fb0eb6db58cd5003b98ef66cdd5c` (sealed repair).
- Archives:
  `/home/hjaber/EmbodimentSemantic_archive/runs/vla_eval_matrix_1921836`;
  `/home/hjaber/EmbodimentSemantic_archive/runs/vla_eval_matrix_1921973`.

### Pi0.5 — standalone 200-episode matrix

- Standalone final job `1922576`: both cells were terminal, **200/200 rows**.
  Sealed-randomized: **51/100 (51%)**; vanilla: **82/100 (82%)**.
- Sealed per task: T0 4/10, T1 6/10, T2 10/10, T3 7/10, T4 8/10,
  T5 2/10, T6 10/10, T7 0/10, T8 4/10, T9 0/10.
- Vanilla per task: T0 10/10, T1 8/10, T2 10/10, T3 10/10, T4 8/10,
  T5 4/10, T6 9/10, T7 8/10, T8 7/10, T9 8/10.
- All 67 failures were `episode_step_budget` terminations; no action-range or
  tokenizer/runtime failure occurred in the final matrix.
- Model: `lerobot/pi05_libero_finetuned_v044`, revision
  `8e174154ef5f6c60a8da12ae99c303d8963138c1`; tokenizer
  `google/paligemma-3b-pt-224`, revision
  `35e4f46485b4d07967e7e9935bc3786aad50687c`.
- Source commit: `11d192d6f29f9eade9e87ac7518a395e044c4531`.
- Dataset manifest SHA-256:
  `da408fc82f9c4b4208929992a559a1169db639551d37f0a36c7dc7be9d7feb9e`.
- Archive: `/home/hjaber/EmbodimentSemantic_archive/runs/vla_eval_matrix_1922576`.

### SmolVLA — final standalone 300-episode matrix

- Full job `1922573`: **FINAL**, 300/300 episodes terminal. The full-matrix
  postcondition passed.
- Base vanilla: **78/100 (78%)**, per-task successes
  `[8, 10, 7, 8, 8, 7, 8, 8, 9, 5]`.
- Base sealed-randomized: **6/100 (6%)**, per-task successes
  `[2, 0, 0, 1, 3, 0, 0, 0, 0, 0]`.
- No-arrow fine-tuned vanilla: **10/100 (10%)**, per-task successes
  `[6, 0, 0, 0, 1, 3, 0, 0, 0, 0]`.
- Combined matrix result: **94/300 (31.3%)**. This is the aggregate across
  the three explicitly named cells, not a replacement for the per-cell scores.
- Terminal archive record: `/home/hjaber/EmbodimentSemantic_archive/runs/smolvla_eval_matrix_1922573/terminal.json` (return code 0).
- Full-matrix source commit: `2eafe0939ad1ff6d43d2e0271c8a0a519f3ce477`;
  base revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`; adapter SHA-256
  `80b3c23fc3987530d57766ab45ed33db918f08983739139c1ff0397184cc7092`.
- Runtime root: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/runs/smolvla_eval_matrix_1922573`.

### Arrow grasp controller — vanilla 100 (single evaluation; in progress)

- This is **one vanilla-100 evaluation** recorded across two non-duplicating
  SLURM segments: initial job `1923501` (task 0) and recovery job `1923521`
  (tasks 1-9). They are not separate ledger experiments.
- The initial submission's comma-delimited `--export` syntax reduced
  `GRASP_TASK_IDS` to task `0`; its verified archive is preserved at
  `/home/hjaber/EmbodimentSemantic_archive/grasp_controller/canonical_grasp_vanilla100_11d192d6_1923501`.
- Recovery job `1923521` is continuing tasks 1-9 on `compute-6-11` without
  rerunning task 0. Its archive target is
  `/home/hjaber/EmbodimentSemantic_archive/grasp_controller/canonical_grasp_vanilla_t1_t9_11d192d6_1923521`.
- Final merged result at the last audit (`2026-09-07T16:01:47Z`):
  **100/100 cells are terminal**, with **86 successes** (**86%**). The final
  outcome breakdown is 13 benchmark failures and 1 unresolved input failure.
  Per task: T0 10/10, T1 10/10, T2 10/10, T3 10/10, T4 7/10, T5 10/10,
  T6 8/10, T7 10/10, T8 10/10, T9 1/10. The two SLURM segments (initial job
  `1923501` and recovery job `1923521`) together form this single final
  vanilla-100 evaluation.
- The three T4 failures did execute and produced preserved manifests:
  seeds 1001 and 1008 had no valid grasp candidates after clearance filtering;
  seed 1005 lost retention after lift (`post_lift_retention`) and its retry had
  no candidates. The episode runner marked these rows completed, but the
  evaluator was not called; under the benchmark success metric, these are
  terminal controller failures, not missing episodes, and are not rerun.
- Scope remains vanilla only; no sealed-randomized episodes were included.
- Release commit: `11d192d6f29f9eade9e87ac7518a395e044c4531` in isolated release
  `/home/hjaber/EmbodimentSemantic_runtime/releases/11d192d6f29f9eade9e87ac7518a395e044c4531`.
- Launcher preflight passed on the allocated A40: CUDA/BF16, pinned MolmoPoint
  runtime, LIBERO, and the canonical policy lock all validated before rollout.
- Canonical configuration SHA-256:
  `01d42abb3c1594bb67e81fad6d21ed035674652dbd3f7ee560aa887616252d84`.

## Reconciliation of unscored attempts

The following archived attempts were checked and deliberately carry no score:
`1921798`, `1921801`, `1921806`, `1921811`, `1921821`, `1922187`, `1922190`,
`1922551`, `1922553`, `1922554`, `1922559`, and `1922561`. They had no terminal
evaluation rows, incomplete archives, tokenizer revision validation failure, an
early action-range failure, or a failed pre-evaluation stage.
