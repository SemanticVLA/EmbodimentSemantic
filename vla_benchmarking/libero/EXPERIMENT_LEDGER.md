# LIBERO VLA Benchmark: Canonical Experiment Ledger

This ledger intentionally tracks only the two canonical results and their two
expanded 500-episode regressions. Earlier campaign history remains recoverable
from Git commit `bf1a6d379dde6540825bc2158d5f808a8598361b` and the immutable
Legion archives; it is not part of the active handoff.

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

- Legion job: `1921211` (`RUNNING`).
- Purpose: confirm that cleanup preserved the 87% controller behavior.
- Suite: `sealed_randomized`; tasks 0-9; 50 episodes per task; seeds 1000-1049.
- Planned: 500 cells.
- Last audited partial result: **123/123 completed successes**; task 0 50/50,
  task 1 50/50, task 2 23/50 with one cell running, and tasks 3-9 not started.
  This is partial, not a final score.
- Immutable execution commit: `d044a5d672092d4322cf7395d91cc1b0085ef496`.
- Canonical configuration SHA-256:
  `37497fd0b2f60346b9ffd1501ccc743046c7fa2370ef6fa9531a7204f69cc044`.
- Archive target:
  `/home/hjaber/EmbodimentSemantic_archive/grasp_controller/canonical_grasp_sealed500_d044a5d_r2_20260905T1752Z_1921211`.

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
