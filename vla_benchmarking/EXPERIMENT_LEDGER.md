# SmolVLA Arrow Experiments: Authoritative Takeover Ledger

This is the human-readable source of truth for the active SmolVLA LoRA
experiments. Its machine-readable companion is
`evaluation_results_tracker.json`. **Both files must always describe the same
state.**

The always-applied startup rule enforcing this requirement is
`.cursor/rules/smolvla-experiment-updates.mdc`.

**Analysis scope:** epoch 15 checkpoints only (`029190`, 29,190 steps). Do not
add any other checkpoints unless the user explicitly changes the scope.

**Last active-run audit represented in both files:** 2026-08-30 19:59:06 UTC.
The active forward run is on PoliTO Legion. The last preserved Lambda-only audit
is 2026-08-20 10:11:17 UTC; its process IDs are historical snapshots, not live
state.

## Mandatory rule whenever the user asks for an update

Before answering any request such as "updates?", "where are we?", "how many?",
"is it done?", or "results so far?", the active thread must complete this exact
transaction:

1. Connect to each active execution host and refresh every running training and
   evaluation job. Current forward runs use Legion through the mp4 gateway.
2. Inspect process state, the latest training log/checkpoints, completed episode
   artifacts, and `eval_info.json` when present.
3. Update `evaluation_results_tracker.json`, including `audited_at_utc`, job
   status, progress, PIDs, per-task counts, and whether each score is partial or
   final.
4. Update this ledger with the identical audit time and identical facts.
5. Parse the JSON and check both files for contradictory statuses or scores.
6. Only after steps 1–5, answer the user with the new numbers.

If an active execution host cannot be reached, do not present its snapshot as
current. Record and report that the refresh failed, retain the last successful
audit timestamp, and label every shown count stale.

## Terminology that must not be mixed up

Never use the word **control** by itself. It has referred to two different
things:

- **Frozen base:** the pinned `HuggingFaceVLA/smolvla_libero` policy with no LoRA
  fine-tuning.
- **No-arrow LoRA:** a LoRA fine-tuned for 15 epochs on normal LIBERO images with
  no arrows. Its dataset directory happens to be named `control`, but this model
  is not the frozen base.

Every result must explicitly state both:

- **Trained on:** the image condition baked into the fine-tuning dataset.
- **Evaluated with:** the live overlay shown during LIBERO rollout.

Image conditions:

- **All arrows:** multiple arrows from `akita_black_bowl_1` to selected visible
  spatial-relation objects.
- **Target arrow:** exactly one arrow from `akita_black_bowl_1` to the task goal,
  `plate_1`.
- **No arrows:** no synthetic arrow overlay.

## Shared experimental contract

- Base revision: `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`
- LoRA rank: 16
- Batch size: 32
- Seed: 1000
- Epoch-15 checkpoint: `029190`
- Save interval: 1,946 steps, one checkpoint per epoch
- Evaluation cameras: `agentview,robot0_eye_in_hand`
- Observation size: 256x256
- Main checkpoint probe: tasks 0 and 7, 10 episodes per task
- Task 0/7 environment variation: canonical LIBERO initial-state variation and
  existing static prompt overrides. These tasks do not receive this repository's
  custom object swaps, removals, or camera-selection interventions. Do not call
  these probes custom randomized-scene evaluations.

## Exact training runs

### A. All-arrow LoRA — epoch 15 selected

- ID: `all_arrows_lora_epoch_15_checkpoint`
- Trained on: all arrows baked into the training images
- Status: complete
- Selected checkpoint:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_runs/treatment_2026_08_17_19_42_03/checkpoints/029190/pretrained_model`
- Dataset:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_datasets/treatment`
- Dataset size: 500 episodes, 62,250 frames
- The source run later continued beyond epoch 15. This ledger uses only
  checkpoint `029190`.

### B. Target-arrow LoRA — 15 epochs

- ID: `target_arrow_lora_15_epochs`
- Trained on: exactly one baked bowl-to-plate target arrow
- Status: complete
- Final checkpoint:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_runs/target_arrow_treatment_2026_08_18_19_47_55/checkpoints/029190/pretrained_model`
- Dataset:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_datasets_target_arrow/target_arrow_treatment`
- Dataset size: 500 episodes, 62,250 frames

### C. No-arrow LoRA — 15 epochs

- ID: `no_arrows_lora_15_epochs`
- Trained on: normal LIBERO images with no arrows
- Status: complete, 29,190 / 29,190 steps
- Run directory:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_runs/no_arrow_treatment_2026_08_19_13_08_20`
- Expected final checkpoint:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_runs/no_arrow_treatment_2026_08_19_13_08_20/checkpoints/029190/pretrained_model`
- Dataset:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_datasets/control`
- Dataset size: 500 episodes, 62,250 frames
- Launcher PID: not running (not a long-lived process)
- Active training PID: not running
- Active worker PIDs: none
- Latest checkpoint observed: 029190
- Log:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/no_arrow_treatment_launch_2026_08_19_13_08_20.log`

## Completed final evaluations

### 1. All-arrow LoRA evaluated with all arrows

- Model: all-arrow LoRA
- Trained on: all arrows
- Evaluated with: all live arrows
- Checkpoint: epoch 15 (`029190`)
- Task 0: 6 / 10 successes
- Task 7: 5 / 10 successes
- **Final score: 11 / 20 (55%)**
- Output:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/checkpoint_probe_029190_tasks_0_7_treatment`

### 2. Frozen base evaluated with all arrows

- Model: pinned frozen base, no LoRA
- Trained on: not fine-tuned
- Evaluated with: all live arrows
- Task 0: 0 / 10 successes
- Task 7: 0 / 10 successes
- **Final score: 0 / 20 (0%)**
- Output:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/checkpoint_probe_base_tasks_0_7`
- Critical clarification: this is not a frozen-base/no-arrow result.
  `visual_relation_audit.jsonl` records `condition=visual_arrows`.

### 3. Target-arrow LoRA evaluated with one target arrow

- Model: target-arrow LoRA
- Trained on: one target arrow
- Evaluated with: one live target arrow
- Checkpoint: epoch 15 (`029190`)
- Task 0: 4 / 10 successes
- Task 7: 1 / 10 successes
- **Final score: 5 / 20 (25%)**
- Evaluation PID: not running (complete)
- Output:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/eval_outputs/target_arrow_epoch15_tasks_0_7`
- Overlay evidence: `visual_relation_audit.jsonl` records
  `visual_goal_arrow` and one bowl-to-plate relation.

### 4. All-arrow LoRA evaluated without arrows

- Model: all-arrow LoRA
- Trained on: all arrows
- Evaluated with: no arrows
- Checkpoint: epoch 15 (`029190`)
- Task 0: 5 / 10 successes
- Task 7: 0 / 10 successes
- **Final score: 5 / 20 (25%)**
- Evaluation PID: not running (complete)
- Output:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/checkpoint_probe_029190_tasks_0_7_treatment_no_arrows`
- Overlay evidence: no `visual_relation_audit.jsonl` is expected because
  `VISUAL_CONDITION=none`.

## Historical Lambda-only evaluation snapshot

### 6. No-arrow LoRA evaluated without arrows

- Trained on: no arrows
- Evaluated with: no arrows
- Planned checkpoint: epoch 15 (`029190`)
- Planned coverage: task 0 and task 7, 10 episodes each
- Status: **stale historical snapshot; not a current running-state claim**.
- Last observed PID on Lambda: 56554 at the preserved 2026-08-20 audit.
- The active forward run moved to Legion, and this Lambda process was not
  refreshed during the current Legion-only audit.
- Planned output:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/eval_outputs/no_arrow_treatment_epoch15_tasks_0_7`

## Verified missing baseline

The **frozen base evaluated without arrows** has not been run. No matching
`eval_info.json` exists on Lambda. The frozen-base 0/20 result above was
evaluated with all arrows.

This is different from the no-arrow LoRA evaluation: the frozen base has no
fine-tuning; the no-arrow LoRA receives 15 additional training epochs.

## Aborted attempts that are not results

These training directories produced no adapter checkpoint and must never be
reported as trained models:

- `pair_2026_08_17_18_46_29`
- `pair_2026_08_17_19_18_06`
- `pair_2026_08_17_19_21_06`
- `treatment_2026_08_17_19_40_32`

These evaluation directories produced no videos or `eval_info.json` and must
never be reported as results:

- `intermediate_eval_029190_live_arrows`

Do not delete these or any other Lambda artifacts without explicit user
authorization.

## Completed Legion no-arrow causal-control experiment

Audit time: **2026-08-30 19:59:06 UTC**.

### Training — `legion_no_arrow_lora_full_s1000_v1`

- Status: **complete** on PoliTO Legion.
- SLURM job: `1910197` on `gpu_a40`; it has left the live queue.
- Source commit: `8579b62e58aad28e131a8b8da370b4c34f2fc013`.
- Trained on: **no arrows**, using the no-arrow half of the exact sealed
  `sealed_lora_control_treatment` pair.
- Base revision: `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.
- Schedule: 15 epochs, 29,190 steps, checkpoint every 1,946 steps, batch 32,
  seed 1000, LoRA rank 16.
- Final checkpoint: `029190`; all 15 scheduled checkpoints exist.
- Completion evidence: step 29,190 and `End of training` at 08:27:49 UTC,
  adapter postcondition and reload smoke passed, launcher finished at 08:28:09
  UTC.
- Final adapter SHA-256:
  `80b3c23fc3987530d57766ab45ed33db918f08983739139c1ff0397184cc7092`.
- Training-manifest SHA-256:
  `95e376aff504265bea2bb53e63cc221fb42d7baa01dd6c3810317de85875c391`.
- Scratch run:
  `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/runs/legion_no_arrow_lora_full_s1000_v1_no_arrow_treatment_1910197`
- Durable archive:
  `/home/hjaber/EmbodimentSemantic_archive/runs/legion_no_arrow_lora_full_s1000_v1_no_arrow_treatment_1910197`
- Score: training itself has no rollout score; final evaluation results are
  recorded below.

The refreshed stage, no-arrow preflight, and two-step A40 smoke completed before
this submission. The smoke wrote and reloaded a no-arrow adapter successfully.

### Evaluation — `legion_no_arrow_trained_live_vs_none_s1000_ep10_v1`

- Status: **complete**, SLURM job `1910198`; it has left the live queue.
- It evaluates the same final no-arrow-trained adapter in exactly this order:
  1. `no_arrow_trained_live_arrows` — trained on no arrows, evaluated with live
     all-object arrows.
  2. `no_arrow_trained_no_arrows` — trained on no arrows, evaluated without
     arrows.
- Coverage per cell: tasks 0–9, 10 episodes per task, seed 1000, batch size 1.
- Randomization and prompt configuration: unchanged sealed current config.
- No frozen-base or arrow-trained model enters this evaluation job.
- Scratch evaluation root:
  `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/eval/legion_no_arrow_trained_live_vs_none_s1000_ep10_v1_no_arrow_treatment_1910198`
- Final scores from each cell's authoritative `eval_info.json`:
  - `no_arrow_trained_live_arrows` — trained on no arrows, evaluated with live
    all-object arrows: tasks 0–9 = `8, 4, 3, 4, 0, 0, 7, 1, 3, 0` successes
    out of 10; **30/100 (30%) final**.
  - `no_arrow_trained_no_arrows` — trained on no arrows, evaluated with no
    arrows: tasks 0–9 = `9, 9, 6, 4, 1, 1, 6, 4, 3, 0` successes out of 10;
    **43/100 (43%) final**.
- Cell completion times were 10:16:18 UTC and 11:45:18 UTC, respectively.
- Durable archive:
  `/home/hjaber/EmbodimentSemantic_archive/eval/legion_no_arrow_trained_live_vs_none_s1000_ep10_v1_no_arrow_treatment_1910198`.
- The final adapter, training manifest, both `eval_info.json` files, and the
  pair-summary CSV have identical SHA-256 hashes in scratch and durable HOME
  storage.

## Blocked graph-text training chain — `legion_graph_treatment_lora_full_s1000_v1_20260830T153645Z`

This chain did not reach training and has no checkpoint or score.

- Source commit: `615f0d23078f6ca36a03cb6fb9ba9bcccb1dc11f`.
- Setup job `1911244`: **failed** with exit code 1 while validating the
  serialized graph base-policy snapshot. The exact error was `base policy
  snapshot file inventory drifted`.
- Setup evidence was preserved and its archive verified with tree SHA-256
  `c0f1a1fb449a607bf520113867e2236a35a1bd2f971bc9b5604e7358469e3e82`.
- Training job `1911247`: still present in `squeue` as **PENDING /
  DependencyNeverSatisfied** because it requires `afterok:1911244(failed)`.
  It has run for 0 seconds and produced no training directory or checkpoint.
- Evaluation job `1911248`: **PENDING / Dependency** behind unstarted training
  job `1911247`; it has no evaluation output or score.
- No cancellation, repair, or resubmission was performed during this audit.

## Failed visual-path setup attempt — `legion_action_visual_lora_no_arrow_s1000_v1_20260830T185643Z`

This is a failed setup attempt, not a training or evaluation result. It must
remain separate from the planned candidate run below.

- Status: **failed during setup; no training or evaluation was submitted**.
- Intended policy: `action_visual_lora_v1`.
- Intended data profile: `no_arrow_treatment`.
- Source commit: `2bbda0bcd7241a68716c23ce73a0ddd1b67d205e`.
- Setup SLURM job: `1911343`.
- Failure causes:
  - the historical schema-1 manifest lacked the current sidecars;
  - its recorded sentinel hash was stale;
  - after the setup `sbatch` call, the launcher `submit_id` parser failed, so
    the setup job ID was not written to the launcher state.
- Scratch setup evidence:
  `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/runs/legion_action_visual_lora_no_arrow_s1000_v1_20260830T185643Z_setup_1911343`
- Durable setup archive:
  `/home/hjaber/EmbodimentSemantic_archive/setup/legion_action_visual_lora_no_arrow_s1000_v1_20260830T185643Z_1911343`
- Preserved scheduler logs:
  `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/legion_action_visual_lora_no_arrow_s1000_v1_20260830T185643Z_setup_1911343.out`
  and
  `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/legion_action_visual_lora_no_arrow_s1000_v1_20260830T185643Z_setup_1911343.err`
- Formal repair: `legacy_action_only_evidence_v1` is **implemented locally**
  as post-hoc/reconstructed evidence; resubmission has **not been launched**.
  It is not a result and does not authorize resubmission.

### Legacy evidence status for the retrospective comparison

- Evidence class: post-hoc/reconstructed `legacy_action_only_evidence_v1` for
  the historical `action_only_lora_v1` checkpoint.
- Raw original sentinel: unavailable; its stale recorded hash is not treated
  as evidence.
- Stable sealed-pair identity: revalidated against the current pair.
- Candidate evidence: native `native_policy_evidence_v1`.
- Comparison type: `retrospective_matched_checkpoint_evaluation`.
- Causal-ablation status: `retrospective_not_strict`.
- Permitted conclusion: exact checkpoint performance under the matched
  evaluation. A strict causal claim requires contemporaneous retraining of
  `action_only_lora_v1` under the same protocol.

## Planned visual-path LoRA diagnostic (not launched)

This section records the next policy experiment without inventing a job,
checkpoint, or result. Status is **implemented/planned, not launched**.

### Policy names

- Historical policy: `action_only_lora_v1`.
- Candidate policy: `action_visual_lora_v1`.
- Candidate framing: **action + late-visual LoRA** / **visual-path LoRA**.
  It retains the historical rank-16 action-expert and action/state targets,
  and adds rank-16 LoRA to the VLM connector plus vision encoder layers 8–11
  q/v. All original base weights, all non-adapted vision weights, and VLM text
  weights remain frozen; only the newly inserted LoRA tensors are trainable.
  Never call this policy “vision-unfrozen”; full vision unfreezing is a
  separate future method.

### Planned matched run

- Train the candidate on the exact `no_arrow_treatment` data used by the
  historical no-arrow `action_only_lora_v1` result.
- Evaluate both policy cells with **no arrows**: candidate
  `action_visual_lora_v1_no_arrows` versus the verified historical
  `historical_action_only_lora_v1_no_arrows` baseline. The training data
  profile remains `no_arrow_treatment` for both cells.
- Cover tasks 0–9 with ten episodes per task and matched seeds, reset/
  randomization configuration, and episode ordering where possible.
- Report the total and every per-task score for both cells.

Hypothesis: late visual-path adaptation can improve spatially grounded action
learning when the vision pathway is frozen. The literal success screen is
**greater than 43/100**; the preferred advancement is approximately **+5
points without a major task 0–1 regression**. This retrospective comparison
is classified as `retrospective_matched_checkpoint_evaluation` with
`causal_ablation_status: retrospective_not_strict`; an ICLR-quality strict
causal claim requires matched contemporaneous retraining seeds and paired
per-task/episode results. Until that run is launched and evaluated, this is a
plan—not a result.

Data profiles (`no_arrow_treatment`, `treatment`, and graph profiles) and
evaluation overlays (no arrows, live arrows, target arrow, or graph overlay)
remain separate axes. Always state both **trained on** and **evaluated with**.

## Permanent takeover checklist

1. Read this ledger and `evaluation_results_tracker.json` before answering any
   experiment question.
2. Apply the mandatory update transaction above on every user update request.
3. Use `eval_info.json` as the authoritative final success record. Logs and
   videos are supporting evidence and progress signals.
4. Report each run as: model, trained on, evaluated with, checkpoint, task 0
   score, task 7 score, total score, and final versus partial.
5. Never use bare `control`; say frozen base or no-arrow LoRA.
6. Keep the analysis at epoch 15 (`029190`). Do not add other checkpoints.
7. Update the Markdown ledger and JSON tracker in the same operation.
8. Preserve raw results, videos, logs, checkpoints, and provenance.

## Next actions, in order

1. Treat no-arrow training job `1910197` and evaluation job `1910198` as
   complete final results; their required durable artifacts match scratch.
2. Decide separately whether to cancel or repair the blocked graph-text jobs
   `1911247` and `1911248`; this audit made no scheduler changes.
3. Do not launch any additional baseline or seed unless the user explicitly
   requests it.

## Current interpretation boundary

Verified: the all-arrow LoRA epoch-15 checkpoint scored 11/20 when evaluated
with all arrows; the frozen base scored 0/20 under that same all-arrow rollout
condition. The Legion no-arrow LoRA epoch-15 checkpoint scored 43/100 when
evaluated with no arrows and 30/100 when evaluated with live all-object arrows.

The same no-arrow-trained checkpoint performed 13 percentage points worse with
live arrows than without them in this seed-1000 paired evaluation. That result
supports an evaluation-time overlay effect for this checkpoint, but it does not
by itself isolate training-data effects or establish a multi-seed causal claim.
