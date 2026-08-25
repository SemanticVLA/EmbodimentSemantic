# SmolVLA Arrow Experiments: Authoritative Takeover Ledger

This is the human-readable source of truth for the active SmolVLA LoRA
experiments. Its machine-readable companion is
`evaluation_results_tracker.json`. **Both files must always describe the same
state.**

The always-applied startup rule enforcing this requirement is
`.cursor/rules/smolvla-experiment-updates.mdc`.

**Analysis scope:** epoch 15 checkpoints only (`029190`, 29,190 steps). Do not
add any other checkpoints unless the user explicitly changes the scope.

**Last Lambda audit represented in both files:** 2026-08-20 10:11:17 UTC.
Anything marked running is a timestamped snapshot, not live state.

## Mandatory rule whenever the user asks for an update

Before answering any request such as "updates?", "where are we?", "how many?",
"is it done?", or "results so far?", the active thread must complete this exact
transaction:

1. Connect to Lambda and refresh every running training and evaluation job.
2. Inspect process state, the latest training log/checkpoints, completed episode
   artifacts, and `eval_info.json` when present.
3. Update `evaluation_results_tracker.json`, including `audited_at_utc`, job
   status, progress, PIDs, per-task counts, and whether each score is partial or
   final.
4. Update this ledger with the identical audit time and identical facts.
5. Parse the JSON and check both files for contradictory statuses or scores.
6. Only after steps 1–5, answer the user with the new numbers.

If Lambda cannot be reached, do not present this snapshot as current. Record and
report that the refresh failed, retain the last successful audit timestamp, and
label every shown count stale.

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

## No evaluation jobs running for these in-scope tasks

- No-arrow LoRA/no-arrow evaluation is running:
  - PID: 56554
  - Output:
    `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/eval_outputs/no_arrow_treatment_epoch15_tasks_0_7`
- Other in-scope evaluations: none running.

## Evaluation pending training completion

### 6. No-arrow LoRA evaluated without arrows

- Trained on: no arrows
- Evaluated with: no arrows
- Planned checkpoint: epoch 15 (`029190`)
- Planned coverage: task 0 and task 7, 10 episodes each
- Status: running (counts partial until eval_info.json completes)
- Evaluation PID: 56554
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

1. Monitor the no-arrow-LoRA/no-arrow evaluation to completion and record final
   `eval_info.json` counts when available.
4. Do not launch the missing frozen-base/no-arrow evaluation unless the user
   explicitly requests it.

## Current interpretation boundary

Verified: the all-arrow LoRA epoch-15 checkpoint scored 11/20 when evaluated
with all arrows; the frozen base scored 0/20 under that same all-arrow rollout
condition.

Not yet isolated: whether the gain is caused by arrow-conditioned learning,
generic additional fine-tuning, arrows at evaluation time, or an interaction.
The no-arrow-LoRA/no-arrow evaluation was still pending when the prior claim was
made, and is now running.
