# SmolVLA Arrow Experiments: Authoritative Takeover Ledger

This is the human-readable source of truth for the active SmolVLA LoRA
experiments. Its machine-readable companion is
`evaluation_results_tracker.json`. **Both files must always describe the same
state.**

The always-applied startup rule enforcing this requirement is
`.cursor/rules/smolvla-experiment-updates.mdc`.

**Analysis scope:** epoch 15 checkpoints only (`029190`, 29,190 steps). Do not
add any other checkpoints unless the user explicitly changes the scope.

**Last successful active-run audit represented in both files:** 2026-09-02 02:52:11 UTC.
The refresh attempt at 2026-09-02 08:24:21 UTC failed because the mp4/Legion
connection was unreachable; any active-job state below is stale until the next
successful refresh.
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

Audit time: **2026-08-30 20:19:37 UTC**.

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

## Graph-text training setup attempts

- Source commit for the final launcher: `e83e3f5e4b7f97b817366acbd89af021001de829`, pinned in the clean
  Legion graph checkout.
- Repair: the graph snapshot manifest excludes runtime `.cache` metadata, and
  the run uses a new immutable `-graph96-v2` snapshot path so the invalid legacy
  `-graph96` artifact is preserved rather than overwritten.
- Setup/smoke job `1911386` **failed at `2026-08-30T20:45:23Z`** after the
  graph-pair verification gate; training and evaluation were not submitted.
  The failure was a stale historical-pair sentinel digest in the existing
  derived graph manifest, not a source-data mismatch.
- Derived-artifact repair job `1911400` completed `0:0` at `21:37:06Z` after
  rebinding and fully verifying the graph-pair sentinel. A backup of the stale
  manifest is retained beside the repaired derived artifact.
- Fresh setup/smoke job `1911425` failed at `22:01:02Z` because its historical
  verify pass deleted the graph sentinel before the stale-hash check; no
  training/evaluation was submitted. A subsequent `1911444` attempt failed
  closed on the now-missing sentinel in 3s.
- Standalone repair verification job `1911451` completed successfully by log
  evidence at `22:52:02Z`: it verified 62,250 frames and 500 episodes, rewrote
  the graph-pair sentinel, and produced no stderr.
- The newest setup/smoke job `1911474`, using source commit `e83e3f5e4b7f97b817366acbd89af021001de829`,
  passed the graph-policy audit but failed at `23:17:40Z` because `job_dir` was
  unbound at line 146 of the generated SLURM script. Its setup archive is
  verified (`78cfafad7f44beef6b75f6dae5235af8199c49e5d92db4941eae1f675fcf2e3c`).
  The state file records `setup_status=FAILED`, `input_bundle_status=FAILED`,
  and no training or evaluation job IDs.
- Repaired setup/smoke job `1912529` completed successfully at
  `2026-09-01T09:49:34Z` from source commit
  `6a717c91a80d7a201ae99bd02d46d37151e3f701`. The existing graph pair passed
  immutable preflight, and the existing `graph96-v2` base snapshot was reused;
  no dataset regeneration or overwrite occurred.
- The repaired setup passed the 2-step GPU LoRA smoke (checkpoint `000002`),
  adapter reload, live checkpoint smoke for tasks 0 and 2, and the real
  terminal-success reset smoke. The previous list/`.ndim` failure was fixed by
  coercing the dummy action to a NumPy `float32` array before `LiberoEnv.step`.
  The setup archive is verified with tree hash
  `bad36313db6f05d60c3424edf7787bb8a63afe819fe28df66bb31f79906f4380`.
- After explicit launch confirmation, training job `1912720` completed and its
  paired evaluation job `1912721` started on `compute-4-11`. The verified setup
  state and sealed templates were reused; no second setup or dataset
  regeneration was performed.
- Fresh Legion audit: `2026-09-02T02:47:32Z`. Training reached step `29,190`
  / epoch `15`, saved all 15 checkpoints through final `029190`, passed the
  adapter reload smoke, and finished at `01:50:19Z`; its archive is verified.
  Evaluation `1912721` is still **RUNNING**, currently in the
  `graph_trained_graph_context` cell. Six task video artifacts (tasks 0–5) are
  present so far; final success metrics are not available yet.
- At the last successful `2026-09-02T02:52:11Z` audit, the evaluator still had no
  complete `eval_info.json` or `randomization_audit.jsonl` rows. Therefore no
  per-task success count is reported yet; the existing task 0–5 artifacts are
  partial debug/video outputs, not evaluation results.
- Refresh attempt `2026-09-02T08:24:21Z` failed at the mp4 gateway, so the
  RUNNING state and partial artifact counts above are stale, not a current
  Legion observation.
- Trained on: graph-text `target_natural_v1` with no visual arrows, using the
  sealed `graph_treatment`/`arrow_graph_treatment` pair.
- Planned contract: 15 epochs, 29,190 steps, checkpoint `029190`, batch 32,
  LoRA rank 16, seed 1000; paired evaluation uses graph-present and
  graph-removed text with no visual arrows, 10 episodes per task for tasks 0–9.
- The failed-attempt state file above is historical. The current verified setup
  state is `/home/hjaber/EmbodimentSemantic_runtime/graph_pilot/legion_graph_treatment_lora_full_s1000_v1_20260901T091457Z/state.env`.
- The setup smoke checkpoint `000002` is diagnostic only. The full-training
  final checkpoint is `029190`; no evaluation score exists yet. The old failed
  chain's jobs `1911247` and `1911248` remain dependency-blocked and are not
  current training progress.

The previous setup job `1911381` was cancelled before completion because it was
still targeting the stale derived snapshot. It produced no training/evaluation
submission; its partial setup evidence remains archived. The follow-up setup
`1911382` was also cancelled before completion after the launcher-state contract
fix was identified; it produced no training/evaluation submission and must not
be reused. Failed setup `1911386` is likewise retired after exposing the stale
historical-pair sentinel. Setup `1911474` is the latest attempt and is terminally
failed; setup `1912529` is the later verified setup. Training `1912720` is
complete and evaluation `1912721` is the active dependent evaluation.

## Completed action-visual training — `legion_action_visual_lora_no_arrow_s1000_v7_20260831T101333Z`

Fresh Legion audit: **2026-09-01 09:02:03 UTC**.

- Training job `1911789` is **COMPLETE**. The workload exited `0`, its durable
  archive is verified, and it is no longer present in `squeue`.
- Trained on: **no arrows**, dataset variant `control`, policy
  `action_visual_lora_v1`, seed `1000`, batch size `32`, LoRA rank `16`.
- Source commit: `98f9e295fe05400cbbce6d1e9cf500222327dac5`.
- Final progress: step `29,190 / 29,190` (**100%**), epoch `15`; the last
  progress line reported loss `0.409` at `2026-09-01 04:28:59 UTC`.
- Final saved checkpoint: `029190`, written at `2026-09-01 04:35:02 UTC`.
  All 15 scheduled checkpoints exist, the post-training adapter audit passed,
  and the final adapter reload smoke test passed.
- Run directory:
  `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/runs/legion_action_visual_lora_no_arrow_s1000_v7_20260831T101333Z_candidate`.
- Training completed at `2026-09-01 04:36:10 UTC`; no fatal signature appears
  in the training log. Verified archive tree SHA-256:
  `33a84486fc8b9c5371d1075a8a74da0020b3b00a48963b1ad1cc91fe3d000885`.
- Dependent evaluation job `1911790` ran after training and completed both
  100-episode no-arrow cells. The new action-visual checkpoint scored
  **26/100 (26%)** with per-task successes `2, 8, 0, 1, 0, 0, 3, 4, 8, 0`;
  the historical action-only checkpoint scored **43/100 (43%)** with per-task
  successes `9, 9, 6, 4, 1, 1, 6, 4, 3, 0`.
- Job `1911790` nevertheless exited `1` because the policy cells did not use
  identical reset/randomization identities. The two individual `eval_info.json`
  results are complete and archived, but **their difference is not a valid
  paired comparison**. The verified evaluation archive tree SHA-256 is
  `77e52ebcf28a150b5e1e7f5f7be09a7690e9824aa2632fa9a90d0625ae9d4f02`.
- Clarification: job `1911790` was **two policy cells under no arrows**
  (new action-visual LoRA versus historical action-only LoRA), not an
  arrows-versus-no-arrows pair. The separate arrows-versus-no-arrows job was
  `1912060` at checkpoint `007784`: live arrows completed at `15/100`, while
  the no-arrow cell was cancelled before any episode. There is no final
  checkpoint `029190` live-arrow evaluation yet.
- Training has no success-rate result. Evaluation scores remain separate.

## Intermediate action-visual checkpoint evaluation — stopped after arrows

Fresh Legion audit: **2026-08-31 19:38:08 UTC**.

- SLURM job `1912060` evaluated checkpoint `007784` (epoch
  `4.0014136546`) from training job `1911789`.
- Trained on: **no arrows** with policy `action_visual_lora_v1`.
- Live-arrow scenario: **complete and final**, tasks 0–9 successes out of 10 =
  `1, 4, 1, 0, 1, 1, 4, 2, 1, 0`; total **15/100 (15%)**.
- No-arrow scenario: initialization started at `2026-08-31 19:02:48 UTC`, but
  it completed **zero episodes**. At the user's request, job `1912060` was
  cancelled at `2026-08-31 19:37:33 UTC`; SLURM reports `CANCELLED`, exit
  `0:15`. There is no no-arrow `eval_info.json` and therefore no no-arrow score.
- The job's exit handler preserved and hash-verified the completed live-arrow
  `eval_info.json`, audit files, checkpoint snapshot, provenance, and empty
  no-arrow audit placeholder under:
  `/home/hjaber/EmbodimentSemantic_archive/eval/legion_action_visual_lora_no_arrow_s1000_v7_20260831T101333Z_checkpoint_007784_arrows_then_none_eval_1912060`.
- Archive tree SHA-256:
  `6c70663ca8715771ed42608629f25e9cded07b187ece6b2b526deeeb73245b70`.
- Training job `1911789` was not cancelled and later completed successfully.
  Its dependent final evaluation job `1911790` also ran; its current status and
  results are recorded in the completed-training section above.

## Historical failed job chains — superseded by active v7 training

Historical Legion audit: **2026-08-31 08:37:13 UTC**. These are retained as
failed-attempt provenance and are not the current active training.

- **Action-visual LoRA:** training job `1911374` (`action_visual_lora_no_arrow_s1000_v2_20260830T200506Z`) is `PENDING` with `DependencyNeverSatisfied` because setup `1911373` failed while building the legacy action-only evidence bundle. Its dependent evaluation job is `1911375`, still dependency-blocked. No training runtime, checkpoint, or `eval_info.json` exists.
- **Graph-text LoRA:** training job `1911247` is `PENDING` with `DependencyNeverSatisfied` because setup `1911244` failed graph-policy inventory validation. Its dependent evaluation job is `1911248`, still dependency-blocked. No training runtime, checkpoint, or `eval_info.json` exists.
- The later graph setup retry `1911474` also failed before issuing a new training
  job. These failed chains have no per-task success rates; active v7 training is
  tracked in the current section above.

## Permanent takeover checklist

1. Read this ledger and `evaluation_results_tracker.json` before answering any
   experiment question.
2. Apply the mandatory update transaction above on every user update request.
3. Use `eval_info.json` as the authoritative final success record. Logs and
   videos are supporting evidence and progress signals.
4. Report each run as: model, trained on, evaluated with, checkpoint, task 0
   score, task 7 score, total score, and final versus partial.
5. Never use bare `control`; say frozen base or no-arrow LoRA.
6. Default final comparisons to epoch 15 (`029190`). Track another checkpoint
   only when the user explicitly requests it; checkpoint `007784` is one such
   recorded exception.
7. Update the Markdown ledger and JSON tracker in the same operation.
8. Preserve raw results, videos, logs, checkpoints, and provenance.

## Next actions, in order

1. Treat action-visual training job `1911789` and checkpoint `029190` as the
   completed final training artifact.
2. Preserve both `eval_info.json` cell results from job `1911790`, but do not
   interpret the 26% versus 43% delta as paired because reset identities differ.
3. If an arrows-versus-no-arrows result for the final checkpoint is needed,
   launch a fresh paired evaluation with identical reset identities; the prior
   arrows result is only for checkpoint `007784`.
3. Keep the failed action-visual and graph-text chains as historical provenance;
   do not reuse their failed states.
4. Verify durable HOME archives match scratch outputs; do not launch any
   additional baseline or seed unless explicitly requested.

## Current interpretation boundary

Verified: the all-arrow LoRA epoch-15 checkpoint scored 11/20 when evaluated
with all arrows; the frozen base scored 0/20 under that same all-arrow rollout
condition.

Not yet isolated: whether the gain is caused by arrow-conditioned learning,
generic additional fine-tuning, arrows at evaluation time, or an interaction.
Historical failed jobs `1911374` and `1911247` never started. Their failures are
not evidence about active action-visual job `1911789`, which is running and has
no final evaluation result yet.
