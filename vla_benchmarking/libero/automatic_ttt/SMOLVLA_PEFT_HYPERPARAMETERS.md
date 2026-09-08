# SmolVLA automatic-TTT PEFT hyperparameter ledger

This run is task-specific SmolVLA LoRA/PEFT behavior cloning on successful
Arrow demonstrations collected from fresh reset environments. It is
deliberately not RoboTTT: RoboTTT trains temporal TTT fast weights with
sequence flow matching and interleaved policy failures/teacher corrections,
while this path trains a native SmolVLA action-side LoRA adapter. The ledger
exists so a checkpoint cannot silently inherit a library default.

## Paper-matched values

| Setting | Value | Category | Reason |
|---|---:|---|---|
| post-training optimizer budget | `ceil(5 × dataset_frames / 8)` updates | `USER_PILOT_DERIVED` | User requested five dataset-equivalent epochs with the existing effective batch of 8; the exact step count is computed after export and recorded. RoboTTT's 20K updates are not used. |
| optimizer | AdamW | `PAPER_EXPLICIT` | Appendix A.2. |
| peak learning rate | 5e-5 | `PAPER_EXPLICIT` | Appendix A.2. |
| weight decay | 1e-5 | `PAPER_EXPLICIT` | Appendix A.2. |
| effective global batch | 8 | `PAPER_DERIVED` | Paper uses 8 GPUs × batch 1/device. This one-GPU pilot uses batch 8 because LeRobot 0.5.2 has no gradient-accumulation field in the pipeline config. |
| cosine scheduler decay span | derived optimizer steps | `USER_PILOT_DERIVED` | Decay equals the exact five-epoch step count; warmup is `floor(steps/30)` to preserve the native SmolVLA warmup fraction. |

The paper does **not** publish LoRA rank/alpha/dropout, warmup, minimum LR,
Adam betas/epsilon, gradient clipping, precision, dataloader settings, action
horizon, or denoising steps. None of those values below are labelled as paper
values.

## Explicit port and environment contract

| Setting | Value | Category | Reason |
|---|---:|---|---|
| LoRA rank | 16 | `SMOLVLA_PINNED` | Existing canonical SmolVLA PEFT workflow (run 1910197); preserves VLA-specific comparability. |
| LoRA alpha | 8 | `SMOLVLA_PINNED` | Existing canonical workflow; explicitly serialized and audited. |
| LoRA dropout | 0 | `SMOLVLA_PINNED` | Existing canonical workflow; no stochastic adapter dropout. |
| LoRA bias | `none` | `SMOLVLA_PINNED` | Existing canonical workflow; base bias is not trained. |
| LoRA initialization | `true` | `SMOLVLA_PINNED` | Existing canonical workflow's standard initialization. |
| rsLoRA | `false` | `SMOLVLA_PINNED` | Existing canonical workflow. |
| fan-in/fan-out | `false` | `SMOLVLA_PINNED` | Existing canonical workflow for linear layers. |
| target modules | SmolVLA action-side q/v projections plus state/action projections | `SMOLVLA_PINNED` | Reuses the official SmolVLA action-side target regex; vision backbone remains frozen. |
| SmolVLA chunk size | 50 | `SMOLVLA_PINNED` | Pinned base config. |
| SmolVLA action steps | 1 | `SMOLVLA_PINNED` | Pinned checkpoint/evaluation contract; evaluation preserves checkpoint semantics. |
| SmolVLA observation steps | 1 | `SMOLVLA_PINNED` | Pinned base config. |
| SmolVLA denoising steps | 10 | `SMOLVLA_PINNED` | Pinned base config. This is not RoboTTT's flow-matching noise sampling. |
| image preprocessing | resize to 512×512 | `SMOLVLA_PINNED` | Pinned SmolVLA processor config. |
| tokenizer max length | 48 | `SMOLVLA_PINNED` | Pinned SmolVLA processor config. |
| AMP | false | `SMOLVLA_PINNED` | Pinned SmolVLA config and explicit `ACCELERATE_MIXED_PRECISION=no`. |
| Adam betas | (0.9, 0.95) | `SMOLVLA_PINNED` | Pinned SmolVLA config; paper is silent. |
| Adam epsilon | 1e-8 | `SMOLVLA_PINNED` | Pinned SmolVLA config; paper is silent. |
| gradient clipping | 10 | `SMOLVLA_PINNED` | Pinned SmolVLA config; paper is silent. |
| resolved warmup | `floor(steps/30)` updates | `USER_PILOT_DERIVED` | Native SmolVLA scheduler fraction (1000/30000), applied to the user-requested dynamic step budget. |
| final LR | 2.5e-6 | `SMOLVLA_PINNED` | Pinned SmolVLA scheduler floor; paper is silent. |
| dataloader workers/prefetch/persistence | 4 / 4 / true | `OPERATIONAL` | Explicit throughput settings, recorded in manifests. |
| checkpoint cadence | `min(2,000, steps)` updates | `OPERATIONAL` | Safe cadence for small datasets; only the exact final derived-step checkpoint is scored. |
| train seed | 1000 | `USER_STUDY_DESIGN` | Fixed reproducible adapter initialization/data order. |
| collection successes | 50 per task | `USER_STUDY_DESIGN` | Requested study design. Failed attempts are discarded from training. |
| collection adaptation seeds | 3000 onward, max 500 attempts | `USER_STUDY_DESIGN` / `OPERATIONAL` | Disjoint from sealed eval seeds; 500 is 10 attempts per required success and a fail-closed safety cap. |
| collection mode | fresh Arrow rollout from reset | `USER_STUDY_DESIGN` | Arrow acts immediately in a new sealed-randomized environment. SmolVLA is not loaded/called during collection; failed attempts are discarded. |
| Arrow correction budget | 1,200 env steps | `CONTROLLER_PINNED` | Canonical Arrow controller budget. |
| collection FPS | 20 | `CONTROLLER_PINNED` | Canonical LIBERO controller rate. |
| eval seeds | 1000–1009 | `USER_STUDY_DESIGN` | Same sealed ten episodes before/after adaptation. |
| eval horizon | 280 env steps | `SMOLVLA_PINNED` | Same horizon as collection and both baseline/adapted evaluations. |
| eval cameras/resolution | agentview + wrist / 256 | `SMOLVLA_PINNED` | Same canonical live/evaluation observation contract. |
| timestamp tolerance | 1e-4 s | `OPERATIONAL` | Explicit LeRobot 0.5.2 synchronization tolerance. |

## Epoch-equivalent reporting

The requested training budget is five dataset-equivalent epochs. For a dataset
with `F` exported frames and effective batch `B=8`, the launcher computes:

```text
steps = ceil(5 × F / 8)
epoch_equivalent = steps × 8 / F
```

Both values are computed after Arrow export and recorded. The achieved value is
always at least 5 and less than `5 + 8/F` due to the ceiling.

## Provenance and fail-closed rules

The launcher passes every value available in LeRobot 0.5.2 and records the
resolved values in `run_context.json`, `training_plan.json`, runtime evidence,
the adapter audit, and the immutable artifact manifest. The audit rejects any
checkpoint whose adapter config differs in rank, alpha, dropout, bias,
initialization, rsLoRA, fan-in/fan-out, target modules, or modules-to-save.
The job asserts `lerobot==0.5.2` and `peft==0.18.0`. Missing runtime fields,
wrong versions, dirty source, or an unverified base snapshot stop the job.
