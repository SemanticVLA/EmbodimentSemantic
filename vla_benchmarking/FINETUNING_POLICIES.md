# SmolVLA fine-tuning policies

This file is the naming contract for future threads. A **fine-tuning policy**
specifies which model parameters receive LoRA adapters. It is independent of
the data profile used to train the adapter and independent of the visual
overlay used during evaluation.

## Canonical policy IDs

### `action_only_lora_v1` (historical)

This is the policy used by the existing arrow and no-arrow LoRA runs. It uses
rank-16 LoRA on the action expert q/v projections and the action/state input
and output projections. The VLM connector, vision encoder, and VLM text
weights remain frozen. When referring to the historical 43/100 no-arrow
baseline, use this exact policy ID.

### `action_visual_lora_v1` (candidate)

This policy retains every `action_only_lora_v1` target and adds rank-16 LoRA
to:

- the VLM connector (`modality_projection.proj`);
- q/v projections in vision encoder layers 8, 9, 10, and 11.

All original base weights remain frozen, including the non-adapted vision
weights and the VLM text weights. Only the newly inserted LoRA tensors on the
listed modules are trainable. This is **action + late-visual LoRA** or
**visual-path LoRA**. Never call it “vision-unfrozen”: full vision unfreezing
is a different future method and must receive a different policy ID.

The implementation records the policy ID, target expression, expected adapter
inventory, rank, and trainable parameter count in run provenance. A run is not
valid for this policy unless its live adapter inventory passes that audit.

## Keep the experimental axes separate

Use these names literally in manifests, job names, evaluations, and reports:

- **Data profile / trained on:** `no_arrow_treatment` (clean LIBERO images),
  `treatment` (baked-in arrows), or a graph profile. A profile describes what
  was present in the fine-tuning data.
- **Fine-tuning policy:** `action_only_lora_v1` or
  `action_visual_lora_v1`. A policy describes what was trainable.
- **Evaluation overlay / evaluated with:** no arrows, live all-object arrows,
  a single live target arrow, or a graph overlay. This describes what the
  policy saw during rollout, not what it was trained on.

For example, a training run records policy `action_visual_lora_v1` with data
profile `no_arrow_treatment`, while its clean evaluation cell is
`action_visual_lora_v1_no_arrows`. The former identifies the training policy
and data profile; the latter identifies the no-arrow evaluation overlay. The
evaluation overlay must always be stated separately.
Do not use the ambiguous word “control” as a model name.

## First planned candidate run

Status: **implemented/planned, not launched**. No job ID, checkpoint, score,
or improvement is claimed here.

The first diagnostic should train `action_visual_lora_v1` on the exact
`no_arrow_treatment` data used for the verified historical
`action_only_lora_v1` baseline. It should then run a paired clean evaluation:

- baseline cell: `action_only_lora_v1_no_arrows`,
- candidate cell: `action_visual_lora_v1_no_arrows`,
- evaluated with: no arrows for both cells,
- tasks: 0–9, ten episodes per task, identical reset/randomization schedule,
  seed, and episode ordering where possible,
- report: total score and every task score for both cells.

The hypothesis is that frozen visual representations are a bottleneck for
spatially grounded action learning, and that adapting only the late visual
path can improve the clean no-arrow result without changing the data
condition. The literal screen is **strictly greater than 43/100**. A useful
advancement target is roughly **+5 percentage points** with no major
regression on tasks 0–1; a smaller or uneven gain should be treated as
inconclusive rather than framed as a win.

For an ICLR-quality claim, rerun the matched baseline and candidate with the
same training seeds and record paired per-task and episode-level results.
The existing 43/100 result is a verified baseline for planning, not evidence
that this candidate has improved. Full vision unfreezing remains a separate
future method/ablation.
