# Active SmolVLA 2x2

This document tracks only the protected SmolVLA 2x2 treatment family. Other
fine-tuning policies remain preserved in their own named sibling folders, but
they are not part of this active comparison.

## Training identities

| Training condition | Source identity | Training artifact |
| --- | --- | --- |
| No arrows | source commit `8579b62`; Legion training job `1910197` | checkpoint `029190`; adapter SHA-256 `80b3c23fc3987530d57766ab45ed33db918f08983739139c1ff0397184cc7092`; training manifest SHA-256 `95e376aff504265bea2bb53e63cc221fb42d7baa01dd6c3810317de85875c391` |
| All arrows | tracker ID `all_arrows_lora_epoch_15_checkpoint` | Lambda artifact `treatment_2026_08_17_19_42_03/.../029190` (the recorded source does not provide a commit or Legion job) |

## Four evaluation cells

The same shared evaluation contracts and LeRobot backend are used for both
training identities:

| Training | Evaluation input | Result |
| --- | --- | --- |
| No arrows | No arrows | 43/100 (Legion job `1910198`) |
| No arrows | Live arrows | 30/100 (Legion job `1910198`) |
| All arrows | No arrows | 5/20 (Lambda; tasks 0 and 7) |
| All arrows | Live arrows | 11/20 (Lambda; tasks 0 and 7) |

The sample sizes are intentionally different: the no-arrow-trained pair is a
full sealed-randomized 100-cell Legion evaluation, while the all-arrow-trained
pair is a 20-cell tasks-0/7 Lambda evaluation. These are preserved results,
not directly interchangeable aggregate scores.

Active workflow entrypoints are under `workflows/`; the common LIBERO
evaluation implementation is under `../evaluation/`. Every retained policy
family has its own named sibling folder under this directory; there is no
generic fine-tuning archive folder.
