# Active SmolVLA cases

Only two matched action-only SmolVLA fine-tuning cases are active in this
checkout: no-arrow training and target-arrow-only training. Both use the same
base revision, LoRA targets and rank, seed, cameras, batch size, step count,
save frequency, task set, and shared evaluation implementation.

## Training identities

| Training condition | Source identity | Training artifact |
| --- | --- | --- |
| No arrows | source commit `8579b62`; Legion training job `1910197` | checkpoint `029190`; adapter SHA-256 `80b3c23fc3987530d57766ab45ed33db918f08983739139c1ff0397184cc7092`; training manifest SHA-256 `95e376aff504265bea2bb53e63cc221fb42d7baa01dd6c3810317de85875c391` |
| Target arrow only | matched active profile | No canonical checkpoint or result yet. Train from the same frozen schedule using only the `target_arrow_treatment` dataset. |

## Evaluation matrix

Each trained checkpoint is evaluated twice with identical randomized cells:

| Training | Evaluation input | Status |
| --- | --- | --- |
| No arrows | No arrows | Historical canonical result: 43/100, Legion job `1910198` |
| No arrows | Live all-scene arrows | Historical comparison: 30/100, Legion job `1910198` |
| Target arrow only | No arrows | Not run for the new matched checkpoint |
| Target arrow only | Live target arrow | Not run for the new matched checkpoint |

The target-arrow visual condition is exactly one `akita_black_bowl_1` to task
goal arrow in `agentview`; the wrist image, actions, state, and task text remain
unchanged. `DISABLE_VISUAL_PROMPT_HINT=1` prevents an unintended text hint from
being added during target-arrow evaluation.

Active launchers live under `smolvla/workflows/`. Removed all-arrow, graph-text,
visual-LoRA, and ArrowStudent profiles remain recoverable from Git history but
are not present or operator-reachable in this checkout.
