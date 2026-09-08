# Shared LIBERO Spatial evaluation

This package is the single evaluation home for the canonical grasp controller,
fine-tuned VLA policies, and baseline LeRobot policies. It owns experiment
conditions, task/seed schedules, randomization, visual inputs, text context,
audits, and result aggregation. Policy packages continue to own inference and
robot actions.

## Independent evaluation factors

- Suite: `normal` or `sealed_randomized`. The direct historical backend records
  `normal` as its native value, `vanilla`, to preserve existing manifests.
- Visual input: `none`, `goal_arrow`, or `relation_arrows`.
- Text input: `none`, `scene_graph`, or `text_triplet` with an explicit format
  from `scene_graph_formats.py`.

`contracts.py` defines the stable condition and cell schedule. `registry.py`
declares which combinations each policy accepts and fails closed on unsupported
inputs.

## Execution backends

- `run_arrow_pick_place_matrix.py` is the direct per-cell backend used by the
  canonical controller and language-free ArrowStudent. Their policy adapters
  retain their own motion/inference loops.
- `run_lerobot_eval_with_context.py` is the LeRobot backend used by base and
  LoRA VLA policies. It retains the existing processor, camera, vector-env,
  reset, and `eval_info.json` behavior.
- `run_policy_eval.py` is the native-policy boundary used by Pi0.5,
  OpenVLA-OFT, and Octo. It validates each adapter's native action chunk
  (`10`, `8`, and `4` respectively) without forcing the models through one
  model-specific loop.
- `native_vla_eval.py` is the plan-consuming bridge for Pi0.5/OpenVLA-OFT.
  Their command builders switch to it whenever a shared v2 plan is supplied;
  the legacy upstream commands remain available only when no plan is passed.

`policy_adapter.py` defines the shared `reset(task_description, episode_seed)` /
`act(observation)` contract, immutable artifact metadata, arrow-free markers,
and action shape/range/finite-value validation. Model packages remain
responsible for their own preprocessing, normalization, and simulator stepping.

New policy plans use `shared_evaluation_plan.v2`; existing SmolVLA and controller
manifests continue to validate as `shared_evaluation_plan.v1`.

The package does not merge these into a universal action loop. Evaluator
outputs are reporting data only and never enter candidate or action selection.

## Matrix outcome reporting

`run_arrow_pick_place_matrix.py` keeps execution lifecycle (`status`) separate
from the benchmark result (`outcome_status`). Every persisted cell and summary
uses these outcome values:

- `success`: the evaluator returned `true`.
- `failure`: the evaluator returned `false`, or the terminal controller
  manifest records a failed grasp, recovery, or candidate-generation outcome.
- `unresolved`: execution ended without a trustworthy benchmark result (for
  example an environment/input/evaluator exception, interruption, or a
  completed row with no evaluator result).
- `not_run`: the planned cell was not executed.
- `not_evaluated`: dry-run evidence only; no benchmark result is claimed.

The summary's `outcome_*` counts and rates are the unambiguous reporting
surface; legacy lifecycle/evaluator fields remain for compatibility.

The former root-level launcher paths were removed. Imports and direct launches
must use `vla_benchmarking.libero.evaluation`.
